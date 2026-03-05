//! Material hot-reload — registry, RON serialization, and change diffing.
//!
//! [`MaterialRegistry`] tracks live material definitions and a dirty flag so
//! the renderer knows when to re-upload the GPU material buffer.
//! [`parse_material_file`] / [`serialize_material_file`] handle round-tripping
//! through RON, and [`diff_material_files`] computes a [`MaterialChangeSet`]
//! between two file versions.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Material Properties ─────────────────────────────────────────────────────

/// PBR + SSS properties for a single material, serializable to RON.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialProperties {
    /// Base color (linear RGB, 0–1).
    pub albedo: [f32; 3],
    /// Roughness (0 = mirror, 1 = diffuse).
    pub roughness: f32,
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Emission strength multiplier.
    pub emission_strength: f32,
    /// Subsurface scattering radius in world units.
    pub sss_radius: f32,
    /// Subsurface scattering tint (linear RGB).
    pub sss_color: [f32; 3],
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            emission_strength: 0.0,
            sss_radius: 0.0,
            sss_color: [1.0, 1.0, 1.0],
        }
    }
}

// ── Material Definition ─────────────────────────────────────────────────────

/// A named, hashed material tracked by the registry.
#[derive(Debug, Clone)]
pub struct MaterialDefinition {
    /// Material table index (0–65535).
    pub id: u16,
    /// Human-readable name.
    pub name: String,
    /// Hash of the serialized properties for change detection.
    pub source_hash: u64,
    /// Current property values.
    pub properties: MaterialProperties,
}

// ── Material File (RON) ─────────────────────────────────────────────────────

/// A single entry in a `.ron` material file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialFileEntry {
    /// Material table index.
    pub id: u16,
    /// Human-readable name.
    pub name: String,
    /// PBR + SSS properties.
    pub properties: MaterialProperties,
}

/// Top-level structure of a `.ron` material definitions file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialFile {
    /// All material entries.
    pub materials: Vec<MaterialFileEntry>,
}

// ── Registry ────────────────────────────────────────────────────────────────

#[deprecated(note = "Use rkf_core::MaterialLibrary instead — file-driven material management with hot-reload")]
/// Central mutable registry of material definitions.
#[derive(Debug, Clone)]
pub struct MaterialRegistry {
    /// All registered definitions, unordered.
    definitions: Vec<MaterialDefinition>,
    /// Optional filesystem path being watched for changes.
    pub watch_path: Option<String>,
    /// Whether any definition has been modified since last clear.
    dirty: bool,
}

impl MaterialRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            watch_path: None,
            dirty: false,
        }
    }

    /// Register a new material definition.
    pub fn register(&mut self, id: u16, name: &str, props: MaterialProperties) {
        let hash = hash_properties(&props);
        self.definitions.push(MaterialDefinition {
            id,
            name: name.to_string(),
            source_hash: hash,
            properties: props,
        });
        self.dirty = true;
    }

    /// Update the properties of an existing material by id.
    /// Returns `true` if the material was found and updated.
    pub fn update(&mut self, id: u16, props: MaterialProperties) -> bool {
        if let Some(def) = self.definitions.iter_mut().find(|d| d.id == id) {
            def.source_hash = hash_properties(&props);
            def.properties = props;
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Look up a material by id.
    pub fn get(&self, id: u16) -> Option<&MaterialDefinition> {
        self.definitions.iter().find(|d| d.id == id)
    }

    /// Look up a material mutably by id.
    pub fn get_mut(&mut self, id: u16) -> Option<&mut MaterialDefinition> {
        self.definitions.iter_mut().find(|d| d.id == id)
    }

    /// Number of registered materials.
    pub fn count(&self) -> usize {
        self.definitions.len()
    }

    /// Whether any definition has changed since the last [`clear_dirty`] call.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag (typically after GPU upload).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Manually set the dirty flag.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Remove a material by id. Returns `true` if found and removed.
    pub fn remove(&mut self, id: u16) -> bool {
        let before = self.definitions.len();
        self.definitions.retain(|d| d.id != id);
        let removed = self.definitions.len() < before;
        if removed {
            self.dirty = true;
        }
        removed
    }

    /// Slice of all registered definitions.
    pub fn all_definitions(&self) -> &[MaterialDefinition] {
        &self.definitions
    }
}

impl Default for MaterialRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parse / Serialize ───────────────────────────────────────────────────────

/// Deserialize a RON material definitions file.
pub fn parse_material_file(ron_str: &str) -> Result<MaterialFile, String> {
    ron::from_str(ron_str).map_err(|e| format!("RON parse error: {e}"))
}

/// Serialize a [`MaterialFile`] to RON.
pub fn serialize_material_file(file: &MaterialFile) -> Result<String, String> {
    ron::ser::to_string_pretty(file, ron::ser::PrettyConfig::default())
        .map_err(|e| format!("RON serialize error: {e}"))
}

// ── Diff ────────────────────────────────────────────────────────────────────

/// Summary of what changed between two material file versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialChangeSet {
    /// Material IDs present in new but not old.
    pub added: Vec<u16>,
    /// Material IDs present in both but with different properties.
    pub modified: Vec<u16>,
    /// Material IDs present in old but not new.
    pub removed: Vec<u16>,
}

impl MaterialChangeSet {
    /// Whether no changes were detected.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.removed.is_empty()
    }

    /// Total number of individual changes.
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.modified.len() + self.removed.len()
    }
}

/// Diff two material file versions and return what changed.
pub fn diff_material_files(old: &MaterialFile, new: &MaterialFile) -> MaterialChangeSet {
    let mut added = Vec::new();
    let mut modified = Vec::new();
    let mut removed = Vec::new();

    // Build lookup for old entries by id.
    let old_map: std::collections::HashMap<u16, &MaterialFileEntry> =
        old.materials.iter().map(|e| (e.id, e)).collect();
    let new_map: std::collections::HashMap<u16, &MaterialFileEntry> =
        new.materials.iter().map(|e| (e.id, e)).collect();

    // Check new entries against old.
    for entry in &new.materials {
        match old_map.get(&entry.id) {
            None => added.push(entry.id),
            Some(old_entry) => {
                if old_entry.properties != entry.properties || old_entry.name != entry.name {
                    modified.push(entry.id);
                }
            }
        }
    }

    // Check for removals.
    for entry in &old.materials {
        if !new_map.contains_key(&entry.id) {
            removed.push(entry.id);
        }
    }

    MaterialChangeSet {
        added,
        modified,
        removed,
    }
}

// ── Internal Helpers ────────────────────────────────────────────────────────

/// Simple FNV-1a hash of the RON representation of material properties.
fn hash_properties(props: &MaterialProperties) -> u64 {
    // Use RON serialization for a deterministic byte sequence.
    let text = ron::to_string(props).unwrap_or_default();
    crate::shader_reload::compute_source_hash(&text)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_props() -> MaterialProperties {
        MaterialProperties::default()
    }

    fn red_props() -> MaterialProperties {
        MaterialProperties {
            albedo: [1.0, 0.0, 0.0],
            roughness: 0.3,
            ..Default::default()
        }
    }

    // -- MaterialRegistry -----------------------------------------------------

    #[test]
    fn registry_starts_empty() {
        let reg = MaterialRegistry::new();
        assert_eq!(reg.count(), 0);
        assert!(!reg.is_dirty());
    }

    #[test]
    fn register_and_get() {
        let mut reg = MaterialRegistry::new();
        reg.register(0, "stone", default_props());
        reg.register(1, "metal", red_props());

        assert_eq!(reg.count(), 2);
        let stone = reg.get(0).unwrap();
        assert_eq!(stone.name, "stone");
        assert!(reg.is_dirty());
    }

    #[test]
    fn update_existing() {
        let mut reg = MaterialRegistry::new();
        reg.register(0, "stone", default_props());
        reg.clear_dirty();

        let updated = reg.update(0, red_props());
        assert!(updated);
        assert!(reg.is_dirty());

        let stone = reg.get(0).unwrap();
        assert!((stone.properties.albedo[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn update_nonexistent_returns_false() {
        let mut reg = MaterialRegistry::new();
        assert!(!reg.update(99, default_props()));
    }

    #[test]
    fn get_mut_modifies() {
        let mut reg = MaterialRegistry::new();
        reg.register(5, "test", default_props());

        let def = reg.get_mut(5).unwrap();
        def.properties.roughness = 1.0;
        assert!((reg.get(5).unwrap().properties.roughness - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn remove_material() {
        let mut reg = MaterialRegistry::new();
        reg.register(0, "a", default_props());
        reg.register(1, "b", default_props());
        reg.clear_dirty();

        assert!(reg.remove(0));
        assert_eq!(reg.count(), 1);
        assert!(reg.is_dirty());
        assert!(reg.get(0).is_none());
    }

    #[test]
    fn remove_nonexistent() {
        let mut reg = MaterialRegistry::new();
        assert!(!reg.remove(42));
        assert!(!reg.is_dirty());
    }

    #[test]
    fn dirty_flag_lifecycle() {
        let mut reg = MaterialRegistry::new();
        assert!(!reg.is_dirty());

        reg.register(0, "x", default_props());
        assert!(reg.is_dirty());

        reg.clear_dirty();
        assert!(!reg.is_dirty());

        reg.mark_dirty();
        assert!(reg.is_dirty());
    }

    #[test]
    fn all_definitions_returns_slice() {
        let mut reg = MaterialRegistry::new();
        reg.register(0, "a", default_props());
        reg.register(1, "b", default_props());
        assert_eq!(reg.all_definitions().len(), 2);
    }

    // -- Parse / Serialize roundtrip ------------------------------------------

    #[test]
    fn parse_serialize_roundtrip() {
        let file = MaterialFile {
            materials: vec![
                MaterialFileEntry {
                    id: 0,
                    name: "stone".into(),
                    properties: default_props(),
                },
                MaterialFileEntry {
                    id: 1,
                    name: "red_metal".into(),
                    properties: red_props(),
                },
            ],
        };

        let ron_str = serialize_material_file(&file).expect("serialize");
        let parsed = parse_material_file(&ron_str).expect("parse");
        assert_eq!(file, parsed);
    }

    #[test]
    fn parse_invalid_ron() {
        let result = parse_material_file("this is not valid RON {{{");
        assert!(result.is_err());
    }

    // -- Diff -----------------------------------------------------------------

    #[test]
    fn diff_identical_files_empty() {
        let file = MaterialFile {
            materials: vec![MaterialFileEntry {
                id: 0,
                name: "stone".into(),
                properties: default_props(),
            }],
        };
        let cs = diff_material_files(&file, &file);
        assert!(cs.is_empty());
        assert_eq!(cs.total_changes(), 0);
    }

    #[test]
    fn diff_detects_addition() {
        let old = MaterialFile {
            materials: vec![MaterialFileEntry {
                id: 0,
                name: "stone".into(),
                properties: default_props(),
            }],
        };
        let new = MaterialFile {
            materials: vec![
                MaterialFileEntry {
                    id: 0,
                    name: "stone".into(),
                    properties: default_props(),
                },
                MaterialFileEntry {
                    id: 1,
                    name: "metal".into(),
                    properties: red_props(),
                },
            ],
        };
        let cs = diff_material_files(&old, &new);
        assert_eq!(cs.added, vec![1]);
        assert!(cs.modified.is_empty());
        assert!(cs.removed.is_empty());
        assert_eq!(cs.total_changes(), 1);
    }

    #[test]
    fn diff_detects_removal() {
        let old = MaterialFile {
            materials: vec![
                MaterialFileEntry {
                    id: 0,
                    name: "stone".into(),
                    properties: default_props(),
                },
                MaterialFileEntry {
                    id: 1,
                    name: "metal".into(),
                    properties: red_props(),
                },
            ],
        };
        let new = MaterialFile {
            materials: vec![MaterialFileEntry {
                id: 0,
                name: "stone".into(),
                properties: default_props(),
            }],
        };
        let cs = diff_material_files(&old, &new);
        assert!(cs.added.is_empty());
        assert!(cs.modified.is_empty());
        assert_eq!(cs.removed, vec![1]);
    }

    #[test]
    fn diff_detects_modification() {
        let old = MaterialFile {
            materials: vec![MaterialFileEntry {
                id: 0,
                name: "stone".into(),
                properties: default_props(),
            }],
        };
        let new = MaterialFile {
            materials: vec![MaterialFileEntry {
                id: 0,
                name: "stone".into(),
                properties: red_props(),
            }],
        };
        let cs = diff_material_files(&old, &new);
        assert!(cs.added.is_empty());
        assert_eq!(cs.modified, vec![0]);
        assert!(cs.removed.is_empty());
    }

    #[test]
    fn diff_mixed_changes() {
        let old = MaterialFile {
            materials: vec![
                MaterialFileEntry {
                    id: 0,
                    name: "a".into(),
                    properties: default_props(),
                },
                MaterialFileEntry {
                    id: 1,
                    name: "b".into(),
                    properties: default_props(),
                },
            ],
        };
        let new = MaterialFile {
            materials: vec![
                MaterialFileEntry {
                    id: 0,
                    name: "a".into(),
                    properties: red_props(), // modified
                },
                MaterialFileEntry {
                    id: 2,
                    name: "c".into(),
                    properties: default_props(), // added
                },
            ],
        };
        let cs = diff_material_files(&old, &new);
        assert_eq!(cs.added, vec![2]);
        assert_eq!(cs.modified, vec![0]);
        assert_eq!(cs.removed, vec![1]);
        assert_eq!(cs.total_changes(), 3);
    }
}
