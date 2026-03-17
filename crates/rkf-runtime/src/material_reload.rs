//! Material hot-reload — RON serialization and change diffing.
//!
//! [`parse_material_file`] / [`serialize_material_file`] handle round-tripping
//! through RON, and [`diff_material_files`] computes a [`MaterialChangeSet`]
//! between two file versions.

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

    let old_map: std::collections::HashMap<u16, &MaterialFileEntry> =
        old.materials.iter().map(|e| (e.id, e)).collect();
    let new_map: std::collections::HashMap<u16, &MaterialFileEntry> =
        new.materials.iter().map(|e| (e.id, e)).collect();

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
                    properties: red_props(),
                },
                MaterialFileEntry {
                    id: 2,
                    name: "c".into(),
                    properties: default_props(),
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
