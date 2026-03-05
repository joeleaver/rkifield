//! Material library — file-driven material management with hot-reload support.
//!
//! Materials are individual `.rkmat` files (RON format), one per material.
//! A `.rkmatlib` palette file maps GPU table slots to `.rkmat` file paths.
//!
//! # File Formats
//!
//! ## `.rkmat` (single material)
//! ```ron
//! MaterialEntry(
//!     name: "Stone",
//!     description: "Gray rough dielectric",
//!     category: "Stone",
//!     properties: ( albedo: (0.45, 0.43, 0.40), roughness: 0.85, ... ),
//! )
//! ```
//!
//! ## `.rkmatlib` (palette)
//! ```ron
//! MaterialPalette(
//!     name: "Default",
//!     slots: [ (index: 0, path: "default.rkmat"), (index: 1, path: "stone.rkmat"), ... ],
//! )
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::material::Material;

// ── MaterialProperties ──────────────────────────────────────────────────────

/// Serializable PBR material properties — all meaningful fields from the
/// 96-byte GPU [`Material`] struct.
///
/// Excludes `_padding` (always zero) since it has no semantic meaning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialProperties {
    /// Base color (linear RGB, 0–1).
    pub albedo: [f32; 3],
    /// Roughness: 0.0 = mirror, 1.0 = fully rough.
    pub roughness: f32,
    /// Metallic factor: 0.0 = dielectric, 1.0 = metal.
    pub metallic: f32,
    /// Emissive color (linear RGB).
    pub emission_color: [f32; 3],
    /// Emissive intensity (HDR, can exceed 1.0).
    pub emission_strength: f32,
    /// Subsurface scattering strength: 0.0 = none, 1.0 = full SSS.
    pub subsurface: f32,
    /// Color of scattered light (skin, wax, leaves).
    pub subsurface_color: [f32; 3],
    /// Opacity: 1.0 = solid, 0.0 = fully transparent.
    pub opacity: f32,
    /// Index of refraction (glass ~1.5, water ~1.33).
    pub ior: f32,
    /// Spatial frequency of procedural noise.
    pub noise_scale: f32,
    /// Amplitude of noise perturbation.
    pub noise_strength: f32,
    /// Noise channel bitfield: bit 0 = albedo, bit 1 = roughness, bit 2 = normal.
    pub noise_channels: u32,
    /// Shader model name (e.g. "pbr", "toon", "unlit"). Resolved to a numeric
    /// `shader_id` on the GPU Material struct via `MaterialLibrary::resolve_shader_ids()`.
    #[serde(default = "default_shader")]
    pub shader: String,
}

fn default_shader() -> String {
    "pbr".to_string()
}

impl Default for MaterialProperties {
    fn default() -> Self {
        let m = Material::default();
        Self::from(m)
    }
}

impl From<Material> for MaterialProperties {
    fn from(m: Material) -> Self {
        Self {
            albedo: m.albedo,
            roughness: m.roughness,
            metallic: m.metallic,
            emission_color: m.emission_color,
            emission_strength: m.emission_strength,
            subsurface: m.subsurface,
            subsurface_color: m.subsurface_color,
            opacity: m.opacity,
            ior: m.ior,
            noise_scale: m.noise_scale,
            noise_strength: m.noise_strength,
            noise_channels: m.noise_channels,
            shader: "pbr".to_string(),
        }
    }
}

impl From<MaterialProperties> for Material {
    fn from(p: MaterialProperties) -> Self {
        Material {
            albedo: p.albedo,
            roughness: p.roughness,
            metallic: p.metallic,
            emission_color: p.emission_color,
            emission_strength: p.emission_strength,
            subsurface: p.subsurface,
            subsurface_color: p.subsurface_color,
            opacity: p.opacity,
            ior: p.ior,
            noise_scale: p.noise_scale,
            noise_strength: p.noise_strength,
            noise_channels: p.noise_channels,
            shader_id: 0, // resolved later via resolve_shader_ids()
            _padding: [0.0; 5],
        }
    }
}

impl From<&Material> for MaterialProperties {
    fn from(m: &Material) -> Self {
        Self::from(*m)
    }
}

// ── MaterialEntry ───────────────────────────────────────────────────────────

/// A single material definition as stored in a `.rkmat` file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialEntry {
    /// Human-readable name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Category for browsing (e.g. "Metal", "Stone", "Organic").
    pub category: String,
    /// PBR material properties.
    pub properties: MaterialProperties,
}

/// Load a [`MaterialEntry`] from a `.rkmat` file (RON format).
pub fn load_material_entry(path: &Path) -> Result<MaterialEntry, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    ron::from_str(&contents)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Save a [`MaterialEntry`] to a `.rkmat` file (RON format).
pub fn save_material_entry(path: &Path, entry: &MaterialEntry) -> Result<(), String> {
    let config = ron::ser::PrettyConfig::default()
        .struct_names(true);
    let contents = ron::ser::to_string_pretty(entry, config)
        .map_err(|e| format!("serialize: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path, contents)
        .map_err(|e| format!("write {}: {e}", path.display()))
}

// ── MaterialPalette ─────────────────────────────────────────────────────────

/// A single slot assignment in a material palette.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PaletteSlot {
    /// GPU material table index (0–65535).
    pub index: u16,
    /// Relative path to the `.rkmat` file.
    pub path: String,
}

/// Maps GPU material table slots to `.rkmat` files.
///
/// Stored as a `.rkmatlib` file (RON format). This is what scenes reference
/// to establish which materials are loaded at which indices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaterialPalette {
    /// Palette name.
    pub name: String,
    /// Slot assignments — each maps a table index to a `.rkmat` path.
    pub slots: Vec<PaletteSlot>,
}

/// Load a [`MaterialPalette`] from a `.rkmatlib` file (RON format).
pub fn load_palette(path: &Path) -> Result<MaterialPalette, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    ron::from_str(&contents)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Save a [`MaterialPalette`] to a `.rkmatlib` file (RON format).
pub fn save_palette(path: &Path, palette: &MaterialPalette) -> Result<(), String> {
    let config = ron::ser::PrettyConfig::default()
        .struct_names(true);
    let contents = ron::ser::to_string_pretty(palette, config)
        .map_err(|e| format!("serialize: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path, contents)
        .map_err(|e| format!("write {}: {e}", path.display()))
}

// ── SlotInfo ────────────────────────────────────────────────────────────────

/// Metadata about a material slot in the library.
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// Material name.
    pub name: String,
    /// Material description.
    pub description: String,
    /// Material category.
    pub category: String,
    /// Relative file path (from palette).
    pub file_path: String,
    /// Shader model name (e.g. "pbr", "toon"). Used by `resolve_shader_ids()`.
    pub shader_name: String,
}

// ── MaterialLibrary ─────────────────────────────────────────────────────────

/// In-memory material library — manages GPU-ready materials with file backing.
///
/// The library holds a flat array of [`Material`] structs indexed by slot number,
/// along with metadata for each occupied slot. A dirty flag tracks whether the
/// GPU material buffer needs re-uploading.
pub struct MaterialLibrary {
    /// GPU-ready materials indexed by slot.
    materials: Vec<Material>,
    /// Metadata per slot (None for unoccupied slots).
    slots: Vec<Option<SlotInfo>>,
    /// Whether any material has changed since last `clear_dirty()`.
    dirty: bool,
    /// Base directory for resolving relative `.rkmat` paths.
    base_dir: Option<std::path::PathBuf>,
}

impl MaterialLibrary {
    /// Create an empty library with the given number of slots.
    pub fn new(slot_count: usize) -> Self {
        Self {
            materials: vec![Material::default(); slot_count],
            slots: vec![None; slot_count],
            dirty: false,
            base_dir: None,
        }
    }

    /// Load a palette file, resolving `.rkmat` paths relative to the palette's directory.
    pub fn load_palette(path: &Path) -> Result<Self, String> {
        let palette = load_palette(path)?;
        let base_dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();

        // Find the max slot index to size the arrays.
        let max_slot = palette.slots.iter().map(|s| s.index as usize).max().unwrap_or(0);
        let slot_count = max_slot + 1;

        let mut lib = Self::new(slot_count.max(16)); // at least 16 slots
        lib.base_dir = Some(base_dir.clone());

        for ps in &palette.slots {
            let mat_path = base_dir.join(&ps.path);
            match load_material_entry(&mat_path) {
                Ok(entry) => {
                    let idx = ps.index as usize;
                    lib.materials[idx] = Material::from(entry.properties.clone());
                    let shader_name = entry.properties.shader.clone();
                    lib.slots[idx] = Some(SlotInfo {
                        name: entry.name,
                        description: entry.description,
                        category: entry.category,
                        file_path: ps.path.clone(),
                        shader_name,
                    });
                }
                Err(e) => {
                    log::warn!("Failed to load material at slot {}: {e}", ps.index);
                }
            }
        }

        lib.dirty = true;
        Ok(lib)
    }

    /// Load a single material file into the given slot.
    pub fn load_material(&mut self, slot: u16, path: &Path) -> Result<(), String> {
        let entry = load_material_entry(path)?;
        let idx = slot as usize;
        self.ensure_slot(idx);
        let shader_name = entry.properties.shader.clone();
        self.materials[idx] = Material::from(entry.properties.clone());
        self.slots[idx] = Some(SlotInfo {
            name: entry.name,
            description: entry.description,
            category: entry.category,
            file_path: path.to_string_lossy().into_owned(),
            shader_name,
        });
        self.dirty = true;
        Ok(())
    }

    /// Set a material directly (GPU struct), marking the library dirty.
    pub fn set_material(&mut self, slot: u16, material: Material) {
        let idx = slot as usize;
        self.ensure_slot(idx);
        self.materials[idx] = material;
        self.dirty = true;
    }

    /// Set material with metadata.
    pub fn set_material_with_info(&mut self, slot: u16, material: Material, info: SlotInfo) {
        let idx = slot as usize;
        self.ensure_slot(idx);
        self.materials[idx] = material;
        self.slots[idx] = Some(info);
        self.dirty = true;
    }

    /// Get the GPU material at a slot.
    pub fn get_material(&self, slot: u16) -> Option<&Material> {
        self.materials.get(slot as usize)
    }

    /// Get a mutable reference to the GPU material at a slot.
    pub fn get_material_mut(&mut self, slot: u16) -> Option<&mut Material> {
        let idx = slot as usize;
        if idx < self.materials.len() {
            self.dirty = true;
            Some(&mut self.materials[idx])
        } else {
            None
        }
    }

    /// Get metadata for a slot.
    pub fn slot_info(&self, slot: u16) -> Option<&SlotInfo> {
        self.slots.get(slot as usize).and_then(|s| s.as_ref())
    }

    /// Get mutable metadata for a slot.
    pub fn slot_info_mut(&mut self, slot: u16) -> Option<&mut SlotInfo> {
        self.slots.get_mut(slot as usize).and_then(|s| s.as_mut())
    }

    /// Reload a material from its file path.
    ///
    /// Used by the file watcher when a `.rkmat` file changes on disk.
    pub fn reload_material(&mut self, file_path: &Path) -> Result<(), String> {
        // Find which slot(s) use this file path.
        let file_str = file_path.to_string_lossy();
        let base = self.base_dir.clone();
        for idx in 0..self.slots.len() {
            if let Some(ref info) = self.slots[idx] {
                let matches = if let Some(ref base_dir) = base {
                    let resolved = base_dir.join(&info.file_path);
                    // Compare canonicalized paths to handle relative vs absolute.
                    match (resolved.canonicalize(), file_path.canonicalize()) {
                        (Ok(a), Ok(b)) => a == b,
                        _ => resolved == file_path,
                    }
                } else {
                    info.file_path == file_str.as_ref()
                };
                if matches {
                    let entry = load_material_entry(file_path)?;
                    let shader_name = entry.properties.shader.clone();
                    self.materials[idx] = Material::from(entry.properties.clone());
                    self.slots[idx] = Some(SlotInfo {
                        name: entry.name,
                        description: entry.description,
                        category: entry.category,
                        file_path: info.file_path.clone(),
                        shader_name,
                    });
                    self.dirty = true;
                    log::info!("Hot-reloaded material at slot {idx} from {}", file_path.display());
                }
            }
        }
        Ok(())
    }

    /// All GPU-ready materials as a slice (for uploading to GPU).
    pub fn all_materials(&self) -> &[Material] {
        &self.materials
    }

    /// Number of slots.
    pub fn slot_count(&self) -> usize {
        self.materials.len()
    }

    /// Whether any material has been modified since last `clear_dirty()`.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear the dirty flag (call after GPU upload).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Mark the library as dirty (force re-upload).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Save the current palette to a `.rkmatlib` file.
    pub fn save_palette(&self, path: &Path) -> Result<(), String> {
        let slots: Vec<PaletteSlot> = self.slots.iter().enumerate()
            .filter_map(|(idx, info)| {
                info.as_ref().map(|si| PaletteSlot {
                    index: idx as u16,
                    path: si.file_path.clone(),
                })
            })
            .collect();
        let palette = MaterialPalette {
            name: path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unnamed".into()),
            slots,
        };
        save_palette(path, &palette)
    }

    /// Save a single material slot to its `.rkmat` file.
    pub fn save_material(&self, slot: u16) -> Result<(), String> {
        let idx = slot as usize;
        let info = self.slots.get(idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| format!("no material at slot {slot}"))?;
        let material = &self.materials[idx];
        let entry = MaterialEntry {
            name: info.name.clone(),
            description: info.description.clone(),
            category: info.category.clone(),
            properties: MaterialProperties::from(material),
        };
        let file_path = if let Some(ref base) = self.base_dir {
            base.join(&info.file_path)
        } else {
            std::path::PathBuf::from(&info.file_path)
        };
        save_material_entry(&file_path, &entry)
    }

    /// Ensure the library has enough slots for the given index.
    fn ensure_slot(&mut self, idx: usize) {
        if idx >= self.materials.len() {
            self.materials.resize(idx + 1, Material::default());
            self.slots.resize(idx + 1, None);
        }
    }

    /// Resolve shader names to numeric IDs for all occupied slots.
    ///
    /// Call after loading materials and after any shader registry change.
    /// The `lookup` closure maps a shader name (e.g. "pbr") to its numeric ID.
    pub fn resolve_shader_ids(&mut self, lookup: impl Fn(&str) -> u32) {
        for idx in 0..self.slots.len() {
            if let Some(ref info) = self.slots[idx] {
                self.materials[idx].shader_id = lookup(&info.shader_name);
            }
        }
        self.dirty = true;
    }

    /// The base directory used for resolving relative paths.
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }

    /// Iterate over all occupied slots with their indices.
    pub fn occupied_slots(&self) -> impl Iterator<Item = (u16, &Material, &SlotInfo)> {
        self.slots.iter().enumerate()
            .filter_map(move |(idx, info)| {
                info.as_ref().map(|si| (idx as u16, &self.materials[idx], si))
            })
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_properties_default_matches_material_default() {
        let props = MaterialProperties::default();
        let mat = Material::default();
        assert_eq!(props.albedo, mat.albedo);
        assert_eq!(props.roughness, mat.roughness);
        assert_eq!(props.metallic, mat.metallic);
        assert_eq!(props.emission_color, mat.emission_color);
        assert_eq!(props.emission_strength, mat.emission_strength);
        assert_eq!(props.subsurface, mat.subsurface);
        assert_eq!(props.subsurface_color, mat.subsurface_color);
        assert_eq!(props.opacity, mat.opacity);
        assert_eq!(props.ior, mat.ior);
        assert_eq!(props.noise_scale, mat.noise_scale);
        assert_eq!(props.noise_strength, mat.noise_strength);
        assert_eq!(props.noise_channels, mat.noise_channels);
    }

    #[test]
    fn material_to_properties_roundtrip() {
        let mat = Material {
            albedo: [1.0, 0.0, 0.0],
            roughness: 0.3,
            metallic: 1.0,
            emission_color: [0.5, 0.5, 0.0],
            emission_strength: 2.0,
            subsurface: 0.8,
            subsurface_color: [1.0, 0.5, 0.25],
            opacity: 0.9,
            ior: 1.33,
            noise_scale: 5.0,
            noise_strength: 0.3,
            noise_channels: 3,
            shader_id: 0,
            _padding: [0.0; 5],
        };
        let props = MaterialProperties::from(mat);
        let back = Material::from(props);
        assert_eq!(back.albedo, mat.albedo);
        assert_eq!(back.roughness, mat.roughness);
        assert_eq!(back.metallic, mat.metallic);
        assert_eq!(back.emission_color, mat.emission_color);
        assert_eq!(back.emission_strength, mat.emission_strength);
        assert_eq!(back.subsurface, mat.subsurface);
        assert_eq!(back.subsurface_color, mat.subsurface_color);
        assert_eq!(back.opacity, mat.opacity);
        assert_eq!(back.ior, mat.ior);
        assert_eq!(back.noise_scale, mat.noise_scale);
        assert_eq!(back.noise_strength, mat.noise_strength);
        assert_eq!(back.noise_channels, mat.noise_channels);
        assert_eq!(back._padding, [0.0; 5]);
    }

    #[test]
    fn material_properties_ron_roundtrip() {
        let props = MaterialProperties {
            albedo: [0.45, 0.43, 0.40],
            roughness: 0.85,
            metallic: 0.0,
            emission_color: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0, 0.8, 0.6],
            opacity: 1.0,
            ior: 1.5,
            noise_scale: 0.0,
            noise_strength: 0.0,
            noise_channels: 0,
            shader: "pbr".into(),
        };
        let ron_str = ron::to_string(&props).unwrap();
        let back: MaterialProperties = ron::from_str(&ron_str).unwrap();
        assert_eq!(props, back);
    }

    #[test]
    fn material_entry_ron_roundtrip() {
        let entry = MaterialEntry {
            name: "Stone".into(),
            description: "Gray rough dielectric".into(),
            category: "Stone".into(),
            properties: MaterialProperties {
                albedo: [0.45, 0.43, 0.40],
                roughness: 0.85,
                ..Default::default()
            },
        };
        let config = ron::ser::PrettyConfig::default().struct_names(true);
        let ron_str = ron::ser::to_string_pretty(&entry, config).unwrap();
        let back: MaterialEntry = ron::from_str(&ron_str).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn palette_ron_roundtrip() {
        let palette = MaterialPalette {
            name: "Default".into(),
            slots: vec![
                PaletteSlot { index: 0, path: "default.rkmat".into() },
                PaletteSlot { index: 1, path: "stone.rkmat".into() },
            ],
        };
        let config = ron::ser::PrettyConfig::default().struct_names(true);
        let ron_str = ron::ser::to_string_pretty(&palette, config).unwrap();
        let back: MaterialPalette = ron::from_str(&ron_str).unwrap();
        assert_eq!(palette, back);
    }

    #[test]
    fn library_new_has_default_materials() {
        let lib = MaterialLibrary::new(14);
        assert_eq!(lib.slot_count(), 14);
        assert!(!lib.is_dirty());
        let m = lib.get_material(0).unwrap();
        assert_eq!(m.albedo, Material::default().albedo);
    }

    #[test]
    fn library_set_material_marks_dirty() {
        let mut lib = MaterialLibrary::new(4);
        assert!(!lib.is_dirty());
        lib.set_material(0, Material { albedo: [1.0, 0.0, 0.0], ..Default::default() });
        assert!(lib.is_dirty());
        lib.clear_dirty();
        assert!(!lib.is_dirty());
    }

    #[test]
    fn library_set_material_grows_if_needed() {
        let mut lib = MaterialLibrary::new(2);
        lib.set_material(10, Material::default());
        assert!(lib.slot_count() >= 11);
        assert!(lib.get_material(10).is_some());
    }

    #[test]
    fn library_set_with_info() {
        let mut lib = MaterialLibrary::new(4);
        let info = SlotInfo {
            name: "Stone".into(),
            description: "Gray rough".into(),
            category: "Stone".into(),
            file_path: "stone.rkmat".into(),
            shader_name: "pbr".into(),
        };
        lib.set_material_with_info(1, Material::default(), info);
        assert!(lib.slot_info(1).is_some());
        assert_eq!(lib.slot_info(1).unwrap().name, "Stone");
    }

    #[test]
    fn library_occupied_slots() {
        let mut lib = MaterialLibrary::new(4);
        let info = SlotInfo {
            name: "A".into(),
            description: "".into(),
            category: "".into(),
            file_path: "a.rkmat".into(),
            shader_name: "pbr".into(),
        };
        lib.set_material_with_info(1, Material::default(), info);
        let occupied: Vec<_> = lib.occupied_slots().collect();
        assert_eq!(occupied.len(), 1);
        assert_eq!(occupied[0].0, 1);
    }

    #[test]
    fn library_dirty_lifecycle() {
        let mut lib = MaterialLibrary::new(4);
        assert!(!lib.is_dirty());
        lib.mark_dirty();
        assert!(lib.is_dirty());
        lib.clear_dirty();
        assert!(!lib.is_dirty());
        lib.set_material(0, Material::default());
        assert!(lib.is_dirty());
    }

    #[test]
    fn material_entry_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("rkf_mat_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_stone.rkmat");

        let entry = MaterialEntry {
            name: "Stone".into(),
            description: "Gray rough dielectric".into(),
            category: "Stone".into(),
            properties: MaterialProperties {
                albedo: [0.45, 0.43, 0.40],
                roughness: 0.85,
                shader: "pbr".into(),
                ..Default::default()
            },
        };

        save_material_entry(&path, &entry).unwrap();
        let loaded = load_material_entry(&path).unwrap();
        assert_eq!(entry, loaded);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn palette_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("rkf_palette_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.rkmatlib");

        let palette = MaterialPalette {
            name: "Test".into(),
            slots: vec![
                PaletteSlot { index: 0, path: "default.rkmat".into() },
                PaletteSlot { index: 5, path: "stone.rkmat".into() },
            ],
        };

        save_palette(&path, &palette).unwrap();
        let loaded = load_palette(&path).unwrap();
        assert_eq!(palette, loaded);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn library_get_material_mut_marks_dirty() {
        let mut lib = MaterialLibrary::new(4);
        lib.clear_dirty();
        let m = lib.get_material_mut(0).unwrap();
        m.albedo = [1.0, 0.0, 0.0];
        assert!(lib.is_dirty());
    }

    #[test]
    fn library_all_materials_slice() {
        let lib = MaterialLibrary::new(14);
        let all = lib.all_materials();
        assert_eq!(all.len(), 14);
    }
}
