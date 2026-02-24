//! Scene save/load data model for the RKIField editor.
//!
//! Defines the serializable scene file format (RON-based), recent files tracking,
//! and unsaved-changes state. This is the data layer only — no filesystem I/O.

#![allow(dead_code)]

use glam::{Quat, Vec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serde helper: deserialize scale as either a single f32 (uniform → Vec3::splat)
/// or a Vec3 tuple `(x, y, z)`. Serializes always as Vec3.
mod scale_serde {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Vec3, s: S) -> Result<S::Ok, S::Error> {
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec3, D::Error> {
        // Try Vec3 first, then fall back to f32 → Vec3::splat
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ScaleValue {
            Vec3(Vec3),
            Uniform(f32),
        }
        match ScaleValue::deserialize(d)? {
            ScaleValue::Vec3(v) => Ok(v),
            ScaleValue::Uniform(f) => Ok(Vec3::splat(f)),
        }
    }
}

/// A component attached to a scene entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentData {
    /// An SDF voxel object loaded from an asset file.
    SdfObject { asset_path: String },
    /// A light source.
    Light {
        light_type: String,
        color: [f32; 3],
        intensity: f32,
        range: f32,
    },
    /// An animated entity with a referenced animation asset.
    AnimatedEntity { animation_path: String },
    /// A physics rigid body.
    RigidBody { body_type: String, mass: f32 },
}

/// A single entity in the scene file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneEntity {
    pub entity_id: u64,
    pub name: String,
    pub parent_id: Option<u64>,
    pub position: Vec3,
    pub rotation: Quat,
    #[serde(with = "scale_serde")]
    pub scale: Vec3,
    pub components: Vec<ComponentData>,
}

/// The top-level scene file structure.
///
/// Serialized to/from RON format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SceneFile {
    /// Format version (currently 1).
    pub version: u32,
    /// Human-readable scene name.
    pub name: String,
    /// All entities in the scene.
    pub entities: Vec<SceneEntity>,
    /// Environment settings serialized as a RON string.
    pub environment_ron: String,
}

/// Serialize a scene to a RON string.
pub fn save_scene(scene: &SceneFile) -> Result<String, String> {
    let config = ron::ser::PrettyConfig::default();
    ron::ser::to_string_pretty(scene, config).map_err(|e| format!("RON serialization error: {e}"))
}

/// Deserialize a scene from a RON string.
pub fn load_scene(ron_str: &str) -> Result<SceneFile, String> {
    ron::from_str(ron_str).map_err(|e| format!("RON deserialization error: {e}"))
}

/// Load a scene file from disk and return the parsed `SceneFile`.
pub fn load_scene_from_path(path: &str) -> Result<SceneFile, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read scene file '{path}': {e}"))?;
    load_scene(&contents)
}

/// Save a scene file to disk.
pub fn save_scene_to_path(scene: &SceneFile, path: &str) -> Result<(), String> {
    let ron_str = save_scene(scene)?;
    std::fs::write(path, ron_str)
        .map_err(|e| format!("Failed to write scene file '{path}': {e}"))
}

// ─── v2 Scene I/O ─────────────────────────────────────────────────────────────

/// Save a v2 `rkf_core::scene::Scene` to a `.rkscene` file using the v2
/// `rkf_runtime::scene_file::SceneFile` format.
///
/// Each `SceneObject` is written as an `ObjectEntry`. If the root node is an
/// analytical primitive the `analytical` and `analytical_params` fields are
/// populated; otherwise `analytical` is left as `None` (asset-backed path
/// would be filled in by a richer serialiser).
///
/// Returns `Ok(())` on success, or an `anyhow::Error` on serialisation or
/// I/O failure.
pub fn save_v2_scene(scene: &rkf_core::scene::Scene, path: &str) -> anyhow::Result<()> {
    use rkf_core::scene_node::SdfSource;
    use rkf_runtime::scene_file::{ObjectEntry, SceneFile};

    let mut sf = SceneFile::new(&scene.name);

    for obj in &scene.root_objects {
        let pos = obj.world_position.local;
        let rot = obj.rotation;
        let (analytical, analytical_params) = match &obj.root_node.sdf_source {
            SdfSource::Analytical { primitive, material_id: _ } => {
                use rkf_core::scene_node::SdfPrimitive;
                let (name, params) = match primitive {
                    SdfPrimitive::Sphere { radius } => ("sphere", vec![*radius]),
                    SdfPrimitive::Box { half_extents } => {
                        ("box", vec![half_extents.x, half_extents.y, half_extents.z])
                    }
                    SdfPrimitive::Capsule { radius, half_height } => {
                        ("capsule", vec![*radius, *half_height])
                    }
                    SdfPrimitive::Torus { major_radius, minor_radius } => {
                        ("torus", vec![*major_radius, *minor_radius])
                    }
                    SdfPrimitive::Cylinder { radius, half_height } => {
                        ("cylinder", vec![*radius, *half_height])
                    }
                    SdfPrimitive::Plane { normal, distance } => {
                        ("plane", vec![normal.x, normal.y, normal.z, *distance])
                    }
                };
                (Some(name.to_string()), Some(params))
            }
            _ => (None, None),
        };

        sf.objects.push(ObjectEntry {
            name: obj.name.clone(),
            asset_path: None,
            position: [pos.x as f64, pos.y as f64, pos.z as f64],
            rotation: [rot.x, rot.y, rot.z, rot.w],
            scale: obj.scale,
            material_id: None,
            analytical,
            analytical_params,
            importance_bias: 1.0,
        });
    }

    rkf_runtime::scene_file::save_scene_file(path, &sf)?;
    Ok(())
}

/// Load a v2 `.rkscene` file and return the parsed `rkf_runtime::scene_file::SceneFile`.
///
/// The caller is responsible for converting the `ObjectEntry` list back into
/// `rkf_core::scene::SceneObject`s (e.g. by looking up `.rkf` assets or
/// reconstructing analytical primitives from `analytical` / `analytical_params`).
pub fn load_v2_scene(path: &str) -> anyhow::Result<rkf_runtime::scene_file::SceneFile> {
    rkf_runtime::scene_file::load_scene_file(path)
}

/// Compute the AABB for an analytical SDF primitive at a given world position and scale.
pub fn aabb_for_analytical(
    prim_name: &str,
    params: &[f32],
    position: Vec3,
    _rotation: Quat,
    scale: f32,
) -> rkf_core::Aabb {
    use rkf_core::Aabb;

    let half = match prim_name {
        "sphere" => {
            let r = params.first().copied().unwrap_or(1.0) * scale;
            Vec3::splat(r)
        }
        "box" => {
            let hx = params.first().copied().unwrap_or(1.0) * scale;
            let hy = params.get(1).copied().unwrap_or(1.0) * scale;
            let hz = params.get(2).copied().unwrap_or(1.0) * scale;
            Vec3::new(hx, hy, hz)
        }
        "capsule" => {
            let r = params.first().copied().unwrap_or(0.2) * scale;
            let hh = params.get(1).copied().unwrap_or(0.5) * scale;
            Vec3::new(r, r + hh, r)
        }
        "torus" => {
            let major = params.first().copied().unwrap_or(0.5) * scale;
            let minor = params.get(1).copied().unwrap_or(0.15) * scale;
            Vec3::new(major + minor, minor, major + minor)
        }
        "cylinder" => {
            let r = params.first().copied().unwrap_or(0.3) * scale;
            let hh = params.get(1).copied().unwrap_or(0.5) * scale;
            Vec3::new(r, hh, r)
        }
        "plane" => Vec3::splat(50.0),
        _ => Vec3::splat(1.0),
    };
    Aabb::new(position - half, position + half)
}

/// Reconstruct a v2 `Scene` from a loaded scene file.
///
/// Only analytical primitives are fully reconstructed; voxelized objects
/// (with `asset_path`) are represented as empty nodes (no brick map data).
pub fn reconstruct_v2_scene(
    sf: &rkf_runtime::scene_file::SceneFile,
) -> rkf_core::scene::Scene {
    use rkf_core::scene::{Scene, SceneObject};
    use rkf_core::scene_node::{SceneNode, SdfPrimitive};
    use rkf_core::WorldPosition;

    let mut scene = Scene::new(&sf.name);

    for entry in &sf.objects {
        let pos = Vec3::new(
            entry.position[0] as f32,
            entry.position[1] as f32,
            entry.position[2] as f32,
        );
        let rot = Quat::from_xyzw(
            entry.rotation[0],
            entry.rotation[1],
            entry.rotation[2],
            entry.rotation[3],
        );
        let scale = entry.scale;

        let root_node = if let (Some(analytical), Some(params)) =
            (&entry.analytical, &entry.analytical_params)
        {
            let material_id = entry.material_id.unwrap_or(0);
            let primitive = match analytical.as_str() {
                "sphere" => SdfPrimitive::Sphere {
                    radius: params.first().copied().unwrap_or(1.0),
                },
                "box" => SdfPrimitive::Box {
                    half_extents: Vec3::new(
                        params.first().copied().unwrap_or(1.0),
                        params.get(1).copied().unwrap_or(1.0),
                        params.get(2).copied().unwrap_or(1.0),
                    ),
                },
                "capsule" => SdfPrimitive::Capsule {
                    radius: params.first().copied().unwrap_or(0.2),
                    half_height: params.get(1).copied().unwrap_or(0.5),
                },
                "torus" => SdfPrimitive::Torus {
                    major_radius: params.first().copied().unwrap_or(0.5),
                    minor_radius: params.get(1).copied().unwrap_or(0.15),
                },
                "cylinder" => SdfPrimitive::Cylinder {
                    radius: params.first().copied().unwrap_or(0.3),
                    half_height: params.get(1).copied().unwrap_or(0.5),
                },
                "plane" => SdfPrimitive::Plane {
                    normal: Vec3::new(
                        params.first().copied().unwrap_or(0.0),
                        params.get(1).copied().unwrap_or(1.0),
                        params.get(2).copied().unwrap_or(0.0),
                    ),
                    distance: params.get(3).copied().unwrap_or(0.0),
                },
                _ => SdfPrimitive::Sphere { radius: 1.0 },
            };
            SceneNode::analytical(&entry.name, primitive, material_id)
        } else {
            SceneNode::new(&entry.name)
        };

        let aabb = if let (Some(analytical), Some(params)) =
            (&entry.analytical, &entry.analytical_params)
        {
            aabb_for_analytical(analytical, params, pos, rot, scale)
        } else {
            rkf_core::Aabb::new(pos - Vec3::splat(1.0), pos + Vec3::splat(1.0))
        };

        let obj = SceneObject {
            id: 0,
            name: entry.name.clone(),
            world_position: WorldPosition::new(glam::IVec3::ZERO, pos),
            rotation: rot,
            scale,
            root_node,
            aabb,
        };
        scene.add_object_full(obj);
    }

    scene
}

/// An entry in the recent files list.
#[derive(Debug, Clone)]
pub struct RecentFileEntry {
    pub path: String,
    pub name: String,
    pub timestamp_ms: u64,
}

/// Tracks recently opened scene files (max 10).
#[derive(Debug)]
pub struct RecentFiles {
    entries: Vec<RecentFileEntry>,
}

const MAX_RECENT: usize = 10;

impl Default for RecentFiles {
    fn default() -> Self {
        Self::new()
    }
}

impl RecentFiles {
    /// Create an empty recent files list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a file to the recent list.
    ///
    /// If the path already exists, it is moved to the front with updated timestamp.
    /// If the list exceeds 10 entries, the oldest is dropped.
    pub fn add(&mut self, path: &str, name: &str, timestamp_ms: u64) {
        // Remove existing entry with the same path.
        self.entries.retain(|e| e.path != path);

        // Insert at the front (most recent first).
        self.entries.insert(
            0,
            RecentFileEntry {
                path: path.to_string(),
                name: name.to_string(),
                timestamp_ms,
            },
        );

        // Enforce max size.
        if self.entries.len() > MAX_RECENT {
            self.entries.truncate(MAX_RECENT);
        }
    }

    /// Remove a file from the recent list by path.
    pub fn remove(&mut self, path: &str) {
        self.entries.retain(|e| e.path != path);
    }

    /// Get the recent files list (most recent first).
    pub fn entries(&self) -> &[RecentFileEntry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Tracks whether the current scene has unsaved modifications.
#[derive(Debug, Clone)]
pub struct UnsavedChangesState {
    pub has_unsaved: bool,
}

impl Default for UnsavedChangesState {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsavedChangesState {
    /// Create a new state with no unsaved changes.
    pub fn new() -> Self {
        Self { has_unsaved: false }
    }

    /// Mark the scene as having unsaved changes.
    pub fn mark_changed(&mut self) {
        self.has_unsaved = true;
    }

    /// Mark the scene as saved (clears the dirty flag).
    pub fn mark_saved(&mut self) {
        self.has_unsaved = false;
    }

    /// Whether the scene needs saving.
    pub fn needs_save(&self) -> bool {
        self.has_unsaved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_scene() -> SceneFile {
        SceneFile {
            version: 1,
            name: "Test Scene".to_string(),
            entities: vec![
                SceneEntity {
                    entity_id: 1,
                    name: "Ground".to_string(),
                    parent_id: None,
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                    components: vec![ComponentData::SdfObject {
                        asset_path: "assets/ground.rkf".to_string(),
                    }],
                },
                SceneEntity {
                    entity_id: 2,
                    name: "Sun".to_string(),
                    parent_id: None,
                    position: Vec3::new(0.0, 10.0, 0.0),
                    rotation: Quat::from_rotation_x(-0.5),
                    scale: Vec3::ONE,
                    components: vec![ComponentData::Light {
                        light_type: "directional".to_string(),
                        color: [1.0, 0.95, 0.8],
                        intensity: 3.0,
                        range: 100.0,
                    }],
                },
            ],
            environment_ron: "(fog_density: 0.01, sky_color: (0.4, 0.6, 0.9))".to_string(),
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let scene = make_test_scene();
        let ron_str = save_scene(&scene).expect("serialize");
        let loaded = load_scene(&ron_str).expect("deserialize");
        assert_eq!(scene, loaded);
    }

    #[test]
    fn test_version_field_preserved() {
        let scene = SceneFile {
            version: 1,
            name: "V1 Scene".to_string(),
            entities: vec![],
            environment_ron: "()".to_string(),
        };
        let ron_str = save_scene(&scene).expect("serialize");
        let loaded = load_scene(&ron_str).expect("deserialize");
        assert_eq!(loaded.version, 1);
    }

    #[test]
    fn test_scene_with_multiple_entities() {
        let scene = SceneFile {
            version: 1,
            name: "Multi".to_string(),
            entities: vec![
                SceneEntity {
                    entity_id: 1,
                    name: "Root".to_string(),
                    parent_id: None,
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                    components: vec![],
                },
                SceneEntity {
                    entity_id: 2,
                    name: "Child".to_string(),
                    parent_id: Some(1),
                    position: Vec3::new(1.0, 2.0, 3.0),
                    rotation: Quat::from_rotation_y(1.57),
                    scale: Vec3::splat(0.5),
                    components: vec![
                        ComponentData::SdfObject {
                            asset_path: "rock.rkf".to_string(),
                        },
                        ComponentData::RigidBody {
                            body_type: "dynamic".to_string(),
                            mass: 5.0,
                        },
                    ],
                },
                SceneEntity {
                    entity_id: 3,
                    name: "Animated".to_string(),
                    parent_id: Some(1),
                    position: Vec3::Y,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                    components: vec![ComponentData::AnimatedEntity {
                        animation_path: "walk.rkanim".to_string(),
                    }],
                },
            ],
            environment_ron: "()".to_string(),
        };

        let ron_str = save_scene(&scene).expect("serialize");
        let loaded = load_scene(&ron_str).expect("deserialize");
        assert_eq!(loaded.entities.len(), 3);
        assert_eq!(loaded.entities[1].parent_id, Some(1));
        assert_eq!(loaded.entities[1].components.len(), 2);
        assert_eq!(loaded.entities[2].name, "Animated");
    }

    #[test]
    fn test_all_component_types_roundtrip() {
        let scene = SceneFile {
            version: 1,
            name: "Components".to_string(),
            entities: vec![SceneEntity {
                entity_id: 1,
                name: "Everything".to_string(),
                parent_id: None,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: vec![
                    ComponentData::SdfObject {
                        asset_path: "mesh.rkf".to_string(),
                    },
                    ComponentData::Light {
                        light_type: "point".to_string(),
                        color: [1.0, 0.5, 0.0],
                        intensity: 10.0,
                        range: 25.0,
                    },
                    ComponentData::AnimatedEntity {
                        animation_path: "idle.rkanim".to_string(),
                    },
                    ComponentData::RigidBody {
                        body_type: "static".to_string(),
                        mass: 0.0,
                    },
                ],
            }],
            environment_ron: "()".to_string(),
        };

        let ron_str = save_scene(&scene).expect("serialize");
        let loaded = load_scene(&ron_str).expect("deserialize");
        assert_eq!(loaded.entities[0].components.len(), 4);
        assert_eq!(scene, loaded);
    }

    #[test]
    fn test_load_invalid_ron() {
        let result = load_scene("this is not valid RON {{{}}}");
        assert!(result.is_err());
    }

    #[test]
    fn test_recent_files_add_and_entries() {
        let mut recent = RecentFiles::new();
        assert!(recent.is_empty());

        recent.add("/scenes/a.rkscene", "Scene A", 1000);
        recent.add("/scenes/b.rkscene", "Scene B", 2000);

        assert_eq!(recent.len(), 2);
        // Most recent first
        assert_eq!(recent.entries()[0].name, "Scene B");
        assert_eq!(recent.entries()[1].name, "Scene A");
    }

    #[test]
    fn test_recent_files_dedup_on_add() {
        let mut recent = RecentFiles::new();
        recent.add("/scenes/a.rkscene", "Scene A", 1000);
        recent.add("/scenes/b.rkscene", "Scene B", 2000);
        // Re-add A with new timestamp — should move to front
        recent.add("/scenes/a.rkscene", "Scene A Updated", 3000);

        assert_eq!(recent.len(), 2);
        assert_eq!(recent.entries()[0].name, "Scene A Updated");
        assert_eq!(recent.entries()[0].timestamp_ms, 3000);
    }

    #[test]
    fn test_recent_files_max_10() {
        let mut recent = RecentFiles::new();
        for i in 0..15 {
            recent.add(&format!("/scenes/{i}.rkscene"), &format!("Scene {i}"), i as u64);
        }
        assert_eq!(recent.len(), 10);
        // Most recent should be 14
        assert_eq!(recent.entries()[0].name, "Scene 14");
    }

    #[test]
    fn test_recent_files_remove() {
        let mut recent = RecentFiles::new();
        recent.add("/a.rkscene", "A", 100);
        recent.add("/b.rkscene", "B", 200);
        recent.add("/c.rkscene", "C", 300);

        recent.remove("/b.rkscene");
        assert_eq!(recent.len(), 2);
        assert_eq!(recent.entries()[0].name, "C");
        assert_eq!(recent.entries()[1].name, "A");
    }

    #[test]
    fn test_recent_files_remove_nonexistent() {
        let mut recent = RecentFiles::new();
        recent.add("/a.rkscene", "A", 100);
        recent.remove("/nonexistent.rkscene");
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_unsaved_changes_state() {
        let mut state = UnsavedChangesState::new();
        assert!(!state.needs_save());

        state.mark_changed();
        assert!(state.needs_save());

        state.mark_saved();
        assert!(!state.needs_save());
    }

    #[test]
    fn test_unsaved_changes_default() {
        let state = UnsavedChangesState::default();
        assert!(!state.needs_save());
    }

    #[test]
    fn test_empty_scene_roundtrip() {
        let scene = SceneFile {
            version: 1,
            name: "Empty".to_string(),
            entities: vec![],
            environment_ron: String::new(),
        };
        let ron_str = save_scene(&scene).expect("serialize");
        let loaded = load_scene(&ron_str).expect("deserialize");
        assert_eq!(loaded.name, "Empty");
        assert!(loaded.entities.is_empty());
        assert_eq!(loaded.version, 1);
    }

    #[test]
    fn test_recent_files_default() {
        let recent = RecentFiles::default();
        assert!(recent.is_empty());
    }

    // ── v2 Scene I/O tests ─────────────────────────────────────────────────────

    #[test]
    fn test_save_v2_scene_creates_file() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("v2test");
        let node = SceneNode::analytical("sphere", SdfPrimitive::Sphere { radius: 1.0 }, 0);
        scene.add_object("SphereObj", WorldPosition::default(), node);

        let path = "/tmp/rkf_v2_save_test.rkscene";
        super::save_v2_scene(&scene, path).expect("save_v2_scene should succeed");

        assert!(
            std::path::Path::new(path).exists(),
            "file should have been written"
        );
    }

    #[test]
    fn test_load_v2_scene_roundtrip() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("roundtrip");
        let node = SceneNode::analytical("box", SdfPrimitive::Box { half_extents: glam::Vec3::ONE }, 3);
        scene.add_object("BoxObj", WorldPosition::default(), node);

        let path = "/tmp/rkf_v2_roundtrip_test.rkscene";
        super::save_v2_scene(&scene, path).expect("save");

        let loaded = super::load_v2_scene(path).expect("load");
        assert_eq!(loaded.name, "roundtrip");
        assert_eq!(loaded.objects.len(), 1);
        assert_eq!(loaded.objects[0].name, "BoxObj");
        assert_eq!(loaded.objects[0].analytical.as_deref(), Some("box"));
        let params = loaded.objects[0].analytical_params.as_ref().unwrap();
        assert_eq!(params.len(), 3);
        // half_extents = (1, 1, 1)
        assert!((params[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_load_v2_scene_nonexistent_path() {
        let result = super::load_v2_scene("/nonexistent/path/scene.rkscene");
        assert!(result.is_err());
    }

    #[test]
    fn test_save_v2_scene_multiple_objects() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("multi");
        scene.add_object(
            "Sphere",
            WorldPosition::default(),
            SceneNode::analytical("s", SdfPrimitive::Sphere { radius: 0.5 }, 1),
        );
        scene.add_object(
            "Capsule",
            WorldPosition::default(),
            SceneNode::analytical("c", SdfPrimitive::Capsule { radius: 0.2, half_height: 0.8 }, 2),
        );

        let path = "/tmp/rkf_v2_multi_test.rkscene";
        super::save_v2_scene(&scene, path).expect("save");

        let loaded = super::load_v2_scene(path).expect("load");
        assert_eq!(loaded.objects.len(), 2);
        assert_eq!(loaded.objects[0].name, "Sphere");
        assert_eq!(loaded.objects[1].name, "Capsule");
    }
}
