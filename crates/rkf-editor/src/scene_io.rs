//! Scene save/load data model for the RKIField editor.
//!
//! Defines the serializable scene file format (RON-based), recent files tracking,
//! and unsaved-changes state. This is the data layer only — no filesystem I/O.

#![allow(dead_code)]

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

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
    pub scale: f32,
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
                    scale: 1.0,
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
                    scale: 1.0,
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
                    scale: 1.0,
                    components: vec![],
                },
                SceneEntity {
                    entity_id: 2,
                    name: "Child".to_string(),
                    parent_id: Some(1),
                    position: Vec3::new(1.0, 2.0, 3.0),
                    rotation: Quat::from_rotation_y(1.57),
                    scale: 0.5,
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
                    scale: 1.0,
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
                scale: 1.0,
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
}
