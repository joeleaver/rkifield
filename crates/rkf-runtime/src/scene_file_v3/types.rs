//! Core types for the v3 scene file format.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Scene file v3 — entity-centric format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV3 {
    /// Format version (always 3).
    pub version: u32,
    /// All entities in the scene.
    pub entities: Vec<EntityRecord>,
    /// Generic property bag for editor/subsystem state.
    ///
    /// Stores camera snapshot, environment, lights, etc. as RON strings.
    /// Not entity-centric — reserved for editor-specific state that doesn't
    /// map to ECS components.
    #[serde(default)]
    pub properties: std::collections::HashMap<String, String>,
}

impl SceneFileV3 {
    /// Create an empty v3 scene file.
    pub fn new() -> Self {
        Self {
            version: 3,
            entities: Vec::new(),
            properties: std::collections::HashMap::new(),
        }
    }
}

impl Default for SceneFileV3 {
    fn default() -> Self {
        Self::new()
    }
}

/// A single entity's serialized state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Persistent entity identity (survives save/load cycles).
    pub stable_id: Uuid,
    /// Parent entity reference (as UUID, resolved at load time).
    pub parent: Option<Uuid>,
    /// Component name -> RON-serialized component value.
    pub components: HashMap<String, String>,
}

impl EntityRecord {
    /// Create a new entity record with the given stable ID and no components.
    pub fn new(stable_id: Uuid) -> Self {
        Self {
            stable_id,
            parent: None,
            components: HashMap::new(),
        }
    }

    /// Insert a serializable component by name.
    pub fn insert_component<T: Serialize>(&mut self, name: &str, value: &T) -> Result<(), ron::Error> {
        let ron_str = ron::to_string(value)?;
        self.components.insert(name.to_string(), ron_str);
        Ok(())
    }

    /// Read a component by name and type.
    pub fn get_component<T: for<'de> Deserialize<'de>>(&self, name: &str) -> Option<Result<T, ron::error::SpannedError>> {
        self.components.get(name).map(|s| ron::from_str(s))
    }

    /// Check if a component exists by name.
    pub fn has_component(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }
}

/// Serialize a [`SceneFileV3`] to a RON string.
pub fn serialize_scene_v3(scene: &SceneFileV3) -> Result<String, ron::Error> {
    ron::ser::to_string_pretty(scene, ron::ser::PrettyConfig::default())
}

/// Deserialize a [`SceneFileV3`] from a RON string.
pub fn deserialize_scene_v3(ron_str: &str) -> Result<SceneFileV3, ron::error::SpannedError> {
    ron::from_str(ron_str)
}
