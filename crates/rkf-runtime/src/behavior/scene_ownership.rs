//! Scene ownership — tracks which scene spawned an entity for lifecycle management.
//!
//! When a scene is unloaded, all entities tagged with that scene are despawned.
//! Persistent entities (`scene: None`) survive scene transitions.

use serde::{Deserialize, Serialize};

/// Tracks which scene owns an entity.
///
/// - `scene: Some("level_01")` — entity belongs to level_01, despawned when level_01 unloads
/// - `scene: None` — persistent entity, survives all scene transitions (player, HUD, music)
///
/// Set automatically by the CommandQueue at flush time based on which spawn variant was used:
/// - `spawn()` → current scene
/// - `spawn_persistent()` → None
/// - `spawn_in_scene(scene)` → explicit scene
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneOwnership {
    /// The scene that owns this entity. `None` means persistent.
    pub scene: Option<String>,
}

impl SceneOwnership {
    /// Create a persistent (scene-independent) ownership.
    pub fn persistent() -> Self {
        Self { scene: None }
    }

    /// Create ownership for a specific scene.
    pub fn for_scene(scene: impl Into<String>) -> Self {
        Self {
            scene: Some(scene.into()),
        }
    }

    /// Returns true if this entity is persistent (survives scene unloads).
    pub fn is_persistent(&self) -> bool {
        self.scene.is_none()
    }

    /// Returns true if this entity belongs to the named scene.
    pub fn belongs_to(&self, scene_name: &str) -> bool {
        self.scene.as_deref() == Some(scene_name)
    }
}

impl Default for SceneOwnership {
    fn default() -> Self {
        Self::persistent()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent() {
        let so = SceneOwnership::persistent();
        assert!(so.is_persistent());
        assert!(!so.belongs_to("any"));
        assert_eq!(so.scene, None);
    }

    #[test]
    fn for_scene() {
        let so = SceneOwnership::for_scene("level_01");
        assert!(!so.is_persistent());
        assert!(so.belongs_to("level_01"));
        assert!(!so.belongs_to("level_02"));
    }

    #[test]
    fn default_is_persistent() {
        let so = SceneOwnership::default();
        assert!(so.is_persistent());
    }

    #[test]
    fn serialization_roundtrip() {
        let cases = vec![
            SceneOwnership::persistent(),
            SceneOwnership::for_scene("level_01"),
        ];

        for so in &cases {
            let ron = ron::to_string(so).unwrap();
            let back: SceneOwnership = ron::from_str(&ron).unwrap();
            assert_eq!(&back, so);
        }
    }

    #[test]
    fn equality() {
        assert_eq!(
            SceneOwnership::for_scene("a"),
            SceneOwnership::for_scene("a")
        );
        assert_ne!(
            SceneOwnership::for_scene("a"),
            SceneOwnership::for_scene("b")
        );
        assert_ne!(
            SceneOwnership::persistent(),
            SceneOwnership::for_scene("a")
        );
    }
}
