//! Engine-automated save/load — camera, loaded scenes, and other engine state.
//!
//! Reserved `engine/` key prefix in the GameStore holds engine-managed state
//! that is automatically saved and restored alongside gameplay data. This
//! ensures the player resumes at the same camera position and in the same
//! scene(s) after loading a save file.

use glam::Quat;
use rkf_core::WorldPosition;

use super::game_store::GameStore;
use super::game_value::GameValue;

// ─── Reserved Keys ────────────────────────────────────────────────────────

/// Store key for the camera world position.
pub const KEY_CAMERA_POSITION: &str = "engine/camera/position";
/// Store key for the camera rotation quaternion.
pub const KEY_CAMERA_ROTATION: &str = "engine/camera/rotation";
/// Store key for the camera field-of-view (degrees).
pub const KEY_CAMERA_FOV: &str = "engine/camera/fov";
/// Store key for the list of currently loaded scene paths.
pub const KEY_SCENES_LOADED: &str = "engine/scenes/loaded";

// ─── EngineStateSnapshot ──────────────────────────────────────────────────

/// Snapshot of engine-managed state restored from the GameStore.
#[derive(Debug, Clone)]
pub struct EngineStateSnapshot {
    /// Camera world position.
    pub camera_position: WorldPosition,
    /// Camera rotation quaternion.
    pub camera_rotation: Quat,
    /// Camera field-of-view in degrees.
    pub camera_fov: f32,
    /// List of scene file paths that were loaded.
    pub loaded_scenes: Vec<String>,
}

// ─── Sync functions ───────────────────────────────────────────────────────

/// Write engine state into the GameStore under reserved `engine/` keys.
///
/// Call this before `store.save_to_ron()` to include engine state in the
/// save file.
pub fn sync_engine_state_to_store(
    store: &mut GameStore,
    camera_pos: WorldPosition,
    camera_rot: Quat,
    camera_fov: f32,
    loaded_scenes: &[String],
) {
    store.set(KEY_CAMERA_POSITION, camera_pos);
    store.set(KEY_CAMERA_ROTATION, camera_rot);
    store.set(KEY_CAMERA_FOV, camera_fov as f64);

    let scene_list = GameValue::List(
        loaded_scenes
            .iter()
            .map(|s| GameValue::String(s.clone()))
            .collect(),
    );
    store.set(KEY_SCENES_LOADED, scene_list);
}

/// Read engine state from the GameStore.
///
/// Returns `None` if any required key is missing (camera position, rotation,
/// or FOV). Missing scene list defaults to an empty vec rather than failing.
pub fn restore_engine_state_from_store(store: &GameStore) -> Option<EngineStateSnapshot> {
    let camera_position = store.get::<WorldPosition>(KEY_CAMERA_POSITION)?;
    let camera_rotation = store.get::<Quat>(KEY_CAMERA_ROTATION)?;
    let camera_fov = store.get::<f32>(KEY_CAMERA_FOV)?;

    let loaded_scenes = match store.get_raw(KEY_SCENES_LOADED) {
        Some(GameValue::List(items)) => items
            .iter()
            .filter_map(|v| {
                if let GameValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => Vec::new(),
    };

    Some(EngineStateSnapshot {
        camera_position,
        camera_rotation,
        camera_fov,
        loaded_scenes,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Vec3};

    fn sample_position() -> WorldPosition {
        WorldPosition {
            chunk: IVec3::new(5, 0, -3),
            local: Vec3::new(1.5, 10.0, -2.0),
        }
    }

    fn sample_rotation() -> Quat {
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)
    }

    #[test]
    fn engine_state_roundtrip() {
        let mut store = GameStore::new();
        let pos = sample_position();
        let rot = sample_rotation();
        let fov = 75.0_f32;
        let scenes = vec!["level_01.rkscene".to_owned(), "hub.rkscene".to_owned()];

        sync_engine_state_to_store(&mut store, pos.clone(), rot, fov, &scenes);

        let snapshot = restore_engine_state_from_store(&store).expect("should restore");

        assert_eq!(snapshot.camera_position, pos);
        // Quat comparison with tolerance
        assert!((snapshot.camera_rotation.x - rot.x).abs() < 1e-6);
        assert!((snapshot.camera_rotation.y - rot.y).abs() < 1e-6);
        assert!((snapshot.camera_rotation.z - rot.z).abs() < 1e-6);
        assert!((snapshot.camera_rotation.w - rot.w).abs() < 1e-6);
        assert!((snapshot.camera_fov - 75.0).abs() < 1e-4);
        assert_eq!(snapshot.loaded_scenes, scenes);
    }

    #[test]
    fn restore_missing_keys_returns_none() {
        let store = GameStore::new();
        assert!(restore_engine_state_from_store(&store).is_none());
    }

    #[test]
    fn partial_restore_missing_position() {
        let mut store = GameStore::new();
        // Only set rotation and fov, skip position
        store.set(KEY_CAMERA_ROTATION, sample_rotation());
        store.set(KEY_CAMERA_FOV, 60.0_f64);

        assert!(restore_engine_state_from_store(&store).is_none());
    }

    #[test]
    fn partial_restore_missing_rotation() {
        let mut store = GameStore::new();
        store.set(KEY_CAMERA_POSITION, sample_position());
        store.set(KEY_CAMERA_FOV, 60.0_f64);

        assert!(restore_engine_state_from_store(&store).is_none());
    }

    #[test]
    fn partial_restore_missing_fov() {
        let mut store = GameStore::new();
        store.set(KEY_CAMERA_POSITION, sample_position());
        store.set(KEY_CAMERA_ROTATION, sample_rotation());

        assert!(restore_engine_state_from_store(&store).is_none());
    }

    #[test]
    fn missing_scenes_defaults_to_empty() {
        let mut store = GameStore::new();
        store.set(KEY_CAMERA_POSITION, sample_position());
        store.set(KEY_CAMERA_ROTATION, sample_rotation());
        store.set(KEY_CAMERA_FOV, 90.0_f64);
        // Don't set scenes

        let snapshot = restore_engine_state_from_store(&store).expect("should restore");
        assert!(snapshot.loaded_scenes.is_empty());
    }

    #[test]
    fn engine_state_survives_ron_roundtrip() {
        let mut store = GameStore::new();
        let pos = sample_position();
        let rot = sample_rotation();
        sync_engine_state_to_store(
            &mut store,
            pos.clone(),
            rot,
            60.0,
            &["test.rkscene".to_owned()],
        );

        // Serialize and deserialize the entire store
        let ron_data = store.save_to_ron();
        let mut store2 = GameStore::new();
        store2.load_from_ron(&ron_data).unwrap();

        let snapshot = restore_engine_state_from_store(&store2).expect("should restore after RON");
        assert_eq!(snapshot.camera_position, pos);
        assert!((snapshot.camera_fov - 60.0).abs() < 1e-4);
        assert_eq!(snapshot.loaded_scenes, vec!["test.rkscene"]);
    }

    #[test]
    fn empty_scenes_list() {
        let mut store = GameStore::new();
        sync_engine_state_to_store(
            &mut store,
            sample_position(),
            sample_rotation(),
            90.0,
            &[],
        );

        let snapshot = restore_engine_state_from_store(&store).expect("should restore");
        assert!(snapshot.loaded_scenes.is_empty());
    }
}
