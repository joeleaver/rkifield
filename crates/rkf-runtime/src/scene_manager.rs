//! Multi-scene management for the v2 RKIField engine.
//!
//! [`SceneManager`] tracks multiple simultaneously-loaded scenes, each represented
//! by a [`SceneHandle`]. Scenes progress through a [`SceneStatus`] lifecycle
//! (`Loading` → `Active` → `Unloading`) and can be flagged as *persistent* so
//! they survive a [`LoadMode::Swap`] operation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// How to load a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Add scene content to existing scenes (additive loading).
    Additive,
    /// Replace all non-persistent scenes with this one.
    Swap,
}

/// Handle to a loaded scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneHandle(u32);

/// Status of a managed scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneStatus {
    /// Scene is loading.
    Loading,
    /// Scene is loaded and active.
    Active,
    /// Scene is being unloaded.
    Unloading,
}

/// Per-scene tracking entry.
#[derive(Debug, Clone)]
pub struct ManagedScene {
    /// Unique handle for this scene.
    pub handle: SceneHandle,
    /// Human-readable name.
    pub name: String,
    /// File path used to load this scene.
    pub path: String,
    /// Current lifecycle status.
    pub status: SceneStatus,
    /// Persistent scenes survive [`LoadMode::Swap`].
    pub persistent: bool,
    /// Object IDs from this scene (used for cleanup on unload).
    pub object_ids: Vec<u32>,
}

// ---------------------------------------------------------------------------
// SceneManager
// ---------------------------------------------------------------------------

/// Manages multiple scenes loaded simultaneously.
pub struct SceneManager {
    scenes: HashMap<SceneHandle, ManagedScene>,
    next_handle: u32,
}

impl SceneManager {
    /// Create a new, empty [`SceneManager`].
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            next_handle: 0,
        }
    }

    /// Begin loading a scene.
    ///
    /// The scene starts in [`SceneStatus::Loading`]. Call [`activate_scene`] once
    /// the data is fully loaded to transition it to [`SceneStatus::Active`].
    ///
    /// In [`LoadMode::Swap`] the *caller* is responsible for calling
    /// [`scenes_to_swap`] and unloading the returned handles before or after
    /// this call. The `SceneManager` does not auto-unload anything here.
    pub fn load_scene(&mut self, name: &str, path: &str, _mode: LoadMode) -> SceneHandle {
        let handle = SceneHandle(self.next_handle);
        self.next_handle += 1;
        self.scenes.insert(
            handle,
            ManagedScene {
                handle,
                name: name.to_string(),
                path: path.to_string(),
                status: SceneStatus::Loading,
                persistent: false,
                object_ids: Vec::new(),
            },
        );
        handle
    }

    /// Mark a scene as fully loaded and active, recording the object IDs it owns.
    ///
    /// Returns `true` if the transition succeeded (the handle was known and in the
    /// `Loading` state).  Returns `false` if the handle is unknown or already in
    /// a different state.
    pub fn activate_scene(&mut self, handle: SceneHandle, object_ids: Vec<u32>) -> bool {
        match self.scenes.get_mut(&handle) {
            Some(scene) if scene.status == SceneStatus::Loading => {
                scene.status = SceneStatus::Active;
                scene.object_ids = object_ids;
                true
            }
            _ => false,
        }
    }

    /// Begin unloading a scene.
    ///
    /// Returns the scene's object IDs so the caller can clean up GPU/CPU
    /// resources.  The entry is removed from the manager after this call.
    ///
    /// Returns `None` if:
    /// - The handle is unknown.
    /// - The scene is persistent and `force` is `false`.
    pub fn unload_scene(&mut self, handle: SceneHandle, force: bool) -> Option<Vec<u32>> {
        // Guard: persistent scenes need force.
        if let Some(scene) = self.scenes.get(&handle) {
            if scene.persistent && !force {
                return None;
            }
        } else {
            return None;
        }

        // Remove the entry and return the object IDs.
        let scene = self.scenes.remove(&handle)?;
        Some(scene.object_ids)
    }

    /// Mark a scene as persistent (or not).
    ///
    /// Persistent scenes are not included in [`scenes_to_swap`] and cannot be
    /// unloaded without `force = true`.
    pub fn set_persistent(&mut self, handle: SceneHandle, persistent: bool) {
        if let Some(scene) = self.scenes.get_mut(&handle) {
            scene.persistent = persistent;
        }
    }

    /// Get an immutable reference to the scene entry for `handle`.
    pub fn get_scene(&self, handle: SceneHandle) -> Option<&ManagedScene> {
        self.scenes.get(&handle)
    }

    /// Find the first scene with the given name, returning its handle.
    pub fn find_by_name(&self, name: &str) -> Option<SceneHandle> {
        self.scenes
            .values()
            .find(|s| s.name == name)
            .map(|s| s.handle)
    }

    /// Return handles of all scenes currently in [`SceneStatus::Active`].
    pub fn active_scenes(&self) -> Vec<SceneHandle> {
        self.scenes
            .values()
            .filter(|s| s.status == SceneStatus::Active)
            .map(|s| s.handle)
            .collect()
    }

    /// Return handles of scenes that should be unloaded during a
    /// [`LoadMode::Swap`] operation: active scenes that are *not* persistent.
    pub fn scenes_to_swap(&self) -> Vec<SceneHandle> {
        self.scenes
            .values()
            .filter(|s| s.status == SceneStatus::Active && !s.persistent)
            .map(|s| s.handle)
            .collect()
    }

    /// Total number of tracked scenes (any status).
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Number of scenes currently in [`SceneStatus::Active`].
    pub fn active_count(&self) -> usize {
        self.scenes
            .values()
            .filter(|s| s.status == SceneStatus::Active)
            .count()
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1 — new_manager_empty
    #[test]
    fn new_manager_empty() {
        let mgr = SceneManager::new();
        assert_eq!(mgr.scene_count(), 0);
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.active_scenes().is_empty());
    }

    // 2 — load_scene_returns_handle
    #[test]
    fn load_scene_returns_handle() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("main", "levels/main.rkscene", LoadMode::Additive);
        assert!(mgr.get_scene(handle).is_some());
        let scene = mgr.get_scene(handle).unwrap();
        assert_eq!(scene.name, "main");
        assert_eq!(scene.path, "levels/main.rkscene");
        assert_eq!(scene.status, SceneStatus::Loading);
    }

    // 3 — activate_makes_active
    #[test]
    fn activate_makes_active() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("level1", "l1.rkscene", LoadMode::Additive);
        assert_eq!(mgr.get_scene(handle).unwrap().status, SceneStatus::Loading);

        let ok = mgr.activate_scene(handle, vec![1, 2, 3]);
        assert!(ok);
        assert_eq!(mgr.get_scene(handle).unwrap().status, SceneStatus::Active);
        assert_eq!(mgr.get_scene(handle).unwrap().object_ids, vec![1, 2, 3]);
    }

    // 4 — unload_returns_object_ids
    #[test]
    fn unload_returns_object_ids() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("level2", "l2.rkscene", LoadMode::Additive);
        mgr.activate_scene(handle, vec![10, 20, 30]);

        let ids = mgr.unload_scene(handle, false);
        assert_eq!(ids, Some(vec![10, 20, 30]));
        // Entry removed.
        assert!(mgr.get_scene(handle).is_none());
        assert_eq!(mgr.scene_count(), 0);
    }

    // 5 — persistent_survives_swap
    #[test]
    fn persistent_survives_swap() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("ui", "ui.rkscene", LoadMode::Additive);
        mgr.activate_scene(handle, vec![]);
        mgr.set_persistent(handle, true);

        // Persistent scene must NOT appear in scenes_to_swap.
        let to_swap = mgr.scenes_to_swap();
        assert!(!to_swap.contains(&handle));
    }

    // 6 — swap_mode_marks_non_persistent
    #[test]
    fn swap_mode_marks_non_persistent() {
        let mut mgr = SceneManager::new();
        let h1 = mgr.load_scene("gameplay", "gameplay.rkscene", LoadMode::Additive);
        mgr.activate_scene(h1, vec![1]);

        let h2 = mgr.load_scene("ui", "ui.rkscene", LoadMode::Additive);
        mgr.activate_scene(h2, vec![2]);
        mgr.set_persistent(h2, true);

        let to_swap = mgr.scenes_to_swap();
        // Only non-persistent active scene is in the list.
        assert_eq!(to_swap.len(), 1);
        assert!(to_swap.contains(&h1));
        assert!(!to_swap.contains(&h2));
    }

    // 7 — unload_persistent_needs_force
    #[test]
    fn unload_persistent_needs_force() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("core", "core.rkscene", LoadMode::Additive);
        mgr.activate_scene(handle, vec![99]);
        mgr.set_persistent(handle, true);

        // Without force → None, scene still present.
        assert!(mgr.unload_scene(handle, false).is_none());
        assert!(mgr.get_scene(handle).is_some());

        // With force → succeeds.
        let ids = mgr.unload_scene(handle, true);
        assert_eq!(ids, Some(vec![99]));
        assert!(mgr.get_scene(handle).is_none());
    }

    // 8 — find_by_name
    #[test]
    fn find_by_name() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("tutorial", "tutorial.rkscene", LoadMode::Additive);

        assert_eq!(mgr.find_by_name("tutorial"), Some(handle));
        assert_eq!(mgr.find_by_name("nonexistent"), None);
    }

    // 9 — active_scenes_list
    #[test]
    fn active_scenes_list() {
        let mut mgr = SceneManager::new();
        let h1 = mgr.load_scene("a", "a.rkscene", LoadMode::Additive);
        let h2 = mgr.load_scene("b", "b.rkscene", LoadMode::Additive);
        let _h3 = mgr.load_scene("c", "c.rkscene", LoadMode::Additive);

        // Activate only h1 and h2.
        mgr.activate_scene(h1, vec![]);
        mgr.activate_scene(h2, vec![]);

        let active = mgr.active_scenes();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&h1));
        assert!(active.contains(&h2));
    }

    // 10 — double_load_same_name
    #[test]
    fn double_load_same_name() {
        let mut mgr = SceneManager::new();
        let h1 = mgr.load_scene("zone", "zone.rkscene", LoadMode::Additive);
        let h2 = mgr.load_scene("zone", "zone.rkscene", LoadMode::Additive);

        // Different handles.
        assert_ne!(h1, h2);
        // Both tracked.
        assert_eq!(mgr.scene_count(), 2);
    }

    // 11 — set_persistent_toggle
    #[test]
    fn set_persistent_toggle() {
        let mut mgr = SceneManager::new();
        let handle = mgr.load_scene("toggleable", "t.rkscene", LoadMode::Additive);

        // Default: not persistent.
        assert!(!mgr.get_scene(handle).unwrap().persistent);

        mgr.set_persistent(handle, true);
        assert!(mgr.get_scene(handle).unwrap().persistent);

        mgr.set_persistent(handle, false);
        assert!(!mgr.get_scene(handle).unwrap().persistent);
    }

    // 12 — scene_count_tracking
    #[test]
    fn scene_count_tracking() {
        let mut mgr = SceneManager::new();
        assert_eq!(mgr.scene_count(), 0);
        assert_eq!(mgr.active_count(), 0);

        let h1 = mgr.load_scene("s1", "s1.rkscene", LoadMode::Additive);
        assert_eq!(mgr.scene_count(), 1);
        assert_eq!(mgr.active_count(), 0); // still Loading

        mgr.activate_scene(h1, vec![]);
        assert_eq!(mgr.scene_count(), 1);
        assert_eq!(mgr.active_count(), 1);

        let h2 = mgr.load_scene("s2", "s2.rkscene", LoadMode::Additive);
        mgr.activate_scene(h2, vec![]);
        assert_eq!(mgr.scene_count(), 2);
        assert_eq!(mgr.active_count(), 2);

        mgr.unload_scene(h1, false);
        assert_eq!(mgr.scene_count(), 1);
        assert_eq!(mgr.active_count(), 1);

        mgr.unload_scene(h2, false);
        assert_eq!(mgr.scene_count(), 0);
        assert_eq!(mgr.active_count(), 0);
    }
}
