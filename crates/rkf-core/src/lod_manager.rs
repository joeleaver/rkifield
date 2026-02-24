//! LOD manager — per-frame LOD selection and brick map transitions.
//!
//! [`LodManager`] tracks the current LOD level per object and handles
//! transitions when the camera moves. When an object's LOD changes:
//! - The new level's brick map is activated
//! - The old level's bricks are marked for deallocation
//!
//! This is a CPU-side manager. The scene's brick map handles are updated
//! in place, and the GPU data is re-uploaded the next frame.

use std::collections::HashMap;

use crate::brick_map::BrickMapAllocator;
use crate::lod::{select_lod, LodSelection, ObjectLod};
use crate::scene_node::BrickMapHandle;

/// Per-object LOD state.
#[derive(Debug, Clone)]
struct ObjectLodState {
    /// Current active LOD level (or Analytical/Hidden).
    current: LodSelection,
    /// Brick map handles per LOD level (indexed same as `ObjectLod::levels`).
    brick_maps: Vec<Option<BrickMapHandle>>,
}

/// Tracks LOD transitions for all objects in the scene.
#[derive(Debug)]
pub struct LodManager {
    /// Per-object LOD state, keyed by object ID.
    objects: HashMap<u32, ObjectLodState>,
    /// Maximum voxel screen-pixels before switching to coarser LOD.
    pub max_voxel_pixels: f32,
}

/// A single LOD transition event.
#[derive(Debug, Clone)]
pub struct LodTransition {
    /// Object ID that changed.
    pub object_id: u32,
    /// Previous LOD selection.
    pub from: LodSelection,
    /// New LOD selection.
    pub to: LodSelection,
    /// Brick map handle to activate (None for Analytical/Hidden).
    pub new_brick_map: Option<BrickMapHandle>,
    /// Brick map handle to deallocate (None if not evicting).
    pub old_brick_map: Option<BrickMapHandle>,
}

impl LodManager {
    /// Create a new LOD manager.
    pub fn new(max_voxel_pixels: f32) -> Self {
        Self {
            objects: HashMap::new(),
            max_voxel_pixels,
        }
    }

    /// Register an object with its LOD config and initial brick map handles.
    ///
    /// `brick_maps` should have one entry per LOD level (matching `lod.levels`).
    /// Pass `None` for levels that haven't been loaded yet.
    pub fn register_object(
        &mut self,
        object_id: u32,
        initial_level: LodSelection,
        brick_maps: Vec<Option<BrickMapHandle>>,
    ) {
        self.objects.insert(
            object_id,
            ObjectLodState {
                current: initial_level,
                brick_maps,
            },
        );
    }

    /// Remove an object from tracking.
    pub fn unregister_object(&mut self, object_id: u32) {
        self.objects.remove(&object_id);
    }

    /// Get the current LOD selection for an object.
    pub fn current_lod(&self, object_id: u32) -> Option<&LodSelection> {
        self.objects.get(&object_id).map(|s| &s.current)
    }

    /// Get the active brick map handle for an object.
    pub fn active_brick_map(&self, object_id: u32) -> Option<BrickMapHandle> {
        let state = self.objects.get(&object_id)?;
        match &state.current {
            LodSelection::Voxelized(level) => {
                state.brick_maps.get(*level).and_then(|h| *h)
            }
            _ => None,
        }
    }

    /// Set the brick map handle for a specific LOD level of an object.
    pub fn set_brick_map(
        &mut self,
        object_id: u32,
        level: usize,
        handle: BrickMapHandle,
    ) -> bool {
        if let Some(state) = self.objects.get_mut(&object_id) {
            if level < state.brick_maps.len() {
                state.brick_maps[level] = Some(handle);
                return true;
            }
        }
        false
    }

    /// Run per-frame LOD selection for all registered objects.
    ///
    /// Returns a list of LOD transitions that occurred. The caller is
    /// responsible for:
    /// - Activating new brick maps (updating scene node SdfSource)
    /// - Deallocating old brick maps from the allocator
    ///
    /// # Parameters
    /// - `lod_configs` — LOD config per object, looked up by object ID
    /// - `distances` — camera distance per object, looked up by object ID
    /// - `viewport_height` — viewport height in pixels
    /// - `fov_y` — vertical FOV in radians
    pub fn update(
        &mut self,
        lod_configs: &HashMap<u32, ObjectLod>,
        distances: &HashMap<u32, f32>,
        aabbs: &HashMap<u32, crate::aabb::Aabb>,
        viewport_height: f32,
        fov_y: f32,
    ) -> Vec<LodTransition> {
        let mut transitions = Vec::new();
        let default_aabb =
            crate::aabb::Aabb::new(glam::Vec3::splat(-1.0), glam::Vec3::splat(1.0));

        for (&object_id, state) in self.objects.iter_mut() {
            let Some(lod_config) = lod_configs.get(&object_id) else {
                continue;
            };
            let Some(&distance) = distances.get(&object_id) else {
                continue;
            };
            let aabb = aabbs.get(&object_id).unwrap_or(&default_aabb);

            let new_selection = select_lod(
                lod_config,
                aabb,
                distance,
                viewport_height,
                fov_y,
                self.max_voxel_pixels,
            );

            if new_selection != state.current {
                let old_brick_map = match &state.current {
                    LodSelection::Voxelized(level) => {
                        state.brick_maps.get(*level).and_then(|h| *h)
                    }
                    _ => None,
                };
                let new_brick_map = match &new_selection {
                    LodSelection::Voxelized(level) => {
                        state.brick_maps.get(*level).and_then(|h| *h)
                    }
                    _ => None,
                };

                transitions.push(LodTransition {
                    object_id,
                    from: state.current.clone(),
                    to: new_selection.clone(),
                    new_brick_map,
                    old_brick_map,
                });

                state.current = new_selection;
            }
        }

        transitions
    }

    /// Apply brick map deallocations from LOD transitions.
    ///
    /// Convenience method: deallocates the `old_brick_map` from each
    /// transition where the old level is no longer needed.
    pub fn deallocate_evicted(transitions: &[LodTransition], allocator: &mut BrickMapAllocator) {
        for t in transitions {
            if let Some(old_handle) = t.old_brick_map {
                // Only deallocate if the new selection doesn't use this handle
                let keep = t
                    .new_brick_map
                    .map_or(false, |new| new.offset == old_handle.offset);
                if !keep {
                    allocator.deallocate(old_handle);
                }
            }
        }
    }

    /// Number of tracked objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aabb::Aabb;
    use crate::lod::{LodLevel, ObjectLod};
    use crate::scene_node::SdfPrimitive;
    use glam::{UVec3, Vec3};
    use std::f32::consts::FRAC_PI_4;

    fn test_lod_config() -> ObjectLod {
        ObjectLod::multi(
            vec![
                LodLevel {
                    voxel_size: 0.005,
                    brick_count: 1000,
                },
                LodLevel {
                    voxel_size: 0.02,
                    brick_count: 100,
                },
                LodLevel {
                    voxel_size: 0.08,
                    brick_count: 20,
                },
            ],
            Some(SdfPrimitive::Sphere { radius: 1.0 }),
        )
    }

    fn handle(offset: u32) -> BrickMapHandle {
        BrickMapHandle {
            offset,
            dims: UVec3::new(4, 4, 4),
        }
    }

    #[test]
    fn register_and_query() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(
            1,
            LodSelection::Voxelized(0),
            vec![Some(handle(0)), Some(handle(64)), Some(handle(128))],
        );

        assert_eq!(mgr.object_count(), 1);
        assert_eq!(mgr.current_lod(1), Some(&LodSelection::Voxelized(0)));
        assert_eq!(mgr.active_brick_map(1), Some(handle(0)));
    }

    #[test]
    fn unregister_removes() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(1, LodSelection::Voxelized(0), vec![Some(handle(0))]);
        mgr.unregister_object(1);
        assert_eq!(mgr.object_count(), 0);
        assert!(mgr.current_lod(1).is_none());
    }

    #[test]
    fn update_detects_lod_change() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(
            1,
            LodSelection::Voxelized(0),
            vec![Some(handle(0)), Some(handle(64)), Some(handle(128))],
        );

        let mut configs = HashMap::new();
        configs.insert(1, test_lod_config());

        let mut aabbs = HashMap::new();
        aabbs.insert(1, Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)));

        // Close distance — should stay at finest (0)
        let mut dists = HashMap::new();
        dists.insert(1, 1.0_f32);
        let transitions = mgr.update(&configs, &dists, &aabbs, 1080.0, FRAC_PI_4);
        assert!(transitions.is_empty(), "should stay at finest when close");

        // Far distance — should switch to coarser
        dists.insert(1, 500.0);
        let transitions = mgr.update(&configs, &dists, &aabbs, 1080.0, FRAC_PI_4);
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].object_id, 1);
        assert_eq!(transitions[0].from, LodSelection::Voxelized(0));
        // Should have selected a coarser level
        match &transitions[0].to {
            LodSelection::Voxelized(level) => assert!(*level > 0),
            _ => panic!("expected Voxelized"),
        }
    }

    #[test]
    fn no_transition_when_unchanged() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(
            1,
            LodSelection::Voxelized(2), // already at coarsest
            vec![Some(handle(0)), Some(handle(64)), Some(handle(128))],
        );

        let mut configs = HashMap::new();
        configs.insert(1, test_lod_config());
        let mut aabbs = HashMap::new();
        aabbs.insert(1, Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)));
        let mut dists = HashMap::new();
        dists.insert(1, 500.0); // very far → coarsest

        let transitions = mgr.update(&configs, &dists, &aabbs, 1080.0, FRAC_PI_4);
        assert!(transitions.is_empty());
    }

    #[test]
    fn set_brick_map() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(
            1,
            LodSelection::Voxelized(0),
            vec![Some(handle(0)), None, None],
        );

        assert!(mgr.set_brick_map(1, 1, handle(64)));
        // Switch to level 1 manually to verify
        mgr.objects.get_mut(&1).unwrap().current = LodSelection::Voxelized(1);
        assert_eq!(mgr.active_brick_map(1), Some(handle(64)));
    }

    #[test]
    fn analytical_has_no_brick_map() {
        let mut mgr = LodManager::new(1.0);
        mgr.register_object(1, LodSelection::Analytical, vec![Some(handle(0))]);
        assert_eq!(mgr.active_brick_map(1), None);
    }

    #[test]
    fn deallocate_evicted_handles() {
        let mut allocator = BrickMapAllocator::new();
        use crate::brick_map::BrickMap;

        // Allocate some maps
        let map = BrickMap::new(UVec3::new(4, 4, 4));
        let h1 = allocator.allocate(&map);
        let h2 = allocator.allocate(&map);

        let transitions = vec![LodTransition {
            object_id: 1,
            from: LodSelection::Voxelized(0),
            to: LodSelection::Voxelized(1),
            new_brick_map: Some(h2),
            old_brick_map: Some(h1),
        }];

        let free_before = allocator.free_region_count();
        LodManager::deallocate_evicted(&transitions, &mut allocator);
        assert_eq!(allocator.free_region_count(), free_before + 1);
    }
}
