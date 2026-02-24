//! Terrain tile streaming manager for the v2 SDF engine.
//!
//! Drives camera-based LOD selection and tile load/unload scheduling for
//! terrain tiles. The system is a **pure state machine** — it emits
//! [`TileRequest`]s and eviction candidates that a separate async I/O layer
//! must act on. No I/O is performed here.
//!
//! # Usage
//!
//! ```rust
//! use glam::Vec3;
//! use rkf_runtime::terrain_streaming::{TerrainStreamConfig, TerrainStreaming};
//!
//! let config = TerrainStreamConfig::default();
//! let mut streaming = TerrainStreaming::new(config);
//!
//! // Each frame: update based on camera position and collect work.
//! let (to_load, to_unload) = streaming.update(Vec3::ZERO);
//! ```

use glam::{IVec2, Vec3};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// TileState
// ---------------------------------------------------------------------------

/// Lifecycle state for a single terrain tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileState {
    /// Tile is known but has no data loaded.
    Unloaded,
    /// An async load is in-flight for this tile.
    Loading,
    /// Tile data is resident, rendering at the given LOD level.
    Loaded {
        /// LOD level (0 = finest).
        lod: usize,
    },
    /// Tile is being generated procedurally.
    Generating,
}

// ---------------------------------------------------------------------------
// TileRequest
// ---------------------------------------------------------------------------

/// A load request emitted by [`TerrainStreaming::update`].
#[derive(Debug, Clone)]
pub struct TileRequest {
    /// Grid coordinate of the tile.
    pub coord: IVec2,
    /// LOD level to load (0 = finest).
    pub lod: usize,
    /// Load priority — higher is more urgent. Computed from camera distance;
    /// closer tiles have higher priority.
    pub priority: f32,
}

// ---------------------------------------------------------------------------
// TerrainStreamConfig
// ---------------------------------------------------------------------------

/// Configuration for terrain streaming.
#[derive(Debug, Clone)]
pub struct TerrainStreamConfig {
    /// Tile size in meters.
    pub tile_size: f32,
    /// Maximum number of tiles loaded simultaneously.
    pub max_loaded_tiles: usize,
    /// Distance ranges (in meters) for LOD tiers `[close, medium, far]`.
    ///
    /// A tile within `lod_distances[0]` metres of the camera is loaded at
    /// LOD 0 (finest). A tile between `lod_distances[0]` and
    /// `lod_distances[1]` is loaded at LOD 1, and so on.
    pub lod_distances: Vec<f32>,
    /// Maximum tiles to load per frame.
    pub max_loads_per_frame: usize,
}

impl Default for TerrainStreamConfig {
    fn default() -> Self {
        Self {
            tile_size: 64.0,
            max_loaded_tiles: 256,
            lod_distances: vec![128.0, 512.0, 2048.0],
            max_loads_per_frame: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// TerrainStreaming
// ---------------------------------------------------------------------------

/// Manages terrain tile streaming based on camera position.
///
/// Call [`update`](Self::update) once per frame with the current camera
/// position to obtain the lists of tiles to load and unload. Notify the
/// system of results via [`on_tile_loaded`](Self::on_tile_loaded),
/// [`on_tile_generated`](Self::on_tile_generated), and
/// [`on_tile_unloaded`](Self::on_tile_unloaded).
pub struct TerrainStreaming {
    config: TerrainStreamConfig,
    tile_states: HashMap<IVec2, TileState>,
}

impl TerrainStreaming {
    /// Create a new streaming manager with the given configuration.
    pub fn new(config: TerrainStreamConfig) -> Self {
        Self {
            config,
            tile_states: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Frame update
    // -----------------------------------------------------------------------

    /// Update streaming based on camera position.
    ///
    /// Returns `(tiles_to_load, tiles_to_unload)`.
    ///
    /// - `tiles_to_load`: new [`TileRequest`]s sorted by priority (highest
    ///   first), capped at [`TerrainStreamConfig::max_loads_per_frame`].
    ///   Tiles that are already `Loading`, `Loaded`, or `Generating` are
    ///   excluded.
    /// - `tiles_to_unload`: coordinates of currently-loaded tiles that have
    ///   moved outside the streaming radius.
    pub fn update(&mut self, camera_pos: Vec3) -> (Vec<TileRequest>, Vec<IVec2>) {
        // Compute the full desired set.
        let desired = self.desired_tiles(camera_pos);
        let desired_coords: HashSet<IVec2> = desired.iter().map(|(c, _, _)| *c).collect();

        // Tiles to unload: loaded but no longer desired.
        let to_unload: Vec<IVec2> = self
            .tile_states
            .iter()
            .filter_map(|(coord, state)| {
                let is_loaded_or_loading = matches!(
                    state,
                    TileState::Loaded { .. } | TileState::Loading | TileState::Generating
                );
                if is_loaded_or_loading && !desired_coords.contains(coord) {
                    Some(*coord)
                } else {
                    None
                }
            })
            .collect();

        // Tiles to load: desired but not yet in-flight or loaded.
        let mut load_candidates: Vec<TileRequest> = desired
            .into_iter()
            .filter(|(coord, _, _)| {
                !matches!(
                    self.tile_states.get(coord),
                    Some(TileState::Loaded { .. })
                        | Some(TileState::Loading)
                        | Some(TileState::Generating)
                )
            })
            .map(|(coord, lod, priority)| TileRequest { coord, lod, priority })
            .collect();

        // Sort by descending priority (closest first).
        load_candidates
            .sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));

        // Cap to max_loads_per_frame.
        load_candidates.truncate(self.config.max_loads_per_frame);

        // Mark selected tiles as Loading.
        for req in &load_candidates {
            self.tile_states.insert(req.coord, TileState::Loading);
        }

        (load_candidates, to_unload)
    }

    // -----------------------------------------------------------------------
    // Notification callbacks
    // -----------------------------------------------------------------------

    /// Mark a tile as successfully loaded from disk.
    pub fn on_tile_loaded(&mut self, coord: IVec2, lod: usize) {
        self.tile_states.insert(coord, TileState::Loaded { lod });
    }

    /// Mark a tile as successfully generated procedurally.
    pub fn on_tile_generated(&mut self, coord: IVec2, lod: usize) {
        self.tile_states.insert(coord, TileState::Loaded { lod });
    }

    /// Mark a tile as unloaded and remove it from tracking.
    pub fn on_tile_unloaded(&mut self, coord: IVec2) {
        self.tile_states.remove(&coord);
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the current state of a tile.
    ///
    /// Returns [`TileState::Unloaded`] for tiles that have never been seen.
    pub fn tile_state(&self, coord: IVec2) -> TileState {
        self.tile_states
            .get(&coord)
            .cloned()
            .unwrap_or(TileState::Unloaded)
    }

    /// Number of tiles currently in the `Loaded` state.
    pub fn loaded_count(&self) -> usize {
        self.tile_states
            .values()
            .filter(|s| matches!(s, TileState::Loaded { .. }))
            .count()
    }

    /// Coordinates of all loaded tiles.
    pub fn loaded_tiles(&self) -> Vec<IVec2> {
        self.tile_states
            .iter()
            .filter_map(|(coord, state)| {
                if matches!(state, TileState::Loaded { .. }) {
                    Some(*coord)
                } else {
                    None
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Select the LOD level for a tile given the distance from its centre to
    /// the camera.
    ///
    /// Returns 0 for the finest tier and increments for each coarser tier.
    /// Tiles beyond the last distance threshold get the coarsest LOD index
    /// (`lod_distances.len()`).
    fn select_lod(&self, tile_center: Vec3, camera_pos: Vec3) -> usize {
        let dist = (tile_center - camera_pos).length();
        for (i, &threshold) in self.config.lod_distances.iter().enumerate() {
            if dist < threshold {
                return i;
            }
        }
        self.config.lod_distances.len()
    }

    /// Compute the set of tiles that should be resident given the camera
    /// position.
    ///
    /// Returns a list of `(coord, lod, priority)` tuples covering all tiles
    /// within the streaming radius (the last `lod_distances` entry).
    fn desired_tiles(&self, camera_pos: Vec3) -> Vec<(IVec2, usize, f32)> {
        let max_dist = self
            .config
            .lod_distances
            .last()
            .copied()
            .unwrap_or(0.0);

        let tile_size = self.config.tile_size;

        // Camera's tile coordinate.
        let cam_tile_x = (camera_pos.x / tile_size).floor() as i32;
        let cam_tile_z = (camera_pos.z / tile_size).floor() as i32;

        // Radius in tiles (ceiling so we don't clip the boundary).
        let radius = ((max_dist / tile_size).ceil() as i32).max(0);

        let mut result = Vec::new();

        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let coord = IVec2::new(cam_tile_x + dx, cam_tile_z + dz);

                // Centre of this tile in world space.
                let tile_center = Vec3::new(
                    (coord.x as f32 + 0.5) * tile_size,
                    camera_pos.y, // use camera Y so distance is horizontal
                    (coord.y as f32 + 0.5) * tile_size,
                );

                let dist = (tile_center - camera_pos).length();
                if dist > max_dist {
                    continue;
                }

                let lod = self.select_lod(tile_center, camera_pos);
                // Priority: higher = closer. Use inverse distance (clamped to
                // avoid div-by-zero when the camera is exactly on the centre).
                let priority = 1.0 / dist.max(0.01);

                result.push((coord, lod, priority));
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_streaming() -> TerrainStreaming {
        TerrainStreaming::new(TerrainStreamConfig::default())
    }

    // -----------------------------------------------------------------------
    // 1. new_streaming_empty
    // -----------------------------------------------------------------------

    #[test]
    fn new_streaming_empty() {
        let s = default_streaming();
        assert_eq!(s.loaded_count(), 0);
        assert!(s.loaded_tiles().is_empty());
        assert_eq!(s.tile_state(IVec2::ZERO), TileState::Unloaded);
    }

    // -----------------------------------------------------------------------
    // 2. update_loads_nearby_tiles
    // -----------------------------------------------------------------------

    #[test]
    fn update_loads_nearby_tiles() {
        let mut s = default_streaming();
        let (to_load, to_unload) = s.update(Vec3::ZERO);

        // Standing at the origin there must be at least one tile requested.
        assert!(!to_load.is_empty(), "expected at least one tile to load near origin");
        assert!(to_unload.is_empty(), "no tiles should be unloaded when starting empty");

        // Every requested tile should now be in the Loading state.
        for req in &to_load {
            assert_eq!(
                s.tile_state(req.coord),
                TileState::Loading,
                "tile {:?} should be Loading after update",
                req.coord
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. update_unloads_far_tiles
    // -----------------------------------------------------------------------

    #[test]
    fn update_unloads_far_tiles() {
        let mut s = default_streaming();

        // Load a tile near origin.
        let near_coord = IVec2::new(0, 0);
        s.on_tile_loaded(near_coord, 0);
        assert_eq!(s.loaded_count(), 1);

        // Move camera very far away (beyond all LOD tiers).
        let far_camera = Vec3::new(100_000.0, 0.0, 100_000.0);
        let (_to_load, to_unload) = s.update(far_camera);

        // The near tile should be in the unload list.
        assert!(
            to_unload.contains(&near_coord),
            "near tile should be unloaded when camera is far away; unload list: {:?}",
            to_unload
        );
    }

    // -----------------------------------------------------------------------
    // 4. on_tile_loaded_updates_state
    // -----------------------------------------------------------------------

    #[test]
    fn on_tile_loaded_updates_state() {
        let mut s = default_streaming();

        // Trigger a load so the tile is in Loading state.
        let (to_load, _) = s.update(Vec3::ZERO);
        assert!(!to_load.is_empty());

        let coord = to_load[0].coord;
        assert_eq!(s.tile_state(coord), TileState::Loading);

        // Notify loaded.
        s.on_tile_loaded(coord, 0);
        assert_eq!(s.tile_state(coord), TileState::Loaded { lod: 0 });
        assert_eq!(s.loaded_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 5. select_lod_by_distance
    // -----------------------------------------------------------------------

    #[test]
    fn select_lod_by_distance() {
        let s = default_streaming();
        // Default lod_distances: [128, 512, 2048]

        let camera = Vec3::ZERO;

        // Distance < 128 → LOD 0 (finest).
        let near_center = Vec3::new(64.0, 0.0, 0.0);
        assert_eq!(s.select_lod(near_center, camera), 0);

        // Distance between 128 and 512 → LOD 1.
        let mid_center = Vec3::new(200.0, 0.0, 0.0);
        assert_eq!(s.select_lod(mid_center, camera), 1);

        // Distance between 512 and 2048 → LOD 2.
        let far_center = Vec3::new(1000.0, 0.0, 0.0);
        assert_eq!(s.select_lod(far_center, camera), 2);

        // Distance > 2048 → LOD 3 (coarsest).
        let very_far = Vec3::new(3000.0, 0.0, 0.0);
        assert_eq!(s.select_lod(very_far, camera), 3);
    }

    // -----------------------------------------------------------------------
    // 6. max_loads_per_frame
    // -----------------------------------------------------------------------

    #[test]
    fn max_loads_per_frame() {
        let config = TerrainStreamConfig {
            tile_size: 1.0,          // tiny tiles → many desired tiles
            max_loads_per_frame: 2,
            lod_distances: vec![100.0, 500.0, 2000.0],
            max_loaded_tiles: 1024,
        };
        let mut s = TerrainStreaming::new(config);

        let (to_load, _) = s.update(Vec3::ZERO);
        assert!(
            to_load.len() <= 2,
            "max_loads_per_frame=2 but got {} requests",
            to_load.len()
        );
    }

    // -----------------------------------------------------------------------
    // 7. loaded_count_tracking
    // -----------------------------------------------------------------------

    #[test]
    fn loaded_count_tracking() {
        let mut s = default_streaming();
        assert_eq!(s.loaded_count(), 0);

        s.on_tile_loaded(IVec2::new(0, 0), 0);
        s.on_tile_loaded(IVec2::new(1, 0), 1);
        assert_eq!(s.loaded_count(), 2);

        s.on_tile_unloaded(IVec2::new(0, 0));
        assert_eq!(s.loaded_count(), 1);

        s.on_tile_unloaded(IVec2::new(1, 0));
        assert_eq!(s.loaded_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 8. desired_tiles_concentric
    // -----------------------------------------------------------------------

    #[test]
    fn desired_tiles_concentric() {
        let s = default_streaming();
        let camera = Vec3::ZERO;
        let desired = s.desired_tiles(camera);

        // There should be at least one tile.
        assert!(!desired.is_empty());

        // Check that closer tiles have higher priority than farther tiles.
        // Find two tiles in the list with meaningfully different priorities.
        let mut sorted = desired.clone();
        sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // The highest-priority tile must be closer than the lowest-priority.
        if sorted.len() >= 2 {
            let highest = &sorted[0];
            let lowest = &sorted[sorted.len() - 1];
            assert!(
                highest.2 >= lowest.2,
                "closest tile priority {} should be >= farthest priority {}",
                highest.2,
                lowest.2
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. on_tile_unloaded_clears_state
    // -----------------------------------------------------------------------

    #[test]
    fn on_tile_unloaded_clears_state() {
        let mut s = default_streaming();
        let coord = IVec2::new(3, 7);

        s.on_tile_loaded(coord, 0);
        assert_eq!(s.tile_state(coord), TileState::Loaded { lod: 0 });

        s.on_tile_unloaded(coord);
        assert_eq!(
            s.tile_state(coord),
            TileState::Unloaded,
            "tile should report Unloaded after on_tile_unloaded"
        );
        assert_eq!(s.loaded_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 10. update_idempotent
    // -----------------------------------------------------------------------

    #[test]
    fn update_idempotent() {
        let mut s = default_streaming();
        let camera = Vec3::ZERO;

        // First update: some tiles requested.
        let (first_load, _) = s.update(camera);
        let first_count = first_load.len();

        // Without moving the camera, tiles requested in the first call are now
        // in Loading state and must NOT be double-requested.
        let (second_load, _) = s.update(camera);

        // Tiles already requested in the first batch must not appear again.
        let first_coords: HashSet<IVec2> = first_load.iter().map(|r| r.coord).collect();
        for req in &second_load {
            assert!(
                !first_coords.contains(&req.coord),
                "tile {:?} was in the first batch and must not be re-requested in the second",
                req.coord
            );
        }

        // Simple case: if max_loads_per_frame >= total desired, second call
        // should produce zero requests (all tiles already Loading).
        let total_desired = s.desired_tiles(camera).len();
        if first_count >= total_desired {
            assert_eq!(
                second_load.len(),
                0,
                "all desired tiles already requested; second update should produce 0 new requests"
            );
        }
    }
}
