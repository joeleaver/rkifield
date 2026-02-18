//! Chunk streaming system — loads and evicts chunks based on camera proximity.
//!
//! Each frame, [`StreamingSystem::update`] recomputes distances from the camera
//! to every tracked chunk and produces load/evict request lists.  An async I/O
//! pipeline (task 14.3) consumes those requests.

use std::collections::HashMap;
use std::path::PathBuf;

use glam::{IVec3, Vec3};
use rkf_core::WorldPosition;
use rkf_core::world_position::CHUNK_SIZE;

/// State of a chunk in the streaming system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkState {
    /// Known but not loaded.
    Unloaded,
    /// Load request queued.
    PendingLoad,
    /// Bricks are in the pool, grid is active.
    Loaded,
    /// Eviction queued (grace period expired).
    PendingEvict,
}

/// Per-chunk tracking data.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// Chunk grid coordinates.
    pub coords: IVec3,
    /// Path to the `.rkf` file on disk.
    pub rkf_path: PathBuf,
    /// Whether an edit journal (`.rkj`) exists.
    pub has_journal: bool,
    /// Current streaming state.
    pub state: ChunkState,
    /// Distance from camera (in metres, f64). Updated each frame.
    pub distance: f64,
    /// Brick pool slots allocated for this chunk's bricks.
    pub brick_slots: Vec<u32>,
    /// Frame number when this chunk was last accessed (for LRU).
    pub last_accessed_frame: u64,
}

/// Configuration for the streaming system.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum streaming radius (metres). Chunks beyond this are evicted.
    pub max_radius: f64,
    /// Radius at which chunks begin loading (metres).
    pub load_radius: f64,
    /// Grace period frames before evicting a chunk that left the radius.
    pub eviction_grace_frames: u64,
    /// Maximum number of chunk loads to request per frame.
    pub max_loads_per_frame: usize,
    /// Maximum number of chunk evictions per frame.
    pub max_evicts_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_radius: 256.0,
            load_radius: 200.0,
            eviction_grace_frames: 120, // ~2 seconds at 60fps
            max_loads_per_frame: 4,
            max_evicts_per_frame: 2,
        }
    }
}

/// Manages chunk loading/unloading based on camera proximity.
///
/// Each frame, call [`StreamingSystem::update`] with the current camera position.
/// The system produces load/evict requests that the async I/O pipeline
/// (task 14.3) will service.
pub struct StreamingSystem {
    /// All known chunks, keyed by chunk coords.
    chunks: HashMap<IVec3, ChunkEntry>,
    /// Configuration.
    config: StreamingConfig,
    /// Current frame number (incremented each update).
    frame: u64,
    /// Load requests produced this frame (chunk coords, sorted nearest-first).
    load_requests: Vec<IVec3>,
    /// Evict requests produced this frame (chunk coords, sorted farthest-first).
    evict_requests: Vec<IVec3>,
}

/// Half the chunk size in metres, used to compute chunk centres.
const HALF_CHUNK: f32 = CHUNK_SIZE / 2.0;

impl StreamingSystem {
    /// Create a new streaming system with the given configuration.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            chunks: HashMap::new(),
            config,
            frame: 0,
            load_requests: Vec::new(),
            evict_requests: Vec::new(),
        }
    }

    /// Register a chunk that exists on disk.
    ///
    /// Called when a scene manifest is loaded.  The chunk starts in
    /// [`ChunkState::Unloaded`] and will be considered for loading on
    /// subsequent [`update`](Self::update) calls.
    pub fn register_chunk(&mut self, coords: IVec3, rkf_path: PathBuf, has_journal: bool) {
        self.chunks.insert(
            coords,
            ChunkEntry {
                coords,
                rkf_path,
                has_journal,
                state: ChunkState::Unloaded,
                distance: f64::MAX,
                brick_slots: Vec::new(),
                last_accessed_frame: 0,
            },
        );
    }

    /// Remove a chunk from tracking entirely.
    pub fn unregister_chunk(&mut self, coords: &IVec3) {
        self.chunks.remove(coords);
    }

    /// Batch-register multiple chunks from an iterator of `(coords, path, has_journal)`.
    pub fn register_chunks(&mut self, chunks: impl IntoIterator<Item = (IVec3, PathBuf, bool)>) {
        for (coords, rkf_path, has_journal) in chunks {
            self.register_chunk(coords, rkf_path, has_journal);
        }
    }

    /// Main per-frame update.
    ///
    /// 1. Increments the frame counter.
    /// 2. Computes distance from `camera` to each chunk centre.
    /// 3. Transitions chunk states based on distance and grace period.
    /// 4. Produces sorted load/evict request lists (capped by config limits).
    pub fn update(&mut self, camera: &WorldPosition) {
        self.frame += 1;
        self.load_requests.clear();
        self.evict_requests.clear();

        // Temporary collectors so we can sort before truncating.
        let mut pending_loads: Vec<(IVec3, f64)> = Vec::new();
        let mut pending_evicts: Vec<(IVec3, f64)> = Vec::new();

        for entry in self.chunks.values_mut() {
            // Chunk centre: chunk coords define the chunk origin; centre is
            // offset by half the chunk size on each axis.
            let chunk_centre =
                WorldPosition::new(entry.coords, Vec3::splat(HALF_CHUNK));
            entry.distance = camera.distance_f64(&chunk_centre);

            match entry.state {
                ChunkState::Unloaded => {
                    if entry.distance <= self.config.load_radius {
                        entry.state = ChunkState::PendingLoad;
                        pending_loads.push((entry.coords, entry.distance));
                    }
                }
                ChunkState::PendingLoad => {
                    // Already queued — keep it in the pending list so it can
                    // be sorted and capped.
                    pending_loads.push((entry.coords, entry.distance));
                }
                ChunkState::Loaded => {
                    if entry.distance <= self.config.max_radius {
                        // Still in range — refresh access time.
                        entry.last_accessed_frame = self.frame;
                    } else {
                        // Out of range.  Check grace period.
                        let frames_since_access =
                            self.frame.saturating_sub(entry.last_accessed_frame);
                        if frames_since_access > self.config.eviction_grace_frames {
                            entry.state = ChunkState::PendingEvict;
                            pending_evicts.push((entry.coords, entry.distance));
                        }
                    }
                }
                ChunkState::PendingEvict => {
                    // Already queued for eviction — keep in list.
                    pending_evicts.push((entry.coords, entry.distance));
                }
            }
        }

        // Sort loads nearest-first.
        pending_loads.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pending_loads.truncate(self.config.max_loads_per_frame);
        self.load_requests = pending_loads.into_iter().map(|(c, _)| c).collect();

        // Sort evicts farthest-first.
        pending_evicts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pending_evicts.truncate(self.config.max_evicts_per_frame);
        self.evict_requests = pending_evicts.into_iter().map(|(c, _)| c).collect();
    }

    /// Take the load requests produced by the last [`update`](Self::update).
    ///
    /// Returns chunk coordinates sorted nearest-first.  The vector is drained
    /// so a second call returns an empty vec.
    pub fn drain_load_requests(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.load_requests)
    }

    /// Take the evict requests produced by the last [`update`](Self::update).
    ///
    /// Returns chunk coordinates sorted farthest-first.  The vector is drained
    /// so a second call returns an empty vec.
    pub fn drain_evict_requests(&mut self) -> Vec<IVec3> {
        std::mem::take(&mut self.evict_requests)
    }

    /// Mark a chunk as fully loaded with the given brick pool slot indices.
    ///
    /// Called by the async I/O pipeline when a load completes.
    pub fn mark_loaded(&mut self, coords: &IVec3, brick_slots: Vec<u32>) {
        if let Some(entry) = self.chunks.get_mut(coords) {
            entry.state = ChunkState::Loaded;
            entry.brick_slots = brick_slots;
            entry.last_accessed_frame = self.frame;
        }
    }

    /// Mark a chunk as evicted, resetting it to [`ChunkState::Unloaded`].
    ///
    /// Called by the async I/O pipeline when eviction completes (brick pool
    /// slots freed, grid cells cleared).
    pub fn mark_evicted(&mut self, coords: &IVec3) {
        if let Some(entry) = self.chunks.get_mut(coords) {
            entry.state = ChunkState::Unloaded;
            entry.brick_slots.clear();
        }
    }

    /// Query the current state of a chunk.
    pub fn chunk_state(&self, coords: &IVec3) -> Option<ChunkState> {
        self.chunks.get(coords).map(|e| e.state)
    }

    /// Number of chunks currently in [`ChunkState::Loaded`].
    pub fn loaded_count(&self) -> usize {
        self.chunks.values().filter(|e| e.state == ChunkState::Loaded).count()
    }

    /// Immutable access to the streaming configuration.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Mutable access to the streaming configuration.
    pub fn config_mut(&mut self) -> &mut StreamingConfig {
        &mut self.config
    }

    /// Current frame number (incremented each [`update`](Self::update) call).
    pub fn frame(&self) -> u64 {
        self.frame
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Vec3};
    use std::path::PathBuf;

    /// Helper: create a default streaming system.
    fn make_system() -> StreamingSystem {
        StreamingSystem::new(StreamingConfig::default())
    }

    /// Helper: dummy path for a chunk.
    fn chunk_path(coords: IVec3) -> PathBuf {
        PathBuf::from(format!("world/chunk_{}_{}_{}.rkf", coords.x, coords.y, coords.z))
    }

    // ── 1. construction ─────────────────────────────────────────────────────

    #[test]
    fn new_streaming_system() {
        let sys = make_system();
        assert_eq!(sys.frame(), 0);
        assert_eq!(sys.loaded_count(), 0);
        assert!(sys.config().max_radius > sys.config().load_radius);
    }

    // ── 2. register / unregister ────────────────────────────────────────────

    #[test]
    fn register_and_unregister() {
        let mut sys = make_system();
        let c = IVec3::new(1, 2, 3);
        sys.register_chunk(c, chunk_path(c), false);

        assert_eq!(sys.chunk_state(&c), Some(ChunkState::Unloaded));
        sys.unregister_chunk(&c);
        assert_eq!(sys.chunk_state(&c), None);
    }

    // ── 3. batch register ───────────────────────────────────────────────────

    #[test]
    fn batch_register() {
        let mut sys = make_system();
        let coords: Vec<IVec3> = (0..5).map(|i| IVec3::new(i, 0, 0)).collect();
        let items: Vec<_> = coords
            .iter()
            .map(|&c| (c, chunk_path(c), false))
            .collect();
        sys.register_chunks(items);

        for c in &coords {
            assert_eq!(sys.chunk_state(c), Some(ChunkState::Unloaded));
        }
    }

    // ── 4. update loads nearby chunks ───────────────────────────────────────

    #[test]
    fn update_loads_nearby_chunks() {
        let mut sys = make_system();
        // Place chunk at origin — camera also at origin → distance ~= 4*sqrt(3) ≈ 6.93m
        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        assert_eq!(sys.chunk_state(&c), Some(ChunkState::PendingLoad));
        let loads = sys.drain_load_requests();
        assert!(loads.contains(&c));
    }

    // ── 5. distant chunks stay unloaded ─────────────────────────────────────

    #[test]
    fn update_ignores_distant_chunks() {
        let mut sys = make_system();
        // Place chunk very far away: 1000 chunks = 8000 metres >> load_radius (200m)
        let c = IVec3::new(1000, 0, 0);
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        assert_eq!(sys.chunk_state(&c), Some(ChunkState::Unloaded));
    }

    // ── 6. load requests sorted by distance ─────────────────────────────────

    #[test]
    fn drain_load_requests_sorted_by_distance() {
        let mut sys = StreamingSystem::new(StreamingConfig {
            load_radius: 500.0,
            max_loads_per_frame: 10,
            ..Default::default()
        });

        // Three chunks at different distances from origin.
        let far = IVec3::new(5, 0, 0); // ~44m
        let mid = IVec3::new(2, 0, 0); // ~20m
        let near = IVec3::new(0, 0, 0); // ~6.9m

        sys.register_chunk(far, chunk_path(far), false);
        sys.register_chunk(mid, chunk_path(mid), false);
        sys.register_chunk(near, chunk_path(near), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        let loads = sys.drain_load_requests();
        assert_eq!(loads.len(), 3);
        assert_eq!(loads[0], near);
        assert_eq!(loads[1], mid);
        assert_eq!(loads[2], far);
    }

    // ── 7. load request limit ───────────────────────────────────────────────

    #[test]
    fn load_request_limit() {
        let mut sys = StreamingSystem::new(StreamingConfig {
            load_radius: 5000.0,
            max_loads_per_frame: 2,
            ..Default::default()
        });

        for i in 0..10 {
            let c = IVec3::new(i, 0, 0);
            sys.register_chunk(c, chunk_path(c), false);
        }

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        let loads = sys.drain_load_requests();
        assert_eq!(loads.len(), 2, "should respect max_loads_per_frame");
    }

    // ── 8. mark_loaded transitions state ────────────────────────────────────

    #[test]
    fn mark_loaded_transitions_state() {
        let mut sys = make_system();
        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);
        assert_eq!(sys.chunk_state(&c), Some(ChunkState::PendingLoad));

        sys.mark_loaded(&c, vec![10, 20, 30]);
        assert_eq!(sys.chunk_state(&c), Some(ChunkState::Loaded));
        assert_eq!(sys.loaded_count(), 1);
    }

    // ── 9. loaded chunk stays loaded in radius ──────────────────────────────

    #[test]
    fn loaded_chunk_stays_loaded_in_radius() {
        let mut sys = make_system();
        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);
        sys.mark_loaded(&c, vec![1]);

        // Run several more updates — chunk should stay Loaded.
        for _ in 0..10 {
            sys.update(&camera);
        }
        assert_eq!(sys.chunk_state(&c), Some(ChunkState::Loaded));
    }

    // ── 10. loaded chunk evicts outside radius ──────────────────────────────

    #[test]
    fn loaded_chunk_evicts_outside_radius() {
        let mut sys = StreamingSystem::new(StreamingConfig {
            max_radius: 50.0,
            load_radius: 40.0,
            eviction_grace_frames: 5,
            max_evicts_per_frame: 10,
            ..Default::default()
        });

        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        // Load it while nearby.
        let near_cam = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&near_cam);
        sys.mark_loaded(&c, vec![1]);
        sys.update(&near_cam); // refresh last_accessed_frame

        // Move camera far away. Chunk is beyond max_radius.
        let far_cam = WorldPosition::new(IVec3::new(100, 0, 0), Vec3::ZERO);

        // Run enough updates to exceed grace period.
        for _ in 0..10 {
            sys.update(&far_cam);
        }

        assert_eq!(
            sys.chunk_state(&c),
            Some(ChunkState::PendingEvict),
            "chunk should be pending eviction after grace period"
        );
    }

    // ── 11. eviction grace period ───────────────────────────────────────────

    #[test]
    fn eviction_grace_period() {
        let grace = 10u64;
        let mut sys = StreamingSystem::new(StreamingConfig {
            max_radius: 50.0,
            load_radius: 40.0,
            eviction_grace_frames: grace,
            max_evicts_per_frame: 10,
            ..Default::default()
        });

        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        // Load chunk.
        let near = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&near);
        sys.mark_loaded(&c, vec![1]);
        sys.update(&near); // frame 2, refreshes last_accessed

        // Move far away.
        let far = WorldPosition::new(IVec3::new(100, 0, 0), Vec3::ZERO);

        // Run exactly `grace` updates (frames 3..grace+2). Should NOT evict yet
        // because we need strictly more than grace frames since last access.
        for _ in 0..grace {
            sys.update(&far);
            assert_eq!(
                sys.chunk_state(&c),
                Some(ChunkState::Loaded),
                "should still be Loaded during grace period (frame {})",
                sys.frame()
            );
        }

        // One more update pushes past grace.
        sys.update(&far);
        assert_eq!(sys.chunk_state(&c), Some(ChunkState::PendingEvict));
    }

    // ── 12. evict requests sorted farthest-first ────────────────────────────

    #[test]
    fn drain_evict_requests_sorted() {
        let mut sys = StreamingSystem::new(StreamingConfig {
            max_radius: 30.0,
            load_radius: 500.0,
            eviction_grace_frames: 1,
            max_loads_per_frame: 100,
            max_evicts_per_frame: 100,
        });

        let c1 = IVec3::new(10, 0, 0); // ~84m from origin
        let c2 = IVec3::new(20, 0, 0); // ~164m
        let c3 = IVec3::new(5, 0, 0); // ~44m

        for c in [c1, c2, c3] {
            sys.register_chunk(c, chunk_path(c), false);
        }

        // Load all chunks (camera at origin, load_radius=500).
        let near = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&near);
        for c in [c1, c2, c3] {
            sys.mark_loaded(&c, vec![1]);
        }
        sys.update(&near); // refresh last_accessed

        // Now all chunks are beyond max_radius (30m). Move past grace period.
        // (last_accessed is frame 2, grace is 1, so frame 4 triggers eviction)
        sys.update(&near);
        sys.update(&near);

        let evicts = sys.drain_evict_requests();
        assert_eq!(evicts.len(), 3);
        // Farthest first: c2 (164m), c1 (84m), c3 (44m).
        assert_eq!(evicts[0], c2);
        assert_eq!(evicts[1], c1);
        assert_eq!(evicts[2], c3);
    }

    // ── 13. mark_evicted resets state ───────────────────────────────────────

    #[test]
    fn mark_evicted_resets_state() {
        let mut sys = make_system();
        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);
        sys.mark_loaded(&c, vec![10, 20, 30]);
        assert_eq!(sys.loaded_count(), 1);

        sys.mark_evicted(&c);
        assert_eq!(sys.chunk_state(&c), Some(ChunkState::Unloaded));
        assert_eq!(sys.loaded_count(), 0);

        // Verify brick_slots are cleared.
        let entry = sys.chunks.get(&c).unwrap();
        assert!(entry.brick_slots.is_empty());
    }

    // ── 14. loaded_count tracks correctly ───────────────────────────────────

    #[test]
    fn loaded_count() {
        let mut sys = StreamingSystem::new(StreamingConfig {
            load_radius: 5000.0,
            max_loads_per_frame: 100,
            ..Default::default()
        });

        let coords: Vec<IVec3> = (0..5).map(|i| IVec3::new(i, 0, 0)).collect();
        for &c in &coords {
            sys.register_chunk(c, chunk_path(c), false);
        }

        assert_eq!(sys.loaded_count(), 0);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        // Mark 3 loaded.
        for &c in &coords[..3] {
            sys.mark_loaded(&c, vec![1]);
        }
        assert_eq!(sys.loaded_count(), 3);

        // Evict one.
        sys.mark_evicted(&coords[0]);
        assert_eq!(sys.loaded_count(), 2);
    }

    // ── 15. drain returns empty on second call ──────────────────────────────

    #[test]
    fn drain_returns_empty_on_second_call() {
        let mut sys = make_system();
        let c = IVec3::ZERO;
        sys.register_chunk(c, chunk_path(c), false);

        let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
        sys.update(&camera);

        let first = sys.drain_load_requests();
        assert!(!first.is_empty());
        let second = sys.drain_load_requests();
        assert!(second.is_empty(), "second drain should be empty");
    }

    // ── 16. config_mut allows runtime changes ───────────────────────────────

    #[test]
    fn config_mut_allows_changes() {
        let mut sys = make_system();
        sys.config_mut().max_radius = 1000.0;
        assert_eq!(sys.config().max_radius, 1000.0);
    }
}
