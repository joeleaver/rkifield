//! Per-object streaming system for the v2 object-centric SDF architecture.
//!
//! Manages loading, unloading, and LOD-level transitions for SDF objects backed
//! by `.rkf` v2 asset files. The system is a **pure state machine** — it emits
//! [`LoadRequest`]s and eviction candidates that a separate async I/O layer
//! (v2-12.3) must act on. No I/O is performed here.
//!
//! # Usage
//!
//! ```rust
//! use std::collections::HashMap;
//! use rkf_runtime::object_streaming::{ObjectStreamingSystem, StreamingConfig};
//!
//! let config = StreamingConfig::default();
//! let mut sys = ObjectStreamingSystem::new(config);
//!
//! // Register objects discovered in the scene.
//! sys.register_object(1, "assets/rock.rkf".into(), 3, 1.0);
//!
//! // Each frame: update priorities then collect work.
//! let aabbs: HashMap<u32, (glam::Vec3, f32)> = HashMap::new();
//! sys.update_priorities(glam::Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);
//!
//! let requests = sys.collect_load_requests();
//! let evictions = sys.collect_eviction_candidates(0);
//! ```

use std::collections::HashMap;

use glam::Vec3;

// ---------------------------------------------------------------------------
// ObjectStreamState
// ---------------------------------------------------------------------------

/// Lifecycle state for a single streamed SDF object.
///
/// State transitions follow this DAG:
///
/// ```text
/// Unloaded ──► Loading(coarsest) ──► Loaded(coarsest)
///                                          │
///                             ┌────────────┘
///                             ▼
///                  Upgrading(current → finer) ──► Loaded(finer)
///
/// Loaded(any) ──► Evicting ──► Unloaded
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectStreamState {
    /// Asset is known but no brick data has been loaded.
    Unloaded,

    /// An async load is in-flight for the given LOD level.
    Loading {
        /// The LOD level currently being loaded (0 = coarsest).
        lod_level: usize,
    },

    /// At least the coarsest LOD is available for rendering.
    Loaded {
        /// The finest LOD level currently resident in GPU memory.
        lod_level: usize,
    },

    /// A finer LOD is being loaded in the background while the object continues
    /// to render at `current_lod`.
    Upgrading {
        /// The LOD level currently resident and in use for rendering.
        current_lod: usize,
        /// The finer LOD level being fetched.
        target_lod: usize,
    },

    /// The object has been marked for eviction and will transition to
    /// [`Unloaded`](ObjectStreamState::Unloaded) once the caller acknowledges
    /// via [`ObjectStreamingSystem::notify_evicted`].
    Evicting,

    /// An error occurred during the most recent load attempt.
    Error(String),
}

// ---------------------------------------------------------------------------
// StreamingObject
// ---------------------------------------------------------------------------

/// Per-object tracking record maintained by [`ObjectStreamingSystem`].
#[derive(Debug, Clone)]
pub struct StreamingObject {
    /// Unique identifier matching the v2 scene object ID.
    pub object_id: u32,

    /// Path to the `.rkf` v2 asset file on disk.
    pub asset_path: String,

    /// Current lifecycle state.
    pub state: ObjectStreamState,

    /// Priority score computed each frame — higher means load sooner.
    pub priority: f32,

    /// Approximate number of screen pixels the object covers this frame.
    pub screen_coverage: f32,

    /// User-set multiplier applied on top of coverage-based priority.
    ///
    /// Values > 1.0 increase priority; values < 1.0 decrease it.
    /// Default is `1.0`.
    pub importance_bias: f32,

    /// The last frame number on which this object was visible.
    ///
    /// Used for LRU eviction eligibility checks.
    pub last_visible_frame: u64,

    /// Total number of LOD levels available in the asset file.
    pub lod_count: usize,
}

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for [`ObjectStreamingSystem`].
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of objects that may be in the [`Loading`] or
    /// [`Upgrading`] state simultaneously.
    ///
    /// [`Loading`]: ObjectStreamState::Loading
    /// [`Upgrading`]: ObjectStreamState::Upgrading
    pub max_concurrent_loads: usize,

    /// Approximate byte budget for loads dispatched in a single frame.
    ///
    /// The system stops emitting [`LoadRequest`]s once the estimated cost of
    /// already-queued requests exceeds this value.
    /// Default: 4 MiB.
    pub load_budget_bytes_per_frame: usize,

    /// Number of frames an object may remain invisible before it becomes an
    /// eviction candidate.
    ///
    /// At 60 fps, the default of 300 gives ~5 seconds of grace time.
    pub eviction_grace_frames: u64,

    /// Objects whose screen coverage (in pixels) falls below this threshold
    /// are not eligible to be loaded.
    ///
    /// Default: 4.0 pixels.
    pub min_screen_coverage_to_load: f32,

    /// Screen coverage (pixels) above which a loaded object should upgrade to
    /// the next finer LOD, if one exists.
    ///
    /// Default: 128.0 pixels (roughly a 16 × 8 area on screen).
    pub lod_upgrade_threshold: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_loads: 4,
            load_budget_bytes_per_frame: 4 * 1024 * 1024,
            eviction_grace_frames: 300,
            min_screen_coverage_to_load: 4.0,
            lod_upgrade_threshold: 128.0,
        }
    }
}

// ---------------------------------------------------------------------------
// LoadRequest
// ---------------------------------------------------------------------------

/// A request for the async I/O layer to load a specific LOD level.
///
/// Requests are emitted by [`ObjectStreamingSystem::collect_load_requests`]
/// in descending priority order. The caller is expected to load the asset and
/// then call [`ObjectStreamingSystem::notify_load_complete`] or
/// [`ObjectStreamingSystem::notify_load_error`].
#[derive(Debug, Clone)]
pub struct LoadRequest {
    /// Object to load.
    pub object_id: u32,

    /// Path to the `.rkf` v2 asset file.
    pub asset_path: String,

    /// Which LOD level to load (0 = coarsest / fastest to display).
    pub lod_level: usize,

    /// Priority at the time the request was generated.
    pub priority: f32,
}

// ---------------------------------------------------------------------------
// ObjectStreamingSystem
// ---------------------------------------------------------------------------

/// Per-object streaming state machine.
///
/// Call [`update_priorities`](Self::update_priorities) once per frame to
/// refresh screen-coverage estimates, then call
/// [`collect_load_requests`](Self::collect_load_requests) and
/// [`collect_eviction_candidates`](Self::collect_eviction_candidates) to
/// obtain the work lists.
pub struct ObjectStreamingSystem {
    /// Configuration applied to all streaming decisions.
    config: StreamingConfig,

    /// All registered objects keyed by object ID.
    objects: HashMap<u32, StreamingObject>,
}

impl ObjectStreamingSystem {
    /// Create a new streaming system with the given configuration.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            objects: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /// Register an object for streaming management.
    ///
    /// If an object with `object_id` is already registered this call is
    /// ignored. The object starts in the [`Unloaded`](ObjectStreamState::Unloaded)
    /// state.
    ///
    /// # Parameters
    ///
    /// * `object_id` — unique ID matching the v2 scene object.
    /// * `asset_path` — path to the `.rkf` v2 file.
    /// * `lod_count` — number of LOD levels in the asset (must be >= 1).
    /// * `importance_bias` — priority multiplier; `1.0` is neutral.
    pub fn register_object(
        &mut self,
        object_id: u32,
        asset_path: String,
        lod_count: usize,
        importance_bias: f32,
    ) {
        self.objects.entry(object_id).or_insert_with(|| StreamingObject {
            object_id,
            asset_path,
            state: ObjectStreamState::Unloaded,
            priority: 0.0,
            screen_coverage: 0.0,
            importance_bias,
            last_visible_frame: 0,
            lod_count: lod_count.max(1),
        });
    }

    /// Remove an object from tracking.
    ///
    /// The object is removed regardless of its current state. If it was in an
    /// in-flight load the caller must ensure the I/O layer is also cancelled or
    /// that any subsequent `notify_*` calls are silently dropped (they will
    /// be — this system simply ignores unknown IDs).
    pub fn unregister_object(&mut self, object_id: u32) {
        self.objects.remove(&object_id);
    }

    // -----------------------------------------------------------------------
    // Priority update
    // -----------------------------------------------------------------------

    /// Recompute per-object priorities for this frame.
    ///
    /// This must be called once per frame **before**
    /// [`collect_load_requests`](Self::collect_load_requests).
    ///
    /// # Parameters
    ///
    /// * `camera_pos` — world-space camera position.
    /// * `viewport_height` — render target height in pixels.
    /// * `fov_y` — vertical field of view in **radians**.
    /// * `frame_number` — monotonically increasing frame counter.
    /// * `object_aabbs` — maps object ID to `(center, radius)` in world space.
    ///   Objects absent from this map receive zero coverage and are treated as
    ///   invisible this frame.
    pub fn update_priorities(
        &mut self,
        camera_pos: Vec3,
        viewport_height: f32,
        fov_y: f32,
        frame_number: u64,
        object_aabbs: &HashMap<u32, (Vec3, f32)>,
    ) {
        let half_fov_tan = (fov_y * 0.5).tan().max(f32::EPSILON);

        for obj in self.objects.values_mut() {
            if let Some(&(center, radius)) = object_aabbs.get(&obj.object_id) {
                let distance = (center - camera_pos).length().max(f32::EPSILON);
                // Projected diameter in pixels.
                let screen_pixels =
                    (radius * 2.0 * viewport_height) / (2.0 * distance * half_fov_tan);

                obj.screen_coverage = screen_pixels.max(0.0);

                if screen_pixels >= self.config.min_screen_coverage_to_load {
                    obj.last_visible_frame = frame_number;
                }
            } else {
                obj.screen_coverage = 0.0;
            }

            obj.priority = obj.screen_coverage * obj.importance_bias;
        }
    }

    // -----------------------------------------------------------------------
    // Work collection
    // -----------------------------------------------------------------------

    /// Return the load requests that should be dispatched this frame.
    ///
    /// Requests are sorted in descending priority order.  The list respects
    /// [`StreamingConfig::max_concurrent_loads`] and will not exceed the
    /// in-flight load count.  Objects below
    /// [`StreamingConfig::min_screen_coverage_to_load`] are skipped.
    ///
    /// Calling this method **mutates state**: objects that are selected for
    /// loading transition to [`Loading`](ObjectStreamState::Loading) or
    /// [`Upgrading`](ObjectStreamState::Upgrading) immediately so that
    /// subsequent calls in the same frame do not double-queue them.
    pub fn collect_load_requests(&mut self) -> Vec<LoadRequest> {
        // Count how many loads are already in flight.
        let in_flight = self.loading_count();
        let slots_available = self.config.max_concurrent_loads.saturating_sub(in_flight);

        if slots_available == 0 {
            return Vec::new();
        }

        // Gather candidates.
        let mut candidates: Vec<u32> = self
            .objects
            .values()
            .filter(|obj| {
                obj.screen_coverage >= self.config.min_screen_coverage_to_load
                    && Self::needs_load(&obj.state, obj.screen_coverage, self.config.lod_upgrade_threshold, obj.lod_count)
            })
            .map(|obj| obj.object_id)
            .collect();

        // Sort descending by priority.
        candidates.sort_by(|a, b| {
            let pa = self.objects[a].priority;
            let pb = self.objects[b].priority;
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut requests = Vec::new();
        for object_id in candidates.into_iter().take(slots_available) {
            let obj = self.objects.get_mut(&object_id).unwrap();
            let (lod_level, new_state) = Self::decide_load(
                &obj.state,
                obj.screen_coverage,
                self.config.lod_upgrade_threshold,
                obj.lod_count,
            );

            requests.push(LoadRequest {
                object_id,
                asset_path: obj.asset_path.clone(),
                lod_level,
                priority: obj.priority,
            });

            obj.state = new_state;
        }

        requests
    }

    /// Return the IDs of objects eligible for eviction this frame.
    ///
    /// An object is a candidate when **all** of the following hold:
    /// - Its state is [`Loaded`](ObjectStreamState::Loaded).
    /// - It has been invisible for at least
    ///   [`StreamingConfig::eviction_grace_frames`] frames.
    ///
    /// The caller must call [`notify_evicted`](Self::notify_evicted) after
    /// freeing the GPU memory for the object.
    pub fn collect_eviction_candidates(&self, frame_number: u64) -> Vec<u32> {
        self.objects
            .values()
            .filter(|obj| {
                matches!(obj.state, ObjectStreamState::Loaded { .. })
                    && frame_number.saturating_sub(obj.last_visible_frame)
                        >= self.config.eviction_grace_frames
            })
            .map(|obj| obj.object_id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Notification callbacks
    // -----------------------------------------------------------------------

    /// Called by the async I/O layer when a load completes successfully.
    ///
    /// Transitions the object from `Loading` to `Loaded` or from `Upgrading`
    /// to `Loaded`. Unknown object IDs are silently ignored.
    pub fn notify_load_complete(&mut self, object_id: u32, lod_level: usize) {
        if let Some(obj) = self.objects.get_mut(&object_id) {
            match obj.state {
                ObjectStreamState::Loading { lod_level: l } if l == lod_level => {
                    obj.state = ObjectStreamState::Loaded { lod_level };
                }
                ObjectStreamState::Upgrading { target_lod, .. } if target_lod == lod_level => {
                    obj.state = ObjectStreamState::Loaded { lod_level };
                }
                _ => {
                    // Stale notification (e.g. object was re-registered while
                    // load was in-flight). Move to Loaded anyway — the data is
                    // now resident.
                    obj.state = ObjectStreamState::Loaded { lod_level };
                }
            }
        }
    }

    /// Called by the async I/O layer when a load fails.
    ///
    /// Transitions the object to the [`Error`](ObjectStreamState::Error) state.
    /// Unknown object IDs are silently ignored.
    pub fn notify_load_error(&mut self, object_id: u32, error: String) {
        if let Some(obj) = self.objects.get_mut(&object_id) {
            obj.state = ObjectStreamState::Error(error);
        }
    }

    /// Called by the GPU memory manager after brick data for an object has been
    /// freed.
    ///
    /// Transitions the object from [`Evicting`](ObjectStreamState::Evicting)
    /// back to [`Unloaded`](ObjectStreamState::Unloaded).
    /// Unknown object IDs are silently ignored.
    pub fn notify_evicted(&mut self, object_id: u32) {
        if let Some(obj) = self.objects.get_mut(&object_id) {
            obj.state = ObjectStreamState::Unloaded;
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return the current state of an object, or `None` if not registered.
    pub fn get_state(&self, object_id: u32) -> Option<&ObjectStreamState> {
        self.objects.get(&object_id).map(|o| &o.state)
    }

    /// Total number of registered objects.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Number of objects currently in the `Loading` or `Upgrading` state.
    pub fn loading_count(&self) -> usize {
        self.objects
            .values()
            .filter(|o| {
                matches!(
                    o.state,
                    ObjectStreamState::Loading { .. } | ObjectStreamState::Upgrading { .. }
                )
            })
            .count()
    }

    /// Number of objects currently in the `Loaded` state at any LOD.
    pub fn loaded_count(&self) -> usize {
        self.objects
            .values()
            .filter(|o| matches!(o.state, ObjectStreamState::Loaded { .. }))
            .count()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Returns `true` if the object needs any kind of load action.
    fn needs_load(
        state: &ObjectStreamState,
        screen_coverage: f32,
        upgrade_threshold: f32,
        lod_count: usize,
    ) -> bool {
        match state {
            ObjectStreamState::Unloaded => true,
            ObjectStreamState::Loaded { lod_level } => {
                // Upgrade if there are finer LODs and coverage warrants it.
                let has_finer = lod_level + 1 < lod_count;
                has_finer && screen_coverage >= upgrade_threshold
            }
            ObjectStreamState::Error(_) => false,
            ObjectStreamState::Loading { .. }
            | ObjectStreamState::Upgrading { .. }
            | ObjectStreamState::Evicting => false,
        }
    }

    /// Determine the LOD level to request and the next state to enter.
    ///
    /// Returns `(lod_level_to_load, new_state)`.
    fn decide_load(
        state: &ObjectStreamState,
        screen_coverage: f32,
        upgrade_threshold: f32,
        lod_count: usize,
    ) -> (usize, ObjectStreamState) {
        match state {
            ObjectStreamState::Unloaded => {
                // Always start with the coarsest LOD (index 0).
                (0, ObjectStreamState::Loading { lod_level: 0 })
            }
            ObjectStreamState::Loaded { lod_level } => {
                let current = *lod_level;
                let target = current + 1;
                debug_assert!(target < lod_count);
                debug_assert!(screen_coverage >= upgrade_threshold);
                (
                    target,
                    ObjectStreamState::Upgrading {
                        current_lod: current,
                        target_lod: target,
                    },
                )
            }
            // Should never be called for these states; return a no-op.
            other => {
                // Preserve whatever LOD information exists.
                let lod = match other {
                    ObjectStreamState::Loading { lod_level } => *lod_level,
                    ObjectStreamState::Upgrading { target_lod, .. } => *target_lod,
                    _ => 0,
                };
                (lod, other.clone())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_system() -> ObjectStreamingSystem {
        ObjectStreamingSystem::new(StreamingConfig::default())
    }

    fn make_aabbs(entries: &[(u32, Vec3, f32)]) -> HashMap<u32, (Vec3, f32)> {
        entries.iter().map(|&(id, c, r)| (id, (c, r))).collect()
    }

    // -----------------------------------------------------------------------
    // 1. new_system_empty
    // -----------------------------------------------------------------------

    #[test]
    fn new_system_empty() {
        let sys = default_system();
        assert_eq!(sys.object_count(), 0);
        assert_eq!(sys.loading_count(), 0);
        assert_eq!(sys.loaded_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 2. register_and_query
    // -----------------------------------------------------------------------

    #[test]
    fn register_and_query() {
        let mut sys = default_system();
        sys.register_object(42, "rock.rkf".into(), 3, 1.0);

        assert_eq!(sys.object_count(), 1);
        assert_eq!(sys.get_state(42), Some(&ObjectStreamState::Unloaded));
        assert_eq!(sys.get_state(99), None);
    }

    /// Registering the same ID twice keeps the original entry.
    #[test]
    fn register_idempotent() {
        let mut sys = default_system();
        sys.register_object(1, "a.rkf".into(), 2, 1.0);
        sys.notify_load_complete(1, 0); // put into Loaded
        // Re-register should be a no-op.
        sys.register_object(1, "b.rkf".into(), 5, 2.0);
        // State must not be reset.
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));
    }

    // -----------------------------------------------------------------------
    // 3. unregister_removes
    // -----------------------------------------------------------------------

    #[test]
    fn unregister_removes() {
        let mut sys = default_system();
        sys.register_object(7, "tree.rkf".into(), 2, 1.0);
        assert_eq!(sys.object_count(), 1);
        sys.unregister_object(7);
        assert_eq!(sys.object_count(), 0);
        assert_eq!(sys.get_state(7), None);
    }

    /// Unregistering an ID that was never registered is a no-op.
    #[test]
    fn unregister_unknown_is_noop() {
        let mut sys = default_system();
        sys.unregister_object(999);
        assert_eq!(sys.object_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 4. update_priorities_computes_coverage
    // -----------------------------------------------------------------------

    #[test]
    fn update_priorities_computes_coverage() {
        let mut sys = default_system();
        sys.register_object(1, "a.rkf".into(), 2, 1.0);

        // Object at distance 10, radius 1, viewport_height 1080, fov_y 60°.
        // Expected: (1*2*1080) / (2*10*tan(30°)) ≈ 2160 / 11.547 ≈ 187 pixels.
        let center = Vec3::new(10.0, 0.0, 0.0);
        let camera = Vec3::ZERO;
        let aabbs = make_aabbs(&[(1, center, 1.0)]);
        sys.update_priorities(camera, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let obj = sys.objects.get(&1).unwrap();
        assert!(
            obj.screen_coverage > 100.0 && obj.screen_coverage < 300.0,
            "coverage {} out of expected range 100–300 pixels",
            obj.screen_coverage
        );
        assert!((obj.priority - obj.screen_coverage).abs() < 1e-3);
    }

    #[test]
    fn update_priorities_zero_coverage_for_absent_object() {
        let mut sys = default_system();
        sys.register_object(5, "b.rkf".into(), 1, 1.0);
        // No AABB provided for object 5.
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &HashMap::new());

        let obj = sys.objects.get(&5).unwrap();
        assert_eq!(obj.screen_coverage, 0.0);
        assert_eq!(obj.priority, 0.0);
    }

    // -----------------------------------------------------------------------
    // 5. collect_load_requests_prioritized
    // -----------------------------------------------------------------------

    #[test]
    fn collect_load_requests_prioritized() {
        let config = StreamingConfig {
            max_concurrent_loads: 10,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);

        // Register three objects at different distances.
        for id in 1_u32..=3 {
            sys.register_object(id, format!("obj{id}.rkf"), 2, 1.0);
        }

        // Object 1 is closest (highest coverage), object 3 is farthest.
        let aabbs = make_aabbs(&[
            (1, Vec3::new(2.0, 0.0, 0.0), 1.0),
            (2, Vec3::new(10.0, 0.0, 0.0), 1.0),
            (3, Vec3::new(50.0, 0.0, 0.0), 1.0),
        ]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let requests = sys.collect_load_requests();

        // All three should be requested (all above min coverage).
        assert_eq!(requests.len(), 3, "expected 3 requests, got {}", requests.len());

        // Must be in descending priority order (closest first).
        for window in requests.windows(2) {
            assert!(
                window[0].priority >= window[1].priority,
                "requests not sorted by priority: {} < {}",
                window[0].priority,
                window[1].priority
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. collect_load_requests_respects_max_concurrent
    // -----------------------------------------------------------------------

    #[test]
    fn collect_load_requests_respects_max_concurrent() {
        let config = StreamingConfig {
            max_concurrent_loads: 2,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);

        for id in 1_u32..=5 {
            sys.register_object(id, format!("obj{id}.rkf"), 2, 1.0);
        }

        let aabbs: HashMap<u32, (Vec3, f32)> = (1_u32..=5)
            .map(|id| (id, (Vec3::new(5.0, 0.0, 0.0), 1.0)))
            .collect();
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let first = sys.collect_load_requests();
        assert_eq!(first.len(), 2, "first batch: expected 2, got {}", first.len());

        // Second call — 2 are already Loading, 0 slots available.
        let second = sys.collect_load_requests();
        assert_eq!(second.len(), 0, "second batch: expected 0, got {}", second.len());
    }

    // -----------------------------------------------------------------------
    // 7. load_complete_transitions_state
    // -----------------------------------------------------------------------

    #[test]
    fn load_complete_transitions_state() {
        let mut sys = default_system();
        sys.register_object(1, "a.rkf".into(), 3, 1.0);

        // Force above min coverage.
        let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let requests = sys.collect_load_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].lod_level, 0, "first load should be coarsest LOD");
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loading { lod_level: 0 }));

        sys.notify_load_complete(1, 0);
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));
        assert_eq!(sys.loaded_count(), 1);
    }

    // -----------------------------------------------------------------------
    // 8. upgrade_triggers_on_high_coverage
    // -----------------------------------------------------------------------

    #[test]
    fn upgrade_triggers_on_high_coverage() {
        let config = StreamingConfig {
            lod_upgrade_threshold: 64.0,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);
        sys.register_object(1, "a.rkf".into(), 3, 1.0);

        // Bring object to Loaded(0).
        sys.notify_load_complete(1, 0);
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));

        // High coverage to trigger upgrade.
        let aabbs = make_aabbs(&[(1, Vec3::new(0.5, 0.0, 0.0), 1.0)]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let requests = sys.collect_load_requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].lod_level, 1, "should request LOD 1 upgrade");

        assert!(
            matches!(sys.get_state(1), Some(ObjectStreamState::Upgrading { current_lod: 0, target_lod: 1 })),
            "expected Upgrading(0→1), got {:?}",
            sys.get_state(1)
        );

        // Complete upgrade.
        sys.notify_load_complete(1, 1);
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 1 }));
    }

    // -----------------------------------------------------------------------
    // 9. eviction_candidates_respect_grace
    // -----------------------------------------------------------------------

    #[test]
    fn eviction_candidates_respect_grace() {
        let config = StreamingConfig {
            eviction_grace_frames: 10,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);
        sys.register_object(1, "a.rkf".into(), 1, 1.0);

        // Simulate: object was visible at frame 0.
        {
            let obj = sys.objects.get_mut(&1).unwrap();
            obj.last_visible_frame = 0;
            obj.state = ObjectStreamState::Loaded { lod_level: 0 };
        }

        // At frame 9 (9 frames invisible) — still within grace.
        let candidates = sys.collect_eviction_candidates(9);
        assert!(candidates.is_empty(), "should not evict within grace period");

        // At frame 10 — exactly at grace boundary.
        let candidates = sys.collect_eviction_candidates(10);
        assert_eq!(candidates, vec![1], "should evict at grace boundary");

        // At frame 100 — well past grace.
        let candidates = sys.collect_eviction_candidates(100);
        assert_eq!(candidates, vec![1]);
    }

    // -----------------------------------------------------------------------
    // 10. no_eviction_when_visible
    // -----------------------------------------------------------------------

    #[test]
    fn no_eviction_when_visible() {
        let config = StreamingConfig {
            eviction_grace_frames: 5,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);
        sys.register_object(1, "a.rkf".into(), 1, 1.0);

        // Bring to Loaded.
        {
            let obj = sys.objects.get_mut(&1).unwrap();
            obj.state = ObjectStreamState::Loaded { lod_level: 0 };
        }

        // Update visibility to frame 50.
        let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 50, &aabbs);

        // Check eviction at frame 52 — only 2 frames since last visible.
        let candidates = sys.collect_eviction_candidates(52);
        assert!(candidates.is_empty(), "visible object should not be evicted");
    }

    // -----------------------------------------------------------------------
    // 11. load_error_handling
    // -----------------------------------------------------------------------

    #[test]
    fn load_error_handling() {
        let mut sys = default_system();
        sys.register_object(1, "missing.rkf".into(), 2, 1.0);

        // Trigger load.
        let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);
        sys.collect_load_requests();

        sys.notify_load_error(1, "file not found".into());
        assert!(
            matches!(sys.get_state(1), Some(ObjectStreamState::Error(msg)) if msg == "file not found"),
            "expected Error state, got {:?}",
            sys.get_state(1)
        );

        // Error objects must not be re-queued by collect_load_requests.
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 1, &aabbs);
        let requests = sys.collect_load_requests();
        assert!(requests.is_empty(), "errored objects must not be re-requested");
    }

    /// notify_load_error for an unknown ID is silently ignored.
    #[test]
    fn load_error_unknown_id_ignored() {
        let mut sys = default_system();
        sys.notify_load_error(999, "boom".into()); // must not panic
        assert_eq!(sys.object_count(), 0);
    }

    // -----------------------------------------------------------------------
    // 12. loading_count_tracking
    // -----------------------------------------------------------------------

    #[test]
    fn loading_count_tracking() {
        let config = StreamingConfig {
            max_concurrent_loads: 10,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);

        for id in 1_u32..=3 {
            sys.register_object(id, format!("o{id}.rkf"), 2, 1.0);
        }

        let aabbs: HashMap<u32, (Vec3, f32)> = (1_u32..=3)
            .map(|id| (id, (Vec3::new(2.0, 0.0, 0.0), 1.0)))
            .collect();
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        assert_eq!(sys.loading_count(), 0);
        sys.collect_load_requests();
        assert_eq!(sys.loading_count(), 3);

        sys.notify_load_complete(1, 0);
        assert_eq!(sys.loading_count(), 2);
        assert_eq!(sys.loaded_count(), 1);

        sys.notify_load_complete(2, 0);
        sys.notify_load_complete(3, 0);
        assert_eq!(sys.loading_count(), 0);
        assert_eq!(sys.loaded_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Additional edge-case tests
    // -----------------------------------------------------------------------

    /// Evicting an object that was Loaded transitions it to Unloaded.
    #[test]
    fn eviction_lifecycle() {
        let config = StreamingConfig {
            eviction_grace_frames: 0,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);
        sys.register_object(1, "a.rkf".into(), 1, 1.0);
        sys.notify_load_complete(1, 0);
        assert_eq!(sys.loaded_count(), 1);

        // last_visible_frame is 0, frame_number is 0 → grace = 0 → evict.
        let candidates = sys.collect_eviction_candidates(0);
        assert_eq!(candidates, vec![1]);

        // Simulate caller marking for eviction (state set externally here since
        // the eviction-initiation step is the caller's responsibility).
        sys.objects.get_mut(&1).unwrap().state = ObjectStreamState::Evicting;
        sys.notify_evicted(1);
        assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Unloaded));
    }

    /// Objects with zero LODs are clamped to lod_count=1.
    #[test]
    fn lod_count_clamped_to_one() {
        let mut sys = default_system();
        sys.register_object(1, "a.rkf".into(), 0, 1.0);
        assert_eq!(sys.objects[&1].lod_count, 1);
    }

    /// Importance bias scales priority correctly.
    #[test]
    fn importance_bias_scales_priority() {
        let config = StreamingConfig {
            max_concurrent_loads: 10,
            ..Default::default()
        };
        let mut sys = ObjectStreamingSystem::new(config);
        // Two objects at the same distance but different importance bias.
        sys.register_object(1, "a.rkf".into(), 2, 2.0); // 2× bias
        sys.register_object(2, "b.rkf".into(), 2, 1.0); // 1× bias

        let aabbs = make_aabbs(&[
            (1, Vec3::new(10.0, 0.0, 0.0), 1.0),
            (2, Vec3::new(10.0, 0.0, 0.0), 1.0),
        ]);
        sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

        let p1 = sys.objects[&1].priority;
        let p2 = sys.objects[&2].priority;
        assert!(
            (p1 - 2.0 * p2).abs() < 1e-3,
            "priority with 2× bias should be double: {p1} vs 2×{p2}"
        );
    }
}
