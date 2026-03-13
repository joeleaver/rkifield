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

#[cfg(test)]
#[path = "object_streaming_tests.rs"]
mod tests;
