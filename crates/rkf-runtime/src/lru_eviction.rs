//! Per-object LRU eviction system for the v2 RKIField SDF engine.
//!
//! When the brick pool approaches capacity, the eviction system selects which
//! objects to remove or demote to a coarser LOD tier. Eviction is per-object
//! (not per-chunk), and the policy prefers LOD demotion over full removal to
//! preserve scene coverage at lower fidelity.
//!
//! # Typical usage
//!
//! ```rust
//! use rkf_runtime::lru_eviction::{EvictionPolicy, LruEvictionTracker, ObjectUsageEntry};
//!
//! let mut tracker = LruEvictionTracker::new(EvictionPolicy::default());
//!
//! tracker.track_object(ObjectUsageEntry {
//!     object_id: 1,
//!     loaded_lod: 0,
//!     lod_count: 3,
//!     brick_count: 512,
//!     load_frame: 0,
//!     last_access_frame: 0,
//!     importance: 1.0,
//! });
//!
//! tracker.touch(1, 10);
//!
//! if tracker.should_evict(0.95) {
//!     let actions = tracker.select_evictions(0.95, 10);
//!     // apply actions to brick pool...
//! }
//! ```

use std::collections::HashMap;

// ─── EvictionPolicy ────────────────────────────────────────────────────────

/// Tuning parameters that control when and how objects are evicted from the
/// brick pool.
#[derive(Debug, Clone)]
pub struct EvictionPolicy {
    /// Pool utilization fraction above which eviction is triggered.
    ///
    /// For example, `0.9` means "start evicting when 90% of the pool is in
    /// use". Must be greater than `low_watermark`.
    pub high_watermark: f32,

    /// Pool utilization fraction at which eviction stops.
    ///
    /// Eviction candidates are accumulated until estimated utilization falls
    /// below this threshold. Must be less than `high_watermark`.
    pub low_watermark: f32,

    /// Minimum number of frames an object must have been loaded before it is
    /// eligible for eviction.
    ///
    /// This prevents thrashing — newly loaded objects are given time to
    /// actually be used before being evicted on the next tight frame.
    pub min_age_frames: u64,

    /// When `true`, objects that have a coarser LOD available are demoted
    /// (swapped to that LOD) instead of being fully evicted.
    ///
    /// Full eviction is used as a fallback when no coarser LOD exists or when
    /// demotion alone is insufficient to reach the low watermark.
    pub prefer_demote: bool,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            high_watermark: 0.9,
            low_watermark: 0.8,
            min_age_frames: 120,
            prefer_demote: true,
        }
    }
}

// ─── ObjectUsageEntry ──────────────────────────────────────────────────────

/// Per-object state maintained by the eviction tracker.
///
/// One entry exists for every object currently occupying bricks in the pool.
#[derive(Debug, Clone)]
pub struct ObjectUsageEntry {
    /// Unique object identifier (matches the v2 scene object ID).
    pub object_id: u32,

    /// LOD level currently loaded into the brick pool.
    ///
    /// `0` is the finest (highest-detail) level; higher indices are coarser.
    pub loaded_lod: usize,

    /// Total number of LOD levels available for this object.
    ///
    /// When `loaded_lod + 1 < lod_count` a coarser LOD is available for
    /// demotion.
    pub lod_count: usize,

    /// Number of bricks this object currently occupies in the pool.
    pub brick_count: u32,

    /// Frame number when this LOD was loaded.
    ///
    /// Used together with [`EvictionPolicy::min_age_frames`] to gate eviction
    /// eligibility.
    pub load_frame: u64,

    /// Frame number of the most recent access (render, touch, or load).
    ///
    /// The primary LRU key: objects with lower values are evicted first.
    pub last_access_frame: u64,

    /// Importance bias in the range `(0, ∞)`.
    ///
    /// Higher values make the object harder to evict. The eviction priority
    /// score is `last_access_frame * importance`, so doubling `importance`
    /// doubles the score and reduces eviction urgency. A value of `1.0`
    /// applies no bias.
    pub importance: f32,
}

// ─── EvictionAction ────────────────────────────────────────────────────────

/// An action the brick-pool manager should perform to reclaim capacity.
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionAction {
    /// Replace the currently loaded LOD with a coarser one.
    ///
    /// The brick-pool manager should unload `from_lod` bricks and load
    /// `to_lod` bricks for the given object.
    Demote {
        /// Object to demote.
        object_id: u32,
        /// Currently loaded LOD (finer).
        from_lod: usize,
        /// Target LOD to switch to (coarser).
        to_lod: usize,
    },

    /// Fully remove the object from the brick pool.
    ///
    /// The brick-pool manager should release all bricks owned by the object.
    Evict {
        /// Object to evict.
        object_id: u32,
    },
}

// ─── LruEvictionTracker ────────────────────────────────────────────────────

/// Tracks per-object brick-pool usage and selects eviction candidates.
///
/// The tracker maintains a [`HashMap`] of [`ObjectUsageEntry`] values, updated
/// each frame via [`touch`](Self::touch) calls from the render loop. When the
/// pool pressure is high, [`select_evictions`](Self::select_evictions) returns
/// a minimal ordered set of [`EvictionAction`]s that bring estimated
/// utilization below the low watermark.
///
/// This is a pure CPU data structure — it does not hold any GPU resources and
/// performs no allocation on the critical path (only `select_evictions` sorts
/// a temporary vector).
pub struct LruEvictionTracker {
    policy: EvictionPolicy,
    entries: HashMap<u32, ObjectUsageEntry>,
}

impl LruEvictionTracker {
    /// Create a new tracker with the supplied eviction policy.
    pub fn new(policy: EvictionPolicy) -> Self {
        Self {
            policy,
            entries: HashMap::new(),
        }
    }

    // ── Mutation ────────────────────────────────────────────────────────

    /// Begin tracking a newly loaded object.
    ///
    /// If an entry with the same `object_id` already exists it is replaced.
    pub fn track_object(&mut self, entry: ObjectUsageEntry) {
        self.entries.insert(entry.object_id, entry);
    }

    /// Stop tracking an object that has been fully removed from the scene.
    ///
    /// Does nothing if `object_id` is not currently tracked.
    pub fn untrack_object(&mut self, object_id: u32) {
        self.entries.remove(&object_id);
    }

    /// Record that `object_id` was accessed on `frame`.
    ///
    /// This updates `last_access_frame` and is the primary signal used by the
    /// LRU ordering. Call once per frame for every object that is visible or
    /// otherwise relevant.
    ///
    /// Does nothing if `object_id` is not currently tracked.
    pub fn touch(&mut self, object_id: u32, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&object_id) {
            entry.last_access_frame = frame;
        }
    }

    /// Update the tracked LOD level and brick count after a demotion completes.
    ///
    /// Call this after the brick-pool manager has finished swapping to the
    /// coarser LOD so that subsequent eviction decisions reflect the new state.
    ///
    /// Does nothing if `object_id` is not currently tracked.
    pub fn update_lod(&mut self, object_id: u32, new_lod: usize, new_brick_count: u32) {
        if let Some(entry) = self.entries.get_mut(&object_id) {
            entry.loaded_lod = new_lod;
            entry.brick_count = new_brick_count;
        }
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Returns `true` when `utilization` exceeds the high watermark and
    /// eviction should begin.
    pub fn should_evict(&self, utilization: f32) -> bool {
        utilization > self.policy.high_watermark
    }

    /// Total number of bricks currently occupied by all tracked objects.
    pub fn total_tracked_bricks(&self) -> u32 {
        self.entries.values().map(|e| e.brick_count).sum()
    }

    /// Number of objects currently tracked.
    pub fn tracked_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a reference to the usage entry for `object_id`, or `None` if
    /// the object is not tracked.
    pub fn get_entry(&self, object_id: u32) -> Option<&ObjectUsageEntry> {
        self.entries.get(&object_id)
    }

    /// Select the minimal set of eviction actions needed to bring estimated
    /// pool utilization below the low watermark.
    ///
    /// # Algorithm
    ///
    /// 1. Filter to objects whose age (`current_frame - load_frame`) is at
    ///    least [`EvictionPolicy::min_age_frames`].
    /// 2. Sort by eviction priority score `last_access_frame * importance`
    ///    (ascending — lowest score is evicted first; higher importance raises
    ///    the score, making an object harder to evict).
    /// 3. For each candidate:
    ///    - If [`EvictionPolicy::prefer_demote`] is `true` and a coarser LOD
    ///      exists (`loaded_lod + 1 < lod_count`): emit
    ///      [`EvictionAction::Demote`] and estimate brick savings as
    ///      `brick_count * 0.75`.
    ///    - Otherwise: emit [`EvictionAction::Evict`] and claim full
    ///      `brick_count` savings.
    /// 4. Stop once estimated utilization falls below the low watermark.
    ///
    /// Returns an empty `Vec` if no eviction is needed or no eligible
    /// candidates exist.
    pub fn select_evictions(
        &self,
        current_utilization: f32,
        current_frame: u64,
    ) -> Vec<EvictionAction> {
        if !self.should_evict(current_utilization) {
            return Vec::new();
        }

        // Collect eviction-eligible entries sorted by priority score.
        let mut candidates: Vec<&ObjectUsageEntry> = self
            .entries
            .values()
            .filter(|e| {
                let age = current_frame.saturating_sub(e.load_frame);
                age >= self.policy.min_age_frames
            })
            .collect();

        // Lower score = evict sooner.  Guard against importance <= 0.
        candidates.sort_by(|a, b| {
            let score_a = priority_score(a);
            let score_b = priority_score(b);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Accumulate actions until estimated utilization drops below the low
        // watermark.  We model utilization as a linear fraction of total pool
        // bricks; savings are subtracted from the numerator.
        let mut actions = Vec::new();
        let mut estimated_utilization = current_utilization;

        for entry in candidates {
            if estimated_utilization <= self.policy.low_watermark {
                break;
            }

            let can_demote = self.policy.prefer_demote
                && entry.loaded_lod + 1 < entry.lod_count;

            let (action, savings_bricks) = if can_demote {
                let action = EvictionAction::Demote {
                    object_id: entry.object_id,
                    from_lod: entry.loaded_lod,
                    to_lod: entry.loaded_lod + 1,
                };
                // Coarser LOD typically uses ~1/4 – 1/8 the bricks; model as
                // 75% savings (retain 25%).
                let savings = (entry.brick_count as f32 * 0.75) as u32;
                (action, savings)
            } else {
                let action = EvictionAction::Evict {
                    object_id: entry.object_id,
                };
                (action, entry.brick_count)
            };

            // Update estimated utilization.  We scale the savings relative to
            // the total tracked brick budget so that the fraction decreases
            // meaningfully as objects are removed.
            let total = self.total_tracked_bricks().max(1) as f32;
            let utilization_delta = savings_bricks as f32 / total
                * (self.policy.high_watermark - self.policy.low_watermark + 0.1);
            estimated_utilization -= utilization_delta;

            actions.push(action);
        }

        actions
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Compute the eviction priority score for an entry.
///
/// Lower score → higher eviction priority (evicted first).
///
/// The score is `last_access_frame * importance`:
/// - A stale (old) object has a small `last_access_frame` → small score →
///   evicted sooner.
/// - A high-importance object multiplies the score upward, making it harder
///   to evict relative to a same-age object with lower importance.
///
/// `importance` is clamped to a minimum of `f32::EPSILON` to avoid
/// zero-multiplication edge cases for non-positive values.
#[inline]
fn priority_score(entry: &ObjectUsageEntry) -> f32 {
    let importance = entry.importance.max(f32::EPSILON);
    entry.last_access_frame as f32 * importance
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        object_id: u32,
        loaded_lod: usize,
        lod_count: usize,
        brick_count: u32,
        load_frame: u64,
        last_access_frame: u64,
        importance: f32,
    ) -> ObjectUsageEntry {
        ObjectUsageEntry {
            object_id,
            loaded_lod,
            lod_count,
            brick_count,
            load_frame,
            last_access_frame,
            importance,
        }
    }

    // 1. new_tracker_empty — no objects tracked initially.
    #[test]
    fn new_tracker_empty() {
        let tracker = LruEvictionTracker::new(EvictionPolicy::default());
        assert_eq!(tracker.tracked_count(), 0);
        assert_eq!(tracker.total_tracked_bricks(), 0);
    }

    // 2. track_and_query — track object, verify entry fields.
    #[test]
    fn track_and_query() {
        let mut tracker = LruEvictionTracker::new(EvictionPolicy::default());
        let entry = make_entry(42, 0, 3, 256, 10, 15, 1.0);
        tracker.track_object(entry);

        assert_eq!(tracker.tracked_count(), 1);
        let queried = tracker.get_entry(42).expect("entry missing");
        assert_eq!(queried.object_id, 42);
        assert_eq!(queried.loaded_lod, 0);
        assert_eq!(queried.lod_count, 3);
        assert_eq!(queried.brick_count, 256);
        assert_eq!(queried.load_frame, 10);
        assert_eq!(queried.last_access_frame, 15);
        assert!((queried.importance - 1.0).abs() < f32::EPSILON);
    }

    // 3. untrack_removes — untrack removes entry, leaving tracker empty.
    #[test]
    fn untrack_removes() {
        let mut tracker = LruEvictionTracker::new(EvictionPolicy::default());
        tracker.track_object(make_entry(1, 0, 1, 100, 0, 0, 1.0));
        assert_eq!(tracker.tracked_count(), 1);

        tracker.untrack_object(1);
        assert_eq!(tracker.tracked_count(), 0);
        assert!(tracker.get_entry(1).is_none());
    }

    // 4. touch_updates_access_frame — touch updates last_access_frame.
    #[test]
    fn touch_updates_access_frame() {
        let mut tracker = LruEvictionTracker::new(EvictionPolicy::default());
        tracker.track_object(make_entry(7, 0, 1, 64, 0, 5, 1.0));

        tracker.touch(7, 100);
        assert_eq!(tracker.get_entry(7).unwrap().last_access_frame, 100);
    }

    // 5. should_evict_above_watermark — returns true when above high watermark.
    #[test]
    fn should_evict_above_watermark() {
        let tracker = LruEvictionTracker::new(EvictionPolicy::default());
        // Default high watermark is 0.9.
        assert!(tracker.should_evict(0.91));
        assert!(tracker.should_evict(1.0));
    }

    // 6. should_evict_below_watermark — returns false when below high watermark.
    #[test]
    fn should_evict_below_watermark() {
        let tracker = LruEvictionTracker::new(EvictionPolicy::default());
        assert!(!tracker.should_evict(0.85));
        assert!(!tracker.should_evict(0.9)); // exactly at watermark is not above it
        assert!(!tracker.should_evict(0.0));
    }

    // 7. select_evictions_oldest_first — least-recently-accessed objects are
    //    selected before more recent ones (single-LOD objects, forced evict).
    #[test]
    fn select_evictions_oldest_first() {
        let policy = EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.0, // keep going until all eligible are chosen
            min_age_frames: 0,
            prefer_demote: false, // force Evict actions so ordering is clear
        };
        let mut tracker = LruEvictionTracker::new(policy);

        // Three objects: last accessed at frames 10, 5, 20.
        tracker.track_object(make_entry(1, 0, 1, 100, 0, 10, 1.0));
        tracker.track_object(make_entry(2, 0, 1, 100, 0, 5, 1.0));
        tracker.track_object(make_entry(3, 0, 1, 100, 0, 20, 1.0));

        let actions = tracker.select_evictions(0.95, 1000);

        // Should be non-empty and the first action should target object 2
        // (last_access_frame=5, lowest score).
        assert!(!actions.is_empty());
        match &actions[0] {
            EvictionAction::Evict { object_id } => assert_eq!(*object_id, 2),
            other => panic!("expected Evict for object 2, got {:?}", other),
        }
        // Second action should target object 1 (last_access=10).
        if actions.len() > 1 {
            match &actions[1] {
                EvictionAction::Evict { object_id } => assert_eq!(*object_id, 1),
                other => panic!("expected Evict for object 1, got {:?}", other),
            }
        }
    }

    // 8. select_evictions_prefer_demote — multi-LOD objects get Demote action
    //    rather than Evict when prefer_demote is true.
    #[test]
    fn select_evictions_prefer_demote() {
        let policy = EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.0,
            min_age_frames: 0,
            prefer_demote: true,
        };
        let mut tracker = LruEvictionTracker::new(policy);

        // Object with 3 LOD levels — currently at LOD 0.
        tracker.track_object(make_entry(10, 0, 3, 512, 0, 1, 1.0));

        let actions = tracker.select_evictions(0.95, 1000);

        assert!(!actions.is_empty());
        match &actions[0] {
            EvictionAction::Demote {
                object_id,
                from_lod,
                to_lod,
            } => {
                assert_eq!(*object_id, 10);
                assert_eq!(*from_lod, 0);
                assert_eq!(*to_lod, 1);
            }
            other => panic!("expected Demote, got {:?}", other),
        }
    }

    // 9. select_evictions_respects_min_age — objects loaded fewer than
    //    min_age_frames frames ago are excluded.
    #[test]
    fn select_evictions_respects_min_age() {
        let policy = EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.0,
            min_age_frames: 120,
            prefer_demote: false,
        };
        let mut tracker = LruEvictionTracker::new(policy);

        let current_frame = 100u64;

        // Object loaded at frame 0 — age is 100, below min_age_frames=120.
        tracker.track_object(make_entry(1, 0, 1, 512, 0, 0, 1.0));

        let actions = tracker.select_evictions(0.95, current_frame);

        // Should produce no actions because the object is too young.
        assert!(
            actions.is_empty(),
            "expected no actions for young object, got {:?}",
            actions
        );
    }

    // 10. select_evictions_stops_at_low_watermark — eviction stops once
    //     estimated utilization would drop below the low watermark.
    #[test]
    fn select_evictions_stops_at_low_watermark() {
        let policy = EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.8,
            min_age_frames: 0,
            prefer_demote: false,
        };
        let mut tracker = LruEvictionTracker::new(policy);

        // Many objects with equal bricks — well above the watermark.
        for id in 1..=20u32 {
            tracker.track_object(make_entry(id, 0, 1, 100, 0, id as u64, 1.0));
        }

        // Utilization just above the high watermark.
        let actions = tracker.select_evictions(0.92, 1000);

        // We should not evict ALL 20 objects — some should remain after we
        // cross the low watermark.
        assert!(!actions.is_empty(), "expected at least one eviction");
        assert!(
            actions.len() < 20,
            "expected fewer than all objects evicted, got {}",
            actions.len()
        );
    }

    // 11. importance_affects_priority — high importance objects are evicted
    //     last compared to low importance objects with the same access frame.
    #[test]
    fn importance_affects_priority() {
        let policy = EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.0,
            min_age_frames: 0,
            prefer_demote: false,
        };
        let mut tracker = LruEvictionTracker::new(policy);

        // Both objects last accessed at the same frame.  Object 2 has 10×
        // higher importance so its score is 10× larger (harder to evict).
        tracker.track_object(make_entry(1, 0, 1, 100, 0, 50, 1.0));
        tracker.track_object(make_entry(2, 0, 1, 100, 0, 50, 10.0));

        let actions = tracker.select_evictions(0.95, 1000);

        assert!(!actions.is_empty());
        // Object 1 (low importance) should appear first.
        match &actions[0] {
            EvictionAction::Evict { object_id } => assert_eq!(*object_id, 1),
            other => panic!("expected Evict for object 1, got {:?}", other),
        }
    }

    // 12. update_lod_changes_tracking — update_lod modifies loaded_lod and
    //     brick_count in the stored entry.
    #[test]
    fn update_lod_changes_tracking() {
        let mut tracker = LruEvictionTracker::new(EvictionPolicy::default());
        tracker.track_object(make_entry(5, 0, 3, 512, 0, 10, 1.0));

        tracker.update_lod(5, 1, 128);

        let entry = tracker.get_entry(5).expect("entry missing after update");
        assert_eq!(entry.loaded_lod, 1);
        assert_eq!(entry.brick_count, 128);
        // Other fields remain unchanged.
        assert_eq!(entry.object_id, 5);
        assert_eq!(entry.lod_count, 3);
    }
}
