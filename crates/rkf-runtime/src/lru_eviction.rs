//! LRU eviction policy for the brick pool.
//!
//! When the brick pool fills up, the streaming system needs to evict the least
//! recently used bricks to make room for new ones.  [`LruTracker`] maintains
//! per-slot access timestamps and, given the pool's current utilization,
//! selects eviction candidates according to an [`EvictionPolicy`].
//!
//! The tracker does **not** own the pool — it only decides *which* slots to
//! evict.  The caller is responsible for calling `pool.deallocate()` on the
//! returned slots and updating the spatial index.

use std::collections::HashMap;

use glam::IVec3;

/// Tracks last-access info for a single pool slot.
#[derive(Debug, Clone, Copy)]
pub struct LruEntry {
    /// Frame number when this slot was last accessed.
    pub last_frame: u64,
    /// Chunk coordinates that own this slot (for eviction callback).
    pub chunk_coords: IVec3,
}

/// Configuration for the LRU eviction policy.
#[derive(Debug, Clone)]
pub struct EvictionPolicy {
    /// Start evicting when pool utilization exceeds this fraction (0.0-1.0).
    ///
    /// Default: 0.9 (90%).
    pub high_watermark: f32,
    /// Stop evicting when utilization drops below this fraction (0.0-1.0).
    ///
    /// Default: 0.8 (80%).
    pub low_watermark: f32,
    /// Minimum frames a brick must be unused before it is eligible for eviction.
    ///
    /// Default: 120 (~2 seconds at 60 fps).
    pub min_age_frames: u64,
    /// Maximum number of slots to evict in a single pass.
    ///
    /// Default: 64.
    pub max_evictions_per_frame: usize,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            high_watermark: 0.9,
            low_watermark: 0.8,
            min_age_frames: 120,
            max_evictions_per_frame: 64,
        }
    }
}

/// Result of an eviction pass.
#[derive(Debug)]
pub struct EvictionResult {
    /// Slots that were selected for eviction, grouped by chunk coordinates.
    pub evicted: Vec<(IVec3, Vec<u32>)>,
    /// Total number of slots evicted.
    pub total_evicted: usize,
}

/// Tracks per-slot LRU data and performs eviction decisions.
///
/// Call [`track`](Self::track) when a brick slot is allocated and
/// [`touch`](Self::touch) each frame the slot is accessed.  Call
/// [`find_eviction_candidates`](Self::find_eviction_candidates) (or the
/// convenience wrapper [`run_eviction`](Self::run_eviction)) to obtain a list
/// of slots eligible for eviction based on the configured [`EvictionPolicy`].
pub struct LruTracker {
    /// Per-slot tracking data.  Only allocated slots are tracked.
    entries: HashMap<u32, LruEntry>,
    /// Eviction policy configuration.
    policy: EvictionPolicy,
}

impl LruTracker {
    /// Create a new tracker with the given eviction policy.
    pub fn new(policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            policy,
        }
    }

    /// Register a newly-allocated slot for LRU tracking.
    ///
    /// Called when a brick is loaded into the pool.
    pub fn track(&mut self, slot: u32, chunk_coords: IVec3, frame: u64) {
        self.entries.insert(
            slot,
            LruEntry {
                last_frame: frame,
                chunk_coords,
            },
        );
    }

    /// Remove a slot from LRU tracking.
    ///
    /// Called when a brick is deallocated from the pool.
    pub fn untrack(&mut self, slot: u32) {
        self.entries.remove(&slot);
    }

    /// Update the last-accessed frame for a single slot.
    ///
    /// Called each frame for every slot that is actively being ray-marched or
    /// otherwise referenced.
    pub fn touch(&mut self, slot: u32, frame: u64) {
        if let Some(entry) = self.entries.get_mut(&slot) {
            entry.last_frame = frame;
        }
    }

    /// Update the last-accessed frame for **all** slots belonging to a chunk.
    ///
    /// Convenience method for the streaming system, which tracks access at
    /// chunk granularity.
    pub fn touch_chunk(&mut self, chunk_coords: &IVec3, frame: u64) {
        for entry in self.entries.values_mut() {
            if entry.chunk_coords == *chunk_coords {
                entry.last_frame = frame;
            }
        }
    }

    /// Find slots eligible for eviction.
    ///
    /// Returns an empty vec if utilization is at or below `high_watermark`.
    /// Otherwise returns up to enough slots to bring utilization down to
    /// `low_watermark`, capped by [`EvictionPolicy::max_evictions_per_frame`].
    /// Candidates are sorted oldest-first (lowest `last_frame`).
    pub fn find_eviction_candidates(
        &self,
        pool_capacity: u32,
        pool_allocated: u32,
        current_frame: u64,
    ) -> Vec<u32> {
        if pool_capacity == 0 {
            return Vec::new();
        }

        let utilization = pool_allocated as f32 / pool_capacity as f32;
        if utilization <= self.policy.high_watermark {
            return Vec::new();
        }

        // How many slots must we free to reach low_watermark?
        let target_allocated = (self.policy.low_watermark * pool_capacity as f32).floor() as u32;
        let slots_to_free = pool_allocated.saturating_sub(target_allocated) as usize;

        // Collect candidates: slots whose age exceeds min_age_frames.
        let mut candidates: Vec<(u32, u64)> = self
            .entries
            .iter()
            .filter_map(|(&slot, entry)| {
                let age = current_frame.saturating_sub(entry.last_frame);
                if age >= self.policy.min_age_frames {
                    Some((slot, entry.last_frame))
                } else {
                    None
                }
            })
            .collect();

        // Sort oldest-first (lowest last_frame first).
        candidates.sort_by_key(|&(_, frame)| frame);

        // Take the minimum of: slots needed, max per frame, available candidates.
        let count = slots_to_free
            .min(self.policy.max_evictions_per_frame)
            .min(candidates.len());

        candidates[..count].iter().map(|&(slot, _)| slot).collect()
    }

    /// Convenience: find candidates, group by chunk, untrack them, return result.
    ///
    /// The caller is responsible for actually calling `pool.deallocate()` on
    /// each returned slot and clearing the corresponding spatial index entries.
    pub fn run_eviction(
        &mut self,
        pool_capacity: u32,
        pool_allocated: u32,
        current_frame: u64,
    ) -> EvictionResult {
        let candidates =
            self.find_eviction_candidates(pool_capacity, pool_allocated, current_frame);

        // Group by chunk coords.
        let mut chunk_map: HashMap<IVec3, Vec<u32>> = HashMap::new();
        for &slot in &candidates {
            if let Some(entry) = self.entries.get(&slot) {
                chunk_map
                    .entry(entry.chunk_coords)
                    .or_default()
                    .push(slot);
            }
        }

        let total_evicted = candidates.len();

        // Untrack all evicted slots.
        for &slot in &candidates {
            self.untrack(slot);
        }

        let evicted: Vec<(IVec3, Vec<u32>)> = chunk_map.into_iter().collect();

        EvictionResult {
            evicted,
            total_evicted,
        }
    }

    /// Number of slots currently being tracked.
    pub fn tracked_count(&self) -> usize {
        self.entries.len()
    }

    /// Look up the LRU entry for a specific slot.
    pub fn entry(&self, slot: u32) -> Option<&LruEntry> {
        self.entries.get(&slot)
    }

    /// Immutable access to the eviction policy.
    pub fn policy(&self) -> &EvictionPolicy {
        &self.policy
    }

    /// Mutable access to the eviction policy for runtime tuning.
    pub fn policy_mut(&mut self) -> &mut EvictionPolicy {
        &mut self.policy
    }

    /// Get all tracked slots belonging to the given chunk.
    pub fn slots_for_chunk(&self, chunk_coords: &IVec3) -> Vec<u32> {
        self.entries
            .iter()
            .filter_map(|(&slot, entry)| {
                if entry.chunk_coords == *chunk_coords {
                    Some(slot)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    /// Helper: create a tracker with default policy.
    fn default_tracker() -> LruTracker {
        LruTracker::new(EvictionPolicy::default())
    }

    // ── 1. new_tracker_empty ────────────────────────────────────────────────

    #[test]
    fn new_tracker_empty() {
        let tracker = default_tracker();
        assert_eq!(tracker.tracked_count(), 0);
    }

    // ── 2. track_and_untrack ────────────────────────────────────────────────

    #[test]
    fn track_and_untrack() {
        let mut tracker = default_tracker();
        let chunk = IVec3::new(1, 2, 3);

        tracker.track(0, chunk, 10);
        assert_eq!(tracker.tracked_count(), 1);
        assert!(tracker.entry(0).is_some());
        assert_eq!(tracker.entry(0).unwrap().last_frame, 10);
        assert_eq!(tracker.entry(0).unwrap().chunk_coords, chunk);

        tracker.untrack(0);
        assert_eq!(tracker.tracked_count(), 0);
        assert!(tracker.entry(0).is_none());
    }

    // ── 3. touch_updates_frame ──────────────────────────────────────────────

    #[test]
    fn touch_updates_frame() {
        let mut tracker = default_tracker();
        tracker.track(5, IVec3::ZERO, 0);
        assert_eq!(tracker.entry(5).unwrap().last_frame, 0);

        tracker.touch(5, 10);
        assert_eq!(tracker.entry(5).unwrap().last_frame, 10);
    }

    // ── 4. touch_chunk_updates_all ──────────────────────────────────────────

    #[test]
    fn touch_chunk_updates_all() {
        let mut tracker = default_tracker();
        let chunk_a = IVec3::new(1, 0, 0);
        let chunk_b = IVec3::new(2, 0, 0);

        // 3 slots for chunk_a, 1 for chunk_b.
        tracker.track(10, chunk_a, 0);
        tracker.track(11, chunk_a, 0);
        tracker.track(12, chunk_a, 0);
        tracker.track(20, chunk_b, 0);

        tracker.touch_chunk(&chunk_a, 50);

        assert_eq!(tracker.entry(10).unwrap().last_frame, 50);
        assert_eq!(tracker.entry(11).unwrap().last_frame, 50);
        assert_eq!(tracker.entry(12).unwrap().last_frame, 50);
        // chunk_b should be untouched.
        assert_eq!(tracker.entry(20).unwrap().last_frame, 0);
    }

    // ── 5. no_eviction_below_watermark ──────────────────────────────────────

    #[test]
    fn no_eviction_below_watermark() {
        let mut tracker = default_tracker(); // high_watermark = 0.9
        // Pool capacity 100, allocated 50 (50% utilization).
        tracker.track(0, IVec3::ZERO, 0);

        let candidates = tracker.find_eviction_candidates(100, 50, 1000);
        assert!(candidates.is_empty(), "should not evict below high watermark");
    }

    // ── 6. eviction_above_watermark ─────────────────────────────────────────

    #[test]
    fn eviction_above_watermark() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.8,
            min_age_frames: 10,
            max_evictions_per_frame: 64,
        });

        // Pool capacity 100, we'll allocate 95 slots (95% utilization).
        // All tracked at frame 0, current frame 100 → age = 100 > min_age 10.
        for i in 0..95u32 {
            tracker.track(i, IVec3::new(i as i32, 0, 0), 0);
        }

        let candidates = tracker.find_eviction_candidates(100, 95, 100);
        assert!(
            !candidates.is_empty(),
            "should evict when above high watermark"
        );
        // Need to drop from 95 to 80 (low_watermark * 100) = free 15 slots.
        assert_eq!(candidates.len(), 15);
    }

    // ── 7. min_age_respected ────────────────────────────────────────────────

    #[test]
    fn min_age_respected() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.8,
            min_age_frames: 120,
            max_evictions_per_frame: 64,
        });

        // All slots accessed at frame 95 — at frame 100, age is only 5.
        for i in 0..95u32 {
            tracker.track(i, IVec3::ZERO, 95);
        }

        let candidates = tracker.find_eviction_candidates(100, 95, 100);
        assert!(
            candidates.is_empty(),
            "should not evict recently-accessed slots"
        );
    }

    // ── 8. eviction_sorted_oldest_first ─────────────────────────────────────

    #[test]
    fn eviction_sorted_oldest_first() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.5,
            low_watermark: 0.0,
            min_age_frames: 0,
            max_evictions_per_frame: 100,
        });

        // Track slots with different access frames (out of order).
        tracker.track(0, IVec3::ZERO, 50); // newest
        tracker.track(1, IVec3::ZERO, 10); // oldest
        tracker.track(2, IVec3::ZERO, 30); // middle

        let candidates = tracker.find_eviction_candidates(4, 3, 100);
        // Should be sorted oldest-first: 1 (frame 10), 2 (frame 30), 0 (frame 50).
        assert_eq!(candidates[0], 1);
        assert_eq!(candidates[1], 2);
        assert_eq!(candidates[2], 0);
    }

    // ── 9. eviction_capped_by_max ───────────────────────────────────────────

    #[test]
    fn eviction_capped_by_max() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.5,
            low_watermark: 0.0,
            min_age_frames: 0,
            max_evictions_per_frame: 3,
        });

        // Track 20 slots, all old.
        for i in 0..20u32 {
            tracker.track(i, IVec3::ZERO, 0);
        }

        let candidates = tracker.find_eviction_candidates(20, 20, 1000);
        assert_eq!(
            candidates.len(),
            3,
            "should respect max_evictions_per_frame"
        );
    }

    // ── 10. eviction_targets_low_watermark ──────────────────────────────────

    #[test]
    fn eviction_targets_low_watermark() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.9,
            low_watermark: 0.7,
            min_age_frames: 0,
            max_evictions_per_frame: 1000,
        });

        // Capacity 200, allocated 190 (95%). Low watermark target = 140.
        // Need to evict 190 - 140 = 50.
        for i in 0..190u32 {
            tracker.track(i, IVec3::ZERO, 0);
        }

        let candidates = tracker.find_eviction_candidates(200, 190, 1000);
        assert_eq!(candidates.len(), 50);
    }

    // ── 11. slots_for_chunk ─────────────────────────────────────────────────

    #[test]
    fn slots_for_chunk() {
        let mut tracker = default_tracker();
        let chunk_a = IVec3::new(1, 0, 0);
        let chunk_b = IVec3::new(2, 0, 0);

        tracker.track(10, chunk_a, 0);
        tracker.track(11, chunk_a, 0);
        tracker.track(12, chunk_b, 0);
        tracker.track(13, chunk_a, 0);

        let mut slots = tracker.slots_for_chunk(&chunk_a);
        slots.sort();
        assert_eq!(slots, vec![10, 11, 13]);

        let slots_b = tracker.slots_for_chunk(&chunk_b);
        assert_eq!(slots_b, vec![12]);

        let slots_none = tracker.slots_for_chunk(&IVec3::new(99, 0, 0));
        assert!(slots_none.is_empty());
    }

    // ── 12. run_eviction_groups_by_chunk ────────────────────────────────────

    #[test]
    fn run_eviction_groups_by_chunk() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.5,
            low_watermark: 0.0,
            min_age_frames: 0,
            max_evictions_per_frame: 100,
        });

        let chunk_a = IVec3::new(1, 0, 0);
        let chunk_b = IVec3::new(2, 0, 0);

        tracker.track(0, chunk_a, 0);
        tracker.track(1, chunk_a, 0);
        tracker.track(2, chunk_b, 0);
        tracker.track(3, chunk_b, 0);

        let result = tracker.run_eviction(4, 4, 1000);
        assert_eq!(result.total_evicted, 4);

        // Verify grouping: should have entries for both chunks.
        let mut chunk_a_found = false;
        let mut chunk_b_found = false;
        for (coords, slots) in &result.evicted {
            if *coords == chunk_a {
                let mut s = slots.clone();
                s.sort();
                assert_eq!(s, vec![0, 1]);
                chunk_a_found = true;
            } else if *coords == chunk_b {
                let mut s = slots.clone();
                s.sort();
                assert_eq!(s, vec![2, 3]);
                chunk_b_found = true;
            }
        }
        assert!(chunk_a_found, "chunk_a group missing");
        assert!(chunk_b_found, "chunk_b group missing");
    }

    // ── 13. run_eviction_untracks ───────────────────────────────────────────

    #[test]
    fn run_eviction_untracks() {
        let mut tracker = LruTracker::new(EvictionPolicy {
            high_watermark: 0.5,
            low_watermark: 0.0,
            min_age_frames: 0,
            max_evictions_per_frame: 100,
        });

        tracker.track(0, IVec3::ZERO, 0);
        tracker.track(1, IVec3::ZERO, 0);
        tracker.track(2, IVec3::ZERO, 0);
        assert_eq!(tracker.tracked_count(), 3);

        let result = tracker.run_eviction(3, 3, 1000);
        assert_eq!(result.total_evicted, 3);
        assert_eq!(
            tracker.tracked_count(),
            0,
            "evicted slots should be untracked"
        );

        // Verify individual entries are gone.
        assert!(tracker.entry(0).is_none());
        assert!(tracker.entry(1).is_none());
        assert!(tracker.entry(2).is_none());
    }

    // ── 14. policy_mut_updates ──────────────────────────────────────────────

    #[test]
    fn policy_mut_updates() {
        let mut tracker = default_tracker();
        assert_eq!(tracker.policy().high_watermark, 0.9);

        tracker.policy_mut().high_watermark = 0.95;
        tracker.policy_mut().low_watermark = 0.85;
        tracker.policy_mut().min_age_frames = 60;
        tracker.policy_mut().max_evictions_per_frame = 128;

        assert_eq!(tracker.policy().high_watermark, 0.95);
        assert_eq!(tracker.policy().low_watermark, 0.85);
        assert_eq!(tracker.policy().min_age_frames, 60);
        assert_eq!(tracker.policy().max_evictions_per_frame, 128);
    }
}
