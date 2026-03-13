//! Copy-on-write brick pool tracking for play mode.
//!
//! During play mode, gameplay systems may modify voxel bricks (e.g. destructible
//! terrain). To preserve the edit-world's bricks for stop-and-restore, we track
//! ownership of each brick pool slot. Play-mode code must copy an edit-owned
//! brick before modifying it (copy-on-write). When play stops, all play-owned
//! bricks are freed.
//!
//! This module provides the tracking data structure only. Actual brick pool
//! integration (copying bricks, hooking into the allocator) happens in the
//! editor/runtime integration layer.

use std::collections::HashMap;

// ─── BrickOwnership ─────────────────────────────────────────────────────────

/// Whether a brick pool slot is owned by the edit world or the play world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickOwnership {
    /// Slot belongs to the edit world — play must copy before modifying.
    Edit,
    /// Slot was allocated or copied by the play world — freed on stop.
    Play,
}

// ─── CowBrickTracker ────────────────────────────────────────────────────────

/// Tracks per-slot brick ownership for copy-on-write during play mode.
///
/// At play start, all existing slots are implicitly edit-owned. When play-mode
/// code needs to modify a brick, the caller checks [`needs_cow`] — if true,
/// the brick is copied to a new slot which is marked play-owned. On stop,
/// [`free_play_bricks`] returns all play-owned slots for deallocation.
#[derive(Debug, Default)]
pub struct CowBrickTracker {
    ownership: HashMap<u32, BrickOwnership>,
}

impl CowBrickTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a slot as edit-owned (existing brick, must not be modified by play).
    pub fn mark_edit_owned(&mut self, slot: u32) {
        self.ownership.insert(slot, BrickOwnership::Edit);
    }

    /// Mark a slot as play-owned (newly allocated or copied for play-mode use).
    pub fn mark_play_owned(&mut self, slot: u32) {
        self.ownership.insert(slot, BrickOwnership::Play);
    }

    /// Returns `true` if the slot is edit-owned and play would need to copy
    /// before modifying it. Returns `false` if the slot is play-owned (already
    /// copied) or unknown (not tracked).
    pub fn needs_cow(&self, slot: u32) -> bool {
        self.ownership.get(&slot) == Some(&BrickOwnership::Edit)
    }

    /// Returns the ownership of a slot, if tracked.
    pub fn ownership(&self, slot: u32) -> Option<BrickOwnership> {
        self.ownership.get(&slot).copied()
    }

    /// Consume all play-owned slots and return them for deallocation.
    ///
    /// After this call, the tracker no longer contains any play-owned entries.
    /// Edit-owned entries are retained (they remain valid in the edit world).
    pub fn free_play_bricks(&mut self) -> Vec<u32> {
        let play_slots: Vec<u32> = self
            .ownership
            .iter()
            .filter_map(|(&slot, &owner)| {
                if owner == BrickOwnership::Play {
                    Some(slot)
                } else {
                    None
                }
            })
            .collect();

        for &slot in &play_slots {
            self.ownership.remove(&slot);
        }

        play_slots
    }

    /// Reset all tracking state.
    pub fn clear(&mut self) {
        self.ownership.clear();
    }

    /// Number of tracked slots.
    pub fn len(&self) -> usize {
        self.ownership.len()
    }

    /// Whether any slots are tracked.
    pub fn is_empty(&self) -> bool {
        self.ownership.is_empty()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tracker_is_empty() {
        let tracker = CowBrickTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn mark_edit_owned() {
        let mut tracker = CowBrickTracker::new();
        tracker.mark_edit_owned(42);
        assert_eq!(tracker.ownership(42), Some(BrickOwnership::Edit));
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn mark_play_owned() {
        let mut tracker = CowBrickTracker::new();
        tracker.mark_play_owned(7);
        assert_eq!(tracker.ownership(7), Some(BrickOwnership::Play));
    }

    #[test]
    fn cow_tracker_needs_cow() {
        let mut tracker = CowBrickTracker::new();

        // Edit-owned slots need COW.
        tracker.mark_edit_owned(10);
        assert!(tracker.needs_cow(10));

        // Play-owned slots do not need COW.
        tracker.mark_play_owned(20);
        assert!(!tracker.needs_cow(20));

        // Unknown slots do not need COW.
        assert!(!tracker.needs_cow(99));
    }

    #[test]
    fn cow_tracker_free_play_bricks() {
        let mut tracker = CowBrickTracker::new();
        tracker.mark_edit_owned(1);
        tracker.mark_edit_owned(2);
        tracker.mark_play_owned(10);
        tracker.mark_play_owned(11);
        tracker.mark_play_owned(12);

        let mut freed = tracker.free_play_bricks();
        freed.sort();
        assert_eq!(freed, vec![10, 11, 12]);

        // Play slots removed, edit slots remain.
        assert_eq!(tracker.len(), 2);
        assert!(tracker.needs_cow(1));
        assert!(tracker.needs_cow(2));
        assert!(!tracker.needs_cow(10));
    }

    #[test]
    fn free_play_bricks_empty_when_none() {
        let mut tracker = CowBrickTracker::new();
        tracker.mark_edit_owned(1);

        let freed = tracker.free_play_bricks();
        assert!(freed.is_empty());
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn clear_resets_all() {
        let mut tracker = CowBrickTracker::new();
        tracker.mark_edit_owned(1);
        tracker.mark_play_owned(2);
        assert_eq!(tracker.len(), 2);

        tracker.clear();
        assert!(tracker.is_empty());
        assert!(!tracker.needs_cow(1));
    }

    #[test]
    fn ownership_transitions() {
        let mut tracker = CowBrickTracker::new();

        // Start as edit-owned.
        tracker.mark_edit_owned(5);
        assert!(tracker.needs_cow(5));

        // After COW copy, mark the new slot as play-owned and the original remains edit.
        tracker.mark_play_owned(50); // new slot for the copy
        assert!(!tracker.needs_cow(50));
        assert!(tracker.needs_cow(5)); // original still edit-owned
    }

    #[test]
    fn default_trait() {
        let tracker = CowBrickTracker::default();
        assert!(tracker.is_empty());
    }
}
