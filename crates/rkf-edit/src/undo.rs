//! Per-object undo/redo system for v2.
//!
//! Each object has its own [`UndoStack`] that tracks [`EditRecord`]s.
//! An `EditRecord` captures the delta (previous voxel values) for all
//! bricks modified by a single edit operation, enabling exact reversal.
//!
//! The undo stack has a configurable maximum depth. When the limit is
//! exceeded, the oldest records are dropped.

use rkf_core::voxel::VoxelSample;

/// Record of a single brick's changed voxels (previous values).
#[derive(Debug, Clone)]
pub struct BrickDelta {
    /// Brick pool slot index.
    pub slot: u32,
    /// Changed voxels: `(linear_index, previous_value)`.
    ///
    /// `linear_index` is `x + y*8 + z*64`, range 0..511.
    pub changed_voxels: Vec<(u16, VoxelSample)>,
}

/// A single undoable edit record — captures all brick deltas from one operation.
#[derive(Debug, Clone)]
pub struct EditRecord {
    /// Object ID this edit targeted.
    pub object_id: u32,
    /// Description of the edit (for display in undo history).
    pub description: String,
    /// Per-brick deltas: the previous state of each modified brick.
    pub brick_deltas: Vec<BrickDelta>,
}

/// Per-object undo/redo stack.
///
/// Maintains a linear history of [`EditRecord`]s with a cursor.
/// Undo moves the cursor backward; redo moves it forward.
/// A new edit after an undo truncates the redo history.
#[derive(Debug)]
pub struct UndoStack {
    /// The edit records (oldest first).
    records: Vec<EditRecord>,
    /// Current cursor position — points to the next record to undo.
    /// `cursor == records.len()` means no undos have been performed.
    cursor: usize,
    /// Maximum number of records to keep.
    max_depth: usize,
}

impl UndoStack {
    /// Create a new undo stack with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            records: Vec::new(),
            cursor: 0,
            max_depth: max_depth.max(1),
        }
    }

    /// Push a new edit record onto the stack.
    ///
    /// Truncates any redo history beyond the current cursor.
    /// Drops the oldest record if the stack exceeds `max_depth`.
    pub fn push(&mut self, record: EditRecord) {
        // Truncate redo history.
        self.records.truncate(self.cursor);

        self.records.push(record);
        self.cursor = self.records.len();

        // Enforce max depth.
        if self.records.len() > self.max_depth {
            let excess = self.records.len() - self.max_depth;
            self.records.drain(0..excess);
            self.cursor = self.records.len();
        }
    }

    /// Undo the most recent edit, returning its record for reversal.
    ///
    /// Returns `None` if there's nothing to undo.
    pub fn undo(&mut self) -> Option<&EditRecord> {
        if self.cursor > 0 {
            self.cursor -= 1;
            Some(&self.records[self.cursor])
        } else {
            None
        }
    }

    /// Redo the next edit, returning its record for re-application.
    ///
    /// Returns `None` if there's nothing to redo.
    pub fn redo(&mut self) -> Option<&EditRecord> {
        if self.cursor < self.records.len() {
            let record = &self.records[self.cursor];
            self.cursor += 1;
            Some(record)
        } else {
            None
        }
    }

    /// Number of edits that can be undone.
    pub fn undo_count(&self) -> usize {
        self.cursor
    }

    /// Number of edits that can be redone.
    pub fn redo_count(&self) -> usize {
        self.records.len() - self.cursor
    }

    /// Total number of records in the stack (undo + redo).
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
        self.cursor = 0;
    }

    /// Description of the next undo operation, if any.
    pub fn undo_description(&self) -> Option<&str> {
        if self.cursor > 0 {
            Some(&self.records[self.cursor - 1].description)
        } else {
            None
        }
    }

    /// Description of the next redo operation, if any.
    pub fn redo_description(&self) -> Option<&str> {
        if self.cursor < self.records.len() {
            Some(&self.records[self.cursor].description)
        } else {
            None
        }
    }
}

/// Registry of per-object undo stacks.
///
/// Each object gets its own independent undo history.
#[derive(Debug, Default)]
pub struct UndoRegistry {
    stacks: std::collections::HashMap<u32, UndoStack>,
    max_depth: usize,
}

impl UndoRegistry {
    /// Create a new registry with the given per-object undo depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            stacks: std::collections::HashMap::new(),
            max_depth: max_depth.max(1),
        }
    }

    /// Get or create the undo stack for an object.
    pub fn stack_for(&mut self, object_id: u32) -> &mut UndoStack {
        self.stacks
            .entry(object_id)
            .or_insert_with(|| UndoStack::new(self.max_depth))
    }

    /// Push an edit record to the appropriate object's stack.
    pub fn push(&mut self, record: EditRecord) {
        let object_id = record.object_id;
        self.stack_for(object_id).push(record);
    }

    /// Undo the most recent edit on the given object.
    pub fn undo(&mut self, object_id: u32) -> Option<&EditRecord> {
        self.stacks.get_mut(&object_id).and_then(|s| s.undo())
    }

    /// Redo the next edit on the given object.
    pub fn redo(&mut self, object_id: u32) -> Option<&EditRecord> {
        self.stacks.get_mut(&object_id).and_then(|s| s.redo())
    }

    /// Remove the undo stack for an object (e.g., when the object is deleted).
    pub fn remove_object(&mut self, object_id: u32) {
        self.stacks.remove(&object_id);
    }

    /// Number of objects with undo stacks.
    pub fn object_count(&self) -> usize {
        self.stacks.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::voxel::VoxelSample;

    fn sample(dist: f32, mat: u16) -> VoxelSample {
        VoxelSample::new(dist, mat, 0)
    }

    fn test_record(object_id: u32, desc: &str) -> EditRecord {
        EditRecord {
            object_id,
            description: desc.into(),
            brick_deltas: vec![BrickDelta {
                slot: 0,
                changed_voxels: vec![(0, sample(1.0, 1))],
            }],
        }
    }

    // -- UndoStack --

    #[test]
    fn empty_stack() {
        let stack = UndoStack::new(10);
        assert!(stack.is_empty());
        assert_eq!(stack.len(), 0);
        assert_eq!(stack.undo_count(), 0);
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn push_and_undo() {
        let mut stack = UndoStack::new(10);
        stack.push(test_record(1, "add sphere"));
        assert_eq!(stack.len(), 1);
        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 0);

        let record = stack.undo().unwrap();
        assert_eq!(record.description, "add sphere");
        assert_eq!(stack.undo_count(), 0);
        assert_eq!(stack.redo_count(), 1);
    }

    #[test]
    fn undo_then_redo() {
        let mut stack = UndoStack::new(10);
        stack.push(test_record(1, "edit A"));
        stack.push(test_record(1, "edit B"));

        // Undo B
        let r = stack.undo().unwrap();
        assert_eq!(r.description, "edit B");
        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 1);

        // Redo B
        let r = stack.redo().unwrap();
        assert_eq!(r.description, "edit B");
        assert_eq!(stack.undo_count(), 2);
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn new_edit_truncates_redo() {
        let mut stack = UndoStack::new(10);
        stack.push(test_record(1, "edit A"));
        stack.push(test_record(1, "edit B"));
        stack.push(test_record(1, "edit C"));

        // Undo C and B
        stack.undo();
        stack.undo();
        assert_eq!(stack.redo_count(), 2);

        // Push new edit — should truncate B and C from redo
        stack.push(test_record(1, "edit D"));
        assert_eq!(stack.len(), 2); // A and D
        assert_eq!(stack.redo_count(), 0);
        assert_eq!(stack.undo_description(), Some("edit D"));
    }

    #[test]
    fn max_depth_enforced() {
        let mut stack = UndoStack::new(3);
        stack.push(test_record(1, "edit A"));
        stack.push(test_record(1, "edit B"));
        stack.push(test_record(1, "edit C"));
        stack.push(test_record(1, "edit D"));

        assert_eq!(stack.len(), 3);
        // Oldest (A) should have been dropped
        let r = stack.undo().unwrap();
        assert_eq!(r.description, "edit D");
        let r = stack.undo().unwrap();
        assert_eq!(r.description, "edit C");
        let r = stack.undo().unwrap();
        assert_eq!(r.description, "edit B");
        assert!(stack.undo().is_none());
    }

    #[test]
    fn undo_empty_returns_none() {
        let mut stack = UndoStack::new(10);
        assert!(stack.undo().is_none());
    }

    #[test]
    fn redo_empty_returns_none() {
        let mut stack = UndoStack::new(10);
        assert!(stack.redo().is_none());
    }

    #[test]
    fn clear_resets() {
        let mut stack = UndoStack::new(10);
        stack.push(test_record(1, "a"));
        stack.push(test_record(1, "b"));
        stack.clear();
        assert!(stack.is_empty());
        assert_eq!(stack.undo_count(), 0);
    }

    #[test]
    fn descriptions() {
        let mut stack = UndoStack::new(10);
        assert!(stack.undo_description().is_none());
        assert!(stack.redo_description().is_none());

        stack.push(test_record(1, "first"));
        assert_eq!(stack.undo_description(), Some("first"));
        assert!(stack.redo_description().is_none());

        stack.undo();
        assert!(stack.undo_description().is_none());
        assert_eq!(stack.redo_description(), Some("first"));
    }

    // -- UndoRegistry --

    #[test]
    fn registry_per_object_stacks() {
        let mut reg = UndoRegistry::new(10);
        reg.push(test_record(1, "obj1 edit"));
        reg.push(test_record(2, "obj2 edit"));

        assert_eq!(reg.object_count(), 2);

        let r = reg.undo(1).unwrap();
        assert_eq!(r.description, "obj1 edit");
        assert_eq!(r.object_id, 1);

        let r = reg.undo(2).unwrap();
        assert_eq!(r.description, "obj2 edit");
        assert_eq!(r.object_id, 2);
    }

    #[test]
    fn registry_independent_histories() {
        let mut reg = UndoRegistry::new(10);
        reg.push(test_record(1, "obj1 A"));
        reg.push(test_record(1, "obj1 B"));
        reg.push(test_record(2, "obj2 A"));

        // Undo on obj1 doesn't affect obj2
        reg.undo(1);
        assert_eq!(reg.stack_for(1).undo_count(), 1);
        assert_eq!(reg.stack_for(2).undo_count(), 1);
    }

    #[test]
    fn registry_remove_object() {
        let mut reg = UndoRegistry::new(10);
        reg.push(test_record(1, "edit"));
        assert_eq!(reg.object_count(), 1);

        reg.remove_object(1);
        assert_eq!(reg.object_count(), 0);
        assert!(reg.undo(1).is_none());
    }

    #[test]
    fn registry_undo_nonexistent_object() {
        let mut reg = UndoRegistry::new(10);
        assert!(reg.undo(999).is_none());
    }

    #[test]
    fn brick_delta_stores_voxels() {
        let delta = BrickDelta {
            slot: 42,
            changed_voxels: vec![
                (0, sample(0.5, 1)),
                (511, sample(-0.3, 2)),
            ],
        };
        assert_eq!(delta.slot, 42);
        assert_eq!(delta.changed_voxels.len(), 2);
    }
}
