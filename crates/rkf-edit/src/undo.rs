//! Delta-based undo/redo system for edit operations.
//!
//! Before an edit modifies the brick pool, [`capture_pre_edit`] snapshots the
//! affected voxels.  After the edit, [`diff_deltas`] keeps only voxels that
//! actually changed, producing an [`EditDelta`] suitable for
//! [`UndoHistory::push`].
//!
//! Undo restores previous voxel values and manages brick allocation/deallocation.
//! Redo re-captures current state and restores the "after" snapshot.

use rkf_core::brick_pool::BrickPool;
use rkf_core::cell_state::CellState;
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::voxel::VoxelSample;

// ---------------------------------------------------------------------------
// BrickDelta — per-brick change record
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// BrickAllocation — allocation/deallocation tracking
// ---------------------------------------------------------------------------

/// Record of a brick allocation or deallocation event.
#[derive(Debug, Clone, Copy)]
pub struct BrickAllocation {
    /// Cell X coordinate in the sparse grid.
    pub cx: u32,
    /// Cell Y coordinate in the sparse grid.
    pub cy: u32,
    /// Cell Z coordinate in the sparse grid.
    pub cz: u32,
    /// Brick pool slot.
    pub slot: u32,
}

// ---------------------------------------------------------------------------
// EditDelta — complete delta for one operation
// ---------------------------------------------------------------------------

/// Complete delta for one edit operation, sufficient to undo it.
#[derive(Debug, Clone)]
pub struct EditDelta {
    /// Unique operation identifier.
    pub operation_id: u64,
    /// Per-brick voxel changes (previous values for undo).
    pub affected_bricks: Vec<BrickDelta>,
    /// Bricks that were newly allocated during this edit.
    pub allocated_bricks: Vec<BrickAllocation>,
    /// Bricks that were deallocated during post-edit cleanup.
    pub deallocated_bricks: Vec<BrickAllocation>,
}

impl EditDelta {
    /// Estimate memory usage of this delta in bytes.
    pub fn memory_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let bricks: usize = self
            .affected_bricks
            .iter()
            .map(|b| {
                std::mem::size_of::<BrickDelta>()
                    + b.changed_voxels.len() * std::mem::size_of::<(u16, VoxelSample)>()
            })
            .sum();
        let alloc =
            self.allocated_bricks.len() * std::mem::size_of::<BrickAllocation>();
        let dealloc =
            self.deallocated_bricks.len() * std::mem::size_of::<BrickAllocation>();
        base + bricks + alloc + dealloc
    }
}

// ---------------------------------------------------------------------------
// Pre-edit capture and diffing
// ---------------------------------------------------------------------------

/// Capture a full snapshot of the specified brick slots before an edit.
///
/// Call this BEFORE the edit modifies the brick pool. Returns one
/// [`BrickDelta`] per slot with all 512 voxels stored as "previous" values.
/// After the edit completes, pass these to [`diff_deltas`] to keep only
/// voxels that actually changed.
pub fn capture_pre_edit(pool: &BrickPool, brick_slots: &[u32]) -> Vec<BrickDelta> {
    brick_slots
        .iter()
        .map(|&slot| {
            let brick = pool.get(slot);
            let mut changed = Vec::with_capacity(512);
            for i in 0u16..512 {
                let vx = (i % 8) as u32;
                let vy = ((i / 8) % 8) as u32;
                let vz = (i / 64) as u32;
                changed.push((i, brick.sample(vx, vy, vz)));
            }
            BrickDelta {
                slot,
                changed_voxels: changed,
            }
        })
        .collect()
}

/// Diff pre-edit snapshots against the current pool state.
///
/// Removes voxels that did not change, and drops bricks with zero changes.
/// The resulting deltas store only the *previous* values of voxels that
/// were actually modified — exactly what is needed to undo the edit.
pub fn diff_deltas(pool: &BrickPool, pre_edit: Vec<BrickDelta>) -> Vec<BrickDelta> {
    pre_edit
        .into_iter()
        .filter_map(|mut bd| {
            let brick = pool.get(bd.slot);
            bd.changed_voxels.retain(|&(idx, prev)| {
                let vx = (idx % 8) as u32;
                let vy = ((idx / 8) % 8) as u32;
                let vz = (idx / 64) as u32;
                brick.sample(vx, vy, vz) != prev
            });
            if bd.changed_voxels.is_empty() {
                None
            } else {
                Some(bd)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// UndoHistory
// ---------------------------------------------------------------------------

/// Undo/redo history with configurable depth and memory budget.
///
/// The undo stack holds [`EditDelta`]s representing previous voxel values.
/// Pushing a new delta clears the redo stack. When the stack exceeds
/// `max_depth` or `max_bytes`, the oldest deltas are discarded.
pub struct UndoHistory {
    stack: Vec<EditDelta>,
    redo_stack: Vec<EditDelta>,
    max_depth: usize,
    max_bytes: usize,
    next_op_id: u64,
    current_bytes: usize,
}

impl UndoHistory {
    /// Create a new undo history.
    ///
    /// * `max_depth` — maximum number of undo steps (default 100).
    /// * `max_bytes` — approximate memory budget in bytes (default 256 MB).
    pub fn new(max_depth: usize, max_bytes: usize) -> Self {
        Self {
            stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
            max_bytes,
            next_op_id: 0,
            current_bytes: 0,
        }
    }

    /// Push a new edit delta onto the undo stack.
    ///
    /// Clears the redo stack (branching history). Evicts the oldest deltas
    /// if the stack exceeds `max_depth` or `max_bytes`.
    pub fn push(&mut self, delta: EditDelta) {
        // Clear redo stack — new edit branches history.
        for rd in self.redo_stack.drain(..) {
            self.current_bytes = self.current_bytes.saturating_sub(rd.memory_bytes());
        }

        self.current_bytes += delta.memory_bytes();
        self.stack.push(delta);

        // Evict oldest deltas if over budget.
        self.evict();
    }

    /// Undo the most recent edit operation.
    ///
    /// Restores previous voxel values in the brick pool and manages brick
    /// allocation/deallocation in the sparse grid. Returns the operation ID
    /// of the undone edit, or `None` if the stack is empty.
    pub fn undo(
        &mut self,
        pool: &mut BrickPool,
        grid: &mut SparseGrid,
    ) -> Option<u64> {
        let delta = self.stack.pop()?;
        let op_id = delta.operation_id;

        // Build a redo delta by capturing the current state of affected voxels.
        let mut redo_bricks = Vec::with_capacity(delta.affected_bricks.len());
        for bd in &delta.affected_bricks {
            let brick = pool.get(bd.slot);
            let current_values: Vec<(u16, VoxelSample)> = bd
                .changed_voxels
                .iter()
                .map(|&(idx, _)| {
                    let vx = (idx % 8) as u32;
                    let vy = ((idx / 8) % 8) as u32;
                    let vz = (idx / 64) as u32;
                    (idx, brick.sample(vx, vy, vz))
                })
                .collect();
            redo_bricks.push(BrickDelta {
                slot: bd.slot,
                changed_voxels: current_values,
            });
        }

        // Restore previous voxel values.
        for bd in &delta.affected_bricks {
            let brick = pool.get_mut(bd.slot);
            for &(idx, prev) in &bd.changed_voxels {
                let vx = (idx % 8) as u32;
                let vy = ((idx / 8) % 8) as u32;
                let vz = (idx / 64) as u32;
                brick.set(vx, vy, vz, prev);
            }
        }

        // Undo brick allocations: deallocate bricks that were allocated during the edit.
        for alloc in &delta.allocated_bricks {
            pool.deallocate(alloc.slot);
            grid.set_cell_state(alloc.cx, alloc.cy, alloc.cz, CellState::Empty);
            grid.clear_brick_slot(alloc.cx, alloc.cy, alloc.cz);
        }

        // Undo brick deallocations: re-allocate bricks that were freed during cleanup.
        // Note: we re-allocate from the pool (which may give a different slot),
        // then restore the voxel data. For simplicity, we allocate new slots.
        // The redo delta for allocated/deallocated is swapped.
        let mut redo_allocated = Vec::new();
        for dealloc in &delta.deallocated_bricks {
            if let Some(new_slot) = pool.allocate() {
                grid.set_cell_state(
                    dealloc.cx,
                    dealloc.cy,
                    dealloc.cz,
                    CellState::Surface,
                );
                grid.set_brick_slot(dealloc.cx, dealloc.cy, dealloc.cz, new_slot);
                redo_allocated.push(BrickAllocation {
                    cx: dealloc.cx,
                    cy: dealloc.cy,
                    cz: dealloc.cz,
                    slot: new_slot,
                });
            }
        }

        let redo_delta = EditDelta {
            operation_id: op_id,
            affected_bricks: redo_bricks,
            // Swap: what was allocated becomes deallocated in redo, and vice versa.
            allocated_bricks: redo_allocated,
            deallocated_bricks: delta.allocated_bricks.clone(),
        };

        self.current_bytes = self.current_bytes.saturating_sub(delta.memory_bytes());
        self.current_bytes += redo_delta.memory_bytes();
        self.redo_stack.push(redo_delta);

        Some(op_id)
    }

    /// Redo the most recently undone edit operation.
    ///
    /// Re-applies voxel values captured during undo. Returns the operation ID
    /// of the redone edit, or `None` if the redo stack is empty.
    pub fn redo(
        &mut self,
        pool: &mut BrickPool,
        grid: &mut SparseGrid,
    ) -> Option<u64> {
        let delta = self.redo_stack.pop()?;
        let op_id = delta.operation_id;

        // Build an undo delta by capturing current state before redo.
        let mut undo_bricks = Vec::with_capacity(delta.affected_bricks.len());
        for bd in &delta.affected_bricks {
            let brick = pool.get(bd.slot);
            let current_values: Vec<(u16, VoxelSample)> = bd
                .changed_voxels
                .iter()
                .map(|&(idx, _)| {
                    let vx = (idx % 8) as u32;
                    let vy = ((idx / 8) % 8) as u32;
                    let vz = (idx / 64) as u32;
                    (idx, brick.sample(vx, vy, vz))
                })
                .collect();
            undo_bricks.push(BrickDelta {
                slot: bd.slot,
                changed_voxels: current_values,
            });
        }

        // Apply redo voxel values.
        for bd in &delta.affected_bricks {
            let brick = pool.get_mut(bd.slot);
            for &(idx, val) in &bd.changed_voxels {
                let vx = (idx % 8) as u32;
                let vy = ((idx / 8) % 8) as u32;
                let vz = (idx / 64) as u32;
                brick.set(vx, vy, vz, val);
            }
        }

        // Undo allocations from the redo delta (reverse of what redo's "allocated" means).
        for alloc in &delta.deallocated_bricks {
            pool.deallocate(alloc.slot);
            grid.set_cell_state(alloc.cx, alloc.cy, alloc.cz, CellState::Empty);
            grid.clear_brick_slot(alloc.cx, alloc.cy, alloc.cz);
        }

        // Re-allocate for redo's allocated bricks.
        let mut new_allocated = Vec::new();
        for alloc in &delta.allocated_bricks {
            if let Some(new_slot) = pool.allocate() {
                grid.set_cell_state(alloc.cx, alloc.cy, alloc.cz, CellState::Surface);
                grid.set_brick_slot(alloc.cx, alloc.cy, alloc.cz, new_slot);
                new_allocated.push(BrickAllocation {
                    cx: alloc.cx,
                    cy: alloc.cy,
                    cz: alloc.cz,
                    slot: new_slot,
                });
            }
        }

        let undo_delta = EditDelta {
            operation_id: op_id,
            affected_bricks: undo_bricks,
            allocated_bricks: new_allocated,
            deallocated_bricks: delta.allocated_bricks.clone(),
        };

        self.current_bytes = self.current_bytes.saturating_sub(delta.memory_bytes());
        self.current_bytes += undo_delta.memory_bytes();
        self.stack.push(undo_delta);

        Some(op_id)
    }

    /// Returns `true` if there is at least one edit to undo.
    pub fn can_undo(&self) -> bool {
        !self.stack.is_empty()
    }

    /// Returns `true` if there is at least one edit to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Number of undo steps currently stored.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Allocate and return the next unique operation ID.
    pub fn next_operation_id(&mut self) -> u64 {
        let id = self.next_op_id;
        self.next_op_id += 1;
        id
    }

    /// Approximate memory usage of all stored deltas in bytes.
    pub fn memory_usage(&self) -> usize {
        self.current_bytes
    }

    /// Evict oldest deltas until within depth and memory budgets.
    fn evict(&mut self) {
        // Evict for depth.
        while self.stack.len() > self.max_depth {
            if let Some(old) = self.stack.first() {
                let bytes = old.memory_bytes();
                self.current_bytes = self.current_bytes.saturating_sub(bytes);
            }
            self.stack.remove(0);
        }
        // Evict for memory.
        while self.current_bytes > self.max_bytes && !self.stack.is_empty() {
            if let Some(old) = self.stack.first() {
                let bytes = old.memory_bytes();
                self.current_bytes = self.current_bytes.saturating_sub(bytes);
            }
            self.stack.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::UVec3;
    use rkf_core::brick_pool::BrickPool;
    use rkf_core::sparse_grid::SparseGrid;
    use rkf_core::voxel::VoxelSample;

    /// Helper: set up a pool, grid, and allocate one brick with a known voxel.
    fn setup() -> (BrickPool, SparseGrid, u32) {
        let mut pool = BrickPool::new(16);
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        let slot = pool.allocate().unwrap();
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, slot);
        // Write a known voxel.
        let v = VoxelSample::new(0.5, 1, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, v);
        (pool, grid, slot)
    }

    // -- capture_pre_edit --

    #[test]
    fn capture_pre_edit_captures_all_512_voxels() {
        let (pool, _, slot) = setup();
        let deltas = capture_pre_edit(&pool, &[slot]);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].slot, slot);
        assert_eq!(deltas[0].changed_voxels.len(), 512);
    }

    #[test]
    fn capture_pre_edit_preserves_values() {
        let (pool, _, slot) = setup();
        let deltas = capture_pre_edit(&pool, &[slot]);
        // Voxel at (0,0,0) = linear index 0 should have our known value.
        let (idx, val) = deltas[0].changed_voxels[0];
        assert_eq!(idx, 0);
        assert_eq!(val.material_id(), 1);
        assert!((val.distance_f32() - 0.5).abs() < 0.01);
    }

    #[test]
    fn capture_pre_edit_empty_slots_list() {
        let (pool, _, _) = setup();
        let deltas = capture_pre_edit(&pool, &[]);
        assert!(deltas.is_empty());
    }

    // -- diff_deltas --

    #[test]
    fn diff_deltas_removes_unchanged_voxels() {
        let (mut pool, _, slot) = setup();
        let pre = capture_pre_edit(&pool, &[slot]);

        // Modify one voxel.
        let new_val = VoxelSample::new(-1.0, 5, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, new_val);

        let diffed = diff_deltas(&pool, pre);
        assert_eq!(diffed.len(), 1);
        // Only the one changed voxel should remain.
        assert_eq!(diffed[0].changed_voxels.len(), 1);
        assert_eq!(diffed[0].changed_voxels[0].0, 0); // linear index 0
    }

    #[test]
    fn diff_deltas_drops_unchanged_bricks() {
        let (pool, _, slot) = setup();
        let pre = capture_pre_edit(&pool, &[slot]);
        // No modifications — diff should be empty.
        let diffed = diff_deltas(&pool, pre);
        assert!(diffed.is_empty());
    }

    #[test]
    fn diff_deltas_multiple_changed_voxels() {
        let (mut pool, _, slot) = setup();
        let pre = capture_pre_edit(&pool, &[slot]);

        // Change 3 voxels.
        pool.get_mut(slot)
            .set(0, 0, 0, VoxelSample::new(-1.0, 2, 0, 0, 0));
        pool.get_mut(slot)
            .set(1, 0, 0, VoxelSample::new(-2.0, 3, 0, 0, 0));
        pool.get_mut(slot)
            .set(7, 7, 7, VoxelSample::new(0.0, 4, 0, 0, 0));

        let diffed = diff_deltas(&pool, pre);
        assert_eq!(diffed.len(), 1);
        assert_eq!(diffed[0].changed_voxels.len(), 3);
    }

    // -- EditDelta::memory_bytes --

    #[test]
    fn edit_delta_memory_bytes_empty() {
        let delta = EditDelta {
            operation_id: 0,
            affected_bricks: vec![],
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        };
        let bytes = delta.memory_bytes();
        // Should be at least the base struct size.
        assert!(bytes >= std::mem::size_of::<EditDelta>());
    }

    #[test]
    fn edit_delta_memory_bytes_with_data() {
        let delta = EditDelta {
            operation_id: 0,
            affected_bricks: vec![BrickDelta {
                slot: 0,
                changed_voxels: vec![(0, VoxelSample::default()); 100],
            }],
            allocated_bricks: vec![BrickAllocation {
                cx: 0,
                cy: 0,
                cz: 0,
                slot: 0,
            }],
            deallocated_bricks: vec![],
        };
        let bytes = delta.memory_bytes();
        // Should be larger than base due to 100 voxel entries + 1 allocation.
        assert!(bytes > std::mem::size_of::<EditDelta>() + 100);
    }

    // -- UndoHistory basic operations --

    #[test]
    fn new_history_is_empty() {
        let h = UndoHistory::new(100, 256 * 1024 * 1024);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.depth(), 0);
        assert_eq!(h.memory_usage(), 0);
    }

    #[test]
    fn push_increases_depth() {
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        let delta = EditDelta {
            operation_id: 0,
            affected_bricks: vec![],
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        };
        h.push(delta);
        assert_eq!(h.depth(), 1);
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn next_operation_id_increments() {
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        assert_eq!(h.next_operation_id(), 0);
        assert_eq!(h.next_operation_id(), 1);
        assert_eq!(h.next_operation_id(), 2);
    }

    // -- Undo restores voxels --

    #[test]
    fn undo_restores_voxel_values() {
        let (mut pool, mut grid, slot) = setup();

        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);

        // Capture pre-edit state.
        let pre = capture_pre_edit(&pool, &[slot]);
        let original_val = pool.get(slot).sample(0, 0, 0);

        // Simulate edit: modify voxel.
        let edited_val = VoxelSample::new(-1.0, 99, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, edited_val);

        // Build delta.
        let diffed = diff_deltas(&pool, pre);
        let op_id = h.next_operation_id();
        let delta = EditDelta {
            operation_id: op_id,
            affected_bricks: diffed,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        };
        h.push(delta);

        // Verify current state is edited.
        assert_eq!(pool.get(slot).sample(0, 0, 0), edited_val);

        // Undo.
        let undone = h.undo(&mut pool, &mut grid);
        assert_eq!(undone, Some(0));
        assert_eq!(pool.get(slot).sample(0, 0, 0), original_val);
    }

    // -- Redo re-applies --

    #[test]
    fn redo_reapplies_voxel_values() {
        let (mut pool, mut grid, slot) = setup();
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);

        let pre = capture_pre_edit(&pool, &[slot]);

        let edited_val = VoxelSample::new(-1.0, 99, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, edited_val);

        let diffed = diff_deltas(&pool, pre);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: diffed,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        // Undo, then redo.
        h.undo(&mut pool, &mut grid);
        assert!(h.can_redo());

        let redone = h.redo(&mut pool, &mut grid);
        assert_eq!(redone, Some(0));
        assert_eq!(pool.get(slot).sample(0, 0, 0), edited_val);
    }

    // -- Push clears redo stack --

    #[test]
    fn push_clears_redo_stack() {
        let (mut pool, mut grid, slot) = setup();
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);

        // First edit.
        let pre = capture_pre_edit(&pool, &[slot]);
        pool.get_mut(slot)
            .set(0, 0, 0, VoxelSample::new(-1.0, 99, 0, 0, 0));
        let diffed = diff_deltas(&pool, pre);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: diffed,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        // Undo.
        h.undo(&mut pool, &mut grid);
        assert!(h.can_redo());

        // New edit — should clear redo.
        let pre2 = capture_pre_edit(&pool, &[slot]);
        pool.get_mut(slot)
            .set(1, 0, 0, VoxelSample::new(-2.0, 50, 0, 0, 0));
        let diffed2 = diff_deltas(&pool, pre2);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: diffed2,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        assert!(!h.can_redo());
    }

    // -- Empty undo/redo returns None --

    #[test]
    fn undo_on_empty_returns_none() {
        let mut pool = BrickPool::new(4);
        let mut grid = SparseGrid::new(UVec3::new(2, 2, 2));
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        assert_eq!(h.undo(&mut pool, &mut grid), None);
    }

    #[test]
    fn redo_on_empty_returns_none() {
        let mut pool = BrickPool::new(4);
        let mut grid = SparseGrid::new(UVec3::new(2, 2, 2));
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        assert_eq!(h.redo(&mut pool, &mut grid), None);
    }

    // -- Depth limit eviction --

    #[test]
    fn depth_limit_evicts_oldest() {
        let mut h = UndoHistory::new(3, 256 * 1024 * 1024);

        for i in 0..5 {
            h.push(EditDelta {
                operation_id: i,
                affected_bricks: vec![],
                allocated_bricks: vec![],
                deallocated_bricks: vec![],
            });
        }

        // Max depth is 3, so only 3 should remain.
        assert_eq!(h.depth(), 3);
        // The oldest remaining should be operation 2 (0 and 1 evicted).
        assert_eq!(h.stack[0].operation_id, 2);
        assert_eq!(h.stack[1].operation_id, 3);
        assert_eq!(h.stack[2].operation_id, 4);
    }

    // -- Memory budget eviction --

    #[test]
    fn memory_budget_evicts_oldest() {
        // Create a very small memory budget.
        let mut h = UndoHistory::new(100, 1); // 1 byte budget

        // Push a delta that exceeds the budget.
        h.push(EditDelta {
            operation_id: 0,
            affected_bricks: vec![BrickDelta {
                slot: 0,
                changed_voxels: vec![(0, VoxelSample::default()); 100],
            }],
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        // The delta exceeds budget, so it gets evicted immediately after push.
        assert_eq!(h.depth(), 0);
    }

    // -- Undo with brick allocation tracking --

    #[test]
    fn undo_deallocates_newly_allocated_bricks() {
        let mut pool = BrickPool::new(16);
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));

        // Simulate an edit that allocates a new brick.
        let slot = pool.allocate().unwrap();
        grid.set_cell_state(1, 1, 1, CellState::Surface);
        grid.set_brick_slot(1, 1, 1, slot);
        pool.get_mut(slot)
            .set(0, 0, 0, VoxelSample::new(0.1, 1, 0, 0, 0));

        let alloc_count_before = pool.allocated_count();

        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: vec![],
            allocated_bricks: vec![BrickAllocation {
                cx: 1,
                cy: 1,
                cz: 1,
                slot,
            }],
            deallocated_bricks: vec![],
        });

        // Undo should deallocate the brick.
        h.undo(&mut pool, &mut grid);

        assert_eq!(pool.allocated_count(), alloc_count_before - 1);
        assert_eq!(grid.cell_state(1, 1, 1), CellState::Empty);
        assert_eq!(grid.brick_slot(1, 1, 1), None);
    }

    // -- Multiple undo/redo cycles --

    #[test]
    fn multiple_undo_redo_cycles() {
        let (mut pool, mut grid, slot) = setup();
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);

        let original = pool.get(slot).sample(0, 0, 0);

        // Edit 1.
        let pre = capture_pre_edit(&pool, &[slot]);
        let v1 = VoxelSample::new(-1.0, 10, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, v1);
        let d1 = diff_deltas(&pool, pre);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: d1,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        // Edit 2.
        let pre = capture_pre_edit(&pool, &[slot]);
        let v2 = VoxelSample::new(-2.0, 20, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, v2);
        let d2 = diff_deltas(&pool, pre);
        let op_id = h.next_operation_id();
        h.push(EditDelta {
            operation_id: op_id,
            affected_bricks: d2,
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        });

        assert_eq!(pool.get(slot).sample(0, 0, 0), v2);

        // Undo edit 2 -> should be v1.
        h.undo(&mut pool, &mut grid);
        assert_eq!(pool.get(slot).sample(0, 0, 0), v1);

        // Undo edit 1 -> should be original.
        h.undo(&mut pool, &mut grid);
        assert_eq!(pool.get(slot).sample(0, 0, 0), original);

        // Redo edit 1 -> v1.
        h.redo(&mut pool, &mut grid);
        assert_eq!(pool.get(slot).sample(0, 0, 0), v1);

        // Redo edit 2 -> v2.
        h.redo(&mut pool, &mut grid);
        assert_eq!(pool.get(slot).sample(0, 0, 0), v2);
    }

    // -- Memory usage tracking --

    #[test]
    fn memory_usage_tracks_pushes() {
        let mut h = UndoHistory::new(100, 256 * 1024 * 1024);
        assert_eq!(h.memory_usage(), 0);

        let delta = EditDelta {
            operation_id: 0,
            affected_bricks: vec![BrickDelta {
                slot: 0,
                changed_voxels: vec![(0, VoxelSample::default()); 50],
            }],
            allocated_bricks: vec![],
            deallocated_bricks: vec![],
        };
        let expected = delta.memory_bytes();
        h.push(delta);
        assert_eq!(h.memory_usage(), expected);
    }
}
