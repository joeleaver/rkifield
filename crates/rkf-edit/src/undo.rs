//! Delta-based undo/redo system — stub pending v2 rewrite.
//!
//! In v2, undo is per-object (not per-chunk). Will be rewritten in Phase 11.

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
