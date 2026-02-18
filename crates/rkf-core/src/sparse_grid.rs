//! Single-LOD sparse grid for voxel occupancy and brick indexing.
//!
//! A [`SparseGrid`] is a two-level spatial index over a 3D grid of cells (bricks):
//!
//! - **Level 2 (root):** 2-bit [`CellState`] per cell, packed into a `Vec<u32>` bitfield
//!   (16 cells per word). Provides coarse occupancy for hierarchical empty-space skipping.
//! - **Level 1 (blocks):** Dense `Vec<u32>` mapping each cell to a brick pool slot.
//!   [`EMPTY_SLOT`] indicates no brick is assigned.
//!
//! The grid dimensions are specified in cells (bricks), not voxels.

use glam::UVec3;

use crate::cell_state::CellState;

/// Sentinel value indicating no brick pool slot is assigned to a cell.
pub const EMPTY_SLOT: u32 = u32::MAX;

/// A single-LOD sparse grid for voxel occupancy and brick pool indexing.
///
/// Dimensions are in cells (bricks). Each cell corresponds to one brick in the pool.
#[derive(Debug, Clone)]
pub struct SparseGrid {
    /// Grid dimensions in cells (bricks) per axis.
    dimensions: UVec3,
    /// Level 2: packed 2-bit CellState per cell. Each u32 holds 16 cells.
    occupancy: Vec<u32>,
    /// Level 1: brick pool slot per cell. [`EMPTY_SLOT`] if no brick.
    slots: Vec<u32>,
}

impl SparseGrid {
    /// Create a new sparse grid with the given dimensions (in cells/bricks).
    ///
    /// All cells start as [`CellState::Empty`] with no brick slot assigned.
    pub fn new(dimensions: UVec3) -> Self {
        let total_cells = (dimensions.x as usize)
            * (dimensions.y as usize)
            * (dimensions.z as usize);
        // Each u32 holds 16 cells (2 bits each). Round up.
        let occupancy_words = total_cells.div_ceil(16);
        Self {
            dimensions,
            occupancy: vec![0u32; occupancy_words],
            slots: vec![EMPTY_SLOT; total_cells],
        }
    }

    /// Grid dimensions in cells per axis.
    #[inline]
    pub fn dimensions(&self) -> UVec3 {
        self.dimensions
    }

    /// Total number of cells in the grid.
    #[inline]
    pub fn total_cells(&self) -> u32 {
        self.dimensions.x * self.dimensions.y * self.dimensions.z
    }

    /// Compute the flat index for a cell at `(x, y, z)`.
    ///
    /// Layout: `x + y * dim_x + z * dim_x * dim_y` (z-major).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if coordinates are out of bounds.
    #[inline]
    fn flat_index(&self, x: u32, y: u32, z: u32) -> usize {
        debug_assert!(x < self.dimensions.x, "x={x} >= dim_x={}", self.dimensions.x);
        debug_assert!(y < self.dimensions.y, "y={y} >= dim_y={}", self.dimensions.y);
        debug_assert!(z < self.dimensions.z, "z={z} >= dim_z={}", self.dimensions.z);
        (x + y * self.dimensions.x + z * self.dimensions.x * self.dimensions.y) as usize
    }

    /// Read the [`CellState`] for the cell at `(x, y, z)`.
    pub fn cell_state(&self, x: u32, y: u32, z: u32) -> CellState {
        let idx = self.flat_index(x, y, z);
        let word = idx / 16;
        let bit_offset = (idx % 16) * 2;
        let bits = (self.occupancy[word] >> bit_offset) & 0b11;
        // SAFETY: bits is always 0..3, which are valid CellState values
        CellState::from_u8(bits as u8).unwrap()
    }

    /// Set the [`CellState`] for the cell at `(x, y, z)`.
    pub fn set_cell_state(&mut self, x: u32, y: u32, z: u32, state: CellState) {
        let idx = self.flat_index(x, y, z);
        let word = idx / 16;
        let bit_offset = (idx % 16) * 2;
        // Clear the 2-bit field, then set it
        self.occupancy[word] &= !(0b11 << bit_offset);
        self.occupancy[word] |= (state.as_u8() as u32) << bit_offset;
    }

    /// Look up the brick pool slot for the cell at `(x, y, z)`.
    ///
    /// Returns `None` if no brick is assigned ([`EMPTY_SLOT`]).
    pub fn brick_slot(&self, x: u32, y: u32, z: u32) -> Option<u32> {
        let idx = self.flat_index(x, y, z);
        let slot = self.slots[idx];
        if slot == EMPTY_SLOT {
            None
        } else {
            Some(slot)
        }
    }

    /// Assign a brick pool slot to the cell at `(x, y, z)`.
    pub fn set_brick_slot(&mut self, x: u32, y: u32, z: u32, slot: u32) {
        let idx = self.flat_index(x, y, z);
        self.slots[idx] = slot;
    }

    /// Clear the brick pool slot for the cell at `(x, y, z)`, resetting to [`EMPTY_SLOT`].
    pub fn clear_brick_slot(&mut self, x: u32, y: u32, z: u32) {
        let idx = self.flat_index(x, y, z);
        self.slots[idx] = EMPTY_SLOT;
    }

    /// The raw occupancy bitfield (Level 2) — for GPU upload.
    #[inline]
    pub fn occupancy_data(&self) -> &[u32] {
        &self.occupancy
    }

    /// The raw slot array (Level 1) — for GPU upload.
    #[inline]
    pub fn slot_data(&self) -> &[u32] {
        &self.slots
    }

    /// Count cells with the given state.
    pub fn count_cells(&self, state: CellState) -> u32 {
        let total = self.total_cells() as usize;
        let mut count = 0u32;
        for i in 0..total {
            let word = i / 16;
            let bit_offset = (i % 16) * 2;
            let bits = (self.occupancy[word] >> bit_offset) & 0b11;
            if bits == state.as_u8() as u32 {
                count += 1;
            }
        }
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ Construction ------

    #[test]
    fn new_grid_dimensions() {
        let grid = SparseGrid::new(UVec3::new(10, 20, 30));
        assert_eq!(grid.dimensions(), UVec3::new(10, 20, 30));
        assert_eq!(grid.total_cells(), 6000);
    }

    #[test]
    fn new_grid_all_empty() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(grid.cell_state(x, y, z), CellState::Empty);
                    assert_eq!(grid.brick_slot(x, y, z), None);
                }
            }
        }
    }

    #[test]
    fn new_grid_count_all_empty() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        assert_eq!(grid.count_cells(CellState::Empty), 64);
        assert_eq!(grid.count_cells(CellState::Surface), 0);
    }

    // ------ Cell state read/write ------

    #[test]
    fn set_cell_state_roundtrip_all_variants() {
        let mut grid = SparseGrid::new(UVec3::new(8, 8, 8));
        let states = [
            CellState::Empty,
            CellState::Surface,
            CellState::Interior,
            CellState::Volumetric,
        ];
        for (i, &state) in states.iter().enumerate() {
            let x = i as u32;
            grid.set_cell_state(x, 0, 0, state);
            assert_eq!(grid.cell_state(x, 0, 0), state, "state mismatch at x={x}");
        }
    }

    #[test]
    fn set_cell_state_does_not_corrupt_neighbors() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        // Set one cell to Surface
        grid.set_cell_state(1, 2, 3, CellState::Surface);
        // Neighbors should remain Empty
        assert_eq!(grid.cell_state(0, 2, 3), CellState::Empty);
        assert_eq!(grid.cell_state(2, 2, 3), CellState::Empty);
        assert_eq!(grid.cell_state(1, 1, 3), CellState::Empty);
        assert_eq!(grid.cell_state(1, 3, 3), CellState::Empty);
        assert_eq!(grid.cell_state(1, 2, 2), CellState::Empty);
        // The set cell should be Surface
        assert_eq!(grid.cell_state(1, 2, 3), CellState::Surface);
    }

    #[test]
    fn overwrite_cell_state() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Surface);
        grid.set_cell_state(0, 0, 0, CellState::Interior);
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Interior);
        grid.set_cell_state(0, 0, 0, CellState::Empty);
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Empty);
    }

    #[test]
    fn cell_states_across_word_boundary() {
        // 16 cells per u32 word. A 17×1×1 grid spans two words.
        let mut grid = SparseGrid::new(UVec3::new(17, 1, 1));
        // Set cell 15 (last in word 0) and cell 16 (first in word 1)
        grid.set_cell_state(15, 0, 0, CellState::Surface);
        grid.set_cell_state(16, 0, 0, CellState::Volumetric);
        assert_eq!(grid.cell_state(15, 0, 0), CellState::Surface);
        assert_eq!(grid.cell_state(16, 0, 0), CellState::Volumetric);
        // Cells 14 and earlier should still be Empty
        assert_eq!(grid.cell_state(14, 0, 0), CellState::Empty);
    }

    #[test]
    fn all_cells_in_same_word_independent() {
        // Test all 16 cells within one u32 word
        let mut grid = SparseGrid::new(UVec3::new(16, 1, 1));
        let states = [
            CellState::Empty,
            CellState::Surface,
            CellState::Interior,
            CellState::Volumetric,
        ];
        // Set each cell to a different state (cycling through 4 states)
        for x in 0..16u32 {
            grid.set_cell_state(x, 0, 0, states[(x % 4) as usize]);
        }
        // Verify all are correct
        for x in 0..16u32 {
            assert_eq!(
                grid.cell_state(x, 0, 0),
                states[(x % 4) as usize],
                "cell {x} has wrong state"
            );
        }
    }

    // ------ Brick slot read/write ------

    #[test]
    fn brick_slot_default_is_none() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        assert_eq!(grid.brick_slot(0, 0, 0), None);
    }

    #[test]
    fn set_brick_slot_roundtrip() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.set_brick_slot(2, 3, 1, 42);
        assert_eq!(grid.brick_slot(2, 3, 1), Some(42));
    }

    #[test]
    fn set_brick_slot_does_not_corrupt_neighbors() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.set_brick_slot(1, 1, 1, 99);
        assert_eq!(grid.brick_slot(0, 1, 1), None);
        assert_eq!(grid.brick_slot(2, 1, 1), None);
        assert_eq!(grid.brick_slot(1, 1, 1), Some(99));
    }

    #[test]
    fn clear_brick_slot() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.set_brick_slot(0, 0, 0, 7);
        assert_eq!(grid.brick_slot(0, 0, 0), Some(7));
        grid.clear_brick_slot(0, 0, 0);
        assert_eq!(grid.brick_slot(0, 0, 0), None);
    }

    // ------ GPU upload data ------

    #[test]
    fn occupancy_data_length() {
        // 4×4×4 = 64 cells, at 16 cells/word = 4 words
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        assert_eq!(grid.occupancy_data().len(), 4);
    }

    #[test]
    fn occupancy_data_rounds_up() {
        // 17 cells → needs 2 words (ceil(17/16))
        let grid = SparseGrid::new(UVec3::new(17, 1, 1));
        assert_eq!(grid.occupancy_data().len(), 2);
    }

    #[test]
    fn slot_data_length() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        assert_eq!(grid.slot_data().len(), 64);
    }

    #[test]
    fn slot_data_all_empty_sentinel() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        assert!(grid.slot_data().iter().all(|&s| s == EMPTY_SLOT));
    }

    // ------ count_cells ------

    #[test]
    fn count_cells_mixed() {
        let mut grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_cell_state(1, 0, 0, CellState::Surface);
        grid.set_cell_state(2, 0, 0, CellState::Interior);
        grid.set_cell_state(3, 0, 0, CellState::Volumetric);
        assert_eq!(grid.count_cells(CellState::Surface), 2);
        assert_eq!(grid.count_cells(CellState::Interior), 1);
        assert_eq!(grid.count_cells(CellState::Volumetric), 1);
        assert_eq!(grid.count_cells(CellState::Empty), 60); // 64 - 4
    }

    // ------ Edge cases ------

    #[test]
    fn grid_1x1x1() {
        let mut grid = SparseGrid::new(UVec3::new(1, 1, 1));
        assert_eq!(grid.total_cells(), 1);
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, 0);
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Surface);
        assert_eq!(grid.brick_slot(0, 0, 0), Some(0));
    }

    #[test]
    fn large_grid_corners() {
        let dim = 50;
        let mut grid = SparseGrid::new(UVec3::new(dim, dim, dim));
        // Set corners
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_cell_state(dim - 1, dim - 1, dim - 1, CellState::Interior);
        grid.set_brick_slot(0, 0, 0, 0);
        grid.set_brick_slot(dim - 1, dim - 1, dim - 1, 999);

        assert_eq!(grid.cell_state(0, 0, 0), CellState::Surface);
        assert_eq!(grid.cell_state(dim - 1, dim - 1, dim - 1), CellState::Interior);
        assert_eq!(grid.brick_slot(0, 0, 0), Some(0));
        assert_eq!(grid.brick_slot(dim - 1, dim - 1, dim - 1), Some(999));
        // Middle cell untouched
        assert_eq!(grid.cell_state(dim / 2, dim / 2, dim / 2), CellState::Empty);
    }

    #[test]
    #[should_panic]
    fn flat_index_out_of_bounds_x() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.cell_state(4, 0, 0);
    }

    #[test]
    #[should_panic]
    fn flat_index_out_of_bounds_y() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.cell_state(0, 4, 0);
    }

    #[test]
    #[should_panic]
    fn flat_index_out_of_bounds_z() {
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        grid.cell_state(0, 0, 4);
    }
}
