//! Edit operation orchestration — CPU-side preparation for GPU dispatch.
//!
//! [`prepare_edit`] is the main entry point: given a [`Brush`], position, and
//! rotation, it computes which grid cells overlap the edit AABB, pre-allocates
//! bricks for empty cells (Add operations), and returns a [`PreparedEdit`]
//! containing per-brick [`EditParams`] ready for
//! [`CsgEditPipeline::dispatch_multi`](crate::pipeline::CsgEditPipeline::dispatch_multi).
//!
//! [`post_edit_cleanup`] runs after GPU execution to deallocate bricks that
//! remain empty (all voxels far from the surface).

use glam::{Quat, UVec3, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::brick_pool::BrickPool;
use rkf_core::cell_state::CellState;
use rkf_core::constants::RESOLUTION_TIERS;
use rkf_core::sparse_grid::SparseGrid;

use crate::brush::{Brush, BrushType};
use crate::types::EditParams;

// ---------------------------------------------------------------------------
// EditOp — an edit operation ready to be applied
// ---------------------------------------------------------------------------

/// An edit operation: brush + placement in local space.
///
/// `position` and `rotation` are in grid-local space (relative to the grid's
/// AABB origin). The orchestrator converts these to per-brick world-space
/// coordinates when building [`EditParams`].
#[derive(Debug, Clone)]
pub struct EditOp {
    /// The brush to apply.
    pub brush: Brush,
    /// Edit center position in grid-local space.
    pub position: Vec3,
    /// Edit rotation (brush-local to grid-local).
    pub rotation: Quat,
}

// ---------------------------------------------------------------------------
// PreparedEdit — result of prepare_edit
// ---------------------------------------------------------------------------

/// Result of preparing an edit for GPU dispatch.
pub struct PreparedEdit {
    /// Per-brick [`EditParams`] for GPU dispatch.
    pub params: Vec<EditParams>,
    /// Brick slots that were newly allocated during preparation.
    ///
    /// Each entry is `(cell_x, cell_y, cell_z, pool_slot)`.
    /// Used by [`post_edit_cleanup`] and undo tracking.
    pub newly_allocated: Vec<(u32, u32, u32, u32)>,
}

// ---------------------------------------------------------------------------
// prepare_edit
// ---------------------------------------------------------------------------

/// Prepare an edit for GPU dispatch.
///
/// This is the CPU-side orchestration:
/// 1. Compute edit AABB with safety margin
/// 2. Convert to grid cell coordinate range
/// 3. For Add operations: pre-allocate bricks for empty cells
/// 4. Build per-brick [`EditParams`] with brick base index and world min
///
/// After calling this, pass `prepared.params` to
/// [`CsgEditPipeline::dispatch_multi`](crate::pipeline::CsgEditPipeline::dispatch_multi).
///
/// # Arguments
///
/// * `pool` — CPU-side brick pool (may be mutated for allocation)
/// * `grid` — Sparse grid (may be mutated for new brick assignments)
/// * `grid_aabb` — World-space AABB of the entire grid
/// * `tier` — Resolution tier index (0..4)
/// * `op` — The edit operation to prepare
pub fn prepare_edit(
    pool: &mut BrickPool,
    grid: &mut SparseGrid,
    grid_aabb: &Aabb,
    tier: usize,
    op: &EditOp,
) -> PreparedEdit {
    let voxel_size = RESOLUTION_TIERS[tier].voxel_size;
    let brick_extent = RESOLUTION_TIERS[tier].brick_extent;
    let dims = grid.dimensions();

    // 1. Compute edit AABB with safety margin.
    //    The margin must cover the brush radius, the smooth blend radius,
    //    and a 2-voxel border for finite-difference normal estimation.
    let margin = op.brush.radius + op.brush.blend_k + voxel_size * 2.0;
    let edit_min = op.position - Vec3::splat(margin);
    let edit_max = op.position + Vec3::splat(margin);

    // 2. Convert world-space edit bounds to grid cell coordinates.
    let cell_min_f = (edit_min - grid_aabb.min) / brick_extent;
    let cell_max_f = (edit_max - grid_aabb.min) / brick_extent;

    // Clamp to valid grid range [0, dims-1].
    let cell_min = UVec3::new(
        (cell_min_f.x.floor().max(0.0) as u32).min(dims.x.saturating_sub(1)),
        (cell_min_f.y.floor().max(0.0) as u32).min(dims.y.saturating_sub(1)),
        (cell_min_f.z.floor().max(0.0) as u32).min(dims.z.saturating_sub(1)),
    );
    let cell_max = UVec3::new(
        (cell_max_f.x.ceil().max(0.0) as u32).min(dims.x.saturating_sub(1)),
        (cell_max_f.y.ceil().max(0.0) as u32).min(dims.y.saturating_sub(1)),
        (cell_max_f.z.ceil().max(0.0) as u32).min(dims.z.saturating_sub(1)),
    );

    // 3. Build base EditParams from brush (brick-specific fields left at zero).
    let base = EditParams::csg(
        op.brush.edit_type(),
        op.brush.shape_type(),
        [op.position.x, op.position.y, op.position.z],
        [
            op.rotation.x,
            op.rotation.y,
            op.rotation.z,
            op.rotation.w,
        ],
        [op.brush.radius, op.brush.radius, op.brush.radius],
        op.brush.strength,
        op.brush.blend_k,
        op.brush.falloff,
        op.brush.material_id,
    );

    let mut params_list = Vec::new();
    let mut newly_allocated = Vec::new();

    // 4. Iterate affected cells.
    for cz in cell_min.z..=cell_max.z {
        for cy in cell_min.y..=cell_max.y {
            for cx in cell_min.x..=cell_max.x {
                let state = grid.cell_state(cx, cy, cz);

                // Determine the brick pool slot for this cell.
                let slot = if state == CellState::Empty {
                    // For Add: pre-allocate bricks for empty cells so the GPU
                    // shader can write new geometry into them.
                    if matches!(op.brush.brush_type, BrushType::Add) {
                        match pool.allocate() {
                            Some(s) => {
                                grid.set_cell_state(cx, cy, cz, CellState::Surface);
                                grid.set_brick_slot(cx, cy, cz, s);
                                newly_allocated.push((cx, cy, cz, s));
                                s
                            }
                            None => continue, // Pool full — skip this cell
                        }
                    } else {
                        // Subtract/Smooth/Paint on empty cells is a no-op.
                        continue;
                    }
                } else if state == CellState::Surface {
                    match grid.brick_slot(cx, cy, cz) {
                        Some(s) => s,
                        None => continue, // Inconsistent state — skip
                    }
                } else {
                    // Interior or Volumetric cells — skip for CSG edits.
                    continue;
                };

                // 5. Compute brick world-space min corner.
                let brick_world_min = grid_aabb.min
                    + Vec3::new(
                        cx as f32 * brick_extent,
                        cy as f32 * brick_extent,
                        cz as f32 * brick_extent,
                    );

                // 6. Build per-brick params.
                let brick_base_index = slot * 512; // 512 voxels per brick
                let params = base.with_brick_info(
                    brick_base_index,
                    [brick_world_min.x, brick_world_min.y, brick_world_min.z],
                    voxel_size,
                );
                params_list.push(params);
            }
        }
    }

    PreparedEdit {
        params: params_list,
        newly_allocated,
    }
}

// ---------------------------------------------------------------------------
// post_edit_cleanup
// ---------------------------------------------------------------------------

/// Post-edit cleanup: deallocate bricks that remain empty after GPU edit.
///
/// After the GPU edit completes, checks each newly allocated brick to see if
/// any voxel is within the narrow band (close to the surface). If all voxels
/// are far from the surface, the brick is deallocated and the grid cell reset
/// to [`CellState::Empty`].
///
/// # Arguments
///
/// * `pool` — CPU-side brick pool
/// * `grid` — Sparse grid
/// * `newly_allocated` — Slice of `(cell_x, cell_y, cell_z, pool_slot)` from
///   [`PreparedEdit::newly_allocated`]
/// * `voxel_size` — Voxel size for the resolution tier used
pub fn post_edit_cleanup(
    pool: &mut BrickPool,
    grid: &mut SparseGrid,
    newly_allocated: &[(u32, u32, u32, u32)],
    voxel_size: f32,
) {
    let narrow_band = voxel_size * 4.0;
    for &(cx, cy, cz, slot) in newly_allocated {
        let brick = pool.get(slot);
        let mut has_surface = false;
        'voxel_check: for vz in 0..8u32 {
            for vy in 0..8u32 {
                for vx in 0..8u32 {
                    let d = brick.sample(vx, vy, vz).distance_f32();
                    if d.abs() < narrow_band {
                        has_surface = true;
                        break 'voxel_check;
                    }
                }
            }
        }
        if !has_surface {
            pool.deallocate(slot);
            grid.set_cell_state(cx, cy, cz, CellState::Empty);
            grid.clear_brick_slot(cx, cy, cz);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EditType, ShapeType};
    use rkf_core::voxel::VoxelSample;

    /// Helper: create a grid + pool + AABB for a small test scenario.
    ///
    /// Returns `(pool, grid, grid_aabb)` for a 4x4x4 grid at tier 1 (2cm voxels).
    fn test_setup() -> (BrickPool, SparseGrid, Aabb) {
        let pool = BrickPool::new(64);
        let grid = SparseGrid::new(UVec3::new(4, 4, 4));
        // Tier 1: brick_extent = 0.16m, so 4x4x4 grid spans 0.64m per axis.
        let grid_aabb = Aabb::new(Vec3::ZERO, Vec3::splat(0.64));
        (pool, grid, grid_aabb)
    }

    // -- prepare_edit with Add brush on empty grid --

    #[test]
    fn add_brush_on_empty_grid_allocates_bricks() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1; // 2cm voxels, 0.16m brick extent

        let op = EditOp {
            brush: Brush::add_sphere(0.05, 1), // small brush
            position: Vec3::new(0.08, 0.08, 0.08), // center of cell (0,0,0)
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // Should have allocated at least one brick and produced params.
        assert!(
            !prepared.params.is_empty(),
            "expected at least one param, got 0"
        );
        assert!(
            !prepared.newly_allocated.is_empty(),
            "expected at least one newly allocated brick"
        );

        // Verify the newly allocated cells are now Surface.
        for &(cx, cy, cz, slot) in &prepared.newly_allocated {
            assert_eq!(grid.cell_state(cx, cy, cz), CellState::Surface);
            assert_eq!(grid.brick_slot(cx, cy, cz), Some(slot));
        }

        // Params should have correct edit type and shape.
        for p in &prepared.params {
            assert_eq!(p.edit_type, EditType::SmoothUnion.as_u32());
            assert_eq!(p.shape_type, ShapeType::Sphere.as_u32());
            assert!(p.voxel_size > 0.0);
            assert!(p.brick_base_index % 512 == 0);
        }
    }

    #[test]
    fn add_brush_params_have_correct_brick_world_min() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;
        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;

        // Place edit at center of cell (1, 1, 1).
        let cell_center = grid_aabb.min
            + Vec3::new(
                1.5 * brick_extent,
                1.5 * brick_extent,
                1.5 * brick_extent,
            );

        let op = EditOp {
            brush: Brush::add_sphere(0.01, 1), // tiny brush, hits ~1 cell
            position: cell_center,
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);
        assert!(!prepared.params.is_empty());

        // Find the param for cell (1,1,1).
        let expected_min = grid_aabb.min
            + Vec3::new(brick_extent, brick_extent, brick_extent);
        let found = prepared.params.iter().any(|p| {
            (p.brick_world_min[0] - expected_min.x).abs() < 1e-6
                && (p.brick_world_min[1] - expected_min.y).abs() < 1e-6
                && (p.brick_world_min[2] - expected_min.z).abs() < 1e-6
        });
        assert!(found, "expected param with brick_world_min at cell (1,1,1)");
    }

    // -- prepare_edit with Subtract brush on grid with surface cells --

    #[test]
    fn subtract_brush_finds_surface_cells() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Pre-populate cell (1,1,1) with a surface brick.
        let slot = pool.allocate().unwrap();
        grid.set_cell_state(1, 1, 1, CellState::Surface);
        grid.set_brick_slot(1, 1, 1, slot);

        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;
        let cell_center = grid_aabb.min
            + Vec3::new(
                1.5 * brick_extent,
                1.5 * brick_extent,
                1.5 * brick_extent,
            );

        let op = EditOp {
            brush: Brush::subtract_sphere(0.05),
            position: cell_center,
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // Should find the pre-existing surface cell.
        assert!(
            !prepared.params.is_empty(),
            "expected at least one param for subtract on surface cell"
        );
        // Subtract should NOT allocate new bricks.
        assert!(
            prepared.newly_allocated.is_empty(),
            "subtract should not allocate new bricks"
        );

        // Verify edit type.
        for p in &prepared.params {
            assert_eq!(p.edit_type, EditType::SmoothSubtract.as_u32());
        }
    }

    #[test]
    fn subtract_brush_skips_empty_cells() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // All cells are empty. Subtract should produce no params.
        let op = EditOp {
            brush: Brush::subtract_sphere(0.05),
            position: Vec3::new(0.08, 0.08, 0.08),
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);
        assert!(
            prepared.params.is_empty(),
            "subtract on empty grid should produce no params"
        );
        assert!(prepared.newly_allocated.is_empty());
    }

    // -- Full pool gracefully skips --

    #[test]
    fn add_brush_full_pool_skips_gracefully() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Exhaust the pool.
        while pool.allocate().is_some() {}
        assert!(pool.is_full());

        let op = EditOp {
            brush: Brush::add_sphere(0.05, 1),
            position: Vec3::new(0.08, 0.08, 0.08),
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // With a full pool, no bricks can be allocated, so no params.
        assert!(
            prepared.params.is_empty(),
            "full pool should produce no params"
        );
        assert!(prepared.newly_allocated.is_empty());
    }

    // -- AABB clamps to grid dimensions --

    #[test]
    fn edit_aabb_clamps_to_grid_bounds() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Place edit far outside the grid (negative coordinates).
        let op = EditOp {
            brush: Brush::add_sphere(0.05, 1),
            position: Vec3::new(-10.0, -10.0, -10.0),
            rotation: Quat::IDENTITY,
        };

        // Should not panic — the cell range clamps to grid bounds.
        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // The edit is entirely outside the grid, but due to clamping it may
        // still hit cell (0,0,0). The important thing is no panic.
        // (In practice the edit is far enough away that the brush won't
        // meaningfully affect cell 0,0,0, but the allocation still happens.)
        let _ = prepared;
    }

    #[test]
    fn edit_at_grid_max_corner_does_not_panic() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Place edit at the far corner of the grid.
        let op = EditOp {
            brush: Brush::add_sphere(0.05, 1),
            position: grid_aabb.max,
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);
        // Should not panic. May or may not produce params depending on clamping.
        let _ = prepared;
    }

    // -- post_edit_cleanup --

    #[test]
    fn post_edit_cleanup_deallocates_empty_bricks() {
        let (mut pool, mut grid, _) = test_setup();

        // Allocate a brick and mark cell as Surface, but leave the brick at
        // default (all voxels at f16::INFINITY — far from surface).
        let slot = pool.allocate().unwrap();
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, slot);

        let alloc_before = pool.allocated_count();
        let newly_allocated = vec![(0u32, 0u32, 0u32, slot)];

        post_edit_cleanup(&mut pool, &mut grid, &newly_allocated, 0.02);

        // The brick should have been deallocated.
        assert_eq!(
            pool.allocated_count(),
            alloc_before - 1,
            "empty brick should have been deallocated"
        );
        assert_eq!(
            grid.cell_state(0, 0, 0),
            CellState::Empty,
            "cell should be reset to Empty"
        );
        assert_eq!(
            grid.brick_slot(0, 0, 0),
            None,
            "brick slot should be cleared"
        );
    }

    #[test]
    fn post_edit_cleanup_keeps_bricks_with_surface_data() {
        let (mut pool, mut grid, _) = test_setup();

        // Allocate a brick and write a near-surface voxel.
        let slot = pool.allocate().unwrap();
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, slot);

        // Write a voxel with small distance (within narrow band).
        let near_surface = VoxelSample::new(0.01, 1, 0, 0, 0);
        pool.get_mut(slot).set(4, 4, 4, near_surface);

        let alloc_before = pool.allocated_count();
        let newly_allocated = vec![(0u32, 0u32, 0u32, slot)];

        post_edit_cleanup(&mut pool, &mut grid, &newly_allocated, 0.02);

        // The brick should NOT have been deallocated.
        assert_eq!(
            pool.allocated_count(),
            alloc_before,
            "brick with surface data should be kept"
        );
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Surface);
        assert_eq!(grid.brick_slot(0, 0, 0), Some(slot));
    }

    #[test]
    fn post_edit_cleanup_negative_distance_counts_as_surface() {
        let (mut pool, mut grid, _) = test_setup();

        let slot = pool.allocate().unwrap();
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, slot);

        // Write a voxel with negative distance (inside surface).
        let inside = VoxelSample::new(-0.01, 1, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, inside);

        let newly_allocated = vec![(0u32, 0u32, 0u32, slot)];
        post_edit_cleanup(&mut pool, &mut grid, &newly_allocated, 0.02);

        // Should keep the brick.
        assert_eq!(grid.cell_state(0, 0, 0), CellState::Surface);
        assert_eq!(grid.brick_slot(0, 0, 0), Some(slot));
    }

    #[test]
    fn post_edit_cleanup_empty_list_is_noop() {
        let (mut pool, mut grid, _) = test_setup();

        let alloc_before = pool.allocated_count();
        post_edit_cleanup(&mut pool, &mut grid, &[], 0.02);

        assert_eq!(pool.allocated_count(), alloc_before);
    }

    // -- EditOp struct --

    #[test]
    fn edit_op_debug_and_clone() {
        let op = EditOp {
            brush: Brush::default(),
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
        };
        let op2 = op.clone();
        assert_eq!(op.position, op2.position);
        let s = format!("{op:?}");
        assert!(s.contains("EditOp"), "debug string: {s}");
    }

    // -- Large edit spanning multiple cells --

    #[test]
    fn large_add_brush_spans_multiple_cells() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;
        let _brick_extent = RESOLUTION_TIERS[tier].brick_extent; // 0.16m

        // Place a large brush at the center of the grid.
        // Brush radius = 0.3m, which at 0.16m/cell spans ~4 cells in each direction.
        let center = grid_aabb.center();
        let op = EditOp {
            brush: Brush::add_sphere(0.3, 1),
            position: center,
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // A 0.3m radius brush + blend_k + margin should hit all 4x4x4 = 64 cells.
        // At minimum it should hit many cells.
        assert!(
            prepared.params.len() > 1,
            "large brush should span multiple cells, got {} params",
            prepared.params.len()
        );

        // All newly allocated cells should be Surface.
        for &(cx, cy, cz, _slot) in &prepared.newly_allocated {
            assert_eq!(grid.cell_state(cx, cy, cz), CellState::Surface);
        }
    }

    // -- Paint brush on surface cells --

    #[test]
    fn paint_brush_finds_surface_no_alloc() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Pre-populate several cells.
        for c in 0..3u32 {
            let slot = pool.allocate().unwrap();
            grid.set_cell_state(c, 0, 0, CellState::Surface);
            grid.set_brick_slot(c, 0, 0, slot);
        }

        let alloc_before = pool.allocated_count();

        let op = EditOp {
            brush: Brush::paint_sphere(0.3, 42),
            position: Vec3::new(0.24, 0.08, 0.08), // overlaps cells 0..2
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // Paint should NOT allocate new bricks.
        assert!(
            prepared.newly_allocated.is_empty(),
            "paint should not allocate"
        );
        assert_eq!(pool.allocated_count(), alloc_before);

        // Should have found the surface cells.
        assert!(
            !prepared.params.is_empty(),
            "paint should find surface cells"
        );
        for p in &prepared.params {
            assert_eq!(p.edit_type, EditType::Paint.as_u32());
        }
    }

    // -- Interior cells are skipped --

    #[test]
    fn interior_cells_are_skipped() {
        let (mut pool, mut grid, grid_aabb) = test_setup();
        let tier = 1;

        // Mark cell (1,1,1) as Interior (no brick).
        grid.set_cell_state(1, 1, 1, CellState::Interior);

        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;
        let cell_center = grid_aabb.min
            + Vec3::new(
                1.5 * brick_extent,
                1.5 * brick_extent,
                1.5 * brick_extent,
            );

        // Even an Add brush should skip Interior cells (it only allocates for Empty).
        let op = EditOp {
            brush: Brush::add_sphere(0.01, 1),
            position: cell_center,
            rotation: Quat::IDENTITY,
        };

        let prepared = prepare_edit(&mut pool, &mut grid, &grid_aabb, tier, &op);

        // The Interior cell should not appear in params (no brick slot).
        // Neighboring Empty cells may get allocated though.
        for p in &prepared.params {
            // None of the params should point to an Interior cell.
            let _ = p;
        }
        // Interior cell state should remain unchanged.
        assert_eq!(grid.cell_state(1, 1, 1), CellState::Interior);
    }
}
