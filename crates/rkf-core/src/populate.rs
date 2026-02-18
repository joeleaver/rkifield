//! Integration between [`BrickPool`] and [`SparseGrid`] via [`populate_grid`].
//!
//! [`populate_grid`] voxelizes an analytic SDF into a brick pool and sparse grid,
//! allocating bricks from the pool rather than creating a standalone `Vec<Brick>`.

use glam::Vec3;

use crate::aabb::Aabb;
use crate::brick_pool::BrickPool;
use crate::cell_state::CellState;
use crate::constants::{BRICK_DIM, RESOLUTION_TIERS};
use crate::sparse_grid::SparseGrid;
use crate::voxel::VoxelSample;

/// Narrow band width in bricks from the surface.
const NARROW_BAND_BRICKS: u32 = 3;

/// Voxelize an analytic SDF into an existing [`BrickPool`] and [`SparseGrid`].
///
/// Allocates bricks from `pool`, fills them with SDF samples, and updates
/// `grid` with cell states and brick slot references.
///
/// # Parameters
/// - `pool`: mutable brick pool to allocate from
/// - `grid`: mutable sparse grid (must have correct dimensions for the AABB + tier)
/// - `sdf_fn`: closure returning signed distance at a world-space point
/// - `tier`: resolution tier index (0–3)
/// - `aabb`: world-space bounding box to voxelize
///
/// # Returns
/// Number of bricks allocated. Returns `Err` if the pool runs out of capacity.
pub fn populate_grid<F>(
    pool: &mut BrickPool,
    grid: &mut SparseGrid,
    sdf_fn: F,
    tier: usize,
    aabb: &Aabb,
) -> Result<u32, PopulateError>
where
    F: Fn(Vec3) -> f32,
{
    let res = &RESOLUTION_TIERS[tier];
    let voxel_size = res.voxel_size;
    let brick_extent = res.brick_extent;
    let narrow_band_dist = NARROW_BAND_BRICKS as f32 * brick_extent;

    let dims = grid.dimensions();
    let mut allocated = 0u32;

    for cz in 0..dims.z {
        for cy in 0..dims.y {
            for cx in 0..dims.x {
                let brick_min = aabb.min
                    + Vec3::new(
                        cx as f32 * brick_extent,
                        cy as f32 * brick_extent,
                        cz as f32 * brick_extent,
                    );
                let brick_center = brick_min + Vec3::splat(brick_extent * 0.5);
                let center_dist = sdf_fn(brick_center);

                if center_dist.abs() <= narrow_band_dist {
                    // Allocate a brick from the pool
                    let slot = pool.allocate().ok_or(PopulateError::PoolExhausted {
                        allocated_so_far: allocated,
                    })?;

                    let brick = pool.get_mut(slot);
                    for vz in 0..BRICK_DIM {
                        for vy in 0..BRICK_DIM {
                            for vx in 0..BRICK_DIM {
                                let voxel_pos = brick_min
                                    + Vec3::new(
                                        (vx as f32 + 0.5) * voxel_size,
                                        (vy as f32 + 0.5) * voxel_size,
                                        (vz as f32 + 0.5) * voxel_size,
                                    );
                                let dist = sdf_fn(voxel_pos);
                                brick.set(vx, vy, vz, VoxelSample::new(dist, 1, 0, 0, 0));
                            }
                        }
                    }

                    grid.set_cell_state(cx, cy, cz, CellState::Surface);
                    grid.set_brick_slot(cx, cy, cz, slot);
                    allocated += 1;
                } else if center_dist < 0.0 {
                    grid.set_cell_state(cx, cy, cz, CellState::Interior);
                }
            }
        }
    }

    Ok(allocated)
}

/// Error from [`populate_grid`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PopulateError {
    /// The brick pool ran out of free slots during population.
    PoolExhausted {
        /// Number of bricks successfully allocated before exhaustion.
        allocated_so_far: u32,
    },
}

impl std::fmt::Display for PopulateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PopulateError::PoolExhausted { allocated_so_far } => {
                write!(
                    f,
                    "brick pool exhausted after allocating {allocated_so_far} bricks"
                )
            }
        }
    }
}

impl std::error::Error for PopulateError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick_pool::Pool;
    use crate::sdf::{sphere_sdf, box_sdf};
    use glam::UVec3;

    /// Helper: create a grid with correct dimensions for an AABB + tier.
    fn grid_for_aabb(aabb: &Aabb, tier: usize) -> SparseGrid {
        let res = &RESOLUTION_TIERS[tier];
        let size = aabb.size();
        let dims = UVec3::new(
            ((size.x / res.brick_extent).ceil() as u32).max(1),
            ((size.y / res.brick_extent).ceil() as u32).max(1),
            ((size.z / res.brick_extent).ceil() as u32).max(1),
        );
        SparseGrid::new(dims)
    }

    #[test]
    fn populate_sphere_allocates_bricks() {
        let aabb = Aabb::new(Vec3::splat(-0.8), Vec3::splat(0.8));
        let mut pool: BrickPool = Pool::new(4096);
        let mut grid = grid_for_aabb(&aabb, 1);

        let count = populate_grid(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::ZERO, 0.5, p),
            1,
            &aabb,
        )
        .unwrap();

        assert!(count > 0, "should allocate some bricks");
        assert_eq!(pool.allocated_count(), count);
        assert_eq!(grid.count_cells(CellState::Surface), count);
    }

    #[test]
    fn populate_sphere_brick_count_matches_voxelize() {
        // populate_grid should produce the same number of bricks as voxelize_sdf
        let aabb = Aabb::new(Vec3::splat(-0.6), Vec3::splat(0.6));
        let sdf = |p: Vec3| sphere_sdf(Vec3::ZERO, 0.3, p);

        let (_ref_grid, ref_bricks) = crate::sdf::voxelize_sdf(sdf, 1, &aabb);

        let mut pool: BrickPool = Pool::new(4096);
        let mut grid = grid_for_aabb(&aabb, 1);
        let count = populate_grid(&mut pool, &mut grid, sdf, 1, &aabb).unwrap();

        assert_eq!(
            count,
            ref_bricks.len() as u32,
            "populate_grid and voxelize_sdf should produce same brick count"
        );
    }

    #[test]
    fn populate_sphere_has_interior_cells() {
        // Narrow band at tier 1 = 3 * 0.16 = 0.48m.
        // Need a sphere whose interior extends beyond 0.48m from the surface.
        // Radius 2.0m → interior region is a sphere of radius 2.0 - 0.48 = 1.52m.
        let aabb = Aabb::new(Vec3::splat(-3.0), Vec3::splat(3.0));
        let mut pool: BrickPool = Pool::new(65536);
        let mut grid = grid_for_aabb(&aabb, 1);

        populate_grid(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::ZERO, 2.0, p),
            1,
            &aabb,
        )
        .unwrap();

        let interior = grid.count_cells(CellState::Interior);
        assert!(interior > 0, "large sphere should have interior cells");
    }

    #[test]
    fn populate_box_allocates_bricks() {
        let aabb = Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5));
        let mut pool: BrickPool = Pool::new(4096);
        let mut grid = grid_for_aabb(&aabb, 1);

        let count = populate_grid(
            &mut pool,
            &mut grid,
            |p| box_sdf(Vec3::splat(0.2), p),
            1,
            &aabb,
        )
        .unwrap();

        assert!(count > 0);
    }

    #[test]
    fn populate_empty_sdf_allocates_nothing() {
        let aabb = Aabb::new(Vec3::splat(-0.1), Vec3::splat(0.1));
        let mut pool: BrickPool = Pool::new(64);
        let mut grid = grid_for_aabb(&aabb, 1);

        let count = populate_grid(&mut pool, &mut grid, |_| 100.0, 1, &aabb).unwrap();

        assert_eq!(count, 0);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn populate_pool_exhaustion_returns_error() {
        let aabb = Aabb::new(Vec3::splat(-0.8), Vec3::splat(0.8));
        let mut pool: BrickPool = Pool::new(2); // Very small pool
        let mut grid = grid_for_aabb(&aabb, 1);

        let result = populate_grid(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::ZERO, 0.5, p),
            1,
            &aabb,
        );

        assert!(result.is_err());
        if let Err(PopulateError::PoolExhausted { allocated_so_far }) = result {
            assert_eq!(allocated_so_far, 2);
        }
    }

    #[test]
    fn populated_bricks_have_valid_distances() {
        let aabb = Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5));
        let mut pool: BrickPool = Pool::new(4096);
        let mut grid = grid_for_aabb(&aabb, 1);

        populate_grid(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::ZERO, 0.2, p),
            1,
            &aabb,
        )
        .unwrap();

        // Check all Surface cells have bricks with finite distances
        let dims = grid.dimensions();
        for cz in 0..dims.z {
            for cy in 0..dims.y {
                for cx in 0..dims.x {
                    if let Some(slot) = grid.brick_slot(cx, cy, cz) {
                        let brick = pool.get(slot);
                        let d = brick.sample(0, 0, 0).distance_f32();
                        assert!(d.is_finite(), "brick at ({cx},{cy},{cz}) has non-finite distance");
                    }
                }
            }
        }
    }

    #[test]
    fn populate_different_tiers() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(1.0));
        let sdf = |p: Vec3| sphere_sdf(Vec3::splat(0.5), 0.3, p);

        let mut pool1: BrickPool = Pool::new(8192);
        let mut grid1 = grid_for_aabb(&aabb, 1);
        let c1 = populate_grid(&mut pool1, &mut grid1, sdf, 1, &aabb).unwrap();

        let mut pool2: BrickPool = Pool::new(8192);
        let mut grid2 = grid_for_aabb(&aabb, 2);
        let c2 = populate_grid(&mut pool2, &mut grid2, sdf, 2, &aabb).unwrap();

        // Finer tier should allocate more bricks
        assert!(
            c1 > c2,
            "tier 1 should allocate more bricks than tier 2: {c1} vs {c2}"
        );
    }
}
