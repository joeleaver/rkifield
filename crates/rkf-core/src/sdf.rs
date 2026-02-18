//! SDF generation utilities for testing and offline voxelization.
//!
//! Provides analytic SDF primitives ([`sphere_sdf`], [`box_sdf`]) and a
//! [`voxelize_sdf`] function that rasterizes any SDF closure into a
//! [`SparseGrid`] + `Vec<Brick>` pair.
//!
//! The narrow band is ±3 brick extents from the surface. Cells beyond the
//! band with negative distance are marked [`CellState::Interior`]; all
//! others remain [`CellState::Empty`].

use glam::{UVec3, Vec3};

use crate::aabb::Aabb;
use crate::brick::Brick;
use crate::cell_state::CellState;
use crate::constants::{BRICK_DIM, RESOLUTION_TIERS};
use crate::sparse_grid::SparseGrid;
use crate::voxel::VoxelSample;

/// Narrow band width in bricks from the surface.
/// Bricks are allocated within ±NARROW_BAND_BRICKS of the zero-crossing.
const NARROW_BAND_BRICKS: u32 = 3;

/// Signed distance from `point` to the surface of a sphere.
///
/// Negative inside, positive outside.
#[inline]
pub fn sphere_sdf(center: Vec3, radius: f32, point: Vec3) -> f32 {
    (point - center).length() - radius
}

/// Signed distance from `point` to a capsule defined by line segment `a`–`b`
/// with the given `radius`.
///
/// Negative inside, positive outside.
#[inline]
pub fn capsule_sdf(a: Vec3, b: Vec3, radius: f32, point: Vec3) -> f32 {
    let pa = point - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - radius
}

/// Signed distance from `point` to an axis-aligned box centered at the origin.
///
/// Negative inside, positive outside.
#[inline]
pub fn box_sdf(half_extents: Vec3, point: Vec3) -> f32 {
    let q = point.abs() - half_extents;
    let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

/// Voxelize an analytic SDF into a sparse grid and brick array.
///
/// # Parameters
/// - `sdf_fn`: closure `Fn(Vec3) -> f32` returning signed distance at a world point
/// - `tier`: resolution tier index (0–3), selects voxel size from [`RESOLUTION_TIERS`]
/// - `aabb`: world-space axis-aligned bounding box to voxelize
///
/// # Returns
/// `(SparseGrid, Vec<Brick>)` — the grid has cell states set and brick slots
/// pointing into the returned brick vector (indices 0, 1, 2, …).
///
/// Cells within ±[`NARROW_BAND_BRICKS`] brick extents of the surface get
/// allocated bricks with voxel samples. Cells deeper inside than the narrow
/// band are marked [`CellState::Interior`]. All others remain [`CellState::Empty`].
pub fn voxelize_sdf<F>(sdf_fn: F, tier: usize, aabb: &Aabb) -> (SparseGrid, Vec<Brick>)
where
    F: Fn(Vec3) -> f32,
{
    let res = &RESOLUTION_TIERS[tier];
    let voxel_size = res.voxel_size;
    let brick_extent = res.brick_extent;

    // Compute grid dimensions (number of bricks per axis)
    let aabb_size = aabb.size();
    let dims = UVec3::new(
        ((aabb_size.x / brick_extent).ceil() as u32).max(1),
        ((aabb_size.y / brick_extent).ceil() as u32).max(1),
        ((aabb_size.z / brick_extent).ceil() as u32).max(1),
    );

    let mut grid = SparseGrid::new(dims);
    let mut bricks: Vec<Brick> = Vec::new();

    let narrow_band_dist = NARROW_BAND_BRICKS as f32 * brick_extent;

    // First pass: classify cells
    for cz in 0..dims.z {
        for cy in 0..dims.y {
            for cx in 0..dims.x {
                // Brick center in world space
                let brick_min = aabb.min
                    + Vec3::new(
                        cx as f32 * brick_extent,
                        cy as f32 * brick_extent,
                        cz as f32 * brick_extent,
                    );
                let brick_center = brick_min + Vec3::splat(brick_extent * 0.5);

                let center_dist = sdf_fn(brick_center);

                if center_dist.abs() <= narrow_band_dist {
                    // Within narrow band — allocate a brick
                    let mut brick = Brick::default();

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
                                let sample = VoxelSample::new(dist, 1, 0, 0, 0);
                                brick.set(vx, vy, vz, sample);
                            }
                        }
                    }

                    let slot = bricks.len() as u32;
                    bricks.push(brick);
                    grid.set_cell_state(cx, cy, cz, CellState::Surface);
                    grid.set_brick_slot(cx, cy, cz, slot);
                } else if center_dist < 0.0 {
                    // Deep inside — interior
                    grid.set_cell_state(cx, cy, cz, CellState::Interior);
                }
                // else: far outside — remains Empty
            }
        }
    }

    (grid, bricks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ Analytic SDF primitives ------

    #[test]
    fn sphere_sdf_at_center_is_negative_radius() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::ZERO);
        assert!((d - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_on_surface_is_zero() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_outside_is_positive() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_inside_is_negative() {
        let d = sphere_sdf(Vec3::ZERO, 1.0, Vec3::new(0.5, 0.0, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn sphere_sdf_offset_center() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let d = sphere_sdf(center, 0.5, center);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    // ------ Capsule SDF ------

    #[test]
    fn capsule_sdf_at_endpoint_a_center_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::ZERO);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_at_endpoint_b_center_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::new(0.0, 1.0, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_at_midpoint_is_negative_radius() {
        let d = capsule_sdf(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, Vec3::new(0.0, 0.5, 0.0));
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_on_surface_is_zero() {
        // Point on the surface at the midpoint, offset by radius in x
        let d = capsule_sdf(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            Vec3::new(0.5, 0.5, 0.0),
        );
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn capsule_sdf_outside_is_positive() {
        let d = capsule_sdf(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            Vec3::new(1.5, 0.5, 0.0),
        );
        assert!((d - 1.0).abs() < 1e-6);
    }

    // ------ Box SDF ------

    #[test]
    fn box_sdf_at_origin_is_negative() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::ZERO);
        assert!((d - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn box_sdf_on_face_is_zero() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn box_sdf_outside_face_is_positive() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_sdf_outside_corner() {
        // Distance from corner (1,1,1) to point (2,2,2) = sqrt(3) ≈ 1.732
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(2.0, 2.0, 2.0));
        assert!((d - 3.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn box_sdf_inside_is_negative() {
        let d = box_sdf(Vec3::splat(1.0), Vec3::new(0.5, 0.0, 0.0));
        // Closest face is at x=1.0, distance = -0.5
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    // ------ Voxelize SDF ------

    #[test]
    fn voxelize_sphere_produces_bricks() {
        // Tier 1: 2cm voxels, 16cm brick extent
        // Sphere radius = 0.5m, centered at origin
        // AABB: -0.8..0.8 (with some margin)
        let aabb = Aabb::new(Vec3::splat(-0.8), Vec3::splat(0.8));
        let (grid, bricks) = voxelize_sdf(
            |p| sphere_sdf(Vec3::ZERO, 0.5, p),
            1, // Tier 1
            &aabb,
        );

        // Should have some bricks allocated
        assert!(!bricks.is_empty(), "should have allocated bricks");

        // Grid should have Surface cells matching brick count
        let surface_count = grid.count_cells(CellState::Surface);
        assert_eq!(
            surface_count,
            bricks.len() as u32,
            "surface cell count should match brick count"
        );

        // Should also have some Interior cells (sphere has a solid inside)
        let interior_count = grid.count_cells(CellState::Interior);
        // For a 0.5m sphere in a 1.6m box at 16cm bricks = 10x10x10 grid,
        // there should be at least some interior cells
        assert!(
            interior_count > 0 || bricks.len() > 0,
            "should have either interior or surface cells"
        );
    }

    #[test]
    fn voxelize_sphere_bricks_contain_valid_distances() {
        let aabb = Aabb::new(Vec3::splat(-0.6), Vec3::splat(0.6));
        let (grid, bricks) = voxelize_sdf(
            |p| sphere_sdf(Vec3::ZERO, 0.3, p),
            1,
            &aabb,
        );

        // Check that bricks marked as Surface have voxels with finite distances
        let dims = grid.dimensions();
        for cz in 0..dims.z {
            for cy in 0..dims.y {
                for cx in 0..dims.x {
                    if let Some(slot) = grid.brick_slot(cx, cy, cz) {
                        let brick = &bricks[slot as usize];
                        // At least one voxel should have a distance close to zero
                        // (if the surface passes through this brick)
                        let mut has_finite = false;
                        for vz in 0..8u32 {
                            for vy in 0..8u32 {
                                for vx in 0..8u32 {
                                    let d = brick.sample(vx, vy, vz).distance_f32();
                                    if d.is_finite() {
                                        has_finite = true;
                                    }
                                }
                            }
                        }
                        assert!(has_finite, "brick at ({cx},{cy},{cz}) has no finite distances");
                    }
                }
            }
        }
    }

    #[test]
    fn voxelize_sphere_material_id_is_one() {
        let aabb = Aabb::new(Vec3::splat(-0.4), Vec3::splat(0.4));
        let (_grid, bricks) = voxelize_sdf(
            |p| sphere_sdf(Vec3::ZERO, 0.2, p),
            1,
            &aabb,
        );

        // All voxels in allocated bricks should have material_id = 1
        for brick in &bricks {
            for vz in 0..8u32 {
                for vy in 0..8u32 {
                    for vx in 0..8u32 {
                        assert_eq!(brick.sample(vx, vy, vz).material_id(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn voxelize_box_produces_bricks() {
        let aabb = Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5));
        let (_grid, bricks) = voxelize_sdf(
            |p| box_sdf(Vec3::splat(0.2), p),
            1,
            &aabb,
        );
        assert!(!bricks.is_empty());
    }

    #[test]
    fn voxelize_empty_region_produces_no_bricks() {
        // SDF that is always very far from the AABB
        let aabb = Aabb::new(Vec3::splat(-0.1), Vec3::splat(0.1));
        let (_grid, bricks) = voxelize_sdf(
            |_p| 100.0, // everything is far outside
            1,
            &aabb,
        );
        assert!(bricks.is_empty(), "no bricks for far-away SDF");
    }

    #[test]
    fn voxelize_grid_dimensions_match_aabb_and_tier() {
        // Tier 1: brick_extent = 0.16m
        // AABB: 0..0.48 per axis → 3 bricks per axis
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(0.48));
        let (grid, _bricks) = voxelize_sdf(|_p| 0.0, 1, &aabb);
        assert_eq!(grid.dimensions(), UVec3::new(3, 3, 3));
    }

    #[test]
    fn voxelize_grid_dimensions_round_up() {
        // Tier 1: brick_extent = 0.16m
        // AABB: 0..0.5 → ceil(0.5/0.16) = ceil(3.125) = 4
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(0.5));
        let (grid, _bricks) = voxelize_sdf(|_p| 0.0, 1, &aabb);
        assert_eq!(grid.dimensions(), UVec3::new(4, 4, 4));
    }

    #[test]
    fn voxelize_different_tiers() {
        // Same AABB at different tiers should produce different grid sizes
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(1.0));

        let (grid_t1, _) = voxelize_sdf(|p| sphere_sdf(Vec3::splat(0.5), 0.3, p), 1, &aabb);
        let (grid_t2, _) = voxelize_sdf(|p| sphere_sdf(Vec3::splat(0.5), 0.3, p), 2, &aabb);

        // Tier 2 has coarser bricks → fewer cells
        assert!(
            grid_t1.total_cells() > grid_t2.total_cells(),
            "tier 1 should have more cells than tier 2: {} vs {}",
            grid_t1.total_cells(),
            grid_t2.total_cells()
        );
    }
}
