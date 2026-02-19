//! LOD pre-computation: downsample voxelized SDF data to coarser resolution tiers.
//!
//! After voxelizing a mesh at a fine resolution tier, this module generates
//! coarser LOD tiers by averaging distance values and picking majority materials
//! from the source data. Each tier step is 4x coarser (matching the engine's
//! [`RESOLUTION_TIERS`] progression).
//!
//! The downsampling reads directly from the source (finest) tier's grid and pool,
//! producing independent [`LodTier`] results for each coarser level.

use glam::UVec3;
use rkf_core::aabb::Aabb;
use rkf_core::brick_pool::BrickPool;
use rkf_core::cell_state::CellState;
use rkf_core::constants::RESOLUTION_TIERS;
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::voxel::VoxelSample;

/// A single LOD tier's data.
#[derive(Debug)]
pub struct LodTier {
    /// Resolution tier index (into [`RESOLUTION_TIERS`]).
    pub tier: usize,
    /// Grid for this tier.
    pub grid: SparseGrid,
    /// Brick pool for this tier.
    pub pool: BrickPool,
    /// World-space AABB.
    pub aabb: Aabb,
    /// Number of allocated bricks.
    pub brick_count: u32,
}

/// Generate LOD tiers by downsampling from a source tier.
///
/// For each coarser tier: each voxel in the coarser tier corresponds to a
/// `ratio x ratio x ratio` region in the source tier (where `ratio = 4^steps`).
/// Distances are averaged and materials are selected by majority vote.
///
/// Each coarser tier is generated directly from the source (finest) tier,
/// not chained through intermediate results. This avoids accumulating
/// quantization error and simplifies ownership (no `Clone` on `BrickPool`).
///
/// Returns an empty `Vec` if `num_coarser_tiers` is 0 or `source_tier` is
/// already at the coarsest tier.
pub fn generate_lod_tiers(
    source_grid: &SparseGrid,
    source_pool: &BrickPool,
    source_aabb: &Aabb,
    source_tier: usize,
    num_coarser_tiers: u32,
) -> Vec<LodTier> {
    let mut tiers = Vec::new();

    for i in 1..=num_coarser_tiers {
        let dst_tier = source_tier + i as usize;
        if dst_tier >= RESOLUTION_TIERS.len() {
            break;
        }

        let result = downsample_from_source(
            source_grid,
            source_pool,
            source_aabb,
            source_tier,
            dst_tier,
        );

        tiers.push(result);
    }

    tiers
}

/// Downsample from the source tier directly to a target coarser tier.
///
/// The voxel ratio between tiers is `4^(dst_tier - src_tier)`. Each destination
/// voxel averages that many source voxels along each axis.
fn downsample_from_source(
    src_grid: &SparseGrid,
    src_pool: &BrickPool,
    src_aabb: &Aabb,
    src_tier: usize,
    dst_tier: usize,
) -> LodTier {
    let src_voxel = RESOLUTION_TIERS[src_tier].voxel_size;
    let dst_voxel = RESOLUTION_TIERS[dst_tier].voxel_size;
    let dst_brick_extent = RESOLUTION_TIERS[dst_tier].brick_extent;

    // Total voxel ratio: how many source voxels per destination voxel per axis
    let voxel_ratio = (dst_voxel / src_voxel).round() as u32;

    // Brick ratio: how many source bricks correspond to one destination brick
    // Since bricks are 8 voxels wide, and voxel_ratio is 4^steps:
    //   dst brick covers 8 * dst_voxel in world space
    //   In source bricks, that's (8 * dst_voxel) / (8 * src_voxel) = voxel_ratio
    let brick_ratio = voxel_ratio;

    let src_dims = src_grid.dimensions();
    let dst_dims = UVec3::new(
        src_dims.x.div_ceil(brick_ratio),
        src_dims.y.div_ceil(brick_ratio),
        src_dims.z.div_ceil(brick_ratio),
    )
    .max(UVec3::ONE);

    let dst_aabb = Aabb::new(
        src_aabb.min,
        src_aabb.min + dst_dims.as_vec3() * dst_brick_extent,
    );

    let max_bricks = (dst_dims.x * dst_dims.y * dst_dims.z).min(65536);
    let mut dst_grid = SparseGrid::new(dst_dims);
    let mut dst_pool = BrickPool::new(max_bricks);
    let mut brick_count = 0u32;

    for dz in 0..dst_dims.z {
        for dy in 0..dst_dims.y {
            for dx in 0..dst_dims.x {
                // Check if any source bricks exist in the corresponding region
                let src_start = UVec3::new(
                    dx * brick_ratio,
                    dy * brick_ratio,
                    dz * brick_ratio,
                );
                let mut has_surface = false;

                for sz in src_start.z..(src_start.z + brick_ratio).min(src_dims.z) {
                    for sy in src_start.y..(src_start.y + brick_ratio).min(src_dims.y) {
                        for sx in src_start.x..(src_start.x + brick_ratio).min(src_dims.x) {
                            if src_grid.cell_state(sx, sy, sz) == CellState::Surface {
                                has_surface = true;
                            }
                        }
                    }
                }

                if !has_surface {
                    continue;
                }

                let slot = match dst_pool.allocate() {
                    Some(s) => s,
                    None => continue,
                };
                dst_grid.set_cell_state(dx, dy, dz, CellState::Surface);
                dst_grid.set_brick_slot(dx, dy, dz, slot);
                brick_count += 1;

                // Fill destination brick by sampling source
                let dst_brick = dst_pool.get_mut(slot);
                for vz in 0..8u32 {
                    for vy in 0..8u32 {
                        for vx in 0..8u32 {
                            // This dst voxel corresponds to a voxel_ratio x voxel_ratio x voxel_ratio
                            // region of source voxels.
                            let mut dist_sum = 0.0f64;
                            let mut count = 0u32;
                            let mut mat_votes: [u32; 16] = [0; 16];

                            let src_base_vx = dx * brick_ratio * 8 + vx * voxel_ratio;
                            let src_base_vy = dy * brick_ratio * 8 + vy * voxel_ratio;
                            let src_base_vz = dz * brick_ratio * 8 + vz * voxel_ratio;

                            for lz in 0..voxel_ratio {
                                for ly in 0..voxel_ratio {
                                    for lx in 0..voxel_ratio {
                                        let svx = src_base_vx + lx;
                                        let svy = src_base_vy + ly;
                                        let svz = src_base_vz + lz;

                                        // Convert global voxel coords to brick + local
                                        let src_bx = svx / 8;
                                        let src_by = svy / 8;
                                        let src_bz = svz / 8;
                                        let local_x = svx % 8;
                                        let local_y = svy % 8;
                                        let local_z = svz % 8;

                                        if src_bx >= src_dims.x
                                            || src_by >= src_dims.y
                                            || src_bz >= src_dims.z
                                        {
                                            continue;
                                        }

                                        if src_grid.cell_state(src_bx, src_by, src_bz)
                                            != CellState::Surface
                                        {
                                            continue;
                                        }

                                        if let Some(src_slot) =
                                            src_grid.brick_slot(src_bx, src_by, src_bz)
                                        {
                                            let sample =
                                                src_pool.get(src_slot).sample(local_x, local_y, local_z);
                                            let d = sample.distance_f32();
                                            if d.is_finite() {
                                                dist_sum += d as f64;
                                                count += 1;
                                                let mid = sample.material_id() as usize;
                                                if mid < 16 {
                                                    mat_votes[mid] += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if count > 0 {
                                let avg_dist = (dist_sum / count as f64) as f32;
                                let mat_id = mat_votes
                                    .iter()
                                    .enumerate()
                                    .max_by_key(|(_, v)| **v)
                                    .map(|(i, _)| i as u16)
                                    .unwrap_or(0);
                                dst_brick.set(
                                    vx,
                                    vy,
                                    vz,
                                    VoxelSample::new(avg_dist, mat_id, 0, 0, 0),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    LodTier {
        tier: dst_tier,
        grid: dst_grid,
        pool: dst_pool,
        aabb: dst_aabb,
        brick_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::brick_pool::BrickPool;
    use rkf_core::cell_state::CellState;
    use rkf_core::sparse_grid::SparseGrid;
    use rkf_core::voxel::VoxelSample;
    use glam::UVec3;

    /// Helper: create a small source grid+pool with some surface bricks at tier 0.
    fn make_source_data() -> (SparseGrid, BrickPool, Aabb, usize) {
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);
        let mut pool = BrickPool::new(64);
        let tier = 0usize;
        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;

        // Allocate a few surface bricks and fill with known distances
        for z in 0..2u32 {
            for y in 0..2u32 {
                for x in 0..2u32 {
                    let slot = pool.allocate().unwrap();
                    grid.set_cell_state(x, y, z, CellState::Surface);
                    grid.set_brick_slot(x, y, z, slot);

                    let brick = pool.get_mut(slot);
                    for vz in 0..8u32 {
                        for vy in 0..8u32 {
                            for vx in 0..8u32 {
                                // Simple distance: distance from center of the 2x2x2 brick region
                                let dist = 0.1 * (vx as f32 + vy as f32 + vz as f32);
                                brick.set(vx, vy, vz, VoxelSample::new(dist, 1, 0, 0, 0));
                            }
                        }
                    }
                }
            }
        }

        let aabb = Aabb::new(
            glam::Vec3::ZERO,
            dims.as_vec3() * brick_extent,
        );

        (grid, pool, aabb, tier)
    }

    #[test]
    fn generate_lod_tiers_zero_coarser_returns_empty() {
        let (grid, pool, aabb, tier) = make_source_data();
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 0);
        assert!(tiers.is_empty());
    }

    #[test]
    fn generate_lod_tiers_at_max_tier_returns_empty() {
        let (grid, pool, aabb, _) = make_source_data();
        // Source at tier 3 (max) -- no coarser tier exists
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, 3, 2);
        assert!(tiers.is_empty());
    }

    #[test]
    fn generate_lod_tiers_one_coarser_has_correct_tier() {
        let (grid, pool, aabb, tier) = make_source_data();
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 1);
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].tier, tier + 1);
    }

    #[test]
    fn downsample_reduces_grid_dimensions() {
        let (grid, pool, aabb, tier) = make_source_data();
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 1);
        assert_eq!(tiers.len(), 1);

        let src_dims = grid.dimensions();
        let dst_dims = tiers[0].grid.dimensions();

        // With 4x voxel ratio, a 4x4x4 source grid -> 1x1x1 destination grid
        assert!(
            dst_dims.x <= src_dims.x,
            "dst_dims.x={} should be <= src_dims.x={}",
            dst_dims.x,
            src_dims.x
        );
        assert!(
            dst_dims.y <= src_dims.y,
            "dst_dims.y={} should be <= src_dims.y={}",
            dst_dims.y,
            src_dims.y
        );
        assert!(
            dst_dims.z <= src_dims.z,
            "dst_dims.z={} should be <= src_dims.z={}",
            dst_dims.z,
            src_dims.z
        );
    }

    #[test]
    fn downsample_brick_count_is_positive() {
        let (grid, pool, aabb, tier) = make_source_data();
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 1);
        assert!(!tiers.is_empty());
        // We had 8 surface bricks in a 2x2x2 region; with 4x ratio, that maps to 1 dst brick
        assert!(
            tiers[0].brick_count > 0,
            "expected at least 1 brick, got {}",
            tiers[0].brick_count
        );
    }

    #[test]
    fn downsample_preserves_material_majority() {
        // Create a source where all voxels have material_id=3
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);
        let mut pool = BrickPool::new(64);
        let tier = 0usize;
        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;

        let slot = pool.allocate().unwrap();
        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, slot);
        let brick = pool.get_mut(slot);
        for vz in 0..8u32 {
            for vy in 0..8u32 {
                for vx in 0..8u32 {
                    brick.set(vx, vy, vz, VoxelSample::new(0.5, 3, 0, 0, 0));
                }
            }
        }

        let aabb = Aabb::new(glam::Vec3::ZERO, dims.as_vec3() * brick_extent);
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 1);
        assert!(!tiers.is_empty());

        // Check that the downsampled voxels have material_id=3
        if tiers[0].brick_count > 0 {
            let dst_dims = tiers[0].grid.dimensions();
            for z in 0..dst_dims.z {
                for y in 0..dst_dims.y {
                    for x in 0..dst_dims.x {
                        if tiers[0].grid.cell_state(x, y, z) == CellState::Surface {
                            if let Some(dst_slot) = tiers[0].grid.brick_slot(x, y, z) {
                                let sample = tiers[0].pool.get(dst_slot).sample(0, 0, 0);
                                if sample.distance_f32().is_finite() {
                                    assert_eq!(
                                        sample.material_id(),
                                        3,
                                        "expected material_id=3, got {}",
                                        sample.material_id()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn generate_multiple_lod_tiers() {
        let (grid, pool, aabb, tier) = make_source_data();
        // tier 0 -> request 3 coarser -> tiers 1, 2, 3
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 3);
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0].tier, 1);
        assert_eq!(tiers[1].tier, 2);
        assert_eq!(tiers[2].tier, 3);
    }

    #[test]
    fn generate_lod_tiers_caps_at_max() {
        let (grid, pool, aabb, _) = make_source_data();
        // tier 2 -> request 5 coarser -> only tier 3 possible
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, 2, 5);
        assert_eq!(tiers.len(), 1);
        assert_eq!(tiers[0].tier, 3);
    }

    #[test]
    fn lod_tier_aabb_starts_at_source_min() {
        let (grid, pool, aabb, tier) = make_source_data();
        let tiers = generate_lod_tiers(&grid, &pool, &aabb, tier, 1);
        assert!(!tiers.is_empty());
        assert_eq!(tiers[0].aabb.min, aabb.min);
    }
}
