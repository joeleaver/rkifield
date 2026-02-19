//! Mesh-to-SDF voxelization: unsigned distance via BVH, sign via winding number.
//!
//! Converts a [`MeshData`] triangle mesh into a sparse grid of signed distance
//! field bricks. The pipeline:
//! 1. Build a BVH over mesh triangles
//! 2. Identify narrow-band cells near the mesh surface
//! 3. Compute unsigned distance per voxel via BVH nearest query
//! 4. Determine sign via generalized winding number (Barill et al. 2018)
//! 5. Write signed distance + material into brick pool

use glam::{IVec3, UVec3, Vec3};
use rayon::prelude::*;
use rkf_core::aabb::Aabb;
use rkf_core::brick::Brick;
use rkf_core::brick_pool::BrickPool;
use rkf_core::cell_state::CellState;
use rkf_core::chunk::{Chunk, TierGrid};
use rkf_core::constants::RESOLUTION_TIERS;
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::voxel::VoxelSample;

use crate::bvh::TriangleBvh;
use crate::lod::LodTier;
use crate::mesh::MeshData;

/// Per-brick voxel data collected during parallel voxelization.
type BrickVoxelData = Vec<(u32, Vec<(u32, u32, u32, VoxelSample)>)>;

/// Configuration for mesh voxelization.
#[derive(Debug, Clone)]
pub struct VoxelizeConfig {
    /// Resolution tier index (0-3).
    pub tier: usize,
    /// Narrow band half-width in number of bricks (default: 3).
    pub narrow_band_bricks: u32,
    /// Whether to compute per-voxel color (not yet implemented).
    pub compute_color: bool,
}

impl Default for VoxelizeConfig {
    fn default() -> Self {
        Self {
            tier: 1,
            narrow_band_bricks: 3,
            compute_color: true,
        }
    }
}

/// Result of voxelization.
pub struct VoxelizeResult {
    /// The grid with populated cells.
    pub grid: SparseGrid,
    /// The brick pool with voxel data.
    pub pool: BrickPool,
    /// World-space AABB of the grid.
    pub aabb: Aabb,
    /// Number of bricks used.
    pub brick_count: u32,
}

/// Voxelize a mesh into a sparse grid + brick pool.
///
/// Steps:
/// 1. Compute grid dimensions from mesh bounds + narrow band margin
/// 2. Build BVH
/// 3. Identify cells within narrow band of the mesh surface
/// 4. For each surface cell: compute signed distance per voxel
/// 5. Sign determined via generalized winding number
pub fn voxelize_mesh(mesh: &MeshData, config: &VoxelizeConfig) -> VoxelizeResult {
    let voxel_size = RESOLUTION_TIERS[config.tier].voxel_size;
    let brick_extent = RESOLUTION_TIERS[config.tier].brick_extent;

    // Expand mesh bounds by narrow band margin
    let margin = brick_extent * config.narrow_band_bricks as f32;
    let grid_min = mesh.bounds_min - Vec3::splat(margin);
    let grid_max = mesh.bounds_max + Vec3::splat(margin);
    let grid_extent = grid_max - grid_min;

    // Grid dimensions in bricks
    let dims = UVec3::new(
        (grid_extent.x / brick_extent).ceil().max(1.0) as u32,
        (grid_extent.y / brick_extent).ceil().max(1.0) as u32,
        (grid_extent.z / brick_extent).ceil().max(1.0) as u32,
    );

    let aabb = Aabb::new(grid_min, grid_min + dims.as_vec3() * brick_extent);
    let mut grid = SparseGrid::new(dims);
    let max_bricks = (dims.x * dims.y * dims.z).min(65536);
    let mut pool = BrickPool::new(max_bricks);

    // Build BVH
    let bvh = TriangleBvh::build(mesh);

    // Narrow band distance threshold: any cell whose center is within this
    // distance of the mesh surface gets a brick allocated.
    let narrow_band_dist = brick_extent * config.narrow_band_bricks as f32;

    // Identify cells that need bricks (within narrow band of surface)
    let cells: Vec<(u32, u32, u32)> = {
        let mut cells = Vec::new();
        for cz in 0..dims.z {
            for cy in 0..dims.y {
                for cx in 0..dims.x {
                    let cell_center = grid_min
                        + Vec3::new(
                            (cx as f32 + 0.5) * brick_extent,
                            (cy as f32 + 0.5) * brick_extent,
                            (cz as f32 + 0.5) * brick_extent,
                        );
                    let nearest = bvh.nearest(cell_center);
                    if nearest.distance < narrow_band_dist + brick_extent {
                        cells.push((cx, cy, cz));
                    }
                }
            }
        }
        cells
    };

    // Allocate bricks for surface cells
    let mut brick_assignments: Vec<(u32, u32, u32, u32)> = Vec::new();
    for &(cx, cy, cz) in &cells {
        if let Some(slot) = pool.allocate() {
            grid.set_cell_state(cx, cy, cz, CellState::Surface);
            grid.set_brick_slot(cx, cy, cz, slot);
            brick_assignments.push((cx, cy, cz, slot));
        }
    }

    // Compute signed distances for each brick's voxels in parallel
    let voxel_data: BrickVoxelData = brick_assignments
        .par_iter()
        .map(|&(cx, cy, cz, slot)| {
            let brick_min = grid_min
                + Vec3::new(
                    cx as f32 * brick_extent,
                    cy as f32 * brick_extent,
                    cz as f32 * brick_extent,
                );

            let mut voxels = Vec::with_capacity(512);
            for vz in 0..8u32 {
                for vy in 0..8u32 {
                    for vx in 0..8u32 {
                        let world_pos = brick_min
                            + (Vec3::new(vx as f32, vy as f32, vz as f32) + 0.5)
                                * voxel_size;

                        let nearest = bvh.nearest(world_pos);
                        let unsigned_dist = nearest.distance;

                        // Sign determination via winding number
                        let winding = compute_winding_number(mesh, world_pos);
                        let signed_dist = if winding > 0.5 {
                            -unsigned_dist
                        } else {
                            unsigned_dist
                        };

                        // Material from nearest triangle
                        let mat_id =
                            if nearest.triangle_index < mesh.material_indices.len() {
                                mesh.material_indices[nearest.triangle_index] as u16
                            } else {
                                0
                            };

                        voxels.push((
                            vx,
                            vy,
                            vz,
                            VoxelSample::new(signed_dist, mat_id, 0, 0, 0),
                        ));
                    }
                }
            }
            (slot, voxels)
        })
        .collect();

    // Write voxel data to pool
    for (slot, voxels) in voxel_data {
        let brick = pool.get_mut(slot);
        for (vx, vy, vz, sample) in voxels {
            brick.set(vx, vy, vz, sample);
        }
    }

    VoxelizeResult {
        grid,
        pool,
        aabb,
        brick_count: brick_assignments.len() as u32,
    }
}

/// Automatically select a resolution tier based on mesh statistics.
///
/// Heuristic: `target_voxel_size = average_edge_length / 3`.
/// Snaps to the nearest resolution tier.
pub fn auto_select_tier(mesh: &MeshData) -> usize {
    let avg_edge = mesh.average_edge_length();
    let target = avg_edge / 3.0;

    if target < 0.01 {
        0 // < 1cm -> Tier 0 (0.5cm)
    } else if target < 0.04 {
        1 // < 4cm -> Tier 1 (2cm)
    } else if target < 0.16 {
        2 // < 16cm -> Tier 2 (8cm)
    } else {
        3 // else -> Tier 3 (32cm)
    }
}

// ---------------------------------------------------------------------------
// VoxelizeResult → Chunk conversion
// ---------------------------------------------------------------------------

/// Convert voxelization results into a [`Chunk`] ready for `.rkf` serialization.
///
/// Takes the primary voxelization result (grid + pool at source tier) and
/// optional LOD tiers, packing them into a single [`Chunk`] with local brick indices.
///
/// For each tier (source + LODs):
/// 1. Walk the grid to find all Surface cells with brick slots
/// 2. Extract bricks from the pool using those slots
/// 3. Remap grid slots to 0-based local indices
/// 4. Create [`TierGrid`] with remapped grid and extracted bricks
pub fn to_chunk(result: &VoxelizeResult, lod_tiers: &[LodTier], tier: usize, coords: IVec3) -> Chunk {
    let mut grids = Vec::with_capacity(1 + lod_tiers.len());

    // Source tier
    grids.push(extract_tier_grid(&result.grid, &result.pool, tier));

    // LOD tiers
    for lod in lod_tiers {
        grids.push(extract_tier_grid(&lod.grid, &lod.pool, lod.tier));
    }

    let brick_count = grids.iter().map(|tg| tg.bricks.len() as u32).sum();

    Chunk {
        coords,
        grids,
        brick_count,
    }
}

/// Extract a [`TierGrid`] from a grid + pool, remapping pool-global slots to
/// contiguous 0-based local indices.
fn extract_tier_grid(grid: &SparseGrid, pool: &BrickPool, tier: usize) -> TierGrid {
    let dims = grid.dimensions();
    let mut remapped_grid = grid.clone();
    let mut bricks: Vec<Brick> = Vec::new();

    // Walk every cell. For Surface cells with a brick slot, copy the brick
    // from the pool and assign a new 0-based local index.
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                if grid.cell_state(x, y, z) == CellState::Surface {
                    if let Some(pool_slot) = grid.brick_slot(x, y, z) {
                        let local_index = bricks.len() as u32;
                        bricks.push(*pool.get(pool_slot));
                        remapped_grid.set_brick_slot(x, y, z, local_index);
                    }
                }
            }
        }
    }

    TierGrid {
        tier: tier as u8,
        grid: remapped_grid,
        bricks,
    }
}

// ---------------------------------------------------------------------------
// Winding number computation
// ---------------------------------------------------------------------------

/// Compute the generalized winding number at a point.
///
/// Barill et al. 2018: sum of solid angles subtended by each triangle.
/// Result > 0.5 means the point is inside the mesh.
///
/// This iterates ALL triangles (no BVH acceleration). For large meshes this
/// is O(triangles) per query point. BVH-accelerated winding number is an
/// upgrade path.
pub fn compute_winding_number(mesh: &MeshData, point: Vec3) -> f32 {
    let mut winding = 0.0f64; // Use f64 for numerical stability

    for tri_idx in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle_positions(tri_idx);
        winding += triangle_solid_angle(
            (a - point).as_dvec3(),
            (b - point).as_dvec3(),
            (c - point).as_dvec3(),
        );
    }

    (winding / (4.0 * std::f64::consts::PI)) as f32
}

/// Solid angle subtended by a triangle at the origin.
///
/// Uses the Van Oosterom-Strackee formula:
///   `omega = 2 * atan2(a . (b x c), |a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|)`
pub fn triangle_solid_angle(a: glam::DVec3, b: glam::DVec3, c: glam::DVec3) -> f64 {
    let la = a.length();
    let lb = b.length();
    let lc = c.length();

    // Degenerate: query point is at a vertex
    if la < 1e-10 || lb < 1e-10 || lc < 1e-10 {
        return 0.0;
    }

    let numerator = a.dot(b.cross(c));
    let denominator =
        la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la;

    2.0 * numerator.atan2(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ImportMaterial;

    /// Helper: build a closed tetrahedron mesh for winding number tests.
    fn make_tetrahedron() -> MeshData {
        // Regular-ish tetrahedron with outward-facing normals
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(0.5, 1.0, 0.0);
        let v3 = Vec3::new(0.5, 0.3, 0.8);

        MeshData {
            positions: vec![v0, v1, v2, v3],
            normals: vec![Vec3::Y; 4],
            uvs: Vec::new(),
            // 4 faces, consistent winding (outward-facing)
            indices: vec![
                0, 2, 1, // bottom face (looking from -Y)
                0, 1, 3, // front face
                1, 2, 3, // right face
                2, 0, 3, // left face
            ],
            material_indices: vec![0, 0, 0, 0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.8),
        }
    }

    #[test]
    fn auto_select_tier_fine_mesh() {
        // Average edge ~0.015m -> target ~0.005 -> Tier 0
        let mesh = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(0.015, 0.0, 0.0),
                Vec3::new(0.0, 0.015, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(0.015, 0.015, 0.0),
        };
        assert_eq!(auto_select_tier(&mesh), 0);
    }

    #[test]
    fn auto_select_tier_standard_mesh() {
        // Average edge ~0.06m -> target ~0.02 -> Tier 1
        let mesh = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(0.06, 0.0, 0.0),
                Vec3::new(0.0, 0.06, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(0.06, 0.06, 0.0),
        };
        assert_eq!(auto_select_tier(&mesh), 1);
    }

    #[test]
    fn auto_select_tier_large_mesh() {
        // Average edge ~0.36m -> target ~0.12 -> Tier 2
        let mesh = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(0.36, 0.0, 0.0),
                Vec3::new(0.0, 0.36, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(0.36, 0.36, 0.0),
        };
        assert_eq!(auto_select_tier(&mesh), 2);
    }

    #[test]
    fn auto_select_tier_huge_mesh() {
        // Average edge ~3.0m -> target ~1.0 -> Tier 3
        let mesh = MeshData {
            positions: vec![
                Vec3::ZERO,
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(0.0, 3.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(3.0, 3.0, 0.0),
        };
        assert_eq!(auto_select_tier(&mesh), 3);
    }

    #[test]
    fn winding_number_inside_tetrahedron() {
        let mesh = make_tetrahedron();
        // Centroid of the tetrahedron vertices
        let centroid = Vec3::new(0.5, 0.325, 0.2);
        let winding = compute_winding_number(&mesh, centroid);
        assert!(
            winding > 0.5,
            "winding number inside should be > 0.5, got {winding}"
        );
    }

    #[test]
    fn winding_number_outside_tetrahedron() {
        let mesh = make_tetrahedron();
        // Point far outside the tetrahedron
        let outside = Vec3::new(10.0, 10.0, 10.0);
        let winding = compute_winding_number(&mesh, outside);
        assert!(
            winding < 0.5,
            "winding number outside should be < 0.5, got {winding}"
        );
    }

    #[test]
    fn triangle_solid_angle_degenerate_zero_area() {
        // Degenerate triangle: all three vertices are the same
        let a = glam::DVec3::new(1.0, 0.0, 0.0);
        let b = glam::DVec3::new(1.0, 0.0, 0.0);
        let c = glam::DVec3::new(1.0, 0.0, 0.0);
        let omega = triangle_solid_angle(a, b, c);
        assert!(
            omega.abs() < 1e-8,
            "solid angle of degenerate triangle should be ~0, got {omega}"
        );
    }

    #[test]
    fn triangle_solid_angle_point_at_vertex() {
        // When the query point is at a vertex (zero-length vector), should return 0
        let a = glam::DVec3::ZERO; // query point at this vertex
        let b = glam::DVec3::new(1.0, 0.0, 0.0);
        let c = glam::DVec3::new(0.0, 1.0, 0.0);
        let omega = triangle_solid_angle(a, b, c);
        assert!(
            omega.abs() < 1e-8,
            "solid angle at vertex should be 0, got {omega}"
        );
    }

    #[test]
    fn voxelize_config_defaults() {
        let config = VoxelizeConfig::default();
        assert_eq!(config.tier, 1);
        assert_eq!(config.narrow_band_bricks, 3);
        assert!(config.compute_color);
    }

    #[test]
    fn winding_number_outside_negative_direction() {
        let mesh = make_tetrahedron();
        // Point in the negative direction, clearly outside
        let outside = Vec3::new(-5.0, -5.0, -5.0);
        let winding = compute_winding_number(&mesh, outside);
        assert!(
            winding.abs() < 0.5,
            "winding number outside (negative) should be < 0.5, got {winding}"
        );
    }

    // ------ to_chunk tests ------

    /// Helper: create a VoxelizeResult with a small grid and some surface bricks.
    fn make_voxelize_result(num_bricks: u32) -> VoxelizeResult {
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);
        let max_bricks = 64;
        let mut pool = rkf_core::brick_pool::Pool::new(max_bricks);
        let tier = 1usize;
        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;

        for i in 0..num_bricks {
            let x = i % 4;
            let y = (i / 4) % 4;
            let z = i / 16;
            let slot = pool.allocate().unwrap();
            grid.set_cell_state(x, y, z, CellState::Surface);
            grid.set_brick_slot(x, y, z, slot);

            // Fill brick with recognizable data
            let brick: &mut rkf_core::brick::Brick = pool.get_mut(slot);
            let mat_id = (i + 1) as u16;
            for vz in 0..8u32 {
                for vy in 0..8u32 {
                    for vx in 0..8u32 {
                        let dist = 0.1 * (vx as f32 + vy as f32 + vz as f32);
                        brick.set(vx, vy, vz, VoxelSample::new(dist, mat_id, 0, 0, 0));
                    }
                }
            }
        }

        let aabb = Aabb::new(Vec3::ZERO, dims.as_vec3() * brick_extent);
        VoxelizeResult {
            grid,
            pool,
            aabb,
            brick_count: num_bricks,
        }
    }

    #[test]
    fn to_chunk_empty_result() {
        let result = make_voxelize_result(0);
        let chunk = to_chunk(&result, &[], 1, IVec3::ZERO);

        assert_eq!(chunk.coords, IVec3::ZERO);
        assert_eq!(chunk.grids.len(), 1); // source tier always present
        assert_eq!(chunk.brick_count, 0);
        assert_eq!(chunk.grids[0].bricks.len(), 0);
        assert_eq!(chunk.grids[0].tier, 1);
    }

    #[test]
    fn to_chunk_preserves_brick_data() {
        let result = make_voxelize_result(3);
        let chunk = to_chunk(&result, &[], 1, IVec3::new(5, -2, 0));

        assert_eq!(chunk.coords, IVec3::new(5, -2, 0));
        assert_eq!(chunk.grids.len(), 1);
        assert_eq!(chunk.brick_count, 3);
        assert_eq!(chunk.grids[0].bricks.len(), 3);

        // Verify brick data preserved — each brick has material_id = i+1
        for (i, brick) in chunk.grids[0].bricks.iter().enumerate() {
            let expected_mat = (i + 1) as u16;
            assert_eq!(
                brick.sample(0, 0, 0).material_id(),
                expected_mat,
                "brick {i} material mismatch"
            );
        }

        // Verify roundtrip through .rkf format
        let mut buf = Vec::new();
        rkf_core::chunk::save_chunk(&chunk, &mut buf).unwrap();
        let loaded = rkf_core::chunk::load_chunk(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.brick_count, 3);
        assert_eq!(loaded.grids[0].bricks[0].sample(0, 0, 0).material_id(), 1);
        assert_eq!(loaded.grids[0].bricks[1].sample(0, 0, 0).material_id(), 2);
        assert_eq!(loaded.grids[0].bricks[2].sample(0, 0, 0).material_id(), 3);
    }

    #[test]
    fn to_chunk_with_lod_tiers() {
        let result = make_voxelize_result(8);

        // Generate LOD tiers from tier 0 so downsampling has data to work with
        // Re-create at tier 0 for LOD generation
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);
        let mut pool = rkf_core::brick_pool::Pool::new(64);
        let tier = 0usize;
        let brick_extent = RESOLUTION_TIERS[tier].brick_extent;

        for z in 0..2u32 {
            for y in 0..2u32 {
                for x in 0..2u32 {
                    let slot = pool.allocate().unwrap();
                    grid.set_cell_state(x, y, z, CellState::Surface);
                    grid.set_brick_slot(x, y, z, slot);
                    let brick: &mut rkf_core::brick::Brick = pool.get_mut(slot);
                    for vz in 0..8u32 {
                        for vy in 0..8u32 {
                            for vx in 0..8u32 {
                                brick.set(
                                    vx, vy, vz,
                                    VoxelSample::new(0.5, 2, 0, 0, 0),
                                );
                            }
                        }
                    }
                }
            }
        }

        let aabb = Aabb::new(Vec3::ZERO, dims.as_vec3() * brick_extent);
        let lod_tiers = crate::lod::generate_lod_tiers(&grid, &pool, &aabb, tier, 2);

        let src_result = VoxelizeResult {
            grid,
            pool,
            aabb,
            brick_count: 8,
        };
        let chunk = to_chunk(&src_result, &lod_tiers, tier, IVec3::ZERO);

        // Source tier + LOD tiers
        assert_eq!(chunk.grids.len(), 1 + lod_tiers.len());
        assert_eq!(chunk.grids[0].tier, 0); // source
        for (i, lod) in lod_tiers.iter().enumerate() {
            assert_eq!(chunk.grids[1 + i].tier, lod.tier as u8);
        }
        // Total brick count should be sum across all tiers
        let expected_total: u32 = chunk.grids.iter().map(|tg| tg.bricks.len() as u32).sum();
        assert_eq!(chunk.brick_count, expected_total);
    }

    #[test]
    fn to_chunk_local_indices_contiguous() {
        let result = make_voxelize_result(5);
        let chunk = to_chunk(&result, &[], 1, IVec3::ZERO);

        let tg = &chunk.grids[0];
        assert_eq!(tg.bricks.len(), 5);

        // Collect all brick slots referenced by Surface cells
        let dims = tg.grid.dimensions();
        let mut referenced_slots: Vec<u32> = Vec::new();
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    if tg.grid.cell_state(x, y, z) == CellState::Surface {
                        if let Some(slot) = tg.grid.brick_slot(x, y, z) {
                            referenced_slots.push(slot);
                        }
                    }
                }
            }
        }

        // Slots should be 0-based contiguous: 0, 1, 2, 3, 4
        referenced_slots.sort();
        let expected: Vec<u32> = (0..5).collect();
        assert_eq!(
            referenced_slots, expected,
            "brick slots should be 0-based contiguous, got {:?}",
            referenced_slots
        );
    }

    /// Integration test: full import pipeline (mesh -> voxelize -> chunk -> .rkf roundtrip)
    #[test]
    fn full_pipeline_roundtrip() {
        use crate::lod::generate_lod_tiers;
        use crate::mesh::{ImportMaterial, MeshData};

        // Build a closed cube mesh (6 faces, 12 triangles)
        let positions = vec![
            // Front face
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            // Back face
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
        ];
        let indices = vec![
            // Front
            0, 1, 2, 0, 2, 3, // Back
            5, 4, 7, 5, 7, 6, // Left
            4, 0, 3, 4, 3, 7, // Right
            1, 5, 6, 1, 6, 2, // Top
            3, 2, 6, 3, 6, 7, // Bottom
            4, 5, 1, 4, 1, 0,
        ];

        let mesh = MeshData {
            positions,
            normals: vec![Vec3::Y; 8],
            uvs: Vec::new(),
            indices,
            material_indices: vec![0; 12],
            materials: vec![ImportMaterial {
                name: "cube".to_string(),
                base_color: [1.0, 0.0, 0.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::splat(-0.5),
            bounds_max: Vec3::splat(0.5),
        };

        // Voxelize at tier 0 (finest, 0.5cm voxels)
        let config = VoxelizeConfig {
            tier: 0,
            narrow_band_bricks: 2,
            compute_color: false,
        };
        let result = voxelize_mesh(&mesh, &config);
        assert!(result.brick_count > 0, "voxelization should produce bricks");

        // Generate 1 LOD tier
        let lod_tiers = generate_lod_tiers(&result.grid, &result.pool, &result.aabb, 0, 1);

        // Convert to chunk
        let chunk = to_chunk(&result, &lod_tiers, 0, IVec3::ZERO);
        assert!(chunk.brick_count > 0);
        assert!(chunk.grids.len() >= 1); // at least source tier

        // Save and reload .rkf
        let mut buf = Vec::new();
        rkf_core::chunk::save_chunk(&chunk, &mut buf).unwrap();
        assert!(!buf.is_empty());

        let loaded = rkf_core::chunk::load_chunk(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.coords, IVec3::ZERO);
        assert_eq!(loaded.brick_count, chunk.brick_count);
        assert_eq!(loaded.grids.len(), chunk.grids.len());

        // Verify brick data survived the roundtrip
        for (i, tg) in loaded.grids.iter().enumerate() {
            assert_eq!(tg.bricks.len(), chunk.grids[i].bricks.len());
        }
    }
}
