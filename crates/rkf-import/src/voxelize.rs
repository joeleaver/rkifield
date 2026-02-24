//! Mesh-to-SDF voxelization — outputs per-object BrickMap for v2 architecture.
//!
//! [`voxelize_mesh`] converts a [`MeshData`] into a compact [`BrickMap`] backed
//! by brick pool allocations. Each voxel stores a signed distance (negative inside,
//! positive outside) computed from the mesh BVH plus a normal-based sign test.
//!
//! [`voxelize_mesh_with_materials`] extends this with per-triangle material transfer
//! via UV barycentric lookup.
//!
//! # Narrow-band optimization
//!
//! Only bricks whose center is within `brick_world_size * 1.8` of the mesh surface
//! are allocated. Interior and exterior air bricks remain as [`EMPTY_SLOT`] entries
//! at zero memory cost.
//!
//! # Sign determination
//!
//! For a query point `p` and the nearest triangle with closest point `c` and
//! interpolated normal `n`, the sign is:
//! - `dot(p - c, n) > 0` → outside → positive distance
//! - `dot(p - c, n) ≤ 0` → inside  → negative distance

use glam::{Vec3, UVec3};

use rkf_core::{
    Aabb, Brick, BrickMap, BrickMapAllocator, BrickMapHandle, Pool, VoxelSample,
    constants::BRICK_DIM,
    brick_map::EMPTY_SLOT,
};

use crate::bvh::TriangleBvh;
use crate::material_transfer::sample_material;
use crate::mesh::MeshData;

// ---------------------------------------------------------------------------
// Public result type
// ---------------------------------------------------------------------------

/// Result of mesh voxelization.
pub struct VoxelizeResult {
    /// Handle to the allocated brick map in the [`BrickMapAllocator`].
    pub handle: BrickMapHandle,
    /// Number of bricks actually allocated (excludes empty-space bricks).
    pub brick_count: u32,
    /// Local-space bounding box of the voxelized region (with margin).
    pub aabb: Aabb,
    /// World-space size of one voxel edge.
    pub voxel_size: f32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Voxelize a mesh into a per-object brick map with a uniform material ID.
///
/// # Arguments
///
/// - `mesh` — source mesh data (positions, normals, indices).
/// - `voxel_size` — world-space size of one voxel edge.
/// - `material_id` — material table index to assign to all surface voxels.
/// - `pool` — brick pool to allocate bricks from.
/// - `map_alloc` — brick map allocator for the packed GPU buffer.
///
/// # Returns
///
/// `Some(VoxelizeResult)` on success, `None` if the pool has insufficient space
/// or the mesh is empty.
pub fn voxelize_mesh(
    mesh: &MeshData,
    voxel_size: f32,
    material_id: u16,
    pool: &mut Pool<Brick>,
    map_alloc: &mut BrickMapAllocator,
) -> Option<VoxelizeResult> {
    if mesh.triangle_count() == 0 {
        return None;
    }

    let bvh = TriangleBvh::build(mesh);
    let normals = precompute_triangle_normals(mesh);
    let aabb = mesh_aabb_with_margin(mesh, voxel_size);

    voxelize_inner(mesh, &bvh, &normals, &aabb, voxel_size, pool, map_alloc, |_pos, _tri_idx| {
        material_id
    })
}

/// Voxelize a mesh with per-triangle material transfer from the mesh's material table.
///
/// Uses [`sample_material`] to select the material ID for each voxel based on
/// the nearest triangle and its barycentric UV interpolation. This allows meshes
/// with multiple materials to produce voxels with different material IDs.
///
/// # Arguments
///
/// - `mesh` — source mesh data (positions, normals, indices, materials).
/// - `voxel_size` — world-space size of one voxel edge.
/// - `pool` — brick pool to allocate bricks from.
/// - `map_alloc` — brick map allocator for the packed GPU buffer.
///
/// # Returns
///
/// `Some(VoxelizeResult)` on success, `None` if the pool has insufficient space
/// or the mesh is empty.
pub fn voxelize_mesh_with_materials(
    mesh: &MeshData,
    voxel_size: f32,
    pool: &mut Pool<Brick>,
    map_alloc: &mut BrickMapAllocator,
) -> Option<VoxelizeResult> {
    if mesh.triangle_count() == 0 {
        return None;
    }

    let bvh = TriangleBvh::build(mesh);
    let normals = precompute_triangle_normals(mesh);
    let aabb = mesh_aabb_with_margin(mesh, voxel_size);

    voxelize_inner(mesh, &bvh, &normals, &aabb, voxel_size, pool, map_alloc, |pos, _tri_idx| {
        sample_material(mesh, &bvh, pos).material_id
    })
}

// ---------------------------------------------------------------------------
// Core voxelization logic
// ---------------------------------------------------------------------------

/// Compute per-triangle face normals from vertex positions.
///
/// For sign determination, a consistent face normal is more robust than
/// interpolating vertex normals (which may point inward on degenerate meshes).
fn precompute_triangle_normals(mesh: &MeshData) -> Vec<Vec3> {
    let tri_count = mesh.triangle_count();
    let mut normals = Vec::with_capacity(tri_count);

    for i in 0..tri_count {
        let [a, b, c] = mesh.triangle_positions(i);
        let edge_ab = b - a;
        let edge_ac = c - a;
        let n = edge_ab.cross(edge_ac);

        // Normalize, or fall back to mesh vertex normal if degenerate.
        let len = n.length();
        if len > 1e-10 {
            normals.push(n / len);
        } else if i < mesh.material_indices.len() {
            // Degenerate triangle — use averaged vertex normals as fallback.
            let base = i * 3;
            let vi0 = mesh.indices[base] as usize;
            let vi1 = mesh.indices[base + 1] as usize;
            let vi2 = mesh.indices[base + 2] as usize;
            let vn = if !mesh.normals.is_empty()
                && vi0 < mesh.normals.len()
                && vi1 < mesh.normals.len()
                && vi2 < mesh.normals.len()
            {
                (mesh.normals[vi0] + mesh.normals[vi1] + mesh.normals[vi2]).normalize_or_zero()
            } else {
                Vec3::Y
            };
            normals.push(vn);
        } else {
            normals.push(Vec3::Y);
        }
    }

    normals
}

/// Compute AABB from mesh bounds with a margin of two voxels on each side.
fn mesh_aabb_with_margin(mesh: &MeshData, voxel_size: f32) -> Aabb {
    let margin = voxel_size * 2.0;
    Aabb::new(
        mesh.bounds_min - Vec3::splat(margin),
        mesh.bounds_max + Vec3::splat(margin),
    )
}

/// Core voxelization loop shared by both public functions.
///
/// The `material_fn(world_pos, triangle_index) -> material_id` closure allows
/// the caller to inject either a uniform material or a per-triangle lookup.
fn voxelize_inner<F>(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    normals: &[Vec3],
    aabb: &Aabb,
    voxel_size: f32,
    pool: &mut Pool<Brick>,
    map_alloc: &mut BrickMapAllocator,
    material_fn: F,
) -> Option<VoxelizeResult>
where
    F: Fn(Vec3, usize) -> u16,
{
    let brick_world_size = voxel_size * BRICK_DIM as f32;
    let narrow_band = brick_world_size * 1.8;

    let aabb_size = aabb.max - aabb.min;
    let dims = UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    // First pass: narrow-band test at brick centers.
    let mut brick_map = BrickMap::new(dims);
    let mut needed_count = 0u32;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = aabb.min
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);

                // Unsigned distance from BVH — fast center probe.
                let nearest = bvh.nearest(brick_center);
                if nearest.distance < narrow_band {
                    // Mark with placeholder (0, which is not EMPTY_SLOT).
                    brick_map.set(bx, by, bz, 0);
                    needed_count += 1;
                }
            }
        }
    }

    // Empty mesh or no bricks near surface.
    if needed_count == 0 {
        let handle = map_alloc.allocate(&brick_map);
        return Some(VoxelizeResult {
            handle,
            brick_count: 0,
            aabb: *aabb,
            voxel_size,
        });
    }

    // Allocate all needed bricks from pool.
    let slots = pool.allocate_range(needed_count)?;
    let mut slot_idx = 0;

    // Second pass: populate each brick.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                if brick_map.get(bx, by, bz) == Some(EMPTY_SLOT) {
                    continue;
                }

                let slot = slots[slot_idx];
                slot_idx += 1;
                brick_map.set(bx, by, bz, slot);

                let brick_min = aabb.min
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );

                populate_brick_mesh(
                    pool.get_mut(slot),
                    bvh,
                    normals,
                    brick_min,
                    voxel_size,
                    &material_fn,
                    mesh,
                );
            }
        }
    }

    debug_assert_eq!(slot_idx, needed_count as usize);

    let handle = map_alloc.allocate(&brick_map);

    Some(VoxelizeResult {
        handle,
        brick_count: needed_count,
        aabb: *aabb,
        voxel_size,
    })
}

/// Populate a single brick by sampling the signed distance at each voxel center.
fn populate_brick_mesh<F>(
    brick: &mut Brick,
    bvh: &TriangleBvh,
    normals: &[Vec3],
    brick_min: Vec3,
    voxel_size: f32,
    material_fn: &F,
    mesh: &MeshData,
) where
    F: Fn(Vec3, usize) -> u16,
{
    let half_voxel = voxel_size * 0.5;

    for vz in 0..BRICK_DIM {
        for vy in 0..BRICK_DIM {
            for vx in 0..BRICK_DIM {
                let pos = brick_min
                    + Vec3::new(
                        vx as f32 * voxel_size + half_voxel,
                        vy as f32 * voxel_size + half_voxel,
                        vz as f32 * voxel_size + half_voxel,
                    );

                let nearest = bvh.nearest(pos);

                // Compute signed distance using normal-dot sign test.
                let tri_normal = if nearest.triangle_index < normals.len() {
                    normals[nearest.triangle_index]
                } else {
                    Vec3::Y
                };
                let to_surface = pos - nearest.closest_point;
                let dot = to_surface.dot(tri_normal);
                let dist = if dot > 0.0 {
                    nearest.distance
                } else {
                    -nearest.distance
                };

                let tri_idx = nearest.triangle_index;
                let mat_idx = if tri_idx < mesh.material_indices.len() {
                    mesh.material_indices[tri_idx] as usize
                } else {
                    0
                };
                let material_id = material_fn(pos, mat_idx);
                let sample = VoxelSample::new(dist, material_id, 0, 0, 0);
                brick.set(vx, vy, vz, sample);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use rkf_core::{BrickMapAllocator, Pool, Brick, brick_map::EMPTY_SLOT};
    use crate::mesh::{ImportMaterial, MeshData};

    // ── Test mesh helpers ────────────────────────────────────────────────────

    /// Single triangle in the XY plane: (0,0,0), (1,0,0), (0,1,0)
    fn single_triangle_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 0.0, 0.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    /// Empty mesh with no geometry.
    fn empty_mesh() -> MeshData {
        MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_indices: Vec::new(),
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    /// Unit cube mesh: 8 vertices, 12 triangles (2 per face × 6 faces).
    fn unit_cube_mesh() -> MeshData {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(1.0, 0.0, 0.0), // 1
            Vec3::new(1.0, 1.0, 0.0), // 2
            Vec3::new(0.0, 1.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 1.0), // 4
            Vec3::new(1.0, 0.0, 1.0), // 5
            Vec3::new(1.0, 1.0, 1.0), // 6
            Vec3::new(0.0, 1.0, 1.0), // 7
        ];

        // 12 triangles (2 per face), winding consistent for outward normals.
        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            // Front face (z=0, normal -Z)
            0, 2, 1,   0, 3, 2,
            // Back face (z=1, normal +Z)
            4, 5, 6,   4, 6, 7,
            // Left face (x=0, normal -X)
            0, 4, 7,   0, 7, 3,
            // Right face (x=1, normal +X)
            1, 2, 6,   1, 6, 5,
            // Bottom face (y=0, normal -Y)
            0, 1, 5,   0, 5, 4,
            // Top face (y=1, normal +Y)
            3, 7, 6,   3, 6, 2,
        ];

        let tri_count = indices.len() / 3;
        let normals = vec![Vec3::Y; 8]; // dummy vertex normals

        MeshData {
            positions,
            normals,
            uvs: Vec::new(),
            indices,
            material_indices: vec![0; tri_count],
            materials: vec![ImportMaterial {
                name: "cube".to_string(),
                base_color: [0.5, 0.5, 0.5],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ONE,
        }
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    /// A single-triangle mesh should produce at least one allocated brick.
    #[test]
    fn voxelize_single_triangle() {
        let mesh = single_triangle_mesh();
        let mut pool: Pool<Brick> = Pool::new(1024);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh(&mesh, 0.1, 1, &mut pool, &mut alloc)
            .expect("should succeed");

        assert!(
            result.brick_count > 0,
            "single triangle should produce at least one brick"
        );
    }

    /// A 12-triangle cube mesh should produce bricks covering the surface.
    #[test]
    fn voxelize_cube_mesh() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh(&mesh, 0.1, 1, &mut pool, &mut alloc)
            .expect("should succeed");

        assert!(
            result.brick_count > 0,
            "cube mesh should produce > 0 bricks, got {}",
            result.brick_count
        );
    }

    /// Bricks far from the surface (empty interior/exterior) should not be allocated.
    ///
    /// Uses a large cube (5m) at a coarse resolution so that inner bricks are
    /// well beyond the narrow-band threshold and get skipped.
    #[test]
    fn narrow_band_skips_empty() {
        // A 5m cube with voxel_size=0.1m → brick_world_size=0.8m, narrow_band=1.44m.
        // The padded AABB spans ~5.4m → ⌈5.4/0.8⌉ = 7 bricks/axis → 343 total.
        // Interior brick centers are ~2.5m from all surfaces, well beyond 1.44m.
        let scale = 5.0_f32;
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(scale, 0.0, 0.0),
            Vec3::new(scale, scale, 0.0),
            Vec3::new(0.0, scale, 0.0),
            Vec3::new(0.0, 0.0, scale),
            Vec3::new(scale, 0.0, scale),
            Vec3::new(scale, scale, scale),
            Vec3::new(0.0, scale, scale),
        ];
        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0, 2, 1,  0, 3, 2,
            4, 5, 6,  4, 6, 7,
            0, 4, 7,  0, 7, 3,
            1, 2, 6,  1, 6, 5,
            0, 1, 5,  0, 5, 4,
            3, 7, 6,  3, 6, 2,
        ];
        let tri_count = indices.len() / 3;
        let mesh = MeshData {
            positions,
            normals: vec![Vec3::Y; 8],
            uvs: Vec::new(),
            indices,
            material_indices: vec![0; tri_count],
            materials: vec![ImportMaterial {
                name: "big_cube".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::splat(scale),
        };

        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh(&mesh, 0.1, 1, &mut pool, &mut alloc)
            .expect("should succeed");

        let total_bricks =
            result.handle.dims.x * result.handle.dims.y * result.handle.dims.z;

        // Narrow-band should allocate only the surface shell (far fewer than total).
        assert!(
            result.brick_count < total_bricks,
            "narrow band should skip interior bricks: {}/{} bricks allocated",
            result.brick_count,
            total_bricks
        );
        // Sanity: at least some surface bricks should be allocated.
        assert!(result.brick_count > 0, "surface bricks must be allocated");
    }

    /// Voxels inside a closed mesh should have negative signed distance.
    #[test]
    fn signed_distance_correct_sign() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh(&mesh, 0.05, 1, &mut pool, &mut alloc)
            .expect("should succeed");

        // Find the center brick (should be inside the cube).
        let handle = &result.handle;
        let cx = handle.dims.x / 2;
        let cy = handle.dims.y / 2;
        let cz = handle.dims.z / 2;

        let slot = alloc.get_entry(handle, cx, cy, cz);
        if let Some(slot) = slot {
            if slot != EMPTY_SLOT {
                let brick = pool.get(slot);
                // Sample the center voxel of the center brick.
                let v = brick.sample(BRICK_DIM / 2, BRICK_DIM / 2, BRICK_DIM / 2);
                // The geometric center of the cube (0.5, 0.5, 0.5) is inside —
                // distance should be negative.
                assert!(
                    v.distance_f32() < 0.0,
                    "center voxel inside the cube should have negative distance, got {}",
                    v.distance_f32()
                );
            }
        }
    }

    /// Voxels should carry the material ID passed to voxelize_mesh.
    #[test]
    fn material_id_assigned() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let expected_mat = 42u16;
        let result = voxelize_mesh(&mesh, 0.1, expected_mat, &mut pool, &mut alloc)
            .expect("should succeed");

        // Check at least one brick's voxels carry the expected material ID.
        let handle = &result.handle;
        let mut found_mat = false;
        'outer: for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = alloc.get_entry(handle, bx, by, bz) {
                        if slot != EMPTY_SLOT {
                            let brick = pool.get(slot);
                            for vz in 0..BRICK_DIM {
                                for vy in 0..BRICK_DIM {
                                    for vx in 0..BRICK_DIM {
                                        if brick.sample(vx, vy, vz).material_id() == expected_mat {
                                            found_mat = true;
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(found_mat, "no voxel with material_id={expected_mat} found");
    }

    /// An empty mesh (no triangles) should return None.
    #[test]
    fn empty_mesh_returns_none() {
        let mesh = empty_mesh();
        let mut pool: Pool<Brick> = Pool::new(1024);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh(&mesh, 0.1, 1, &mut pool, &mut alloc);
        assert!(result.is_none(), "empty mesh should return None");
    }

    /// Finer voxel size should result in more bricks for the same geometry.
    #[test]
    fn voxel_size_affects_resolution() {
        let mesh = unit_cube_mesh();
        let mut pool_coarse: Pool<Brick> = Pool::new(4096);
        let mut alloc_coarse = BrickMapAllocator::new();
        let mut pool_fine: Pool<Brick> = Pool::new(8192);
        let mut alloc_fine = BrickMapAllocator::new();

        let coarse = voxelize_mesh(&mesh, 0.2, 1, &mut pool_coarse, &mut alloc_coarse)
            .expect("coarse should succeed");
        let fine = voxelize_mesh(&mesh, 0.05, 1, &mut pool_fine, &mut alloc_fine)
            .expect("fine should succeed");

        assert!(
            fine.brick_count > coarse.brick_count,
            "finer voxel size should produce more bricks: fine={} coarse={}",
            fine.brick_count,
            coarse.brick_count
        );
    }

    /// `voxelize_mesh_with_materials` should complete without error and produce bricks.
    #[test]
    fn voxelize_with_materials_produces_bricks() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh_with_materials(&mesh, 0.1, &mut pool, &mut alloc)
            .expect("should succeed");

        assert!(result.brick_count > 0);
    }

    /// `voxelize_mesh_with_materials` on an empty mesh should return None.
    #[test]
    fn voxelize_with_materials_empty_mesh_returns_none() {
        let mesh = empty_mesh();
        let mut pool: Pool<Brick> = Pool::new(1024);
        let mut alloc = BrickMapAllocator::new();

        let result = voxelize_mesh_with_materials(&mesh, 0.1, &mut pool, &mut alloc);
        assert!(result.is_none());
    }

    /// The returned AABB should be larger than the original mesh bounds.
    #[test]
    fn aabb_includes_margin() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(4096);
        let mut alloc = BrickMapAllocator::new();

        let voxel_size = 0.1;
        let result = voxelize_mesh(&mesh, voxel_size, 1, &mut pool, &mut alloc)
            .expect("should succeed");

        // AABB should extend beyond the unit cube [0,1]^3 by at least one voxel.
        assert!(result.aabb.min.x < 0.0, "AABB min.x should have margin");
        assert!(result.aabb.max.x > 1.0, "AABB max.x should have margin");
    }

    /// Pool exhaustion should return None gracefully.
    #[test]
    fn pool_exhaustion_returns_none() {
        let mesh = unit_cube_mesh();
        let mut pool: Pool<Brick> = Pool::new(2); // tiny pool
        let mut alloc = BrickMapAllocator::new();

        // Fine voxel size needs many bricks — should fail with tiny pool.
        let result = voxelize_mesh(&mesh, 0.02, 1, &mut pool, &mut alloc);
        assert!(result.is_none(), "should fail when pool is too small");
    }
}
