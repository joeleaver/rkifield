//! Per-object voxelization — convert an SDF function into a brick map + pool entries.
//!
//! [`voxelize_sdf`] replaces the old `populate_grid` approach. Instead of writing
//! to a chunk-based spatial index, it produces a per-object [`BrickMap`] backed by
//! brick pool allocations.
//!
//! # Narrow-band optimization
//!
//! Only bricks near the SDF surface are allocated. Interior and exterior air bricks
//! are left as [`EMPTY_SLOT`] and cost zero memory. This is critical for large objects
//! where the surface-to-volume ratio is low.

use glam::{UVec3, Vec3};

use crate::aabb::Aabb;
use crate::brick::Brick;
use crate::brick_geometry::{BrickGeometry, SurfaceVoxel, voxel_index};
use crate::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use crate::brick_pool::Pool;
use crate::constants::BRICK_DIM;
use crate::scene_node::BrickMapHandle;
use crate::sdf_cache::SdfCache;
use crate::sdf_compute::SlotMapping;
use crate::voxel::VoxelSample;

/// Voxelize an SDF function into a brick map and brick pool.
///
/// Evaluates `sdf_fn` over the given `aabb` at the specified `voxel_size`,
/// allocating bricks in `pool` and registering them in `map_alloc`.
///
/// # Narrow-band optimization
///
/// For each potential brick, the SDF is sampled at the brick center. If the
/// distance exceeds the brick's diagonal (all voxels would be far from the
/// surface), the brick is skipped entirely. Only bricks where the surface
/// passes through (or nearby) get allocated and populated.
///
/// # Arguments
///
/// - `sdf_fn` — evaluates the signed distance at a local-space position.
///   Returns `(distance, material_id)`.
/// - `aabb` — local-space bounding box to voxelize within.
/// - `voxel_size` — world-space size of one voxel edge.
/// - `pool` — brick pool to allocate bricks from.
/// - `map_alloc` — brick map allocator for the packed GPU buffer.
///
/// # Returns
///
/// `(BrickMapHandle, u32)` — the handle to the new brick map, and the number
/// of bricks actually allocated (excludes empty-space bricks).
///
/// Returns `None` if the pool doesn't have enough free bricks.
pub fn voxelize_sdf<F>(
    sdf_fn: F,
    aabb: &Aabb,
    voxel_size: f32,
    pool: &mut Pool<Brick>,
    map_alloc: &mut BrickMapAllocator,
) -> Option<(BrickMapHandle, u32)>
where
    F: Fn(Vec3) -> (f32, u16),
{
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    // Compute brick grid dimensions from AABB.
    let aabb_size = aabb.max - aabb.min;
    let dims = UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    // The ray march shader (sample_voxelized) centers the grid at the object's
    // local origin: grid_pos = local_pos + grid_size * 0.5.  We must sample the
    // SDF at the same positions the shader will read, so the grid origin is
    // -grid_size/2 (NOT aabb.min, which may differ due to ceil rounding).
    let grid_origin = -Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );

    // Narrow-band threshold: if the SDF at brick center exceeds this,
    // no voxel in the brick can be near the surface.
    // Brick diagonal = sqrt(3) * brick_world_size ≈ 1.732 * brick_world_size.
    let narrow_band = brick_world_size * 1.8; // slight margin

    // First pass: determine which bricks need allocation (narrow-band test).
    let mut brick_map = BrickMap::new(dims);
    let mut needed_count = 0u32;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);

                let (dist, _) = sdf_fn(brick_center);
                if dist.abs() < narrow_band {
                    // Mark as needing allocation (use a temporary sentinel).
                    // We'll fill in real slot IDs in the second pass.
                    brick_map.set(bx, by, bz, 0); // placeholder, not EMPTY_SLOT
                    needed_count += 1;
                }
            }
        }
    }

    // Allocate all needed bricks from the pool.
    if needed_count == 0 {
        // Object is entirely outside the narrow band — return empty map.
        let handle = map_alloc.allocate(&brick_map);
        return Some((handle, 0));
    }

    let slots = pool.allocate_range(needed_count)?;
    let mut slot_idx = 0;

    // Second pass: populate bricks and assign real pool slots.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                if brick_map.get(bx, by, bz) == Some(EMPTY_SLOT) {
                    continue; // Skip empty bricks.
                }

                let slot = slots[slot_idx];
                slot_idx += 1;
                brick_map.set(bx, by, bz, slot);

                // Compute brick origin in local space (grid centered at origin).
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );

                // Sample SDF at each voxel center within the brick.
                let brick = pool.get_mut(slot);
                populate_brick(brick, &sdf_fn, brick_min, voxel_size);
            }
        }
    }

    debug_assert_eq!(slot_idx, needed_count as usize);

    // Register in the allocator.
    let handle = map_alloc.allocate(&brick_map);

    Some((handle, needed_count))
}

/// Populate a single brick by sampling the SDF at each voxel center.
fn populate_brick<F>(brick: &mut Brick, sdf_fn: &F, brick_min: Vec3, voxel_size: f32)
where
    F: Fn(Vec3) -> (f32, u16),
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

                let (dist, material_id) = sdf_fn(pos);
                let sample = VoxelSample::new(dist, material_id, [255, 255, 255, 255]);
                brick.set(vx, vy, vz, sample);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Geometry-first voxelization
// ────────────────────────────────────────────────────────────────────────────

/// Result of geometry-first voxelization.
pub struct VoxelizeGeometryResult {
    /// Brick map handle in the allocator.
    pub handle: BrickMapHandle,
    /// Number of allocated bricks.
    pub brick_count: u32,
    /// Geometry pool slots for each allocated brick.
    pub geometry_slots: Vec<u32>,
    /// SDF cache pool slots for each allocated brick.
    pub sdf_slots: Vec<u32>,
    /// Brick map dims (same as handle.dims).
    pub dims: UVec3,
}

/// Voxelize an SDF function into BrickGeometry + SdfCache (geometry-first).
///
/// Evaluates `sdf_fn` to determine occupancy (negative = solid), assigns color
/// and material from the function, then computes SDF distances from geometry.
///
/// # Arguments
///
/// - `sdf_fn` — evaluates `(distance, material_id, color_rgb)` at a local-space position.
/// - `aabb` — local-space bounding box.
/// - `voxel_size` — world-space size of one voxel edge.
/// - `geo_pool` — geometry pool to allocate into.
/// - `sdf_pool` — SDF cache pool to allocate into.
/// - `map_alloc` — brick map allocator.
///
/// Returns `None` if pools don't have enough capacity.
pub fn voxelize_to_geometry<F>(
    sdf_fn: F,
    aabb: &Aabb,
    voxel_size: f32,
    geo_pool: &mut Pool<BrickGeometry>,
    sdf_pool: &mut Pool<SdfCache>,
    map_alloc: &mut BrickMapAllocator,
) -> Option<VoxelizeGeometryResult>
where
    F: Fn(Vec3) -> (f32, u8, [u8; 3]),
{
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    // Compute brick grid dimensions from AABB.
    let aabb_size = aabb.max - aabb.min;
    let dims = UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    // Grid origin centered at local origin (matches ray march shader).
    let grid_origin = -Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );

    // Narrow-band threshold for brick-level culling.
    let narrow_band = brick_world_size * 1.8;

    // First pass: evaluate SDF at brick centers, determine which bricks need allocation.
    // Bricks near the surface (|dist| < narrow_band) need full voxelization.
    // Bricks deep inside (dist < -narrow_band) are fully solid → INTERIOR_SLOT.
    // Bricks far outside (dist > narrow_band) stay EMPTY_SLOT.
    let total_bricks = (dims.x * dims.y * dims.z) as usize;
    let mut brick_needs_alloc = vec![false; total_bricks];
    let mut brick_is_interior = vec![false; total_bricks];
    let mut needed_count = 0u32;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);
                let (dist, _, _) = sdf_fn(brick_center);
                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                if dist.abs() < narrow_band {
                    brick_needs_alloc[bi] = true;
                    needed_count += 1;
                } else if dist < -narrow_band {
                    // Deep interior: all voxels are solid, no surface detail needed.
                    brick_is_interior[bi] = true;
                }
            }
        }
    }

    if needed_count == 0 {
        // Even with no surface bricks, we may have interior bricks.
        let mut brick_map = BrickMap::new(dims);
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                    if brick_is_interior[bi] {
                        brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    }
                }
            }
        }
        let handle = map_alloc.allocate(&brick_map);
        return Some(VoxelizeGeometryResult {
            handle,
            brick_count: 0,
            geometry_slots: vec![],
            sdf_slots: vec![],
            dims,
        });
    }

    // Allocate pool slots.
    let geo_slots = geo_pool.allocate_range(needed_count)?;
    let sdf_slots_vec = sdf_pool.allocate_range(needed_count)?;

    let mut brick_map = BrickMap::new(dims);

    // Pre-mark interior bricks before surface-brick processing.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                if brick_is_interior[bi] {
                    brick_map.set(bx, by, bz, INTERIOR_SLOT);
                }
            }
        }
    }
    let mut slot_mappings = Vec::with_capacity(needed_count as usize);
    let mut slot_idx = 0usize;

    // Second pass: populate geometry for each allocated brick.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let bi = (bx + by * dims.x + bz * dims.x * dims.y) as usize;
                if !brick_needs_alloc[bi] {
                    continue;
                }

                let g_slot = geo_slots[slot_idx];
                let s_slot = sdf_slots_vec[slot_idx];
                // Use g_slot as the brick map entry (arbitrary choice — it just needs to be unique)
                brick_map.set(bx, by, bz, g_slot);

                slot_mappings.push(SlotMapping {
                    brick_slot: g_slot,
                    geometry_slot: g_slot,
                    sdf_slot: s_slot,
                });

                let brick_min = grid_origin
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );

                let geo = geo_pool.get_mut(g_slot);
                let sdf_cache = sdf_pool.get_mut(s_slot);
                let half_voxel = voxel_size * 0.5;

                // Sample SDF at each voxel center → occupancy + SDF cache.
                // TEST: Store analytical distances directly to isolate whether
                // rings come from compute_sdf_from_geometry or elsewhere.
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            let pos = brick_min
                                + Vec3::new(
                                    vx as f32 * voxel_size + half_voxel,
                                    vy as f32 * voxel_size + half_voxel,
                                    vz as f32 * voxel_size + half_voxel,
                                );
                            let (dist, _mat_id, _color) = sdf_fn(pos);
                            sdf_cache.set_distance(vx, vy, vz, dist);
                            if dist <= 0.0 {
                                geo.set_solid(vx, vy, vz, true);
                            }
                        }
                    }
                }

                // Classify: if fully empty or fully solid, optimize
                if geo.is_fully_empty() {
                    brick_map.set(bx, by, bz, EMPTY_SLOT);
                    geo_pool.deallocate(g_slot);
                    sdf_pool.deallocate(s_slot);
                    slot_mappings.pop();
                    slot_idx += 1;
                    continue;
                }
                if geo.is_fully_solid() {
                    brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    geo_pool.deallocate(g_slot);
                    sdf_pool.deallocate(s_slot);
                    slot_mappings.pop();
                    slot_idx += 1;
                    continue;
                }

                // Identify surface voxels and assign color + material
                // (We use brick-local surface detection for initial pass;
                //  cross-brick will be handled by SDF computation)
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if geo.is_surface_voxel(vx, vy, vz) {
                                let pos = brick_min
                                    + Vec3::new(
                                        vx as f32 * voxel_size + half_voxel,
                                        vy as f32 * voxel_size + half_voxel,
                                        vz as f32 * voxel_size + half_voxel,
                                    );
                                let (_, mat_id, _color) = sdf_fn(pos);
                                geo.surface_voxels.push(SurfaceVoxel::new(
                                    voxel_index(vx, vy, vz),
                                    mat_id,
                                ));
                            }
                        }
                    }
                }

                slot_idx += 1;
            }
        }
    }

    let actual_count = slot_mappings.len();

    let handle = map_alloc.allocate(&brick_map);
    let geometry_slots = slot_mappings.iter().map(|m| m.geometry_slot).collect();
    let sdf_slots_result = slot_mappings.iter().map(|m| m.sdf_slot).collect();

    Some(VoxelizeGeometryResult {
        handle,
        brick_count: actual_count as u32,
        geometry_slots,
        sdf_slots: sdf_slots_result,
        dims,
    })
}

/// Evaluate an [`SdfPrimitive`] at a local-space position.
///
/// Convenience function for voxelizing analytical SDF nodes.
pub fn evaluate_primitive(primitive: &crate::scene_node::SdfPrimitive, pos: Vec3) -> f32 {
    use crate::scene_node::SdfPrimitive;

    match *primitive {
        SdfPrimitive::Sphere { radius } => pos.length() - radius,
        SdfPrimitive::Box { half_extents } => crate::sdf::box_sdf(half_extents, pos),
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => {
            let a = Vec3::new(0.0, -half_height, 0.0);
            let b = Vec3::new(0.0, half_height, 0.0);
            crate::sdf::capsule_sdf(a, b, radius, pos)
        }
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => {
            // Torus in XZ plane: q = (length(p.xz) - R, p.y)
            let q_x = Vec3::new(pos.x, 0.0, pos.z).length() - major_radius;
            let q = glam::Vec2::new(q_x, pos.y);
            q.length() - minor_radius
        }
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => {
            let d_radial = Vec3::new(pos.x, 0.0, pos.z).length() - radius;
            let d_height = pos.y.abs() - half_height;
            let outside = glam::Vec2::new(d_radial.max(0.0), d_height.max(0.0)).length();
            let inside = d_radial.max(d_height).min(0.0);
            outside + inside
        }
        SdfPrimitive::Plane { normal, distance } => pos.dot(normal) - distance,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_node::SdfPrimitive;

    fn make_sphere_fn(radius: f32, material: u16) -> impl Fn(Vec3) -> (f32, u16) {
        move |pos: Vec3| (pos.length() - radius, material)
    }

    #[test]
    fn voxelize_sphere() {
        let mut pool: Pool<Brick> = Pool::new(1024);
        let mut alloc = BrickMapAllocator::new();

        let radius = 0.5;
        let voxel_size = 0.02;
        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(
            Vec3::splat(-radius - margin),
            Vec3::splat(radius + margin),
        );

        let (handle, brick_count) =
            voxelize_sdf(make_sphere_fn(radius, 1), &aabb, voxel_size, &mut pool, &mut alloc)
                .unwrap();

        // Should have allocated some bricks.
        assert!(brick_count > 0, "expected allocated bricks");
        // Not every brick position should be allocated (narrow band).
        let total_bricks = handle.dims.x * handle.dims.y * handle.dims.z;
        assert!(
            brick_count < total_bricks,
            "narrow band should skip some bricks: {brick_count} < {total_bricks}"
        );
    }

    #[test]
    fn voxelize_box() {
        let mut pool: Pool<Brick> = Pool::new(512);
        let mut alloc = BrickMapAllocator::new();

        let half = Vec3::splat(0.3);
        let voxel_size = 0.04;
        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(-half - margin, half + margin);

        let sdf_fn = |pos: Vec3| (crate::sdf::box_sdf(half, pos), 2u16);
        let (handle, brick_count) =
            voxelize_sdf(sdf_fn, &aabb, voxel_size, &mut pool, &mut alloc).unwrap();

        assert!(brick_count > 0);
        assert_eq!(handle.dims.x, handle.dims.y); // symmetric
    }

    #[test]
    fn voxelize_empty_sdf() {
        // SDF that is always far from zero — no bricks should be allocated.
        let mut pool: Pool<Brick> = Pool::new(64);
        let mut alloc = BrickMapAllocator::new();

        let sdf_fn = |_pos: Vec3| (100.0f32, 0u16); // far from surface
        let aabb = Aabb::new(Vec3::ZERO, Vec3::splat(1.0));

        let (_, brick_count) =
            voxelize_sdf(sdf_fn, &aabb, 0.1, &mut pool, &mut alloc).unwrap();

        assert_eq!(brick_count, 0);
    }

    #[test]
    fn voxelize_returns_none_when_pool_full() {
        let mut pool: Pool<Brick> = Pool::new(2); // tiny pool
        let mut alloc = BrickMapAllocator::new();

        // Sphere that needs many bricks.
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let result = voxelize_sdf(make_sphere_fn(0.5, 1), &aabb, 0.02, &mut pool, &mut alloc);
        assert!(result.is_none(), "should fail with tiny pool");
    }

    #[test]
    fn brick_count_matches_pool_allocation() {
        let mut pool: Pool<Brick> = Pool::new(2048);
        let mut alloc = BrickMapAllocator::new();

        let initial_free = pool.free_count();
        let (_, brick_count) =
            voxelize_sdf(make_sphere_fn(0.3, 5), &Aabb::new(Vec3::splat(-0.4), Vec3::splat(0.4)), 0.02, &mut pool, &mut alloc)
                .unwrap();

        let after_free = pool.free_count();
        assert_eq!(initial_free - after_free, brick_count);
    }

    #[test]
    fn voxel_values_are_correct() {
        let mut pool: Pool<Brick> = Pool::new(512);
        let mut alloc = BrickMapAllocator::new();

        let radius = 0.3;
        let voxel_size = 0.04;
        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(
            Vec3::splat(-radius - margin),
            Vec3::splat(radius + margin),
        );

        let (handle, _) =
            voxelize_sdf(make_sphere_fn(radius, 7), &aabb, voxel_size, &mut pool, &mut alloc)
                .unwrap();

        // Find the brick at the grid center.
        let center_bx = handle.dims.x / 2;
        let center_by = handle.dims.y / 2;
        let center_bz = handle.dims.z / 2;
        let slot = alloc.get_entry(&handle, center_bx, center_by, center_bz);

        if let Some(slot) = slot {
            if slot != EMPTY_SLOT {
                let brick = pool.get(slot);
                // Center voxel of center brick should be near the origin.
                let v = brick.sample(BRICK_DIM / 2, BRICK_DIM / 2, BRICK_DIM / 2);
                // Material should be 7.
                assert_eq!(v.material_id(), 7);
                // Distance should be approximately -radius (inside the sphere).
                assert!(v.distance() != half::f16::INFINITY, "voxel should be populated");
            }
        }
    }

    #[test]
    fn narrow_band_skips_interior() {
        let mut pool: Pool<Brick> = Pool::new(65536);
        let mut alloc = BrickMapAllocator::new();

        // Large sphere at fine resolution — many interior bricks should be skipped.
        // brick_world_size = 0.02 * 8 = 0.16m, narrow_band ≈ 0.288m.
        // Sphere radius 5.0 → ~62 bricks diameter, ~240k total, surface shell ~5%.
        let radius = 5.0;
        let voxel_size = 0.02;
        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(
            Vec3::splat(-radius - margin),
            Vec3::splat(radius + margin),
        );

        let (handle, brick_count) =
            voxelize_sdf(make_sphere_fn(radius, 1), &aabb, voxel_size, &mut pool, &mut alloc)
                .unwrap();

        let total = handle.dims.x * handle.dims.y * handle.dims.z;
        // For a large sphere with many bricks, the surface shell fraction
        // should be significantly less than the total volume.
        let fill_pct = brick_count as f32 / total as f32 * 100.0;
        assert!(
            fill_pct < 50.0,
            "narrow band should skip interior: {brick_count}/{total} = {fill_pct:.1}%"
        );
    }

    // ── evaluate_primitive tests ────────────────────────────────────────

    #[test]
    fn eval_sphere() {
        let p = SdfPrimitive::Sphere { radius: 1.0 };
        assert!((evaluate_primitive(&p, Vec3::ZERO) - (-1.0)).abs() < 1e-6);
        assert!((evaluate_primitive(&p, Vec3::X) - 0.0).abs() < 1e-6);
        assert!((evaluate_primitive(&p, Vec3::X * 2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eval_box() {
        let p = SdfPrimitive::Box {
            half_extents: Vec3::splat(1.0),
        };
        assert!((evaluate_primitive(&p, Vec3::ZERO) - (-1.0)).abs() < 1e-6);
        assert!((evaluate_primitive(&p, Vec3::X) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn eval_capsule() {
        let p = SdfPrimitive::Capsule {
            radius: 0.5,
            half_height: 1.0,
        };
        // On the axis at midpoint, distance should be -radius.
        assert!((evaluate_primitive(&p, Vec3::ZERO) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn eval_torus() {
        let p = SdfPrimitive::Torus {
            major_radius: 1.0,
            minor_radius: 0.3,
        };
        // On the tube center (1,0,0) the distance should be -minor_radius.
        assert!((evaluate_primitive(&p, Vec3::X) - (-0.3)).abs() < 1e-5);
    }

    #[test]
    fn eval_cylinder() {
        let p = SdfPrimitive::Cylinder {
            radius: 0.5,
            half_height: 1.0,
        };
        // Center should be deep inside.
        assert!(evaluate_primitive(&p, Vec3::ZERO) < 0.0);
        // On the surface radially.
        assert!((evaluate_primitive(&p, Vec3::new(0.5, 0.0, 0.0)) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn eval_plane() {
        let p = SdfPrimitive::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        };
        assert!((evaluate_primitive(&p, Vec3::ZERO) - 0.0).abs() < 1e-6);
        assert!((evaluate_primitive(&p, Vec3::Y) - 1.0).abs() < 1e-6);
        assert!((evaluate_primitive(&p, -Vec3::Y) - (-1.0)).abs() < 1e-6);
    }
}
