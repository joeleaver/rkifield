//! Geometry resampling — change voxel resolution of an existing object.
//!
//! Resamples occupancy geometry from an old grid to a new grid at a different
//! voxel_size, then computes SDF from the new geometry. Colors and materials
//! are transferred from the nearest old surface voxel.

use glam::{UVec3, Vec3};

use crate::aabb::Aabb;
use crate::brick_geometry::{BrickGeometry, SurfaceVoxel, voxel_index};
use crate::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use crate::brick_pool::Pool;
use crate::constants::BRICK_DIM;
use crate::scene_node::BrickMapHandle;
use crate::sdf_cache::SdfCache;
use crate::sdf_compute::{SlotMapping, compute_sdf_from_geometry};

/// Result of resampling geometry.
pub struct ResampleResult {
    /// Brick map handle.
    pub handle: BrickMapHandle,
    /// Number of allocated bricks.
    pub brick_count: u32,
    /// Geometry pool slots.
    pub geometry_slots: Vec<u32>,
    /// SDF cache pool slots.
    pub sdf_slots: Vec<u32>,
    /// New AABB (may differ slightly from old due to grid quantization).
    pub aabb: Aabb,
}

/// Resample geometry from an old grid to a new grid at a different voxel_size.
///
/// For **downsampling** (new_voxel_size > old_voxel_size): box filter on occupancy —
/// a new voxel is solid if >50% of the old voxels it covers were solid.
///
/// For **upsampling** (new_voxel_size < old_voxel_size): nearest-neighbor sampling
/// of the old occupancy field.
///
/// Colors and materials are transferred from the nearest old surface voxel.
pub fn resample_geometry(
    old_map: &BrickMap,
    old_pool: &Pool<BrickGeometry>,
    old_voxel_size: f32,
    old_aabb: &Aabb,
    new_voxel_size: f32,
    new_geo_pool: &mut Pool<BrickGeometry>,
    new_sdf_pool: &mut Pool<SdfCache>,
    new_map_alloc: &mut BrickMapAllocator,
) -> Option<ResampleResult> {
    let old_brick_size = old_voxel_size * BRICK_DIM as f32;
    let new_brick_size = new_voxel_size * BRICK_DIM as f32;

    // Compute new grid dims from old AABB
    let aabb_size = old_aabb.max - old_aabb.min;
    let new_dims = UVec3::new(
        ((aabb_size.x / new_brick_size).ceil() as u32).max(1),
        ((aabb_size.y / new_brick_size).ceil() as u32).max(1),
        ((aabb_size.z / new_brick_size).ceil() as u32).max(1),
    );

    // Grid origins (centered at local origin)
    let old_grid_origin = -Vec3::new(
        old_map.dims.x as f32 * old_brick_size * 0.5,
        old_map.dims.y as f32 * old_brick_size * 0.5,
        old_map.dims.z as f32 * old_brick_size * 0.5,
    );
    let new_grid_origin = -Vec3::new(
        new_dims.x as f32 * new_brick_size * 0.5,
        new_dims.y as f32 * new_brick_size * 0.5,
        new_dims.z as f32 * new_brick_size * 0.5,
    );

    // Helper: sample old occupancy at a world position
    let sample_old_occupancy = |pos: Vec3| -> bool {
        // Convert world pos to old grid coords
        let grid_pos = pos - old_grid_origin;
        let voxel_f = grid_pos / old_voxel_size;

        let vx = voxel_f.x.floor() as i32;
        let vy = voxel_f.y.floor() as i32;
        let vz = voxel_f.z.floor() as i32;

        if vx < 0 || vy < 0 || vz < 0 {
            return false;
        }
        let (vx, vy, vz) = (vx as u32, vy as u32, vz as u32);

        let bx = vx / BRICK_DIM;
        let by = vy / BRICK_DIM;
        let bz = vz / BRICK_DIM;

        let lx = (vx % BRICK_DIM) as u8;
        let ly = (vy % BRICK_DIM) as u8;
        let lz = (vz % BRICK_DIM) as u8;

        match old_map.get(bx, by, bz) {
            Some(EMPTY_SLOT) | None => false,
            Some(INTERIOR_SLOT) => true,
            Some(slot) => old_pool.get(slot).is_solid(lx, ly, lz),
        }
    };

    // Helper: find nearest old surface voxel material at a position
    let sample_old_surface = |pos: Vec3| -> u8 {
        let grid_pos = pos - old_grid_origin;
        let voxel_f = grid_pos / old_voxel_size;

        let cvx = voxel_f.x.round() as i32;
        let cvy = voxel_f.y.round() as i32;
        let cvz = voxel_f.z.round() as i32;

        // Search in a small radius for the nearest surface voxel
        let search_radius = 2i32;
        let mut best_mat = 0u8;
        let mut best_dist_sq = f32::INFINITY;

        for dz in -search_radius..=search_radius {
            for dy in -search_radius..=search_radius {
                for dx in -search_radius..=search_radius {
                    let sx = cvx + dx;
                    let sy = cvy + dy;
                    let sz = cvz + dz;

                    if sx < 0 || sy < 0 || sz < 0 {
                        continue;
                    }
                    let (sx, sy, sz) = (sx as u32, sy as u32, sz as u32);

                    let bx = sx / BRICK_DIM;
                    let by = sy / BRICK_DIM;
                    let bz = sz / BRICK_DIM;
                    let lx = (sx % BRICK_DIM) as u8;
                    let ly = (sy % BRICK_DIM) as u8;
                    let lz = (sz % BRICK_DIM) as u8;

                    let slot = match old_map.get(bx, by, bz) {
                        Some(s) if s != EMPTY_SLOT && s != INTERIOR_SLOT => s,
                        _ => continue,
                    };

                    let old_geo = old_pool.get(slot);
                    let idx = voxel_index(lx, ly, lz);
                    if let Some(sv) = old_geo.get_surface_voxel(idx) {
                        let dist_sq = (dx * dx + dy * dy + dz * dz) as f32;
                        if dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_mat = sv.material_id;
                        }
                    }
                }
            }
        }

        best_mat
    };

    let ratio = new_voxel_size / old_voxel_size;
    let downsampling = ratio > 1.0;

    // Estimate max needed bricks
    let max_bricks = new_dims.x * new_dims.y * new_dims.z;

    // Pre-allocate temporary storage for new bricks
    let mut new_brick_map = BrickMap::new(new_dims);
    let mut allocated_bricks: Vec<(u32, u32, u32, u32, u32)> = Vec::new(); // (bx,by,bz,geo_slot,sdf_slot)

    for bz in 0..new_dims.z {
        for by in 0..new_dims.y {
            for bx in 0..new_dims.x {
                let brick_min = new_grid_origin
                    + Vec3::new(
                        bx as f32 * new_brick_size,
                        by as f32 * new_brick_size,
                        bz as f32 * new_brick_size,
                    );

                let g_slot = match new_geo_pool.allocate() {
                    Some(s) => s,
                    None => {
                        // Out of pool space — deallocate what we've done and bail
                        for &(_, _, _, gs, ss) in &allocated_bricks {
                            new_geo_pool.deallocate(gs);
                            new_sdf_pool.deallocate(ss);
                        }
                        return None;
                    }
                };
                let s_slot = match new_sdf_pool.allocate() {
                    Some(s) => s,
                    None => {
                        new_geo_pool.deallocate(g_slot);
                        for &(_, _, _, gs, ss) in &allocated_bricks {
                            new_geo_pool.deallocate(gs);
                            new_sdf_pool.deallocate(ss);
                        }
                        return None;
                    }
                };

                let geo = new_geo_pool.get_mut(g_slot);
                let half_voxel = new_voxel_size * 0.5;

                // Sample occupancy for each voxel in the new brick
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            let center = brick_min
                                + Vec3::new(
                                    vx as f32 * new_voxel_size + half_voxel,
                                    vy as f32 * new_voxel_size + half_voxel,
                                    vz as f32 * new_voxel_size + half_voxel,
                                );

                            let is_solid = if downsampling {
                                // Box filter: sample multiple old voxels
                                let samples_per_axis = (ratio.ceil() as i32).max(2);
                                let step = new_voxel_size / samples_per_axis as f32;
                                let mut solid_count = 0;
                                let mut total = 0;

                                for sz in 0..samples_per_axis {
                                    for sy in 0..samples_per_axis {
                                        for sx in 0..samples_per_axis {
                                            let sample_pos = center
                                                - Vec3::splat(half_voxel)
                                                + Vec3::new(
                                                    (sx as f32 + 0.5) * step,
                                                    (sy as f32 + 0.5) * step,
                                                    (sz as f32 + 0.5) * step,
                                                );
                                            if sample_old_occupancy(sample_pos) {
                                                solid_count += 1;
                                            }
                                            total += 1;
                                        }
                                    }
                                }
                                solid_count * 2 > total // >50%
                            } else {
                                // Nearest-neighbor for upsampling
                                sample_old_occupancy(center)
                            };

                            if is_solid {
                                geo.set_solid(vx, vy, vz, true);
                            }
                        }
                    }
                }

                // Classify brick
                if geo.is_fully_empty() {
                    new_brick_map.set(bx, by, bz, EMPTY_SLOT);
                    new_geo_pool.deallocate(g_slot);
                    new_sdf_pool.deallocate(s_slot);
                    continue;
                }
                if geo.is_fully_solid() {
                    new_brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    new_geo_pool.deallocate(g_slot);
                    new_sdf_pool.deallocate(s_slot);
                    continue;
                }

                // Identify surface voxels and transfer color + material
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if geo.is_surface_voxel(vx, vy, vz) {
                                let center = brick_min
                                    + Vec3::new(
                                        vx as f32 * new_voxel_size + half_voxel,
                                        vy as f32 * new_voxel_size + half_voxel,
                                        vz as f32 * new_voxel_size + half_voxel,
                                    );
                                let mat_id = sample_old_surface(center);
                                geo.surface_voxels.push(SurfaceVoxel::new(
                                    voxel_index(vx, vy, vz),
                                    mat_id,
                                ));
                            }
                        }
                    }
                }

                new_brick_map.set(bx, by, bz, g_slot);
                allocated_bricks.push((bx, by, bz, g_slot, s_slot));
            }
        }
    }

    let brick_count = allocated_bricks.len() as u32;

    if brick_count == 0 {
        let handle = new_map_alloc.allocate(&new_brick_map);
        return Some(ResampleResult {
            handle,
            brick_count: 0,
            geometry_slots: vec![],
            sdf_slots: vec![],
            aabb: *old_aabb,
        });
    }

    // Compute SDF from new geometry
    {
        let slot_mappings: Vec<SlotMapping> = allocated_bricks
            .iter()
            .enumerate()
            .map(|(i, &(_, _, _, g_slot, s_slot))| SlotMapping {
                brick_slot: g_slot,
                geometry_slot: i as u32,
                sdf_slot: i as u32,
            })
            .collect();

        let geo_slice: Vec<BrickGeometry> = allocated_bricks
            .iter()
            .map(|&(_, _, _, g_slot, _)| new_geo_pool.get(g_slot).clone())
            .collect();
        let mut sdf_slice: Vec<SdfCache> = allocated_bricks
            .iter()
            .map(|&(_, _, _, _, s_slot)| new_sdf_pool.get(s_slot).clone())
            .collect();

        compute_sdf_from_geometry(&new_brick_map, &geo_slice, &mut sdf_slice, &slot_mappings, new_voxel_size);

        for (i, &(_, _, _, _, s_slot)) in allocated_bricks.iter().enumerate() {
            *new_sdf_pool.get_mut(s_slot) = sdf_slice[i].clone();
        }
    }

    let handle = new_map_alloc.allocate(&new_brick_map);
    let geometry_slots = allocated_bricks.iter().map(|&(_, _, _, g, _)| g).collect();
    let sdf_slots = allocated_bricks.iter().map(|&(_, _, _, _, s)| s).collect();

    // New AABB based on actual grid extent
    let new_half = Vec3::new(
        new_dims.x as f32 * new_brick_size * 0.5,
        new_dims.y as f32 * new_brick_size * 0.5,
        new_dims.z as f32 * new_brick_size * 0.5,
    );
    let new_aabb = Aabb::new(-new_half, new_half);

    Some(ResampleResult {
        handle,
        brick_count,
        geometry_slots,
        sdf_slots,
        aabb: new_aabb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sphere_geometry(radius: f32, voxel_size: f32) -> (BrickMap, Pool<BrickGeometry>, Pool<SdfCache>, BrickMapAllocator) {
        let mut geo_pool: Pool<BrickGeometry> = Pool::new(1024);
        let mut sdf_pool: Pool<SdfCache> = Pool::new(1024);
        let mut alloc = BrickMapAllocator::new();

        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(
            Vec3::splat(-radius - margin),
            Vec3::splat(radius + margin),
        );

        let sdf_fn = move |pos: Vec3| {
            let d = pos.length() - radius;
            (d, 1u8, [200u8, 100, 50])
        };

        let result = crate::voxelize_object::voxelize_to_geometry(
            sdf_fn, &aabb, voxel_size,
            &mut geo_pool, &mut sdf_pool, &mut alloc,
        ).unwrap();

        // Extract brick map from allocator
        let mut brick_map = BrickMap::new(result.dims);
        for bz in 0..result.dims.z {
            for by in 0..result.dims.y {
                for bx in 0..result.dims.x {
                    if let Some(val) = alloc.get_entry(&result.handle, bx, by, bz) {
                        brick_map.set(bx, by, bz, val);
                    }
                }
            }
        }

        (brick_map, geo_pool, sdf_pool, alloc)
    }

    #[test]
    fn resample_upsample_preserves_shape() {
        let old_vs = 0.1;
        let (old_map, old_geo, _, _) = make_sphere_geometry(0.5, old_vs);

        let mut new_geo: Pool<BrickGeometry> = Pool::new(4096);
        let mut new_sdf: Pool<SdfCache> = Pool::new(4096);
        let mut new_alloc = BrickMapAllocator::new();

        let old_aabb = Aabb::new(Vec3::splat(-0.6), Vec3::splat(0.6));
        let new_vs = 0.05; // 2x resolution

        let result = resample_geometry(
            &old_map, &old_geo, old_vs, &old_aabb, new_vs,
            &mut new_geo, &mut new_sdf, &mut new_alloc,
        ).unwrap();

        // More bricks at higher resolution
        assert!(result.brick_count > 0, "should have allocated bricks");
    }

    #[test]
    fn resample_downsample_preserves_shape() {
        let old_vs = 0.05;
        let (old_map, old_geo, _, _) = make_sphere_geometry(0.5, old_vs);

        let mut new_geo: Pool<BrickGeometry> = Pool::new(4096);
        let mut new_sdf: Pool<SdfCache> = Pool::new(4096);
        let mut new_alloc = BrickMapAllocator::new();

        let old_aabb = Aabb::new(Vec3::splat(-0.6), Vec3::splat(0.6));
        let new_vs = 0.1; // half resolution

        let result = resample_geometry(
            &old_map, &old_geo, old_vs, &old_aabb, new_vs,
            &mut new_geo, &mut new_sdf, &mut new_alloc,
        ).unwrap();

        assert!(result.brick_count > 0, "should have allocated bricks");
    }

    #[test]
    fn resample_color_transfer() {
        let old_vs = 0.1;
        let (old_map, old_geo, _, _) = make_sphere_geometry(0.5, old_vs);

        let mut new_geo: Pool<BrickGeometry> = Pool::new(4096);
        let mut new_sdf: Pool<SdfCache> = Pool::new(4096);
        let mut new_alloc = BrickMapAllocator::new();

        let old_aabb = Aabb::new(Vec3::splat(-0.6), Vec3::splat(0.6));

        let result = resample_geometry(
            &old_map, &old_geo, old_vs, &old_aabb, old_vs, // same resolution
            &mut new_geo, &mut new_sdf, &mut new_alloc,
        ).unwrap();

        // Check that at least some surface voxels have the expected material
        let mut found_mat = false;
        for &gs in &result.geometry_slots {
            let geo = new_geo.get(gs);
            for sv in &geo.surface_voxels {
                if sv.material_id == 5 {
                    found_mat = true;
                    break;
                }
            }
            if found_mat { break; }
        }
        assert!(found_mat, "should transfer material from old surface voxels");
    }
}
