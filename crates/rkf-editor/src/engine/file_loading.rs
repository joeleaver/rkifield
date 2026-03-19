//! File loading helpers for the editor engine.
//!
//! Standalone functions for loading `.rkf` files (v2 and v3), building demo scenes,
//! and computing analytical primitive bounding diameters.

use glam::Vec3;

use rkf_core::{
    Aabb, BrickMapAllocator, BrickPool, Scene, SceneNode, SceneObject,
    SdfPrimitive, SdfSource,
};
use rkf_core::brick_pool::{GeometryPool, SdfCachePool};

use super::GeometryFirstData;

/// Build result containing the scene and CPU brick data for GPU upload.
pub(super) struct DemoScene {
    pub(super) scene: Scene,
    pub(super) brick_pool: BrickPool,
    pub(super) brick_map_alloc: BrickMapAllocator,
}

/// Detect whether a .rkf file uses v2 or v3 format by reading the magic bytes.
fn detect_rkf_version(path: &str) -> Result<u32, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut file, &mut magic).map_err(|e| format!("read magic: {e}"))?;
    match &magic {
        b"RKF2" => Ok(2),
        b"RKF3" => Ok(3),
        _ => Err(format!("unknown .rkf magic: {:?}", magic)),
    }
}

/// Try to load a voxelized object from a .rkf file into the brick pool.
///
/// Returns `(BrickMapHandle, voxel_size, grid_aabb, brick_count)` on success.
/// Supports both v2 and v3 formats. For v3, also populates geometry-first pools.
pub(crate) fn load_rkf_into_pool(
    path: &str,
    pool: &mut rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
    alloc: &mut BrickMapAllocator,
) -> Result<(rkf_core::scene_node::BrickMapHandle, f32, Aabb, u32), String> {
    use rkf_core::asset_file::{load_object_header, load_object_lod};
    use rkf_core::brick_map::EMPTY_SLOT;
    use std::io::BufReader;

    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut reader = BufReader::new(file);

    let header = load_object_header(&mut reader).map_err(|e| format!("header: {e}"))?;
    if header.lod_entries.is_empty() {
        return Err("no LOD levels in .rkf".into());
    }

    // Load the finest LOD (last entry, since they're sorted coarsest-first).
    let finest_idx = header.lod_entries.len() - 1;
    let lod = load_object_lod(&mut reader, &header, finest_idx)
        .map_err(|e| format!("lod: {e}"))?;

    let voxel_size = header.lod_entries[finest_idx].voxel_size;
    let brick_count = lod.brick_data.len() as u32;

    // Allocate pool slots for all bricks.
    let slots = pool.allocate_range(brick_count)
        .ok_or_else(|| format!("pool full: need {brick_count} bricks"))?;

    // Build a new BrickMap with real pool slot indices, and copy brick data.
    let dims = lod.brick_map.dims;
    let mut brick_map = rkf_core::brick_map::BrickMap::new(dims);
    let mut slot_idx = 0usize;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let local_idx = lod.brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if local_idx == EMPTY_SLOT {
                    continue;
                }

                let pool_slot = slots[slot_idx];
                slot_idx += 1;
                brick_map.set(bx, by, bz, pool_slot);

                // Copy voxel data into the pool brick.
                let src = &lod.brick_data[local_idx as usize];
                let dst = pool.get_mut(pool_slot);
                dst.voxels.copy_from_slice(src);
            }
        }
    }

    // Register the brick map in the allocator.
    let handle = alloc.allocate(&brick_map);

    // Compute grid-aligned AABB from dims.
    let brick_world_size = voxel_size * 8.0;
    let grid_half = Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );
    let grid_aabb = Aabb::new(-grid_half, grid_half);

    log::info!(
        "Loaded {path} (v2): {brick_count} bricks, dims={dims:?}, voxel_size={voxel_size}"
    );

    Ok((handle, voxel_size, grid_aabb, brick_count))
}

/// Load a .rkf v3 (geometry-first) file into all three pool tiers.
///
/// Returns `(BrickMapHandle, voxel_size, grid_aabb, brick_count, GeometryFirstData)`.
/// Color data loaded from a v3 file: (brick_pool_slot, ColorBrick) pairs.
pub(crate) type LoadedColorBricks = Vec<(u32, rkf_core::companion::ColorBrick)>;

pub(crate) fn load_rkf_v3_into_pools(
    path: &str,
    pool: &mut rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
    alloc: &mut BrickMapAllocator,
    geo_pool: &mut GeometryPool,
    sdf_pool: &mut SdfCachePool,
) -> Result<(rkf_core::scene_node::BrickMapHandle, f32, Aabb, u32, GeometryFirstData, LoadedColorBricks), String> {
    use rkf_core::asset_file_v3::{load_object_header_v3, load_object_lod_v3};
    use rkf_core::brick::Brick;
    use rkf_core::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};
    use rkf_core::sdf_compute::{compute_sdf_region, SlotMapping};
    use std::io::BufReader;

    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut reader = BufReader::new(file);

    let header = load_object_header_v3(&mut reader).map_err(|e| format!("v3 header: {e}"))?;
    if header.lod_entries.is_empty() {
        return Err("no LOD levels in .rkf v3".into());
    }

    // Load the finest LOD (last entry, sorted coarsest-first).
    let finest_idx = header.lod_entries.len() - 1;
    let lod = load_object_lod_v3(&mut reader, &header, finest_idx)
        .map_err(|e| format!("v3 lod: {e}"))?;

    let voxel_size = header.lod_entries[finest_idx].voxel_size;
    let brick_count = lod.geometry.len() as u32;
    let dims = lod.brick_map.dims;

    // Pre-grow pools if needed.
    if geo_pool.free_count() < brick_count {
        let new_cap = (geo_pool.capacity() * 2).max(geo_pool.capacity() + brick_count);
        geo_pool.grow(new_cap);
    }
    if sdf_pool.free_count() < brick_count {
        let new_cap = (sdf_pool.capacity() * 2).max(sdf_pool.capacity() + brick_count);
        sdf_pool.grow(new_cap);
    }
    if pool.free_count() < brick_count {
        let new_cap = (pool.capacity() * 2).max(pool.capacity() + brick_count);
        pool.grow(new_cap);
    }

    // Allocate pool slots for all bricks.
    let geo_slots: Vec<u32> = (0..brick_count)
        .map(|_| geo_pool.allocate().expect("geo pool alloc"))
        .collect();
    let sdf_slots: Vec<u32> = (0..brick_count)
        .map(|_| sdf_pool.allocate().expect("sdf pool alloc"))
        .collect();
    let brick_slots: Vec<u32> = pool.allocate_range(brick_count)
        .ok_or_else(|| format!("brick pool full: need {brick_count}"))?;

    // Copy geometry data into pools.
    for (i, geo) in lod.geometry.iter().enumerate() {
        *geo_pool.get_mut(geo_slots[i]) = geo.clone();
    }

    // Copy SDF cache if present, otherwise we'll compute it below.
    let has_sdf_cache = lod.sdf_cache.is_some();
    if let Some(ref caches) = lod.sdf_cache {
        for (i, cache) in caches.iter().enumerate() {
            *sdf_pool.get_mut(sdf_slots[i]) = cache.clone();
        }
    }

    // Build geo brick map (entries = geo pool slot indices) and
    // GPU brick map (entries = brick pool slot indices).
    let mut geo_brick_map = BrickMap::new(dims);
    let mut gpu_brick_map = BrickMap::new(dims);
    let mut slot_map = std::collections::HashMap::new();

    // Map local indices (from file) to pool slot indices.
    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let local_idx = lod.brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if local_idx == INTERIOR_SLOT {
                    geo_brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    gpu_brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    continue;
                }
                if local_idx == EMPTY_SLOT {
                    continue;
                }

                let geo_slot = geo_slots[local_idx as usize];
                let sdf_slot = sdf_slots[local_idx as usize];
                let brick_slot = brick_slots[local_idx as usize];

                geo_brick_map.set(bx, by, bz, geo_slot);
                gpu_brick_map.set(bx, by, bz, brick_slot);
                slot_map.insert(geo_slot, (sdf_slot, brick_slot));
            }
        }
    }

    // If no SDF cache was in the file, compute from geometry.
    if !has_sdf_cache && brick_count > 0 {
        let mappings: Vec<SlotMapping> = slot_map.iter()
            .map(|(&geo_slot, &(sdf_slot, _))| SlotMapping {
                brick_slot: geo_slot,
                geometry_slot: geo_slot,
                sdf_slot,
            })
            .collect();

        let region_min = glam::UVec3::ZERO;
        let region_max = dims;
        compute_sdf_region(
            &geo_brick_map,
            geo_pool.as_slice(),
            sdf_pool.as_slice_mut(),
            &mappings,
            region_min,
            region_max,
            voxel_size,
            None,
        );
    }

    // Convert geometry + SDF -> Brick for GPU.
    for (&geo_slot, &(sdf_slot, brick_slot)) in &slot_map {
        let geo = geo_pool.get(geo_slot);
        let cache = sdf_pool.get(sdf_slot);
        let brick = Brick::from_geometry(geo, cache);
        *pool.get_mut(brick_slot) = brick;
    }

    // Register the GPU brick map in the allocator.
    let handle = alloc.allocate(&gpu_brick_map);

    // Compute grid-aligned AABB.
    let brick_world_size = voxel_size * 8.0;
    let grid_half = Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );
    let grid_aabb = Aabb::new(-grid_half, grid_half);

    let gf_data = GeometryFirstData {
        geo_brick_map,
        slot_map: slot_map.clone(),
        voxel_size,
    };

    // Collect color bricks mapped to brick pool slots.
    let loaded_colors: LoadedColorBricks = if let Some(ref colors) = lod.color_bricks {
        let mut out = Vec::new();
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let local_idx = lod.brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                    if local_idx == EMPTY_SLOT || local_idx == INTERIOR_SLOT {
                        continue;
                    }
                    if let Some(color_brick) = colors.get(local_idx as usize) {
                        let brick_slot = brick_slots[local_idx as usize];
                        // Only include if any voxel has non-zero intensity.
                        let has_color = color_brick.data.iter().any(|cv| cv.intensity() > 0);
                        if has_color {
                            out.push((brick_slot, color_brick.clone()));
                        }
                    }
                }
            }
        }
        out
    } else {
        Vec::new()
    };

    log::info!(
        "Loaded {path} (v3): {brick_count} bricks, dims={dims:?}, voxel_size={voxel_size}, sdf_cache={}",
        has_sdf_cache
    );

    Ok((handle, voxel_size, grid_aabb, brick_count, gf_data, loaded_colors))
}

/// Load a .rkf file, auto-detecting v2 or v3 format.
///
/// Returns `(BrickMapHandle, voxel_size, grid_aabb, brick_count, Option<GeometryFirstData>, LoadedColorBricks)`.
/// For v3 files, geometry-first data and color bricks are populated; for v2 files they are empty/None.
pub(crate) fn load_rkf_auto(
    path: &str,
    pool: &mut rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
    alloc: &mut BrickMapAllocator,
    geo_pool: &mut GeometryPool,
    sdf_pool: &mut SdfCachePool,
) -> Result<(rkf_core::scene_node::BrickMapHandle, f32, Aabb, u32, Option<GeometryFirstData>, LoadedColorBricks), String> {
    let version = detect_rkf_version(path)?;
    match version {
        3 => {
            let (handle, vs, aabb, count, gfd, colors) =
                load_rkf_v3_into_pools(path, pool, alloc, geo_pool, sdf_pool)?;
            Ok((handle, vs, aabb, count, Some(gfd), colors))
        }
        _ => {
            let (handle, vs, aabb, count) = load_rkf_into_pool(path, pool, alloc)?;
            Ok((handle, vs, aabb, count, None, Vec::new()))
        }
    }
}

/// Build the demo scene.
///
/// If `scenes/test_cross.rkf` exists, loads it as the primary voxelized object.
/// Otherwise falls back to an inline voxelized sphere.
pub(super) fn build_demo_scene() -> DemoScene {
    use glam::Quat;

    let mut scene = Scene::new("editor_demo");

    // Ground plane — large, flat, stone material.
    let ground = SceneNode::analytical("ground", SdfPrimitive::Box {
        half_extents: Vec3::new(12.0, 0.15, 12.0),
    }, 1);
    let ground_obj = SceneObject {
        id: 0,
        name: "ground".into(),
        parent_id: None,
        position: Vec3::new(0.0, -0.8, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: ground,
        aabb: Aabb::new(Vec3::new(-12.0, -0.15, -12.0), Vec3::new(12.0, 0.15, 12.0)),
    };
    scene.add_object_full(ground_obj);

    let mut brick_pool = BrickPool::new(4096);
    let mut brick_map_alloc = BrickMapAllocator::new();

    // Central hero object — try .rkf, fallback to sphere.
    let rkf_path = "scenes/test_cross.rkf";
    match load_rkf_into_pool(rkf_path, &mut brick_pool, &mut brick_map_alloc) {
        Ok((handle, voxel_size, grid_aabb, _brick_count)) => {
            let mut vox_node = SceneNode::new("vox_cross");
            vox_node.sdf_source = SdfSource::Voxelized {
                brick_map_handle: handle,
                voxel_size,
                aabb: grid_aabb,
            };
            let vox_obj = SceneObject {
                id: 0,
                name: "vox_cross".into(),
                parent_id: None,
                position: Vec3::new(0.0, 0.0, -2.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: vox_node,
                aabb: grid_aabb,
            };
            scene.add_object_full(vox_obj);
        }
        Err(e) => {
            log::warn!("Failed to load {rkf_path}: {e} -- using analytical primitives");

            // Hero sphere — metallic material.
            let radius = 0.5;
            let sphere_node = SceneNode::analytical("hero_sphere", SdfPrimitive::Sphere { radius }, 6);
            let sphere_aabb = Aabb::new(Vec3::splat(-radius), Vec3::splat(radius));
            let sphere_obj = SceneObject {
                id: 0,
                name: "hero_sphere".into(),
                parent_id: None,
                position: Vec3::new(0.0, 0.0, -2.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: sphere_node,
                aabb: sphere_aabb,
            };
            scene.add_object_full(sphere_obj);
        }
    }

    // Pillar — tall box with stone material.
    let pillar_he = Vec3::new(0.2, 1.0, 0.2);
    let pillar_node = SceneNode::analytical("pillar", SdfPrimitive::Box {
        half_extents: pillar_he,
    }, 2);
    scene.add_object_full(SceneObject {
        id: 0,
        name: "pillar_left".into(),
        parent_id: None,
        position: Vec3::new(-3.0, 0.2, -4.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: pillar_node.clone(),
        aabb: Aabb::new(-pillar_he, pillar_he),
    });
    scene.add_object_full(SceneObject {
        id: 0,
        name: "pillar_right".into(),
        parent_id: None,
        position: Vec3::new(3.0, 0.2, -4.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: pillar_node,
        aabb: Aabb::new(-pillar_he, pillar_he),
    });

    // Floating orbs at different heights — different materials.
    let orb_r = 0.25;
    let orb_aabb = Aabb::new(Vec3::splat(-orb_r), Vec3::splat(orb_r));
    let orb_positions = [
        (Vec3::new(-1.5, 0.5, -3.0), 3),  // red-ish
        (Vec3::new(1.5, 0.8, -3.0), 4),   // blue-ish
        (Vec3::new(0.0, 1.2, -4.5), 5),   // green-ish
    ];
    for (pos, mat_id) in orb_positions {
        let node = SceneNode::analytical("orb", SdfPrimitive::Sphere { radius: orb_r }, mat_id);
        scene.add_object_full(SceneObject {
            id: 0,
            name: format!("orb_mat{}", mat_id),
            parent_id: None,
            position: pos,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            root_node: node,
            aabb: orb_aabb,
        });
    }

    // Capsule archway piece.
    let cap_node = SceneNode::analytical("capsule", SdfPrimitive::Capsule {
        radius: 0.15,
        half_height: 0.8,
    }, 7);
    let cap_aabb = Aabb::new(Vec3::new(-0.15, -0.95, -0.15), Vec3::new(0.15, 0.95, 0.15));
    scene.add_object_full(SceneObject {
        id: 0,
        name: "archway".into(),
        parent_id: None,
        position: Vec3::new(0.0, 0.5, -6.0),
        rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        scale: Vec3::new(1.0, 2.0, 1.0),
        root_node: cap_node,
        aabb: cap_aabb,
    });

    DemoScene {
        scene,
        brick_pool,
        brick_map_alloc,
    }
}

/// Compute the diameter of an analytical SDF primitive's bounding sphere.
pub(crate) fn primitive_diameter(prim: &rkf_core::SdfPrimitive) -> f32 {
    let he = primitive_half_extents(prim);
    he.length() * 2.0
}

/// Compute per-axis half-extents of an analytical SDF primitive's tight AABB.
///
/// Returns a `Vec3` where each component is the half-extent along that axis.
/// Used by convert-to-voxel to build a tight (possibly non-cubic) AABB,
/// especially important when non-uniform scale is baked in.
/// Create a new primitive with per-axis scale baked into its dimensions.
///
/// Only primitives where per-axis scaling can be expressed exactly get a
/// scaled primitive (currently only Box). Returns `None` for primitives that
/// can't be exactly scaled (spheres, capsules, etc.) — caller should fall
/// back to the inverse-transform approach.
pub(crate) fn scale_primitive(
    prim: &rkf_core::SdfPrimitive,
    scale: glam::Vec3,
) -> Option<rkf_core::SdfPrimitive> {
    use rkf_core::SdfPrimitive;
    match *prim {
        SdfPrimitive::Box { half_extents } => Some(SdfPrimitive::Box {
            half_extents: half_extents * scale,
        }),
        _ => None,
    }
}

pub(crate) fn primitive_half_extents(prim: &rkf_core::SdfPrimitive) -> glam::Vec3 {
    use rkf_core::SdfPrimitive;
    match *prim {
        SdfPrimitive::Sphere { radius } => glam::Vec3::splat(radius),
        SdfPrimitive::Box { half_extents } => half_extents,
        SdfPrimitive::Capsule { radius, half_height } => {
            // Capsule: cylinder + hemispheres along Y
            glam::Vec3::new(radius, half_height + radius, radius)
        }
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            // Torus in XZ plane
            let r = major_radius + minor_radius;
            glam::Vec3::new(r, minor_radius, r)
        }
        SdfPrimitive::Cylinder { radius, half_height } => {
            glam::Vec3::new(radius, half_height, radius)
        }
        SdfPrimitive::Plane { .. } => glam::Vec3::splat(1.0),
    }
}
