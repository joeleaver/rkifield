//! .rkf asset export functions.

/// Export a voxelized object to .rkf v3 (geometry-first) format.
pub fn export_voxelized_to_rkf_v3(
    path: &std::path::Path,
    gfd: &crate::engine::GeometryFirstData,
    voxel_size: f32,
    aabb: &rkf_core::Aabb,
    geo_pool: &rkf_core::brick_pool::GeometryPool,
    sdf_pool: &rkf_core::brick_pool::SdfCachePool,
) -> anyhow::Result<()> {
    use rkf_core::asset_file_v3::{save_object_v3, SaveLodV3};
    use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
    use std::io::BufWriter;

    let dims = gfd.geo_brick_map.dims;

    // Collect geometry and SDF cache data in brick-map traversal order,
    // building a remapped brick map with local indices.
    let mut geometry = Vec::new();
    let mut sdf_caches = Vec::new();
    let mut material_ids_set = std::collections::HashSet::new();
    let mut remap = std::collections::HashMap::new();
    let mut brick_map = rkf_core::brick_map::BrickMap::new(dims);

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let geo_slot = gfd.geo_brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if geo_slot == EMPTY_SLOT {
                    continue;
                }
                if geo_slot == INTERIOR_SLOT {
                    brick_map.set(bx, by, bz, INTERIOR_SLOT);
                    continue;
                }

                let local_idx = if let Some(&idx) = remap.get(&geo_slot) {
                    idx
                } else {
                    let idx = geometry.len() as u32;
                    remap.insert(geo_slot, idx);

                    let geo = geo_pool.get(geo_slot);
                    geometry.push(geo.clone());

                    // Collect material IDs from surface voxels.
                    for sv in &geo.surface_voxels {
                        material_ids_set.insert(sv.material_id);
                        if sv.secondary_material_id != 0 {
                            material_ids_set.insert(sv.secondary_material_id);
                        }
                    }

                    // Get SDF cache if available.
                    if let Some(&(sdf_slot, _)) = gfd.slot_map.get(&geo_slot) {
                        sdf_caches.push(sdf_pool.get(sdf_slot).clone());
                    }

                    idx
                };
                brick_map.set(bx, by, bz, local_idx);
            }
        }
    }

    let sdf_cache = if sdf_caches.len() == geometry.len() {
        Some(sdf_caches)
    } else {
        None
    };

    let mut material_ids: Vec<u8> = material_ids_set.into_iter().collect();
    material_ids.sort();

    let lod = SaveLodV3 {
        voxel_size,
        brick_map,
        geometry,
        sdf_cache,
        color_bricks: None, // TODO: save per-voxel color from editor
    };

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_object_v3(&mut writer, aabb, None, &material_ids, &[lod])?;

    let save_brick_count = remap.len();
    log::info!("Saved v3 .rkf: {} ({} bricks)", path.display(), save_brick_count);

    Ok(())
}
