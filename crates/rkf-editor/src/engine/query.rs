//! SDF spatial query and inspection functions for the editor engine.

use glam::Vec3;
use rkf_core::SdfSource;
use super::EditorEngine;

impl EditorEngine {
    /// Sample a 2D XZ slice of SDF distances from an object's voxel brick data.
    ///
    /// Walks the object-local XZ extent at the given Y coordinate, sampling
    /// the CPU brick pool at each voxel center. Returns a `VoxelSliceResult`
    /// with the distance grid and per-sample slot status.
    pub fn sample_voxel_slice(
        &self,
        scene: &rkf_core::Scene,
        object_id: u32,
        y_coord: f32,
    ) -> Result<rkf_core::automation::VoxelSliceResult, String> {
        use rkf_core::automation::VoxelSliceResult;
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

        // Find the object in the scene.
        let obj = scene.objects.iter()
            .find(|o| o.id == object_id)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        // Must be voxelized.
        let (handle, voxel_size, _aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (brick_map_handle, *voxel_size, aabb)
            }
            _ => return Err(format!("object {object_id} is not voxelized")),
        };

        let dims = handle.dims;
        let brick_extent = voxel_size * 8.0;
        let grid_size_x = dims.x as f32 * brick_extent;
        let grid_size_z = dims.z as f32 * brick_extent;
        let x_min = -(grid_size_x * 0.5);
        let z_min = -(grid_size_z * 0.5);

        let total_voxels_x = dims.x * 8;
        let total_voxels_z = dims.z * 8;

        let mut distances = Vec::with_capacity((total_voxels_x * total_voxels_z) as usize);
        let mut slot_status = Vec::with_capacity((total_voxels_x * total_voxels_z) as usize);

        // Sample at voxel centers across the XZ extent.
        for vz in 0..total_voxels_z {
            for vx in 0..total_voxels_x {
                let _local_x = x_min + (vx as f32 + 0.5) * voxel_size;
                let _local_z = z_min + (vz as f32 + 0.5) * voxel_size;

                // Determine brick and local voxel coordinates.
                let bx = vx / 8;
                let bz = vz / 8;
                // Compute brick Y from y_coord.
                let grid_y = y_coord + (dims.y as f32 * brick_extent * 0.5);
                let vy_f = grid_y / voxel_size;
                let by = (vy_f / 8.0).floor() as u32;
                let ly = ((vy_f as u32) % 8).min(7);
                let lx = vx % 8;
                let lz = vz % 8;

                if by >= dims.y {
                    distances.push(f32::MAX);
                    slot_status.push(0); // EMPTY
                    continue;
                }

                match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                    Some(slot) if slot == EMPTY_SLOT => {
                        distances.push(voxel_size * 2.0);
                        slot_status.push(0);
                    }
                    Some(slot) if slot == INTERIOR_SLOT => {
                        distances.push(-(voxel_size * 2.0));
                        slot_status.push(1);
                    }
                    Some(slot) => {
                        let sample = self.cpu_brick_pool.get(slot).sample(lx, ly, lz);
                        distances.push(sample.distance_f32());
                        slot_status.push(2);
                    }
                    None => {
                        distances.push(f32::MAX);
                        slot_status.push(0);
                    }
                }
            }
        }

        Ok(VoxelSliceResult {
            origin: [x_min, z_min],
            spacing: voxel_size,
            width: total_voxels_x,
            height: total_voxels_z,
            y_coord,
            distances,
            slot_status,
        })
    }

    /// Sample the SDF distance at a world-space position by iterating scene objects.
    ///
    /// Transforms the query position into each object's local space and evaluates
    /// the SDF. Returns the closest result across all objects.
    pub fn sample_spatial_query(
        &self,
        scene: &rkf_core::Scene,
        world_pos: Vec3,
    ) -> rkf_core::automation::SpatialQueryResult {
        use rkf_core::automation::SpatialQueryResult;

        let mut best_dist = f32::MAX;
        let mut best_mat: u16 = 0;

        for obj in &scene.objects {
            // Transform world_pos to object-local space.
            let inv = glam::Mat4::from_scale_rotation_translation(
                obj.scale, obj.rotation, obj.position,
            ).inverse();
            let local_pos = inv.transform_point3(world_pos);

            let dist = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                    // Simple nearest-voxel lookup from CPU brick pool.
                    let vs = *voxel_size;
                    let dims = brick_map_handle.dims;
                    let grid_size = Vec3::new(
                        dims.x as f32 * vs * 8.0,
                        dims.y as f32 * vs * 8.0,
                        dims.z as f32 * vs * 8.0,
                    );
                    let grid_pos = local_pos + grid_size * 0.5;
                    let vx = (grid_pos.x / vs).floor() as i32;
                    let vy = (grid_pos.y / vs).floor() as i32;
                    let vz = (grid_pos.z / vs).floor() as i32;
                    let total_x = (dims.x * 8) as i32;
                    let total_y = (dims.y * 8) as i32;
                    let total_z = (dims.z * 8) as i32;
                    if vx < 0 || vy < 0 || vz < 0
                        || vx >= total_x || vy >= total_y || vz >= total_z
                    {
                        // Outside grid — treat as far exterior.
                        let half = (aabb.max - aabb.min) * 0.5;
                        (local_pos.abs() - half).max(Vec3::ZERO).length()
                    } else {
                        let bx = (vx / 8) as u32;
                        let by = (vy / 8) as u32;
                        let bz = (vz / 8) as u32;
                        let lx = (vx % 8) as u32;
                        let ly = (vy % 8) as u32;
                        let lz = (vz % 8) as u32;
                        let entry = self.cpu_brick_map_alloc.get_entry(brick_map_handle, bx, by, bz);
                        log::warn!(
                            "[spatial_query] handle={{offset:{}, dims:{:?}}}, brick=({},{},{}), \
                             alloc_buf_len={}, entry={:?}, pool_cap={}",
                            brick_map_handle.offset, brick_map_handle.dims,
                            bx, by, bz,
                            self.cpu_brick_map_alloc.buffer_len(),
                            entry,
                            self.cpu_brick_pool.capacity(),
                        );
                        match entry {
                            Some(slot) if !Self::is_unallocated(slot) => {
                                self.cpu_brick_pool.get(slot).sample(lx, ly, lz).distance_f32()
                            }
                            _ => vs * 2.0,
                        }
                    }
                }
                SdfSource::Analytical { primitive, .. } => {
                    rkf_core::evaluate_primitive(primitive, local_pos)
                }
                SdfSource::None => continue,
            };

            // Apply conservative scale correction.
            let scale_min = obj.scale.min_element();
            let scaled_dist = dist * scale_min;

            if scaled_dist.abs() < best_dist.abs() {
                best_dist = scaled_dist;
                best_mat = match &obj.root_node.sdf_source {
                    SdfSource::Analytical { material_id, .. } => *material_id,
                    _ => 0,
                };
            }
        }

        SpatialQueryResult {
            distance: best_dist,
            material_id: best_mat,
            inside: best_dist < 0.0,
        }
    }

    /// Sample a compact brick-level 3D shape overview of an object.
    ///
    /// Each brick is categorized as empty (`.`), interior (`#`), or
    /// surface/allocated (`+`). Returns per-Y-level ASCII slices.
    pub fn sample_object_shape(
        &self,
        scene: &rkf_core::Scene,
        object_id: u32,
    ) -> Result<rkf_core::automation::ObjectShapeResult, String> {
        use rkf_core::automation::ObjectShapeResult;
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

        let obj = scene.objects.iter()
            .find(|o| o.id == object_id)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        let (handle, voxel_size, aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (brick_map_handle, *voxel_size, aabb)
            }
            _ => return Err(format!("object {object_id} is not voxelized")),
        };

        let dims = handle.dims;
        let brick_extent = voxel_size * 8.0;
        let _half_x = dims.x as f32 * brick_extent * 0.5;
        let _half_y = dims.y as f32 * brick_extent * 0.5;
        let _half_z = dims.z as f32 * brick_extent * 0.5;

        let mut empty_count = 0u32;
        let mut interior_count = 0u32;
        let mut surface_count = 0u32;
        let mut y_slices = Vec::with_capacity(dims.y as usize);

        for by in 0..dims.y {
            let mut slice = String::new();
            for bz in 0..dims.z {
                for bx in 0..dims.x {
                    let ch = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(slot) if slot == EMPTY_SLOT => {
                            empty_count += 1;
                            '.'
                        }
                        Some(slot) if slot == INTERIOR_SLOT => {
                            interior_count += 1;
                            '#'
                        }
                        Some(_) => {
                            surface_count += 1;
                            '+'
                        }
                        None => {
                            empty_count += 1;
                            '.'
                        }
                    };
                    slice.push(ch);
                }
                if bz + 1 < dims.z {
                    slice.push('\n');
                }
            }
            y_slices.push(slice);
        }

        Ok(ObjectShapeResult {
            object_id,
            dims: [dims.x, dims.y, dims.z],
            voxel_size,
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z],
            empty_count,
            interior_count,
            surface_count,
            y_slices,
        })
    }
}
