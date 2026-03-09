//! Brick map and pool operations for the editor engine.

use glam::Vec3;
use rkf_core::{Aabb, Scene, SdfSource};
use super::EditorEngine;

impl EditorEngine {
    /// Grow the brick map if the edit extends beyond the current grid.
    ///
    /// The grid is always centered at the object's local origin. When a sculpt
    /// edit extends past the current grid boundary, this method:
    /// 1. Computes new (larger) grid dimensions that contain the edit
    /// 2. Creates a new BrickMap with expanded dims
    /// 3. Copies existing slot entries (with coordinate offset for centering)
    /// 4. Deallocates old map, allocates new map
    /// 5. Updates SdfSource with new handle/AABB
    /// 6. Re-uploads brick maps to GPU
    ///
    /// Returns true if the map was grown.
    pub(super) fn grow_brick_map_if_needed(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
    ) -> bool {
        use rkf_core::brick_map::BrickMap;

        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return false,
        };

        let (handle, voxel_size) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return false,
        };

        let brick_size = voxel_size * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let grid_origin = -Vec3::new(
            handle.dims.x as f32 * brick_size * 0.5,
            handle.dims.y as f32 * brick_size * 0.5,
            handle.dims.z as f32 * brick_size * 0.5,
        );
        let grid_end = -grid_origin;

        // Check if edit fits within current grid (with padding margin).
        // 3 bricks: 2 for SDF propagation padding + 1 for gradient sampling.
        let margin = brick_size * 3.0;
        if edit_min.x >= grid_origin.x + margin
            && edit_min.y >= grid_origin.y + margin
            && edit_min.z >= grid_origin.z + margin
            && edit_max.x <= grid_end.x - margin
            && edit_max.y <= grid_end.y - margin
            && edit_max.z <= grid_end.z - margin
        {
            return false; // Fits fine, no growth needed.
        }

        // Compute required extent: union of current grid and edit AABB, plus margin.
        // 4 bricks: 2 for SDF propagation + 1 for Catmull-Rom gradient + 1 safety.
        // The brick MAP (flat u32 index array) is cheap; actual GPU cost is only allocated
        // brick pool slots (4KB each), which are proportional to surface area, not grid dims.
        let growth_margin = brick_size * 4.0;
        let required_min = edit_min.min(grid_origin) - Vec3::splat(growth_margin);
        let required_max = edit_max.max(grid_end) + Vec3::splat(growth_margin);

        // New grid must be symmetric about origin (shader assumes centered).
        // The dimension change must be even so that integer pad = (new - old) / 2
        // exactly matches the floating-point centering offset. An odd delta
        // would shift old data by half a brick, scrambling the SDF.
        let half_extent = required_min.abs().max(required_max.abs());
        let raw_dims = glam::UVec3::new(
            ((half_extent.x * 2.0 / brick_size).ceil() as u32).max(handle.dims.x),
            ((half_extent.y * 2.0 / brick_size).ceil() as u32).max(handle.dims.y),
            ((half_extent.z * 2.0 / brick_size).ceil() as u32).max(handle.dims.z),
        );
        // Ensure (new_dims - old_dims) is even on each axis.
        let fix_parity = |new: u32, old: u32| -> u32 {
            if (new.wrapping_sub(old)) % 2 != 0 { new + 1 } else { new }
        };
        let new_dims = glam::UVec3::new(
            fix_parity(raw_dims.x, handle.dims.x),
            fix_parity(raw_dims.y, handle.dims.y),
            fix_parity(raw_dims.z, handle.dims.z),
        );

        if new_dims == handle.dims {
            return false; // Already big enough.
        }

        // Compute offset: old brick (0,0,0) maps to new brick at this offset.
        let pad_x = (new_dims.x - handle.dims.x) / 2;
        let pad_y = (new_dims.y - handle.dims.y) / 2;
        let pad_z = (new_dims.z - handle.dims.z) / 2;

        // Build new BrickMap and copy existing entries.
        // IMPORTANT: copy both allocated bricks AND INTERIOR_SLOT markers.
        // is_unallocated() returns true for INTERIOR_SLOT, but interior markers
        // must be preserved — they tell the GPU that these bricks are deep inside
        // the object (returns -vs*2.0). If lost during growth, they'd become
        // EMPTY_SLOT (returns +vs*2.0), flipping the SDF sign and creating
        // concentric ring artifacts.
        let mut new_map = BrickMap::new(new_dims);
        for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if slot != rkf_core::brick_map::EMPTY_SLOT {
                            new_map.set(bx + pad_x, by + pad_y, bz + pad_z, slot);
                        }
                    }
                }
            }
        }

        // Deallocate old, allocate new.
        self.cpu_brick_map_alloc.deallocate(handle);
        let new_handle = self.cpu_brick_map_alloc.allocate(&new_map);

        // Update AABB to match new grid extent (centered at origin).
        let new_half = Vec3::new(
            new_dims.x as f32 * brick_size * 0.5,
            new_dims.y as f32 * brick_size * 0.5,
            new_dims.z as f32 * brick_size * 0.5,
        );
        let new_aabb = Aabb::new(-new_half, new_half);

        // Update the scene object.
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id).unwrap();
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: new_aabb,
        };
        obj.aabb = new_aabb;

        // Re-upload brick maps to GPU.
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }

        true
    }

    /// Sample the dominant material_id from the CPU brick pool near `local_pos`.
    ///
    /// Checks a small neighborhood of voxels (3x3x3 around the nearest voxel)
    /// and returns the most common non-zero material_id. Falls back to 1 if
    /// no material is found (all voxels are empty or material 0).
    pub(super) fn sample_dominant_material(
        &self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        _aabb_min: Vec3,
        local_pos: Vec3,
    ) -> u16 {
        let vs = voxel_size;
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(
            dims.x as f32 * brick_extent,
            dims.y as f32 * brick_extent,
            dims.z as f32 * brick_extent,
        );

        // Convert local_pos to grid-space (grid is centered at origin).
        let grid_pos = local_pos + grid_size * 0.5;
        let center_voxel = (grid_pos / vs).floor();
        let cx = center_voxel.x as i32;
        let cy = center_voxel.y as i32;
        let cz = center_voxel.z as i32;

        let total_x = (dims.x * 8) as i32;
        let total_y = (dims.y * 8) as i32;
        let total_z = (dims.z * 8) as i32;

        // Sample a 3x3x3 neighborhood and count material occurrences.
        // Only count voxels near the surface (negative SDF, i.e. inside).
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let vx = (cx + dx).clamp(0, total_x - 1);
                    let vy = (cy + dy).clamp(0, total_y - 1);
                    let vz = (cz + dz).clamp(0, total_z - 1);

                    let bx = (vx / 8) as u32;
                    let by = (vy / 8) as u32;
                    let bz = (vz / 8) as u32;
                    let lx = (vx % 8) as u32;
                    let ly = (vy % 8) as u32;
                    let lz = (vz % 8) as u32;

                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let sample = self.cpu_brick_pool.get(slot).sample(lx, ly, lz);
                            let mid = sample.material_id();
                            // Only count non-zero materials from voxels near/on surface.
                            if mid != 0 && sample.distance_f32() < voxel_size {
                                *counts.entry(mid).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Return the most common material, or 1 as fallback.
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(mid, _)| mid)
            .unwrap_or(1)
    }

    /// Compute the tight local-space AABB of allocated bricks for a flat node.
    ///
    /// Used by the render pass to set `geometry_aabb_min/max` on `GpuObject`,
    /// enabling early-out in `sample_voxelized` for the expanded empty region.
    pub(super) fn compute_geometry_aabb_for_flat_node(
        &self,
        flat: &rkf_core::transform_flatten::FlatNode,
    ) -> ([f32; 3], [f32; 3]) {
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
        let (handle, vs) = match &flat.sdf_source {
            rkf_core::SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return ([0.0; 3], [0.0; 3]),
        };
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(dims.x as f32, dims.y as f32, dims.z as f32) * brick_extent;
        let mut gmin = Vec3::splat(f32::MAX);
        let mut gmax = Vec3::splat(f32::MIN);
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) => s,
                        None => continue,
                    };
                    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                        continue;
                    }
                    let brick_min = Vec3::new(bx as f32, by as f32, bz as f32) * brick_extent - grid_size * 0.5;
                    let brick_max = Vec3::new((bx + 1) as f32, (by + 1) as f32, (bz + 1) as f32) * brick_extent - grid_size * 0.5;
                    gmin = gmin.min(brick_min);
                    gmax = gmax.max(brick_max);
                }
            }
        }
        if gmin.x > gmax.x {
            return ([0.0; 3], [0.0; 3]);
        }
        (gmin.to_array(), gmax.to_array())
    }

    /// Grow the CPU brick pool (and matching GPU buffer) if free slots are low.
    ///
    /// Called before sculpt allocation. Doubles capacity whenever fewer
    /// than `min_free` slots remain.
    pub(super) fn grow_brick_pool_if_needed(&mut self, min_free: u32) {
        if self.cpu_brick_pool.free_count() >= min_free {
            return;
        }
        let old_cap = self.cpu_brick_pool.capacity();
        let new_cap = (old_cap * 2).max(old_cap + min_free);
        log::info!(
            "grow_brick_pool_if_needed: {} → {} slots ({} free, needed {})",
            old_cap, new_cap, self.cpu_brick_pool.free_count(), min_free,
        );
        self.cpu_brick_pool.grow(new_cap);

        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: pool_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
        self.gpu_scene.set_brick_pool(&self.ctx.device, new_buf);
    }

    /// Re-upload the entire CPU brick pool and brick maps to the GPU.
    pub(crate) fn reupload_brick_data(&mut self) {
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        let gpu_buf = self.gpu_scene.brick_pool_buffer();

        if pool_data.len() as u64 > gpu_buf.size() {
            let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brick_pool"),
                size: pool_data.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
            self.gpu_scene.set_brick_pool(&self.ctx.device, new_buf);
        } else {
            self.ctx.queue.write_buffer(gpu_buf, 0, pool_data);
        }

        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }
    }
}

