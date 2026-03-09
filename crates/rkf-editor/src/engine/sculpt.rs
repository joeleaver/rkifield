//! Geometry-first sculpt brush operations for the editor engine.
//!
//! All sculpting goes through the geometry-first pipeline:
//! 1. `apply_geometry_edit()` — modifies occupancy bitmasks
//! 2. `compute_sdf_region()` — recomputes SDF distances from geometry
//! 3. `Brick::from_geometry()` — converts to GPU format
//! 4. Targeted GPU upload of changed bricks

use glam::Vec3;
use rkf_core::brick::Brick;
use rkf_core::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::{Aabb, Scene, SdfSource};
use rkf_core::sdf_compute::{compute_sdf_region, SlotMapping};

use super::EditorEngine;

impl EditorEngine {
    /// Auto-voxelize an analytical object to geometry-first for sculpting.
    ///
    /// If the object's root node is `SdfSource::Analytical`, converts it to
    /// geometry-first voxelized form. Returns the voxel_size if conversion occurred.
    pub fn ensure_object_voxelized(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
    ) -> Option<(f32, Aabb, rkf_core::scene_node::BrickMapHandle)> {
        use super::primitive_diameter;
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id)?;

        let (primitive, material_id) = match &obj.root_node.sdf_source {
            SdfSource::Analytical { primitive, material_id } => (*primitive, *material_id),
            SdfSource::Voxelized { .. } => return None, // Already voxelized.
            SdfSource::None => return None,
        };

        let diameter = primitive_diameter(&primitive);
        let voxel_size = (diameter / 48.0).clamp(0.005, 0.5);

        if let Some((handle, vs, grid_aabb, _count)) =
            self.convert_to_geometry_first(&primitive, material_id as u8, voxel_size, object_id)
        {
            let obj = scene.objects.iter_mut().find(|o| o.id == object_id)?;
            obj.root_node.sdf_source = SdfSource::Voxelized {
                brick_map_handle: handle,
                voxel_size: vs,
                aabb: grid_aabb,
            };
            obj.aabb = grid_aabb;
            self.reupload_brick_data();
            log::info!(
                "Auto-voxelized object {} to geometry-first: vs={:.4}",
                object_id, vs
            );
            Some((vs, grid_aabb, handle))
        } else {
            None
        }
    }

    /// Apply a batch of sculpt edit requests using the geometry-first pipeline.
    ///
    /// Each request modifies occupancy bitmasks, then SDF distances are recomputed
    /// from geometry via Dijkstra propagation. Changed bricks are uploaded to GPU.
    pub fn apply_sculpt_edits(
        &mut self,
        scene: &mut Scene,
        edits: &[crate::sculpt::SculptEditRequest],
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> Vec<u32> {
        use rkf_edit::types::{EditType, FalloffCurve, ShapeType};

        let mut all_modified_slots = Vec::new();

        for req in edits {
            log::info!(
                "sculpt edit: object_id={}, world_pos={:?}, brush_type={:?}, radius={}",
                req.object_id, req.world_position, req.settings.brush_type, req.settings.radius,
            );

            // 1. Auto-voxelize if analytical.
            self.ensure_object_voxelized(scene, req.object_id);

            // 2. Look up the object to get transform + SDF source info.
            let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                Some(o) => o,
                None => { log::warn!("sculpt: object {} not found in scene", req.object_id); continue; },
            };

            let (voxel_size, handle, _aabb_min) = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { voxel_size, brick_map_handle, aabb } => {
                    (*voxel_size, *brick_map_handle, aabb.min)
                }
                _ => { log::warn!("sculpt: object {} not voxelized after ensure", req.object_id); continue; },
            };

            // 3. Ensure geometry-first data exists for this object.
            if !self.geometry_first_data.contains_key(&req.object_id) {
                log::warn!("No geometry-first data for object {} — skipping sculpt", req.object_id);
                continue;
            }
            log::info!("sculpt: voxel_size={}, handle.dims={:?}", voxel_size, handle.dims);

            // 4. Transform world position to object-local space.
            let local_pos = crate::sculpt::world_to_object_local_v2(
                req.world_position,
                obj,
            );

            // 5. Sample dominant material from the surface at the brush hit.
            let sampled_material = self.sample_dominant_material(
                &handle, voxel_size, _aabb_min, local_pos,
            );

            // 6. Convert BrushSettings → EditOp.
            let min_scale = obj.scale.x.min(obj.scale.y.min(obj.scale.z)).max(1e-6);
            let inv_scale = 1.0 / min_scale;

            let edit_type = match req.settings.brush_type {
                crate::sculpt::BrushType::Add => EditType::SmoothUnion,
                crate::sculpt::BrushType::Subtract => EditType::SmoothSubtract,
                crate::sculpt::BrushType::Smooth => EditType::Smooth,
                crate::sculpt::BrushType::Flatten => EditType::Flatten,
                crate::sculpt::BrushType::Sharpen => EditType::Smooth,
            };

            let shape_type = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => ShapeType::Sphere,
                crate::sculpt::BrushShape::Cube => ShapeType::Box,
                crate::sculpt::BrushShape::Cylinder => ShapeType::Cylinder,
            };

            let local_radius = req.settings.radius * inv_scale;
            let local_dims = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => Vec3::new(local_radius, 0.0, 0.0),
                crate::sculpt::BrushShape::Cube => Vec3::splat(local_radius),
                crate::sculpt::BrushShape::Cylinder => Vec3::new(local_radius, local_radius, 0.0),
            };

            let blend_k = local_radius * 0.3;

            let op = rkf_edit::edit_op::EditOp {
                object_id: req.object_id,
                position: local_pos,
                rotation: glam::Quat::IDENTITY,
                edit_type,
                shape_type,
                dimensions: local_dims,
                strength: req.settings.strength,
                blend_k,
                falloff: FalloffCurve::Smooth,
                material_id: sampled_material,
                secondary_id: 0,
                color_packed: 0,
            };

            // 7. Grow brick map if brush extends beyond grid (for add operations).
            let is_add = matches!(edit_type, EditType::SmoothUnion | EditType::CsgUnion);
            if is_add {
                self.grow_brick_map_if_needed(scene, req.object_id, &op);
                // Also grow the geometry brick map to match.
                self.grow_geo_brick_map_if_needed(scene, req.object_id);
            }

            // Re-lookup handle after potential growth.
            let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                Some(o) => o,
                None => continue,
            };
            let handle = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, .. } => *brick_map_handle,
                _ => continue,
            };

            // 8. Undo: snapshot brick pool data before modification.
            {
                let gfd = match self.geometry_first_data.get(&req.object_id) {
                    Some(d) => d,
                    None => continue,
                };
                if let Some(ref mut acc) = undo_acc {
                    for (_, &(_, brick_slot)) in &gfd.slot_map {
                        if acc.captured_slots.insert(brick_slot) {
                            acc.snapshots.push((brick_slot, self.cpu_brick_pool.get(brick_slot).clone()));
                        }
                    }
                }
            }

            // 9. Pre-grow the geometry pool so apply_geometry_edit never
            //    silently drops bricks due to pool exhaustion.
            {
                let brick_size = voxel_size * 8.0;
                let (edit_min, edit_max) = op.local_aabb();
                let grid_origin = -Vec3::new(
                    handle.dims.x as f32 * brick_size * 0.5,
                    handle.dims.y as f32 * brick_size * 0.5,
                    handle.dims.z as f32 * brick_size * 0.5,
                );
                let bmin = ((edit_min - grid_origin) / brick_size).floor().max(Vec3::ZERO);
                let bmax = ((edit_max - grid_origin) / brick_size).ceil();
                let bmax = Vec3::new(
                    bmax.x.min(handle.dims.x as f32),
                    bmax.y.min(handle.dims.y as f32),
                    bmax.z.min(handle.dims.z as f32),
                );
                let max_new = ((bmax.x - bmin.x) * (bmax.y - bmin.y) * (bmax.z - bmin.z)) as u32;
                if self.cpu_geometry_pool.free_count() < max_new {
                    let new_cap = (self.cpu_geometry_pool.capacity() * 2).max(
                        self.cpu_geometry_pool.capacity() + max_new,
                    );
                    self.cpu_geometry_pool.grow(new_cap);
                    log::info!("Grew geometry pool to {} for sculpt edit", new_cap);
                }
            }

            // 10. Apply geometry edit — take geo_brick_map out temporarily
            //     to avoid borrow conflicts with self.
            let mut geo_brick_map = {
                let gfd = self.geometry_first_data.get_mut(&req.object_id).unwrap();
                std::mem::replace(&mut gfd.geo_brick_map, BrickMap::new(glam::UVec3::ONE))
            };

            let result = rkf_edit::geometry_edit::apply_geometry_edit(
                &mut self.cpu_geometry_pool,
                &mut geo_brick_map,
                &mut self.cpu_brick_map_alloc,
                &handle,
                &op,
                voxel_size,
            );

            log::info!(
                "geometry_edit result: modified={}, new={}, removed={}, sdf_region={:?}..{:?}",
                result.modified_bricks.len(), result.new_bricks.len(), result.removed_bricks.len(),
                result.sdf_region_min, result.sdf_region_max,
            );

            // 10. Allocate SDF cache + brick pool slots for new bricks.
            // Collect new bricks' geo_slots first, then allocate pools.
            let new_geo_slots: Vec<(glam::UVec3, u32)> = result.new_bricks.iter()
                .filter_map(|&bc| {
                    let geo_slot = geo_brick_map.get(bc.x, bc.y, bc.z)?;
                    if geo_slot == EMPTY_SLOT || geo_slot == INTERIOR_SLOT { return None; }
                    Some((bc, geo_slot))
                })
                .collect();

            // Pre-grow pools if needed.
            let needed = new_geo_slots.len() as u32;
            if needed > 0 {
                if self.cpu_sdf_cache_pool.free_count() < needed {
                    let new_cap = (self.cpu_sdf_cache_pool.capacity() * 2).max(
                        self.cpu_sdf_cache_pool.capacity() + needed,
                    );
                    self.cpu_sdf_cache_pool.grow(new_cap);
                }
                if self.cpu_brick_pool.free_count() < needed {
                    self.grow_brick_pool_if_needed(needed.max(64));
                }
            }

            // Now get gfd back and perform the allocations.
            {
                let gfd = self.geometry_first_data.get_mut(&req.object_id).unwrap();
                // Put the brick map back.
                gfd.geo_brick_map = geo_brick_map;

                for (bc, geo_slot) in &new_geo_slots {
                    if gfd.slot_map.contains_key(geo_slot) {
                        continue;
                    }
                    let sdf_slot = self.cpu_sdf_cache_pool.allocate().unwrap();
                    let brick_slot = self.cpu_brick_pool.allocate().unwrap();
                    gfd.slot_map.insert(*geo_slot, (sdf_slot, brick_slot));

                    // Update the GPU brick map allocator.
                    self.cpu_brick_map_alloc.set_entry(&handle, bc.x, bc.y, bc.z, brick_slot);

                    // Snapshot new brick for undo.
                    if let Some(ref mut acc) = undo_acc {
                        if acc.captured_slots.insert(brick_slot) {
                            acc.snapshots.push((brick_slot, self.cpu_brick_pool.get(brick_slot).clone()));
                        }
                    }
                }

                // 11. Sync removed bricks to the allocator.
                for &bc in &result.removed_bricks {
                    let entry = gfd.geo_brick_map.get(bc.x, bc.y, bc.z).unwrap_or(EMPTY_SLOT);
                    self.cpu_brick_map_alloc.set_entry(&handle, bc.x, bc.y, bc.z, entry);
                }

                // Clean up stale slot_map entries (geo_slots that were deallocated).
                let live_geo_slots: std::collections::HashSet<u32> = {
                    let dims = gfd.geo_brick_map.dims;
                    let mut live = std::collections::HashSet::new();
                    for bz in 0..dims.z {
                        for by in 0..dims.y {
                            for bx in 0..dims.x {
                                if let Some(s) = gfd.geo_brick_map.get(bx, by, bz) {
                                    if s != EMPTY_SLOT && s != INTERIOR_SLOT {
                                        live.insert(s);
                                    }
                                }
                            }
                        }
                    }
                    live
                };
                let stale: Vec<u32> = gfd.slot_map.keys()
                    .filter(|k| !live_geo_slots.contains(k))
                    .copied()
                    .collect();
                for geo_slot in stale {
                    if let Some((sdf_slot, brick_slot)) = gfd.slot_map.remove(&geo_slot) {
                        self.cpu_sdf_cache_pool.deallocate(sdf_slot);
                        self.cpu_brick_pool.deallocate(brick_slot);
                    }
                }
            }

            if result.modified_bricks.is_empty() && result.new_bricks.is_empty() {
                continue; // No changes.
            }

            // 11. Build SlotMappings and take geo_brick_map out for SDF computation.
            // We need to avoid simultaneous borrows of geometry_first_data (immut)
            // and cpu_sdf_cache_pool (mut), which are both fields of self.
            let (geo_brick_map, mappings) = {
                let gfd = self.geometry_first_data.get_mut(&req.object_id).unwrap();
                let mappings: Vec<SlotMapping> = gfd.slot_map.iter()
                    .map(|(&geo_slot, &(sdf_slot, _))| SlotMapping {
                        brick_slot: geo_slot,
                        geometry_slot: geo_slot,
                        sdf_slot,
                    })
                    .collect();
                let map = std::mem::replace(&mut gfd.geo_brick_map, BrickMap::new(glam::UVec3::ONE));
                (map, mappings)
            };

            // 12. Compute SDF for the affected region.
            // Only write back SDF to actually-modified bricks — preserve existing
            // analytical SDF values on margin/untouched bricks.
            let dirty_set: std::collections::HashSet<glam::UVec3> = result.modified_bricks.iter()
                .chain(result.new_bricks.iter())
                .copied()
                .collect();
            compute_sdf_region(
                &geo_brick_map,
                self.cpu_geometry_pool.as_slice(),
                self.cpu_sdf_cache_pool.as_slice_mut(),
                &mappings,
                result.sdf_region_min,
                result.sdf_region_max,
                voxel_size,
                Some(&dirty_set),
            );

            // Put the brick map back.
            self.geometry_first_data.get_mut(&req.object_id).unwrap().geo_brick_map = geo_brick_map;

            // 13. Convert geometry + SDF → Brick for dirty bricks and upload to GPU.
            let mut modified_brick_slots = Vec::new();

            let gfd = self.geometry_first_data.get(&req.object_id).unwrap();
            for brick_coord in &dirty_set {
                let geo_slot = match gfd.geo_brick_map.get(brick_coord.x, brick_coord.y, brick_coord.z) {
                    Some(s) if s != EMPTY_SLOT && s != INTERIOR_SLOT => s,
                    _ => continue,
                };
                if let Some(&(sdf_slot, brick_slot)) = gfd.slot_map.get(&geo_slot) {
                    let geo = self.cpu_geometry_pool.get(geo_slot);
                    let cache = self.cpu_sdf_cache_pool.get(sdf_slot);
                    let brick = Brick::from_geometry(geo, cache);
                    *self.cpu_brick_pool.get_mut(brick_slot) = brick;
                    modified_brick_slots.push(brick_slot);
                }
            }

            // 14. Targeted GPU upload.
            {
                let brick_byte_size = std::mem::size_of::<Brick>() as u64;
                let gpu_buf_size = self.gpu_scene.brick_pool_buffer().size();

                for &slot in &modified_brick_slots {
                    let offset = slot as u64 * brick_byte_size;
                    if offset + brick_byte_size <= gpu_buf_size {
                        self.ctx.queue.write_buffer(
                            self.gpu_scene.brick_pool_buffer(),
                            offset,
                            bytemuck::bytes_of(self.cpu_brick_pool.get(slot)),
                        );
                    }
                }

                // Brick maps always need re-upload (new allocations may have changed entries).
                let map_data = self.cpu_brick_map_alloc.as_slice();
                if !map_data.is_empty() {
                    self.gpu_scene.upload_brick_maps(
                        &self.ctx.device, &self.ctx.queue, map_data,
                    );
                }
            }

            log::info!(
                "sculpt geometry-first: {} modified bricks, {} new, {} removed",
                result.modified_bricks.len(), result.new_bricks.len(), result.removed_bricks.len(),
            );

            all_modified_slots.extend(&modified_brick_slots);
        }

        all_modified_slots
    }

    /// Grow the geometry brick map to match the GPU brick map's dimensions.
    ///
    /// When `grow_brick_map_if_needed` expands the GPU-side brick map,
    /// the geometry-layer brick map must be expanded to match.
    fn grow_geo_brick_map_if_needed(&mut self, scene: &Scene, object_id: u32) {
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return,
        };
        let handle = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, .. } => *brick_map_handle,
            _ => return,
        };

        let gfd = match self.geometry_first_data.get_mut(&object_id) {
            Some(d) => d,
            None => return,
        };

        let gpu_dims = handle.dims;
        let geo_dims = gfd.geo_brick_map.dims;

        if gpu_dims == geo_dims {
            return; // Already in sync.
        }

        // Grow: create a new geo brick map with the larger dims, copy entries with offset.
        let pad_x = (gpu_dims.x - geo_dims.x) / 2;
        let pad_y = (gpu_dims.y - geo_dims.y) / 2;
        let pad_z = (gpu_dims.z - geo_dims.z) / 2;

        let mut new_map = BrickMap::new(gpu_dims);
        for bz in 0..geo_dims.z {
            for by in 0..geo_dims.y {
                for bx in 0..geo_dims.x {
                    if let Some(slot) = gfd.geo_brick_map.get(bx, by, bz) {
                        if slot != EMPTY_SLOT {
                            new_map.set(bx + pad_x, by + pad_y, bz + pad_z, slot);
                        }
                    }
                }
            }
        }

        gfd.geo_brick_map = new_map;
        log::info!(
            "Grew geo brick map for object {}: {:?} → {:?}",
            object_id, geo_dims, gpu_dims
        );
    }
}
