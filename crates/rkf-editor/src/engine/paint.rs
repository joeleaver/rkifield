//! Geometry-first paint operations for the editor engine.
//!
//! Paint modifies surface voxel properties (material_id, color) without
//! changing occupancy or SDF distances. The pipeline:
//! 1. Geodesic flood fill from brush hit point along the surface
//! 2. Apply paint to visited voxels with distance-based falloff
//! 3. `Brick::from_geometry()` → GPU format, targeted upload

use glam::Vec3;
use rkf_core::brick::Brick;
use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::surface_flood::{self, FloodFillParams};
use rkf_core::{Scene, SdfSource};

use super::EditorEngine;

impl EditorEngine {
    /// Apply a batch of paint edit requests using geodesic surface flood fill.
    ///
    /// The brush follows the object's surface contour: a Dijkstra flood fill
    /// expands from the hit point along face-adjacent surface voxels,
    /// accumulating geodesic distance. Paint weight falls off with geodesic
    /// distance, matching the wireframe brush preview.
    pub fn apply_paint_edits(
        &mut self,
        scene: &mut Scene,
        edits: &[crate::paint::PaintEditRequest],
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> Vec<u32> {
        let mut all_modified_slots = Vec::new();

        for req in edits {
            // 1. Look up the object — must already be voxelized.
            let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                Some(o) => o,
                None => continue,
            };

            let voxel_size = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { voxel_size, .. } => *voxel_size,
                _ => continue,
            };

            // 2. Check geometry-first data exists.
            if !self.geometry_first_data.contains_key(&req.object_id) {
                continue;
            }

            // 3. Transform brush hit to object-local space.
            let local_pos = crate::paint::world_to_object_local_v2(req.world_position, obj);

            // 4. Flood fill parameters.
            // Brush radius is in world units. Step cost accounts for scale so
            // geodesic distance is measured in world units, matching the brush.
            let step_cost = Vec3::new(
                voxel_size * obj.scale.x,
                voxel_size * obj.scale.y,
                voxel_size * obj.scale.z,
            );
            let params = FloodFillParams {
                max_distance: req.settings.radius,
                step_cost,
            };

            let is_material_paint = matches!(req.settings.mode, crate::paint::PaintMode::Material);
            let is_erase = matches!(req.settings.mode, crate::paint::PaintMode::Erase);
            let falloff_frac = req.settings.falloff.clamp(0.0, 1.0);

            let color_bytes = if is_erase {
                [0u8, 0, 0] // erase clears intensity, color doesn't matter
            } else if !is_material_paint {
                let r = (req.settings.color.x * 255.0).clamp(0.0, 255.0) as u8;
                let g = (req.settings.color.y * 255.0).clamp(0.0, 255.0) as u8;
                let b = (req.settings.color.z * 255.0).clamp(0.0, 255.0) as u8;
                [r, g, b]
            } else {
                [0, 0, 0]
            };

            // 5. Undo: snapshot geometry-first data before modification.
            {
                let gfd = match self.geometry_first_data.get(&req.object_id) {
                    Some(d) => d,
                    None => continue,
                };
                if let Some(ref mut acc) = undo_acc {
                    for (&geo_slot, &(sdf_slot, brick_slot)) in &gfd.slot_map {
                        if acc.captured_slots.insert(geo_slot) {
                            acc.snapshots.push(crate::editor_state::GeometryUndoEntry {
                                geo_slot,
                                geometry: self.cpu_geometry_pool.get(geo_slot).clone(),
                                sdf_cache: self.cpu_sdf_cache_pool.get(sdf_slot).clone(),
                                brick_slot,
                            });
                        }
                    }
                }
            }

            // 6. Run geodesic flood fill on the geometry brick map.
            let geo_brick_map = &self.geometry_first_data.get(&req.object_id).unwrap().geo_brick_map;
            let flood_results = surface_flood::surface_flood_fill(
                local_pos,
                geo_brick_map,
                &self.cpu_geometry_pool,
                voxel_size,
                &params,
            );

            if flood_results.is_empty() {
                continue;
            }

            // 7. Apply paint to each visited voxel.
            let brush_radius = req.settings.radius;
            let mut modified_brick_set = std::collections::HashSet::new();

            for entry in &flood_results {
                let geo = self.cpu_geometry_pool.get_mut(entry.geo_slot);
                let sv = &mut geo.surface_voxels[entry.surface_index];

                let normalized = if brush_radius > 0.0 {
                    entry.geodesic_distance / brush_radius
                } else {
                    0.0
                };

                if is_material_paint {
                    // Compute paint weight with smoothstep falloff.
                    let w = if falloff_frac < 0.001 {
                        1.0_f32
                    } else {
                        let edge_start = 1.0 - falloff_frac;
                        if normalized <= edge_start {
                            1.0_f32
                        } else {
                            let t = ((normalized - edge_start) / falloff_frac).clamp(0.0, 1.0);
                            let smooth_t = t * t * (3.0 - 2.0 * t);
                            1.0 - smooth_t
                        }
                    };

                    let desired_blend = (w * 255.0) as u8;
                    let new_mat = req.settings.material_id as u8;

                    // Commit existing blend if painting a different material.
                    if sv.secondary_material_id != 0
                        && sv.secondary_material_id != new_mat
                        && sv.blend_weight > 0
                    {
                        if sv.blend_weight > 127 {
                            sv.material_id = sv.secondary_material_id;
                        }
                        sv.secondary_material_id = 0;
                        sv.blend_weight = 0;
                    }

                    // Apply new paint.
                    if sv.material_id == new_mat {
                        sv.secondary_material_id = 0;
                        sv.blend_weight = 0;
                    } else {
                        sv.secondary_material_id = new_mat;
                        sv.blend_weight = sv.blend_weight.max(desired_blend);
                    }
                } else if is_erase {
                    // Erase: reduce intensity toward 0 in the color companion pool.
                    let w = if falloff_frac < 0.001 {
                        1.0_f32
                    } else {
                        let edge_start = 1.0 - falloff_frac;
                        if normalized <= edge_start {
                            1.0_f32
                        } else {
                            let t = ((normalized - edge_start) / falloff_frac).clamp(0.0, 1.0);
                            let smooth_t = t * t * (3.0 - 2.0 * t);
                            1.0 - smooth_t
                        }
                    };
                    let brick_slot = self.resolve_brick_slot(req.object_id, entry.brick_coord);
                    if let Some(brick_slot) = brick_slot {
                        let color_slot = self.ensure_color_brick(brick_slot);
                        let vi = entry.voxel_index as usize;
                        let cv = &mut self.cpu_color_bricks[color_slot as usize].data[vi];
                        let old_intensity = cv.intensity();
                        let new_intensity = ((old_intensity as f32) * (1.0 - w)).round() as u8;
                        *cv = rkf_core::companion::ColorVoxel::new(
                            cv.red(), cv.green(), cv.blue(), new_intensity,
                        );
                    }
                } else {
                    // Color paint: write to ColorBrick companion pool with intensity falloff.
                    let w = if falloff_frac < 0.001 {
                        1.0_f32
                    } else {
                        let edge_start = 1.0 - falloff_frac;
                        if normalized <= edge_start {
                            1.0_f32
                        } else {
                            let t = ((normalized - edge_start) / falloff_frac).clamp(0.0, 1.0);
                            let smooth_t = t * t * (3.0 - 2.0 * t);
                            1.0 - smooth_t
                        }
                    };
                    let brick_slot = self.resolve_brick_slot(req.object_id, entry.brick_coord);
                    if let Some(brick_slot) = brick_slot {
                        let color_slot = self.ensure_color_brick(brick_slot);
                        let vi = entry.voxel_index as usize;
                        let cv = &mut self.cpu_color_bricks[color_slot as usize].data[vi];
                        let desired_intensity = (w * 255.0) as u8;
                        let new_intensity = cv.intensity().max(desired_intensity);
                        *cv = rkf_core::companion::ColorVoxel::new(
                            color_bytes[0], color_bytes[1], color_bytes[2], new_intensity,
                        );
                    }
                }

                modified_brick_set.insert(entry.brick_coord);
            }

            // 8. Convert geometry + SDF → Brick for modified bricks and upload to GPU.
            let mut modified_brick_slots = Vec::new();
            let gfd = self.geometry_first_data.get(&req.object_id).unwrap();

            for brick_coord in &modified_brick_set {
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

            // 9. Targeted GPU upload.
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
            }

            // 10. Upload modified color bricks to GPU.
            for brick_coord in &modified_brick_set {
                let geo_slot = match gfd.geo_brick_map.get(brick_coord.x, brick_coord.y, brick_coord.z) {
                    Some(s) if s != EMPTY_SLOT && s != INTERIOR_SLOT => s,
                    _ => continue,
                };
                if let Some(&(_, brick_slot)) = gfd.slot_map.get(&geo_slot) {
                    if let Some(&color_slot) = self.color_companion_map.get(brick_slot as usize) {
                        if color_slot != 0xFFFFFFFF {
                            let brick_data: &[u32; 512] = bytemuck::cast_ref(
                                &self.cpu_color_bricks[color_slot as usize],
                            );
                            self.gpu_color_pool.write_color_brick(
                                &self.ctx.queue, color_slot, brick_data,
                            );
                        }
                    }
                }
            }

            all_modified_slots.extend(&modified_brick_slots);
        }

        all_modified_slots
    }

    /// Look up the brick_pool_slot for a given object and brick coordinate.
    fn resolve_brick_slot(&self, object_id: u32, brick_coord: glam::UVec3) -> Option<u32> {
        let gfd = self.geometry_first_data.get(&object_id)?;
        let geo_slot = gfd.geo_brick_map.get(brick_coord.x, brick_coord.y, brick_coord.z)?;
        if geo_slot == EMPTY_SLOT || geo_slot == INTERIOR_SLOT {
            return None;
        }
        gfd.slot_map.get(&geo_slot).map(|&(_, brick_slot)| brick_slot)
    }

    /// Ensure a ColorBrick exists for the given brick_pool_slot. Returns the color_slot.
    ///
    /// If no ColorBrick exists yet, allocates one (zeroed = no paint, intensity=0)
    /// and updates the companion map on both CPU and GPU.
    fn ensure_color_brick(&mut self, brick_slot: u32) -> u32 {
        // Grow companion map if needed.
        if (brick_slot as usize) >= self.color_companion_map.len() {
            self.color_companion_map.resize(brick_slot as usize + 1, 0xFFFFFFFF);
        }

        let existing = self.color_companion_map[brick_slot as usize];
        if existing != 0xFFFFFFFF {
            return existing;
        }

        // Allocate a new color brick.
        let color_slot = self.cpu_color_bricks.len() as u32;
        self.cpu_color_bricks.push(rkf_core::companion::ColorBrick::default());
        self.color_companion_map[brick_slot as usize] = color_slot;

        // Check if GPU buffers need to grow.
        if color_slot >= self.gpu_color_pool.color_brick_capacity()
            || (brick_slot + 1) > self.gpu_color_pool.companion_map_capacity()
        {
            // Recreate GPU buffers with larger capacity and rebuild bind group.
            let new_color_cap = (color_slot + 1).max(64).next_power_of_two();
            let new_map_cap = self.color_companion_map.len() as u32;
            let color_data: &[u8] = bytemuck::cast_slice(&self.cpu_color_bricks);
            self.gpu_color_pool = rkf_render::GpuColorPool::upload(
                &self.ctx.device,
                color_data,
                &self.color_companion_map,
            );
            // Rebuild group 3 bind group since buffers changed.
            self.shading_pass.rebuild_group3(
                &self.ctx.device, &self.brush_overlay, &self.gpu_color_pool,
            );
            let _ = (new_color_cap, new_map_cap); // used for sizing above
        } else {
            // Incremental: write companion map entry.
            self.gpu_color_pool.write_companion_entry(
                &self.ctx.queue, brick_slot, color_slot,
            );
        }

        color_slot
    }

    /// Update the GPU brush overlay with geodesic flood fill distances.
    ///
    /// Runs the same flood fill used for painting, then uploads the per-voxel
    /// geodesic distances to the shader so it can draw a pixel-perfect ring.
    pub fn update_brush_overlay_from_flood(
        &mut self,
        scene: &rkf_core::Scene,
        object_id: u32,
        world_pos: glam::Vec3,
        brush_radius: f32,
        brush_falloff: f32,
        brush_color: [f32; 4],
    ) {
        use rkf_core::surface_flood::{self, FloodFillParams};
        use rkf_render::brush_overlay::BrushOverlayUniforms;

        // Look up object and its voxel size.
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => {
                self.deactivate_brush_overlay();
                return;
            }
        };
        let voxel_size = match &obj.root_node.sdf_source {
            rkf_core::SdfSource::Voxelized { voxel_size, .. } => *voxel_size,
            _ => {
                self.deactivate_brush_overlay();
                return;
            }
        };
        let gfd = match self.geometry_first_data.get(&object_id) {
            Some(d) => d,
            None => {
                self.deactivate_brush_overlay();
                return;
            }
        };

        // Transform to object-local space.
        let local_pos = crate::paint::world_to_object_local_v2(world_pos, obj);

        // Flood fill parameters — same as paint pipeline.
        let step_cost = glam::Vec3::new(
            voxel_size * obj.scale.x,
            voxel_size * obj.scale.y,
            voxel_size * obj.scale.z,
        );
        let params = FloodFillParams {
            max_distance: brush_radius,
            step_cost,
        };

        let flood_results = surface_flood::surface_flood_fill(
            local_pos,
            &gfd.geo_brick_map,
            &self.cpu_geometry_pool,
            voxel_size,
            &params,
        );

        if flood_results.is_empty() {
            self.deactivate_brush_overlay();
            return;
        }

        // Build companion map: brick_pool_slot → overlay_slot.
        // The map must be sized to cover all brick pool slots.
        let pool_capacity = self.cpu_brick_pool.capacity() as usize;
        let mut companion_map = vec![0xFFFFFFFFu32; pool_capacity];
        let mut overlay_slot_counter = 0u32;

        // Collect which geo_slots are touched, map them to overlay slots.
        let mut geo_slot_to_overlay: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();

        for entry in &flood_results {
            if geo_slot_to_overlay.contains_key(&entry.geo_slot) {
                continue;
            }
            // Translate geo_slot → brick_pool_slot via gfd.slot_map.
            if let Some(&(_sdf_slot, brick_slot)) = gfd.slot_map.get(&entry.geo_slot) {
                let os = overlay_slot_counter;
                overlay_slot_counter += 1;
                geo_slot_to_overlay.insert(entry.geo_slot, os);
                if (brick_slot as usize) < companion_map.len() {
                    companion_map[brick_slot as usize] = os;
                }
            }
        }

        if overlay_slot_counter == 0 {
            self.deactivate_brush_overlay();
            return;
        }

        // Build distance data: overlay_slot * 512 + voxel_index.
        // Initialize with a large sentinel (anything > brush_radius means "no data").
        let total_floats = overlay_slot_counter as usize * 512;
        let mut distances = vec![1e10f32; total_floats];

        for entry in &flood_results {
            if let Some(&os) = geo_slot_to_overlay.get(&entry.geo_slot) {
                let idx = os as usize * 512 + entry.voxel_index as usize;
                if idx < distances.len() {
                    distances[idx] = entry.geodesic_distance;
                }
            }
        }

        let uniforms = BrushOverlayUniforms {
            brush_radius,
            brush_falloff,
            brush_object_id: object_id,
            brush_active: 1,
            brush_color,
            brush_center_local: [local_pos.x, local_pos.y, local_pos.z, 0.0],
        };

        self.brush_overlay.update(
            &self.ctx.device,
            &self.ctx.queue,
            &distances,
            &companion_map,
            &uniforms,
        );
        // Rebuild group 3 bind group since update() recreated data/map buffers.
        self.shading_pass.rebuild_group3(&self.ctx.device, &self.brush_overlay, &self.gpu_color_pool);
    }

    /// Deactivate the brush overlay (hide the cursor ring).
    pub fn deactivate_brush_overlay(&mut self) {
        use rkf_render::brush_overlay::BrushOverlayUniforms;

        let uniforms = BrushOverlayUniforms {
            brush_radius: 0.0,
            brush_falloff: 0.0,
            brush_object_id: 0,
            brush_active: 0,
            brush_color: [1.0, 1.0, 1.0, 1.0],
            brush_center_local: [0.0; 4],
        };
        self.ctx.queue.write_buffer(
            &self.brush_overlay.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }
}
