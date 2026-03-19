//! Render frame methods for the editor engine.

use glam::Vec3;
use rkf_core::{Aabb, Scene};
use rkf_render::{
    DebugMode, GpuObject, Light, SceneUniforms, ShadeUniforms,
};
use rkf_render::radiance_inject::InjectUniforms;
use rkf_core::transform_flatten::flatten_object;

use super::EditorEngine;

impl EditorEngine {
    /// Update GPU scene data incrementally.
    ///
    /// Unified path — all changes flow through dirty_objects, spawned_objects,
    /// or despawned_objects. Full rebuild only for scene open / new project.
    ///
    /// Returns the number of GPU object slots (including tombstones).
    pub(super) fn update_scene_gpu(&mut self, scene: &Scene) -> u32 {
        // Path 0: Full rebuild — only for scene open / new project.
        if self.topology_changed {
            self.full_rebuild(scene);
            return self.cached_gpu_objects.len() as u32;
        }

        // Static frame — no changes at all.
        if self.dirty_objects.is_empty()
            && self.spawned_objects.is_empty()
            && self.despawned_objects.is_empty()
        {
            return self.cached_gpu_objects.len() as u32;
        }

        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        // Build O(1) lookup for scene objects.
        let scene_map: std::collections::HashMap<u32, &rkf_core::SceneObject> =
            scene.objects.iter().map(|o| (o.id, o)).collect();

        let mut bvh_structural_change = false;

        // --- Despawns: tombstone removed objects ---
        for obj_id in std::mem::take(&mut self.despawned_objects) {
            if let Some((start_idx, count)) = self.object_gpu_ranges.remove(&obj_id) {
                // Mark old AABB dirty in coarse field before removing.
                if let Some(old_aabb) = self.cached_world_aabbs.remove(&obj_id) {
                    self.coarse_field.mark_dirty(old_aabb.min, old_aabb.max);
                }
                // Zero out GPU slots — sdf_type=0 means invisible to ray march.
                let zeroed: GpuObject = bytemuck::Zeroable::zeroed();
                let empty_aabb = Aabb::new(Vec3::ZERO, Vec3::ZERO);
                for i in start_idx..start_idx + count {
                    self.cached_gpu_objects[i] = zeroed;
                    self.cached_bvh_pairs[i].1 = empty_aabb;
                }
                self.gpu_scene.upload_object_range(
                    &self.ctx.queue,
                    &self.cached_gpu_objects[start_idx..start_idx + count],
                    start_idx,
                );
                self.tombstone_count += count;
                bvh_structural_change = true;
            }
        }

        // --- Spawns: append new objects ---
        for obj_id in std::mem::take(&mut self.spawned_objects) {
            let Some(obj) = scene_map.get(&obj_id) else { continue };
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let world_aabb = Self::compute_world_aabb(obj, wt);
            let flat_nodes = flatten_object(obj, wt.position);

            let start_idx = self.cached_gpu_objects.len();
            for flat in &flat_nodes {
                let gpu_idx = self.cached_gpu_objects.len() as u32;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                self.cached_gpu_objects.push(GpuObject::from_flat_node(
                    flat, obj.id,
                    [world_aabb.min.x, world_aabb.min.y, world_aabb.min.z, 0.0],
                    [world_aabb.max.x, world_aabb.max.y, world_aabb.max.z, 0.0],
                    geom_min, geom_max,
                ));
                self.cached_bvh_pairs.push((gpu_idx, world_aabb));
            }
            let count = self.cached_gpu_objects.len() - start_idx;
            self.object_gpu_ranges.insert(obj.id, (start_idx, count));
            self.cached_world_aabbs.insert(obj.id, world_aabb);

            // Upload the full buffer (may have grown past capacity).
            self.gpu_scene.upload_objects(
                &self.ctx.device, &self.ctx.queue, &self.cached_gpu_objects,
            );

            // Mark new AABB dirty in coarse field.
            self.coarse_field.mark_dirty(world_aabb.min, world_aabb.max);
            bvh_structural_change = true;
        }

        // --- Dirty objects: update in-place or classify as spawn ---
        let dirty_ids: Vec<u32> = self.dirty_objects.drain().collect();
        let mut deferred_spawns: Vec<u32> = Vec::new();

        for &obj_id in &dirty_ids {
            let Some(obj) = scene_map.get(&obj_id) else { continue };
            let Some(&(start_idx, count)) = self.object_gpu_ranges.get(&obj_id) else {
                // Object not in cache — treat as spawn.
                deferred_spawns.push(obj_id);
                continue;
            };
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let world_aabb = Self::compute_world_aabb(obj, wt);
            let flat_nodes = flatten_object(obj, wt.position);

            if flat_nodes.len() != count {
                // Node count changed — need full rebuild.
                self.topology_changed = true;
                return self.full_rebuild_and_return(scene);
            }

            // Mark old AABB dirty, then new AABB dirty (for coarse field).
            if let Some(old_aabb) = self.cached_world_aabbs.get(&obj_id) {
                if *old_aabb != world_aabb {
                    self.coarse_field.mark_dirty(old_aabb.min, old_aabb.max);
                    self.coarse_field.mark_dirty(world_aabb.min, world_aabb.max);
                }
            }

            // Patch cached GPU objects in-place.
            for (i, flat) in flat_nodes.iter().enumerate() {
                let idx = start_idx + i;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                self.cached_gpu_objects[idx] = GpuObject::from_flat_node(
                    flat, obj.id,
                    [world_aabb.min.x, world_aabb.min.y, world_aabb.min.z, 0.0],
                    [world_aabb.max.x, world_aabb.max.y, world_aabb.max.z, 0.0],
                    geom_min, geom_max,
                );
                self.cached_bvh_pairs[idx].1 = world_aabb;
            }

            // Partial GPU upload for this object's range.
            self.gpu_scene.upload_object_range(
                &self.ctx.queue,
                &self.cached_gpu_objects[start_idx..start_idx + count],
                start_idx,
            );

            self.cached_world_aabbs.insert(obj_id, world_aabb);
        }

        // Process any dirty objects that weren't in cache (treat as spawns).
        if !deferred_spawns.is_empty() {
            for obj_id in deferred_spawns {
                let Some(obj) = scene_map.get(&obj_id) else { continue };
                let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
                let world_aabb = Self::compute_world_aabb(obj, wt);
                let flat_nodes = flatten_object(obj, wt.position);

                let start_idx = self.cached_gpu_objects.len();
                for flat in &flat_nodes {
                    let gpu_idx = self.cached_gpu_objects.len() as u32;
                    let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                    self.cached_gpu_objects.push(GpuObject::from_flat_node(
                        flat, obj.id,
                        [world_aabb.min.x, world_aabb.min.y, world_aabb.min.z, 0.0],
                        [world_aabb.max.x, world_aabb.max.y, world_aabb.max.z, 0.0],
                        geom_min, geom_max,
                    ));
                    self.cached_bvh_pairs.push((gpu_idx, world_aabb));
                }
                let count = self.cached_gpu_objects.len() - start_idx;
                self.object_gpu_ranges.insert(obj.id, (start_idx, count));
                self.cached_world_aabbs.insert(obj.id, world_aabb);
                self.coarse_field.mark_dirty(world_aabb.min, world_aabb.max);
                bvh_structural_change = true;
            }
            // Upload full buffer (may have grown).
            self.gpu_scene.upload_objects(
                &self.ctx.device, &self.ctx.queue, &self.cached_gpu_objects,
            );
        }

        // --- BVH: rebuild if topology changed, refit otherwise ---
        if bvh_structural_change {
            // Filter out tombstoned entries for a clean BVH build.
            let live_pairs: Vec<(u32, Aabb)> = self.cached_bvh_pairs.iter()
                .filter(|(_, aabb)| aabb.min != aabb.max || aabb.min != Vec3::ZERO)
                .copied()
                .collect();
            let bvh = rkf_core::Bvh::build(&live_pairs);
            self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);
            self.cached_bvh = Some(bvh);
        } else if !dirty_ids.is_empty() {
            if let Some(ref mut bvh) = self.cached_bvh {
                // Refit with live pairs only.
                let live_pairs: Vec<(u32, Aabb)> = self.cached_bvh_pairs.iter()
                    .filter(|(_, aabb)| aabb.min != aabb.max || aabb.min != Vec3::ZERO)
                    .copied()
                    .collect();
                *bvh = rkf_core::Bvh::build(&live_pairs);
                self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, bvh);
            }
        }

        // --- Coarse field: incremental update from dirty regions ---
        if self.coarse_field.is_dirty() {
            let aabb_pairs: Vec<(Vec3, Vec3)> = self.cached_world_aabbs
                .values()
                .map(|a| (a.min, a.max))
                .collect();
            self.coarse_field.update_dirty(&self.ctx.queue, &aabb_pairs);
        }

        // --- Compaction: if too many tombstones, do a full rebuild ---
        let total = self.cached_gpu_objects.len();
        if total > 0 && self.tombstone_count > 0
            && self.tombstone_count as f32 / total as f32 > 0.25
        {
            log::info!(
                "Compacting GPU objects: {} tombstones / {} total",
                self.tombstone_count, total,
            );
            self.full_rebuild(scene);
        }

        self.cached_gpu_objects.len() as u32
    }

    /// Full rebuild of all GPU scene data from the scene.
    /// Used for scene open / new project / compaction.
    fn full_rebuild(&mut self, scene: &Scene) {
        self.cached_gpu_objects.clear();
        self.cached_bvh_pairs.clear();
        self.cached_world_aabbs.clear();
        self.object_gpu_ranges.clear();
        self.dirty_objects.clear();
        self.spawned_objects.clear();
        self.despawned_objects.clear();
        self.tombstone_count = 0;

        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let world_aabb = Self::compute_world_aabb(obj, wt);
            self.cached_world_aabbs.insert(obj.id, world_aabb);

            let start_idx = self.cached_gpu_objects.len();
            let flat_nodes = flatten_object(obj, wt.position);
            for flat in &flat_nodes {
                let gpu_idx = self.cached_gpu_objects.len() as u32;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                self.cached_gpu_objects.push(GpuObject::from_flat_node(
                    flat, obj.id,
                    [world_aabb.min.x, world_aabb.min.y, world_aabb.min.z, 0.0],
                    [world_aabb.max.x, world_aabb.max.y, world_aabb.max.z, 0.0],
                    geom_min, geom_max,
                ));
                self.cached_bvh_pairs.push((gpu_idx, world_aabb));
            }
            let count = self.cached_gpu_objects.len() - start_idx;
            self.object_gpu_ranges.insert(obj.id, (start_idx, count));
        }

        self.gpu_scene.upload_objects(
            &self.ctx.device, &self.ctx.queue, &self.cached_gpu_objects,
        );

        let bvh = rkf_core::Bvh::build(&self.cached_bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);
        self.cached_bvh = Some(bvh);

        let aabb_pairs: Vec<(Vec3, Vec3)> = self.cached_world_aabbs
            .values()
            .map(|a| (a.min, a.max))
            .collect();
        self.coarse_field.populate(&aabb_pairs);
        self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

        self.topology_changed = false;
    }

    /// Convenience: set topology_changed and do a full rebuild, returning the count.
    fn full_rebuild_and_return(&mut self, scene: &Scene) -> u32 {
        self.full_rebuild(scene);
        self.cached_gpu_objects.len() as u32
    }

    /// Compute world-space AABB from a scene object and its baked world transform.
    fn compute_world_aabb(
        obj: &rkf_core::SceneObject,
        wt: &rkf_core::transform_bake::WorldTransform,
    ) -> rkf_core::Aabb {
        let local_aabb = obj.aabb;
        let smin = local_aabb.min * wt.scale;
        let smax = local_aabb.max * wt.scale;
        let corners = [
            Vec3::new(smin.x, smin.y, smin.z), Vec3::new(smax.x, smin.y, smin.z),
            Vec3::new(smin.x, smax.y, smin.z), Vec3::new(smax.x, smax.y, smin.z),
            Vec3::new(smin.x, smin.y, smax.z), Vec3::new(smax.x, smin.y, smax.z),
            Vec3::new(smin.x, smax.y, smax.z), Vec3::new(smax.x, smax.y, smax.z),
        ];
        let mut wmin = Vec3::splat(f32::MAX);
        let mut wmax = Vec3::splat(f32::MIN);
        for c in &corners {
            let r = wt.rotation * *c + wt.position;
            wmin = wmin.min(r);
            wmax = wmax.max(r);
        }
        rkf_core::Aabb::new(wmin, wmax)
    }

    /// Render one frame without UI overlay (full-screen engine blit).
    pub fn render_frame(&mut self, scene: &Scene) {
        self.render_frame_composited(scene, |_, _, _| {});
    }

    /// Render one frame with an overlay compositing callback.
    ///
    /// The callback runs after the engine has submitted its compute passes
    /// and blit to the swapchain, but before present. The callback receives
    /// the device, queue, and swapchain target view so it can render an
    /// overlay (e.g. rinch UI) on top.
    ///
    /// The engine blit fills the full swapchain. Use `render_frame_viewport`
    /// for sub-region rendering with panels.
    pub fn render_frame_composited<F>(&mut self, scene: &Scene, post_engine: F)
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(scene, None, post_engine);
    }

    /// Render one frame with engine output constrained to a viewport sub-region.
    ///
    /// The engine blit is drawn into `viewport` (x, y, width, height) in pixels.
    /// Areas outside the viewport are cleared to black (covered by UI panels).
    /// The `post_engine` callback composites the UI overlay on top.
    pub fn render_frame_viewport<F>(
        &mut self,
        scene: &Scene,
        viewport: (f32, f32, f32, f32),
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(scene, Some(viewport), post_engine);
    }

    pub(super) fn render_frame_inner<F>(
        &mut self,
        scene: &Scene,
        viewport: Option<(f32, f32, f32, f32)>,
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        let camera_pos_vec = self.camera.position;

        // Incremental scene-to-GPU update: only re-flatten dirty objects,
        // skip entirely on static frames (camera-only movement).
        let num_objects = self.update_scene_gpu(scene);

        // Update camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        // Scene uniforms.
        let scene_uniforms = SceneUniforms {
            num_objects,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Light buffer: only re-upload when lights or camera position changed.
        let total_lights = 1 + self.world_lights.len() as u32;
        let cam = self.camera.position;
        let cam_moved = cam != self.last_light_cam_pos;
        if self.lights_dirty || cam_moved {
            let sun_light = Light {
                light_type: 0, // directional
                pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
                dir_x: self.env_sun_dir[0],
                dir_y: self.env_sun_dir[1],
                dir_z: self.env_sun_dir[2],
                color_r: self.env_sun_color[0],
                color_g: self.env_sun_color[1],
                color_b: self.env_sun_color[2],
                intensity: 1.0,
                range: 0.0,
                inner_angle: 0.0,
                outer_angle: 0.0,
                cookie_index: -1,
                shadow_caster: 1,
            };
            let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
                let mut cl = *l;
                if cl.light_type != 0 {
                    cl.pos_x -= cam.x;
                    cl.pos_y -= cam.y;
                    cl.pos_z -= cam.z;
                }
                cl
            }).collect();
            let mut all_lights = vec![sun_light];
            all_lights.extend(cam_rel_lights);
            self.light_buffer.update(&self.ctx.queue, &all_lights);
            self.lights_dirty = false;
            self.last_light_cam_pos = cam;
        }

        // Shade uniforms — includes atmosphere + camera basis for sky rendering.
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = self.render_width as f32 / self.render_height as f32;
        let fwd = self.camera.forward();
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;

        self.shading_pass.update_uniforms(&self.ctx.queue, &ShadeUniforms {
            debug_mode: self.shade_debug_mode,
            num_lights: total_lights,
            _pad0: 0,
            shadow_budget_k: 0,
            camera_pos: [camera_pos_vec.x, camera_pos_vec.y, camera_pos_vec.z, 0.0],
            sun_dir: [self.env_sun_dir[0], self.env_sun_dir[1], self.env_sun_dir[2], self.env_sun_intensity],
            sun_color: [self.env_sun_color_raw[0], self.env_sun_color_raw[1], self.env_sun_color_raw[2], 0.0],
            sky_params: [self.env_rayleigh_scale, self.env_mie_scale, if self.env_atmosphere_enabled { 1.0 } else { 0.0 }, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
        });

        // Debug view (kept for fallback).
        let debug_mode = match self.shade_debug_mode {
            0 => DebugMode::Lambert,
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, debug_mode);

        // Store VP for next frame's motion vectors.
        self.prev_vp = self.camera.view_projection(self.render_width, self.render_height)
            .to_cols_array_2d();

        // Poll the GPU device to acknowledge completed work from previous frames.
        // Without this, heavy compute (cloud FBM) can fill the swapchain during
        // rapid slider drags, causing get_current_texture() to block permanently.
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        // Get swapchain texture (surface-based path).
        let surface = self.surface.as_ref()
            .expect("render_frame_inner requires a surface (use render_frame_offscreen for compositor path)");
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(
                    self.window_width.max(64),
                    self.window_height.max(64),
                );
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let target_view = frame.texture.create_view(&Default::default());

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // Update per-frame uniforms.
        self.coarse_field.update_uniforms(&self.ctx.queue, self.camera.position);
        self.radiance_volume.update_center(
            &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
        );
        self.radiance_inject.update_uniforms(&self.ctx.queue, &InjectUniforms {
            num_lights: total_lights,
            max_shadow_lights: 1,
            _pad: [0; 2],
        });

        // --- Core rendering ---
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        self.radiance_inject.dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.radiance_mip.dispatch(&mut encoder);
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );

        // --- Volumetric pipeline (uses cached env values) ---
        // Upload cloud parameters each frame (time advances for wind animation).
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

        // DEBUG: Save cloud params for post-frame diagnostic.
        let cloud_params_snapshot = [
            cloud_params.flags[0],     // enabled
            cloud_params.altitude[0],  // cloud_min
            cloud_params.altitude[1],  // cloud_max
            cloud_params.altitude[2],  // threshold
            cloud_params.altitude[3],  // density_scale
        ];
        let cam_snapshot = [cam.x, cam.y, cam.z];

        self.vol_march.set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow.set_cloud_params(&self.ctx.queue, &cloud_params);

        let sun_dir = self.env_sun_dir;
        // Skip vol_shadow dispatch when camera and sun haven't changed.
        let vs_cam_delta = Vec3::new(
            cam.x - self.last_vol_shadow_cam_pos.x,
            cam.y - self.last_vol_shadow_cam_pos.y,
            cam.z - self.last_vol_shadow_cam_pos.z,
        );
        let vs_sun_changed = sun_dir != self.last_vol_shadow_sun_dir;
        if vs_cam_delta.length_squared() > 0.01 || vs_sun_changed {
            self.vol_shadow.dispatch(
                &mut encoder, &self.ctx.queue,
                [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
            );
            self.last_vol_shadow_cam_pos = cam;
            self.last_vol_shadow_sun_dir = sun_dir;
        }
        // Use the actual cloud altitude from cloud_params, not the defaults
        // (DEFAULT_CLOUD_MIN=1000, DEFAULT_CLOUD_MAX=3000).  The shadow map must
        // march through the same altitude band as the visible clouds.
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],  // cloud_min
            cloud_params.altitude[1],  // cloud_max
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);
        let sc = self.env_sun_color;
        let fc = self.env_fog_color;
        let fog_alpha = if self.env_fog_density > 0.0 { 1.0 } else { 0.0 };
        // Reuse the FOV-scaled camera basis from the shade pass so vol march
        // rays match the G-buffer exactly (same fwd/right/up from line ~990).
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [sc[0], sc[1], sc[2], 0.0],
            width: self.render_width / 2,
            height: self.render_height / 2,
            full_width: self.render_width,
            full_height: self.render_height,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [fc[0], fc[1], fc[2], fog_alpha],
            fog_height: [self.env_fog_density, -0.5, self.env_fog_height_falloff, 0.0],
            fog_distance: [0.0, 0.01, self.env_ambient_dust, self.env_dust_g],
            frame_index: self.frame_index,
            vol_ambient_color: {
                let i = self.env_vol_ambient_intensity;
                [self.env_vol_ambient_color[0] * i, self.env_vol_ambient_color[1] * i, self.env_vol_ambient_color[2] * i]
            },
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // --- Post-processing pipeline ---
        // Project sun to screen UV for radial blur god rays.
        {
            let sun_dir = glam::Vec3::from(self.env_sun_dir).normalize_or_zero();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir.dot(right) / sun_dot;
                let ndc_y = -sun_dir.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur.update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // --- Blit engine output to full swapchain ---
        // Always blit to the full window. The rinch overlay will paint opaque
        // panels over non-viewport areas (like the game-embed and video player
        // examples). This avoids sub-pixel alignment gaps between the viewport
        // blit region and the overlay's transparent hole.
        let _ = viewport; // viewport used only for resize_render, not blit positioning
        self.blit.draw(&mut encoder, &target_view);

        // Submit engine work.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Let the caller composite overlay (e.g. rinch UI) on top.
        // This runs after engine submit (Vello needs the queue), before present.
        post_engine(&self.ctx.device, &self.ctx.queue, &target_view);

        // --- GPU pick readback (single pixel from material G-buffer) ---
        let pending_pick = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_pick.take());

        if let Some((px, py)) = pending_pick {
            // Clamp to internal resolution bounds.
            let px = px.min(self.render_width.saturating_sub(1));
            let py = py.min(self.render_height.saturating_sub(1));

            let mut pick_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("pick_readback") },
            );
            pick_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256), // must be 256-aligned
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(pick_enc.finish()));

            // Synchronous readback — fast for 1 pixel.
            let slice = self.pick_readback_buffer.slice(..4);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let object_id = packed >> 24; // bits 24-31
                drop(data);
                self.pick_readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.pick_result = Some(object_id);
                }
            } else {
                self.pick_readback_buffer.unmap();
            }
        }

        // --- GPU brush hit readback (position + object_id from G-buffer) ---
        let pending_brush = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_brush_hit.take());

        if let Some((bx, by)) = pending_brush {
            let bx = bx.min(self.render_width.saturating_sub(1));
            let by = by.min(self.render_height.saturating_sub(1));

            let mut brush_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("brush_readback") },
            );

            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.position_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.brush_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            self.ctx.queue.submit(std::iter::once(brush_enc.finish()));

            let pos_slice = self.brush_readback_buffer.slice(..16);
            let (tx_pos, rx_pos) = std::sync::mpsc::channel();
            pos_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_pos.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut hit_pos = [0.0f32; 4];
            let pos_ok = if let Ok(Ok(())) = rx_pos.recv() {
                let data = pos_slice.get_mapped_range();
                hit_pos = [
                    f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                    f32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                    f32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                    f32::from_le_bytes([data[12], data[13], data[14], data[15]]),
                ];
                drop(data);
                self.brush_readback_buffer.unmap();
                true
            } else {
                self.brush_readback_buffer.unmap();
                false
            };

            let mat_slice = self.pick_readback_buffer.slice(..4);
            let (tx_mat, rx_mat) = std::sync::mpsc::channel();
            mat_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_mat.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut object_id = 0u32;
            if let Ok(Ok(())) = rx_mat.recv() {
                let data = mat_slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                object_id = packed >> 24;
                drop(data);
                self.pick_readback_buffer.unmap();
            } else {
                self.pick_readback_buffer.unmap();
            }

            if pos_ok && hit_pos[3] < 1e30 {
                let result = crate::automation::BrushHitResult {
                    position: Vec3::new(hit_pos[0], hit_pos[1], hit_pos[2]),
                    object_id,
                };
                if let Ok(mut state) = self.shared_state.lock() {
                    state.brush_hit_result = Some(result);
                }
            }
        }

        // --- Screenshot readback (after overlay composite, captures full UI) ---
        let do_readback = self.shared_state.lock()
            .map(|s| s.screenshot_requested)
            .unwrap_or(false);

        if do_readback {
            let w = self.window_width;
            let h = self.window_height;
            let bytes_per_pixel = 4u32;
            let unpadded_row = w * bytes_per_pixel;
            let padded_row = (unpadded_row + 255) & !255;

            // Copy from composited swapchain texture to readback buffer.
            let mut readback_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("screenshot_readback") },
            );
            readback_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffers[0],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            self.ctx.queue.submit(std::iter::once(readback_enc.finish()));

            // Map and read back the pixels.
            let buffer_slice = self.readback_buffers[0].slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let pixel_count = (w * h) as usize;
                let mut rgba8 = vec![0u8; pixel_count * 4];
                let is_bgra = matches!(
                    self.surface_format,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                );

                for y in 0..h as usize {
                    let src_row_offset = y * padded_row as usize;
                    let dst_row_offset = y * w as usize * 4;
                    let row_bytes = w as usize * 4;
                    rgba8[dst_row_offset..dst_row_offset + row_bytes]
                        .copy_from_slice(&data[src_row_offset..src_row_offset + row_bytes]);
                }

                // Convert BGRA → RGBA if the surface format is BGRA.
                if is_bgra {
                    for pixel in rgba8.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                }

                drop(data);
                self.readback_buffers[0].unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_pixels = rgba8;
                    state.frame_width = w;
                    state.frame_height = h;
                    state.screenshot_requested = false;
                }
            } else {
                self.readback_buffers[0].unmap();
                if let Ok(mut state) = self.shared_state.lock() {
                    state.screenshot_requested = false;
                }
            }
        }

        frame.present();

        // DEBUG: Mark frame completed (pair with pre-dispatch write above).
        {
            let diag = format!(
                "f={} en={} min={:.1} max={:.1} thr={:.4} dens={:.2} cam=[{:.1},{:.1},{:.1}] res={}x{} OK\n",
                self.frame_index,
                cloud_params_snapshot[0] > 0.5,
                cloud_params_snapshot[1],
                cloud_params_snapshot[2],
                cloud_params_snapshot[3],
                cloud_params_snapshot[4],
                cam_snapshot[0], cam_snapshot[1], cam_snapshot[2],
                self.render_width / 2, self.render_height / 2,
            );
            let _ = std::fs::write("/tmp/rkf-cloud-diag.txt", &diag);
        }

        self.frame_index += 1;
    }
}
