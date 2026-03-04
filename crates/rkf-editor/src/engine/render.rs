//! Render frame methods for the editor engine.

use glam::Vec3;
use rkf_core::Scene;
use rkf_render::{
    DebugMode, GpuObject, Light, SceneUniforms, ShadeUniforms,
};
use rkf_render::radiance_inject::InjectUniforms;
use rkf_core::transform_flatten::flatten_object;

use super::EditorEngine;

impl EditorEngine {
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

        // Bake world transforms for parent-child hierarchy.
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        // Flatten all objects → GPU object list + BVH.
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        let mut world_aabbs_for_coarse: Vec<(Vec3, Vec3)> = Vec::new();
        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos_vec;
            // Transform local AABB to world space via baked transform.
            let local_aabb = obj.aabb;
            let world_aabb = {
                let smin = local_aabb.min * wt.scale;
                let smax = local_aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z),
                    Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z),
                    Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z),
                    Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z),
                    Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                rkf_core::Aabb::new(wmin, wmax)
            };
            world_aabbs_for_coarse.push((world_aabb.min, world_aabb.max));
            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - self.camera.position;
                let cam_rel_max = world_aabb.max - self.camera.position;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                    geom_min,
                    geom_max,
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        self.gpu_scene.upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);

        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Repopulate coarse field every frame so moved objects stay visible.
        self.coarse_field.populate(&world_aabbs_for_coarse);
        self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

        // Update camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        // Scene uniforms.
        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Synthesize directional light from environment sun settings.
        let sun_light = Light {
            light_type: 0, // directional
            pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: self.env_sun_dir[0],
            dir_y: self.env_sun_dir[1],
            dir_z: self.env_sun_dir[2],
            color_r: self.env_sun_color[0],
            color_g: self.env_sun_color[1],
            color_b: self.env_sun_color[2],
            intensity: 1.0, // already baked into env_sun_color (sun_color * sun_intensity)
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: 1,
        };

        // Camera-relative point/spot lights.
        let cam = self.camera.position;
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            if cl.light_type != 0 {
                cl.pos_x -= cam.x;
                cl.pos_y -= cam.y;
                cl.pos_z -= cam.z;
            }
            cl
        }).collect();

        // Sun (directional) + point/spot lights.
        let total_lights = 1 + cam_rel_lights.len() as u32;
        let mut all_lights = vec![sun_light];
        all_lights.extend(cam_rel_lights);
        self.light_buffer.update(&self.ctx.queue, &all_lights);

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
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
        );
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
            _pad0: 0, _pad1: 0, _pad2: 0,
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
                    buffer: &self.readback_buffer,
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
            let buffer_slice = self.readback_buffer.slice(..);
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
                self.readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_pixels = rgba8;
                    state.frame_width = w;
                    state.frame_height = h;
                    state.screenshot_requested = false;
                }
            } else {
                self.readback_buffer.unmap();
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
