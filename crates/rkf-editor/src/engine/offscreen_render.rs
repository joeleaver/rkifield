//! Offscreen render frame and readback methods for EditorEngine.

use glam::Vec3;
use rkf_core::Scene;
use rkf_render::{
    DebugMode, SceneUniforms, ShadeUniforms, Light,
};
use rkf_render::radiance_inject::InjectUniforms;
use super::EditorEngine;

impl EditorEngine {
    /// Render one frame to the offscreen texture (compositor path).
    ///
    /// Runs the full compute pipeline at internal resolution, then blits to the
    /// offscreen render target at viewport resolution. The compositor reads
    /// the offscreen texture directly — no CPU readback needed per frame.
    ///
    /// Screenshot readback only happens when `SharedState.screenshot_requested`
    /// is true.
    pub fn render_frame_offscreen(&mut self, scene: &Scene) {
        let camera_pos_vec = self.camera.position;

        // Incremental scene-to-GPU update: only re-flatten dirty objects,
        // skip entirely on static frames (camera-only movement).
        let num_objects = self.update_scene_gpu(scene);

        // Camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

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
                light_type: 0, pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
                dir_x: self.env_sun_dir[0], dir_y: self.env_sun_dir[1], dir_z: self.env_sun_dir[2],
                color_r: self.env_sun_color[0], color_g: self.env_sun_color[1], color_b: self.env_sun_color[2],
                intensity: 1.0, range: 0.0, inner_angle: 0.0, outer_angle: 0.0,
                cookie_index: -1, shadow_caster: 1,
            };
            let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
                let mut cl = *l;
                if cl.light_type != 0 { cl.pos_x -= cam.x; cl.pos_y -= cam.y; cl.pos_z -= cam.z; }
                cl
            }).collect();
            let mut all_lights = vec![sun_light];
            all_lights.extend(cam_rel_lights);
            self.light_buffer.update(&self.ctx.queue, &all_lights);
            self.lights_dirty = false;
            self.last_light_cam_pos = cam;
        }

        // Shade uniforms.
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

        let debug_mode = match self.shade_debug_mode {
            0 => DebugMode::Lambert,
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, debug_mode);

        self.prev_vp = self.camera.view_projection(self.render_width, self.render_height)
            .to_cols_array_2d();

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_frame"),
        });

        self.gpu_profiler.begin_frame();

        // Per-frame uniforms.
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
        self.gpu_profiler.begin_pass(&mut encoder, "tile_cull");
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "ray_march");
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "radiance_inject");
        self.radiance_inject.dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "radiance_mip");
        self.radiance_mip.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "shading");
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Volumetric pipeline ---
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

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
            self.gpu_profiler.begin_pass(&mut encoder, "vol_shadow");
            self.vol_shadow.dispatch(
                &mut encoder, &self.ctx.queue,
                [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
            );
            self.gpu_profiler.end_pass(&mut encoder);
            self.last_vol_shadow_cam_pos = cam;
            self.last_vol_shadow_sun_dir = sun_dir;
        }

        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.gpu_profiler.begin_pass(&mut encoder, "cloud_shadow");
        self.cloud_shadow.dispatch_only(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        let sc = self.env_sun_color;
        let fc = self.env_fog_color;
        let fog_alpha = if self.env_fog_density > 0.0 { 1.0 } else { 0.0 };
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
        self.gpu_profiler.begin_pass(&mut encoder, "vol_march");
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "vol_upscale");
        self.vol_upscale.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "vol_composite");
        self.vol_composite.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Post-processing pipeline ---
        {
            let sun_dir_v = glam::Vec3::from(self.env_sun_dir).normalize_or_zero();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir_v.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir_v.dot(right) / sun_dot;
                let ndc_y = -sun_dir_v.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur.update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.gpu_profiler.begin_pass(&mut encoder, "god_rays");
        self.god_rays_blur.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "bloom");
        self.bloom.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "auto_exposure");
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "dof");
        self.dof.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "motion_blur");
        self.motion_blur.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "bloom_composite");
        self.bloom_composite.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "tone_map");
        self.tone_map.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "color_grade");
        self.color_grade.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "cosmetics");
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Blit to offscreen render target ---
        let offscreen_view = self.offscreen_view.as_ref()
            .expect("offscreen_view must exist in offscreen mode");
        self.gpu_profiler.begin_pass(&mut encoder, "blit");
        if let Some(ref offscreen_blit) = self.offscreen_blit {
            offscreen_blit.draw(&mut encoder, offscreen_view);
        }
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Wireframe overlays (same encoder — no extra GPU submit) ---
        if !self.pending_wireframe.is_empty() {
            let vp_matrix = self.camera.view_projection(
                self.viewport_width, self.viewport_height,
            );
            let viewport = (
                0.0f32, 0.0f32,
                self.viewport_width as f32, self.viewport_height as f32,
            );
            if let Some(ref mut wf) = self.wireframe_pass {
                wf.draw(
                    &self.ctx.device, &self.ctx.queue, &mut encoder,
                    offscreen_view, vp_matrix, viewport, &self.pending_wireframe,
                );
            }
            self.pending_wireframe.clear();
        }

        // --- Frame readback (double-buffered CPU pixel path) ---
        // Copy to the CURRENT parity buffer. The previous parity buffer
        // holds last frame's data and can be mapped without waiting.
        {
            let w = self.viewport_width;
            let h = self.viewport_height;
            let padded_row = (w * 4 + 255) & !255;
            let offscreen_tex = self.offscreen_texture.as_ref()
                .expect("offscreen_texture must exist");
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: offscreen_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffers[self.readback_parity],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        // Resolve GPU timestamps before submit.
        self.gpu_profiler.resolve(&mut encoder);

        // Single submit for render + wireframe + readback copy.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // --- GPU pick readback (same as surface path) ---
        let pending_pick = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_pick.take());

        if let Some((px, py)) = pending_pick {
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
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(pick_enc.finish()));

            let slice = self.pick_readback_buffer.slice(..4);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let object_id = packed >> 24;
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

            // Copy 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
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

            // Also copy 1 pixel from material G-buffer for object_id (reuse pick buffer).
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

            // Read position (4xf32 = 16 bytes).
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

            // Read object_id from material G-buffer (bits 24-31).
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

            // Only produce a hit result if the ray actually hit geometry (not sky).
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

        // Screenshot readback is now handled by the caller via map_readback(),
        // which reads the readback_buffer populated by the copy above.

        self.frame_index += 1;
    }

    /// Synchronous readback: wait for GPU to finish, map buffer, read pixels.
    ///
    /// Called after `render_frame_offscreen()` which submitted render + copy
    /// to `readback_buffers[0]`. This waits for the GPU, reads the pixels,
    /// and returns them. The engine loop runs at GPU frame rate.
    pub fn map_readback(&mut self) -> (Vec<u8>, u32, u32) {
        let w = self.viewport_width;
        let h = self.viewport_height;
        let padded_row = (w * 4 + 255) & !255;

        let t0 = std::time::Instant::now();

        let buffer_slice = self.readback_buffers[0].slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let t1 = std::time::Instant::now();

        let mut rgba8 = vec![0u8; (w * h * 4) as usize];
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..h as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * w as usize * 4;
                let row_bytes = w as usize * 4;
                if src_offset + row_bytes <= data.len()
                    && dst_offset + row_bytes <= rgba8.len()
                {
                    rgba8[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&data[src_offset..src_offset + row_bytes]);
                }
            }
            drop(data);
            self.readback_buffers[0].unmap();
        } else {
            self.readback_buffers[0].unmap();
        }

        let t2 = std::time::Instant::now();

        // GPU timestamps are ready after poll — read them.
        self.gpu_profiler.read_results(&self.ctx.device);

        // Log timing every 60 frames.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // (Profiling log suppressed — uncomment to debug GPU timing.)
        // if n % 60 == 0 { ... }
        let _ = (n, t0, t1, t2, w, h);

        (rgba8, w, h)
    }
}
