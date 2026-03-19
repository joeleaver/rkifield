//! Per-frame rendering — `render_to_surface`, `render_offscreen`, `render_frame`.

use glam::Vec3;

use rkf_core::transform_bake;
use rkf_core::transform_flatten::flatten_object;
use rkf_render::radiance_inject::InjectUniforms;
use rkf_render::{
    CoarseField, DebugMode, GpuObject, Light, SceneUniforms, ShadeUniforms, COARSE_VOXEL_SIZE,
};

use super::helpers::transform_aabb;
use super::Renderer;
use crate::api::world::World;

impl Renderer {
    // ── Per-frame rendering ────────────────────────────────────────────────

    /// Render a frame from the given world to a surface.
    pub fn render_to_surface(&mut self, world: &World, surface: &wgpu::Surface<'_>) {
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let target_view = frame.texture.create_view(&Default::default());
        self.render_frame(world, Some(&target_view));
        frame.present();
    }

    /// Render a frame to the internal offscreen texture (for editor compositing).
    ///
    /// Runs the full 21-pass compute pipeline, then blits to the offscreen render
    /// target at viewport resolution. Returns the offscreen texture view for the
    /// compositor. If no offscreen target exists (surface-based mode), returns the
    /// cosmetics compute output directly.
    pub fn render_offscreen(&mut self, world: &World) -> &wgpu::TextureView {
        if self.offscreen_view.is_some() {
            // Offscreen path: render compute passes, then blit to offscreen target.
            self.render_frame(world, None);
            // Blit from cosmetics output to offscreen sRGB target.
            if let Some(ref offscreen_blit) = self.offscreen_blit {
                if let Some(ref offscreen_view) = self.offscreen_view {
                    let mut encoder = self.ctx.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("offscreen_blit"),
                        },
                    );
                    offscreen_blit.draw(&mut encoder, offscreen_view);
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                }
            }
            self.offscreen_view.as_ref().unwrap()
        } else {
            // No offscreen target — return compute output directly.
            self.render_frame(world, None);
            &self.cosmetics.output_view
        }
    }

    /// Core render dispatch — executes all 21 passes.
    fn render_frame(&mut self, world: &World, blit_target: Option<&wgpu::TextureView>) {
        let scene = world.build_render_scene();
        let camera_pos = self.camera.position;
        let iw = self.internal_width;
        let ih = self.internal_height;

        // Bake world transforms for parent-child hierarchy
        let world_transforms = transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = transform_bake::WorldTransform::default();

        // Flatten all objects to GPU representation + build BVH pairs
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        let mut scene_aabbs = Vec::new();

        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos;

            let world_aabb = transform_aabb(&obj.aabb, wt);
            scene_aabbs.push((world_aabb.min, world_aabb.max));

            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - camera_pos;
                let cam_rel_max = world_aabb.max - camera_pos;
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                    [0.0; 3],
                    [0.0; 3],
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        // Upload brick pool if needed
        let pool_data: &[u8] = bytemuck::cast_slice(world.brick_pool().as_slice());
        if !pool_data.is_empty() {
            let gpu_buf = self.gpu_scene.brick_pool_buffer();
            if pool_data.len() as u64 > gpu_buf.size() {
                let new_buf =
                    self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("brick_pool"),
                        size: pool_data.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
                self.gpu_scene
                    .set_brick_pool(&self.ctx.device, new_buf);
            } else {
                self.ctx
                    .queue
                    .write_buffer(gpu_buf, 0, pool_data);
            }
        }

        // Upload brick maps
        let brick_map_data = world.brick_map_alloc().as_slice();
        if !brick_map_data.is_empty() {
            self.gpu_scene
                .upload_brick_maps(&self.ctx.device, &self.ctx.queue, brick_map_data);
        }

        // Upload GPU objects + BVH
        self.gpu_scene
            .upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);
        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene
            .upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Camera uniforms
        let cam_uniforms = self.camera.uniforms(iw, ih, self.frame_index, self.prev_vp);
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene
            .update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // ── Compute environment-derived values ───────────────────────────────

        let env = &self.render_env;
        let sun_dir = env.sun_direction.normalize();

        let sun_color_tinted = env.sun_color * env.sun_intensity;
        let sun_dir_arr = [sun_dir.x, sun_dir.y, sun_dir.z];

        let fog_density = if env.fog_density > 0.0 { env.fog_density } else { 0.0 };
        let fog_alpha = if fog_density > 0.0 { 1.0 } else { 0.0 };

        // Camera basis for shade + vol march
        let fwd = self.camera.forward();
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = iw as f32 / ih as f32;
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;
        let cam = self.camera.position;

        // Synthesize directional sun light.
        let sun_light = Light {
            light_type: 0,
            pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: sun_dir_arr[0],
            dir_y: sun_dir_arr[1],
            dir_z: sun_dir_arr[2],
            color_r: sun_color_tinted.x,
            color_g: sun_color_tinted.y,
            color_b: sun_color_tinted.z,
            intensity: 1.0,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: 1,
        };

        // Camera-relative point/spot lights.
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            if cl.light_type != 0 {
                cl.pos_x -= cam.x;
                cl.pos_y -= cam.y;
                cl.pos_z -= cam.z;
            }
            cl
        }).collect();

        let total_lights = 1 + cam_rel_lights.len() as u32;
        let mut all_lights = vec![sun_light];
        all_lights.extend(cam_rel_lights);
        self.light_buffer.update(&self.ctx.queue, &all_lights);

        // Shade uniforms
        self.shading_pass.update_uniforms(
            &self.ctx.queue,
            &ShadeUniforms {
                debug_mode: self.shade_debug_mode,
                num_lights: total_lights,
                _pad0: 0,
                shadow_budget_k: 0,
                camera_pos: [cam.x, cam.y, cam.z, 0.0],
                sun_dir: [sun_dir_arr[0], sun_dir_arr[1], sun_dir_arr[2], env.sun_intensity],
                sun_color: [env.sun_color.x, env.sun_color.y, env.sun_color.z, 0.0],
                sky_params: [
                    env.rayleigh_scale,
                    env.mie_scale,
                    if env.atmosphere_enabled { 1.0 } else { 0.0 },
                    0.0,
                ],
                cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
                cam_right: [right.x, right.y, right.z, 0.0],
                cam_up: [up.x, up.y, up.z, 0.0],
            },
        );

        // Debug view
        let dm = match self.shade_debug_mode {
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, dm);

        // Store VP for motion vectors
        self.prev_vp = self.camera.view_projection(iw, ih).to_cols_array_2d();

        // Poll device
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        // Build command buffer
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Update coarse field
        if !scene_aabbs.is_empty() {
            self.coarse_field = CoarseField::from_scene_aabbs(
                &self.ctx.device,
                &scene_aabbs,
                COARSE_VOXEL_SIZE,
                1.0,
            );
            self.coarse_field.populate(&scene_aabbs);
        }
        self.coarse_field.upload(&self.ctx.queue, camera_pos);

        self.radiance_volume
            .update_center(&self.ctx.queue, [cam.x, cam.y, cam.z]);

        self.radiance_inject.update_uniforms(
            &self.ctx.queue,
            &InjectUniforms {
                num_lights: total_lights,
                max_shadow_lights: 1,
                _pad: [0; 2],
            },
        );

        // === Core rendering ===
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.ray_march.dispatch(
            &mut encoder,
            &self.gpu_scene,
            &self.gbuffer,
            &self.tile_cull,
            &self.coarse_field,
        );
        self.radiance_inject
            .dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.radiance_mip.dispatch(&mut encoder);
        self.shading_pass.dispatch(
            &mut encoder,
            &self.gbuffer,
            &self.gpu_scene,
            &self.coarse_field,
            &self.radiance_volume,
        );

        // === Volumetrics ===
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &env.cloud_settings,
            self.accumulated_time,
        );
        self.vol_march
            .set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow
            .set_cloud_params(&self.ctx.queue, &cloud_params);

        self.vol_shadow.dispatch(
            &mut encoder,
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
            &self.coarse_field.bind_group,
        );
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);

        let sc = [sun_color_tinted.x, sun_color_tinted.y, sun_color_tinted.z];
        let fc = [env.fog_color.x, env.fog_color.y, env.fog_color.z];
        let half_w = iw / 2;
        let half_h = ih / 2;
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir_arr[0], sun_dir_arr[1], sun_dir_arr[2], 0.0],
            sun_color: [sc[0], sc[1], sc[2], 0.0],
            width: half_w,
            height: half_h,
            full_width: iw,
            full_height: ih,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [fc[0], fc[1], fc[2], fog_alpha],
            fog_height: [fog_density, -0.5, env.fog_height_falloff, 0.0],
            fog_distance: [0.0, 0.01, env.ambient_dust, env.dust_asymmetry],
            frame_index: self.frame_index,
            vol_ambient_color: {
                let va = env.vol_ambient_color * env.vol_ambient_intensity;
                [va.x, va.y, va.z]
            },
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march
            .dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // === Post-processing ===
        // Project sun to screen UV for radial blur god rays.
        {
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir.dot(right) / sun_dot;
                let ndc_y = -sun_dir.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur
                .update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure
            .dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics
            .dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // Blit to target if provided
        if let Some(target) = blit_target {
            self.blit.draw(&mut encoder, target);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
    }
}
