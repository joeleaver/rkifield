//! Offscreen rendering path for the editor engine (compositor/rinch integration).

use std::sync::{Arc, Mutex};
use glam::Vec3;
use rkf_core::{Aabb, Scene};
use rkf_core::transform_flatten::flatten_object;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuObject, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::{RadianceInjectPass, InjectUniforms};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::{MaterialTable, create_test_materials};
use super::EditorEngine;
use super::{RENDER_SCALE, OFFSCREEN_FORMAT, build_demo_scene};
use crate::automation::SharedState;

impl EditorEngine {
    /// Create an engine using a shared wgpu device (from rinch's `GpuHandle`).
    ///
    /// Renders to an offscreen texture instead of a swapchain surface. The
    /// compositor reads the offscreen texture directly via `TextureView` —
    /// zero-copy GPU compositing.
    ///
    /// Returns `(engine, scene)` — the caller stores the scene in EditorState.
    pub fn new_with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        viewport_width: u32,
        viewport_height: u32,
        shared_state: Arc<Mutex<SharedState>>,
    ) -> (Self, Scene) {
        let ctx = RenderContext::from_shared(device, queue);

        // Compute internal render resolution from viewport size.
        let vp_w = viewport_width.max(64);
        let vp_h = viewport_height.max(64);
        let internal_w = ((vp_w as f32 * RENDER_SCALE) as u32).max(64);
        let internal_h = ((vp_h as f32 * RENDER_SCALE) as u32).max(64);

        // Build demo scene.
        let demo = build_demo_scene();
        let scene = demo.scene;

        // Upload brick pool to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(demo.brick_pool.as_slice());
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: pool_data.len().max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !pool_data.is_empty() {
            ctx.queue.write_buffer(&brick_pool_buffer, 0, pool_data);
        }

        let mut gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);
        let brick_map_data = demo.brick_map_alloc.as_slice();
        if !brick_map_data.is_empty() {
            gpu_scene.upload_brick_maps(&ctx.device, &ctx.queue, brick_map_data);
        }

        let gbuffer = GBuffer::new(&ctx.device, internal_w, internal_h);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, internal_w, internal_h,
        );

        // Coarse acceleration field.
        let init_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let init_default_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = init_transforms.get(&obj.id).unwrap_or(&init_default_wt);
                let smin = obj.aabb.min * wt.scale;
                let smax = obj.aabb.max * wt.scale;
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
                (wmin, wmax)
            })
            .collect();
        let mut coarse_field = CoarseField::from_scene_aabbs(
            &ctx.device, &scene_aabbs, COARSE_VOXEL_SIZE, 1.0,
        );
        coarse_field.populate(&scene_aabbs);
        coarse_field.upload(&ctx.queue, Vec3::ZERO);

        let ray_march = RayMarchPass::new(
            &ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field,
        );
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material table.
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights.
        let world_lights = vec![
            Light::point([2.0, 1.5, -1.0], [1.0, 0.8, 0.5], 5.0, 8.0, true),
            Light::point([-2.0, 1.0, -3.0], [0.5, 0.7, 1.0], 3.0, 6.0, false),
        ];
        let mut init_lights = Vec::with_capacity(64);
        init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        init_lights.extend(&world_lights);
        while init_lights.len() < 64 {
            init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        }
        let light_buffer = LightBuffer::upload(&ctx.device, &init_lights);

        // GI radiance volume.
        let radiance_volume = RadianceVolume::new(&ctx.device);
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device, &gpu_scene, &material_table.buffer,
            &light_buffer, &radiance_volume, &coarse_field,
        );
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Shading pass.
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            internal_w, internal_h,
        );

        // Volumetric pipeline.
        let vol_shadow = VolShadowPass::new(
            &ctx.device, &ctx.queue, &coarse_field.bind_group_layout,
        );
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = internal_w / 2;
        let half_h = internal_h / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device, &ctx.queue,
            &gbuffer.position_view, &vol_shadow.shadow_view, &cloud_shadow.shadow_view,
            half_w, half_h, internal_w, internal_h,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device, &vol_march.output_view, &gbuffer.position_view,
            internal_w, internal_h, half_w, half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device, &shading_pass.hdr_view, &vol_upscale.output_view,
            internal_w, internal_h,
        );

        // Post-processing pipeline.
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
            internal_w, internal_h,
        );
        let bloom = BloomPass::new(
            &ctx.device, &god_rays_blur.output_view, internal_w, internal_h,
        );
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &god_rays_blur.output_view, internal_w, internal_h,
        );
        let dof = DofPass::new(
            &ctx.device, &god_rays_blur.output_view, &gbuffer.position_view,
            internal_w, internal_h,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device, &dof.output_view, &gbuffer.motion_view,
            internal_w, internal_h,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device, &motion_blur.output_view, bloom.mip_views(),
            internal_w, internal_h,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device, &bloom_composite.output_view,
            internal_w, internal_h,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device, &ctx.queue, &tone_map.ldr_view,
            internal_w, internal_h,
        );
        let cosmetics = CosmeticsPass::new(
            &ctx.device, &color_grade.output_view, internal_w, internal_h,
        );
        let sharpen = SharpenPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer,
            internal_w, internal_h,
        );

        // Offscreen render target at viewport resolution.
        let (offscreen_tex, offscreen_view) = Self::create_offscreen_target(
            &ctx.device, vp_w, vp_h,
        );

        // Blit from post-processed output (internal res) to offscreen target
        // (viewport res). The BlitPass uses bilinear sampling for upscale.
        let offscreen_blit = BlitPass::new(
            &ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT,
        );

        // Legacy blit (unused in offscreen path, but struct requires it).
        let blit = BlitPass::new(
            &ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT,
        );

        // Wireframe pass for selection highlights and gizmos (targets offscreen texture).
        let wireframe_pass = crate::wireframe::WireframePass::new(&ctx.device, OFFSCREEN_FORMAT);
        // Render camera.
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Pick readback buffer.
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback buffer (Rgba32Float position = 16 bytes, 256-aligned).
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Screenshot readback at viewport resolution.
        let readback_buffer = Self::create_readback_buffer(
            &ctx.device, vp_w, vp_h,
        );

        let jfa_sdf = crate::jfa_sdf::JfaSdfPass::new(&ctx.device);
        let eikonal_repair = crate::eikonal_repair::EikonalRepairPass::new(&ctx.device);

        let engine = Self {
            ctx,
            surface: None,
            surface_format: OFFSCREEN_FORMAT,
            offscreen_texture: Some(offscreen_tex),
            offscreen_view: Some(offscreen_view),
            offscreen_blit: Some(offscreen_blit),
            viewport_width: vp_w,
            viewport_height: vp_h,
            gpu_scene,
            gbuffer,
            tile_cull,
            coarse_field,
            ray_march,
            debug_view,
            shading_pass,
            radiance_volume,
            radiance_inject,
            radiance_mip,
            vol_shadow,
            cloud_shadow,
            vol_march,
            vol_upscale,
            vol_composite,
            god_rays_blur,
            bloom,
            auto_exposure,
            dof,
            motion_blur,
            bloom_composite,
            tone_map,
            color_grade,
            cosmetics,
            sharpen,
            blit,
            camera,
            world_lights,
            light_buffer,
            material_buffer: material_table.buffer,
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            env_sun_dir: {
                let d = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
                [d.x, d.y, d.z]
            },
            env_sun_color: [3.0, 2.85, 2.55],
            env_fog_color: [0.7, 0.75, 0.8],
            env_fog_density: 0.0,
            env_fog_height_falloff: 0.1,
            env_sun_intensity: 3.0,
            env_sun_color_raw: [1.0, 0.95, 0.85],
            env_rayleigh_scale: 1.0,
            env_mie_scale: 1.0,
            env_atmosphere_enabled: true,
            env_ambient_dust: 0.005,
            env_dust_g: 0.3,
            env_cloud_settings: rkf_render::CloudSettings::default(),
            accumulated_time: 0.0,
            render_width: internal_w,
            render_height: internal_h,
            pick_readback_buffer,
            brush_readback_buffer,
            readback_buffer,
            window_width: vp_w,
            window_height: vp_h,
            shared_state,
            wireframe_pass: Some(wireframe_pass),
            character: Some(demo.character),
            character_obj_index: Some(demo.character_obj_index),
            last_frame_time: std::time::Instant::now(),
            cpu_brick_pool: demo.brick_pool,
            cpu_brick_map_alloc: demo.brick_map_alloc,
            jfa_sdf,
            eikonal_repair,
        };
        (engine, scene)
    }

    /// Create the offscreen render target texture at the given resolution.
    pub(super) fn create_offscreen_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// Get the offscreen texture view for compositor integration.
    ///
    /// Returns `None` if the engine is using the surface-based path.
    /// The compositor calls `set_texture_source()` with this view.
    pub fn offscreen_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.offscreen_view.as_ref()
    }

    /// Current viewport dimensions (physical pixels).
    pub fn viewport_size(&self) -> (u32, u32) {
        (self.viewport_width, self.viewport_height)
    }

    /// Resize the viewport and recreate resolution-dependent resources.
    ///
    /// Called when the `RenderSurface` layout changes (e.g., panel resize).
    /// Recreates the offscreen texture, internal render resources, and
    /// readback buffer. Returns the new offscreen `TextureView` for the
    /// compositor to call `set_texture_source()`.
    pub fn resize_viewport(&mut self, viewport_w: u32, viewport_h: u32) -> Option<wgpu::TextureView> {
        let vp_w = viewport_w.max(64);
        let vp_h = viewport_h.max(64);
        if vp_w == self.viewport_width && vp_h == self.viewport_height {
            return None;
        }

        log::info!("Viewport resize: {}x{} → {}x{}", self.viewport_width, self.viewport_height, vp_w, vp_h);
        self.viewport_width = vp_w;
        self.viewport_height = vp_h;

        // Recreate offscreen target at new viewport resolution.
        let (tex, view) = Self::create_offscreen_target(&self.ctx.device, vp_w, vp_h);
        self.offscreen_texture = Some(tex);
        self.offscreen_view = Some(view);

        // Recreate readback buffer at new viewport resolution.
        self.readback_buffer = Self::create_readback_buffer(&self.ctx.device, vp_w, vp_h);
        self.window_width = vp_w;
        self.window_height = vp_h;

        // Resize internal render resolution.
        let internal_w = ((vp_w as f32 * RENDER_SCALE) as u32).max(64);
        let internal_h = ((vp_h as f32 * RENDER_SCALE) as u32).max(64);
        self.resize_render(internal_w, internal_h);

        // Update offscreen blit source (cosmetics output may have changed).
        if let Some(ref mut offscreen_blit) = self.offscreen_blit {
            offscreen_blit.update_source(&self.ctx.device, &self.cosmetics.output_view);
        }

        // Update SharedState.
        if let Ok(mut state) = self.shared_state.lock() {
            state.frame_width = vp_w;
            state.frame_height = vp_h;
        }

        // Return new view for set_texture_source().
        self.offscreen_view.clone()
    }

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
            let local_aabb = obj.aabb;
            let world_aabb = {
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
                Aabb::new(wmin, wmax)
            };
            world_aabbs_for_coarse.push((world_aabb.min, world_aabb.max));
            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - self.camera.position;
                let cam_rel_max = world_aabb.max - self.camera.position;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                gpu_objects.push(GpuObject::from_flat_node(
                    flat, obj.id,
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

        self.coarse_field.populate(&world_aabbs_for_coarse);
        self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

        // Camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Synthesize directional light from environment sun.
        let sun_light = Light {
            light_type: 0, pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: self.env_sun_dir[0], dir_y: self.env_sun_dir[1], dir_z: self.env_sun_dir[2],
            color_r: self.env_sun_color[0], color_g: self.env_sun_color[1], color_b: self.env_sun_color[2],
            intensity: 1.0, range: 0.0, inner_angle: 0.0, outer_angle: 0.0,
            cookie_index: -1, shadow_caster: 1,
        };

        let cam = self.camera.position;
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            if cl.light_type != 0 { cl.pos_x -= cam.x; cl.pos_y -= cam.y; cl.pos_z -= cam.z; }
            cl
        }).collect();

        let total_lights = 1 + cam_rel_lights.len() as u32;
        let mut all_lights = vec![sun_light];
        all_lights.extend(cam_rel_lights);
        self.light_buffer.update(&self.ctx.queue, &all_lights);

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

        // Poll GPU to prevent command buffer buildup.
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_frame"),
        });

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

        // --- Volumetric pipeline ---
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

        self.vol_march.set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow.set_cloud_params(&self.ctx.queue, &cloud_params);

        let sun_dir = self.env_sun_dir;
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
        );
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);

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
            _pad0: 0, _pad1: 0, _pad2: 0,
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

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
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // --- Blit to offscreen render target ---
        let offscreen_view = self.offscreen_view.as_ref()
            .expect("offscreen_view must exist in offscreen mode");
        if let Some(ref offscreen_blit) = self.offscreen_blit {
            offscreen_blit.draw(&mut encoder, offscreen_view);
        }

        // Submit all GPU work. The compositor reads the offscreen texture
        // on the next paint — no readback needed per frame.
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

            // Read position (4×f32 = 16 bytes).
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

        // --- Screenshot readback (on demand only) ---
        let do_readback = self.shared_state.lock()
            .map(|s| s.screenshot_requested)
            .unwrap_or(false);

        if do_readback {
            let w = self.viewport_width;
            let h = self.viewport_height;
            let bytes_per_pixel = 4u32;
            let unpadded_row = w * bytes_per_pixel;
            let padded_row = (unpadded_row + 255) & !255;

            let offscreen_tex = self.offscreen_texture.as_ref()
                .expect("offscreen_texture must exist");
            let mut readback_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("screenshot_readback") },
            );
            readback_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: offscreen_tex,
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
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(readback_enc.finish()));

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

                for y in 0..h as usize {
                    let src_row_offset = y * padded_row as usize;
                    let dst_row_offset = y * w as usize * 4;
                    let row_bytes = w as usize * 4;
                    rgba8[dst_row_offset..dst_row_offset + row_bytes]
                        .copy_from_slice(&data[src_row_offset..src_row_offset + row_bytes]);
                }

                // Offscreen format is RGBA — no BGRA swap needed.
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

        self.frame_index += 1;
    }
}
