//! EditorEngine constructor and resize methods.

use std::sync::{Arc, Mutex};

use glam::Vec3;
use winit::window::Window;

use rkf_core::Scene;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShadingPass, SharpenPass,
    TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::{MaterialTable, create_test_materials};

use super::{
    EditorEngine, INTERNAL_WIDTH, INTERNAL_HEIGHT, build_demo_scene,
};
use crate::automation::SharedState;

impl EditorEngine {
    /// Initialize the render engine with wgpu, all passes, and a demo scene.
    ///
    /// Returns `(engine, scene)` — the caller stores the scene in EditorState.
    pub fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> (Self, Scene) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let ctx = RenderContext::new(&instance, &surface);

        let size = window.inner_size();
        // Configure surface with COPY_SRC so we can read back the composited
        // frame (engine + UI overlay) for MCP screenshots.
        let surface_format = Self::configure_surface_copy(
            &ctx, &surface, size.width, size.height,
        );

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

        // Upload brick maps.
        let brick_map_data = demo.brick_map_alloc.as_slice();
        if !brick_map_data.is_empty() {
            gpu_scene.upload_brick_maps(&ctx.device, &ctx.queue, brick_map_data);
        }

        let gbuffer = GBuffer::new(&ctx.device, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Coarse acceleration field — compute world-space AABBs from local AABBs + baked transforms.
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
        log::info!(
            "Coarse field: {}x{}x{} cells, voxel_size={}m",
            coarse_field.dims.x, coarse_field.dims.y, coarse_field.dims.z,
            coarse_field.voxel_size,
        );

        let ray_march = RayMarchPass::new(
            &ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field,
        );
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material table.
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);
        log::info!("Material table: {} materials uploaded", material_table.count);

        // Lights (point/spot only — directional sun is synthesized from environment each frame).
        let world_lights = vec![
            Light::point([2.0, 1.5, -1.0], [1.0, 0.8, 0.5], 5.0, 8.0, true),
            Light::point([-2.0, 1.0, -3.0], [0.5, 0.7, 1.0], 3.0, 6.0, false),
        ];
        // Allocate light buffer with headroom (1 sun + up to 63 point/spot).
        // Slot 0 is reserved for the sun light synthesized from environment each frame.
        let mut init_lights = Vec::with_capacity(64);
        init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false)); // placeholder for sun
        init_lights.extend(&world_lights);
        // Pad to 64 lights so the buffer can accommodate runtime additions.
        while init_lights.len() < 64 {
            init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        }
        let light_buffer = LightBuffer::upload(&ctx.device, &init_lights);
        log::info!("Lights: {} point/spot + 1 env sun", world_lights.len());

        // Radiance volume for GI.
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
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Volumetric pipeline.
        let vol_shadow = VolShadowPass::new(
            &ctx.device, &ctx.queue, &coarse_field.bind_group_layout,
        );
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = INTERNAL_WIDTH / 2;
        let half_h = INTERNAL_HEIGHT / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device, &ctx.queue,
            &gbuffer.position_view, &vol_shadow.shadow_view, &cloud_shadow.shadow_view,
            half_w, half_h, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device, &vol_march.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT, half_w, half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device, &shading_pass.hdr_view, &vol_upscale.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Post-processing pipeline.
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let bloom = BloomPass::new(
            &ctx.device, &god_rays_blur.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &god_rays_blur.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let dof = DofPass::new(
            &ctx.device, &god_rays_blur.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device, &dof.output_view, &gbuffer.motion_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device, &motion_blur.output_view, bloom.mip_views(),
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device, &bloom_composite.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device, &ctx.queue, &tone_map.ldr_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let cosmetics = CosmeticsPass::new(
            &ctx.device, &color_grade.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let sharpen = SharpenPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Blit to swapchain.
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, surface_format);

        // Render camera (synced from EditorCamera each frame).
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Pick readback buffer — 1 pixel from the R32Uint material texture.
        // bytes_per_row must be aligned to 256, so buffer is 256 bytes.
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback buffer — 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
        // bytes_per_row must be 256-aligned, so buffer is 256 bytes.
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Readback buffer for MCP screenshots — at window resolution so we
        // capture the composited output (engine viewport + rinch UI panels).
        let window_width = size.width.max(1);
        let window_height = size.height.max(1);
        let readback_buffers = Self::create_readback_buffers(
            &ctx.device, window_width, window_height,
        );

        let jfa_sdf = crate::jfa_sdf::JfaSdfPass::new(&ctx.device);
        let eikonal_repair = crate::eikonal_repair::EikonalRepairPass::new(&ctx.device);
        let gpu_profiler = rkf_render::gpu_profiler::GpuProfiler::new(&ctx.device, &ctx.queue);

        let engine = Self {
            ctx,
            surface: Some(surface),
            surface_format,
            offscreen_texture: None,
            offscreen_view: None,
            offscreen_blit: None,
            viewport_width: 0,
            viewport_height: 0,
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
            // Matches EnvironmentState defaults; first frame apply_environment
            // overwrites these with actual data model values.
            env_sun_dir: {
                let d = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
                [d.x, d.y, d.z]
            },
            env_sun_color: [3.0, 2.85, 2.55], // sun_color * sun_intensity ([1.0,0.95,0.85] * 3.0)
            env_fog_color: [0.7, 0.75, 0.8],
            env_fog_density: 0.0, // fog disabled by default
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
            render_width: INTERNAL_WIDTH,
            render_height: INTERNAL_HEIGHT,
            pick_readback_buffer,
            brush_readback_buffer,
            readback_buffers,
            readback_parity: 0,
            readback_pending: None,
            prev_frame_pixels: None,
            window_width,
            window_height,
            shared_state,
            wireframe_pass: None,
            cpu_brick_pool: demo.brick_pool,
            cpu_brick_map_alloc: demo.brick_map_alloc,
            jfa_sdf,
            eikonal_repair,
            // Incremental update cache — first frame triggers full rebuild.
            cached_gpu_objects: Vec::new(),
            cached_bvh: None,
            cached_bvh_pairs: Vec::new(),
            cached_world_aabbs: Vec::new(),
            object_gpu_ranges: std::collections::HashMap::new(),
            dirty_objects: std::collections::HashSet::new(),
            topology_changed: true,
            lights_dirty: true,
            last_light_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            pending_wireframe: Vec::new(),
            gpu_profiler,
            last_vol_shadow_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            last_vol_shadow_sun_dir: [f32::NAN; 3],
        };
        (engine, scene)
    }

    /// Configure the surface with COPY_SRC for screenshot readback.
    pub(super) fn configure_surface_copy(
        ctx: &RenderContext,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let adapter = ctx.adapter.as_ref()
            .expect("configure_surface_copy requires an adapter");
        let caps = surface.get_capabilities(adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctx.device, &config);

        log::info!("Surface configured: {width}x{height}, format={format:?} (COPY_SRC enabled)");
        format
    }

    /// Create a pair of readback buffers for double-buffered async readback.
    pub(super) fn create_readback_buffers(device: &wgpu::Device, width: u32, height: u32) -> [wgpu::Buffer; 2] {
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;
        let size = (padded_row * height) as u64;
        [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback_0"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback_1"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        ]
    }

    /// Reconfigure the surface for a new window size (surface-based path only).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            if let Some(ref surface) = self.surface {
                Self::configure_surface_copy(&self.ctx, surface, width, height);
            }
            self.window_width = width;
            self.window_height = height;
            self.readback_buffers = Self::create_readback_buffers(
                &self.ctx.device, width, height,
            );
            self.readback_pending = None;
            self.prev_frame_pixels = None;
            // Update SharedState dimensions for MCP screenshot encoding.
            if let Ok(mut state) = self.shared_state.lock() {
                state.frame_width = width;
                state.frame_height = height;
            }
        }
    }

    /// Resize the internal render resolution.
    ///
    /// Recreates all size-dependent GPU resources: GBuffer, tile cull, shading,
    /// volumetric pipeline, post-processing, and blit pass. Called when the
    /// viewport dimensions change (window resize or panel layout change).
    pub fn resize_render(&mut self, width: u32, height: u32) {
        let width = width.max(64);
        let height = height.max(64);
        if width == self.render_width && height == self.render_height {
            return;
        }
        log::info!("Render resolution: {}x{} → {}x{}", self.render_width, self.render_height, width, height);
        self.render_width = width;
        self.render_height = height;

        let device = &self.ctx.device;

        // --- Core passes ---
        self.gbuffer = GBuffer::new(device, width, height);
        self.tile_cull = TileObjectCullPass::new(device, &self.gpu_scene, width, height);
        // RayMarchPass pipeline is resolution-agnostic (dispatch from gbuffer dims).
        // DebugViewPass stores bind groups to gbuffer.
        self.debug_view = DebugViewPass::new(device, &self.gbuffer);
        self.shading_pass = ShadingPass::new(
            device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
            &self.coarse_field, &self.radiance_volume, &self.material_buffer,
            width, height,
        );

        // --- Volumetric pipeline (half-res march, full-res upscale) ---
        // vol_shadow and cloud_shadow are resolution-independent.
        let half_w = width / 2;
        let half_h = height / 2;
        self.vol_march = VolMarchPass::new(
            device, &self.ctx.queue,
            &self.gbuffer.position_view, &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w, half_h, width, height,
        );
        self.vol_upscale = VolUpscalePass::new(
            device, &self.vol_march.output_view, &self.gbuffer.position_view,
            width, height, half_w, half_h,
        );
        self.vol_composite = VolCompositePass::new(
            device, &self.shading_pass.hdr_view, &self.vol_upscale.output_view,
            width, height,
        );

        // --- Post-processing pipeline ---
        self.god_rays_blur = GodRaysBlurPass::new(
            device, &self.vol_composite.output_view, &self.gbuffer.position_view,
            width, height,
        );
        self.bloom = BloomPass::new(
            device, &self.god_rays_blur.output_view, width, height,
        );
        self.auto_exposure = AutoExposurePass::new(
            device, &self.god_rays_blur.output_view, width, height,
        );
        self.dof = DofPass::new(
            device, &self.god_rays_blur.output_view, &self.gbuffer.position_view,
            width, height,
        );
        self.motion_blur = MotionBlurPass::new(
            device, &self.dof.output_view, &self.gbuffer.motion_view,
            width, height,
        );
        self.bloom_composite = BloomCompositePass::new(
            device, &self.motion_blur.output_view, self.bloom.mip_views(),
            width, height,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            device, &self.bloom_composite.output_view,
            width, height,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            device, &self.ctx.queue, &self.tone_map.ldr_view,
            width, height,
        );
        self.cosmetics = CosmeticsPass::new(
            device, &self.color_grade.output_view, width, height,
        );
        self.sharpen = SharpenPass::new(
            device, &self.vol_composite.output_view, &self.gbuffer,
            width, height,
        );

        // --- Blit (update source view to new cosmetics output) ---
        self.blit.update_source(device, &self.cosmetics.output_view);
    }

    /// Refresh the coarse acceleration field from current scene AABBs.
    ///
    /// If the scene still fits within the existing coarse field bounds, this
    /// only repopulates the CPU data and re-uploads the texture — no GPU
    /// pipeline or bind group rebuilds needed.  If the scene has grown beyond
    /// the current bounds, the field is reallocated and passes are rebuilt.
    pub fn refresh_coarse_field(&mut self, scene: &Scene) {
        let wts = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let def_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = wts.get(&obj.id).unwrap_or(&def_wt);
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

        // Check if current coarse field covers the scene (with margin).
        let margin = 1.0_f32;
        let (scene_min, scene_max) = if scene_aabbs.is_empty() {
            (Vec3::ZERO, Vec3::splat(self.coarse_field.voxel_size))
        } else {
            let mut lo = Vec3::splat(f32::MAX);
            let mut hi = Vec3::splat(f32::MIN);
            for (amin, amax) in &scene_aabbs {
                lo = lo.min(*amin);
                hi = hi.max(*amax);
            }
            (lo, hi)
        };

        let needed_min = scene_min - Vec3::splat(margin);
        let needed_max = scene_max + Vec3::splat(margin);
        let current_max = self.coarse_field.origin + Vec3::new(
            self.coarse_field.dims.x as f32,
            self.coarse_field.dims.y as f32,
            self.coarse_field.dims.z as f32,
        ) * self.coarse_field.voxel_size;

        let fits = needed_min.cmpge(self.coarse_field.origin).all()
            && needed_max.cmple(current_max).all();

        if fits {
            // Fast path: repopulate + upload, same texture — bind groups stay valid.
            self.coarse_field.populate(&scene_aabbs);
            self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);
        } else {
            // Scene outgrew the field — reallocate texture and rebuild passes.
            self.coarse_field = CoarseField::from_scene_aabbs(
                &self.ctx.device, &scene_aabbs, COARSE_VOXEL_SIZE, margin,
            );
            self.coarse_field.populate(&scene_aabbs);
            self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

            self.ray_march = RayMarchPass::new(
                &self.ctx.device, &self.gpu_scene, &self.gbuffer,
                &self.tile_cull, &self.coarse_field,
            );
            self.radiance_inject = RadianceInjectPass::new(
                &self.ctx.device, &self.gpu_scene, &self.material_buffer,
                &self.light_buffer, &self.radiance_volume, &self.coarse_field,
            );
            self.shading_pass = ShadingPass::new(
                &self.ctx.device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
                &self.coarse_field, &self.radiance_volume, &self.material_buffer,
                self.render_width, self.render_height,
            );
            self.vol_shadow = VolShadowPass::new(
                &self.ctx.device, &self.ctx.queue, &self.coarse_field.bind_group_layout,
            );
            log::info!("Coarse field resized for {} objects", scene.objects.len());
        }
    }

    /// Fill 1 layer of SDF padding bricks around each voxelized object.
    ///
    /// At load time, narrow-band bricks go straight to EMPTY_SLOT with no
    /// transition. This causes gradient discontinuities (faceted normals) at
    /// the narrow-band boundary because `sample_voxel_at` returns a constant
    /// for EMPTY_SLOT bricks. This method allocates padding bricks and fills
    /// them with properly trilinear-sampled SDF values, making the gradient
    /// smooth across the boundary.
    pub fn init_sdf_padding(&mut self, scene: &mut Scene) {
        use rkf_core::SdfSource;
        use glam::Vec3;

        for obj in &mut scene.objects {
            let (handle, voxel_size, aabb_min) = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                    (*brick_map_handle, *voxel_size, aabb.min)
                }
                _ => continue,
            };

            // Mark interior empties first (needed for correct sign in padding).
            self.mark_interior_empties(&handle);

            // Create a fake EditOp covering the entire object so
            // ensure_sdf_consistency fills all boundary bricks.
            let half = Vec3::new(
                handle.dims.x as f32 * voxel_size * 4.0,
                handle.dims.y as f32 * voxel_size * 4.0,
                handle.dims.z as f32 * voxel_size * 4.0,
            );
            let fake_op = rkf_edit::edit_op::EditOp {
                object_id: obj.id,
                position: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                edit_type: rkf_edit::types::EditType::SmoothUnion,
                shape_type: rkf_edit::types::ShapeType::Sphere,
                dimensions: half,
                strength: 1.0,
                blend_k: 0.0,
                falloff: rkf_edit::types::FalloffCurve::Smooth,
                material_id: 0,
                secondary_id: 0,
                color_packed: 0,
            };

            let slots = self.ensure_sdf_consistency(&handle, voxel_size, aabb_min, &fake_op);
            if !slots.is_empty() {
                log::info!(
                    "init_sdf_padding: filled {} padding bricks for '{}'",
                    slots.len(), obj.name,
                );
            }
        }

        // Reupload all brick data + maps after padding.
        self.reupload_brick_data();
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }
    }
}
