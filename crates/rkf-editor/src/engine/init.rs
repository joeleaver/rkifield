//! EditorEngine resize and coarse field refresh methods.

use glam::Vec3;

use rkf_core::Scene;
use rkf_render::{
    AutoExposurePass, BloomCompositePass, BloomPass, BrushOverlay,
    CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, MotionBlurPass,
    RayMarchPass, RenderContext, ShadingPass, SharpenPass,
    TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::RadianceInjectPass;

use super::EditorEngine;

impl EditorEngine {
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
        self.debug_view = DebugViewPass::new(device, &self.gbuffer);
        self.brush_overlay = BrushOverlay::empty(device);
        let composed = self.shader_composer.compose().to_string();
        self.gpu_color_pool = rkf_render::GpuColorPool::empty(device);
        self.shading_pass = ShadingPass::new(
            device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
            &self.coarse_field, &self.radiance_volume, &self.material_buffer,
            width, height,
            Some(&composed),
            &self.brush_overlay,
            &self.gpu_color_pool,
        );

        // --- Volumetric pipeline (half-res march, full-res upscale) ---
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
            let composed = self.shader_composer.compose().to_string();
            // Preserve existing gpu_color_pool — it may contain imported color data.
            self.shading_pass = ShadingPass::new(
                &self.ctx.device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
                &self.coarse_field, &self.radiance_volume, &self.material_buffer,
                self.render_width, self.render_height,
                Some(&composed),
                &self.brush_overlay,
                &self.gpu_color_pool,
            );
            self.vol_shadow = VolShadowPass::new(
                &self.ctx.device, &self.ctx.queue, &self.coarse_field.bind_group_layout,
            );
            log::info!("Coarse field resized for {} objects", scene.objects.len());
        }
    }
}
