//! Resize operations — `resize_render`, `resize_viewport`.

use rkf_render::{
    AutoExposurePass, BloomCompositePass, BloomPass, ColorGradePass, CosmeticsPass,
    DebugViewPass, DofPass, GBuffer, GodRaysBlurPass, MotionBlurPass, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolUpscalePass,
};

use super::helpers::{create_offscreen_target, create_readback_buffer};
use super::Renderer;

impl Renderer {
    // ── Resize ────────────────────────────────────────────────────────────

    /// Resize the internal render resolution.
    ///
    /// Recreates all size-dependent GPU resources.
    pub fn resize_render(&mut self, width: u32, height: u32) {
        let width = width.max(64);
        let height = height.max(64);
        if width == self.internal_width && height == self.internal_height {
            return;
        }
        log::info!(
            "Render resolution: {}x{} -> {}x{}",
            self.internal_width,
            self.internal_height,
            width,
            height
        );
        self.internal_width = width;
        self.internal_height = height;

        let device = &self.ctx.device;

        // Core passes
        self.gbuffer = GBuffer::new(device, width, height);
        self.tile_cull = TileObjectCullPass::new(device, &self.gpu_scene, width, height);
        self.debug_view = DebugViewPass::new(device, &self.gbuffer);
        self.brush_overlay = rkf_render::BrushOverlay::empty(device);
        self.gpu_color_pool = rkf_render::GpuColorPool::empty(device);
        self.shading_pass = ShadingPass::new(
            device,
            &self.gbuffer,
            &self.gpu_scene,
            &self.light_buffer,
            &self.coarse_field,
            &self.radiance_volume,
            &self.material_table.buffer,
            width,
            height,
            None,
            &self.brush_overlay,
            &self.gpu_color_pool,
        );

        // Volumetric pipeline
        let half_w = width / 2;
        let half_h = height / 2;
        self.vol_march = VolMarchPass::new(
            device,
            &self.ctx.queue,
            &self.gbuffer.position_view,
            &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w,
            half_h,
            width,
            height,
        );
        self.vol_upscale = VolUpscalePass::new(
            device,
            &self.vol_march.output_view,
            &self.gbuffer.position_view,
            width,
            height,
            half_w,
            half_h,
        );
        self.vol_composite = VolCompositePass::new(
            device,
            &self.shading_pass.hdr_view,
            &self.vol_upscale.output_view,
            width,
            height,
        );

        // Post-processing pipeline
        self.god_rays_blur = GodRaysBlurPass::new(
            device,
            &self.vol_composite.output_view,
            &self.gbuffer.position_view,
            width,
            height,
        );
        self.bloom = BloomPass::new(device, &self.god_rays_blur.output_view, width, height);
        self.auto_exposure =
            AutoExposurePass::new(device, &self.god_rays_blur.output_view, width, height);
        self.dof = DofPass::new(
            device,
            &self.god_rays_blur.output_view,
            &self.gbuffer.position_view,
            width,
            height,
        );
        self.motion_blur = MotionBlurPass::new(
            device,
            &self.dof.output_view,
            &self.gbuffer.motion_view,
            width,
            height,
        );
        self.bloom_composite = BloomCompositePass::new(
            device,
            &self.motion_blur.output_view,
            self.bloom.mip_views(),
            width,
            height,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            device,
            &self.bloom_composite.output_view,
            width,
            height,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            device,
            &self.ctx.queue,
            &self.tone_map.ldr_view,
            width,
            height,
        );
        self.cosmetics = CosmeticsPass::new(device, &self.color_grade.output_view, width, height);
        self.sharpen = SharpenPass::new(
            device,
            &self.vol_composite.output_view,
            &self.gbuffer,
            width,
            height,
        );

        // Update blit source
        self.blit.update_source(device, &self.cosmetics.output_view);
        if let Some(ref mut os_blit) = self.offscreen_blit {
            os_blit.update_source(device, &self.cosmetics.output_view);
        }

        // Re-apply post-processing settings
        let env = self.render_env.clone();
        self.apply_post_process_settings(&env);
    }

    /// Resize the viewport (offscreen render target) and internal resolution.
    ///
    /// Called when the viewport dimensions change (e.g., panel resize).
    /// Returns the new offscreen `TextureView` for the compositor to use,
    /// or `None` if the viewport didn't actually change.
    pub fn resize_viewport(
        &mut self,
        viewport_w: u32,
        viewport_h: u32,
        render_scale: f32,
    ) -> Option<wgpu::TextureView> {
        let vp_w = viewport_w.max(64);
        let vp_h = viewport_h.max(64);
        if vp_w == self.display_width && vp_h == self.display_height {
            return None;
        }

        log::info!(
            "Viewport resize: {}x{} -> {}x{}",
            self.display_width,
            self.display_height,
            vp_w,
            vp_h
        );
        self.display_width = vp_w;
        self.display_height = vp_h;

        // Recreate offscreen target.
        let (tex, view) = create_offscreen_target(&self.ctx.device, vp_w, vp_h);
        self.offscreen_texture = Some(tex);
        self.offscreen_view = Some(view);

        // Recreate readback buffer at new viewport resolution.
        self.readback_buffer = create_readback_buffer(&self.ctx.device, vp_w, vp_h);

        // Resize internal render resolution.
        let internal_w = ((vp_w as f32 * render_scale) as u32).max(64);
        let internal_h = ((vp_h as f32 * render_scale) as u32).max(64);
        self.resize_render(internal_w, internal_h);

        // Return new view for compositor.
        self.offscreen_view.clone()
    }
}
