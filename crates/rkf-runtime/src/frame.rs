//! Frame scheduling for the RKIField render pipeline.
//!
//! This module defines the static pass ordering for each frame and provides
//! [`FrameSettings`] to gate optional passes, [`FrameContext`] to bundle all
//! per-frame resources, and [`execute_frame`] to drive the full pipeline in the
//! correct dependency order.

use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, ClipmapGpuData, CloudShadowPass,
    ColorGradePass, CosmeticsPass, DofPass, GBuffer, GpuColorPool, GpuScene, MaterialTable,
    LightBuffer, MotionBlurPass, RadianceInjectPass, RadianceMipPass, RadianceVolume,
    RayMarchPass, ShadingPass, SharpenPass, TileCullPass, ToneMapPass, UpscalePass,
    VolCompositePass, VolMarchParams, VolMarchPass, VolShadowPass, VolTemporalPass,
    VolUpscalePass,
};

/// Controls which optional render passes are enabled for a frame.
///
/// All fields default to `true` — every pass is active unless explicitly
/// disabled. Pass individual fields to `false` to skip that subsystem.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSettings {
    /// Enable volumetric fog passes (shadow map, march, temporal, upscale, composite).
    pub volumetrics_enabled: bool,
    /// Enable cloud shadow map pass.
    pub cloud_shadows_enabled: bool,
    /// Enable global illumination (radiance inject + mip gen).
    pub gi_enabled: bool,
    /// Enable depth of field pass.
    pub dof_enabled: bool,
    /// Enable motion blur pass.
    pub motion_blur_enabled: bool,
    /// Enable bloom passes (extract/blur pre-upscale + composite post-upscale).
    pub bloom_enabled: bool,
    /// Enable automatic exposure adaptation.
    pub auto_exposure_enabled: bool,
    /// Enable color grading LUT pass.
    pub color_grade_enabled: bool,
    /// Enable cosmetic effects (vignette, grain, chromatic aberration).
    pub cosmetics_enabled: bool,
    /// Enable edge-aware sharpening (only meaningful when using the custom upscaler).
    pub sharpen_enabled: bool,
}

impl Default for FrameSettings {
    fn default() -> Self {
        Self {
            volumetrics_enabled: true,
            cloud_shadows_enabled: true,
            gi_enabled: true,
            dof_enabled: true,
            motion_blur_enabled: true,
            bloom_enabled: true,
            auto_exposure_enabled: true,
            color_grade_enabled: true,
            cosmetics_enabled: true,
            sharpen_enabled: true,
        }
    }
}

/// All resources required to execute a single render frame.
///
/// This is a large struct, but it is always passed by mutable reference.
/// It simply aggregates references to the render pass objects and GPU resources
/// that are owned elsewhere (typically by the engine's `RenderState`), avoiding
/// any ownership transfer.
///
/// `vol_temporal` requires `&mut` because its `dispatch` performs a ping-pong
/// swap of internal history textures.
pub struct FrameContext<'a> {
    /// Active GPU command encoder for the frame.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// GPU queue for buffer uploads performed during dispatch.
    pub queue: &'a wgpu::Queue,
    /// Per-frame feature toggles.
    pub settings: &'a FrameSettings,

    // ── Scene data ────────────────────────────────────────────────────────────
    /// GPU scene uniforms and bind group.
    pub scene: &'a GpuScene,
    /// G-buffer render targets.
    pub gbuffer: &'a GBuffer,
    /// Global PBR material table.
    pub material_table: &'a MaterialTable,
    /// GPU light buffer (not used directly in `execute_frame`, held for callers).
    pub light_buffer: &'a LightBuffer,

    // ── Clipmap LOD ────────────────────────────────────────────────────────────
    /// Clipmap GPU data for multi-level LOD traversal.
    pub clipmap: &'a ClipmapGpuData,

    // ── Core passes ───────────────────────────────────────────────────────────
    /// Ray march pass — produces the G-buffer.
    pub ray_march: &'a RayMarchPass,
    /// Tiled light culling pass.
    pub tile_cull: &'a TileCullPass,
    /// PBR shading pass.
    pub shading: &'a ShadingPass,
    /// Tone mapping pass (HDR → LDR).
    pub tone_map: &'a ToneMapPass,
    /// Final blit to swapchain.
    pub blit: &'a BlitPass,

    /// Pre-computed light bind group from [`TileCullPass`].
    ///
    /// Extracted here to avoid dual-borrow of `tile_cull` and `shading` in
    /// `execute_frame`. Callers should pass `&tile_cull.light_list_bind_group`.
    pub shade_light_bind_group: &'a wgpu::BindGroup,

    // ── Global illumination ───────────────────────────────────────────────────
    /// Radiance volume (voxel cone tracing GI data).
    pub radiance_volume: &'a RadianceVolume,
    /// Radiance injection pass (L0 fill).
    pub radiance_inject: &'a RadianceInjectPass,
    /// Radiance mip-chain generation pass (L0 → L1-L3).
    pub radiance_mip: &'a RadianceMipPass,

    // ── Color pool ────────────────────────────────────────────────────────────
    /// GPU color pool for per-voxel color companion data.
    pub color_pool: &'a GpuColorPool,

    // ── Volumetrics ───────────────────────────────────────────────────────────
    /// Volumetric shadow map pass.
    pub vol_shadow: &'a VolShadowPass,
    /// Volumetric ray march pass (half-resolution).
    pub vol_march: &'a VolMarchPass,
    /// Parameters for the volumetric march pass.
    pub vol_march_params: &'a VolMarchParams,
    /// Volumetric temporal reprojection pass (ping-pong, requires `&mut`).
    pub vol_temporal: &'a mut VolTemporalPass,
    /// Bilateral upscale for the volumetric buffer.
    pub vol_upscale: &'a VolUpscalePass,
    /// Volumetric compositing pass.
    pub vol_composite: &'a VolCompositePass,
    /// Cloud shadow map pass.
    pub cloud_shadow: &'a CloudShadowPass,

    // ── Pre-upscale post-processing ───────────────────────────────────────────
    /// Bloom extract and blur pass.
    pub bloom: &'a BloomPass,
    /// Depth of field pass.
    pub dof: &'a DofPass,
    /// Motion blur pass.
    pub motion_blur: &'a MotionBlurPass,

    // ── Upscale ───────────────────────────────────────────────────────────────
    /// Upscale pass (DLSS or custom temporal).
    pub upscale: &'a UpscalePass,
    /// Edge-aware sharpening pass (custom upscaler only).
    pub sharpen: &'a SharpenPass,
    /// Index into the history ping-pong buffer for this frame's read-back.
    pub history_read_idx: usize,

    // ── Post-upscale post-processing ──────────────────────────────────────────
    /// Bloom composite pass (post-upscale).
    pub bloom_composite: &'a BloomCompositePass,
    /// Auto-exposure luminance adaptation pass.
    pub auto_exposure: &'a AutoExposurePass,
    /// Color grading LUT pass.
    pub color_grade: &'a ColorGradePass,
    /// Cosmetic effects pass (vignette, grain, chromatic aberration).
    pub cosmetics: &'a CosmeticsPass,

    // ── Frame info ────────────────────────────────────────────────────────────
    /// Monotonically increasing frame counter (wraps at `u32::MAX`).
    pub frame_index: u32,
    /// Delta time in seconds since the previous frame.
    pub dt: f32,
    /// Camera world position, camera-relative (used for volumetric passes).
    pub camera_pos: [f32; 3],
    /// Sun/directional-light direction vector (normalised).
    pub sun_dir: [f32; 3],
    /// Swapchain texture view to blit the final image into.
    pub swapchain_view: &'a wgpu::TextureView,
}

/// Execute a complete render frame following the engine's static pass order.
///
/// The pass order mirrors the pipeline defined in the architecture document:
///
/// 1. Ray march → G-buffer
/// 2. Tile light cull (independent)
/// 3. GI: radiance inject + mip gen (conditional on [`FrameSettings::gi_enabled`])
/// 4. Volumetric shadow map (conditional)
/// 5. Cloud shadow map (conditional)
/// 6. Shade (depends on tile lights + GI + G-buffer)
/// 7. Volumetric march (conditional, depends on vol shadow)
/// 8. Volumetric temporal reprojection (conditional)
/// 9. Volumetric bilateral upscale (conditional)
/// 10. Volumetric composite (conditional)
/// 11. Pre-upscale: bloom extract/blur (conditional)
/// 12. Pre-upscale: depth of field (conditional)
/// 13. Pre-upscale: motion blur (conditional)
/// 14. Upscale (DLSS or custom temporal)
/// 15. Edge sharpen (conditional on [`FrameSettings::sharpen_enabled`])
/// 16. Post-upscale: bloom composite (conditional)
/// 17. Post-upscale: auto exposure (conditional)
/// 18. Tone map
/// 19. Post-upscale: color grade (conditional)
/// 20. Post-upscale: cosmetics (conditional)
/// 21. Blit to swapchain
///
/// Animation rebake (step 1 in the full pipeline), particle simulate (step 2),
/// and screen-space particles (step 18) are not yet implemented; they are
/// skipped here and will be wired in later phases.
pub fn execute_frame(ctx: &mut FrameContext) {
    // ── 1. Ray march → G-buffer ───────────────────────────────────────────────
    ctx.ray_march
        .dispatch(ctx.encoder, ctx.scene, ctx.gbuffer, ctx.clipmap);

    // ── 2. Tile light cull (independent of GI) ───────────────────────────────
    ctx.tile_cull.dispatch(ctx.encoder, ctx.gbuffer);

    // ── 3. GI: radiance inject + mip gen ─────────────────────────────────────
    if ctx.settings.gi_enabled {
        ctx.radiance_inject
            .dispatch(ctx.encoder, ctx.scene, ctx.material_table);
        ctx.radiance_mip.dispatch(ctx.encoder);
    }

    // ── 4. Volumetric shadow map ──────────────────────────────────────────────
    if ctx.settings.volumetrics_enabled {
        ctx.vol_shadow
            .dispatch(ctx.encoder, ctx.queue, ctx.camera_pos, ctx.sun_dir);
    }

    // ── 5. Cloud shadow map ───────────────────────────────────────────────────
    if ctx.settings.cloud_shadows_enabled {
        ctx.cloud_shadow
            .dispatch(ctx.encoder, ctx.queue, ctx.camera_pos, ctx.sun_dir);
    }

    // ── 6. Shade ──────────────────────────────────────────────────────────────
    ctx.shading.dispatch(
        ctx.encoder,
        ctx.gbuffer,
        ctx.material_table,
        ctx.scene,
        ctx.shade_light_bind_group,
        ctx.radiance_volume,
        ctx.color_pool,
    );

    // ── 7-10. Volumetrics ─────────────────────────────────────────────────────
    if ctx.settings.volumetrics_enabled {
        ctx.vol_march
            .dispatch(ctx.encoder, ctx.queue, ctx.vol_march_params);
        ctx.vol_temporal
            .dispatch(ctx.encoder, ctx.queue, ctx.frame_index);
        ctx.vol_upscale.dispatch(ctx.encoder);
        ctx.vol_composite.dispatch(ctx.encoder);
    }

    // ── 11-13. Pre-upscale post-processing ───────────────────────────────────
    if ctx.settings.bloom_enabled {
        ctx.bloom.dispatch(ctx.encoder);
    }
    if ctx.settings.dof_enabled {
        ctx.dof.dispatch(ctx.encoder);
    }
    if ctx.settings.motion_blur_enabled {
        ctx.motion_blur.dispatch(ctx.encoder);
    }

    // ── 14. Upscale ───────────────────────────────────────────────────────────
    ctx.upscale.dispatch(ctx.encoder, ctx.history_read_idx);

    // ── 15. Edge sharpen ──────────────────────────────────────────────────────
    if ctx.settings.sharpen_enabled {
        ctx.sharpen.dispatch(ctx.encoder);
    }

    // ── 16-20. Post-upscale post-processing ──────────────────────────────────
    if ctx.settings.bloom_enabled {
        ctx.bloom_composite.dispatch(ctx.encoder);
    }
    if ctx.settings.auto_exposure_enabled {
        ctx.auto_exposure.dispatch(ctx.encoder, ctx.queue, ctx.dt);
    }

    // Tone map always runs (HDR → LDR is mandatory).
    ctx.tone_map.dispatch(ctx.encoder);

    if ctx.settings.color_grade_enabled {
        ctx.color_grade.dispatch(ctx.encoder);
    }
    if ctx.settings.cosmetics_enabled {
        ctx.cosmetics
            .dispatch(ctx.encoder, ctx.queue, ctx.frame_index);
    }

    // ── 21. Blit to swapchain ─────────────────────────────────────────────────
    ctx.blit.draw(ctx.encoder, ctx.swapchain_view);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_settings_default_all_enabled() {
        let s = FrameSettings::default();
        assert!(s.volumetrics_enabled);
        assert!(s.cloud_shadows_enabled);
        assert!(s.gi_enabled);
        assert!(s.dof_enabled);
        assert!(s.motion_blur_enabled);
        assert!(s.bloom_enabled);
        assert!(s.auto_exposure_enabled);
        assert!(s.color_grade_enabled);
        assert!(s.cosmetics_enabled);
        assert!(s.sharpen_enabled);
    }

    #[test]
    fn frame_settings_field_count() {
        // There are exactly 10 optional passes tracked in FrameSettings.
        // If this count changes, update execute_frame accordingly.
        let s = FrameSettings::default();
        let enabled: usize = [
            s.volumetrics_enabled,
            s.cloud_shadows_enabled,
            s.gi_enabled,
            s.dof_enabled,
            s.motion_blur_enabled,
            s.bloom_enabled,
            s.auto_exposure_enabled,
            s.color_grade_enabled,
            s.cosmetics_enabled,
            s.sharpen_enabled,
        ]
        .iter()
        .filter(|&&v| v)
        .count();
        assert_eq!(enabled, 10, "all 10 settings should be enabled by default");
    }

    #[test]
    fn frame_settings_individual_toggle() {
        let mut s = FrameSettings::default();

        s.volumetrics_enabled = false;
        assert!(!s.volumetrics_enabled);
        assert!(s.gi_enabled, "other settings must be unaffected");

        s.gi_enabled = false;
        assert!(!s.gi_enabled);

        s.bloom_enabled = false;
        assert!(!s.bloom_enabled);

        s.sharpen_enabled = false;
        assert!(!s.sharpen_enabled);
    }

    #[test]
    fn frame_settings_minimal() {
        let s = FrameSettings {
            volumetrics_enabled: false,
            cloud_shadows_enabled: false,
            gi_enabled: false,
            dof_enabled: false,
            motion_blur_enabled: false,
            bloom_enabled: false,
            auto_exposure_enabled: false,
            color_grade_enabled: false,
            cosmetics_enabled: false,
            sharpen_enabled: false,
        };
        let enabled: usize = [
            s.volumetrics_enabled,
            s.cloud_shadows_enabled,
            s.gi_enabled,
            s.dof_enabled,
            s.motion_blur_enabled,
            s.bloom_enabled,
            s.auto_exposure_enabled,
            s.color_grade_enabled,
            s.cosmetics_enabled,
            s.sharpen_enabled,
        ]
        .iter()
        .filter(|&&v| v)
        .count();
        assert_eq!(enabled, 0, "all settings disabled — only mandatory passes run");
    }

    #[test]
    fn frame_settings_clone_eq() {
        let a = FrameSettings::default();
        let b = a.clone();
        assert_eq!(a, b);

        let mut c = a.clone();
        c.gi_enabled = false;
        assert_ne!(a, c);
    }
}
