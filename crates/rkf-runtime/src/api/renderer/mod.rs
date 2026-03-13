//! Renderer — the GPU rendering pipeline.
//!
//! Owns all GPU resources (brick pool buffer, G-buffer, 21 compute passes,
//! readback buffers). Takes a [`&World`](super::world::World) per frame — does
//! NOT own the scene.
//!
//! # Features
//!
//! - Full 21-pass compute pipeline (ray march, shading, GI, volumetrics, post-FX)
//! - Wireframe overlay rendering (selection highlights, gizmos, debug)
//! - GPU pick readback (object ID at pixel)
//! - GPU brush hit readback (world position + object ID at pixel)
//! - Environment-driven rendering (sun, fog, clouds, post-processing)
//! - Offscreen rendering with compositor integration
//!
//! # Usage
//!
//! ```ignore
//! let mut renderer = Renderer::new(&window, RendererConfig::default());
//! renderer.add_light(Light::directional([0.5, 1.0, 0.3], [1.0, 0.95, 0.85], 3.0, true));
//! // Per frame:
//! renderer.render_to_surface(&world, &surface);
//! ```

mod accessors;
mod helpers;
mod init;
mod readback;
mod render;
mod resize;

use glam::Vec3;

use rkf_core::material::Material;
#[allow(deprecated)]
use rkf_render::material_table::MaterialTable;
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass, GBuffer,
    GodRaysBlurPass, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass, VolShadowPass,
    VolUpscalePass, WireframePass,
};

pub use helpers::transform_aabb;

// ── Configuration ────────────────────────────────────────────────────────────

/// Renderer configuration.
pub struct RendererConfig {
    /// Internal render resolution width (default 960).
    pub internal_width: u32,
    /// Internal render resolution height (default 540).
    pub internal_height: u32,
    /// Display/viewport resolution width (default 1280).
    pub display_width: u32,
    /// Display/viewport resolution height (default 720).
    pub display_height: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            internal_width: 960,
            internal_height: 540,
            display_width: 1280,
            display_height: 720,
        }
    }
}

/// Where to render the final image.
pub enum RenderTarget<'a> {
    /// Render to a window surface (swapchain).
    Surface(&'a wgpu::Surface<'a>),
    /// Render to an offscreen texture view.
    Texture(&'a wgpu::TextureView),
}

/// Rendering environment configuration.
///
/// Contains all parameters that drive the rendering pipeline: sun, atmosphere,
/// fog, clouds, and post-processing. Set via [`Renderer::set_render_environment`].
#[derive(Debug, Clone)]
pub struct RenderEnvironment {
    // Atmosphere
    /// Sun direction (normalized).
    pub sun_direction: Vec3,
    /// Sun base color (before tinting by elevation).
    pub sun_color: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Rayleigh scattering scale.
    pub rayleigh_scale: f32,
    /// Mie scattering scale.
    pub mie_scale: f32,
    /// Whether the atmosphere is enabled.
    pub atmosphere_enabled: bool,

    // Fog
    /// Ground fog density (0.0 = off).
    pub fog_density: f32,
    /// Fog color.
    pub fog_color: Vec3,
    /// Height falloff rate for fog.
    pub fog_height_falloff: f32,
    /// Ambient dust density for volumetric scattering.
    pub ambient_dust: f32,
    /// Henyey-Greenstein asymmetry parameter for dust scattering.
    pub dust_asymmetry: f32,

    // Clouds
    /// Cloud simulation settings.
    pub cloud_settings: rkf_render::CloudSettings,

    // Post-processing
    /// Bloom threshold (HDR).
    pub bloom_threshold: f32,
    /// Bloom intensity.
    pub bloom_intensity: f32,
    /// Tone mapping mode (0 = ACES, 1 = AgX).
    pub tone_map_mode: u32,
    /// Manual exposure multiplier.
    pub exposure: f32,
    /// Depth of field enabled.
    pub dof_enabled: bool,
    /// DoF focus distance in metres.
    pub dof_focus_distance: f32,
    /// DoF focus range (transition zone).
    pub dof_focus_range: f32,
    /// DoF maximum circle of confusion radius.
    pub dof_max_coc: f32,
    /// Sharpen strength.
    pub sharpen_strength: f32,
    /// Motion blur intensity.
    pub motion_blur_intensity: f32,
    /// God rays intensity.
    pub god_rays_intensity: f32,
    /// Vignette intensity.
    pub vignette_intensity: f32,
    /// Film grain intensity.
    pub grain_intensity: f32,
    /// Chromatic aberration strength.
    pub chromatic_aberration: f32,
}

impl Default for RenderEnvironment {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.5, 1.0, 0.3).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85),
            sun_intensity: 3.0,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            atmosphere_enabled: true,
            fog_density: 0.0,
            fog_color: Vec3::new(0.7, 0.75, 0.8),
            fog_height_falloff: 0.5,
            ambient_dust: 0.005,
            dust_asymmetry: 0.6,
            cloud_settings: rkf_render::CloudSettings::default(),
            bloom_threshold: 1.0,
            bloom_intensity: 0.3,
            tone_map_mode: 0,
            exposure: 1.0,
            dof_enabled: false,
            dof_focus_distance: 5.0,
            dof_focus_range: 2.0,
            dof_max_coc: 6.0,
            sharpen_strength: 0.5,
            motion_blur_intensity: 0.5,
            god_rays_intensity: 0.3,
            vignette_intensity: 0.15,
            grain_intensity: 0.02,
            chromatic_aberration: 0.0,
        }
    }
}

/// Result of a brush hit readback (world position + object ID).
#[derive(Debug, Clone)]
pub struct BrushHitResult {
    /// World-space position of the hit point (camera-relative f32).
    pub position: Vec3,
    /// Object ID from the G-buffer material texture.
    pub object_id: u32,
}

/// Offscreen render target format for the compositor path.
///
/// sRGB variant so the blit pass's linear-to-sRGB conversion happens in hardware.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// ── Renderer ─────────────────────────────────────────────────────────────────

/// The GPU rendering pipeline.
///
/// Owns all GPU resources and dispatches 21 compute passes per frame.
/// Takes `&World` to read the scene — does NOT own scene data.
pub struct Renderer {
    // GPU context
    pub(crate) ctx: RenderContext,

    // Scene GPU data
    pub(crate) gpu_scene: GpuSceneV2,
    pub(crate) brick_pool_buffer: wgpu::Buffer,

    // Core rendering
    pub(crate) gbuffer: GBuffer,
    pub(crate) tile_cull: TileObjectCullPass,
    pub(crate) coarse_field: CoarseField,
    pub(crate) ray_march: RayMarchPass,
    pub(crate) debug_view: DebugViewPass,
    pub(crate) shading_pass: ShadingPass,
    pub(crate) brush_overlay: rkf_render::BrushOverlay,
    pub(crate) gpu_color_pool: rkf_render::GpuColorPool,
    pub(crate) material_table: MaterialTable,

    // Global illumination
    pub(crate) radiance_volume: RadianceVolume,
    pub(crate) radiance_inject: RadianceInjectPass,
    pub(crate) radiance_mip: RadianceMipPass,

    // Volumetrics
    pub(crate) vol_shadow: VolShadowPass,
    pub(crate) cloud_shadow: CloudShadowPass,
    pub(crate) vol_march: VolMarchPass,
    pub(crate) vol_upscale: VolUpscalePass,
    pub(crate) vol_composite: VolCompositePass,

    // Post-processing
    pub(crate) god_rays_blur: GodRaysBlurPass,
    pub(crate) bloom: BloomPass,
    pub(crate) auto_exposure: AutoExposurePass,
    pub(crate) dof: DofPass,
    pub(crate) motion_blur: MotionBlurPass,
    pub(crate) bloom_composite: BloomCompositePass,
    pub(crate) tone_map: ToneMapPass,
    pub(crate) color_grade: ColorGradePass,
    pub(crate) cosmetics: CosmeticsPass,
    #[allow(dead_code)]
    pub(crate) sharpen: SharpenPass,
    pub(crate) blit: BlitPass,

    // Wireframe overlay
    pub(crate) wireframe_pass: WireframePass,

    // Offscreen rendering (compositor path)
    pub(crate) offscreen_texture: Option<wgpu::Texture>,
    pub(crate) offscreen_view: Option<wgpu::TextureView>,
    pub(crate) offscreen_blit: Option<BlitPass>,

    // GPU readback
    pub(crate) readback_buffer: wgpu::Buffer,
    pub(crate) pick_readback_buffer: wgpu::Buffer,
    pub(crate) brush_readback_buffer: wgpu::Buffer,

    // CPU state
    pub(crate) camera: Camera,
    pub(crate) world_lights: Vec<Light>,
    pub(crate) materials: Vec<Material>,
    pub(crate) light_buffer: LightBuffer,
    pub(crate) render_env: RenderEnvironment,
    pub(crate) frame_index: u32,
    pub(crate) prev_vp: [[f32; 4]; 4],
    pub(crate) shade_debug_mode: u32,
    pub(crate) accumulated_time: f32,

    // Config
    pub(crate) internal_width: u32,
    pub(crate) internal_height: u32,
    pub(crate) display_width: u32,
    pub(crate) display_height: u32,
    pub(crate) surface_format: wgpu::TextureFormat,
}
