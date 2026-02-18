//! # rkf-render
//!
//! Compute-shader rendering pipeline for the RKIField SDF engine.
//!
//! All rendering is performed via compute shader dispatches -- no rasterization
//! pipeline is used. This crate implements:
//! - Ray marching through voxel-backed signed distance fields
//! - G-buffer generation at internal resolution
//! - PBR shading with SDF soft shadows and ambient occlusion
//! - Global illumination via voxel cone tracing
//! - Atmospheric scattering and volumetric effects compositing
//! - Upscaling (DLSS or custom temporal) and post-processing

#![warn(missing_docs)]

/// GPU device and queue wrapper.
pub mod context;
/// Camera system for ray generation.
pub mod camera;
/// GPU buffer upload for scene data.
pub mod gpu_scene;

/// Blit pass — fullscreen copy to swapchain.
pub mod blit;
/// G-buffer textures for deferred shading.
pub mod gbuffer;
/// GPU material table for PBR shading.
pub mod material_table;
/// Ray march compute pass.
pub mod ray_march;
/// Light types and GPU light buffer.
pub mod light;
/// PBR shading compute pass.
pub mod shading;
/// Tiled light culling compute pass.
pub mod tile_cull;
/// Tone mapping compute pass (HDR → LDR).
pub mod tone_map;
/// Radiance volume for voxel cone tracing global illumination.
pub mod radiance_volume;
/// Radiance injection compute pass for GI.
pub mod radiance_inject;
/// Radiance mip generation pass (downsample L0 → L1 → L2 → L3).
pub mod radiance_mip;
/// Volumetric shadow map compute pass.
pub mod vol_shadow;
/// Volumetric ray march compute pass (half resolution).
pub mod vol_march;
/// History buffers for temporal upscaling (ping-pong at display resolution).
pub mod history;
/// Custom temporal upscaler compute pass.
pub mod upscale;
/// Edge-aware sharpening compute pass.
pub mod sharpen;
/// Post-processing stack architecture.
pub mod post_process;
/// Bloom compute pass (pre-upscale).
pub mod bloom;
/// Depth of field compute pass (pre-upscale).
pub mod dof;
/// Motion blur compute pass (pre-upscale).
pub mod motion_blur;
/// Bloom composite compute pass (post-upscale).
pub mod bloom_composite;
/// Auto-exposure compute pass.
pub mod auto_exposure;
/// Color grading compute pass (post-upscale).
pub mod color_grade;
/// Cosmetic post-processing effects (vignette, grain, chromatic aberration).
pub mod cosmetics;
/// DLSS integration layer (stub — falls back to custom upscaler).
pub mod dlss;
/// Analytic fog settings for height fog and distance fog.
pub mod fog;
/// Local fog volume types and GPU buffer.
pub mod fog_volume;

pub use camera::{Camera, CameraUniforms};
pub use context::RenderContext;
pub use gbuffer::GBuffer;
pub use gpu_scene::{GpuScene, SceneUniforms};
pub use material_table::MaterialTable;
pub use blit::BlitPass;
pub use ray_march::RayMarchPass;
pub use light::{Light, LightBuffer, MAX_LIGHTS, MAX_LIGHTS_PER_TILE, TILE_SIZE};
pub use shading::{ShadeUniforms, ShadingPass};
pub use tile_cull::{CullUniforms, TileCullPass};
pub use tone_map::{ToneMapPass, ToneMapMode, ToneMapParams, DEFAULT_EXPOSURE, LDR_FORMAT};
pub use radiance_volume::{RadianceVolume, RadianceVolumeUniforms, RADIANCE_DIM, RADIANCE_LEVELS};
pub use radiance_inject::{InjectUniforms, RadianceInjectPass};
pub use radiance_mip::RadianceMipPass;
pub use vol_shadow::{VolShadowPass, VolShadowParams, VOL_SHADOW_DIM_X, VOL_SHADOW_DIM_Y, VOL_SHADOW_DIM_Z, VOL_SHADOW_FORMAT, DEFAULT_VOL_SHADOW_RANGE, DEFAULT_VOL_SHADOW_HEIGHT};
pub use vol_march::{VolMarchPass, VolMarchParams, VOL_MARCH_FORMAT, DEFAULT_VOL_STEP_SIZE, DEFAULT_VOL_MAX_STEPS, DEFAULT_VOL_NEAR, DEFAULT_VOL_FAR, DEFAULT_AMBIENT_DUST, DEFAULT_AMBIENT_DUST_G};
pub use fog::{FogSettings, FogParams, DEFAULT_FOG_BASE_DENSITY, DEFAULT_FOG_BASE_HEIGHT, DEFAULT_FOG_HEIGHT_FALLOFF, DEFAULT_FOG_COLOR, DEFAULT_DISTANCE_FOG_DENSITY, DEFAULT_DISTANCE_FOG_FALLOFF};
pub use history::{HistoryBuffers, HistoryUniforms, HISTORY_COLOR_FORMAT, HISTORY_METADATA_FORMAT};
pub use upscale::{QualityMode, ResolutionConfig, UpscaleBackend, UpscalePass, UpscaleUniforms};
pub use sharpen::{SharpenPass, SharpenUniforms, DEFAULT_SHARPEN_STRENGTH};
pub use post_process::{PostProcessPassId, PostProcessConfig, PostProcessContext, PingPongBuffers, PP_FORMAT};
pub use bloom::{BloomPass, BloomParams, BLOOM_MIP_LEVELS, DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_KNEE};
pub use dof::{DofPass, DofParams, DEFAULT_FOCUS_DISTANCE, DEFAULT_FOCUS_RANGE, DEFAULT_MAX_COC, COC_FORMAT};
pub use motion_blur::{MotionBlurPass, MotionBlurParams, DEFAULT_MOTION_BLUR_INTENSITY, DEFAULT_MAX_MOTION_SAMPLES};
pub use bloom_composite::{BloomCompositePass, BloomCompositeParams, DEFAULT_BLOOM_INTENSITY};
pub use auto_exposure::{AutoExposurePass, ExposureParams, HISTOGRAM_BINS, DEFAULT_MIN_EV, DEFAULT_MAX_EV, DEFAULT_ADAPT_SPEED};
pub use color_grade::{ColorGradePass, ColorGradeParams, DEFAULT_LUT_SIZE, DEFAULT_COLOR_GRADE_INTENSITY};
pub use cosmetics::{CosmeticsPass, CosmeticsParams, DEFAULT_VIGNETTE_INTENSITY, DEFAULT_GRAIN_INTENSITY, DEFAULT_CHROMATIC_ABERRATION};
pub use dlss::{DlssContext, DlssQuality};
pub use fog_volume::{GpuFogVolume, FogVolumeHeader, FogVolumeBuffer, MAX_FOG_VOLUMES};
