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
/// DLSS integration layer (stub — falls back to custom upscaler).
pub mod dlss;

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
pub use tone_map::ToneMapPass;
pub use radiance_volume::{RadianceVolume, RadianceVolumeUniforms, RADIANCE_DIM, RADIANCE_LEVELS};
pub use radiance_inject::{InjectUniforms, RadianceInjectPass};
pub use radiance_mip::RadianceMipPass;
pub use history::{HistoryBuffers, HistoryUniforms, HISTORY_COLOR_FORMAT, HISTORY_METADATA_FORMAT};
pub use upscale::{QualityMode, ResolutionConfig, UpscaleBackend, UpscalePass, UpscaleUniforms};
pub use sharpen::{SharpenPass, SharpenUniforms, DEFAULT_SHARPEN_STRENGTH};
pub use post_process::{PostProcessPassId, PostProcessConfig, PostProcessContext, PingPongBuffers, PP_FORMAT};
pub use bloom::{BloomPass, BloomParams, BLOOM_MIP_LEVELS, DEFAULT_BLOOM_THRESHOLD, DEFAULT_BLOOM_KNEE};
pub use dof::{DofPass, DofParams, DEFAULT_FOCUS_DISTANCE, DEFAULT_FOCUS_RANGE, DEFAULT_MAX_COC, COC_FORMAT};
pub use dlss::{DlssContext, DlssQuality};
