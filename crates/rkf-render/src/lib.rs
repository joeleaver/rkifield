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

pub use camera::{Camera, CameraUniforms};
pub use context::RenderContext;
