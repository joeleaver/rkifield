//! # rkf-particles
//!
//! GPU particle system for the RKIField SDF engine.
//!
//! Supports three distinct rendering backends, none of which use traditional
//! billboard quads:
//! - **Volumetric density splats** for glowing/emissive effects
//! - **SDF micro-objects** for solid particles with proper lighting
//! - **Screen-space overlay** for weather effects (rain, snow)
//!
//! All particle simulation runs in compute shaders. Emitters define spawn
//! rules, and the GPU handles integration, collision (via SDF queries),
//! and sorting.

#![warn(missing_docs)]

pub mod binning;
pub mod emitter;
pub mod micro_object;
pub mod particle;
pub mod screen_space;
pub mod simulate;
pub mod volumetric;
