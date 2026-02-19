//! # rkf-animation
//!
//! Skeletal animation system for the RKIField SDF engine.
//!
//! Unlike traditional mesh-based engines, skeletal animation here operates on
//! segmented rigid body parts with joint rebaking. Bones are NOT evaluated
//! during ray marching -- instead, a compute shader blends joint regions using
//! smooth-min operations and writes the result back to the brick pool.
//!
//! This crate provides:
//! - Segmented joint rebaking pipeline
//! - Blend shape support via delta-SDF application
//! - Animation clip playback and blending
//! - Skeleton definition and bone hierarchy types

#![warn(missing_docs)]

pub mod blend_shape;
pub mod character;
pub mod clip;
pub mod player;
pub mod rebake;
pub mod segment;
pub mod skeleton;
