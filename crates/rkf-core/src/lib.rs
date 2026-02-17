//! # rkf-core
//!
//! Core data structures for the RKIField SDF graphics engine.
//!
//! This crate provides the foundational types shared across the engine:
//! - [`WorldPosition`] for float-precision-safe world-space coordinates
//! - Voxel sample types and brick data structures (8x8x8 voxel bricks)
//! - Brick pool management and GPU resource coordination
//! - Sparse spatial index for world occupancy
//! - Material table types and constants
//! - Resolution tier definitions and chunk geometry

#![warn(missing_docs)]

pub mod automation;
pub mod cell_state;
/// Companion brick types: bone, volumetric, and color data pools.
pub mod companion;
pub mod constants;
/// Voxel sample type and flag constants for GPU-packed voxel data.
pub mod voxel;
pub mod world_position;

pub use cell_state::CellState;
pub use companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel, VolumetricBrick, VolumetricVoxel};
pub use voxel::VoxelSample;
pub use world_position::WorldPosition;
