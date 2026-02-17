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

pub mod aabb;
pub mod automation;
/// 8×8×8 voxel brick — the fundamental unit of SDF storage.
pub mod brick;
/// CPU-side brick pool with free-list allocation.
pub mod brick_pool;
pub mod cell_state;
/// Companion brick types: bone, volumetric, and color data pools.
pub mod companion;
pub mod constants;
/// Material properties for the global GPU material table.
pub mod material;
/// Single-LOD sparse grid for voxel occupancy and brick indexing.
pub mod sparse_grid;
/// Voxel sample type and flag constants for GPU-packed voxel data.
pub mod voxel;
pub mod world_position;

pub use aabb::{Aabb, WorldAabb};
pub use brick::Brick;
pub use brick_pool::{BonePool, BrickPool, ColorPool, Pool, VolumetricPool};
pub use cell_state::CellState;
pub use sparse_grid::{SparseGrid, EMPTY_SLOT};
pub use companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel, VolumetricBrick, VolumetricVoxel};
pub use material::Material;
pub use voxel::VoxelSample;
pub use world_position::WorldPosition;
