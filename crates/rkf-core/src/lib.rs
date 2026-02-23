//! # rkf-core
//!
//! Core data structures for the RKIField SDF graphics engine.
//!
//! This crate provides the foundational types shared across the engine:
//! - [`WorldPosition`] for float-precision-safe world-space coordinates
//! - Voxel sample types and brick data structures (8x8x8 voxel bricks)
//! - Brick pool management and GPU resource coordination
//! - Material table types and constants
//! - SDF primitives and AABB geometry

#![warn(missing_docs)]

pub mod aabb;
pub mod automation;
/// 8×8×8 voxel brick — the fundamental unit of SDF storage.
pub mod brick;
/// Per-object brick maps — flat 3D arrays mapping brick coordinates to pool slots.
pub mod brick_map;
/// CPU-side brick pool with free-list allocation.
pub mod brick_pool;
/// Companion brick types: bone, volumetric, and color data pools.
pub mod companion;
pub mod constants;
/// Material properties for the global GPU material table.
pub mod material;
/// CPU reference trilinear sampling within a brick.
pub mod sampling;
/// v2 scene and root object container.
pub mod scene;
/// v2 scene hierarchy node — object SDF tree with transforms and blending.
pub mod scene_node;
/// SDF generation utilities for testing and offline voxelization.
pub mod sdf;
/// v2 transform flattening — depth-first traversal producing GPU-ready flat nodes.
pub mod transform_flatten;
/// Per-object voxelization — SDF to brick map conversion.
pub mod voxelize_object;
/// Voxel sample type and flag constants for GPU-packed voxel data.
pub mod voxel;
pub mod world_position;

pub use aabb::{Aabb, WorldAabb};
pub use scene::{Scene, SceneObject};
pub use scene_node::{
    BlendMode, BrickMapHandle, NodeMetadata, SceneNode, SdfPrimitive, SdfSource, Transform,
};
pub use brick::Brick;
pub use brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT};
pub use brick_pool::{BonePool, BrickPool, ColorPool, Pool, VolumetricPool};
pub use companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel, VolumetricBrick, VolumetricVoxel};
pub use material::Material;
pub use voxel::VoxelSample;
pub use world_position::WorldPosition;
