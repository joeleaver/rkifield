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
/// .rkf v2 file format — serialization and deserialization for voxelized objects.
pub mod asset_file;
/// .rkf v3 file format — geometry-first serialization (occupancy + surface voxels + SDF cache).
pub mod asset_file_v3;
pub mod automation;
/// Geometry-first brick representation — occupancy bitmask + surface voxels.
pub mod brick_geometry;
/// Bounding volume hierarchy over scene objects for spatial acceleration.
pub mod bvh;
/// 8×8×8 voxel brick — the fundamental unit of SDF storage.
pub mod brick;
/// Per-object brick maps — flat 3D arrays mapping brick coordinates to pool slots.
pub mod brick_map;
/// CPU-side brick pool with free-list allocation.
pub mod brick_pool;
/// Companion brick types: bone, volumetric, and color data pools.
pub mod companion;
pub mod constants;
/// Per-object LOD (Level of Detail) selection — screen-space driven.
pub mod lod;
/// LOD manager — per-frame LOD selection and brick map transitions.
pub mod lod_manager;
/// Material properties for the global GPU material table.
pub mod material;
/// File-driven material library — `.rkmat` files, palettes, hot-reload support.
pub mod material_library;
/// Geometry resampling — change voxel resolution of an existing object.
pub mod resample;
/// CPU reference trilinear sampling within a brick.
pub mod sampling;
/// v2 scene and root object container.
pub mod scene;
/// Grid-based terrain container — O(1) tile lookup, procedural SDF fallback.
pub mod terrain;
/// v2 scene hierarchy node — object SDF tree with transforms and blending.
pub mod scene_node;
/// Cached SDF distances derived from brick geometry.
pub mod sdf_cache;
/// Compute SDF distances from geometry (occupancy → signed distances).
pub mod sdf_compute;
/// SDF generation utilities for testing and offline voxelization.
pub mod sdf;
/// Geodesic surface flood fill over voxel brick geometry.
pub mod surface_flood;
/// Transform baking — compute world transforms from parent-local hierarchy.
pub mod transform_bake;
/// v2 transform flattening — depth-first traversal producing GPU-ready flat nodes.
pub mod transform_flatten;
/// Per-object voxelization — SDF to brick map conversion.
pub mod voxelize_object;
/// Voxel sample type and flag constants for GPU-packed voxel data.
pub mod voxel;
pub mod world_position;

pub use aabb::{Aabb, WorldAabb};
pub use asset_file::{
    AssetError, LodData, LodEntryInfo, ObjectHeader, SaveLodLevel, load_object_header,
    load_object_lod, save_object,
};
pub use bvh::{Bvh, BvhNode};
pub use scene::{Scene, SceneObject};
pub use terrain::{TerrainConfig, TerrainLodTier, TerrainNode, TerrainTile};
pub use scene_node::{
    BlendMode, BrickMapHandle, NodeMetadata, SceneNode, SdfPrimitive, SdfSource, Transform,
};
pub use brick::Brick;
pub use brick_geometry::{BrickGeometry, NeighborContext, SurfaceVoxel, index_to_xyz, voxel_index};
pub use brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT};
pub use brick_pool::{BonePool, BrickPool, ColorPool, GeometryPool, Pool, SdfCachePool, VolumetricPool};
pub use companion::{BoneBrick, BoneVoxel, ColorBrick, ColorVoxel, VolumetricBrick, VolumetricVoxel};
pub use lod::{LodLevel, LodSelection, ObjectLod, select_lod};
pub use lod_manager::{LodManager, LodTransition};
pub use material::Material;
pub use material_library::{MaterialEntry, MaterialLibrary, MaterialPalette, MaterialProperties};
pub use sdf_cache::SdfCache;
pub use sdf_compute::{SlotMapping, compute_sdf_from_geometry, compute_sdf_region};
pub use resample::{ResampleResult, resample_geometry};
pub use voxel::VoxelSample;
pub use voxelize_object::{VoxelizeGeometryResult, evaluate_primitive, voxelize_sdf, voxelize_to_geometry};
pub use world_position::WorldPosition;
