//! # rkf-import
//!
//! Mesh-to-SDF conversion library for the RKIField engine.
//!
//! Converts traditional polygon meshes into signed distance field
//! representations stored as voxel bricks. Uses BVH acceleration and
//! generalized winding number computation for robust inside/outside
//! classification.
//!
//! This crate provides:
//! - Mesh loading and triangle soup processing
//! - BVH construction for accelerated distance queries
//! - Winding number computation for sign determination
//! - Multi-resolution voxelization into brick format
//! - `.rkf` asset file writing with LZ4 compression

#![warn(missing_docs)]

pub mod animated_chunk;
pub mod bvh;
pub mod lod;
pub mod material_transfer;
pub mod mesh;
pub mod segment_voxelize;
pub mod skeleton_extract;
pub mod voxelize;
