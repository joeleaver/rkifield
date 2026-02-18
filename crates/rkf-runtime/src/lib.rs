//! # rkf-runtime
//!
//! Engine runtime for the RKIField SDF engine.
//!
//! This crate ties together all subsystems into a coherent frame loop.
//! It manages:
//! - Frame scheduling and render graph execution
//! - ECS world (via hecs) for scene entity management
//! - Chunk streaming and LOD management
//! - Asset loading and caching
//! - Configuration and quality preset management
//! - Window and input handling integration

#![warn(missing_docs)]

/// ECS component types for the scene graph.
pub mod components;
/// Scene management via hecs ECS.
pub mod scene;
/// Transform hierarchy update system.
pub mod transform_system;

pub use components::{
    AnimatedCharacter, CameraComponent, ChunkRef, EditorMetadata, FogVolumeComponent, Parent,
    SdfObject, Transform, WorldTransform,
};
pub use scene::Scene;
pub use transform_system::update_transforms;
