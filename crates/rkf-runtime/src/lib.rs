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

/// Async I/O pipeline for background chunk loading.
pub mod async_io;
/// Engine configuration system with quality presets and RON serialization.
pub mod config;
/// ECS component types for the scene graph.
pub mod components;
/// Double-buffered GPU uniform buffers.
pub mod double_buffer;
/// Frame scheduling and render pass ordering.
pub mod frame;
/// LRU eviction for the brick pool.
pub mod lru_eviction;
/// Scene management via hecs ECS.
pub mod scene;
/// Chunk streaming system — camera-distance-based load/evict management.
pub mod streaming;
/// Transform hierarchy update system.
pub mod transform_system;

pub use config::{
    ConfigError, EngineConfig, GiSettings, PostProcessSettings, QualityPreset, RayMarchSettings,
    ShadingSettings, UpscaleSettings, VolumetricSettings,
};
pub use double_buffer::{DoubleBuffer, DoubleBufferSet};
pub use components::{
    AnimatedCharacter, CameraComponent, ChunkRef, EditorMetadata, FogVolumeComponent, Parent,
    SdfObject, Transform, WorldTransform,
};
pub use frame::{execute_frame, FrameContext, FrameSettings};
pub use async_io::{AsyncIoPipeline, ChunkLoadResult};
pub use scene::Scene;
pub use lru_eviction::{EvictionPolicy, EvictionResult, LruEntry, LruTracker};
pub use streaming::{ChunkEntry, ChunkState, StreamingConfig, StreamingSystem};
pub use transform_system::update_transforms;
