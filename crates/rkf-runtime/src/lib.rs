//! # rkf-runtime
//!
//! Engine runtime for the RKIField SDF engine.
//!
//! This crate ties together all subsystems into a coherent frame loop.
//! It manages:
//! - Frame scheduling and render graph execution
//! - ECS world (via hecs) for scene entity management
//! - Asset loading and caching
//! - Configuration and quality preset management
//! - Window and input handling integration

#![warn(missing_docs)]

/// Asset registry with generational handles and state tracking.
pub mod asset_registry;
/// ECS component types for the scene graph.
pub mod components;
/// Engine configuration system with quality presets and RON serialization.
pub mod config;
/// Double-buffered GPU uniform buffers.
pub mod double_buffer;
/// Frame scheduling and render pass ordering (stub — pending v2 rewrite).
pub mod frame;
/// Scene management — v2 object-centric SDF scene + hecs ECS for non-SDF entities.
pub mod scene;
/// Transform hierarchy update system.
pub mod transform_system;
/// GPU memory audit and leak detection.
pub mod memory_audit;
/// Performance profiling: CPU timing, per-pass breakdown, frame history.
pub mod profiler;
/// Quality preset validation — checks engine configs for out-of-range values.
pub mod preset_validator;
/// Stress test framework — scenario definitions, result tracking, evaluation.
pub mod stress_test;
/// Edge case handling — graceful degradation, chunk boundaries, LOD transitions, window resize.
pub mod edge_cases;
/// Shader hot-reload — registry, change detection, compilation state.
pub mod shader_reload;
/// Material hot-reload — registry, RON serialization, change diffing.
pub mod material_reload;

pub use asset_registry::{AssetEntry, AssetRegistry, AssetState, Handle};
pub use components::{
    CameraComponent, EditorMetadata, FogVolumeComponent, Parent,
    Transform, WorldTransform,
};
pub use config::{
    ConfigError, EngineConfig, GiSettings, PostProcessSettings, QualityPreset, RayMarchSettings,
    ShadingSettings, UpscaleSettings, VolumetricSettings,
};
pub use double_buffer::{DoubleBuffer, DoubleBufferSet};
pub use frame::FrameSettings;
pub use scene::RuntimeScene;
pub use transform_system::{flatten_sdf_scene, update_all_transforms, update_transforms};
pub use memory_audit::{
    LeakReport, MemoryAudit, MemoryHistory, PoolStats, detect_leaks, POOL_BONE_BRICKS,
    POOL_COLOR_BRICKS, POOL_SDF_BRICKS, POOL_STAGING, POOL_VOLUMETRIC_BRICKS,
};
pub use profiler::{
    CpuTimer, FrameProfile, PassTiming, ProfileHistory, Profiler, ProfilingConfig,
};
pub use preset_validator::{
    PresetComparison, PresetValidation, Severity, SettingDiff, ValidationIssue, compare_presets,
    validate_all_presets, validate_config,
};
pub use stress_test::{
    StressConfig, StressResult, StressScenario, StressSuite, StressThresholds,
    build_default_suite, evaluate_result,
};
pub use edge_cases::{
    BoundaryFix, ChunkBoundaryFixer, ConfigChange, DegradationLevel, GracefulDegradation,
    LodTransitionBlender, ResizeAction, WindowResizeHandler,
};
pub use shader_reload::{
    ChangeType, CompileError, ShaderChangeEvent, ShaderRegistry, ShaderReloadState, ShaderSource,
    check_shader_changed, compute_source_hash,
};
pub use material_reload::{
    MaterialChangeSet, MaterialDefinition, MaterialFile, MaterialFileEntry, MaterialProperties,
    MaterialRegistry, diff_material_files, parse_material_file, serialize_material_file,
};
