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

// Allow `rkf_runtime::` paths in macro-generated code within this crate.
extern crate self as rkf_runtime;

/// Behavior system types — components, systems, registry, and execution.
pub mod behavior;

/// Public engine API — World, Entity, Renderer, SpawnBuilder.
pub mod api;

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
/// Async I/O pipeline for loading .rkf v2 object files on background threads.
pub mod async_io;
/// Per-object LRU eviction system — tracks brick-pool usage and selects
/// eviction/demotion candidates when the pool is under pressure.
pub mod lru_eviction;
/// Per-object streaming state machine for the v2 object-centric SDF architecture.
pub mod object_streaming;
/// Multi-scene management — load, activate, unload, and query simultaneous scenes.
pub mod scene_manager;
/// Project file format (.rkproject) — RON-serialized project descriptor.
pub mod project;
/// Scene file format v3 — entity-centric scene format with StableId UUIDs.
pub mod scene_file_v3;
/// Environment profiles (.rkenv) — sky, fog, ambient, volumetric, and post-process hints
/// with blending, overrides, and RON serialisation.
pub mod environment;
/// Game state management — scene tracking, environment control, and typed key-value store.
pub mod game_manager;
/// Main camera with environment ownership and zone-based environment transitions.
pub mod main_camera;
/// Save/load system (.rksave) — full game-state snapshots with RON serialization.
pub mod save_system;
/// Terrain tile streaming manager — camera-based LOD selection and tile load/unload scheduling.
pub mod terrain_streaming;
/// File watcher — monitors asset and shader directories for changes (notify-based).
pub mod file_watcher;

pub use asset_registry::{AssetEntry, AssetRegistry, AssetState, Handle};
pub use components::{
    CameraComponent, EditorMetadata, FogVolumeComponent, Parent,
    SdfTree, Transform, WorldTransform,
};
pub use config::{
    ConfigError, EngineConfig, GiSettings, PostProcessSettings, QualityPreset, RayMarchSettings,
    ShadingSettings, UpscaleSettings, VolumetricSettings,
};
pub use double_buffer::{DoubleBuffer, DoubleBufferSet};
pub use frame::FrameSettings;
pub use scene::RuntimeScene;
pub use transform_system::{
    build_scene_bvh, flatten_sdf_scene, refit_scene_bvh, update_all_transforms, update_transforms,
};
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
pub use lru_eviction::{
    EvictionAction, EvictionPolicy, LruEvictionTracker, ObjectUsageEntry,
};
pub use object_streaming::{
    LoadRequest, ObjectStreamState, ObjectStreamingSystem, StreamingConfig, StreamingObject,
};
pub use scene_manager::{
    LoadMode, ManagedScene, SceneHandle, SceneManager, SceneStatus,
};
pub use project::{
    ProjectFile, SceneRef, load_project, save_project,
    create_project, project_root, resolve_scene_path, ENGINE_SHADERS,
};
pub use environment::{
    AmbientConfig, EnvironmentOverrides, EnvironmentProfile, FogConfig, PostProcessHints,
    SkyMode, VolumetricHints, apply_overrides, lerp_profiles, load_environment,
    resolve_environment, save_environment,
};
pub use behavior::{
    BehaviorExecutor, Blueprint, BlueprintCatalog, BuildState, BuildWatcher, CommandQueue,
    ComponentEntry, ComponentMeta, DylibError, DylibLoader, EditError, EditOp, EditPipeline,
    EntityNameIndex, EntityTagIndex, FieldMeta, FieldType, GameStore, GameplayRegistry,
    LookupError, Phase, PlayModeManager, PlayState, QueryError, ReloadQueue, ScaffoldError,
    Schedule, ScheduleError, SceneOwnership, Sequence, SequenceBuilder, SequenceStep, StableId,
    StableIdIndex, StoreEvent, SystemContext, SystemMeta, TempEntity, UndoAction, UndoStack,
    build_schedule, children_of, clone_world_for_play, create_blueprint_from_entity, descendants_of,
    deserialize_blueprint, deserialize_entity, find_path, find_tagged, parent_of, root_of,
    scaffold_game_crate, serialize_blueprint, serialize_entity,
};
pub use behavior::engine_components::{engine_register, ENGINE_COMPONENT_NAMES};
pub use behavior::game_value::GameValue as BehaviorGameValue;
pub use behavior::game_value::GameValueTypeError;
pub use game_manager::{GameEvent, GameManager, GameState, GameValue};
pub use main_camera::{EnvironmentZone, MainCamera};
pub use save_system::{
    CameraSnapshot, EntityOverride, EnvironmentSnapshot, SaveFile, SaveInfo,
    create_save, load_game, list_saves, save_game,
};
pub use terrain_streaming::{
    TerrainStreamConfig, TerrainStreaming, TileRequest, TileState,
};
pub use file_watcher::{FileEvent, FileWatcher};
