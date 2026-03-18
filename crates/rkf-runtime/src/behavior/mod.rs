//! Behavior system types — components, systems, registry, and execution.
//!
//! This module contains the core types for the ECS behavior system:
//! - [`GameValue`] — dynamically-typed values for the game state store
//! - [`ComponentMeta`] / [`FieldMeta`] — component introspection for inspector, MCP, JS bindings
//! - [`ComponentEntry`] — type-erased component operations for hot-reload and serialization
//! - [`GameplayRegistry`] — central catalog of registered components, systems, and blueprints
//! - [`SystemMeta`] — system registration metadata (phase, ordering, function pointer)
//! - [`QueryError`] — error type for entity lookup operations

pub mod blueprint;
pub mod command_queue;
pub mod cow_bricks;
pub mod dylib_loader;
pub mod edit_pipeline;
pub mod engine_components;
pub mod engine_persist;
pub mod entity_lookup;
pub mod hot_reload;
pub mod entity_names;
pub mod entity_ref;
pub mod entity_serde;
pub mod executor;
pub mod game_store;
pub mod game_value;
pub mod registry;
pub mod scene_ownership;
pub mod scheduler;
pub mod sequence;
pub mod sequence_system;
#[allow(missing_docs)]
pub mod stable_id;
#[allow(missing_docs)]
pub mod stable_id_index;
pub mod build_watcher;
pub mod console;
pub mod engine_access;
pub mod inspector;
pub mod persist;
pub mod play_mode;
pub mod reload_queue;
pub mod scaffold;
pub mod system_context;
pub mod systems_panel;
pub mod tool_routing;
pub mod uniform_treatment;

#[cfg(test)]
mod integration_tests;
#[cfg(test)]
mod play_mode_tests;
#[cfg(test)]
mod edit_pipeline_tests;
#[cfg(test)]
mod command_queue_tests;
#[cfg(test)]
mod sequence_system_tests;

pub use blueprint::{
    Blueprint, BlueprintCatalog, create_blueprint_from_entity, deserialize_blueprint,
    serialize_blueprint,
};
pub use command_queue::{CommandQueue, TempEntity};
pub use edit_pipeline::{EditError, EditOp, EditPipeline, UndoAction, UndoStack};
pub use entity_names::{
    EntityNameIndex, EntityTagIndex, children_of, descendants_of, parent_of, root_of,
};
pub use entity_lookup::{LookupError, find_one, find_path, find_relative, find_tagged};
pub use entity_ref::{deserialize_entity, serialize_entity};
pub use executor::BehaviorExecutor;
pub use game_store::{GameStore, StoreEvent, StoreSnapshot};
pub use game_value::{GameValue, get_nested_field, set_nested_field};
pub use registry::{
    ComponentEntry, ComponentMeta, FieldMeta, FieldType, GameplayRegistry, Phase, QueryError,
    StructMeta, SystemMeta,
};
pub use build_watcher::{
    BuildState, BuildWatcher, CompileError, parse_cargo_errors,
    hash_source_tree, read_build_stamp, write_build_stamp,
};
pub use inspector::{
    InspectorData, ComponentInspectorData, FieldInspectorData, MANDATORY_COMPONENTS,
    build_inspector_data, available_components_for_entity, add_component_default, remove_component,
};
pub use systems_panel::{SystemPanelEntry, build_systems_panel};
pub use engine_persist::{EngineStateSnapshot, restore_engine_state_from_store, sync_engine_state_to_store};
pub use persist::{
    Persistable, persist_key, persist_component_key, persisted_field_names,
    auto_sync_to_store, auto_sync_from_store,
    auto_sync_entity_to_store, auto_sync_entity_from_store,
};
pub use dylib_loader::{DylibError, DylibLoader};
pub use hot_reload::{HotReloadError, ReloadReport, hot_reload};
pub use reload_queue::ReloadQueue;
pub use scaffold::ScaffoldError;
pub use scene_ownership::SceneOwnership;
pub use scheduler::{Schedule, ScheduleError, build_schedule};
pub use sequence::{Ease, Sequence, SequenceBuilder, SequenceStep, StepStartValues};
pub use stable_id::StableId;
pub use stable_id_index::StableIdIndex;
/// Re-exported for use in `#[component]` macro-generated code (Entity field UUID resolution).
pub use uuid::Uuid as _MacroUuid;
pub use cow_bricks::{BrickOwnership, CowBrickTracker};
pub use play_mode::{
    PlayModeManager, PlayState, PlayFrameContext, clone_world_for_play,
    run_play_frame, should_run_systems, is_tool_allowed, DISABLED_PLAY_TOOLS,
    load_scene_into_play_world, unload_scene_from_play_world, push_field_to_edit,
    build_play_inspector_data,
};
pub use console::{ConsoleBuffer, ConsoleEntry, ConsoleFilter, ConsoleLevel};
pub use engine_access::{EngineAccess, TransformUpdate, WorldEngineAccess};
pub use system_context::SystemContext;
pub use tool_routing::ToolEditMapping;
pub use uniform_treatment::{
    verify_engine_components_in_registry, verify_engine_component_fields,
    verify_inspector_renders_engine_components,
};
pub use blueprint::{save_entity_as_blueprint, spawn_from_blueprint};

// Collect all `#[component]`-registered ComponentEntry items via `inventory`.
inventory::collect!(ComponentEntry);

#[cfg(test)]
mod macro_tests;
