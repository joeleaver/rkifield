//! Behavior system types — components, systems, registry, and execution.
//!
//! This module contains the core types for the ECS behavior system:
//! - [`GameValue`] — dynamically-typed values for the game state store
//! - [`ComponentMeta`] / [`FieldMeta`] — component introspection for inspector, MCP, JS bindings
//! - [`ComponentEntry`] — type-erased component operations for hot-reload and serialization
//! - [`GameplayRegistry`] — central catalog of registered components, systems, and blueprints
//! - [`SystemMeta`] — system registration metadata (phase, ordering, function pointer)
//! - [`QueryError`] — error type for entity lookup operations

pub mod command_queue;
pub mod entity_lookup;
pub mod entity_names;
pub mod entity_ref;
pub mod executor;
pub mod game_store;
pub mod game_value;
pub mod registry;
pub mod scene_ownership;
pub mod scheduler;
pub mod sequence;
pub mod sequence_system;
pub mod stable_id;
pub mod stable_id_index;
pub mod system_context;

pub use command_queue::{CommandQueue, TempEntity};
pub use entity_names::{
    EntityNameIndex, EntityTagIndex, children_of, descendants_of, parent_of, root_of,
};
pub use entity_lookup::{LookupError, find_path, find_tagged};
pub use entity_ref::{deserialize_entity, serialize_entity};
pub use executor::BehaviorExecutor;
pub use game_store::{GameStore, StoreEvent, StoreSnapshot};
pub use game_value::GameValue;
pub use registry::{
    ComponentEntry, ComponentMeta, FieldMeta, FieldType, GameplayRegistry, Phase, QueryError,
    SystemMeta,
};
pub use scene_ownership::SceneOwnership;
pub use scheduler::{Schedule, ScheduleError, build_schedule};
pub use sequence::{Ease, Sequence, SequenceBuilder, SequenceStep};
pub use stable_id::StableId;
pub use stable_id_index::StableIdIndex;
pub use system_context::SystemContext;

// Collect all `#[component]`-registered ComponentEntry items via `inventory`.
inventory::collect!(ComponentEntry);

#[cfg(test)]
mod macro_tests;
