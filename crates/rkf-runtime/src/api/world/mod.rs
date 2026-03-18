//! World — the unified game state container.
//!
//! [`World`] wraps a [`hecs::World`] (the sole authority for game state) and
//! optionally a brick pool for voxelized SDF objects. Users interact with
//! entities via their [`Uuid`] identity.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use uuid::Uuid;

use rkf_core::brick_map::BrickMapAllocator;
use rkf_core::brick_pool::BrickPool;
use rkf_core::WorldPosition;

use super::error::WorldError;
use super::spawn::SpawnBuilder;

mod accessors;
mod camera;
mod components;
mod entity_ops;
mod helpers;
mod hierarchy;
mod node_tree;
mod query;
mod scene_io;
mod scene_management;
mod transforms;

#[cfg(test)]
mod tests;

/// Internal ECS component linking an hecs entity to an SDF scene object.
#[allow(dead_code)]
pub(crate) struct SceneLink {
    pub object_id: u32,
}

/// Record tracking an entity's state in the World.
pub struct EntityRecord {
    /// Every entity gets a parallel hecs entity for component storage.
    pub ecs_entity: hecs::Entity,
    /// SDF object ID (None for ECS-only entities).
    pub sdf_object_id: Option<u32>,
    /// Parent entity UUID for hierarchy.
    pub(crate) parent_id: Option<Uuid>,
    /// Full-precision world position.
    pub(crate) position: WorldPosition,
    /// Rotation.
    pub(crate) rotation: Quat,
    /// Per-axis scale.
    pub(crate) scale: Vec3,
    /// Entity name.
    pub(crate) name: String,
}

/// Metadata wrapping a scene slot for multi-scene support.
struct SceneMeta {
    name: String,
    persistent: bool,
}

/// The unified game state container.
///
/// Wraps a [`hecs::World`] (the sole authority) behind a single API. Users
/// identify entities by [`Uuid`] (from the [`StableId`] component).
pub struct World {
    scenes: Vec<SceneMeta>,
    active_scene: usize,
    next_sdf_id: u32,
    ecs: hecs::World,
    brick_pool: BrickPool,
    brick_map_alloc: BrickMapAllocator,

    // Entity tracking — keyed by StableId UUID
    pub(crate) entities: HashMap<Uuid, EntityRecord>,
    sdf_to_entity: HashMap<u32, Uuid>,
    /// Which scene index each entity was spawned into.
    entity_scene: HashMap<Uuid, usize>,
}

impl World {
    /// Create a new empty world with one default scene.
    pub fn new(name: impl Into<String>) -> Self {
        let mut w = Self {
            scenes: vec![SceneMeta {
                name: name.into(),
                persistent: false,
            }],
            active_scene: 0,
            next_sdf_id: 1, // 0 is reserved as "no object" sentinel for GPU picking
            ecs: hecs::World::new(),
            brick_pool: BrickPool::new(4096),
            brick_map_alloc: BrickMapAllocator::new(),
            entities: HashMap::new(),
            sdf_to_entity: HashMap::new(),
            entity_scene: HashMap::new(),
        };
        w.ensure_scene_environment();
        w
    }

    /// Ensure the singleton scene environment entity exists.
    ///
    /// If no entity with `SceneEnvironment` + `EnvironmentSettings` exists,
    /// spawns one as a proper World entity (with UUID) so it can be targeted
    /// by `SetComponentField` commands.
    pub fn ensure_scene_environment(&mut self) {
        use crate::environment::{EnvironmentSettings, SceneEnvironment};
        let has = self.ecs.query::<&SceneEnvironment>().iter().next().is_some();
        if !has {
            let uuid = self.finalize_ecs_spawn("SceneEnvironment".to_string());
            if let Some(record) = self.entities.get(&uuid) {
                let _ = self.ecs.insert(record.ecs_entity, (SceneEnvironment, EnvironmentSettings::default()));
            }
        }
    }

    /// Find the singleton scene environment hecs entity.
    pub fn scene_environment_entity(&self) -> Option<hecs::Entity> {
        use crate::environment::SceneEnvironment;
        self.ecs.query::<&SceneEnvironment>().iter().next().map(|(e, _)| e)
    }

    /// Find the UUID of the singleton scene environment entity.
    pub fn scene_environment_uuid(&self) -> Option<Uuid> {
        use crate::environment::SceneEnvironment;
        let hecs_entity = self.ecs.query::<&SceneEnvironment>().iter().next().map(|(e, _)| e)?;
        self.entities.iter()
            .find(|(_, r)| r.ecs_entity == hecs_entity)
            .map(|(uuid, _)| *uuid)
    }
}
