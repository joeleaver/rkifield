//! World — the unified game state container.
//!
//! [`World`] wraps a [`Scene`] (SDF objects), a [`hecs::World`] (game logic
//! components), and optionally a brick pool for voxelized SDF objects. Users
//! interact with a single entity handle ([`Entity`]) that transparently spans
//! both storage backends.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use glam::{Quat, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::brick_map::BrickMapAllocator;
use rkf_core::brick_pool::BrickPool;
use rkf_core::scene::{Scene, SceneObject};
use rkf_core::scene_node::{BlendMode, SceneNode, SdfSource, Transform as NodeTransform};
use rkf_core::WorldPosition;

use crate::components::CameraComponent;

use super::entity::Entity;
use super::error::WorldError;
use super::spawn::SpawnBuilder;

/// Internal ECS component linking an hecs entity to an SDF scene object.
#[allow(dead_code)]
pub(crate) struct SceneLink {
    pub object_id: u32,
}

/// Internal record tracking an entity's state.
struct EntityRecord {
    /// Every entity gets a parallel hecs entity for component storage.
    ecs_entity: hecs::Entity,
    /// SDF object ID in the Scene (None for ECS-only entities).
    sdf_object_id: Option<u32>,
    /// Full-precision world position.
    position: WorldPosition,
    /// Rotation.
    rotation: Quat,
    /// Per-axis scale.
    scale: Vec3,
    /// Entity name.
    name: String,
}

/// Metadata wrapping a Scene for multi-scene support.
struct SceneMeta {
    scene: Scene,
    persistent: bool,
}

/// The unified game state container.
///
/// Wraps one or more [`Scene`]s (SDF rendering data) and a [`hecs::World`]
/// (game logic components) behind a single API. Users spawn objects, add
/// components, query, and mutate transforms through one handle type
/// ([`Entity`]).
pub struct World {
    scenes: Vec<SceneMeta>,
    active_scene: usize,
    next_sdf_id: u32,
    ecs: hecs::World,
    brick_pool: BrickPool,
    brick_map_alloc: BrickMapAllocator,

    // Entity tracking
    next_generation: u32,
    entities: HashMap<Entity, EntityRecord>,
    sdf_to_entity: HashMap<u32, Entity>,
    /// Which scene index each entity was spawned into.
    entity_scene: HashMap<Entity, usize>,
}

impl World {
    /// Create a new empty world with one default scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            scenes: vec![SceneMeta {
                scene: Scene::new(name),
                persistent: false,
            }],
            active_scene: 0,
            next_sdf_id: 0,
            ecs: hecs::World::new(),
            brick_pool: BrickPool::new(4096),
            brick_map_alloc: BrickMapAllocator::new(),
            next_generation: 0,
            entities: HashMap::new(),
            sdf_to_entity: HashMap::new(),
            entity_scene: HashMap::new(),
        }
    }

    /// Internal reference to the active scene.
    fn active_scene_ref(&self) -> &Scene {
        &self.scenes[self.active_scene].scene
    }

    /// Internal mutable reference to the active scene.
    fn active_scene_mut(&mut self) -> &mut Scene {
        &mut self.scenes[self.active_scene].scene
    }

    /// Find a SceneObject by ID across all scenes.
    fn find_object_by_id(&self, obj_id: u32) -> Option<&SceneObject> {
        for sm in &self.scenes {
            if let Some(obj) = sm.scene.find_by_id(obj_id) {
                return Some(obj);
            }
        }
        None
    }

    /// Find a mutable SceneObject by ID across all scenes.
    fn find_object_by_id_mut(&mut self, obj_id: u32) -> Option<&mut SceneObject> {
        for sm in &mut self.scenes {
            if let Some(obj) = sm.scene.find_by_id_mut(obj_id) {
                return Some(obj);
            }
        }
        None
    }

    /// Find the scene containing a given object ID.
    fn scene_containing_object(&self, obj_id: u32) -> Option<usize> {
        for (i, sm) in self.scenes.iter().enumerate() {
            if sm.scene.find_by_id(obj_id).is_some() {
                return Some(i);
            }
        }
        None
    }

    // ── Multi-scene management ──────────────────────────────────────────

    /// Create a new empty scene and return its index.
    pub fn create_scene(&mut self, name: impl Into<String>) -> usize {
        let idx = self.scenes.len();
        self.scenes.push(SceneMeta {
            scene: Scene::new(name),
            persistent: false,
        });
        idx
    }

    /// Number of loaded scenes.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Index of the currently active scene.
    pub fn active_scene_index(&self) -> usize {
        self.active_scene
    }

    /// Switch the active scene.
    ///
    /// All spawn operations target the active scene. Existing entities
    /// remain accessible regardless of which scene is active.
    pub fn set_active_scene(&mut self, index: usize) {
        assert!(index < self.scenes.len(), "scene index out of range");
        self.active_scene = index;
    }

    /// Get the name of a scene by index.
    pub fn scene_name(&self, index: usize) -> Option<&str> {
        self.scenes.get(index).map(|s| s.scene.name.as_str())
    }

    /// Iterate over all objects from all loaded scenes.
    ///
    /// The renderer uses this to build the BVH and populate GPU scene data
    /// from every scene, not just the active one.
    pub fn all_objects(&self) -> impl Iterator<Item = &SceneObject> {
        self.scenes.iter().flat_map(|sm| sm.scene.objects.iter())
    }

    /// Total number of SDF objects across all scenes.
    pub fn total_object_count(&self) -> usize {
        self.scenes.iter().map(|sm| sm.scene.object_count()).sum()
    }

    // ── Scene persistence ───────────────────────────────────────────────

    /// Mark a scene as persistent (survives scene swaps).
    pub fn set_scene_persistent(&mut self, index: usize, persistent: bool) {
        if let Some(sm) = self.scenes.get_mut(index) {
            sm.persistent = persistent;
        }
    }

    /// Check if a scene is persistent.
    pub fn is_scene_persistent(&self, index: usize) -> bool {
        self.scenes.get(index).map(|sm| sm.persistent).unwrap_or(false)
    }

    /// Unload all non-persistent scenes, despawning their entities.
    ///
    /// Returns the names of the removed scenes. Adjusts the active scene
    /// index to remain valid (falls back to 0).
    pub fn swap_scenes(&mut self) -> Vec<String> {
        let mut removed_names = Vec::new();

        // Collect indices of non-persistent scenes (reverse order for removal)
        let to_remove: Vec<usize> = self
            .scenes
            .iter()
            .enumerate()
            .filter(|(_, sm)| !sm.persistent)
            .map(|(i, _)| i)
            .collect();

        // Despawn entities belonging to non-persistent scenes
        let entities_to_despawn: Vec<Entity> = self
            .entity_scene
            .iter()
            .filter(|(_, si)| to_remove.contains(si))
            .map(|(e, _)| *e)
            .collect();

        for entity in entities_to_despawn {
            let _ = self.despawn(entity);
        }

        // Remove scenes in reverse order to preserve indices
        for &idx in to_remove.iter().rev() {
            removed_names.push(self.scenes[idx].scene.name.clone());
            self.scenes.remove(idx);
        }

        // If all scenes were removed, create a default empty one
        if self.scenes.is_empty() {
            self.scenes.push(SceneMeta {
                scene: Scene::new("default"),
                persistent: false,
            });
        }

        // Adjust active scene index
        if self.active_scene >= self.scenes.len() {
            self.active_scene = 0;
        }

        removed_names
    }

    /// Remove a scene by index.
    ///
    /// If the scene is persistent, `force` must be true. Cannot remove the
    /// last scene. Despawns all entities belonging to the removed scene.
    pub fn remove_scene(&mut self, index: usize, force: bool) -> Result<String, WorldError> {
        if index >= self.scenes.len() {
            return Err(WorldError::SceneOutOfRange(index));
        }
        if self.scenes.len() == 1 {
            return Err(WorldError::CannotRemoveLastScene);
        }
        if self.scenes[index].persistent && !force {
            return Err(WorldError::Parse(
                "scene is persistent; use force=true".to_string(),
            ));
        }

        // Despawn entities belonging to this scene
        let entities_to_despawn: Vec<Entity> = self
            .entity_scene
            .iter()
            .filter(|(_, si)| **si == index)
            .map(|(e, _)| *e)
            .collect();
        for entity in entities_to_despawn {
            let _ = self.despawn(entity);
        }

        let name = self.scenes[index].scene.name.clone();
        self.scenes.remove(index);

        // Update entity_scene indices for scenes after the removed one
        for si in self.entity_scene.values_mut() {
            if *si > index {
                *si -= 1;
            }
        }

        // Adjust active scene
        if self.active_scene >= self.scenes.len() {
            self.active_scene = self.scenes.len() - 1;
        }

        Ok(name)
    }

    // ── Entity tracking resync ──────────────────────────────────────────────

    /// Resynchronize entity tracking with the contents of all scenes.
    ///
    /// Scans every scene for objects that are not yet tracked in
    /// `sdf_to_entity` and creates entity records for them. Also removes
    /// stale records whose objects no longer exist in any scene.
    ///
    /// Call this after directly mutating a scene (e.g. assigning a demo
    /// scene or loading objects via `scene_mut()`) to bring entity tracking
    /// back in sync.
    pub fn resync_entity_tracking(&mut self) {
        // 1. Register any scene objects not yet tracked.
        for scene_idx in 0..self.scenes.len() {
            for obj_idx in 0..self.scenes[scene_idx].scene.objects.len() {
                let obj = &self.scenes[scene_idx].scene.objects[obj_idx];
                let obj_id = obj.id;

                if self.sdf_to_entity.contains_key(&obj_id) {
                    continue;
                }

                let position = WorldPosition::new(
                    glam::IVec3::ZERO,
                    obj.position,
                );
                let rotation = obj.rotation;
                let scale = obj.scale;
                let name = obj.name.clone();

                let generation = self.next_generation;
                self.next_generation += 1;
                let entity = Entity::sdf(obj_id, generation);

                let ecs_entity = self.ecs.spawn((SceneLink { object_id: obj_id },));

                let record = EntityRecord {
                    ecs_entity,
                    sdf_object_id: Some(obj_id),
                    position,
                    rotation,
                    scale,
                    name,
                };

                self.entities.insert(entity, record);
                self.sdf_to_entity.insert(obj_id, entity);
                self.entity_scene.insert(entity, scene_idx);

                if obj_id >= self.next_sdf_id {
                    self.next_sdf_id = obj_id + 1;
                }
            }
        }

        // 2. Remove stale entries (objects no longer in any scene).
        let all_obj_ids: HashSet<u32> = self
            .scenes
            .iter()
            .flat_map(|s| s.scene.objects.iter().map(|o| o.id))
            .collect();

        let stale_ids: Vec<u32> = self
            .sdf_to_entity
            .keys()
            .copied()
            .filter(|id| !all_obj_ids.contains(id))
            .collect();

        for obj_id in stale_ids {
            if let Some(entity) = self.sdf_to_entity.remove(&obj_id) {
                if let Some(record) = self.entities.remove(&entity) {
                    let _ = self.ecs.despawn(record.ecs_entity);
                }
                self.entity_scene.remove(&entity);
            }
        }
    }

    // ── Spawning ───────────────────────────────────────────────────────────

    /// Begin building a new entity with the given name.
    ///
    /// Call methods on the returned [`SpawnBuilder`] to configure the entity,
    /// then call `.build()` to finalize.
    pub fn spawn(&mut self, name: impl Into<String>) -> SpawnBuilder<'_> {
        SpawnBuilder::new(self, name.into())
    }

    /// Despawn an entity, removing it from the world entirely.
    ///
    /// If the entity has children, they are also despawned recursively.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), WorldError> {
        if !self.entities.contains_key(&entity) {
            return Err(WorldError::NoSuchEntity(entity));
        }

        // Collect children to despawn recursively
        let children: Vec<Entity> = self.children(entity).collect();
        for child in children {
            let _ = self.despawn(child);
        }

        let record = self.entities.remove(&entity).unwrap();

        // Remove from hecs
        let _ = self.ecs.despawn(record.ecs_entity);

        // Remove from Scene if SDF
        if let Some(obj_id) = record.sdf_object_id {
            // Find the scene containing this object and remove from it
            if let Some(scene_idx) = self.scene_containing_object(obj_id) {
                self.scenes[scene_idx].scene.remove_object(obj_id);
            }
            self.sdf_to_entity.remove(&obj_id);
        }
        self.entity_scene.remove(&entity);

        Ok(())
    }

    // ── Internal spawning ──────────────────────────────────────────────────

    /// Finalize an SDF entity spawn (called by SpawnBuilder).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize_sdf_spawn(
        &mut self,
        name: String,
        position: WorldPosition,
        rotation: Quat,
        scale: Vec3,
        root_node: SceneNode,
        material_id: u16,
        blend_mode: Option<rkf_core::scene_node::BlendMode>,
        parent: Option<Entity>,
        aabb: Aabb,
    ) -> Entity {
        let _ = material_id; // Material is already set on the root_node

        let mut obj = SceneObject {
            id: 0, // overwritten by add_object_full
            name: name.clone(),
            parent_id: None,
            position: position.to_vec3(),
            rotation,
            scale,
            root_node,
            aabb,
        };

        // Resolve parent
        if let Some(parent_entity) = parent {
            if let Some(parent_record) = self.entities.get(&parent_entity) {
                obj.parent_id = parent_record.sdf_object_id;
            }
        }

        if let Some(bm) = blend_mode {
            obj.root_node.blend_mode = bm;
        }

        // Use global ID counter to prevent collisions across scenes
        self.active_scene_mut().next_id = self.next_sdf_id;
        let obj_id = self.active_scene_mut().add_object_full(obj);
        self.next_sdf_id = obj_id + 1;

        let generation = self.next_generation;
        self.next_generation += 1;
        let entity = Entity::sdf(obj_id, generation);

        // Create parallel hecs entity
        let ecs_entity = self.ecs.spawn((SceneLink { object_id: obj_id },));

        let record = EntityRecord {
            ecs_entity,
            sdf_object_id: Some(obj_id),
            position,
            rotation,
            scale,
            name,
        };

        self.entities.insert(entity, record);
        self.sdf_to_entity.insert(obj_id, entity);
        self.entity_scene.insert(entity, self.active_scene);

        entity
    }

    /// Finalize an ECS-only entity spawn (called by SpawnBuilder).
    pub(crate) fn finalize_ecs_spawn(&mut self, name: String) -> Entity {
        let ecs_entity = self.ecs.spawn(());
        let generation = self.next_generation;
        self.next_generation += 1;
        let entity = Entity::ecs_only(ecs_entity, generation);

        let record = EntityRecord {
            ecs_entity,
            sdf_object_id: None,
            position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            name,
        };

        self.entities.insert(entity, record);
        self.entity_scene.insert(entity, self.active_scene);
        entity
    }

    // ── Transforms ─────────────────────────────────────────────────────────

    /// Get the world position of an entity.
    pub fn position(&self, entity: Entity) -> Result<WorldPosition, WorldError> {
        self.entities
            .get(&entity)
            .map(|r| r.position)
            .ok_or(WorldError::NoSuchEntity(entity))
    }

    /// Set the world position of an entity.
    pub fn set_position(
        &mut self,
        entity: Entity,
        pos: WorldPosition,
    ) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        record.position = pos;

        // Sync to Scene
        if let Some(obj_id) = record.sdf_object_id {
            if let Some(obj) = self.find_object_by_id_mut(obj_id) {
                obj.position = pos.to_vec3();
            }
        }
        Ok(())
    }

    /// Get the rotation of an entity.
    pub fn rotation(&self, entity: Entity) -> Result<Quat, WorldError> {
        self.entities
            .get(&entity)
            .map(|r| r.rotation)
            .ok_or(WorldError::NoSuchEntity(entity))
    }

    /// Set the rotation of an entity.
    pub fn set_rotation(&mut self, entity: Entity, rot: Quat) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        record.rotation = rot;

        if let Some(obj_id) = record.sdf_object_id {
            if let Some(obj) = self.find_object_by_id_mut(obj_id) {
                obj.rotation = rot;
            }
        }
        Ok(())
    }

    /// Get the scale of an entity.
    pub fn scale(&self, entity: Entity) -> Result<Vec3, WorldError> {
        self.entities
            .get(&entity)
            .map(|r| r.scale)
            .ok_or(WorldError::NoSuchEntity(entity))
    }

    /// Set the per-axis scale of an entity.
    pub fn set_scale(&mut self, entity: Entity, scale: Vec3) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        record.scale = scale;

        if let Some(obj_id) = record.sdf_object_id {
            if let Some(obj) = self.find_object_by_id_mut(obj_id) {
                obj.scale = scale;
            }
        }
        Ok(())
    }

    // ── Hierarchy ──────────────────────────────────────────────────────────

    /// Set the parent of an entity.
    pub fn set_parent(
        &mut self,
        child: Entity,
        parent: Entity,
    ) -> Result<(), WorldError> {
        let child_record = self
            .entities
            .get(&child)
            .ok_or(WorldError::NoSuchEntity(child))?;
        let parent_record = self
            .entities
            .get(&parent)
            .ok_or(WorldError::NoSuchEntity(parent))?;

        // Both must be SDF entities for Scene hierarchy
        match (child_record.sdf_object_id, parent_record.sdf_object_id) {
            (Some(child_id), Some(parent_id)) => {
                // Find the scene containing the child and reparent within it
                if let Some(scene_idx) = self.scene_containing_object(child_id) {
                    if !self.scenes[scene_idx].scene.reparent(child_id, Some(parent_id)) {
                        return Err(WorldError::CycleDetected);
                    }
                }
                Ok(())
            }
            _ => {
                // ECS-only entities don't participate in Scene hierarchy
                // For now, we don't support parenting ECS-only entities
                Err(WorldError::CycleDetected)
            }
        }
    }

    /// Remove an entity from its parent (make it a root).
    pub fn unparent(&mut self, entity: Entity) -> Result<(), WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;

        if let Some(obj_id) = record.sdf_object_id {
            if let Some(scene_idx) = self.scene_containing_object(obj_id) {
                self.scenes[scene_idx].scene.reparent(obj_id, None);
            }
        }
        Ok(())
    }

    /// Iterate over the children of an entity.
    pub fn children(&self, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
        let sdf_obj_id = self
            .entities
            .get(&entity)
            .and_then(|r| r.sdf_object_id);

        // Collect child SDF object IDs across all scenes
        let child_ids: Vec<u32> = if let Some(parent_id) = sdf_obj_id {
            self.scenes
                .iter()
                .flat_map(|sm| sm.scene.children_of(parent_id).map(|o| o.id))
                .collect()
        } else {
            Vec::new()
        };

        child_ids
            .into_iter()
            .filter_map(move |obj_id| self.sdf_to_entity.get(&obj_id).copied())
    }

    /// Get the parent of an entity, if it has one.
    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        let record = self.entities.get(&entity)?;
        let obj_id = record.sdf_object_id?;
        let obj = self.find_object_by_id(obj_id)?;
        let parent_id = obj.parent_id?;
        self.sdf_to_entity.get(&parent_id).copied()
    }

    // ── Query ──────────────────────────────────────────────────────────────

    /// Find the first entity with the given name.
    pub fn find(&self, name: &str) -> Option<Entity> {
        self.entities
            .iter()
            .find(|(_, r)| r.name == name)
            .map(|(e, _)| *e)
    }

    /// Find all entities with the given name.
    pub fn find_all(&self, name: &str) -> Vec<Entity> {
        self.entities
            .iter()
            .filter(|(_, r)| r.name == name)
            .map(|(e, _)| *e)
            .collect()
    }

    /// Iterate over all live entities.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.keys().copied()
    }

    /// Get the name of an entity.
    pub fn name(&self, entity: Entity) -> Result<&str, WorldError> {
        self.entities
            .get(&entity)
            .map(|r| r.name.as_str())
            .ok_or(WorldError::NoSuchEntity(entity))
    }

    /// Check if an entity is still alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.contains_key(&entity)
    }

    /// Number of live entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Remove all entities from the world.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.sdf_to_entity.clear();
        self.entity_scene.clear();
        self.ecs = hecs::World::new();
        let name = self.active_scene_ref().name.clone();
        self.scenes = vec![SceneMeta {
            scene: Scene::new(name),
            persistent: false,
        }];
        self.active_scene = 0;
        self.next_sdf_id = 0;
    }

    // ── ECS Components ─────────────────────────────────────────────────────

    /// Get a shared reference to a component on an entity.
    pub fn get<C: hecs::Component>(&self, entity: Entity) -> Result<hecs::Ref<'_, C>, WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        self.ecs
            .get::<&C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity, std::any::type_name::<C>()))
    }

    /// Get a mutable reference to a component on an entity.
    pub fn get_mut<C: hecs::Component>(
        &self,
        entity: Entity,
    ) -> Result<hecs::RefMut<'_, C>, WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        self.ecs
            .get::<&mut C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity, std::any::type_name::<C>()))
    }

    /// Insert a component on an entity (overwrites if already present).
    pub fn insert<C: hecs::Component>(
        &mut self,
        entity: Entity,
        component: C,
    ) -> Result<(), WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        self.ecs
            .insert_one(record.ecs_entity, component)
            .map_err(|_| WorldError::NoSuchEntity(entity))
    }

    /// Remove a component from an entity, returning it.
    pub fn remove<C: hecs::Component>(&mut self, entity: Entity) -> Result<C, WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        self.ecs
            .remove_one::<C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity, std::any::type_name::<C>()))
    }

    /// Check if an entity has a component.
    pub fn has<C: hecs::Component>(&self, entity: Entity) -> bool {
        if let Some(record) = self.entities.get(&entity) {
            self.ecs.get::<&C>(record.ecs_entity).is_ok()
        } else {
            false
        }
    }

    // ── Scene I/O ──────────────────────────────────────────────────────────

    /// Load entities from a `.rkscene` file into a specific or new scene.
    ///
    /// - `target_scene: None` — creates a new scene and loads into it.
    /// - `target_scene: Some(idx)` — loads into the existing scene at `idx`.
    ///
    /// Returns `(scene_index, loaded_entities)`.
    pub fn load_scene_into(
        &mut self,
        path: impl AsRef<Path>,
        target_scene: Option<usize>,
    ) -> Result<(usize, Vec<Entity>), WorldError> {
        let path_ref = path.as_ref();
        let scene_name = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("loaded")
            .to_string();

        let scene_idx = match target_scene {
            Some(idx) => {
                if idx >= self.scenes.len() {
                    return Err(WorldError::SceneOutOfRange(idx));
                }
                idx
            }
            None => self.create_scene(&scene_name),
        };

        let prev_active = self.active_scene;
        self.active_scene = scene_idx;
        let result = self.load_scene(path_ref);
        self.active_scene = prev_active;

        result.map(|entities| (scene_idx, entities))
    }

    /// Save the entities in a specific scene to a `.rkscene` file.
    pub fn save_scene_at(
        &self,
        index: usize,
        path: impl AsRef<Path>,
    ) -> Result<(), WorldError> {
        if index >= self.scenes.len() {
            return Err(WorldError::SceneOutOfRange(index));
        }

        use crate::scene_file::{ObjectEntry, SceneFile};
        use rkf_core::scene_node::SdfSource;

        let scene = &self.scenes[index].scene;
        let mut objects = Vec::new();

        for obj in &scene.objects {
            let entity = self.sdf_to_entity.get(&obj.id);
            let record = entity.and_then(|e| self.entities.get(e));

            let (analytical, analytical_params, material_id) =
                match &obj.root_node.sdf_source {
                    SdfSource::Analytical {
                        primitive,
                        material_id,
                    } => {
                        let (name, params) = primitive_to_analytical(primitive);
                        (Some(name), Some(params), Some(*material_id))
                    }
                    _ => (None, None, None),
                };

            let pos = record
                .map(|r| r.position)
                .unwrap_or_default();
            let pos_f64 = [
                pos.chunk.x as f64 * 8.0 + pos.local.x as f64,
                pos.chunk.y as f64 * 8.0 + pos.local.y as f64,
                pos.chunk.z as f64 * 8.0 + pos.local.z as f64,
            ];

            objects.push(ObjectEntry {
                name: obj.name.clone(),
                asset_path: None,
                position: pos_f64,
                rotation: [
                    obj.rotation.x,
                    obj.rotation.y,
                    obj.rotation.z,
                    obj.rotation.w,
                ],
                scale: [obj.scale.x, obj.scale.y, obj.scale.z],
                material_id,
                analytical,
                analytical_params,
                importance_bias: 1.0,
            });
        }

        let scene_file = SceneFile {
            name: scene.name.clone(),
            objects,
            cameras: Vec::new(),
            lights: Vec::new(),
            environment: None,
            material_palette: None,
            properties: std::collections::HashMap::new(),
        };

        let path_str = path.as_ref().to_str().unwrap_or("");
        crate::scene_file::save_scene_file(path_str, &scene_file)
            .map_err(|e| WorldError::Io(std::io::Error::other(e.to_string())))
    }

    /// Load entities from a `.rkscene` file, adding them to the active scene.
    ///
    /// Returns the entities that were loaded.
    pub fn load_scene(&mut self, path: impl AsRef<Path>) -> Result<Vec<Entity>, WorldError> {
        let path_str = path.as_ref().to_str().unwrap_or("");
        let scene_file = crate::scene_file::load_scene_file(path_str)
            .map_err(|e| WorldError::Parse(e.to_string()))?;

        let mut loaded_entities = Vec::new();

        for obj_entry in &scene_file.objects {
            let position = WorldPosition::from_world_f64(
                obj_entry.position[0],
                obj_entry.position[1],
                obj_entry.position[2],
            );
            let rotation = Quat::from_xyzw(
                obj_entry.rotation[0],
                obj_entry.rotation[1],
                obj_entry.rotation[2],
                obj_entry.rotation[3],
            );
            let scale = Vec3::new(
                obj_entry.scale[0],
                obj_entry.scale[1],
                obj_entry.scale[2],
            );

            // Build SDF node from analytical params or default
            let root_node = if let Some(ref analytical_type) = obj_entry.analytical {
                let primitive = parse_analytical_primitive(
                    analytical_type,
                    obj_entry.analytical_params.as_deref(),
                );
                let mat_id = obj_entry.material_id.unwrap_or(0);
                SceneNode::analytical(&obj_entry.name, primitive, mat_id)
            } else {
                SceneNode::new(&obj_entry.name)
            };

            let entity = self.finalize_sdf_spawn(
                obj_entry.name.clone(),
                position,
                rotation,
                scale,
                root_node,
                obj_entry.material_id.unwrap_or(0),
                None,
                None,
                Aabb::new(Vec3::ZERO, Vec3::ZERO),
            );
            loaded_entities.push(entity);
        }

        Ok(loaded_entities)
    }

    /// Save the world's entities to a `.rkscene` file.
    pub fn save_scene(&self, path: impl AsRef<Path>) -> Result<(), WorldError> {
        use crate::scene_file::{ObjectEntry, SceneFile};
        use rkf_core::scene_node::SdfSource;

        let mut objects = Vec::new();

        for (entity, record) in &self.entities {
            if let Some(obj_id) = record.sdf_object_id {
                if let Some(obj) = self.find_object_by_id(obj_id) {
                    let (analytical, analytical_params, material_id) =
                        match &obj.root_node.sdf_source {
                            SdfSource::Analytical {
                                primitive,
                                material_id,
                            } => {
                                let (name, params) = primitive_to_analytical(primitive);
                                (Some(name), Some(params), Some(*material_id))
                            }
                            _ => (None, None, None),
                        };

                    let pos = record.position;
                    let pos_f64 = [
                        pos.chunk.x as f64 * 8.0 + pos.local.x as f64,
                        pos.chunk.y as f64 * 8.0 + pos.local.y as f64,
                        pos.chunk.z as f64 * 8.0 + pos.local.z as f64,
                    ];

                    objects.push(ObjectEntry {
                        name: record.name.clone(),
                        asset_path: None,
                        position: pos_f64,
                        rotation: [
                            obj.rotation.x,
                            obj.rotation.y,
                            obj.rotation.z,
                            obj.rotation.w,
                        ],
                        scale: [obj.scale.x, obj.scale.y, obj.scale.z],
                        material_id,
                        analytical,
                        analytical_params,
                        importance_bias: 1.0,
                    });
                }
            }
            let _ = entity; // suppress unused warning
        }

        let scene_file = SceneFile {
            name: self.active_scene_ref().name.clone(),
            objects,
            cameras: Vec::new(),
            lights: Vec::new(),
            environment: None,
            material_palette: None,
            properties: std::collections::HashMap::new(),
        };

        let path_str = path.as_ref().to_str().unwrap_or("");
        crate::scene_file::save_scene_file(path_str, &scene_file)
            .map_err(|e| WorldError::Io(std::io::Error::other(e.to_string())))
    }

    // ── Internal accessors ─────────────────────────────────────────────────

    /// Get a reference to the active Scene.
    ///
    /// Used by the Renderer internally and by the editor for snapshot cloning.
    pub fn scene(&self) -> &Scene {
        self.active_scene_ref()
    }

    /// Get a reference to the brick pool.
    pub(crate) fn brick_pool(&self) -> &BrickPool {
        &self.brick_pool
    }

    /// Get a reference to the brick map allocator.
    pub(crate) fn brick_map_alloc(&self) -> &BrickMapAllocator {
        &self.brick_map_alloc
    }

    /// Get the hecs entity for an API entity (for SpawnBuilder component insertion).
    pub(crate) fn ecs_entity_for(&self, entity: Entity) -> Option<hecs::Entity> {
        self.entities.get(&entity).map(|r| r.ecs_entity)
    }

    /// Get a mutable reference to the hecs world.
    pub(crate) fn ecs_mut(&mut self) -> &mut hecs::World {
        &mut self.ecs
    }

    // ── Advanced access ─────────────────────────────────────────────────────

    /// Get a mutable reference to the underlying Scene.
    ///
    /// Advanced: use this for animation updates, custom voxelization, or
    /// other operations not covered by the standard World API.
    pub fn scene_mut(&mut self) -> &mut Scene {
        self.active_scene_mut()
    }

    /// Voxelize an SDF function into this world's brick pool.
    ///
    /// Returns the brick map handle and brick count on success.
    pub fn voxelize<F>(
        &mut self,
        sdf_fn: F,
        aabb: &Aabb,
        voxel_size: f32,
    ) -> Result<(rkf_core::BrickMapHandle, u32), WorldError>
    where
        F: Fn(Vec3) -> (f32, u16),
    {
        rkf_core::voxelize_sdf(
            sdf_fn,
            aabb,
            voxel_size,
            &mut self.brick_pool,
            &mut self.brick_map_alloc,
        )
        .ok_or(WorldError::Voxelize("voxelization produced no bricks".into()))
    }

    /// Get a mutable reference to the root SDF scene node of an entity.
    ///
    /// Advanced: use this for per-frame animation updates that modify the
    /// scene node tree (e.g. skeletal animation bone transforms).
    pub fn root_node_mut(&mut self, entity: Entity) -> Result<&mut SceneNode, WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        let sdf_id = record
            .sdf_object_id
            .ok_or(WorldError::MissingComponent(entity, "SdfObject"))?;
        let obj = self
            .find_object_by_id_mut(sdf_id)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        Ok(&mut obj.root_node)
    }

    // ── Node tree access ────────────────────────────────────────────────

    /// Get a reference to the root scene node of an entity.
    pub fn root_node(&self, entity: Entity) -> Result<&SceneNode, WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        let sdf_id = record
            .sdf_object_id
            .ok_or(WorldError::MissingComponent(entity, "SdfObject"))?;
        let obj = self
            .find_object_by_id(sdf_id)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        Ok(&obj.root_node)
    }

    /// Find a node by name within an entity's scene node tree.
    pub fn find_node(&self, entity: Entity, name: &str) -> Result<&SceneNode, WorldError> {
        let root = self.root_node(entity)?;
        root.find_by_name(name)
            .ok_or_else(|| WorldError::NodeNotFound(name.to_string()))
    }

    /// Find a node by name (mutable) within an entity's scene node tree.
    pub fn find_node_mut(
        &mut self,
        entity: Entity,
        name: &str,
    ) -> Result<&mut SceneNode, WorldError> {
        let root = self.root_node_mut(entity)?;
        root.find_by_name_mut(name)
            .ok_or_else(|| WorldError::NodeNotFound(name.to_string()))
    }

    /// Find a node by slash-separated path within an entity's scene node tree.
    pub fn find_node_by_path(
        &self,
        entity: Entity,
        path: &str,
    ) -> Result<&SceneNode, WorldError> {
        let root = self.root_node(entity)?;
        root.find_by_path(path)
            .ok_or_else(|| WorldError::NodeNotFound(path.to_string()))
    }

    /// Find a node by slash-separated path (mutable) within an entity's scene node tree.
    pub fn find_node_by_path_mut(
        &mut self,
        entity: Entity,
        path: &str,
    ) -> Result<&mut SceneNode, WorldError> {
        let root = self.root_node_mut(entity)?;
        root.find_by_path_mut(path)
            .ok_or_else(|| WorldError::NodeNotFound(path.to_string()))
    }

    /// Count the total number of nodes in an entity's scene node tree.
    pub fn node_count(&self, entity: Entity) -> Result<usize, WorldError> {
        let root = self.root_node(entity)?;
        Ok(root.node_count())
    }

    // ── Node tree mutation ──────────────────────────────────────────────

    /// Set the local transform of a named node within an entity's tree.
    pub fn set_node_transform(
        &mut self,
        entity: Entity,
        node_name: &str,
        transform: NodeTransform,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity, node_name)?;
        node.local_transform = transform;
        Ok(())
    }

    /// Add a child node to a named parent node within an entity's tree.
    pub fn add_child_node(
        &mut self,
        entity: Entity,
        parent_name: &str,
        child: SceneNode,
    ) -> Result<(), WorldError> {
        let parent = self.find_node_mut(entity, parent_name)?;
        parent.add_child(child);
        Ok(())
    }

    /// Remove a named node from an entity's tree, returning it.
    ///
    /// Searches the tree for the node's parent and removes it from the
    /// parent's children. Cannot remove the root node itself.
    pub fn remove_child_node(
        &mut self,
        entity: Entity,
        node_name: &str,
    ) -> Result<SceneNode, WorldError> {
        let root = self.root_node_mut(entity)?;
        // Cannot remove root itself
        if root.name == node_name {
            return Err(WorldError::NodeNotFound(format!(
                "cannot remove root node '{}'",
                node_name
            )));
        }
        remove_named_child(root, node_name)
            .ok_or_else(|| WorldError::NodeNotFound(node_name.to_string()))
    }

    /// Set the blend mode of a named node within an entity's tree.
    pub fn set_node_blend_mode(
        &mut self,
        entity: Entity,
        node_name: &str,
        mode: BlendMode,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity, node_name)?;
        node.blend_mode = mode;
        Ok(())
    }

    /// Set the SDF source of a named node within an entity's tree.
    pub fn set_node_sdf_source(
        &mut self,
        entity: Entity,
        node_name: &str,
        source: SdfSource,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity, node_name)?;
        node.sdf_source = source;
        Ok(())
    }

    // ── Camera entities ─────────────────────────────────────────────────

    /// Spawn a camera entity (ECS-only, no SDF geometry).
    pub fn spawn_camera(
        &mut self,
        label: impl Into<String>,
        position: WorldPosition,
        yaw: f32,
        pitch: f32,
        fov_degrees: f32,
    ) -> Entity {
        let label = label.into();
        let entity = self.finalize_ecs_spawn(label.clone());
        let cam = CameraComponent {
            fov_degrees,
            active: false,
            label,
            yaw,
            pitch,
            ..Default::default()
        };
        let record = self.entities.get(&entity).unwrap();
        let _ = self.ecs.insert_one(record.ecs_entity, cam);
        // Store position
        if let Some(r) = self.entities.get_mut(&entity) {
            r.position = position;
        }
        entity
    }

    /// List all camera entities.
    pub fn cameras(&self) -> Vec<Entity> {
        self.entities
            .iter()
            .filter(|(_, r)| {
                self.ecs.get::<&CameraComponent>(r.ecs_entity).is_ok()
            })
            .map(|(e, _)| *e)
            .collect()
    }

    /// Find the active camera entity, if any.
    pub fn active_camera(&self) -> Option<Entity> {
        self.entities
            .iter()
            .find(|(_, r)| {
                self.ecs
                    .get::<&CameraComponent>(r.ecs_entity)
                    .map(|c| c.active)
                    .unwrap_or(false)
            })
            .map(|(e, _)| *e)
    }

    /// Set a camera entity as the active camera (deactivates all others).
    pub fn set_active_camera(&mut self, entity: Entity) -> Result<(), WorldError> {
        let record = self
            .entities
            .get(&entity)
            .ok_or(WorldError::NoSuchEntity(entity))?;
        // Verify it has a CameraComponent
        if self.ecs.get::<&CameraComponent>(record.ecs_entity).is_err() {
            return Err(WorldError::MissingComponent(entity, "CameraComponent"));
        }

        // Deactivate all cameras
        let ecs_entities: Vec<hecs::Entity> = self
            .entities
            .values()
            .filter(|r| self.ecs.get::<&CameraComponent>(r.ecs_entity).is_ok())
            .map(|r| r.ecs_entity)
            .collect();
        for ee in ecs_entities {
            if let Ok(mut cam) = self.ecs.get::<&mut CameraComponent>(ee) {
                cam.active = false;
            }
        }

        // Activate target
        let record = self.entities.get(&entity).unwrap();
        if let Ok(mut cam) = self.ecs.get::<&mut CameraComponent>(record.ecs_entity) {
            cam.active = true;
        }
        Ok(())
    }

    // ── Automation ID conversion ───────────────────────────────────────

    /// Look up an entity by its automation-API u64 identifier.
    ///
    /// SDF objects (bit 63 = 0) are looked up by object_id.
    /// ECS-only entities (bit 63 = 1) are looked up by hecs entity id.
    pub fn find_entity_by_id(&self, id: u64) -> Option<Entity> {
        if id & (1u64 << 63) != 0 {
            // ECS-only entity
            let hecs_id = (id & !(1u64 << 63)) as u32;
            self.entities
                .iter()
                .find(|(_, r)| r.ecs_entity.id() == hecs_id)
                .map(|(e, _)| *e)
        } else {
            // SDF object
            self.sdf_to_entity.get(&(id as u32)).copied()
        }
    }
}

// ── Node tree helpers ──────────────────────────────────────────────────────

/// Recursively find and remove a child with the given name from the tree.
/// Returns the removed node, or None if not found.
fn remove_named_child(parent: &mut SceneNode, name: &str) -> Option<SceneNode> {
    // Check direct children first
    if let Some(pos) = parent.children.iter().position(|c| c.name == name) {
        return Some(parent.children.remove(pos));
    }
    // Recurse into children
    for child in &mut parent.children {
        if let Some(removed) = remove_named_child(child, name) {
            return Some(removed);
        }
    }
    None
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn parse_analytical_primitive(
    name: &str,
    params: Option<&[f32]>,
) -> rkf_core::scene_node::SdfPrimitive {
    use rkf_core::scene_node::SdfPrimitive;

    match name.to_lowercase().as_str() {
        "sphere" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.5);
            SdfPrimitive::Sphere { radius }
        }
        "box" => {
            let half = params
                .map(|p| {
                    Vec3::new(
                        *p.first().unwrap_or(&0.5),
                        *p.get(1).unwrap_or(&0.5),
                        *p.get(2).unwrap_or(&0.5),
                    )
                })
                .unwrap_or(Vec3::splat(0.5));
            SdfPrimitive::Box {
                half_extents: half,
            }
        }
        "capsule" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.2);
            let half_height = params.and_then(|p| p.get(1).copied()).unwrap_or(0.5);
            SdfPrimitive::Capsule {
                radius,
                half_height,
            }
        }
        "torus" => {
            let major = params.and_then(|p| p.first().copied()).unwrap_or(0.5);
            let minor = params.and_then(|p| p.get(1).copied()).unwrap_or(0.1);
            SdfPrimitive::Torus {
                major_radius: major,
                minor_radius: minor,
            }
        }
        "cylinder" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.3);
            let half_height = params.and_then(|p| p.get(1).copied()).unwrap_or(0.5);
            SdfPrimitive::Cylinder {
                radius,
                half_height,
            }
        }
        "plane" => {
            let normal = params
                .map(|p| {
                    Vec3::new(
                        *p.first().unwrap_or(&0.0),
                        *p.get(1).unwrap_or(&1.0),
                        *p.get(2).unwrap_or(&0.0),
                    )
                })
                .unwrap_or(Vec3::Y);
            let distance = params.and_then(|p| p.get(3).copied()).unwrap_or(0.0);
            SdfPrimitive::Plane { normal, distance }
        }
        _ => SdfPrimitive::Sphere { radius: 0.5 },
    }
}

fn primitive_to_analytical(
    primitive: &rkf_core::scene_node::SdfPrimitive,
) -> (String, Vec<f32>) {
    use rkf_core::scene_node::SdfPrimitive;

    match primitive {
        SdfPrimitive::Sphere { radius } => ("sphere".to_string(), vec![*radius]),
        SdfPrimitive::Box { half_extents } => (
            "box".to_string(),
            vec![half_extents.x, half_extents.y, half_extents.z],
        ),
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => ("capsule".to_string(), vec![*radius, *half_height]),
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => ("torus".to_string(), vec![*major_radius, *minor_radius]),
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => ("cylinder".to_string(), vec![*radius, *half_height]),
        SdfPrimitive::Plane { normal, distance } => (
            "plane".to_string(),
            vec![normal.x, normal.y, normal.z, *distance],
        ),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::scene_node::SdfPrimitive;

    // ── World core ─────────────────────────────────────────────────────────

    #[test]
    fn world_new_empty() {
        let world = World::new("test");
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn spawn_returns_entity() {
        let mut world = World::new("test");
        let e = world
            .spawn("cube")
            .sdf(SdfPrimitive::Box {
                half_extents: Vec3::splat(0.5),
            })
            .material(1)
            .build();
        assert!(world.is_alive(e));
    }

    #[test]
    fn spawn_increments_count() {
        let mut world = World::new("test");
        world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.entity_count(), 2);
    }

    #[test]
    fn despawn_removes_entity() {
        let mut world = World::new("test");
        let e = world
            .spawn("cube")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.despawn(e).unwrap();
        assert!(!world.is_alive(e));
        assert_eq!(world.entity_count(), 0);
    }

    #[test]
    fn despawn_invalid_entity_errors() {
        let mut world = World::new("test");
        let e = Entity::sdf(999, 0);
        assert!(matches!(
            world.despawn(e),
            Err(WorldError::NoSuchEntity(_))
        ));
    }

    #[test]
    fn double_despawn_errors() {
        let mut world = World::new("test");
        let e = world
            .spawn("cube")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.despawn(e).unwrap();
        assert!(matches!(
            world.despawn(e),
            Err(WorldError::NoSuchEntity(_))
        ));
    }

    #[test]
    fn name_round_trip() {
        let mut world = World::new("test");
        let e = world
            .spawn("my_cube")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.name(e).unwrap(), "my_cube");
    }

    #[test]
    fn find_by_name() {
        let mut world = World::new("test");
        let e = world
            .spawn("target")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.find("target"), Some(e));
    }

    #[test]
    fn find_by_name_missing() {
        let world = World::new("test");
        assert_eq!(world.find("nope"), None);
    }

    #[test]
    fn find_all_multiple() {
        let mut world = World::new("test");
        world
            .spawn("dup")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("dup")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("other")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.find_all("dup").len(), 2);
    }

    #[test]
    fn entities_iterator() {
        let mut world = World::new("test");
        world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.entities().count(), 2);
    }

    #[test]
    fn clear_removes_all() {
        let mut world = World::new("test");
        world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.clear();
        assert_eq!(world.entity_count(), 0);
    }

    // ── Transforms ─────────────────────────────────────────────────────────

    #[test]
    fn position_default_origin() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let pos = world.position(e).unwrap();
        assert_eq!(pos, WorldPosition::default());
    }

    #[test]
    fn set_position_read_back() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let target = WorldPosition::new(glam::IVec3::new(1, 0, 0), Vec3::new(3.0, 2.0, 1.0));
        world.set_position(e, target).unwrap();
        assert_eq!(world.position(e).unwrap(), target);
    }

    #[test]
    fn rotation_default_identity() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let rot = world.rotation(e).unwrap();
        assert!((rot - Quat::IDENTITY).length() < 1e-5);
    }

    #[test]
    fn set_rotation_read_back() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let target = Quat::from_rotation_y(1.5);
        world.set_rotation(e, target).unwrap();
        let got = world.rotation(e).unwrap();
        assert!((got - target).length() < 1e-5);
    }

    #[test]
    fn scale_default_one() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.scale(e).unwrap(), Vec3::ONE);
    }

    #[test]
    fn set_scale_read_back() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let target = Vec3::new(2.0, 3.0, 4.0);
        world.set_scale(e, target).unwrap();
        assert_eq!(world.scale(e).unwrap(), target);
    }

    #[test]
    fn transform_on_despawned_errors() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.despawn(e).unwrap();
        assert!(matches!(
            world.position(e),
            Err(WorldError::NoSuchEntity(_))
        ));
    }

    // ── Hierarchy ──────────────────────────────────────────────────────────

    #[test]
    fn set_parent_establishes_relationship() {
        let mut world = World::new("test");
        let parent = world
            .spawn("parent")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let child = world
            .spawn("child")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .build();
        world.set_parent(child, parent).unwrap();
        assert_eq!(world.parent(child), Some(parent));
    }

    #[test]
    fn unparent_removes_relationship() {
        let mut world = World::new("test");
        let parent = world
            .spawn("parent")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let child = world
            .spawn("child")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .build();
        world.set_parent(child, parent).unwrap();
        world.unparent(child).unwrap();
        assert_eq!(world.parent(child), None);
    }

    #[test]
    fn children_lists_children() {
        let mut world = World::new("test");
        let parent = world
            .spawn("parent")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let child1 = world
            .spawn("c1")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .build();
        let child2 = world
            .spawn("c2")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .build();
        world.set_parent(child1, parent).unwrap();
        world.set_parent(child2, parent).unwrap();

        let children: Vec<Entity> = world.children(parent).collect();
        assert_eq!(children.len(), 2);
        assert!(children.contains(&child1));
        assert!(children.contains(&child2));
    }

    #[test]
    fn children_empty_for_leaf() {
        let mut world = World::new("test");
        let e = world
            .spawn("leaf")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.children(e).count(), 0);
    }

    #[test]
    fn despawn_parent_despawns_children() {
        let mut world = World::new("test");
        let parent = world
            .spawn("parent")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let child = world
            .spawn("child")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .build();
        world.set_parent(child, parent).unwrap();
        world.despawn(parent).unwrap();
        assert!(!world.is_alive(child));
    }

    #[test]
    fn cycle_detection() {
        let mut world = World::new("test");
        let a = world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let b = world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.set_parent(b, a).unwrap();
        assert!(matches!(
            world.set_parent(a, b),
            Err(WorldError::CycleDetected)
        ));
    }

    // ── SpawnBuilder ───────────────────────────────────────────────────────

    #[test]
    fn spawn_with_sdf_primitive() {
        let mut world = World::new("test");
        let e = world
            .spawn("sphere")
            .sdf(SdfPrimitive::Sphere { radius: 1.0 })
            .material(3)
            .build();
        assert!(world.is_alive(e));
        assert_eq!(world.name(e).unwrap(), "sphere");
    }

    #[test]
    fn spawn_with_sdf_tree() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::analytical(
            "child",
            SdfPrimitive::Sphere { radius: 0.3 },
            1,
        ));
        let e = world.spawn("composite").sdf_tree(root).build();
        assert!(world.is_alive(e));
    }

    #[test]
    fn spawn_with_position() {
        let mut world = World::new("test");
        let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(5.0, 2.0, -3.0));
        let e = world
            .spawn("obj")
            .position(pos)
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.position(e).unwrap(), pos);
    }

    #[test]
    fn spawn_with_position_vec3() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .position_vec3(Vec3::new(1.0, 2.0, 3.0))
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let pos = world.position(e).unwrap();
        // Should be normalized
        assert!((pos.to_vec3() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn spawn_with_rotation() {
        let mut world = World::new("test");
        let rot = Quat::from_rotation_z(1.0);
        let e = world
            .spawn("obj")
            .rotation(rot)
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let got = world.rotation(e).unwrap();
        assert!((got - rot).length() < 1e-5);
    }

    #[test]
    fn spawn_with_scale() {
        let mut world = World::new("test");
        let s = Vec3::new(2.0, 3.0, 4.0);
        let e = world
            .spawn("obj")
            .scale(s)
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.scale(e).unwrap(), s);
    }

    #[test]
    fn spawn_with_material() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(42)
            .build();
        // Material is set on the scene node — verify via Scene
        let record = world.entities.get(&e).unwrap();
        let obj = world.scene().find_by_id(record.sdf_object_id.unwrap()).unwrap();
        match &obj.root_node.sdf_source {
            rkf_core::scene_node::SdfSource::Analytical { material_id, .. } => {
                assert_eq!(*material_id, 42);
            }
            _ => panic!("expected analytical source"),
        }
    }

    #[test]
    fn spawn_with_blend_mode() {
        use rkf_core::scene_node::BlendMode;
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .blend(BlendMode::Subtract)
            .build();
        let record = world.entities.get(&e).unwrap();
        let obj = world.scene().find_by_id(record.sdf_object_id.unwrap()).unwrap();
        assert!(matches!(obj.root_node.blend_mode, BlendMode::Subtract));
    }

    #[test]
    fn spawn_with_parent() {
        let mut world = World::new("test");
        let parent = world
            .spawn("parent")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let child = world
            .spawn("child")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(1)
            .parent(parent)
            .build();
        assert_eq!(world.parent(child), Some(parent));
    }

    #[test]
    fn spawn_with_component() {
        #[derive(Debug, PartialEq)]
        struct Velocity(Vec3);

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .with(Velocity(Vec3::new(1.0, 0.0, 0.0)))
            .build();
        let vel = world.get::<Velocity>(e).unwrap();
        assert_eq!(*vel, Velocity(Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn spawn_without_sdf() {
        let mut world = World::new("test");
        let e = world.spawn("ecs_only").build();
        assert!(world.is_alive(e));
        // No SDF object in scene
        assert_eq!(world.scene().object_count(), 0);
    }

    // ── ECS Components ─────────────────────────────────────────────────────

    #[test]
    fn insert_and_get() {
        #[derive(Debug, PartialEq)]
        struct Health(i32);

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.insert(e, Health(100)).unwrap();
        let h = world.get::<Health>(e).unwrap();
        assert_eq!(*h, Health(100));
    }

    #[test]
    fn insert_replaces_existing() {
        #[derive(Debug, PartialEq)]
        struct Health(i32);

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.insert(e, Health(100)).unwrap();
        world.insert(e, Health(50)).unwrap();
        let h = world.get::<Health>(e).unwrap();
        assert_eq!(*h, Health(50));
    }

    #[test]
    fn get_missing_errors() {
        #[derive(Debug)]
        struct Missing;

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert!(matches!(
            world.get::<Missing>(e),
            Err(WorldError::MissingComponent(_, _))
        ));
    }

    #[test]
    fn remove_returns_component() {
        #[derive(Debug, PartialEq)]
        struct Tag(u32);

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.insert(e, Tag(42)).unwrap();
        let removed = world.remove::<Tag>(e).unwrap();
        assert_eq!(removed, Tag(42));
        assert!(!world.has::<Tag>(e));
    }

    #[test]
    fn has_returns_true() {
        #[derive(Debug)]
        struct Marker;

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.insert(e, Marker).unwrap();
        assert!(world.has::<Marker>(e));
    }

    #[test]
    fn has_returns_false() {
        #[derive(Debug)]
        struct Marker;

        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert!(!world.has::<Marker>(e));
    }

    // ── Per-scene load/save (C.4) ────────────────────────────────────────

    #[test]
    fn load_scene_into_new_creates_scene() {
        let dir = std::env::temp_dir().join("rkf_api_test_load_into.rkscene");
        let path = dir.to_str().unwrap();

        let mut source = World::new("src");
        source
            .spawn("sphere")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        source.save_scene(path).unwrap();

        let mut world = World::new("main");
        let (idx, entities) = world.load_scene_into(path, None).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(entities.len(), 1);
        assert_eq!(world.scene_count(), 2);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_scene_into_existing_appends() {
        let dir = std::env::temp_dir().join("rkf_api_test_load_existing.rkscene");
        let path = dir.to_str().unwrap();

        let mut source = World::new("src");
        source
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        source.save_scene(path).unwrap();

        let mut world = World::new("main");
        world
            .spawn("existing")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let (idx, entities) = world.load_scene_into(path, Some(0)).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(entities.len(), 1);
        assert_eq!(world.total_object_count(), 2);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn save_scene_at_writes_file() {
        let dir = std::env::temp_dir().join("rkf_api_test_save_at.rkscene");
        let path = dir.to_str().unwrap();

        let mut world = World::new("main");
        world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.save_scene_at(0, path).unwrap();
        assert!(std::path::Path::new(path).exists());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_then_save_roundtrip() {
        let dir1 = std::env::temp_dir().join("rkf_api_test_rt1.rkscene");
        let dir2 = std::env::temp_dir().join("rkf_api_test_rt2.rkscene");
        let p1 = dir1.to_str().unwrap();
        let p2 = dir2.to_str().unwrap();

        let mut world = World::new("test");
        world
            .spawn("ball")
            .position_vec3(Vec3::new(1.0, 2.0, 3.0))
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(2)
            .build();
        world.save_scene(p1).unwrap();

        let mut world2 = World::new("empty");
        let (idx, _) = world2.load_scene_into(p1, None).unwrap();
        world2.save_scene_at(idx, p2).unwrap();

        let mut world3 = World::new("verify");
        let loaded = world3.load_scene(p2).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(world3.name(loaded[0]).unwrap(), "ball");

        let _ = std::fs::remove_file(p1);
        let _ = std::fs::remove_file(p2);
    }

    #[test]
    fn save_scene_at_out_of_range_errors() {
        let world = World::new("test");
        let result = world.save_scene_at(99, "/tmp/nope.rkscene");
        assert!(matches!(result, Err(WorldError::SceneOutOfRange(99))));
    }

    // ── Scene I/O ──────────────────────────────────────────────────────────

    #[test]
    fn save_load_round_trip() {
        let dir = std::env::temp_dir().join("rkf_api_test_scene.rkscene");
        let path = dir.to_str().unwrap();

        let mut world = World::new("test");
        world
            .spawn("sphere")
            .position_vec3(Vec3::new(1.0, 2.0, 3.0))
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(2)
            .build();

        world.save_scene(path).unwrap();

        let mut world2 = World::new("loaded");
        let loaded = world2.load_scene(path).unwrap();
        assert_eq!(loaded.len(), 1);

        let e = loaded[0];
        assert_eq!(world2.name(e).unwrap(), "sphere");

        // Clean up
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_nonexistent_errors() {
        let mut world = World::new("test");
        let result = world.load_scene("/nonexistent/path.rkscene");
        assert!(result.is_err());
    }

    // ── Node tree access (B.1) ──────────────────────────────────────────

    #[test]
    fn find_node_returns_matching_child() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::analytical(
            "arm",
            SdfPrimitive::Sphere { radius: 0.2 },
            1,
        ));
        let e = world.spawn("obj").sdf_tree(root).build();
        let node = world.find_node(e, "arm").unwrap();
        assert_eq!(node.name, "arm");
    }

    #[test]
    fn find_node_not_found_returns_error() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert!(matches!(
            world.find_node(e, "nope"),
            Err(WorldError::NodeNotFound(_))
        ));
    }

    #[test]
    fn find_node_on_ecs_entity_errors() {
        let mut world = World::new("test");
        let e = world.spawn("ecs_only").build();
        assert!(matches!(
            world.find_node(e, "anything"),
            Err(WorldError::MissingComponent(_, _))
        ));
    }

    #[test]
    fn find_node_mut_allows_modification() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child"));
        let e = world.spawn("obj").sdf_tree(root).build();
        world.find_node_mut(e, "child").unwrap().metadata.locked = true;
        assert!(world.find_node(e, "child").unwrap().metadata.locked);
    }

    #[test]
    fn find_node_by_path_multi_level() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        let mut spine = SceneNode::new("spine");
        spine.add_child(SceneNode::new("chest"));
        root.add_child(spine);
        let e = world.spawn("obj").sdf_tree(root).build();
        let node = world.find_node_by_path(e, "spine/chest").unwrap();
        assert_eq!(node.name, "chest");
    }

    #[test]
    fn root_node_immutable() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let root = world.root_node(e).unwrap();
        assert_eq!(root.name, "obj");
    }

    #[test]
    fn node_count_matches_tree() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("a"));
        root.add_child(SceneNode::new("b"));
        let e = world.spawn("obj").sdf_tree(root).build();
        assert_eq!(world.node_count(e).unwrap(), 3);
    }

    // ── Node tree mutation (B.2) ─────────────────────────────────────────

    #[test]
    fn set_node_transform_updates_child() {
        use rkf_core::scene_node::Transform;
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("arm"));
        let e = world.spawn("obj").sdf_tree(root).build();
        let t = Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
        world.set_node_transform(e, "arm", t).unwrap();
        let node = world.find_node(e, "arm").unwrap();
        assert_eq!(node.local_transform.position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn set_node_transform_nonexistent_errors() {
        use rkf_core::scene_node::Transform;
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let t = Transform::default();
        assert!(matches!(
            world.set_node_transform(e, "nope", t),
            Err(WorldError::NodeNotFound(_))
        ));
    }

    #[test]
    fn add_child_node_appends() {
        let mut world = World::new("test");
        let root = SceneNode::new("root");
        let e = world.spawn("obj").sdf_tree(root).build();
        world
            .add_child_node(e, "root", SceneNode::new("new_child"))
            .unwrap();
        assert_eq!(world.node_count(e).unwrap(), 2);
        assert!(world.find_node(e, "new_child").is_ok());
    }

    #[test]
    fn add_child_node_to_nonexistent_parent_errors() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert!(matches!(
            world.add_child_node(e, "nope", SceneNode::new("c")),
            Err(WorldError::NodeNotFound(_))
        ));
    }

    #[test]
    fn remove_child_node_returns_removed() {
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("to_remove"));
        root.add_child(SceneNode::new("keep"));
        let e = world.spawn("obj").sdf_tree(root).build();
        let removed = world.remove_child_node(e, "to_remove").unwrap();
        assert_eq!(removed.name, "to_remove");
        assert_eq!(world.node_count(e).unwrap(), 2); // root + keep
    }

    #[test]
    fn remove_child_node_nonexistent_errors() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert!(matches!(
            world.remove_child_node(e, "nope"),
            Err(WorldError::NodeNotFound(_))
        ));
    }

    #[test]
    fn set_node_blend_mode_changes_mode() {
        use rkf_core::scene_node::BlendMode;
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child"));
        let e = world.spawn("obj").sdf_tree(root).build();
        world
            .set_node_blend_mode(e, "child", BlendMode::Subtract)
            .unwrap();
        let node = world.find_node(e, "child").unwrap();
        assert!(matches!(node.blend_mode, BlendMode::Subtract));
    }

    #[test]
    fn set_node_sdf_source_changes_source() {
        use rkf_core::scene_node::SdfSource;
        let mut world = World::new("test");
        let mut root = SceneNode::new("root");
        root.add_child(SceneNode::new("child"));
        let e = world.spawn("obj").sdf_tree(root).build();
        let source = SdfSource::Analytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
            material_id: 5,
        };
        world.set_node_sdf_source(e, "child", source).unwrap();
        let node = world.find_node(e, "child").unwrap();
        assert!(matches!(node.sdf_source, SdfSource::Analytical { material_id: 5, .. }));
    }

    // ── Camera entities (D.2) ───────────────────────────────────────────

    #[test]
    fn spawn_camera_creates_entity() {
        let mut world = World::new("test");
        let cam = world.spawn_camera("Main", WorldPosition::default(), 0.0, 0.0, 60.0);
        assert!(world.is_alive(cam));
        assert_eq!(world.name(cam).unwrap(), "Main");
    }

    #[test]
    fn cameras_lists_camera_entities() {
        let mut world = World::new("test");
        world.spawn_camera("Cam1", WorldPosition::default(), 0.0, 0.0, 60.0);
        world.spawn_camera("Cam2", WorldPosition::default(), 0.0, 0.0, 90.0);
        // Also spawn a non-camera entity
        world
            .spawn("cube")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.cameras().len(), 2);
    }

    #[test]
    fn active_camera_finds_one() {
        let mut world = World::new("test");
        let cam = world.spawn_camera("Main", WorldPosition::default(), 0.0, 0.0, 60.0);
        assert!(world.active_camera().is_none()); // starts inactive
        world.set_active_camera(cam).unwrap();
        assert_eq!(world.active_camera(), Some(cam));
    }

    #[test]
    fn set_active_camera_deactivates_others() {
        let mut world = World::new("test");
        let cam1 = world.spawn_camera("Cam1", WorldPosition::default(), 0.0, 0.0, 60.0);
        let cam2 = world.spawn_camera("Cam2", WorldPosition::default(), 0.0, 0.0, 90.0);
        world.set_active_camera(cam1).unwrap();
        world.set_active_camera(cam2).unwrap();
        assert_eq!(world.active_camera(), Some(cam2));
        // cam1 should be inactive
        let c1 = world.get::<CameraComponent>(cam1).unwrap();
        assert!(!c1.active);
    }

    #[test]
    fn camera_round_trips_through_position() {
        let mut world = World::new("test");
        let pos = WorldPosition::new(glam::IVec3::new(1, 0, 0), Vec3::new(3.0, 2.0, 1.0));
        let cam = world.spawn_camera("Main", pos, 45.0, -15.0, 75.0);
        assert_eq!(world.position(cam).unwrap(), pos);
        let c = world.get::<CameraComponent>(cam).unwrap();
        assert!((c.fov_degrees - 75.0).abs() < 1e-6);
        assert!((c.yaw - 45.0).abs() < 1e-6);
        assert!((c.pitch - -15.0).abs() < 1e-6);
    }

    // ── Multi-scene (C.1) ──────────────────────────────────────────────

    #[test]
    fn world_starts_with_one_scene() {
        let world = World::new("test");
        assert_eq!(world.scene_count(), 1);
        assert_eq!(world.scene_name(0), Some("test"));
    }

    #[test]
    fn create_scene_increments_count() {
        let mut world = World::new("test");
        world.create_scene("scene2");
        world.create_scene("scene3");
        assert_eq!(world.scene_count(), 3);
    }

    #[test]
    fn active_scene_defaults_to_zero() {
        let world = World::new("test");
        assert_eq!(world.active_scene_index(), 0);
    }

    #[test]
    fn set_active_scene_changes_target() {
        let mut world = World::new("test");
        world.create_scene("scene2");
        world.set_active_scene(1);
        assert_eq!(world.active_scene_index(), 1);
    }

    #[test]
    fn scene_name_returns_correct() {
        let mut world = World::new("main");
        world.create_scene("overlay");
        assert_eq!(world.scene_name(0), Some("main"));
        assert_eq!(world.scene_name(1), Some("overlay"));
        assert_eq!(world.scene_name(99), None);
    }

    #[test]
    fn spawn_targets_active_scene() {
        let mut world = World::new("scene0");
        let _e0 = world
            .spawn("obj_s0")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.scene().object_count(), 1);

        world.create_scene("scene1");
        world.set_active_scene(1);
        let _e1 = world
            .spawn("obj_s1")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.scene().object_count(), 1); // scene1 has 1

        world.set_active_scene(0);
        assert_eq!(world.scene().object_count(), 1); // scene0 still has 1
    }

    #[test]
    fn existing_apis_still_work_after_refactor() {
        let mut world = World::new("test");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        world.set_position(e, pos).unwrap();
        assert_eq!(world.position(e).unwrap(), pos);

        let rot = Quat::from_rotation_y(1.0);
        world.set_rotation(e, rot).unwrap();
        assert!((world.rotation(e).unwrap() - rot).length() < 1e-5);
    }

    // ── Combined scene view (C.2) ──────────────────────────────────────

    #[test]
    fn all_objects_spans_scenes() {
        let mut world = World::new("s0");
        world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.create_scene("s1");
        world.set_active_scene(1);
        world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        let names: Vec<&str> = world.all_objects().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }

    #[test]
    fn total_object_count_sums_scenes() {
        let mut world = World::new("s0");
        world
            .spawn("a")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.create_scene("s1");
        world.set_active_scene(1);
        world
            .spawn("b")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world
            .spawn("c")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        assert_eq!(world.total_object_count(), 3);
    }

    #[test]
    fn renderer_sees_all_scenes() {
        let mut world = World::new("s0");
        world
            .spawn("obj0")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.create_scene("s1");
        world.set_active_scene(1);
        world
            .spawn("obj1")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        // Renderer would call all_objects() — verify it contains both
        assert_eq!(world.all_objects().count(), 2);
        assert_eq!(world.total_object_count(), 2);
    }

    // ── Persistent scenes and swap (C.3) ─────────────────────────────────

    #[test]
    fn set_persistent_flag() {
        let mut world = World::new("test");
        assert!(!world.is_scene_persistent(0));
        world.set_scene_persistent(0, true);
        assert!(world.is_scene_persistent(0));
    }

    #[test]
    fn swap_removes_non_persistent() {
        let mut world = World::new("gameplay");
        world.create_scene("ui");
        world.set_scene_persistent(1, true); // ui is persistent
        let removed = world.swap_scenes();
        assert_eq!(removed, vec!["gameplay"]);
        assert_eq!(world.scene_count(), 1);
        assert_eq!(world.scene_name(0), Some("ui"));
    }

    #[test]
    fn swap_despawns_entities() {
        let mut world = World::new("temp");
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.create_scene("persistent");
        world.set_scene_persistent(1, true);
        world.swap_scenes();
        assert!(!world.is_alive(e));
    }

    #[test]
    fn swap_preserves_persistent_entities() {
        let mut world = World::new("temp");
        world
            .spawn("temp_obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.create_scene("persistent");
        world.set_scene_persistent(1, true);
        world.set_active_scene(1);
        let keeper = world
            .spawn("keeper")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        world.swap_scenes();
        assert!(world.is_alive(keeper));
    }

    #[test]
    fn remove_scene_by_index() {
        let mut world = World::new("s0");
        world.create_scene("s1");
        let name = world.remove_scene(0, false).unwrap();
        assert_eq!(name, "s0");
        assert_eq!(world.scene_count(), 1);
    }

    #[test]
    fn remove_persistent_needs_force() {
        let mut world = World::new("s0");
        world.create_scene("s1");
        world.set_scene_persistent(0, true);
        assert!(world.remove_scene(0, false).is_err());
        assert!(world.remove_scene(0, true).is_ok());
    }

    #[test]
    fn cannot_remove_last_scene() {
        let mut world = World::new("only");
        assert!(matches!(
            world.remove_scene(0, false),
            Err(WorldError::CannotRemoveLastScene)
        ));
    }

    #[test]
    fn active_index_adjusts_after_removal() {
        let mut world = World::new("s0");
        world.create_scene("s1");
        world.create_scene("s2");
        world.set_active_scene(2);
        world.remove_scene(0, false).unwrap();
        // Active was 2, scene 0 removed → active should be adjusted
        assert!(world.active_scene_index() < world.scene_count());
    }

    #[test]
    fn entities_span_scenes() {
        let mut world = World::new("scene0");
        let e0 = world
            .spawn("obj_s0")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();

        world.create_scene("scene1");
        world.set_active_scene(1);
        let e1 = world
            .spawn("obj_s1")
            .sdf(SdfPrimitive::Sphere { radius: 0.3 })
            .material(2)
            .build();

        // Both entities accessible regardless of active scene
        assert!(world.is_alive(e0));
        assert!(world.is_alive(e1));
        assert_eq!(world.name(e0).unwrap(), "obj_s0");
        assert_eq!(world.name(e1).unwrap(), "obj_s1");

        // Transforms work across scenes
        let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(5.0, 0.0, 0.0));
        world.set_position(e0, pos).unwrap();
        assert_eq!(world.position(e0).unwrap(), pos);
    }

    #[test]
    fn save_empty_world() {
        let dir = std::env::temp_dir().join("rkf_api_test_empty.rkscene");
        let path = dir.to_str().unwrap();

        let world = World::new("empty");
        world.save_scene(path).unwrap();

        let mut world2 = World::new("loaded");
        let loaded = world2.load_scene(path).unwrap();
        assert_eq!(loaded.len(), 0);

        let _ = std::fs::remove_file(path);
    }

    // ── Resync entity tracking ────────────────────────────────────────────

    #[test]
    fn resync_registers_untracked() {
        use rkf_core::scene::SceneObject;
        use rkf_core::scene_node::SceneNode;
        use rkf_core::aabb::Aabb;

        let mut world = World::new("test");
        assert_eq!(world.entity_count(), 0);

        // Directly push objects into the scene (bypassing World::spawn).
        {
            let scene = world.scene_mut();
            scene.objects.push(SceneObject {
                id: 10,
                name: "direct_a".into(),
                parent_id: None,
                position: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: SceneNode::new("direct_a"),
                aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
            });
            scene.objects.push(SceneObject {
                id: 20,
                name: "direct_b".into(),
                parent_id: None,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: SceneNode::new("direct_b"),
                aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
            });
        }

        // Before resync: no entity records exist.
        assert_eq!(world.entity_count(), 0);

        world.resync_entity_tracking();

        // After resync: both objects are tracked.
        assert_eq!(world.entity_count(), 2);
        assert!(world.find("direct_a").is_some());
        assert!(world.find("direct_b").is_some());

        // Entity position should match the object position.
        let entity_a = world.find("direct_a").unwrap();
        let pos = world.position(entity_a).unwrap();
        assert_eq!(pos.to_vec3(), Vec3::new(1.0, 2.0, 3.0));

        // find_entity_by_id should work for automation API.
        assert!(world.find_entity_by_id(10).is_some());
        assert!(world.find_entity_by_id(20).is_some());
    }

    #[test]
    fn resync_removes_stale() {
        let mut world = World::new("test");

        // Spawn normally.
        let e = world
            .spawn("obj")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(0)
            .build();
        assert_eq!(world.entity_count(), 1);

        // Directly remove the object from the scene (bypassing World::despawn).
        {
            let scene = world.scene_mut();
            scene.objects.clear();
        }

        // Entity record still exists (stale).
        assert_eq!(world.entity_count(), 1);

        world.resync_entity_tracking();

        // After resync: stale record removed.
        assert_eq!(world.entity_count(), 0);
        assert!(!world.is_alive(e));
    }

    #[test]
    fn resync_idempotent() {
        use rkf_core::scene::SceneObject;
        use rkf_core::scene_node::SceneNode;
        use rkf_core::aabb::Aabb;

        let mut world = World::new("test");

        // Directly push an object.
        {
            let scene = world.scene_mut();
            scene.objects.push(SceneObject {
                id: 5,
                name: "obj".into(),
                parent_id: None,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: SceneNode::new("obj"),
                aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
            });
        }

        world.resync_entity_tracking();
        assert_eq!(world.entity_count(), 1);
        let entity_first = world.find("obj").unwrap();

        // Calling resync again should not create duplicates.
        world.resync_entity_tracking();
        assert_eq!(world.entity_count(), 1);
        let entity_second = world.find("obj").unwrap();

        // Same entity handle — not a new one.
        assert_eq!(entity_first, entity_second);
    }
}
