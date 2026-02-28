//! World — the unified game state container.
//!
//! [`World`] wraps a [`Scene`] (SDF objects), a [`hecs::World`] (game logic
//! components), and optionally a brick pool for voxelized SDF objects. Users
//! interact with a single entity handle ([`Entity`]) that transparently spans
//! both storage backends.

use std::collections::HashMap;
use std::path::Path;

use glam::{Quat, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::brick_map::BrickMapAllocator;
use rkf_core::brick_pool::BrickPool;
use rkf_core::scene::{Scene, SceneObject};
use rkf_core::scene_node::SceneNode;
use rkf_core::WorldPosition;

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

/// The unified game state container.
///
/// Wraps [`Scene`] (SDF rendering data) and [`hecs::World`] (game logic
/// components) behind a single API. Users spawn objects, add components,
/// query, and mutate transforms through one handle type ([`Entity`]).
pub struct World {
    scene: Scene,
    ecs: hecs::World,
    brick_pool: BrickPool,
    brick_map_alloc: BrickMapAllocator,

    // Entity tracking
    next_generation: u32,
    entities: HashMap<Entity, EntityRecord>,
    sdf_to_entity: HashMap<u32, Entity>,
}

impl World {
    /// Create a new empty world.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            scene: Scene::new(name),
            ecs: hecs::World::new(),
            brick_pool: BrickPool::new(4096),
            brick_map_alloc: BrickMapAllocator::new(),
            next_generation: 0,
            entities: HashMap::new(),
            sdf_to_entity: HashMap::new(),
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
            self.scene.remove_object(obj_id);
            self.sdf_to_entity.remove(&obj_id);
        }

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

        let obj_id = self.scene.add_object_full(obj);

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
            if let Some(obj) = self.scene.find_by_id_mut(obj_id) {
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
            if let Some(obj) = self.scene.find_by_id_mut(obj_id) {
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
            if let Some(obj) = self.scene.find_by_id_mut(obj_id) {
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
                if !self.scene.reparent(child_id, Some(parent_id)) {
                    return Err(WorldError::CycleDetected);
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
            self.scene.reparent(obj_id, None);
        }
        Ok(())
    }

    /// Iterate over the children of an entity.
    pub fn children(&self, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
        let sdf_obj_id = self
            .entities
            .get(&entity)
            .and_then(|r| r.sdf_object_id);

        // Collect child SDF object IDs, then map to Entities
        let child_ids: Vec<u32> = if let Some(parent_id) = sdf_obj_id {
            self.scene
                .children_of(parent_id)
                .map(|o| o.id)
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
        let obj = self.scene.find_by_id(obj_id)?;
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
        self.ecs = hecs::World::new();
        self.scene = Scene::new(self.scene.name.clone());
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

    /// Load entities from a `.rkscene` file, adding them to this world.
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
                if let Some(obj) = self.scene.find_by_id(obj_id) {
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
            name: self.scene.name.clone(),
            objects,
            cameras: Vec::new(),
            lights: Vec::new(),
            environment: None,
        };

        let path_str = path.as_ref().to_str().unwrap_or("");
        crate::scene_file::save_scene_file(path_str, &scene_file)
            .map_err(|e| WorldError::Io(std::io::Error::other(e.to_string())))
    }

    // ── Internal accessors ─────────────────────────────────────────────────

    /// Get a reference to the underlying Scene.
    ///
    /// Used by the Renderer internally and by the editor for snapshot cloning.
    pub fn scene(&self) -> &Scene {
        &self.scene
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
        &mut self.scene
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
            .scene
            .objects
            .iter_mut()
            .find(|o| o.id == sdf_id)
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
            .scene
            .find_by_id(sdf_id)
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
        let obj = world.scene.find_by_id(record.sdf_object_id.unwrap()).unwrap();
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
        let obj = world.scene.find_by_id(record.sdf_object_id.unwrap()).unwrap();
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
        assert_eq!(world.scene.object_count(), 0);
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
}
