//! CommandQueue — deferred structural ECS mutations.
//!
//! Rust's ownership rules prevent modifying the hecs World while iterating.
//! The CommandQueue collects spawn/despawn/insert/remove operations that are
//! applied between phases when the world is not being iterated.

use std::any::TypeId;
use std::collections::HashMap;

use super::entity_names::ensure_unique_name;
use super::stable_id::StableId;
use super::stable_id_index::StableIdIndex;
use crate::components::{EditorMetadata, Transform};

/// Opaque handle to a not-yet-materialized entity.
///
/// Returned by `CommandQueue::spawn*()`. Can be used with `insert()` to
/// batch additional components onto the pending entity. The entity
/// materializes atomically when the queue flushes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TempEntity(usize);

/// Which scene to assign to a spawned entity.
#[derive(Debug, Clone)]
enum SpawnScene {
    /// Use the current scene (error if multiple scenes loaded).
    Current,
    /// Persistent — survives scene transitions.
    Persistent,
    /// Explicit scene name.
    Explicit(String),
}

/// Source of components for a pending spawn.
enum SpawnSource {
    /// Explicit EntityBuilder with pre-added components.
    Builder,
    /// Blueprint name — resolved during flush() via BlueprintCatalog.
    Blueprint {
        name: String,
        position: rkf_core::WorldPosition,
    },
}

/// A pending spawn operation.
struct PendingSpawn {
    /// The entity builder accumulating components.
    builder: hecs::EntityBuilder,
    /// Scene ownership assignment.
    scene: SpawnScene,
    /// How to populate this entity's components.
    source: SpawnSource,
}

/// A type-erased component insertion.
struct PendingInsert {
    /// The component data, boxed.
    data: Box<dyn ComponentBox>,
}

/// Trait object for type-erased component insertion.
trait ComponentBox: Send + 'static {
    fn insert_into(self: Box<Self>, world: &mut hecs::World, entity: hecs::Entity);
    fn type_id(&self) -> TypeId;
}

struct TypedComponent<C: hecs::Component> {
    component: C,
}

impl<C: hecs::Component> ComponentBox for TypedComponent<C> {
    fn insert_into(self: Box<Self>, world: &mut hecs::World, entity: hecs::Entity) {
        // Insert or replace the component on the entity.
        let _ = world.insert_one(entity, self.component);
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<C>()
    }
}

/// A type-erased component removal.
struct PendingRemove {
    /// Function that removes the component from the entity.
    remove_fn: fn(&mut hecs::World, hecs::Entity),
}

/// Collects deferred structural mutations applied between phases.
///
/// # Usage
/// ```ignore
/// let guard = ctx.commands.spawn(builder);
/// ctx.commands.insert(guard, Alerted { target: player });
/// // Both materialize together on flush
/// ```
pub struct CommandQueue {
    /// Pending entity spawns.
    spawns: Vec<PendingSpawn>,
    /// Components to insert on TempEntity handles (index → components).
    temp_inserts: HashMap<usize, Vec<PendingInsert>>,
    /// Components to insert on existing entities.
    entity_inserts: Vec<(hecs::Entity, PendingInsert)>,
    /// Entities to despawn.
    despawns: Vec<hecs::Entity>,
    /// Components to remove from existing entities.
    removes: Vec<(hecs::Entity, PendingRemove)>,
    /// Currently loaded scenes (set by engine). Used for `spawn()` default scene.
    loaded_scenes: Vec<String>,
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandQueue {
    /// Create a new empty command queue.
    pub fn new() -> Self {
        Self {
            spawns: Vec::new(),
            temp_inserts: HashMap::new(),
            entity_inserts: Vec::new(),
            despawns: Vec::new(),
            removes: Vec::new(),
            loaded_scenes: Vec::new(),
        }
    }

    /// Set the currently loaded scenes. Called by the engine each frame.
    pub fn set_loaded_scenes(&mut self, scenes: Vec<String>) {
        self.loaded_scenes = scenes;
    }

    // ─── Spawn variants ───────────────────────────────────────────────

    /// Spawn an entity in the current scene.
    ///
    /// Panics if multiple scenes are loaded — use `spawn_in_scene()` instead.
    pub fn spawn(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder,
            scene: SpawnScene::Current,
            source: SpawnSource::Builder,
        });
        TempEntity(idx)
    }

    /// Spawn a persistent entity (survives scene transitions).
    pub fn spawn_persistent(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder,
            scene: SpawnScene::Persistent,
            source: SpawnSource::Builder,
        });
        TempEntity(idx)
    }

    /// Spawn an entity in an explicit scene.
    pub fn spawn_in_scene(
        &mut self,
        builder: hecs::EntityBuilder,
        scene: &str,
    ) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder,
            scene: SpawnScene::Explicit(scene.to_owned()),
            source: SpawnSource::Builder,
        });
        TempEntity(idx)
    }

    // ─── Blueprint spawn variants ────────────────────────────────────

    /// Spawn an entity from a blueprint in the current scene.
    ///
    /// The blueprint name and position are stored for deferred resolution.
    /// During [`flush_with_catalog`], the blueprint is looked up in the
    /// [`BlueprintCatalog`] and its components are deserialized onto the entity.
    ///
    /// Panics if multiple scenes are loaded — use `spawn_blueprint_in_scene()`.
    pub fn spawn_blueprint(
        &mut self,
        name: &str,
        position: impl Into<rkf_core::WorldPosition>,
    ) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder: hecs::EntityBuilder::new(),
            scene: SpawnScene::Current,
            source: SpawnSource::Blueprint {
                name: name.to_owned(),
                position: position.into(),
            },
        });
        TempEntity(idx)
    }

    /// Spawn a persistent entity from a blueprint (survives scene transitions).
    ///
    /// See [`spawn_blueprint`](Self::spawn_blueprint) for details.
    pub fn spawn_blueprint_persistent(
        &mut self,
        name: &str,
        position: impl Into<rkf_core::WorldPosition>,
    ) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder: hecs::EntityBuilder::new(),
            scene: SpawnScene::Persistent,
            source: SpawnSource::Blueprint {
                name: name.to_owned(),
                position: position.into(),
            },
        });
        TempEntity(idx)
    }

    /// Spawn an entity from a blueprint in an explicit scene.
    ///
    /// See [`spawn_blueprint`](Self::spawn_blueprint) for details.
    pub fn spawn_blueprint_in_scene(
        &mut self,
        name: &str,
        position: impl Into<rkf_core::WorldPosition>,
        scene: &str,
    ) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder: hecs::EntityBuilder::new(),
            scene: SpawnScene::Explicit(scene.to_owned()),
            source: SpawnSource::Blueprint {
                name: name.to_owned(),
                position: position.into(),
            },
        });
        TempEntity(idx)
    }

    // ─── Insert / Remove / Despawn ────────────────────────────────────

    /// Insert a component onto a pending spawn (TempEntity).
    pub fn insert_temp<C: hecs::Component>(&mut self, temp: TempEntity, component: C) {
        self.temp_inserts
            .entry(temp.0)
            .or_default()
            .push(PendingInsert {
                data: Box::new(TypedComponent { component }),
            });
    }

    /// Insert a component onto an existing entity.
    pub fn insert<C: hecs::Component>(&mut self, entity: hecs::Entity, component: C) {
        self.entity_inserts.push((
            entity,
            PendingInsert {
                data: Box::new(TypedComponent { component }),
            },
        ));
    }

    /// Queue an entity for despawn. Cascading — descendants are also despawned.
    pub fn despawn(&mut self, entity: hecs::Entity) {
        self.despawns.push(entity);
    }

    /// Queue removal of a component type from an entity.
    pub fn remove<C: hecs::Component>(&mut self, entity: hecs::Entity) {
        self.removes.push((
            entity,
            PendingRemove {
                remove_fn: |world, entity| {
                    let _ = world.remove_one::<C>(entity);
                },
            },
        ));
    }

    // ─── Flush ────────────────────────────────────────────────────────

    /// Apply all pending operations to the world.
    ///
    /// Order: spawns (with temp inserts + auto-injected defaults) → inserts on existing → removes → despawns.
    ///
    /// Auto-injects on spawn (if not already provided by caller):
    /// - `Transform::default()`
    /// - `StableId::new()`
    /// - `EditorMetadata::default()`
    ///
    /// Maintains `stable_ids` index: registers on spawn, removes on despawn.
    ///
    /// Blueprint spawns are ignored by this method — use [`flush_with_catalog`]
    /// if the queue may contain blueprint spawns.
    pub fn flush(&mut self, world: &mut hecs::World, stable_ids: &mut StableIdIndex) {
        self.flush_inner(world, stable_ids, None, None);
    }

    /// Apply all pending operations, resolving blueprint spawns via the catalog.
    ///
    /// Blueprint spawns look up the named blueprint in `catalog` and use
    /// `registry` to deserialize the blueprint's components onto the entity.
    /// If a blueprint name is not found in the catalog, the spawn is skipped
    /// with a warning.
    pub fn flush_with_catalog(
        &mut self,
        world: &mut hecs::World,
        stable_ids: &mut StableIdIndex,
        catalog: &super::blueprint::BlueprintCatalog,
        registry: &super::registry::GameplayRegistry,
    ) {
        self.flush_inner(world, stable_ids, Some(catalog), Some(registry));
    }

    /// Auto-inject default components onto a newly spawned entity if not
    /// already present, and register it in the StableIdIndex.
    fn auto_inject_defaults(
        world: &mut hecs::World,
        entity: hecs::Entity,
        stable_ids: &mut StableIdIndex,
    ) {
        // Auto-inject Transform if not present
        if world.get::<&Transform>(entity).is_err() {
            let _ = world.insert_one(entity, Transform::default());
        }
        // Auto-inject StableId if not present
        if world.get::<&StableId>(entity).is_err() {
            let _ = world.insert_one(entity, StableId::new());
        }
        // Auto-inject EditorMetadata if not present
        if world.get::<&EditorMetadata>(entity).is_err() {
            let _ = world.insert_one(entity, EditorMetadata::default());
        }

        // Register in StableIdIndex
        if let Ok(stable_id) = world.get::<&StableId>(entity) {
            let uuid = stable_id.uuid();
            stable_ids.insert(uuid, entity);
        }
    }

    fn flush_inner(
        &mut self,
        world: &mut hecs::World,
        stable_ids: &mut StableIdIndex,
        catalog: Option<&super::blueprint::BlueprintCatalog>,
        registry: Option<&super::registry::GameplayRegistry>,
    ) {
        // 1. Process spawns
        let spawns = std::mem::take(&mut self.spawns);
        let mut temp_inserts = std::mem::take(&mut self.temp_inserts);

        for (idx, mut pending) in spawns.into_iter().enumerate() {
            // Resolve scene ownership
            let scene = match &pending.scene {
                SpawnScene::Current => {
                    if self.loaded_scenes.len() == 1 {
                        Some(self.loaded_scenes[0].clone())
                    } else if self.loaded_scenes.is_empty() {
                        log::warn!(
                            "spawn() called with no scenes loaded — entity will be persistent"
                        );
                        None
                    } else {
                        panic!(
                            "spawn() called with {} scenes loaded — use spawn_in_scene() \
                             to specify which scene. Loaded: {:?}",
                            self.loaded_scenes.len(),
                            self.loaded_scenes
                        );
                    }
                }
                SpawnScene::Persistent => None,
                SpawnScene::Explicit(name) => Some(name.clone()),
            };

            // Add SceneOwnership component
            pending
                .builder
                .add(super::scene_ownership::SceneOwnership { scene });

            // Handle blueprint vs builder source
            let spawned_entity = match pending.source {
                SpawnSource::Builder => {
                    let entity = world.spawn(pending.builder.build());

                    // Apply TempEntity inserts atomically with the spawn
                    if let Some(inserts) = temp_inserts.remove(&idx) {
                        for insert in inserts {
                            insert.data.insert_into(world, entity);
                        }
                    }
                    Some(entity)
                }
                SpawnSource::Blueprint { name, position } => {
                    let (Some(cat), Some(reg)) = (catalog, registry) else {
                        log::warn!(
                            "Blueprint spawn '{}' ignored — flush() called without catalog/registry. \
                             Use flush_with_catalog() instead.",
                            name
                        );
                        continue;
                    };
                    let Some(blueprint) = cat.get(&name) else {
                        log::warn!(
                            "Blueprint '{}' not found in catalog — spawn skipped",
                            name
                        );
                        continue;
                    };

                    // Spawn the entity with the scene ownership from the builder.
                    let entity = world.spawn(pending.builder.build());

                    // Deserialize blueprint components onto the entity.
                    for (comp_name, ron_data) in &blueprint.components {
                        if let Some(entry) = reg.component_entry(comp_name) {
                            if let Err(e) = (entry.deserialize_insert)(world, entity, ron_data) {
                                log::warn!(
                                    "Failed to deserialize blueprint component '{}': {}",
                                    comp_name, e
                                );
                            }
                        } else {
                            log::warn!(
                                "Unknown component '{}' in blueprint '{}' — skipped",
                                comp_name, name
                            );
                        }
                    }

                    // Set transform position.
                    let _ = world.insert_one(entity, Transform {
                        position,
                        rotation: glam::Quat::IDENTITY,
                        scale: glam::Vec3::ONE,
                    });

                    // Apply TempEntity inserts.
                    if let Some(inserts) = temp_inserts.remove(&idx) {
                        for insert in inserts {
                            insert.data.insert_into(world, entity);
                        }
                    }
                    Some(entity)
                }
            };

            // Auto-inject defaults and register in StableIdIndex
            if let Some(entity) = spawned_entity {
                Self::auto_inject_defaults(world, entity, stable_ids);

                // Enforce sibling name uniqueness (spec 4.3).
                let needs_rename = world
                    .get::<&EditorMetadata>(entity)
                    .ok()
                    .map(|m| m.name.clone());
                if let Some(name) = needs_rename {
                    if !name.is_empty() {
                        let unique = ensure_unique_name(world, entity, &name);
                        if unique != name {
                            if let Ok(mut meta) = world.get::<&mut EditorMetadata>(entity) {
                                meta.name = unique;
                            }
                        }
                    }
                }
            }
        }

        // 2. Inserts on existing entities
        let entity_inserts = std::mem::take(&mut self.entity_inserts);
        let mut inserted_entities = Vec::new();
        for (entity, insert) in entity_inserts {
            if world.contains(entity) {
                let is_metadata = insert.data.type_id() == std::any::TypeId::of::<EditorMetadata>();
                insert.data.insert_into(world, entity);
                if is_metadata {
                    inserted_entities.push(entity);
                }
            }
        }

        // Enforce sibling name uniqueness on renamed entities (spec 4.3).
        for entity in inserted_entities {
            let needs_rename = world
                .get::<&EditorMetadata>(entity)
                .ok()
                .map(|m| m.name.clone());
            if let Some(name) = needs_rename {
                if !name.is_empty() {
                    let unique = ensure_unique_name(world, entity, &name);
                    if unique != name {
                        if let Ok(mut meta) = world.get::<&mut EditorMetadata>(entity) {
                            meta.name = unique;
                        }
                    }
                }
            }
        }

        // 3. Removes
        let removes = std::mem::take(&mut self.removes);
        for (entity, remove) in removes {
            if world.contains(entity) {
                (remove.remove_fn)(world, entity);
            }
        }

        // 4. Despawns (with cascading)
        let despawns = std::mem::take(&mut self.despawns);
        for entity in despawns {
            self.despawn_cascading(world, entity, stable_ids);
        }
    }

    /// Despawn an entity and all its descendants.
    fn despawn_cascading(
        &self,
        world: &mut hecs::World,
        entity: hecs::Entity,
        stable_ids: &mut StableIdIndex,
    ) {
        if !world.contains(entity) {
            return;
        }

        // Collect all descendants first (children whose Parent points to this entity)
        let children: Vec<hecs::Entity> = world
            .query::<&crate::components::Parent>()
            .iter()
            .filter_map(|(child, parent)| {
                if parent.entity == entity {
                    Some(child)
                } else {
                    None
                }
            })
            .collect();

        // Recursively despawn children
        for child in children {
            self.despawn_cascading(world, child, stable_ids);
        }

        // Remove from StableIdIndex before despawning
        stable_ids.remove_by_entity(entity);

        // Despawn the entity itself
        let _ = world.despawn(entity);
    }

    /// Returns true if the queue has no pending operations.
    pub fn is_empty(&self) -> bool {
        self.spawns.is_empty()
            && self.temp_inserts.is_empty()
            && self.entity_inserts.is_empty()
            && self.despawns.is_empty()
            && self.removes.is_empty()
    }
}

