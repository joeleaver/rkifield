//! CommandQueue — deferred structural ECS mutations.
//!
//! Rust's ownership rules prevent modifying the hecs World while iterating.
//! The CommandQueue collects spawn/despawn/insert/remove operations that are
//! applied between phases when the world is not being iterated.

use std::any::TypeId;
use std::collections::HashMap;

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

/// A pending spawn operation.
struct PendingSpawn {
    /// The entity builder accumulating components.
    builder: hecs::EntityBuilder,
    /// Scene ownership assignment.
    scene: SpawnScene,
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
        });
        TempEntity(idx)
    }

    /// Spawn a persistent entity (survives scene transitions).
    pub fn spawn_persistent(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        let idx = self.spawns.len();
        self.spawns.push(PendingSpawn {
            builder,
            scene: SpawnScene::Persistent,
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
    /// Order: spawns (with temp inserts) → inserts on existing → removes → despawns.
    pub fn flush(&mut self, world: &mut hecs::World) {
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

            // Spawn the entity
            let entity = world.spawn(pending.builder.build());

            // Apply TempEntity inserts atomically with the spawn
            if let Some(inserts) = temp_inserts.remove(&idx) {
                for insert in inserts {
                    insert.data.insert_into(world, entity);
                }
            }
        }

        // 2. Inserts on existing entities
        let entity_inserts = std::mem::take(&mut self.entity_inserts);
        for (entity, insert) in entity_inserts {
            if world.contains(entity) {
                insert.data.insert_into(world, entity);
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
            self.despawn_cascading(world, entity);
        }
    }

    /// Despawn an entity and all its descendants.
    fn despawn_cascading(&self, world: &mut hecs::World, entity: hecs::Entity) {
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
            self.despawn_cascading(world, child);
        }

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

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{EditorMetadata, Transform};
    use super::super::scene_ownership::SceneOwnership;

    #[test]
    fn spawn_and_flush() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();
        queue.set_loaded_scenes(vec!["level_01".into()]);

        let mut builder = hecs::EntityBuilder::new();
        builder.add(Transform::default());
        queue.spawn(builder);

        assert!(!queue.is_empty());
        queue.flush(&mut world);
        assert!(queue.is_empty());

        // Should have one entity with Transform + SceneOwnership
        let count = world.query::<&Transform>().iter().count();
        assert_eq!(count, 1);

        let count = world.query::<&SceneOwnership>().iter().count();
        assert_eq!(count, 1);

        // Check scene ownership
        let mut q = world.query::<&SceneOwnership>();
        let (_, so) = q.iter().next().unwrap();
        assert!(so.belongs_to("level_01"));
    }

    #[test]
    fn spawn_persistent() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();
        queue.set_loaded_scenes(vec!["level_01".into()]);

        let builder = hecs::EntityBuilder::new();
        queue.spawn_persistent(builder);
        queue.flush(&mut world);

        let mut q = world.query::<&SceneOwnership>();
        let (_, so) = q.iter().next().unwrap();
        assert!(so.is_persistent());
    }

    #[test]
    fn spawn_in_explicit_scene() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();

        let builder = hecs::EntityBuilder::new();
        queue.spawn_in_scene(builder, "level_02");
        queue.flush(&mut world);

        let mut q = world.query::<&SceneOwnership>();
        let (_, so) = q.iter().next().unwrap();
        assert!(so.belongs_to("level_02"));
    }

    #[test]
    #[should_panic(expected = "spawn() called with 2 scenes loaded")]
    fn spawn_panics_with_multiple_scenes() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();
        queue.set_loaded_scenes(vec!["level_01".into(), "level_02".into()]);

        let builder = hecs::EntityBuilder::new();
        queue.spawn(builder);
        queue.flush(&mut world);
    }

    #[test]
    fn despawn() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut queue = CommandQueue::new();
        queue.despawn(entity);
        queue.flush(&mut world);

        assert!(!world.contains(entity));
    }

    #[test]
    fn despawn_cascading() {
        let mut world = hecs::World::new();
        let parent = world.spawn((Transform::default(),));
        let child = world.spawn((
            Transform::default(),
            crate::components::Parent {
                entity: parent,
                bone_index: None,
            },
        ));
        let grandchild = world.spawn((
            Transform::default(),
            crate::components::Parent {
                entity: child,
                bone_index: None,
            },
        ));

        let mut queue = CommandQueue::new();
        queue.despawn(parent);
        queue.flush(&mut world);

        assert!(!world.contains(parent));
        assert!(!world.contains(child));
        assert!(!world.contains(grandchild));
    }

    #[test]
    fn insert_on_existing_entity() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut queue = CommandQueue::new();
        queue.insert(entity, EditorMetadata {
            name: "Test".into(),
            tags: vec![],
            locked: false,
        });
        queue.flush(&mut world);

        let meta = world.get::<&EditorMetadata>(entity).unwrap();
        assert_eq!(meta.name, "Test");
    }

    #[test]
    fn remove_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            EditorMetadata::default(),
        ));

        let mut queue = CommandQueue::new();
        queue.remove::<EditorMetadata>(entity);
        queue.flush(&mut world);

        assert!(world.get::<&EditorMetadata>(entity).is_err());
        assert!(world.get::<&Transform>(entity).is_ok());
    }

    #[test]
    fn empty_flush_is_noop() {
        let mut world = hecs::World::new();
        let mut queue = CommandQueue::new();
        assert!(queue.is_empty());
        queue.flush(&mut world);
        assert_eq!(world.query::<()>().iter().count(), 0);
    }
}
