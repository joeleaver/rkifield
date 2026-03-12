//! SystemContext — the single parameter passed to every gameplay system function.
//!
//! Provides controlled access to the ECS world, command queue, game state store,
//! time, and entity lookup. Systems receive `&mut SystemContext` and use it for
//! all world interaction.

use super::command_queue::{CommandQueue, TempEntity};
use super::registry::QueryError;

/// The single parameter passed to every gameplay system function.
///
/// Provides controlled access to the ECS world (read-only queries), a deferred
/// command queue (spawn/despawn/insert), and frame timing. Systems receive
/// `&mut SystemContext` and use it for all world interaction.
///
/// # Design
///
/// - **World access is read-only** via `query()`, `get()`, `has()`, `entity_exists()`.
///   Mutations go through the [`CommandQueue`] which flushes between phases.
/// - **Time** is immutable for the frame — `delta_time`, `total_time`, `frame`.
/// - **GameStore** will be wired in a future commit when GameStore merges.
///
/// # Example
///
/// ```ignore
/// fn patrol_system(ctx: &mut SystemContext) {
///     for (entity, (transform, patrol)) in ctx.query::<(&Transform, &Patrol)>().iter() {
///         // read components...
///     }
///     ctx.despawn(stale_entity);
/// }
/// ```
pub struct SystemContext<'a> {
    world: &'a mut hecs::World,
    commands: &'a mut CommandQueue,
    // game_store: &'a mut GameStore,  // Skip for now — will wire when GameStore merges
    delta_time: f32,
    total_time: f64,
    frame: u64,
}

impl<'a> SystemContext<'a> {
    /// Create a new SystemContext for a single frame tick.
    pub fn new(
        world: &'a mut hecs::World,
        commands: &'a mut CommandQueue,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) -> Self {
        Self {
            world,
            commands,
            delta_time,
            total_time,
            frame,
        }
    }

    // ─── World access (read-only queries) ────────────────────────────────

    /// Iterate entities matching a query archetype.
    ///
    /// Returns a [`hecs::QueryBorrow`] that can be iterated with `.iter()`.
    ///
    /// ```ignore
    /// for (entity, (t, h)) in ctx.query::<(&Transform, &Health)>().iter() {
    ///     // ...
    /// }
    /// ```
    pub fn query<Q: hecs::Query>(&self) -> hecs::QueryBorrow<'_, Q> {
        self.world.query::<Q>()
    }

    /// Read a single component from an entity.
    ///
    /// Returns `Err` if the entity doesn't exist or lacks the component.
    pub fn get<C: hecs::Component>(
        &self,
        entity: hecs::Entity,
    ) -> Result<hecs::Ref<'_, C>, hecs::ComponentError> {
        self.world.get::<&C>(entity)
    }

    /// Check whether an entity has a specific component type.
    pub fn has<C: hecs::Component>(&self, entity: hecs::Entity) -> bool {
        self.world.get::<&C>(entity).is_ok()
    }

    /// Check whether an entity exists in the world (regardless of components).
    pub fn entity_exists(&self, entity: hecs::Entity) -> bool {
        self.world.contains(entity)
    }

    // ─── Entity lookup helpers ───────────────────────────────────────────

    // TODO: Add `find_by_name(&self, name: &str)` and `find_by_tag(&self, tag: &str)`
    // convenience methods once EntityNameIndex and EntityTagIndex are wired into
    // SystemContext (requires full runtime assembly to pass indexes through).
    // For now, use `entity_lookup::find_path` and `entity_lookup::find_tagged`
    // directly with the indexes.

    /// Find exactly one entity matching a query. Returns error if 0 or 2+ match.
    ///
    /// Useful for singletons (e.g., the player entity, a game manager).
    ///
    /// ```ignore
    /// let player = ctx.find_one_entity::<(&Player, &Transform)>()?;
    /// let pos = ctx.get::<Transform>(player)?;
    /// ```
    pub fn find_one_entity<Q: hecs::Query>(&self) -> Result<hecs::Entity, QueryError> {
        let mut query = self.world.query::<Q>();
        let mut iter = query.iter();

        let first = iter.next();
        if first.is_none() {
            return Err(QueryError::NotFound);
        }

        let (entity, _) = first.unwrap();

        if iter.next().is_some() {
            return Err(QueryError::Multiple);
        }

        Ok(entity)
    }

    // ─── Commands (deferred mutations) ───────────────────────────────────

    /// Access the command queue directly for advanced operations.
    pub fn commands(&mut self) -> &mut CommandQueue {
        self.commands
    }

    /// Shortcut: spawn an entity via the command queue.
    ///
    /// Returns a [`TempEntity`] handle that can be used with
    /// `commands().insert_temp()` before the queue flushes.
    pub fn spawn(&mut self, builder: hecs::EntityBuilder) -> TempEntity {
        self.commands.spawn(builder)
    }

    /// Shortcut: queue an entity for despawn (cascading — descendants too).
    pub fn despawn(&mut self, entity: hecs::Entity) {
        self.commands.despawn(entity);
    }

    /// Shortcut: queue a component insertion on an existing entity.
    pub fn insert<C: hecs::Component>(&mut self, entity: hecs::Entity, component: C) {
        self.commands.insert(entity, component);
    }

    // ─── Time ────────────────────────────────────────────────────────────

    /// Seconds elapsed since the previous frame (variable timestep).
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    /// Total seconds elapsed since engine start (high precision).
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Current frame number (monotonically increasing from 0).
    pub fn frame(&self) -> u64 {
        self.frame
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{EditorMetadata, Transform};

    /// Helper: create a world + command queue + context for testing.
    fn make_ctx<'a>(
        world: &'a mut hecs::World,
        commands: &'a mut CommandQueue,
    ) -> SystemContext<'a> {
        SystemContext::new(world, commands, 1.0 / 60.0, 10.5, 630)
    }

    // ─── World access ────────────────────────────────────────────────────

    #[test]
    fn query_iterates_matching_entities() {
        let mut world = hecs::World::new();
        world.spawn((Transform::default(),));
        world.spawn((Transform::default(), EditorMetadata::default()));
        world.spawn((EditorMetadata::default(),));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        // Query for Transform — should match 2 entities
        let count = ctx.query::<&Transform>().iter().count();
        assert_eq!(count, 2);

        // Query for both — should match 1
        let count = ctx.query::<(&Transform, &EditorMetadata)>().iter().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn get_returns_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((EditorMetadata {
            name: "TestEntity".into(),
            tags: vec![],
            locked: false,
        },));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        let meta = ctx.get::<EditorMetadata>(entity).unwrap();
        assert_eq!(meta.name, "TestEntity");
    }

    #[test]
    fn get_returns_error_for_missing_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        assert!(ctx.get::<EditorMetadata>(entity).is_err());
    }

    #[test]
    fn has_returns_true_when_present() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        assert!(ctx.has::<Transform>(entity));
        assert!(!ctx.has::<EditorMetadata>(entity));
    }

    #[test]
    fn entity_exists_checks_liveness() {
        let mut world = hecs::World::new();
        let alive = world.spawn((Transform::default(),));
        let dead = world.spawn((Transform::default(),));
        world.despawn(dead).unwrap();

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        assert!(ctx.entity_exists(alive));
        assert!(!ctx.entity_exists(dead));
    }

    // ─── find_one_entity ─────────────────────────────────────────────────

    /// Marker component for singleton tests.
    struct Player;

    #[test]
    fn find_one_entity_with_exactly_one() {
        let mut world = hecs::World::new();
        let expected = world.spawn((Player, Transform::default()));
        world.spawn((Transform::default(),)); // not a Player

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        let found = ctx.find_one_entity::<&Player>().unwrap();
        assert_eq!(found, expected);
    }

    #[test]
    fn find_one_entity_with_zero_returns_not_found() {
        let mut world = hecs::World::new();
        world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        let result = ctx.find_one_entity::<&Player>();
        assert!(matches!(result, Err(QueryError::NotFound)));
    }

    #[test]
    fn find_one_entity_with_multiple_returns_error() {
        let mut world = hecs::World::new();
        world.spawn((Player,));
        world.spawn((Player,));

        let mut commands = CommandQueue::new();
        let ctx = make_ctx(&mut world, &mut commands);

        let result = ctx.find_one_entity::<&Player>();
        assert!(matches!(result, Err(QueryError::Multiple)));
    }

    // ─── Command shortcuts ───────────────────────────────────────────────

    #[test]
    fn spawn_shortcut_adds_to_queue() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test_scene".into()]);

        {
            let mut ctx = make_ctx(&mut world, &mut commands);
            let mut builder = hecs::EntityBuilder::new();
            builder.add(Transform::default());
            let _temp = ctx.spawn(builder);
        }

        // Flush and verify
        commands.flush(&mut world);
        assert_eq!(world.query::<&Transform>().iter().count(), 1);
    }

    #[test]
    fn despawn_shortcut_queues_removal() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let mut commands = CommandQueue::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands);
            ctx.despawn(entity);
        }

        commands.flush(&mut world);
        assert!(!world.contains(entity));
    }

    #[test]
    fn insert_shortcut_queues_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let mut commands = CommandQueue::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands);
            ctx.insert(entity, EditorMetadata {
                name: "Inserted".into(),
                tags: vec![],
                locked: false,
            });
        }

        commands.flush(&mut world);
        let meta = world.get::<&EditorMetadata>(entity).unwrap();
        assert_eq!(meta.name, "Inserted");
    }

    // ─── Time accessors ──────────────────────────────────────────────────

    #[test]
    fn time_accessors() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let ctx = SystemContext::new(&mut world, &mut commands, 0.016, 42.5, 2550);

        assert!((ctx.delta_time() - 0.016).abs() < 1e-6);
        assert!((ctx.total_time() - 42.5).abs() < 1e-12);
        assert_eq!(ctx.frame(), 2550);
    }

    // ─── Commands accessor ───────────────────────────────────────────────

    #[test]
    fn commands_accessor_gives_mutable_access() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let mut commands = CommandQueue::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands);
            // Use the raw commands() accessor to queue a remove
            ctx.commands().remove::<Transform>(entity);
        }

        commands.flush(&mut world);
        assert!(world.get::<&Transform>(entity).is_err());
    }
}
