//! SystemContext — the single parameter passed to every gameplay system function.
//!
//! Provides controlled access to the ECS world, command queue, game state store,
//! time, and entity lookup. Systems receive `&mut SystemContext` and use it for
//! all world interaction.

use glam::{Quat, Vec3};
use rkf_core::WorldPosition;

use super::command_queue::{CommandQueue, TempEntity};
use super::console::ConsoleBuffer;
use super::engine_access::{EngineAccess, TransformUpdate};
use super::game_store::GameStore;
use super::persist::Persistable;
use super::registry::QueryError;
use super::stable_id_index::StableIdIndex;

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
    store: &'a mut GameStore,
    stable_ids: &'a StableIdIndex,
    engine: &'a dyn EngineAccess,
    console: ConsoleBuffer,
    transform_updates: Vec<TransformUpdate>,
    delta_time: f32,
    total_time: f64,
    frame: u64,
}

impl<'a> SystemContext<'a> {
    /// Create a new SystemContext for a single frame tick.
    pub fn new(
        world: &'a mut hecs::World,
        commands: &'a mut CommandQueue,
        store: &'a mut GameStore,
        stable_ids: &'a StableIdIndex,
        engine: &'a dyn EngineAccess,
        console: ConsoleBuffer,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) -> Self {
        Self {
            world,
            commands,
            store,
            stable_ids,
            engine,
            console,
            transform_updates: Vec::new(),
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

    // ─── Game store ────────────────────────────────────────────────────────

    /// Access the game state store for reading and writing gameplay state.
    pub fn store(&mut self) -> &mut GameStore {
        self.store
    }

    /// Read-only access to the game state store.
    pub fn store_ref(&self) -> &GameStore {
        self.store
    }

    // ─── Persistence sync ────────────────────────────────────────────────

    /// Sync a `Persistable` component's fields from the entity into the store.
    ///
    /// Looks up the entity's `StableId` component and calls
    /// `T::sync_to_store()` with the entity's stable ID string.
    ///
    /// No-op if the entity lacks a `StableId` or the component `T`.
    pub fn sync_to_store<T: Persistable + hecs::Component>(&mut self, entity: hecs::Entity) {
        // Look up the StableId via the index (avoids borrow conflict with world)
        let stable_uuid = match self.stable_ids.get_stable(entity) {
            Some(uuid) => uuid,
            None => return,
        };
        let stable_str = stable_uuid.to_string();

        // Read the component and call sync_to_store. The Ref borrow from world
        // is compatible with the &mut store borrow since sync_to_store takes
        // &self (immutable component reference).
        if let Ok(c) = self.world.get::<&T>(entity) {
            c.sync_to_store(&stable_str, self.store);
        }
    }

    /// Sync a `Persistable` component's fields from the store into the entity.
    ///
    /// Looks up the entity's `StableId` component and calls
    /// `T::sync_from_store()` with the entity's stable ID string.
    ///
    /// No-op if the entity lacks a `StableId` or the component `T`.
    pub fn sync_from_store<T: Persistable + hecs::Component>(&mut self, entity: hecs::Entity) {
        let stable_uuid = match self.stable_ids.get_stable(entity) {
            Some(uuid) => uuid,
            None => return,
        };
        let stable_str = stable_uuid.to_string();

        // Get mutable access to the component
        if let Ok(mut c) = self.world.get::<&mut T>(entity) {
            c.sync_from_store(&stable_str, self.store);
        }
    }

    // ─── Engine component access (cross-dylib safe) ───────────────────────

    /// Access the engine bridge for reading engine components.
    ///
    /// Use this instead of querying `Transform` directly — direct hecs queries
    /// for engine types won't work across the dylib boundary.
    pub fn engine(&self) -> &dyn EngineAccess {
        self.engine
    }

    /// Read an entity's position.
    pub fn position(&self, entity: hecs::Entity) -> Option<WorldPosition> {
        self.engine.position(entity)
    }

    /// Read an entity's full transform.
    pub fn get_transform(&self, entity: hecs::Entity) -> Option<(WorldPosition, Quat, Vec3)> {
        self.engine.transform(entity)
    }

    /// Queue a position update (applied after the current system returns).
    pub fn set_position(&mut self, entity: hecs::Entity, position: WorldPosition) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: Some(position),
            rotation: None,
            scale: None,
        });
    }

    /// Queue a rotation update.
    pub fn set_rotation(&mut self, entity: hecs::Entity, rotation: Quat) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: None,
            rotation: Some(rotation),
            scale: None,
        });
    }

    /// Queue a full transform update.
    pub fn set_transform(
        &mut self,
        entity: hecs::Entity,
        position: WorldPosition,
        rotation: Quat,
        scale: Vec3,
    ) {
        self.transform_updates.push(TransformUpdate {
            entity,
            position: Some(position),
            rotation: Some(rotation),
            scale: Some(scale),
        });
    }

    /// Drain pending transform updates (called by the executor).
    pub(crate) fn take_transform_updates(&mut self) -> Vec<TransformUpdate> {
        std::mem::take(&mut self.transform_updates)
    }

    // ─── Console (cross-dylib safe) ──────────────────────────────────────

    /// Log an info message to the console.
    pub fn log(&self, message: impl Into<String>) {
        self.console.info(message);
    }

    /// Log a warning to the console.
    pub fn warn(&self, message: impl Into<String>) {
        self.console.warn(message);
    }

    /// Log an error to the console.
    pub fn error(&self, message: impl Into<String>) {
        self.console.error(message);
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
    use crate::behavior::game_store::GameStore;
    use crate::behavior::persist::{Persistable, persist_key};
    use crate::behavior::stable_id::StableId;
    use crate::behavior::stable_id_index::StableIdIndex;
    use crate::components::{EditorMetadata, Transform};

    /// Stub EngineAccess for tests (no transforms).
    struct StubEngineAccess;
    impl EngineAccess for StubEngineAccess {
        fn position(&self, _: hecs::Entity) -> Option<rkf_core::WorldPosition> { None }
        fn rotation(&self, _: hecs::Entity) -> Option<glam::Quat> { None }
        fn scale(&self, _: hecs::Entity) -> Option<glam::Vec3> { None }
        fn transform(&self, _: hecs::Entity) -> Option<(rkf_core::WorldPosition, glam::Quat, glam::Vec3)> { None }
        fn all_transforms(&self) -> Vec<(hecs::Entity, rkf_core::WorldPosition, glam::Quat, glam::Vec3)> { Vec::new() }
    }

    static STUB_ENGINE: StubEngineAccess = StubEngineAccess;

    /// Helper: create a world + command queue + context for testing.
    fn make_ctx<'a>(
        world: &'a mut hecs::World,
        commands: &'a mut CommandQueue,
        store: &'a mut GameStore,
        stable_ids: &'a StableIdIndex,
    ) -> SystemContext<'a> {
        SystemContext::new(world, commands, store, stable_ids, &STUB_ENGINE, ConsoleBuffer::new(), 1.0 / 60.0, 10.5, 630)
    }

    // ─── World access ────────────────────────────────────────────────────

    #[test]
    fn query_iterates_matching_entities() {
        let mut world = hecs::World::new();
        world.spawn((Transform::default(),));
        world.spawn((Transform::default(), EditorMetadata::default()));
        world.spawn((EditorMetadata::default(),));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

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
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

        let meta = ctx.get::<EditorMetadata>(entity).unwrap();
        assert_eq!(meta.name, "TestEntity");
    }

    #[test]
    fn get_returns_error_for_missing_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

        assert!(ctx.get::<EditorMetadata>(entity).is_err());
    }

    #[test]
    fn has_returns_true_when_present() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

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
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

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
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

        let found = ctx.find_one_entity::<&Player>().unwrap();
        assert_eq!(found, expected);
    }

    #[test]
    fn find_one_entity_with_zero_returns_not_found() {
        let mut world = hecs::World::new();
        world.spawn((Transform::default(),));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

        let result = ctx.find_one_entity::<&Player>();
        assert!(matches!(result, Err(QueryError::NotFound)));
    }

    #[test]
    fn find_one_entity_with_multiple_returns_error() {
        let mut world = hecs::World::new();
        world.spawn((Player,));
        world.spawn((Player,));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);

        let result = ctx.find_one_entity::<&Player>();
        assert!(matches!(result, Err(QueryError::Multiple)));
    }

    // ─── Command shortcuts ───────────────────────────────────────────────

    #[test]
    fn spawn_shortcut_adds_to_queue() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test_scene".into()]);
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            let mut builder = hecs::EntityBuilder::new();
            builder.add(Transform::default());
            let _temp = ctx.spawn(builder);
        }

        // Flush and verify
        commands.flush(&mut world, &mut stable_ids);
        assert_eq!(world.query::<&Transform>().iter().count(), 1);
    }

    #[test]
    fn despawn_shortcut_queues_removal() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            ctx.despawn(entity);
        }

        commands.flush(&mut world, &mut stable_ids);
        assert!(!world.contains(entity));
    }

    #[test]
    fn insert_shortcut_queues_component() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            ctx.insert(entity, EditorMetadata {
                name: "Inserted".into(),
                tags: vec![],
                locked: false,
            });
        }

        commands.flush(&mut world, &mut stable_ids);
        let meta = world.get::<&EditorMetadata>(entity).unwrap();
        assert_eq!(meta.name, "Inserted");
    }

    // ─── Time accessors ──────────────────────────────────────────────────

    #[test]
    fn time_accessors() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        let ctx = SystemContext::new(&mut world, &mut commands, &mut store, &stable_ids, &STUB_ENGINE, ConsoleBuffer::new(), 0.016, 42.5, 2550);

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
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            // Use the raw commands() accessor to queue a remove
            ctx.commands().remove::<Transform>(entity);
        }

        commands.flush(&mut world, &mut stable_ids);
        assert!(world.get::<&Transform>(entity).is_err());
    }

    // ─── Persistence sync ────────────────────────────────────────────────

    /// Test component implementing Persistable.
    struct Score {
        value: i64,
        label: String,
    }

    impl Persistable for Score {
        fn sync_to_store(&self, stable_id: &str, store: &mut GameStore) {
            store.set(&persist_key(stable_id, "value"), self.value);
            store.set(&persist_key(stable_id, "label"), self.label.as_str());
        }
        fn sync_from_store(&mut self, stable_id: &str, store: &GameStore) {
            if let Some(v) = store.get::<i64>(&persist_key(stable_id, "value")) {
                self.value = v;
            }
            if let Some(v) = store.get::<String>(&persist_key(stable_id, "label")) {
                self.label = v;
            }
        }
    }

    #[test]
    fn context_sync_to_store() {
        let mut world = hecs::World::new();
        let stable_id = StableId::new();
        let entity = world.spawn((
            stable_id,
            Score { value: 42, label: "high".into() },
        ));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        stable_ids.insert(stable_id.uuid(), entity);

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            ctx.sync_to_store::<Score>(entity);
        }

        let key = persist_key(&stable_id.to_string(), "value");
        assert_eq!(store.get::<i64>(&key), Some(42));
        let key2 = persist_key(&stable_id.to_string(), "label");
        assert_eq!(store.get::<String>(&key2), Some("high".into()));
    }

    #[test]
    fn context_sync_from_store() {
        let mut world = hecs::World::new();
        let stable_id = StableId::new();
        let entity = world.spawn((
            stable_id,
            Score { value: 0, label: "none".into() },
        ));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();
        stable_ids.insert(stable_id.uuid(), entity);

        // Pre-populate store
        store.set(&persist_key(&stable_id.to_string(), "value"), 99_i64);
        store.set(&persist_key(&stable_id.to_string(), "label"), "restored");

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            ctx.sync_from_store::<Score>(entity);
        }

        let score = world.get::<&Score>(entity).unwrap();
        assert_eq!(score.value, 99);
        assert_eq!(score.label, "restored");
    }

    #[test]
    fn context_sync_noop_without_stable_id() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Score { value: 10, label: "x".into() },));

        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new(); // entity not registered

        {
            let mut ctx = make_ctx(&mut world, &mut commands, &mut store, &stable_ids);
            // Should be a no-op — no panic
            ctx.sync_to_store::<Score>(entity);
            ctx.sync_from_store::<Score>(entity);
        }

        // Store should be empty
        assert_eq!(store.list("").count(), 0);
    }
}
