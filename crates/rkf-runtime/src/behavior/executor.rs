//! BehaviorExecutor — frame execution loop for gameplay systems.
//!
//! Ties together [`Schedule`], [`GameplayRegistry`], [`SystemContext`],
//! [`CommandQueue`], and [`GameStore`] into a single `tick()` call that
//! runs one frame of gameplay logic.

use super::command_queue::CommandQueue;
use super::game_store::GameStore;
use super::registry::GameplayRegistry;
use super::scheduler::{Schedule, ScheduleError, build_schedule};
use super::system_context::SystemContext;

/// Runs one frame of gameplay systems in schedule order.
///
/// The executor owns a [`Schedule`] (topologically sorted system indices)
/// and drives the per-frame loop:
///
/// 1. Run all `Update` phase systems in dependency order
/// 2. Flush the command queue (spawns/despawns/inserts materialize)
/// 3. Run all `LateUpdate` phase systems in dependency order
/// 4. Flush the command queue again
/// 5. Drain events from the game store
///
/// # Safety invariant
///
/// [`SystemMeta::fn_ptr`] is stored as `*const ()` because the proc macro
/// crate cannot depend on `SystemContext`. The executor transmutes it back
/// to `fn(&mut SystemContext)` at call time. This is safe **if and only if**
/// every registered system was originally a `fn(&mut SystemContext)` cast to
/// `*const ()`. The `#[system]` proc macro guarantees this — manual
/// registration must uphold the same contract.
pub struct BehaviorExecutor {
    schedule: Schedule,
}

impl BehaviorExecutor {
    /// Build an executor from the registry's systems.
    pub fn new(registry: &GameplayRegistry) -> Result<Self, ScheduleError> {
        let schedule = build_schedule(registry.system_list())?;
        Ok(Self { schedule })
    }

    /// Rebuild the schedule (e.g. after hot-reload changes systems).
    pub fn rebuild(&mut self, registry: &GameplayRegistry) -> Result<(), ScheduleError> {
        self.schedule = build_schedule(registry.system_list())?;
        Ok(())
    }

    /// Execute one frame: run all systems in schedule order, flush commands, drain events.
    ///
    /// # Arguments
    ///
    /// * `registry` — the system catalog (provides `SystemMeta` with fn pointers)
    /// * `world` — the ECS world (passed to each system via `SystemContext`)
    /// * `commands` — deferred mutation queue (flushed between phases)
    /// * `store` — gameplay state store (events drained at frame end)
    /// * `delta_time` — seconds since last frame
    /// * `total_time` — total elapsed seconds (high precision)
    /// * `frame` — monotonic frame counter
    ///
    /// # Safety
    ///
    /// This function uses `unsafe` to transmute `SystemMeta::fn_ptr` from
    /// `*const ()` back to `fn(&mut SystemContext)`. See the struct-level
    /// safety invariant documentation. Every `fn_ptr` **must** have been
    /// originally a `fn(&mut SystemContext)` — the `#[system]` macro
    /// guarantees this for generated registrations.
    pub fn tick(
        &self,
        registry: &GameplayRegistry,
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        let systems = registry.system_list();

        // ── Phase 1: Update ──────────────────────────────────────────────
        for &idx in &self.schedule.update {
            let meta = &systems[idx];
            // SAFETY: fn_ptr was produced by casting `fn(&mut SystemContext)` to
            // `*const ()` during system registration (see SystemMeta docs).
            // The #[system] proc macro guarantees this invariant.
            let system_fn: fn(&mut SystemContext) =
                unsafe { std::mem::transmute(meta.fn_ptr) };
            let mut ctx = SystemContext::new(world, commands, delta_time, total_time, frame);
            system_fn(&mut ctx);
        }

        // Flush commands between phases — entities spawned in Update are
        // visible to LateUpdate systems.
        commands.flush(world);

        // ── Phase 2: LateUpdate ──────────────────────────────────────────
        for &idx in &self.schedule.late_update {
            let meta = &systems[idx];
            // SAFETY: same invariant as above.
            let system_fn: fn(&mut SystemContext) =
                unsafe { std::mem::transmute(meta.fn_ptr) };
            let mut ctx = SystemContext::new(world, commands, delta_time, total_time, frame);
            system_fn(&mut ctx);
        }

        // Final flush + drain events
        commands.flush(world);
        store.drain_events();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::registry::{Phase, SystemMeta};

    /// Marker component for test systems to interact with.
    struct Counter(i32);

    /// Helper: create a SystemMeta with a real fn pointer.
    fn sys_meta(
        name: &'static str,
        phase: Phase,
        after: &'static [&'static str],
        before: &'static [&'static str],
        f: fn(&mut SystemContext),
    ) -> SystemMeta {
        SystemMeta {
            name,
            module_path: name,
            phase,
            after,
            before,
            fn_ptr: f as *const (),
        }
    }

    // ── Test: empty schedule tick is a no-op ─────────────────────────────

    #[test]
    fn empty_schedule_is_noop() {
        let registry = GameplayRegistry::new();
        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        // World should be unchanged
        assert_eq!(world.query::<()>().iter().count(), 0);
    }

    // ── Test: single system executes and modifies world via commands ─────

    fn increment_system(ctx: &mut SystemContext) {
        let updates: Vec<_> = ctx
            .query::<&Counter>()
            .iter()
            .map(|(entity, counter)| (entity, counter.0 + 1))
            .collect();
        for (entity, new_val) in updates {
            ctx.insert(entity, Counter(new_val));
        }
    }

    #[test]
    fn single_system_modifies_world() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta(
            "increment",
            Phase::Update,
            &[],
            &[],
            increment_system,
        ));

        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        world.spawn((Counter(0),));
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        // After tick, flush happened, so the insert should be applied
        let mut q = world.query::<&Counter>();
        let (_, counter) = q.iter().next().unwrap();
        assert_eq!(counter.0, 1);
    }

    // ── Test: systems run in dependency order ────────────────────────────

    // We use thread_local to track execution order, since system fns have a
    // fixed signature and can't capture state.
    thread_local! {
        static EXEC_ORDER: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());
    }

    fn system_a(_ctx: &mut SystemContext) {
        EXEC_ORDER.with(|v| v.borrow_mut().push("a".into()));
    }

    fn system_b(_ctx: &mut SystemContext) {
        EXEC_ORDER.with(|v| v.borrow_mut().push("b".into()));
    }

    fn system_c(_ctx: &mut SystemContext) {
        EXEC_ORDER.with(|v| v.borrow_mut().push("c".into()));
    }

    #[test]
    fn systems_run_in_dependency_order() {
        EXEC_ORDER.with(|v| v.borrow_mut().clear());

        let mut registry = GameplayRegistry::new();
        // c depends on b, b depends on a → order must be a, b, c
        registry.register_system(sys_meta("c", Phase::Update, &["b"], &[], system_c));
        registry.register_system(sys_meta("a", Phase::Update, &[], &["b"], system_a));
        registry.register_system(sys_meta("b", Phase::Update, &["a"], &["c"], system_b));

        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        EXEC_ORDER.with(|v| {
            let order = v.borrow();
            assert_eq!(*order, vec!["a", "b", "c"]);
        });
    }

    // ── Test: Update phase runs before LateUpdate ────────────────────────

    fn update_system(_ctx: &mut SystemContext) {
        EXEC_ORDER.with(|v| v.borrow_mut().push("update".into()));
    }

    fn late_update_system(_ctx: &mut SystemContext) {
        EXEC_ORDER.with(|v| v.borrow_mut().push("late_update".into()));
    }

    #[test]
    fn update_runs_before_late_update() {
        EXEC_ORDER.with(|v| v.borrow_mut().clear());

        let mut registry = GameplayRegistry::new();
        // Register LateUpdate first to ensure phase ordering, not registration ordering
        registry.register_system(sys_meta(
            "late",
            Phase::LateUpdate,
            &[],
            &[],
            late_update_system,
        ));
        registry.register_system(sys_meta(
            "early",
            Phase::Update,
            &[],
            &[],
            update_system,
        ));

        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        EXEC_ORDER.with(|v| {
            let order = v.borrow();
            assert_eq!(*order, vec!["update", "late_update"]);
        });
    }

    // ── Test: commands flush between phases ──────────────────────────────

    /// Marker for entities spawned by the Update phase system.
    struct SpawnedInUpdate;

    fn spawn_in_update_system(ctx: &mut SystemContext) {
        let mut builder = hecs::EntityBuilder::new();
        builder.add(SpawnedInUpdate);
        ctx.spawn(builder);
    }

    fn count_spawned_in_late_update(ctx: &mut SystemContext) {
        let count = ctx.query::<&SpawnedInUpdate>().iter().count();
        // Find the pre-existing "result" entity and store the count.
        let result_entity: Option<hecs::Entity> = ctx
            .query::<&Counter>()
            .iter()
            .map(|(e, _)| e)
            .next();
        if let Some(entity) = result_entity {
            ctx.insert(entity, Counter(count as i32));
        }
    }

    #[test]
    fn commands_flush_between_phases() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta(
            "spawner",
            Phase::Update,
            &[],
            &[],
            spawn_in_update_system,
        ));
        registry.register_system(sys_meta(
            "counter",
            Phase::LateUpdate,
            &[],
            &[],
            count_spawned_in_late_update,
        ));

        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        // Pre-spawn a result entity to hold the count
        world.spawn((Counter(-1),));
        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test".into()]);
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        // The LateUpdate system should have seen the entity spawned in Update
        // (because commands flush between phases). The count should be 1.
        let mut q = world.query::<&Counter>();
        let (_, counter) = q.iter().next().unwrap();
        assert_eq!(counter.0, 1, "LateUpdate should see entity spawned in Update");
    }

    // ── Test: events are drained at end of tick ─────────────────────────

    #[test]
    fn events_drained_at_end_of_tick() {
        let registry = GameplayRegistry::new();
        let executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // Emit some events before the tick
        store.emit("test_event", None, None);
        store.emit("test_event", None, None);
        assert_eq!(store.events("test_event").count(), 2);

        executor.tick(&registry, &mut world, &mut commands, &mut store, 0.016, 0.0, 0);

        // Events should be drained after tick
        assert_eq!(store.events("test_event").count(), 0);
    }

    // ── Test: rebuild schedule ───────────────────────────────────────────

    #[test]
    fn rebuild_updates_schedule() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta("a", Phase::Update, &[], &[], system_a));

        let mut executor = BehaviorExecutor::new(&registry).unwrap();
        assert_eq!(executor.schedule.update.len(), 1);

        // Add another system and rebuild
        registry.register_system(sys_meta("b", Phase::Update, &[], &[], system_b));
        executor.rebuild(&registry).unwrap();
        assert_eq!(executor.schedule.update.len(), 2);
    }
}
