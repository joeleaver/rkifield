//! BehaviorExecutor — frame execution loop for gameplay systems.
//!
//! Ties together [`Schedule`], [`GameplayRegistry`], [`SystemContext`],
//! [`CommandQueue`], and [`GameStore`] into a single `tick()` call that
//! runs one frame of gameplay logic.

use std::time::Instant;

use super::command_queue::CommandQueue;
use super::console::ConsoleBuffer;
use super::engine_access::WorldEngineAccess;
use super::game_store::GameStore;
use super::registry::GameplayRegistry;
use super::scheduler::{Schedule, ScheduleError, build_schedule};
use super::stable_id_index::StableIdIndex;
use super::system_context::SystemContext;
use crate::components::Transform;

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
    /// Set of system indices (into the registry's system list) that faulted
    /// (panicked) during the most recent tick. Accumulates across frames
    /// until explicitly cleared via [`clear_faults`].
    faulted_systems: std::collections::HashSet<usize>,
    /// Per-system frame timing in microseconds, indexed by system position
    /// in the registry's system list. `None` if the system was skipped
    /// (e.g. faulted) or has not run yet.
    system_timings: Vec<Option<u64>>,
}

impl BehaviorExecutor {
    /// Build an executor from the registry's systems.
    pub fn new(registry: &GameplayRegistry) -> Result<Self, ScheduleError> {
        let schedule = build_schedule(registry.system_list())?;
        let count = registry.system_list().len();
        Ok(Self {
            schedule,
            faulted_systems: std::collections::HashSet::new(),
            system_timings: vec![None; count],
        })
    }

    /// Rebuild the schedule (e.g. after hot-reload changes systems).
    pub fn rebuild(&mut self, registry: &GameplayRegistry) -> Result<(), ScheduleError> {
        self.schedule = build_schedule(registry.system_list())?;
        self.faulted_systems.clear();
        self.system_timings = vec![None; registry.system_list().len()];
        Ok(())
    }

    /// Returns the set of system indices that have faulted (panicked).
    pub fn faulted_systems(&self) -> &std::collections::HashSet<usize> {
        &self.faulted_systems
    }

    /// Returns `true` if the system at the given index has faulted.
    pub fn is_faulted(&self, system_index: usize) -> bool {
        self.faulted_systems.contains(&system_index)
    }

    /// Clear all recorded faults, allowing faulted systems to run again.
    pub fn clear_faults(&mut self) {
        self.faulted_systems.clear();
    }

    /// Last recorded frame time for the system at `index`, in microseconds.
    ///
    /// Returns `None` if the index is out of range, the system was skipped
    /// (faulted), or no tick has run yet.
    pub fn system_timing(&self, index: usize) -> Option<u64> {
        self.system_timings.get(index).copied().flatten()
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
        &mut self,
        registry: &GameplayRegistry,
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        stable_ids: &mut StableIdIndex,
        console: &ConsoleBuffer,
        delta_time: f32,
        total_time: f64,
        frame: u64,
    ) {
        let systems = registry.system_list();

        // Reset timings for this frame.
        for t in self.system_timings.iter_mut() {
            *t = None;
        }

        // ── Phase 1: Update ──────────────────────────────────────────────
        for &idx in &self.schedule.update {
            if self.faulted_systems.contains(&idx) {
                continue; // skip faulted systems until faults are cleared
            }
            let meta = &systems[idx];
            // SAFETY: fn_ptr was produced by casting `fn(&mut SystemContext)` to
            // `*const ()` during system registration (see SystemMeta docs).
            // The #[system] proc macro guarantees this invariant.
            let system_fn: fn(&mut SystemContext) =
                unsafe { std::mem::transmute(meta.fn_ptr) };
            // SAFETY: WorldEngineAccess holds a raw pointer for read-only access
            // through the EngineAccess trait. The pointer is valid for the scope
            // of this system call, and no mutable aliasing occurs — the trait
            // methods only read from the world.
            // SAFETY: Raw pointer for read-only EngineAccess. The pointer is
            // derived before the &mut borrow for SystemContext, and EngineAccess
            // methods only read from the world (no mutation through this pointer).
            let world_ptr: *const hecs::World = &raw const *world;
            let engine_access = unsafe { WorldEngineAccess::new(world_ptr) };
            let mut ctx = SystemContext::new(world, commands, store, stable_ids, &engine_access, console.clone(), delta_time, total_time, frame);
            let start = Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                system_fn(&mut ctx);
            }));
            let elapsed_us = start.elapsed().as_micros() as u64;
            let pending_updates = ctx.take_transform_updates();
            drop(ctx);
            if result.is_err() {
                log::error!("system '{}' panicked during Update phase", meta.name);
                self.faulted_systems.insert(idx);
            } else {
                apply_transform_updates(world, pending_updates);
                if idx < self.system_timings.len() {
                    self.system_timings[idx] = Some(elapsed_us);
                }
            }
        }

        // Flush commands between phases — entities spawned in Update are
        // visible to LateUpdate systems.
        commands.flush_with_catalog(world, stable_ids, &registry.blueprint_catalog, registry);

        // ── Phase 2: LateUpdate ──────────────────────────────────────────
        for &idx in &self.schedule.late_update {
            if self.faulted_systems.contains(&idx) {
                continue;
            }
            let meta = &systems[idx];
            // SAFETY: same invariant as above.
            let system_fn: fn(&mut SystemContext) =
                unsafe { std::mem::transmute(meta.fn_ptr) };
            // SAFETY: Raw pointer for read-only EngineAccess. The pointer is
            // derived before the &mut borrow for SystemContext, and EngineAccess
            // methods only read from the world (no mutation through this pointer).
            let world_ptr: *const hecs::World = &raw const *world;
            let engine_access = unsafe { WorldEngineAccess::new(world_ptr) };
            let mut ctx = SystemContext::new(world, commands, store, stable_ids, &engine_access, console.clone(), delta_time, total_time, frame);
            let start = Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                system_fn(&mut ctx);
            }));
            let elapsed_us = start.elapsed().as_micros() as u64;
            let pending_updates = ctx.take_transform_updates();
            drop(ctx);
            if result.is_err() {
                log::error!("system '{}' panicked during LateUpdate phase", meta.name);
                self.faulted_systems.insert(idx);
            } else {
                apply_transform_updates(world, pending_updates);
                if idx < self.system_timings.len() {
                    self.system_timings[idx] = Some(elapsed_us);
                }
            }
        }

        // Final flush + drain events
        commands.flush_with_catalog(world, stable_ids, &registry.blueprint_catalog, registry);
        store.drain_events();
    }
}

/// Apply buffered transform updates using host TypeIds.
fn apply_transform_updates(
    world: &mut hecs::World,
    updates: Vec<super::engine_access::TransformUpdate>,
) {
    for update in updates {
        if let Ok(mut t) = world.get::<&mut Transform>(update.entity) {
            if let Some(pos) = update.position {
                t.position = pos;
            }
            if let Some(rot) = update.rotation {
                t.rotation = rot;
            }
            if let Some(scale) = update.scale {
                t.scale = scale;
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::registry::{Phase, SystemMeta};
    use crate::behavior::stable_id_index::StableIdIndex;

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
        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        world.spawn((Counter(0),));
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        // Pre-spawn a result entity to hold the count
        world.spawn((Counter(-1),));
        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test".into()]);
        let mut store = GameStore::new();

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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
        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // Emit some events before the tick
        store.emit("test_event", None, None);
        store.emit("test_event", None, None);
        assert_eq!(store.events("test_event").count(), 2);

        executor.tick(&registry, &mut world, &mut commands, &mut store, &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0);

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

    // ── Test: fault tracking ────────────────────────────────────────────

    fn panicking_system(_ctx: &mut SystemContext) {
        panic!("intentional test panic");
    }

    #[test]
    fn faulted_system_is_recorded_and_skipped() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta(
            "panicker",
            Phase::Update,
            &[],
            &[],
            panicking_system,
        ));

        let mut executor = BehaviorExecutor::new(&registry).unwrap();
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // First tick: system panics, gets recorded as faulted.
        executor.tick(
            &registry, &mut world, &mut commands, &mut store,
            &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0,
        );
        assert!(executor.is_faulted(0));
        assert_eq!(executor.faulted_systems().len(), 1);

        // Second tick: faulted system is skipped (no panic propagation).
        executor.tick(
            &registry, &mut world, &mut commands, &mut store,
            &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.016, 1,
        );
        assert!(executor.is_faulted(0));
    }

    #[test]
    fn clear_faults_allows_system_to_run_again() {
        EXEC_ORDER.with(|v| v.borrow_mut().clear());

        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta("a", Phase::Update, &[], &[], system_a));

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        // Manually mark system 0 as faulted.
        executor.faulted_systems.insert(0);
        assert!(executor.is_faulted(0));

        // Tick: system_a should NOT run because it's faulted.
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();
        executor.tick(
            &registry, &mut world, &mut commands, &mut store,
            &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0,
        );
        EXEC_ORDER.with(|v| assert!(v.borrow().is_empty(), "faulted system should not run"));

        // Clear faults.
        executor.clear_faults();
        assert!(!executor.is_faulted(0));
        assert!(executor.faulted_systems().is_empty());

        // Tick again: system_a should run now.
        executor.tick(
            &registry, &mut world, &mut commands, &mut store,
            &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.016, 1,
        );
        EXEC_ORDER.with(|v| assert_eq!(*v.borrow(), vec!["a"]));
    }

    #[test]
    fn system_timing_recorded_after_tick() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta("a", Phase::Update, &[], &[], system_a));
        registry.register_system(sys_meta(
            "late",
            Phase::LateUpdate,
            &[],
            &[],
            late_update_system,
        ));

        let mut executor = BehaviorExecutor::new(&registry).unwrap();
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // Before tick, timings are None.
        assert!(executor.system_timing(0).is_none());
        assert!(executor.system_timing(1).is_none());

        executor.tick(
            &registry, &mut world, &mut commands, &mut store,
            &mut StableIdIndex::new(), &ConsoleBuffer::new(), 0.016, 0.0, 0,
        );

        // After tick, timings should be recorded (non-None).
        assert!(executor.system_timing(0).is_some());
        assert!(executor.system_timing(1).is_some());

        // Out-of-range returns None.
        assert!(executor.system_timing(99).is_none());
    }

    #[test]
    fn rebuild_clears_faults() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(sys_meta("a", Phase::Update, &[], &[], system_a));

        let mut executor = BehaviorExecutor::new(&registry).unwrap();
        executor.faulted_systems.insert(0);
        assert!(executor.is_faulted(0));

        executor.rebuild(&registry).unwrap();
        assert!(executor.faulted_systems().is_empty());
    }
}
