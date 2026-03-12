//! Sequence system — advances Sequence components each frame.
//!
//! Designed to run as a LateUpdate system. Processes each entity with a
//! [`Sequence`] component, advancing the timer, executing completed steps,
//! and removing the component when all steps finish.

use super::command_queue::CommandQueue;
use super::game_store::GameStore;
use super::sequence::{Sequence, SequenceStep};
use crate::components::Transform;

/// Advance all Sequence components by `delta_time`.
///
/// This is designed to be called as a LateUpdate system. It processes each
/// entity with a Sequence component, advancing the timer and executing
/// completed steps. When a sequence finishes, the Sequence component is
/// removed via the command queue.
pub fn tick_sequences(
    world: &mut hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    delta_time: f32,
) {
    // Collect entities with sequences to avoid borrow issues
    let entities: Vec<(hecs::Entity, Sequence)> = world
        .query::<&Sequence>()
        .iter()
        .filter(|(_, seq)| !seq.is_complete())
        .map(|(e, seq)| (e, seq.clone()))
        .collect();

    for (entity, mut seq) in entities {
        advance_sequence(entity, &mut seq, world, commands, store, delta_time);

        if seq.is_complete() {
            // Remove the Sequence component when done
            commands.remove::<Sequence>(entity);
        } else {
            // Write back the advanced sequence
            commands.insert(entity, seq);
        }
    }
}

/// Process a single sequence, advancing through steps by `delta_time`.
fn advance_sequence(
    entity: hecs::Entity,
    seq: &mut Sequence,
    world: &hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    delta_time: f32,
) {
    seq.timer += delta_time;

    while !seq.is_complete() {
        let step = &seq.steps[seq.current];
        let step_duration = step.duration();

        if seq.timer >= step_duration {
            // Step complete — execute it at t=1.0
            execute_step(entity, step, 1.0, world, commands, store);
            seq.timer -= step_duration;
            seq.current += 1;
        } else {
            // Step in progress — compute progress and apply
            let t = if step_duration > 0.0 {
                seq.timer / step_duration
            } else {
                1.0
            };
            execute_step(entity, step, t, world, commands, store);
            break;
        }
    }
}

/// Execute a single step at progress `t` (0.0..1.0).
///
/// Instant steps (Emit, SetState, Despawn, etc.) only fire when `t >= 1.0`.
/// Interpolated steps (MoveTo, MoveBy, RotateTo, ScaleTo) will eventually
/// lerp/slerp based on eased `t`, but for now only apply the final value
/// at completion.
fn execute_step(
    entity: hecs::Entity,
    step: &SequenceStep,
    t: f32,
    world: &hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
) {
    match step {
        SequenceStep::Wait { .. } => {
            // Nothing to do — just wait for the timer to advance.
        }

        SequenceStep::MoveTo { target, .. } => {
            // TODO: Full interpolation requires storing the start position when the
            // step begins. For now, set the final position at completion.
            if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position = target.clone();
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::MoveBy { offset, .. } => {
            if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position.local += *offset;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::RotateTo { target, .. } => {
            if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = *target;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::RotateBy { rotation, .. } => {
            if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let cur_rotation = transform.rotation;
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = *rotation * cur_rotation;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::ScaleTo { target, .. } => {
            if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.scale = *target;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::Emit { name } => {
            if t >= 1.0 {
                store.emit(name, Some(entity), None);
            }
        }

        SequenceStep::EmitWith { name, data } => {
            if t >= 1.0 {
                store.emit(name, Some(entity), Some(data.clone()));
            }
        }

        SequenceStep::EmitFrom {
            name,
            source,
            data,
        } => {
            if t >= 1.0 {
                store.emit(name, *source, data.clone());
            }
        }

        SequenceStep::SetState { key, value } => {
            if t >= 1.0 {
                store.set(key, value.clone());
            }
        }

        SequenceStep::SpawnBlueprint { .. } => {
            // TODO: Requires GameplayRegistry access for blueprint lookup.
            // Will be wired when blueprint system is implemented.
        }

        SequenceStep::Despawn => {
            if t >= 1.0 {
                commands.despawn(entity);
            }
        }

        SequenceStep::Repeat { .. } => {
            // TODO: Repeat requires flattening into sub-steps or tracking inner
            // progress. Will be implemented when needed.
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;
    use glam::Vec3;
    use rkf_core::WorldPosition;

    /// Helper: run tick_sequences and flush commands.
    fn tick_and_flush(
        world: &mut hecs::World,
        commands: &mut CommandQueue,
        store: &mut GameStore,
        dt: f32,
    ) {
        tick_sequences(world, commands, store, dt);
        commands.flush(world);
    }

    #[test]
    fn wait_step_advances_and_completes() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new().wait(1.0).build();
        let entity = world.spawn((Transform::default(), seq));

        // Tick 0.5s — still in progress
        tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
        assert!(world.get::<&Sequence>(entity).is_ok(), "sequence should still exist");

        // Tick another 0.6s — should complete (total 1.1s > 1.0s)
        tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
        assert!(
            world.get::<&Sequence>(entity).is_err(),
            "sequence should be removed after completion"
        );
    }

    #[test]
    fn emit_step_fires_event_at_completion() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new().emit("test_event").build();
        let entity = world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 0.0);

        // Emit is instant (duration=0), so it fires immediately
        let events: Vec<_> = store.events("test_event").collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, Some(entity));
        assert!(events[0].data.is_none());
    }

    #[test]
    fn emit_with_step_fires_event_with_data() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        use super::super::game_value::GameValue;
        let seq = Sequence::new()
            .emit_with("damage", 25.0_f32)
            .build();
        let entity = world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 0.0);

        let events: Vec<_> = store.events("damage").collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].source, Some(entity));
        assert_eq!(events[0].data, Some(GameValue::Float(25.0)));
    }

    #[test]
    fn set_state_step_sets_store_value() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new()
            .set_state("door_open", true)
            .build();
        world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 0.0);

        assert_eq!(store.get::<bool>("door_open"), Some(true));
    }

    #[test]
    fn despawn_step_queues_entity_removal() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new().despawn().build();
        let entity = world.spawn((Transform::default(), seq));

        assert!(world.contains(entity));
        tick_and_flush(&mut world, &mut commands, &mut store, 0.0);
        assert!(!world.contains(entity), "entity should be despawned");
    }

    #[test]
    fn multi_step_sequence_progresses_through_steps() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new()
            .wait(1.0)
            .emit("step_1_done")
            .wait(1.0)
            .emit("step_2_done")
            .build();
        world.spawn((Transform::default(), seq));

        // After 0.5s: in Wait step 0, no events
        tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
        assert_eq!(store.events("step_1_done").count(), 0);
        assert_eq!(store.events("step_2_done").count(), 0);

        // After 0.6s more (total 1.1s): Wait 0 done, Emit 1 fires, now in Wait 2
        tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
        assert_eq!(store.events("step_1_done").count(), 1);
        assert_eq!(store.events("step_2_done").count(), 0);

        store.drain_events();

        // After 1.0s more (total 2.1s): Wait 2 done, Emit 3 fires, sequence complete
        tick_and_flush(&mut world, &mut commands, &mut store, 1.0);
        assert_eq!(store.events("step_2_done").count(), 1);
    }

    #[test]
    fn completed_sequence_has_component_removed() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new().wait(0.5).build();
        let entity = world.spawn((Transform::default(), seq));

        // Not yet complete
        tick_and_flush(&mut world, &mut commands, &mut store, 0.3);
        assert!(world.get::<&Sequence>(entity).is_ok());

        // Complete
        tick_and_flush(&mut world, &mut commands, &mut store, 0.3);
        assert!(
            world.get::<&Sequence>(entity).is_err(),
            "Sequence component should be removed after completion"
        );
        // Entity itself should still exist
        assert!(world.contains(entity));
    }

    #[test]
    fn zero_duration_steps_execute_immediately() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // All instant steps — should all fire in a single tick
        let seq = Sequence::new()
            .emit("a")
            .emit("b")
            .set_state("x", 42_i32)
            .emit("c")
            .build();
        world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 0.016);

        assert_eq!(store.events("a").count(), 1);
        assert_eq!(store.events("b").count(), 1);
        assert_eq!(store.events("c").count(), 1);
        assert_eq!(store.get::<i32>("x"), Some(42));
    }

    #[test]
    fn move_to_sets_position_at_completion() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let target = WorldPosition {
            chunk: glam::IVec3::ZERO,
            local: Vec3::new(10.0, 20.0, 30.0),
        };
        let seq = Sequence::new().move_to(target.clone(), 1.0).build();
        let entity = world.spawn((Transform::default(), seq));

        // Not yet complete
        tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
        let pos = world.get::<&Transform>(entity).unwrap().position.clone();
        // Position unchanged during partial progress (interpolation not yet implemented)
        assert_eq!(pos, WorldPosition::default());

        // Complete
        tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
        let pos = world.get::<&Transform>(entity).unwrap().position.clone();
        assert_eq!(pos.local, Vec3::new(10.0, 20.0, 30.0));
    }

    #[test]
    fn move_by_applies_offset_at_completion() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let seq = Sequence::new()
            .move_by(Vec3::new(5.0, 0.0, 0.0), 0.5)
            .build();
        let entity = world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 1.0);
        let pos = world.get::<&Transform>(entity).unwrap().position.clone();
        assert_eq!(pos.local, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn emit_from_step_uses_custom_source() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // EmitFrom with source=None (UI-originated)
        let seq = Sequence::new()
            .emit_from("ui_event", None, None)
            .build();
        world.spawn((Transform::default(), seq));

        tick_and_flush(&mut world, &mut commands, &mut store, 0.0);

        let events: Vec<_> = store.events("ui_event").collect();
        assert_eq!(events.len(), 1);
        assert!(events[0].source.is_none());
    }

    #[test]
    fn already_complete_sequence_is_skipped() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        // An empty sequence is immediately complete
        let seq = Sequence::default();
        assert!(seq.is_complete());
        let entity = world.spawn((Transform::default(), seq));

        // tick_sequences filters out complete sequences, so nothing happens
        tick_and_flush(&mut world, &mut commands, &mut store, 0.016);

        // The entity still has its Sequence (it was never processed)
        // This is by design — the filter skips already-complete sequences
        assert!(world.contains(entity));
    }

    #[test]
    fn entity_without_transform_skips_move_to() {
        let mut world = hecs::World::new();
        let mut commands = CommandQueue::new();
        let mut store = GameStore::new();

        let target = WorldPosition {
            chunk: glam::IVec3::ZERO,
            local: Vec3::new(10.0, 0.0, 0.0),
        };
        // Entity has Sequence but no Transform
        let seq = Sequence::new()
            .move_to(target, 0.5)
            .emit("after_move")
            .build();
        let entity = world.spawn((seq,));

        tick_and_flush(&mut world, &mut commands, &mut store, 1.0);

        // MoveTo silently skipped (no Transform), Emit still fires
        assert_eq!(store.events("after_move").count(), 1);
        assert!(world.contains(entity));
    }
}
