use super::command_queue::CommandQueue;
use super::game_store::GameStore;
use super::registry::GameplayRegistry;
use super::sequence::{Ease, Sequence};
use super::sequence_system::*;
use super::stable_id_index::StableIdIndex;
use crate::components::Transform;
use glam::{Quat, Vec3};
use rkf_core::WorldPosition;

/// Helper: run tick_sequences and flush commands.
fn tick_and_flush(
    world: &mut hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    dt: f32,
) {
    let mut stable_ids = StableIdIndex::new();
    tick_sequences(world, commands, store, dt);
    commands.flush(world, &mut stable_ids);
}

/// Helper: run tick_sequences_with_registry and flush commands.
fn tick_and_flush_with_registry(
    world: &mut hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    dt: f32,
    registry: Option<&GameplayRegistry>,
) {
    let mut stable_ids = StableIdIndex::new();
    tick_sequences_with_registry(world, commands, store, dt, registry);
    commands.flush(world, &mut stable_ids);
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

    use super::game_value::GameValue;
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
fn move_to_interpolates_position() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let target = WorldPosition {
        chunk: glam::IVec3::ZERO,
        local: Vec3::new(10.0, 20.0, 30.0),
    };
    let seq = Sequence::new().move_to(target.clone(), 1.0).build();
    let entity = world.spawn((Transform::default(), seq));

    // At t=0.5, position should be halfway
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    let expected_half = Vec3::new(5.0, 10.0, 15.0);
    assert!(
        (pos.to_vec3() - expected_half).length() < 0.1,
        "position at t=0.5 should be halfway, got {:?}",
        pos.to_vec3()
    );

    // At t=1.0, position should be at target
    tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    assert!(
        (pos.to_vec3() - target.to_vec3()).length() < 0.01,
        "position at t=1.0 should be at target, got {:?}",
        pos.to_vec3()
    );
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

    // Complete in one tick
    tick_and_flush(&mut world, &mut commands, &mut store, 1.1);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    assert!(
        (pos.to_vec3() - Vec3::new(10.0, 20.0, 30.0)).length() < 0.01,
        "position should be at target, got {:?} (world: {:?})",
        pos.local,
        pos.to_vec3()
    );
}

#[test]
fn move_by_interpolates_offset() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let seq = Sequence::new()
        .move_by(Vec3::new(10.0, 0.0, 0.0), 1.0)
        .build();
    let entity = world.spawn((Transform::default(), seq));

    // At t=0.5, should be halfway
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    assert!(
        (pos.to_vec3().x - 5.0).abs() < 0.1,
        "x at t=0.5 should be ~5.0, got {}",
        pos.to_vec3().x
    );

    // Complete
    tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    assert!(
        (pos.to_vec3().x - 10.0).abs() < 0.01,
        "x at t=1.0 should be ~10.0, got {}",
        pos.to_vec3().x
    );
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
    assert!((pos.to_vec3().x - 5.0).abs() < 0.01);
}

#[test]
fn rotate_to_interpolates_rotation() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let target = Quat::from_rotation_y(std::f32::consts::PI);
    let seq = Sequence::new().rotate_to(target, 1.0).build();
    let entity = world.spawn((Transform::default(), seq));

    // At t=0.5, should be halfway slerp
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    let rot = world.get::<&Transform>(entity).unwrap().rotation;
    let expected_half = Quat::IDENTITY.slerp(target, 0.5);
    assert!(
        rot.dot(expected_half).abs() > 0.99,
        "rotation at t=0.5 should be halfway slerp"
    );

    // Complete
    tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
    let rot = world.get::<&Transform>(entity).unwrap().rotation;
    assert!(
        rot.dot(target).abs() > 0.99,
        "rotation at t=1.0 should be at target"
    );
}

#[test]
fn scale_to_interpolates_scale() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let target = Vec3::new(2.0, 3.0, 4.0);
    let seq = Sequence::new().scale_to(target, 1.0).build();
    let entity = world.spawn((Transform::default(), seq));

    // At t=0.5, should be halfway lerp from (1,1,1) to (2,3,4)
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    let scale = world.get::<&Transform>(entity).unwrap().scale;
    let expected_half = Vec3::ONE.lerp(target, 0.5);
    assert!(
        (scale - expected_half).length() < 0.01,
        "scale at t=0.5 should be halfway, got {:?}",
        scale
    );

    // Complete
    tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
    let scale = world.get::<&Transform>(entity).unwrap().scale;
    assert!(
        (scale - target).length() < 0.01,
        "scale at t=1.0 should be at target, got {:?}",
        scale
    );
}

#[test]
fn move_to_with_easing() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let target = WorldPosition {
        chunk: glam::IVec3::ZERO,
        local: Vec3::new(10.0, 0.0, 0.0),
    };
    let seq = Sequence::new()
        .move_to(target.clone(), 1.0)
        .ease(Ease::InQuad)
        .build();
    let entity = world.spawn((Transform::default(), seq));

    // InQuad at t=0.5 => 0.25 (slower start)
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    let pos = world.get::<&Transform>(entity).unwrap().position.clone();
    // InQuad(0.5) = 0.25, so x should be ~2.5
    assert!(
        (pos.to_vec3().x - 2.5).abs() < 0.2,
        "InQuad at t=0.5 should give x~2.5, got {}",
        pos.to_vec3().x
    );
}

#[test]
fn repeat_step_expands_and_executes() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let seq = Sequence::new()
        .repeat(3, |s| s.emit("ping"))
        .emit("done")
        .build();
    world.spawn((Transform::default(), seq));

    tick_and_flush(&mut world, &mut commands, &mut store, 0.016);

    assert_eq!(store.events("ping").count(), 3);
    assert_eq!(store.events("done").count(), 1);
}

#[test]
fn repeat_with_waits() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let seq = Sequence::new()
        .repeat(2, |s| s.wait(1.0).emit("tick"))
        .build();
    let entity = world.spawn((Transform::default(), seq));

    // After 0.5s: in first wait, no events
    tick_and_flush(&mut world, &mut commands, &mut store, 0.5);
    assert_eq!(store.events("tick").count(), 0);

    // After 0.6s more (total 1.1s): first wait done, first emit fires
    tick_and_flush(&mut world, &mut commands, &mut store, 0.6);
    assert_eq!(store.events("tick").count(), 1);

    store.drain_events();

    // After 1.0s more (total 2.1s): second wait done, second emit fires
    tick_and_flush(&mut world, &mut commands, &mut store, 1.0);
    assert_eq!(store.events("tick").count(), 1);

    store.drain_events();

    // Sequence should be complete
    tick_and_flush(&mut world, &mut commands, &mut store, 0.016);
    assert!(
        world.get::<&Sequence>(entity).is_err(),
        "sequence should be removed after all repeats"
    );
}

#[test]
fn repeat_zero_count_does_nothing() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    let seq = Sequence::new()
        .repeat(0, |s| s.emit("never"))
        .emit("after")
        .build();
    world.spawn((Transform::default(), seq));

    tick_and_flush(&mut world, &mut commands, &mut store, 0.016);

    assert_eq!(store.events("never").count(), 0);
    assert_eq!(store.events("after").count(), 1);
}

#[test]
fn spawn_blueprint_with_registry() {
    use super::blueprint::Blueprint;

    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();
    let mut registry = GameplayRegistry::new();

    // Register the Transform component so spawn_from_blueprint can work.
    // We use engine_components registration for this.
    super::engine_components::engine_register(&mut registry);

    // Add a blueprint to the catalog.
    let bp = Blueprint::new("TestEntity");
    // Blueprint is empty (no gameplay components), so just a Transform will be created.
    registry.blueprint_catalog.insert(bp);

    let start_pos = WorldPosition {
        chunk: glam::IVec3::ZERO,
        local: Vec3::new(5.0, 5.0, 5.0),
    };
    let transform = Transform {
        position: start_pos,
        ..Default::default()
    };

    let seq = Sequence::new().spawn_blueprint("TestEntity").build();
    let _entity = world.spawn((transform, seq));

    let entity_count_before = world.len();

    tick_and_flush_with_registry(
        &mut world,
        &mut commands,
        &mut store,
        0.016,
        Some(&registry),
    );

    // A new entity should have been spawned
    assert!(
        world.len() > entity_count_before,
        "spawn_blueprint should have created a new entity"
    );
}

#[test]
fn spawn_blueprint_missing_logs_warning() {
    // SpawnBlueprint with a name that doesn't exist should not crash
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();
    let registry = GameplayRegistry::new();

    let seq = Sequence::new()
        .spawn_blueprint("NonExistent")
        .emit("after")
        .build();
    world.spawn((Transform::default(), seq));

    // Should not panic, just log a warning
    tick_and_flush_with_registry(
        &mut world,
        &mut commands,
        &mut store,
        0.016,
        Some(&registry),
    );

    // The emit after should still fire
    assert_eq!(store.events("after").count(), 1);
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

#[test]
fn emit_from_despawned_source_substitutes_none() {
    let mut world = hecs::World::new();
    let mut commands = CommandQueue::new();
    let mut store = GameStore::new();

    // Spawn a source entity, then build a sequence referencing it.
    let source_entity = world.spawn((Transform::default(),));
    let seq = Sequence::new()
        .emit_from("test_event", Some(source_entity), None)
        .build();
    let _seq_entity = world.spawn((Transform::default(), seq));

    // Despawn the source before the sequence ticks.
    world.despawn(source_entity).unwrap();
    assert!(!world.contains(source_entity));

    tick_and_flush(&mut world, &mut commands, &mut store, 0.0);

    let events: Vec<_> = store.events("test_event").collect();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].source, None,
        "source should be None for despawned entity, got {:?}",
        events[0].source
    );
}
