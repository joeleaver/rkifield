//! Tests for play_mode module.

use super::play_mode::*;
use super::command_queue::CommandQueue;
use super::executor::BehaviorExecutor;
use super::game_store::GameStore;
use super::game_value::GameValue;
use super::registry::{ComponentEntry, GameplayRegistry, Phase, SystemMeta};
use super::reload_queue::ReloadQueue;
use super::scene_ownership::SceneOwnership;
use super::stable_id::StableId;
use super::stable_id_index::StableIdIndex;
use crate::components::{
    CameraComponent, EditorMetadata, FogVolumeComponent, Parent, SdfTree, Transform,
};
use glam::{IVec3, Quat, Vec3};
use rkf_core::WorldPosition;
use std::path::PathBuf;

/// Helper: build an edit world with several entities for testing.
fn build_test_world() -> (hecs::World, StableIdIndex) {
    let mut world = hecs::World::new();
    let mut index = StableIdIndex::new();

    // Entity A: StableId + Transform + EditorMetadata
    let id_a = StableId::new();
    let entity_a = world.spawn((
        id_a,
        Transform {
            position: WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
            rotation: Quat::from_rotation_y(1.0),
            scale: Vec3::new(2.0, 2.0, 2.0),
        },
        EditorMetadata {
            name: "ObjectA".to_string(),
            tags: vec!["tag1".to_string()],
            locked: false,
        },
    ));
    index.insert(id_a.uuid(), entity_a);

    // Entity B: StableId + Transform + CameraComponent
    let id_b = StableId::new();
    let entity_b = world.spawn((
        id_b,
        Transform::default(),
        CameraComponent {
            fov_degrees: 90.0,
            near: 0.5,
            far: 500.0,
            active: true,
            label: "MainCam".to_string(),
            yaw: 45.0,
            pitch: -10.0,
        },
    ));
    index.insert(id_b.uuid(), entity_b);

    // Entity C: StableId + Transform, child of A
    let id_c = StableId::new();
    let entity_c = world.spawn((
        id_c,
        Transform::default(),
        Parent {
            entity: entity_a,
            bone_index: Some(3),
        },
    ));
    index.insert(id_c.uuid(), entity_c);

    // Entity D: NO StableId — should be skipped
    let _entity_d = world.spawn((Transform::default(),));

    (world, index)
}

// ── 12.1: clone_world_for_play ───────────────────────────────────────

#[test]
fn clone_copies_all_stable_id_entities() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    // 3 entities with StableId cloned (D is skipped).
    assert_eq!(entity_map.len(), 3);
    assert_eq!(play_index.len(), 3);
    // Play world should have exactly 3 entities with StableId.
    let play_count = play_world.query::<&StableId>().iter().count();
    assert_eq!(play_count, 3);
}

#[test]
fn clone_preserves_transform() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, _play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    for (edit_entity, play_entity) in &entity_map {
        if let Ok(edit_t) = edit_world.get::<&Transform>(*edit_entity) {
            let play_t = play_world
                .get::<&Transform>(*play_entity)
                .expect("play entity should have Transform");
            assert_eq!(edit_t.position, play_t.position);
            assert_eq!(edit_t.scale, play_t.scale);
            assert!(
                (edit_t.rotation.x - play_t.rotation.x).abs() < 1e-6
                    && (edit_t.rotation.y - play_t.rotation.y).abs() < 1e-6
                    && (edit_t.rotation.z - play_t.rotation.z).abs() < 1e-6
                    && (edit_t.rotation.w - play_t.rotation.w).abs() < 1e-6
            );
        }
    }
}

#[test]
fn clone_preserves_editor_metadata() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, _play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    for (edit_entity, play_entity) in &entity_map {
        if let Ok(edit_m) = edit_world.get::<&EditorMetadata>(*edit_entity) {
            let play_m = play_world
                .get::<&EditorMetadata>(*play_entity)
                .expect("play entity should have EditorMetadata");
            assert_eq!(edit_m.name, play_m.name);
            assert_eq!(edit_m.tags, play_m.tags);
            assert_eq!(edit_m.locked, play_m.locked);
        }
    }
}

#[test]
fn clone_preserves_camera_component() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, _play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    let mut found_camera = false;
    for (edit_entity, play_entity) in &entity_map {
        if let Ok(edit_c) = edit_world.get::<&CameraComponent>(*edit_entity) {
            let play_c = play_world
                .get::<&CameraComponent>(*play_entity)
                .expect("play entity should have CameraComponent");
            assert!((edit_c.fov_degrees - play_c.fov_degrees).abs() < 1e-6);
            assert_eq!(edit_c.active, play_c.active);
            assert_eq!(edit_c.label, play_c.label);
            assert!((edit_c.yaw - play_c.yaw).abs() < 1e-6);
            assert!((edit_c.pitch - play_c.pitch).abs() < 1e-6);
            found_camera = true;
        }
    }
    assert!(found_camera, "should have found at least one CameraComponent");
}

#[test]
fn clone_remaps_parent_references() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, _play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    // Find entity C (has Parent) in the play world.
    let mut found_parent = false;
    for (_play_entity, (_, parent)) in play_world.query::<(&StableId, &Parent)>().iter() {
        // The parent entity should be a play-world entity, not an edit-world entity.
        assert!(
            play_world.contains(parent.entity),
            "Parent entity should exist in play world"
        );
        // It should NOT be an edit-world entity handle that somehow matches.
        // Verify it maps to the correct edit entity's play counterpart.
        found_parent = true;

        // Verify bone_index is preserved.
        assert_eq!(parent.bone_index, Some(3));
    }
    assert!(found_parent, "should have found a Parent component");

    // Also verify: the play parent entity is the mapped version of the edit parent.
    for (edit_entity, edit_parent) in edit_world.query::<(&StableId, &Parent)>().iter().map(
        |(e, (_, p))| (e, p),
    ) {
        let play_child = entity_map[&edit_entity];
        let play_parent = play_world
            .get::<&Parent>(play_child)
            .expect("play child should have Parent");
        assert_eq!(play_parent.entity, entity_map[&edit_parent.entity]);
    }
}

#[test]
fn clone_stable_ids_match() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    for (edit_entity, play_entity) in &entity_map {
        let edit_id = edit_world
            .get::<&StableId>(*edit_entity)
            .expect("edit entity has StableId");
        let play_id = play_world
            .get::<&StableId>(*play_entity)
            .expect("play entity has StableId");
        assert_eq!(*edit_id, *play_id);

        // Also verify the index maps correctly.
        assert_eq!(
            play_index.get_entity(edit_id.uuid()),
            Some(*play_entity)
        );
    }
}

#[test]
fn clone_skips_entities_without_stable_id() {
    let (edit_world, edit_index) = build_test_world();
    let (play_world, _play_index, entity_map) =
        clone_world_for_play(&edit_world, &edit_index, &GameplayRegistry::new());

    // Edit world has 4 entities total, but only 3 have StableId.
    let edit_total = edit_world.iter().count();
    assert_eq!(edit_total, 4);
    assert_eq!(entity_map.len(), 3);

    let play_total = play_world.iter().count();
    assert_eq!(play_total, 3);
}

#[test]
fn clone_preserves_fog_volume() {
    let mut world = hecs::World::new();
    let mut index = StableIdIndex::new();

    let id = StableId::new();
    let entity = world.spawn((
        id,
        FogVolumeComponent {
            density: 0.7,
            color: [0.1, 0.2, 0.3],
            phase_g: 0.5,
            half_extents: Vec3::new(10.0, 20.0, 30.0),
        },
    ));
    index.insert(id.uuid(), entity);

    let (play_world, _play_index, entity_map) = clone_world_for_play(&world, &index, &GameplayRegistry::new());
    let play_entity = entity_map[&entity];
    let play_fog = play_world
        .get::<&FogVolumeComponent>(play_entity)
        .expect("play entity should have FogVolumeComponent");
    assert!((play_fog.density - 0.7).abs() < 1e-6);
    assert_eq!(play_fog.color, [0.1, 0.2, 0.3]);
    assert!((play_fog.phase_g - 0.5).abs() < 1e-6);
    assert_eq!(play_fog.half_extents, Vec3::new(10.0, 20.0, 30.0));
}

#[test]
fn clone_empty_world() {
    let world = hecs::World::new();
    let index = StableIdIndex::new();

    let (play_world, play_index, entity_map) = clone_world_for_play(&world, &index, &GameplayRegistry::new());
    assert!(entity_map.is_empty());
    assert!(play_index.is_empty());
    assert_eq!(play_world.iter().count(), 0);
}

// ── 12.2: Store snapshot/restore (already in game_store.rs) ──────────

#[test]
fn store_snapshot_captures_all_keys() {
    let mut store = GameStore::new();
    store.set("health", 100_i64);
    store.set("name", "Hero");
    store.set("speed", 3.5_f64);

    let snap = store.snapshot();

    // Mutate store.
    store.set("health", 0_i64);
    store.remove("name");

    // Restore.
    store.restore(snap);
    assert_eq!(store.get::<i64>("health"), Some(100));
    assert_eq!(store.get::<String>("name"), Some("Hero".to_owned()));
    assert_eq!(store.get::<f64>("speed"), Some(3.5));
}

#[test]
fn store_restore_clears_events() {
    let mut store = GameStore::new();
    store.set("x", 1_i64);
    let snap = store.snapshot();

    store.emit("noise", None, None);
    assert_eq!(store.events("noise").count(), 1);

    store.restore(snap);
    assert_eq!(store.events("noise").count(), 0);
}

// ── 12.3: PlayModeManager ────────────────────────────────────────────

#[test]
fn manager_starts_in_edit() {
    let mgr = PlayModeManager::new();
    assert_eq!(mgr.state(), PlayState::Edit);
    assert!(!mgr.is_playing());
}

#[test]
fn manager_start_play_transitions() {
    let (edit_world, edit_index) = build_test_world();
    let store = GameStore::new();
    let mut mgr = PlayModeManager::new();

    let result = mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new());
    assert!(result.is_ok());
    assert_eq!(mgr.state(), PlayState::Playing);
    assert!(mgr.is_playing());
}

#[test]
fn manager_start_play_produces_valid_world() {
    let (edit_world, edit_index) = build_test_world();
    let store = GameStore::new();
    let mut mgr = PlayModeManager::new();

    let (play_world, play_index, entity_map) =
        mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();

    assert_eq!(entity_map.len(), 3);
    assert_eq!(play_index.len(), 3);
    assert_eq!(play_world.query::<&StableId>().iter().count(), 3);
}

#[test]
fn manager_start_play_twice_fails() {
    let (edit_world, edit_index) = build_test_world();
    let store = GameStore::new();
    let mut mgr = PlayModeManager::new();

    mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();
    match mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()) {
        Err(e) => assert!(e.contains("already")),
        Ok(_) => panic!("expected error when starting play twice"),
    }
}

#[test]
fn manager_stop_play_restores_store() {
    let (edit_world, edit_index) = build_test_world();
    let mut store = GameStore::new();
    store.set("score", 0_i64);
    store.set("level", 1_i64);

    let mut mgr = PlayModeManager::new();
    mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();

    // Simulate gameplay modifying the store.
    store.set("score", 9999_i64);
    store.set("level", 42_i64);
    store.set("new_key", true);

    // Stop play — store should be restored to pre-play state.
    mgr.stop_play(&mut store).unwrap();
    assert_eq!(mgr.state(), PlayState::Edit);
    assert!(!mgr.is_playing());
    assert_eq!(store.get::<i64>("score"), Some(0));
    assert_eq!(store.get::<i64>("level"), Some(1));
    assert_eq!(store.get::<bool>("new_key"), None);
}

#[test]
fn manager_stop_play_without_playing_fails() {
    let mut store = GameStore::new();
    let mut mgr = PlayModeManager::new();

    let err = mgr.stop_play(&mut store).unwrap_err();
    assert!(err.contains("not in Play mode"));
}

#[test]
fn manager_full_cycle() {
    let (edit_world, edit_index) = build_test_world();
    let mut store = GameStore::new();
    store.set("health", 100_i64);

    let mut mgr = PlayModeManager::new();

    // Start play.
    let (_play_world, _play_index, _map) =
        mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();
    assert!(mgr.is_playing());

    // Modify store during play.
    store.set("health", 0_i64);

    // Stop play.
    mgr.stop_play(&mut store).unwrap();
    assert!(!mgr.is_playing());
    assert_eq!(store.get::<i64>("health"), Some(100));

    // Can start play again.
    let result = mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new());
    assert!(result.is_ok());
    assert!(mgr.is_playing());
}

#[test]
fn manager_default_trait() {
    let mgr = PlayModeManager::default();
    assert_eq!(mgr.state(), PlayState::Edit);
}

// ── 12.4: Frame execution integration ────────────────────────────────

#[test]
fn should_run_systems_only_during_play() {
    assert!(!should_run_systems(&PlayState::Edit));
    assert!(should_run_systems(&PlayState::Playing));
}

#[test]
fn run_play_frame_executes_tick() {
    // Track that a system ran.
    thread_local! {
        static PLAY_RAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    fn mark_ran(_ctx: &mut super::system_context::SystemContext) {
        PLAY_RAN.with(|c| c.set(true));
    }

    let mut registry = GameplayRegistry::new();
    registry.register_system(SystemMeta {
        name: "mark_ran",
        module_path: "test::mark_ran",
        phase: Phase::Update,
        after: &[],
        before: &[],
        fn_ptr: mark_ran as *const (),
    });

    let mut executor = BehaviorExecutor::new(&registry).unwrap();
    let mut world = hecs::World::new();
    let mut store = GameStore::new();
    let mut stable_ids = StableIdIndex::new();
    let mut commands = CommandQueue::new();

    PLAY_RAN.with(|c| c.set(false));
    run_play_frame(
        &mut executor,
        &mut world,
        &mut store,
        &mut stable_ids,
        &registry,
        &mut commands,
        1.0 / 60.0,
        0.0,
        0,
    );
    assert!(PLAY_RAN.with(|c| c.get()), "system should have run");
}

// ── 12.5: Edit tool disabling during play ────────────────────────────

#[test]
fn is_tool_allowed_blocks_during_play() {
    // During Edit, everything is allowed.
    assert!(is_tool_allowed(&PlayState::Edit, "sculpt"));
    assert!(is_tool_allowed(&PlayState::Edit, "paint"));
    assert!(is_tool_allowed(&PlayState::Edit, "gizmo"));
    assert!(is_tool_allowed(&PlayState::Edit, "screenshot"));

    // During Playing, sculpt/paint/gizmo/inspector_edit are blocked.
    assert!(!is_tool_allowed(&PlayState::Playing, "sculpt"));
    assert!(!is_tool_allowed(&PlayState::Playing, "paint"));
    assert!(!is_tool_allowed(&PlayState::Playing, "gizmo"));
    assert!(!is_tool_allowed(&PlayState::Playing, "inspector_edit"));

    // Observation tools remain allowed during play.
    assert!(is_tool_allowed(&PlayState::Playing, "screenshot"));
    assert!(is_tool_allowed(&PlayState::Playing, "camera_orbit"));
    assert!(is_tool_allowed(&PlayState::Playing, "view"));
}

// ── 12.6: Stop and check reload ──────────────────────────────────────

#[test]
fn stop_and_check_reload_no_pending() {
    let (edit_world, edit_index) = build_test_world();
    let mut store = GameStore::new();
    let mut mgr = PlayModeManager::new();
    let mut reload_queue = ReloadQueue::new();

    mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();

    let result = mgr.stop_and_check_reload(&mut store, &mut reload_queue).unwrap();
    assert!(result.is_none());
    assert_eq!(mgr.state(), PlayState::Edit);
}

#[test]
fn stop_and_check_reload_integration() {
    let (edit_world, edit_index) = build_test_world();
    let mut store = GameStore::new();
    let mut mgr = PlayModeManager::new();
    let mut reload_queue = ReloadQueue::new();

    mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();

    // Simulate a build completing during play.
    reload_queue.queue_reload(PathBuf::from("/tmp/libgame_v2.so"));

    let result = mgr.stop_and_check_reload(&mut store, &mut reload_queue).unwrap();
    assert_eq!(result, Some(PathBuf::from("/tmp/libgame_v2.so")));
    assert_eq!(mgr.state(), PlayState::Edit);

    // Queue should be consumed.
    assert!(!reload_queue.has_pending());
}

#[test]
fn stop_and_check_reload_fails_when_not_playing() {
    let mut store = GameStore::new();
    let mut mgr = PlayModeManager::new();
    let mut reload_queue = ReloadQueue::new();

    let err = mgr.stop_and_check_reload(&mut store, &mut reload_queue).unwrap_err();
    assert!(err.contains("not in Play mode"));
}

// ── 12.7: Scene transitions during play ──────────────────────────────

#[test]
fn load_scene_empty_data_returns_empty() {
    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let result = load_scene_into_play_world(&mut world, b"", &registry);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn load_scene_into_play_world_v3_data() {
    use crate::scene_file_v3::{EntityRecord, SceneFileV3, component_names, serialize_scene_v3};

    let mut scene = SceneFileV3::new();
    let id = uuid::Uuid::from_u128(101);
    let mut record = EntityRecord::new(id);
    record
        .insert_component(
            component_names::TRANSFORM,
            &Transform {
                position: WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        )
        .unwrap();
    record
        .insert_component(
            component_names::EDITOR_METADATA,
            &EditorMetadata {
                name: "TestObj".to_string(),
                tags: vec![],
                locked: false,
            },
        )
        .unwrap();
    scene.entities.push(record);

    let ron_str = serialize_scene_v3(&scene).unwrap();

    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let result = load_scene_into_play_world(&mut world, ron_str.as_bytes(), &registry);
    assert!(result.is_ok());

    let spawned = result.unwrap();
    assert_eq!(spawned.len(), 1);

    // Entity exists in world
    let entity = spawned[0];
    assert!(world.contains(entity));

    // Has SceneOwnership
    let ownership = world.get::<&SceneOwnership>(entity).unwrap();
    assert!(ownership.belongs_to("loaded"));

    // Has StableId
    let sid = world.get::<&StableId>(entity).unwrap();
    assert_eq!(sid.uuid(), id);

    // Has Transform
    let t = world.get::<&Transform>(entity).unwrap();
    assert!((t.position.local.x - 1.0).abs() < 1e-6);

    // Has EditorMetadata
    let m = world.get::<&EditorMetadata>(entity).unwrap();
    assert_eq!(m.name, "TestObj");
}


#[test]
fn load_scene_into_play_world_invalid_utf8() {
    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let result = load_scene_into_play_world(&mut world, &[0xFF, 0xFE], &registry);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("UTF-8"));
}

#[test]
fn loaded_scene_entities_unloadable() {
    // Load a scene, then unload it by scene name
    use crate::scene_file_v3::{EntityRecord, SceneFileV3, component_names, serialize_scene_v3};

    let mut scene = SceneFileV3::new();
    let mut record = EntityRecord::new(uuid::Uuid::from_u128(200));
    record
        .insert_component(
            component_names::TRANSFORM,
            &Transform::default(),
        )
        .unwrap();
    scene.entities.push(record);

    let ron_str = serialize_scene_v3(&scene).unwrap();

    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let spawned = load_scene_into_play_world(&mut world, ron_str.as_bytes(), &registry).unwrap();
    assert_eq!(spawned.len(), 1);
    assert!(world.contains(spawned[0]));

    // Unload by the "loaded" tag
    unload_scene_from_play_world(&mut world, "loaded");
    assert!(!world.contains(spawned[0]));
}

#[test]
fn unload_scene_removes_owned_entities() {
    let mut world = hecs::World::new();

    // Entities owned by "level_01"
    let e1 = world.spawn((SceneOwnership::for_scene("level_01"),));
    let e2 = world.spawn((SceneOwnership::for_scene("level_01"),));
    // Entity owned by "level_02" — should survive
    let e3 = world.spawn((SceneOwnership::for_scene("level_02"),));
    // Persistent entity — should survive
    let e4 = world.spawn((SceneOwnership::persistent(),));
    // Entity without SceneOwnership — should survive
    let e5 = world.spawn((Transform::default(),));

    unload_scene_from_play_world(&mut world, "level_01");

    // level_01 entities removed
    assert!(!world.contains(e1));
    assert!(!world.contains(e2));
    // Others survive
    assert!(world.contains(e3));
    assert!(world.contains(e4));
    assert!(world.contains(e5));
}

#[test]
fn unload_scene_noop_for_unknown_scene() {
    let mut world = hecs::World::new();
    let e1 = world.spawn((SceneOwnership::for_scene("level_01"),));

    unload_scene_from_play_world(&mut world, "nonexistent");

    // Nothing removed
    assert!(world.contains(e1));
}

// ── 12.9: Push field to edit ─────────────────────────────────────────

#[test]
fn push_field_to_edit_fails_unknown_component() {
    let play_world = hecs::World::new();
    let mut edit_world = hecs::World::new();
    let registry = GameplayRegistry::new();

    let play_entity = play_world.reserve_entity();
    let edit_entity = edit_world.reserve_entity();

    let result = push_field_to_edit(
        &play_world,
        &mut edit_world,
        play_entity,
        edit_entity,
        "NonexistentComponent",
        "field",
        &registry,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ── 12.9: Push field to edit — success path ─────────────────────────

/// A minimal component for testing push_field_to_edit success.
struct Health {
    value: i64,
}

/// Build a registry with a "Health" component that supports get/set on "value".
fn registry_with_health() -> GameplayRegistry {
    let entry = ComponentEntry {
        name: "Health",
        serialize: |_, _| None,
        deserialize_insert: |_, _, _| Ok(()),
        remove: |_, _| {},
        has: |world, entity| world.get::<&Health>(entity).is_ok(),
        get_field: |world, entity, field| {
            if field != "value" {
                return Err(format!("unknown field '{}'", field));
            }
            let h = world
                .get::<&Health>(entity)
                .map_err(|_| "entity missing Health".to_string())?;
            Ok(GameValue::Int(h.value))
        },
        set_field: |world, entity, field, val| {
            if field != "value" {
                return Err(format!("unknown field '{}'", field));
            }
            let new_val = match val {
                GameValue::Int(v) => v,
                _ => return Err("expected Int".to_string()),
            };
            let mut h = world
                .get::<&mut Health>(entity)
                .map_err(|_| "entity missing Health".to_string())?;
            h.value = new_val;
            Ok(())
        },
        meta: &[],
    };

    let mut registry = GameplayRegistry::new();
    registry.register_component(entry).unwrap();
    registry
}

#[test]
fn push_field_to_edit_success() {
    let registry = registry_with_health();

    // Edit world: entity with Health { value: 100 }.
    let mut edit_world = hecs::World::new();
    let edit_entity = edit_world.spawn((Health { value: 100 },));

    // Play world: cloned entity with modified health.
    let mut play_world = hecs::World::new();
    let play_entity = play_world.spawn((Health { value: 42 },));

    // Push the modified value from play to edit.
    push_field_to_edit(
        &play_world,
        &mut edit_world,
        play_entity,
        edit_entity,
        "Health",
        "value",
        &registry,
    )
    .expect("push_field_to_edit should succeed");

    // Edit world should now have the play world's value.
    let h = edit_world.get::<&Health>(edit_entity).unwrap();
    assert_eq!(h.value, 42);
}

// ── 10.6: Reload play-blocking (embedded queue) ─────────────────────

#[test]
fn reload_blocked_during_play_then_applied_on_stop() {
    let (edit_world, edit_index) = build_test_world();
    let mut store = GameStore::new();
    let mut mgr = PlayModeManager::new();

    mgr.start_play(&edit_world, &edit_index, &store, &GameplayRegistry::new()).unwrap();

    // Simulate a build completing during play — queue via embedded reload queue.
    mgr.reload_queue_mut()
        .queue_reload(PathBuf::from("/tmp/libgame_v3.so"));

    // The reload is pending but not applied (we're still playing).
    assert!(mgr.is_playing());

    // Stop play and drain the embedded reload queue.
    let pending = mgr.stop_and_drain_reload(&mut store).unwrap();
    assert_eq!(pending, Some(PathBuf::from("/tmp/libgame_v3.so")));
    assert_eq!(mgr.state(), PlayState::Edit);

    // Queue is consumed — no double-apply.
    assert!(!mgr.reload_queue_mut().has_pending());
}
