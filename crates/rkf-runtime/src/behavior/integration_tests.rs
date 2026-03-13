//! End-to-end integration tests for the behavior system.
//!
//! These tests exercise the full pipeline: registry, components, systems,
//! executor, inspector, blueprints, edit pipeline, and play mode isolation.
//!
//! All component and system definitions are test-local — no dependency on
//! example_game or example_game_systems.

#[cfg(test)]
mod tests {
    use crate::behavior::blueprint::{Blueprint, create_blueprint_from_entity};
    use crate::behavior::command_queue::CommandQueue;
    use crate::behavior::edit_pipeline::{EditOp, EditPipeline};
    use crate::behavior::engine_components::engine_register;
    use crate::behavior::engine_persist::{
        restore_engine_state_from_store, sync_engine_state_to_store,
    };
    use crate::behavior::executor::BehaviorExecutor;
    use crate::behavior::game_store::GameStore;
    use crate::behavior::game_value::GameValue;
    use crate::behavior::inspector::build_inspector_data;
    use crate::behavior::registry::{
        ComponentEntry, FieldMeta, FieldType, GameplayRegistry, Phase, SystemMeta,
    };
    use crate::behavior::scheduler::build_schedule;
    use crate::behavior::stable_id::StableId;
    use crate::behavior::stable_id_index::StableIdIndex;
    use crate::behavior::system_context::SystemContext;
    use crate::components::{EditorMetadata, Transform};
    use glam::Vec3;
    use rkf_core::WorldPosition;
    use std::collections::HashMap;

    // ─── Test-only component structs ─────────────────────────────────────

    struct TestHealth {
        current: f32,
        max: f32,
    }

    struct TestCounter {
        value: i32,
    }

    struct TestDoor {
        open: bool,
    }

    // ─── Field metadata (static) ─────────────────────────────────────────

    static TEST_HEALTH_FIELDS: [FieldMeta; 2] = [
        FieldMeta {
            name: "current",
            field_type: FieldType::Float,
            transient: false,
            range: Some((0.0, 1000.0)),
            default: None,
            persist: true,
        },
        FieldMeta {
            name: "max",
            field_type: FieldType::Float,
            transient: false,
            range: Some((0.0, 1000.0)),
            default: None,
            persist: false,
        },
    ];

    static TEST_COUNTER_FIELDS: [FieldMeta; 1] = [FieldMeta {
        name: "value",
        field_type: FieldType::Int,
        transient: false,
        range: Some((0.0, 9999.0)),
        default: None,
        persist: true,
    }];

    static TEST_DOOR_FIELDS: [FieldMeta; 1] = [FieldMeta {
        name: "open",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: true,
    }];

    // ─── ComponentEntry constructors ─────────────────────────────────────

    fn test_health_entry() -> ComponentEntry {
        ComponentEntry {
            name: "TestHealth",
            meta: &TEST_HEALTH_FIELDS,
            serialize: |world, entity| {
                world
                    .get::<&TestHealth>(entity)
                    .ok()
                    .map(|c| format!("(current: {}, max: {})", c.current, c.max))
            },
            deserialize_insert: |world, entity, _ron_str| {
                world
                    .insert_one(entity, TestHealth { current: 100.0, max: 100.0 })
                    .map_err(|e| e.to_string())
            },
            has: |world, entity| world.get::<&TestHealth>(entity).is_ok(),
            remove: |world, entity| {
                let _ = world.remove_one::<TestHealth>(entity);
            },
            get_field: |world, entity, field_name| {
                let c = world
                    .get::<&TestHealth>(entity)
                    .map_err(|_| "entity does not have component 'TestHealth'".to_string())?;
                match field_name {
                    "current" => Ok(GameValue::Float(c.current as f64)),
                    "max" => Ok(GameValue::Float(c.max as f64)),
                    _ => Err(format!("unknown field '{}' on component 'TestHealth'", field_name)),
                }
            },
            set_field: |world, entity, field_name, value| {
                let mut c = world
                    .get::<&mut TestHealth>(entity)
                    .map_err(|_| "entity does not have component 'TestHealth'".to_string())?;
                match field_name {
                    "current" => match value {
                        GameValue::Float(f) => c.current = f as f32,
                        _ => return Err("type mismatch for field 'current'".into()),
                    },
                    "max" => match value {
                        GameValue::Float(f) => c.max = f as f32,
                        _ => return Err("type mismatch for field 'max'".into()),
                    },
                    _ => {
                        return Err(format!(
                            "unknown field '{}' on component 'TestHealth'",
                            field_name
                        ))
                    }
                }
                Ok(())
            },
        }
    }

    fn test_counter_entry() -> ComponentEntry {
        ComponentEntry {
            name: "TestCounter",
            meta: &TEST_COUNTER_FIELDS,
            serialize: |world, entity| {
                world
                    .get::<&TestCounter>(entity)
                    .ok()
                    .map(|c| format!("(value: {})", c.value))
            },
            deserialize_insert: |world, entity, _ron_str| {
                world
                    .insert_one(entity, TestCounter { value: 0 })
                    .map_err(|e| e.to_string())
            },
            has: |world, entity| world.get::<&TestCounter>(entity).is_ok(),
            remove: |world, entity| {
                let _ = world.remove_one::<TestCounter>(entity);
            },
            get_field: |world, entity, field_name| {
                let c = world
                    .get::<&TestCounter>(entity)
                    .map_err(|_| "entity does not have component 'TestCounter'".to_string())?;
                match field_name {
                    "value" => Ok(GameValue::Int(c.value as i64)),
                    _ => Err(format!(
                        "unknown field '{}' on component 'TestCounter'",
                        field_name
                    )),
                }
            },
            set_field: |world, entity, field_name, value| {
                let mut c = world
                    .get::<&mut TestCounter>(entity)
                    .map_err(|_| "entity does not have component 'TestCounter'".to_string())?;
                match field_name {
                    "value" => match value {
                        GameValue::Int(i) => c.value = i as i32,
                        _ => return Err("type mismatch for field 'value'".into()),
                    },
                    _ => {
                        return Err(format!(
                            "unknown field '{}' on component 'TestCounter'",
                            field_name
                        ))
                    }
                }
                Ok(())
            },
        }
    }

    fn test_door_entry() -> ComponentEntry {
        ComponentEntry {
            name: "TestDoor",
            meta: &TEST_DOOR_FIELDS,
            serialize: |world, entity| {
                world
                    .get::<&TestDoor>(entity)
                    .ok()
                    .map(|c| format!("(open: {})", c.open))
            },
            deserialize_insert: |world, entity, _ron_str| {
                world
                    .insert_one(entity, TestDoor { open: false })
                    .map_err(|e| e.to_string())
            },
            has: |world, entity| world.get::<&TestDoor>(entity).is_ok(),
            remove: |world, entity| {
                let _ = world.remove_one::<TestDoor>(entity);
            },
            get_field: |world, entity, field_name| {
                let c = world
                    .get::<&TestDoor>(entity)
                    .map_err(|_| "entity does not have component 'TestDoor'".to_string())?;
                match field_name {
                    "open" => Ok(GameValue::Bool(c.open)),
                    _ => Err(format!(
                        "unknown field '{}' on component 'TestDoor'",
                        field_name
                    )),
                }
            },
            set_field: |world, entity, field_name, value| {
                let mut c = world
                    .get::<&mut TestDoor>(entity)
                    .map_err(|_| "entity does not have component 'TestDoor'".to_string())?;
                match field_name {
                    "open" => match value {
                        GameValue::Bool(b) => c.open = b,
                        _ => return Err("type mismatch for field 'open'".into()),
                    },
                    _ => {
                        return Err(format!(
                            "unknown field '{}' on component 'TestDoor'",
                            field_name
                        ))
                    }
                }
                Ok(())
            },
        }
    }

    // ─── Test system functions ────────────────────────────────────────────

    /// Counts TestCounter entities and writes the count to the game store.
    fn count_system(ctx: &mut SystemContext) {
        let count = ctx.query::<(&TestCounter,)>().iter().count();
        if count > 0 {
            ctx.store().set("test_counter_count", count as i64);
        }
    }

    /// Despawns entities with TestHealth.current <= 0.
    fn death_test_system(ctx: &mut SystemContext) {
        let dead: Vec<_> = ctx
            .query::<(&TestHealth,)>()
            .iter()
            .filter(|(_, (h,))| h.current <= 0.0)
            .map(|(entity, _)| entity)
            .collect();

        for entity in dead {
            ctx.despawn(entity);
        }
    }

    /// No-op system for stress/scheduling tests.
    fn noop_system(_ctx: &mut SystemContext) {}

    // ─── Registry helper ──────────────────────────────────────────────────

    /// Create a registry with engine components + test-only components and systems.
    fn test_registry() -> GameplayRegistry {
        let mut reg = GameplayRegistry::new();
        engine_register(&mut reg);
        reg.register_component(test_health_entry()).unwrap();
        reg.register_component(test_counter_entry()).unwrap();
        reg.register_component(test_door_entry()).unwrap();

        reg.register_system(SystemMeta {
            name: "count_system",
            module_path: "integration_tests::count_system",
            phase: Phase::LateUpdate,
            after: &[],
            before: &[],
            fn_ptr: count_system as *const (),
        });

        reg.register_system(SystemMeta {
            name: "death_test_system",
            module_path: "integration_tests::death_test_system",
            phase: Phase::Update,
            after: &[],
            before: &[],
            fn_ptr: death_test_system as *const (),
        });

        reg
    }

    // ── 1. Full workflow test ───────────────────────────────────────────

    #[test]
    fn full_workflow_tick_10_frames() {
        let registry = test_registry();
        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();

        // Spawn an entity with TestHealth and TestCounter
        let entity_a = world.spawn((
            Transform::default(),
            EditorMetadata {
                name: "Entity_A".into(),
                tags: vec!["test".into()],
                locked: false,
            },
            TestHealth { current: 100.0, max: 100.0 },
            TestCounter { value: 42 },
        ));

        // Spawn another TestCounter entity
        let entity_b = world.spawn((TestCounter { value: 7 },));

        // Spawn a door entity
        let door = world.spawn((TestDoor { open: false },));

        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test".into()]);
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        // Tick for 10 frames
        for frame in 0..10 {
            executor.tick(
                &registry,
                &mut world,
                &mut commands,
                &mut store,
                &mut stable_ids,
                1.0 / 60.0,
                frame as f64 / 60.0,
                frame,
            );
        }

        // Verify entities still valid (no panics during execution)
        assert!(world.contains(entity_a));
        assert!(world.contains(entity_b));
        assert!(world.contains(door));

        // Verify count_system wrote to store (2 TestCounter entities)
        let count = store.get::<i64>("test_counter_count");
        assert_eq!(count, Some(2));
    }

    // ── 2. Play isolation test ──────────────────────────────────────────

    #[test]
    fn play_world_isolation() {
        let mut edit_world = hecs::World::new();
        let stable_id = StableId::new();
        let edit_entity = edit_world.spawn((
            stable_id,
            Transform {
                position: WorldPosition::default(),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            EditorMetadata {
                name: "Original".into(),
                tags: vec![],
                locked: false,
            },
        ));

        let mut edit_stable_index = StableIdIndex::new();
        edit_stable_index.insert(stable_id.uuid(), edit_entity);

        // Clone for play
        use crate::behavior::play_mode::clone_world_for_play;
        let (mut play_world, _play_stable_index, entity_map) =
            clone_world_for_play(&edit_world, &edit_stable_index, &GameplayRegistry::new());

        let play_entity = entity_map[&edit_entity];

        // Modify play world
        {
            let mut meta = play_world.get::<&mut EditorMetadata>(play_entity).unwrap();
            meta.name = "Modified".into();
        }

        // Verify edit world is unchanged
        let edit_meta = edit_world.get::<&EditorMetadata>(edit_entity).unwrap();
        assert_eq!(edit_meta.name, "Original");

        // Verify play world was modified
        let play_meta = play_world.get::<&EditorMetadata>(play_entity).unwrap();
        assert_eq!(play_meta.name, "Modified");
    }

    // ── 3. Save/load test ───────────────────────────────────────────────

    #[test]
    fn engine_state_save_load_roundtrip() {
        let mut store = GameStore::new();

        let cam_pos = WorldPosition::new(glam::IVec3::new(1, 2, 3), Vec3::new(0.5, 1.0, 1.5));
        let cam_rot = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let cam_fov = 75.0_f32;
        let scenes = vec!["scene_a.rkscene".to_string(), "scene_b.rkscene".to_string()];

        sync_engine_state_to_store(&mut store, cam_pos.clone(), cam_rot, cam_fov, &scenes);

        let snapshot = restore_engine_state_from_store(&store).expect("should restore");

        assert_eq!(snapshot.camera_position, cam_pos);
        // Quat comparison with tolerance
        let q = snapshot.camera_rotation;
        assert!((q.x - cam_rot.x).abs() < 1e-6);
        assert!((q.y - cam_rot.y).abs() < 1e-6);
        assert!((q.z - cam_rot.z).abs() < 1e-6);
        assert!((q.w - cam_rot.w).abs() < 1e-6);
        assert!((snapshot.camera_fov - cam_fov).abs() < 1e-3);
        assert_eq!(snapshot.loaded_scenes, scenes);
    }

    // ── 4. Inspector test ───────────────────────────────────────────────

    #[test]
    fn inspector_shows_all_components_and_fields() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            EditorMetadata {
                name: "InspectorTest".into(),
                tags: vec![],
                locked: false,
            },
            TestHealth { current: 80.0, max: 100.0 },
        ));

        let data = build_inspector_data(&world, entity, &registry);
        assert_eq!(data.entity, entity);

        // Should have 3 components: Transform, EditorMetadata, TestHealth
        assert_eq!(data.components.len(), 3);

        let comp_names: Vec<&str> = data.components.iter().map(|c| c.name.as_str()).collect();
        assert!(comp_names.contains(&"Transform"));
        assert!(comp_names.contains(&"EditorMetadata"));
        assert!(comp_names.contains(&"TestHealth"));

        // Verify TestHealth fields
        let health_data = data
            .components
            .iter()
            .find(|c| c.name == "TestHealth")
            .expect("should have TestHealth");
        assert_eq!(health_data.fields.len(), 2);

        let current_field = health_data
            .fields
            .iter()
            .find(|f| f.name == "current")
            .expect("should have 'current' field");
        assert!((current_field.value.as_float().unwrap() - 80.0).abs() < 1e-6);

        // TestHealth should be removable (not mandatory)
        assert!(health_data.removable);
    }

    // ── 5. Blueprint test ───────────────────────────────────────────────

    #[test]
    fn blueprint_create_and_spawn() {
        let registry = test_registry();
        let mut world = hecs::World::new();

        // Spawn original entity
        let original = world.spawn((
            StableId::new(),
            TestHealth { current: 100.0, max: 100.0 },
            TestCounter { value: 50 },
        ));

        // Create blueprint from entity
        let bp = create_blueprint_from_entity(
            "TestBlueprint".to_string(),
            original,
            &world,
            &registry,
        );

        assert_eq!(bp.name, "TestBlueprint");
        // StableId is excluded from blueprints
        assert!(!bp.components.contains_key("StableId"));
        // TestHealth and TestCounter should be present
        assert!(bp.components.contains_key("TestHealth"));
        assert!(bp.components.contains_key("TestCounter"));

        // Verify inline test blueprints are structurally correct
        let health_bp = {
            let mut components = HashMap::new();
            components.insert(
                "TestHealth".to_owned(),
                "(current: 100.0, max: 100.0)".to_owned(),
            );
            components.insert(
                "TestCounter".to_owned(),
                "(value: 0)".to_owned(),
            );
            Blueprint {
                name: "TestEntity".to_owned(),
                components,
            }
        };
        assert_eq!(health_bp.components.len(), 2);

        let counter_bp = {
            let mut components = HashMap::new();
            components.insert(
                "TestCounter".to_owned(),
                "(value: 10)".to_owned(),
            );
            Blueprint {
                name: "TestCounter".to_owned(),
                components,
            }
        };
        assert_eq!(counter_bp.components.len(), 1);
    }

    // ── 6. Edit pipeline test ───────────────────────────────────────────

    #[test]
    fn edit_pipeline_set_property_undo_redo() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((TestHealth { current: 100.0, max: 100.0 },));

        let mut pipeline = EditPipeline::new();

        // Apply: set current to 50
        pipeline
            .apply(
                EditOp::SetProperty {
                    entity,
                    component_name: "TestHealth".to_string(),
                    field_name: "current".to_string(),
                    value: GameValue::Float(50.0),
                },
                &mut world,
                &registry,
            )
            .unwrap();

        // Verify the new value
        let entry = registry.component_entry("TestHealth").unwrap();
        let val = (entry.get_field)(&world, entity, "current").unwrap();
        assert!((val.as_float().unwrap() - 50.0).abs() < 1e-6);
        assert!(pipeline.undo_stack().can_undo());

        // Undo: should restore to 100
        let undone = pipeline.undo(&mut world, &registry).unwrap();
        assert!(undone.is_some());
        let val = (entry.get_field)(&world, entity, "current").unwrap();
        assert!((val.as_float().unwrap() - 100.0).abs() < 1e-6);
        assert!(pipeline.undo_stack().can_redo());

        // Redo: should go back to 50
        let redone = pipeline.redo(&mut world, &registry).unwrap();
        assert!(redone.is_some());
        let val = (entry.get_field)(&world, entity, "current").unwrap();
        assert!((val.as_float().unwrap() - 50.0).abs() < 1e-6);
    }

    // ── 7. Scheduler stress test ────────────────────────────────────────

    #[test]
    fn scheduler_stress_20_systems_100_entities() {
        let mut registry = GameplayRegistry::new();
        registry.register_component(test_health_entry()).unwrap();
        registry.register_component(test_counter_entry()).unwrap();

        // Register a couple of real test systems
        registry.register_system(SystemMeta {
            name: "count_system",
            module_path: "integration_tests::count_system",
            phase: Phase::LateUpdate,
            after: &[],
            before: &[],
            fn_ptr: count_system as *const (),
        });
        registry.register_system(SystemMeta {
            name: "death_test_system",
            module_path: "integration_tests::death_test_system",
            phase: Phase::Update,
            after: &[],
            before: &[],
            fn_ptr: death_test_system as *const (),
        });

        // Add dummy systems to reach 20 total
        let existing = registry.system_list().len();
        let needed = 20 - existing;
        for i in 0..needed {
            let name: &'static str = Box::leak(format!("stress_system_{i}").into_boxed_str());
            registry.register_system(SystemMeta {
                name,
                module_path: name,
                phase: if i % 2 == 0 {
                    Phase::Update
                } else {
                    Phase::LateUpdate
                },
                after: &[],
                before: &[],
                fn_ptr: noop_system as *const (),
            });
        }

        assert_eq!(registry.system_list().len(), 20);

        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        // Spawn 100 entities with TestHealth
        for i in 0..100 {
            world.spawn((TestHealth {
                current: 100.0 - i as f32,
                max: 100.0,
            },));
        }

        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test".into()]);
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        let start = std::time::Instant::now();
        executor.tick(
            &registry,
            &mut world,
            &mut commands,
            &mut store,
            &mut stable_ids,
            1.0 / 60.0,
            0.0,
            0,
        );
        let elapsed = start.elapsed();

        // Assert < 10ms per tick (generous for CI)
        assert!(
            elapsed.as_millis() < 10,
            "tick took {}ms, expected < 10ms",
            elapsed.as_millis()
        );
    }

    // ── 8. Death system actually despawns ────────────────────────────────

    #[test]
    fn death_system_despawns_dead_entities() {
        let registry = test_registry();
        let mut executor = BehaviorExecutor::new(&registry).unwrap();

        let mut world = hecs::World::new();
        let alive = world.spawn((TestHealth { current: 50.0, max: 100.0 },));
        let dead = world.spawn((TestHealth { current: 0.0, max: 100.0 },));
        let also_dead = world.spawn((TestHealth { current: -10.0, max: 100.0 },));

        let mut commands = CommandQueue::new();
        commands.set_loaded_scenes(vec!["test".into()]);
        let mut store = GameStore::new();
        let mut stable_ids = StableIdIndex::new();

        executor.tick(
            &registry,
            &mut world,
            &mut commands,
            &mut store,
            &mut stable_ids,
            1.0 / 60.0,
            0.0,
            0,
        );

        // Alive entity should still exist
        assert!(world.contains(alive));
        // Dead entities should be despawned
        assert!(!world.contains(dead));
        assert!(!world.contains(also_dead));
    }

    // ── 9. Mixed engine + gameplay components in inspector ──────────────

    #[test]
    fn inspector_mixed_engine_and_gameplay() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            TestHealth { current: 100.0, max: 100.0 },
            TestCounter { value: 5 },
            TestDoor { open: true },
        ));

        let data = build_inspector_data(&world, entity, &registry);

        // Should see all 4 components
        assert_eq!(data.components.len(), 4);

        let comp_names: Vec<&str> = data.components.iter().map(|c| c.name.as_str()).collect();
        assert!(comp_names.contains(&"Transform"));
        assert!(comp_names.contains(&"TestHealth"));
        assert!(comp_names.contains(&"TestCounter"));
        assert!(comp_names.contains(&"TestDoor"));

        // Transform is mandatory (not removable), gameplay components are removable
        let transform_comp = data.components.iter().find(|c| c.name == "Transform").unwrap();
        assert!(!transform_comp.removable);

        let health_comp = data.components.iter().find(|c| c.name == "TestHealth").unwrap();
        assert!(health_comp.removable);
    }

    // ── 10. Schedule builds correctly with dependencies ─────────────────

    #[test]
    fn schedule_respects_test_system_dependencies() {
        let mut registry = GameplayRegistry::new();

        // Register systems with explicit ordering constraints
        registry.register_system(SystemMeta {
            name: "setup_system",
            module_path: "integration_tests::setup_system",
            phase: Phase::Update,
            after: &[],
            before: &["death_test_system"],
            fn_ptr: noop_system as *const (),
        });

        registry.register_system(SystemMeta {
            name: "death_test_system",
            module_path: "integration_tests::death_test_system",
            phase: Phase::Update,
            after: &["setup_system"],
            before: &[],
            fn_ptr: death_test_system as *const (),
        });

        registry.register_system(SystemMeta {
            name: "count_system",
            module_path: "integration_tests::count_system",
            phase: Phase::LateUpdate,
            after: &[],
            before: &[],
            fn_ptr: count_system as *const (),
        });

        let schedule = build_schedule(registry.system_list()).unwrap();

        // setup_system has before: ["death_test_system"]
        // death_test_system has after: ["setup_system"]
        // So in the Update phase, setup must come before death.
        let systems = registry.system_list();
        let setup_idx = schedule
            .update
            .iter()
            .position(|&i| systems[i].name == "setup_system")
            .expect("setup_system should be in update");
        let death_idx = schedule
            .update
            .iter()
            .position(|&i| systems[i].name == "death_test_system")
            .expect("death_test_system should be in update");

        assert!(
            setup_idx < death_idx,
            "setup_system should run before death_test_system"
        );

        // count_system should be in late_update
        assert!(
            schedule
                .late_update
                .iter()
                .any(|&i| systems[i].name == "count_system"),
            "count_system should be in late_update"
        );
    }
}
