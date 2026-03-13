use super::edit_pipeline::*;
use super::game_value::GameValue;
use super::registry::GameplayRegistry;
use super::stable_id::StableId;
use uuid::Uuid;

// ── UndoStack unit tests ─────────────────────────────────────────────

fn dummy_action(label: &str) -> UndoAction {
    UndoAction::GeometryEdit {
        entity: hecs::Entity::DANGLING,
        description: label.to_string(),
    }
}

fn prop_action(old: i64, new: i64) -> UndoAction {
    UndoAction::PropertyChange {
        entity: hecs::Entity::DANGLING,
        component_name: "Comp".to_string(),
        field_name: "field".to_string(),
        old_value: GameValue::Int(old),
        new_value: GameValue::Int(new),
    }
}

#[test]
fn stack_empty_initial() {
    let stack = UndoStack::new();
    assert!(!stack.can_undo());
    assert!(!stack.can_redo());
    assert_eq!(stack.undo_count(), 0);
    assert_eq!(stack.redo_count(), 0);
}

#[test]
fn stack_push_and_undo() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("a"));
    stack.push(dummy_action("b"));
    assert_eq!(stack.undo_count(), 2);
    assert!(stack.can_undo());

    let action = stack.undo().unwrap();
    match &action {
        UndoAction::GeometryEdit { description, .. } => assert_eq!(description, "b"),
        _ => panic!("wrong variant"),
    }
    assert_eq!(stack.undo_count(), 1);
    assert_eq!(stack.redo_count(), 1);
}

#[test]
fn stack_undo_then_redo() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("a"));
    stack.undo();
    assert!(stack.can_redo());

    let action = stack.redo().unwrap();
    match &action {
        UndoAction::GeometryEdit { description, .. } => assert_eq!(description, "a"),
        _ => panic!("wrong variant"),
    }
    assert_eq!(stack.undo_count(), 1);
    assert_eq!(stack.redo_count(), 0);
}

#[test]
fn stack_push_clears_redo() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("a"));
    stack.push(dummy_action("b"));
    stack.undo(); // redo has "b"
    assert!(stack.can_redo());

    stack.push(dummy_action("c")); // should clear redo
    assert!(!stack.can_redo());
    assert_eq!(stack.redo_count(), 0);
    assert_eq!(stack.undo_count(), 2); // "a" and "c"
}

#[test]
fn stack_undo_empty_returns_none() {
    let mut stack = UndoStack::new();
    assert!(stack.undo().is_none());
}

#[test]
fn stack_redo_empty_returns_none() {
    let mut stack = UndoStack::new();
    assert!(stack.redo().is_none());
}

#[test]
fn stack_clear() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("a"));
    stack.push(dummy_action("b"));
    stack.undo();
    stack.clear();
    assert!(!stack.can_undo());
    assert!(!stack.can_redo());
    assert_eq!(stack.undo_count(), 0);
    assert_eq!(stack.redo_count(), 0);
}

#[test]
fn stack_coalesce_multiple() {
    let mut stack = UndoStack::new();
    stack.begin_coalesce();
    stack.push(dummy_action("a"));
    stack.push(dummy_action("b"));
    stack.push(dummy_action("c"));
    stack.end_coalesce();

    assert_eq!(stack.undo_count(), 1);
    let action = stack.undo().unwrap();
    match action {
        UndoAction::Coalesced { actions } => assert_eq!(actions.len(), 3),
        _ => panic!("expected Coalesced"),
    }
}

#[test]
fn stack_coalesce_single_unwraps() {
    let mut stack = UndoStack::new();
    stack.begin_coalesce();
    stack.push(dummy_action("only"));
    stack.end_coalesce();

    assert_eq!(stack.undo_count(), 1);
    let action = stack.undo().unwrap();
    match action {
        UndoAction::GeometryEdit { description, .. } => assert_eq!(description, "only"),
        _ => panic!("single coalesce should unwrap"),
    }
}

#[test]
fn stack_coalesce_empty_noop() {
    let mut stack = UndoStack::new();
    stack.begin_coalesce();
    stack.end_coalesce();
    assert_eq!(stack.undo_count(), 0);
}

#[test]
fn stack_coalesce_clears_redo() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("a"));
    stack.undo();
    assert!(stack.can_redo());

    stack.begin_coalesce();
    stack.push(dummy_action("b"));
    stack.end_coalesce();
    assert!(!stack.can_redo());
}

#[test]
fn stack_multiple_undo_redo_cycles() {
    let mut stack = UndoStack::new();
    stack.push(dummy_action("1"));
    stack.push(dummy_action("2"));
    stack.push(dummy_action("3"));

    // Undo all
    stack.undo();
    stack.undo();
    stack.undo();
    assert_eq!(stack.undo_count(), 0);
    assert_eq!(stack.redo_count(), 3);

    // Redo all
    stack.redo();
    stack.redo();
    stack.redo();
    assert_eq!(stack.undo_count(), 3);
    assert_eq!(stack.redo_count(), 0);
}

// ── UndoAction variant tests ─────────────────────────────────────────

#[test]
fn property_change_stores_both_values() {
    let action = prop_action(10, 20);
    match &action {
        UndoAction::PropertyChange {
            old_value,
            new_value,
            ..
        } => {
            assert_eq!(old_value, &GameValue::Int(10));
            assert_eq!(new_value, &GameValue::Int(20));
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn component_add_action() {
    let action = UndoAction::ComponentAdd {
        entity: hecs::Entity::DANGLING,
        component_name: "Health".to_string(),
        data: "(hp: 100)".to_string(),
    };
    match &action {
        UndoAction::ComponentAdd {
            component_name,
            data,
            ..
        } => {
            assert_eq!(component_name, "Health");
            assert_eq!(data, "(hp: 100)");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn component_remove_action() {
    let action = UndoAction::ComponentRemove {
        entity: hecs::Entity::DANGLING,
        component_name: "Health".to_string(),
        data: "(hp: 50)".to_string(),
    };
    match &action {
        UndoAction::ComponentRemove {
            component_name,
            data,
            ..
        } => {
            assert_eq!(component_name, "Health");
            assert_eq!(data, "(hp: 50)");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn entity_spawn_action() {
    let uuid = Uuid::new_v4();
    let action = UndoAction::EntitySpawn {
        entity: hecs::Entity::DANGLING,
        components: vec![("StableId".to_string(), uuid.to_string())],
        stable_id: uuid,
        parent: None,
    };
    match &action {
        UndoAction::EntitySpawn {
            stable_id,
            parent,
            components,
            ..
        } => {
            assert_eq!(*stable_id, uuid);
            assert!(parent.is_none());
            assert_eq!(components.len(), 1);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn entity_despawn_action() {
    let uuid = Uuid::new_v4();
    let parent_uuid = Uuid::new_v4();
    let action = UndoAction::EntityDespawn {
        entity: hecs::Entity::DANGLING,
        components: vec![
            ("StableId".to_string(), uuid.to_string()),
            ("Transform".to_string(), "(pos: (0,0,0))".to_string()),
        ],
        stable_id: uuid,
        parent: Some(parent_uuid),
    };
    match &action {
        UndoAction::EntityDespawn {
            stable_id,
            parent,
            components,
            ..
        } => {
            assert_eq!(*stable_id, uuid);
            assert_eq!(*parent, Some(parent_uuid));
            assert_eq!(components.len(), 2);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn geometry_edit_action() {
    let action = UndoAction::GeometryEdit {
        entity: hecs::Entity::DANGLING,
        description: "sculpt add sphere r=2".to_string(),
    };
    match &action {
        UndoAction::GeometryEdit { description, .. } => {
            assert_eq!(description, "sculpt add sphere r=2");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn coalesced_action_nesting() {
    let inner = vec![prop_action(1, 2), prop_action(2, 3)];
    let action = UndoAction::Coalesced {
        actions: inner.clone(),
    };
    match &action {
        UndoAction::Coalesced { actions } => {
            assert_eq!(actions.len(), 2);
        }
        _ => panic!("wrong variant"),
    }
}

// ── EditPipeline integration tests ───────────────────────────────────

#[test]
fn pipeline_new_empty() {
    let pipeline = EditPipeline::new();
    assert!(!pipeline.undo_stack().can_undo());
    assert!(!pipeline.undo_stack().can_redo());
}

#[test]
fn pipeline_geometry_edit_pushes_undo() {
    let mut pipeline = EditPipeline::new();
    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let entity = world.spawn(());

    pipeline
        .apply(
            EditOp::GeometryEdit {
                entity,
                edit_data: "carve sphere".to_string(),
            },
            &mut world,
            &registry,
        )
        .unwrap();

    assert_eq!(pipeline.undo_stack().undo_count(), 1);
}

#[test]
fn pipeline_spawn_and_despawn() {
    let mut pipeline = EditPipeline::new();
    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();

    // Spawn
    pipeline
        .apply(
            EditOp::SpawnEntity {
                name: "TestEntity".to_string(),
                parent: None,
            },
            &mut world,
            &registry,
        )
        .unwrap();

    assert_eq!(pipeline.undo_stack().undo_count(), 1);
    // World should have one entity with a StableId.
    let count = world.iter().count();
    assert_eq!(count, 1);

    // Find the spawned entity.
    let entity = world.iter().next().unwrap().entity();

    // Despawn
    pipeline
        .apply(
            EditOp::DespawnEntity { entity },
            &mut world,
            &registry,
        )
        .unwrap();

    assert_eq!(pipeline.undo_stack().undo_count(), 2);
    assert_eq!(world.iter().count(), 0);
}

#[test]
fn pipeline_unknown_component_errors() {
    let mut pipeline = EditPipeline::new();
    let mut world = hecs::World::new();
    let registry = GameplayRegistry::new();
    let entity = world.spawn(());

    let result = pipeline.apply(
        EditOp::SetProperty {
            entity,
            component_name: "NonExistent".to_string(),
            field_name: "x".to_string(),
            value: GameValue::Int(1),
        },
        &mut world,
        &registry,
    );

    assert!(result.is_err());
    match result.unwrap_err() {
        EditError::UnknownComponent(name) => assert_eq!(name, "NonExistent"),
        e => panic!("wrong error: {e}"),
    }
}

#[test]
fn pipeline_default_trait() {
    let pipeline = EditPipeline::default();
    assert_eq!(pipeline.undo_stack().undo_count(), 0);
}

#[test]
fn stack_default_trait() {
    let stack = UndoStack::default();
    assert_eq!(stack.undo_count(), 0);
    assert_eq!(stack.redo_count(), 0);
}

#[test]
fn stack_coalesce_deduplicates_same_entity_field() {
    let mut world = hecs::World::new();
    let entity = world.spawn(());

    let mut stack = UndoStack::new();
    stack.begin_coalesce();

    // Simulate dragging a position slider: 3 edits to the same field.
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_x".to_string(),
        old_value: GameValue::Float(0.0),
        new_value: GameValue::Float(1.0),
    });
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_x".to_string(),
        old_value: GameValue::Float(1.0),
        new_value: GameValue::Float(2.0),
    });
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_x".to_string(),
        old_value: GameValue::Float(2.0),
        new_value: GameValue::Float(3.0),
    });

    stack.end_coalesce();

    // Should produce exactly 1 undo entry (single action, not Coalesced wrapper).
    assert_eq!(stack.undo_count(), 1);

    let action = stack.undo().unwrap();
    // With only one deduplicated edit, end_coalesce unwraps it (no Coalesced wrapper).
    match action {
        UndoAction::PropertyChange {
            old_value,
            new_value,
            ..
        } => {
            // old_value is from the FIRST edit (original state).
            assert_eq!(old_value, GameValue::Float(0.0));
            // new_value is from the LAST edit (final state).
            assert_eq!(new_value, GameValue::Float(3.0));
        }
        _ => panic!("expected PropertyChange, got {action:?}"),
    }
}

#[test]
fn stack_coalesce_dedup_preserves_different_fields() {
    let mut world = hecs::World::new();
    let entity = world.spawn(());

    let mut stack = UndoStack::new();
    stack.begin_coalesce();

    // Two different fields on the same entity — both should survive.
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_x".to_string(),
        old_value: GameValue::Float(0.0),
        new_value: GameValue::Float(1.0),
    });
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_y".to_string(),
        old_value: GameValue::Float(0.0),
        new_value: GameValue::Float(5.0),
    });
    // Duplicate of position_x — should merge with the first.
    stack.push(UndoAction::PropertyChange {
        entity,
        component_name: "Transform".to_string(),
        field_name: "position_x".to_string(),
        old_value: GameValue::Float(1.0),
        new_value: GameValue::Float(2.0),
    });

    stack.end_coalesce();

    assert_eq!(stack.undo_count(), 1);
    let action = stack.undo().unwrap();
    match action {
        UndoAction::Coalesced { actions } => {
            // Two distinct fields remain after dedup.
            assert_eq!(actions.len(), 2);
        }
        _ => panic!("expected Coalesced with 2 actions"),
    }
}

#[test]
fn stack_coalesce_dedup_non_property_actions_pass_through() {
    let mut stack = UndoStack::new();
    stack.begin_coalesce();

    stack.push(dummy_action("geo1"));
    stack.push(dummy_action("geo2"));

    stack.end_coalesce();

    assert_eq!(stack.undo_count(), 1);
    let action = stack.undo().unwrap();
    match action {
        UndoAction::Coalesced { actions } => {
            // GeometryEdit actions are not deduplicated.
            assert_eq!(actions.len(), 2);
        }
        _ => panic!("expected Coalesced"),
    }
}
