use super::command_queue::*;
use super::stable_id::StableId;
use super::stable_id_index::StableIdIndex;
use super::scene_ownership::SceneOwnership;
use crate::components::{EditorMetadata, Transform};

#[test]
fn spawn_and_flush() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    let mut builder = hecs::EntityBuilder::new();
    builder.add(Transform::default());
    queue.spawn(builder);

    assert!(!queue.is_empty());
    queue.flush(&mut world, &mut stable_ids);
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
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    let builder = hecs::EntityBuilder::new();
    queue.spawn_persistent(builder);
    queue.flush(&mut world, &mut stable_ids);

    let mut q = world.query::<&SceneOwnership>();
    let (_, so) = q.iter().next().unwrap();
    assert!(so.is_persistent());
}

#[test]
fn spawn_in_explicit_scene() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();

    let builder = hecs::EntityBuilder::new();
    queue.spawn_in_scene(builder, "level_02");
    queue.flush(&mut world, &mut stable_ids);

    let mut q = world.query::<&SceneOwnership>();
    let (_, so) = q.iter().next().unwrap();
    assert!(so.belongs_to("level_02"));
}

#[test]
#[should_panic(expected = "spawn() called with 2 scenes loaded")]
fn spawn_panics_with_multiple_scenes() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into(), "level_02".into()]);

    let builder = hecs::EntityBuilder::new();
    queue.spawn(builder);
    queue.flush(&mut world, &mut stable_ids);
}

#[test]
fn despawn() {
    let mut world = hecs::World::new();
    let mut stable_ids = StableIdIndex::new();
    let entity = world.spawn((Transform::default(),));

    let mut queue = CommandQueue::new();
    queue.despawn(entity);
    queue.flush(&mut world, &mut stable_ids);

    assert!(!world.contains(entity));
}

#[test]
fn despawn_cascading() {
    let mut world = hecs::World::new();
    let mut stable_ids = StableIdIndex::new();
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
    queue.flush(&mut world, &mut stable_ids);

    assert!(!world.contains(parent));
    assert!(!world.contains(child));
    assert!(!world.contains(grandchild));
}

#[test]
fn insert_on_existing_entity() {
    let mut world = hecs::World::new();
    let mut stable_ids = StableIdIndex::new();
    let entity = world.spawn((Transform::default(),));

    let mut queue = CommandQueue::new();
    queue.insert(entity, EditorMetadata {
        name: "Test".into(),
        tags: vec![],
        locked: false,
    });
    queue.flush(&mut world, &mut stable_ids);

    let meta = world.get::<&EditorMetadata>(entity).unwrap();
    assert_eq!(meta.name, "Test");
}

#[test]
fn remove_component() {
    let mut world = hecs::World::new();
    let mut stable_ids = StableIdIndex::new();
    let entity = world.spawn((
        Transform::default(),
        EditorMetadata::default(),
    ));

    let mut queue = CommandQueue::new();
    queue.remove::<EditorMetadata>(entity);
    queue.flush(&mut world, &mut stable_ids);

    assert!(world.get::<&EditorMetadata>(entity).is_err());
    assert!(world.get::<&Transform>(entity).is_ok());
}

#[test]
fn empty_flush_is_noop() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    assert!(queue.is_empty());
    queue.flush(&mut world, &mut stable_ids);
    assert_eq!(world.query::<()>().iter().count(), 0);
}

// ─── Auto-inject tests ──────────────────────────────────────────

#[test]
fn spawn_auto_injects_transform_stable_id_editor_metadata() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["test".into()]);

    // Spawn with empty builder — no components provided
    let builder = hecs::EntityBuilder::new();
    queue.spawn(builder);
    queue.flush(&mut world, &mut stable_ids);

    // Should have exactly one entity
    assert_eq!(world.query::<()>().iter().count(), 1);

    let mut q = world.query::<(&Transform, &StableId, &EditorMetadata)>();
    let (entity, (_, stable_id, meta)) = q.iter().next()
        .expect("entity should have all three auto-injected components");

    // EditorMetadata should be default
    assert_eq!(meta.name, "Entity");

    // StableId should be registered in the index
    assert!(stable_ids.contains_entity(entity));
    assert_eq!(stable_ids.get_entity(stable_id.uuid()), Some(entity));
}

#[test]
fn spawn_does_not_overwrite_caller_provided_components() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["test".into()]);

    let custom_id = StableId::new();
    let custom_transform = Transform {
        position: rkf_core::WorldPosition::new(glam::IVec3::new(1, 2, 3), glam::Vec3::ZERO),
        rotation: glam::Quat::IDENTITY,
        scale: glam::Vec3::ONE,
    };
    let custom_meta = EditorMetadata {
        name: "CustomName".into(),
        tags: vec!["special".into()],
        locked: true,
    };

    let mut builder = hecs::EntityBuilder::new();
    builder.add(custom_id);
    builder.add(custom_transform);
    builder.add(custom_meta);
    queue.spawn(builder);
    queue.flush(&mut world, &mut stable_ids);

    let mut q = world.query::<(&Transform, &StableId, &EditorMetadata)>();
    let (entity, (transform, stable_id, meta)) = q.iter().next().unwrap();

    // Caller-provided values should be preserved
    assert_eq!(stable_id.uuid(), custom_id.uuid());
    assert_eq!(transform.position.chunk, glam::IVec3::new(1, 2, 3));
    assert_eq!(meta.name, "CustomName");
    assert!(meta.locked);

    // Should still be registered in the index
    assert!(stable_ids.contains_entity(entity));
    assert_eq!(stable_ids.get_entity(custom_id.uuid()), Some(entity));
}

#[test]
fn spawned_entity_registered_in_stable_id_index() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["test".into()]);

    let builder = hecs::EntityBuilder::new();
    queue.spawn(builder);
    queue.flush(&mut world, &mut stable_ids);

    assert_eq!(stable_ids.len(), 1);

    let mut q = world.query::<&StableId>();
    let (entity, stable_id) = q.iter().next().unwrap();
    assert_eq!(stable_ids.get_entity(stable_id.uuid()), Some(entity));
    assert_eq!(stable_ids.get_stable(entity), Some(stable_id.uuid()));
}

#[test]
fn despawned_entity_removed_from_stable_id_index() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["test".into()]);

    // Spawn an entity
    let builder = hecs::EntityBuilder::new();
    queue.spawn(builder);
    queue.flush(&mut world, &mut stable_ids);

    assert_eq!(stable_ids.len(), 1);

    // Get the entity and its StableId
    let mut q = world.query::<&StableId>();
    let (entity, stable_id) = q.iter().next().unwrap();
    let uuid = stable_id.uuid();
    drop(q);

    // Despawn it
    let mut queue2 = CommandQueue::new();
    queue2.despawn(entity);
    queue2.flush(&mut world, &mut stable_ids);

    assert!(!world.contains(entity));
    assert_eq!(stable_ids.len(), 0);
    assert!(stable_ids.get_entity(uuid).is_none());
}

// ─── Blueprint spawn tests ───────────────────────────────────────

#[test]
fn spawn_blueprint_current_scene() {
    use crate::behavior::blueprint::{Blueprint, BlueprintCatalog};
    use crate::behavior::engine_components::engine_register;
    use crate::behavior::registry::GameplayRegistry;

    let mut registry = GameplayRegistry::new();
    engine_register(&mut registry);

    let mut catalog = BlueprintCatalog::new();
    let mut bp = Blueprint::new("Guard");
    bp.components.insert(
        "EditorMetadata".to_owned(),
        "(name: \"Guard\", tags: [\"enemy\"], locked: false)".to_owned(),
    );
    catalog.insert(bp);

    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    let pos = glam::Vec3::new(1.0, 2.0, 3.0);
    let temp = queue.spawn_blueprint("Guard", pos);
    // Can still add extra components via insert_temp
    assert!(!queue.is_empty());
    let _ = temp; // used to verify TempEntity is returned

    queue.flush_with_catalog(&mut world, &mut stable_ids, &catalog, &registry);
    assert!(queue.is_empty());

    // Should have one entity with EditorMetadata + Transform + SceneOwnership
    let count = world.query::<&EditorMetadata>().iter().count();
    assert_eq!(count, 1);

    let mut q = world.query::<(&EditorMetadata, &Transform, &SceneOwnership)>();
    let (_, (meta, transform, so)) = q.iter().next().unwrap();
    assert_eq!(meta.name, "Guard");
    assert!(so.belongs_to("level_01"));
    // Position should be set
    let expected_wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
    assert_eq!(transform.position.chunk, expected_wp.chunk);
}

#[test]
fn spawn_blueprint_persistent() {
    use crate::behavior::blueprint::{Blueprint, BlueprintCatalog};
    use crate::behavior::engine_components::engine_register;
    use crate::behavior::registry::GameplayRegistry;

    let mut registry = GameplayRegistry::new();
    engine_register(&mut registry);

    let mut catalog = BlueprintCatalog::new();
    catalog.insert(Blueprint::new("Empty"));

    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();

    queue.spawn_blueprint_persistent("Empty", glam::Vec3::ZERO);
    queue.flush_with_catalog(&mut world, &mut stable_ids, &catalog, &registry);

    let mut q = world.query::<&SceneOwnership>();
    let (_, so) = q.iter().next().unwrap();
    assert!(so.is_persistent());
}

#[test]
fn spawn_blueprint_in_scene() {
    use crate::behavior::blueprint::{Blueprint, BlueprintCatalog};
    use crate::behavior::engine_components::engine_register;
    use crate::behavior::registry::GameplayRegistry;

    let mut registry = GameplayRegistry::new();
    engine_register(&mut registry);

    let mut catalog = BlueprintCatalog::new();
    catalog.insert(Blueprint::new("Empty"));

    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();

    queue.spawn_blueprint_in_scene("Empty", glam::Vec3::ZERO, "level_02");
    queue.flush_with_catalog(&mut world, &mut stable_ids, &catalog, &registry);

    let mut q = world.query::<&SceneOwnership>();
    let (_, so) = q.iter().next().unwrap();
    assert!(so.belongs_to("level_02"));
}

#[test]
fn spawn_blueprint_missing_skipped() {
    use crate::behavior::blueprint::BlueprintCatalog;
    use crate::behavior::registry::GameplayRegistry;

    let registry = GameplayRegistry::new();
    let catalog = BlueprintCatalog::new(); // empty

    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    queue.spawn_blueprint("NonExistent", glam::Vec3::ZERO);
    queue.flush_with_catalog(&mut world, &mut stable_ids, &catalog, &registry);

    // Entity should not have been spawned
    assert_eq!(world.query::<()>().iter().count(), 0);
}

#[test]
fn spawn_blueprint_without_catalog_skipped() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    queue.spawn_blueprint("Guard", glam::Vec3::ZERO);
    // Using plain flush() without catalog — blueprint spawn should be skipped
    queue.flush(&mut world, &mut stable_ids);

    assert_eq!(world.query::<()>().iter().count(), 0);
}

// ─── Name uniqueness tests (spec 4.3) ─────────────────────────────

#[test]
fn duplicate_sibling_name_gets_suffix() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    // Spawn first entity named "Guard"
    let mut b1 = hecs::EntityBuilder::new();
    b1.add(EditorMetadata {
        name: "Guard".into(),
        tags: vec![],
        locked: false,
    });
    queue.spawn(b1);
    queue.flush(&mut world, &mut stable_ids);

    // Spawn second entity with same name "Guard"
    let mut b2 = hecs::EntityBuilder::new();
    b2.add(EditorMetadata {
        name: "Guard".into(),
        tags: vec![],
        locked: false,
    });
    queue.spawn(b2);
    queue.flush(&mut world, &mut stable_ids);

    let mut names: Vec<String> = world
        .query::<&EditorMetadata>()
        .iter()
        .map(|(_, m)| m.name.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["Guard", "Guard_2"]);
}

#[test]
fn non_siblings_can_share_names() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    // Spawn parent with name "Guard"
    let mut b1 = hecs::EntityBuilder::new();
    b1.add(EditorMetadata {
        name: "Guard".into(),
        tags: vec![],
        locked: false,
    });
    queue.spawn(b1);
    queue.flush(&mut world, &mut stable_ids);

    // Get the parent entity
    let parent_entity = world
        .query::<&EditorMetadata>()
        .iter()
        .next()
        .unwrap()
        .0;

    // Spawn child under the parent with same name "Guard"
    let mut b2 = hecs::EntityBuilder::new();
    b2.add(EditorMetadata {
        name: "Guard".into(),
        tags: vec![],
        locked: false,
    });
    b2.add(crate::components::Parent {
        entity: parent_entity,
        bone_index: None,
    });
    queue.spawn(b2);
    queue.flush(&mut world, &mut stable_ids);

    // Both should keep "Guard" since they are not siblings
    let mut names: Vec<String> = world
        .query::<&EditorMetadata>()
        .iter()
        .map(|(_, m)| m.name.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["Guard", "Guard"]);
}

#[test]
fn sequential_spawns_increment_suffix() {
    let mut world = hecs::World::new();
    let mut queue = CommandQueue::new();
    let mut stable_ids = StableIdIndex::new();
    queue.set_loaded_scenes(vec!["level_01".into()]);

    // Spawn three entities all named "Guard" at root level
    for _ in 0..3 {
        let mut b = hecs::EntityBuilder::new();
        b.add(EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        });
        queue.spawn(b);
        queue.flush(&mut world, &mut stable_ids);
    }

    let mut names: Vec<String> = world
        .query::<&EditorMetadata>()
        .iter()
        .map(|(_, m)| m.name.clone())
        .collect();
    names.sort();
    assert_eq!(names, vec!["Guard", "Guard_2", "Guard_3"]);
}
