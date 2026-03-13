//! Tests for the v3 scene file format.

use super::*;
use std::collections::HashMap;
use uuid::Uuid;

#[test]
fn empty_scene_roundtrip() {
    let scene = SceneFileV3::new();
    let ron = serialize_scene_v3(&scene).unwrap();
    let scene2 = deserialize_scene_v3(&ron).unwrap();
    assert_eq!(scene2.version, 3);
    assert!(scene2.entities.is_empty());
}

#[test]
fn entity_record_component_insert_and_get() {
    let mut record = EntityRecord::new(Uuid::nil());
    let transform = crate::components::Transform::default();
    record.insert_component("Transform", &transform).unwrap();

    assert!(record.has_component("Transform"));
    assert!(!record.has_component("Light"));

    let t2: crate::components::Transform = record.get_component("Transform").unwrap().unwrap();
    assert_eq!(t2.position, transform.position);
    assert_eq!(t2.scale, transform.scale);
}

#[test]
fn scene_with_entities_roundtrip() {
    let mut scene = SceneFileV3::new();

    let id1 = Uuid::from_u128(1);
    let id2 = Uuid::from_u128(2);

    let mut e1 = EntityRecord::new(id1);
    e1.insert_component(
        component_names::TRANSFORM,
        &crate::components::Transform::default(),
    ).unwrap();
    e1.insert_component(
        component_names::EDITOR_METADATA,
        &crate::components::EditorMetadata {
            name: "Guard".to_string(),
            tags: vec!["npc".to_string()],
            locked: false,
        },
    ).unwrap();

    let mut e2 = EntityRecord::new(id2);
    e2.parent = Some(id1);
    e2.insert_component(
        component_names::CAMERA,
        &crate::components::CameraComponent {
            label: "MainCam".to_string(),
            fov_degrees: 75.0,
            active: true,
            ..Default::default()
        },
    ).unwrap();

    scene.entities.push(e1);
    scene.entities.push(e2);

    let ron = serialize_scene_v3(&scene).unwrap();
    let scene2 = deserialize_scene_v3(&ron).unwrap();

    assert_eq!(scene2.entities.len(), 2);
    assert_eq!(scene2.entities[0].stable_id, id1);
    assert_eq!(scene2.entities[1].stable_id, id2);
    assert_eq!(scene2.entities[1].parent, Some(id1));

    // Verify component data survived
    let meta: crate::components::EditorMetadata = scene2.entities[0]
        .get_component(component_names::EDITOR_METADATA)
        .unwrap()
        .unwrap();
    assert_eq!(meta.name, "Guard");
    assert_eq!(meta.tags, vec!["npc"]);

    let cam: crate::components::CameraComponent = scene2.entities[1]
        .get_component(component_names::CAMERA)
        .unwrap()
        .unwrap();
    assert_eq!(cam.label, "MainCam");
    assert!((cam.fov_degrees - 75.0).abs() < 1e-6);
    assert!(cam.active);
}

#[test]
fn sdf_tree_component_roundtrip() {
    let mut record = EntityRecord::new(Uuid::from_u128(42));
    let tree = crate::components::SdfTree {
        root: rkf_core::scene_node::SceneNode::new("complex"),
        asset_path: Some("models/hero.rkf".to_string()),
        aabb: rkf_core::aabb::Aabb::new(
            glam::Vec3::new(-2.0, -1.0, -3.0),
            glam::Vec3::new(2.0, 1.0, 3.0),
        ),
    };
    record.insert_component(component_names::SDF_TREE, &tree).unwrap();

    let ron_str = &record.components[component_names::SDF_TREE];
    let tree2: crate::components::SdfTree = ron::from_str(ron_str).unwrap();
    assert_eq!(tree2.asset_path, Some("models/hero.rkf".to_string()));
    assert_eq!(tree2.aabb.min, glam::Vec3::new(-2.0, -1.0, -3.0));
    // root is reconstructed as default (SceneNode::new("root"))
    assert_eq!(tree2.root.name, "root");
}

#[test]
fn fog_volume_component_roundtrip() {
    let mut record = EntityRecord::new(Uuid::from_u128(99));
    let fog = crate::components::FogVolumeComponent {
        density: 0.7,
        color: [0.5, 0.6, 0.7],
        phase_g: 0.4,
        half_extents: glam::Vec3::new(10.0, 20.0, 30.0),
    };
    record.insert_component(component_names::FOG_VOLUME, &fog).unwrap();

    let fog2: crate::components::FogVolumeComponent = record
        .get_component(component_names::FOG_VOLUME)
        .unwrap()
        .unwrap();
    assert!((fog2.density - 0.7).abs() < 1e-6);
    assert_eq!(fog2.color, [0.5, 0.6, 0.7]);
}

#[test]
fn unknown_components_preserved() {
    let mut scene = SceneFileV3::new();
    let mut record = EntityRecord::new(Uuid::from_u128(1));
    // Simulate an unknown component from a gameplay dylib
    record.components.insert(
        "Health".to_string(),
        "(current:80,max:100)".to_string(),
    );
    record.components.insert(
        "Transform".to_string(),
        ron::to_string(&crate::components::Transform::default()).unwrap(),
    );
    scene.entities.push(record);

    let ron = serialize_scene_v3(&scene).unwrap();
    let scene2 = deserialize_scene_v3(&ron).unwrap();

    // Both known and unknown components survive
    assert!(scene2.entities[0].has_component("Transform"));
    assert!(scene2.entities[0].has_component("Health"));
    assert_eq!(
        scene2.entities[0].components["Health"],
        "(current:80,max:100)"
    );
}

#[test]
fn version_is_always_3() {
    let scene = SceneFileV3::new();
    assert_eq!(scene.version, 3);

    let ron = serialize_scene_v3(&scene).unwrap();
    assert!(ron.contains("version: 3"));
}

// -- save/load integration tests ------------------------------------------------

#[test]
fn save_load_roundtrip() {
    use crate::behavior::{GameplayRegistry, StableId, StableIdIndex};
    use crate::components::*;

    let mut ecs = hecs::World::new();
    let mut stable_index = StableIdIndex::new();
    let registry = GameplayRegistry::new();

    let id1 = Uuid::from_u128(100);
    let id2 = Uuid::from_u128(200);

    let e1 = ecs.spawn((
        StableId(id1),
        Transform {
            position: rkf_core::WorldPosition::new(
                glam::IVec3::new(1, 0, 0),
                glam::Vec3::new(5.0, 10.0, 15.0),
            ),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },
        EditorMetadata {
            name: "Hero".to_string(),
            tags: vec!["player".to_string()],
            locked: false,
        },
    ));
    stable_index.insert(id1, e1);

    let e2 = ecs.spawn((
        StableId(id2),
        Transform::default(),
        CameraComponent {
            label: "MainCam".to_string(),
            fov_degrees: 90.0,
            active: true,
            ..Default::default()
        },
        Parent {
            entity: e1,
            bone_index: None,
        },
    ));
    stable_index.insert(id2, e2);

    // Save
    let scene = save_scene(&ecs, &stable_index, &registry);
    assert_eq!(scene.entities.len(), 2);

    // Serialize to RON and back
    let ron = serialize_scene_v3(&scene).unwrap();
    let scene2 = deserialize_scene_v3(&ron).unwrap();

    // Load into fresh world
    let mut ecs2 = hecs::World::new();
    let mut stable_index2 = StableIdIndex::new();
    load_scene(&scene2, &mut ecs2, &mut stable_index2, &registry);

    // Verify entities exist
    assert_eq!(stable_index2.len(), 2);
    let loaded_e1 = stable_index2.get_entity(id1).unwrap();
    let loaded_e2 = stable_index2.get_entity(id2).unwrap();

    // Verify Transform (WorldPosition::new normalizes local into [0,8))
    let t = ecs2.get::<&Transform>(loaded_e1).unwrap();
    assert!((t.position.local.x - 5.0).abs() < 1e-6);
    assert!((t.position.local.y - 2.0).abs() < 1e-6); // 10.0 mod 8.0 = 2.0
    assert_eq!(t.position.chunk, glam::IVec3::new(1, 1, 1)); // (1,0,0) + (0,1,1) from normalization

    // Verify EditorMetadata
    let m = ecs2.get::<&EditorMetadata>(loaded_e1).unwrap();
    assert_eq!(m.name, "Hero");
    assert_eq!(m.tags, vec!["player"]);

    // Verify CameraComponent
    let c = ecs2.get::<&CameraComponent>(loaded_e2).unwrap();
    assert_eq!(c.label, "MainCam");
    assert!((c.fov_degrees - 90.0).abs() < 1e-6);
    assert!(c.active);

    // Verify Parent resolved correctly
    let p = ecs2.get::<&Parent>(loaded_e2).unwrap();
    assert_eq!(p.entity, loaded_e1);
}

#[test]
fn save_skips_entities_without_stable_id() {
    use crate::behavior::{GameplayRegistry, StableId, StableIdIndex};
    use crate::components::*;

    let mut ecs = hecs::World::new();
    let _stable_index = StableIdIndex::new();
    let registry = GameplayRegistry::new();

    // Entity without StableId -- should be skipped
    ecs.spawn((Transform::default(),));

    // Entity with StableId
    let id = Uuid::from_u128(42);
    let e = ecs.spawn((
        StableId(id),
        Transform::default(),
    ));
    let mut si = StableIdIndex::new();
    si.insert(id, e);

    let scene = save_scene(&ecs, &si, &registry);
    assert_eq!(scene.entities.len(), 1);
    assert_eq!(scene.entities[0].stable_id, id);
}

#[test]
fn unknown_components_survive_save_load() {
    use crate::behavior::{GameplayRegistry, StableId, StableIdIndex};

    let mut ecs = hecs::World::new();
    let mut stable_index = StableIdIndex::new();
    let registry = GameplayRegistry::new();

    let id = Uuid::from_u128(77);
    let e = ecs.spawn((
        StableId(id),
        UnknownComponents {
            data: {
                let mut m = HashMap::new();
                m.insert("Health".to_string(), "(current:80,max:100)".to_string());
                m
            },
        },
    ));
    stable_index.insert(id, e);

    let scene = save_scene(&ecs, &stable_index, &registry);
    assert!(scene.entities[0].has_component("Health"));

    // Load into fresh world
    let mut ecs2 = hecs::World::new();
    let mut si2 = StableIdIndex::new();
    load_scene(&scene, &mut ecs2, &mut si2, &registry);

    let loaded_e = si2.get_entity(id).unwrap();
    let unknown = ecs2.get::<&UnknownComponents>(loaded_e).unwrap();
    assert_eq!(unknown.data["Health"], "(current:80,max:100)");
}

#[test]
fn sdf_tree_asset_path_roundtrip_through_save_load() {
    use crate::behavior::{GameplayRegistry, StableId, StableIdIndex};
    use crate::components::*;

    let mut ecs = hecs::World::new();
    let mut stable_index = StableIdIndex::new();
    let registry = GameplayRegistry::new();

    let id = Uuid::from_u128(55);
    let e = ecs.spawn((
        StableId(id),
        SdfTree {
            root: rkf_core::scene_node::SceneNode::new("hero"),
            asset_path: Some("models/hero.rkf".to_string()),
            aabb: rkf_core::aabb::Aabb::new(
                glam::Vec3::splat(-2.0),
                glam::Vec3::splat(2.0),
            ),
        },
    ));
    stable_index.insert(id, e);

    let scene = save_scene(&ecs, &stable_index, &registry);
    let ron = serialize_scene_v3(&scene).unwrap();
    let scene2 = deserialize_scene_v3(&ron).unwrap();

    let mut ecs2 = hecs::World::new();
    let mut si2 = StableIdIndex::new();
    load_scene(&scene2, &mut ecs2, &mut si2, &registry);

    let loaded_e = si2.get_entity(id).unwrap();
    let sdf = ecs2.get::<&SdfTree>(loaded_e).unwrap();
    assert_eq!(sdf.asset_path, Some("models/hero.rkf".to_string()));
    assert_eq!(sdf.aabb.min, glam::Vec3::splat(-2.0));
    // root is default (streaming resolves later)
    assert_eq!(sdf.root.name, "root");
}

