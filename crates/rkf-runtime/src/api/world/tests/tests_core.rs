//! Core world tests: spawn, despawn, query, transforms, hierarchy, components.

use std::collections::HashSet;

use glam::{Quat, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::scene_node::{SceneNode, SdfPrimitive};
use rkf_core::WorldPosition;

use uuid::Uuid;
use crate::api::error::WorldError;
use crate::api::world::World;
use crate::components::CameraComponent;

// ── World core ─────────────────────────────────────────────────────────

#[test]
fn world_new_empty() {
    let world = World::new("test");
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn spawn_returns_entity() {
    let mut world = World::new("test");
    let e = world
        .spawn("cube")
        .sdf(SdfPrimitive::Box {
            half_extents: Vec3::splat(0.5),
        })
        .material(1)
        .build();
    assert!(world.is_alive(e));
}

#[test]
fn spawn_increments_count() {
    let mut world = World::new("test");
    world
        .spawn("a")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    world
        .spawn("b")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    assert_eq!(world.entity_count(), 2);
}

#[test]
fn despawn_removes_entity() {
    let mut world = World::new("test");
    let e = world
        .spawn("cube")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    world.despawn(e).unwrap();
    assert!(!world.is_alive(e));
    assert_eq!(world.entity_count(), 0);
}

#[test]
fn despawn_invalid_entity_errors() {
    let mut world = World::new("test");
    let e = Uuid::from_u128(999);
    assert!(matches!(
        world.despawn(e),
        Err(WorldError::NoSuchEntity(_))
    ));
}

#[test]
fn double_despawn_errors() {
    let mut world = World::new("test");
    let e = world
        .spawn("cube")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    world.despawn(e).unwrap();
    assert!(matches!(
        world.despawn(e),
        Err(WorldError::NoSuchEntity(_))
    ));
}

#[test]
fn name_round_trip() {
    let mut world = World::new("test");
    let e = world
        .spawn("my_cube")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    assert_eq!(world.name(e).unwrap(), "my_cube");
}

#[test]
fn find_by_name() {
    let mut world = World::new("test");
    let e = world
        .spawn("target")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(1)
        .build();
    assert_eq!(world.find("target"), Some(e));
}

#[test]
fn find_by_name_missing() {
    let world = World::new("test");
    assert_eq!(world.find("nope"), None);
}

#[test]
fn find_all_multiple() {
    let mut world = World::new("test");
    world.spawn("dup").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    // Second "dup" gets deduplicated to "dup_2".
    world.spawn("dup").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.spawn("other").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.find_all("dup").len(), 1);
    assert_eq!(world.find_all("dup_2").len(), 1);
}

#[test]
fn entities_iterator() {
    let mut world = World::new("test");
    world.spawn("a").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.spawn("b").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.entities().count(), 2);
}

#[test]
fn clear_removes_all() {
    let mut world = World::new("test");
    world.spawn("a").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.spawn("b").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.clear();
    assert_eq!(world.entity_count(), 0);
}

// ── Transforms ─────────────────────────────────────────────────────────

#[test]
fn position_default_origin() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let pos = world.position(e).unwrap();
    assert_eq!(pos, WorldPosition::default());
}

#[test]
fn set_position_read_back() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let target = WorldPosition::new(glam::IVec3::new(1, 0, 0), Vec3::new(3.0, 2.0, 1.0));
    world.set_position(e, target).unwrap();
    assert_eq!(world.position(e).unwrap(), target);
}

#[test]
fn rotation_default_identity() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let rot = world.rotation(e).unwrap();
    assert!((rot - Quat::IDENTITY).length() < 1e-5);
}

#[test]
fn set_rotation_read_back() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let target = Quat::from_rotation_y(1.5);
    world.set_rotation(e, target).unwrap();
    let got = world.rotation(e).unwrap();
    assert!((got - target).length() < 1e-5);
}

#[test]
fn scale_default_one() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.scale(e).unwrap(), Vec3::ONE);
}

#[test]
fn set_scale_read_back() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let target = Vec3::new(2.0, 3.0, 4.0);
    world.set_scale(e, target).unwrap();
    assert_eq!(world.scale(e).unwrap(), target);
}

#[test]
fn transform_on_despawned_errors() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.despawn(e).unwrap();
    assert!(matches!(world.position(e), Err(WorldError::NoSuchEntity(_))));
}

// ── Hierarchy ──────────────────────────────────────────────────────────

#[test]
fn set_parent_establishes_relationship() {
    let mut world = World::new("test");
    let parent = world.spawn("parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let child = world.spawn("child").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).build();
    world.set_parent(child, parent).unwrap();
    assert_eq!(world.parent(child), Some(parent));
}

#[test]
fn unparent_removes_relationship() {
    let mut world = World::new("test");
    let parent = world.spawn("parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let child = world.spawn("child").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).build();
    world.set_parent(child, parent).unwrap();
    world.unparent(child).unwrap();
    assert_eq!(world.parent(child), None);
}

#[test]
fn children_lists_children() {
    let mut world = World::new("test");
    let parent = world.spawn("parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let child1 = world.spawn("c1").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).build();
    let child2 = world.spawn("c2").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).build();
    world.set_parent(child1, parent).unwrap();
    world.set_parent(child2, parent).unwrap();
    let children: Vec<Uuid> = world.children(parent).collect();
    assert_eq!(children.len(), 2);
    assert!(children.contains(&child1));
    assert!(children.contains(&child2));
}

#[test]
fn children_empty_for_leaf() {
    let mut world = World::new("test");
    let e = world.spawn("leaf").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.children(e).count(), 0);
}

#[test]
fn despawn_parent_despawns_children() {
    let mut world = World::new("test");
    let parent = world.spawn("parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let child = world.spawn("child").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).build();
    world.set_parent(child, parent).unwrap();
    world.despawn(parent).unwrap();
    assert!(!world.is_alive(child));
}

#[test]
fn cycle_detection() {
    let mut world = World::new("test");
    let a = world.spawn("a").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let b = world.spawn("b").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.set_parent(b, a).unwrap();
    assert!(matches!(world.set_parent(a, b), Err(WorldError::CycleDetected)));
}

// ── SpawnBuilder ───────────────────────────────────────────────────────

#[test]
fn spawn_with_sdf_primitive() {
    let mut world = World::new("test");
    let e = world.spawn("sphere").sdf(SdfPrimitive::Sphere { radius: 1.0 }).material(3).build();
    assert!(world.is_alive(e));
    assert_eq!(world.name(e).unwrap(), "sphere");
}

#[test]
fn spawn_with_sdf_tree() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::analytical("child", SdfPrimitive::Sphere { radius: 0.3 }, 1));
    let e = world.spawn("composite").sdf_tree(root).build();
    assert!(world.is_alive(e));
}

#[test]
fn spawn_with_position() {
    let mut world = World::new("test");
    let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(5.0, 2.0, -3.0));
    let e = world.spawn("obj").position(pos).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.position(e).unwrap(), pos);
}

#[test]
fn spawn_with_position_vec3() {
    let mut world = World::new("test");
    let e = world.spawn("obj").position_vec3(Vec3::new(1.0, 2.0, 3.0)).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let pos = world.position(e).unwrap();
    assert!((pos.to_vec3() - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
}

#[test]
fn spawn_with_rotation() {
    let mut world = World::new("test");
    let rot = Quat::from_rotation_z(1.0);
    let e = world.spawn("obj").rotation(rot).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let got = world.rotation(e).unwrap();
    assert!((got - rot).length() < 1e-5);
}

#[test]
fn spawn_with_scale() {
    let mut world = World::new("test");
    let s = Vec3::new(2.0, 3.0, 4.0);
    let e = world.spawn("obj").scale(s).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.scale(e).unwrap(), s);
}

#[test]
fn spawn_with_material() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(42).build();
    let root = world.root_node(e).unwrap();
    match &root.sdf_source {
        rkf_core::scene_node::SdfSource::Analytical { material_id, .. } => {
            assert_eq!(*material_id, 42);
        }
        _ => panic!("expected analytical source"),
    }
}

#[test]
fn spawn_with_blend_mode() {
    use rkf_core::scene_node::BlendMode;
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).blend(BlendMode::Subtract).build();
    let root = world.root_node(e).unwrap();
    assert!(matches!(root.blend_mode, BlendMode::Subtract));
}

#[test]
fn spawn_with_parent() {
    let mut world = World::new("test");
    let parent = world.spawn("parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let child = world.spawn("child").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(1).parent(parent).build();
    assert_eq!(world.parent(child), Some(parent));
}

#[test]
fn spawn_with_component() {
    #[derive(Debug, PartialEq)]
    struct Velocity(Vec3);
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).with(Velocity(Vec3::new(1.0, 0.0, 0.0))).build();
    let vel = world.get::<Velocity>(e).unwrap();
    assert_eq!(*vel, Velocity(Vec3::new(1.0, 0.0, 0.0)));
}

#[test]
fn spawn_without_sdf() {
    let mut world = World::new("test");
    let e = world.spawn("ecs_only").build();
    assert!(world.is_alive(e));
    assert_eq!(world.total_object_count(), 0);
}

// ── ECS Components ─────────────────────────────────────────────────────

#[test]
fn insert_and_get() {
    #[derive(Debug, PartialEq)]
    struct Health(i32);
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.insert(e, Health(100)).unwrap();
    let h = world.get::<Health>(e).unwrap();
    assert_eq!(*h, Health(100));
}

#[test]
fn insert_replaces_existing() {
    #[derive(Debug, PartialEq)]
    struct Health(i32);
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.insert(e, Health(100)).unwrap();
    world.insert(e, Health(50)).unwrap();
    let h = world.get::<Health>(e).unwrap();
    assert_eq!(*h, Health(50));
}

#[test]
fn get_missing_errors() {
    #[derive(Debug)]
    struct Missing;
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert!(matches!(world.get::<Missing>(e), Err(WorldError::MissingComponent(_, _))));
}

#[test]
fn remove_returns_component() {
    #[derive(Debug, PartialEq)]
    struct Tag(u32);
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.insert(e, Tag(42)).unwrap();
    let removed = world.remove::<Tag>(e).unwrap();
    assert_eq!(removed, Tag(42));
    assert!(!world.has::<Tag>(e));
}

#[test]
fn has_returns_true() {
    #[derive(Debug)]
    struct Marker;
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.insert(e, Marker).unwrap();
    assert!(world.has::<Marker>(e));
}

#[test]
fn has_returns_false() {
    #[derive(Debug)]
    struct Marker;
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert!(!world.has::<Marker>(e));
}

// ── Node tree access (B.1) ──────────────────────────────────────────

#[test]
fn find_node_returns_matching_child() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::analytical("arm", SdfPrimitive::Sphere { radius: 0.2 }, 1));
    let e = world.spawn("obj").sdf_tree(root).build();
    let node = world.find_node(e, "arm").unwrap();
    assert_eq!(node.name, "arm");
}

#[test]
fn find_node_not_found_returns_error() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert!(matches!(world.find_node(e, "nope"), Err(WorldError::NodeNotFound(_))));
}

#[test]
fn find_node_on_ecs_entity_errors() {
    let mut world = World::new("test");
    let e = world.spawn("ecs_only").build();
    assert!(matches!(world.find_node(e, "anything"), Err(WorldError::MissingComponent(_, _))));
}

#[test]
fn find_node_mut_allows_modification() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("child"));
    let e = world.spawn("obj").sdf_tree(root).build();
    world.find_node_mut(e, "child").unwrap().metadata.locked = true;
    assert!(world.find_node(e, "child").unwrap().metadata.locked);
}

#[test]
fn find_node_by_path_multi_level() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    let mut spine = SceneNode::new("spine");
    spine.add_child(SceneNode::new("chest"));
    root.add_child(spine);
    let e = world.spawn("obj").sdf_tree(root).build();
    let node = world.find_node_by_path(e, "spine/chest").unwrap();
    assert_eq!(node.name, "chest");
}

#[test]
fn root_node_immutable() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let root = world.root_node(e).unwrap();
    assert_eq!(root.name, "obj");
}

#[test]
fn node_count_matches_tree() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("a"));
    root.add_child(SceneNode::new("b"));
    let e = world.spawn("obj").sdf_tree(root).build();
    assert_eq!(world.node_count(e).unwrap(), 3);
}

// ── Node tree mutation (B.2) ─────────────────────────────────────────

#[test]
fn set_node_transform_updates_child() {
    use rkf_core::scene_node::Transform;
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("arm"));
    let e = world.spawn("obj").sdf_tree(root).build();
    let t = Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
    world.set_node_transform(e, "arm", t).unwrap();
    let node = world.find_node(e, "arm").unwrap();
    assert_eq!(node.local_transform.position, Vec3::new(1.0, 0.0, 0.0));
}

#[test]
fn set_node_transform_nonexistent_errors() {
    use rkf_core::scene_node::Transform;
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let t = Transform::default();
    assert!(matches!(world.set_node_transform(e, "nope", t), Err(WorldError::NodeNotFound(_))));
}

#[test]
fn add_child_node_appends() {
    let mut world = World::new("test");
    let root = SceneNode::new("root");
    let e = world.spawn("obj").sdf_tree(root).build();
    world.add_child_node(e, "root", SceneNode::new("new_child")).unwrap();
    assert_eq!(world.node_count(e).unwrap(), 2);
    assert!(world.find_node(e, "new_child").is_ok());
}

#[test]
fn add_child_node_to_nonexistent_parent_errors() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert!(matches!(world.add_child_node(e, "nope", SceneNode::new("c")), Err(WorldError::NodeNotFound(_))));
}

#[test]
fn remove_child_node_returns_removed() {
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("to_remove"));
    root.add_child(SceneNode::new("keep"));
    let e = world.spawn("obj").sdf_tree(root).build();
    let removed = world.remove_child_node(e, "to_remove").unwrap();
    assert_eq!(removed.name, "to_remove");
    assert_eq!(world.node_count(e).unwrap(), 2);
}

#[test]
fn remove_child_node_nonexistent_errors() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert!(matches!(world.remove_child_node(e, "nope"), Err(WorldError::NodeNotFound(_))));
}

#[test]
fn set_node_blend_mode_changes_mode() {
    use rkf_core::scene_node::BlendMode;
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("child"));
    let e = world.spawn("obj").sdf_tree(root).build();
    world.set_node_blend_mode(e, "child", BlendMode::Subtract).unwrap();
    let node = world.find_node(e, "child").unwrap();
    assert!(matches!(node.blend_mode, BlendMode::Subtract));
}

#[test]
fn set_node_sdf_source_changes_source() {
    use rkf_core::scene_node::SdfSource;
    let mut world = World::new("test");
    let mut root = SceneNode::new("root");
    root.add_child(SceneNode::new("child"));
    let e = world.spawn("obj").sdf_tree(root).build();
    let source = SdfSource::Analytical { primitive: SdfPrimitive::Sphere { radius: 1.0 }, material_id: 5 };
    world.set_node_sdf_source(e, "child", source).unwrap();
    let node = world.find_node(e, "child").unwrap();
    assert!(matches!(node.sdf_source, SdfSource::Analytical { material_id: 5, .. }));
}

// ── Camera entities (D.2) ───────────────────────────────────────────

#[test]
fn spawn_camera_creates_entity() {
    let mut world = World::new("test");
    let cam = world.spawn_camera("Main", WorldPosition::default(), 0.0, 0.0, 60.0, None);
    assert!(world.is_alive(cam));
    assert_eq!(world.name(cam).unwrap(), "Main");
}

#[test]
fn cameras_lists_camera_entities() {
    let mut world = World::new("test");
    world.spawn_camera("Cam1", WorldPosition::default(), 0.0, 0.0, 60.0, None);
    world.spawn_camera("Cam2", WorldPosition::default(), 0.0, 0.0, 90.0, None);
    world.spawn("cube").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.cameras().len(), 2);
}

#[test]
fn active_camera_finds_one() {
    let mut world = World::new("test");
    let cam = world.spawn_camera("Main", WorldPosition::default(), 0.0, 0.0, 60.0, None);
    assert!(world.active_camera().is_none());
    world.set_active_camera(cam).unwrap();
    assert_eq!(world.active_camera(), Some(cam));
}

#[test]
fn set_active_camera_deactivates_others() {
    let mut world = World::new("test");
    let cam1 = world.spawn_camera("Cam1", WorldPosition::default(), 0.0, 0.0, 60.0, None);
    let cam2 = world.spawn_camera("Cam2", WorldPosition::default(), 0.0, 0.0, 90.0, None);
    world.set_active_camera(cam1).unwrap();
    world.set_active_camera(cam2).unwrap();
    assert_eq!(world.active_camera(), Some(cam2));
    let c1 = world.get::<CameraComponent>(cam1).unwrap();
    assert!(!c1.active);
}

#[test]
fn camera_round_trips_through_position() {
    let mut world = World::new("test");
    let pos = WorldPosition::new(glam::IVec3::new(1, 0, 0), Vec3::new(3.0, 2.0, 1.0));
    let cam = world.spawn_camera("Main", pos, 45.0, -15.0, 75.0, None);
    assert_eq!(world.position(cam).unwrap(), pos);
    let c = world.get::<CameraComponent>(cam).unwrap();
    assert!((c.fov_degrees - 75.0).abs() < 1e-6);
    assert!((c.yaw - 45.0).abs() < 1e-6);
    assert!((c.pitch - -15.0).abs() < 1e-6);
}

// ── Name deduplication ──────────────────────────────────────────────────

#[test]
fn spawn_duplicate_name_gets_suffix() {
    let mut world = World::new("test");
    let a = world.spawn("Guard").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let b = world.spawn("Guard").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let c = world.spawn("Guard").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();

    assert_eq!(world.name(a).unwrap(), "Guard");
    assert_eq!(world.name(b).unwrap(), "Guard_2");
    assert_eq!(world.name(c).unwrap(), "Guard_3");
}

#[test]
fn spawn_ecs_only_duplicate_name_gets_suffix() {
    let mut world = World::new("test");
    let a = world.spawn("Empty").build();
    let b = world.spawn("Empty").build();

    assert_eq!(world.name(a).unwrap(), "Empty");
    assert_eq!(world.name(b).unwrap(), "Empty_2");
}

#[test]
fn non_siblings_can_share_names() {
    let mut world = World::new("test");
    let parent = world.spawn("Parent").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let child = world.spawn("Guard").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).parent(parent).build();
    // Root-level "Guard" should not conflict with child "Guard".
    let root_guard = world.spawn("Guard").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();

    assert_eq!(world.name(child).unwrap(), "Guard");
    assert_eq!(world.name(root_guard).unwrap(), "Guard");
}

#[test]
fn strip_numeric_suffix_on_dedup() {
    let mut world = World::new("test");
    world.spawn("Box").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    world.spawn("Box_2").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    // Spawning another "Box" should skip _2 (taken) and use _3.
    let c = world.spawn("Box").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    assert_eq!(world.name(c).unwrap(), "Box_3");

    // Spawning "Box_2" again should also produce "Box_3" — wait, _3 is taken now, so _4.
    let d = world.spawn("Box_2").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    assert_eq!(world.name(d).unwrap(), "Box_4");
}
