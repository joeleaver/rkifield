//! Scene management, scene I/O, build_render_scene, and hecs sync tests.

use std::collections::HashSet;

use glam::{Quat, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::scene::SceneObject;
use rkf_core::scene_node::{SceneNode, SdfPrimitive};
use rkf_core::WorldPosition;

use uuid::Uuid;
use crate::api::error::WorldError;
use crate::api::world::World;

// ── Per-scene load/save (C.4) ────────────────────────────────────────

#[test]
fn load_scene_into_new_creates_scene() {
    let dir = std::env::temp_dir().join("rkf_api_test_load_into.rkscene");
    let path = dir.to_str().unwrap();

    let mut source = World::new("src");
    source.spawn("sphere").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    source.save_scene(path).unwrap();

    let mut world = World::new("main");
    let (idx, entities) = world.load_scene_into(path, None).unwrap();
    assert_eq!(idx, 1);
    assert_eq!(entities.len(), 1);
    assert_eq!(world.scene_count(), 2);

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_scene_into_existing_appends() {
    let dir = std::env::temp_dir().join("rkf_api_test_load_existing.rkscene");
    let path = dir.to_str().unwrap();

    let mut source = World::new("src");
    source.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    source.save_scene(path).unwrap();

    let mut world = World::new("main");
    world.spawn("existing").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let (idx, entities) = world.load_scene_into(path, Some(0)).unwrap();
    assert_eq!(idx, 0);
    assert_eq!(entities.len(), 1);
    assert_eq!(world.total_object_count(), 2);

    let _ = std::fs::remove_file(path);
}

#[test]
fn save_scene_at_writes_file() {
    let dir = std::env::temp_dir().join("rkf_api_test_save_at.rkscene");
    let path = dir.to_str().unwrap();

    let mut world = World::new("main");
    world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.save_scene_at(0, path).unwrap();
    assert!(std::path::Path::new(path).exists());

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_then_save_roundtrip() {
    let dir1 = std::env::temp_dir().join("rkf_api_test_rt1.rkscene");
    let dir2 = std::env::temp_dir().join("rkf_api_test_rt2.rkscene");
    let p1 = dir1.to_str().unwrap();
    let p2 = dir2.to_str().unwrap();

    let mut world = World::new("test");
    world.spawn("ball").position_vec3(Vec3::new(1.0, 2.0, 3.0)).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(2).build();
    world.save_scene(p1).unwrap();

    let mut world2 = World::new("empty");
    let (idx, _) = world2.load_scene_into(p1, None).unwrap();
    world2.save_scene_at(idx, p2).unwrap();

    let mut world3 = World::new("verify");
    let loaded = world3.load_scene(p2).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(world3.name(loaded[0]).unwrap(), "ball");

    let _ = std::fs::remove_file(p1);
    let _ = std::fs::remove_file(p2);
}

#[test]
fn save_scene_at_out_of_range_errors() {
    let mut world = World::new("test");
    let result = world.save_scene_at(99, "/tmp/nope.rkscene");
    assert!(matches!(result, Err(WorldError::SceneOutOfRange(99))));
}

// ── Scene I/O ──────────────────────────────────────────────────────────

#[test]
fn save_load_round_trip() {
    let dir = std::env::temp_dir().join("rkf_api_test_scene.rkscene");
    let path = dir.to_str().unwrap();

    let mut world = World::new("test");
    world.spawn("sphere").position_vec3(Vec3::new(1.0, 2.0, 3.0)).sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(2).build();
    world.save_scene(path).unwrap();

    let mut world2 = World::new("loaded");
    let loaded = world2.load_scene(path).unwrap();
    assert_eq!(loaded.len(), 1);
    let e = loaded[0];
    assert_eq!(world2.name(e).unwrap(), "sphere");

    let _ = std::fs::remove_file(path);
}

#[test]
fn load_nonexistent_errors() {
    let mut world = World::new("test");
    let result = world.load_scene("/nonexistent/path.rkscene");
    assert!(result.is_err());
}

// ── Multi-scene (C.1) ──────────────────────────────────────────────

#[test]
fn world_starts_with_one_scene() {
    let world = World::new("test");
    assert_eq!(world.scene_count(), 1);
    assert_eq!(world.scene_name(0), Some("test"));
}

#[test]
fn create_scene_increments_count() {
    let mut world = World::new("test");
    world.create_scene("scene2");
    world.create_scene("scene3");
    assert_eq!(world.scene_count(), 3);
}

#[test]
fn active_scene_defaults_to_zero() {
    let world = World::new("test");
    assert_eq!(world.active_scene_index(), 0);
}

#[test]
fn set_active_scene_changes_target() {
    let mut world = World::new("test");
    world.create_scene("scene2");
    world.set_active_scene(1);
    assert_eq!(world.active_scene_index(), 1);
}

#[test]
fn scene_name_returns_correct() {
    let mut world = World::new("main");
    world.create_scene("overlay");
    assert_eq!(world.scene_name(0), Some("main"));
    assert_eq!(world.scene_name(1), Some("overlay"));
    assert_eq!(world.scene_name(99), None);
}

#[test]
fn spawn_targets_active_scene() {
    let mut world = World::new("scene0");
    world.spawn("obj_s0").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    // Count entities in scene 0
    let scene0_count = world.entity_count();
    assert_eq!(scene0_count, 1);

    world.create_scene("scene1");
    world.set_active_scene(1);
    world.spawn("obj_s1").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.entity_count(), 2); // Both scenes' entities are tracked
}

#[test]
fn existing_apis_still_work_after_refactor() {
    let mut world = World::new("test");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
    world.set_position(e, pos).unwrap();
    assert_eq!(world.position(e).unwrap(), pos);
    let rot = Quat::from_rotation_y(1.0);
    world.set_rotation(e, rot).unwrap();
    assert!((world.rotation(e).unwrap() - rot).length() < 1e-5);
}

// ── Combined scene view (C.2) ──────────────────────────────────────

#[test]
fn total_object_count_sums_scenes() {
    let mut world = World::new("s0");
    world.spawn("a").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.create_scene("s1");
    world.set_active_scene(1);
    world.spawn("b").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.spawn("c").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    assert_eq!(world.total_object_count(), 3);
}

#[test]
fn renderer_sees_all_scenes() {
    let mut world = World::new("s0");
    world.spawn("obj0").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.create_scene("s1");
    world.set_active_scene(1);
    world.spawn("obj1").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    let scene = world.build_render_scene();
    assert_eq!(scene.objects.len(), 2);
    assert_eq!(world.total_object_count(), 2);
}

// ── Persistent scenes and swap (C.3) ─────────────────────────────────

#[test]
fn set_persistent_flag() {
    let mut world = World::new("test");
    assert!(!world.is_scene_persistent(0));
    world.set_scene_persistent(0, true);
    assert!(world.is_scene_persistent(0));
}

#[test]
fn swap_removes_non_persistent() {
    let mut world = World::new("gameplay");
    world.create_scene("ui");
    world.set_scene_persistent(1, true);
    let removed = world.swap_scenes();
    assert_eq!(removed, vec!["gameplay"]);
    assert_eq!(world.scene_count(), 1);
    assert_eq!(world.scene_name(0), Some("ui"));
}

#[test]
fn swap_despawns_entities() {
    let mut world = World::new("temp");
    let e = world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.create_scene("persistent");
    world.set_scene_persistent(1, true);
    world.swap_scenes();
    assert!(!world.is_alive(e));
}

#[test]
fn swap_preserves_persistent_entities() {
    let mut world = World::new("temp");
    world.spawn("temp_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.create_scene("persistent");
    world.set_scene_persistent(1, true);
    world.set_active_scene(1);
    let keeper = world.spawn("keeper").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.swap_scenes();
    assert!(world.is_alive(keeper));
}

#[test]
fn remove_scene_by_index() {
    let mut world = World::new("s0");
    world.create_scene("s1");
    let name = world.remove_scene(0, false).unwrap();
    assert_eq!(name, "s0");
    assert_eq!(world.scene_count(), 1);
}

#[test]
fn remove_persistent_needs_force() {
    let mut world = World::new("s0");
    world.create_scene("s1");
    world.set_scene_persistent(0, true);
    assert!(world.remove_scene(0, false).is_err());
    assert!(world.remove_scene(0, true).is_ok());
}

#[test]
fn cannot_remove_last_scene() {
    let mut world = World::new("only");
    assert!(matches!(world.remove_scene(0, false), Err(WorldError::CannotRemoveLastScene)));
}

#[test]
fn active_index_adjusts_after_removal() {
    let mut world = World::new("s0");
    world.create_scene("s1");
    world.create_scene("s2");
    world.set_active_scene(2);
    world.remove_scene(0, false).unwrap();
    assert!(world.active_scene_index() < world.scene_count());
}

#[test]
fn entities_span_scenes() {
    let mut world = World::new("scene0");
    let e0 = world.spawn("obj_s0").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(1).build();
    world.create_scene("scene1");
    world.set_active_scene(1);
    let e1 = world.spawn("obj_s1").sdf(SdfPrimitive::Sphere { radius: 0.3 }).material(2).build();
    assert!(world.is_alive(e0));
    assert!(world.is_alive(e1));
    assert_eq!(world.name(e0).unwrap(), "obj_s0");
    assert_eq!(world.name(e1).unwrap(), "obj_s1");
    let pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(5.0, 0.0, 0.0));
    world.set_position(e0, pos).unwrap();
    assert_eq!(world.position(e0).unwrap(), pos);
}

#[test]
fn save_empty_world() {
    let dir = std::env::temp_dir().join("rkf_api_test_empty.rkscene");
    let path = dir.to_str().unwrap();

    let mut world = World::new("empty");
    world.save_scene(path).unwrap();

    let mut world2 = World::new("loaded");
    let loaded = world2.load_scene(path).unwrap();
    assert_eq!(loaded.len(), 0);

    let _ = std::fs::remove_file(path);
}

// ── rebuild_entity_tracking_from_ecs ─────────────────────────────────

#[test]
fn rebuild_tracking_registers_entities() {
    let mut world = World::new("test");
    world.spawn("a").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    world.spawn("b").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    assert_eq!(world.entity_count(), 2);

    // Rebuild tracking from scratch
    world.rebuild_entity_tracking_from_ecs();
    assert_eq!(world.entity_count(), 2);
    assert!(world.find("a").is_some());
    assert!(world.find("b").is_some());
}

#[test]
fn rebuild_tracking_is_idempotent() {
    let mut world = World::new("test");
    world.spawn("obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();

    world.rebuild_entity_tracking_from_ecs();
    assert_eq!(world.entity_count(), 1);
    let entity_first = world.find("obj").unwrap();

    world.rebuild_entity_tracking_from_ecs();
    assert_eq!(world.entity_count(), 1);
    // Entity handle may differ (new generation) but the same object is found
    assert!(world.find("obj").is_some());
}

// ── build_render_scene ──────────────────────────────────────────────

#[test]
fn build_render_scene_includes_spawned_objects() {
    let mut world = World::new("test");
    world.spawn("sphere").sdf(SdfPrimitive::Sphere { radius: 1.0 }).material(0)
        .position(WorldPosition::new(glam::IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0))).build();
    world.spawn("box").sdf(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) }).material(1).build();

    let scene = world.build_render_scene();
    assert_eq!(scene.objects.len(), 2);

    let sphere = scene.objects.iter().find(|o| o.name == "sphere").unwrap();
    assert!((sphere.position.x - 1.0).abs() < 1e-6);
    assert!((sphere.position.y - 2.0).abs() < 1e-6);
    assert!((sphere.position.z - 3.0).abs() < 1e-6);

    let box_obj = scene.objects.iter().find(|o| o.name == "box").unwrap();
    assert!((box_obj.position.x).abs() < 1e-6);
}

// ── hecs sync tests ──────────────────────────────────────────────────

#[test]
fn position_reads_from_hecs() {
    let mut world = World::new("test");
    let pos = WorldPosition::new(glam::IVec3::new(1, 0, 0), Vec3::new(5.0, 3.0, 2.0));
    let e = world.spawn("test_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).position(pos).build();
    let read_pos = world.position(e).unwrap();
    assert_eq!(read_pos, pos);
}

#[test]
fn set_position_syncs_to_hecs() {
    let mut world = World::new("test");
    let e = world.spawn("test_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let new_pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(3.0, 4.0, 5.0));
    world.set_position(e, new_pos).unwrap();
    assert_eq!(world.position(e).unwrap(), new_pos);
}

#[test]
fn set_rotation_syncs_to_hecs() {
    let mut world = World::new("test");
    let e = world.spawn("test_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
    world.set_rotation(e, rot).unwrap();
    let read_rot = world.rotation(e).unwrap();
    assert!((read_rot.x - rot.x).abs() < 1e-6);
    assert!((read_rot.y - rot.y).abs() < 1e-6);
    assert!((read_rot.z - rot.z).abs() < 1e-6);
    assert!((read_rot.w - rot.w).abs() < 1e-6);
}

#[test]
fn set_scale_syncs_to_hecs() {
    let mut world = World::new("test");
    let e = world.spawn("test_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let new_scale = Vec3::new(2.0, 3.0, 4.0);
    world.set_scale(e, new_scale).unwrap();
    let read_scale = world.scale(e).unwrap();
    assert!((read_scale - new_scale).length() < 1e-6);
}

#[test]
fn ecs_only_entity_has_transform() {
    let mut world = World::new("test");
    let e = world.finalize_ecs_spawn("pure_ecs".to_string());
    let pos = world.position(e).unwrap();
    assert_eq!(pos, WorldPosition::default());
}

#[test]
fn build_render_scene_reflects_set_position() {
    let mut world = World::new("test");
    let e = world.spawn("moved_obj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    let new_pos = WorldPosition::new(glam::IVec3::ZERO, Vec3::new(5.0, 0.0, 0.0));
    world.set_position(e, new_pos).unwrap();
    let scene = world.build_render_scene();
    assert!(!scene.objects.is_empty());
    let obj = scene.objects.iter().find(|o| o.name == "moved_obj").unwrap();
    assert!((obj.position.x - 5.0).abs() < 0.01);
}
