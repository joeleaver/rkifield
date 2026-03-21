//! Tests for EditorState.

use super::*;
use glam::{Quat, Vec3};
use rkf_runtime::scene_file_v3::{
    EntityRecord, SceneFileV3, serialize_scene_v3,
};

fn write_test_scene_v3(path: &str) {
    let mut scene = SceneFileV3::new();

    // Entity 1: Ground (SDF object)
    let id1 = rkf_runtime::behavior::StableId::new();
    let mut e1 = EntityRecord::new(id1.uuid());
    e1.insert_component(
        "Transform",
        &rkf_runtime::components::Transform {
            position: rkf_core::WorldPosition::new(glam::IVec3::ZERO, Vec3::new(0.0, -0.5, 0.0)),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        },
    ).unwrap();
    e1.insert_component(
        "EditorMetadata",
        &rkf_runtime::components::EditorMetadata {
            name: "Ground".to_string(),
            tags: vec![],
            locked: false,
        },
    ).unwrap();
    e1.insert_component(
        "SdfTree",
        &rkf_runtime::components::SdfTree {
            root: rkf_core::scene_node::SceneNode::new("procedural://ground"),
            asset_path: None,
            aabb: rkf_core::aabb::Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        },
    ).unwrap();
    scene.entities.push(e1);

    // Entity 2: Child (SDF object, child of Ground)
    let id2 = rkf_runtime::behavior::StableId::new();
    let mut e2 = EntityRecord::new(id2.uuid());
    e2.parent = Some(id1.uuid());
    e2.insert_component(
        "Transform",
        &rkf_runtime::components::Transform {
            position: rkf_core::WorldPosition::new(glam::IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
            rotation: Quat::from_rotation_y(1.0),
            scale: Vec3::splat(0.5),
        },
    ).unwrap();
    e2.insert_component(
        "EditorMetadata",
        &rkf_runtime::components::EditorMetadata {
            name: "Child".to_string(),
            tags: vec![],
            locked: false,
        },
    ).unwrap();
    e2.insert_component(
        "SdfTree",
        &rkf_runtime::components::SdfTree {
            root: rkf_core::scene_node::SceneNode::new("procedural://pillar"),
            asset_path: None,
            aabb: rkf_core::aabb::Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        },
    ).unwrap();
    scene.entities.push(e2);

    // Store lights in properties
    use crate::light_editor::{SceneLight, SceneLightType};
    let lights = vec![SceneLight {
        id: 1,
        light_type: SceneLightType::Point,
        position: Vec3::ZERO,
        direction: Vec3::new(0.0, -1.0, 0.0),
        color: Vec3::new(1.0, 0.95, 0.8),
        intensity: 3.0,
        range: 10.0,
        spot_inner_angle: 0.0,
        spot_outer_angle: 0.0,
        cast_shadows: true,
        cookie_path: None,
    }];
    if let Ok(s) = ron::to_string(&lights) {
        scene.properties.insert("lights".into(), s);
    }

    let ron_str = serialize_scene_v3(&scene).unwrap();
    std::fs::write(path, ron_str).unwrap();
}

#[test]
fn test_load_scene_populates_world() {
    let path = "/tmp/rkf_test_load_tree_v3.rkscene";
    write_test_scene_v3(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    // 2 SDF entities go in world
    let render = state.world.build_render_scene();
    assert_eq!(render.objects.len(), 2);

    let names: Vec<&str> = render.objects.iter().map(|o| o.name.as_str()).collect();
    assert!(names.contains(&"Ground"));
    assert!(names.contains(&"Child"));
}

#[test]
fn test_load_scene_populates_light_editor() {
    let path = "/tmp/rkf_test_load_lights_v3.rkscene";
    write_test_scene_v3(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    assert_eq!(state.light_editor.all_lights().len(), 1);
    let light = &state.light_editor.all_lights()[0];
    assert!((light.intensity - 3.0).abs() < 1e-6);
}

#[test]
fn test_load_scene_sets_path_and_clears_dirty() {
    let path = "/tmp/rkf_test_load_path_v3.rkscene";
    write_test_scene_v3(path);
    let mut state = EditorState::new();
    state.unsaved_changes.mark_changed();
    state.load_scene(path).unwrap();
    assert_eq!(state.current_scene_path.as_deref(), Some(path));
    assert!(!state.unsaved_changes.needs_save());
}

#[test]
fn test_load_nonexistent_file() {
    let mut state = EditorState::new();
    let result = state.load_scene("/nonexistent/path.rkscene");
    assert!(result.is_err());
}

#[test]
fn test_save_empty_scene() {
    let state = EditorState::new();
    let saved = state.save_current_scene();
    assert_eq!(saved.version, 3);
    assert!(saved.entities.is_empty());
}

#[test]
fn test_save_roundtrip_preserves_entities() {
    let path = "/tmp/rkf_test_save_rt_v3.rkscene";
    write_test_scene_v3(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();

    let saved = state.save_current_scene();
    assert_eq!(saved.version, 3);
    // 2 SDF entities
    assert_eq!(saved.entities.len(), 2);

    // Check entity metadata
    let ground = saved.entities.iter().find(|e| {
        e.get_component::<rkf_runtime::components::EditorMetadata>(
            "EditorMetadata",
        )
        .and_then(|r| r.ok())
        .map(|m| m.name == "Ground")
        .unwrap_or(false)
    });
    assert!(ground.is_some());
}

#[test]
fn test_world_defaults_to_empty() {
    let state = EditorState::new();
    let render = state.world.build_render_scene();
    assert_eq!(render.objects.len(), 0);
}

#[test]
fn test_world_scene_is_single_source_of_truth() {
    use rkf_core::scene_node::SdfPrimitive;
    let mut state = EditorState::new();
    state.world.spawn("SphereObj").sdf(SdfPrimitive::Sphere { radius: 0.5 }).material(0).build();
    state.world.spawn("BoxObj").sdf(SdfPrimitive::Box { half_extents: glam::Vec3::splat(0.5) }).material(0).build();

    let render = state.world.build_render_scene();
    assert_eq!(render.objects.len(), 2);
    let names: Vec<&str> = render.objects.iter().map(|o| o.name.as_str()).collect();
    assert!(names.contains(&"SphereObj"));
    assert!(names.contains(&"BoxObj"));
}

#[test]
fn test_save_scene_name_from_path() {
    let path = "/tmp/rkf_test_save_name_v3.rkscene";
    write_test_scene_v3(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    let saved = state.save_current_scene();
    let name = saved.properties.get("name").cloned().unwrap_or_default();
    assert_eq!(name, "rkf_test_save_name_v3");
}

// ── Undo/redo application tests ───────────────────────────────────────

#[test]
fn undo_transform_restores_position() {
    use rkf_core::scene_node::SdfPrimitive;
    let mut state = EditorState::new();
    let entity = state.world.spawn("obj")
        .position_vec3(Vec3::new(1.0, 2.0, 3.0))
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(0)
        .build();

    let action = crate::undo::UndoAction {
        kind: crate::undo::UndoActionKind::Transform {
            entity_id: entity,
            old_pos: Vec3::new(1.0, 2.0, 3.0),
            old_rot: Quat::IDENTITY,
            old_scale: Vec3::ONE,
            new_pos: Vec3::new(10.0, 20.0, 30.0),
            new_rot: Quat::from_rotation_y(1.0),
            new_scale: Vec3::splat(2.0),
        },
        timestamp_ms: 0,
        description: "Move".into(),
    };

    // Apply redo (new values).
    state.apply_undo_action(&action, false);
    let pos = state.world.position(entity).unwrap().to_vec3();
    assert!(pos.abs_diff_eq(Vec3::new(10.0, 20.0, 30.0), 1e-6));

    // Apply undo (old values).
    state.apply_undo_action(&action, true);
    let pos = state.world.position(entity).unwrap().to_vec3();
    assert!(pos.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
}

#[test]
fn redo_transform_reapplies() {
    use rkf_core::scene_node::SdfPrimitive;
    let mut state = EditorState::new();
    let entity = state.world.spawn("obj")
        .position_vec3(Vec3::ZERO)
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(0)
        .build();

    let action = crate::undo::UndoAction {
        kind: crate::undo::UndoActionKind::Transform {
            entity_id: entity,
            old_pos: Vec3::ZERO,
            old_rot: Quat::IDENTITY,
            old_scale: Vec3::ONE,
            new_pos: Vec3::new(5.0, 0.0, 0.0),
            new_rot: Quat::IDENTITY,
            new_scale: Vec3::splat(3.0),
        },
        timestamp_ms: 0,
        description: "Move".into(),
    };

    // Undo then redo should restore new values.
    state.apply_undo_action(&action, true);
    state.apply_undo_action(&action, false);
    let scale = state.world.scale(entity).unwrap();
    assert!(scale.abs_diff_eq(Vec3::splat(3.0), 1e-6));
}

#[test]
fn undo_spawn_despawns_entity() {
    use rkf_core::scene_node::SdfPrimitive;
    let mut state = EditorState::new();
    let entity = state.world.spawn("spawned")
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(0)
        .build();
    assert!(state.world.is_alive(entity));

    let action = crate::undo::UndoAction {
        kind: crate::undo::UndoActionKind::SpawnEntity {
            entity_id: entity,
        },
        timestamp_ms: 0,
        description: "Spawn".into(),
    };

    // Undo spawn = despawn.
    state.apply_undo_action(&action, true);
    assert!(!state.world.is_alive(entity));
}

#[test]
fn undo_with_empty_stack_noop() {
    let mut state = EditorState::new();
    // No actions pushed — undo/redo should return None silently.
    assert!(state.undo.undo().is_none());
    assert!(state.undo.redo().is_none());
}
