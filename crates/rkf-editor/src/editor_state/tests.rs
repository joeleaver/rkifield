//! Tests for EditorState.

use super::*;
use crate::scene_io::{save_scene, ComponentData, SceneEntity, SceneFile};
use glam::{Quat, Vec3};

fn write_test_scene(path: &str) {
    let scene = SceneFile {
        version: 1,
        name: "Test".to_string(),
        entities: vec![
            SceneEntity {
                entity_id: 1,
                name: "Ground".to_string(),
                parent_id: None,
                position: Vec3::new(0.0, -0.5, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: vec![ComponentData::SdfObject {
                    asset_path: "procedural://ground".to_string(),
                }],
            },
            SceneEntity {
                entity_id: 2,
                name: "Child".to_string(),
                parent_id: Some(1),
                position: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::from_rotation_y(1.0),
                scale: Vec3::splat(0.5),
                components: vec![ComponentData::SdfObject {
                    asset_path: "procedural://pillar".to_string(),
                }],
            },
            SceneEntity {
                entity_id: 3,
                name: "Sun".to_string(),
                parent_id: None,
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: vec![ComponentData::Light {
                    light_type: "directional".to_string(),
                    color: [1.0, 0.95, 0.8],
                    intensity: 3.0,
                    range: 0.0,
                }],
            },
        ],
        environment_ron: String::new(),
    };
    let ron_str = save_scene(&scene).unwrap();
    std::fs::write(path, ron_str).unwrap();
}

#[test]
fn test_load_scene_populates_world() {
    let path = "/tmp/rkf_test_load_tree.rkscene";
    write_test_scene(path);
    let mut state = EditorState::new();
    let scene = state.load_scene(path).unwrap();
    assert_eq!(scene.entities.len(), 3);
    // Only SDF entities go in world scene (lights go to light_editor)
    let objects = &state.world.scene().objects;
    assert_eq!(objects.len(), 2); // "Ground" + "Child"
    assert_eq!(objects[0].name, "Ground");
    assert_eq!(objects[1].name, "Child");
    // "Child" has parent_id pointing to Ground
    assert_eq!(objects[1].parent_id, Some(objects[0].id));
}

#[test]
fn test_load_scene_stores_transform_and_asset_path() {
    let path = "/tmp/rkf_test_load_xform.rkscene";
    write_test_scene(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    let objects = &state.world.scene().objects;
    let ground = &objects[0];
    assert_eq!(ground.position, Vec3::new(0.0, -0.5, 0.0));
    // Asset path stored in root_node.name
    assert_eq!(ground.root_node.name, "procedural://ground");
    let child = &objects[1];
    assert_eq!(child.position, Vec3::new(1.0, 2.0, 3.0));
    assert!(child.scale.abs_diff_eq(Vec3::splat(0.5), 1e-6));
    assert_eq!(child.root_node.name, "procedural://pillar");
}

#[test]
fn test_load_scene_populates_light_editor() {
    let path = "/tmp/rkf_test_load_lights.rkscene";
    write_test_scene(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    assert_eq!(state.light_editor.all_lights().len(), 1);
    let light = &state.light_editor.all_lights()[0];
    assert_eq!(light.intensity, 3.0);
}

#[test]
fn test_load_scene_sets_path_and_clears_dirty() {
    let path = "/tmp/rkf_test_load_path.rkscene";
    write_test_scene(path);
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
    assert_eq!(saved.version, 1);
    assert_eq!(saved.name, "Untitled");
    assert!(saved.entities.is_empty());
}

#[test]
fn test_save_roundtrip_preserves_entities() {
    let path = "/tmp/rkf_test_save_rt.rkscene";
    write_test_scene(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();

    let saved = state.save_current_scene();
    assert_eq!(saved.version, 1);
    // 2 SDF entities (Ground + Child) + 1 light = 3
    assert_eq!(saved.entities.len(), 3);

    // Check SDF entities
    let ground = saved.entities.iter().find(|e| e.name == "Ground").unwrap();
    assert_eq!(ground.position, Vec3::new(0.0, -0.5, 0.0));
    assert!(ground.components.iter().any(|c| matches!(
        c,
        ComponentData::SdfObject { asset_path } if asset_path == "procedural://ground"
    )));

    let child = saved.entities.iter().find(|e| e.name == "Child").unwrap();
    assert_eq!(child.parent_id, Some(1));
    assert!(child.scale.abs_diff_eq(Vec3::splat(0.5), 1e-6));

    // Check light entity
    let light_ent = saved
        .entities
        .iter()
        .find(|e| e.components.iter().any(|c| matches!(c, ComponentData::Light { .. })))
        .unwrap();
    assert!(light_ent
        .components
        .iter()
        .any(|c| matches!(c, ComponentData::Light { intensity, .. } if (*intensity - 3.0).abs() < 1e-6)));
}

#[test]
fn test_world_defaults_to_empty() {
    let state = EditorState::new();
    assert_eq!(state.world.scene().objects.len(), 0);
}

#[test]
fn test_world_scene_is_single_source_of_truth() {
    use rkf_core::scene_node::SceneNode as CoreNode;
    let mut state = EditorState::new();
    let scene = state.world.scene_mut();
    scene.add_object("SphereObj", Vec3::ZERO, CoreNode::new("root"));
    scene.add_object("BoxObj", Vec3::ZERO, CoreNode::new("root2"));

    // World scene is the single source of truth — no sync needed.
    assert_eq!(state.world.scene().objects.len(), 2);
    assert_eq!(state.world.scene().objects[0].name, "SphereObj");
    assert_eq!(state.world.scene().objects[1].name, "BoxObj");
}

#[test]
fn test_save_scene_name_from_path() {
    let path = "/tmp/rkf_test_save_name.rkscene";
    write_test_scene(path);
    let mut state = EditorState::new();
    state.load_scene(path).unwrap();
    let saved = state.save_current_scene();
    assert_eq!(saved.name, "rkf_test_save_name");
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
            entity_id: entity.to_u64(),
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
            entity_id: entity.to_u64(),
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
            entity_id: entity.to_u64(),
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
