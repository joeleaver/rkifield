//! Tests for EditorAutomationApi.

use super::*;
use rkf_core::automation::AutomationApi;

/// Create a test EditorAutomationApi with empty state.
fn make_api() -> EditorAutomationApi {
    let state = Arc::new(Mutex::new(SharedState::new(4096, 0, 128, 128)));
    let editor_state = Arc::new(Mutex::new(EditorState::new()));
    EditorAutomationApi::new(state, editor_state)
}

/// Spawn a test object and return its object_id.
fn spawn_test_object(api: &EditorAutomationApi) -> u32 {
    api.object_spawn("test_obj", "sphere", &[0.5], [0.0, 0.0, 0.0], 0)
        .unwrap()
}

#[test]
fn automation_node_find_returns_json() {
    let api = make_api();
    let oid = spawn_test_object(&api);
    let result = api.node_find(oid, "test_obj").unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(json["name"], "test_obj");
    assert!(json["sdf_source"].as_str().unwrap().contains("Analytical"));
}

#[test]
fn automation_node_add_child_adds_to_tree() {
    let api = make_api();
    let oid = spawn_test_object(&api);
    api.node_add_child(oid, "test_obj", "box", &[0.3, 0.3, 0.3], "child_box", 1)
        .unwrap();

    let result = api.node_find(oid, "test_obj").unwrap();
    let json: serde_json::Value = serde_json::from_str(&result).unwrap();
    let children = json["children"].as_array().unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0], "child_box");
}

#[test]
fn automation_scene_create_adds_scene() {
    let api = make_api();
    let result = api.scene_list().unwrap();
    let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    assert_eq!(scenes.len(), 1);

    let idx = api.scene_create("level2").unwrap();
    assert_eq!(idx, 1);

    let result = api.scene_list().unwrap();
    let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    assert_eq!(scenes.len(), 2);
    assert_eq!(scenes[1]["name"], "level2");
}

#[test]
fn automation_scene_set_active_changes_target() {
    let api = make_api();
    api.scene_create("second").unwrap();
    api.scene_set_active(1).unwrap();

    let result = api.scene_list().unwrap();
    let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
    assert_eq!(scenes[1]["active"], true);
    assert_eq!(scenes[0]["active"], false);
}

#[test]
fn automation_camera_spawn_creates_entity() {
    let api = make_api();
    let cam_id = api
        .camera_spawn("Main", [1.0, 2.0, 3.0], 45.0, -10.0, 75.0)
        .unwrap();
    assert!(cam_id != 0); // Should get a valid non-zero ID

    let list = api.camera_list().unwrap();
    let cameras: Vec<serde_json::Value> = serde_json::from_str(&list).unwrap();
    assert_eq!(cameras.len(), 1);
    assert_eq!(cameras[0]["label"], "Main");
    assert!((cameras[0]["fov_degrees"].as_f64().unwrap() - 75.0).abs() < 0.01);
}

#[test]
fn automation_camera_list_includes_spawned() {
    let api = make_api();
    api.camera_spawn("CamA", [0.0; 3], 0.0, 0.0, 60.0).unwrap();
    api.camera_spawn("CamB", [5.0, 0.0, 0.0], 90.0, 0.0, 90.0)
        .unwrap();

    let list = api.camera_list().unwrap();
    let cameras: Vec<serde_json::Value> = serde_json::from_str(&list).unwrap();
    assert_eq!(cameras.len(), 2);
    let labels: Vec<&str> = cameras
        .iter()
        .map(|c| c["label"].as_str().unwrap())
        .collect();
    assert!(labels.contains(&"CamA"));
    assert!(labels.contains(&"CamB"));
}
