//! Tests for the automation API types, trait, and stub.

use std::collections::HashMap;

use super::*;

// Compile-time object-safety check — if the trait is not object-safe this
// function will fail to compile.
fn _assert_object_safe(_: &dyn AutomationApi) {}

// Verify StubAutomationApi can be constructed.
fn make_stub() -> StubAutomationApi {
    StubAutomationApi
}

// Helper: assert that an AutomationResult is the NotImplemented variant.
fn assert_not_implemented<T: std::fmt::Debug>(result: AutomationResult<T>) {
    match result {
        Err(AutomationError::NotImplemented(_)) => {}
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

#[test]
fn stub_screenshot_not_implemented() {
    assert_not_implemented(make_stub().screenshot(1920, 1080));
}

#[test]
fn stub_scene_graph_not_implemented() {
    assert_not_implemented(make_stub().scene_graph());
}

#[test]
fn stub_entity_inspect_not_implemented() {
    assert_not_implemented(make_stub().entity_inspect("42"));
}

#[test]
fn stub_render_stats_not_implemented() {
    assert_not_implemented(make_stub().render_stats());
}

#[test]
fn stub_asset_status_not_implemented() {
    assert_not_implemented(make_stub().asset_status());
}

#[test]
fn stub_read_log_not_implemented() {
    assert_not_implemented(make_stub().read_log(10));
}

#[test]
fn stub_camera_state_not_implemented() {
    assert_not_implemented(make_stub().camera_state());
}

#[test]
fn stub_brick_pool_stats_not_implemented() {
    assert_not_implemented(make_stub().brick_pool_stats());
}

#[test]
fn stub_spatial_query_not_implemented() {
    assert_not_implemented(make_stub().spatial_query([0, 0, 0], [0.0, 0.0, 0.0]));
}

#[test]
fn stub_entity_spawn_not_implemented() {
    let def = EntityDef {
        name: "test".into(),
        components: vec![],
    };
    assert_not_implemented(make_stub().entity_spawn(def));
}

#[test]
fn stub_entity_despawn_not_implemented() {
    assert_not_implemented(make_stub().entity_despawn("1"));
}

#[test]
fn stub_entity_set_component_not_implemented() {
    let comp = ComponentDef::Custom {
        type_name: "Foo".into(),
        data: serde_json::Value::Null,
    };
    assert_not_implemented(make_stub().entity_set_component("1", comp));
}

#[test]
fn stub_material_set_not_implemented() {
    assert_not_implemented(make_stub().material_set(0, MaterialDef::default()));
}

#[test]
fn stub_brush_apply_not_implemented() {
    assert_not_implemented(make_stub().brush_apply(serde_json::json!({})));
}

#[test]
fn stub_scene_load_not_implemented() {
    assert_not_implemented(make_stub().scene_load("foo.rkscene"));
}

#[test]
fn stub_scene_save_not_implemented() {
    assert_not_implemented(make_stub().scene_save("foo.rkscene"));
}

#[test]
fn stub_camera_set_not_implemented() {
    assert_not_implemented(make_stub().camera_set([0, 0, 0], [0.0; 3], [0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn stub_quality_preset_not_implemented() {
    assert_not_implemented(make_stub().quality_preset(QualityPreset::High));
}

#[test]
fn stub_execute_command_not_implemented() {
    assert_not_implemented(make_stub().execute_command("help"));
}

// --- Serde round-trip tests for all result types -----------------------

#[test]
fn scene_graph_snapshot_serde_roundtrip() {
    let snap = SceneGraphSnapshot {
        entities: vec![EntityNode {
            id: "1".into(),
            name: "root".into(),
            parent: None,
            entity_type: "sdf_object".into(),
            transform: [0.0; 10],
        }],
    };
    let json = serde_json::to_string(&snap).unwrap();
    let back: SceneGraphSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.entities.len(), 1);
    assert_eq!(back.entities[0].id, "1");
}

#[test]
fn entity_snapshot_serde_roundtrip() {
    let snap = EntitySnapshot {
        id: "7".into(),
        name: "lamp".into(),
        components: HashMap::from([("light".into(), serde_json::json!({"intensity": 100.0}))]),
    };
    let json = serde_json::to_string(&snap).unwrap();
    let back: EntitySnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id, "7");
    assert!(back.components.contains_key("light"));
}

#[test]
fn render_stats_serde_roundtrip() {
    let stats = RenderStats {
        frame_time_ms: 16.6,
        pass_timings: HashMap::from([("ray_march".into(), 8.3)]),
        brick_pool_usage: 0.42,
        memory_mb: 512.0,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let back: RenderStats = serde_json::from_str(&json).unwrap();
    assert!((back.frame_time_ms - 16.6).abs() < 1e-9);
}

#[test]
fn camera_snapshot_serde_roundtrip() {
    let cam = CameraSnapshot {
        chunk: [1, 2, 3],
        local: [0.5, 1.0, -0.5],
        rotation: [0.0, 0.0, 0.0, 1.0],
        fov_degrees: 75.0,
    };
    let json = serde_json::to_string(&cam).unwrap();
    let back: CameraSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.chunk, [1, 2, 3]);
    assert!((back.fov_degrees - 75.0).abs() < 1e-6);
}

#[test]
fn asset_status_serde_roundtrip() {
    let report = AssetStatusReport {
        loaded_chunks: 16,
        pending_uploads: 4,
        total_bricks: 8192,
        pool_capacity: 65536,
    };
    let json = serde_json::to_string(&report).unwrap();
    let back: AssetStatusReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back.loaded_chunks, 16);
}

#[test]
fn log_entry_serde_roundtrip() {
    let entry = LogEntry {
        level: LogLevel::Warn,
        message: "brick pool near capacity".into(),
        timestamp_ms: 12345,
    };
    let json = serde_json::to_string(&entry).unwrap();
    let back: LogEntry = serde_json::from_str(&json).unwrap();
    assert_eq!(back.level, LogLevel::Warn);
    assert_eq!(back.timestamp_ms, 12345);
}

#[test]
fn brick_pool_stats_serde_roundtrip() {
    let stats = BrickPoolStats {
        capacity: 65536,
        allocated: 1024,
        free_list_size: 64512,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let back: BrickPoolStats = serde_json::from_str(&json).unwrap();
    assert_eq!(back.allocated, 1024);
}

#[test]
fn object_shape_result_serde_roundtrip() {
    let res = ObjectShapeResult {
        object_id: 2,
        dims: [32, 16, 32],
        voxel_size: 0.03,
        aabb_min: [-3.84, -1.92, -3.84],
        aabb_max: [3.84, 1.92, 3.84],
        empty_count: 14000,
        interior_count: 200,
        surface_count: 2184,
        y_slices: vec!["....++++....".to_string()],
    };
    let json = serde_json::to_string(&res).unwrap();
    let back: ObjectShapeResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.object_id, 2);
    assert_eq!(back.dims, [32, 16, 32]);
    assert_eq!(back.y_slices.len(), 1);
}

#[test]
fn voxel_slice_result_serde_roundtrip() {
    let res = VoxelSliceResult {
        origin: [-0.5, -0.5],
        spacing: 0.04,
        width: 25,
        height: 25,
        y_coord: 0.0,
        distances: vec![0.1; 625],
        slot_status: vec![2; 625],
    };
    let json = serde_json::to_string(&res).unwrap();
    let back: VoxelSliceResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.width, 25);
    assert_eq!(back.height, 25);
    assert!((back.spacing - 0.04).abs() < 1e-6);
    assert_eq!(back.distances.len(), 625);
}

#[test]
fn spatial_query_result_serde_roundtrip() {
    let res = SpatialQueryResult {
        distance: -0.05,
        material_id: 7,
        inside: true,
    };
    let json = serde_json::to_string(&res).unwrap();
    let back: SpatialQueryResult = serde_json::from_str(&json).unwrap();
    assert!(back.inside);
    assert_eq!(back.material_id, 7);
}

#[test]
fn stub_material_list_not_implemented() {
    assert_not_implemented(make_stub().material_list());
}

#[test]
fn stub_material_get_not_implemented() {
    assert_not_implemented(make_stub().material_get(0));
}

#[test]
fn material_info_serde_roundtrip() {
    let info = MaterialInfo {
        slot: 1,
        name: "Stone".into(),
        category: "Stone".into(),
        albedo: [0.45, 0.43, 0.40],
        roughness: 0.85,
        metallic: 0.0,
        is_emissive: false,
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: MaterialInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.slot, 1);
    assert_eq!(back.name, "Stone");
}

#[test]
fn material_snapshot_serde_roundtrip() {
    let snap = MaterialSnapshot {
        slot: 2,
        name: "Gold".into(),
        description: "Shiny metallic gold".into(),
        category: "Metal".into(),
        albedo: [1.0, 0.84, 0.0],
        roughness: 0.2,
        metallic: 1.0,
        emission_color: [0.0, 0.0, 0.0],
        emission_strength: 0.0,
        subsurface: 0.0,
        subsurface_color: [1.0, 1.0, 1.0],
        opacity: 1.0,
        ior: 1.5,
        noise_scale: 0.0,
        noise_strength: 0.0,
        noise_channels: 0,
    };
    let json = serde_json::to_string(&snap).unwrap();
    let back: MaterialSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.slot, 2);
    assert_eq!(back.name, "Gold");
    assert!((back.metallic - 1.0).abs() < 1e-6);
}

#[test]
fn material_def_serde_roundtrip() {
    let mat = MaterialDef {
        base_color: Some([1.0, 0.0, 0.0]),
        metallic: Some(0.0),
        roughness: Some(0.5),
        ..Default::default()
    };
    let json = serde_json::to_string(&mat).unwrap();
    let back: MaterialDef = serde_json::from_str(&json).unwrap();
    assert_eq!(back.base_color, Some([1.0, 0.0, 0.0]));
    assert_eq!(back.sss_radius, None);
}

#[test]
fn component_def_custom_serde_roundtrip() {
    let comp = ComponentDef::Custom {
        type_name: "rkf_runtime::Foo".into(),
        data: serde_json::json!({"bar": 42}),
    };
    let json = serde_json::to_string(&comp).unwrap();
    let back: ComponentDef = serde_json::from_str(&json).unwrap();
    match back {
        ComponentDef::Custom { type_name, .. } => {
            assert_eq!(type_name, "rkf_runtime::Foo");
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn quality_preset_serde_roundtrip() {
    for preset in [
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Ultra,
        QualityPreset::Custom,
    ] {
        let json = serde_json::to_string(&preset).unwrap();
        let back: QualityPreset = serde_json::from_str(&json).unwrap();
        assert_eq!(back, preset);
    }
}

// --- Tool discovery default method tests --------------------------------

#[test]
fn default_list_tools_json_returns_empty_array() {
    let result = make_stub().list_tools_json().unwrap();
    assert_eq!(result, serde_json::json!([]));
}

#[test]
fn default_call_tool_json_returns_not_implemented() {
    assert_not_implemented(make_stub().call_tool_json("foo", serde_json::json!({})));
}

// --- v2 default method tests -------------------------------------------

// Helper: assert a Result<_, String> is Err containing the expected text.
fn assert_err_contains<T: std::fmt::Debug>(result: Result<T, String>, needle: &str) {
    match result {
        Err(msg) => assert!(
            msg.contains(needle),
            "expected error containing {:?}, got {:?}",
            needle,
            msg
        ),
        Ok(v) => panic!("expected Err, got Ok({v:?})"),
    }
}

#[test]
fn default_object_spawn_returns_error() {
    assert_err_contains(
        make_stub().object_spawn("cube", "box", &[0.5, 0.5, 0.5], [0.0; 3], 0),
        "not supported",
    );
}

#[test]
fn default_object_despawn_returns_error() {
    assert_err_contains(make_stub().object_despawn(1), "not supported");
}

#[test]
fn default_node_set_transform_returns_error() {
    assert_err_contains(
        make_stub().node_set_transform(1, [0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
        "not supported",
    );
}

#[test]
fn default_environment_get_returns_error() {
    assert_err_contains(make_stub().environment_get(), "not supported");
}

#[test]
fn default_environment_blend_returns_error() {
    assert_err_contains(make_stub().environment_blend(0, 2.0), "not supported");
}

#[test]
fn default_env_override_returns_error() {
    assert_err_contains(make_stub().env_override("sun.intensity", 1.0), "not supported");
}

#[test]
fn automation_error_display() {
    let e = AutomationError::EntityNotFound("99".into());
    assert!(e.to_string().contains("99"));

    let e2 = AutomationError::InvalidParameter("negative size".into());
    assert!(e2.to_string().contains("negative size"));

    let e3 = AutomationError::NotImplemented("screenshot");
    assert!(e3.to_string().contains("screenshot"));
}
