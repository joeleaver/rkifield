//! Tool handler structs for observation tools.
//!
//! Each struct implements [`ToolHandler`] and calls the corresponding
//! [`AutomationApi`] method.

use crate::registry::*;
use rkf_core::automation::AutomationApi;
use serde_json::Value;

// --- Screenshot tool ---

pub(super) struct ScreenshotHandler;

impl ToolHandler for ScreenshotHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let width = params.get("width").and_then(|v| v.as_u64()).unwrap_or(1920) as u32;
        let height = params.get("height").and_then(|v| v.as_u64()).unwrap_or(1080) as u32;

        match api.screenshot(width, height) {
            Ok(data) => Ok(ToolResponse::Image {
                data,
                mime_type: "image/png".to_string(),
            }),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Screenshot Window tool (full editor UI + viewport) ---

pub(super) struct ScreenshotWindowHandler;

impl ToolHandler for ScreenshotWindowHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.screenshot_window() {
            Ok(data) => Ok(ToolResponse::Image {
                data,
                mime_type: "image/png".to_string(),
            }),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Scene Graph tool ---

pub(super) struct SceneGraphHandler;

impl ToolHandler for SceneGraphHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.scene_graph() {
            Ok(snapshot) => Ok(serde_json::to_value(snapshot).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Entity Inspect tool ---

pub(super) struct EntityInspectHandler;

impl ToolHandler for EntityInspectHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".to_string()))?;

        match api.entity_inspect(entity_id) {
            Ok(snapshot) => Ok(serde_json::to_value(snapshot).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Render Stats tool ---

pub(super) struct RenderStatsHandler;

impl ToolHandler for RenderStatsHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.render_stats() {
            Ok(stats) => Ok(serde_json::to_value(stats).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Log Read tool ---

pub(super) struct LogReadHandler;

impl ToolHandler for LogReadHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let lines = params.get("lines").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        match api.read_log(lines) {
            Ok(entries) => Ok(serde_json::to_value(entries).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Camera Get tool ---

pub(super) struct CameraGetHandler;

impl ToolHandler for CameraGetHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.camera_state() {
            Ok(state) => Ok(serde_json::to_value(state).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Brick Pool Stats tool ---

pub(super) struct BrickPoolStatsHandler;

impl ToolHandler for BrickPoolStatsHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.brick_pool_stats() {
            Ok(stats) => Ok(serde_json::to_value(stats).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Spatial Query tool ---

pub(super) struct SpatialQueryHandler;

impl ToolHandler for SpatialQueryHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let chunk = [
            params.get("chunk_x").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            params.get("chunk_y").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            params.get("chunk_z").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
        ];
        let local = [
            params.get("local_x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("local_y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("local_z").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        ];

        match api.spatial_query(chunk, local) {
            Ok(result) => Ok(serde_json::to_value(result).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Asset Status tool ---

pub(super) struct AssetStatusHandler;

impl ToolHandler for AssetStatusHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.asset_status() {
            Ok(status) => Ok(serde_json::to_value(status).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Debug Mode tool ---

pub(super) struct DebugModeHandler;

impl ToolHandler for DebugModeHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let mode = params
            .get("mode")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("mode is required (0-5)".to_string()))?
            as u32;

        match api.execute_command(&format!("debug_mode {mode}")) {
            Ok(msg) => Ok(serde_json::json!({ "status": "ok", "message": msg }).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Camera Set tool ---

pub(super) struct CameraSetHandler;

impl ToolHandler for CameraSetHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let x = params.get("x").and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::InvalidParams("x is required".to_string()))? as f32;
        let y = params.get("y").and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::InvalidParams("y is required".to_string()))? as f32;
        let z = params.get("z").and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::InvalidParams("z is required".to_string()))? as f32;
        let yaw_deg = params.get("yaw").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let pitch_deg = params.get("pitch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        // Route through execute_command (same pattern as debug_mode) so it
        // works over IPC without needing a separate bridge method.
        let cmd = format!("camera_set {x} {y} {z} {yaw_deg} {pitch_deg}");
        match api.execute_command(&cmd) {
            Ok(msg) => Ok(serde_json::json!({
                "status": "ok",
                "message": msg,
                "position": [x, y, z],
                "yaw": yaw_deg,
                "pitch": pitch_deg,
            }).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Environment Set tool ---

pub(super) struct EnvSetHandler;

impl ToolHandler for EnvSetHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let property = params
            .get("property")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("property is required".to_string()))?;
        // Accept both numbers and strings as value.
        let value = params
            .get("value")
            .ok_or_else(|| ToolError::InvalidParams("value is required".to_string()))?;
        let value_str = if let Some(s) = value.as_str() {
            s.to_string()
        } else if let Some(b) = value.as_bool() {
            b.to_string()
        } else {
            // Numeric — use Display to get clean float/int representation.
            value.to_string()
        };

        let cmd = format!("env_set {property} {value_str}");
        match api.execute_command(&cmd) {
            Ok(msg) => Ok(serde_json::json!({ "status": "ok", "message": msg }).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Environment Get tool ---

pub(super) struct EnvGetHandler;

impl ToolHandler for EnvGetHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let property = params
            .get("property")
            .and_then(|v| v.as_str())
            .unwrap_or("all");

        let cmd = format!("env_get {property}");
        match api.execute_command(&cmd) {
            Ok(val) => Ok(serde_json::json!({ "property": property, "value": val }).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Node Find tool ---

pub(super) struct NodeFindHandler;

impl ToolHandler for NodeFindHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;
        let node_name = params.get("node_name").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("node_name is required".to_string()))?;

        match api.node_find(object_id, node_name) {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(val.into())
            }
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Node Add Child tool ---

pub(super) struct NodeAddChildHandler;

impl ToolHandler for NodeAddChildHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;
        let parent_node = params.get("parent_node").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("parent_node is required".to_string()))?;
        let child_primitive = params.get("child_primitive").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("child_primitive is required".to_string()))?;
        let p: Vec<f32> = params.get("params").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .unwrap_or_default();
        let name = params.get("name").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("name is required".to_string()))?;
        let material_id = params.get("material_id").and_then(|v| v.as_u64()).unwrap_or(0) as u16;

        match api.node_add_child(object_id, parent_node, child_primitive, &p, name, material_id) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Node Remove tool ---

pub(super) struct NodeRemoveHandler;

impl ToolHandler for NodeRemoveHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;
        let node_name = params.get("node_name").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("node_name is required".to_string()))?;

        match api.node_remove(object_id, node_name) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Scene Create tool ---

pub(super) struct SceneCreateHandler;

impl ToolHandler for SceneCreateHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let name = params.get("name").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("name is required".to_string()))?;

        match api.scene_create(name) {
            Ok(index) => Ok(serde_json::json!({"status": "ok", "index": index}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Scene List tool ---

pub(super) struct SceneListHandler;

impl ToolHandler for SceneListHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.scene_list() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(val.into())
            }
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Scene Set Active tool ---

pub(super) struct SceneSetActiveHandler;

impl ToolHandler for SceneSetActiveHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let index = params.get("index").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("index is required".to_string()))? as usize;

        match api.scene_set_active(index) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Scene Set Persistent tool ---

pub(super) struct SceneSetPersistentHandler;

impl ToolHandler for SceneSetPersistentHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let index = params.get("index").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("index is required".to_string()))? as usize;
        let persistent = params.get("persistent").and_then(|v| v.as_bool())
            .ok_or_else(|| ToolError::InvalidParams("persistent is required".to_string()))?;

        match api.scene_set_persistent(index, persistent) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Scene Swap tool ---

pub(super) struct SceneSwapHandler;

impl ToolHandler for SceneSwapHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.scene_swap() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(val.into())
            }
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Camera Spawn tool ---

pub(super) struct CameraSpawnHandler;

impl ToolHandler for CameraSpawnHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let label = params.get("label").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("label is required".to_string()))?;
        let position = [
            params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        ];
        let yaw = params.get("yaw").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let pitch = params.get("pitch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let fov = params.get("fov").and_then(|v| v.as_f64()).unwrap_or(60.0) as f32;

        match api.camera_spawn(label, position, yaw, pitch, fov) {
            Ok(entity_id) => Ok(serde_json::json!({"status": "ok", "entity_id": entity_id}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Camera List tool ---

pub(super) struct CameraListHandler;

impl ToolHandler for CameraListHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.camera_list() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(val.into())
            }
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Camera Snap To tool ---

pub(super) struct CameraSnapToHandler;

impl ToolHandler for CameraSnapToHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let entity_id = params.get("entity_id").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".to_string()))?;

        match api.camera_snap_to(entity_id) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Light Spawn tool ---

pub(super) struct LightSpawnHandler;

impl ToolHandler for LightSpawnHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let light_type = params.get("light_type").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("light_type is required ('point' or 'spot')".to_string()))?;
        let position = [
            params.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            params.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        ];

        match api.light_spawn(light_type, position) {
            Ok(id) => Ok(serde_json::json!({"status": "ok", "light_id": id}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- Voxel Slice tool ---

pub(super) struct VoxelSliceHandler;

impl ToolHandler for VoxelSliceHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;
        let y_coord = params.get("y_coord").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        match api.voxel_slice(object_id, y_coord) {
            Ok(result) => Ok(serde_json::to_value(result).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Sculpt Apply tool ---

pub(super) struct SculptApplyHandler;

impl ToolHandler for SculptApplyHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;

        let position = match params.get("position").and_then(|v| v.as_array()) {
            Some(arr) if arr.len() >= 3 => [
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
            ],
            _ => return Err(ToolError::InvalidParams("position is required as [x, y, z]".to_string())),
        };

        let mode = params.get("mode").and_then(|v| v.as_str()).unwrap_or("add");
        let radius = params.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let strength = params.get("strength").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let material_id = params.get("material_id").and_then(|v| v.as_u64()).unwrap_or(1) as u16;

        match api.sculpt_apply(object_id, position, mode, radius, strength, material_id) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Voxelize / Save / Open tools ---

pub(super) struct VoxelizeHandler;

impl ToolHandler for VoxelizeHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;
        match api.execute_command(&format!("voxelize {object_id}")) {
            Ok(msg) => Ok(serde_json::json!({"status": "ok", "message": msg}).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

pub(super) struct SceneSaveHandler;

impl ToolHandler for SceneSaveHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let cmd = if let Some(path) = params.get("path").and_then(|v| v.as_str()) {
            format!("save {path}")
        } else {
            "save".to_string()
        };
        match api.execute_command(&cmd) {
            Ok(msg) => Ok(serde_json::json!({"status": "ok", "message": msg}).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

pub(super) struct SceneOpenHandler;

impl ToolHandler for SceneOpenHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let path = params.get("path").and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("path is required".to_string()))?;
        match api.execute_command(&format!("open {path}")) {
            Ok(msg) => Ok(serde_json::json!({"status": "ok", "message": msg}).into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Object Shape tool ---

pub(super) struct ObjectShapeHandler;

impl ToolHandler for ObjectShapeHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let object_id = params.get("object_id").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("object_id is required".to_string()))? as u32;

        match api.object_shape(object_id) {
            Ok(result) => Ok(serde_json::to_value(result).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Material / Shader tools ---

pub(super) struct ShaderListHandler;

impl ToolHandler for ShaderListHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.shader_list() {
            Ok(list) => Ok(serde_json::to_value(list).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

pub(super) struct MaterialListHandler;

impl ToolHandler for MaterialListHandler {
    fn call(&self, api: &dyn AutomationApi, _params: Value) -> Result<ToolResponse, ToolError> {
        match api.material_list() {
            Ok(list) => Ok(serde_json::to_value(list).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

pub(super) struct MaterialGetHandler;

impl ToolHandler for MaterialGetHandler {
    fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<ToolResponse, ToolError> {
        let slot = params.get("slot").and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("slot is required".to_string()))? as u16;

        match api.material_get(slot) {
            Ok(snapshot) => Ok(serde_json::to_value(snapshot).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}
