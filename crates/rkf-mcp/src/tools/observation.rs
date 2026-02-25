//! Observation tool handlers and shared dispatch functions.
//!
//! Each tool calls the corresponding `AutomationApi` method and returns the result
//! as JSON. The shared functions `standard_tool_definitions()` and
//! `dispatch_tool_call()` allow any `AutomationApi` implementor to serve all 13
//! observation tools without needing a `ToolRegistry`.

use crate::registry::*;
use rkf_core::automation::{AutomationApi, AutomationError, AutomationResult};
use serde_json::Value;
use std::sync::Arc;

// --- Screenshot tool ---

struct ScreenshotHandler;

impl ToolHandler for ScreenshotHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

// --- Scene Graph tool ---

struct SceneGraphHandler;

impl ToolHandler for SceneGraphHandler {
    fn call(&self, api: &dyn AutomationApi, _params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        match api.scene_graph() {
            Ok(snapshot) => Ok(serde_json::to_value(snapshot).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Entity Inspect tool ---

struct EntityInspectHandler;

impl ToolHandler for EntityInspectHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".to_string()))?;

        match api.entity_inspect(entity_id) {
            Ok(snapshot) => Ok(serde_json::to_value(snapshot).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Render Stats tool ---

struct RenderStatsHandler;

impl ToolHandler for RenderStatsHandler {
    fn call(&self, api: &dyn AutomationApi, _params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        match api.render_stats() {
            Ok(stats) => Ok(serde_json::to_value(stats).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Log Read tool ---

struct LogReadHandler;

impl ToolHandler for LogReadHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        let lines = params.get("lines").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        match api.read_log(lines) {
            Ok(entries) => Ok(serde_json::to_value(entries).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Camera Get tool ---

struct CameraGetHandler;

impl ToolHandler for CameraGetHandler {
    fn call(&self, api: &dyn AutomationApi, _params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        match api.camera_state() {
            Ok(state) => Ok(serde_json::to_value(state).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Brick Pool Stats tool ---

struct BrickPoolStatsHandler;

impl ToolHandler for BrickPoolStatsHandler {
    fn call(&self, api: &dyn AutomationApi, _params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        match api.brick_pool_stats() {
            Ok(stats) => Ok(serde_json::to_value(stats).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Spatial Query tool ---

struct SpatialQueryHandler;

impl ToolHandler for SpatialQueryHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

struct AssetStatusHandler;

impl ToolHandler for AssetStatusHandler {
    fn call(&self, api: &dyn AutomationApi, _params: serde_json::Value) -> Result<ToolResponse, ToolError> {
        match api.asset_status() {
            Ok(status) => Ok(serde_json::to_value(status).map_err(ToolError::from)?.into()),
            Err(e) => Err(ToolError::EngineError(e.to_string())),
        }
    }
}

// --- Debug Mode tool ---

struct DebugModeHandler;

impl ToolHandler for DebugModeHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

struct CameraSetHandler;

impl ToolHandler for CameraSetHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

struct EnvSetHandler;

impl ToolHandler for EnvSetHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

struct EnvGetHandler;

impl ToolHandler for EnvGetHandler {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<ToolResponse, ToolError> {
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

/// Register all built-in observation tools with the registry.
pub fn register_observation_tools(registry: &mut ToolRegistry) {
    registry.register(
        ToolDefinition {
            name: "screenshot".to_string(),
            description: "Capture current viewport as PNG image".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "width".to_string(),
                    description: "Image width in pixels".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1920)),
                },
                ParameterDef {
                    name: "height".to_string(),
                    description: "Image height in pixels".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1080)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Base64-encoded PNG image data".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ScreenshotHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_graph".to_string(),
            description: "List all entities with hierarchy, types, and transforms".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON entity tree".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SceneGraphHandler),
    );

    registry.register(
        ToolDefinition {
            name: "entity_inspect".to_string(),
            description: "Read all components of a specific entity".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "entity_id".to_string(),
                description: "Entity ID to inspect".to_string(),
                param_type: ParamType::Integer,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "JSON component data".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EntityInspectHandler),
    );

    registry.register(
        ToolDefinition {
            name: "render_stats".to_string(),
            description: "Frame time, pass timings, brick pool usage, memory stats".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON stats object".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(RenderStatsHandler),
    );

    registry.register(
        ToolDefinition {
            name: "log_read".to_string(),
            description: "Read recent engine log entries".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "lines".to_string(),
                description: "Number of log lines to return".to_string(),
                param_type: ParamType::Integer,
                required: false,
                default: Some(serde_json::json!(50)),
            }],
            return_type: ReturnTypeDef {
                description: "JSON array of log entries".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(LogReadHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_get".to_string(),
            description: "Current camera position, orientation, and FOV".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON camera state".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(CameraGetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "brick_pool_stats".to_string(),
            description: "Brick pool occupancy, free list size, LRU state".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON pool stats".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(BrickPoolStatsHandler),
    );

    registry.register(
        ToolDefinition {
            name: "spatial_query".to_string(),
            description: "Query SDF distance and material at a world position".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "chunk_x".to_string(),
                    description: "Chunk X coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "chunk_y".to_string(),
                    description: "Chunk Y coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "chunk_z".to_string(),
                    description: "Chunk Z coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_x".to_string(),
                    description: "Local X position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_y".to_string(),
                    description: "Local Y position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_z".to_string(),
                    description: "Local Z position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "SDF distance, material ID, inside flag".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SpatialQueryHandler),
    );

    registry.register(
        ToolDefinition {
            name: "asset_status".to_string(),
            description: "Loading progress, loaded chunks, pending uploads".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON status report".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(AssetStatusHandler),
    );

    registry.register(
        ToolDefinition {
            name: "debug_mode".to_string(),
            description: "Set shading debug visualization mode. 0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse only, 5=specular only".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "mode".to_string(),
                description: "Debug mode (0=normal shading, 1=normals, 2=positions, 3=material IDs, 4=diffuse only, 5=specular only)".to_string(),
                param_type: ParamType::Integer,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "Confirmation of mode change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(DebugModeHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_set".to_string(),
            description: "Set camera position and orientation".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "x".to_string(),
                    description: "X position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "y".to_string(),
                    description: "Y position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "z".to_string(),
                    description: "Z position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "yaw".to_string(),
                    description: "Yaw in degrees".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                },
                ParameterDef {
                    name: "pitch".to_string(),
                    description: "Pitch in degrees".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of camera position change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(CameraSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "env_set".to_string(),
            description: "Set an environment property (atmosphere, fog, clouds, post-processing). Use env_get with property='all' to see available properties.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "property".to_string(),
                    description: "Property path, e.g. 'clouds.enabled', 'atmosphere.sun_intensity', 'post_process.exposure', 'fog.density'".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "value".to_string(),
                    description: "Value to set (number, boolean, or string)".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of property change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EnvSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "env_get".to_string(),
            description: "Get an environment property value. Use property='all' for a summary of all settings.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "property".to_string(),
                description: "Property path (e.g. 'clouds.enabled', 'atmosphere.sun_intensity') or 'all' for summary".to_string(),
                param_type: ParamType::String,
                required: false,
                default: Some(serde_json::json!("all")),
            }],
            return_type: ReturnTypeDef {
                description: "Property value".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EnvGetHandler),
    );
}

// ---------------------------------------------------------------------------
// Shared dispatch functions — used by EditorAutomationApi & TestbedAutomationApi
// ---------------------------------------------------------------------------

/// Wrap a successful JSON value as an MCP `ToolsCallResult` text content block.
fn tool_ok_json(value: Value) -> Value {
    let text = serde_json::to_string_pretty(&value).unwrap_or_default();
    serde_json::json!({
        "content": [{ "type": "text", "text": text }]
    })
}

/// Wrap an error message as an MCP `ToolsCallResult` with `isError: true`.
fn tool_err_json(msg: &str) -> Value {
    serde_json::json!({
        "content": [{ "type": "text", "text": format!("Error: {msg}") }],
        "isError": true
    })
}

/// Wrap raw PNG bytes as an MCP `ToolsCallResult` image content block.
fn tool_image_json(data: Vec<u8>) -> Value {
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
    serde_json::json!({
        "content": [{ "type": "image", "data": b64, "mimeType": "image/png" }]
    })
}

/// Return a JSON array of all 13 standard observation tool definitions.
///
/// Each element has `name`, `description`, and `inputSchema` fields matching
/// the MCP `tools/list` response schema. Engines that implement
/// `AutomationApi::list_tools_json` should return this directly.
pub fn standard_tool_definitions() -> Value {
    serde_json::json!([
        {
            "name": "screenshot",
            "description": "Capture current viewport as PNG image",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "width":  { "type": "integer", "description": "Image width in pixels",  "default": 1920 },
                    "height": { "type": "integer", "description": "Image height in pixels", "default": 1080 }
                },
                "required": []
            }
        },
        {
            "name": "scene_graph",
            "description": "List all entities with hierarchy, types, and transforms",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "entity_inspect",
            "description": "Read all components of a specific entity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "integer", "description": "Entity ID to inspect" }
                },
                "required": ["entity_id"]
            }
        },
        {
            "name": "render_stats",
            "description": "Frame time, pass timings, brick pool usage, memory stats",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "log_read",
            "description": "Read recent engine log entries",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "lines": { "type": "integer", "description": "Number of log lines to return", "default": 50 }
                },
                "required": []
            }
        },
        {
            "name": "camera_get",
            "description": "Current camera position, orientation, and FOV",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "brick_pool_stats",
            "description": "Brick pool occupancy, free list size, LRU state",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "spatial_query",
            "description": "Query SDF distance and material at a world position",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "chunk_x": { "type": "integer", "description": "Chunk X coordinate" },
                    "chunk_y": { "type": "integer", "description": "Chunk Y coordinate" },
                    "chunk_z": { "type": "integer", "description": "Chunk Z coordinate" },
                    "local_x": { "type": "number",  "description": "Local X position within chunk" },
                    "local_y": { "type": "number",  "description": "Local Y position within chunk" },
                    "local_z": { "type": "number",  "description": "Local Z position within chunk" }
                },
                "required": ["chunk_x", "chunk_y", "chunk_z", "local_x", "local_y", "local_z"]
            }
        },
        {
            "name": "asset_status",
            "description": "Loading progress, loaded chunks, pending uploads",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "debug_mode",
            "description": "Set shading debug visualization mode. 0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse only, 5=specular only",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": { "type": "integer", "description": "Debug mode (0=normal shading, 1=normals, 2=positions, 3=material IDs, 4=diffuse only, 5=specular only)" }
                },
                "required": ["mode"]
            }
        },
        {
            "name": "camera_set",
            "description": "Set camera position and orientation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "x":     { "type": "number", "description": "X position" },
                    "y":     { "type": "number", "description": "Y position" },
                    "z":     { "type": "number", "description": "Z position" },
                    "yaw":   { "type": "number", "description": "Yaw in degrees",   "default": 0.0 },
                    "pitch": { "type": "number", "description": "Pitch in degrees", "default": 0.0 }
                },
                "required": ["x", "y", "z"]
            }
        },
        {
            "name": "env_set",
            "description": "Set an environment property (atmosphere, fog, clouds, post-processing). Use env_get with property='all' to see available properties.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "property": { "type": "string", "description": "Property path, e.g. 'clouds.enabled', 'atmosphere.sun_intensity', 'post_process.exposure', 'fog.density'" },
                    "value":    { "type": "string", "description": "Value to set (number, boolean, or string)" }
                },
                "required": ["property", "value"]
            }
        },
        {
            "name": "env_get",
            "description": "Get an environment property value. Use property='all' for a summary of all settings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "property": { "type": "string", "description": "Property path (e.g. 'clouds.enabled', 'atmosphere.sun_intensity') or 'all' for summary", "default": "all" }
                },
                "required": []
            }
        }
    ])
}

/// Dispatch a tool call to the corresponding `AutomationApi` method.
///
/// Returns the full `ToolsCallResult` JSON (content array + optional isError flag).
/// Returns `AutomationError::NotImplemented` for unknown tool names.
pub fn dispatch_tool_call(
    api: &dyn AutomationApi,
    name: &str,
    args: Value,
) -> AutomationResult<Value> {
    match name {
        "screenshot" => {
            let width = args.get("width").and_then(|v| v.as_u64()).unwrap_or(1920) as u32;
            let height = args.get("height").and_then(|v| v.as_u64()).unwrap_or(1080) as u32;
            match api.screenshot(width, height) {
                Ok(data) => Ok(tool_image_json(data)),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "scene_graph" => match api.scene_graph() {
            Ok(snap) => Ok(tool_ok_json(serde_json::to_value(snap).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "entity_inspect" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_u64()) {
                Some(id) => id,
                None => return Ok(tool_err_json("entity_id is required")),
            };
            match api.entity_inspect(entity_id) {
                Ok(snap) => Ok(tool_ok_json(serde_json::to_value(snap).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "render_stats" => match api.render_stats() {
            Ok(stats) => Ok(tool_ok_json(serde_json::to_value(stats).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "log_read" => {
            let lines = args.get("lines").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
            match api.read_log(lines) {
                Ok(entries) => Ok(tool_ok_json(serde_json::to_value(entries).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "camera_get" => match api.camera_state() {
            Ok(state) => Ok(tool_ok_json(serde_json::to_value(state).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "brick_pool_stats" => match api.brick_pool_stats() {
            Ok(stats) => Ok(tool_ok_json(serde_json::to_value(stats).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "spatial_query" => {
            let chunk = [
                args.get("chunk_x").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                args.get("chunk_y").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                args.get("chunk_z").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            ];
            let local = [
                args.get("local_x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                args.get("local_y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                args.get("local_z").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            ];
            match api.spatial_query(chunk, local) {
                Ok(result) => Ok(tool_ok_json(serde_json::to_value(result).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "asset_status" => match api.asset_status() {
            Ok(status) => Ok(tool_ok_json(serde_json::to_value(status).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "debug_mode" => {
            let mode = match args.get("mode").and_then(|v| v.as_u64()) {
                Some(m) => m as u32,
                None => return Ok(tool_err_json("mode is required (0-5)")),
            };
            match api.execute_command(&format!("debug_mode {mode}")) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({ "status": "ok", "message": msg }))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "camera_set" => {
            let x = match args.get("x").and_then(|v| v.as_f64()) {
                Some(v) => v as f32,
                None => return Ok(tool_err_json("x is required")),
            };
            let y = match args.get("y").and_then(|v| v.as_f64()) {
                Some(v) => v as f32,
                None => return Ok(tool_err_json("y is required")),
            };
            let z = match args.get("z").and_then(|v| v.as_f64()) {
                Some(v) => v as f32,
                None => return Ok(tool_err_json("z is required")),
            };
            let yaw_deg = args.get("yaw").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let pitch_deg = args.get("pitch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let cmd = format!("camera_set {x} {y} {z} {yaw_deg} {pitch_deg}");
            match api.execute_command(&cmd) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({
                    "status": "ok",
                    "message": msg,
                    "position": [x, y, z],
                    "yaw": yaw_deg,
                    "pitch": pitch_deg,
                }))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "env_set" => {
            let property = match args.get("property").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return Ok(tool_err_json("property is required")),
            };
            let value = match args.get("value") {
                Some(v) => v,
                None => return Ok(tool_err_json("value is required")),
            };
            let value_str = if let Some(s) = value.as_str() {
                s.to_string()
            } else if let Some(b) = value.as_bool() {
                b.to_string()
            } else {
                value.to_string()
            };
            let cmd = format!("env_set {property} {value_str}");
            match api.execute_command(&cmd) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({ "status": "ok", "message": msg }))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "env_get" => {
            let property = args.get("property").and_then(|v| v.as_str()).unwrap_or("all");
            let cmd = format!("env_get {property}");
            match api.execute_command(&cmd) {
                Ok(val) => Ok(tool_ok_json(serde_json::json!({ "property": property, "value": val }))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        _ => Err(AutomationError::NotImplemented("unknown tool")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn register_all_observation_tools() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        assert_eq!(registry.len(), 13);
    }

    #[test]
    fn all_observation_tools_visible_in_debug_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Debug);
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn all_observation_tools_visible_in_editor_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Editor);
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn screenshot_tool_returns_engine_error() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call("screenshot", ToolMode::Editor, &api, serde_json::json!({}));
        // StubAutomationApi returns NotImplemented, which becomes EngineError
        assert!(matches!(result, Err(ToolError::EngineError(_))));
    }

    #[test]
    fn entity_inspect_requires_entity_id() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let api = StubAutomationApi;
        // Missing entity_id
        let result = registry.call("entity_inspect", ToolMode::Editor, &api, serde_json::json!({}));
        assert!(matches!(result, Err(ToolError::InvalidParams(_))));
    }

    #[test]
    fn scene_graph_tool_returns_engine_error() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call("scene_graph", ToolMode::Editor, &api, serde_json::json!({}));
        assert!(matches!(result, Err(ToolError::EngineError(_))));
    }

    #[test]
    fn tool_names_match_architecture() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let expected = [
            "screenshot", "scene_graph", "entity_inspect", "render_stats",
            "log_read", "camera_get", "brick_pool_stats", "spatial_query",
            "asset_status", "debug_mode", "camera_set",
        ];
        for name in &expected {
            assert!(registry.get_tool(name).is_some(), "missing tool: {name}");
        }
    }
}
