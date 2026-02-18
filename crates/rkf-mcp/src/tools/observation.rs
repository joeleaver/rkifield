//! Observation tool stubs — read-only tools available in both Editor and Debug modes.
//!
//! Each tool calls the corresponding `AutomationApi` method and returns the result
//! as JSON. Tools return placeholder data until engine features are built.

use crate::registry::*;
use rkf_core::automation::AutomationApi;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn register_all_observation_tools() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        assert_eq!(registry.len(), 9);
    }

    #[test]
    fn all_observation_tools_visible_in_debug_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Debug);
        assert_eq!(tools.len(), 9);
    }

    #[test]
    fn all_observation_tools_visible_in_editor_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Editor);
        assert_eq!(tools.len(), 9);
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
            "asset_status",
        ];
        for name in &expected {
            assert!(registry.get_tool(name).is_some(), "missing tool: {name}");
        }
    }
}
