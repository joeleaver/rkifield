//! Behavior system MCP tool handlers (Phase 14.1–14.4).
//!
//! Tools for inspecting and manipulating the ECS behavior system:
//! components, systems, blueprints, game state, and play control.

pub mod dispatch;

use crate::registry::*;
use rkf_core::automation::AutomationApi;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Handler structs
// ---------------------------------------------------------------------------

// --- Component tools (14.1) ---

struct ComponentListHandler;

impl ToolHandler for ComponentListHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let components = api.component_list();
        Ok(serde_json::to_value(components)
            .map_err(ToolError::from)?
            .into())
    }
}

struct ComponentGetHandler;

impl ToolHandler for ComponentGetHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".into()))?;
        let component_name = params
            .get("component_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("component_name is required".into()))?;

        match api.component_get(entity_id, component_name) {
            Ok(fields) => Ok(serde_json::to_value(fields)
                .map_err(ToolError::from)?
                .into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct ComponentSetHandler;

impl ToolHandler for ComponentSetHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".into()))?;
        let component_name = params
            .get("component_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("component_name is required".into()))?;
        let fields = parse_string_map(&params, "fields")?;

        match api.component_set(entity_id, component_name, fields) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct ComponentAddHandler;

impl ToolHandler for ComponentAddHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".into()))?;
        let component_name = params
            .get("component_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("component_name is required".into()))?;
        let fields = parse_string_map(&params, "fields")?;

        match api.component_add(entity_id, component_name, fields) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct ComponentRemoveHandler;

impl ToolHandler for ComponentRemoveHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let entity_id = params
            .get("entity_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("entity_id is required".into()))?;
        let component_name = params
            .get("component_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("component_name is required".into()))?;

        match api.component_remove(entity_id, component_name) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- System + Blueprint tools (14.2) ---

struct SystemListHandler;

impl ToolHandler for SystemListHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let systems = api.system_list();
        Ok(serde_json::to_value(systems)
            .map_err(ToolError::from)?
            .into())
    }
}

struct BlueprintListHandler;

impl ToolHandler for BlueprintListHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let blueprints = api.blueprint_list();
        Ok(serde_json::to_value(blueprints)
            .map_err(ToolError::from)?
            .into())
    }
}

struct BlueprintSpawnHandler;

impl ToolHandler for BlueprintSpawnHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("name is required".into()))?;
        let position = match params.get("position").and_then(|v| v.as_array()) {
            Some(arr) if arr.len() >= 3 => [
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
            ],
            _ => [0.0, 0.0, 0.0],
        };

        match api.blueprint_spawn(name, position) {
            Ok(entity_id) => {
                Ok(serde_json::json!({"status": "ok", "entity_id": entity_id}).into())
            }
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

// --- State tools (14.3) ---

struct StateGetHandler;

impl ToolHandler for StateGetHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let key = params
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("key is required".into()))?;

        match api.state_get(key) {
            Ok(value) => Ok(serde_json::json!({"key": key, "value": value}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct StateSetHandler;

impl ToolHandler for StateSetHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let key = params
            .get("key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("key is required".into()))?;
        let value = params
            .get("value")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidParams("value is required".into()))?;
        let value_type = params
            .get("value_type")
            .and_then(|v| v.as_str())
            .unwrap_or("string");

        match api.state_set(key, value, value_type) {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct StateListHandler;

impl ToolHandler for StateListHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let prefix = params
            .get("prefix")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let keys = api.state_list(prefix);
        Ok(serde_json::json!({"prefix": prefix, "keys": keys}).into())
    }
}

// --- Play control tools (14.4) ---

struct PlayStartHandler;

impl ToolHandler for PlayStartHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        match api.play_start() {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct PlayStopHandler;

impl ToolHandler for PlayStopHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        match api.play_stop() {
            Ok(()) => Ok(serde_json::json!({"status": "ok"}).into()),
            Err(e) => Err(ToolError::EngineError(e)),
        }
    }
}

struct PlayStateHandler;

impl ToolHandler for PlayStateHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        _params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let state = api.play_state();
        Ok(serde_json::json!({"state": state}).into())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a `HashMap<String, String>` from a JSON object parameter.
fn parse_string_map(params: &Value, key: &str) -> Result<HashMap<String, String>, ToolError> {
    let obj = params
        .get(key)
        .and_then(|v| v.as_object())
        .ok_or_else(|| ToolError::InvalidParams(format!("{key} is required as a JSON object")))?;

    let mut map = HashMap::new();
    for (k, v) in obj {
        let val = if let Some(s) = v.as_str() {
            s.to_string()
        } else {
            v.to_string()
        };
        map.insert(k.clone(), val);
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all behavior system tools with the registry.
pub fn register_behavior_tools(registry: &mut ToolRegistry) {
    // --- Component tools (14.1) ---
    registry.register(
        ToolDefinition {
            name: "component_list".to_string(),
            description: "List all registered component types and their fields".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Array of ComponentInfo with name and fields".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ComponentListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "component_get".to_string(),
            description: "Get all field values of a component on an entity".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "entity_id".to_string(),
                    description: "Entity ID".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "component_name".to_string(),
                    description: "Component type name (e.g. \"Health\")".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Map of field name to string value".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ComponentGetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "component_set".to_string(),
            description: "Set field values on an existing component of an entity".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "entity_id".to_string(),
                    description: "Entity ID".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "component_name".to_string(),
                    description: "Component type name".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "fields".to_string(),
                    description: "Object mapping field names to string values".to_string(),
                    param_type: ParamType::Object,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of update".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(ComponentSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "component_add".to_string(),
            description: "Add a new component to an entity with initial field values".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "entity_id".to_string(),
                    description: "Entity ID".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "component_name".to_string(),
                    description: "Component type name".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "fields".to_string(),
                    description: "Object mapping field names to string values".to_string(),
                    param_type: ParamType::Object,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of component addition".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(ComponentAddHandler),
    );

    registry.register(
        ToolDefinition {
            name: "component_remove".to_string(),
            description: "Remove a component from an entity by name".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "entity_id".to_string(),
                    description: "Entity ID".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "component_name".to_string(),
                    description: "Component type name to remove".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of component removal".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(ComponentRemoveHandler),
    );

    // --- System + Blueprint tools (14.2) ---
    registry.register(
        ToolDefinition {
            name: "system_list".to_string(),
            description: "List all registered behavior systems with phase and fault status"
                .to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Array of SystemInfo with name, phase, and faulted flag".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SystemListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "blueprint_list".to_string(),
            description: "List all available blueprints (prefabs) and their components".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Array of BlueprintInfo with name and component_names".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(BlueprintListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "blueprint_spawn".to_string(),
            description: "Spawn an entity from a named blueprint at a position".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "name".to_string(),
                    description: "Blueprint name".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "position".to_string(),
                    description: "Spawn position as [x, y, z]".to_string(),
                    param_type: ParamType::Array,
                    required: false,
                    default: Some(serde_json::json!([0.0, 0.0, 0.0])),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Object with entity_id of spawned entity".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(BlueprintSpawnHandler),
    );

    // --- State tools (14.3) ---
    registry.register(
        ToolDefinition {
            name: "state_get".to_string(),
            description: "Get a value from the game state store by key".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "key".to_string(),
                description: "State key to look up".to_string(),
                param_type: ParamType::String,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "Object with key and value (null if not found)".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(StateGetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "state_set".to_string(),
            description: "Set a value in the game state store".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "key".to_string(),
                    description: "State key".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "value".to_string(),
                    description: "Value as string".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "value_type".to_string(),
                    description: "Type hint (f32, i32, string, bool)".to_string(),
                    param_type: ParamType::String,
                    required: false,
                    default: Some(serde_json::json!("string")),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of state update".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(StateSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "state_list".to_string(),
            description: "List all keys in the game state store matching a prefix".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "prefix".to_string(),
                description: "Key prefix filter (empty = all keys)".to_string(),
                param_type: ParamType::String,
                required: false,
                default: Some(serde_json::json!("")),
            }],
            return_type: ReturnTypeDef {
                description: "Object with prefix and matching keys array".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(StateListHandler),
    );

    // --- Play control tools (14.4) ---
    registry.register(
        ToolDefinition {
            name: "play_start".to_string(),
            description: "Start play mode (begin running behavior systems)".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Confirmation of play start".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(PlayStartHandler),
    );

    registry.register(
        ToolDefinition {
            name: "play_stop".to_string(),
            description: "Stop play mode (pause all behavior systems, revert to edit state)"
                .to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Confirmation of play stop".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(PlayStopHandler),
    );

    registry.register(
        ToolDefinition {
            name: "play_state".to_string(),
            description: "Get current play state (stopped, playing, paused)".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Object with state string".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(PlayStateHandler),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn register_all_behavior_tools() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        assert_eq!(registry.len(), 14);
    }

    #[test]
    fn behavior_tools_visible_in_debug_mode() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Debug);
        // Debug mode sees "Both" tools: component_list, component_get,
        // system_list, blueprint_list, state_get, state_list, play_state = 7
        assert_eq!(tools.len(), 7);
    }

    #[test]
    fn all_behavior_tools_visible_in_editor_mode() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Editor);
        assert_eq!(tools.len(), 14);
    }

    #[test]
    fn component_list_returns_empty_on_stub() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call(
            "component_list",
            ToolMode::Editor,
            &api,
            serde_json::json!({}),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn component_get_requires_params() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call(
            "component_get",
            ToolMode::Editor,
            &api,
            serde_json::json!({}),
        );
        assert!(matches!(result, Err(ToolError::InvalidParams(_))));
    }

    #[test]
    fn play_state_returns_stopped_on_stub() {
        let mut registry = ToolRegistry::new();
        register_behavior_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call(
            "play_state",
            ToolMode::Editor,
            &api,
            serde_json::json!({}),
        );
        match result.unwrap() {
            ToolResponse::Json(value) => {
                assert_eq!(value["state"], "stopped");
            }
            _ => panic!("expected Json response"),
        }
    }
}
