//! Shared dispatch for behavior tool calls — used by EditorAutomationApi & TestbedAutomationApi.

use rkf_core::automation::AutomationApi;
use serde_json::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
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

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Return a JSON array of behavior system tool definitions for `standard_tool_definitions`.
pub fn behavior_tool_definitions() -> Value {
    serde_json::json!([
        {
            "name": "component_list",
            "description": "List all registered component types and their fields",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "component_get",
            "description": "Get all field values of a component on an entity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity UUID" },
                    "component_name": { "type": "string", "description": "Component type name" }
                },
                "required": ["entity_id", "component_name"]
            }
        },
        {
            "name": "component_set",
            "description": "Set field values on an existing component of an entity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity UUID" },
                    "component_name": { "type": "string", "description": "Component type name" },
                    "fields": { "type": "object", "description": "Field name to string value map" }
                },
                "required": ["entity_id", "component_name", "fields"]
            }
        },
        {
            "name": "component_add",
            "description": "Add a new component to an entity with initial field values",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity UUID" },
                    "component_name": { "type": "string", "description": "Component type name" },
                    "fields": { "type": "object", "description": "Field name to string value map" }
                },
                "required": ["entity_id", "component_name", "fields"]
            }
        },
        {
            "name": "component_remove",
            "description": "Remove a component from an entity by name",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity UUID" },
                    "component_name": { "type": "string", "description": "Component type name" }
                },
                "required": ["entity_id", "component_name"]
            }
        },
        {
            "name": "system_list",
            "description": "List all registered behavior systems with phase and fault status",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "blueprint_list",
            "description": "List all available blueprints (prefabs) and their components",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "blueprint_spawn",
            "description": "Spawn an entity from a named blueprint at a position",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Blueprint name" },
                    "position": { "type": "array", "description": "Spawn position [x, y, z]", "items": { "type": "number" } }
                },
                "required": ["name"]
            }
        },
        {
            "name": "state_get",
            "description": "Get a value from the game state store by key",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "State key to look up" }
                },
                "required": ["key"]
            }
        },
        {
            "name": "state_set",
            "description": "Set a value in the game state store",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "State key" },
                    "value": { "type": "string", "description": "Value as string" },
                    "value_type": { "type": "string", "description": "Type hint (f32, i32, string, bool)" }
                },
                "required": ["key", "value"]
            }
        },
        {
            "name": "state_list",
            "description": "List all keys in the game state store matching a prefix",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prefix": { "type": "string", "description": "Key prefix filter (empty = all)" }
                }
            }
        },
        {
            "name": "play_start",
            "description": "Start play mode (begin running behavior systems)",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "play_stop",
            "description": "Stop play mode (pause all behavior systems, revert to edit state)",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "play_state",
            "description": "Get current play state (stopped, playing, paused)",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Dispatch a behavior tool call to the corresponding `AutomationApi` method.
///
/// Returns `None` if the tool name is not a behavior tool (caller should try
/// other dispatch functions). Returns `Some(result)` on match.
pub fn dispatch_behavior_tool_call(
    api: &dyn AutomationApi,
    name: &str,
    args: Value,
) -> Option<rkf_core::automation::AutomationResult<Value>> {
    let result = match name {
        "component_list" => {
            let components = api.component_list();
            Ok(tool_ok_json(serde_json::to_value(components).unwrap()))
        }
        "component_get" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => return Some(Ok(tool_err_json("entity_id is required"))),
            };
            let component_name = match args.get("component_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(Ok(tool_err_json("component_name is required"))),
            };
            match api.component_get(entity_id, component_name) {
                Ok(fields) => Ok(tool_ok_json(serde_json::to_value(fields).unwrap())),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "component_set" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => return Some(Ok(tool_err_json("entity_id is required"))),
            };
            let component_name = match args.get("component_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(Ok(tool_err_json("component_name is required"))),
            };
            let fields = match args
                .get("fields")
                .and_then(|v| v.as_object())
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| {
                            (
                                k.clone(),
                                v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                            )
                        })
                        .collect::<HashMap<String, String>>()
                }) {
                Some(f) => f,
                None => return Some(Ok(tool_err_json("fields is required as a JSON object"))),
            };
            match api.component_set(entity_id, component_name, fields) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "component_add" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => return Some(Ok(tool_err_json("entity_id is required"))),
            };
            let component_name = match args.get("component_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(Ok(tool_err_json("component_name is required"))),
            };
            let fields = match args
                .get("fields")
                .and_then(|v| v.as_object())
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| {
                            (
                                k.clone(),
                                v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                            )
                        })
                        .collect::<HashMap<String, String>>()
                }) {
                Some(f) => f,
                None => return Some(Ok(tool_err_json("fields is required as a JSON object"))),
            };
            match api.component_add(entity_id, component_name, fields) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "component_remove" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => return Some(Ok(tool_err_json("entity_id is required"))),
            };
            let component_name = match args.get("component_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(Ok(tool_err_json("component_name is required"))),
            };
            match api.component_remove(entity_id, component_name) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "system_list" => {
            let systems = api.system_list();
            Ok(tool_ok_json(serde_json::to_value(systems).unwrap()))
        }
        "blueprint_list" => {
            let blueprints = api.blueprint_list();
            Ok(tool_ok_json(serde_json::to_value(blueprints).unwrap()))
        }
        "blueprint_spawn" => {
            let name = match args.get("name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Some(Ok(tool_err_json("name is required"))),
            };
            let position = match args.get("position").and_then(|v| v.as_array()) {
                Some(arr) if arr.len() >= 3 => [
                    arr[0].as_f64().unwrap_or(0.0) as f32,
                    arr[1].as_f64().unwrap_or(0.0) as f32,
                    arr[2].as_f64().unwrap_or(0.0) as f32,
                ],
                _ => [0.0, 0.0, 0.0],
            };
            match api.blueprint_spawn(name, position) {
                Ok(entity_id) => Ok(tool_ok_json(
                    serde_json::json!({"status": "ok", "entity_id": entity_id}),
                )),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "state_get" => {
            let key = match args.get("key").and_then(|v| v.as_str()) {
                Some(k) => k,
                None => return Some(Ok(tool_err_json("key is required"))),
            };
            match api.state_get(key) {
                Ok(value) => Ok(tool_ok_json(serde_json::json!({"key": key, "value": value}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "state_set" => {
            let key = match args.get("key").and_then(|v| v.as_str()) {
                Some(k) => k,
                None => return Some(Ok(tool_err_json("key is required"))),
            };
            let value = match args.get("value").and_then(|v| v.as_str()) {
                Some(v) => v,
                None => return Some(Ok(tool_err_json("value is required"))),
            };
            let value_type = args
                .get("value_type")
                .and_then(|v| v.as_str())
                .unwrap_or("string");
            match api.state_set(key, value, value_type) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "state_list" => {
            let prefix = args
                .get("prefix")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let keys = api.state_list(prefix);
            Ok(tool_ok_json(
                serde_json::json!({"prefix": prefix, "keys": keys}),
            ))
        }
        "play_start" => match api.play_start() {
            Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
            Err(e) => Ok(tool_err_json(&e)),
        },
        "play_stop" => match api.play_stop() {
            Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
            Err(e) => Ok(tool_err_json(&e)),
        },
        "play_state" => {
            let state = api.play_state();
            Ok(tool_ok_json(serde_json::json!({"state": state})))
        }
        _ => return None,
    };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn dispatch_behavior_tool_call_matches() {
        let api = StubAutomationApi;
        // Known behavior tool
        let result = dispatch_behavior_tool_call(&api, "component_list", serde_json::json!({}));
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());

        // Unknown tool returns None
        let result = dispatch_behavior_tool_call(&api, "screenshot", serde_json::json!({}));
        assert!(result.is_none());
    }

    #[test]
    fn behavior_tool_definitions_has_14_entries() {
        let defs = behavior_tool_definitions();
        assert_eq!(defs.as_array().unwrap().len(), 14);
    }
}
