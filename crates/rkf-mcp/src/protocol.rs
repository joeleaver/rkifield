//! MCP protocol layer — JSON-RPC 2.0 message handling.
//!
//! Implements the subset of MCP needed for tool discovery and execution:
//! - `initialize` — handshake with client capabilities
//! - `tools/list` — return registered tools with full metadata
//! - `tools/call` — dispatch tool invocation to registry

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

// Standard JSON-RPC error codes
pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: Option<Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }

    /// Create an error response with additional data.
    pub fn error_with_data(
        id: Option<Value>,
        code: i64,
        message: impl Into<String>,
        data: Value,
    ) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: Some(data),
            }),
        }
    }
}

/// MCP server info returned during initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// MCP server capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub tools: ToolsCapability,
}

/// Tools capability declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCapability {
    /// Whether the tool list can change during the session.
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// MCP initialize response result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// MCP tool definition as returned in tools/list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP tools/list response result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsListResult {
    pub tools: Vec<McpToolDef>,
}

/// MCP tools/call request params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// MCP content block in tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

/// MCP tools/call response result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCallResult {
    pub content: Vec<ContentBlock>,
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Convert a [`ToolDefinition`] from the registry into MCP's tool format.
pub fn tool_def_to_mcp(def: &crate::registry::ToolDefinition) -> McpToolDef {
    use crate::registry::ParamType;

    // Build JSON Schema from parameter definitions
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for param in &def.parameters {
        let type_str = match &param.param_type {
            ParamType::String => "string",
            ParamType::Number => "number",
            ParamType::Integer => "integer",
            ParamType::Boolean => "boolean",
            ParamType::Object => "object",
            ParamType::Array => "array",
            ParamType::Enum(_) => "string",
        };

        let mut prop = serde_json::json!({
            "type": type_str,
            "description": param.description,
        });

        if let ParamType::Enum(values) = &param.param_type {
            prop["enum"] = serde_json::json!(values);
        }

        if let Some(default) = &param.default {
            prop["default"] = default.clone();
        }

        properties.insert(param.name.clone(), prop);

        if param.required {
            required.push(serde_json::Value::String(param.name.clone()));
        }
    }

    let input_schema = serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    });

    McpToolDef {
        name: def.name.clone(),
        description: def.description.clone(),
        input_schema,
    }
}

/// Handle an incoming JSON-RPC request and produce a response.
///
/// This is the core MCP dispatch: it handles `initialize`, `tools/list`,
/// `tools/call`, and `notifications/initialized`.
pub fn handle_request(
    request: &JsonRpcRequest,
    registry: &crate::registry::ToolRegistry,
    mode: crate::registry::ToolMode,
    api: &dyn rkf_core::automation::AutomationApi,
) -> Option<JsonRpcResponse> {
    match request.method.as_str() {
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    tools: ToolsCapability {
                        list_changed: false,
                    },
                },
                server_info: ServerInfo {
                    name: "rkf-mcp".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
            };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "notifications/initialized" => {
            // Notification — no response
            None
        }

        "tools/list" => {
            let tool_defs: Vec<McpToolDef> = registry
                .list_tools(mode)
                .into_iter()
                .map(tool_def_to_mcp)
                .collect();

            let result = ToolsListResult { tools: tool_defs };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "tools/call" => {
            let params = match &request.params {
                Some(p) => p.clone(),
                None => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        "missing params for tools/call",
                    ));
                }
            };

            let call_params: ToolsCallParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        format!("invalid tools/call params: {e}"),
                    ));
                }
            };

            let tool_args = call_params.arguments.unwrap_or(Value::Null);

            match registry.call(&call_params.name, mode, api, tool_args) {
                Ok(response) => {
                    let content = tool_response_to_content(response);
                    let result = ToolsCallResult {
                        content,
                        is_error: None,
                    };
                    Some(JsonRpcResponse::success(
                        request.id.clone(),
                        serde_json::to_value(result).unwrap(),
                    ))
                }
                Err(e) => {
                    let result = ToolsCallResult {
                        content: vec![ContentBlock::Text {
                            text: format!("Error: {e}"),
                        }],
                        is_error: Some(true),
                    };
                    Some(JsonRpcResponse::success(
                        request.id.clone(),
                        serde_json::to_value(result).unwrap(),
                    ))
                }
            }
        }

        _ => Some(JsonRpcResponse::error(
            request.id.clone(),
            METHOD_NOT_FOUND,
            format!("method not found: {}", request.method),
        )),
    }
}

/// Convert a [`ToolResponse`] into MCP content blocks.
fn tool_response_to_content(response: crate::registry::ToolResponse) -> Vec<ContentBlock> {
    use base64::Engine;
    use crate::registry::ToolResponse;

    match response {
        ToolResponse::Json(value) => {
            let text = serde_json::to_string_pretty(&value).unwrap_or_default();
            vec![ContentBlock::Text { text }]
        }
        ToolResponse::Image { data, mime_type } => {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
            vec![ContentBlock::Image {
                data: b64,
                mime_type,
            }]
        }
    }
}

/// Type alias for the swappable API slot used by the stdio server.
///
/// The inner `Arc<dyn AutomationApi>` allows callers to clone the API handle
/// and release the `RwLock` before dispatching tool calls. This prevents
/// deadlocks when meta tools (connect, disconnect) need to acquire a write
/// lock on the same slot from within their handler.
pub type ApiSlot = std::sync::Arc<std::sync::RwLock<std::sync::Arc<dyn rkf_core::automation::AutomationApi>>>;

/// Handle an incoming JSON-RPC request using a lock-guarded API slot.
///
/// Unlike [`handle_request`], this variant accepts an [`ApiSlot`] and clones
/// the inner `Arc<dyn AutomationApi>` so the lock is released before tool
/// dispatch. This prevents deadlocks when meta tools need write access.
///
/// Use this in the stdio server main loop instead of [`handle_request`].
pub fn handle_request_locked(
    request: &JsonRpcRequest,
    registry: &crate::registry::ToolRegistry,
    mode: crate::registry::ToolMode,
    api_slot: &ApiSlot,
) -> Option<JsonRpcResponse> {
    match request.method.as_str() {
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    tools: ToolsCapability {
                        list_changed: false,
                    },
                },
                server_info: ServerInfo {
                    name: "rkf-mcp".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
            };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "notifications/initialized" => None,

        "tools/list" => {
            // Local meta tools (connect, disconnect, status)
            let local_defs: Vec<McpToolDef> = registry
                .list_tools(mode)
                .into_iter()
                .map(tool_def_to_mcp)
                .collect();

            // Remote engine tools (fetched via IPC)
            let api: std::sync::Arc<dyn rkf_core::automation::AutomationApi> =
                api_slot.read().unwrap().clone();
            let remote_defs: Vec<McpToolDef> = api
                .list_tools_json()
                .ok()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();

            // Merge: local tools take priority over remote tools with the same name
            let local_names: HashSet<String> =
                local_defs.iter().map(|t| t.name.clone()).collect();
            let mut all_tools = local_defs;
            all_tools.extend(
                remote_defs
                    .into_iter()
                    .filter(|t| !local_names.contains(&t.name)),
            );

            let result = ToolsListResult { tools: all_tools };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "tools/call" => {
            let params = match &request.params {
                Some(p) => p.clone(),
                None => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        "missing params for tools/call",
                    ));
                }
            };

            let call_params: ToolsCallParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        format!("invalid tools/call params: {e}"),
                    ));
                }
            };

            let tool_args = call_params.arguments.unwrap_or(Value::Null);

            // Clone the inner Arc so we can drop the read lock before dispatch.
            // This allows meta tool handlers to acquire a write lock on api_slot
            // without deadlocking.
            let api: std::sync::Arc<dyn rkf_core::automation::AutomationApi> =
                api_slot.read().unwrap().clone();

            // Try local registry first (meta tools: connect, disconnect, status)
            match registry.call(&call_params.name, mode, &*api, tool_args.clone()) {
                Ok(response) => {
                    let content = tool_response_to_content(response);
                    let result = ToolsCallResult {
                        content,
                        is_error: None,
                    };
                    Some(JsonRpcResponse::success(
                        request.id.clone(),
                        serde_json::to_value(result).unwrap(),
                    ))
                }
                Err(crate::registry::ToolError::NotFound(_)) => {
                    // Not a local tool — forward to engine via IPC
                    match api.call_tool_json(&call_params.name, tool_args) {
                        Ok(result) => {
                            Some(JsonRpcResponse::success(request.id.clone(), result))
                        }
                        Err(e) => {
                            let result = ToolsCallResult {
                                content: vec![ContentBlock::Text {
                                    text: format!("Error: {e}"),
                                }],
                                is_error: Some(true),
                            };
                            Some(JsonRpcResponse::success(
                                request.id.clone(),
                                serde_json::to_value(result).unwrap(),
                            ))
                        }
                    }
                }
                Err(e) => {
                    let result = ToolsCallResult {
                        content: vec![ContentBlock::Text {
                            text: format!("Error: {e}"),
                        }],
                        is_error: Some(true),
                    };
                    Some(JsonRpcResponse::success(
                        request.id.clone(),
                        serde_json::to_value(result).unwrap(),
                    ))
                }
            }
        }

        _ => Some(JsonRpcResponse::error(
            request.id.clone(),
            METHOD_NOT_FOUND,
            format!("method not found: {}", request.method),
        )),
    }
}

/// Parse a raw JSON string into a [`JsonRpcRequest`].
pub fn parse_request(input: &str) -> Result<JsonRpcRequest, JsonRpcResponse> {
    serde_json::from_str::<JsonRpcRequest>(input).map_err(|e| {
        JsonRpcResponse::error(None, PARSE_ERROR, format!("parse error: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::*;
    use rkf_core::automation::StubAutomationApi;
    use std::sync::Arc;

    struct EchoHandler;
    impl ToolHandler for EchoHandler {
        fn call(
            &self,
            _api: &dyn rkf_core::automation::AutomationApi,
            params: Value,
        ) -> Result<ToolResponse, ToolError> {
            Ok(serde_json::json!({ "echo": params }).into())
        }
    }

    fn test_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(
            ToolDefinition {
                name: "screenshot".to_string(),
                description: "Take a screenshot".to_string(),
                category: ToolCategory::Observation,
                parameters: vec![ParameterDef {
                    name: "width".to_string(),
                    description: "Width in pixels".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1920)),
                }],
                return_type: ReturnTypeDef {
                    description: "PNG image data".to_string(),
                    return_type: ParamType::String,
                },
                mode: ToolMode::Both,
            },
            Arc::new(EchoHandler),
        );
        reg.register(
            ToolDefinition {
                name: "entity_spawn".to_string(),
                description: "Spawn an entity".to_string(),
                category: ToolCategory::Mutation,
                parameters: vec![],
                return_type: ReturnTypeDef {
                    description: "Entity ID".to_string(),
                    return_type: ParamType::Integer,
                },
                mode: ToolMode::Editor,
            },
            Arc::new(EchoHandler),
        );
        reg
    }

    #[test]
    fn parse_valid_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = parse_request(json);
        assert!(req.is_ok());
        let req = req.unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(Value::Number(1.into())));
    }

    #[test]
    fn parse_invalid_json() {
        let result = parse_request("not json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.error.as_ref().unwrap().code, PARSE_ERROR);
    }

    #[test]
    fn handle_initialize() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: "initialize".to_string(),
            params: Some(serde_json::json!({})),
        };

        let resp = handle_request(&req, &registry, ToolMode::Editor, &api);
        assert!(resp.is_some());
        let resp = resp.unwrap();
        assert!(resp.error.is_none());
        let result: InitializeResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.server_info.name, "rkf-mcp");
        assert_eq!(result.protocol_version, "2024-11-05");
    }

    #[test]
    fn handle_notifications_initialized() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: "notifications/initialized".to_string(),
            params: None,
        };
        let resp = handle_request(&req, &registry, ToolMode::Editor, &api);
        assert!(resp.is_none()); // Notifications don't get responses
    }

    #[test]
    fn handle_tools_list_editor() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(2.into())),
            method: "tools/list".to_string(),
            params: None,
        };

        let resp = handle_request(&req, &registry, ToolMode::Editor, &api).unwrap();
        assert!(resp.error.is_none());
        let result: ToolsListResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.tools.len(), 2); // Both tools visible in editor mode
    }

    #[test]
    fn handle_tools_list_debug_filters_mutations() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(3.into())),
            method: "tools/list".to_string(),
            params: None,
        };

        let resp = handle_request(&req, &registry, ToolMode::Debug, &api).unwrap();
        assert!(resp.error.is_none());
        let result: ToolsListResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.tools.len(), 1); // Only screenshot (Both mode)
        assert_eq!(result.tools[0].name, "screenshot");
    }

    #[test]
    fn handle_tools_call() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(4.into())),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": "screenshot",
                "arguments": {"width": 800}
            })),
        };

        let resp = handle_request(&req, &registry, ToolMode::Editor, &api).unwrap();
        assert!(resp.error.is_none());
        let result: ToolsCallResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert!(result.is_error.is_none());
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn handle_tools_call_nonexistent() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(5.into())),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({ "name": "nonexistent" })),
        };

        let resp = handle_request(&req, &registry, ToolMode::Editor, &api).unwrap();
        // Tool errors are returned as successful JSON-RPC responses with isError=true
        assert!(resp.error.is_none());
        let result: ToolsCallResult =
            serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn handle_unknown_method() {
        let registry = test_registry();
        let api = StubAutomationApi;
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(6.into())),
            method: "unknown/method".to_string(),
            params: None,
        };

        let resp = handle_request(&req, &registry, ToolMode::Editor, &api).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.as_ref().unwrap().code, METHOD_NOT_FOUND);
    }

    #[test]
    fn tool_def_to_mcp_generates_schema() {
        let def = ToolDefinition {
            name: "test".to_string(),
            description: "test tool".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "width".to_string(),
                    description: "width".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "format".to_string(),
                    description: "output format".to_string(),
                    param_type: ParamType::Enum(vec!["png".into(), "jpeg".into()]),
                    required: false,
                    default: Some(serde_json::json!("png")),
                },
            ],
            return_type: ReturnTypeDef {
                description: "result".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        };

        let mcp = tool_def_to_mcp(&def);
        assert_eq!(mcp.name, "test");
        let schema = &mcp.input_schema;
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["width"]["type"], "integer");
        assert_eq!(
            schema["properties"]["format"]["enum"],
            serde_json::json!(["png", "jpeg"])
        );
        assert_eq!(schema["required"], serde_json::json!(["width"]));
    }

    #[test]
    fn json_rpc_response_serialization() {
        let resp = JsonRpcResponse::success(
            Some(Value::Number(1.into())),
            serde_json::json!({"ok": true}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"ok\":true"));
        assert!(!json.contains("\"error\"")); // error field should be skipped

        let err_resp = JsonRpcResponse::error(
            Some(Value::Number(2.into())),
            METHOD_NOT_FOUND,
            "not found",
        );
        let json = serde_json::to_string(&err_resp).unwrap();
        assert!(json.contains("-32601"));
        assert!(!json.contains("\"result\"")); // result field should be skipped
    }
}
