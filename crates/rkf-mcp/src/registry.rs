//! Tool discovery registry — self-describing tools with dynamic registration.
//!
//! Tools register themselves with full metadata (name, description, parameters,
//! return type, mode). The MCP server generates `tools/list` responses dynamically
//! from this registry. Adding a new tool = implement ToolHandler + register.

use rkf_core::automation::AutomationApi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Tool category for organizational grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// Read-only observation of engine state
    Observation,
    /// State-modifying mutation operations
    Mutation,
    /// Debug/diagnostic tools
    Debug,
}

/// Which server mode(s) a tool is available in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolMode {
    /// Available in editor mode only
    Editor,
    /// Available in debug mode only
    Debug,
    /// Available in both modes
    Both,
}

/// Parameter type for JSON Schema compatibility.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    String,
    Number,
    Integer,
    Boolean,
    Object,
    Array,
    /// Enum with specific allowed values
    Enum(Vec<std::string::String>),
}

/// Definition of a tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDef {
    /// Parameter name
    pub name: std::string::String,
    /// Human/agent-readable description
    pub description: std::string::String,
    /// Parameter type
    pub param_type: ParamType,
    /// Whether this parameter is required
    pub required: bool,
    /// Default value (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// Description of what a tool returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnTypeDef {
    /// Human-readable description of the return value
    pub description: std::string::String,
    /// The JSON type of the return value
    pub return_type: ParamType,
}

/// Complete definition of an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool name (e.g., "screenshot")
    pub name: std::string::String,
    /// Human/agent-readable description
    pub description: std::string::String,
    /// Organizational category
    pub category: ToolCategory,
    /// Parameter definitions
    pub parameters: Vec<ParameterDef>,
    /// Return type description
    pub return_type: ReturnTypeDef,
    /// Which mode(s) this tool is available in
    pub mode: ToolMode,
}

/// Response from a tool handler.
///
/// Most tools return JSON data (wrapped as a text content block in MCP).
/// Tools that produce binary data (e.g., screenshots) return typed variants
/// that the protocol layer converts to the appropriate MCP content block.
pub enum ToolResponse {
    /// Regular JSON result — serialized as a text content block.
    Json(serde_json::Value),
    /// Binary image data — returned as an MCP image content block.
    Image {
        /// Raw image bytes (e.g., PNG-encoded).
        data: Vec<u8>,
        /// MIME type (e.g., "image/png").
        mime_type: String,
    },
}

impl From<serde_json::Value> for ToolResponse {
    fn from(v: serde_json::Value) -> Self {
        ToolResponse::Json(v)
    }
}

/// Trait for tool handler implementations.
///
/// Each tool implements this trait to handle invocations.
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with the given parameters.
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: serde_json::Value,
    ) -> Result<ToolResponse, ToolError>;
}

/// Errors that can occur during tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The requested tool was not found
    #[error("tool not found: {0}")]
    NotFound(std::string::String),
    /// Invalid parameters provided
    #[error("invalid parameters: {0}")]
    InvalidParams(std::string::String),
    /// The tool encountered an engine error
    #[error("engine error: {0}")]
    EngineError(std::string::String),
    /// Serialization/deserialization error
    #[error("serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

/// A registered tool: definition + handler.
struct RegisteredTool {
    definition: ToolDefinition,
    handler: Arc<dyn ToolHandler>,
}

/// Central registry for MCP tools.
///
/// Tools self-register at startup. The MCP server queries this registry
/// to generate `tools/list` responses and dispatch `tools/call` requests.
pub struct ToolRegistry {
    tools: HashMap<std::string::String, RegisteredTool>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool with its definition and handler.
    ///
    /// # Panics
    /// Panics if a tool with the same name is already registered.
    pub fn register(&mut self, definition: ToolDefinition, handler: Arc<dyn ToolHandler>) {
        let name = definition.name.clone();
        if self.tools.contains_key(&name) {
            panic!("tool already registered: {name}");
        }
        self.tools
            .insert(name, RegisteredTool { definition, handler });
    }

    /// List all tool definitions available in the given mode.
    ///
    /// - Editor mode: returns tools with mode Editor or Both
    /// - Debug mode: returns tools with mode Debug or Both
    pub fn list_tools(&self, mode: ToolMode) -> Vec<&ToolDefinition> {
        self.tools
            .values()
            .filter(|t| match mode {
                ToolMode::Editor => {
                    t.definition.mode == ToolMode::Editor
                        || t.definition.mode == ToolMode::Both
                }
                ToolMode::Debug => {
                    t.definition.mode == ToolMode::Debug
                        || t.definition.mode == ToolMode::Both
                }
                ToolMode::Both => true,
            })
            .map(|t| &t.definition)
            .collect()
    }

    /// Get a specific tool's definition.
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name).map(|t| &t.definition)
    }

    /// Call a tool by name, checking mode access.
    ///
    /// Returns an error if the tool doesn't exist or isn't available in the given mode.
    pub fn call(
        &self,
        name: &str,
        mode: ToolMode,
        api: &dyn AutomationApi,
        params: serde_json::Value,
    ) -> Result<ToolResponse, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?;

        // Check mode access
        let allowed = match mode {
            ToolMode::Editor => {
                tool.definition.mode == ToolMode::Editor
                    || tool.definition.mode == ToolMode::Both
            }
            ToolMode::Debug => {
                tool.definition.mode == ToolMode::Debug
                    || tool.definition.mode == ToolMode::Both
            }
            ToolMode::Both => true,
        };

        if !allowed {
            return Err(ToolError::NotFound(format!(
                "tool '{}' is not available in {:?} mode",
                name, mode
            )));
        }

        tool.handler.call(api, params)
    }

    /// Total number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    /// A simple test tool handler that echoes params back.
    struct EchoHandler;

    impl ToolHandler for EchoHandler {
        fn call(
            &self,
            _api: &dyn AutomationApi,
            params: serde_json::Value,
        ) -> Result<ToolResponse, ToolError> {
            Ok(serde_json::json!({ "echo": params }).into())
        }
    }

    fn make_observation_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Test observation tool: {name}"),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "test result".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        }
    }

    fn make_mutation_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Test mutation tool: {name}"),
            category: ToolCategory::Mutation,
            parameters: vec![ParameterDef {
                name: "value".to_string(),
                description: "test param".to_string(),
                param_type: ParamType::String,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "mutation result".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        }
    }

    #[test]
    fn empty_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.list_tools(ToolMode::Editor).is_empty());
        assert!(registry.list_tools(ToolMode::Debug).is_empty());
    }

    #[test]
    fn register_and_list() {
        let mut registry = ToolRegistry::new();
        registry.register(make_observation_tool("screenshot"), Arc::new(EchoHandler));
        registry.register(make_mutation_tool("entity_spawn"), Arc::new(EchoHandler));

        assert_eq!(registry.len(), 2);

        // Editor mode sees both
        let editor_tools = registry.list_tools(ToolMode::Editor);
        assert_eq!(editor_tools.len(), 2);

        // Debug mode sees only observation (mode=Both)
        let debug_tools = registry.list_tools(ToolMode::Debug);
        assert_eq!(debug_tools.len(), 1);
        assert_eq!(debug_tools[0].name, "screenshot");
    }

    #[test]
    fn call_tool() {
        let mut registry = ToolRegistry::new();
        registry.register(make_observation_tool("screenshot"), Arc::new(EchoHandler));

        let api = StubAutomationApi;
        let params = serde_json::json!({"width": 1920});
        let result = registry.call("screenshot", ToolMode::Editor, &api, params.clone());
        assert!(result.is_ok());
        match result.unwrap() {
            ToolResponse::Json(value) => {
                assert_eq!(value, serde_json::json!({"echo": {"width": 1920}}));
            }
            _ => panic!("expected ToolResponse::Json"),
        }
    }

    #[test]
    fn call_nonexistent_tool() {
        let registry = ToolRegistry::new();
        let api = StubAutomationApi;
        let result =
            registry.call("nonexistent", ToolMode::Editor, &api, serde_json::Value::Null);
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[test]
    fn mode_filtering_blocks_mutation_in_debug() {
        let mut registry = ToolRegistry::new();
        registry.register(make_mutation_tool("entity_spawn"), Arc::new(EchoHandler));

        let api = StubAutomationApi;
        // Editor mode allows it
        let result =
            registry.call("entity_spawn", ToolMode::Editor, &api, serde_json::Value::Null);
        assert!(result.is_ok());

        // Debug mode blocks it
        let result =
            registry.call("entity_spawn", ToolMode::Debug, &api, serde_json::Value::Null);
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[test]
    fn get_tool_definition() {
        let mut registry = ToolRegistry::new();
        registry.register(make_observation_tool("screenshot"), Arc::new(EchoHandler));

        let def = registry.get_tool("screenshot");
        assert!(def.is_some());
        let def = def.unwrap();
        assert_eq!(def.name, "screenshot");
        assert_eq!(def.category, ToolCategory::Observation);
        assert_eq!(def.mode, ToolMode::Both);

        assert!(registry.get_tool("nonexistent").is_none());
    }

    #[test]
    #[should_panic(expected = "tool already registered")]
    fn duplicate_registration_panics() {
        let mut registry = ToolRegistry::new();
        registry.register(make_observation_tool("screenshot"), Arc::new(EchoHandler));
        registry.register(make_observation_tool("screenshot"), Arc::new(EchoHandler));
    }

    #[test]
    fn parameter_def_serialization() {
        let param = ParameterDef {
            name: "width".to_string(),
            description: "Image width in pixels".to_string(),
            param_type: ParamType::Integer,
            required: true,
            default: Some(serde_json::json!(1920)),
        };
        let json = serde_json::to_string(&param).unwrap();
        let roundtrip: ParameterDef = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.name, "width");
        assert_eq!(roundtrip.param_type, ParamType::Integer);
        assert!(roundtrip.required);
    }

    #[test]
    fn tool_definition_serialization() {
        let def = make_observation_tool("screenshot");
        let json = serde_json::to_string(&def).unwrap();
        let roundtrip: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.name, "screenshot");
        assert_eq!(roundtrip.mode, ToolMode::Both);
    }
}
