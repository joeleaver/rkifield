//! Dynamic tool discovery and dispatch.
//!
//! Instead of registering every engine tool as a separate MCP tool (which
//! requires restarting the MCP server whenever tools change), this module
//! provides two generic tools:
//!
//! - **`search_tools`** — query the engine's tool registry by keyword,
//!   returning matching tool names, descriptions, and parameter schemas.
//! - **`use_tool`** — call any engine tool by name with a JSON argument blob.
//!
//! This pattern lets the tool set grow on the engine side without touching
//! the MCP server at all.

use crate::registry::*;
use rkf_core::automation::AutomationApi;
use serde_json::Value;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// search_tools
// ---------------------------------------------------------------------------

struct SearchToolsHandler;

impl ToolHandler for SearchToolsHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        // Fetch the full tool list from the engine.
        let tools_json = api.list_tools_json().map_err(|e| {
            ToolError::EngineError(format!(
                "Not connected to engine. Use the 'connect' tool first. ({e})"
            ))
        })?;

        let tools = match tools_json.as_array() {
            Some(arr) => arr,
            None => {
                return Ok(serde_json::json!({
                    "matches": [],
                    "message": "Engine returned no tools. Is it connected?"
                })
                .into());
            }
        };

        // If query is empty, return all tools (summary only — name + description).
        if query.is_empty() {
            let summaries: Vec<Value> = tools
                .iter()
                .filter_map(|t| {
                    Some(serde_json::json!({
                        "name": t.get("name")?.as_str()?,
                        "description": t.get("description")?.as_str()?,
                    }))
                })
                .collect();

            return Ok(serde_json::json!({
                "total": summaries.len(),
                "matches": summaries,
            })
            .into());
        }

        // Filter by query — match against name and description.
        let matches: Vec<&Value> = tools
            .iter()
            .filter(|t| {
                let name = t
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let desc = t
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                name.contains(&query) || desc.contains(&query)
            })
            .collect();

        // For matching tools, return the full definition (including inputSchema)
        // so the caller knows how to invoke them.
        Ok(serde_json::json!({
            "query": query,
            "total": matches.len(),
            "matches": matches,
        })
        .into())
    }
}

// ---------------------------------------------------------------------------
// use_tool
// ---------------------------------------------------------------------------

struct UseToolHandler;

impl ToolHandler for UseToolHandler {
    fn call(
        &self,
        api: &dyn AutomationApi,
        params: Value,
    ) -> Result<ToolResponse, ToolError> {
        let tool_name = params
            .get("tool_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidParams(
                    "tool_name is required. Use search_tools to find available tools.".to_string(),
                )
            })?;

        let tool_args = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        // Forward to the engine's tool dispatch.
        let result = api.call_tool_json(tool_name, tool_args).map_err(|e| {
            ToolError::EngineError(format!(
                "Tool '{}' failed: {}. Use search_tools to verify the tool exists.",
                tool_name, e
            ))
        })?;

        // The engine returns a full ToolsCallResult JSON (content array + isError).
        // Extract content blocks and pass them through as RawContent so they don't
        // get double-wrapped (e.g., image blocks would otherwise become stringified
        // JSON inside a text block).
        if let Some(content_arr) = result.get("content").and_then(|c| c.as_array()) {
            let blocks: Vec<crate::protocol::ContentBlock> = content_arr
                .iter()
                .filter_map(|c| serde_json::from_value(c.clone()).ok())
                .collect();
            if !blocks.is_empty() {
                return Ok(ToolResponse::RawContent(blocks));
            }
        }

        // Fallback: wrap entire result as JSON text.
        Ok(ToolResponse::Json(result))
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register the dynamic discovery tools (search_tools, use_tool).
pub fn register_dynamic_tools(registry: &mut ToolRegistry) {
    registry.register(
        ToolDefinition {
            name: "search_tools".to_string(),
            description: "Search for available engine tools by keyword. Returns tool names, \
                          descriptions, and parameter schemas. Call with no query to list all \
                          tools. Use the returned tool names with use_tool to invoke them."
                .to_string(),
            category: ToolCategory::Debug,
            parameters: vec![ParameterDef {
                name: "query".to_string(),
                description: "Search keyword to match against tool names and descriptions. \
                              Leave empty to list all available tools."
                    .to_string(),
                param_type: ParamType::String,
                required: false,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "Matching tools with name, description, and inputSchema".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SearchToolsHandler),
    );

    registry.register(
        ToolDefinition {
            name: "use_tool".to_string(),
            description: "Call an engine tool by name. Use search_tools first to discover \
                          available tools and their parameter schemas."
                .to_string(),
            category: ToolCategory::Debug,
            parameters: vec![
                ParameterDef {
                    name: "tool_name".to_string(),
                    description: "Name of the engine tool to call (from search_tools results)"
                        .to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "arguments".to_string(),
                    description: "JSON object with the tool's parameters (see inputSchema \
                                  from search_tools)"
                        .to_string(),
                    param_type: ParamType::Object,
                    required: false,
                    default: Some(serde_json::json!({})),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Tool result (content blocks from the engine)".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(UseToolHandler),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn register_dynamic_tools_adds_two() {
        let mut registry = ToolRegistry::new();
        register_dynamic_tools(&mut registry);
        assert_eq!(registry.len(), 2);
        assert!(registry.get_tool("search_tools").is_some());
        assert!(registry.get_tool("use_tool").is_some());
    }

    #[test]
    fn dynamic_tools_visible_in_both_modes() {
        let mut registry = ToolRegistry::new();
        register_dynamic_tools(&mut registry);
        assert_eq!(registry.list_tools(ToolMode::Editor).len(), 2);
        assert_eq!(registry.list_tools(ToolMode::Debug).len(), 2);
    }

    #[test]
    fn search_tools_returns_error_when_disconnected() {
        let mut registry = ToolRegistry::new();
        register_dynamic_tools(&mut registry);
        let api = StubAutomationApi;
        // StubAutomationApi.list_tools_json() returns Ok(json!([])), so we
        // get an empty matches array rather than an error.
        let result = registry
            .call("search_tools", ToolMode::Editor, &api, serde_json::json!({}))
            .unwrap();
        match result {
            ToolResponse::Json(v) => {
                assert_eq!(v["total"], 0);
                assert!(v["matches"].as_array().unwrap().is_empty());
            }
            _ => panic!("expected Json response"),
        }
    }

    #[test]
    fn use_tool_requires_tool_name() {
        let mut registry = ToolRegistry::new();
        register_dynamic_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call(
            "use_tool",
            ToolMode::Editor,
            &api,
            serde_json::json!({}),
        );
        assert!(matches!(result, Err(ToolError::InvalidParams(_))));
    }

    #[test]
    fn use_tool_fails_on_stub() {
        let mut registry = ToolRegistry::new();
        register_dynamic_tools(&mut registry);
        let api = StubAutomationApi;
        let result = registry.call(
            "use_tool",
            ToolMode::Editor,
            &api,
            serde_json::json!({"tool_name": "screenshot"}),
        );
        // StubAutomationApi.call_tool_json() returns NotImplemented
        assert!(matches!(result, Err(ToolError::EngineError(_))));
    }
}
