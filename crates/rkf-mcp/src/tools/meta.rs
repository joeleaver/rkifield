//! Meta tools — server management (connect, disconnect, status).
//!
//! These tools manage the MCP server's connection to the engine,
//! not the engine itself. They work independently of the AutomationApi.

use crate::protocol::ApiSlot;
use crate::registry::*;
use rkf_core::automation::{AutomationApi, StubAutomationApi};
use std::sync::Arc;

// --- Connect tool ---

struct ConnectHandler {
    api_slot: ApiSlot,
}

impl ToolHandler for ConnectHandler {
    fn call(
        &self,
        _api: &dyn AutomationApi,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        let socket_path = params
            .get("socket_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if socket_path.is_empty() {
            // Auto-discover: look for /tmp/rkifield-*.sock
            let mut found = Vec::new();
            if let Ok(entries) = std::fs::read_dir("/tmp") {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("rkifield-") && name.ends_with(".sock") {
                        found.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }

            if found.is_empty() {
                return Ok(serde_json::json!({
                    "status": "no_engine_found",
                    "message": "No running RKIField engine found. Start rkf-testbed, rkf-editor, or rkf-game first, then call connect again.",
                    "hint": "You can also specify socket_path directly if the engine uses a custom path."
                }));
            }

            return Ok(serde_json::json!({
                "status": "engines_found",
                "sockets": found,
                "message": "Found engine socket(s). Call connect again with socket_path to attach.",
            }));
        }

        // Check if socket exists
        if !std::path::Path::new(&socket_path).exists() {
            return Ok(serde_json::json!({
                "status": "socket_not_found",
                "socket_path": socket_path,
                "message": format!("Socket file not found: {socket_path}"),
            }));
        }

        // TODO: When engine IPC bridge is implemented, this will:
        // 1. Connect to the engine via Unix socket
        // 2. Create a BridgeAutomationApi that proxies calls over IPC
        // 3. Swap it into self.api_slot
        // For now, report that connection infrastructure exists but engine bridge isn't built yet.
        let _api_slot = &self.api_slot;

        Ok(serde_json::json!({
            "status": "not_yet_implemented",
            "socket_path": socket_path,
            "message": "Socket found but engine IPC bridge is not yet implemented. Engine tools will return stub data. The bridge will be built as engine features come online.",
        }))
    }
}

// --- Disconnect tool ---

struct DisconnectHandler {
    api_slot: ApiSlot,
}

impl ToolHandler for DisconnectHandler {
    fn call(
        &self,
        _api: &dyn AutomationApi,
        _params: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        // Swap back to stub API
        let mut slot = self.api_slot.write().map_err(|e| {
            ToolError::EngineError(format!("failed to acquire write lock: {e}"))
        })?;
        *slot = Arc::new(StubAutomationApi);

        Ok(serde_json::json!({
            "status": "disconnected",
            "message": "Disconnected from engine. All tools now return stub data.",
        }))
    }
}

// --- Status tool ---

struct StatusHandler {
    api_slot: ApiSlot,
}

impl ToolHandler for StatusHandler {
    fn call(
        &self,
        _api: &dyn AutomationApi,
        _params: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        // Check if we're using the stub or a real connection
        // For now, we can only be in stub mode since engine bridge isn't built
        let _api = self.api_slot.read().map_err(|e| {
            ToolError::EngineError(format!("failed to acquire read lock: {e}"))
        })?;

        // Auto-discover available engines
        let mut available_sockets = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/tmp") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("rkifield-") && name.ends_with(".sock") {
                    available_sockets.push(entry.path().to_string_lossy().to_string());
                }
            }
        }

        Ok(serde_json::json!({
            "connected": false,
            "api_mode": "stub",
            "message": "Running with stub API. Engine tools return placeholder data. Use the 'connect' tool to attach to a running engine.",
            "available_engines": available_sockets,
            "server_version": env!("CARGO_PKG_VERSION"),
        }))
    }
}

/// Register meta tools (connect, disconnect, status) with the registry.
pub fn register_meta_tools(registry: &mut ToolRegistry, api_slot: ApiSlot) {
    registry.register(
        ToolDefinition {
            name: "connect".to_string(),
            description: "Connect to a running RKIField engine via IPC socket. Call without arguments to auto-discover, or provide socket_path.".to_string(),
            category: ToolCategory::Debug,
            parameters: vec![ParameterDef {
                name: "socket_path".to_string(),
                description: "Path to engine's Unix socket (e.g., /tmp/rkifield-12345.sock). Leave empty to auto-discover.".to_string(),
                param_type: ParamType::String,
                required: false,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "Connection status".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ConnectHandler {
            api_slot: Arc::clone(&api_slot),
        }),
    );

    registry.register(
        ToolDefinition {
            name: "disconnect".to_string(),
            description: "Disconnect from the engine, reverting to stub API".to_string(),
            category: ToolCategory::Debug,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Disconnection status".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(DisconnectHandler {
            api_slot: Arc::clone(&api_slot),
        }),
    );

    registry.register(
        ToolDefinition {
            name: "status".to_string(),
            description: "Check MCP server connection status and discover available engines".to_string(),
            category: ToolCategory::Debug,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Server status including connection state and available engines".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(StatusHandler {
            api_slot: Arc::clone(&api_slot),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;

    fn make_api_slot() -> ApiSlot {
        Arc::new(RwLock::new(Arc::new(StubAutomationApi) as Arc<dyn AutomationApi>))
    }

    #[test]
    fn register_all_meta_tools() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, api_slot);
        assert_eq!(registry.len(), 3);
        assert!(registry.get_tool("connect").is_some());
        assert!(registry.get_tool("disconnect").is_some());
        assert!(registry.get_tool("status").is_some());
    }

    #[test]
    fn meta_tools_visible_in_both_modes() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, api_slot);
        assert_eq!(registry.list_tools(ToolMode::Editor).len(), 3);
        assert_eq!(registry.list_tools(ToolMode::Debug).len(), 3);
    }

    #[test]
    fn status_tool_shows_stub_mode() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, Arc::clone(&api_slot));
        // Clone the inner Arc so the read lock is released before tool dispatch
        let api: Arc<dyn AutomationApi> = api_slot.read().unwrap().clone();
        let result = registry.call("status", ToolMode::Editor, &*api, serde_json::json!({}));
        assert!(result.is_ok());
        let value = result.unwrap();
        assert_eq!(value["connected"], false);
        assert_eq!(value["api_mode"], "stub");
    }

    #[test]
    fn connect_tool_without_args_discovers() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, Arc::clone(&api_slot));
        let api: Arc<dyn AutomationApi> = api_slot.read().unwrap().clone();
        let result = registry.call("connect", ToolMode::Editor, &*api, serde_json::json!({}));
        assert!(result.is_ok());
        let value = result.unwrap();
        // Will be either "no_engine_found" or "engines_found" depending on machine state
        let status = value["status"].as_str().unwrap();
        assert!(status == "no_engine_found" || status == "engines_found");
    }

    #[test]
    fn connect_tool_with_missing_socket() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, Arc::clone(&api_slot));
        let api: Arc<dyn AutomationApi> = api_slot.read().unwrap().clone();
        let result = registry.call(
            "connect",
            ToolMode::Editor,
            &*api,
            serde_json::json!({"socket_path": "/tmp/nonexistent.sock"}),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["status"], "socket_not_found");
    }

    #[test]
    fn disconnect_tool_resets_to_stub() {
        let api_slot = make_api_slot();
        let mut registry = ToolRegistry::new();
        register_meta_tools(&mut registry, Arc::clone(&api_slot));
        // Clone the inner Arc and drop the lock before calling disconnect,
        // which needs to acquire a write lock on the same slot.
        let api: Arc<dyn AutomationApi> = api_slot.read().unwrap().clone();
        let result = registry.call("disconnect", ToolMode::Editor, &*api, serde_json::json!({}));
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["status"], "disconnected");
    }
}
