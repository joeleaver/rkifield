//! IPC bridge — proxies [`AutomationApi`] calls over a Unix socket to the engine.
//!
//! [`BridgeAutomationApi`] is created by the `connect` tool when an agent attaches
//! to a running engine. Each method sends a JSON-RPC `tools/call` request over the
//! IPC socket and parses the response back into the typed result.

use rkf_core::automation::*;

use crate::ipc::IpcClient;
use crate::protocol::JsonRpcRequest;

/// Automation API implementation that proxies calls over IPC to the engine.
pub struct BridgeAutomationApi {
    socket_path: String,
}

impl BridgeAutomationApi {
    /// Create a new bridge connected to the given Unix socket path.
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    /// Send a `tools/call` request over IPC and return the raw result JSON.
    fn call_tool_raw(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> AutomationResult<serde_json::Value> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.call_tool_raw_async(name, args))
        })
    }

    /// Send a `tools/call` request over IPC and return the parsed JSON result
    /// (extracts text content and parses as JSON).
    fn call_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> AutomationResult<serde_json::Value> {
        let result = self.call_tool_raw(name, args)?;

        // Extract text content and parse as JSON
        let text = result["content"][0]["text"]
            .as_str()
            .ok_or_else(|| AutomationError::EngineError("missing content text".into()))?;

        serde_json::from_str(text)
            .map_err(|e| AutomationError::EngineError(format!("parse IPC result: {e}")))
    }

    async fn call_tool_raw_async(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> AutomationResult<serde_json::Value> {
        let mut client = IpcClient::connect_unix(&self.socket_path)
            .await
            .map_err(|e| AutomationError::EngineError(format!("IPC connect failed: {e}")))?;

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": name,
                "arguments": args
            })),
        };

        let resp = client
            .request(&req)
            .await
            .map_err(|e| AutomationError::EngineError(format!("IPC request failed: {e}")))?;

        if let Some(err) = resp.error {
            return Err(AutomationError::EngineError(err.message));
        }

        let result = resp
            .result
            .ok_or_else(|| AutomationError::EngineError("empty IPC response".into()))?;

        // Check for tool-level error
        if result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let text = result["content"][0]["text"]
                .as_str()
                .unwrap_or("unknown tool error");
            return Err(AutomationError::EngineError(text.to_string()));
        }

        Ok(result)
    }
}

impl AutomationApi for BridgeAutomationApi {
    fn screenshot(&self, width: u32, height: u32) -> AutomationResult<Vec<u8>> {
        let result = self.call_tool_raw(
            "screenshot",
            serde_json::json!({"width": width, "height": height}),
        )?;

        // Screenshot returns an image content block, not text
        let content = &result["content"][0];
        let b64 = content["data"]
            .as_str()
            .ok_or_else(|| AutomationError::EngineError("missing screenshot image data".into()))?;

        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| AutomationError::EngineError(format!("base64 decode: {e}")))
    }

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        let result = self.call_tool("scene_graph", serde_json::json!({}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn entity_inspect(&self, entity_id: u64) -> AutomationResult<EntitySnapshot> {
        let result = self.call_tool(
            "entity_inspect",
            serde_json::json!({"entity_id": entity_id}),
        )?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn render_stats(&self) -> AutomationResult<RenderStats> {
        let result = self.call_tool("render_stats", serde_json::json!({}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn asset_status(&self) -> AutomationResult<AssetStatusReport> {
        let result = self.call_tool("asset_status", serde_json::json!({}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn read_log(&self, lines: usize) -> AutomationResult<Vec<LogEntry>> {
        let result = self.call_tool("log_read", serde_json::json!({"lines": lines}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn camera_state(&self) -> AutomationResult<CameraSnapshot> {
        let result = self.call_tool("camera_get", serde_json::json!({}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats> {
        let result = self.call_tool("brick_pool_stats", serde_json::json!({}))?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    fn spatial_query(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
        let result = self.call_tool(
            "spatial_query",
            serde_json::json!({
                "chunk_x": chunk[0], "chunk_y": chunk[1], "chunk_z": chunk[2],
                "local_x": local[0], "local_y": local[1], "local_z": local[2],
            }),
        )?;
        serde_json::from_value(result).map_err(|e| AutomationError::EngineError(e.to_string()))
    }

    // --- Mutation methods (not supported over bridge) ---

    fn entity_spawn(&self, _def: EntityDef) -> AutomationResult<u64> {
        Err(AutomationError::NotImplemented(
            "entity_spawn (bridge is observation-only)",
        ))
    }

    fn entity_despawn(&self, _entity_id: u64) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "entity_despawn (bridge is observation-only)",
        ))
    }

    fn entity_set_component(
        &self,
        _entity_id: u64,
        _component: ComponentDef,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "entity_set_component (bridge is observation-only)",
        ))
    }

    fn material_set(&self, _id: u16, _material: MaterialDef) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "material_set (bridge is observation-only)",
        ))
    }

    fn brush_apply(&self, _op: serde_json::Value) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "brush_apply (bridge is observation-only)",
        ))
    }

    fn scene_load(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "scene_load (bridge is observation-only)",
        ))
    }

    fn scene_save(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "scene_save (bridge is observation-only)",
        ))
    }

    fn camera_set(
        &self,
        _chunk: [i32; 3],
        _local: [f32; 3],
        _rotation: [f32; 4],
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "camera_set (bridge is observation-only)",
        ))
    }

    fn quality_preset(&self, _preset: QualityPreset) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented(
            "quality_preset (bridge is observation-only)",
        ))
    }

    fn execute_command(&self, command: &str) -> AutomationResult<String> {
        // Route known commands to their MCP tools
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.as_slice() {
            ["debug_mode", mode_str] => {
                let mode: u64 = mode_str.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid debug mode: {mode_str}"))
                })?;
                let result =
                    self.call_tool("debug_mode", serde_json::json!({"mode": mode}))?;
                Ok(result["message"]
                    .as_str()
                    .unwrap_or("ok")
                    .to_string())
            }
            _ => Err(AutomationError::InvalidParameter(format!(
                "unknown command: {command}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_mutation_methods_return_not_implemented() {
        let bridge = BridgeAutomationApi::new("/tmp/nonexistent.sock".to_string());
        assert!(bridge
            .entity_spawn(EntityDef {
                name: "test".into(),
                components: vec![]
            })
            .is_err());
        assert!(bridge.entity_despawn(1).is_err());
        assert!(bridge.scene_load("foo").is_err());
        assert!(bridge.scene_save("foo").is_err());
        assert!(bridge
            .camera_set([0, 0, 0], [0.0; 3], [0.0, 0.0, 0.0, 1.0])
            .is_err());
        assert!(bridge.quality_preset(QualityPreset::High).is_err());
        assert!(bridge.execute_command("test").is_err());
    }
}
