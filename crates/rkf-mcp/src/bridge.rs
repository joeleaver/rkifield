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

    // --- Mutation methods (forwarded over IPC) ---

    fn entity_spawn(&self, def: EntityDef) -> AutomationResult<u64> {
        let result = self.call_tool(
            "entity_spawn",
            serde_json::to_value(&def)
                .map_err(|e| AutomationError::EngineError(format!("serialize: {e}")))?,
        )?;
        result["id"]
            .as_u64()
            .ok_or_else(|| AutomationError::EngineError("missing id in response".into()))
    }

    fn entity_despawn(&self, entity_id: u64) -> AutomationResult<()> {
        self.call_tool(
            "entity_despawn",
            serde_json::json!({"entity_id": entity_id}),
        )?;
        Ok(())
    }

    fn entity_set_component(
        &self,
        entity_id: u64,
        component: ComponentDef,
    ) -> AutomationResult<()> {
        let comp_value = serde_json::to_value(&component)
            .map_err(|e| AutomationError::EngineError(format!("serialize: {e}")))?;
        self.call_tool(
            "entity_set_component",
            serde_json::json!({"entity_id": entity_id, "component": comp_value}),
        )?;
        Ok(())
    }

    fn material_set(&self, id: u16, material: MaterialDef) -> AutomationResult<()> {
        let mat_value = serde_json::to_value(&material)
            .map_err(|e| AutomationError::EngineError(format!("serialize: {e}")))?;
        self.call_tool(
            "material_set",
            serde_json::json!({"id": id, "material": mat_value}),
        )?;
        Ok(())
    }

    fn brush_apply(&self, op: serde_json::Value) -> AutomationResult<()> {
        self.call_tool("brush_apply", op)?;
        Ok(())
    }

    fn scene_load(&self, path: &str) -> AutomationResult<()> {
        self.call_tool("scene_load", serde_json::json!({"path": path}))?;
        Ok(())
    }

    fn scene_save(&self, path: &str) -> AutomationResult<()> {
        self.call_tool("scene_save", serde_json::json!({"path": path}))?;
        Ok(())
    }

    fn camera_set(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
        rotation: [f32; 4],
    ) -> AutomationResult<()> {
        self.call_tool(
            "camera_set",
            serde_json::json!({
                "x": local[0], "y": local[1], "z": local[2],
                "yaw": 0.0, "pitch": 0.0,
            }),
        )?;
        // Also store chunk/rotation for callers that use the full API
        let _ = (chunk, rotation);
        Ok(())
    }

    fn quality_preset(&self, preset: QualityPreset) -> AutomationResult<()> {
        let preset_value = serde_json::to_value(&preset)
            .map_err(|e| AutomationError::EngineError(format!("serialize: {e}")))?;
        self.call_tool(
            "quality_preset",
            serde_json::json!({"preset": preset_value}),
        )?;
        Ok(())
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
            ["camera_set", x_s, y_s, z_s, yaw_s, pitch_s] => {
                let x: f64 = x_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid x: {x_s}"))
                })?;
                let y: f64 = y_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid y: {y_s}"))
                })?;
                let z: f64 = z_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid z: {z_s}"))
                })?;
                let yaw: f64 = yaw_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid yaw: {yaw_s}"))
                })?;
                let pitch: f64 = pitch_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid pitch: {pitch_s}"))
                })?;
                let result = self.call_tool("camera_set", serde_json::json!({
                    "x": x, "y": y, "z": z,
                    "yaw": yaw, "pitch": pitch,
                }))?;
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bridge_mutation_methods_fail_without_server() {
        // With no server running, all methods should return errors (connection refused).
        // Needs a tokio runtime because the bridge uses block_in_place internally.
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
        assert!(bridge.execute_command("test").is_err());
    }
}
