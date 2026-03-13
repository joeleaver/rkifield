//! Shared dispatch functions for observation tools.
//!
//! These functions allow any `AutomationApi` implementor to serve all
//! observation tools without needing a `ToolRegistry`.

use rkf_core::automation::{AutomationApi, AutomationError, AutomationResult};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON response helpers
// ---------------------------------------------------------------------------

/// Wrap a successful JSON value as an MCP `ToolsCallResult` text content block.
pub(crate) fn tool_ok_json(value: Value) -> Value {
    let text = serde_json::to_string_pretty(&value).unwrap_or_default();
    serde_json::json!({
        "content": [{ "type": "text", "text": text }]
    })
}

/// Wrap an error message as an MCP `ToolsCallResult` with `isError: true`.
pub(crate) fn tool_err_json(msg: &str) -> Value {
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

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Return a JSON array of all standard observation tool definitions.
///
/// Each element has `name`, `description`, and `inputSchema` fields matching
/// the MCP `tools/list` response schema. Engines that implement
/// `AutomationApi::list_tools_json` should return this directly.
pub fn standard_tool_definitions() -> Value {
    let mut defs = serde_json::json!([
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
            "name": "screenshot_window",
            "description": "Capture full editor window (UI + viewport composited) as PNG image",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
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
                    "entity_id": { "type": "string", "description": "Entity UUID" }
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
            "description": "Set shading debug visualization mode. 0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular, 6=GI, 7=SDF distance, 8=brick boundaries",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": { "type": "integer", "description": "Debug mode (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular, 6=GI, 7=SDF distance, 8=brick boundaries)" }
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
        },
        // --- Node tree tools ---
        {
            "name": "node_find",
            "description": "Find a node by name within an object's SDF tree and return its properties",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "SDF object ID" },
                    "node_name": { "type": "string", "description": "Node name to find" }
                },
                "required": ["object_id", "node_name"]
            }
        },
        {
            "name": "node_add_child",
            "description": "Add a child SDF primitive node to a named parent node within an object",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "SDF object ID" },
                    "parent_node": { "type": "string", "description": "Parent node name" },
                    "child_primitive": { "type": "string", "description": "Primitive type: sphere, box, capsule, torus, cylinder, plane" },
                    "params": { "type": "array", "description": "Shape parameters", "default": [] },
                    "name": { "type": "string", "description": "Name for the new child node" },
                    "material_id": { "type": "integer", "description": "Material table index", "default": 0 }
                },
                "required": ["object_id", "parent_node", "child_primitive", "name"]
            }
        },
        {
            "name": "node_remove",
            "description": "Remove a named node from an object's SDF tree",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "SDF object ID" },
                    "node_name": { "type": "string", "description": "Node name to remove" }
                },
                "required": ["object_id", "node_name"]
            }
        },
        // --- Multi-scene tools ---
        {
            "name": "scene_create",
            "description": "Create a new empty scene and return its index",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Scene name" }
                },
                "required": ["name"]
            }
        },
        {
            "name": "scene_list",
            "description": "List all scenes with names, active state, and persistence flags",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "scene_set_active",
            "description": "Set the active scene by index",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": { "type": "integer", "description": "Scene index" }
                },
                "required": ["index"]
            }
        },
        {
            "name": "scene_set_persistent",
            "description": "Mark a scene as persistent (survives scene swaps) or not",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "index": { "type": "integer", "description": "Scene index" },
                    "persistent": { "type": "boolean", "description": "Whether the scene is persistent" }
                },
                "required": ["index", "persistent"]
            }
        },
        {
            "name": "scene_swap",
            "description": "Unload all non-persistent scenes, keeping persistent ones",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        // --- Camera entity tools ---
        {
            "name": "camera_spawn",
            "description": "Spawn a camera entity at a position with orientation and FOV",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "label": { "type": "string", "description": "Camera display name" },
                    "x": { "type": "number", "description": "X position", "default": 0.0 },
                    "y": { "type": "number", "description": "Y position", "default": 0.0 },
                    "z": { "type": "number", "description": "Z position", "default": 0.0 },
                    "yaw": { "type": "number", "description": "Yaw in degrees", "default": 0.0 },
                    "pitch": { "type": "number", "description": "Pitch in degrees", "default": 0.0 },
                    "fov": { "type": "number", "description": "Vertical FOV in degrees", "default": 60.0 }
                },
                "required": ["label"]
            }
        },
        {
            "name": "camera_list",
            "description": "List all camera entities with positions, orientations, and FOV",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "camera_snap_to",
            "description": "Snap the rendering viewport camera to a camera entity's position and orientation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity_id": { "type": "string", "description": "Entity UUID" }
                },
                "required": ["entity_id"]
            }
        },
        // --- Diagnostic tools ---
        {
            "name": "voxel_slice",
            "description": "Sample a 2D XZ slice of raw SDF distances from an object's voxel data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "Scene object ID (from scene_graph)" },
                    "y_coord": { "type": "number", "description": "Object-local Y coordinate for the slice", "default": 0.0 }
                },
                "required": ["object_id"]
            }
        },
        {
            "name": "sculpt_apply",
            "description": "Apply a single sculpt brush hit to an object. Creates an undo entry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "Scene object ID" },
                    "position": { "type": "array", "description": "World-space [x, y, z]" },
                    "mode": { "type": "string", "description": "add/subtract/smooth", "default": "add" },
                    "radius": { "type": "number", "description": "Brush radius", "default": 0.5 },
                    "strength": { "type": "number", "description": "Brush strength 0-1", "default": 0.5 },
                    "material_id": { "type": "integer", "description": "Material index", "default": 1 }
                },
                "required": ["object_id", "position"]
            }
        },
        {
            "name": "object_shape",
            "description": "Return a compact brick-level 3D shape overview with ASCII y-slices",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object_id": { "type": "integer", "description": "Scene object ID" }
                },
                "required": ["object_id"]
            }
        },
        {
            "name": "material_list",
            "description": "List all materials in the library with slot, name, category, albedo, roughness, metallic, is_emissive",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        },
        {
            "name": "material_get",
            "description": "Get full properties of a material at a given slot index",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "slot": { "type": "integer", "description": "Material table slot index (0-65535)" }
                },
                "required": ["slot"]
            }
        },
        {
            "name": "shader_list",
            "description": "List available shading models (name, id, built_in)",
            "inputSchema": { "type": "object", "properties": {}, "required": [] }
        }
    ]);

    // Append behavior system tool definitions
    if let Some(arr) = defs.as_array_mut() {
        if let Some(behavior_arr) = super::super::behavior::dispatch::behavior_tool_definitions().as_array().cloned()
        {
            arr.extend(behavior_arr);
        }
    }

    defs
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

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
        "screenshot_window" => match api.screenshot_window() {
            Ok(data) => Ok(tool_image_json(data)),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "scene_graph" => match api.scene_graph() {
            Ok(snap) => Ok(tool_ok_json(serde_json::to_value(snap).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "entity_inspect" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
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
        // --- Node tree tools ---
        "node_find" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            let node_name = match args.get("node_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Ok(tool_err_json("node_name is required")),
            };
            match api.node_find(object_id, node_name) {
                Ok(json_str) => {
                    let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                    Ok(tool_ok_json(val))
                }
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "node_add_child" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            let parent_node = match args.get("parent_node").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Ok(tool_err_json("parent_node is required")),
            };
            let child_primitive = match args.get("child_primitive").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return Ok(tool_err_json("child_primitive is required")),
            };
            let p: Vec<f32> = args.get("params").and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                .unwrap_or_default();
            let name = match args.get("name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Ok(tool_err_json("name is required")),
            };
            let material_id = args.get("material_id").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
            match api.node_add_child(object_id, parent_node, child_primitive, &p, name, material_id) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "node_remove" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            let node_name = match args.get("node_name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Ok(tool_err_json("node_name is required")),
            };
            match api.node_remove(object_id, node_name) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        // --- Multi-scene tools ---
        "scene_create" => {
            let name = match args.get("name").and_then(|v| v.as_str()) {
                Some(n) => n,
                None => return Ok(tool_err_json("name is required")),
            };
            match api.scene_create(name) {
                Ok(index) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "index": index}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "scene_list" => match api.scene_list() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(tool_ok_json(val))
            }
            Err(e) => Ok(tool_err_json(&e)),
        },
        "scene_set_active" => {
            let index = match args.get("index").and_then(|v| v.as_u64()) {
                Some(i) => i as usize,
                None => return Ok(tool_err_json("index is required")),
            };
            match api.scene_set_active(index) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "scene_set_persistent" => {
            let index = match args.get("index").and_then(|v| v.as_u64()) {
                Some(i) => i as usize,
                None => return Ok(tool_err_json("index is required")),
            };
            let persistent = match args.get("persistent").and_then(|v| v.as_bool()) {
                Some(p) => p,
                None => return Ok(tool_err_json("persistent is required")),
            };
            match api.scene_set_persistent(index, persistent) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "scene_swap" => match api.scene_swap() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(tool_ok_json(val))
            }
            Err(e) => Ok(tool_err_json(&e)),
        },
        // --- Camera entity tools ---
        "camera_spawn" => {
            let label = match args.get("label").and_then(|v| v.as_str()) {
                Some(l) => l,
                None => return Ok(tool_err_json("label is required")),
            };
            let position = [
                args.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                args.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                args.get("z").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            ];
            let yaw = args.get("yaw").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let pitch = args.get("pitch").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let fov = args.get("fov").and_then(|v| v.as_f64()).unwrap_or(60.0) as f32;
            match api.camera_spawn(label, position, yaw, pitch, fov) {
                Ok(entity_id) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "entity_id": entity_id}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "camera_list" => match api.camera_list() {
            Ok(json_str) => {
                let val: Value = serde_json::from_str(&json_str).unwrap_or(Value::String(json_str));
                Ok(tool_ok_json(val))
            }
            Err(e) => Ok(tool_err_json(&e)),
        },
        "camera_snap_to" => {
            let entity_id = match args.get("entity_id").and_then(|v| v.as_str()) {
                Some(id) => id,
                None => return Ok(tool_err_json("entity_id is required")),
            };
            match api.camera_snap_to(entity_id) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        // --- Diagnostic tools ---
        "voxel_slice" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            let y_coord = args.get("y_coord").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            match api.voxel_slice(object_id, y_coord) {
                Ok(result) => Ok(tool_ok_json(serde_json::to_value(result).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "sculpt_apply" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            let position = match args.get("position").and_then(|v| v.as_array()) {
                Some(arr) if arr.len() >= 3 => [
                    arr[0].as_f64().unwrap_or(0.0) as f32,
                    arr[1].as_f64().unwrap_or(0.0) as f32,
                    arr[2].as_f64().unwrap_or(0.0) as f32,
                ],
                _ => return Ok(tool_err_json("position is required as [x, y, z]")),
            };
            let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("add");
            let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let strength = args.get("strength").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let material_id = args.get("material_id").and_then(|v| v.as_u64()).unwrap_or(1) as u16;
            match api.sculpt_apply(object_id, position, mode, radius, strength, material_id) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "object_shape" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            match api.object_shape(object_id) {
                Ok(result) => Ok(tool_ok_json(serde_json::to_value(result).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "material_list" => match api.material_list() {
            Ok(list) => Ok(tool_ok_json(serde_json::to_value(list).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "material_get" => {
            let slot = match args.get("slot").and_then(|v| v.as_u64()) {
                Some(s) => s as u16,
                None => return Ok(tool_err_json("slot is required")),
            };
            match api.material_get(slot) {
                Ok(snapshot) => Ok(tool_ok_json(serde_json::to_value(snapshot).unwrap())),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "shader_list" => match api.shader_list() {
            Ok(list) => Ok(tool_ok_json(serde_json::to_value(list).unwrap())),
            Err(e) => Ok(tool_err_json(&e.to_string())),
        },
        "object_spawn" => {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("Object");
            let primitive_type = match args.get("primitive_type").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return Ok(tool_err_json("primitive_type is required (sphere, box, capsule, etc.)")),
            };
            let params: Vec<f32> = args.get("params").and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                .unwrap_or_default();
            let position = match args.get("position").and_then(|v| v.as_array()) {
                Some(arr) if arr.len() >= 3 => [
                    arr[0].as_f64().unwrap_or(0.0) as f32,
                    arr[1].as_f64().unwrap_or(0.0) as f32,
                    arr[2].as_f64().unwrap_or(0.0) as f32,
                ],
                _ => [0.0, 0.0, 0.0],
            };
            let material_id = args.get("material_id").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
            match api.object_spawn(name, primitive_type, &params, position, material_id) {
                Ok(id) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "object_id": id}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "object_despawn" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            match api.object_despawn(object_id) {
                Ok(()) => Ok(tool_ok_json(serde_json::json!({"status": "ok"}))),
                Err(e) => Ok(tool_err_json(&e)),
            }
        }
        "voxelize" => {
            let object_id = match args.get("object_id").and_then(|v| v.as_u64()) {
                Some(id) => id as u32,
                None => return Ok(tool_err_json("object_id is required")),
            };
            match api.execute_command(&format!("voxelize {object_id}")) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "message": msg}))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "scene_save" => {
            let cmd = if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
                format!("save {path}")
            } else {
                "save".to_string()
            };
            match api.execute_command(&cmd) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "message": msg}))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        "scene_open" => {
            let path = match args.get("path").and_then(|v| v.as_str()) {
                Some(p) => p,
                None => return Ok(tool_err_json("path is required")),
            };
            match api.execute_command(&format!("open {path}")) {
                Ok(msg) => Ok(tool_ok_json(serde_json::json!({"status": "ok", "message": msg}))),
                Err(e) => Ok(tool_err_json(&e.to_string())),
            }
        }
        _ => {
            // Try behavior system tools before giving up.
            if let Some(result) =
                super::super::behavior::dispatch::dispatch_behavior_tool_call(api, name, args)
            {
                return result;
            }
            Err(AutomationError::NotImplemented("unknown tool"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::*;
    use super::super::registration::register_observation_tools;
    use rkf_core::automation::StubAutomationApi;

    #[test]
    fn register_all_observation_tools() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        assert_eq!(registry.len(), 34);
    }

    #[test]
    fn observation_tools_visible_in_debug_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Debug);
        // Debug mode sees "Both" tools only (+object_shape, +material_list, +material_get, +shader_list)
        assert_eq!(tools.len(), 22);
    }

    #[test]
    fn all_tools_visible_in_editor_mode() {
        let mut registry = ToolRegistry::new();
        register_observation_tools(&mut registry);
        let tools = registry.list_tools(ToolMode::Editor);
        assert_eq!(tools.len(), 34);
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
            "screenshot", "screenshot_window", "scene_graph", "entity_inspect",
            "render_stats", "log_read", "camera_get", "brick_pool_stats",
            "spatial_query", "asset_status", "debug_mode", "camera_set",
        ];
        for name in &expected {
            assert!(registry.get_tool(name).is_some(), "missing tool: {name}");
        }
    }
}
