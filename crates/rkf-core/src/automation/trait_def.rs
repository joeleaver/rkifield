//! The [`AutomationApi`] trait definition.

use std::collections::HashMap;

use super::types::*;

/// The engine's automation surface — called by `rkf-mcp`.
///
/// Implemented by `rkf-runtime` to provide read and write access to engine
/// state from MCP tools and AI agents.
///
/// # Mode restrictions
/// Observation methods are available in **Editor** and **Debug** modes.
/// Mutation methods (`entity_spawn`, `material_set`, etc.) are only available
/// in **Editor** mode; they return [`AutomationError::EngineError`] otherwise.
pub trait AutomationApi: Send + Sync {
    // --- Observation -------------------------------------------------------

    /// Capture the current viewport as a PNG-encoded byte vector.
    ///
    /// The engine will render an off-screen frame at the requested resolution.
    fn screenshot(&self, width: u32, height: u32) -> AutomationResult<Vec<u8>>;

    /// Capture the full editor window (UI + viewport composited) as a PNG-encoded byte vector.
    ///
    /// Uses the rinch debug protocol to read the composited window texture.
    fn screenshot_window(&self) -> AutomationResult<Vec<u8>> {
        Err(AutomationError::NotImplemented("screenshot_window"))
    }

    /// Return the full scene entity hierarchy.
    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot>;

    /// Return all components attached to the entity with the given id (UUID string).
    fn entity_inspect(&self, entity_id: &str) -> AutomationResult<EntitySnapshot>;

    /// Return the most recent frame timing and resource statistics.
    fn render_stats(&self) -> AutomationResult<RenderStats>;

    /// Return brick pool streaming and upload queue status.
    fn asset_status(&self) -> AutomationResult<AssetStatusReport>;

    /// Return the last `lines` log entries (most recent last).
    fn read_log(&self, lines: usize) -> AutomationResult<Vec<LogEntry>>;

    /// Return the current camera position, orientation, and projection.
    fn camera_state(&self) -> AutomationResult<CameraSnapshot>;

    /// Return raw brick pool occupancy counters.
    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats>;

    /// Sample the signed distance field at a world-space position.
    ///
    /// `chunk` is the integer chunk coordinate; `local` is the sub-chunk
    /// position in metres.
    fn spatial_query(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult>;

    // --- Mutation (editor mode only) ---------------------------------------

    /// Spawn a new entity described by `def` and return its UUID string.
    fn entity_spawn(&self, def: EntityDef) -> AutomationResult<String>;

    /// Remove the entity with the given id (UUID string) from the scene.
    fn entity_despawn(&self, entity_id: &str) -> AutomationResult<()>;

    /// Add or replace a component on the given entity (UUID string).
    fn entity_set_component(
        &self,
        entity_id: &str,
        component: ComponentDef,
    ) -> AutomationResult<()>;

    /// Update fields of the material at slot `id` in the global material table.
    fn material_set(&self, id: u16, material: MaterialDef) -> AutomationResult<()>;

    /// Apply a CSG / sculpt brush operation described by `op`.
    ///
    /// The `op` JSON object is forwarded to `rkf-edit`; its schema is defined
    /// in that crate's brush operation registry.
    fn brush_apply(&self, op: serde_json::Value) -> AutomationResult<()>;

    /// Load and activate the scene at the given `.rkscene` path.
    fn scene_load(&self, path: &str) -> AutomationResult<()>;

    /// Serialise the current scene to the given `.rkscene` path.
    fn scene_save(&self, path: &str) -> AutomationResult<()>;

    /// Teleport the camera to the given world-space position and orientation.
    fn camera_set(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
        rotation: [f32; 4],
    ) -> AutomationResult<()>;

    /// Switch the renderer to the given quality preset.
    fn quality_preset(&self, preset: QualityPreset) -> AutomationResult<()>;

    /// Execute an engine console command and return its output string.
    fn execute_command(&self, command: &str) -> AutomationResult<String>;

    // --- Tool discovery (default: empty/unsupported) -----------------------

    /// List available MCP tools from the engine as a JSON array of tool definition objects.
    ///
    /// Each element has `name`, `description`, and `inputSchema` fields matching
    /// the MCP `tools/list` response schema.
    fn list_tools_json(&self) -> AutomationResult<serde_json::Value> {
        Ok(serde_json::json!([]))
    }

    /// Forward a raw MCP tool call to the engine.
    ///
    /// Returns the full `ToolsCallResult` JSON (content array + isError flag).
    #[allow(unused_variables)]
    fn call_tool_json(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> AutomationResult<serde_json::Value> {
        Err(AutomationError::NotImplemented("call_tool_json"))
    }

    // --- v2 object-centric methods (default: unsupported) ------------------

    /// Spawn a new SDF object in the v2 scene and return its object ID.
    ///
    /// `primitive_type` is a string such as `"sphere"`, `"box"`, `"capsule"`, etc.
    /// `params` are primitive-specific shape parameters (radius, half-extents, ...).
    /// `position` is the initial world-space position in metres.
    /// `material_id` is the index into the global material table.
    #[allow(unused_variables)]
    fn object_spawn(
        &self,
        name: &str,
        primitive_type: &str,
        params: &[f32],
        position: [f32; 3],
        material_id: u16,
    ) -> Result<u32, String> {
        Err("object_spawn not supported".into())
    }

    /// Despawn an SDF object by ID, removing it from the v2 scene.
    #[allow(unused_variables)]
    fn object_despawn(&self, object_id: u32) -> Result<(), String> {
        Err("object_despawn not supported".into())
    }

    /// Set a node's local transform in the v2 scene.
    ///
    /// `position` is the local translation in metres, `rotation` is a unit
    /// quaternion `[x, y, z, w]`, and `scale` is per-axis `[sx, sy, sz]`.
    #[allow(unused_variables)]
    fn node_set_transform(
        &self,
        object_id: u32,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) -> Result<(), String> {
        Err("node_set_transform not supported".into())
    }

    /// Return a human-readable description of the current environment profile.
    fn environment_get(&self) -> Result<String, String> {
        Err("environment_get not supported".into())
    }

    /// Begin blending the environment toward the profile at `target_index` over
    /// `duration` seconds.
    #[allow(unused_variables)]
    fn environment_blend(&self, target_index: usize, duration: f32) -> Result<(), String> {
        Err("environment_blend not supported".into())
    }

    /// Override a single environment property by name.
    ///
    /// `property` is a dot-separated path such as `"sun.intensity"` or
    /// `"fog.density"`.  `value` is the new scalar value.
    #[allow(unused_variables)]
    fn env_override(&self, property: &str, value: f32) -> Result<(), String> {
        Err("env_override not supported".into())
    }

    // --- Node tree operations (default: unsupported) -------------------------

    /// Find a node by name within an object's scene node tree.
    /// Returns a JSON string describing the node.
    #[allow(unused_variables)]
    fn node_find(&self, object_id: u32, node_name: &str) -> Result<String, String> {
        Err("node_find not supported".into())
    }

    /// Add a child node to a named parent node within an object's tree.
    #[allow(unused_variables)]
    fn node_add_child(
        &self,
        object_id: u32,
        parent_node: &str,
        child_primitive: &str,
        params: &[f32],
        name: &str,
        material_id: u16,
    ) -> Result<(), String> {
        Err("node_add_child not supported".into())
    }

    /// Remove a named node from an object's tree.
    #[allow(unused_variables)]
    fn node_remove(&self, object_id: u32, node_name: &str) -> Result<(), String> {
        Err("node_remove not supported".into())
    }

    // --- Multi-scene operations (default: unsupported) -----------------------

    /// Create a new empty scene, returning its index.
    #[allow(unused_variables)]
    fn scene_create(&self, name: &str) -> Result<usize, String> {
        Err("scene_create not supported".into())
    }

    /// List all scenes as a JSON string.
    fn scene_list(&self) -> Result<String, String> {
        Err("scene_list not supported".into())
    }

    /// Set the active scene by index.
    #[allow(unused_variables)]
    fn scene_set_active(&self, index: usize) -> Result<(), String> {
        Err("scene_set_active not supported".into())
    }

    /// Mark a scene as persistent or not.
    #[allow(unused_variables)]
    fn scene_set_persistent(&self, index: usize, persistent: bool) -> Result<(), String> {
        Err("scene_set_persistent not supported".into())
    }

    /// Swap scenes: unload non-persistent scenes. Returns names of removed scenes.
    fn scene_swap(&self) -> Result<String, String> {
        Err("scene_swap not supported".into())
    }

    // --- Camera entity operations (default: unsupported) ---------------------

    /// Spawn a camera entity. Returns the entity's UUID string.
    #[allow(unused_variables)]
    fn camera_spawn(
        &self,
        label: &str,
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        fov: f32,
    ) -> Result<String, String> {
        Err("camera_spawn not supported".into())
    }

    /// List all camera entities as a JSON string.
    fn camera_list(&self) -> Result<String, String> {
        Err("camera_list not supported".into())
    }

    /// Snap the rendering camera to a camera entity (UUID string).
    #[allow(unused_variables)]
    fn camera_snap_to(&self, entity_id: &str) -> Result<(), String> {
        Err("camera_snap_to not supported".into())
    }

    // --- Diagnostic tools (default: unsupported) -----------------------------

    /// Sample a 2D XZ slice of raw SDF distances from an object's voxel data.
    ///
    /// `object_id` identifies the scene object. `y_coord` is the object-local
    /// Y coordinate for the slice. Returns a grid of distance values and
    /// per-sample brick slot status.
    #[allow(unused_variables)]
    fn voxel_slice(
        &self,
        object_id: u32,
        y_coord: f32,
    ) -> AutomationResult<VoxelSliceResult> {
        Err(AutomationError::NotImplemented("voxel_slice"))
    }

    /// Apply a single sculpt brush hit to an object via MCP.
    ///
    /// Creates a one-shot undo entry. `position` is world-space `[x, y, z]`.
    /// `mode` is `"add"`, `"subtract"`, or `"smooth"`.
    #[allow(unused_variables)]
    fn sculpt_apply(
        &self,
        object_id: u32,
        position: [f32; 3],
        mode: &str,
        radius: f32,
        strength: f32,
        material_id: u16,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("sculpt_apply"))
    }

    /// Return a compact brick-level 3D shape overview of an object.
    ///
    /// Each brick is categorized as empty, interior, or surface/allocated.
    /// Returns per-Y-level ASCII slices for spatial reasoning.
    #[allow(unused_variables)]
    fn object_shape(&self, object_id: u32) -> AutomationResult<ObjectShapeResult> {
        Err(AutomationError::NotImplemented("object_shape"))
    }

    // --- Material library methods (default: unsupported) ---------------------

    /// List all materials in the library with summary info.
    fn material_list(&self) -> AutomationResult<Vec<MaterialInfo>> {
        Err(AutomationError::NotImplemented("material_list"))
    }

    /// Get full properties of a material at the given slot.
    #[allow(unused_variables)]
    fn material_get(&self, slot: u16) -> AutomationResult<MaterialSnapshot> {
        Err(AutomationError::NotImplemented("material_get"))
    }

    /// List available shader models (name, id, built_in).
    fn shader_list(&self) -> AutomationResult<Vec<ShaderInfo>> {
        Err(AutomationError::NotImplemented("shader_list"))
    }

    // --- Behavior system: component tools (Phase 14.1) -----------------------

    /// List all registered component types and their fields.
    fn component_list(&self) -> Vec<ComponentInfo> {
        vec![]
    }

    /// Get all field values of a component on an entity (UUID string).
    #[allow(unused_variables)]
    fn component_get(
        &self,
        entity_id: &str,
        component_name: &str,
    ) -> Result<HashMap<String, String>, String> {
        Err("component_get not implemented".into())
    }

    /// Set field values on an existing component of an entity (UUID string).
    #[allow(unused_variables)]
    fn component_set(
        &self,
        entity_id: &str,
        component_name: &str,
        fields: HashMap<String, String>,
    ) -> Result<(), String> {
        Err("component_set not implemented".into())
    }

    /// Add a new component to an entity with the given field values (UUID string).
    #[allow(unused_variables)]
    fn component_add(
        &self,
        entity_id: &str,
        component_name: &str,
        fields: HashMap<String, String>,
    ) -> Result<(), String> {
        Err("component_add not implemented".into())
    }

    /// Remove a component from an entity by name (UUID string).
    #[allow(unused_variables)]
    fn component_remove(&self, entity_id: &str, component_name: &str) -> Result<(), String> {
        Err("component_remove not implemented".into())
    }

    // --- Behavior system: system + blueprint tools (Phase 14.2) --------------

    /// List all registered behavior systems.
    fn system_list(&self) -> Vec<SystemInfo> {
        vec![]
    }

    /// List all available blueprints (prefabs).
    fn blueprint_list(&self) -> Vec<BlueprintInfo> {
        vec![]
    }

    /// Spawn an entity from a named blueprint at the given position.
    /// Returns the new entity's UUID string.
    #[allow(unused_variables)]
    fn blueprint_spawn(&self, name: &str, position: [f32; 3]) -> Result<String, String> {
        Err("blueprint_spawn not implemented".into())
    }

    // --- Behavior system: state tools (Phase 14.3) ---------------------------

    /// Get a value from the game state store by key.
    #[allow(unused_variables)]
    fn state_get(&self, key: &str) -> Result<Option<String>, String> {
        Err("state_get not implemented".into())
    }

    /// Set a value in the game state store.
    /// `value_type` hints the type (e.g. `"f32"`, `"i32"`, `"string"`, `"bool"`).
    #[allow(unused_variables)]
    fn state_set(&self, key: &str, value: &str, value_type: &str) -> Result<(), String> {
        Err("state_set not implemented".into())
    }

    /// List all keys in the game state store matching a prefix.
    /// An empty prefix returns all keys.
    #[allow(unused_variables)]
    fn state_list(&self, prefix: &str) -> Vec<String> {
        vec![]
    }

    // --- Behavior system: play control (Phase 14.4) --------------------------

    /// Start play mode (begin running behavior systems).
    fn play_start(&self) -> Result<(), String> {
        Err("play_start not implemented".into())
    }

    /// Stop play mode (pause/stop all behavior systems, revert to edit state).
    fn play_stop(&self) -> Result<(), String> {
        Err("play_stop not implemented".into())
    }

    /// Return the current play state as a string (e.g. `"stopped"`, `"playing"`, `"paused"`).
    fn play_state(&self) -> String {
        "stopped".into()
    }
}
