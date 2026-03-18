//! Editor automation API — implements [`AutomationApi`] backed by shared engine state.
//!
//! [`SharedState`] is updated by the render loop each frame. [`EditorAutomationApi`]
//! reads it to serve MCP tool requests over IPC.

mod api_behavior;
mod api_helpers;
mod api_impl;
pub(crate) mod rinch_debug_client;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;
use rkf_core::automation::*;

use rkf_core::material_library::MaterialLibrary;

use crate::editor_state::EditorState;

/// Camera position + orientation set via MCP camera_set tool.
pub struct PendingCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Result of a GPU brush hit readback from the G-buffer.
#[derive(Debug, Clone, Copy)]
pub struct BrushHitResult {
    /// World-space hit position (from position G-buffer).
    pub position: Vec3,
    /// Object ID at the hit pixel (bits 24-31 of material G-buffer).
    pub object_id: u32,
}

/// Whether the behavior system is currently in play mode.
/// Written by the engine loop, read by MCP tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlayModeState {
    #[default]
    Stopped,
    Playing,
}

/// Shared mutable state between the render loop and the automation API.
pub struct SharedState {
    /// Camera position in world space.
    pub camera_position: Vec3,
    /// Camera yaw in radians.
    pub camera_yaw: f32,
    /// Camera pitch in radians.
    pub camera_pitch: f32,
    /// Camera vertical FOV in degrees.
    pub camera_fov: f32,
    /// Brick pool total capacity.
    pub pool_capacity: u64,
    /// Brick pool allocated count.
    pub pool_allocated: u64,
    /// Last frame time in milliseconds.
    pub frame_time_ms: f64,
    /// Last rendered frame pixels (RGBA, display resolution).
    pub frame_pixels: Vec<u8>,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Pending debug mode change (set by MCP, consumed by render loop).
    pub pending_debug_mode: Option<u32>,
    /// Pending camera teleport (set by MCP, consumed by render loop).
    pub pending_camera: Option<PendingCamera>,
    /// Set true by MCP screenshot(); render loop performs readback and clears.
    pub screenshot_requested: bool,
    /// GPU pick request: internal-resolution pixel to sample for object_id.
    /// Set by click handler, consumed by render loop.
    pub pending_pick: Option<(u32, u32)>,
    /// GPU pick result: object_id at the requested pixel (0 = miss/sky).
    /// Set by render loop after readback, consumed by engine thread section l.
    pub pick_result: Option<u32>,
    /// GPU brush hit request: internal-resolution pixel to sample for position.
    /// Set by mouse handler in Sculpt/Paint modes, consumed by render loop.
    pub pending_brush_hit: Option<(u32, u32)>,
    /// GPU brush hit result: world-space position + object_id at the requested pixel.
    /// Set by render loop after readback, consumed by engine thread.
    pub brush_hit_result: Option<BrushHitResult>,
    /// Brush preview position for wireframe sphere rendering.
    /// Set by engine thread after processing brush hit, read by wireframe builder.
    pub brush_preview_pos: Option<Vec3>,
    /// Object ID that the brush is hovering over (for surface-conforming cursor).
    pub brush_preview_object_id: Option<u32>,
    /// Ring buffer of recent log entries for MCP `read_log` tool.
    pub log_entries: VecDeque<LogEntry>,
    /// Engine start time for log timestamps.
    pub start_time: Instant,
    /// Pending voxel slice request (set by MCP, consumed by render loop).
    pub pending_voxel_slice: Option<VoxelSliceRequest>,
    /// Voxel slice result (set by render loop, consumed by MCP polling).
    pub voxel_slice_result: Option<rkf_core::automation::VoxelSliceResult>,
    /// Pending spatial query request (set by MCP, consumed by render loop).
    pub pending_spatial_query: Option<SpatialQueryRequest>,
    /// Spatial query result (set by render loop, consumed by MCP polling).
    pub spatial_query_result: Option<rkf_core::automation::SpatialQueryResult>,
    /// Pending MCP sculpt request (set by MCP, consumed by render loop).
    pub pending_mcp_sculpt: Option<McpSculptRequest>,
    /// MCP sculpt result (set by render loop, consumed by MCP polling).
    pub mcp_sculpt_result: Option<Result<(), String>>,
    /// Pending object_shape request (set by MCP, consumed by render loop).
    pub pending_object_shape: Option<u32>,
    /// Object shape result (set by render loop, consumed by MCP polling).
    pub object_shape_result: Option<rkf_core::automation::ObjectShapeResult>,
    /// Available shader model names (published by engine thread from ShaderComposer).
    pub shader_names: Vec<(String, u32)>,
    /// Which material slot to preview (None = no preview).
    pub preview_material_slot: Option<u16>,
    /// Preview primitive type: 0=sphere, 1=box, 2=capsule, 3=torus, 4=cylinder, 5=plane.
    pub preview_primitive_type: u32,
    /// Whether the preview needs re-rendering.
    pub preview_dirty: bool,
    /// Current play mode state (written by engine loop, read by MCP).
    pub play_mode_state: PlayModeState,
}

/// Request for a voxel slice diagnostic.
#[derive(Debug, Clone)]
pub struct VoxelSliceRequest {
    pub object_id: u32,
    pub y_coord: f32,
}

/// Single MCP sculpt brush hit request.
#[derive(Debug, Clone)]
pub struct McpSculptRequest {
    pub object_id: u32,
    pub position: Vec3,
    pub mode: String,
    pub radius: f32,
    pub strength: f32,
    pub material_id: u16,
}

/// Request for a spatial query at a world position.
#[derive(Debug, Clone)]
pub struct SpatialQueryRequest {
    pub world_pos: Vec3,
}

/// Maximum number of log entries kept in the ring buffer.
const MAX_LOG_ENTRIES: usize = 500;

impl SharedState {
    /// Create shared state with the given pool stats and frame dimensions.
    pub fn new(pool_capacity: u64, pool_allocated: u64, width: u32, height: u32) -> Self {
        Self {
            camera_position: Vec3::ZERO,
            camera_yaw: 0.0,
            camera_pitch: 0.0,
            camera_fov: 60.0,
            pool_capacity,
            pool_allocated,
            frame_time_ms: 0.0,
            frame_pixels: vec![0u8; (width * height * 4) as usize],
            frame_width: width,
            frame_height: height,
            pending_debug_mode: None,
            pending_camera: None,
            screenshot_requested: false,
            pending_pick: None,
            pick_result: None,
            pending_brush_hit: None,
            brush_hit_result: None,
            brush_preview_pos: None,
            brush_preview_object_id: None,
            log_entries: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            start_time: Instant::now(),
            pending_voxel_slice: None,
            voxel_slice_result: None,
            pending_spatial_query: None,
            spatial_query_result: None,
            pending_mcp_sculpt: None,
            mcp_sculpt_result: None,
            pending_object_shape: None,
            object_shape_result: None,
            shader_names: Vec::new(),
            preview_material_slot: None,
            preview_primitive_type: 0,
            preview_dirty: false,
            play_mode_state: PlayModeState::Stopped,
        }
    }

    /// Push a log entry into the ring buffer, evicting the oldest if full.
    pub fn push_log(&mut self, level: LogLevel, message: impl Into<String>) {
        let timestamp_ms = self.start_time.elapsed().as_millis() as u64;
        if self.log_entries.len() >= MAX_LOG_ENTRIES {
            self.log_entries.pop_front();
        }
        self.log_entries.push_back(LogEntry {
            level,
            message: message.into(),
            timestamp_ms,
        });
    }
}

/// Editor implementation of [`AutomationApi`] backed by shared engine state.
pub struct EditorAutomationApi {
    state: Arc<Mutex<SharedState>>,
    editor_state: Arc<Mutex<EditorState>>,
    material_library: Arc<Mutex<MaterialLibrary>>,
    gameplay_registry: Arc<Mutex<rkf_runtime::behavior::GameplayRegistry>>,
    game_store: Arc<Mutex<rkf_runtime::behavior::GameStore>>,
}

impl EditorAutomationApi {
    pub fn new(
        state: Arc<Mutex<SharedState>>,
        editor_state: Arc<Mutex<EditorState>>,
        material_library: Arc<Mutex<MaterialLibrary>>,
        gameplay_registry: Arc<Mutex<rkf_runtime::behavior::GameplayRegistry>>,
    ) -> Self {
        Self::with_game_store(
            state,
            editor_state,
            material_library,
            gameplay_registry,
            Arc::new(Mutex::new(rkf_runtime::behavior::GameStore::new())),
        )
    }

    pub fn with_game_store(
        state: Arc<Mutex<SharedState>>,
        editor_state: Arc<Mutex<EditorState>>,
        material_library: Arc<Mutex<MaterialLibrary>>,
        gameplay_registry: Arc<Mutex<rkf_runtime::behavior::GameplayRegistry>>,
        game_store: Arc<Mutex<rkf_runtime::behavior::GameStore>>,
    ) -> Self {
        Self {
            state,
            editor_state,
            material_library,
            gameplay_registry,
            game_store,
        }
    }

    /// Get a reference to the shared game store.
    pub fn game_store(&self) -> &Arc<Mutex<rkf_runtime::behavior::GameStore>> {
        &self.game_store
    }

    /// Parse a float value from a string, returning an AutomationError on failure.
    fn parse_f32(val: &str, prop: &str) -> AutomationResult<f32> {
        val.parse::<f32>().map_err(|_| {
            AutomationError::InvalidParameter(format!("invalid float for {prop}: {val}"))
        })
    }

    /// Parse a bool value from a string ("true"/"false"/"0"/"1").
    fn parse_bool(val: &str, prop: &str) -> AutomationResult<bool> {
        match val {
            "true" | "1" | "on" => Ok(true),
            "false" | "0" | "off" => Ok(false),
            _ => Err(AutomationError::InvalidParameter(format!(
                "invalid bool for {prop}: {val} (expected true/false/0/1/on/off)"
            ))),
        }
    }

    /// Set an environment property by name via the ECS singleton.
    fn env_set(&self, prop: &str, val: &str) -> AutomationResult<String> {
        use rkf_runtime::behavior::game_value::GameValue;

        let mut es = self.editor_state.lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        let env_entity = es.world.scene_environment_entity()
            .ok_or_else(|| AutomationError::EngineError("no scene environment entity".into()))?;

        // Parse value based on field type.
        let game_value = if prop.ends_with(".enabled") {
            GameValue::Bool(Self::parse_bool(val, prop)?)
        } else if prop == "post_process.tone_map_mode" {
            let v = val.parse::<u32>().map_err(|_| {
                AutomationError::InvalidParameter(format!("invalid u32 for {prop}: {val}"))
            })?;
            GameValue::Int(v as i64)
        } else if prop == "atmosphere.sun_direction" {
            // Accept "x,y,z" format.
            let parts: Vec<f32> = val.split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect::<Result<_, _>>()
                .map_err(|_| AutomationError::InvalidParameter(format!("invalid Vec3 for {prop}: {val}")))?;
            if parts.len() != 3 {
                return Err(AutomationError::InvalidParameter(format!("expected 3 components for {prop}")));
            }
            GameValue::Vec3(glam::Vec3::new(parts[0], parts[1], parts[2]))
        } else {
            GameValue::Float(Self::parse_f32(val, prop)? as f64)
        };

        let reg = self.gameplay_registry.lock()
            .map_err(|e| AutomationError::EngineError(format!("registry lock: {e}")))?;
        if let Some(entry) = reg.component_entry("EnvironmentSettings") {
            (entry.set_field)(es.world.ecs_mut(), env_entity, prop, game_value)
                .map_err(|e| AutomationError::EngineError(format!("set_field: {e}")))?;
        }

        Ok(format!("{prop} = {val}"))
    }

    /// Get an environment property value by name via the ECS singleton.
    fn env_get(&self, prop: &str) -> AutomationResult<String> {
        let es = self.editor_state.lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        let env_entity = es.world.scene_environment_entity()
            .ok_or_else(|| AutomationError::EngineError("no scene environment entity".into()))?;

        let reg = self.gameplay_registry.lock()
            .map_err(|e| AutomationError::EngineError(format!("registry lock: {e}")))?;
        let entry = reg.component_entry("EnvironmentSettings")
            .ok_or_else(|| AutomationError::EngineError("EnvironmentSettings not registered".into()))?;

        if prop == "all" {
            // Return a summary of key fields.
            let mut parts = Vec::new();
            for field in &["atmosphere.enabled", "atmosphere.sun_intensity",
                           "fog.enabled", "fog.density",
                           "clouds.enabled", "clouds.coverage",
                           "post_process.bloom_enabled", "post_process.exposure"]
            {
                if let Ok(v) = (entry.get_field)(es.world.ecs_ref(), env_entity, field) {
                    parts.push(format!("{field}={}", crate::engine_loop_commands::format_game_value_short(&v)));
                }
            }
            return Ok(parts.join(", "));
        }

        let val = (entry.get_field)(es.world.ecs_ref(), env_entity, prop)
            .map_err(|e| AutomationError::InvalidParameter(e))?;
        Ok(crate::engine_loop_commands::format_game_value_short(&val))
    }
}
