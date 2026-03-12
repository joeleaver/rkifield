//! Editor automation API — implements [`AutomationApi`] backed by shared engine state.
//!
//! [`SharedState`] is updated by the render loop each frame. [`EditorAutomationApi`]
//! reads it to serve MCP tool requests over IPC.

mod api_helpers;
mod api_impl;
pub(crate) mod rinch_debug_client;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;
use image::ImageEncoder;
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
    pub shader_names: Vec<(String, u32, bool)>,
    /// Which material slot to preview (None = no preview).
    pub preview_material_slot: Option<u16>,
    /// Preview primitive type: 0=sphere, 1=box, 2=capsule, 3=torus, 4=cylinder, 5=plane.
    pub preview_primitive_type: u32,
    /// Whether the preview needs re-rendering.
    pub preview_dirty: bool,
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
}

impl EditorAutomationApi {
    pub fn new(
        state: Arc<Mutex<SharedState>>,
        editor_state: Arc<Mutex<EditorState>>,
        material_library: Arc<Mutex<MaterialLibrary>>,
    ) -> Self {
        Self {
            state,
            editor_state,
            material_library,
        }
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

    /// Set an environment property by name.
    fn env_set(&self, prop: &str, val: &str) -> AutomationResult<String> {
        let mut es = self.editor_state.lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
        let env = &mut es.environment;
        match prop {
            // --- Atmosphere ---
            "atmosphere.enabled" => {
                env.atmosphere.enabled = Self::parse_bool(val, prop)?;
            }
            "atmosphere.rayleigh_scale" => {
                env.atmosphere.rayleigh_scale = Self::parse_f32(val, prop)?;
            }
            "atmosphere.mie_scale" => {
                env.atmosphere.mie_scale = Self::parse_f32(val, prop)?;
            }
            "atmosphere.sun_intensity" => {
                env.atmosphere.sun_intensity = Self::parse_f32(val, prop)?;
            }
            "atmosphere.sun_azimuth" => {
                let az = Self::parse_f32(val, prop)?.to_radians();
                let el = env.atmosphere.sun_direction.y.asin();
                env.atmosphere.sun_direction = glam::Vec3::new(
                    el.cos() * az.sin(), el.sin(), el.cos() * az.cos(),
                ).normalize();
            }
            "atmosphere.sun_elevation" => {
                let el = Self::parse_f32(val, prop)?.to_radians();
                let cur = env.atmosphere.sun_direction;
                let az = cur.x.atan2(cur.z);
                env.atmosphere.sun_direction = glam::Vec3::new(
                    el.cos() * az.sin(), el.sin(), el.cos() * az.cos(),
                ).normalize();
            }
            // --- Fog ---
            "fog.enabled" => {
                env.fog.enabled = Self::parse_bool(val, prop)?;
            }
            "fog.density" => {
                env.fog.density = Self::parse_f32(val, prop)?;
            }
            "fog.start_distance" => {
                env.fog.start_distance = Self::parse_f32(val, prop)?;
            }
            "fog.end_distance" => {
                env.fog.end_distance = Self::parse_f32(val, prop)?;
            }
            "fog.height_falloff" => {
                env.fog.height_falloff = Self::parse_f32(val, prop)?;
            }
            "fog.ambient_dust_density" => {
                env.fog.ambient_dust_density = Self::parse_f32(val, prop)?;
            }
            "fog.dust_asymmetry" => {
                env.fog.dust_asymmetry = Self::parse_f32(val, prop)?;
            }
            // --- Clouds ---
            "clouds.enabled" => {
                env.clouds.enabled = Self::parse_bool(val, prop)?;
            }
            "clouds.coverage" => {
                env.clouds.coverage = Self::parse_f32(val, prop)?;
            }
            "clouds.density" => {
                env.clouds.density = Self::parse_f32(val, prop)?;
            }
            "clouds.altitude" => {
                env.clouds.altitude = Self::parse_f32(val, prop)?;
            }
            "clouds.thickness" => {
                env.clouds.thickness = Self::parse_f32(val, prop)?;
            }
            "clouds.wind_speed" => {
                env.clouds.wind_speed = Self::parse_f32(val, prop)?;
            }
            // --- Post-processing ---
            "post_process.bloom_enabled" => {
                env.post_process.bloom_enabled = Self::parse_bool(val, prop)?;
            }
            "post_process.bloom_intensity" => {
                env.post_process.bloom_intensity = Self::parse_f32(val, prop)?;
            }
            "post_process.bloom_threshold" => {
                env.post_process.bloom_threshold = Self::parse_f32(val, prop)?;
            }
            "post_process.exposure" => {
                env.post_process.exposure = Self::parse_f32(val, prop)?;
            }
            "post_process.contrast" => {
                env.post_process.contrast = Self::parse_f32(val, prop)?;
            }
            "post_process.saturation" => {
                env.post_process.saturation = Self::parse_f32(val, prop)?;
            }
            "post_process.vignette_intensity" => {
                env.post_process.vignette_intensity = Self::parse_f32(val, prop)?;
            }
            "post_process.tone_map_mode" => {
                env.post_process.tone_map_mode = val.parse::<u32>().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid u32 for {prop}: {val}"))
                })?;
            }
            "post_process.sharpen_strength" => {
                env.post_process.sharpen_strength = Self::parse_f32(val, prop)?;
            }
            "post_process.dof_enabled" => {
                env.post_process.dof_enabled = Self::parse_bool(val, prop)?;
            }
            "post_process.dof_focus_distance" => {
                env.post_process.dof_focus_distance = Self::parse_f32(val, prop)?;
            }
            "post_process.dof_focus_range" => {
                env.post_process.dof_focus_range = Self::parse_f32(val, prop)?;
            }
            "post_process.dof_max_coc" => {
                env.post_process.dof_max_coc = Self::parse_f32(val, prop)?;
            }
            "post_process.motion_blur_intensity" => {
                env.post_process.motion_blur_intensity = Self::parse_f32(val, prop)?;
            }
            "post_process.grain_intensity" => {
                env.post_process.grain_intensity = Self::parse_f32(val, prop)?;
            }
            "post_process.chromatic_aberration" => {
                env.post_process.chromatic_aberration = Self::parse_f32(val, prop)?;
            }
            _ => {
                return Err(AutomationError::InvalidParameter(format!(
                    "unknown environment property: {prop}"
                )));
            }
        }
        env.mark_dirty();
        Ok(format!("{prop} = {val}"))
    }

    /// Get an environment property value by name.
    fn env_get(&self, prop: &str) -> AutomationResult<String> {
        let es = self.editor_state.lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
        let env = &es.environment;
        let val = match prop {
            // --- Atmosphere ---
            "atmosphere.enabled" => format!("{}", env.atmosphere.enabled),
            "atmosphere.rayleigh_scale" => format!("{}", env.atmosphere.rayleigh_scale),
            "atmosphere.mie_scale" => format!("{}", env.atmosphere.mie_scale),
            "atmosphere.sun_intensity" => format!("{}", env.atmosphere.sun_intensity),
            "atmosphere.sun_direction" => format!(
                "[{}, {}, {}]",
                env.atmosphere.sun_direction.x,
                env.atmosphere.sun_direction.y,
                env.atmosphere.sun_direction.z,
            ),
            // --- Fog ---
            "fog.enabled" => format!("{}", env.fog.enabled),
            "fog.density" => format!("{}", env.fog.density),
            "fog.start_distance" => format!("{}", env.fog.start_distance),
            "fog.end_distance" => format!("{}", env.fog.end_distance),
            "fog.height_falloff" => format!("{}", env.fog.height_falloff),
            "fog.ambient_dust_density" => format!("{}", env.fog.ambient_dust_density),
            "fog.dust_asymmetry" => format!("{}", env.fog.dust_asymmetry),
            // --- Clouds ---
            "clouds.enabled" => format!("{}", env.clouds.enabled),
            "clouds.coverage" => format!("{}", env.clouds.coverage),
            "clouds.density" => format!("{}", env.clouds.density),
            "clouds.altitude" => format!("{}", env.clouds.altitude),
            "clouds.thickness" => format!("{}", env.clouds.thickness),
            "clouds.wind_speed" => format!("{}", env.clouds.wind_speed),
            // --- Post-processing ---
            "post_process.bloom_enabled" => format!("{}", env.post_process.bloom_enabled),
            "post_process.bloom_intensity" => format!("{}", env.post_process.bloom_intensity),
            "post_process.bloom_threshold" => format!("{}", env.post_process.bloom_threshold),
            "post_process.exposure" => format!("{}", env.post_process.exposure),
            "post_process.contrast" => format!("{}", env.post_process.contrast),
            "post_process.saturation" => format!("{}", env.post_process.saturation),
            "post_process.vignette_intensity" => format!("{}", env.post_process.vignette_intensity),
            "post_process.tone_map_mode" => format!("{}", env.post_process.tone_map_mode),
            "post_process.sharpen_strength" => format!("{}", env.post_process.sharpen_strength),
            "post_process.dof_enabled" => format!("{}", env.post_process.dof_enabled),
            "post_process.dof_focus_distance" => format!("{}", env.post_process.dof_focus_distance),
            "post_process.dof_focus_range" => format!("{}", env.post_process.dof_focus_range),
            "post_process.dof_max_coc" => format!("{}", env.post_process.dof_max_coc),
            "post_process.motion_blur_intensity" => format!("{}", env.post_process.motion_blur_intensity),
            "post_process.grain_intensity" => format!("{}", env.post_process.grain_intensity),
            "post_process.chromatic_aberration" => format!("{}", env.post_process.chromatic_aberration),
            "all" => {
                // Return all properties as JSON-like summary.
                format!(
                    "atmosphere: enabled={} rayleigh={} mie={} sun_intensity={}\n\
                     fog: enabled={} density={} dust={} dust_g={}\n\
                     clouds: enabled={} coverage={} density={} altitude={} thickness={}\n\
                     post_process: bloom={}/{} exposure={} contrast={} saturation={} sharpen={} dof={} vignette={}",
                    env.atmosphere.enabled, env.atmosphere.rayleigh_scale,
                    env.atmosphere.mie_scale, env.atmosphere.sun_intensity,
                    env.fog.enabled, env.fog.density,
                    env.fog.ambient_dust_density, env.fog.dust_asymmetry,
                    env.clouds.enabled, env.clouds.coverage, env.clouds.density,
                    env.clouds.altitude, env.clouds.thickness,
                    env.post_process.bloom_enabled, env.post_process.bloom_intensity,
                    env.post_process.exposure, env.post_process.contrast,
                    env.post_process.saturation, env.post_process.sharpen_strength,
                    env.post_process.dof_enabled, env.post_process.vignette_intensity,
                )
            }
            _ => {
                return Err(AutomationError::InvalidParameter(format!(
                    "unknown environment property: {prop}"
                )));
            }
        };
        Ok(val)
    }
}
