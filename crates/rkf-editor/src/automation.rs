//! Editor automation API — implements [`AutomationApi`] backed by shared engine state.
//!
//! [`SharedState`] is updated by the render loop each frame. [`EditorAutomationApi`]
//! reads it to serve MCP tool requests over IPC.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;
use image::ImageEncoder;
use rkf_core::automation::*;

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
    /// Set by the engine thread after processing a pick result.
    /// Cleared by the UI thread after bumping UiRevision.
    pub pick_completed: bool,
    /// Set by the engine thread when a gizmo drag finishes (transform changed).
    /// Cleared by the UI thread after bumping UiRevision.
    pub ui_revision_needed: bool,
    /// GPU brush hit request: internal-resolution pixel to sample for position.
    /// Set by mouse handler in Sculpt/Paint modes, consumed by render loop.
    pub pending_brush_hit: Option<(u32, u32)>,
    /// GPU brush hit result: world-space position + object_id at the requested pixel.
    /// Set by render loop after readback, consumed by engine thread.
    pub brush_hit_result: Option<BrushHitResult>,
    /// Brush preview position for wireframe sphere rendering.
    /// Set by engine thread after processing brush hit, read by wireframe builder.
    pub brush_preview_pos: Option<Vec3>,
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
    /// Pending fix_sdfs request (set by MCP, consumed by render loop).
    pub pending_fix_sdfs: Option<u32>,
    /// fix_sdfs result: Ok(()) on success, Err(msg) on failure.
    pub fix_sdfs_result: Option<Result<(), String>>,
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
            pick_completed: false,
            ui_revision_needed: false,
            pending_brush_hit: None,
            brush_hit_result: None,
            brush_preview_pos: None,
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
            pending_fix_sdfs: None,
            fix_sdfs_result: None,
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
}

impl EditorAutomationApi {
    pub fn new(state: Arc<Mutex<SharedState>>, editor_state: Arc<Mutex<EditorState>>) -> Self {
        Self {
            state,
            editor_state,
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

impl AutomationApi for EditorAutomationApi {
    fn list_tools_json(&self) -> AutomationResult<serde_json::Value> {
        Ok(rkf_mcp::tools::observation::standard_tool_definitions())
    }

    fn call_tool_json(&self, name: &str, args: serde_json::Value) -> AutomationResult<serde_json::Value> {
        rkf_mcp::tools::observation::dispatch_tool_call(self, name, args)
    }

    fn screenshot(&self, _width: u32, _height: u32) -> AutomationResult<Vec<u8>> {
        // Request the render loop to capture pixels.
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.screenshot_requested = true;
        }

        // Wait for the render loop to fulfill the request (up to ~2000ms).
        // The render loop must receive a redraw event before it can process
        // the screenshot, so allow extra time for event-loop latency.
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            if !state.screenshot_requested {
                if state.frame_pixels.is_empty() {
                    return Err(AutomationError::EngineError(
                        "no frame captured yet".into(),
                    ));
                }
                let mut png_bytes = Vec::new();
                let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
                encoder
                    .write_image(
                        &state.frame_pixels,
                        state.frame_width,
                        state.frame_height,
                        image::ExtendedColorType::Rgba8,
                    )
                    .map_err(|e| AutomationError::EngineError(format!("PNG encode failed: {e}")))?;
                return Ok(png_bytes);
            }
        }

        Err(AutomationError::EngineError(
            "screenshot timeout — render loop did not respond within 2000ms".into(),
        ))
    }

    fn camera_state(&self) -> AutomationResult<CameraSnapshot> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        let quat = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            state.camera_yaw,
            state.camera_pitch,
            0.0,
        );

        Ok(CameraSnapshot {
            chunk: [0, 0, 0],
            local: [
                state.camera_position.x,
                state.camera_position.y,
                state.camera_position.z,
            ],
            rotation: [quat.x, quat.y, quat.z, quat.w],
            fov_degrees: state.camera_fov,
        })
    }

    fn render_stats(&self) -> AutomationResult<RenderStats> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        Ok(RenderStats {
            frame_time_ms: state.frame_time_ms,
            pass_timings: HashMap::from([("frame".to_string(), state.frame_time_ms)]),
            brick_pool_usage: if state.pool_capacity > 0 {
                state.pool_allocated as f32 / state.pool_capacity as f32
            } else {
                0.0
            },
            memory_mb: (state.pool_allocated * 4096) as f32 / (1024.0 * 1024.0),
        })
    }

    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        Ok(BrickPoolStats {
            capacity: state.pool_capacity,
            allocated: state.pool_allocated,
            free_list_size: state.pool_capacity - state.pool_allocated,
        })
    }

    fn asset_status(&self) -> AutomationResult<AssetStatusReport> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        Ok(AssetStatusReport {
            loaded_chunks: 1,
            pending_uploads: 0,
            total_bricks: state.pool_allocated,
            pool_capacity: state.pool_capacity,
        })
    }

    fn spatial_query(
        &self,
        _chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
        // Submit request to the engine thread and poll for result.
        let world_pos = Vec3::new(local[0], local[1], local[2]);
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.spatial_query_result = None;
            state.pending_spatial_query = Some(SpatialQueryRequest { world_pos });
        }

        // Poll for up to 2 seconds.
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Ok(mut state) = self.state.lock() {
                if let Some(result) = state.spatial_query_result.take() {
                    return Ok(result);
                }
            }
        }
        Err(AutomationError::EngineError(
            "spatial_query timed out waiting for engine".into(),
        ))
    }

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        // Read directly from world.scene() — the authoritative source.
        let scene = es.world.scene();
        let mut entities = Vec::new();
        for obj in &scene.objects {
            let p = obj.position;
            let r = obj.rotation;
            let entity_type = match &obj.root_node.sdf_source {
                rkf_core::scene_node::SdfSource::None => "entity",
                _ => "sdf_object",
            };
            entities.push(EntityNode {
                id: obj.id as u64,
                name: obj.name.clone(),
                parent: obj.parent_id.map(|pid| pid as u64),
                entity_type: entity_type.to_string(),
                transform: [p.x, p.y, p.z, r.x, r.y, r.z, r.w, obj.scale.x, obj.scale.y, obj.scale.z],
            });
        }

        // Include light entities from light editor.
        use crate::light_editor::EditorLightType;
        for (idx, light) in es.light_editor.all_lights().iter().enumerate() {
            let (light_type, type_name) = match light.light_type {
                EditorLightType::Point => ("point_light", "Point"),
                EditorLightType::Spot => ("spot_light", "Spot"),
            };
            let p = light.position;
            entities.push(EntityNode {
                id: light.id,
                name: format!("{} Light {}", type_name, idx + 1),
                parent: None,
                entity_type: light_type.to_string(),
                transform: [p.x, p.y, p.z, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            });
        }

        Ok(SceneGraphSnapshot { entities })
    }

    fn entity_inspect(&self, entity_id: u64) -> AutomationResult<EntitySnapshot> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        // Look up from world.scene() — the authoritative source.
        let scene = es.world.scene();
        let obj = scene.objects.iter()
            .find(|o| o.id as u64 == entity_id)
            .ok_or(AutomationError::EntityNotFound(entity_id))?;

        let is_selected = matches!(
            es.selected_entity,
            Some(crate::editor_state::SelectedEntity::Object(eid)) if eid == entity_id
        );

        let child_count = scene.objects.iter()
            .filter(|o| o.parent_id == Some(obj.id))
            .count();

        let mut components = HashMap::new();
        components.insert(
            "scene_object".to_string(),
            serde_json::json!({
                "selected": is_selected,
                "children": child_count,
                "position": [obj.position.x, obj.position.y, obj.position.z],
                "rotation": [obj.rotation.x, obj.rotation.y, obj.rotation.z, obj.rotation.w],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
                "sdf_type": format!("{:?}", obj.root_node.sdf_source),
            }),
        );

        Ok(EntitySnapshot {
            id: entity_id,
            name: obj.name.clone(),
            components,
        })
    }

    fn read_log(&self, lines: usize) -> AutomationResult<Vec<LogEntry>> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        let total = state.log_entries.len();
        let skip = total.saturating_sub(lines);
        Ok(state.log_entries.iter().skip(skip).cloned().collect())
    }

    fn entity_spawn(&self, _def: EntityDef) -> AutomationResult<u64> {
        Err(AutomationError::NotImplemented("entity_spawn"))
    }

    fn entity_despawn(&self, _entity_id: u64) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_despawn"))
    }

    fn entity_set_component(
        &self,
        _entity_id: u64,
        _component: ComponentDef,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_set_component"))
    }

    fn material_set(&self, _id: u16, _material: MaterialDef) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("material_set"))
    }

    fn brush_apply(&self, _op: serde_json::Value) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("brush_apply"))
    }

    fn scene_load(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_load"))
    }

    fn scene_save(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_save"))
    }

    fn camera_set(
        &self,
        _chunk: [i32; 3],
        local: [f32; 3],
        rotation: [f32; 4],
    ) -> AutomationResult<()> {
        let quat = glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
        let (yaw, pitch, _roll) = quat.to_euler(glam::EulerRot::YXZ);
        let mut state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
        state.pending_camera = Some(PendingCamera {
            position: Vec3::new(local[0], local[1], local[2]),
            yaw,
            pitch,
        });
        Ok(())
    }

    fn quality_preset(&self, _preset: QualityPreset) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("quality_preset"))
    }

    fn execute_command(&self, command: &str) -> AutomationResult<String> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        match parts.as_slice() {
            ["debug_mode", mode_str] => {
                let mode: u32 = mode_str.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!(
                        "invalid debug mode: {mode_str} (expected 0-8)"
                    ))
                })?;
                if mode > 8 {
                    return Err(AutomationError::InvalidParameter(format!(
                        "debug mode {mode} out of range (expected 0-8)"
                    )));
                }
                let mut state = self
                    .state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                state.pending_debug_mode = Some(mode);
                let mode_name = match mode {
                    0 => "normal shading",
                    1 => "surface normals",
                    2 => "world positions",
                    3 => "material IDs",
                    4 => "diffuse only",
                    5 => "specular only",
                    6 => "GI only",
                    7 => "SDF distance",
                    8 => "brick boundaries",
                    _ => "unknown",
                };
                Ok(format!("debug mode set to {mode} ({mode_name})"))
            }
            ["debug_mode"] => Err(AutomationError::InvalidParameter(
                "usage: debug_mode <0-8>".to_string(),
            )),
            ["camera_set", x_s, y_s, z_s, yaw_s, pitch_s] => {
                let x: f32 = x_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid x: {x_s}"))
                })?;
                let y: f32 = y_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid y: {y_s}"))
                })?;
                let z: f32 = z_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid z: {z_s}"))
                })?;
                let yaw_deg: f32 = yaw_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid yaw: {yaw_s}"))
                })?;
                let pitch_deg: f32 = pitch_s.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid pitch: {pitch_s}"))
                })?;
                let mut state = self
                    .state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                state.pending_camera = Some(PendingCamera {
                    position: Vec3::new(x, y, z),
                    yaw: yaw_deg.to_radians(),
                    pitch: pitch_deg.to_radians(),
                });
                Ok(format!(
                    "camera set to ({x}, {y}, {z}) yaw={yaw_deg} pitch={pitch_deg}"
                ))
            }
            ["camera_set", ..] => Err(AutomationError::InvalidParameter(
                "usage: camera_set <x> <y> <z> <yaw_deg> <pitch_deg>".to_string(),
            )),
            ["env_set", prop, val_str] => {
                self.env_set(prop, val_str)
            }
            ["env_set", ..] => Err(AutomationError::InvalidParameter(
                "usage: env_set <property> <value>".to_string(),
            )),
            ["env_get", prop] => {
                self.env_get(prop)
            }
            ["env_get"] => Err(AutomationError::InvalidParameter(
                "usage: env_get <property>".to_string(),
            )),
            _ => Err(AutomationError::InvalidParameter(format!(
                "unknown command: {command}"
            ))),
        }
    }

    // --- v2 object-centric methods -----------------------------------------

    fn object_spawn(
        &self,
        name: &str,
        primitive_type: &str,
        params: &[f32],
        position: [f32; 3],
        material_id: u16,
    ) -> Result<u32, String> {
        use rkf_core::SdfPrimitive;

        let primitive = match primitive_type {
            "sphere" => {
                let radius = params.first().copied().unwrap_or(0.5);
                SdfPrimitive::Sphere { radius }
            }
            "box" => {
                let hx = params.first().copied().unwrap_or(0.5);
                let hy = params.get(1).copied().unwrap_or(hx);
                let hz = params.get(2).copied().unwrap_or(hx);
                SdfPrimitive::Box { half_extents: Vec3::new(hx, hy, hz) }
            }
            "capsule" => {
                let radius = params.first().copied().unwrap_or(0.2);
                let half_height = params.get(1).copied().unwrap_or(0.4);
                SdfPrimitive::Capsule { radius, half_height }
            }
            "torus" => {
                let major = params.first().copied().unwrap_or(0.4);
                let minor = params.get(1).copied().unwrap_or(0.12);
                SdfPrimitive::Torus { major_radius: major, minor_radius: minor }
            }
            "cylinder" => {
                let radius = params.first().copied().unwrap_or(0.3);
                let half_height = params.get(1).copied().unwrap_or(0.5);
                SdfPrimitive::Cylinder { radius, half_height }
            }
            "plane" => SdfPrimitive::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            other => return Err(format!("unknown primitive type: {other}")),
        };

        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.spawn(name)
            .position_vec3(Vec3::new(position[0], position[1], position[2]))
            .sdf(primitive)
            .material(material_id)
            .build();

        Ok(entity.to_u64() as u32)
    }

    fn object_despawn(&self, object_id: u32) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.find_entity_by_id(object_id as u64)
            .ok_or_else(|| format!("object {object_id} not found"))?;
        es.world.despawn(entity)
            .map_err(|e| format!("{e}"))?;

        Ok(())
    }

    fn node_set_transform(
        &self,
        object_id: u32,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.find_entity_by_id(object_id as u64)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        let pos = rkf_core::WorldPosition::new(
            glam::IVec3::ZERO,
            Vec3::new(position[0], position[1], position[2]),
        );
        let rot = glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
        let scl = Vec3::new(scale[0], scale[1], scale[2]);

        es.world.set_position(entity, pos).map_err(|e| format!("{e}"))?;
        es.world.set_rotation(entity, rot).map_err(|e| format!("{e}"))?;
        es.world.set_scale(entity, scl).map_err(|e| format!("{e}"))?;

        Ok(())
    }

    fn environment_get(&self) -> Result<String, String> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        Ok(format!(
            "environment: sun_intensity={:.2}, fog_enabled={}, fog_density={:.4}, \
             bloom_enabled={}, clouds_enabled={}",
            es.environment.atmosphere.sun_intensity,
            es.environment.fog.enabled,
            es.environment.fog.density,
            es.environment.post_process.bloom_enabled,
            es.environment.clouds.enabled,
        ))
    }

    fn environment_blend(&self, _target_index: usize, _duration: f32) -> Result<(), String> {
        // EnvironmentState has no multi-profile blend concept yet.
        // Mark dirty so the render loop knows something changed.
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        es.environment.mark_dirty();
        Ok(())
    }

    // --- Node tree operations -----------------------------------------------

    fn node_find(&self, object_id: u32, node_name: &str) -> Result<String, String> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.find_entity_by_id(object_id as u64)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        let node = es.world.find_node(entity, node_name)
            .map_err(|e| format!("{e}"))?;

        Ok(serde_json::json!({
            "name": node.name,
            "child_count": node.children.len(),
            "sdf_source": format!("{:?}", node.sdf_source),
            "blend_mode": format!("{:?}", node.blend_mode),
            "transform": {
                "position": [node.local_transform.position.x, node.local_transform.position.y, node.local_transform.position.z],
                "rotation": [node.local_transform.rotation.x, node.local_transform.rotation.y, node.local_transform.rotation.z, node.local_transform.rotation.w],
                "scale": [node.local_transform.scale.x, node.local_transform.scale.y, node.local_transform.scale.z],
            },
            "children": node.children.iter().map(|c| c.name.clone()).collect::<Vec<_>>(),
        })
        .to_string())
    }

    fn node_add_child(
        &self,
        object_id: u32,
        parent_node: &str,
        child_primitive: &str,
        params: &[f32],
        name: &str,
        material_id: u16,
    ) -> Result<(), String> {
        use rkf_core::scene_node::{SceneNode as CoreNode, SdfSource};
        use rkf_core::SdfPrimitive;

        let primitive = match child_primitive {
            "sphere" => {
                let radius = params.first().copied().unwrap_or(0.5);
                SdfPrimitive::Sphere { radius }
            }
            "box" => {
                let hx = params.first().copied().unwrap_or(0.5);
                let hy = params.get(1).copied().unwrap_or(hx);
                let hz = params.get(2).copied().unwrap_or(hx);
                SdfPrimitive::Box {
                    half_extents: Vec3::new(hx, hy, hz),
                }
            }
            "capsule" => {
                let radius = params.first().copied().unwrap_or(0.2);
                let half_height = params.get(1).copied().unwrap_or(0.4);
                SdfPrimitive::Capsule {
                    radius,
                    half_height,
                }
            }
            "torus" => {
                let major = params.first().copied().unwrap_or(0.4);
                let minor = params.get(1).copied().unwrap_or(0.12);
                SdfPrimitive::Torus {
                    major_radius: major,
                    minor_radius: minor,
                }
            }
            "cylinder" => {
                let radius = params.first().copied().unwrap_or(0.3);
                let half_height = params.get(1).copied().unwrap_or(0.5);
                SdfPrimitive::Cylinder {
                    radius,
                    half_height,
                }
            }
            "plane" => SdfPrimitive::Plane {
                normal: Vec3::Y,
                distance: 0.0,
            },
            other => return Err(format!("unknown primitive type: {other}")),
        };

        let mut child = CoreNode::new(name);
        child.sdf_source = SdfSource::Analytical {
            primitive,
            material_id,
        };

        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.find_entity_by_id(object_id as u64)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        es.world.add_child_node(entity, parent_node, child)
            .map_err(|e| format!("{e}"))
    }

    fn node_remove(&self, object_id: u32, node_name: &str) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let entity = es.world.find_entity_by_id(object_id as u64)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        es.world.remove_child_node(entity, node_name)
            .map_err(|e| format!("{e}"))?;

        Ok(())
    }

    // --- Multi-scene operations -----------------------------------------------

    fn scene_create(&self, name: &str) -> Result<usize, String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        Ok(es.world.create_scene(name))
    }

    fn scene_list(&self) -> Result<String, String> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let mut scenes = Vec::new();
        for i in 0..es.world.scene_count() {
            scenes.push(serde_json::json!({
                "index": i,
                "name": es.world.scene_name(i).unwrap_or("unknown"),
                "active": i == es.world.active_scene_index(),
                "persistent": es.world.is_scene_persistent(i),
            }));
        }

        Ok(serde_json::Value::Array(scenes).to_string())
    }

    fn scene_set_active(&self, index: usize) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        if index >= es.world.scene_count() {
            return Err(format!(
                "scene index {index} out of range (count: {})",
                es.world.scene_count()
            ));
        }

        es.world.set_active_scene(index);
        Ok(())
    }

    fn scene_set_persistent(&self, index: usize, persistent: bool) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        if index >= es.world.scene_count() {
            return Err(format!(
                "scene index {index} out of range (count: {})",
                es.world.scene_count()
            ));
        }

        es.world.set_scene_persistent(index, persistent);
        Ok(())
    }

    fn scene_swap(&self) -> Result<String, String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let removed = es.world.swap_scenes();
        Ok(serde_json::json!({
            "removed_scenes": removed,
            "remaining_count": es.world.scene_count(),
        })
        .to_string())
    }

    // --- Camera entity operations -----------------------------------------------

    fn camera_spawn(
        &self,
        label: &str,
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        fov: f32,
    ) -> Result<u64, String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let wp = rkf_core::WorldPosition::new(
            glam::IVec3::ZERO,
            Vec3::new(position[0], position[1], position[2]),
        );
        let entity = es.world.spawn_camera(label, wp, yaw, pitch, fov);
        Ok(entity.to_u64())
    }

    fn camera_list(&self) -> Result<String, String> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        let cameras = es.world.cameras();
        let mut result = Vec::new();
        for entity in &cameras {
            let pos = es
                .world
                .position(*entity)
                .map(|p| {
                    let v = p.to_vec3();
                    [v.x, v.y, v.z]
                })
                .unwrap_or([0.0; 3]);
            // Extract CameraComponent fields while the hecs::Ref is alive.
            let (label, fov, yaw, pitch, active) = es
                .world
                .get::<rkf_runtime::components::CameraComponent>(*entity)
                .map(|c| (c.label.clone(), c.fov_degrees, c.yaw, c.pitch, c.active))
                .unwrap_or_else(|_| (String::new(), 60.0, 0.0, 0.0, false));
            result.push(serde_json::json!({
                "entity_id": entity.to_u64(),
                "position": pos,
                "label": label,
                "fov_degrees": fov,
                "yaw": yaw,
                "pitch": pitch,
                "active": active,
            }));
        }

        Ok(serde_json::Value::Array(result).to_string())
    }

    fn camera_snap_to(&self, entity_id: u64) -> Result<(), String> {
        // Read camera data while holding editor_state lock, then release it.
        let (pos_vec3, yaw_rad, pitch_rad) = {
            let es = self
                .editor_state
                .lock()
                .map_err(|e| format!("lock poisoned: {e}"))?;

            let entity = es
                .world
                .find_entity_by_id(entity_id)
                .ok_or_else(|| format!("entity {entity_id} not found"))?;

            let pos = es
                .world
                .position(entity)
                .map_err(|e| format!("cannot read position: {e}"))?;

            let cam = es
                .world
                .get::<rkf_runtime::components::CameraComponent>(entity)
                .map_err(|_| format!("entity {entity_id} has no CameraComponent"))?;

            (pos.to_vec3(), cam.yaw.to_radians(), cam.pitch.to_radians())
        };

        // Set pending camera on SharedState to update the viewport camera.
        let mut state = self
            .state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;
        state.pending_camera = Some(PendingCamera {
            position: pos_vec3,
            yaw: yaw_rad,
            pitch: pitch_rad,
        });

        Ok(())
    }

    fn env_override(&self, property: &str, value: f32) -> Result<(), String> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| format!("lock poisoned: {e}"))?;

        match property {
            "sun.intensity" | "atmosphere.sun_intensity" => {
                es.environment.atmosphere.sun_intensity = value;
            }
            "fog.density" => {
                es.environment.fog.density = value;
            }
            "fog.enabled" => {
                es.environment.fog.enabled = value != 0.0;
            }
            "fog.start_distance" => {
                es.environment.fog.start_distance = value;
            }
            "fog.end_distance" => {
                es.environment.fog.end_distance = value;
            }
            "fog.height_falloff" => {
                es.environment.fog.height_falloff = value;
            }
            "clouds.enabled" => {
                es.environment.clouds.enabled = value != 0.0;
            }
            "clouds.coverage" => {
                es.environment.clouds.coverage = value;
            }
            "clouds.density" => {
                es.environment.clouds.density = value;
            }
            "post.bloom_enabled" | "post_process.bloom_enabled" => {
                es.environment.post_process.bloom_enabled = value != 0.0;
            }
            "post.bloom_intensity" | "post_process.bloom_intensity" => {
                es.environment.post_process.bloom_intensity = value;
            }
            "post.exposure" | "post_process.exposure" => {
                es.environment.post_process.exposure = value;
            }
            "post.contrast" | "post_process.contrast" => {
                es.environment.post_process.contrast = value;
            }
            "post.saturation" | "post_process.saturation" => {
                es.environment.post_process.saturation = value;
            }
            other => {
                return Err(format!(
                    "unknown environment property: {other}. \
                     Known properties: sun.intensity, fog.density, fog.enabled, \
                     fog.start_distance, fog.end_distance, fog.height_falloff, \
                     clouds.enabled, clouds.coverage, clouds.density, \
                     post.bloom_enabled, post.bloom_intensity, post.exposure, \
                     post.contrast, post.saturation"
                ));
            }
        }

        es.environment.mark_dirty();
        Ok(())
    }

    fn voxel_slice(
        &self,
        object_id: u32,
        y_coord: f32,
    ) -> AutomationResult<VoxelSliceResult> {
        // Submit request to the engine thread and poll for result.
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.voxel_slice_result = None;
            state.pending_voxel_slice = Some(VoxelSliceRequest { object_id, y_coord });
        }

        // Poll for up to 2 seconds.
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Ok(mut state) = self.state.lock() {
                if let Some(result) = state.voxel_slice_result.take() {
                    return Ok(result);
                }
            }
        }
        Err(AutomationError::EngineError(
            "voxel_slice timed out waiting for engine".into(),
        ))
    }

    fn sculpt_apply(
        &self,
        object_id: u32,
        position: [f32; 3],
        mode: &str,
        radius: f32,
        strength: f32,
        material_id: u16,
    ) -> AutomationResult<()> {
        // Validate mode.
        match mode {
            "add" | "subtract" | "smooth" => {}
            other => {
                return Err(AutomationError::InvalidParameter(format!(
                    "invalid sculpt mode: {other} (expected add/subtract/smooth)"
                )));
            }
        }

        // Submit request to the engine thread and poll for result.
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.mcp_sculpt_result = None;
            state.pending_mcp_sculpt = Some(McpSculptRequest {
                object_id,
                position: Vec3::new(position[0], position[1], position[2]),
                mode: mode.to_string(),
                radius,
                strength,
                material_id,
            });
        }

        // Poll for up to 2 seconds.
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Ok(mut state) = self.state.lock() {
                if let Some(result) = state.mcp_sculpt_result.take() {
                    return result.map_err(|e| AutomationError::EngineError(e));
                }
            }
        }
        Err(AutomationError::EngineError(
            "sculpt_apply timed out waiting for engine".into(),
        ))
    }

    fn object_shape(&self, object_id: u32) -> AutomationResult<ObjectShapeResult> {
        // Submit request to the engine thread and poll for result.
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.object_shape_result = None;
            state.pending_object_shape = Some(object_id);
        }

        // Poll for up to 2 seconds.
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Ok(mut state) = self.state.lock() {
                if let Some(result) = state.object_shape_result.take() {
                    return Ok(result);
                }
            }
        }
        Err(AutomationError::EngineError(
            "object_shape timed out waiting for engine".into(),
        ))
    }

    fn fix_sdfs(&self, object_id: u32) -> AutomationResult<()> {
        {
            let mut state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            state.fix_sdfs_result = None;
            state.pending_fix_sdfs = Some(object_id);
        }

        // fix_sdfs can be slow (full BFS over large grids) — poll for up to 10 s.
        for _ in 0..1000 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            if let Ok(mut state) = self.state.lock() {
                if let Some(result) = state.fix_sdfs_result.take() {
                    return result.map_err(AutomationError::EngineError);
                }
            }
        }
        Err(AutomationError::EngineError(
            "fix_sdfs timed out waiting for engine".into(),
        ))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::automation::AutomationApi;

    /// Create a test EditorAutomationApi with empty state.
    fn make_api() -> EditorAutomationApi {
        let state = Arc::new(Mutex::new(SharedState::new(4096, 0, 128, 128)));
        let editor_state = Arc::new(Mutex::new(EditorState::new()));
        EditorAutomationApi::new(state, editor_state)
    }

    /// Spawn a test object and return its object_id.
    fn spawn_test_object(api: &EditorAutomationApi) -> u32 {
        api.object_spawn("test_obj", "sphere", &[0.5], [0.0, 0.0, 0.0], 0)
            .unwrap()
    }

    #[test]
    fn automation_node_find_returns_json() {
        let api = make_api();
        let oid = spawn_test_object(&api);
        let result = api.node_find(oid, "test_obj").unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(json["name"], "test_obj");
        assert!(json["sdf_source"].as_str().unwrap().contains("Analytical"));
    }

    #[test]
    fn automation_node_add_child_adds_to_tree() {
        let api = make_api();
        let oid = spawn_test_object(&api);
        api.node_add_child(oid, "test_obj", "box", &[0.3, 0.3, 0.3], "child_box", 1)
            .unwrap();

        let result = api.node_find(oid, "test_obj").unwrap();
        let json: serde_json::Value = serde_json::from_str(&result).unwrap();
        let children = json["children"].as_array().unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], "child_box");
    }

    #[test]
    fn automation_scene_create_adds_scene() {
        let api = make_api();
        let result = api.scene_list().unwrap();
        let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(scenes.len(), 1);

        let idx = api.scene_create("level2").unwrap();
        assert_eq!(idx, 1);

        let result = api.scene_list().unwrap();
        let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[1]["name"], "level2");
    }

    #[test]
    fn automation_scene_set_active_changes_target() {
        let api = make_api();
        api.scene_create("second").unwrap();
        api.scene_set_active(1).unwrap();

        let result = api.scene_list().unwrap();
        let scenes: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(scenes[1]["active"], true);
        assert_eq!(scenes[0]["active"], false);
    }

    #[test]
    fn automation_camera_spawn_creates_entity() {
        let api = make_api();
        let cam_id = api
            .camera_spawn("Main", [1.0, 2.0, 3.0], 45.0, -10.0, 75.0)
            .unwrap();
        assert!(cam_id != 0); // Should get a valid non-zero ID

        let list = api.camera_list().unwrap();
        let cameras: Vec<serde_json::Value> = serde_json::from_str(&list).unwrap();
        assert_eq!(cameras.len(), 1);
        assert_eq!(cameras[0]["label"], "Main");
        assert!((cameras[0]["fov_degrees"].as_f64().unwrap() - 75.0).abs() < 0.01);
    }

    #[test]
    fn automation_camera_list_includes_spawned() {
        let api = make_api();
        api.camera_spawn("CamA", [0.0; 3], 0.0, 0.0, 60.0).unwrap();
        api.camera_spawn("CamB", [5.0, 0.0, 0.0], 90.0, 0.0, 90.0)
            .unwrap();

        let list = api.camera_list().unwrap();
        let cameras: Vec<serde_json::Value> = serde_json::from_str(&list).unwrap();
        assert_eq!(cameras.len(), 2);
        let labels: Vec<&str> = cameras
            .iter()
            .map(|c| c["label"].as_str().unwrap())
            .collect();
        assert!(labels.contains(&"CamA"));
        assert!(labels.contains(&"CamB"));
    }
}
