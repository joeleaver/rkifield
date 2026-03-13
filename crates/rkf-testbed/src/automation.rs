//! Testbed automation API — implements [`AutomationApi`] backed by shared engine state.
//!
//! [`SharedState`] is updated by the render loop each frame. [`TestbedAutomationApi`]
//! reads it to serve MCP tool requests over IPC.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::Vec3;
use image::ImageEncoder;
use rkf_core::automation::*;

/// Camera position + orientation set via MCP camera_set tool.
pub struct PendingCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
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
    /// Last rendered frame pixels (RGBA, internal resolution).
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
}

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
        }
    }
}

/// Testbed implementation of [`AutomationApi`] backed by shared engine state.
pub struct TestbedAutomationApi {
    state: Arc<Mutex<SharedState>>,
}

impl TestbedAutomationApi {
    /// Create a new testbed automation API backed by the given shared state.
    pub fn new(state: Arc<Mutex<SharedState>>) -> Self {
        Self { state }
    }
}

impl AutomationApi for TestbedAutomationApi {
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

        // Wait for the render loop to fulfill the request (up to ~500ms).
        for _ in 0..50 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            let state = self
                .state
                .lock()
                .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
            if !state.screenshot_requested {
                // Render loop has fulfilled the request.
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
            "screenshot timeout — render loop did not respond within 500ms".into(),
        ))
    }

    fn camera_state(&self) -> AutomationResult<CameraSnapshot> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        // Convert yaw/pitch to unit quaternion
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
            pass_timings: HashMap::from([("ray_march".to_string(), state.frame_time_ms)]),
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
        chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
        // Testbed has a single sphere at origin with radius 0.5
        if chunk != [0, 0, 0] {
            return Ok(SpatialQueryResult {
                distance: f32::MAX,
                material_id: 0,
                inside: false,
            });
        }
        let pos = Vec3::new(local[0], local[1], local[2]);
        let distance = pos.length() - 0.5;
        Ok(SpatialQueryResult {
            distance,
            material_id: if distance < 0.0 { 1 } else { 0 },
            inside: distance < 0.0,
        })
    }

    // --- Not yet implemented in testbed ---

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        Err(AutomationError::NotImplemented("scene_graph"))
    }

    fn entity_inspect(&self, _entity_id: &str) -> AutomationResult<EntitySnapshot> {
        Err(AutomationError::NotImplemented("entity_inspect"))
    }

    fn read_log(&self, _lines: usize) -> AutomationResult<Vec<LogEntry>> {
        Err(AutomationError::NotImplemented("read_log"))
    }

    // --- Mutation methods (not supported in testbed) ---

    fn entity_spawn(&self, _def: EntityDef) -> AutomationResult<String> {
        Err(AutomationError::NotImplemented("entity_spawn"))
    }

    fn entity_despawn(&self, _entity_id: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_despawn"))
    }

    fn entity_set_component(
        &self,
        _entity_id: &str,
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
                        "invalid debug mode: {mode_str} (expected 0-6)"
                    ))
                })?;
                if mode > 6 {
                    return Err(AutomationError::InvalidParameter(format!(
                        "debug mode {mode} out of range (expected 0-6)"
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
                    _ => "unknown",
                };
                Ok(format!("debug mode set to {mode} ({mode_name})"))
            }
            ["debug_mode"] => Err(AutomationError::InvalidParameter(
                "usage: debug_mode <0-6>".to_string(),
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
                Ok(format!("camera set to ({x}, {y}, {z}) yaw={yaw_deg} pitch={pitch_deg}"))
            }
            ["camera_set", ..] => Err(AutomationError::InvalidParameter(
                "usage: camera_set <x> <y> <z> <yaw_deg> <pitch_deg>".to_string(),
            )),
            _ => Err(AutomationError::InvalidParameter(format!(
                "unknown command: {command}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_shared_state() -> Arc<Mutex<SharedState>> {
        Arc::new(Mutex::new(SharedState::new(4096, 100, 4, 4)))
    }

    #[test]
    fn camera_state_returns_position() {
        let state = make_shared_state();
        {
            let mut s = state.lock().unwrap();
            s.camera_position = Vec3::new(1.0, 2.0, 3.0);
            s.camera_fov = 75.0;
        }
        let api = TestbedAutomationApi::new(state);
        let cam = api.camera_state().unwrap();
        assert_eq!(cam.local, [1.0, 2.0, 3.0]);
        assert!((cam.fov_degrees - 75.0).abs() < 1e-6);
    }

    #[test]
    fn render_stats_returns_frame_time() {
        let state = make_shared_state();
        state.lock().unwrap().frame_time_ms = 16.6;
        let api = TestbedAutomationApi::new(state);
        let stats = api.render_stats().unwrap();
        assert!((stats.frame_time_ms - 16.6).abs() < 1e-9);
    }

    #[test]
    fn brick_pool_stats_matches_state() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        let stats = api.brick_pool_stats().unwrap();
        assert_eq!(stats.capacity, 4096);
        assert_eq!(stats.allocated, 100);
        assert_eq!(stats.free_list_size, 3996);
    }

    #[test]
    fn asset_status_reports_one_chunk() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        let status = api.asset_status().unwrap();
        assert_eq!(status.loaded_chunks, 1);
        assert_eq!(status.pending_uploads, 0);
    }

    #[test]
    fn spatial_query_inside_sphere() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        let result = api.spatial_query([0, 0, 0], [0.0, 0.0, 0.0]).unwrap();
        assert!(result.inside);
        assert!(result.distance < 0.0);
    }

    #[test]
    fn spatial_query_outside_sphere() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        let result = api.spatial_query([0, 0, 0], [2.0, 0.0, 0.0]).unwrap();
        assert!(!result.inside);
        assert!(result.distance > 0.0);
    }

    #[test]
    fn spatial_query_wrong_chunk() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        let result = api.spatial_query([1, 0, 0], [0.0, 0.0, 0.0]).unwrap();
        assert!(!result.inside);
        assert_eq!(result.distance, f32::MAX);
    }

    #[test]
    fn screenshot_encodes_png() {
        let state = make_shared_state();
        {
            let mut s = state.lock().unwrap();
            // Fill with a 4x4 red image and pre-clear the request flag
            // (simulates render loop having already fulfilled the capture).
            for pixel in s.frame_pixels.chunks_exact_mut(4) {
                pixel[0] = 255; // R
                pixel[1] = 0; // G
                pixel[2] = 0; // B
                pixel[3] = 255; // A
            }
        }
        let api = TestbedAutomationApi::new(Arc::clone(&state));
        // Simulate: spawn a thread that clears the flag after a short delay
        // (mimics what the render loop does).
        let state2 = Arc::clone(&state);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            state2.lock().unwrap().screenshot_requested = false;
        });
        let png = api.screenshot(4, 4).unwrap();
        // PNG magic bytes
        assert_eq!(&png[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[test]
    fn execute_command_debug_mode() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(Arc::clone(&state));
        let result = api.execute_command("debug_mode 3").unwrap();
        assert!(result.contains("material IDs"));
        assert_eq!(state.lock().unwrap().pending_debug_mode, Some(3));
    }

    #[test]
    fn execute_command_debug_mode_out_of_range() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        assert!(api.execute_command("debug_mode 9").is_err());
    }

    #[test]
    fn execute_command_unknown() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        assert!(api.execute_command("unknown_cmd").is_err());
    }

    #[test]
    fn mutation_methods_not_implemented() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(state);
        assert!(api.entity_spawn(EntityDef { name: "test".into(), components: vec![] }).is_err());
        assert!(api.entity_despawn("00000000-0000-0000-0000-000000000001").is_err());
        assert!(api.scene_load("foo").is_err());
    }

    #[test]
    fn camera_set_stores_pending() {
        let state = make_shared_state();
        let api = TestbedAutomationApi::new(Arc::clone(&state));
        let quat = glam::Quat::from_euler(glam::EulerRot::YXZ, 1.0, 0.5, 0.0);
        api.camera_set([0, 0, 0], [1.0, 2.0, 3.0], [quat.x, quat.y, quat.z, quat.w]).unwrap();
        let s = state.lock().unwrap();
        let cam = s.pending_camera.as_ref().unwrap();
        assert!((cam.position.x - 1.0).abs() < 1e-5);
        assert!((cam.position.y - 2.0).abs() < 1e-5);
        assert!((cam.position.z - 3.0).abs() < 1e-5);
        assert!((cam.yaw - 1.0).abs() < 1e-4);
        assert!((cam.pitch - 0.5).abs() < 1e-4);
    }
}
