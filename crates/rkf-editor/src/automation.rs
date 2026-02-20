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
    /// Ring buffer of recent log entries for MCP `read_log` tool.
    pub log_entries: VecDeque<LogEntry>,
    /// Engine start time for log timestamps.
    pub start_time: Instant,
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
            log_entries: VecDeque::with_capacity(MAX_LOG_ENTRIES),
            start_time: Instant::now(),
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
}

impl AutomationApi for EditorAutomationApi {
    fn screenshot(&self, _width: u32, _height: u32) -> AutomationResult<Vec<u8>> {
        let state = self
            .state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

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

        Ok(png_bytes)
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
        chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
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

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        fn collect_nodes(
            node: &crate::scene_tree::SceneNode,
            parent_id: Option<u64>,
            out: &mut Vec<EntityNode>,
        ) {
            let entity_type = if node.asset_path.is_some() {
                "sdf_object"
            } else {
                "entity"
            };
            let p = node.position;
            let r = node.rotation;
            out.push(EntityNode {
                id: node.entity_id,
                name: node.name.clone(),
                parent: parent_id,
                entity_type: entity_type.to_string(),
                transform: [p.x, p.y, p.z, r.x, r.y, r.z, r.w, node.scale.x, node.scale.y, node.scale.z],
            });
            for child in &node.children {
                collect_nodes(child, Some(node.entity_id), out);
            }
        }

        let mut entities = Vec::new();
        for root in &es.scene_tree.roots {
            collect_nodes(root, None, &mut entities);
        }

        // Include light entities from light editor
        use crate::light_editor::EditorLightType;
        for (idx, light) in es.light_editor.all_lights().iter().enumerate() {
            let (light_type, type_name) = match light.light_type {
                EditorLightType::Point => ("point_light", "Point"),
                EditorLightType::Spot => ("spot_light", "Spot"),
                EditorLightType::Directional => ("directional_light", "Directional"),
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

        let node = es
            .scene_tree
            .find_node(entity_id)
            .ok_or(AutomationError::EntityNotFound(entity_id))?;

        let mut components = HashMap::new();
        components.insert(
            "scene_node".to_string(),
            serde_json::json!({
                "visible": node.visible,
                "expanded": node.expanded,
                "selected": node.selected,
                "children": node.children.len(),
            }),
        );

        Ok(EntitySnapshot {
            id: node.entity_id,
            name: node.name.clone(),
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
                Ok(format!(
                    "camera set to ({x}, {y}, {z}) yaw={yaw_deg} pitch={pitch_deg}"
                ))
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
