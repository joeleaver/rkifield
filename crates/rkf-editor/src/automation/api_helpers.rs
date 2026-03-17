//! Large method bodies extracted from the AutomationApi impl to keep api_impl.rs under 900 lines.
//!
//! These are `pub(super)` helpers on `EditorAutomationApi`; the trait impl in
//! `api_impl.rs` delegates to them.

use glam::Vec3;
use rkf_core::automation::*;

use super::*;

impl EditorAutomationApi {

    pub(super) fn execute_command_impl(&self, command: &str) -> AutomationResult<String> {
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
            ["voxelize", id_str] => {
                let object_id: u32 = id_str.parse().map_err(|_| {
                    AutomationError::InvalidParameter(format!("invalid object id: {id_str}"))
                })?;
                let mut es = self
                    .editor_state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                let entity_uuid = es.world.find_by_sdf_id(object_id)
                    .ok_or_else(|| AutomationError::EngineError(
                        format!("object {object_id} not found")
                    ))?;
                // MCP uses a default voxel_size of 0 to signal auto-compute.
                es.pending_convert_to_voxel = Some((entity_uuid, 0.0));
                Ok(format!("queued voxelization for object {object_id}"))
            }
            ["voxelize"] => Err(AutomationError::InvalidParameter(
                "usage: voxelize <object_id>".to_string(),
            )),
            ["save"] => {
                let mut es = self
                    .editor_state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                es.pending_save = true;
                Ok("queued scene save".to_string())
            }
            ["save", path] => {
                let mut es = self
                    .editor_state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                es.pending_save = true;
                es.pending_save_path = Some(path.to_string());
                Ok(format!("queued scene save to {path}"))
            }
            ["open", path] => {
                let mut es = self
                    .editor_state
                    .lock()
                    .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
                es.pending_open = true;
                es.pending_open_path = Some(path.to_string());
                Ok(format!("queued scene open: {path}"))
            }
            ["open"] => Err(AutomationError::InvalidParameter(
                "usage: open <scene_path>".to_string(),
            )),
            _ => Err(AutomationError::InvalidParameter(format!(
                "unknown command: {command}"
            ))),
        }
    }

    pub(super) fn node_add_child_impl(
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

        let entity = es.world.find_by_sdf_id(object_id)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        es.world.add_child_node(entity, parent_node, child)
            .map_err(|e| format!("{e}"))
    }


    pub(super) fn env_override_impl(&self, property: &str, value: f32) -> Result<(), String> {
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


    pub(super) fn sculpt_apply_impl(
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


    pub(super) fn scene_graph_impl(&self) -> AutomationResult<SceneGraphSnapshot> {
        let es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;

        // Read from hecs — the authoritative source.
        let render_scene = es.world.build_render_scene();
        let mut entities = Vec::new();
        for obj in &render_scene.objects {
            let p = obj.position;
            let r = obj.rotation;
            let entity_type = match &obj.root_node.sdf_source {
                rkf_core::scene_node::SdfSource::None => "entity",
                _ => "sdf_object",
            };
            // Look up UUID for this SDF object.
            let uuid_str = es.world.find_by_sdf_id(obj.id)
                .map(|u| u.to_string())
                .unwrap_or_else(|| format!("sdf:{}", obj.id));
            let parent_uuid_str = obj.parent_id.and_then(|pid| {
                es.world.find_by_sdf_id(pid).map(|u| u.to_string())
            });
            entities.push(EntityNode {
                id: uuid_str,
                name: obj.name.clone(),
                parent: parent_uuid_str,
                entity_type: entity_type.to_string(),
                transform: [p.x, p.y, p.z, r.x, r.y, r.z, r.w, obj.scale.x, obj.scale.y, obj.scale.z],
            });
        }

        // Include light entities from light editor.
        use crate::light_editor::SceneLightType;
        for (idx, light) in es.light_editor.all_lights().iter().enumerate() {
            let (light_type, type_name) = match light.light_type {
                SceneLightType::Point => ("point_light", "Point"),
                SceneLightType::Spot => ("spot_light", "Spot"),
            };
            let p = light.position;
            entities.push(EntityNode {
                id: format!("light:{}", light.id),
                name: format!("{} Light {}", type_name, idx + 1),
                parent: None,
                entity_type: light_type.to_string(),
                transform: [p.x, p.y, p.z, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            });
        }

        Ok(SceneGraphSnapshot { entities })
    }

}
