//! AutomationApi trait implementation for EditorAutomationApi.
//!
//! Large method bodies are delegated to `api_helpers.rs` to keep this file under 900 lines.

use std::sync::{Arc, Mutex};
use std::time::Instant;
use glam::Vec3;
use image::ImageEncoder;
use rkf_core::automation::*;

use super::*;

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
        self.scene_graph_impl()
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

    fn scene_load(&self, path: &str) -> AutomationResult<()> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
        es.pending_open = true;
        es.pending_open_path = Some(path.to_string());
        Ok(())
    }

    fn scene_save(&self, path: &str) -> AutomationResult<()> {
        let mut es = self
            .editor_state
            .lock()
            .map_err(|e| AutomationError::EngineError(format!("lock poisoned: {e}")))?;
        es.pending_save = true;
        es.pending_save_path = Some(path.to_string());
        Ok(())
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
        self.execute_command_impl(command)
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
        self.node_add_child_impl(object_id, parent_node, child_primitive, params, name, material_id)
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
        self.env_override_impl(property, value)
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
        self.sculpt_apply_impl(object_id, position, mode, radius, strength, material_id)
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
