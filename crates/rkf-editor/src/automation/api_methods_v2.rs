// v2 object-centric AutomationApi methods — included into impl AutomationApi block in api_impl.rs.
// Do NOT add an `impl` wrapper here; this file is `include!`'d directly.

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

