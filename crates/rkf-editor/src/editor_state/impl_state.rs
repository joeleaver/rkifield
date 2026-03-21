//! `impl EditorState` methods.

use glam::Vec3;
use super::*;


impl EditorState {
    /// Create a new editor state with default values and fly-mode camera
    /// positioned to match the engine's initial viewpoint.
    pub fn new() -> Self {
        let mut world = World::new("editor");

        // Spawn the editor's own camera entity with EditorCameraMarker.
        let editor_cam_pos = rkf_core::WorldPosition::new(
            glam::IVec3::ZERO,
            Vec3::new(0.0, 2.5, 5.0),
        );
        let editor_cam_uuid = world.spawn_camera(
            "Editor Camera",
            editor_cam_pos,
            0.0,   // yaw
            -0.15_f32.to_degrees(), // pitch (radians→degrees for CameraComponent)
            70.0,  // fov_degrees
            None,
        );
        // Insert EditorCameraMarker on the entity (EnvironmentSettings added by spawn_camera).
        if let Some(ecs_entity) = world.ecs_entity_for(editor_cam_uuid) {
            let _ = world.ecs_mut().insert_one(ecs_entity, rkf_runtime::components::EditorCameraMarker);
        }

        let camera_control = CameraControlState {
            mode: CameraMode::Fly,
            fly_speed: 5.0,
            ..CameraControlState::default()
        };

        Self {
            mode: EditorMode::Default,
            camera_control,
            editor_input: InputState::new(),
            selected_entity: None,
            selected_properties: None,
            gizmo: GizmoState::new(),
            sculpt: SculptState::new(),
            paint: PaintState::new(),
            light_editor: LightManager::new(),
            overlay_config: OverlayConfig::default(),
            debug_viz: DebugOverlay::new(),
            frame_time_history: FrameTimeHistory::new(),
            undo: UndoStack::new(100),
            material_browser: MaterialBrowserState::default(),
            unsaved_changes: UnsavedChangesState::new(),
            recent_files: RecentFiles::new(),
            current_scene_path: None,
            current_project: None,
            current_project_path: None,
            pending_new_project: false,
            pending_open_project: false,
            pending_open_project_path: None,
            pending_save_project: false,
            viewport: ViewportRect::default(),
            left_panel_width: 251,  // 250px content + 1px border-right
            right_panel_width: 301, // 300px content + 1px border-left
            top_bar_height: 37,    // titlebar 36px + 1px border
            bottom_bar_height: 25, // status bar 24px + 1px border-top
            pending_debug_mode: None,
            debug_mode: 0,
            wants_exit: false,
            pending_open: false,
            pending_open_path: None,
            pending_save: false,
            pending_save_as: false,
            pending_save_path: None,
            pending_spawn: None,
            pending_spawn_camera: false,
            pending_spawn_point_light: false,
            pending_spawn_spot_light: false,
            pending_place_model: None,
            pending_delete: false,
            pending_duplicate: false,
            pending_undo: false,
            pending_redo: false,
            show_grid: false,
            show_shortcuts: false,
            pending_play_start: false,
            pending_play_stop: false,
            pending_drag: false,
            pending_minimize: false,
            pending_maximize: false,
            pending_convert_to_voxel: None,
            pending_remap_material: None,
            pending_set_primitive_material: None,
            pending_sculpt_edits: Vec::new(),
            sculpt_undo_accumulator: None,
            pending_sculpt_undo: None,
            pending_paint_edits: Vec::new(),
            paint_undo_accumulator: None,
            pending_paint_undo: None,
            world,
            editor_camera_entity: Some(editor_cam_uuid),
            viewport_camera: None,
            piloting: None,
            linked_env_camera: None,
            last_env_profile_key: None,
        }
    }

    /// Recompute the viewport rect from current panel sizes and window dimensions.
    ///
    /// Returns `true` if the viewport changed (callers should rebuild GPU passes).
    pub fn compute_viewport(&mut self, window_width: u32, window_height: u32) -> bool {
        let new = ViewportRect {
            x: self.left_panel_width,
            y: self.top_bar_height,
            width: window_width
                .saturating_sub(self.left_panel_width + self.right_panel_width)
                .max(64),
            height: window_height
                .saturating_sub(self.top_bar_height + self.bottom_bar_height)
                .max(64),
        };
        let changed = new != self.viewport;
        self.viewport = new;
        changed
    }

    /// Update camera from current input state.
    ///
    /// Reads position/yaw/pitch from the editor camera entity, applies
    /// `CameraControlState` controls, then writes the result back to the entity.
    pub fn update_camera(&mut self, dt: f32) {
        let uuid = match self.editor_camera_entity {
            Some(u) => u,
            None => return,
        };

        let ecs_entity = self.world.ecs_entity_for(uuid);

        let mut position = ecs_entity
            .and_then(|e| {
                self.world.ecs_ref()
                    .get::<&rkf_runtime::components::Transform>(e)
                    .ok()
                    .map(|t| t.position.to_vec3())
            })
            .unwrap_or(Vec3::new(0.0, 2.5, 5.0));

        let (mut yaw, mut pitch) = ecs_entity
            .and_then(|e| {
                self.world.ecs_ref()
                    .get::<&rkf_runtime::components::CameraComponent>(e)
                    .ok()
                    .map(|c| (c.yaw.to_radians(), c.pitch.to_radians()))
            })
            .unwrap_or((0.0, 0.0));

        // Apply control state.
        self.camera_control.update(&self.editor_input, dt, &mut position, &mut yaw, &mut pitch);

        // Write back to entity.
        let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, position);
        let _ = self.world.set_position(uuid, wp);
        if let Some(e) = ecs_entity {
            if let Ok(mut cam) = self.world.ecs_mut()
                .get::<&mut rkf_runtime::components::CameraComponent>(e)
            {
                cam.yaw = yaw.to_degrees();
                cam.pitch = pitch.to_degrees();
            }
        }
    }

    /// Read fov_degrees from the editor camera entity's CameraComponent.
    pub fn editor_camera_fov_degrees(&self) -> f32 {
        self.editor_camera_component_field(|c| c.fov_degrees).unwrap_or(70.0)
    }

    /// Read near clip from the editor camera entity's CameraComponent.
    pub fn editor_camera_near(&self) -> f32 {
        self.editor_camera_component_field(|c| c.near).unwrap_or(0.1)
    }

    /// Read far clip from the editor camera entity's CameraComponent.
    pub fn editor_camera_far(&self) -> f32 {
        self.editor_camera_component_field(|c| c.far).unwrap_or(1000.0)
    }

    /// Read fov_y in radians from the editor camera entity.
    pub fn editor_camera_fov_y(&self) -> f32 {
        self.editor_camera_fov_degrees().to_radians()
    }

    /// Helper to read a field from the editor camera entity's CameraComponent.
    fn editor_camera_component_field<T>(
        &self,
        f: impl FnOnce(&rkf_runtime::components::CameraComponent) -> T,
    ) -> Option<T> {
        let uuid = self.editor_camera_entity?;
        let ecs_entity = self.world.ecs_entity_for(uuid)?;
        let cam = self.world.ecs_ref()
            .get::<&rkf_runtime::components::CameraComponent>(ecs_entity)
            .ok()?;
        Some(f(&cam))
    }

    /// Extract a `CameraSnapshot` from the editor camera entity.
    ///
    /// Reads position from Transform, yaw/pitch/fov/near/far from CameraComponent.
    pub fn extract_camera_snapshot(&self) -> CameraSnapshot {
        self.extract_camera_snapshot_for(
            self.editor_camera_entity.expect("editor camera entity must exist"),
        )
    }

    /// Extract a `CameraSnapshot` from any camera entity by UUID.
    pub fn extract_camera_snapshot_for(&self, uuid: uuid::Uuid) -> CameraSnapshot {
        let ecs_entity = self.world.ecs_entity_for(uuid);
        let position = ecs_entity
            .and_then(|e| {
                self.world.ecs_ref()
                    .get::<&rkf_runtime::components::Transform>(e)
                    .ok()
                    .map(|t| t.position.to_vec3())
            })
            .unwrap_or(Vec3::new(0.0, 2.5, 5.0));
        let (yaw, pitch, fov_degrees, near, far) = ecs_entity
            .and_then(|e| {
                self.world.ecs_ref()
                    .get::<&rkf_runtime::components::CameraComponent>(e)
                    .ok()
                    .map(|c| (c.yaw.to_radians(), c.pitch.to_radians(), c.fov_degrees, c.near, c.far))
            })
            .unwrap_or((0.0, 0.0, 70.0, 0.1, 1000.0));
        CameraSnapshot { position, yaw, pitch, fov_degrees, near, far }
    }

    /// Write a field on the editor camera entity's CameraComponent.
    pub fn set_editor_camera_component_field(
        &mut self,
        f: impl FnOnce(&mut rkf_runtime::components::CameraComponent),
    ) {
        if let Some(uuid) = self.editor_camera_entity {
            if let Some(ecs_entity) = self.world.ecs_entity_for(uuid) {
                if let Ok(mut cam) = self.world.ecs_mut()
                    .get::<&mut rkf_runtime::components::CameraComponent>(ecs_entity)
                {
                    f(&mut cam);
                }
            }
        }
    }

    /// Re-spawn the editor camera entity after a world clear.
    ///
    /// `saved` must be captured BEFORE `world.clear()` since the old entity
    /// will be gone. If `None`, uses defaults.
    pub fn respawn_editor_camera(&mut self, saved: Option<CameraSnapshot>) {
        let snap = saved.unwrap_or(CameraSnapshot {
            position: Vec3::new(0.0, 2.5, 5.0),
            yaw: 0.0,
            pitch: -0.15,
            fov_degrees: 70.0,
            near: 0.1,
            far: 1000.0,
        });

        let editor_cam_pos = rkf_core::WorldPosition::new(
            glam::IVec3::ZERO,
            snap.position,
        );
        let editor_cam_uuid = self.world.spawn_camera(
            "Editor Camera",
            editor_cam_pos,
            snap.yaw.to_degrees(),
            snap.pitch.to_degrees(),
            snap.fov_degrees,
            None,
        );
        if let Some(ecs_entity) = self.world.ecs_entity_for(editor_cam_uuid) {
            let _ = self.world.ecs_mut().insert(
                ecs_entity,
                (
                    rkf_runtime::components::EditorCameraMarker,
                    rkf_runtime::environment::EnvironmentSettings::default(),
                ),
            );
            if let Ok(mut cam) = self.world.ecs_mut()
                .get::<&mut rkf_runtime::components::CameraComponent>(ecs_entity)
            {
                cam.near = snap.near;
                cam.far = snap.far;
            }
        }
        self.editor_camera_entity = Some(editor_cam_uuid);
    }

    /// Name of the current debug visualization mode (empty for normal shading).
    pub fn debug_mode_name(&self) -> &'static str {
        match self.debug_mode {
            0 => "",
            1 => "Normals",
            2 => "Positions",
            3 => "Material IDs",
            4 => "Diffuse",
            5 => "Specular",
            6 => "GI Only",
            _ => "Debug",
        }
    }

    /// Reset per-frame input deltas (mouse delta, scroll) after processing.
    pub fn reset_frame_deltas(&mut self) {
        self.editor_input.mouse_delta = glam::Vec2::ZERO;
        self.editor_input.scroll_delta = 0.0;
        self.editor_input.keys_just_pressed.clear();
    }

    /// Apply a single undo/redo action to the world.
    ///
    /// When `reverse` is `true` (undo), applies old values.
    /// When `reverse` is `false` (redo), applies new values.
    pub fn apply_undo_action(&mut self, action: &crate::undo::UndoAction, reverse: bool) {
        use crate::undo::UndoActionKind;

        match &action.kind {
            UndoActionKind::Transform {
                entity_id,
                old_pos, old_rot, old_scale,
                new_pos, new_rot, new_scale,
            } => {
                let (pos, rot, scale) = if reverse {
                    (*old_pos, *old_rot, *old_scale)
                } else {
                    (*new_pos, *new_rot, *new_scale)
                };
                if self.world.is_alive(*entity_id) {
                    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
                    let _ = self.world.set_position(*entity_id, wp);
                    let _ = self.world.set_rotation(*entity_id, rot);
                    let _ = self.world.set_scale(*entity_id, scale);
                }
            }
            UndoActionKind::SpawnEntity { entity_id } => {
                if reverse {
                    // Undo spawn = despawn
                    if self.world.is_alive(*entity_id) {
                        let _ = self.world.despawn(*entity_id);
                    }
                }
                // Redo spawn would need stored object data — not supported yet.
            }
            UndoActionKind::DespawnEntity { entity_id: _ } => {
                // Undo despawn would need stored object data — not supported yet.
            }
            UndoActionKind::SculptStroke { object_id, geometry_snapshots } => {
                if reverse {
                    // Undo: queue geometry restoration for the render loop.
                    self.pending_sculpt_undo = Some((*object_id, geometry_snapshots.clone()));
                }
                // Redo is not supported for sculpt strokes.
            }
            UndoActionKind::PaintStroke { object_id, geometry_snapshots } => {
                if reverse {
                    // Undo: queue geometry restoration for the render loop (same as sculpt undo).
                    self.pending_paint_undo = Some((*object_id, geometry_snapshots.clone()));
                }
            }
            _ => {
                // VoxelEdit, PropertyChange, EnvironmentChange — future work.
            }
        }
    }

    /// Pick a scene object via ray-AABB intersection (CPU fallback).
    ///
    /// Returns the nearest hit object's id, or `None`. This is a rough
    /// approximation — the primary pick path uses GPU readback from the
    /// material G-buffer (see `pending_pick`/`pick_result` in SharedState).
    pub fn pick_object_aabb(
        &self,
        pixel_x: f32,
        pixel_y: f32,
        vp_width: f32,
        vp_height: f32,
    ) -> Option<uuid::Uuid> {
        let snap = self.extract_camera_snapshot();
        let (ray_o, ray_d) = crate::camera::screen_to_ray_snapshot(
            &snap,
            pixel_x,
            pixel_y,
            vp_width,
            vp_height,
        );

        let render_scene = self.world.build_render_scene();
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&render_scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        let mut best_t = f32::MAX;
        let mut best_id = None;

        for obj in &render_scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let smin = obj.aabb.min * wt.scale;
            let smax = obj.aabb.max * wt.scale;
            let corners = [
                glam::Vec3::new(smin.x, smin.y, smin.z), glam::Vec3::new(smax.x, smin.y, smin.z),
                glam::Vec3::new(smin.x, smax.y, smin.z), glam::Vec3::new(smax.x, smax.y, smin.z),
                glam::Vec3::new(smin.x, smin.y, smax.z), glam::Vec3::new(smax.x, smin.y, smax.z),
                glam::Vec3::new(smin.x, smax.y, smax.z), glam::Vec3::new(smax.x, smax.y, smax.z),
            ];
            let mut wmin = glam::Vec3::splat(f32::MAX);
            let mut wmax = glam::Vec3::splat(f32::MIN);
            for c in &corners {
                let r = wt.rotation * *c + wt.position;
                wmin = wmin.min(r);
                wmax = wmax.max(r);
            }
            if let Some(t) = ray_aabb_distance(ray_o, ray_d, wmin, wmax) {
                if t < best_t {
                    best_t = t;
                    best_id = self.world.find_by_sdf_id(obj.id);
                }
            }
        }

        best_id
    }

    /// Load a v3 scene file and populate the editor state from it.
    ///
    /// Spawns entities into hecs via `scene_file_v3::load_scene()`,
    /// rebuilds World entity tracking, and restores editor state
    /// (camera, environment, lights) from properties.
    pub fn load_scene(&mut self, path: &str) -> Result<(), String> {
        use rkf_runtime::behavior::{GameplayRegistry, StableIdIndex};
        use rkf_runtime::scene_file_v3;

        let ron_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read scene file '{path}': {e}"))?;
        let scene_v3 = scene_file_v3::deserialize_scene_v3(&ron_str)
            .map_err(|e| format!("Failed to parse scene file: {e}"))?;

        let saved_snap = if self.editor_camera_entity.is_some() {
            Some(self.extract_camera_snapshot())
        } else {
            None
        };
        self.world.clear();
        self.respawn_editor_camera(saved_snap);

        let mut registry = GameplayRegistry::new();
        rkf_runtime::engine_register(&mut registry);
        let mut stable_index = StableIdIndex::new();
        scene_file_v3::load_scene(&scene_v3, self.world.ecs_mut(), &mut stable_index, &registry);
        self.world.rebuild_entity_tracking_from_ecs();

        // Restore editor camera environment settings.
        if let Some(env_str) = scene_v3.properties.get("editor_environment") {
            if let Ok(settings) = ron::from_str::<rkf_runtime::environment::EnvironmentSettings>(env_str) {
                if let Some(editor_cam) = self.editor_camera_entity {
                    if let Some(ee) = self.world.ecs_entity_for(editor_cam) {
                        let _ = self.world.ecs_mut().insert_one(ee, settings);
                    }
                }
            }
        }
        // Restore linked environment camera.
        if let Some(linked_str) = scene_v3.properties.get("linked_env_camera") {
            if let Ok(uuid) = uuid::Uuid::parse_str(linked_str) {
                self.linked_env_camera = Some(uuid);
            }
        } else {
            self.linked_env_camera = None;
        }
        self.last_env_profile_key = None;

        if let Some(s) = scene_v3.properties.get("lights") {
            if let Ok(lights) = ron::from_str::<Vec<crate::light_editor::SceneLight>>(s) {
                self.light_editor.replace_lights(lights);
            }
        }

        self.current_scene_path = Some(path.to_string());
        self.unsaved_changes.mark_saved();

        Ok(())
    }

    /// Save the current editor state to a v3 scene file.
    ///
    /// Serializes all hecs entities plus editor state (camera, environment,
    /// lights) into a `SceneFileV3`.
    pub fn save_current_scene(&self) -> rkf_runtime::scene_file_v3::SceneFileV3 {
        use rkf_runtime::behavior::{GameplayRegistry, StableIdIndex};
        use rkf_runtime::scene_file_v3;

        // Build a stable index from current hecs entities.
        let mut stable_index = StableIdIndex::new();
        for entity_ref in self.world.ecs_ref().iter() {
            let e = entity_ref.entity();
            if let Ok(sid) = self.world.ecs_ref().get::<&rkf_runtime::behavior::StableId>(e) {
                stable_index.insert(sid.0, e);
            }
        }

        let mut registry = GameplayRegistry::new();
        rkf_runtime::engine_register(&mut registry);
        let mut scene_v3 = scene_file_v3::save_scene(self.world.ecs_ref(), &stable_index, &registry);

        // Filter out internal entities — editor camera.
        if let Some(editor_cam_id) = self.editor_camera_entity {
            scene_v3.entities.retain(|e| e.stable_id != editor_cam_id);
        }

        // Persist editor camera's environment settings.
        if let Some(editor_cam) = self.editor_camera_entity {
            if let Some(ee) = self.world.ecs_entity_for(editor_cam) {
                if let Ok(env) = self.world.ecs_ref()
                    .get::<&rkf_runtime::environment::EnvironmentSettings>(ee)
                {
                    if let Ok(s) = ron::to_string(&*env) {
                        scene_v3.properties.insert("editor_environment".into(), s);
                    }
                }
            }
        }
        // Persist linked environment camera UUID.
        if let Some(linked) = self.linked_env_camera {
            scene_v3.properties.insert("linked_env_camera".into(), linked.to_string());
        }

        if let Ok(s) = ron::to_string(self.light_editor.all_lights()) {
            scene_v3.properties.insert("lights".into(), s);
        }

        // Set scene name from path.
        let name = self
            .current_scene_path
            .as_ref()
            .and_then(|p| std::path::Path::new(p).file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".to_string());
        scene_v3.properties.insert("name".into(), name);

        scene_v3
    }

    /// Build object summaries from hecs (authoritative source of truth).
    pub fn build_object_summaries(&self) -> Vec<crate::ui_snapshot::ObjectSummary> {
        use crate::ui_snapshot::{ObjectSummary, ObjectType};

        let mut summaries = Vec::new();

        // SDF entities (have geometry).
        let render_scene = self.world.build_render_scene();
        for obj in &render_scene.objects {
            let (yaw, pitch, roll) = obj.rotation.to_euler(glam::EulerRot::XYZ);
            let entity_uuid = self.world.find_by_sdf_id(obj.id)
                .unwrap_or(uuid::Uuid::nil());
            let parent_uuid = obj.parent_id.and_then(|pid| self.world.find_by_sdf_id(pid));
            summaries.push(ObjectSummary {
                id: entity_uuid,
                name: obj.name.clone(),
                position: obj.position,
                rotation_degrees: Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees()),
                scale: obj.scale,
                parent_id: parent_uuid,
                object_type: match &obj.root_node.sdf_source {
                    rkf_core::scene_node::SdfSource::None => ObjectType::None,
                    rkf_core::scene_node::SdfSource::Analytical { .. } => ObjectType::Analytical,
                    rkf_core::scene_node::SdfSource::Voxelized { .. } => ObjectType::Voxelized,
                },
                primitive: match &obj.root_node.sdf_source {
                    rkf_core::scene_node::SdfSource::Analytical { primitive, .. } => Some(*primitive),
                    _ => None,
                },
                is_camera: false,
            });
        }

        // ECS-only entities (no geometry — empties, cameras, etc.).
        for (uuid, record) in self.world.entity_records() {
            if record.sdf_object_id.is_some() {
                continue; // Already included above.
            }
            let ecs = self.world.ecs_ref();
            // Skip internal entities (EditorCameraMarker).
            if ecs.get::<&rkf_runtime::components::EditorCameraMarker>(record.ecs_entity).is_ok() {
                continue;
            }
            let transform = ecs.get::<&rkf_runtime::components::Transform>(record.ecs_entity).ok();
            let meta = ecs.get::<&rkf_runtime::components::EditorMetadata>(record.ecs_entity).ok();
            let is_camera = ecs.get::<&rkf_runtime::components::CameraComponent>(record.ecs_entity).is_ok();
            let name = meta.map(|m| m.name.clone()).unwrap_or_default();
            let (pos, rot, scl) = match transform {
                Some(t) => (t.position.to_vec3(), t.rotation, t.scale),
                None => (Vec3::ZERO, glam::Quat::IDENTITY, Vec3::ONE),
            };
            let (yaw, pitch, roll) = rot.to_euler(glam::EulerRot::XYZ);
            summaries.push(ObjectSummary {
                id: *uuid,
                name,
                position: pos,
                rotation_degrees: Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees()),
                scale: scl,
                parent_id: None,
                object_type: ObjectType::None,
                primitive: None,
                is_camera,
            });
        }

        summaries
    }

    /// Build light summaries from the light editor.
    pub fn build_light_summaries(&self) -> Vec<crate::ui_snapshot::LightSummary> {
        use crate::ui_snapshot::LightSummary;
        self.light_editor.all_lights().iter().map(|l| {
            LightSummary {
                id: l.id,
                light_type: l.light_type,
                position: l.position,
                intensity: l.intensity,
                range: l.range,
            }
        }).collect()
    }
}

/// Ray-AABB intersection returning the entry distance along the ray.
///
/// Returns `Some(t)` where `t >= 0` if the ray hits the box, `None` otherwise.
fn ray_aabb_distance(ray_o: Vec3, ray_d: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv_d = Vec3::new(
        if ray_d.x.abs() > 1e-8 { 1.0 / ray_d.x } else { f32::MAX.copysign(ray_d.x) },
        if ray_d.y.abs() > 1e-8 { 1.0 / ray_d.y } else { f32::MAX.copysign(ray_d.y) },
        if ray_d.z.abs() > 1e-8 { 1.0 / ray_d.z } else { f32::MAX.copysign(ray_d.z) },
    );
    let t1 = (min - ray_o) * inv_d;
    let t2 = (max - ray_o) * inv_d;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    if t_exit >= t_enter.max(0.0) {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}
