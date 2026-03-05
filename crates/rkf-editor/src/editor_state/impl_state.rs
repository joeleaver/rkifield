//! `impl EditorState` methods.

use glam::Vec3;
use rinch::prelude::Signal;

use super::*;


impl EditorState {
    /// Create a new editor state with default values and fly-mode camera
    /// positioned to match the engine's initial viewpoint.
    pub fn new() -> Self {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.position = Vec3::new(0.0, 2.5, 5.0);
        cam.fly_yaw = 0.0;
        cam.fly_pitch = -0.15;
        cam.fov_y = 70.0_f32.to_radians();
        cam.fly_speed = 5.0;

        Self {
            mode: EditorMode::Default,
            editor_camera: cam,
            editor_input: InputState::new(),
            selected_entity: None,
            selected_properties: None,
            gizmo: GizmoState::new(),
            sculpt: SculptState::new(),
            paint: PaintState::new(),
            placement_queue: PlacementQueue::new(),
            asset_browser: AssetBrowser::new(),
            grid_snap: GridSnap::default(),
            light_editor: LightManager::new(),
            environment: {
                let mut env = EnvironmentState::new();
                env.mark_dirty(); // Ensure first frame applies defaults to engine
                env
            },
            animation: AnimationPreview::new(),
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
            pending_delete: false,
            pending_duplicate: false,
            pending_undo: false,
            pending_redo: false,
            show_grid: false,
            show_shortcuts: false,
            pending_drag: false,
            pending_minimize: false,
            pending_maximize: false,
            pending_revoxelize: None,
            pending_fix_sdfs: None,
            pending_remap_material: None,
            pending_sculpt_edits: Vec::new(),
            sculpt_undo_accumulator: None,
            pending_sculpt_undo: None,
            world: World::new("editor"),
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
    /// This method exists to work around Rust's borrow checker: calling
    /// `self.editor_camera.update(&self.editor_input, dt)` through a
    /// `MutexGuard` doesn't allow simultaneous mutable + immutable borrows,
    /// but a method on `Self` can borrow separate fields.
    pub fn update_camera(&mut self, dt: f32) {
        self.editor_camera.update(&self.editor_input, dt);
    }

    /// Sync editor camera state to an engine `Camera`.
    ///
    /// Copies position, orientation, and FOV from the `SceneCamera` to
    /// the render engine's `Camera` struct.
    pub fn sync_to_engine_camera(&self, engine_cam: &mut rkf_render::camera::Camera) {
        engine_cam.position = self.editor_camera.position;
        engine_cam.yaw = self.editor_camera.fly_yaw;
        engine_cam.pitch = self.editor_camera.fly_pitch;
        engine_cam.fov_degrees = self.editor_camera.fov_y.to_degrees();
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
                if let Some(entity) = self.world.find_entity_by_id(*entity_id) {
                    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
                    let _ = self.world.set_position(entity, wp);
                    let _ = self.world.set_rotation(entity, rot);
                    let _ = self.world.set_scale(entity, scale);
                }
            }
            UndoActionKind::SpawnEntity { entity_id } => {
                if reverse {
                    // Undo spawn = despawn
                    if let Some(entity) = self.world.find_entity_by_id(*entity_id) {
                        let _ = self.world.despawn(entity);
                    }
                }
                // Redo spawn would need stored object data — not supported yet.
            }
            UndoActionKind::DespawnEntity { entity_id: _ } => {
                // Undo despawn would need stored object data — not supported yet.
            }
            UndoActionKind::SculptStroke { brick_snapshots, .. } => {
                if reverse {
                    // Undo: queue brick restoration for the render loop.
                    self.pending_sculpt_undo = Some(brick_snapshots.clone());
                }
                // Redo is not supported for sculpt strokes.
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
    ) -> Option<u64> {
        let (ray_o, ray_d) = crate::camera::screen_to_ray(
            &self.editor_camera,
            pixel_x,
            pixel_y,
            vp_width,
            vp_height,
        );

        let scene = self.world.scene();
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        let mut best_t = f32::MAX;
        let mut best_id = None;

        for obj in &scene.objects {
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
                    best_id = Some(obj.id as u64);
                }
            }
        }

        best_id
    }

    /// Load a scene file and populate the editor state from it.
    ///
    /// Populates `world.scene()` with SDF objects from the file, lights into
    /// the light editor, and applies environment settings. Returns the loaded
    /// `SceneFile` for further processing (e.g. setting up engine geometry).
    pub fn load_scene(&mut self, path: &str) -> Result<crate::scene_io::SceneFile, String> {
        use crate::scene_io::{load_scene_from_path, ComponentData};

        let scene_file = load_scene_from_path(path)?;

        // Clear existing world scene.
        let world_scene = self.world.scene_mut();
        world_scene.objects.clear();

        // Populate world scene from non-light entities.
        for entity in &scene_file.entities {
            let is_light_only = !entity.components.is_empty()
                && entity
                    .components
                    .iter()
                    .all(|c| matches!(c, ComponentData::Light { .. }));
            if is_light_only {
                continue;
            }

            let root_node = rkf_core::scene_node::SceneNode::new(&entity.name);
            let mut obj = rkf_core::scene::SceneObject {
                id: entity.entity_id as u32,
                name: entity.name.clone(),
                parent_id: entity.parent_id.map(|p| p as u32),
                position: entity.position,
                rotation: entity.rotation,
                scale: entity.scale,
                root_node,
                aabb: rkf_core::aabb::Aabb::new(Vec3::ZERO, Vec3::ZERO),
            };
            // Store asset path in the root node's name for roundtrip.
            for comp in &entity.components {
                if let ComponentData::SdfObject { asset_path } = comp {
                    obj.root_node.name = asset_path.clone();
                }
            }
            world_scene.objects.push(obj);
        }

        // Update next_id to avoid collisions with loaded objects.
        if let Some(max_id) = world_scene.objects.iter().map(|o| o.id).max() {
            world_scene.next_id = max_id + 1;
        }

        // Populate light editor from Light components.
        self.light_editor = crate::light_editor::LightManager::new();
        for entity in &scene_file.entities {
            for comp in &entity.components {
                if let ComponentData::Light {
                    light_type,
                    color,
                    intensity,
                    range,
                } = comp
                {
                    use crate::light_editor::SceneLightType;
                    let lt = match light_type.as_str() {
                        "spot" => SceneLightType::Spot,
                        _ => SceneLightType::Point,
                    };
                    let id = self.light_editor.add_light(lt);
                    self.light_editor.set_position(id, entity.position);
                    self.light_editor.set_color(
                        id,
                        Vec3::new(color[0], color[1], color[2]),
                    );
                    self.light_editor.set_intensity(id, *intensity);
                    if *range > 0.0 {
                        self.light_editor.set_range(id, *range);
                    }
                }
            }
        }

        // Apply environment settings if present.
        if !scene_file.environment_ron.is_empty() {
            if let Ok(env) =
                crate::environment::EnvironmentState::deserialize_from_ron(&scene_file.environment_ron)
            {
                self.environment = env;
                self.environment.mark_dirty();
            }
        }

        // Resync entity tracking to register all loaded objects.
        self.world.resync_entity_tracking();

        // Track the current scene path.
        self.current_scene_path = Some(path.to_string());
        self.unsaved_changes.mark_saved();

        Ok(scene_file)
    }

    /// Construct a [`SceneFile`] from the current editor state.
    ///
    /// Reads SDF objects directly from `world.scene()`, then appends light
    /// entities from the light editor. The environment is serialized via RON.
    /// The resulting `SceneFile` can be passed to
    /// [`crate::scene_io::save_scene_to_path`].
    pub fn save_current_scene(&self) -> crate::scene_io::SceneFile {
        use crate::light_editor::SceneLightType;
        use crate::scene_io::{ComponentData, SceneEntity, SceneFile};
        use glam::Quat;

        let mut entities: Vec<SceneEntity> = Vec::new();

        // Collect SDF objects from world scene.
        let scene = self.world.scene();
        for obj in &scene.objects {
            let mut components = Vec::new();
            // If the root node name looks like an asset path, store it.
            if !obj.root_node.name.is_empty() && obj.root_node.name.contains("://") {
                components.push(ComponentData::SdfObject {
                    asset_path: obj.root_node.name.clone(),
                });
            }
            entities.push(SceneEntity {
                entity_id: obj.id as u64,
                name: obj.name.clone(),
                parent_id: obj.parent_id.map(|p| p as u64),
                position: obj.position,
                rotation: obj.rotation,
                scale: obj.scale,
                components,
            });
        }

        // Append light entities from the light editor.
        for (idx, light) in self.light_editor.all_lights().iter().enumerate() {
            let light_type_str = match light.light_type {
                SceneLightType::Point => "point",
                SceneLightType::Spot => "spot",
            };
            let name = format!(
                "{} Light {}",
                match light.light_type {
                    SceneLightType::Point => "Point",
                    SceneLightType::Spot => "Spot",
                },
                idx + 1
            );
            let range = if light.range.is_infinite() {
                0.0
            } else {
                light.range
            };
            entities.push(SceneEntity {
                entity_id: light.id,
                name,
                parent_id: None,
                position: light.position,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: vec![ComponentData::Light {
                    light_type: light_type_str.to_string(),
                    color: [light.color.x, light.color.y, light.color.z],
                    intensity: light.intensity,
                    range,
                }],
            });
        }

        let environment_ron = self.environment.serialize_to_ron().unwrap_or_default();

        let name = self
            .current_scene_path
            .as_ref()
            .and_then(|p| std::path::Path::new(p).file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".to_string());

        SceneFile {
            version: 1,
            name,
            entities,
            environment_ron,
        }
    }

    /// Build a `UiSnapshot` from the current editor state.
    ///
    /// Called by the render thread after processing commands. Published
    /// via `ArcSwap<UiSnapshot>` for the UI thread to read lock-free.
    pub fn to_snapshot(&self, scene_revision: u64, fps_ms: f64) -> crate::ui_snapshot::UiSnapshot {
        use crate::ui_snapshot::{LightSummary, ObjectSummary, UiSnapshot};

        let d = &self.environment.atmosphere.sun_direction;
        let sun_azimuth = d.x.atan2(d.z).to_degrees().rem_euclid(360.0);
        let sun_elevation = d.y.asin().to_degrees();

        let objects: Vec<ObjectSummary> = self.world.scene().objects.iter().map(|obj| {
            let (yaw, pitch, roll) = obj.rotation.to_euler(glam::EulerRot::XYZ);
            ObjectSummary {
                id: obj.id as u64,
                name: obj.name.clone(),
                position: obj.position,
                rotation_degrees: Vec3::new(yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees()),
                scale: obj.scale,
                parent_id: obj.parent_id,
            }
        }).collect();

        let lights: Vec<LightSummary> = self.light_editor.all_lights().iter().map(|l| {
            LightSummary {
                id: l.id,
                light_type: l.light_type,
                position: l.position,
                intensity: l.intensity,
                range: l.range,
            }
        }).collect();

        let object_count = objects.len();

        UiSnapshot {
            camera_position: self.editor_camera.position,
            camera_yaw: self.editor_camera.fly_yaw,
            camera_pitch: self.editor_camera.fly_pitch,
            camera_fov: self.editor_camera.fov_y.to_degrees(),
            camera_speed: self.editor_camera.fly_speed,
            camera_near: self.editor_camera.near,
            camera_far: self.editor_camera.far,
            selected_entity: self.selected_entity,
            mode: self.mode,
            gizmo_mode: self.gizmo.mode,
            debug_mode: self.debug_mode,
            show_grid: self.show_grid,
            show_shortcuts: self.show_shortcuts,
            scene_name: self.world.scene().name.clone(),
            objects,
            lights,
            scene_revision,
            brush_radius: self.sculpt.current_settings.radius,
            brush_strength: self.sculpt.current_settings.strength,
            brush_falloff: self.sculpt.current_settings.falloff,
            animation_state: match self.animation.playback_state {
                crate::animation_preview::PlaybackState::Stopped => 0,
                crate::animation_preview::PlaybackState::Playing => 1,
                crate::animation_preview::PlaybackState::Paused => 2,
            },
            animation_speed: self.animation.speed,
            atmo_enabled: self.environment.atmosphere.enabled,
            sun_azimuth,
            sun_elevation,
            sun_intensity: self.environment.atmosphere.sun_intensity,
            rayleigh_scale: self.environment.atmosphere.rayleigh_scale,
            mie_scale: self.environment.atmosphere.mie_scale,
            fog_enabled: self.environment.fog.enabled,
            fog_density: self.environment.fog.density,
            fog_height_falloff: self.environment.fog.height_falloff,
            dust_density: self.environment.fog.ambient_dust_density,
            dust_asymmetry: self.environment.fog.dust_asymmetry,
            clouds_enabled: self.environment.clouds.enabled,
            cloud_coverage: self.environment.clouds.coverage,
            cloud_density: self.environment.clouds.density,
            cloud_altitude: self.environment.clouds.altitude,
            cloud_thickness: self.environment.clouds.thickness,
            cloud_wind_speed: self.environment.clouds.wind_speed,
            bloom_enabled: self.environment.post_process.bloom_enabled,
            bloom_intensity: self.environment.post_process.bloom_intensity,
            bloom_threshold: self.environment.post_process.bloom_threshold,
            dof_enabled: self.environment.post_process.dof_enabled,
            dof_focus_distance: self.environment.post_process.dof_focus_distance,
            dof_focus_range: self.environment.post_process.dof_focus_range,
            dof_max_coc: self.environment.post_process.dof_max_coc,
            exposure: self.environment.post_process.exposure,
            sharpen: self.environment.post_process.sharpen_strength,
            motion_blur: self.environment.post_process.motion_blur_intensity,
            god_rays: self.environment.post_process.god_rays_intensity,
            vignette: self.environment.post_process.vignette_intensity,
            grain: self.environment.post_process.grain_intensity,
            chromatic_aberration: self.environment.post_process.chromatic_aberration,
            tone_map_mode: self.environment.post_process.tone_map_mode,
            current_scene_path: self.current_scene_path.clone(),
            materials: Vec::new(), // Populated by engine loop from MaterialLibrary.
            material_revision: 0,
            shaders: Vec::new(), // Populated by engine loop from ShaderComposer.
            selected_object_materials: Vec::new(), // Populated by engine loop from brick pool.
            fps_ms,
            object_count,
        }
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
