//! Editor command dispatch and material usage computation.
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use crate::editor_state::{EditorMode, EditorState};
use crate::engine::EditorEngine;

/// Compute per-material voxel counts for the selected object.
pub(crate) fn compute_material_usage(
    es: &EditorState,
    engine: &EditorEngine,
    eid: u64,
) -> Vec<crate::ui_snapshot::ObjectMaterialUsage> {
    use rkf_core::scene_node::SdfSource;
    let scene = es.world.scene();
    if let Some(obj) = scene.objects.iter().find(|o| o.id as u64 == eid) {
        if let SdfSource::Analytical { material_id, .. } = &obj.root_node.sdf_source {
            return vec![crate::ui_snapshot::ObjectMaterialUsage {
                material_id: *material_id,
                voxel_count: 0,
            }];
        }
        if let SdfSource::Voxelized { brick_map_handle, .. } = &obj.root_node.sdf_source {
            let handle = *brick_map_handle;
            let dims = handle.dims;
            let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
            for bz in 0..dims.z {
                for by in 0..dims.y {
                    for bx in 0..dims.x {
                        if let Some(slot) = engine.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            if EditorEngine::is_unallocated(slot) {
                                continue;
                            }
                            let brick = engine.cpu_brick_pool.get(slot);
                            for vz in 0..8u32 {
                                for vy in 0..8u32 {
                                    for vx in 0..8u32 {
                                        let sample = brick.sample(vx, vy, vz);
                                        if sample.distance_f32() <= 0.0 {
                                            *counts.entry(sample.material_id()).or_insert(0) += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut usage: Vec<crate::ui_snapshot::ObjectMaterialUsage> = counts
                .into_iter()
                .map(|(material_id, voxel_count)| crate::ui_snapshot::ObjectMaterialUsage { material_id, voxel_count })
                .collect();
            usage.sort_by(|a, b| b.voxel_count.cmp(&a.voxel_count));
            return usage;
        }
    }
    Vec::new()
}

/// Apply a single UI command to the editor state.
///
/// Called by the engine thread while holding the EditorState lock.
/// This is the command consumer for the UI->engine channel.
pub(crate) fn apply_editor_command(es: &mut EditorState, cmd: crate::editor_command::EditorCommand) {
    use crate::editor_command::EditorCommand::*;
    match cmd {
        // -- Input -------------------------------------------------------
        MouseMove { x, y, dx, dy } => {
            es.editor_input.mouse_pos = glam::Vec2::new(x, y);
            es.editor_input.mouse_delta += glam::Vec2::new(dx, dy);
        }
        MouseDown { button, x, y } => {
            if button < 3 {
                es.editor_input.mouse_buttons[button] = true;
            }
            es.editor_input.mouse_pos = glam::Vec2::new(x, y);
        }
        MouseUp { button, .. } => {
            if button < 3 {
                es.editor_input.mouse_buttons[button] = false;
            }
        }
        Scroll { delta } => {
            es.editor_input.scroll_delta += delta;
        }
        KeyDown { key, modifiers } => {
            es.editor_input.keys_pressed.insert(key);
            es.editor_input.keys_just_pressed.insert(key);
            es.editor_input.modifiers = modifiers;
            // Gizmo mode switching: G/R/L keys.
            if es.mode == crate::editor_state::EditorMode::Default {
                use crate::input::KeyCode as KC;
                match key {
                    KC::G => es.gizmo.mode = crate::gizmo::GizmoMode::Translate,
                    KC::R => es.gizmo.mode = crate::gizmo::GizmoMode::Rotate,
                    KC::L => es.gizmo.mode = crate::gizmo::GizmoMode::Scale,
                    _ => {}
                }
            }
        }
        KeyUp { key, modifiers } => {
            es.editor_input.keys_pressed.remove(&key);
            es.editor_input.modifiers = modifiers;
        }

        // -- Scene mutations ---------------------------------------------
        SpawnPrimitive { name } => {
            es.pending_spawn = Some(name);
        }
        DeleteSelected => {
            es.pending_delete = true;
        }
        DuplicateSelected => {
            es.pending_duplicate = true;
        }
        Undo => {
            es.pending_undo = true;
        }
        Redo => {
            es.pending_redo = true;
        }
        SelectEntity { entity } => {
            es.selected_entity = entity;
        }

        // -- Gizmo -------------------------------------------------------
        SetGizmoMode { mode } => {
            es.gizmo.mode = mode;
        }

        // -- Tool settings -----------------------------------------------
        SetEditorMode { mode } => {
            es.mode = mode;
            // Sync selected material into paint settings when entering Paint mode.
            if mode == EditorMode::Paint {
                if let Some(slot) = es.material_browser.selected_slot {
                    es.paint.current_settings.material_id = slot;
                }
            }
        }
        SetSculptSettings { radius, strength, falloff } => {
            es.sculpt.set_radius(radius);
            es.sculpt.set_strength(strength);
            es.sculpt.current_settings.falloff = falloff;
        }
        SetPaintSettings { radius, strength, falloff } => {
            es.paint.current_settings.radius = radius;
            es.paint.current_settings.strength = strength;
            es.paint.current_settings.falloff = falloff;
        }
        SetPaintMode { mode } => {
            es.paint.current_settings.mode = mode;
        }
        SetPaintColor { r, g, b } => {
            es.paint.set_color(glam::Vec3::new(r, g, b));
        }

        // -- Camera settings ---------------------------------------------
        SetCameraFov { fov } => {
            es.editor_camera.fov_y = fov.to_radians();
        }
        SetCameraSpeed { speed } => {
            es.editor_camera.fly_speed = speed;
        }
        SetCameraNearFar { near, far } => {
            es.editor_camera.near = near;
            es.editor_camera.far = far;
        }

        // -- Environment -------------------------------------------------
        SetAtmosphere { sun_direction, sun_intensity, rayleigh_scale, mie_scale } => {
            es.environment.atmosphere.sun_direction = sun_direction;
            es.environment.atmosphere.sun_intensity = sun_intensity;
            es.environment.atmosphere.rayleigh_scale = rayleigh_scale;
            es.environment.atmosphere.mie_scale = mie_scale;
            es.environment.mark_dirty();
        }
        SetFog { density, height_falloff, dust_density, dust_asymmetry } => {
            es.environment.fog.density = density;
            es.environment.fog.height_falloff = height_falloff;
            es.environment.fog.ambient_dust_density = dust_density;
            es.environment.fog.dust_asymmetry = dust_asymmetry;
            es.environment.mark_dirty();
        }
        SetClouds { coverage, density, altitude, thickness, wind_speed } => {
            es.environment.clouds.coverage = coverage;
            es.environment.clouds.density = density;
            es.environment.clouds.altitude = altitude;
            es.environment.clouds.thickness = thickness;
            es.environment.clouds.wind_speed = wind_speed;
            es.environment.mark_dirty();
        }
        SetPostProcess {
            bloom_intensity, bloom_threshold, exposure, sharpen,
            dof_focus_distance, dof_focus_range, dof_max_coc,
            motion_blur, god_rays, vignette, grain, chromatic_aberration,
        } => {
            es.environment.post_process.bloom_intensity = bloom_intensity;
            es.environment.post_process.bloom_threshold = bloom_threshold;
            es.environment.post_process.exposure = exposure;
            es.environment.post_process.sharpen_strength = sharpen;
            es.environment.post_process.dof_focus_distance = dof_focus_distance;
            es.environment.post_process.dof_focus_range = dof_focus_range;
            es.environment.post_process.dof_max_coc = dof_max_coc;
            es.environment.post_process.motion_blur_intensity = motion_blur;
            es.environment.post_process.god_rays_intensity = god_rays;
            es.environment.post_process.vignette_intensity = vignette;
            es.environment.post_process.grain_intensity = grain;
            es.environment.post_process.chromatic_aberration = chromatic_aberration;
            es.environment.mark_dirty();
        }
        ToggleAtmosphere { enabled } => {
            es.environment.atmosphere.enabled = enabled;
            es.environment.mark_dirty();
        }
        ToggleFog { enabled } => {
            es.environment.fog.enabled = enabled;
            es.environment.mark_dirty();
        }
        ToggleClouds { enabled } => {
            es.environment.clouds.enabled = enabled;
            es.environment.mark_dirty();
        }
        ToggleBloom { enabled } => {
            es.environment.post_process.bloom_enabled = enabled;
            es.environment.mark_dirty();
        }
        ToggleDof { enabled } => {
            es.environment.post_process.dof_enabled = enabled;
            es.environment.mark_dirty();
        }
        SetToneMapMode { mode } => {
            es.environment.post_process.tone_map_mode = mode;
            es.environment.mark_dirty();
        }

        // -- Lights ------------------------------------------------------
        SetLightPosition { light_id, position } => {
            if let Some(light) = es.light_editor.get_light_mut(light_id) {
                light.position = position;
            }
            es.light_editor.mark_dirty();
        }
        SetLightIntensity { light_id, intensity } => {
            if let Some(light) = es.light_editor.get_light_mut(light_id) {
                light.intensity = intensity;
            }
            es.light_editor.mark_dirty();
        }
        SetLightRange { light_id, range } => {
            if let Some(light) = es.light_editor.get_light_mut(light_id) {
                light.range = range;
            }
            es.light_editor.mark_dirty();
        }

        // -- Debug / view ------------------------------------------------
        SetDebugMode { mode } => {
            es.pending_debug_mode = Some(mode);
        }
        ToggleGrid => {
            es.show_grid = !es.show_grid;
        }
        ToggleShortcuts => {
            es.show_shortcuts = !es.show_shortcuts;
        }

        // -- Object properties -------------------------------------------
        SetObjectPosition { entity_id, position } => {
            let sc = es.world.scene_mut();
            if let Some(obj) = sc.objects.iter_mut().find(|o| o.id as u64 == entity_id) {
                obj.position = position;
            }
        }
        SetObjectRotation { entity_id, rotation } => {
            let sc = es.world.scene_mut();
            if let Some(obj) = sc.objects.iter_mut().find(|o| o.id as u64 == entity_id) {
                obj.rotation = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    rotation.x.to_radians(),
                    rotation.y.to_radians(),
                    rotation.z.to_radians(),
                );
            }
        }
        SetObjectScale { entity_id, scale } => {
            let sc = es.world.scene_mut();
            if let Some(obj) = sc.objects.iter_mut().find(|o| o.id as u64 == entity_id) {
                obj.scale = scale;
            }
        }

        // -- Scene I/O ---------------------------------------------------
        OpenScene { path } => {
            es.pending_open = true;
            if !path.is_empty() {
                es.pending_open_path = Some(path);
            }
        }
        SaveScene { path } => {
            if let Some(p) = path {
                es.pending_save = true;
                es.pending_save_path = Some(p);
            } else {
                es.pending_save = true;
            }
        }

        // -- Project I/O -------------------------------------------------
        NewProject => {
            es.pending_new_project = true;
        }
        OpenProject { path } => {
            es.pending_open_project = true;
            if !path.is_empty() {
                es.pending_open_project_path = Some(path);
            }
        }
        SaveProject => {
            es.pending_save_project = true;
        }

        // -- Voxel ops ---------------------------------------------------
        ConvertToVoxel { object_id } => {
            es.pending_convert_to_voxel = Some(object_id);
        }
        // -- Animation ---------------------------------------------------
        SetAnimationState { state } => {
            es.animation.playback_state = match state {
                1 => crate::animation_preview::PlaybackState::Playing,
                2 => crate::animation_preview::PlaybackState::Paused,
                _ => crate::animation_preview::PlaybackState::Stopped,
            };
        }
        SetAnimationSpeed { speed } => {
            es.animation.speed = speed;
        }

        // -- Materials ---------------------------------------------------
        SelectMaterial { slot } => {
            es.material_browser.selected_slot = Some(slot);
            // Always sync to paint settings — user may select material before entering Paint mode.
            es.paint.current_settings.material_id = slot;
        }
        RemapMaterial { object_id, from_material, to_material } => {
            es.pending_remap_material = Some((object_id, from_material, to_material));
        }
        SetPrimitiveMaterial { object_id, material_id } => {
            es.pending_set_primitive_material = Some((object_id, material_id));
        }
        SetMaterial { .. } | SetMaterialShader { .. } => {
            // Handled separately in engine loop (needs MaterialLibrary / brick pool access).
        }

        // -- Window management -------------------------------------------
        WindowDrag => {
            es.pending_drag = true;
        }
        WindowMinimize => {
            es.pending_minimize = true;
        }
        WindowMaximize => {
            es.pending_maximize = true;
        }
        RequestExit => {
            es.wants_exit = true;
        }
    }
}
