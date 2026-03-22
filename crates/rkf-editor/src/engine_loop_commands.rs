//! Editor command dispatch and material usage computation.
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use uuid::Uuid;

use crate::editor_state::{
    ComponentSnapshot, EditorMode, EditorState, FieldSnapshot, InspectorSnapshot,
};
use crate::engine::EditorEngine;

/// Compute per-material voxel counts for the selected object.
pub(crate) fn compute_material_usage(
    es: &EditorState,
    engine: &EditorEngine,
    entity_uuid: Uuid,
) -> Vec<crate::ui_snapshot::ObjectMaterialUsage> {
    use rkf_core::scene_node::SdfSource;
    // Read from hecs SdfTree (authoritative source of truth).
    let hecs_entity = match es.world.ecs_entity_for(entity_uuid) {
        Some(e) => e,
        None => return Vec::new(),
    };
    let sdf_tree = match es.world.ecs_ref().get::<&rkf_runtime::components::SdfTree>(hecs_entity) {
        Ok(sdf) => sdf,
        Err(_) => return Vec::new(),
    };
    let sdf_source = &sdf_tree.root.sdf_source;
    {
        if let SdfSource::Analytical { material_id, .. } = sdf_source {
            return vec![crate::ui_snapshot::ObjectMaterialUsage {
                material_id: *material_id,
                voxel_count: 0,
            }];
        }
        if let SdfSource::Voxelized { brick_map_handle, .. } = sdf_source {
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
        SpawnCamera => {
            es.pending_spawn_camera = true;
        }
        SpawnPointLight => {
            es.pending_spawn_point_light = true;
        }
        SpawnSpotLight => {
            es.pending_spawn_spot_light = true;
        }
        PlaceModel { asset_path } => {
            es.pending_place_model = Some(asset_path);
        }
        DragModelEnter { asset_path } => {
            es.pending_drag_model_enter = Some(asset_path);
        }
        DragModelMove { x, y } => {
            // Store viewport-relative mouse position for drag placement.
            es.drag_model_global_mouse = Some((x, y));
        }
        DragModelDrop => {
            // Finalize — push undo and clear drag state.
            es.drag_model_global_mouse = None;
            if let Some(drag) = es.drag_placing.take() {
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SpawnEntity { entity_id: drag.entity_id },
                    timestamp_ms: 0,
                    description: "Place model".into(),
                });
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Object(drag.entity_id));
            }
        }
        DragModelCancel => {
            // Cancel — despawn the entity.
            es.drag_model_global_mouse = None;
            if let Some(drag) = es.drag_placing.take() {
                let _ = es.world.despawn(drag.entity_id);
            }
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
            es.set_editor_camera_component_field(|c| c.fov_degrees = fov);
        }
        SetCameraSpeed { speed } => {
            es.camera_control.fly_speed = speed;
        }
        SetCameraNearFar { near, far } => {
            es.set_editor_camera_component_field(|c| { c.near = near; c.far = far; });
        }
        SetCameraOrbitAngles { yaw, pitch } => {
            let mut pos = glam::Vec3::ZERO;
            let mut cur_yaw = 0.0f32;
            let mut cur_pitch = 0.0f32;
            // Read current transform, then set_orbit_angles writes back.
            if let Some(uuid) = es.editor_camera_entity {
                let snap = es.extract_camera_snapshot_for(uuid);
                pos = snap.position;
                cur_yaw = snap.yaw;
                cur_pitch = snap.pitch;
            }
            es.camera_control.set_orbit_angles(yaw, pitch, &mut pos, &mut cur_yaw, &mut cur_pitch);
            // Write back to entity.
            if let Some(uuid) = es.editor_camera_entity {
                let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
                let _ = es.world.set_position(uuid, wp);
                if let Some(e) = es.world.ecs_entity_for(uuid) {
                    if let Ok(mut cam) = es.world.ecs_mut()
                        .get::<&mut rkf_runtime::components::CameraComponent>(e)
                    {
                        cam.yaw = cur_yaw.to_degrees();
                        cam.pitch = cur_pitch.to_degrees();
                    }
                }
            }
        }

        // -- Environment -------------------------------------------------
        // Environment settings flow through SetComponentField targeting the
        // active camera entity's EnvironmentSettings component.

        // -- Lights ------------------------------------------------------
        SetLightPosition { light_id, position } => {
            if let Some(light) = es.light_editor.get_light_mut(light_id) {
                // NaN sentinel = keep existing axis (for per-axis store edits).
                if !position.x.is_nan() { light.position.x = position.x; }
                if !position.y.is_nan() { light.position.y = position.y; }
                if !position.z.is_nan() { light.position.z = position.z; }
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
            let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, position);
            let _ = es.world.set_position(entity_id, wp);
        }
        SetObjectRotation { entity_id, rotation } => {
            let rot = glam::Quat::from_euler(
                glam::EulerRot::XYZ,
                rotation.x.to_radians(),
                rotation.y.to_radians(),
                rotation.z.to_radians(),
            );
            let _ = es.world.set_rotation(entity_id, rot);
        }
        SetObjectScale { entity_id, scale } => {
            let _ = es.world.set_scale(entity_id, scale);
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
        RemoveRecentProject { path } => {
            crate::editor_config::remove_recent_project(&path);
            // Refresh the UI signal with updated list.
            let config = crate::editor_config::load_editor_config();
            let recents = config.recent_projects;
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
                    ui.recent_projects.set(recents);
                }
            });
        }

        // -- Voxel ops ---------------------------------------------------
        ConvertToVoxel { object_id, voxel_size } => {
            es.pending_convert_to_voxel = Some((object_id, voxel_size));
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

        // -- Component inspector -----------------------------------------
        SetComponentField { .. } | AddComponent { .. } | RemoveComponent { .. } => {
            // Handled by engine loop with registry access (needs GameplayRegistry).
        }

        // -- Play mode ---------------------------------------------------
        PlayStart => {
            es.pending_play_start = true;
        }
        PlayStop => {
            es.pending_play_stop = true;
        }

        // -- Camera viewport -----------------------------------------------
        SetViewportCamera { camera_id } => {
            es.viewport_camera = camera_id;
        }
        SetLinkedEnvCamera { camera_id } => {
            es.linked_env_camera = camera_id;
            // Reset profile key so the engine loop re-evaluates the linked profile.
            es.last_env_profile_key = None;
        }
        SnapToCamera { camera_id } => {
            // Read target camera's transform and apply to editor camera entity.
            let snap = es.extract_camera_snapshot_for(camera_id);
            if let Some(uuid) = es.editor_camera_entity {
                let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, snap.position);
                let _ = es.world.set_position(uuid, wp);
                if let Some(e) = es.world.ecs_entity_for(uuid) {
                    if let Ok(mut cam) = es.world.ecs_mut()
                        .get::<&mut rkf_runtime::components::CameraComponent>(e)
                    {
                        cam.yaw = snap.yaw.to_degrees();
                        cam.pitch = snap.pitch.to_degrees();
                    }
                }
            }
        }
        CreateCameraFromView => {
            let snap = es.extract_camera_snapshot();
            let pos = rkf_core::WorldPosition::new(glam::IVec3::ZERO, snap.position);
            let _uuid = es.world.spawn_camera(
                "Camera", pos, snap.yaw.to_degrees(), snap.pitch.to_degrees(),
                snap.fov_degrees, None,
            );
        }
        PilotCamera { camera_id } => {
            es.piloting = camera_id;
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

/// Build an [`InspectorSnapshot`] for the given entity, suitable for the UI thread.
///
/// Uses the gameplay registry to introspect all components on the entity,
/// then converts each field's `GameValue` to typed optional fields.
pub(crate) fn build_inspector_snapshot(
    es: &EditorState,
    entity_uuid: Uuid,
    registry: &rkf_runtime::behavior::GameplayRegistry,
) -> Option<InspectorSnapshot> {
    let hecs_entity = es.world.ecs_entity_for(entity_uuid)?;
    let ecs = es.world.ecs_ref();

    let data = rkf_runtime::behavior::inspector::build_inspector_data(ecs, hecs_entity, registry);

    let components = data
        .components
        .into_iter()
        .map(|comp| {
            let fields = comp
                .fields
                .into_iter()
                .map(|f| field_inspector_to_snapshot(f))
                .collect();

            ComponentSnapshot {
                name: comp.name,
                fields,
                removable: comp.removable,
            }
        })
        .collect();

    Some(InspectorSnapshot {
        entity_id: entity_uuid,
        components,
    })
}

/// Build a list of component names available to add to the given entity.
pub(crate) fn build_available_components(
    es: &EditorState,
    entity_uuid: Uuid,
    registry: &rkf_runtime::behavior::GameplayRegistry,
) -> Vec<String> {
    let hecs_entity = match es.world.ecs_entity_for(entity_uuid) {
        Some(e) => e,
        None => return Vec::new(),
    };
    let ecs = es.world.ecs_ref();
    rkf_runtime::behavior::inspector::available_components_for_entity(ecs, hecs_entity, registry)
}

/// Apply a component inspector command (requires the gameplay registry).
///
/// Returns `true` if the command was a component inspector command
/// (consumed), `false` if it was something else (not consumed).
pub(crate) fn apply_component_command(
    es: &mut EditorState,
    cmd: &crate::editor_command::EditorCommand,
    registry: &rkf_runtime::behavior::GameplayRegistry,
) -> bool {
    use crate::editor_command::EditorCommand;
    match cmd {
        EditorCommand::SetComponentField {
            entity_id,
            component_name,
            field_name,
            value,
        } => {
            if let Some(hecs_entity) = es.world.ecs_entity_for(*entity_id) {
                if let Some(entry) = registry.component_entry(component_name) {
                    if field_name.contains('.') {
                        // Dot-notation: first try the full path directly (components
                        // like EnvironmentSettings handle flattened dot-paths in
                        // their set_field). Fall back to the struct read-modify-write
                        // pattern if the direct path fails.
                        let direct_ok = (entry.set_field)(
                            es.world.ecs_mut(),
                            hecs_entity,
                            field_name,
                            value.clone(),
                        ).is_ok();
                        if !direct_ok {
                            let top_field = field_name.split('.').next().unwrap();
                            let rest = &field_name[top_field.len() + 1..];
                            if let Ok(mut parent_val) =
                                (entry.get_field)(es.world.ecs_ref(), hecs_entity, top_field)
                            {
                                if let Ok(()) = rkf_runtime::behavior::set_nested_field(
                                    &mut parent_val,
                                    rest,
                                    value.clone(),
                                ) {
                                    let _ = (entry.set_field)(
                                        es.world.ecs_mut(),
                                        hecs_entity,
                                        top_field,
                                        parent_val,
                                    );
                                }
                            }
                        }
                    } else {
                        // Simple flat field.
                        let _ = (entry.set_field)(
                            es.world.ecs_mut(),
                            hecs_entity,
                            field_name,
                            value.clone(),
                        );
                    }

                    // Console warning for component ref pointing to entity without required component.
                    if let Some(field_meta) = entry
                        .meta
                        .iter()
                        .find(|m| m.name == field_name.split('.').next().unwrap_or(field_name))
                    {
                        if let (Some(filter), Some(uuid_str)) =
                            (field_meta.component_filter, value.as_string())
                        {
                            if !uuid_str.is_empty() {
                                if let Ok(target_uuid) = uuid::Uuid::parse_str(uuid_str) {
                                    if let Some(target_entity) =
                                        es.world.ecs_entity_for(target_uuid)
                                    {
                                        if let Some(target_entry) =
                                            registry.component_entry(filter)
                                        {
                                            if !(target_entry.has)(
                                                es.world.ecs_ref(),
                                                target_entity,
                                            ) {
                                                eprintln!(
                                                    "[warn] component ref '{}' on '{}': \
                                                     target entity {} does not have '{}'",
                                                    field_name,
                                                    component_name,
                                                    uuid_str,
                                                    filter
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Auto-save environment profile: if we just edited EnvironmentSettings
                    // on the editor camera and a linked_env_camera is set, write the
                    // updated settings back to the linked camera's .rkenv file.
                    if component_name == "EnvironmentSettings" {
                        // First check: direct profile on this entity's CameraComponent.
                        let mut saved = false;
                        if let Ok(cam) = es.world.ecs_ref()
                            .get::<&rkf_runtime::components::CameraComponent>(hecs_entity)
                        {
                            let profile_path = cam.environment_profile.clone();
                            if !profile_path.is_empty() {
                                if let Ok(env_settings) = es.world.ecs_ref()
                                    .get::<&rkf_runtime::environment::EnvironmentSettings>(hecs_entity)
                                {
                                    let profile = env_settings.to_profile("auto");
                                    if let Err(e) = rkf_runtime::save_environment(&profile_path, &profile) {
                                        eprintln!("[warn] failed to auto-save environment profile '{}': {}", profile_path, e);
                                    }
                                    saved = true;
                                }
                            }
                        }
                        // Second check: if this is the editor camera and we have a linked
                        // env camera, save to the linked camera's profile instead.
                        if !saved {
                            if let Some(editor_cam) = es.editor_camera_entity {
                                if es.world.ecs_entity_for(editor_cam) == Some(hecs_entity) {
                                    if let Some(linked_uuid) = es.linked_env_camera {
                                        if let Some(linked_e) = es.world.ecs_entity_for(linked_uuid) {
                                            if let Ok(linked_cam) = es.world.ecs_ref()
                                                .get::<&rkf_runtime::components::CameraComponent>(linked_e)
                                            {
                                                let profile_path = linked_cam.environment_profile.clone();
                                                if !profile_path.is_empty() {
                                                    if let Ok(env_settings) = es.world.ecs_ref()
                                                        .get::<&rkf_runtime::environment::EnvironmentSettings>(hecs_entity)
                                                    {
                                                        let profile = env_settings.to_profile("auto");
                                                        if let Err(e) = rkf_runtime::save_environment(&profile_path, &profile) {
                                                            eprintln!("[warn] failed to auto-save environment profile '{}': {}", profile_path, e);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            true
        }
        EditorCommand::AddComponent {
            entity_id,
            component_name,
        } => {
            if let Some(hecs_entity) = es.world.ecs_entity_for(*entity_id) {
                let _ = rkf_runtime::behavior::inspector::add_component_default(
                    es.world.ecs_mut(),
                    hecs_entity,
                    component_name,
                    registry,
                );
            }
            true
        }
        EditorCommand::RemoveComponent {
            entity_id,
            component_name,
        } => {
            if let Some(hecs_entity) = es.world.ecs_entity_for(*entity_id) {
                let _ = rkf_runtime::behavior::inspector::remove_component(
                    es.world.ecs_mut(),
                    hecs_entity,
                    component_name,
                    registry,
                );
            }
            true
        }
        _ => false,
    }
}

/// Convert a `FieldInspectorData` to a `FieldSnapshot`, recursing into sub-fields.
fn field_inspector_to_snapshot(
    f: rkf_runtime::behavior::FieldInspectorData,
) -> FieldSnapshot {
    let display_value = format_game_value(&f.value);
    let float_value = f.value.as_float();
    let int_value = f.value.as_int();
    let bool_value = f.value.as_bool();
    let vec3_value = f.value.as_vec3();
    let string_value = f.value.as_string().map(|s| s.to_string());
    let color_value = f.value.as_color();

    let sub_fields = f.sub_fields.map(|subs| {
        subs.into_iter()
            .map(|sf| field_inspector_to_snapshot(sf))
            .collect()
    });

    FieldSnapshot {
        name: f.name,
        field_type: f.field_type,
        display_value,
        float_value,
        int_value,
        bool_value,
        vec3_value,
        string_value,
        color_value,
        range: f.range,
        transient: f.transient,
        sub_fields,
        asset_filter: f.asset_filter,
        component_filter: f.component_filter,
    }
}

/// Format a `GameValue` as a short display string for the inspector.
pub(crate) fn format_game_value_short(val: &rkf_runtime::behavior::game_value::GameValue) -> String {
    format_game_value(val)
}

fn format_game_value(val: &rkf_runtime::behavior::game_value::GameValue) -> String {
    use rkf_runtime::behavior::game_value::GameValue;
    match val {
        GameValue::Bool(b) => b.to_string(),
        GameValue::Int(i) => i.to_string(),
        GameValue::Float(f) => format!("{f:.3}"),
        GameValue::String(s) => s.clone(),
        GameValue::Vec3(v) => format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z),
        GameValue::WorldPosition(wp) => {
            let v = wp.to_vec3();
            format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z)
        }
        GameValue::Quat(q) => {
            let (x, y, z) = q.to_euler(glam::EulerRot::XYZ);
            format!(
                "({:.1}, {:.1}, {:.1})",
                x.to_degrees(),
                y.to_degrees(),
                z.to_degrees()
            )
        }
        GameValue::Color(c) => format!("({:.2}, {:.2}, {:.2}, {:.2})", c[0], c[1], c[2], c[3]),
        GameValue::List(v) => format!("[{} items]", v.len()),
        GameValue::Struct(fields) => format!("{{{} fields}}", fields.len()),
        GameValue::Ron(s) => {
            if s.len() > 40 {
                format!("{}...", &s[..37])
            } else {
                s.clone()
            }
        }
    }
}
