//! Engine background thread: render loop, sculpt processing, wireframe overlays.

use std::sync::{Arc, Mutex};

use rinch::render_surface::{GpuTextureRegistrar, SurfaceWriter};

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorState, UiSignals};
use crate::engine::EditorEngine;
use crate::engine_loop_commands::apply_editor_command;
use crate::engine_loop_edits;
use crate::engine_loop_io;
use crate::engine_loop_play;
use crate::engine_loop_ui;
use crate::engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};

pub(crate) struct EngineThreadData {
    pub(crate) editor_state: Arc<Mutex<EditorState>>,
    pub(crate) shared_state: Arc<Mutex<SharedState>>,
    pub(crate) cmd_rx: crossbeam::channel::Receiver<EditorCommand>,
    pub(crate) layout_backing: crate::layout::state::LayoutBacking,
    pub(crate) surface_writer: SurfaceWriter,
    pub(crate) gpu_registrar: GpuTextureRegistrar,
    pub(crate) preview_writer: SurfaceWriter,
    pub(crate) gameplay_registry: Arc<Mutex<rkf_runtime::behavior::GameplayRegistry>>,
    pub(crate) game_store: Arc<Mutex<rkf_runtime::behavior::GameStore>>,
    pub(crate) console: rkf_runtime::behavior::ConsoleBuffer,
    pub(crate) store_push_buffer: crate::store::signals::PushBuffer,
}

/// Tracks which categories of data changed this frame, so we only
/// push signals that actually need updating.
#[derive(Default)]
pub(crate) struct DirtyFlags {
    /// Scene objects added/removed/renamed/reparented/transformed.
    pub(crate) scene: bool,
    /// Light list changed (add/remove/modify).
    pub(crate) lights: bool,
    /// Material table changed.
    pub(crate) materials: bool,
    /// Shader registry changed.
    pub(crate) shaders: bool,
}

/// Map a user-facing primitive name to an SdfPrimitive.
pub(crate) fn primitive_from_name(name: &str) -> rkf_core::SdfPrimitive {
    use rkf_core::SdfPrimitive;
    match name {
        "Sphere" => SdfPrimitive::Sphere { radius: 0.5 },
        "Box" | "Cube" => SdfPrimitive::Box {
            half_extents: glam::Vec3::splat(0.5),
        },
        "Capsule" => SdfPrimitive::Capsule {
            radius: 0.2,
            half_height: 0.4,
        },
        "Torus" => SdfPrimitive::Torus {
            major_radius: 0.4,
            minor_radius: 0.12,
        },
        "Cylinder" => SdfPrimitive::Cylinder {
            radius: 0.3,
            half_height: 0.5,
        },
        _ => SdfPrimitive::Sphere { radius: 0.5 }, // fallback
    }
}

pub(crate) fn engine_thread(data: EngineThreadData) {
    let EngineThreadData {
        editor_state,
        shared_state,
        cmd_rx,
        layout_backing,
        surface_writer,
        gpu_registrar,
        preview_writer,
        gameplay_registry,
        game_store,
        console,
        store_push_buffer,
    } = data;

    log::info!("Engine thread: creating dedicated GPU device");

    // Create engine with its own wgpu device (no sharing with compositor).
    // Starts with an empty scene — content loaded via project/scene path.
    // hecs is the single source of truth for all scene data.
    let mut engine = EditorEngine::new_headless(
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        Arc::clone(&shared_state),
    );
    engine.console = console;

    // Game plugin dylib handle — must outlive the GameplayRegistry since system
    // fn_ptrs point into the loaded library. Dropping unloads the dylib.
    let mut game_dylib_loader: Option<rkf_runtime::behavior::DylibLoader> = None;

    // Load recent projects and push to UI for the welcome screen.
    {
        let config = crate::editor_config::load_editor_config();
        let recents = config.recent_projects;
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.recent_projects.set(recents);
            }
        });
    }

    /// Push a loading status message to the UI from the engine thread.
    /// Pass `None` to clear the loading indicator.
    fn set_loading_status(msg: Option<String>) {
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.loading_status.set(msg);
            }
        });
    }

    log::info!("Engine thread: engine ready, entering render loop");

    // ── Behavior system initialization ───────────────────────────────────
    let mut behavior_executor = {
        let reg = gameplay_registry.lock().expect("gameplay registry lock");
        rkf_runtime::behavior::BehaviorExecutor::new(&reg)
            .expect("failed to build behavior schedule")
    };
    let game_store_arc = game_store;
    // The game store lock is held briefly per-frame for behavior ticks.
    // Clone Arc for passing to play state functions.
    let stable_ids = rkf_runtime::behavior::StableIdIndex::new();
    let mut play_state = engine_loop_play::PlayState::new();

    let mut last_frame = std::time::Instant::now();
    let mut last_fps_push = std::time::Instant::now();
    let mut last_camera_push = std::time::Instant::now();
    // Push everything on the first frame so the UI has initial data.
    let mut first_frame = true;
    let mut current_vp = engine.viewport_size();
    // Tracks whether the preview texture has been registered with the compositor.
    let preview_writer = preview_writer;

    loop {
        let frame_start = std::time::Instant::now();
        let dt = frame_start.duration_since(last_frame).as_secs_f32();
        last_frame = frame_start;

        // Track scene changes for incremental GPU updates.
        let mut frame_topology_changed = false; // Only for scene open / new project.
        let mut viewport_resized = false;
        let mut frame_dirty_objects: Vec<u32> = Vec::new();
        let mut frame_spawned: Vec<u32> = Vec::new();
        let mut frame_despawned: Vec<u32> = Vec::new();

        // a. Check viewport size from layout.
        let desired_vp = {
            let (w, h) = gpu_registrar.layout_size();
            (w.max(64), h.max(64))
        };
        if desired_vp != current_vp {
            if let Some(_view) = engine.resize_viewport(desired_vp.0, desired_vp.1) {
                current_vp = desired_vp;
                log::debug!("Engine: resized to {}x{}", desired_vp.0, desired_vp.1);
                viewport_resized = true;
            }
        }

        // b. Drain pending MCP commands from shared_state.
        let (pending_camera, pending_debug, pending_voxel_slice, pending_spatial_query, pending_mcp_sculpt, pending_object_shape) =
            if let Ok(mut ss) = shared_state.lock() {
                (ss.pending_camera.take(), ss.pending_debug_mode.take(),
                 ss.pending_voxel_slice.take(), ss.pending_spatial_query.take(),
                 ss.pending_mcp_sculpt.take(), ss.pending_object_shape.take())
            } else {
                (None, None, None, None, None, None)
            };

        // b2. Drain UI commands and apply to EditorState (brief lock).
        let mut dirty = DirtyFlags::default();
        if viewport_resized || first_frame {
            dirty.scene = true;
            dirty.lights = true;
            dirty.materials = true;
            dirty.shaders = true;
        }
        {
            if let Ok(mut es) = editor_state.lock() {
                let mut key_downs = Vec::new();
                while let Ok(cmd) = cmd_rx.try_recv() {
                    // Track transform commands that need GPU dirty marking.
                    match &cmd {
                        EditorCommand::SetObjectPosition { entity_id, .. }
                        | EditorCommand::SetObjectRotation { entity_id, .. }
                        | EditorCommand::SetObjectScale { entity_id, .. } => {
                            // Look up SDF object ID from UUID for GPU dirty tracking.
                            if let Some((_, record)) = es.world.entity_records().find(|(uid, _)| **uid == *entity_id) {
                                if let Some(obj_id) = record.sdf_object_id {
                                    frame_dirty_objects.push(obj_id);
                                }
                            }
                            // Push updated summaries to UiSignals so the properties
                            // panel (which reads via Memo) reflects the change.
                            dirty.scene = true;
                        }
                        // Structural scene changes.
                        EditorCommand::SpawnPrimitive { .. }
                        | EditorCommand::SpawnCamera
                        | EditorCommand::SpawnPointLight
                        | EditorCommand::SpawnSpotLight
                        | EditorCommand::PlaceModel { .. }
                        | EditorCommand::DragModelEnter { .. }
                        | EditorCommand::DragModelDrop
                        | EditorCommand::DragModelCancel
                        | EditorCommand::DeleteSelected
                        | EditorCommand::DuplicateSelected
                        | EditorCommand::Undo
                        | EditorCommand::Redo
                        | EditorCommand::OpenScene { .. }
                        | EditorCommand::OpenProject { .. }
                        | EditorCommand::NewProject
                        | EditorCommand::ConvertToVoxel { .. }
                        | EditorCommand::RemapMaterial { .. }
                        | EditorCommand::SetPrimitiveMaterial { .. } => {
                            dirty.scene = true;
                            dirty.lights = true;
                        }
                        // Light property edits.
                        EditorCommand::SetLightPosition { .. }
                        | EditorCommand::SetLightIntensity { .. }
                        | EditorCommand::SetLightRange { .. } => {
                            dirty.lights = true;
                        }
                        // Camera/environment source changes.
                        EditorCommand::SetViewportCamera { .. }
                        | EditorCommand::PilotCamera { .. }
                        | EditorCommand::SetLinkedEnvCamera { .. } => {
                            dirty.scene = true;
                        }
                        _ => {}
                    }
                    // Handle SetMaterial directly with engine's MaterialLibrary.
                    if let EditorCommand::SetMaterial { slot, material } = cmd {
                        if let Ok(mut lib) = engine.material_library.lock() {
                            let old = lib.get_material(slot).copied().unwrap_or_default();
                            if old != material {
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::MaterialChange {
                                        slot,
                                        old,
                                        new: material,
                                    },
                                    timestamp_ms: 0,
                                    description: format!("Change material #{slot}"),
                                });
                            }
                            lib.set_material(slot, material);
                        }
                        dirty.materials = true;
                        continue;
                    }
                    if let EditorCommand::SetMaterialShader { slot, shader_name } = cmd {
                        let shader_id = engine.shader_composer.shader_id(&shader_name);
                        if let Ok(mut lib) = engine.material_library.lock() {
                            if let Some(info) = lib.slot_info_mut(slot) {
                                info.shader_name = shader_name;
                            }
                            if let Some(mat) = lib.get_material_mut(slot) {
                                mat.shader_id = shader_id;
                            }
                            lib.mark_dirty();
                        }
                        dirty.materials = true;
                        continue;
                    }
                    // Handle component inspector commands (need registry).
                    if matches!(
                        cmd,
                        EditorCommand::SetComponentField { .. }
                        | EditorCommand::AddComponent { .. }
                        | EditorCommand::RemoveComponent { .. }
                    ) {
                        // Check if this is an SdfTree.asset_path change — need to
                        // trigger .rkf loading after the field is written.
                        let sdf_load_info = if let EditorCommand::SetComponentField {
                            entity_id, ref component_name, ref field_name, ref value,
                        } = cmd {
                            if component_name == "SdfTree" && field_name == "asset_path" {
                                value.as_string().map(|s| (entity_id, s.to_string()))
                            } else { None }
                        } else { None };

                        if let Ok(reg) = gameplay_registry.lock() {
                            crate::engine_loop_commands::apply_component_command(
                                &mut es, &cmd, &reg,
                            );
                        }

                        // Post-hook: load .rkf if asset_path was changed.
                        if let Some((entity_uuid, new_path)) = sdf_load_info {
                            if !new_path.is_empty() {
                                let resolved = crate::engine_loop_io::resolve_rkf_path(
                                    &new_path, &es, &engine,
                                );
                                let path_str = resolved.to_string_lossy().to_string();
                                if let Some(hecs_entity) = es.world.ecs_entity_for(entity_uuid) {
                                    let obj_id = es.world.entity_records()
                                        .find(|(uid, _)| **uid == entity_uuid)
                                        .and_then(|(_, r)| r.sdf_object_id);
                                    match engine.load_rkf_for_entity(
                                        &path_str, es.world.ecs_mut(), hecs_entity, obj_id,
                                    ) {
                                        Ok(_) => {
                                            engine.finish_rkf_upload();
                                            engine.topology_changed = true;
                                        }
                                        Err(e) => log::error!("Failed to load .rkf '{}': {e}", new_path),
                                    }
                                }
                            }
                        }

                        dirty.scene = true; // trigger inspector refresh
                        continue;
                    }
                    match cmd {
                        EditorCommand::KeyDown { .. } => key_downs.push(cmd),
                        EditorCommand::SelectEntity { .. } => {
                            let prev = es.selected_entity;
                            apply_editor_command(&mut es, cmd);
                            if es.selected_entity != prev {
                                dirty.scene = true; // rebuild inspector for new selection
                            }
                        }
                        _ => apply_editor_command(&mut es, cmd),
                    }
                }
                for cmd in key_downs {
                    apply_editor_command(&mut es, cmd);
                }
            }
        }

        // c. Single brief lock: read all data needed for this frame, then release.
        let (
            camera_snap, f_debug_mode, f_convert_to_voxel, f_remap_material,
            f_set_prim_mat,
            f_environment, f_lights,
            mut scene_clone, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius, f_brush_strength, f_brush_falloff,
            f_sculpt_edits, f_sculpt_undo, f_sculpting_active,
            f_paint_edits, f_paint_undo,
            f_play_start, f_play_stop,
            f_camera_store, f_viewport_camera,
        ) = {
            let mut es = match editor_state.lock() {
                Ok(es) => es,
                Err(_) => continue, // poisoned -- skip frame
            };

            // Apply pending MCP commands (mutate before read).
            if let Some(cam) = pending_camera {
                // Write to the editor camera entity (source of truth).
                if let Some(uuid) = es.editor_camera_entity {
                    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, cam.position);
                    let _ = es.world.set_position(uuid, wp);
                    if let Some(e) = es.world.ecs_entity_for(uuid) {
                        if let Ok(mut cc) = es.world.ecs_mut()
                            .get::<&mut rkf_runtime::components::CameraComponent>(e)
                        {
                            cc.yaw = cam.yaw.to_degrees();
                            cc.pitch = cam.pitch.to_degrees();
                        }
                    }
                }
                // Update orbit target so spawned objects appear at the right place.
                let dir = glam::Vec3::new(
                    -cam.yaw.sin() * cam.pitch.cos(),
                    cam.pitch.sin(),
                    -cam.yaw.cos() * cam.pitch.cos(),
                );
                es.camera_control.target = cam.position
                    + dir * es.camera_control.orbit_distance;
            }
            if let Some(mode) = pending_debug {
                es.pending_debug_mode = Some(mode);
            }

            // Camera update: reads entity, applies controls, writes back to entity.
            es.update_camera(dt);

            // Pilot mode: write editor camera state back to the piloted entity.
            if let Some(pilot_uuid) = es.piloting {
                if let Some(hecs_entity) = es.world.ecs_entity_for(pilot_uuid) {
                    let snap = es.extract_camera_snapshot();
                    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, snap.position);
                    let _ = es.world.set_position(pilot_uuid, wp);
                    if let Ok(mut cam) = es.world.ecs_mut()
                        .get::<&mut rkf_runtime::components::CameraComponent>(hecs_entity)
                    {
                        cam.yaw = snap.yaw.to_degrees();
                        cam.pitch = snap.pitch.to_degrees();
                        cam.fov_degrees = snap.fov_degrees;
                    }
                } else {
                    // Piloted camera was deleted — stop piloting.
                    es.piloting = None;
                }
            }

            // Determine the active camera for rendering: piloting > viewport > editor.
            let active_camera_uuid = if es.piloting.is_some() {
                // In piloting mode, editor camera drives the piloted entity,
                // and we render from the editor camera's perspective.
                es.editor_camera_entity.unwrap_or_default()
            } else if let Some(vp_uuid) = es.viewport_camera {
                if es.world.ecs_entity_for(vp_uuid).is_some() {
                    vp_uuid
                } else {
                    es.viewport_camera = None;
                    es.editor_camera_entity.unwrap_or_default()
                }
            } else {
                es.editor_camera_entity.unwrap_or_default()
            };
            let camera_snapshot = es.extract_camera_snapshot_for(active_camera_uuid);

            // Consume pending commands.
            let debug_mode = es.pending_debug_mode.take();
            if let Some(mode) = debug_mode {
                es.debug_mode = mode;
            }
            let convert_to_voxel = es.pending_convert_to_voxel.take();
            let remap_material = es.pending_remap_material.take();
            let set_prim_mat = es.pending_set_primitive_material.take();

            // Consume undo/redo -- classify by action kind.
            if es.pending_undo {
                es.pending_undo = false;
                if let Some(action) = es.undo.undo() {
                    es.apply_undo_action(&action, true);
                    // Helper: look up SDF object ID from UUID.
                    let sdf_id = |uuid: &uuid::Uuid| -> Option<u32> {
                        es.world.entity_records()
                            .find(|(uid, _)| *uid == uuid)
                            .and_then(|(_, r)| r.sdf_object_id)
                    };
                    match &action.kind {
                        crate::undo::UndoActionKind::SpawnEntity { entity_id } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_despawned.push(oid); }
                        }
                        crate::undo::UndoActionKind::DespawnEntity { entity_id } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_spawned.push(oid); }
                        }
                        crate::undo::UndoActionKind::Transform { entity_id, .. } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_dirty_objects.push(oid); }
                        }
                        crate::undo::UndoActionKind::SculptStroke { object_id, .. } => {
                            if let Some(oid) = sdf_id(object_id) { frame_dirty_objects.push(oid); }
                        }
                        _ => {}
                    }
                }
            }
            if es.pending_redo {
                es.pending_redo = false;
                if let Some(action) = es.undo.redo() {
                    es.apply_undo_action(&action, false);
                    let sdf_id = |uuid: &uuid::Uuid| -> Option<u32> {
                        es.world.entity_records()
                            .find(|(uid, _)| *uid == uuid)
                            .and_then(|(_, r)| r.sdf_object_id)
                    };
                    match &action.kind {
                        crate::undo::UndoActionKind::SpawnEntity { entity_id } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_spawned.push(oid); }
                        }
                        crate::undo::UndoActionKind::DespawnEntity { entity_id } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_despawned.push(oid); }
                        }
                        crate::undo::UndoActionKind::Transform { entity_id, .. } => {
                            if let Some(oid) = sdf_id(entity_id) { frame_dirty_objects.push(oid); }
                        }
                        crate::undo::UndoActionKind::SculptStroke { object_id, .. } => {
                            if let Some(oid) = sdf_id(object_id) { frame_dirty_objects.push(oid); }
                        }
                        _ => {}
                    }
                }
            }

            // Consume pending spawn.
            if let Some(prim_name) = es.pending_spawn.take() {
                let pos = es.camera_control.target;
                let uuid = if prim_name == "Empty" {
                    es.world.spawn("Empty")
                        .position_vec3(pos)
                        .build()
                } else {
                    let primitive = primitive_from_name(&prim_name);
                    es.world.spawn(&prim_name)
                        .position_vec3(pos)
                        .sdf(primitive)
                        .material(0)
                        .build()
                };
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Object(uuid));
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SpawnEntity {
                        entity_id: uuid,
                    },
                    timestamp_ms: 0,
                    description: format!("Spawn {prim_name}"),
                });
                // Look up SDF object ID for the new entity.
                if let Some((_, record)) = es.world.entity_records().find(|(uid, _)| **uid == uuid) {
                    if let Some(obj_id) = record.sdf_object_id {
                        frame_spawned.push(obj_id);
                    }
                }
            }

            // Consume pending camera spawn.
            if es.pending_spawn_camera {
                es.pending_spawn_camera = false;
                let snap = es.extract_camera_snapshot();
                let pos = rkf_core::WorldPosition::new(glam::IVec3::ZERO, snap.position);
                let uuid = es.world.spawn_camera(
                    "Camera", pos, snap.yaw.to_degrees(), snap.pitch.to_degrees(),
                    snap.fov_degrees, None,
                );
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Object(uuid));
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SpawnEntity { entity_id: uuid },
                    timestamp_ms: 0,
                    description: "Spawn Camera".into(),
                });
            }

            // Consume pending light spawns.
            if es.pending_spawn_point_light {
                es.pending_spawn_point_light = false;
                let pos = es.camera_control.target;
                let light = crate::light_editor::SceneLight {
                    id: 0,
                    light_type: crate::light_editor::SceneLightType::Point,
                    position: pos,
                    direction: glam::Vec3::NEG_Y,
                    color: glam::Vec3::ONE,
                    intensity: 1.0,
                    range: 10.0,
                    spot_inner_angle: 0.0,
                    spot_outer_angle: 0.0,
                    cast_shadows: true,
                    cookie_path: None,
                };
                let id = es.light_editor.add_light_full(light);
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Light(id));
            }
            if es.pending_spawn_spot_light {
                es.pending_spawn_spot_light = false;
                let pos = es.camera_control.target;
                let light = crate::light_editor::SceneLight {
                    id: 0,
                    light_type: crate::light_editor::SceneLightType::Spot,
                    position: pos,
                    direction: glam::Vec3::NEG_Y,
                    color: glam::Vec3::ONE,
                    intensity: 1.0,
                    range: 15.0,
                    spot_inner_angle: 0.3,
                    spot_outer_angle: 0.5,
                    cast_shadows: true,
                    cookie_path: None,
                };
                let id = es.light_editor.add_light_full(light);
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Light(id));
            }

            // Helper: spawn an entity with a loaded .rkf model, returns (uuid, obj_id).
            let spawn_model = |asset_path: &str,
                               pos: glam::Vec3,
                               es: &mut EditorState,
                               engine: &mut EditorEngine,
                               frame_spawned: &mut Vec<u32>|
             -> Option<(uuid::Uuid, Option<u32>)> {
                let name = std::path::Path::new(asset_path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Model".into());
                let uuid = es.world.spawn(&name).position_vec3(pos).build();
                let hecs_entity = es.world.ecs_entity_for(uuid)?;
                let sdf_tree = rkf_runtime::components::SdfTree {
                    asset_path: Some(asset_path.to_string()),
                    ..Default::default()
                };
                let _ = es.world.ecs_mut().insert_one(hecs_entity, sdf_tree);
                es.world.rebuild_entity_tracking_from_ecs();
                let resolved = crate::engine_loop_io::resolve_rkf_path(
                    asset_path, es, &*engine,
                );
                let path_str = resolved.to_string_lossy().to_string();
                let obj_id = es.world.entity_records()
                    .find(|(uid, _)| **uid == uuid)
                    .and_then(|(_, r)| r.sdf_object_id);
                match engine.load_rkf_for_entity(
                    &path_str, es.world.ecs_mut(), hecs_entity, obj_id,
                ) {
                    Ok(_) => {
                        engine.finish_rkf_upload();
                        engine.topology_changed = true;
                    }
                    Err(e) => log::error!("Failed to load model '{}': {e}", asset_path),
                }
                if let Some(oid) = obj_id {
                    frame_spawned.push(oid);
                }
                Some((uuid, obj_id))
            };

            // Consume pending model placement (click "Place" button).
            if let Some(asset_path) = es.pending_place_model.take() {
                let pos = es.camera_control.target;
                if let Some((uuid, _)) = spawn_model(&asset_path, pos, &mut es, &mut engine, &mut frame_spawned) {
                    es.selected_entity = Some(crate::editor_state::SelectedEntity::Object(uuid));
                    es.undo.push(crate::undo::UndoAction {
                        kind: crate::undo::UndoActionKind::SpawnEntity { entity_id: uuid },
                        timestamp_ms: 0,
                        description: "Place model".into(),
                    });
                }
            }

            // Consume pending drag-model-enter (drag from Models panel into viewport).
            if let Some(asset_path) = es.pending_drag_model_enter.take() {
                let pos = es.camera_control.target;
                if let Some((uuid, obj_id)) = spawn_model(&asset_path, pos, &mut es, &mut engine, &mut frame_spawned) {
                    es.drag_placing = Some(crate::editor_state::DragPlacement {
                        entity_id: uuid,
                        sdf_object_id: obj_id,
                        asset_path: asset_path.clone(),
                    });
                }
            }

            // Update drag-placing entity position from mouse ray-plane intersection.
            if let Some(drag_info) = es.drag_placing.as_ref().map(|d| (d.entity_id, d.sdf_object_id)) {
                if let Some((gx, gy)) = es.drag_model_global_mouse.take() {
                    let (drag_uuid, drag_oid) = drag_info;
                    // Convert global mouse to normalized coords using window size.
                    let (win_w, win_h) = (engine.window_width.max(1) as f32, engine.window_height.max(1) as f32);
                    let ndc_x = (gx / win_w) * 2.0 - 1.0;
                    let ndc_y = 1.0 - (gy / win_h) * 2.0;
                    // Build a ray from the camera through the mouse position.
                    let snap = es.extract_camera_snapshot();
                    let aspect = win_w / win_h;
                    let fov_rad = snap.fov_degrees.to_radians();
                    let half_h = (fov_rad * 0.5).tan();
                    let half_w = half_h * aspect;
                    let forward = glam::Vec3::new(
                        snap.yaw.cos() * snap.pitch.cos(),
                        snap.pitch.sin(),
                        snap.yaw.sin() * snap.pitch.cos(),
                    ).normalize();
                    let right = forward.cross(glam::Vec3::Y).normalize();
                    let up = right.cross(forward).normalize();
                    let ray_dir = (forward + right * (ndc_x * half_w) + up * (ndc_y * half_h)).normalize();
                    let ray_origin = snap.position;
                    // Intersect with Y=0 ground plane (or camera target Y).
                    let plane_y = 0.0f32;
                    if ray_dir.y.abs() > 1e-6 {
                        let t = (plane_y - ray_origin.y) / ray_dir.y;
                        if t > 0.0 {
                            let hit = ray_origin + ray_dir * t;
                            let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, hit);
                            let _ = es.world.set_position(drag_uuid, wp);
                            if let Some(oid) = drag_oid {
                                frame_dirty_objects.push(oid);
                            }
                        }
                    }
                }
            }

            // Consume pending delete.
            if es.pending_delete {
                es.pending_delete = false;
                if let Some(crate::editor_state::SelectedEntity::Object(uuid)) = es.selected_entity {
                    if es.world.is_alive(uuid) {
                        // Get SDF object ID before despawning.
                        let sdf_oid = es.world.entity_records()
                            .find(|(uid, _)| **uid == uuid)
                            .and_then(|(_, r)| r.sdf_object_id);
                        es.undo.push(crate::undo::UndoAction {
                            kind: crate::undo::UndoActionKind::DespawnEntity {
                                entity_id: uuid,
                            },
                            timestamp_ms: 0,
                            description: "Delete object".into(),
                        });
                        let _ = es.world.despawn(uuid);
                        es.selected_entity = None;
                        // Push cleared selection to UI.
                        rinch::shell::rinch_runtime::run_on_main_thread(|| {
                            if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
                                ui.selection.set(None);
                            }
                        });
                        if let Some(oid) = sdf_oid {
                            frame_despawned.push(oid);
                        }
                    }
                }
            }

            // Consume pending duplicate.
            if es.pending_duplicate {
                es.pending_duplicate = false;
                if let Some(crate::editor_state::SelectedEntity::Object(src_uuid)) = es.selected_entity {
                    if es.world.is_alive(src_uuid) {
                        if let (Ok(pos), Ok(rot), Ok(scale)) = (
                            es.world.position(src_uuid),
                            es.world.rotation(src_uuid),
                            es.world.scale(src_uuid),
                        ) {
                            let root = es.world.root_node(src_uuid).ok().cloned();
                            if let Some(root_node) = root {
                                let offset = rkf_core::WorldPosition::new(
                                    glam::IVec3::ZERO,
                                    pos.to_vec3() + glam::Vec3::new(1.0, 0.0, 0.0),
                                );
                                let new_uuid = es.world.spawn(&root_node.name)
                                    .position(offset)
                                    .rotation(rot)
                                    .scale(scale)
                                    .sdf_tree(root_node)
                                    .build();
                                es.selected_entity = Some(
                                    crate::editor_state::SelectedEntity::Object(new_uuid),
                                );
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::SpawnEntity {
                                        entity_id: new_uuid,
                                    },
                                    timestamp_ms: 0,
                                    description: "Duplicate object".into(),
                                });
                                if let Some((_, record)) = es.world.entity_records().find(|(uid, _)| **uid == new_uuid) {
                                    if let Some(obj_id) = record.sdf_object_id {
                                        frame_spawned.push(obj_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // -- Scene/Project I/O (delegated to engine_loop_io) --------
            if es.pending_save {
                es.pending_save = false;
                let reg = gameplay_registry.lock().expect("registry lock");
                engine_loop_io::handle_scene_save(&mut es, &engine, &reg);
                drop(reg);
            }
            if es.pending_save_as {
                es.pending_save_as = false;
                let reg = gameplay_registry.lock().expect("registry lock");
                engine_loop_io::handle_scene_save_as(es, &editor_state, &engine, &reg);
                // Re-acquire lock — es was consumed by handle_scene_save_as.
                es = editor_state.lock().expect("editor_state lock after save-as");
            }

            if es.pending_open {
                es.pending_open = false;
                let open_path = es.pending_open_path.take();
                let mat_lib = engine.material_library.clone();
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                    gameplay_registry: &gameplay_registry,
                    material_library: &mat_lib,
                };
                engine_loop_io::handle_scene_open_impl(es, open_path, &mut io_ctx);
                // Re-acquire lock — es was consumed by handle_scene_open_impl.
                // Fall through to render the newly loaded scene this frame.
                es = editor_state.lock().expect("editor_state lock after scene open");
                frame_topology_changed = true;
                dirty.scene = true;
                dirty.lights = true;
                dirty.materials = true;
            }

            if es.pending_new_project {
                es.pending_new_project = false;
                drop(es);
                let mat_lib = engine.material_library.clone();
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                    gameplay_registry: &gameplay_registry,
                    material_library: &mat_lib,
                };
                engine_loop_io::handle_new_project(&mut io_ctx, &mut frame_topology_changed);

                // Push project-loaded to UI if a project was actually created.
                if editor_state.lock().ok().map_or(false, |es| es.current_project.is_some()) {
                    engine_loop_io::push_project_loaded_to_ui();
                }

                // Build and load game plugin for the new project (non-blocking).
                let project_root = editor_state.lock().ok().and_then(|es_guard| {
                    let pp = es_guard.current_project_path.as_ref()?;
                    Some(rkf_runtime::project::project_root(std::path::Path::new(pp)))
                });
                if let Some(project_root) = project_root {
                    // Signal dylib not ready while build runs.
                    rinch::shell::rinch_runtime::run_on_main_thread(|| {
                        if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                            ui.dylib_ready.set(false);
                        }
                    });
                    let (loader, bw) = engine_loop_io::build_and_load_game_dylib(
                        &project_root, &gameplay_registry, game_dylib_loader.take(),
                        Some(&engine.console),
                    );
                    game_dylib_loader = loader;
                    if let Some(bw) = bw {
                        engine.build_watcher = Some(bw);
                    }
                    let reg = gameplay_registry.lock().expect("registry lock");
                    behavior_executor = rkf_runtime::behavior::BehaviorExecutor::new(&reg)
                        .expect("failed to rebuild behavior schedule");
                    // If we loaded an existing dylib, signal ready immediately.
                    if game_dylib_loader.is_some() {
                        rinch::shell::rinch_runtime::run_on_main_thread(|| {
                            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                                ui.dylib_ready.set(true);
                            }
                        });
                    }
                }

                engine.topology_changed = true;
                dirty.scene = true;
                dirty.lights = true;
                dirty.materials = true;
                dirty.shaders = true;
                // Re-acquire lock — es was dropped above.
                // Fall through to render the newly created project this frame.
                es = editor_state.lock().expect("editor_state lock after new project");
            }

            if es.pending_open_project {
                es.pending_open_project = false;
                let open_path = es.pending_open_project_path.take();
                drop(es);
                let mat_lib = engine.material_library.clone();
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                    gameplay_registry: &gameplay_registry,
                    material_library: &mat_lib,
                };
                engine_loop_io::handle_open_project(open_path, &mut io_ctx, &mut frame_topology_changed);

                // Push project-loaded to UI if a project was actually opened.
                if editor_state.lock().ok().map_or(false, |es| es.current_project.is_some()) {
                    engine_loop_io::push_project_loaded_to_ui();
                }

                // Build and load game plugin for the opened project (non-blocking).
                let project_root = editor_state.lock().ok().and_then(|es_guard| {
                    let pp = es_guard.current_project_path.as_ref()?;
                    Some(rkf_runtime::project::project_root(std::path::Path::new(pp)))
                });
                if let Some(project_root) = project_root {
                    // Signal dylib not ready while build runs.
                    rinch::shell::rinch_runtime::run_on_main_thread(|| {
                        if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                            ui.dylib_ready.set(false);
                        }
                    });
                    let (loader, bw) = engine_loop_io::build_and_load_game_dylib(
                        &project_root, &gameplay_registry, game_dylib_loader.take(),
                        Some(&engine.console),
                    );
                    game_dylib_loader = loader;
                    if let Some(bw) = bw {
                        engine.build_watcher = Some(bw);
                    }
                    let reg = gameplay_registry.lock().expect("registry lock");
                    behavior_executor = rkf_runtime::behavior::BehaviorExecutor::new(&reg)
                        .expect("failed to rebuild behavior schedule");
                    // If we loaded an existing dylib, signal ready immediately.
                    if game_dylib_loader.is_some() {
                        rinch::shell::rinch_runtime::run_on_main_thread(|| {
                            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                                ui.dylib_ready.set(true);
                            }
                        });
                    }
                }

                engine.topology_changed = true;
                dirty.scene = true;
                dirty.lights = true;
                dirty.materials = true;
                dirty.shaders = true;
                // Re-acquire lock — es was dropped above.
                // Fall through to render the newly opened project this frame.
                es = editor_state.lock().expect("editor_state lock after open project");
            }

            if es.pending_save_project {
                es.pending_save_project = false;
                let reg = gameplay_registry.lock().expect("registry lock");
                engine_loop_io::handle_save_project(&mut es, &engine, &layout_backing, &reg);
                drop(reg);
            }

            // Detect gizmo-driven transform changes.
            if es.gizmo.dragging {
                if let Some(crate::editor_state::SelectedEntity::Object(uuid)) = es.selected_entity {
                    if let Some((_, record)) = es.world.entity_records().find(|(uid, _)| **uid == uuid) {
                        if let Some(obj_id) = record.sdf_object_id {
                            frame_dirty_objects.push(obj_id);
                        }
                    }
                }
            }

            // ── Environment: per-camera → renderer ─────────────────────
            //
            // Environment settings live on each camera entity. The active
            // camera is the viewport camera (if set) or the editor camera.

            // Environment: two separate concepts.
            //
            // 1. "Env source" — which camera's EnvironmentSettings drives the renderer.
            //    Priority: piloting > viewport camera > editor camera.
            //    Each camera has its own EnvironmentSettings component.
            //
            // 2. "Profile linking" — copies a .rkenv file onto the editor camera.
            //    This only affects the editor camera, not scene cameras.

            let env_source_uuid = es.piloting
                .or(es.viewport_camera)
                .or(es.linked_env_camera)
                .or(es.editor_camera_entity);
            let env_source_entity = env_source_uuid
                .and_then(|uuid| es.world.ecs_entity_for(uuid));

            // Load .rkenv profile onto the EDITOR camera when linked camera changes.
            let editor_cam_entity = es.editor_camera_entity
                .and_then(|uuid| es.world.ecs_entity_for(uuid));
            if let Some(linked_uuid) = es.linked_env_camera {
                if let Some(linked_entity) = es.world.ecs_entity_for(linked_uuid) {
                    let profile_path = es.world.ecs_ref()
                        .get::<&rkf_runtime::components::CameraComponent>(linked_entity)
                        .ok()
                        .map(|c| c.environment_profile.clone())
                        .unwrap_or_default();
                    let key = if profile_path.is_empty() {
                        None
                    } else {
                        Some((linked_uuid, profile_path.clone()))
                    };
                    if key != es.last_env_profile_key {
                        es.last_env_profile_key = key.clone();
                        if let Some((_, ref path)) = key {
                            if let Ok(profile) = rkf_runtime::load_environment(path) {
                                let settings = rkf_runtime::environment::EnvironmentSettings::from_profile(&profile);
                                if let Some(ee) = editor_cam_entity {
                                    let _ = es.world.ecs_mut().insert_one(ee, settings);
                                    dirty.scene = true;
                                }
                            }
                        }
                    }
                }
            } else if es.last_env_profile_key.is_some() {
                es.last_env_profile_key = None;
            }

            // Read EnvironmentSettings from the env source camera for the renderer.
            let environment: Option<rkf_runtime::environment::EnvironmentSettings> = if dirty.scene {
                if let Some(ee) = env_source_entity {
                    es.world.ecs_ref()
                        .get::<&rkf_runtime::environment::EnvironmentSettings>(ee)
                        .ok()
                        .map(|s| (*s).clone())
                } else {
                    None
                }
            } else {
                None
            };

            // Lights: convert only when dirty.
            let lights = if es.light_editor.is_dirty() {
                let converted = es.light_editor.all_lights().iter().map(|el| {
                    use crate::light_editor::SceneLightType;
                    rkf_render::Light {
                        light_type: match el.light_type {
                            SceneLightType::Point => 1,
                            SceneLightType::Spot => 2,
                        },
                        pos_x: el.position.x,
                        pos_y: el.position.y,
                        pos_z: el.position.z,
                        dir_x: el.direction.x,
                        dir_y: el.direction.y,
                        dir_z: el.direction.z,
                        color_r: el.color.x,
                        color_g: el.color.y,
                        color_b: el.color.z,
                        intensity: el.intensity,
                        range: el.range,
                        inner_angle: el.spot_inner_angle,
                        outer_angle: el.spot_outer_angle,
                        cookie_index: -1,
                        shadow_caster: el.cast_shadows as u32,
                    }
                }).collect();
                es.light_editor.clear_dirty();
                Some(converted)
            } else {
                None
            };

            // Build a render scene from hecs (authoritative source of truth).
            let scene = {
                let mut merged = es.world.build_render_scene();
                for obj in &mut merged.objects {
                    obj.aabb = crate::placement::compute_object_local_aabb(obj);
                }
                merged
            };

            // Wireframe overlay data (all Copy).
            let gizmo_axis = if es.gizmo.dragging {
                es.gizmo.active_axis
            } else {
                es.gizmo.hovered_axis
            };
            let brush_radius = match es.mode {
                crate::editor_state::EditorMode::Sculpt => es.sculpt.current_settings.radius,
                crate::editor_state::EditorMode::Paint => es.paint.current_settings.radius,
                _ => 1.0,
            };
            let brush_strength = match es.mode {
                crate::editor_state::EditorMode::Sculpt => es.sculpt.current_settings.strength,
                crate::editor_state::EditorMode::Paint => es.paint.current_settings.strength,
                _ => 0.5,
            };
            let brush_falloff = match es.mode {
                crate::editor_state::EditorMode::Paint => es.paint.current_settings.falloff,
                crate::editor_state::EditorMode::Sculpt => es.sculpt.current_settings.falloff,
                _ => 0.0,
            };

            let cam_snap = camera_snapshot;
            let sel = es.selected_entity;
            let gm = es.gizmo.mode;
            let grid = es.show_grid;
            let emode = es.mode;
            let sculpting_active = es.sculpt.active_stroke.is_some();

            // Drain pending sculpt edits and undo.
            let sculpt_edits = std::mem::take(&mut es.pending_sculpt_edits);
            let sculpt_undo = es.pending_sculpt_undo.take();

            // Drain pending paint edits and undo.
            let paint_edits = std::mem::take(&mut es.pending_paint_edits);
            let paint_undo = es.pending_paint_undo.take();

            // Drain play mode requests.
            let f_play_start = std::mem::take(&mut es.pending_play_start);
            let f_play_stop = std::mem::take(&mut es.pending_play_stop);

            // Camera store values (for bound widgets).
            let camera_store = crate::engine_loop_store::CameraStoreValues {
                fov_degrees: es.editor_camera_fov_degrees(),
                fly_speed: es.camera_control.fly_speed,
                near: es.editor_camera_near(),
                far: es.editor_camera_far(),
            };

            // Reset per-frame deltas last.
            es.reset_frame_deltas();

            (cam_snap, debug_mode, convert_to_voxel, remap_material,
             set_prim_mat,
             environment, lights,
             scene, sel, gm, gizmo_axis, grid, emode, brush_radius, brush_strength, brush_falloff,
             sculpt_edits, sculpt_undo, sculpting_active,
             paint_edits, paint_undo,
             f_play_start, f_play_stop,
             camera_store, es.viewport_camera)
        };

        // d. Apply extracted data to engine (no lock held).
        engine.sync_camera_snapshot(&camera_snap);
        crate::engine_loop_store::push_camera_to_store(&store_push_buffer, &f_camera_store);
        crate::engine_loop_store::push_modes_to_store(&store_push_buffer, f_editor_mode, f_gizmo_mode);
        crate::engine_loop_store::push_viewport_camera_to_store(&store_push_buffer, f_viewport_camera);
        if let Some(mode) = f_debug_mode {
            engine.set_debug_mode(mode);
        }
        crate::engine_loop_store::push_debug_and_grid_to_store(
            &store_push_buffer, engine.debug_mode(), f_show_grid,
        );
        crate::engine_loop_store::push_brush_to_store(
            &store_push_buffer, f_brush_radius, f_brush_strength, f_brush_falloff,
        );
        if let Some(ref env) = f_environment {
            engine.apply_environment_settings(env);
            engine.lights_dirty = true;
            // Push environment fields to UI Store for bound widgets.
            let active_cam = editor_state.lock().ok()
                .and_then(|es| es.editor_camera_entity);
            crate::engine_loop_store::push_environment_to_store(
                &store_push_buffer, env, active_cam,
            );
        }
        if let Some(lights) = f_lights {
            engine.world_lights = lights;
            engine.lights_dirty = true;
        }

        // e0. Process pending convert-to-voxel.
        if let Some((obj_id, voxel_size)) = f_convert_to_voxel {
            engine_loop_edits::process_convert_to_voxel(
                obj_id, voxel_size, &mut engine, &editor_state, &mut scene_clone, &mut frame_dirty_objects,
            );
        }

        // e1b. Process pending material remap.
        if let Some((object_id, from_material, to_material)) = f_remap_material {
            engine_loop_edits::process_remap_material(
                object_id, from_material, to_material,
                &mut engine, &editor_state, &scene_clone, &mut frame_dirty_objects,
            );
        }

        // e1c. Process pending primitive material change.
        if let Some((object_id, new_mat_id)) = f_set_prim_mat {
            engine_loop_edits::process_set_primitive_material(
                object_id, new_mat_id, &editor_state, &mut scene_clone, &mut frame_dirty_objects,
            );
        }

        // e2. Process sculpt undo.
        if let Some((undo_uuid, snapshots)) = f_sculpt_undo {
            engine.apply_sculpt_undo(&snapshots);
            // Look up SDF object ID from UUID.
            if let Ok(es_ref) = editor_state.lock() {
                if let Some((_, record)) = es_ref.world.entity_records().find(|(uid, _)| **uid == undo_uuid) {
                    if let Some(obj_id) = record.sdf_object_id {
                        frame_dirty_objects.push(obj_id);
                    }
                }
            }
        }

        // e2b. Process paint undo.
        if let Some((undo_uuid, snapshots)) = f_paint_undo {
            engine.apply_sculpt_undo(&snapshots);
            if let Ok(es_ref) = editor_state.lock() {
                if let Some((_, record)) = es_ref.world.entity_records().find(|(uid, _)| **uid == undo_uuid) {
                    if let Some(obj_id) = record.sdf_object_id {
                        frame_dirty_objects.push(obj_id);
                    }
                }
            }
        }

        // e3. Process sculpt edits.
        if !f_sculpt_edits.is_empty() {
            engine_loop_edits::process_sculpt_edits(
                &f_sculpt_edits, &mut engine, &editor_state, &mut scene_clone, &mut frame_dirty_objects,
            );
        }

        // e3b. Process paint edits.
        if !f_paint_edits.is_empty() {
            engine_loop_edits::process_paint_edits(
                &f_paint_edits, &mut engine, &editor_state, &mut frame_dirty_objects, &mut dirty.scene,
            );
        }

        // e4. Process pending voxel_slice request.
        if let Some(req) = pending_voxel_slice {
            engine_loop_edits::process_voxel_slice(req, &engine, &scene_clone, &shared_state);
        }

        // e5. Process pending spatial_query request.
        if let Some(req) = pending_spatial_query {
            engine_loop_edits::process_spatial_query(req, &engine, &scene_clone, &shared_state);
        }

        // e6. Process pending MCP sculpt request.
        if let Some(req) = pending_mcp_sculpt {
            engine_loop_edits::process_mcp_sculpt(
                req, &mut engine, &editor_state, &mut scene_clone, &shared_state, &mut frame_dirty_objects,
            );
        }

        // e7. Process pending object_shape request.
        if let Some(obj_id) = pending_object_shape {
            engine_loop_edits::process_object_shape(obj_id, &engine, &scene_clone, &shared_state);
        }

        // f. Apply incremental dirty tracking to the engine.
        if frame_topology_changed {
            engine.topology_changed = true;
        }
        for obj_id in &frame_dirty_objects {
            engine.dirty_objects.insert(*obj_id);
        }
        engine.spawned_objects.extend(frame_spawned);
        engine.despawned_objects.extend(frame_despawned);

        // g. Build wireframe overlays (delegated to engine_loop_ui).
        engine_loop_ui::build_wireframe_overlays(
            &mut engine, &editor_state, &shared_state, &scene_clone,
            camera_snap.position, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius, f_brush_falloff,
            f_sculpting_active, play_state.is_playing(),
        );

        // g2. Process file watcher events (material + shader + source hot-reload).
        if let Some(built_dylib_path) = engine.process_file_events() {
            // Game crate build completed — hot-reload the dylib.
            game_dylib_loader = engine_loop_io::load_game_dylib_public(
                &built_dylib_path, &gameplay_registry, game_dylib_loader.take(),
            );
            if let Ok(reg) = gameplay_registry.lock() {
                if let Ok(()) = behavior_executor.rebuild(&reg) {
                    log::info!("Behavior schedule rebuilt after hot-reload");
                }
            }
            dirty.scene = true;
            engine.console.info("Scripts reloaded");
            rinch::shell::rinch_runtime::run_on_main_thread(|| {
                if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
                    ui.dylib_ready.set(true);
                }
            });
        }
        // g3. Sync material library to GPU if dirty.
        engine.sync_materials();

        // g4. Play mode transitions and behavior tick.
        //     Block play start if no dylib is loaded.
        let safe_play_start = f_play_start && game_dylib_loader.is_some();
        {
            let mut game_store = game_store_arc.lock().expect("game_store lock");
            engine_loop_play::tick_play_mode(
                &mut play_state,
                safe_play_start,
                f_play_stop,
                &mut scene_clone,
                &mut engine,
                &shared_state,
                &gameplay_registry,
                &mut behavior_executor,
                &mut game_store,
                &editor_state,
                &stable_ids,
                dt,
            );
        }

        // h. Render frame to offscreen texture (skip if no project loaded).
        let t_render_start = std::time::Instant::now();
        let has_project = engine.project_root.is_some();
        if has_project {
            engine.render_frame_offscreen(&scene_clone);
        }
        let t_render_end = std::time::Instant::now();

        // i. Synchronous readback -- wait for GPU, read pixels, submit to compositor.
        let (pixels, px_w, px_h, t_submit_start, t_submit_end) = if has_project {
            let (pixels, px_w, px_h) = engine.map_readback();
            let ts = std::time::Instant::now();
            surface_writer.submit_frame(&pixels, px_w, px_h);
            let te = std::time::Instant::now();
            (pixels, px_w, px_h, ts, te)
        } else {
            (Vec::new(), 0, 0, t_render_end, t_render_end)
        };

        // Log frame breakdown every 60 frames.
        static FRAME_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let fn_ = FRAME_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // (Frame profiling log suppressed — uncomment to debug frame timing.)
        // if fn_ % 60 == 0 {
        //     eprintln!(
        //         "[FRAME] cpu_submit: {:.2}ms  submit_frame: {:.2}ms  dt: {:.2}ms",
        //         (t_render_end - t_render_start).as_secs_f64() * 1000.0,
        //         (t_submit_end - t_submit_start).as_secs_f64() * 1000.0,
        //         dt as f64 * 1000.0,
        //     );
        // }
        let _ = fn_;

        // i2. If screenshot was requested, store the pixels in shared_state.
        if let Ok(mut ss) = shared_state.lock() {
            if ss.screenshot_requested {
                ss.frame_pixels = pixels;
                ss.frame_width = px_w;
                ss.frame_height = px_h;
                ss.screenshot_requested = false;
            }
        }

        // i3. Material preview -- dispatch + readback.
        let (preview_slot, preview_prim) = if let Ok(mut ss) = shared_state.lock() {
            let slot = ss.preview_material_slot;
            let prim = ss.preview_primitive_type;
            if ss.preview_dirty { ss.preview_dirty = false; }
            (slot, prim)
        } else {
            (None, 0)
        };
        if let Some(slot) = preview_slot {
            let materials = engine.material_library.lock().unwrap().all_materials().to_vec();
            engine.material_preview.dispatch_render(
                &engine.ctx.device, &engine.ctx.queue, &materials, slot, preview_prim,
            );
            if let Some((pixels, pw, ph)) = engine.material_preview.read_pixels(&engine.ctx.device) {
                preview_writer.submit_frame(&pixels, pw, ph);
            }
        }

        // j. Update shared_state for MCP observation.
        if let Ok(mut ss) = shared_state.lock() {
            ss.camera_position = camera_snap.position;
            ss.camera_yaw = camera_snap.yaw;
            ss.camera_pitch = camera_snap.pitch;
            ss.camera_fov = camera_snap.fov_degrees;
            ss.frame_time_ms = dt as f64 * 1000.0;
            ss.frame_width = current_vp.0;
            ss.frame_height = current_vp.1;
            ss.shader_names = engine.shader_composer.shader_info();
        }

        // k. GPU pick is handled inside render_frame_offscreen.

        // l. Process GPU pick result.
        engine_loop_ui::process_gpu_pick(
            &shared_state, &editor_state, &engine, f_sculpting_active, &gameplay_registry,
        );

        // m. Process GPU brush hit result.
        engine_loop_ui::process_brush_hit(&shared_state, &editor_state);

        // n0. Push dirty data into UI signals.
        if first_frame {
            first_frame = false;
        }
        engine_loop_ui::push_dirty_ui_signals(&dirty, &editor_state, &engine, &gameplay_registry, &store_push_buffer);

        // n-store. Drain UI Store push buffer on main thread.
        rinch::shell::rinch_runtime::run_on_main_thread(|| {
            if let Some(store) = rinch::core::context::try_use_context::<crate::store::UiStore>() {
                store.drain_pushes();
            }
        });

        // n. Periodically push FPS + camera position to the main thread.
        if last_fps_push.elapsed() >= std::time::Duration::from_millis(500) {
            last_fps_push = std::time::Instant::now();
            let fps_ms = dt as f64 * 1000.0;
            // Push to store for status bar.
            crate::engine_loop_store::push_fps_to_store(&store_push_buffer, fps_ms);
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                    ui.fps.set(fps_ms);
                }
            });
        }
        if last_camera_push.elapsed() >= std::time::Duration::from_millis(250) {
            last_camera_push = std::time::Instant::now();
            if let Ok(es) = editor_state.lock() {
                let snap = es.extract_camera_snapshot();
                // Push to store for status bar.
                crate::engine_loop_store::push_camera_position_to_store(
                    &store_push_buffer, snap.position,
                );
                rinch::shell::rinch_runtime::run_on_main_thread(move || {
                    if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                        ui.camera_display_pos.set(snap.position);
                    }
                });
            }
        }

        // o. Frame pacing — ensure minimum 2ms per frame (~500fps cap).
        //    When a project is loaded, GPU readback provides natural pacing.
        //    Without it the loop spins at 100K+ fps, pegging a CPU core.
        const MIN_FRAME_DURATION: std::time::Duration = std::time::Duration::from_millis(2);
        let elapsed = frame_start.elapsed();
        if elapsed < MIN_FRAME_DURATION {
            std::thread::sleep(MIN_FRAME_DURATION - elapsed);
        }
    }
}
