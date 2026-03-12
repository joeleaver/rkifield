//! Engine background thread: render loop, sculpt processing, wireframe overlays.

use std::sync::{Arc, Mutex};

use rinch::render_surface::{GpuTextureRegistrar, SurfaceWriter};

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, UiSignals};
use crate::engine::EditorEngine;
use crate::engine_loop_commands::{apply_editor_command, compute_material_usage};
use crate::engine_loop_edits;
use crate::engine_loop_io;
use crate::engine_loop_ui;
use crate::engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};

/// Data bundle passed to the engine thread.
pub(crate) struct EngineThreadData {
    pub(crate) editor_state: Arc<Mutex<EditorState>>,
    pub(crate) shared_state: Arc<Mutex<SharedState>>,
    /// Receiver for UI->engine commands.
    pub(crate) cmd_rx: crossbeam::channel::Receiver<EditorCommand>,
    /// Lock-free layout config shared between UI and engine threads.
    pub(crate) layout_backing: crate::layout::state::LayoutBacking,
    /// CPU pixel writer for submitting rendered frames to the compositor.
    pub(crate) surface_writer: SurfaceWriter,
    /// GPU texture registrar -- provides layout size and texture submission.
    pub(crate) gpu_registrar: GpuTextureRegistrar,
    /// CPU pixel writer for the material preview surface.
    pub(crate) preview_writer: SurfaceWriter,
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
    } = data;

    log::info!("Engine thread: creating dedicated GPU device");

    // Create engine with its own wgpu device (no sharing with compositor).
    let (mut engine, demo_scene) = EditorEngine::new_headless(
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        Arc::clone(&shared_state),
    );

    // 4. Store demo scene in editor_state (set world's scene directly).
    if let Ok(mut es) = editor_state.lock() {
        *es.world.scene_mut() = demo_scene;
        es.world.resync_entity_tracking();

        // Seed editor light list from render lights (point/spot only).
        for rl in &engine.world_lights {
            use crate::light_editor::{SceneLight, SceneLightType};
            if rl.light_type == 0 {
                continue; // Skip directional -- driven by environment
            }
            let light_type = match rl.light_type {
                2 => SceneLightType::Spot,
                _ => SceneLightType::Point,
            };
            es.light_editor.add_light_full(SceneLight {
                id: 0, // overwritten by add_light_full
                light_type,
                position: glam::Vec3::new(rl.pos_x, rl.pos_y, rl.pos_z),
                direction: glam::Vec3::new(rl.dir_x, rl.dir_y, rl.dir_z),
                color: glam::Vec3::new(rl.color_r, rl.color_g, rl.color_b),
                intensity: rl.intensity,
                range: rl.range,
                spot_inner_angle: rl.inner_angle,
                spot_outer_angle: rl.outer_angle,
                cast_shadows: rl.shadow_caster != 0,
                cookie_path: None,
            });
        }
        es.light_editor.clear_dirty();
    }

    // Auto-open last project if configured.
    {
        let config = crate::editor_config::load_editor_config();
        if let Some(ref project_path_str) = config.last_project_path {
            if std::path::Path::new(project_path_str).exists() {
                log::info!("Auto-opening last project: {project_path_str}");
                match rkf_runtime::load_project(project_path_str) {
                    Ok(pf) => {
                        let project_path = std::path::Path::new(project_path_str);
                        let project_root = rkf_runtime::project::project_root(project_path);
                        let default_scene_path = engine_loop_io::resolve_default_scene_path(&pf, &project_root);

                        if let Some(scene_path) = default_scene_path {
                            let sp = scene_path.to_string_lossy().to_string();
                            match crate::scene_io::load_v2_scene(&sp) {
                                Ok(sf) => {
                                    let mut new_scene = crate::scene_io::reconstruct_v2_scene(&sf);

                                    // Load voxelized objects from .rkf files.
                                    engine.clear_scene();
                                    let scene_dir = scene_path.parent()
                                        .unwrap_or(std::path::Path::new("."));
                                    engine_loop_io::load_scene_rkf_assets(
                                        &mut engine, &sf, &mut new_scene, scene_dir,
                                    );

                                    // Upload brick pool + brick map data to GPU.
                                    engine.reupload_brick_data();

                                    if let Ok(mut es) = editor_state.lock() {
                                        *es.world.scene_mut() = new_scene;
                                        es.current_scene_path = Some(sp);
                                        // Restore editor layout if saved in project.
                                        if let Some(ref layout_ron) = pf.editor_layout {
                                            layout_backing.from_ron(layout_ron);
                                            let lb = layout_backing.clone();
                                            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                                                let layout = rinch::core::use_context::<crate::layout::state::LayoutState>();
                                                layout.load_from_backing(&lb);
                                            });
                                        }
                                        es.current_project = Some(pf);
                                        es.current_project_path = Some(project_path_str.clone());
                                        es.world.resync_entity_tracking();

                                        // Restore subsystem state from property bag.
                                        engine_loop_io::restore_scene_properties(&mut es, &sf);
                                    }
                                    log::info!("Last project restored: {project_path_str}");
                                }
                                Err(e) => log::error!("Failed to load last project's scene: {e}"),
                            }
                        }
                    }
                    Err(e) => log::warn!("Failed to load last project '{}': {e}", project_path_str),
                }
            } else {
                log::info!("Last project path no longer exists: {project_path_str}");
            }
        }
    }

    log::info!("Engine thread: engine ready, entering render loop");

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
                if let Ok(mut es) = editor_state.lock() {
                    es.environment.mark_dirty();
                }
            }
        }

        // b. Drain pending MCP commands from shared_state.
        let pending_camera;
        let pending_debug;
        let pending_voxel_slice;
        let pending_spatial_query;
        let pending_mcp_sculpt;
        let pending_object_shape;
        {
            if let Ok(mut ss) = shared_state.lock() {
                pending_camera = ss.pending_camera.take();
                pending_debug = ss.pending_debug_mode.take();
                pending_voxel_slice = ss.pending_voxel_slice.take();
                pending_spatial_query = ss.pending_spatial_query.take();
                pending_mcp_sculpt = ss.pending_mcp_sculpt.take();
                pending_object_shape = ss.pending_object_shape.take();
            } else {
                pending_camera = None;
                pending_debug = None;
                pending_voxel_slice = None;
                pending_spatial_query = None;
                pending_mcp_sculpt = None;
                pending_object_shape = None;
            }
        }

        // b2. Drain UI commands and apply to EditorState (brief lock).
        let mut dirty = DirtyFlags::default();
        {
            if let Ok(mut es) = editor_state.lock() {
                let mut key_downs = Vec::new();
                while let Ok(cmd) = cmd_rx.try_recv() {
                    // Track transform commands that need GPU dirty marking.
                    match &cmd {
                        EditorCommand::SetObjectPosition { entity_id, .. }
                        | EditorCommand::SetObjectRotation { entity_id, .. }
                        | EditorCommand::SetObjectScale { entity_id, .. } => {
                            frame_dirty_objects.push(*entity_id as u32);
                        }
                        // Structural scene changes.
                        EditorCommand::SpawnPrimitive { .. }
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
                    match cmd {
                        EditorCommand::KeyDown { .. } => key_downs.push(cmd),
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
            camera, f_debug_mode, f_convert_to_voxel, f_remap_material,
            f_set_prim_mat,
            f_environment, f_lights,
            mut scene_clone, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius, f_brush_falloff,
            f_sculpt_edits, f_sculpt_undo, f_sculpting_active,
            f_paint_edits, f_paint_undo,
        ) = {
            let mut es = match editor_state.lock() {
                Ok(es) => es,
                Err(_) => continue, // poisoned -- skip frame
            };

            // Apply pending MCP commands (mutate before read).
            if let Some(cam) = pending_camera {
                es.editor_camera.position = cam.position;
                es.editor_camera.fly_yaw = cam.yaw;
                es.editor_camera.fly_pitch = cam.pitch;
                let dir = glam::Vec3::new(
                    -cam.yaw.sin() * cam.pitch.cos(),
                    cam.pitch.sin(),
                    -cam.yaw.cos() * cam.pitch.cos(),
                );
                es.editor_camera.target = es.editor_camera.position
                    + dir * es.editor_camera.orbit_distance;
            }
            if let Some(mode) = pending_debug {
                es.pending_debug_mode = Some(mode);
            }

            // Camera update (mutates EditorState input state).
            es.update_camera(dt);

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
                    match &action.kind {
                        crate::undo::UndoActionKind::SpawnEntity { entity_id } => {
                            frame_despawned.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::DespawnEntity { entity_id } => {
                            frame_spawned.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::Transform { entity_id, .. } => {
                            frame_dirty_objects.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::SculptStroke { object_id, .. } => {
                            frame_dirty_objects.push(*object_id as u32);
                        }
                        _ => {}
                    }
                }
            }
            if es.pending_redo {
                es.pending_redo = false;
                if let Some(action) = es.undo.redo() {
                    es.apply_undo_action(&action, false);
                    match &action.kind {
                        crate::undo::UndoActionKind::SpawnEntity { entity_id } => {
                            frame_spawned.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::DespawnEntity { entity_id } => {
                            frame_despawned.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::Transform { entity_id, .. } => {
                            frame_dirty_objects.push(*entity_id as u32);
                        }
                        crate::undo::UndoActionKind::SculptStroke { object_id, .. } => {
                            frame_dirty_objects.push(*object_id as u32);
                        }
                        _ => {}
                    }
                }
            }

            // Consume pending spawn.
            if let Some(prim_name) = es.pending_spawn.take() {
                let pos = es.editor_camera.target;
                let primitive = primitive_from_name(&prim_name);
                let entity = es.world.spawn(&prim_name)
                    .position_vec3(pos)
                    .sdf(primitive)
                    .material(0)
                    .build();
                es.selected_entity = Some(crate::editor_state::SelectedEntity::Object(entity.to_u64()));
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SpawnEntity {
                        entity_id: entity.to_u64(),
                    },
                    timestamp_ms: 0,
                    description: format!("Spawn {prim_name}"),
                });
                frame_spawned.push(entity.to_u64() as u32);
            }

            // Consume pending delete.
            if es.pending_delete {
                es.pending_delete = false;
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                    if let Some(entity) = es.world.find_entity_by_id(eid) {
                        es.undo.push(crate::undo::UndoAction {
                            kind: crate::undo::UndoActionKind::DespawnEntity {
                                entity_id: eid,
                            },
                            timestamp_ms: 0,
                            description: "Delete object".into(),
                        });
                        let _ = es.world.despawn(entity);
                        es.selected_entity = None;
                        frame_despawned.push(eid as u32);
                    }
                }
            }

            // Consume pending duplicate.
            if es.pending_duplicate {
                es.pending_duplicate = false;
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                    if let Some(src_entity) = es.world.find_entity_by_id(eid) {
                        if let (Ok(pos), Ok(rot), Ok(scale)) = (
                            es.world.position(src_entity),
                            es.world.rotation(src_entity),
                            es.world.scale(src_entity),
                        ) {
                            let root = es.world.root_node(src_entity).ok().cloned();
                            if let Some(root_node) = root {
                                let offset = rkf_core::WorldPosition::new(
                                    glam::IVec3::ZERO,
                                    pos.to_vec3() + glam::Vec3::new(1.0, 0.0, 0.0),
                                );
                                let new_entity = es.world.spawn(&root_node.name)
                                    .position(offset)
                                    .rotation(rot)
                                    .scale(scale)
                                    .sdf_tree(root_node)
                                    .build();
                                es.selected_entity = Some(
                                    crate::editor_state::SelectedEntity::Object(new_entity.to_u64()),
                                );
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::SpawnEntity {
                                        entity_id: new_entity.to_u64(),
                                    },
                                    timestamp_ms: 0,
                                    description: "Duplicate object".into(),
                                });
                                frame_spawned.push(new_entity.to_u64() as u32);
                            }
                        }
                    }
                }
            }

            // -- Scene/Project I/O (delegated to engine_loop_io) --------
            if es.pending_save {
                es.pending_save = false;
                engine_loop_io::handle_scene_save(&mut es, &engine);
            }
            if es.pending_save_as {
                es.pending_save_as = false;
                engine_loop_io::handle_scene_save_as(es, &editor_state, &engine);
                continue;
            }

            if es.pending_open {
                es.pending_open = false;
                let open_path = es.pending_open_path.take();
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                };
                engine_loop_io::handle_scene_open_impl(es, open_path, &mut io_ctx);
                continue;
            }

            if es.pending_new_project {
                es.pending_new_project = false;
                drop(es);
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                };
                engine_loop_io::handle_new_project(&mut io_ctx, &mut frame_topology_changed);
                continue;
            }

            if es.pending_open_project {
                es.pending_open_project = false;
                let open_path = es.pending_open_project_path.take();
                drop(es);
                let mut io_ctx = engine_loop_io::IoContext {
                    engine: &mut engine,
                    editor_state: &editor_state,
                    layout_backing: &layout_backing,
                };
                engine_loop_io::handle_open_project(open_path, &mut io_ctx, &mut frame_topology_changed);
                continue;
            }

            if es.pending_save_project {
                es.pending_save_project = false;
                engine_loop_io::handle_save_project(&mut es, &engine, &layout_backing);
            }

            // Detect gizmo-driven transform changes.
            if es.gizmo.dragging {
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                    frame_dirty_objects.push(eid as u32);
                }
            }

            // Environment: clone only when dirty.
            let environment = if es.environment.is_dirty() {
                let env = es.environment.clone();
                es.environment.clear_dirty();
                Some(env)
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

            // Build a merged scene from all scenes.
            let scene = {
                let mut merged = rkf_core::scene::Scene::new("merged");
                merged.objects = es.world.all_objects().cloned().collect();
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
            let brush_falloff = match es.mode {
                crate::editor_state::EditorMode::Paint => es.paint.current_settings.falloff,
                crate::editor_state::EditorMode::Sculpt => es.sculpt.current_settings.falloff,
                _ => 0.0,
            };

            let cam = es.editor_camera;
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

            // Reset per-frame deltas last.
            es.reset_frame_deltas();

            (cam, debug_mode, convert_to_voxel, remap_material,
             set_prim_mat,
             environment, lights,
             scene, sel, gm, gizmo_axis, grid, emode, brush_radius, brush_falloff,
             sculpt_edits, sculpt_undo, sculpting_active,
             paint_edits, paint_undo)
        };

        // d. Apply extracted data to engine (no lock held).
        engine.sync_camera(&camera);
        if let Some(mode) = f_debug_mode {
            engine.set_debug_mode(mode);
        }
        if let Some(ref env) = f_environment {
            engine.apply_environment_snapshot(env);
            engine.lights_dirty = true;
        }
        if let Some(lights) = f_lights {
            engine.world_lights = lights;
            engine.lights_dirty = true;
        }

        // e0. Process pending convert-to-voxel.
        if let Some(obj_id) = f_convert_to_voxel {
            engine_loop_edits::process_convert_to_voxel(
                obj_id, &mut engine, &editor_state, &mut scene_clone, &mut frame_dirty_objects,
            );
        }

        // e1b. Process pending material remap.
        if let Some((object_id, from_material, to_material)) = f_remap_material {
            engine_loop_edits::process_remap_material(
                object_id, from_material, to_material,
                &mut engine, &scene_clone, &mut frame_dirty_objects,
            );
        }

        // e1c. Process pending primitive material change.
        if let Some((object_id, new_mat_id)) = f_set_prim_mat {
            engine_loop_edits::process_set_primitive_material(
                object_id, new_mat_id, &editor_state, &mut scene_clone, &mut frame_dirty_objects,
            );
        }

        // e2. Process sculpt undo.
        if let Some((undo_object_id, snapshots)) = f_sculpt_undo {
            engine.apply_sculpt_undo(&snapshots);
            frame_dirty_objects.push(undo_object_id as u32);
        }

        // e2b. Process paint undo.
        if let Some((undo_object_id, snapshots)) = f_paint_undo {
            engine.apply_sculpt_undo(&snapshots);
            frame_dirty_objects.push(undo_object_id as u32);
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
            camera.position, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius, f_brush_falloff,
            f_sculpting_active,
        );

        // g2. Process file watcher events (material + shader hot-reload).
        engine.process_file_events();
        // g3. Sync material library to GPU if dirty.
        engine.sync_materials();

        // h. Render frame to offscreen texture.
        let t_render_start = std::time::Instant::now();
        engine.render_frame_offscreen(&scene_clone);
        let t_render_end = std::time::Instant::now();

        // i. Synchronous readback -- wait for GPU, read pixels, submit to compositor.
        let (pixels, px_w, px_h) = engine.map_readback();
        let t_submit_start = std::time::Instant::now();
        surface_writer.submit_frame(&pixels, px_w, px_h);
        let t_submit_end = std::time::Instant::now();

        // Log frame breakdown every 60 frames.
        static FRAME_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let fn_ = FRAME_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if fn_ % 60 == 0 {
            eprintln!(
                "[FRAME] cpu_submit: {:.2}ms  submit_frame: {:.2}ms  dt: {:.2}ms",
                (t_render_end - t_render_start).as_secs_f64() * 1000.0,
                (t_submit_end - t_submit_start).as_secs_f64() * 1000.0,
                dt as f64 * 1000.0,
            );
        }

        // i2. If screenshot was requested, store the pixels in shared_state.
        {
            let do_screenshot = shared_state.lock()
                .map(|s| s.screenshot_requested)
                .unwrap_or(false);
            if do_screenshot {
                if let Ok(mut ss) = shared_state.lock() {
                    ss.frame_pixels = pixels;
                    ss.frame_width = px_w;
                    ss.frame_height = px_h;
                    ss.screenshot_requested = false;
                }
            }
        }

        // i3. Material preview -- dispatch + readback.
        {
            let (preview_slot, preview_prim) = {
                if let Ok(mut ss) = shared_state.lock() {
                    let slot = ss.preview_material_slot;
                    let prim = ss.preview_primitive_type;
                    if ss.preview_dirty {
                        ss.preview_dirty = false;
                    }
                    (slot, prim)
                } else {
                    (None, 0)
                }
            };
            if let Some(slot) = preview_slot {
                let materials = {
                    let lib = engine.material_library.lock().unwrap();
                    lib.all_materials().to_vec()
                };
                engine.material_preview.dispatch_render(
                    &engine.ctx.device, &engine.ctx.queue,
                    &materials, slot, preview_prim,
                );
                if let Some((pixels, pw, ph)) = engine.material_preview.read_pixels(&engine.ctx.device) {
                    preview_writer.submit_frame(&pixels, pw, ph);
                }
            }
        }

        // j. Update shared_state for MCP observation.
        if let Ok(mut ss) = shared_state.lock() {
            ss.camera_position = camera.position;
            ss.camera_yaw = camera.fly_yaw;
            ss.camera_pitch = camera.fly_pitch;
            ss.camera_fov = camera.fov_y.to_degrees();
            ss.frame_time_ms = dt as f64 * 1000.0;
            ss.frame_width = current_vp.0;
            ss.frame_height = current_vp.1;
            ss.shader_names = engine.shader_composer.shader_info();
        }

        // k. GPU pick is handled inside render_frame_offscreen.

        // l. Process GPU pick result.
        engine_loop_ui::process_gpu_pick(
            &shared_state, &editor_state, &engine, f_sculpting_active,
        );

        // m. Process GPU brush hit result.
        engine_loop_ui::process_brush_hit(&shared_state, &editor_state);

        // n0. Push dirty data into UI signals.
        if first_frame {
            dirty.scene = true;
            dirty.lights = true;
            dirty.materials = true;
            dirty.shaders = true;
            first_frame = false;
        }
        engine_loop_ui::push_dirty_ui_signals(&dirty, &editor_state, &engine);

        // n. Periodically push FPS + camera position to the main thread.
        if last_fps_push.elapsed() >= std::time::Duration::from_millis(500) {
            last_fps_push = std::time::Instant::now();
            let fps_ms = dt as f64 * 1000.0;
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                    ui.fps.set(fps_ms);
                }
            });
        }
        if last_camera_push.elapsed() >= std::time::Duration::from_millis(250) {
            last_camera_push = std::time::Instant::now();
            if let Ok(es) = editor_state.lock() {
                let cam_pos = es.editor_camera.position;
                rinch::shell::rinch_runtime::run_on_main_thread(move || {
                    if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                        ui.camera_display_pos.set(cam_pos);
                    }
                });
            }
        }

        // o. Yield to let OS schedule other threads between frames.
        std::thread::yield_now();
    }
}
