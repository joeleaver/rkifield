//! Engine background thread: render loop, sculpt processing, wireframe overlays.

use std::sync::{Arc, Mutex};

use rinch::render_surface::{GpuTextureRegistrar, SurfaceWriter};

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorState, UiSignals};
use crate::engine::EditorEngine;
use crate::engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};
use crate::ui_snapshot::UiSnapshot;

/// Data bundle passed to the engine thread.
pub(crate) struct EngineThreadData {
    pub(crate) editor_state: Arc<Mutex<EditorState>>,
    pub(crate) shared_state: Arc<Mutex<SharedState>>,
    /// Receiver for UI→engine commands (replaces many direct EditorState locks).
    pub(crate) cmd_rx: crossbeam::channel::Receiver<EditorCommand>,
    /// Lock-free snapshot published each frame for the UI thread to read.
    pub(crate) ui_snapshot: Arc<arc_swap::ArcSwap<UiSnapshot>>,
    /// CPU pixel writer for submitting rendered frames to the compositor.
    pub(crate) surface_writer: SurfaceWriter,
    /// GPU texture registrar — provides layout size and texture submission.
    pub(crate) gpu_registrar: GpuTextureRegistrar,
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
        ui_snapshot,
        surface_writer,
        gpu_registrar,
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

        // Fill SDF padding bricks for smooth normals at narrow-band boundaries.
        engine.init_sdf_padding(es.world.scene_mut());

        // Seed editor light list from render lights (point/spot only).
        for rl in &engine.world_lights {
            use crate::light_editor::{EditorLight, EditorLightType};
            if rl.light_type == 0 {
                continue; // Skip directional — driven by environment
            }
            let light_type = match rl.light_type {
                2 => EditorLightType::Spot,
                _ => EditorLightType::Point,
            };
            es.light_editor.add_light_full(EditorLight {
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

    log::info!("Engine thread: engine ready, entering render loop");

    let mut last_frame = std::time::Instant::now();
    let mut last_fps_push = std::time::Instant::now();
    let mut current_vp = engine.viewport_size();

    loop {
        let frame_start = std::time::Instant::now();
        let dt = frame_start.duration_since(last_frame).as_secs_f32();
        last_frame = frame_start;

        // Track whether scene topology or individual objects changed this frame.
        // Used to drive incremental GPU updates (skip re-flatten on static frames).
        let mut frame_topology_changed = false;
        let mut frame_dirty_objects: Vec<u32> = Vec::new();

        // a. Check viewport size from layout — the compositor updates
        //    layout_size each frame with the physical pixel dimensions of
        //    the RenderSurface element in the DOM.
        let desired_vp = {
            let (w, h) = gpu_registrar.layout_size();
            (w.max(64), h.max(64))
        };
        if desired_vp != current_vp {
            if let Some(_view) = engine.resize_viewport(desired_vp.0, desired_vp.1) {
                current_vp = desired_vp;
                log::debug!("Engine: resized to {}x{}", desired_vp.0, desired_vp.1);

                // Resize recreates all GPU passes with default uniforms.
                // Mark environment dirty so apply_environment re-pushes
                // all post-process settings (DOF, bloom, tone map, etc.)
                // on the next frame.
                if let Ok(mut es) = editor_state.lock() {
                    es.environment.mark_dirty();
                }
            }
        }

        // b. Drain pending MCP commands from shared_state (lock SS briefly,
        //    then release before touching editor_state — avoids nested locks).
        let pending_camera;
        let pending_debug;
        let pending_voxel_slice;
        let pending_spatial_query;
        let pending_mcp_sculpt;
        let pending_object_shape;
        let pending_mcp_fix_sdfs;
        {
            if let Ok(mut ss) = shared_state.lock() {
                pending_camera = ss.pending_camera.take();
                pending_debug = ss.pending_debug_mode.take();
                pending_voxel_slice = ss.pending_voxel_slice.take();
                pending_spatial_query = ss.pending_spatial_query.take();
                pending_mcp_sculpt = ss.pending_mcp_sculpt.take();
                pending_object_shape = ss.pending_object_shape.take();
                pending_mcp_fix_sdfs = ss.pending_fix_sdfs.take();
            } else {
                pending_camera = None;
                pending_debug = None;
                pending_voxel_slice = None;
                pending_spatial_query = None;
                pending_mcp_sculpt = None;
                pending_object_shape = None;
                pending_mcp_fix_sdfs = None;
            }
        }

        // b2. Drain UI commands and apply to EditorState (brief lock).
        //
        // Process KeyDown AFTER KeyUp so that X11/Wayland synthetic
        // key-repeat pairs (Release+Press) don't flicker keys_pressed.
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
                        _ => {}
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
        //    We inline what was formerly `take_engine_snapshot()` to avoid the
        //    intermediate `EngineSnapshot` struct.
        let (
            camera, f_debug_mode, f_revoxelize, f_fix_sdfs, f_environment, f_lights,
            mut scene_clone, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius,
            f_sculpt_edits, f_sculpt_undo, f_sculpting_active,
        ) = {
            let mut es = match editor_state.lock() {
                Ok(es) => es,
                Err(_) => continue, // poisoned — skip frame
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
            let revoxelize = es.pending_revoxelize.take();
            let fix_sdfs = es.pending_fix_sdfs.take();

            // Consume undo/redo.
            if es.pending_undo {
                es.pending_undo = false;
                if let Some(action) = es.undo.undo() {
                    es.apply_undo_action(&action, true);
                    frame_topology_changed = true; // Conservative — undo can be spawn/despawn/transform
                }
            }
            if es.pending_redo {
                es.pending_redo = false;
                if let Some(action) = es.undo.redo() {
                    es.apply_undo_action(&action, false);
                    frame_topology_changed = true;
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
                frame_topology_changed = true;
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
                        frame_topology_changed = true;
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
                                frame_topology_changed = true;
                            }
                        }
                    }
                }
            }

            // Detect gizmo-driven transform changes: if gizmo is dragging,
            // the selected object's transform is being modified by the UI thread.
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
                    use crate::light_editor::EditorLightType;
                    rkf_render::Light {
                        light_type: match el.light_type {
                            EditorLightType::Point => 1,
                            EditorLightType::Spot => 2,
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

            // Build a merged scene from all scenes (multi-scene rendering).
            // Recompute AABBs on the clone — character animation mutates
            // the clone anyway, so we discard it after render.
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

            let cam = es.editor_camera;
            let sel = es.selected_entity;
            let gm = es.gizmo.mode;
            let grid = es.show_grid;
            let emode = es.mode;
            let sculpting_active = es.sculpt.active_stroke.is_some();

            // Drain pending sculpt edits and undo.
            let sculpt_edits = std::mem::take(&mut es.pending_sculpt_edits);
            let sculpt_undo = es.pending_sculpt_undo.take();

            // Reset per-frame deltas last.
            es.reset_frame_deltas();

            (cam, debug_mode, revoxelize, fix_sdfs, environment, lights,
             scene, sel, gm, gizmo_axis, grid, emode, brush_radius,
             sculpt_edits, sculpt_undo, sculpting_active)
        };

        // d. Apply extracted data to engine (no lock held — UI thread runs freely).
        engine.sync_camera(&camera);
        if let Some(mode) = f_debug_mode {
            engine.set_debug_mode(mode);
        }
        if let Some(ref env) = f_environment {
            engine.apply_environment_snapshot(env);
            engine.lights_dirty = true; // sun direction/color changed
        }
        if let Some(lights) = f_lights {
            engine.world_lights = lights;
            engine.lights_dirty = true;
        }

        // e. Process pending re-voxelize (needs brief lock for scene mutation).
        if let Some(obj_id) = f_revoxelize {
            if let Ok(mut es) = editor_state.lock() {
                engine.process_revoxelize(es.world.scene_mut(), obj_id);
            }
            frame_dirty_objects.push(obj_id);
        }

        // e1. Process pending Fix SDFs (BFS SDF re-initialization).
        if let Some(obj_id) = f_fix_sdfs {
            if let Ok(mut es) = editor_state.lock() {
                engine.process_fix_sdfs(es.world.scene_mut(), obj_id);
            }
            frame_dirty_objects.push(obj_id);
        }

        // e2. Process sculpt undo (restore brick snapshots to CPU pool + GPU).
        if let Some(snapshots) = f_sculpt_undo {
            engine.apply_sculpt_undo(&snapshots);
            // Sculpt undo changes brick data — mark topology changed to rebake AABBs.
            frame_topology_changed = true;
        }

        // e3. Process sculpt edits (CPU-side CSG, targeted GPU upload).
        if !f_sculpt_edits.is_empty() {
            if let Ok(mut es) = editor_state.lock() {
                // Ensure undo accumulator exists for this stroke.
                if es.sculpt_undo_accumulator.is_none() {
                    let obj_id = f_sculpt_edits[0].object_id as u64;
                    es.sculpt_undo_accumulator = Some(
                        crate::editor_state::SculptUndoAccumulator {
                            object_id: obj_id,
                            captured_slots: std::collections::HashSet::new(),
                            snapshots: Vec::new(),
                        },
                    );
                }

                let mut undo_acc = es.sculpt_undo_accumulator.take();
                let scene = es.world.scene_mut();
                let _modified = engine.apply_sculpt_edits(
                    scene,
                    &f_sculpt_edits,
                    undo_acc.as_mut(),
                );
                es.sculpt_undo_accumulator = undo_acc;

                // Rebuild the render clone since scene may have changed.
                scene_clone = {
                    let mut merged = rkf_core::scene::Scene::new("merged");
                    merged.objects = es.world.all_objects().cloned().collect();
                    for obj in &mut merged.objects {
                        obj.aabb = crate::placement::compute_object_local_aabb(obj);
                    }
                    merged
                };
            }
            // Mark sculpted objects dirty for incremental GPU update.
            for edit in &f_sculpt_edits {
                frame_dirty_objects.push(edit.object_id);
            }
        }

        // e4. Process pending voxel_slice request (CPU-side brick pool lookup).
        if let Some(req) = pending_voxel_slice {
            let result = engine.sample_voxel_slice(&scene_clone, req.object_id, req.y_coord);
            if let Ok(mut ss) = shared_state.lock() {
                match result {
                    Ok(slice) => ss.voxel_slice_result = Some(slice),
                    Err(e) => {
                        ss.push_log(rkf_core::automation::LogLevel::Error,
                            format!("voxel_slice error: {e}"));
                        // Store an empty result so the polling doesn't hang.
                        ss.voxel_slice_result = Some(rkf_core::automation::VoxelSliceResult {
                            origin: [0.0, 0.0],
                            spacing: 0.0,
                            width: 0,
                            height: 0,
                            y_coord: req.y_coord,
                            distances: vec![],
                            slot_status: vec![],
                        });
                    }
                }
            }
        }

        // e5. Process pending spatial_query request (CPU-side SDF evaluation).
        if let Some(req) = pending_spatial_query {
            let result = engine.sample_spatial_query(&scene_clone, req.world_pos);
            if let Ok(mut ss) = shared_state.lock() {
                ss.spatial_query_result = Some(result);
            }
        }

        // e6. Process pending MCP sculpt request (one-shot brush hit + undo).
        if let Some(req) = pending_mcp_sculpt {
            let mcp_sculpt_obj_id = req.object_id;
            let sculpt_result = (|| -> Result<(), String> {
                let brush_type = match req.mode.as_str() {
                    "add" => crate::sculpt::BrushType::Add,
                    "subtract" => crate::sculpt::BrushType::Subtract,
                    "smooth" => crate::sculpt::BrushType::Smooth,
                    other => return Err(format!("invalid mode: {other}")),
                };

                let edit_request = crate::sculpt::SculptEditRequest {
                    object_id: req.object_id,
                    world_position: req.position,
                    settings: crate::sculpt::BrushSettings {
                        brush_type,
                        shape: crate::sculpt::BrushShape::Sphere,
                        radius: req.radius,
                        strength: req.strength,
                        material_id: req.material_id,
                        falloff: 0.5,
                    },
                };

                // Create one-shot undo accumulator.
                let mut undo_acc = crate::editor_state::SculptUndoAccumulator {
                    object_id: req.object_id as u64,
                    captured_slots: std::collections::HashSet::new(),
                    snapshots: Vec::new(),
                };

                if let Ok(mut es) = editor_state.lock() {
                    let scene = es.world.scene_mut();
                    engine.apply_sculpt_edits(
                        scene,
                        &[edit_request],
                        Some(&mut undo_acc),
                    );

                    // Push undo entry immediately.
                    if !undo_acc.snapshots.is_empty() {
                        es.undo.push(crate::undo::UndoAction {
                            kind: crate::undo::UndoActionKind::SculptStroke {
                                object_id: req.object_id as u64,
                                brick_snapshots: undo_acc.snapshots,
                            },
                            timestamp_ms: 0,
                            description: format!("MCP sculpt ({}) on obj {}", req.mode, req.object_id),
                        });
                    }

                    // Rebuild scene_clone since voxel data changed.
                    scene_clone = {
                        let mut merged = rkf_core::scene::Scene::new("merged");
                        merged.objects = es.world.all_objects().cloned().collect();
                        for obj in &mut merged.objects {
                            obj.aabb = crate::placement::compute_object_local_aabb(obj);
                        }
                        merged
                    };
                }

                Ok(())
            })();

            if let Ok(mut ss) = shared_state.lock() {
                ss.mcp_sculpt_result = Some(sculpt_result);
            }
            frame_dirty_objects.push(mcp_sculpt_obj_id);
        }

        // e7. Process pending object_shape request (CPU-side brick map lookup).
        if let Some(obj_id) = pending_object_shape {
            let result = engine.sample_object_shape(&scene_clone, obj_id);
            if let Ok(mut ss) = shared_state.lock() {
                match result {
                    Ok(shape) => ss.object_shape_result = Some(shape),
                    Err(e) => {
                        ss.push_log(rkf_core::automation::LogLevel::Error,
                            format!("object_shape error: {e}"));
                        // Store a minimal result so polling doesn't hang.
                        ss.object_shape_result = Some(rkf_core::automation::ObjectShapeResult {
                            object_id: obj_id,
                            dims: [0, 0, 0],
                            voxel_size: 0.0,
                            aabb_min: [0.0; 3],
                            aabb_max: [0.0; 3],
                            empty_count: 0,
                            interior_count: 0,
                            surface_count: 0,
                            y_slices: vec![],
                        });
                    }
                }
            }
        }

        // e8. Process pending MCP fix_sdfs request.
        if let Some(obj_id) = pending_mcp_fix_sdfs {
            let ok = engine.process_fix_sdfs(&mut scene_clone, obj_id);
            if let Ok(mut ss) = shared_state.lock() {
                if ok {
                    ss.fix_sdfs_result = Some(Ok(()));
                } else {
                    ss.fix_sdfs_result = Some(Err(format!(
                        "fix_sdfs: object {obj_id} not found or not voxelized"
                    )));
                }
            }
            frame_dirty_objects.push(obj_id);
        }

        // f. Apply incremental dirty tracking to the engine.
        if frame_topology_changed {
            engine.topology_changed = true;
        }
        for obj_id in &frame_dirty_objects {
            engine.dirty_objects.insert(*obj_id);
        }

        // g. Build wireframe overlays (selection, gizmos, grid) — staged
        //    so they are drawn in the same GPU submit as the main render.
        {
            let mut wf_verts: Vec<crate::wireframe::LineVertex> = Vec::new();
            let cam_pos = camera.position;

            // Light wireframe + translate gizmo for selected light.
            if let Some(crate::editor_state::SelectedEntity::Light(lid)) = f_selected {
                if let Ok(es) = editor_state.lock() {
                    if let Some(light) = es.light_editor.get_light(lid) {
                        let lc = [1.0, 0.9, 0.5, 1.0];
                        match light.light_type {
                            crate::light_editor::EditorLightType::Point => {
                                wf_verts.extend(crate::wireframe::point_light_wireframe(
                                    light.position, light.range, lc,
                                ));
                            }
                            crate::light_editor::EditorLightType::Spot => {
                                wf_verts.extend(crate::wireframe::spot_light_wireframe(
                                    light.position, light.direction,
                                    light.range, light.spot_outer_angle, lc,
                                ));
                            }
                        }
                        // Translate gizmo at light position.
                        let gc = light.position;
                        let cam_dist = (gc - cam_pos).length();
                        let gizmo_size = cam_dist * 0.12;
                        wf_verts.extend(crate::wireframe::translate_gizmo_wireframe(
                            gc, gizmo_size, f_gizmo_axis, cam_pos,
                        ));
                    }
                }
            }

            // Selection AABB + transform gizmo for selected objects.
            // Hidden while actively sculpting to avoid visual clutter.
            if let Some(crate::editor_state::SelectedEntity::Object(eid)) = f_selected.filter(|_| !f_sculpting_active) {
                let color = [0.3, 0.7, 1.0, 1.0]; // Light blue
                let mut gizmo_center: Option<glam::Vec3> = None;

                for obj in &scene_clone.objects {
                    let obj_id = obj.id as u64;
                    let root_world = glam::Mat4::from_scale_rotation_translation(
                        obj.scale,
                        obj.rotation,
                        obj.position,
                    );

                    if eid == obj_id {
                        if let Some((lmin, lmax)) =
                            crate::wireframe::compute_node_tree_aabb(&obj.root_node, glam::Mat4::IDENTITY)
                        {
                            wf_verts.extend(crate::wireframe::obb_wireframe(
                                lmin, lmax, obj.position, obj.rotation, obj.scale, color,
                            ));
                            let center = obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale);
                            gizmo_center = Some(center);
                        }
                    } else if let Some((child_node, child_world)) =
                        crate::wireframe::find_child_node_and_transform(
                            eid, obj_id, &obj.root_node, root_world,
                        )
                    {
                        if let Some((lmin, lmax)) =
                            crate::wireframe::compute_node_tree_aabb(child_node, glam::Mat4::IDENTITY)
                        {
                            let (child_scale, child_rot, child_pos) =
                                child_world.to_scale_rotation_translation();
                            wf_verts.extend(crate::wireframe::obb_wireframe(
                                lmin, lmax, child_pos, child_rot, child_scale, color,
                            ));
                            let center = child_pos + child_rot * ((lmin + lmax) * 0.5 * child_scale);
                            gizmo_center = Some(center);
                        }
                    }
                }

                // Transform gizmo at the center of the selected object.
                if let Some(gc) = gizmo_center {
                    let cam_dist = (gc - cam_pos).length();
                    let gizmo_size = cam_dist * 0.12;
                    match f_gizmo_mode {
                        crate::gizmo::GizmoMode::Translate => {
                            wf_verts.extend(crate::wireframe::translate_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                        crate::gizmo::GizmoMode::Rotate => {
                            wf_verts.extend(crate::wireframe::rotate_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                        crate::gizmo::GizmoMode::Scale => {
                            wf_verts.extend(crate::wireframe::scale_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                    }
                }
            }

            // Ground grid overlay.
            if f_show_grid {
                wf_verts.extend(crate::wireframe::ground_grid_wireframe(
                    cam_pos,
                    40.0,
                    1.0,
                    [0.3, 0.3, 0.3, 0.4],
                ));
            }

            // Brush preview circle in Sculpt/Paint modes — view-facing circle
            // showing the screen-space footprint of the brush.
            if matches!(f_editor_mode, crate::editor_state::EditorMode::Sculpt | crate::editor_state::EditorMode::Paint) {
                let brush_pos = shared_state.lock().ok()
                    .and_then(|s| s.brush_preview_pos);
                if let Some(pos) = brush_pos {
                    let brush_color = match f_editor_mode {
                        crate::editor_state::EditorMode::Sculpt => [0.0, 1.0, 1.0, 0.8],
                        crate::editor_state::EditorMode::Paint  => [1.0, 0.8, 0.0, 0.8],
                        _ => [1.0, 1.0, 1.0, 0.8],
                    };
                    let view_dir = (cam_pos - pos).normalize();
                    wf_verts.extend(crate::wireframe::circle_wireframe(
                        pos, view_dir, f_brush_radius, brush_color, 48,
                    ));
                }
            }

            engine.set_wireframe_vertices(wf_verts);
        }

        // h. Render frame to offscreen texture (wireframe drawn in same encoder).
        //    The readback copy is appended to the same encoder — one GPU submit.
        let t_render_start = std::time::Instant::now();
        engine.render_frame_offscreen(&scene_clone);
        let t_render_end = std::time::Instant::now();

        // i. Synchronous readback — wait for GPU, read pixels, submit to compositor.
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

        // j. Update shared_state for MCP observation (no editor_state lock).
        if let Ok(mut ss) = shared_state.lock() {
            ss.camera_position = camera.position;
            ss.camera_yaw = camera.fly_yaw;
            ss.camera_pitch = camera.fly_pitch;
            ss.camera_fov = camera.fov_y.to_degrees();
            ss.frame_time_ms = dt as f64 * 1000.0;
            ss.frame_width = current_vp.0;
            ss.frame_height = current_vp.1;
        }

        // k. GPU pick is handled inside render_frame_offscreen.

        // l. Process GPU pick result — sets selected_entity, signals main thread.
        //    Suppressed while actively sculpting to prevent accidental selection changes.
        {
            let pick = shared_state.lock().ok()
                .and_then(|mut ss| ss.pick_result.take());
            if let Some(object_id) = pick {
                if !f_sculpting_active {
                    let picked_entity = if object_id > 0 {
                        Some(crate::editor_state::SelectedEntity::Object(object_id as u64))
                    } else {
                        None
                    };
                    if let Ok(mut es) = editor_state.lock() {
                        es.selected_entity = picked_entity;
                    }
                    // Signal the main thread to update selection signal.
                    if let Ok(mut ss) = shared_state.lock() {
                        ss.pick_completed = true;
                    }
                }
            }
        }

        // m. Process GPU brush hit result — drive sculpt/paint stroke lifecycle.
        {
            let brush_hit = shared_state.lock().ok()
                .and_then(|mut ss| ss.brush_hit_result.take());
            if let Some(hit) = brush_hit {
                let (left_down, mode, selected_obj_id) = editor_state.lock().ok()
                    .map(|es| {
                        let sel_id = match es.selected_entity {
                            Some(crate::editor_state::SelectedEntity::Object(eid)) => Some(eid as u32),
                            _ => None,
                        };
                        (es.editor_input.mouse_buttons[0], es.mode, sel_id)
                    })
                    .unwrap_or((false, crate::editor_state::EditorMode::Default, None));

                // Only allow sculpt/paint on the currently selected object.
                let hit_on_selected = selected_obj_id == Some(hit.object_id);

                if left_down {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            crate::editor_state::EditorMode::Sculpt if hit_on_selected => {
                                if es.sculpt.active_stroke.is_some() {
                                    es.sculpt.continue_stroke(hit.position);
                                } else {
                                    es.sculpt.begin_stroke(hit.position);
                                }
                                // Queue a real-time sculpt edit for this point.
                                let settings = es.sculpt.current_settings.clone();
                                es.pending_sculpt_edits.push(
                                    crate::sculpt::SculptEditRequest {
                                        object_id: hit.object_id,
                                        world_position: hit.position,
                                        settings,
                                    },
                                );
                            }
                            crate::editor_state::EditorMode::Sculpt => {
                                // Hit a non-selected object — ignore.
                            }
                            crate::editor_state::EditorMode::Paint if hit_on_selected => {
                                if es.paint.active_stroke.is_some() {
                                    es.paint.continue_stroke(hit.position);
                                } else {
                                    es.paint.begin_stroke(hit.position);
                                }
                            }
                            crate::editor_state::EditorMode::Paint => {
                                // Hit a non-selected object — ignore.
                            }
                            _ => {}
                        }
                    }
                }

                if let Ok(mut ss) = shared_state.lock() {
                    ss.brush_preview_pos = Some(hit.position);
                }
            } else {
                let (left_down, mode) = editor_state.lock().ok()
                    .map(|es| (es.editor_input.mouse_buttons[0], es.mode))
                    .unwrap_or((false, crate::editor_state::EditorMode::Default));

                if !left_down && matches!(mode, crate::editor_state::EditorMode::Sculpt | crate::editor_state::EditorMode::Paint) {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            crate::editor_state::EditorMode::Sculpt => {
                                es.sculpt.end_stroke();
                                // Finalize sculpt undo: push accumulated brick snapshots.
                                if let Some(acc) = es.sculpt_undo_accumulator.take() {
                                    if !acc.snapshots.is_empty() {
                                        es.undo.push(crate::undo::UndoAction {
                                            kind: crate::undo::UndoActionKind::SculptStroke {
                                                object_id: acc.object_id,
                                                brick_snapshots: acc.snapshots,
                                            },
                                            timestamp_ms: 0,
                                            description: "Sculpt stroke".into(),
                                        });
                                    }
                                }
                            }
                            crate::editor_state::EditorMode::Paint => es.paint.end_stroke(),
                            _ => {}
                        }
                    }
                }
            }
        }

        // n0. Publish UiSnapshot for lock-free UI reads.
        {
            if let Ok(es) = editor_state.lock() {
                let snapshot = es.to_snapshot(0, dt as f64 * 1000.0);
                ui_snapshot.store(Arc::new(snapshot));
            }
        }

        // n. Periodically push the FPS value to the main thread (~4/sec).
        if last_fps_push.elapsed() >= std::time::Duration::from_millis(500) {
            last_fps_push = std::time::Instant::now();
            let fps_ms = dt as f64 * 1000.0;
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                    ui.fps.set(fps_ms);
                }
            });
        }

        // o. Yield to let OS schedule other threads between frames.
        std::thread::yield_now();
    }
}

/// Apply a single UI command to the editor state.
///
/// Called by the engine thread while holding the EditorState lock.
/// This is the command consumer for the UI→engine channel.
fn apply_editor_command(es: &mut EditorState, cmd: EditorCommand) {
    use crate::editor_command::EditorCommand::*;
    match cmd {
        // ── Input ────────────────────────────────────────────────
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

        // ── Scene mutations ──────────────────────────────────────
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

        // ── Gizmo ────────────────────────────────────────────────
        SetGizmoMode { mode } => {
            es.gizmo.mode = mode;
        }

        // ── Tool settings ────────────────────────────────────────
        SetEditorMode { mode } => {
            es.mode = mode;
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

        // ── Camera settings ──────────────────────────────────────
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

        // ── Environment ──────────────────────────────────────────
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

        // ── Lights ───────────────────────────────────────────────
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

        // ── Debug / view ─────────────────────────────────────────
        SetDebugMode { mode } => {
            es.pending_debug_mode = Some(mode);
        }
        ToggleGrid => {
            es.show_grid = !es.show_grid;
        }
        ToggleShortcuts => {
            es.show_shortcuts = !es.show_shortcuts;
        }

        // ── Object properties ────────────────────────────────────
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

        // ── Scene I/O ────────────────────────────────────────────
        OpenScene { path: _ } => {
            es.pending_open = true;
        }
        SaveScene { path: _ } => {
            es.pending_save = true;
        }

        // ── Voxel ops ────────────────────────────────────────────
        Revoxelize { object_id } => {
            es.pending_revoxelize = Some(object_id);
        }
        FixSdfs { object_id } => {
            es.pending_fix_sdfs = Some(object_id);
        }

        // ── Animation ────────────────────────────────────────────
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

        // ── Window management ────────────────────────────────────
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
