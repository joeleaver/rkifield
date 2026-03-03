//! RKIField editor — v2 object-centric render pipeline with rinch shell.
//!
//! Architecture:
//! - `rinch::shell::run_with_window_props_and_menu()` owns the window and event loop.
//! - The engine runs on a background thread, sharing the wgpu Device/Queue from `GpuHandle`.
//! - Engine output goes to an offscreen texture; `RenderSurfaceHandle::set_texture_source()`
//!   feeds it to rinch's compositor for zero-copy GPU compositing.
//! - Input flows through `SurfaceEvent` → `EditorState.editor_input` → engine thread.

#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod animation_preview;
mod automation;
mod camera;
mod debug_viz;
mod editor_state;
mod engine;
mod engine_viewport;
mod jfa_sdf;
mod environment;
mod gizmo;
mod input;
mod light_editor;
mod overlay;
mod paint;
mod placement;
mod properties;
mod scene_io;
mod sculpt;
mod ui;
mod undo;
mod wireframe;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use automation::SharedState;
use editor_state::{EditorState, SliderSignals, UiRevision, UiSignals};
use engine::EditorEngine;
use engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};

// ---------------------------------------------------------------------------
// IPC server setup
// ---------------------------------------------------------------------------

fn start_ipc_server(
    rt: &tokio::runtime::Runtime,
    api: Arc<dyn rkf_core::automation::AutomationApi>,
) -> (tokio::task::JoinHandle<()>, String) {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();

    // Write discovery metadata.
    let meta_path = socket_path.replace(".sock", ".json");
    let meta = serde_json::json!({
        "type": "editor",
        "pid": std::process::id(),
        "socket": &socket_path,
        "name": "RKIField Editor",
        "version": "0.1.0"
    });
    let _ = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap());

    let config = rkf_mcp::ipc::IpcConfig {
        socket_path: Some(socket_path.clone()),
        tcp_port: 0,
        mode: rkf_mcp::registry::ToolMode::Editor,
    };

    let handle = rt.spawn(async move {
        if let Err(e) = rkf_mcp::ipc::run_server(config, api).await {
            log::error!("IPC server error: {e}");
        }
    });

    (handle, socket_path)
}

/// Map a user-facing primitive name to an SdfPrimitive.
fn primitive_from_name(name: &str) -> rkf_core::SdfPrimitive {
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

fn cleanup_ipc(socket_path: &str) {
    let _ = std::fs::remove_file(socket_path);
    let meta_path = socket_path.replace(".sock", ".json");
    let _ = std::fs::remove_file(&meta_path);
}

// ---------------------------------------------------------------------------
// Engine background thread
// ---------------------------------------------------------------------------

/// Data bundle passed to the engine thread.
struct EngineThreadData {
    editor_state: Arc<Mutex<EditorState>>,
    shared_state: Arc<Mutex<SharedState>>,
    /// Send-able handle for registering the engine's GPU texture with the compositor.
    gpu_registrar: GpuTextureRegistrar,
}

fn engine_thread(data: EngineThreadData) {
    let EngineThreadData {
        editor_state,
        shared_state,
        gpu_registrar,
    } = data;

    // 1. Wait for GpuHandle — rinch initialises it during window creation.
    //    Poll every 10 ms; typically resolves in < 100 ms.
    let gpu = loop {
        if let Some(h) = gpu_handle() {
            break h.clone();
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };

    log::info!("Engine thread: got GpuHandle, initialising engine");

    // Cloning Arc<Device> / Arc<Queue> is cheap.
    let device = (*gpu.device).clone();
    let queue  = (*gpu.queue).clone();

    // 2. Create engine via new_with_device.
    let (mut engine, demo_scene) = EditorEngine::new_with_device(
        device,
        queue,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
        Arc::clone(&shared_state),
    );

    // 3. Register offscreen texture with the RenderSurface.
    //    TextureView implements Clone, so we can clone the initial view directly.
    {
        let (w, h) = engine.viewport_size();
        if let Some(view) = engine.offscreen_texture_view().cloned() {
            gpu_registrar.set_texture_source(view, w, h);
        }
    }

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

    // Frame pacing: cap the engine thread to ~120 fps so it doesn't
    // monopolise the editor_state mutex and starve the UI thread.
    let target_frame_time = std::time::Duration::from_micros(8333); // ~120 fps

    loop {
        let frame_start = std::time::Instant::now();
        let dt = frame_start.duration_since(last_frame).as_secs_f32();
        last_frame = frame_start;

        // a. Check viewport size from layout — the compositor updates
        //    gpu_registrar.layout_size() each frame with the physical pixel
        //    dimensions of the RenderSurface element in the DOM.
        let desired_vp = {
            let (w, h) = gpu_registrar.layout_size();
            (w.max(64), h.max(64))
        };
        if desired_vp != current_vp {
            if let Some(view) = engine.resize_viewport(desired_vp.0, desired_vp.1) {
                gpu_registrar.set_texture_source(view, desired_vp.0, desired_vp.1);
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
                }
            }
            if es.pending_redo {
                es.pending_redo = false;
                if let Some(action) = es.undo.redo() {
                    es.apply_undo_action(&action, false);
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
                es.selected_entity = Some(editor_state::SelectedEntity::Object(entity.to_u64()));
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SpawnEntity {
                        entity_id: entity.to_u64(),
                    },
                    timestamp_ms: 0,
                    description: format!("Spawn {prim_name}"),
                });
            }

            // Consume pending delete.
            if es.pending_delete {
                es.pending_delete = false;
                if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
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
                    }
                }
            }

            // Consume pending duplicate.
            if es.pending_duplicate {
                es.pending_duplicate = false;
                if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
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
                                    editor_state::SelectedEntity::Object(new_entity.to_u64()),
                                );
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::SpawnEntity {
                                        entity_id: new_entity.to_u64(),
                                    },
                                    timestamp_ms: 0,
                                    description: "Duplicate object".into(),
                                });
                            }
                        }
                    }
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
                editor_state::EditorMode::Sculpt => es.sculpt.current_settings.radius,
                editor_state::EditorMode::Paint => es.paint.current_settings.radius,
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
        }
        if let Some(lights) = f_lights {
            engine.world_lights = lights;
        }

        // e. Process pending re-voxelize (needs brief lock for scene mutation).
        if let Some(obj_id) = f_revoxelize {
            if let Ok(mut es) = editor_state.lock() {
                engine.process_revoxelize(es.world.scene_mut(), obj_id);
            }
        }

        // e1. Process pending Fix SDFs (BFS SDF re-initialization).
        if let Some(obj_id) = f_fix_sdfs {
            if let Ok(mut es) = editor_state.lock() {
                engine.process_fix_sdfs(es.world.scene_mut(), obj_id);
            }
        }

        // e2. Process sculpt undo (restore brick snapshots to CPU pool + GPU).
        if let Some(snapshots) = f_sculpt_undo {
            engine.apply_sculpt_undo(&snapshots);
        }

        // e3. Process sculpt edits (CPU-side CSG, targeted GPU upload).
        if !f_sculpt_edits.is_empty() {
            if let Ok(mut es) = editor_state.lock() {
                // Ensure undo accumulator exists for this stroke.
                if es.sculpt_undo_accumulator.is_none() {
                    let obj_id = f_sculpt_edits[0].object_id as u64;
                    es.sculpt_undo_accumulator = Some(
                        editor_state::SculptUndoAccumulator {
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
                let mut undo_acc = editor_state::SculptUndoAccumulator {
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
        }

        // f. Advance character animation on the render clone.
        engine.advance_character(&mut scene_clone);

        // g. Render frame to offscreen texture.
        engine.render_frame_offscreen(&scene_clone);

        // h. Build and draw wireframe overlays (selection, gizmos, grid).
        {
            let mut wf_verts: Vec<wireframe::LineVertex> = Vec::new();
            let cam_pos = camera.position;

            // Light gizmo for selected light.
            if let Some(editor_state::SelectedEntity::Light(lid)) = f_selected {
                if let Ok(es) = editor_state.lock() {
                    if let Some(light) = es.light_editor.get_light(lid) {
                        let lc = [1.0, 0.9, 0.5, 1.0];
                        match light.light_type {
                            light_editor::EditorLightType::Point => {
                                wf_verts.extend(wireframe::point_light_wireframe(
                                    light.position, light.range, lc,
                                ));
                            }
                            light_editor::EditorLightType::Spot => {
                                wf_verts.extend(wireframe::spot_light_wireframe(
                                    light.position, light.direction,
                                    light.range, light.spot_outer_angle, lc,
                                ));
                            }
                        }
                    }
                }
            }

            // Selection AABB + transform gizmo for selected objects.
            // Hidden while actively sculpting to avoid visual clutter.
            if let Some(editor_state::SelectedEntity::Object(eid)) = f_selected.filter(|_| !f_sculpting_active) {
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
                            wireframe::compute_node_tree_aabb(&obj.root_node, glam::Mat4::IDENTITY)
                        {
                            wf_verts.extend(wireframe::obb_wireframe(
                                lmin, lmax, obj.position, obj.rotation, obj.scale, color,
                            ));
                            let center = obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale);
                            gizmo_center = Some(center);
                        }
                    } else if let Some((child_node, child_world)) =
                        wireframe::find_child_node_and_transform(
                            eid, obj_id, &obj.root_node, root_world,
                        )
                    {
                        if let Some((lmin, lmax)) =
                            wireframe::compute_node_tree_aabb(child_node, glam::Mat4::IDENTITY)
                        {
                            let (child_scale, child_rot, child_pos) =
                                child_world.to_scale_rotation_translation();
                            wf_verts.extend(wireframe::obb_wireframe(
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
                        gizmo::GizmoMode::Translate => {
                            wf_verts.extend(wireframe::translate_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                        gizmo::GizmoMode::Rotate => {
                            wf_verts.extend(wireframe::rotate_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                        gizmo::GizmoMode::Scale => {
                            wf_verts.extend(wireframe::scale_gizmo_wireframe(
                                gc, gizmo_size, f_gizmo_axis, cam_pos,
                            ));
                        }
                    }
                }
            }

            // Ground grid overlay.
            if f_show_grid {
                wf_verts.extend(wireframe::ground_grid_wireframe(
                    cam_pos,
                    40.0,
                    1.0,
                    [0.3, 0.3, 0.3, 0.4],
                ));
            }

            // Brush preview sphere in Sculpt/Paint modes.
            if matches!(f_editor_mode, editor_state::EditorMode::Sculpt | editor_state::EditorMode::Paint) {
                let brush_pos = shared_state.lock().ok()
                    .and_then(|s| s.brush_preview_pos);
                if let Some(pos) = brush_pos {
                    let brush_color = match f_editor_mode {
                        editor_state::EditorMode::Sculpt => [0.0, 1.0, 1.0, 0.8],
                        editor_state::EditorMode::Paint  => [1.0, 0.8, 0.0, 0.8],
                        _ => [1.0, 1.0, 1.0, 0.8],
                    };
                    wf_verts.extend(wireframe::sphere_wireframe(
                        pos, f_brush_radius, brush_color,
                    ));
                }
            }

            engine.draw_wireframe(&wf_verts);
        }

        // i. Tell the compositor that new content is ready so it repaints.
        gpu_registrar.notify_frame_ready();

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

        // k. Screenshot readback and GPU pick are handled inside render_frame_offscreen,
        //    which reads shared_state.screenshot_requested and shared_state.pending_pick
        //    directly, writing results back to shared_state before returning.

        // l. Process GPU pick result — sets selected_entity, signals main thread.
        //    Suppressed while actively sculpting to prevent accidental selection changes.
        {
            let pick = shared_state.lock().ok()
                .and_then(|mut ss| ss.pick_result.take());
            if let Some(object_id) = pick {
                if !f_sculpting_active {
                    let picked_entity = if object_id > 0 {
                        Some(editor_state::SelectedEntity::Object(object_id as u64))
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
                            Some(editor_state::SelectedEntity::Object(eid)) => Some(eid as u32),
                            _ => None,
                        };
                        (es.editor_input.mouse_buttons[0], es.mode, sel_id)
                    })
                    .unwrap_or((false, editor_state::EditorMode::Default, None));

                // Only allow sculpt/paint on the currently selected object.
                let hit_on_selected = selected_obj_id == Some(hit.object_id);

                if left_down {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            editor_state::EditorMode::Sculpt if hit_on_selected => {
                                if es.sculpt.active_stroke.is_some() {
                                    es.sculpt.continue_stroke(hit.position);
                                } else {
                                    es.sculpt.begin_stroke(hit.position);
                                }
                                // Queue a real-time sculpt edit for this point.
                                let settings = es.sculpt.current_settings.clone();
                                es.pending_sculpt_edits.push(
                                    sculpt::SculptEditRequest {
                                        object_id: hit.object_id,
                                        world_position: hit.position,
                                        settings,
                                    },
                                );
                            }
                            editor_state::EditorMode::Sculpt => {
                                // Hit a non-selected object — ignore.
                            }
                            editor_state::EditorMode::Paint if hit_on_selected => {
                                if es.paint.active_stroke.is_some() {
                                    es.paint.continue_stroke(hit.position);
                                } else {
                                    es.paint.begin_stroke(hit.position);
                                }
                            }
                            editor_state::EditorMode::Paint => {
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
                    .unwrap_or((false, editor_state::EditorMode::Default));

                if !left_down && matches!(mode, editor_state::EditorMode::Sculpt | editor_state::EditorMode::Paint) {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            editor_state::EditorMode::Sculpt => {
                                es.sculpt.end_stroke();
                                // Finalize sculpt undo: push accumulated brick snapshots.
                                if let Some(acc) = es.sculpt_undo_accumulator.take() {
                                    if !acc.snapshots.is_empty() {
                                        es.undo.push(undo::UndoAction {
                                            kind: undo::UndoActionKind::SculptStroke {
                                                object_id: acc.object_id,
                                                brick_snapshots: acc.snapshots,
                                            },
                                            timestamp_ms: 0,
                                            description: "Sculpt stroke".into(),
                                        });
                                    }
                                }
                            }
                            editor_state::EditorMode::Paint => es.paint.end_stroke(),
                            _ => {}
                        }
                    }
                }
            }
        }

        // n. Periodically push the FPS value to the main thread (~4/sec).
        if last_fps_push.elapsed() >= std::time::Duration::from_millis(250) {
            last_fps_push = std::time::Instant::now();
            let fps_ms = dt as f64 * 1000.0;
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                    ui.fps.set(fps_ms);
                }
            });
        }

        // o. Frame pacing — sleep if we finished early so the engine thread
        //    doesn't spin-lock the editor_state mutex and starve the UI thread.
        let elapsed = frame_start.elapsed();
        if elapsed < target_frame_time {
            std::thread::sleep(target_frame_time - elapsed);
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    env_logger::init();
    log::info!("RKIField Editor — v2 render pipeline + rinch shell");

    // 1. Create shared state.
    let editor_state = Arc::new(Mutex::new(EditorState::new()));
    let shared_state = Arc::new(Mutex::new(SharedState::new(
        0, 0,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT,
    )));

    // 2. Create tokio runtime and start IPC server.
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let api = automation::EditorAutomationApi::new(
        Arc::clone(&shared_state),
        Arc::clone(&editor_state),
    );
    let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
    let (_ipc_handle, socket_path) = start_ipc_server(&rt, api);

    // 3. Create the RenderSurface — identifies where in the rinch layout
    //    the engine's offscreen texture will be composited.
    let surface_handle = create_render_surface();

    // 4. Clones for the engine thread.
    //    GpuTextureRegistrar is Send + Sync (wraps Arc<Mutex<>> fields only).
    let editor_state_for_thread = Arc::clone(&editor_state);
    let shared_state_for_thread = Arc::clone(&shared_state);
    let gpu_registrar = surface_handle.gpu_registrar();
    let socket_path_for_cleanup = socket_path.clone();

    // 5. Spawn engine thread.
    //    The thread will poll for gpu_handle() until rinch has created the
    //    wgpu device (happens inside run_with_window_props_and_menu).
    std::thread::spawn(move || {
        engine_thread(EngineThreadData {
            editor_state: editor_state_for_thread,
            shared_state: shared_state_for_thread,
            gpu_registrar,
        });
    });

    // 6. Build WindowProps for a borderless transparent window.
    //    Matches the previous `with_decorations(false).with_transparent(true)`.
    let sp = socket_path_for_cleanup.clone();
    let props = WindowProps {
        title: "RKIField Editor".into(),
        width: DISPLAY_WIDTH,
        height: DISPLAY_HEIGHT,
        borderless: true,
        resizable: true,
        transparent: true,
        menu_in_titlebar: true,
        on_close_requested: Some(std::sync::Arc::new(move || {
            cleanup_ipc(&sp);
            true // proceed with exit
        })),
        ..Default::default()
    };

    let theme = Some(ThemeProviderProps {
        dark_mode: true,
        primary_color: Some("blue".into()),
        ..Default::default()
    });

    // Context Arc clones for the component closure (main thread).
    let editor_state_ctx = Arc::clone(&editor_state);
    let shared_state_ctx = Arc::clone(&shared_state);

    // 7. Call rinch shell — BLOCKING until window close.
    //    create_context() and UiRevision::new() must be called on the main
    //    thread inside this closure (both use thread-local rinch state).
    rinch::shell::run_with_window_props_and_menu(
        move |_scope| {
            // Register contexts so components can call use_context().
            // Centralized slider signals — one batch sync Effect replaces 33 lock closures.
            // Must be created before editor_state_ctx is moved into create_context.
            let slider_signals = SliderSignals::new(&editor_state_ctx.lock().unwrap());
            create_context(editor_state_ctx);
            create_context(shared_state_ctx);
            // Surface handle context — editor_ui uses it to wire SurfaceEvent → editor input.
            create_context(surface_handle);
            // Per-property UI signals — replaces the single UiRevision counter.
            let ui_signals = UiSignals::new();
            create_context(ui_signals);
            create_context(slider_signals);
            // Legacy: UiRevision and FpsSignal kept during migration.
            let ui_revision = UiRevision::new();
            create_context(ui_revision);
            create_context(crate::editor_state::FpsSignal::new());

            // Build the editor UI tree.
            ui::editor_ui(_scope)
        },
        props,
        theme,
        None, // Menus are rendered inside the borderless titlebar by editor_ui
    );

    // 8. Cleanup IPC on normal exit (on_close_requested may have already done this).
    cleanup_ipc(&socket_path);

    Ok(())
}
