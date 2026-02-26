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
mod environment;
mod gizmo;
mod input;
mod light_editor;
mod overlay;
mod paint;
mod placement;
mod properties;
mod scene_io;
mod scene_tree;
mod sculpt;
mod ui;
mod undo;
mod wireframe;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use automation::SharedState;
use editor_state::{EditorState, UiRevision};
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

    // 4. Store demo scene in editor_state.
    if let Ok(mut es) = editor_state.lock() {
        es.v2_scene = Some(demo_scene);
        es.sync_v2_scene();

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
    let mut current_vp = engine.viewport_size();
    let mut prev_left_down = false;

    loop {
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

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
        {
            if let Ok(mut ss) = shared_state.lock() {
                pending_camera = ss.pending_camera.take();
                pending_debug = ss.pending_debug_mode.take();
            } else {
                pending_camera = None;
                pending_debug = None;
            }
        }

        // Wireframe data — extracted from editor_state while the lock is held,
        // used to build wireframe vertices after the lock is released.
        let mut wf_selected: Option<editor_state::SelectedEntity> = None;
        let mut wf_gizmo_mode = gizmo::GizmoMode::Translate;
        let mut wf_hovered_axis = gizmo::GizmoAxis::None;
        let mut wf_cam_pos = glam::Vec3::ZERO;
        let mut wf_show_grid = false;
        let mut wf_mode = editor_state::EditorMode::Default;
        let mut wf_brush_radius = 1.0f32;
        let mut gizmo_drag_ended = false;

        // c. Apply pending commands + per-frame updates to editor_state.
        {
            if let Ok(mut es) = editor_state.lock() {
                // Apply pending MCP camera teleport.
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

                // Camera fly/orbit update (WASD input wired in Step 3).
                es.update_camera(dt);

                // Sync editor camera → render camera.
                engine.sync_camera(&es.editor_camera);

                // Apply pending debug mode.
                if let Some(mode) = es.pending_debug_mode.take() {
                    es.debug_mode = mode;
                    engine.set_debug_mode(mode);
                }

                // d. Apply environment settings.
                engine.apply_environment(&mut es.environment);

                // e. Sync editor lights → render lights when dirty.
                if es.light_editor.is_dirty() {
                    engine.world_lights = es.light_editor.all_lights().iter().map(|el| {
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
                }

                // f. Recompute AABBs from current transforms each frame.
                if let Some(ref mut scene) = es.v2_scene {
                    for obj in &mut scene.objects {
                        obj.aabb = placement::compute_object_local_aabb(obj);
                    }
                }

                // g-gizmo. Gizmo mode switching + drag interaction.
                {
                    // Mode switching: G=Translate, R=Rotate, T/L=Scale.
                    if es.editor_input.keys_just_pressed.contains(&input::KeyCode::G)
                        && es.mode == editor_state::EditorMode::Default
                    {
                        es.gizmo.mode = gizmo::GizmoMode::Translate;
                    }
                    if es.editor_input.keys_just_pressed.contains(&input::KeyCode::R)
                        && es.mode == editor_state::EditorMode::Default
                    {
                        es.gizmo.mode = gizmo::GizmoMode::Rotate;
                    }
                    if es.editor_input.keys_just_pressed.contains(&input::KeyCode::L)
                        && es.mode == editor_state::EditorMode::Default
                    {
                        es.gizmo.mode = gizmo::GizmoMode::Scale;
                    }

                    let left_down = es.editor_input.mouse_buttons[0];
                    let right_down = es.editor_input.mouse_buttons[1];
                    let left_just_pressed = left_down && !prev_left_down;
                    let left_just_released = !left_down && prev_left_down;

                    // Hover detection: update hovered_axis each frame when not dragging.
                    if !es.gizmo.dragging {
                        if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                            let gc = es.v2_scene.as_ref().and_then(|scene| {
                                let obj = scene.objects.iter().find(|o| o.id as u64 == eid)?;
                                let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                    &obj.root_node, glam::Mat4::IDENTITY,
                                )?;
                                Some(obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale))
                            });
                            if let Some(gc) = gc {
                                let cam_dist = (gc - es.editor_camera.position).length();
                                let gizmo_size = cam_dist * 0.12;
                                let (ray_o, ray_d) = camera::screen_to_ray(
                                    &es.editor_camera,
                                    es.editor_input.mouse_pos.x,
                                    es.editor_input.mouse_pos.y,
                                    current_vp.0 as f32,
                                    current_vp.1 as f32,
                                );
                                es.gizmo.hovered_axis = gizmo::pick_gizmo_axis_for_mode(
                                    ray_o, ray_d, gc, gizmo_size, es.gizmo.mode,
                                );
                            } else {
                                es.gizmo.hovered_axis = gizmo::GizmoAxis::None;
                            }
                        } else {
                            es.gizmo.hovered_axis = gizmo::GizmoAxis::None;
                        }
                    }

                    // Drag start: left-click in Default mode with an object selected.
                    if left_just_pressed
                        && !right_down
                        && es.mode == editor_state::EditorMode::Default
                        && !es.gizmo.dragging
                    {
                        if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                            // Compute gizmo center from selected object's AABB.
                            let gc = es.v2_scene.as_ref().and_then(|scene| {
                                let obj = scene.objects.iter().find(|o| o.id as u64 == eid)?;
                                let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                    &obj.root_node, glam::Mat4::IDENTITY,
                                )?;
                                let center = obj.position
                                    + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale);
                                Some((center, obj.position, obj.rotation, obj.scale))
                            });

                            if let Some((gc, obj_pos, obj_rot, obj_scale)) = gc {
                                let cam_dist = (gc - es.editor_camera.position).length();
                                let gizmo_size = cam_dist * 0.12;
                                let (ray_o, ray_d) = camera::screen_to_ray(
                                    &es.editor_camera,
                                    es.editor_input.mouse_pos.x,
                                    es.editor_input.mouse_pos.y,
                                    current_vp.0 as f32,
                                    current_vp.1 as f32,
                                );
                                let axis = gizmo::pick_gizmo_axis_for_mode(
                                    ray_o, ray_d, gc, gizmo_size, es.gizmo.mode,
                                );
                                if axis != gizmo::GizmoAxis::None {
                                    let start_point = match es.gizmo.mode {
                                        gizmo::GizmoMode::Translate | gizmo::GizmoMode::Scale => {
                                            if axis == gizmo::GizmoAxis::View {
                                                let vn = (es.editor_camera.position - gc).normalize();
                                                gizmo::project_to_plane(ray_o, ray_d, gc, vn)
                                                    .unwrap_or(gc)
                                            } else {
                                                let t = gizmo::ray_axis_closest_point(
                                                    ray_o, ray_d, gc, axis.direction(),
                                                );
                                                gc + axis.direction() * t
                                            }
                                        }
                                        gizmo::GizmoMode::Rotate => {
                                            gizmo::project_to_plane(
                                                ray_o, ray_d, gc, axis.plane_normal(),
                                            )
                                            .unwrap_or(gc)
                                        }
                                    };
                                    let view_normal =
                                        (es.editor_camera.position - gc).normalize();
                                    es.gizmo.begin_drag(
                                        axis,
                                        start_point,
                                        obj_pos,
                                        obj_rot,
                                        obj_scale,
                                        view_normal,
                                    );
                                    es.gizmo.pivot = gc;
                                }
                            }
                        }
                    }

                    // Drag continue: update object transform from ray projection.
                    if es.gizmo.dragging && !left_just_released {
                        if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity
                        {
                            let (ray_o, ray_d) = camera::screen_to_ray(
                                &es.editor_camera,
                                es.editor_input.mouse_pos.x,
                                es.editor_input.mouse_pos.y,
                                current_vp.0 as f32,
                                current_vp.1 as f32,
                            );
                            let gizmo_mode = es.gizmo.mode;
                            let pivot = es.gizmo.pivot;
                            let initial_pos = es.gizmo.initial_position;
                            let initial_rot = es.gizmo.initial_rotation;
                            let initial_scale = es.gizmo.initial_scale;

                            let (new_pos, new_rot, new_scale) = match gizmo_mode {
                                gizmo::GizmoMode::Translate => {
                                    let delta =
                                        gizmo::compute_translate_delta(&es.gizmo, ray_o, ray_d);
                                    (
                                        initial_pos + delta,
                                        initial_rot,
                                        initial_scale,
                                    )
                                }
                                gizmo::GizmoMode::Rotate => {
                                    let rot_delta = gizmo::compute_rotate_delta(
                                        &es.gizmo, ray_o, ray_d, pivot,
                                    );
                                    let new_rot = rot_delta * initial_rot;
                                    let offset = initial_pos - pivot;
                                    let new_pos = pivot + rot_delta * offset;
                                    (new_pos, new_rot, initial_scale)
                                }
                                gizmo::GizmoMode::Scale => {
                                    let scale_delta =
                                        gizmo::compute_scale_delta(&es.gizmo, ray_o, ray_d);
                                    // Per-axis: multiply initial scale by delta per-component.
                                    (initial_pos, initial_rot, initial_scale * scale_delta)
                                }
                            };

                            if let Some(ref mut scene) = es.v2_scene {
                                if let Some(obj) =
                                    scene.objects.iter_mut().find(|o| o.id as u64 == eid)
                                {
                                    obj.position = new_pos;
                                    obj.rotation = new_rot;
                                    obj.scale = new_scale;
                                }
                            }
                        }
                    }

                    // Drag end: push undo action and end the drag.
                    if left_just_released && es.gizmo.dragging {
                        if let Some(editor_state::SelectedEntity::Object(eid)) = es.selected_entity
                        {
                            // Read current (final) transform from scene.
                            let final_transform = es.v2_scene.as_ref().and_then(|scene| {
                                let obj = scene.objects.iter().find(|o| o.id as u64 == eid)?;
                                Some((obj.position, obj.rotation, obj.scale))
                            });

                            if let Some((new_pos, new_rot, new_scale)) = final_transform {
                                let desc = match es.gizmo.mode {
                                    gizmo::GizmoMode::Translate => "Move object",
                                    gizmo::GizmoMode::Rotate => "Rotate object",
                                    gizmo::GizmoMode::Scale => "Scale object",
                                };
                                let old_pos = es.gizmo.initial_position;
                                let old_rot = es.gizmo.initial_rotation;
                                let old_scale = es.gizmo.initial_scale;
                                es.undo.push(undo::UndoAction {
                                    kind: undo::UndoActionKind::Transform {
                                        entity_id: eid,
                                        old_pos,
                                        old_rot,
                                        old_scale,
                                        new_pos,
                                        new_rot,
                                        new_scale,
                                    },
                                    timestamp_ms: 0,
                                    description: desc.to_string(),
                                });
                            }
                        }
                        es.gizmo.end_drag();
                        gizmo_drag_ended = true;
                    }

                    prev_left_down = left_down;
                }

                // Extract wireframe data while we hold the lock.
                wf_selected = es.selected_entity;
                wf_gizmo_mode = es.gizmo.mode;
                wf_hovered_axis = if es.gizmo.dragging {
                    es.gizmo.active_axis
                } else {
                    es.gizmo.hovered_axis
                };
                wf_cam_pos = es.editor_camera.position;
                wf_show_grid = es.show_grid;
                wf_mode = es.mode;
                wf_brush_radius = match es.mode {
                    editor_state::EditorMode::Sculpt => es.sculpt.current_settings.radius,
                    editor_state::EditorMode::Paint => es.paint.current_settings.radius,
                    _ => 1.0,
                };

                // Reset per-frame input deltas.
                es.reset_frame_deltas();
            }
        }

        // Notify UI of gizmo transform changes (after editor_state lock released).
        if gizmo_drag_ended {
            if let Ok(mut ss) = shared_state.lock() {
                ss.ui_revision_needed = true;
            }
        }

        // g-revox. Process pending re-voxelize request (must happen before scene clone).
        {
            let revox_id = editor_state.lock().ok()
                .and_then(|mut es| es.pending_revoxelize.take());
            if let Some(obj_id) = revox_id {
                if let Ok(mut es) = editor_state.lock() {
                    if let Some(ref mut scene) = es.v2_scene {
                        engine.process_revoxelize(scene, obj_id);
                    }
                    es.sync_v2_scene();
                }
            }
        }

        // g. Clone v2_scene for render — character animation mutates bone
        //    transforms on the clone, discarded after render.
        let mut render_scene = editor_state.lock()
            .ok()
            .and_then(|es| es.v2_scene.clone())
            .unwrap_or_else(|| rkf_core::Scene::new("empty"));

        // h. Advance character animation on the render clone.
        engine.advance_character(&mut render_scene);

        // i. Render frame to offscreen texture.
        engine.render_frame_offscreen(&render_scene);

        // i-wf. Build and draw wireframe overlays (selection, gizmos, grid).
        {
            let mut wf_verts: Vec<wireframe::LineVertex> = Vec::new();

            // Light gizmo for selected light.
            if let Some(editor_state::SelectedEntity::Light(lid)) = wf_selected {
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
            if let Some(editor_state::SelectedEntity::Object(eid)) = wf_selected {
                let color = [0.3, 0.7, 1.0, 1.0]; // Light blue
                let mut gizmo_center: Option<glam::Vec3> = None;

                for obj in &render_scene.objects {
                    let obj_id = obj.id as u64;
                    let root_world = glam::Mat4::from_scale_rotation_translation(
                        obj.scale,
                        obj.rotation,
                        obj.position,
                    );

                    if eid == obj_id {
                        // Root object selected: OBB follows object rotation.
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
                        // Child node selected: compute local AABB, draw as OBB
                        // using the accumulated parent transform.
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
                    let cam_dist = (gc - wf_cam_pos).length();
                    let gizmo_size = cam_dist * 0.12;
                    match wf_gizmo_mode {
                        gizmo::GizmoMode::Translate => {
                            wf_verts.extend(wireframe::translate_gizmo_wireframe(
                                gc, gizmo_size, wf_hovered_axis, wf_cam_pos,
                            ));
                        }
                        gizmo::GizmoMode::Rotate => {
                            wf_verts.extend(wireframe::rotate_gizmo_wireframe(
                                gc, gizmo_size, wf_hovered_axis, wf_cam_pos,
                            ));
                        }
                        gizmo::GizmoMode::Scale => {
                            wf_verts.extend(wireframe::scale_gizmo_wireframe(
                                gc, gizmo_size, wf_hovered_axis, wf_cam_pos,
                            ));
                        }
                    }
                }
            }

            // Ground grid overlay.
            if wf_show_grid {
                wf_verts.extend(wireframe::ground_grid_wireframe(
                    wf_cam_pos,
                    40.0,
                    1.0,
                    [0.3, 0.3, 0.3, 0.4],
                ));
            }

            // Brush preview sphere in Sculpt/Paint modes.
            if matches!(wf_mode, editor_state::EditorMode::Sculpt | editor_state::EditorMode::Paint) {
                let brush_pos = shared_state.lock().ok()
                    .and_then(|s| s.brush_preview_pos);
                if let Some(pos) = brush_pos {
                    let brush_color = match wf_mode {
                        editor_state::EditorMode::Sculpt => [0.0, 1.0, 1.0, 0.8], // Cyan
                        editor_state::EditorMode::Paint  => [1.0, 0.8, 0.0, 0.8], // Yellow
                        _ => [1.0, 1.0, 1.0, 0.8],
                    };
                    wf_verts.extend(wireframe::sphere_wireframe(
                        pos, wf_brush_radius, brush_color,
                    ));
                }
            }

            engine.draw_wireframe(&wf_verts);
        }

        // i2. Tell the compositor that new content is ready so it repaints.
        gpu_registrar.notify_frame_ready();

        // j. Update shared_state for MCP observation.
        //    Read camera snapshot from editor_state first, then write to shared_state.
        //    Never hold both locks simultaneously.
        let cam_snapshot = editor_state.lock().ok().map(|es| {
            (es.editor_camera.position, es.editor_camera.fly_yaw,
             es.editor_camera.fly_pitch, es.editor_camera.fov_y.to_degrees())
        });
        if let Some((pos, yaw, pitch, fov)) = cam_snapshot {
            if let Ok(mut ss) = shared_state.lock() {
                ss.camera_position = pos;
                ss.camera_yaw = yaw;
                ss.camera_pitch = pitch;
                ss.camera_fov = fov;
                ss.frame_time_ms = dt as f64 * 1000.0;
                ss.frame_width = current_vp.0;
                ss.frame_height = current_vp.1;
            }
        }

        // k. Screenshot readback and GPU pick are handled inside render_frame_offscreen,
        //    which reads shared_state.screenshot_requested and shared_state.pending_pick
        //    directly, writing results back to shared_state before returning.

        // l. Process GPU pick result (fulfilled each frame by render_frame_offscreen).
        //    Sets selected_entity immediately; UI notification (rev.bump()) happens
        //    on the main thread when it detects pick_completed in the event handler.
        {
            let pick = shared_state.lock().ok()
                .and_then(|mut ss| ss.pick_result.take());
            if let Some(object_id) = pick {
                if let Ok(mut es) = editor_state.lock() {
                    if object_id > 0 {
                        es.selected_entity = Some(
                            editor_state::SelectedEntity::Object(object_id as u64),
                        );
                    } else {
                        es.selected_entity = None;
                    }
                }
                // Signal the main thread to bump UiRevision.
                if let Ok(mut ss) = shared_state.lock() {
                    ss.pick_completed = true;
                }
            }
        }

        // l2. Process GPU brush hit result — drive sculpt/paint stroke lifecycle.
        {
            let brush_hit = shared_state.lock().ok()
                .and_then(|mut ss| ss.brush_hit_result.take());
            if let Some(hit) = brush_hit {
                // Read left-mouse state and mode from editor_state (brief lock).
                let (left_down, mode) = editor_state.lock().ok()
                    .map(|es| (es.editor_input.mouse_buttons[0], es.mode))
                    .unwrap_or((false, editor_state::EditorMode::Default));

                if left_down {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            editor_state::EditorMode::Sculpt => {
                                if es.sculpt.active_stroke.is_some() {
                                    es.sculpt.continue_stroke(hit.position);
                                } else {
                                    es.sculpt.begin_stroke(hit.position);
                                }
                            }
                            editor_state::EditorMode::Paint => {
                                if es.paint.active_stroke.is_some() {
                                    es.paint.continue_stroke(hit.position);
                                } else {
                                    es.paint.begin_stroke(hit.position);
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Update brush preview position for wireframe sphere.
                if let Ok(mut ss) = shared_state.lock() {
                    ss.brush_preview_pos = Some(hit.position);
                }
            } else {
                // No brush hit this frame — end active strokes if mouse released.
                let (left_down, mode) = editor_state.lock().ok()
                    .map(|es| (es.editor_input.mouse_buttons[0], es.mode))
                    .unwrap_or((false, editor_state::EditorMode::Default));

                if !left_down && matches!(mode, editor_state::EditorMode::Sculpt | editor_state::EditorMode::Paint) {
                    if let Ok(mut es) = editor_state.lock() {
                        match mode {
                            editor_state::EditorMode::Sculpt => es.sculpt.end_stroke(),
                            editor_state::EditorMode::Paint => es.paint.end_stroke(),
                            _ => {}
                        }
                    }
                }

                // Clear brush preview when hovering over sky (no hit).
                if matches!(mode, editor_state::EditorMode::Sculpt | editor_state::EditorMode::Paint) {
                    // Only clear if we're actively in brush mode but got no hit.
                    // Keep the last position if there's no pending request.
                    let had_pending = shared_state.lock().ok()
                        .map(|s| s.pending_brush_hit.is_some())
                        .unwrap_or(false);
                    if !had_pending {
                        // No pending request means no mouse movement this frame — keep preview.
                    }
                }
            }
        }

        // m. Cap frame rate at ~60 fps when GPU is faster.
        let elapsed = last_frame.elapsed();
        let frame_budget = std::time::Duration::from_millis(16);
        if elapsed < frame_budget {
            std::thread::sleep(frame_budget - elapsed);
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
            create_context(editor_state_ctx);
            create_context(shared_state_ctx);
            // Surface handle context — editor_ui uses it to wire SurfaceEvent → editor input.
            create_context(surface_handle);
            // UiRevision wraps a Signal — must be created on the main thread.
            let ui_revision = UiRevision::new();
            create_context(ui_revision);

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
