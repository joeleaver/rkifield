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
        {
            if let Ok(mut ss) = shared_state.lock() {
                pending_camera = ss.pending_camera.take();
                pending_debug = ss.pending_debug_mode.take();
            } else {
                pending_camera = None;
                pending_debug = None;
            }
        }

        // c. Single brief lock: read all data needed for this frame, then release.
        //    We inline what was formerly `take_engine_snapshot()` to avoid the
        //    intermediate `EngineSnapshot` struct.
        let (
            camera, f_debug_mode, f_revoxelize, f_environment, f_lights,
            mut scene_clone, f_selected, f_gizmo_mode, f_gizmo_axis,
            f_show_grid, f_editor_mode, f_brush_radius,
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

            // Reset per-frame deltas last.
            es.reset_frame_deltas();

            (cam, debug_mode, revoxelize, environment, lights,
             scene, sel, gm, gizmo_axis, grid, emode, brush_radius)
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
            if let Some(editor_state::SelectedEntity::Object(eid)) = f_selected {
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
        {
            let pick = shared_state.lock().ok()
                .and_then(|mut ss| ss.pick_result.take());
            if let Some(object_id) = pick {
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

        // m. Process GPU brush hit result — drive sculpt/paint stroke lifecycle.
        {
            let brush_hit = shared_state.lock().ok()
                .and_then(|mut ss| ss.brush_hit_result.take());
            if let Some(hit) = brush_hit {
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
                            editor_state::EditorMode::Sculpt => es.sculpt.end_stroke(),
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
