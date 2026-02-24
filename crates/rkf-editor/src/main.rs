//! RKIField editor — v2 object-centric render pipeline with rinch UI overlay.
//!
//! Provides the full v2 compute-shader rendering pipeline, rinch-based UI
//! panels (scene tree, properties, toolbar, status bar), orbit/fly camera
//! controls, and IPC server for MCP agent tooling.

#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod animation_preview;
mod automation;
mod camera;
mod debug_viz;
mod editor_state;
mod engine;
mod engine_viewport;
mod environment;
mod event_bridge;
mod gizmo;
mod input;
mod light_editor;
mod overlay;
mod overlay_blit;
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
use std::time::Instant;

use rinch::prelude::*;
use rinch_platform::PlatformEvent;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode as WinitKeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use automation::SharedState;
use editor_state::{EditorState, UiRevision};
use engine::EditorEngine;
use engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};
use overlay_blit::OverlayBlit;

// ---------------------------------------------------------------------------
// Winit key → editor key translation
// ---------------------------------------------------------------------------

fn translate_key(key: WinitKeyCode) -> Option<input::KeyCode> {
    use input::KeyCode as Ek;
    match key {
        WinitKeyCode::KeyW => Some(Ek::W),
        WinitKeyCode::KeyA => Some(Ek::A),
        WinitKeyCode::KeyS => Some(Ek::S),
        WinitKeyCode::KeyD => Some(Ek::D),
        WinitKeyCode::KeyQ => Some(Ek::Q),
        WinitKeyCode::KeyE => Some(Ek::E),
        WinitKeyCode::KeyG => Some(Ek::G),
        WinitKeyCode::KeyR => Some(Ek::R),
        WinitKeyCode::KeyT => Some(Ek::T),
        WinitKeyCode::KeyX => Some(Ek::X),
        WinitKeyCode::KeyY => Some(Ek::Y),
        WinitKeyCode::KeyZ => Some(Ek::Z),
        WinitKeyCode::KeyF => Some(Ek::F),
        WinitKeyCode::Delete => Some(Ek::Delete),
        WinitKeyCode::Escape => Some(Ek::Escape),
        WinitKeyCode::Space => Some(Ek::Space),
        WinitKeyCode::Tab => Some(Ek::Tab),
        WinitKeyCode::Enter => Some(Ek::Return),
        WinitKeyCode::ShiftLeft => Some(Ek::ShiftLeft),
        WinitKeyCode::F5 => Some(Ek::F5),
        WinitKeyCode::F12 => Some(Ek::F12),
        WinitKeyCode::Digit1 => Some(Ek::Num1),
        WinitKeyCode::Digit2 => Some(Ek::Num2),
        WinitKeyCode::Digit3 => Some(Ek::Num3),
        _ => None,
    }
}

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

    let mut registry = rkf_mcp::registry::ToolRegistry::new();
    rkf_mcp::tools::observation::register_observation_tools(&mut registry);
    let registry = Arc::new(registry);

    let handle = rt.spawn(async move {
        if let Err(e) = rkf_mcp::ipc::run_server(config, registry, api).await {
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
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    engine: Option<EditorEngine>,
    editor_state: Arc<Mutex<EditorState>>,
    shared_state: Arc<Mutex<SharedState>>,
    last_frame: Instant,
    _ipc_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: Option<String>,
    rt: tokio::runtime::Runtime,

    // Rinch UI
    rinch_ctx: Option<RinchContext>,
    overlay_renderer: Option<RinchOverlayRenderer>,
    overlay_blit: Option<OverlayBlit>,
    wireframe_pass: Option<wireframe::WireframePass>,

    // Input state for event routing
    mouse_phys: (f32, f32),
    modifiers: winit::keyboard::ModifiersState,
    pending_events: Vec<PlatformEvent>,

    // Shared reactive revision — bumped on any editor state mutation
    // to trigger rinch component re-renders.
    ui_revision: Option<UiRevision>,
}

impl App {
    fn new() -> Self {
        let editor_state = Arc::new(Mutex::new(EditorState::new()));
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )));
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        Self {
            window: None,
            engine: None,
            editor_state,
            shared_state,
            last_frame: Instant::now(),
            _ipc_handle: None,
            socket_path: None,
            rt,

            rinch_ctx: None,
            overlay_renderer: None,
            overlay_blit: None,
            wireframe_pass: None,

            mouse_phys: (0.0, 0.0),
            modifiers: winit::keyboard::ModifiersState::empty(),
            pending_events: Vec::new(),
            ui_revision: None,
        }
    }

    fn scale_factor(&self) -> f64 {
        self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0)
    }

    fn logical_mouse(&self) -> (f32, f32) {
        let sf = self.scale_factor() as f32;
        (self.mouse_phys.0 / sf, self.mouse_phys.1 / sf)
    }

    /// Check if rinch wants mouse input at the current position.
    fn ui_wants_mouse(&self) -> bool {
        let (lx, ly) = self.logical_mouse();
        self.rinch_ctx
            .as_ref()
            .is_some_and(|ctx| ctx.wants_mouse(lx, ly))
    }

    /// Check if rinch wants keyboard input (text field focused).
    fn ui_wants_keyboard(&self) -> bool {
        self.rinch_ctx
            .as_ref()
            .is_some_and(|ctx| ctx.wants_keyboard())
    }

    /// Save the current scene to .rkscene.
    fn save_scene(&mut self) {
        let engine = match self.engine.as_ref() {
            Some(e) => e,
            None => return,
        };

        let path = {
            let es = self.editor_state.lock().unwrap();
            es.current_scene_path.clone()
        };

        let path = if let Some(p) = path {
            p
        } else {
            let dialog = rfd::FileDialog::new()
                .set_title("Save Scene")
                .add_filter("RKIField Scene", &["rkscene"])
                .set_file_name("scene.rkscene");
            match dialog.save_file() {
                Some(p) => p.to_string_lossy().to_string(),
                None => return,
            }
        };

        match scene_io::save_v2_scene(&engine.scene, &path) {
            Ok(()) => {
                log::info!("Scene saved to: {path}");
                if let Ok(mut es) = self.editor_state.lock() {
                    es.current_scene_path = Some(path.clone());
                    es.unsaved_changes.mark_saved();
                    es.recent_files.add(&path, &engine.scene.name, 0);
                }
                if let Some(window) = &self.window {
                    let name = std::path::Path::new(&path)
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "Untitled".to_string());
                    window.set_title(&format!("RKIField Editor — {name}"));
                }
            }
            Err(e) => log::error!("Failed to save scene: {e}"),
        }
    }

    /// Open a .rkscene file.
    fn open_scene(&mut self) {
        let dialog = rfd::FileDialog::new()
            .set_title("Open Scene")
            .add_filter("RKIField Scene", &["rkscene"]);
        let path = match dialog.pick_file() {
            Some(p) => p.to_string_lossy().to_string(),
            None => return,
        };

        let sf = match scene_io::load_v2_scene(&path) {
            Ok(sf) => sf,
            Err(e) => {
                log::error!("Failed to load scene: {e}");
                return;
            }
        };

        let new_scene = scene_io::reconstruct_v2_scene(&sf);

        if let Some(engine) = self.engine.as_mut() {
            engine.replace_scene(new_scene.clone());
        }

        if let Ok(mut es) = self.editor_state.lock() {
            es.v2_scene = Some(new_scene);
            es.sync_v2_scene();
            es.current_scene_path = Some(path.clone());
            es.unsaved_changes.mark_saved();
            es.recent_files.add(&path, &sf.name, 0);
        }

        if let Some(rev) = &self.ui_revision {
            rev.bump();
        }

        if let Some(window) = &self.window {
            let name = std::path::Path::new(&path)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "Untitled".to_string());
            window.set_title(&format!("RKIField Editor — {name}"));
        }

        log::info!("Scene loaded from: {path}");
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = winit::window::WindowAttributes::default()
            .with_title("RKIField Editor")
            .with_inner_size(winit::dpi::PhysicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        let size = window.inner_size();

        // Initialize engine (wgpu, all render passes, demo scene).
        let engine_inst = EditorEngine::new(window.clone(), Arc::clone(&self.shared_state));

        // Sync engine's demo scene into editor state.
        if let Ok(mut es) = self.editor_state.lock() {
            es.v2_scene = Some(engine_inst.scene.clone());
            es.sync_v2_scene();
        }

        // Set up rinch context sharing — components use use_context to access these.
        create_context(Arc::clone(&self.editor_state));
        create_context(Arc::clone(&self.shared_state));
        let ui_revision = UiRevision::new();
        create_context(ui_revision);
        self.ui_revision = Some(ui_revision);

        // Initialize rinch UI context.
        let rinch_ctx = RinchContext::new(
            RinchContextConfig {
                width: size.width,
                height: size.height,
                scale_factor: window.scale_factor(),
                theme: Some(ThemeProviderProps {
                    dark_mode: true,
                    primary_color: Some("blue".into()),
                    ..Default::default()
                }),
            },
            ui::editor_ui,
        );

        // Initialize overlay renderer (Vello → transparent texture).
        let overlay_renderer = RinchOverlayRenderer::new(
            engine_inst.device(),
            size.width,
            size.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // Initialize overlay blit (premultiplied alpha composite onto swapchain).
        let overlay_blit_pass = OverlayBlit::new(
            engine_inst.device(),
            engine_inst.surface_format(),
        );

        // Initialize wireframe pass for selection highlighting.
        let wireframe_pass = wireframe::WireframePass::new(
            engine_inst.device(),
            engine_inst.surface_format(),
        );

        self.engine = Some(engine_inst);
        self.rinch_ctx = Some(rinch_ctx);
        self.overlay_renderer = Some(overlay_renderer);
        self.overlay_blit = Some(overlay_blit_pass);
        self.wireframe_pass = Some(wireframe_pass);

        // Start IPC server for MCP agent tooling.
        let api = automation::EditorAutomationApi::new(
            Arc::clone(&self.shared_state),
            Arc::clone(&self.editor_state),
        );
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
        let (handle, path) = start_ipc_server(&self.rt, api);
        self._ipc_handle = Some(handle);
        self.socket_path = Some(path);

        log::info!("RKIField Editor initialized — v2 render pipeline + rinch UI active");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        // Raw mouse motion for camera rotation (fires even when cursor is at edge).
        if let DeviceEvent::MouseMotion { delta } = event {
            // Only process camera delta when UI doesn't want the mouse.
            if !self.ui_wants_mouse() {
                if let Ok(mut es) = self.editor_state.lock() {
                    // Only accumulate delta when right mouse is held (camera rotation).
                    if es.editor_input.is_mouse_button_down(1) {
                        es.editor_input.mouse_delta.x += delta.0 as f32;
                        es.editor_input.mouse_delta.y += delta.1 as f32;
                    }
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Always collect platform events for rinch (hover effects, etc).
        let platform_events = event_bridge::translate_window_event(&event, self.modifiers);
        self.pending_events.extend(platform_events);

        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref path) = self.socket_path {
                    cleanup_ipc(path);
                }
                event_loop.exit();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_phys = (position.x as f32, position.y as f32);

                // Always update editor input mouse pos (for viewport bounds checking).
                if let Ok(mut es) = self.editor_state.lock() {
                    es.editor_input.mouse_pos =
                        glam::Vec2::new(position.x as f32, position.y as f32);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                // Update rinch pending events with actual mouse position.
                // (translate_window_event uses 0,0 — fix up the last event)
                if let Some(last) = self.pending_events.last_mut() {
                    let (px, py) = self.mouse_phys;
                    match last {
                        PlatformEvent::MouseDown { x, y, .. }
                        | PlatformEvent::MouseUp { x, y, .. } => {
                            *x = px;
                            *y = py;
                        }
                        _ => {}
                    }
                }

                let wants_ui = self.ui_wants_mouse();

                // Only route to engine if UI doesn't want the click.
                if !wants_ui {
                    let mut should_bump_revision = false;

                    if let Ok(mut es) = self.editor_state.lock() {
                        let idx = match button {
                            MouseButton::Left => 0,
                            MouseButton::Right => 1,
                            MouseButton::Middle => 2,
                            _ => return,
                        };
                        es.editor_input.mouse_buttons[idx] = state == ElementState::Pressed;

                        // Left-click release in viewport → ray-pick to select object.
                        if idx == 0 && state == ElementState::Released {
                            let viewport = self.rinch_ctx.as_ref().and_then(|ctx| {
                                ctx.viewport_rect("main").map(|r| {
                                    let sf = self.scale_factor() as f32;
                                    (r.x * sf, r.y * sf, r.width * sf, r.height * sf)
                                })
                            });
                            if let Some((vp_x, vp_y, vp_w, vp_h)) = viewport {
                                let (mx, my) = self.mouse_phys;
                                let px = mx - vp_x;
                                let py = my - vp_y;
                                if px >= 0.0 && py >= 0.0 && px < vp_w && py < vp_h {
                                    if let Some(entity_id) =
                                        es.pick_object(px, py, vp_w, vp_h)
                                    {
                                        es.scene_tree.select_node(entity_id);
                                    } else {
                                        es.scene_tree.clear_selection();
                                    }
                                    should_bump_revision = true;
                                }
                            }
                        }

                        // Hide cursor when right-mouse is held (fly/orbit rotation).
                        if idx == 1 {
                            if let Some(window) = &self.window {
                                let _ = window.set_cursor_visible(state != ElementState::Pressed);
                            }
                        }
                    }
                    // Lock released — now safe to bump revision (triggers reactive
                    // Effects that re-lock editor_state for reading).
                    if should_bump_revision {
                        if let Some(rev) = &self.ui_revision {
                            rev.bump();
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Fix up mouse position in pending events.
                if let Some(last) = self.pending_events.last_mut() {
                    let (px, py) = self.mouse_phys;
                    if let PlatformEvent::MouseWheel { x, y, .. } = last {
                        *x = px;
                        *y = py;
                    }
                }

                let wants_ui = self.ui_wants_mouse();

                // Only route scroll to engine if UI doesn't want it.
                if !wants_ui {
                    if let Ok(mut es) = self.editor_state.lock() {
                        let scroll = match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                            winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                        };
                        es.editor_input.scroll_delta += scroll;
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    // Escape always exits.
                    if key == WinitKeyCode::Escape && event.state == ElementState::Pressed {
                        if let Some(ref path) = self.socket_path {
                            cleanup_ipc(path);
                        }
                        event_loop.exit();
                        return;
                    }

                    // Ctrl+S: Save scene, Ctrl+O: Open scene.
                    if event.state == ElementState::Pressed && self.modifiers.control_key() {
                        match key {
                            WinitKeyCode::KeyS => {
                                self.save_scene();
                                return;
                            }
                            WinitKeyCode::KeyO => {
                                self.open_scene();
                                return;
                            }
                            _ => {}
                        }
                    }

                    let wants_kb = self.ui_wants_keyboard();

                    // Only route to engine when UI doesn't want keyboard.
                    if !wants_kb {
                        if let Some(ek) = translate_key(key) {
                            if let Ok(mut es) = self.editor_state.lock() {
                                match event.state {
                                    ElementState::Pressed => {
                                        es.editor_input.keys_pressed.insert(ek);
                                    }
                                    ElementState::Released => {
                                        es.editor_input.keys_pressed.remove(&ek);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                if let Ok(mut es) = self.editor_state.lock() {
                    let m = modifiers.state();
                    es.editor_input.modifiers.shift = m.shift_key();
                    es.editor_input.modifiers.ctrl = m.control_key();
                    es.editor_input.modifiers.alt = m.alt_key();
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size.width, size.height);
                }
                if let Some(ctx) = self.rinch_ctx.as_mut() {
                    ctx.resize(size.width, size.height);
                }
                if let Some(overlay) = self.overlay_renderer.as_mut() {
                    if let Some(engine) = self.engine.as_ref() {
                        overlay.resize(engine.device(), size.width, size.height);
                    }
                }
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(ctx) = self.rinch_ctx.as_mut() {
                    ctx.set_scale_factor(scale_factor);
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // 1. Update rinch UI with collected platform events.
                let events: Vec<_> = self.pending_events.drain(..).collect();
                if let Some(ctx) = self.rinch_ctx.as_mut() {
                    ctx.update(&events);
                }

                // 2. Query viewport rect from rinch layout (logical pixels → physical).
                let viewport = self.rinch_ctx.as_ref().and_then(|ctx| {
                    ctx.viewport_rect("main").map(|rect| {
                        let sf = self.scale_factor() as f32;
                        (rect.x * sf, rect.y * sf, rect.width * sf, rect.height * sf)
                    })
                });

                // 3. Resize render resolution to match viewport (if changed).
                if let Some(vp) = viewport {
                    let vp_w = (vp.2 as u32).max(64);
                    let vp_h = (vp.3 as u32).max(64);
                    if let Some(engine) = self.engine.as_mut() {
                        engine.resize_render(vp_w, vp_h);
                    }
                }

                // 4. Lock editor state, update camera from input.
                if let Ok(mut es) = self.editor_state.lock() {
                    // Apply pending MCP commands.
                    if let Ok(mut ss) = self.shared_state.lock() {
                        if let Some(cam) = ss.pending_camera.take() {
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
                        if let Some(mode) = ss.pending_debug_mode.take() {
                            es.pending_debug_mode = Some(mode);
                        }
                    }

                    // Update camera from input state.
                    es.update_camera(dt);

                    // Sync editor camera → render camera + apply pending debug mode.
                    if let Some(engine) = self.engine.as_mut() {
                        engine.sync_camera(&es.editor_camera);

                        if let Some(mode) = es.pending_debug_mode.take() {
                            engine.set_debug_mode(mode);
                        }
                    }

                    // Reset per-frame deltas.
                    es.reset_frame_deltas();
                }

                // 5. Build wireframe vertices for selected object OBBs.
                //    Read from engine.scene (live animated transforms) rather than
                //    es.v2_scene (static snapshot) so wireframes track animation
                //    and future in-editor transforms.
                let wireframe_verts = {
                    let mut verts = Vec::new();
                    if let (Some(engine), Ok(es)) =
                        (self.engine.as_ref(), self.editor_state.lock())
                    {
                        let selected = es.scene_tree.selected_entities();
                        let color = [0.3, 0.7, 1.0, 1.0]; // Light blue
                        let origin = rkf_core::WorldPosition::default();
                        for &eid in &selected {
                            for obj in &engine.scene.root_objects {
                                let obj_id = obj.id as u64;
                                let world_pos = obj.world_position.relative_to(&origin);
                                let root_world = glam::Mat4::from_scale_rotation_translation(
                                    glam::Vec3::splat(obj.scale),
                                    obj.rotation,
                                    world_pos,
                                );

                                if eid == obj_id {
                                    // Root object selected: full hierarchy AABB.
                                    if let Some((amin, amax)) = wireframe::compute_node_tree_aabb(
                                        &obj.root_node, root_world,
                                    ) {
                                        verts.extend(wireframe::aabb_wireframe(
                                            amin, amax, color,
                                        ));
                                    }
                                } else if let Some((child_node, child_world)) =
                                    wireframe::find_child_node_and_transform(
                                        eid, obj_id, &obj.root_node, root_world,
                                    )
                                {
                                    // Child node selected: AABB of that subtree.
                                    if let Some((amin, amax)) = wireframe::compute_node_tree_aabb(
                                        child_node, child_world,
                                    ) {
                                        verts.extend(wireframe::aabb_wireframe(
                                            amin, amax, color,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    verts
                };

                // 6. Render frame with overlay compositing.
                //    Take overlay state out temporarily to avoid borrow conflict
                //    with engine (both live in self).
                let mut rinch_ctx = self.rinch_ctx.take();
                let mut overlay_renderer = self.overlay_renderer.take();
                let overlay_blit = self.overlay_blit.take();
                let wireframe = self.wireframe_pass.take();

                if let Some(engine) = self.engine.as_mut() {
                    let vp_matrix = engine.view_projection();

                    if let Some(vp) = viewport {
                        engine.render_frame_viewport(vp, |device, queue, target_view| {
                            // Draw wireframe selection highlights.
                            if let Some(ref wf) = wireframe {
                                if !wireframe_verts.is_empty() {
                                    let mut enc = device.create_command_encoder(
                                        &wgpu::CommandEncoderDescriptor {
                                            label: Some("wireframe"),
                                        },
                                    );
                                    wf.draw(
                                        device, queue, &mut enc, target_view,
                                        vp_matrix, vp, &wireframe_verts,
                                    );
                                    queue.submit(std::iter::once(enc.finish()));
                                }
                            }

                            // Render rinch UI to overlay texture, then composite.
                            if let (Some(ctx), Some(renderer), Some(blit)) =
                                (&mut rinch_ctx, &mut overlay_renderer, &overlay_blit)
                            {
                                let scene = ctx.scene();
                                let overlay_view = renderer.render(device, queue, scene);
                                let mut enc = device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("overlay"),
                                    },
                                );
                                blit.draw(device, &mut enc, target_view, &overlay_view);
                                queue.submit(std::iter::once(enc.finish()));
                            }
                        });
                    } else {
                        // No viewport rect yet — render fullscreen (fallback).
                        engine.render_frame();
                    }
                }

                // Restore overlay state.
                self.rinch_ctx = rinch_ctx;
                self.overlay_renderer = overlay_renderer;
                self.overlay_blit = overlay_blit;
                self.wireframe_pass = wireframe;

                // 7. Update shared state for MCP observation.
                if let Ok(es) = self.editor_state.lock() {
                    if let Ok(mut ss) = self.shared_state.lock() {
                        ss.camera_position = es.editor_camera.position;
                        ss.camera_yaw = es.editor_camera.fly_yaw;
                        ss.camera_pitch = es.editor_camera.fly_pitch;
                        ss.camera_fov = es.editor_camera.fov_y.to_degrees();
                        ss.frame_time_ms = dt as f64 * 1000.0;
                    }
                }

                // Request next frame.
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    log::info!("RKIField Editor — v2 render pipeline + rinch UI");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
