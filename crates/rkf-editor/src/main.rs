//! RKIField editor — v2 object-centric render pipeline with camera controls.
//!
//! Provides a full rendering window with the v2 compute-shader pipeline,
//! orbit/fly camera controls, and IPC server for MCP agent tooling.

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
mod undo;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode as WinitKeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use automation::SharedState;
use editor_state::EditorState;
use engine::EditorEngine;
use engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};

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
}

impl App {
    fn new() -> Self {
        let editor_state = Arc::new(Mutex::new(EditorState::new()));
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0,
            engine::INTERNAL_WIDTH,
            engine::INTERNAL_HEIGHT,
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
        }
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

        // Initialize engine (wgpu, all render passes, demo scene).
        let engine_inst = EditorEngine::new(window.clone(), Arc::clone(&self.shared_state));
        self.engine = Some(engine_inst);

        // Start IPC server for MCP agent tooling.
        let api = automation::EditorAutomationApi::new(
            Arc::clone(&self.shared_state),
            Arc::clone(&self.editor_state),
        );
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
        let (handle, path) = start_ipc_server(&self.rt, api);
        self._ipc_handle = Some(handle);
        self.socket_path = Some(path);

        log::info!("RKIField Editor initialized — v2 render pipeline active");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        // Raw mouse motion for camera rotation (fires even when cursor is at edge).
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Ok(mut es) = self.editor_state.lock() {
                // Only accumulate delta when right mouse is held (camera rotation).
                if es.editor_input.is_mouse_button_down(1) {
                    es.editor_input.mouse_delta.x += delta.0 as f32;
                    es.editor_input.mouse_delta.y += delta.1 as f32;
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
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref path) = self.socket_path {
                    cleanup_ipc(path);
                }
                event_loop.exit();
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Ok(mut es) = self.editor_state.lock() {
                    es.editor_input.mouse_pos =
                        glam::Vec2::new(position.x as f32, position.y as f32);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Ok(mut es) = self.editor_state.lock() {
                    let idx = match button {
                        MouseButton::Left => 0,
                        MouseButton::Right => 1,
                        MouseButton::Middle => 2,
                        _ => return,
                    };
                    es.editor_input.mouse_buttons[idx] = state == ElementState::Pressed;

                    // Hide cursor when right-mouse is held (fly/orbit rotation).
                    if idx == 1 {
                        if let Some(window) = &self.window {
                            let _ = window.set_cursor_visible(state != ElementState::Pressed);
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Ok(mut es) = self.editor_state.lock() {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
                    };
                    es.editor_input.scroll_delta += scroll;
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    // Escape exits.
                    if key == WinitKeyCode::Escape && event.state == ElementState::Pressed {
                        if let Some(ref path) = self.socket_path {
                            cleanup_ipc(path);
                        }
                        event_loop.exit();
                        return;
                    }

                    // Translate to editor key codes.
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

            WindowEvent::ModifiersChanged(modifiers) => {
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
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // 1. Lock editor state, update camera from input, get camera params.
                if let Ok(mut es) = self.editor_state.lock() {
                    // Apply pending MCP commands.
                    if let Ok(mut ss) = self.shared_state.lock() {
                        if let Some(cam) = ss.pending_camera.take() {
                            es.editor_camera.position = cam.position;
                            es.editor_camera.fly_yaw = cam.yaw;
                            es.editor_camera.fly_pitch = cam.pitch;
                            // Update orbit target to match for mode-switch consistency.
                            let dir = glam::Vec3::new(
                                -cam.yaw.sin() * cam.pitch.cos(),
                                cam.pitch.sin(),
                                -cam.yaw.cos() * cam.pitch.cos(),
                            );
                            es.editor_camera.target = es.editor_camera.position + dir * es.editor_camera.orbit_distance;
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

                // 2. Render frame (engine internally locks SharedState for readback).
                if let Some(engine) = self.engine.as_mut() {
                    engine.render_frame();
                }

                // 3. Update shared state for MCP observation.
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
    log::info!("RKIField Editor — v2 object-centric render pipeline");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
