#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod animation_preview;
mod automation;
mod camera;
mod debug_viz;
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
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use rinch::embed::{RinchContext, RinchContextConfig};
use rinch::prelude::*;
use rinch_platform::{
    KeyCode as PlatformKeyCode, Modifiers as PlatformModifiers,
    MouseButton as PlatformMouseButton, PlatformEvent,
};

use automation::{EditorAutomationApi, SharedState};
use engine_viewport::{EngineState, DISPLAY_HEIGHT, DISPLAY_WIDTH};

// ---------------------------------------------------------------------------
// Key translation (winit → rinch platform)
// ---------------------------------------------------------------------------

fn translate_key(key: winit::keyboard::KeyCode) -> PlatformKeyCode {
    use winit::keyboard::KeyCode as WK;
    match key {
        WK::ArrowLeft => PlatformKeyCode::ArrowLeft,
        WK::ArrowRight => PlatformKeyCode::ArrowRight,
        WK::ArrowUp => PlatformKeyCode::ArrowUp,
        WK::ArrowDown => PlatformKeyCode::ArrowDown,
        WK::Home => PlatformKeyCode::Home,
        WK::End => PlatformKeyCode::End,
        WK::PageUp => PlatformKeyCode::PageUp,
        WK::PageDown => PlatformKeyCode::PageDown,
        WK::Enter | WK::NumpadEnter => PlatformKeyCode::Enter,
        WK::Backspace => PlatformKeyCode::Backspace,
        WK::Delete => PlatformKeyCode::Delete,
        WK::Tab => PlatformKeyCode::Tab,
        WK::Escape => PlatformKeyCode::Escape,
        WK::Space => PlatformKeyCode::Space,
        WK::KeyA => PlatformKeyCode::KeyA,
        WK::KeyB => PlatformKeyCode::KeyB,
        WK::KeyC => PlatformKeyCode::KeyC,
        WK::KeyD => PlatformKeyCode::KeyD,
        WK::KeyE => PlatformKeyCode::KeyE,
        WK::KeyF => PlatformKeyCode::KeyF,
        WK::KeyG => PlatformKeyCode::KeyG,
        WK::KeyH => PlatformKeyCode::KeyH,
        WK::KeyI => PlatformKeyCode::KeyI,
        WK::KeyJ => PlatformKeyCode::KeyJ,
        WK::KeyK => PlatformKeyCode::KeyK,
        WK::KeyL => PlatformKeyCode::KeyL,
        WK::KeyM => PlatformKeyCode::KeyM,
        WK::KeyN => PlatformKeyCode::KeyN,
        WK::KeyO => PlatformKeyCode::KeyO,
        WK::KeyP => PlatformKeyCode::KeyP,
        WK::KeyQ => PlatformKeyCode::KeyQ,
        WK::KeyR => PlatformKeyCode::KeyR,
        WK::KeyS => PlatformKeyCode::KeyS,
        WK::KeyT => PlatformKeyCode::KeyT,
        WK::KeyU => PlatformKeyCode::KeyU,
        WK::KeyV => PlatformKeyCode::KeyV,
        WK::KeyW => PlatformKeyCode::KeyW,
        WK::KeyX => PlatformKeyCode::KeyX,
        WK::KeyY => PlatformKeyCode::KeyY,
        WK::KeyZ => PlatformKeyCode::KeyZ,
        WK::Digit0 => PlatformKeyCode::Digit0,
        WK::Digit1 => PlatformKeyCode::Digit1,
        WK::Digit2 => PlatformKeyCode::Digit2,
        WK::Digit3 => PlatformKeyCode::Digit3,
        WK::Digit4 => PlatformKeyCode::Digit4,
        WK::Digit5 => PlatformKeyCode::Digit5,
        WK::Digit6 => PlatformKeyCode::Digit6,
        WK::Digit7 => PlatformKeyCode::Digit7,
        WK::Digit8 => PlatformKeyCode::Digit8,
        WK::Digit9 => PlatformKeyCode::Digit9,
        WK::F1 => PlatformKeyCode::F1,
        WK::F2 => PlatformKeyCode::F2,
        WK::F3 => PlatformKeyCode::F3,
        WK::F4 => PlatformKeyCode::F4,
        WK::F5 => PlatformKeyCode::F5,
        WK::F6 => PlatformKeyCode::F6,
        WK::F7 => PlatformKeyCode::F7,
        WK::F8 => PlatformKeyCode::F8,
        WK::F9 => PlatformKeyCode::F9,
        WK::F10 => PlatformKeyCode::F10,
        WK::F11 => PlatformKeyCode::F11,
        WK::F12 => PlatformKeyCode::F12,
        WK::Equal => PlatformKeyCode::Equal,
        WK::Minus => PlatformKeyCode::Minus,
        _ => PlatformKeyCode::Other,
    }
}

fn translate_modifiers(m: winit::keyboard::ModifiersState) -> PlatformModifiers {
    PlatformModifiers {
        shift: m.shift_key(),
        ctrl: m.control_key(),
        alt: m.alt_key(),
        meta: m.super_key(),
    }
}

// ---------------------------------------------------------------------------
// Editor UI component
// ---------------------------------------------------------------------------

#[component]
fn editor_ui() -> NodeHandle {
    // Root fills the entire window
    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "display: flex; flex-direction: column; width: 100%; height: 100%;",
    );

    // Menu bar
    let menu = rsx! {
        div {
            style: "display: flex; flex-direction: row; height: 32px; background: #2b2b2b; \
                    align-items: center; padding: 0 8px; gap: 12px; \
                    border-bottom: 1px solid #333;",
            span { style: "color: #aaa; font-weight: bold;", "RKIField Editor" }
            span { style: "color: #888; cursor: pointer;", "File" }
            span { style: "color: #888; cursor: pointer;", "Edit" }
            span { style: "color: #888; cursor: pointer;", "View" }
            span { style: "color: #888; cursor: pointer;", "Tools" }
        }
    };
    root.append_child(&menu);

    // Main content area
    let content = __scope.create_element("div");
    content.set_attribute(
        "style",
        "display: flex; flex-direction: row; flex: 1; overflow: hidden;",
    );

    // Left panel (scene hierarchy)
    let left_panel = rsx! {
        div {
            style: "width: 250px; background: #1e1e1e; border-right: 1px solid #333; \
                    overflow-y: auto; padding: 8px;",
            span { style: "color: #ccc; font-weight: bold;", "Scene Hierarchy" }
        }
    };
    content.append_child(&left_panel);

    // Engine viewport — transparent hole with data-viewport for input routing
    let viewport = __scope.create_element("div");
    viewport.set_attribute("data-viewport", "main");
    viewport.set_attribute(
        "style",
        "flex: 1; pointer-events: none; background: transparent;",
    );
    content.append_child(&viewport);

    // Right panel (properties)
    let right_panel = rsx! {
        div {
            style: "width: 300px; background: #1e1e1e; border-left: 1px solid #333; \
                    overflow-y: auto; padding: 8px;",
            span { style: "color: #ccc; font-weight: bold;", "Properties" }
        }
    };
    content.append_child(&right_panel);

    root.append_child(&content);

    // Status bar
    let status = rsx! {
        div {
            style: "height: 24px; background: #2b2b2b; display: flex; align-items: center; \
                    padding: 0 8px; border-top: 1px solid #333;",
            span { style: "color: #666; font-size: 12px;", "Ready" }
        }
    };
    root.append_child(&status);

    root
}

// ---------------------------------------------------------------------------
// IPC server
// ---------------------------------------------------------------------------

fn spawn_ipc_server(api: Arc<dyn rkf_core::automation::AutomationApi>) -> String {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let path_clone = socket_path.clone();

    std::thread::Builder::new()
        .name("ipc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime for IPC server");

            rt.block_on(async {
                let mut registry = rkf_mcp::registry::ToolRegistry::new();
                rkf_mcp::tools::observation::register_observation_tools(&mut registry);
                let registry = Arc::new(registry);

                let config = rkf_mcp::ipc::IpcConfig {
                    socket_path: Some(path_clone),
                    tcp_port: 0,
                    mode: rkf_mcp::registry::ToolMode::Debug,
                };

                if let Err(e) = rkf_mcp::ipc::run_server(config, registry, api).await {
                    log::error!("IPC server error: {e}");
                }
            });
        })
        .expect("failed to spawn IPC server thread");

    socket_path
}

// ---------------------------------------------------------------------------
// Camera input state
// ---------------------------------------------------------------------------

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl InputState {
    fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }

    /// Process a key event for camera movement. Returns Some(debug_mode) if a
    /// number key was pressed.
    fn process_key(&mut self, key: winit::keyboard::KeyCode, pressed: bool) -> Option<u32> {
        use winit::keyboard::KeyCode;
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            KeyCode::Digit0 if pressed => return Some(0),
            KeyCode::Digit1 if pressed => return Some(1),
            KeyCode::Digit2 if pressed => return Some(2),
            KeyCode::Digit3 if pressed => return Some(3),
            KeyCode::Digit4 if pressed => return Some(4),
            KeyCode::Digit5 if pressed => return Some(5),
            KeyCode::Digit6 if pressed => return Some(6),
            _ => {}
        }
        None
    }

    fn apply_to_camera(&self, cam: &mut rkf_render::camera::Camera, dt: f32) {
        let speed = cam.move_speed * dt;
        if self.forward {
            cam.translate_forward(speed);
        }
        if self.backward {
            cam.translate_forward(-speed);
        }
        if self.right {
            cam.translate_right(speed);
        }
        if self.left {
            cam.translate_right(-speed);
        }
        if self.up {
            cam.translate_up(speed);
        }
        if self.down {
            cam.translate_up(-speed);
        }
    }
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    // Window
    window: Option<Arc<Window>>,

    // wgpu core
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,

    // Rinch embed (headless — overlay rendering deferred to wgpu 27 upgrade)
    // When overlay is available, set this to true to enable UI input routing.
    rinch_ctx: Option<RinchContext>,
    ui_visible: bool,

    // Engine
    engine: Option<EngineState>,

    // Input
    cam_input: InputState,
    mouse_phys: (f32, f32),
    prev_mouse: (f32, f32),
    dragging: bool,
    modifiers: winit::keyboard::ModifiersState,
    pending_events: Vec<PlatformEvent>,

    // MCP
    shared_state: Arc<Mutex<SharedState>>,
    socket_path: Option<String>,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    last_title_update: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            surface_format: wgpu::TextureFormat::Bgra8Unorm,
            surface_width: DISPLAY_WIDTH,
            surface_height: DISPLAY_HEIGHT,
            rinch_ctx: None,
            ui_visible: false, // No overlay until wgpu 27 upgrade
            engine: None,
            cam_input: InputState::new(),
            mouse_phys: (0.0, 0.0),
            prev_mouse: (0.0, 0.0),
            dragging: false,
            modifiers: winit::keyboard::ModifiersState::empty(),
            pending_events: Vec::new(),
            shared_state: Arc::new(Mutex::new(SharedState::new(
                0,
                0,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            ))),
            socket_path: None,
            last_frame: Instant::now(),
            frame_count: 0,
            last_title_update: Instant::now(),
        }
    }

    fn scale_factor(&self) -> f64 {
        self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0)
    }

    fn logical_mouse(&self) -> (f32, f32) {
        let sf = self.scale_factor() as f32;
        (self.mouse_phys.0 / sf, self.mouse_phys.1 / sf)
    }

    // ── GPU initialization ─────────────────────────────────────────────────

    fn init_gpu(&mut self) {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let w = size.width.max(1);
        let h = size.height.max(1);

        // Create wgpu instance + surface
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        // Use RenderContext for device creation (gets rkf features: FLOAT32_FILTERABLE, high limits)
        let render_ctx = rkf_render::RenderContext::new(&instance, &surface);
        let device = render_ctx.device;
        let queue = render_ctx.queue;

        // Configure surface with COPY_SRC for screenshot support
        let caps = surface.get_capabilities(&render_ctx.adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        self.surface_format = format;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        self.surface_width = w;
        self.surface_height = h;

        // ── Rinch embed (headless UI for input routing) ────────────────────
        // NOTE: RinchOverlayRenderer requires wgpu 27 (rinch's version) but
        // our workspace uses wgpu 24. The RinchContext is headless and doesn't
        // cross the wgpu boundary, so input routing works. Visual overlay
        // compositing is deferred until we upgrade wgpu to 27.

        let rinch_ctx = RinchContext::new(
            RinchContextConfig {
                width: w,
                height: h,
                scale_factor: self.scale_factor(),
                theme: Some(ThemeProviderProps {
                    dark_mode: true,
                    primary_color: Some("blue".into()),
                    ..Default::default()
                }),
            },
            editor_ui,
        );

        // ── Engine ─────────────────────────────────────────────────────────

        let engine = EngineState::new(
            &device,
            &queue,
            format,
            Arc::clone(&self.shared_state),
        );

        // Store everything
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.rinch_ctx = Some(rinch_ctx);
        self.engine = Some(engine);

        // Spawn MCP IPC server
        let api: Arc<dyn rkf_core::automation::AutomationApi> =
            Arc::new(EditorAutomationApi::new(Arc::clone(&self.shared_state)));
        let socket_path = spawn_ipc_server(api);
        log::info!("IPC server listening on {socket_path}");
        self.socket_path = Some(socket_path);

        log::info!("Editor initialized — engine viewport active");
    }

    // ── Render one frame ───────────────────────────────────────────────────

    fn render(&mut self) {
        let surface = self.surface.as_ref().unwrap();

        // Acquire surface texture
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Skip this frame — will reconfigure on next resize
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e:?}");
                return;
            }
        };
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Timing
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_count += 1;

        // Update rinch context (headless — for input routing only)
        if let Some(ctx) = &mut self.rinch_ctx {
            let events: Vec<_> = self.pending_events.drain(..).collect();
            let _actions = ctx.update(&events);
        }

        // Apply camera movement from keyboard input
        if let Some(engine) = &mut self.engine {
            self.cam_input.apply_to_camera(&mut engine.camera, dt);
        }

        // Engine renders full pipeline + blits to surface
        if let Some(engine) = &mut self.engine {
            engine.render(&surface_view, dt);
        }

        frame.present();

        // Update title bar with FPS every 500ms
        if now.duration_since(self.last_title_update).as_millis() > 500 {
            if let Some(window) = &self.window {
                let elapsed = now.duration_since(self.last_title_update).as_secs_f64();
                let fps = self.frame_count as f64 / elapsed;
                window.set_title(&format!(
                    "RKIField Editor — {fps:.0} fps ({:.2} ms)",
                    1000.0 / fps
                ));
                self.frame_count = 0;
                self.last_title_update = now;
            }
        }
    }

    // ── Handle resize ──────────────────────────────────────────────────────

    fn handle_resize(&mut self, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        self.surface_width = w;
        self.surface_height = h;

        if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: self.surface_format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };
            surface.configure(device, &config);
        }

        if let Some(ctx) = &mut self.rinch_ctx {
            ctx.resize(w, h);
        }
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("RKIField Editor")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            ));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window);

        self.init_gpu();
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.handle_resize(size.width, size.height);
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(ctx) = &mut self.rinch_ctx {
                    ctx.set_scale_factor(scale_factor);
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                self.pending_events.push(PlatformEvent::ModifiersChanged(
                    translate_modifiers(self.modifiers),
                ));
            }

            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;

                // Right-drag rotates camera in viewport
                if self.dragging {
                    let dx = px - self.prev_mouse.0;
                    let dy = py - self.prev_mouse.1;
                    if let Some(engine) = &mut self.engine {
                        engine.camera.yaw -= dx * 0.003;
                        engine.camera.pitch -= dy * 0.003;
                        engine.camera.pitch = engine.camera.pitch.clamp(-1.5, 1.5);
                    }
                }

                self.prev_mouse = self.mouse_phys;
                self.mouse_phys = (px, py);

                // Always send mouse move to rinch (for hover effects)
                self.pending_events
                    .push(PlatformEvent::MouseMove { x: px, y: py });
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let platform_btn = match button {
                    winit::event::MouseButton::Left => PlatformMouseButton::Left,
                    winit::event::MouseButton::Right => PlatformMouseButton::Right,
                    winit::event::MouseButton::Middle => PlatformMouseButton::Middle,
                    _ => return,
                };
                let (px, py) = self.mouse_phys;

                match state {
                    ElementState::Pressed => {
                        let (lx, ly) = self.logical_mouse();
                        let wants_ui = self.ui_visible
                            && self
                                .rinch_ctx
                                .as_ref()
                                .is_some_and(|ctx| ctx.wants_mouse(lx, ly));

                        if wants_ui {
                            // Send to rinch UI
                            self.pending_events.push(PlatformEvent::MouseDown {
                                x: px,
                                y: py,
                                button: platform_btn,
                            });
                        } else if platform_btn == PlatformMouseButton::Right {
                            // Right-drag for camera rotation in viewport
                            self.dragging = true;
                        }
                    }
                    ElementState::Released => {
                        if self.dragging && platform_btn == PlatformMouseButton::Right {
                            self.dragging = false;
                        } else {
                            self.pending_events.push(PlatformEvent::MouseUp {
                                x: px,
                                y: py,
                                button: platform_btn,
                            });
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y,
                };
                let (lx, ly) = self.logical_mouse();
                let wants_ui = self.ui_visible
                    && self
                        .rinch_ctx
                        .as_ref()
                        .is_some_and(|ctx| ctx.wants_mouse(lx, ly));

                if wants_ui {
                    let (px, py) = self.mouse_phys;
                    self.pending_events.push(PlatformEvent::MouseWheel {
                        x: px,
                        y: py,
                        delta_x: 0.0,
                        delta_y: scroll_y,
                    });
                } else {
                    // Scroll to move camera forward/backward
                    if let Some(engine) = &mut self.engine {
                        engine.camera.translate_forward(scroll_y as f32 * 0.02);
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let key_code = match event.physical_key {
                    PhysicalKey::Code(k) => k,
                    _ => return,
                };

                // Escape always exits
                if key_code == winit::keyboard::KeyCode::Escape && pressed {
                    event_loop.exit();
                    return;
                }

                let wants_kb = self.ui_visible
                    && self
                        .rinch_ctx
                        .as_ref()
                        .is_some_and(|ctx| ctx.wants_keyboard());

                if wants_kb {
                    // Route to rinch for text input
                    let platform_key = translate_key(key_code);
                    let text = event.text.as_ref().map(|s| s.to_string());
                    let mods = translate_modifiers(self.modifiers);
                    if pressed {
                        self.pending_events.push(PlatformEvent::KeyDown {
                            key: platform_key,
                            text,
                            modifiers: mods,
                        });
                    }
                } else {
                    // Camera movement + debug shortcuts
                    if let Some(debug_mode) = self.cam_input.process_key(key_code, pressed) {
                        if let Some(engine) = &self.engine {
                            engine.shading.set_debug_mode(
                                self.queue.as_ref().unwrap(),
                                debug_mode,
                            );
                            log::info!("Debug mode: {debug_mode}");
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
