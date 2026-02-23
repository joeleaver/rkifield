//! RKIField visual testbed — stubbed pending v2 rewrite.
//!
//! In v2, this will be rebuilt with object-centric rendering (Phase 5).
//! Currently provides a minimal window with an empty render loop and
//! IPC server for MCP agent tooling.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod automation;
use automation::{SharedState, TestbedAutomationApi};

/// Display (output) resolution width — the window size.
const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height — the window size.
const DISPLAY_HEIGHT: u32 = 720;

/// Internal render resolution width (v2 stub — not yet rendering).
#[allow(dead_code)]
const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height (v2 stub — not yet rendering).
#[allow(dead_code)]
const INTERNAL_HEIGHT: u32 = 540;

// ---------------------------------------------------------------------------
// IPC server setup
// ---------------------------------------------------------------------------

/// Start the IPC server for MCP agent tooling.
fn start_ipc_server(
    rt: &tokio::runtime::Runtime,
    api: Arc<dyn rkf_core::automation::AutomationApi>,
) -> (tokio::task::JoinHandle<()>, String) {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();

    // Write discovery metadata
    let meta_path = socket_path.replace(".sock", ".json");
    let meta = serde_json::json!({
        "type": "testbed",
        "pid": std::process::id(),
        "socket": &socket_path,
        "name": "RKIField Testbed",
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

/// Clean up IPC socket and discovery metadata on exit.
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
    shared_state: Arc<Mutex<SharedState>>,
    last_frame: Instant,
    _ipc_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: Option<String>,
    rt: tokio::runtime::Runtime,
}

impl App {
    fn new() -> Self {
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        )));
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        Self {
            window: None,
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

        let attrs = WindowAttributes::default()
            .with_title("RKIField Testbed (v2 stub)")
            .with_inner_size(PhysicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        // Start IPC server for MCP agent tooling
        let api = TestbedAutomationApi::new(Arc::clone(&self.shared_state));
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
        let (handle, path) = start_ipc_server(&self.rt, api);
        self._ipc_handle = Some(handle);
        self.socket_path = Some(path);

        log::info!("Testbed window created (v2 stub — no rendering yet)");
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
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                        if let Some(ref path) = self.socket_path {
                            cleanup_ipc(path);
                        }
                        event_loop.exit();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame);
                self.last_frame = now;

                // Update shared state for MCP
                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_time_ms = dt.as_secs_f64() * 1000.0;
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    log::info!("RKIField Testbed v2 (stub) — no rendering until Phase 5");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
