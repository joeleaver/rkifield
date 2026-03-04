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
mod eikonal_repair;
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
mod engine_loop;
mod ui;
mod undo;
mod wireframe;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use automation::SharedState;
use editor_state::{EditorState, SliderSignals, UiRevision, UiSignals};
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
        engine_loop::engine_thread(engine_loop::EngineThreadData {
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
