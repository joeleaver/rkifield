//! RKIField editor — v2 object-centric render pipeline with rinch shell.
//!
//! Architecture:
//! - `rinch::shell::run_with_window_props_and_menu()` owns the window and event loop.
//! - The engine runs on a background thread with its own wgpu Device (separate from rinch).
//! - Engine renders to an offscreen texture, reads back pixels to CPU, and submits them
//!   via `SurfaceWriter::submit_frame()` for compositor integration.
//! - Input flows through `SurfaceEvent` → `EditorState.editor_input` → engine thread.

#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod animation_preview;
mod automation;
mod camera;
mod debug_viz;
mod editor_config;
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
mod editor_command;
mod engine_loop;
pub(crate) mod layout;
mod ui;
mod ui_snapshot;
mod undo;
mod wireframe;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use automation::SharedState;
use editor_command::EditorCommand;
use editor_state::{EditorState, SliderSignals, UiSignals};
use engine_viewport::{DISPLAY_HEIGHT, DISPLAY_WIDTH};
use ui_snapshot::UiSnapshot;

/// Sender half of the UI→engine command channel.
/// Stored in rinch context for UI components to send commands.
#[derive(Clone)]
pub(crate) struct CommandSender(pub crossbeam::channel::Sender<EditorCommand>);

/// Lock-free snapshot of editor state for the UI thread.
/// Published by the engine thread each frame, read by UI components.
#[derive(Clone)]
pub(crate) struct SnapshotReader(pub Arc<arc_swap::ArcSwap<UiSnapshot>>);


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

    // 1b. Create UI→engine command channel and lock-free snapshot.
    let (cmd_tx, cmd_rx) = crossbeam::channel::unbounded::<EditorCommand>();
    let ui_snapshot = Arc::new(arc_swap::ArcSwap::from_pointee(UiSnapshot::default()));
    let layout_backing = layout::state::LayoutBacking::new(layout::default_layout());

    // 1c. Create shared material library (pre-populated with test materials).
    let material_library = {
        use rkf_core::material_library::{MaterialLibrary, SlotInfo};
        #[allow(deprecated)]
        use rkf_render::material_table::create_test_materials;
        #[allow(deprecated)]
        let mats = create_test_materials();
        let mut lib = MaterialLibrary::new(mats.len().max(16));
        for (i, mat) in mats.into_iter().enumerate() {
            lib.set_material_with_info(
                i as u16,
                mat,
                SlotInfo {
                    name: format!("Material {i}"),
                    description: String::new(),
                    category: "Default".into(),
                    file_path: String::new(),
                    shader_name: "pbr".into(),
                },
            );
        }
        lib.clear_dirty(); // initial state is clean
        Arc::new(Mutex::new(lib))
    };

    // 2. Create tokio runtime and start IPC server.
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let api = automation::EditorAutomationApi::new(
        Arc::clone(&shared_state),
        Arc::clone(&editor_state),
        Arc::clone(&material_library),
    );
    let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
    let (_ipc_handle, socket_path) = start_ipc_server(&rt, api);

    // 3. Create the RenderSurface — identifies where in the rinch layout
    //    the engine's offscreen texture will be composited.
    let surface_handle = create_render_surface();

    // 4. Clones for the engine thread.
    let editor_state_for_thread = Arc::clone(&editor_state);
    let shared_state_for_thread = Arc::clone(&shared_state);
    let ui_snapshot_for_thread = Arc::clone(&ui_snapshot);
    let layout_backing_for_thread = layout_backing.clone();
    let surface_writer = surface_handle.writer();
    let gpu_registrar = surface_handle.gpu_registrar();
    let socket_path_for_cleanup = socket_path.clone();

    // 5. Spawn engine thread.
    //    Engine creates its own wgpu device — no dependency on rinch's device.
    std::thread::spawn(move || {
        engine_loop::engine_thread(engine_loop::EngineThreadData {
            editor_state: editor_state_for_thread,
            shared_state: shared_state_for_thread,
            cmd_rx,
            ui_snapshot: ui_snapshot_for_thread,
            layout_backing: layout_backing_for_thread,
            surface_writer,
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
    let cmd_sender = CommandSender(cmd_tx);
    let snapshot_reader = SnapshotReader(ui_snapshot);

    // 7. Call rinch shell — BLOCKING until window close.
    //    create_context() calls must be on the main
    //    thread inside this closure (both use thread-local rinch state).
    rinch::shell::run_with_window_props_and_menu(
        move |_scope| {
            // Register contexts so components can call use_context().
            // Centralized slider signals — one batch sync Effect replaces 33 lock closures.
            // Must be created before editor_state_ctx is moved into create_context.
            let slider_signals = SliderSignals::new(&editor_state_ctx.lock().expect("EditorState lock at init"));
            create_context(editor_state_ctx);
            create_context(shared_state_ctx);
            // Surface handle context — editor_ui uses it to wire SurfaceEvent → editor input.
            create_context(surface_handle);
            // Per-property UI signals for fine-grained reactivity.
            let ui_signals = UiSignals::new();
            create_context(ui_signals);
            create_context(slider_signals);
            create_context(crate::editor_state::FpsSignal::new());
            // Command channel + snapshot for lock-free UI ↔ engine communication.
            create_context(cmd_sender);
            create_context(snapshot_reader);
            // Layout state — zone-based configurable layout.
            let layout_state = layout::state::LayoutState::new(
                layout::default_layout(),
                DISPLAY_WIDTH as f32,
                DISPLAY_HEIGHT as f32,
            );
            create_context(layout_state);
            // Layout backing — cross-thread shared config for project save/load.
            create_context(layout_backing);

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
