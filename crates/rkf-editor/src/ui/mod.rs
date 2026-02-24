//! Editor UI root component.
//!
//! Defines the rinch UI layout: left panel (scene tree), center viewport hole,
//! right panel (properties/tools), top toolbar, and bottom status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod properties_panel;
pub mod scene_tree_panel;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_state::{EditorState, UiRevision};
use properties_panel::PropertiesPanel;
use scene_tree_panel::SceneTreePanel;

// ── Style constants ─────────────────────────────────────────────────────────

const PANEL_BG: &str = "background:rgb(32,32,38);";
const PANEL_BORDER: &str = "border:1px solid rgba(255,255,255,0.06);";
const TOOLBAR_HEIGHT: &str = "height:38px;";
const STATUS_HEIGHT: &str = "height:25px;";
const LEFT_PANEL_WIDTH: &str = "width:250px;";
const RIGHT_PANEL_WIDTH: &str = "width:300px;";

// ── Root component ──────────────────────────────────────────────────────────

/// Root editor UI component.
///
/// Layout:
/// ```text
/// ┌──────────────────────────────────────────────────┐
/// │                   Toolbar (38px)                  │
/// ├────────┬─────────────────────────────┬────────────┤
/// │  Left  │       GameViewport          │   Right    │
/// │ 250px  │        (flex: 1)            │  300px     │
/// │        │                             │            │
/// ├────────┴─────────────────────────────┴────────────┤
/// │                  Status Bar (25px)                 │
/// └──────────────────────────────────────────────────┘
/// ```
///
/// Expects `Arc<Mutex<EditorState>>`, `Arc<Mutex<SharedState>>`, and
/// `UiRevision` to be available via `use_context` (set up in main.rs).
#[component]
pub fn editor_ui() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    // Read initial mode name for toolbar (static — mode switching is future work).
    let mode_name = {
        let es = editor_state.lock().unwrap();
        es.mode.name().to_string()
    };

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;\
                    background:rgb(24,24,28);color:white;font-family:system-ui,sans-serif;",

            // ── Top toolbar ──
            div {
                style: {format!("display:flex;align-items:center;{TOOLBAR_HEIGHT}\
                    {PANEL_BG}border-bottom:1px solid rgba(255,255,255,0.08);\
                    padding:0 12px;gap:8px;")},
                div {
                    style: "font-size:13px;font-weight:600;color:rgba(255,255,255,0.8);",
                    "RKIField Editor"
                }
                div { style: "flex:1;" }
                div {
                    style: "font-size:11px;color:rgba(255,255,255,0.35);",
                    {mode_name.clone()}
                }
            }

            // ── Main content row (left + viewport + right) ──
            div {
                style: "display:flex;flex:1;min-height:0;",

                // Left panel — scene tree (live component)
                div {
                    style: {format!("{LEFT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-right:1px solid rgba(255,255,255,0.08);\
                        display:flex;flex-direction:column;overflow:hidden;")},
                    SceneTreePanel {}
                }

                // Center viewport
                GameViewport { name: "main", style: "flex:1;" }

                // Right panel — properties (live component)
                div {
                    style: {format!("{RIGHT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-left:1px solid rgba(255,255,255,0.08);\
                        display:flex;flex-direction:column;overflow:hidden;")},
                    PropertiesPanel {}
                }
            }

            // ── Bottom status bar ──
            StatusBar {}
        }
    }
}

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, and current mode.
#[component]
pub fn StatusBar() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();
    let revision = use_context::<UiRevision>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        &format!(
            "display:flex;align-items:center;{STATUS_HEIGHT}\
            {PANEL_BG}border-top:1px solid rgba(255,255,255,0.08);\
            padding:0 12px;gap:16px;\
            font-size:11px;color:rgba(255,255,255,0.4);"
        ),
    );

    let es = editor_state.clone();
    let ss = shared_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        revision.track();

        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display:flex;align-items:center;width:100%;gap:16px;",
        );

        let (obj_count, mode_name) = {
            let es = es.lock().unwrap();
            (es.scene_tree.roots.len(), es.mode.name().to_string())
        };
        let frame_time_ms = ss
            .lock()
            .map(|s| s.frame_time_ms)
            .unwrap_or(0.0);
        let fps = if frame_time_ms > 0.1 {
            format!("{:.0} fps", 1000.0 / frame_time_ms)
        } else {
            "-- fps".to_string()
        };

        let obj_div = __scope.create_element("div");
        obj_div.append_child(&__scope.create_text(&format!("{obj_count} objects")));
        container.append_child(&obj_div);

        let fps_div = __scope.create_element("div");
        fps_div.append_child(&__scope.create_text(&fps));
        container.append_child(&fps_div);

        let spacer = __scope.create_element("div");
        spacer.set_attribute("style", "flex:1;");
        container.append_child(&spacer);

        let mode_div = __scope.create_element("div");
        mode_div.append_child(&__scope.create_text(&format!("{mode_name} mode")));
        container.append_child(&mode_div);

        container
    });

    root
}
