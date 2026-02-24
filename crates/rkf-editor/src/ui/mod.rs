//! Editor UI root component.
//!
//! Defines the rinch UI layout: left panel (scene tree), center viewport hole,
//! right panel (properties/tools), top toolbar, and bottom status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod scene_tree_panel;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_state::EditorState;
use scene_tree_panel::SceneTreePanel;

// ── Style constants ─────────────────────────────────────────────────────────

const PANEL_BG: &str = "background:rgb(32,32,38);";
const PANEL_BORDER: &str = "border:1px solid rgba(255,255,255,0.06);";
const TOOLBAR_HEIGHT: &str = "height:38px;";
const STATUS_HEIGHT: &str = "height:25px;";
const LEFT_PANEL_WIDTH: &str = "width:250px;";
const RIGHT_PANEL_WIDTH: &str = "width:300px;";
const LABEL_STYLE: &str = "font-size:11px;color:rgba(255,255,255,0.45);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";
const SECTION_STYLE: &str = "font-size:12px;color:rgba(255,255,255,0.7);padding:6px 12px;";

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
/// Expects `Arc<Mutex<EditorState>>` and `Arc<Mutex<SharedState>>` to be
/// available via `use_context` (set up by `create_context` in main.rs).
#[component]
pub fn editor_ui() -> NodeHandle {
    // Read editor state for toolbar mode display.
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();

    // Read current mode and stats for status bar / toolbar.
    let (mode_name, obj_count, frame_time_ms, selected_name) = {
        let es = editor_state.lock().unwrap();
        let sel_name = es
            .scene_tree
            .selected_entities()
            .first()
            .and_then(|id| es.scene_tree.find_node(*id))
            .map(|n| n.name.clone());
        let count = es.scene_tree.roots.len();
        let mode = es.mode.name();
        let ft = shared_state
            .lock()
            .map(|ss| ss.frame_time_ms)
            .unwrap_or(0.0);
        (mode.to_string(), count, ft, sel_name)
    };

    let fps = if frame_time_ms > 0.1 {
        format!("{:.0} fps", 1000.0 / frame_time_ms)
    } else {
        "-- fps".to_string()
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

                // Right panel — properties / tools
                div {
                    style: {format!("{RIGHT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-left:1px solid rgba(255,255,255,0.08);\
                        display:flex;flex-direction:column;overflow:hidden;")},
                    div { style: LABEL_STYLE, "Properties" }
                    div {
                        style: {format!("{SECTION_STYLE}color:rgba(255,255,255,0.35);")},
                        {selected_name.unwrap_or_else(|| "No object selected".to_string())}
                    }
                }
            }

            // ── Bottom status bar ──
            div {
                style: {format!("display:flex;align-items:center;{STATUS_HEIGHT}\
                    {PANEL_BG}border-top:1px solid rgba(255,255,255,0.08);\
                    padding:0 12px;gap:16px;\
                    font-size:11px;color:rgba(255,255,255,0.4);")},
                div { {format!("{obj_count} objects")} }
                div { {fps} }
                div { style: "flex:1;" }
                div { {format!("{mode_name} mode")} }
            }
        }
    }
}
