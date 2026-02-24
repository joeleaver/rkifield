//! Editor UI root component.
//!
//! Defines the rinch UI layout: left panel (scene tree), center viewport hole,
//! right panel (properties/tools), top toolbar, and bottom status bar.
//! Panels are placeholder shells — content wired in later phases.

use rinch::prelude::*;

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
#[component]
pub fn editor_ui() -> NodeHandle {
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
                    "Navigate"
                }
            }

            // ── Main content row (left + viewport + right) ──
            div {
                style: "display:flex;flex:1;min-height:0;",

                // Left panel — scene tree
                div {
                    style: {format!("{LEFT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-right:1px solid rgba(255,255,255,0.08);\
                        display:flex;flex-direction:column;overflow:hidden;")},
                    div { style: LABEL_STYLE, "Scene" }
                    div {
                        style: SECTION_STYLE,
                        "ground"
                    }
                    div {
                        style: SECTION_STYLE,
                        "sphere"
                    }
                    div {
                        style: SECTION_STYLE,
                        "box"
                    }
                    div {
                        style: SECTION_STYLE,
                        "capsule"
                    }
                    div {
                        style: SECTION_STYLE,
                        "torus"
                    }
                    div {
                        style: SECTION_STYLE,
                        "vox_sphere"
                    }
                    div {
                        style: SECTION_STYLE,
                        "humanoid"
                    }
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
                        "No object selected"
                    }
                }
            }

            // ── Bottom status bar ──
            div {
                style: {format!("display:flex;align-items:center;{STATUS_HEIGHT}\
                    {PANEL_BG}border-top:1px solid rgba(255,255,255,0.08);\
                    padding:0 12px;gap:16px;\
                    font-size:11px;color:rgba(255,255,255,0.4);")},
                div { "7 objects" }
                div { style: "flex:1;" }
                div { "Navigate mode" }
            }
        }
    }
}
