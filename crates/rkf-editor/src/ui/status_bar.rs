//! Status bar component — object count, FPS, selection info, mode display.

use rinch::prelude::*;

use crate::editor_state::{SelectedEntity, UiSignals};

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
///
/// Uses a lightweight reactive_component_dom for layout changes (selection
/// presence, debug mode, grid toggle) with an independent FPS Effect to
/// avoid rebuilding on every frame time update.
#[component]
pub fn StatusBar() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let root = rsx! {
        div {
            style: "display:flex;align-items:center;height:25px;\
                background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
                padding:0 12px;gap:16px;\
                font-size:11px;color:var(--rinch-color-dimmed);",
        }
    };

    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking — only rebuild when these specific values change.
        let selection = ui.selection.get();
        let editor_mode = ui.editor_mode.get();
        let gizmo_mode = ui.gizmo_mode.get();
        let debug_mode = ui.debug_mode.get();
        let show_grid = ui.show_grid.get();
        let objects = ui.objects.get();
        let lights = ui.lights.get();

        let obj_count = objects.len();
        let mode_name = editor_mode.name().to_string();
        let selected_name = selection.as_ref().map(|sel| match sel {
            SelectedEntity::Object(eid) => {
                objects.iter()
                    .find(|o| o.id == *eid)
                    .map(|o| o.name.clone())
                    .unwrap_or_else(|| format!("Object {eid}"))
            }
            SelectedEntity::Light(lid) => {
                lights.iter()
                    .find(|l| l.id == *lid)
                    .map(|l| match l.light_type {
                        crate::light_editor::SceneLightType::Point => format!("Point Light {lid}"),
                        crate::light_editor::SceneLightType::Spot => format!("Spot Light {lid}"),
                    })
                    .unwrap_or_else(|| format!("Light {lid}"))
            }
            SelectedEntity::Camera => "Camera".to_string(),
            SelectedEntity::Scene => "Scene".to_string(),
            SelectedEntity::Project => "Project".to_string(),
        });
        let debug_name = crate::ui_snapshot::debug_mode_name(debug_mode).to_string();
        let gizmo_mode_name = format!("{gizmo_mode:?}");

        let container = rsx! {
            div {
                style: "display:flex;align-items:center;width:100%;gap:16px;",

                // Object count.
                div { {format!("{obj_count} objects")} }
            }
        };

        // FPS: own reactive text node tracking UiSignals.fps (not revision).
        let fps_div = __scope.create_element("div");
        rinch::core::reactive_component_dom(__scope, &fps_div, move |__scope| {
            let ms = ui.fps.get();
            let label = if ms > 0.1 {
                format!("{:.0} fps", 1000.0 / ms)
            } else {
                "-- fps".to_string()
            };
            __scope.create_text(&label)
        });
        container.append_child(&fps_div);

        // Selected object name.
        if let Some(name) = &selected_name {
            let sel_div = rsx! {
                div {
                    style: "color:var(--rinch-primary-color);",
                    {name.clone()}
                }
            };
            container.append_child(&sel_div);
        }

        // Debug mode indicator.
        if !debug_name.is_empty() {
            let dbg_div = rsx! {
                div {
                    style: "color:var(--rinch-color-yellow-5, #fcc419);",
                    {debug_name.clone()}
                }
            };
            container.append_child(&dbg_div);
        }

        let spacer = rsx! {
            div { style: "flex:1;", }
        };
        container.append_child(&spacer);

        // Gizmo mode indicator.
        let gizmo_div = rsx! {
            div {
                style: "color:var(--rinch-color-dimmed);",
                {gizmo_mode_name.clone()}
            }
        };
        container.append_child(&gizmo_div);

        // Show active tool mode (only when Sculpt/Paint active).
        if !mode_name.is_empty() {
            let mode_div = rsx! {
                div {
                    style: "color:var(--rinch-primary-color);",
                    {format!("{mode_name} mode")}
                }
            };
            container.append_child(&mode_div);
        }

        // Grid indicator.
        if show_grid {
            let grid_div = rsx! {
                div {
                    style: "color:var(--rinch-color-dimmed);",
                    "Grid"
                }
            };
            container.append_child(&grid_div);
        }

        // F1 shortcut reference overlay.
        // TODO: add show_shortcuts signal to UiSignals.
        if false {
            let overlay = rsx! {
                div {
                    style: "position:fixed;bottom:30px;right:310px;z-index:100;\
                         background:var(--rinch-color-dark-9);border:1px solid var(--rinch-color-border);\
                         border-radius:6px;padding:12px 16px;font-size:11px;\
                         color:var(--rinch-color-text);line-height:1.8;white-space:pre;",
                    "Keyboard Shortcuts  (F1 to close)\n\
                     \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\n\
                     G/R/L      Grab / Rotate / scaLe\n\
                     B / N      Sculpt / Paint mode\n\
                     G          Toggle grid\n\
                     F3         Cycle debug mode\n\
                     Delete     Delete selected\n\
                     Ctrl+D     Duplicate selected\n\
                     Ctrl+S     Save scene\n\
                     Ctrl+O     Open scene\n\
                     RMB+WASD   Fly camera\n\
                     Scroll     Zoom (orbit)\n\
                     MMB        Pan\n\
                     F1         This reference"
                }
            };
            container.append_child(&overlay);
        }

        container
    });

    root
}
