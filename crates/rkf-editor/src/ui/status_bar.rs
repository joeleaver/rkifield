//! Status bar component — object count, FPS, selection info, mode display.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_state::{EditorState, SelectedEntity, UiRevision, UiSignals};

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
#[component]
pub fn StatusBar() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let _shared_state = use_context::<Arc<Mutex<SharedState>>>();
    let revision = use_context::<UiRevision>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        &format!(
            "display:flex;align-items:center;height:25px;\
            background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
            padding:0 12px;gap:16px;\
            font-size:11px;color:var(--rinch-color-dimmed);"
        ),
    );

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking — only rebuild when these specific values change.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        let _ = ui.gizmo_mode.get();
        let _ = ui.debug_mode.get();
        let _ = ui.show_grid.get();
        let _ = ui.object_count.get();
        // Legacy fallback.
        revision.track();

        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display:flex;align-items:center;width:100%;gap:16px;",
        );

        let (obj_count, mode_name, selected_name, debug_name, gizmo_mode_name) = {
            let es = es.lock().unwrap();
            let sel_name = es.selected_entity.as_ref().map(|sel| match sel {
                SelectedEntity::Object(eid) => {
                    es.world.scene().objects.iter()
                        .find(|o| o.id as u64 == *eid)
                        .map(|o| o.name.clone())
                        .unwrap_or_else(|| format!("Object {eid}"))
                }
                SelectedEntity::Light(lid) => {
                    es.light_editor.get_light(*lid)
                        .map(|l| match l.light_type {
                            crate::light_editor::EditorLightType::Point => format!("Point Light {lid}"),
                            crate::light_editor::EditorLightType::Spot => format!("Spot Light {lid}"),
                        })
                        .unwrap_or_else(|| format!("Light {lid}"))
                }
                SelectedEntity::Camera => "Camera".to_string(),
                SelectedEntity::Scene => "Scene".to_string(),
                SelectedEntity::Project => "Project".to_string(),
            });
            let gizmo_name = match es.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => "Translate (W)",
                crate::gizmo::GizmoMode::Rotate => "Rotate (E)",
                crate::gizmo::GizmoMode::Scale => "Scale (R)",
            };
            (
                es.world.scene().objects.len(),
                es.mode.name().to_string(),
                sel_name,
                es.debug_mode_name().to_string(),
                gizmo_name.to_string(),
            )
        };
        let obj_div = __scope.create_element("div");
        obj_div.append_child(&__scope.create_text(&format!("{obj_count} objects")));
        container.append_child(&obj_div);

        // FPS: own reactive text node tracking UiSignals.fps (not revision).
        let fps_div = __scope.create_element("div");
        let fps_text = __scope.create_text("-- fps");
        fps_div.append_child(&fps_text);
        container.append_child(&fps_div);
        {
            let fps_text = fps_text.clone();
            Effect::new(move || {
                let ms = ui.fps.get();
                let label = if ms > 0.1 {
                    format!("{:.0} fps", 1000.0 / ms)
                } else {
                    "-- fps".to_string()
                };
                fps_text.set_text(&label);
            });
        }

        // Selected object name.
        if let Some(name) = &selected_name {
            let sel_div = __scope.create_element("div");
            sel_div.set_attribute("style", "color:var(--rinch-primary-color);");
            sel_div.append_child(&__scope.create_text(name));
            container.append_child(&sel_div);
        }

        // Debug mode indicator.
        if !debug_name.is_empty() {
            let dbg_div = __scope.create_element("div");
            dbg_div.set_attribute("style", "color:var(--rinch-color-yellow-5, #fcc419);");
            dbg_div.append_child(&__scope.create_text(&debug_name));
            container.append_child(&dbg_div);
        }

        let spacer = __scope.create_element("div");
        spacer.set_attribute("style", "flex:1;");
        container.append_child(&spacer);

        // Gizmo mode indicator.
        {
            let gizmo_div = __scope.create_element("div");
            gizmo_div.set_attribute("style", "color:var(--rinch-color-dimmed);");
            gizmo_div.append_child(&__scope.create_text(&gizmo_mode_name));
            container.append_child(&gizmo_div);
        }

        // Show active tool mode (only when Sculpt/Paint active).
        if !mode_name.is_empty() {
            let mode_div = __scope.create_element("div");
            mode_div.set_attribute("style", "color:var(--rinch-primary-color);");
            mode_div.append_child(&__scope.create_text(&format!("{mode_name} mode")));
            container.append_child(&mode_div);
        }

        // Grid indicator.
        {
            let show_grid = es.lock().map(|e| e.show_grid).unwrap_or(false);
            if show_grid {
                let grid_div = __scope.create_element("div");
                grid_div.set_attribute("style", "color:var(--rinch-color-dimmed);");
                grid_div.append_child(&__scope.create_text("Grid"));
                container.append_child(&grid_div);
            }
        }

        // F1 shortcut reference overlay.
        {
            let show_shortcuts = es.lock().map(|e| e.show_shortcuts).unwrap_or(false);
            if show_shortcuts {
                let overlay = __scope.create_element("div");
                overlay.set_attribute("style",
                    "position:fixed;bottom:30px;right:310px;z-index:100;\
                     background:var(--rinch-color-dark-9);border:1px solid var(--rinch-color-border);\
                     border-radius:6px;padding:12px 16px;font-size:11px;\
                     color:var(--rinch-color-text);line-height:1.8;white-space:pre;",
                );
                overlay.append_child(&__scope.create_text(
                    "Keyboard Shortcuts  (F1 to close)\n\
                     ─────────────────────────────────\n\
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
                     F1         This reference",
                ));
                container.append_child(&overlay);
            }
        }

        container
    });

    root
}
