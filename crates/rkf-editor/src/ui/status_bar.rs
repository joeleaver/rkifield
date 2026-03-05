//! Status bar component — object count, FPS, selection info, mode display.

use rinch::prelude::*;

use crate::editor_state::{SelectedEntity, UiSignals};
use crate::SnapshotReader;

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
///
/// Uses a lightweight reactive_component_dom for layout changes (selection
/// presence, debug mode, grid toggle) with an independent FPS Effect to
/// avoid rebuilding on every frame time update.
#[component]
pub fn StatusBar() -> NodeHandle {
    let snap = use_context::<SnapshotReader>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "display:flex;align-items:center;height:25px;\
        background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
        padding:0 12px;gap:16px;\
        font-size:11px;color:var(--rinch-color-dimmed);",
    );

    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking — only rebuild when these specific values change.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        let _ = ui.gizmo_mode.get();
        let _ = ui.debug_mode.get();
        let _ = ui.show_grid.get();
        let _ = ui.scene_revision.get();

        // Read from lock-free snapshot — no mutex.
        let guard = snap.0.load();

        let obj_count = guard.objects.len();
        let mode_name = guard.mode.name().to_string();
        let selected_name = guard.selected_entity.as_ref().map(|sel| match sel {
            SelectedEntity::Object(eid) => {
                guard.objects.iter()
                    .find(|o| o.id == *eid)
                    .map(|o| o.name.clone())
                    .unwrap_or_else(|| format!("Object {eid}"))
            }
            SelectedEntity::Light(lid) => {
                guard.lights.iter()
                    .find(|l| l.id == *lid)
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
        let debug_name = guard.debug_mode_name().to_string();
        let gizmo_mode_name = guard.gizmo_mode_name().to_string();

        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display:flex;align-items:center;width:100%;gap:16px;",
        );

        // Object count.
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
        if guard.show_grid {
            let grid_div = __scope.create_element("div");
            grid_div.set_attribute("style", "color:var(--rinch-color-dimmed);");
            grid_div.append_child(&__scope.create_text("Grid"));
            container.append_child(&grid_div);
        }

        // F1 shortcut reference overlay.
        if guard.show_shortcuts {
            let overlay = __scope.create_element("div");
            overlay.set_attribute("style",
                "position:fixed;bottom:30px;right:310px;z-index:100;\
                 background:var(--rinch-color-dark-9);border:1px solid var(--rinch-color-border);\
                 border-radius:6px;padding:12px 16px;font-size:11px;\
                 color:var(--rinch-color-text);line-height:1.8;white-space:pre;",
            );
            overlay.append_child(&__scope.create_text(
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
                 F1         This reference",
            ));
            container.append_child(&overlay);
        }

        container
    });

    root
}
