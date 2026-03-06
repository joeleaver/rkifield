//! Shaders panel — shows registered shaders as a card grid.
//!
//! Displays shader cards with built-in/custom indicators. Click to select,
//! double-click to open the `.wgsl` source file.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use rinch::prelude::*;

use crate::editor_state::UiSignals;
use crate::SnapshotReader;

/// Double-click detection threshold in milliseconds.
const DOUBLE_CLICK_MS: u128 = 400;

/// Shaders panel component.
///
/// Shows a horizontal grid of shader cards from the engine's shader registry.
#[component]
pub fn ShadersPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();

    let last_shader_click: Rc<RefCell<(String, Instant)>> =
        Rc::new(RefCell::new((String::new(), Instant::now() - std::time::Duration::from_secs(10))));

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;min-height:0;display:flex;flex-direction:column;",
    );

    let snap = snapshot.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        let _ = ui.selected_shader.get();

        let snap_guard = snap.0.load();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;height:100%;");

        // ── Count badge ──
        let count_bar = __scope.create_element("div");
        count_bar.set_attribute(
            "style",
            "font-size:10px;color:var(--rinch-color-placeholder);\
             font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
             border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
        );
        count_bar.append_child(
            &__scope.create_text(&format!("{} shaders", snap_guard.shaders.len())),
        );
        container.append_child(&count_bar);

        // ── Shaders grid ──
        let grid = __scope.create_element("div");
        grid.set_attribute(
            "style",
            "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
             overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",
        );

        let selected_shader = ui.selected_shader.get();

        for shader in &snap_guard.shaders {
            let is_selected = selected_shader.as_deref() == Some(&shader.name);

            let card = __scope.create_element("div");
            let border_color = if is_selected {
                "var(--rinch-primary-color)"
            } else {
                "var(--rinch-color-border)"
            };
            card.set_attribute(
                "style",
                &format!(
                    "width:64px;display:flex;flex-direction:column;\
                     border:1px solid {border_color};border-radius:4px;\
                     background:var(--rinch-color-dark-7);cursor:pointer;\
                     overflow:hidden;flex-shrink:0;"
                ),
            );

            // Icon area.
            let icon = __scope.create_element("div");
            let (bg_color, icon_text) = if shader.built_in {
                ("rgba(60,120,200,0.3)", "B")
            } else {
                ("rgba(60,180,100,0.3)", "C")
            };
            icon.set_attribute(
                "style",
                &format!(
                    "width:100%;height:32px;background:{bg_color};\
                     display:flex;align-items:center;justify-content:center;\
                     font-size:14px;color:var(--rinch-color-dimmed);font-weight:600;"
                ),
            );
            icon.append_child(&__scope.create_text(icon_text));
            card.append_child(&icon);

            // Name + type label.
            let info = __scope.create_element("div");
            info.set_attribute(
                "style",
                "padding:2px 3px;display:flex;flex-direction:column;gap:1px;",
            );
            let name_el = __scope.create_element("div");
            name_el.set_attribute(
                "style",
                "font-size:9px;color:var(--rinch-color-text);\
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
            );
            name_el.append_child(&__scope.create_text(&shader.name));
            info.append_child(&name_el);

            let type_el = __scope.create_element("div");
            type_el.set_attribute(
                "style",
                "font-size:8px;color:var(--rinch-color-placeholder);\
                 font-family:var(--rinch-font-family-monospace);",
            );
            let type_label = if shader.built_in { "built-in" } else { "custom" };
            type_el.append_child(&__scope.create_text(type_label));
            info.append_child(&type_el);
            card.append_child(&info);

            // Make card draggable.
            card.set_attribute("draggable", "true");

            let shader_name = shader.name.clone();
            let file_path = shader.file_path.clone();

            let drag_start_hid = __scope.register_handler({
                let ui = ui;
                let shader_name = shader_name.clone();
                move || {
                    ui.shader_drag.set(shader_name.clone());
                }
            });
            card.set_attribute("data-ondragstart", &drag_start_hid.to_string());

            let drag_end_hid = __scope.register_handler({
                let ui = ui;
                move || {
                    ui.shader_drag.clear();
                    ui.shader_drop_highlight.set(false);
                    ui.drag_drop_generation.update(|g| *g += 1);
                }
            });
            card.set_attribute("data-ondragend", &drag_end_hid.to_string());

            // Click handler — select shader, double-click opens file.
            let hid = __scope.register_handler({
                let ui = ui;
                let last_click = last_shader_click.clone();
                let shader_name = shader_name.clone();
                let file_path = file_path.clone();
                move || {
                    let now = Instant::now();
                    let mut prev = last_click.borrow_mut();
                    let is_double = prev.0 == shader_name
                        && now.duration_since(prev.1).as_millis() < DOUBLE_CLICK_MS;

                    if is_double {
                        if !file_path.is_empty() {
                            let _ = std::process::Command::new("xdg-open")
                                .arg(&file_path)
                                .spawn();
                        }
                        *prev = (String::new(), now);
                    } else {
                        ui.selected_shader.set(Some(shader_name.clone()));
                        ui.selected_material.set(None);
                        ui.properties_tab.set(1);
                        *prev = (shader_name.clone(), now);
                    }
                }
            });
            card.set_attribute("data-rid", &hid.to_string());

            grid.append_child(&card);
        }

        container.append_child(&grid);
        container
    });

    root
}
