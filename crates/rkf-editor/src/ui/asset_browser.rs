//! Asset browser panel — bottom-center panel showing material/shader cards.
//!
//! Displays a tabbed view with "Materials" and "Shaders" tabs. The Materials
//! tab shows a horizontal grid of material swatches. The Shaders tab shows
//! registered shader cards. Double-clicking a shader opens its `.wgsl` file.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::{CommandSender, SnapshotReader};

use super::PANEL_BG;

/// Height of the asset browser panel in pixels.
const BROWSER_HEIGHT: &str = "height:180px;";

/// Double-click detection threshold in milliseconds.
const DOUBLE_CLICK_MS: u128 = 400;

/// Asset browser panel component.
///
/// Shows a tabbed panel with Materials and Shaders grids.
#[component]
pub fn AssetBrowser() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();

    // Double-click tracking for shader cards: (shader_name, last_click_time).
    let last_shader_click: Rc<RefCell<(String, Instant)>> =
        Rc::new(RefCell::new((String::new(), Instant::now() - std::time::Duration::from_secs(10))));

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        &format!(
            "{BROWSER_HEIGHT}{PANEL_BG}\
             border-top:1px solid var(--rinch-color-border);\
             display:flex;flex-direction:column;flex-shrink:0;min-height:0;"
        ),
    );

    let snap = snapshot.clone();
    let cmd2 = cmd.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        let _ = ui.material_revision.get();
        let _ = ui.selected_material.get();
        let _ = ui.editor_mode.get();
        let active_tab = ui.asset_browser_tab.get();
        let _ = ui.selected_shader.get();

        let snap_guard = snap.0.load();
        let selected_slot = ui.selected_material.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;height:100%;");

        // ── Tab bar ──
        let tab_bar = __scope.create_element("div");
        tab_bar.set_attribute(
            "style",
            "display:flex;align-items:center;gap:0;flex-shrink:0;\
             border-bottom:1px solid var(--rinch-color-border);",
        );

        let tabs = [("Materials", 0u32), ("Shaders", 1u32)];
        for (label, idx) in &tabs {
            let tab = __scope.create_element("div");
            let is_active = active_tab == *idx;
            let border_bottom = if is_active {
                "border-bottom:2px solid var(--rinch-primary-color);"
            } else {
                "border-bottom:2px solid transparent;"
            };
            let color = if is_active {
                "color:var(--rinch-color-text);"
            } else {
                "color:var(--rinch-color-dimmed);"
            };
            tab.set_attribute(
                "style",
                &format!(
                    "padding:4px 12px;font-size:11px;text-transform:uppercase;\
                     letter-spacing:1px;cursor:pointer;{border_bottom}{color}"
                ),
            );
            tab.append_child(&__scope.create_text(label));

            let tab_idx = *idx;
            let hid = __scope.register_handler({
                let ui = ui;
                move || {
                    ui.asset_browser_tab.set(tab_idx);
                }
            });
            tab.set_attribute("data-rid", &hid.to_string());
            tab_bar.append_child(&tab);
        }

        // Count badge on right side of tab bar.
        let count = __scope.create_element("span");
        count.set_attribute(
            "style",
            "font-size:10px;color:var(--rinch-color-placeholder);\
             font-family:var(--rinch-font-family-monospace);margin-left:auto;padding-right:12px;",
        );
        if active_tab == 0 {
            count.append_child(
                &__scope.create_text(&format!("{} slots", snap_guard.materials.len())),
            );
        } else {
            count.append_child(
                &__scope.create_text(&format!("{} shaders", snap_guard.shaders.len())),
            );
        }
        tab_bar.append_child(&count);
        container.append_child(&tab_bar);

        if active_tab == 0 {
            // ── Materials grid ──
            let grid = __scope.create_element("div");
            grid.set_attribute(
                "style",
                "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                 overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",
            );

            for mat in &snap_guard.materials {
                let slot = mat.slot;
                let is_selected = selected_slot == Some(slot);

                // Card container.
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

                // Color swatch (albedo).
                let swatch = __scope.create_element("div");
                let r = (mat.albedo[0] * 255.0).round() as u8;
                let g = (mat.albedo[1] * 255.0).round() as u8;
                let b = (mat.albedo[2] * 255.0).round() as u8;
                let mut swatch_style = format!(
                    "width:100%;height:32px;background:rgb({r},{g},{b});"
                );
                // Emissive glow overlay.
                if mat.emission_strength > 0.01 {
                    let er = (mat.emission_color[0] * 255.0).round() as u8;
                    let eg = (mat.emission_color[1] * 255.0).round() as u8;
                    let eb = (mat.emission_color[2] * 255.0).round() as u8;
                    swatch_style.push_str(&format!(
                        "box-shadow:inset 0 0 8px rgb({er},{eg},{eb});"
                    ));
                }
                swatch.set_attribute("style", &swatch_style);

                // Indicator row (metallic/emissive icons).
                let indicators = __scope.create_element("div");
                indicators.set_attribute(
                    "style",
                    "display:flex;gap:2px;padding:1px 2px;height:10px;align-items:center;",
                );
                if mat.metallic > 0.5 {
                    let m_dot = __scope.create_element("div");
                    m_dot.set_attribute(
                        "style",
                        "width:6px;height:6px;border-radius:50%;\
                         background:#a0a0c0;border:1px solid #8080a0;",
                    );
                    indicators.append_child(&m_dot);
                }
                if mat.emission_strength > 0.01 {
                    let e_dot = __scope.create_element("div");
                    e_dot.set_attribute(
                        "style",
                        "width:6px;height:6px;border-radius:50%;\
                         background:#ffcc44;border:1px solid #cc9900;",
                    );
                    indicators.append_child(&e_dot);
                }
                swatch.append_child(&indicators);
                card.append_child(&swatch);

                // Name + slot label.
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
                name_el.append_child(&__scope.create_text(&mat.name));
                info.append_child(&name_el);

                let slot_el = __scope.create_element("div");
                slot_el.set_attribute(
                    "style",
                    "font-size:8px;color:var(--rinch-color-placeholder);\
                     font-family:var(--rinch-font-family-monospace);",
                );
                slot_el.append_child(&__scope.create_text(&format!("#{slot}")));
                info.append_child(&slot_el);
                card.append_child(&info);

                // Make material card draggable for drag-and-drop to object material rows.
                card.set_attribute("draggable", "true");

                let drag_start_hid = __scope.register_handler({
                    let ui = ui;
                    move || {
                        ui.material_drag.set(slot);
                    }
                });
                card.set_attribute("data-ondragstart", &drag_start_hid.to_string());

                let drag_end_hid = __scope.register_handler({
                    let ui = ui;
                    move || {
                        ui.material_drag.clear();
                        ui.material_drop_highlight.set(None);
                        // Bump generation so the viewport's MouseUp handler
                        // (which rinch dispatches immediately after ondragend)
                        // knows to suppress the GPU pick.
                        ui.drag_drop_generation.update(|g| *g += 1);
                    }
                });
                card.set_attribute("data-ondragend", &drag_end_hid.to_string());

                // Click handler — select material (and in Paint mode, set paint material).
                let hid = __scope.register_handler({
                    let cmd = cmd2.clone();
                    let ui = ui;
                    move || {
                        ui.selected_material.set(Some(slot));
                        ui.selected_shader.set(None); // clear shader selection
                        ui.properties_tab.set(1); // switch to Asset tab
                        let _ = cmd.0.send(EditorCommand::SelectMaterial { slot });
                    }
                });
                card.set_attribute("data-rid", &hid.to_string());

                grid.append_child(&card);
            }

            container.append_child(&grid);
        } else {
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

                // Icon area — colored block indicating built-in vs custom.
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

                // Make card draggable for drag-and-drop to material shader slot.
                card.set_attribute("draggable", "true");

                let shader_name = shader.name.clone();
                let file_path = shader.file_path.clone();

                // ondragstart: set drag context data.
                let drag_start_hid = __scope.register_handler({
                    let ui = ui;
                    let shader_name = shader_name.clone();
                    move || {
                        ui.shader_drag.set(shader_name.clone());
                    }
                });
                card.set_attribute("data-ondragstart", &drag_start_hid.to_string());

                // ondragend: clear drag context.
                let drag_end_hid = __scope.register_handler({
                    let ui = ui;
                    move || {
                        ui.shader_drag.clear();
                        ui.shader_drop_highlight.set(false);
                        ui.drag_drop_generation.update(|g| *g += 1);
                    }
                });
                card.set_attribute("data-ondragend", &drag_end_hid.to_string());

                // Click handler — select shader (also serves as fallback when
                // mousedown→mouseup without crossing drag threshold).
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
                            // Double-click: open shader file.
                            if !file_path.is_empty() {
                                let _ = std::process::Command::new("xdg-open")
                                    .arg(&file_path)
                                    .spawn();
                            }
                            *prev = (String::new(), now);
                        } else {
                            // Single click: select shader.
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
        }

        container
    });

    root
}
