//! Materials panel — shows material library as a grid of swatches.
//!
//! Displays material cards with albedo color swatches, metallic/emissive indicators,
//! and slot numbers. Supports click-to-select and drag-to-assign.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::{CommandSender, SnapshotReader};

/// Materials panel component.
///
/// Shows a horizontal grid of material swatches from the engine's material table.
#[component]
pub fn MaterialsPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;min-height:0;display:flex;flex-direction:column;",
    );

    let snap = snapshot.clone();
    let cmd2 = cmd.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        let _ = ui.material_revision.get();
        let _ = ui.selected_material.get();

        let snap_guard = snap.0.load();
        let selected_slot = ui.selected_material.get();

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
            &__scope.create_text(&format!("{} slots", snap_guard.materials.len())),
        );
        container.append_child(&count_bar);

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

            // Make material card draggable.
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
                    ui.drag_drop_generation.update(|g| *g += 1);
                }
            });
            card.set_attribute("data-ondragend", &drag_end_hid.to_string());

            // Click handler — select material.
            let hid = __scope.register_handler({
                let cmd = cmd2.clone();
                let ui = ui;
                move || {
                    ui.selected_material.set(Some(slot));
                    ui.selected_shader.set(None);
                    ui.properties_tab.set(1);
                    let _ = cmd.0.send(EditorCommand::SelectMaterial { slot });
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
