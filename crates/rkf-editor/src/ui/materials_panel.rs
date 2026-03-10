//! Materials panel — shows material library as a grid of swatches.
//!
//! Displays material cards with albedo color swatches, metallic/emissive indicators,
//! and slot numbers. Supports click-to-select and drag-to-assign.
//!
//! Uses `for_each_dom_typed` so the grid only rebuilds when the materials list
//! changes. Selection border updates are handled by reactive style closures on
//! each card, avoiding a full grid rebuild on every selection change.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

/// Materials panel component.
///
/// Shows a horizontal grid of material swatches from the engine's material table.
#[component]
pub fn MaterialsPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();

    let root = rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;" }
    };

    // Container: count badge + grid.
    let container = rsx! {
        div { style: "display:flex;flex-direction:column;height:100%;" }
    };

    // Count badge — reactive text tracks materials list length.
    let count_div = rsx! {
        div {
            style: "font-size:10px;color:var(--rinch-color-placeholder);\
                    font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                    border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
            {move || format!("{} slots", ui.materials.get().len())}
        }
    };
    container.append_child(&count_div);

    // Grid — for_each_dom_typed rebuilds only when materials list changes.
    let grid = rsx! {
        div {
            style: "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                    overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",
        }
    };

    let cmd2 = cmd.clone();
    rinch::core::for_each_dom_typed(
        __scope,
        &grid,
        move || ui.materials.get(),
        |mat| format!("{}", mat.slot),
        move |mat, __scope| {
            let slot = mat.slot;

            // Build swatch style from material properties (static per-item).
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

            // Build indicator dots.
            let indicators = rsx! {
                div { style: "display:flex;gap:2px;padding:1px 2px;height:10px;align-items:center;" }
            };
            if mat.metallic > 0.5 {
                indicators.append_child(&rsx! {
                    div {
                        style: "width:6px;height:6px;border-radius:50%;\
                                background:#a0a0c0;border:1px solid #8080a0;",
                    }
                });
            }
            if mat.emission_strength > 0.01 {
                indicators.append_child(&rsx! {
                    div {
                        style: "width:6px;height:6px;border-radius:50%;\
                                background:#ffcc44;border:1px solid #cc9900;",
                    }
                });
            }

            // Build swatch with indicators overlay.
            let swatch = rsx! {
                div { style: {swatch_style.as_str()} }
            };
            swatch.append_child(&indicators);

            // Card — reactive style closure for selection border.
            let card = rsx! {
                div {
                    style: {
                        let ui = ui;
                        move || {
                            let is_selected = ui.selected_material.get() == Some(slot);
                            let border_color = if is_selected {
                                "var(--rinch-primary-color)"
                            } else {
                                "var(--rinch-color-border)"
                            };
                            format!(
                                "width:64px;display:flex;flex-direction:column;\
                                 border:1px solid {border_color};border-radius:4px;\
                                 background:var(--rinch-color-dark-7);cursor:pointer;\
                                 overflow:hidden;flex-shrink:0;"
                            )
                        }
                    },
                    draggable: "true",
                    ondragstart: { let ui = ui; move || ui.material_drag.set(slot) },
                    ondragend: {
                        let ui = ui;
                        move || {
                            ui.material_drag.clear();
                            ui.material_drop_highlight.set(None);
                            ui.drag_drop_generation.update(|g| *g += 1);
                        }
                    },
                    onclick: {
                        let cmd = cmd2.clone();
                        let ui = ui;
                        let shared_state = shared_state.clone();
                        move || {
                            ui.selected_material.set(Some(slot));
                            ui.selected_shader.set(None);
                            ui.properties_tab.set(1);
                            let _ = cmd.0.send(EditorCommand::SelectMaterial { slot });
                            if let Ok(mut ss) = shared_state.lock() {
                                ss.preview_material_slot = Some(slot);
                                ss.preview_dirty = true;
                            }
                        }
                    },
                }
            };

            // Append swatch (with indicators), then info.
            card.append_child(&swatch);
            let mat_name = mat.name.clone();
            let info = rsx! {
                div { style: "padding:2px 3px;display:flex;flex-direction:column;gap:1px;",
                    div {
                        style: "font-size:9px;color:var(--rinch-color-text);\
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                        {mat_name}
                    }
                    div {
                        style: "font-size:8px;color:var(--rinch-color-placeholder);\
                                font-family:var(--rinch-font-family-monospace);",
                        {format!("#{slot}")}
                    }
                }
            };
            card.append_child(&info);

            card
        },
    );

    container.append_child(&grid);
    root.append_child(&container);
    root
}
