//! Materials panel — shows material library as a grid of swatches.
//!
//! Displays material cards with albedo color swatches, metallic/emissive indicators,
//! and slot numbers. Supports click-to-select and drag-to-assign.
//!
//! Uses `for` in rsx for keyed reconciliation — the grid only rebuilds when the
//! materials list changes. Selection border updates are handled by reactive style
//! closures on each card, avoiding a full grid rebuild on every selection change.
//!
//! ## Store migration status
//!
//! The materials panel reads `ui.materials` (Vec<MaterialSummary>) which is a
//! complex typed collection. This remains on UiSignals because `UiValue` doesn't
//! support typed lists. The material count is mirrored to `editor/material_count`
//! in the store for read-only use by other widgets. Individual material property
//! editing uses `material:{slot}/{field}` store paths (see routing.rs).

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

    rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;",
            div { style: "display:flex;flex-direction:column;height:100%;",

                // Count badge — reactive text tracks materials list length.
                div {
                    style: "font-size:10px;color:var(--rinch-color-placeholder);\
                            font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                            border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
                    {move || format!("{} slots", ui.materials.get().len())}
                }

                // Grid — keyed for loop rebuilds only when materials list changes.
                div {
                    style: "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                            overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",

                    for mat in ui.materials.get() {
                        div {
                            key: format!("{}", mat.slot),
                            style: {
                                let ui = ui;
                                let slot = mat.slot;
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
                            ondragstart: { let ui = ui; let slot = mat.slot; move || ui.material_drag.set(slot) },
                            ondragend: {
                                let ui = ui;
                                move || {
                                    ui.material_drag.clear();
                                    ui.material_drop_highlight.set(None);
                                    ui.drag_drop_generation.update(|g| *g += 1);
                                }
                            },
                            onclick: {
                                let cmd = cmd.clone();
                                let ui = ui;
                                let shared_state = shared_state.clone();
                                let slot = mat.slot;
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

                            // Swatch with indicators overlay.
                            div {
                                style: {
                                    let r = (mat.albedo[0] * 255.0).round() as u8;
                                    let g = (mat.albedo[1] * 255.0).round() as u8;
                                    let b = (mat.albedo[2] * 255.0).round() as u8;
                                    let mut s = format!("width:100%;height:32px;background:rgb({r},{g},{b});");
                                    if mat.emission_strength > 0.01 {
                                        let er = (mat.emission_color[0] * 255.0).round() as u8;
                                        let eg = (mat.emission_color[1] * 255.0).round() as u8;
                                        let eb = (mat.emission_color[2] * 255.0).round() as u8;
                                        s.push_str(&format!("box-shadow:inset 0 0 8px rgb({er},{eg},{eb});"));
                                    }
                                    s
                                },

                                // Indicator dots.
                                div { style: "display:flex;gap:2px;padding:1px 2px;height:10px;align-items:center;",
                                    if mat.metallic > 0.5 {
                                        div {
                                            style: "width:6px;height:6px;border-radius:50%;\
                                                    background:#a0a0c0;border:1px solid #8080a0;",
                                        }
                                    }
                                    if mat.emission_strength > 0.01 {
                                        div {
                                            style: "width:6px;height:6px;border-radius:50%;\
                                                    background:#ffcc44;border:1px solid #cc9900;",
                                        }
                                    }
                                }
                            }

                            // Info: name + slot number.
                            div { style: "padding:2px 3px;display:flex;flex-direction:column;gap:1px;",
                                div {
                                    style: "font-size:9px;color:var(--rinch-color-text);\
                                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                                    {mat.name.clone()}
                                }
                                div {
                                    style: "font-size:8px;color:var(--rinch-color-placeholder);\
                                            font-family:var(--rinch-font-family-monospace);",
                                    {format!("#{}", mat.slot)}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
