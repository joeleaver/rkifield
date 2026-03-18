//! Shaders panel — shows registered shaders as a card grid.
//!
//! Displays shader cards with built-in/custom indicators. Click to select,
//! double-click to open the `.wgsl` source file.
//!
//! Uses `for` in rsx for keyed reconciliation — the grid only rebuilds when the
//! shaders list changes. Selection border updates are handled by reactive style
//! closures on each card, avoiding a full grid rebuild on every selection change.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use rinch::prelude::*;

use crate::editor_state::UiSignals;

/// Double-click detection threshold in milliseconds.
const DOUBLE_CLICK_MS: u128 = 400;

/// Shaders panel component.
///
/// Shows a horizontal grid of shader cards from the engine's shader registry.
#[component]
pub fn ShadersPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;",
            div { style: "display:flex;flex-direction:column;height:100%;",

                // Count badge — reactive text tracks shaders list length.
                div {
                    style: "font-size:10px;color:var(--rinch-color-placeholder);\
                            font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                            border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
                    {move || format!("{} shaders", ui.shaders.get().len())}
                }

                // Grid — keyed for loop rebuilds only when shaders list changes.
                div {
                    style: "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                            overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",

                    for shader in ui.shaders.get() {
                        ShaderCard {
                            key: shader.name.clone(),
                            shader_name: shader.name.clone(),
                            file_path: shader.file_path.clone(),
                        }
                    }
                }
            }
        }
    }
}

/// Individual shader card component.
#[component]
fn ShaderCard(
    shader_name: String,
    file_path: String,
) -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let last_click: Rc<RefCell<(String, Instant)>> =
        Rc::new(RefCell::new((String::new(), Instant::now() - std::time::Duration::from_secs(10))));

    let sn1 = shader_name.clone();
    let sn2 = shader_name.clone();
    let sn3 = shader_name.clone();
    let sn4 = shader_name;
    let fp1 = file_path;

    rsx! {
        div {
            style: || {
                let is_selected = ui.selected_shader.get().as_deref() == Some(sn1.as_str());
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
            },
            draggable: "true",
            ondragstart: {
                let ui = ui;
                move || {
                    ui.shader_drag.set(sn2.clone());
                }
            },
            ondragend: {
                let ui = ui;
                move || {
                    ui.shader_drag.clear();
                    ui.shader_drop_highlight.set(false);
                    ui.drag_drop_generation.update(|g| *g += 1);
                }
            },
            onclick: {
                let ui = ui;
                move || {
                    let now = Instant::now();
                    let mut prev = last_click.borrow_mut();
                    let is_double = prev.0 == sn3
                        && now.duration_since(prev.1).as_millis() < DOUBLE_CLICK_MS;

                    if is_double {
                        if !fp1.is_empty() {
                            let _ = std::process::Command::new("xdg-open")
                                .arg(&fp1)
                                .spawn();
                        }
                        *prev = (String::new(), now);
                    } else {
                        ui.selected_shader.set(Some(sn3.clone()));
                        ui.selected_material.set(None);
                        ui.properties_tab.set(1);
                        *prev = (sn3.clone(), now);
                    }
                }
            },

            // Icon area.
            div {
                style: "width:100%;height:32px;background:rgba(60,180,100,0.3);\
                        display:flex;align-items:center;justify-content:center;\
                        font-size:14px;color:var(--rinch-color-dimmed);font-weight:600;",
                "S"
            }

            // Name label.
            div { style: "padding:2px 3px;",
                div {
                    style: "font-size:9px;color:var(--rinch-color-text);\
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                    {sn4}
                }
            }
        }
    }
}
