//! Asset browser panel — bottom-center panel showing material/shader cards.
//!
//! Displays a tabbed view with "Materials" and "Shaders" tabs. The Materials
//! tab shows a horizontal grid of material swatches. The Shaders tab shows
//! registered shader cards. Double-clicking a shader opens its `.wgsl` file.
//!
//! Uses fine-grained reactivity:
//! - Tab buttons have reactive style closures for active state
//! - Count badge uses a reactive text closure
//! - Tab content uses reactive `display:none` wrappers
//! - Material and shader grids use `for` in rsx with reactive selection borders

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

use super::PANEL_BG;

/// Height of the asset browser panel in pixels.
const BROWSER_HEIGHT: &str = "height:180px;";

/// Double-click detection threshold in milliseconds.
const DOUBLE_CLICK_MS: u128 = 400;

/// Asset browser panel component.
///
/// Shows a tabbed panel with Materials and Shaders grids.
/// Each sub-section uses fine-grained reactivity so that changing selection,
/// switching tabs, or updating material/shader lists only rebuilds the
/// affected DOM nodes.
#[component]
pub fn AssetBrowser() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let root_style = format!(
        "{BROWSER_HEIGHT}{PANEL_BG}\
         border-top:1px solid var(--rinch-color-border);\
         display:flex;flex-direction:column;flex-shrink:0;min-height:0;"
    );

    rsx! {
        div { style: {root_style.as_str()},
            div { style: "display:flex;flex-direction:column;height:100%;",
                // ── Tab bar ──
                div {
                    style: "display:flex;align-items:center;gap:0;flex-shrink:0;\
                            border-bottom:1px solid var(--rinch-color-border);",

                    // Materials tab button — reactive style for active state.
                    div {
                        style: {
                            let ui = ui;
                            move || {
                                let is_active = ui.asset_browser_tab.get() == 0;
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
                                format!(
                                    "padding:4px 12px;font-size:11px;text-transform:uppercase;\
                                     letter-spacing:1px;cursor:pointer;{border_bottom}{color}"
                                )
                            }
                        },
                        onclick: { let ui = ui; move || ui.asset_browser_tab.set(0) },
                        "Materials"
                    }

                    // Shaders tab button — reactive style for active state.
                    div {
                        style: {
                            let ui = ui;
                            move || {
                                let is_active = ui.asset_browser_tab.get() == 1;
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
                                format!(
                                    "padding:4px 12px;font-size:11px;text-transform:uppercase;\
                                     letter-spacing:1px;cursor:pointer;{border_bottom}{color}"
                                )
                            }
                        },
                        onclick: { let ui = ui; move || ui.asset_browser_tab.set(1) },
                        "Shaders"
                    }

                    // Count badge — reactive text.
                    span {
                        style: "font-size:10px;color:var(--rinch-color-placeholder);\
                                font-family:var(--rinch-font-family-monospace);margin-left:auto;padding-right:12px;",
                        {move || {
                            if ui.asset_browser_tab.get() == 0 {
                                format!("{} slots", ui.materials.get().len())
                            } else {
                                format!("{} shaders", ui.shaders.get().len())
                            }
                        }}
                    }
                }

                // ── Materials grid (hidden when shaders tab active) ──
                MaterialsGrid {}

                // ── Shaders grid (hidden when materials tab active) ──
                ShadersGrid {}
            }
        }
    }
}

/// Materials grid with reactive visibility and keyed for-loop.
#[component]
fn MaterialsGrid() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();

    rsx! {
        div {
            style: {
                let ui = ui;
                move || {
                    if ui.asset_browser_tab.get() == 0 {
                        "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                         overflow-y:auto;flex:1;min-height:0;align-content:flex-start;"
                    } else {
                        "display:none;"
                    }
                }
            },

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
                            // Bump generation so the viewport's MouseUp handler
                            // (which rinch dispatches immediately after ondragend)
                            // knows to suppress the GPU pick.
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
                            // Trigger material preview update.
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

/// Shaders grid with reactive visibility and keyed for-loop.
#[component]
fn ShadersGrid() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    rsx! {
        div {
            style: {
                let ui = ui;
                move || {
                    if ui.asset_browser_tab.get() == 1 {
                        "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                         overflow-y:auto;flex:1;min-height:0;align-content:flex-start;"
                    } else {
                        "display:none;"
                    }
                }
            },

            for shader in ui.shaders.get() {
                BrowserShaderCard {
                    key: shader.name.clone(),
                    shader_name: shader.name.clone(),
                    file_path: shader.file_path.clone(),
                }
            }
        }
    }
}

/// Individual shader card for the asset browser.
#[component]
fn BrowserShaderCard(
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
