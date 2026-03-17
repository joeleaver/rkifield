//! Shaders panel — shows registered shaders as a card grid.
//!
//! Displays shader cards with built-in/custom indicators. Click to select,
//! double-click to open the `.wgsl` source file.
//!
//! Uses `for_each_dom_typed` so the grid only rebuilds when the shaders list
//! changes. Selection border updates are handled by reactive style closures on
//! each card, avoiding a full grid rebuild on every selection change.

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

    let last_shader_click: Rc<RefCell<(String, Instant)>> =
        Rc::new(RefCell::new((String::new(), Instant::now() - std::time::Duration::from_secs(10))));

    let root = rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;" }
    };

    // Container: count badge + grid.
    let container = rsx! {
        div { style: "display:flex;flex-direction:column;height:100%;" }
    };

    // Count badge — reactive text tracks shaders list length.
    let count_div = rsx! {
        div {
            style: "font-size:10px;color:var(--rinch-color-placeholder);\
                    font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                    border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
            {move || format!("{} shaders", ui.shaders.get().len())}
        }
    };
    container.append_child(&count_div);

    // Grid — for_each_dom_typed rebuilds only when shaders list changes.
    let grid = rsx! {
        div {
            style: "display:flex;flex-wrap:wrap;gap:6px;padding:4px 12px;\
                    overflow-y:auto;flex:1;min-height:0;align-content:flex-start;",
        }
    };

    rinch::core::for_each_dom_typed(
        __scope,
        &grid,
        move || ui.shaders.get(),
        |shader| shader.name.clone(),
        move |shader, __scope| {
            let shader_name_for_style = shader.name.clone();
            let shader_name = shader.name.clone();
            let file_path = shader.file_path.clone();

            let icon_style = format!(
                "width:100%;height:32px;background:rgba(60,180,100,0.3);\
                 display:flex;align-items:center;justify-content:center;\
                 font-size:14px;color:var(--rinch-color-dimmed);font-weight:600;"
            );

            // Card — reactive style closure for selection border.
            let card = rsx! {
                div {
                    style: {
                        let ui = ui;
                        let shader_name = shader_name_for_style.clone();
                        move || {
                            let is_selected = ui.selected_shader.get().as_deref() == Some(shader_name.as_str());
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
                    ondragstart: {
                        let ui = ui;
                        let shader_name = shader_name.clone();
                        move || {
                            ui.shader_drag.set(shader_name.clone());
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
                    },

                    // Icon area.
                    div { style: {icon_style.as_str()}, "S" }

                    // Name label.
                    div { style: "padding:2px 3px;",
                        div {
                            style: "font-size:9px;color:var(--rinch-color-text);\
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                            {shader_name.clone()}
                        }
                    }
                }
            };

            card
        },
    );

    container.append_child(&grid);
    root.append_child(&container);
    root
}
