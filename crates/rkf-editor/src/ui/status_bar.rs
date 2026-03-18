//! Status bar component — object count, FPS, selection info, mode display.
//!
//! All content is built in a single rsx! call. Each section uses its own
//! reactive closure (`{move || ...}`) so only that section re-renders when
//! its specific signals change. Conditional visibility uses reactive style
//! closures (`display:none`) instead of rsx! `if` blocks, avoiding DOM
//! teardown/rebuild.

use rinch::prelude::*;

use crate::editor_state::{SelectedEntity, UiSignals};

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
///
/// Each section reads only the signals it needs via fine-grained reactive
/// closures, so e.g. FPS updates don't cause the selection label to rebuild.
#[component]
pub fn StatusBar() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            style: "display:flex;align-items:center;height:25px;\
                background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
                padding:0 12px;gap:16px;\
                font-size:11px;color:var(--rinch-color-dimmed);",

            div {
                style: "display:flex;align-items:center;width:100%;gap:16px;",

                // Object count — reads only ui.objects.
                div { {move || {
                    let objects = ui.objects.get();
                    format!("{} objects", objects.len())
                }} }

                // FPS — reads only ui.fps.
                div { {move || {
                    let ms = ui.fps.get();
                    if ms > 0.1 {
                        format!("{:.0} fps", 1000.0 / ms)
                    } else {
                        "-- fps".to_string()
                    }
                }} }

                // Diagnostics indicator — reads ui.diagnostics.
                // Shows error/warning count in red. Hidden when no diagnostics.
                div {
                    style: {move || {
                        let diags = ui.diagnostics.get();
                        if diags.is_empty() {
                            "display:none;".to_string()
                        } else {
                            "color:var(--rinch-color-red-5, #ff6b6b);".to_string()
                        }
                    }},
                    {move || {
                        let diags = ui.diagnostics.get();
                        let errors = diags.iter()
                            .filter(|d| d.severity == crate::ui_snapshot::DiagnosticSeverity::Error)
                            .count();
                        let warnings = diags.iter()
                            .filter(|d| d.severity == crate::ui_snapshot::DiagnosticSeverity::Warning)
                            .count();
                        let mut parts = Vec::new();
                        if errors > 0 {
                            parts.push(format!("{errors} err"));
                        }
                        if warnings > 0 {
                            parts.push(format!("{warnings} warn"));
                        }
                        parts.join(" ")
                    }}
                }

                // Selected entity name — reads ui.selection, ui.objects, ui.lights.
                // Hidden via display:none when nothing is selected.
                div {
                    style: {move || {
                        let selection = ui.selection.get();
                        if selection.is_some() {
                            "color:var(--rinch-primary-color);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    {move || {
                        let selection = ui.selection.get();
                        let objects = ui.objects.get();
                        let lights = ui.lights.get();
                        selection.as_ref().map(|sel| match sel {
                            SelectedEntity::Object(eid) => {
                                objects.iter()
                                    .find(|o| o.id == *eid)
                                    .map(|o| o.name.clone())
                                    .unwrap_or_else(|| format!("Object {}", &eid.to_string()[..8]))
                            }
                            SelectedEntity::Light(lid) => {
                                lights.iter()
                                    .find(|l| l.id == *lid)
                                    .map(|l| match l.light_type {
                                        crate::light_editor::SceneLightType::Point => format!("Point Light {lid}"),
                                        crate::light_editor::SceneLightType::Spot => format!("Spot Light {lid}"),
                                    })
                                    .unwrap_or_else(|| format!("Light {lid}"))
                            }
                            SelectedEntity::Scene => "Scene".to_string(),
                            SelectedEntity::Project => "Project".to_string(),
                        }).unwrap_or_default()
                    }}
                }

                // Debug mode indicator — reads only ui.debug_mode.
                // Hidden via display:none when debug mode is off (mode 0 = normal).
                div {
                    style: {move || {
                        let debug_mode = ui.debug_mode.get();
                        let debug_name = crate::ui_snapshot::debug_mode_name(debug_mode);
                        if !debug_name.is_empty() {
                            "color:var(--rinch-color-yellow-5, #fcc419);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    {move || {
                        let debug_mode = ui.debug_mode.get();
                        crate::ui_snapshot::debug_mode_name(debug_mode).to_string()
                    }}
                }

                // Spacer.
                div { style: "flex:1;", }

                // Gizmo mode indicator — reads only ui.gizmo_mode.
                div {
                    style: "color:var(--rinch-color-dimmed);",
                    {move || {
                        let gizmo_mode = ui.gizmo_mode.get();
                        format!("{gizmo_mode:?}")
                    }}
                }

                // Active tool mode — reads only ui.editor_mode.
                // Hidden via display:none when no tool mode is active.
                div {
                    style: {move || {
                        let editor_mode = ui.editor_mode.get();
                        let mode_name = editor_mode.name();
                        if !mode_name.is_empty() {
                            "color:var(--rinch-primary-color);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    {move || {
                        let editor_mode = ui.editor_mode.get();
                        let mode_name = editor_mode.name();
                        if !mode_name.is_empty() {
                            format!("{mode_name} mode")
                        } else {
                            String::new()
                        }
                    }}
                }

                // Grid indicator — reads only ui.show_grid.
                // Hidden via display:none when grid is off.
                div {
                    style: {move || {
                        if ui.show_grid.get() {
                            "color:var(--rinch-color-dimmed);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    "Grid"
                }
            }
        }
    }
}
