//! Status bar component — object count, FPS, selection info, mode display.
//!
//! Most sections read from the UI Store for simple scalar/string values.
//! Selection name lookup and diagnostics remain on UiSignals because they
//! require complex typed data (Vec<ObjectSummary>, DiagnosticEntry enums).

use rinch::prelude::*;

use crate::editor_state::{SelectedEntity, UiSignals};
use crate::store::UiStore;

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
///
/// Each section reads only the signals it needs via fine-grained reactive
/// closures, so e.g. FPS updates don't cause the selection label to rebuild.
#[component]
pub fn StatusBar() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let store = use_context::<UiStore>();

    let object_count_signal = store.read("editor/object_count");
    let fps_signal = store.read("editor/fps");
    let debug_mode_signal = store.read("editor/debug_mode");
    let gizmo_mode_signal = store.read("gizmo/mode");
    let editor_mode_signal = store.read("editor/mode");
    let show_grid_signal = store.read("editor/show_grid");

    rsx! {
        div {
            style: "display:flex;align-items:center;height:25px;\
                background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
                padding:0 12px;gap:16px;\
                font-size:11px;color:var(--rinch-color-dimmed);",

            div {
                style: "display:flex;align-items:center;width:100%;gap:16px;",

                // Object count — reads from store.
                div { {move || {
                    let count = object_count_signal.get().as_int().unwrap_or(0);
                    format!("{count} objects")
                }} }

                // FPS — reads from store.
                div { {move || {
                    let ms = fps_signal.get().as_float().unwrap_or(0.0);
                    if ms > 0.1 {
                        format!("{:.0} fps", 1000.0 / ms)
                    } else {
                        "-- fps".to_string()
                    }
                }} }

                // Diagnostics indicator — reads ui.diagnostics (complex typed data).
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
                // Stays on UiSignals: requires Vec<ObjectSummary>/Vec<LightSummary> lookup.
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

                // Debug mode indicator — reads from store.
                // Hidden via display:none when debug mode is off (mode 0 = normal).
                div {
                    style: {move || {
                        let mode = debug_mode_signal.get().as_int().unwrap_or(0) as u32;
                        let name = crate::ui_snapshot::debug_mode_name(mode);
                        if !name.is_empty() {
                            "color:var(--rinch-color-yellow-5, #fcc419);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    {move || {
                        let mode = debug_mode_signal.get().as_int().unwrap_or(0) as u32;
                        crate::ui_snapshot::debug_mode_name(mode).to_string()
                    }}
                }

                // Spacer.
                div { style: "flex:1;", }

                // Gizmo mode indicator — reads from store.
                div {
                    style: "color:var(--rinch-color-dimmed);",
                    {move || {
                        let mode = gizmo_mode_signal.get();
                        match mode.as_string().unwrap_or_default() {
                            "translate" => "Translate",
                            "rotate" => "Rotate",
                            "scale" => "Scale",
                            _ => "Translate",
                        }.to_string()
                    }}
                }

                // Active tool mode — reads from store.
                // Hidden via display:none when no tool mode is active (default mode).
                div {
                    style: {move || {
                        let mode = editor_mode_signal.get();
                        let name = mode.as_string().unwrap_or_default();
                        if name == "sculpt" || name == "paint" {
                            "color:var(--rinch-primary-color);".to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }},
                    {move || {
                        let mode = editor_mode_signal.get();
                        match mode.as_string().unwrap_or_default() {
                            "sculpt" => "Sculpt mode".to_string(),
                            "paint" => "Paint mode".to_string(),
                            _ => String::new(),
                        }
                    }}
                }

                // Grid indicator — reads from store.
                // Hidden via display:none when grid is off.
                div {
                    style: {move || {
                        if show_grid_signal.get().as_bool().unwrap_or(false) {
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
