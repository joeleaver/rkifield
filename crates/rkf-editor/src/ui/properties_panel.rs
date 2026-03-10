//! Properties panel — shows details of the currently selected entity.
//!
//! Tracks `UiSignals::selection` for fine-grained reactivity — only rebuilds
//! when the selection changes (from scene tree clicks or viewport picks).

use rinch::prelude::*;

use crate::editor_state::{SelectedEntity, UiSignals};

// ── Style constants ──────────────────────────────────────────────────────────

const LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

const SECTION_STYLE: &str = "font-size:12px;color:var(--rinch-color-text);padding:6px 12px;";

const VALUE_STYLE: &str = "font-size:12px;color:var(--rinch-color-dimmed);padding:2px 12px;\
    font-family:var(--rinch-font-family-monospace);";

const NO_SELECTION_STYLE: &str = "font-size:12px;color:var(--rinch-color-placeholder);padding:6px 12px;";

// ── Component ────────────────────────────────────────────────────────────────

/// Properties panel component.
///
/// Displays the name and basic info about the currently selected entity.
/// Re-renders reactively when the `UiSignals::selection` signal changes.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    // Reactive content — rebuilds when selection changes.
    // Keyed by selection state so the view fully rebuilds on selection change.
    rinch::core::for_each_dom_typed(
        __scope,
        &root,
        move || vec![ui.selection.get()],
        |sel| match &sel {
            None => "none".to_string(),
            Some(SelectedEntity::Camera) => "camera".to_string(),
            Some(SelectedEntity::Object(eid)) => format!("obj-{eid}"),
            Some(SelectedEntity::Light(lid)) => format!("light-{lid}"),
            Some(SelectedEntity::Scene) => "scene".to_string(),
            Some(SelectedEntity::Project) => "project".to_string(),
        },
        move |sel, __scope| {
            match sel {
                Some(SelectedEntity::Object(entity_id)) => {
                    let objects = ui.objects.get();
                    if let Some(obj) = objects.iter().find(|o| o.id == entity_id) {
                        let name = obj.name.clone();
                        let id_text = format!("Entity ID: {entity_id}");
                        let child_count = objects.iter()
                            .filter(|o| o.parent_id.map(|p| p as u64) == Some(entity_id))
                            .count();

                        rsx! {
                            div { style: "display:flex;flex-direction:column;",
                                div { style: {LABEL_STYLE}, "Properties" }
                                div { style: {SECTION_STYLE}, {name} }
                                div { style: {VALUE_STYLE}, {id_text} }
                                if child_count > 0 {
                                    div { style: {VALUE_STYLE},
                                        {format!("Children: {child_count}")}
                                    }
                                }
                            }
                        }
                    } else {
                        rsx! {
                            div { style: "display:flex;flex-direction:column;",
                                div { style: {LABEL_STYLE}, "Properties" }
                            }
                        }
                    }
                }
                Some(SelectedEntity::Light(lid)) => {
                    let lights = ui.lights.get();
                    if let Some(light) = lights.iter().find(|l| l.id == lid) {
                        let type_name = match light.light_type {
                            crate::light_editor::SceneLightType::Point => "Point Light",
                            crate::light_editor::SceneLightType::Spot => "Spot Light",
                        };
                        let detail = format!(
                            "Intensity: {:.2}  Range: {:.1}",
                            light.intensity, light.range
                        );

                        rsx! {
                            div { style: "display:flex;flex-direction:column;",
                                div { style: {LABEL_STYLE}, "Properties" }
                                div { style: {SECTION_STYLE}, {type_name} }
                                div { style: {VALUE_STYLE}, {detail} }
                            }
                        }
                    } else {
                        rsx! {
                            div { style: "display:flex;flex-direction:column;",
                                div { style: {LABEL_STYLE}, "Properties" }
                            }
                        }
                    }
                }
                Some(SelectedEntity::Camera) => {
                    let pos = ui.camera_display_pos.get();
                    let pos_text = format!(
                        "Pos: ({:.1}, {:.1}, {:.1})",
                        pos.x, pos.y, pos.z
                    );

                    rsx! {
                        div { style: "display:flex;flex-direction:column;",
                            div { style: {LABEL_STYLE}, "Properties" }
                            div { style: {SECTION_STYLE}, "Camera" }
                            div { style: {VALUE_STYLE}, {pos_text} }
                        }
                    }
                }
                Some(_) => {
                    rsx! {
                        div { style: "display:flex;flex-direction:column;",
                            div { style: {LABEL_STYLE}, "Properties" }
                            div { style: {NO_SELECTION_STYLE}, "No properties" }
                        }
                    }
                }
                None => {
                    rsx! {
                        div { style: "display:flex;flex-direction:column;",
                            div { style: {LABEL_STYLE}, "Properties" }
                            div { style: {NO_SELECTION_STYLE}, "No object selected" }
                        }
                    }
                }
            }
        },
    );

    root
}
