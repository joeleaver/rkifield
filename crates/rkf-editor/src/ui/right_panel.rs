//! Right panel — composes property components with fine-grained reactivity.
//!
//! Uses rsx! with reactive if/match for conditional rendering. Each section
//! (camera, environment, object, light, material) is its own component that
//! tracks only its relevant signals.

use rinch::prelude::*;
use rinch::render_surface::RenderSurface;

use crate::editor_state::{SelectedEntity, UiSignals};
use crate::PreviewSurfaceHandle;

use super::component_inspector::ComponentInspector;
use super::light_properties::LightProperties;
use super::material_properties::MaterialProperties;
use super::object_properties::ObjectProperties;
use super::shader_properties::ShaderProperties;
use super::{SECTION_STYLE, VALUE_STYLE};

// ── Panel components for zone-based layout ──────────────────────────────────

/// Object Properties panel — dispatches to entity-specific components.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    rsx! {
        div { style: "display:flex;flex-direction:column;min-height:0;flex:1;overflow-y:auto;",
            EntityContent {}
        }
    }
}

/// Asset Properties panel — shows material or shader properties,
/// with a material preview surface at the top.
#[component]
pub fn AssetPropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let preview_handle = use_context::<PreviewSurfaceHandle>();

    rsx! {
        div { style: "display:flex;flex-direction:column;min-height:0;flex:1;overflow-y:auto;",
            // Material preview — visible when a material is selected.
            RenderSurface { surface: Some(preview_handle.0.clone()),
                style: {
                    let ui = ui;
                    move || {
                        if ui.selected_material.get().is_some() {
                            "flex-shrink:0;width:200px;height:200px;\
                             align-self:center;\
                             border-radius:4px;\
                             border:1px solid var(--rinch-color-border);\
                             overflow:hidden;margin:8px auto;"
                        } else {
                            "width:0;height:0;overflow:hidden;"
                        }
                    }
                },
            }
            AssetContent {}
        }
    }
}

// ── Shared components ──────────────────────────────────────────────────────

/// Tab bar with Object/Asset tabs and reactive active-tab styling.
#[component]
fn TabBar() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            style: "display:flex;gap:0;border-bottom:1px solid var(--rinch-color-border);\
                     margin-bottom:4px;flex-shrink:0;",

            // Object tab
            div {
                style: {
                    let ui = ui;
                    move || {
                        let is_active = ui.properties_tab.get() == 0;
                        let bg = if is_active { "var(--rinch-color-dark-7)" } else { "transparent" };
                        let color = if is_active { "var(--rinch-color-text)" } else { "var(--rinch-color-dimmed)" };
                        let border_bottom = if is_active { "2px solid var(--rinch-primary-color)" } else { "2px solid transparent" };
                        format!(
                            "flex:1;text-align:center;padding:4px 0;font-size:11px;\
                             cursor:pointer;background:{bg};color:{color};\
                             border-bottom:{border_bottom};text-transform:uppercase;\
                             letter-spacing:0.5px;"
                        )
                    }
                },
                onclick: { let ui = ui; move || ui.properties_tab.set(0) },
                "Object"
            }

            // Asset tab
            div {
                style: {
                    let ui = ui;
                    move || {
                        let is_active = ui.properties_tab.get() == 1;
                        let bg = if is_active { "var(--rinch-color-dark-7)" } else { "transparent" };
                        let color = if is_active { "var(--rinch-color-text)" } else { "var(--rinch-color-dimmed)" };
                        let border_bottom = if is_active { "2px solid var(--rinch-primary-color)" } else { "2px solid transparent" };
                        format!(
                            "flex:1;text-align:center;padding:4px 0;font-size:11px;\
                             cursor:pointer;background:{bg};color:{color};\
                             border-bottom:{border_bottom};text-transform:uppercase;\
                             letter-spacing:0.5px;"
                        )
                    }
                },
                onclick: { let ui = ui; move || ui.properties_tab.set(1) },
                "Asset"
            }
        }
    }
}

/// Entity-specific content based on the current selection.
///
/// Uses a `for` loop with a single-element vec keyed by a selection
/// fingerprint. When the selection changes (different entity, different
/// variant), the key changes, the old subtree is torn down, and a fresh
/// one is built — guaranteeing components see the current entity.
#[component]
#[allow(unused_variables)]
fn EntityContent() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            for sel_key in vec![selection_key(ui.selection.get())] {
                div {
                    key: sel_key.clone(),
                    SelectionPanel {}
                }
            }
        }
    }
}

/// Produce a unique string key for the current selection.
fn selection_key(sel: Option<SelectedEntity>) -> String {
    match sel {
        Some(SelectedEntity::Object(id)) => format!("obj:{id}"),
        Some(SelectedEntity::Light(id)) => format!("light:{id}"),
        Some(SelectedEntity::Scene) => "scene".to_string(),
        Some(SelectedEntity::Project) => "project".to_string(),
        None => "none".to_string(),
    }
}

/// Inner panel that reads `ui.selection` once at render time.
///
/// Because the parent tears down and rebuilds this component whenever the
/// selection key changes, `ui.selection.get()` here always returns the
/// current value — no stale captures.
#[component]
fn SelectionPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    match ui.selection.get() {
        Some(SelectedEntity::Object(_)) => {
            rsx! {
                div {
                    ObjectProperties {}
                    ComponentInspector {}
                }
            }
        }
        Some(SelectedEntity::Light(_)) => {
            rsx! { div { LightProperties {} } }
        }
        Some(SelectedEntity::Scene) => {
            rsx! {
                div {
                    div { style: {SECTION_STYLE},
                        {move || ui.scene_name.get()}
                    }
                    div { style: {VALUE_STYLE},
                        {move || format!("{} objects", ui.objects.get().len())}
                    }
                }
            }
        }
        Some(SelectedEntity::Project) => {
            rsx! { div { div { style: {SECTION_STYLE}, "Project" } } }
        }
        None => {
            rsx! {
                div {
                    div {
                        style: {format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);")},
                        "No object selected"
                    }
                }
            }
        }
    }
}

/// Asset-specific content (material or shader properties).
#[component]
/// Compute a fingerprint for the current asset selection (keyed rebuild).
fn asset_selection_key(mat: Option<u16>, shader: Option<String>) -> String {
    if let Some(slot) = mat {
        format!("mat:{slot}")
    } else if let Some(ref name) = shader {
        format!("shader:{name}")
    } else {
        "none".to_string()
    }
}

#[component]
#[allow(unused_variables)]
fn AssetContent() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            for key in vec![{
                let mat = ui.selected_material.get();
                let shader = ui.selected_shader.get();
                if let Some(slot) = mat { format!("mat:{slot}") }
                else if let Some(ref name) = shader { format!("shader:{name}") }
                else { "none".to_string() }
            }] {
                div {
                    key: key,
                    if let Some(slot) = ui.selected_material.get() {
                        MaterialProperties { slot: slot }
                    } else if let Some(shader_name) = ui.selected_shader.get() {
                        ShaderProperties { shader_name: shader_name }
                    } else {
                        div {
                            style: {format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);")},
                            "No asset selected"
                        }
                    }
                }
            }
        }
    }
}
