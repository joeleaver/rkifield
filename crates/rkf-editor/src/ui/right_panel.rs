//! Right panel — composes property components with fine-grained reactivity.
//!
//! Uses rsx! with reactive if/match for conditional rendering. Each section
//! (camera, environment, object, light, material) is its own component that
//! tracks only its relevant signals.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;
use rinch::render_surface::RenderSurface;

use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiSignals};
use crate::PreviewSurfaceHandle;
use crate::CommandSender;

use super::camera_properties::CameraProperties;
use super::environment_panel::EnvironmentPanel;
use super::slider_helpers::build_slider_row;
use super::light_properties::LightProperties;
use super::material_properties::MaterialProperties;
use super::object_properties::ObjectProperties;
use super::shader_properties::ShaderProperties;
use super::{LABEL_STYLE, SECTION_STYLE, VALUE_STYLE};

// ── Right panel (legacy single-panel layout) ────────────────────────────────

/// Right panel — shows properties + environment in a single scrollable column.
///
/// For the zone-based layout, use `PropertiesPanel` and `AssetPropertiesPanel`
/// instead (they are independent panels that can be placed in separate zones).
#[component]
pub fn RightPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let preview_handle = use_context::<PreviewSurfaceHandle>();

    rsx! {
        div { style: "flex:1;overflow-y:scroll;min-height:0;height:0;display:flex;flex-direction:column;",

            // ── Material preview surface ──
            RenderSurface { surface: Some(preview_handle.0.clone()),
                style: {
                    let ui = ui;
                    move || {
                        let show = ui.properties_tab.get() == 1
                            && ui.selected_material.get().is_some();
                        if show {
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

            // ── Tool-specific settings (Sculpt/Paint) ──
            if matches!(ui.editor_mode.get(), EditorMode::Sculpt) {
                SculptPanel {}
                div { style: {super::DIVIDER_STYLE} }
            }
            if matches!(ui.editor_mode.get(), EditorMode::Paint) {
                PaintPanel {}
                div { style: {super::DIVIDER_STYLE} }
            }

            // ── Tab bar ──
            {build_tab_bar(__scope, ui)}

            // ── Tab content ──
            if ui.properties_tab.get() == 0 {
                // Object tab
                {build_entity_content(__scope, ui, &editor_state)}
            }
            if ui.properties_tab.get() == 1 {
                // Asset tab
                {build_asset_content(__scope, ui)}
            }
        }
    }
}

// ── Standalone panel components for zone-based layout ───────────────────────

/// Object Properties panel — dispatches to entity-specific components.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            if matches!(ui.editor_mode.get(), EditorMode::Sculpt) {
                SculptPanel {}
                div { style: {super::DIVIDER_STYLE} }
            }
            if matches!(ui.editor_mode.get(), EditorMode::Paint) {
                PaintPanel {}
                div { style: {super::DIVIDER_STYLE} }
            }
            {build_entity_content(__scope, ui, &editor_state)}
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
        div { style: "display:flex;flex-direction:column;",
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
            {build_asset_content(__scope, ui)}
        }
    }
}

/// Sculpt settings panel — brush type, radius, strength, falloff.
#[component]
pub fn SculptPanel() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "padding:4px 0;");

    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Sculpt Brush"));
    root.append_child(&header);

    // Show brush type from editor state.
    if let Ok(es_lock) = editor_state.lock() {
        let type_name = match es_lock.sculpt.current_settings.brush_type {
            crate::sculpt::BrushType::Add => "Add",
            crate::sculpt::BrushType::Subtract => "Subtract",
            crate::sculpt::BrushType::Smooth => "Smooth",
            crate::sculpt::BrushType::Flatten => "Flatten",
            crate::sculpt::BrushType::Sharpen => "Sharpen",
        };
        let row = __scope.create_element("div");
        row.set_attribute("style", VALUE_STYLE);
        row.append_child(&__scope.create_text(&format!("Type: {type_name}")));
        root.append_child(&row);
    }

    build_slider_row(
        __scope, &root, "Radius", "",
        sliders.brush_radius, 0.01, 10.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Strength", "",
        sliders.brush_strength, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Falloff", "",
        sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );

    root.into()
}

/// Paint settings panel — material, radius, strength, falloff.
#[component]
pub fn PaintPanel() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "padding:4px 0;");

    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Paint Brush"));
    root.append_child(&header);

    build_slider_row(
        __scope, &root, "Radius", "",
        sliders.brush_radius, 0.01, 10.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Strength", "",
        sliders.brush_strength, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Falloff", "",
        sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );

    root.into()
}

// ── Shared helpers ──────────────────────────────────────────────────────────

/// Build the Object/Asset tab bar with reactive active-tab styling.
fn build_tab_bar(scope: &mut RenderScope, ui: UiSignals) -> NodeHandle {
    let tab_bar = scope.create_element("div");
    tab_bar.set_attribute(
        "style",
        "display:flex;gap:0;border-bottom:1px solid var(--rinch-color-border);\
         margin-bottom:4px;flex-shrink:0;",
    );
    for (idx, label) in [(0u32, "Object"), (1u32, "Asset")] {
        let tab = scope.create_element("div");
        // Reactive style — updates when properties_tab changes without rebuild.
        {
            let tab_clone = tab.clone();
            scope.create_effect(move || {
                let is_active = ui.properties_tab.get() == idx;
                let bg = if is_active { "var(--rinch-color-dark-7)" } else { "transparent" };
                let color = if is_active { "var(--rinch-color-text)" } else { "var(--rinch-color-dimmed)" };
                let border_bottom = if is_active { "2px solid var(--rinch-primary-color)" } else { "2px solid transparent" };
                tab_clone.set_attribute(
                    "style",
                    &format!(
                        "flex:1;text-align:center;padding:4px 0;font-size:11px;\
                         cursor:pointer;background:{bg};color:{color};\
                         border-bottom:{border_bottom};text-transform:uppercase;\
                         letter-spacing:0.5px;"
                    ),
                );
            });
        }
        tab.append_child(&scope.create_text(label));
        let hid = scope.register_handler(move || { ui.properties_tab.set(idx); });
        tab.set_attribute("data-rid", &hid.to_string());
        tab_bar.append_child(&tab);
    }
    tab_bar
}

/// Build entity-specific content based on the current selection.
fn build_entity_content(
    scope: &mut RenderScope,
    ui: UiSignals,
    _es: &Arc<Mutex<EditorState>>,
) -> NodeHandle {
    let container = scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    rinch::core::for_each_dom_typed(
        scope, &container,
        move || vec![ui.selection.get()],
        |sel| match &sel {
            None => "none".to_string(),
            Some(SelectedEntity::Camera) => "camera".to_string(),
            Some(SelectedEntity::Object(eid)) => format!("obj-{eid}"),
            Some(SelectedEntity::Light(lid)) => format!("light-{lid}"),
            Some(SelectedEntity::Scene) => "scene".to_string(),
            Some(SelectedEntity::Project) => "project".to_string(),
        },
        move |selected_entity, scope| {
            let inner = scope.create_element("div");

            match selected_entity {
                Some(SelectedEntity::Camera) => {
                    let camera = CameraProperties::default();
                    inner.append_child(&camera.render(scope, &[]));
                    let div = scope.create_element("div");
                    div.set_attribute("style", super::DIVIDER_STYLE);
                    inner.append_child(&div);
                    let env = EnvironmentPanel::default();
                    inner.append_child(&env.render(scope, &[]));
                }
                Some(SelectedEntity::Object(eid)) => {
                    let obj = ObjectProperties { entity_id: eid, ..Default::default() };
                    inner.append_child(&obj.render(scope, &[]));
                }
                Some(SelectedEntity::Light(lid)) => {
                    let light = LightProperties { light_id: lid, ..Default::default() };
                    inner.append_child(&light.render(scope, &[]));
                }
                Some(SelectedEntity::Scene) => {
                    let scene_name = ui.scene_name.get();
                    let name_row = scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    name_row.append_child(&scope.create_text(&scene_name));
                    inner.append_child(&name_row);

                    let object_count = ui.objects.get().len();
                    let detail = scope.create_element("div");
                    detail.set_attribute("style", VALUE_STYLE);
                    detail.append_child(
                        &scope.create_text(&format!("{} objects", object_count)),
                    );
                    inner.append_child(&detail);
                }
                Some(SelectedEntity::Project) => {
                    let hdr = scope.create_element("div");
                    hdr.set_attribute("style", SECTION_STYLE);
                    hdr.append_child(&scope.create_text("Project"));
                    inner.append_child(&hdr);
                }
                None => {
                    let msg = scope.create_element("div");
                    msg.set_attribute(
                        "style",
                        &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                    );
                    msg.append_child(&scope.create_text("No object selected"));
                    inner.append_child(&msg);
                }
            }

            inner
        },
    );

    container
}

/// Build asset-specific content (material or shader properties).
fn build_asset_content(
    scope: &mut RenderScope,
    ui: UiSignals,
) -> NodeHandle {
    let container = scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    rinch::core::for_each_dom_typed(
        scope, &container,
        move || {
            let mat = ui.selected_material.get();
            let shader = ui.selected_shader.get();
            vec![(mat, shader)]
        },
        |(mat, shader)| {
            match (mat, shader) {
                (Some(slot), _) => format!("mat-{slot}"),
                (None, Some(name)) => format!("shader-{name}"),
                (None, None) => "none".to_string(),
            }
        },
        move |(selected_mat_slot, selected_shader_name), scope| {
            let inner = scope.create_element("div");

            if let Some(slot) = selected_mat_slot {
                let mat = MaterialProperties { slot, ..Default::default() };
                inner.append_child(&mat.render(scope, &[]));
            } else if let Some(ref shader_name) = selected_shader_name {
                let shaders = ui.shaders.get();
                if shaders.iter().any(|s| &s.name == shader_name) {
                    let shader = ShaderProperties {
                        shader_name: shader_name.clone(),
                        ..Default::default()
                    };
                    inner.append_child(&shader.render(scope, &[]));
                }
            } else {
                let msg = scope.create_element("div");
                msg.set_attribute(
                    "style",
                    &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                );
                msg.append_child(&scope.create_text("No asset selected"));
                inner.append_child(&msg);
            }

            inner
        },
    );

    container
}
