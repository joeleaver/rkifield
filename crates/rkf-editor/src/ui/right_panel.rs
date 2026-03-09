//! Right panel — composes property components with fine-grained reactivity.
//!
//! Each section (camera, environment, object, light, material) is its own
//! component that tracks only its relevant signals. The panels here dispatch
//! to those components based on selection and editor mode.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiSignals};
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

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:scroll;min-height:0;height:0;",
    );

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Only track signals that affect WHICH components to show.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        let _ = ui.properties_tab.get();
        let _ = ui.selected_material.get();
        let _ = ui.selected_shader.get();

        let mode = ui.editor_mode.get();
        let selected_entity = ui.selection.get();
        let active_tab = ui.properties_tab.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // ── Tool-specific settings (when Sculpt/Paint active) ──
        match mode {
            EditorMode::Sculpt => {
                let panel = SculptPanel::default();
                container.append_child(&panel.render(__scope, &[]));
                append_divider(__scope, &container);
            }
            EditorMode::Paint => {
                let panel = PaintPanel::default();
                container.append_child(&panel.render(__scope, &[]));
                append_divider(__scope, &container);
            }
            EditorMode::Default => {}
        }

        // ── Tab bar ──
        let tab_bar = build_tab_bar(__scope, ui, active_tab);
        container.append_child(&tab_bar);

        if active_tab == 0 {
            // ── Object tab ──
            build_entity_content(__scope, &container, selected_entity, &ui, &es);
        } else {
            // ── Asset tab ──
            build_asset_content(__scope, &container, &ui);
        }

        container
    });

    root
}

// ── Standalone panel components for zone-based layout ───────────────────────

/// Object Properties panel — dispatches to entity-specific components.
///
/// Each entity type (camera, object, light) is rendered by its own component
/// with fine-grained reactivity — toggling fog doesn't rebuild object properties.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex-direction:column;");

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Only track selection and editor mode — NOT toggles.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();

        let mode = ui.editor_mode.get();
        let selected_entity = ui.selection.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // ── Tool-specific settings ──
        match mode {
            EditorMode::Sculpt => {
                let panel = SculptPanel::default();
                container.append_child(&panel.render(__scope, &[]));
                append_divider(__scope, &container);
            }
            EditorMode::Paint => {
                let panel = PaintPanel::default();
                container.append_child(&panel.render(__scope, &[]));
                append_divider(__scope, &container);
            }
            EditorMode::Default => {}
        }

        // ── Entity properties ──
        build_entity_content(__scope, &container, selected_entity, &ui, &es);

        container
    });

    root
}

/// Asset Properties panel — shows material or shader properties.
#[component]
pub fn AssetPropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex-direction:column;");

    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        let _ = ui.selected_material.get();
        let _ = ui.selected_shader.get();
        let _ = ui.materials.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        build_asset_content(__scope, &container, &ui);

        container
    });

    root
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

fn append_divider(scope: &mut RenderScope, container: &NodeHandle) {
    let div = scope.create_element("div");
    div.set_attribute("style", super::DIVIDER_STYLE);
    container.append_child(&div);
}

/// Build the Object/Asset tab bar.
fn build_tab_bar(scope: &mut RenderScope, ui: UiSignals, active_tab: u32) -> NodeHandle {
    let tab_bar = scope.create_element("div");
    tab_bar.set_attribute(
        "style",
        "display:flex;gap:0;border-bottom:1px solid var(--rinch-color-border);\
         margin-bottom:4px;flex-shrink:0;",
    );
    for (idx, label) in [(0u32, "Object"), (1u32, "Asset")] {
        let tab = scope.create_element("div");
        let is_active = active_tab == idx;
        let bg = if is_active { "var(--rinch-color-dark-7)" } else { "transparent" };
        let color = if is_active { "var(--rinch-color-text)" } else { "var(--rinch-color-dimmed)" };
        let border_bottom = if is_active { "2px solid var(--rinch-primary-color)" } else { "2px solid transparent" };
        tab.set_attribute(
            "style",
            &format!(
                "flex:1;text-align:center;padding:4px 0;font-size:11px;\
                 cursor:pointer;background:{bg};color:{color};\
                 border-bottom:{border_bottom};text-transform:uppercase;\
                 letter-spacing:0.5px;"
            ),
        );
        tab.append_child(&scope.create_text(label));
        let hid = scope.register_handler(move || { ui.properties_tab.set(idx); });
        tab.set_attribute("data-rid", &hid.to_string());
        tab_bar.append_child(&tab);
    }
    tab_bar
}

/// Build entity-specific content based on the current selection.
///
/// Dispatches to dedicated property components — each component manages
/// its own fine-grained reactivity internally.
fn build_entity_content(
    scope: &mut RenderScope,
    container: &NodeHandle,
    selected_entity: Option<SelectedEntity>,
    ui: &UiSignals,
    _es: &Arc<Mutex<EditorState>>,
) {
    match selected_entity {
        Some(SelectedEntity::Camera) => {
            let camera = CameraProperties::default();
            container.append_child(&camera.render(scope, &[]));
            append_divider(scope, container);
            let env = EnvironmentPanel::default();
            container.append_child(&env.render(scope, &[]));
        }
        Some(SelectedEntity::Object(eid)) => {
            let obj = ObjectProperties { entity_id: eid, ..Default::default() };
            container.append_child(&obj.render(scope, &[]));
        }
        Some(SelectedEntity::Light(lid)) => {
            let light = LightProperties { light_id: lid, ..Default::default() };
            container.append_child(&light.render(scope, &[]));
        }
        Some(SelectedEntity::Scene) => {
            let scene_name = ui.scene_name.get();
            let name_row = scope.create_element("div");
            name_row.set_attribute("style", SECTION_STYLE);
            name_row.append_child(&scope.create_text(&scene_name));
            container.append_child(&name_row);

            let object_count = ui.objects.get().len();
            let detail = scope.create_element("div");
            detail.set_attribute("style", VALUE_STYLE);
            detail.append_child(
                &scope.create_text(&format!("{} objects", object_count)),
            );
            container.append_child(&detail);
        }
        Some(SelectedEntity::Project) => {
            let hdr = scope.create_element("div");
            hdr.set_attribute("style", SECTION_STYLE);
            hdr.append_child(&scope.create_text("Project"));
            container.append_child(&hdr);
        }
        None => {
            let msg = scope.create_element("div");
            msg.set_attribute(
                "style",
                &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
            );
            msg.append_child(&scope.create_text("No object selected"));
            container.append_child(&msg);
        }
    }
}

/// Build asset-specific content (material or shader properties).
fn build_asset_content(
    scope: &mut RenderScope,
    container: &NodeHandle,
    ui: &UiSignals,
) {
    let selected_mat_slot = ui.selected_material.get();
    let selected_shader_name = ui.selected_shader.get();

    if let Some(slot) = selected_mat_slot {
        let mat = MaterialProperties { slot, ..Default::default() };
        container.append_child(&mat.render(scope, &[]));
    } else if let Some(ref shader_name) = selected_shader_name {
        let shaders = ui.shaders.get();
        if shaders.iter().any(|s| &s.name == shader_name) {
            let shader = ShaderProperties {
                shader_name: shader_name.clone(),
                ..Default::default()
            };
            container.append_child(&shader.render(scope, &[]));
        }
    } else {
        let msg = scope.create_element("div");
        msg.set_attribute(
            "style",
            &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
        );
        msg.append_child(&scope.create_text("No asset selected"));
        container.append_child(&msg);
    }
}
