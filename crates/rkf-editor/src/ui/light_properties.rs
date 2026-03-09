//! Light properties panel — shows type, position, intensity, and range
//! for the selected light.

use rinch::prelude::*;

use crate::editor_state::{SliderSignals, UiSignals};
use crate::CommandSender;

use super::components::{DragValue, Vec3Editor};
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE};

/// Light properties panel — shows type name, position Vec3Editor,
/// intensity DragValue, and range DragValue.
#[component]
pub fn LightProperties(
    light_id: u64,
) -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let on_change = ValueCallback::new(move |_v: f64| {
        sliders.send_light_commands(&cmd);
    });

    let container = __scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    let lights = ui.lights.get();
    let lid = light_id;

    let light_data = lights.iter().find(|l| l.id == lid);
    if let Some(light) = light_data {
        let type_name = match light.light_type {
            crate::light_editor::SceneLightType::Point => "Point Light",
            crate::light_editor::SceneLightType::Spot => "Spot Light",
        };
        let hdr = __scope.create_element("div");
        hdr.set_attribute("style", SECTION_STYLE);
        hdr.append_child(&__scope.create_text(type_name));
        container.append_child(&hdr);

        // Position — Vec3Editor with colored axis labels.
        let pos_label = __scope.create_element("div");
        pos_label.set_attribute("style", LABEL_STYLE);
        pos_label.append_child(&__scope.create_text("Position"));
        container.append_child(&pos_label);

        let pos_editor = Vec3Editor {
            x: sliders.light_pos_x,
            y: sliders.light_pos_y,
            z: sliders.light_pos_z,
            step: 0.01,
            min: -500.0,
            max: 500.0,
            decimals: 2,
            suffix: String::new(),
            on_change: Some(on_change.clone()),
        };
        let pos_row = __scope.create_element("div");
        pos_row.set_attribute("style", "padding: 2px 12px;");
        let pos_node = rinch::core::untracked(|| pos_editor.render(__scope, &[]));
        pos_row.append_child(&pos_node);
        container.append_child(&pos_row);

        append_divider(__scope, &container);

        // Intensity — DragValue.
        let int_label_row = __scope.create_element("div");
        int_label_row.set_attribute(
            "style",
            "display:flex;align-items:center;padding:2px 12px;gap:8px;",
        );
        let int_label = __scope.create_element("span");
        int_label.set_attribute(
            "style",
            "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;",
        );
        int_label.append_child(&__scope.create_text("Intensity"));
        int_label_row.append_child(&int_label);

        let int_dv = DragValue {
            value: sliders.light_intensity,
            step: 0.1,
            min: 0.0,
            max: 50.0,
            decimals: 1,
            on_change: Some(on_change.clone()),
            ..Default::default()
        };
        let int_node = rinch::core::untracked(|| int_dv.render(__scope, &[]));
        int_label_row.append_child(&int_node);
        container.append_child(&int_label_row);

        // Range — DragValue.
        let range_label_row = __scope.create_element("div");
        range_label_row.set_attribute(
            "style",
            "display:flex;align-items:center;padding:2px 12px;gap:8px;",
        );
        let range_label = __scope.create_element("span");
        range_label.set_attribute(
            "style",
            "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;",
        );
        range_label.append_child(&__scope.create_text("Range"));
        range_label_row.append_child(&range_label);

        let range_dv = DragValue {
            value: sliders.light_range,
            step: 0.5,
            min: 0.1,
            max: 100.0,
            decimals: 1,
            suffix: "m".to_string(),
            on_change: Some(on_change.clone()),
            ..Default::default()
        };
        let range_node = rinch::core::untracked(|| range_dv.render(__scope, &[]));
        range_label_row.append_child(&range_node);
        container.append_child(&range_label_row);
    }

    container
}

fn append_divider(scope: &mut RenderScope, container: &NodeHandle) {
    let div = scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    container.append_child(&div);
}
