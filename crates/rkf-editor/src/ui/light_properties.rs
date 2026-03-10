//! Light properties panel — shows type, position, intensity, and range
//! for the selected light.

use rinch::prelude::*;

use crate::editor_state::{SliderSignals, UiSignals};
use crate::CommandSender;

use super::components::{DragValue, Vec3Editor};
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE};

const DRAG_ROW_STYLE: &str =
    "display:flex;align-items:center;padding:2px 12px;gap:8px;";
const DRAG_LABEL_STYLE: &str =
    "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;";

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

    let lights = ui.lights.get();
    let lid = light_id;

    let light_data = lights.iter().find(|l| l.id == lid);

    if let Some(light) = light_data {
        let type_name = match light.light_type {
            crate::light_editor::SceneLightType::Point => "Point Light",
            crate::light_editor::SceneLightType::Spot => "Spot Light",
        };

        // Build component instances for embedding via untracked render.
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
        let pos_node = rinch::core::untracked(|| pos_editor.render(__scope, &[]));

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

        rsx! {
            div { style: "display:flex;flex-direction:column;",
                div { style: {SECTION_STYLE}, {type_name} }

                // Position — Vec3Editor with colored axis labels.
                div { style: {LABEL_STYLE}, "Position" }
                div { style: "padding: 2px 12px;", {pos_node} }

                div { style: {DIVIDER_STYLE} }

                // Intensity — DragValue.
                div { style: {DRAG_ROW_STYLE},
                    span { style: {DRAG_LABEL_STYLE}, "Intensity" }
                    {int_node}
                }

                // Range — DragValue.
                div { style: {DRAG_ROW_STYLE},
                    span { style: {DRAG_LABEL_STYLE}, "Range" }
                    {range_node}
                }
            }
        }
    } else {
        rsx! { div { style: "display:flex;flex-direction:column;" } }
    }
}
