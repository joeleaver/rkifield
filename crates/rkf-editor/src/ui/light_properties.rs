//! Light properties panel — shows type, position, intensity, and range
//! for the selected light. Uses store-bound sliders for all fields.

use rinch::prelude::*;

use crate::editor_state::UiSignals;
use crate::ui::bound::bound_slider::BoundSlider;

use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE};

const DRAG_ROW_STYLE: &str =
    "display:flex;align-items:center;padding:2px 12px;gap:8px;";
const DRAG_LABEL_STYLE: &str =
    "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;";

/// Light properties panel — shows type name, position sliders,
/// intensity slider, and range slider, all bound to the UI Store.
#[component]
pub fn LightProperties() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let lid = match ui.selection.get() {
        Some(crate::editor_state::SelectedEntity::Light(id)) => id,
        _ => return rsx! { div {} },
    };

    let lights = ui.lights.get();
    let light_data = lights.iter().find(|l| l.id == lid);

    if let Some(light) = light_data {
        let type_name = match light.light_type {
            crate::light_editor::SceneLightType::Point => "Point Light",
            crate::light_editor::SceneLightType::Spot => "Spot Light",
        };

        // Position — three per-axis BoundSliders.
        let pos_x = BoundSlider {
            path: format!("light:{lid}/position.x"), label: "X".into(),
            min: -500.0, max: 500.0, step: 0.01, decimals: 2, suffix: String::new(),
        }.render(__scope, &[]);
        let pos_y = BoundSlider {
            path: format!("light:{lid}/position.y"), label: "Y".into(),
            min: -500.0, max: 500.0, step: 0.01, decimals: 2, suffix: String::new(),
        }.render(__scope, &[]);
        let pos_z = BoundSlider {
            path: format!("light:{lid}/position.z"), label: "Z".into(),
            min: -500.0, max: 500.0, step: 0.01, decimals: 2, suffix: String::new(),
        }.render(__scope, &[]);

        // Intensity — BoundSlider.
        let int_slider = BoundSlider {
            path: format!("light:{lid}/intensity"), label: "Intensity".into(),
            min: 0.0, max: 50.0, step: 0.1, decimals: 1, suffix: String::new(),
        }.render(__scope, &[]);

        // Range — BoundSlider.
        let range_slider = BoundSlider {
            path: format!("light:{lid}/range"), label: "Range".into(),
            min: 0.1, max: 100.0, step: 0.5, decimals: 1, suffix: "m".into(),
        }.render(__scope, &[]);

        rsx! {
            div { style: "display:flex;flex-direction:column;",
                div { style: {SECTION_STYLE}, {type_name} }

                // Position — three axis sliders.
                div { style: {LABEL_STYLE}, "Position" }
                div { style: "padding: 2px 12px;",
                    {pos_x}
                    {pos_y}
                    {pos_z}
                }

                div { style: {DIVIDER_STYLE} }

                // Intensity.
                div { style: {DRAG_ROW_STYLE},
                    span { style: {DRAG_LABEL_STYLE}, "Intensity" }
                    {int_slider}
                }

                // Range.
                div { style: {DRAG_ROW_STYLE},
                    span { style: {DRAG_LABEL_STYLE}, "Range" }
                    {range_slider}
                }
            }
        }
    } else {
        rsx! { div { style: "display:flex;flex-direction:column;" } }
    }
}
