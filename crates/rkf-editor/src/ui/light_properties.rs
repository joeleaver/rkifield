//! Light properties panel — shows type, position, intensity, and range
//! for the selected light.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
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
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    let lid = light_id;

    let lights = ui.lights.get();
    let light_data = lights.iter().find(|l| l.id == lid);

    if let Some(light) = light_data {
        let type_name = match light.light_type {
            crate::light_editor::SceneLightType::Point => "Point Light",
            crate::light_editor::SceneLightType::Spot => "Spot Light",
        };

        // Memos derived from UiSignals — reactive, Copy, single source of truth.
        let pos_x = Memo::new(move || ui.lights.get().iter().find(|l| l.id == lid).map(|l| l.position.x as f64).unwrap_or(0.0));
        let pos_y = Memo::new(move || ui.lights.get().iter().find(|l| l.id == lid).map(|l| l.position.y as f64).unwrap_or(0.0));
        let pos_z = Memo::new(move || ui.lights.get().iter().find(|l| l.id == lid).map(|l| l.position.z as f64).unwrap_or(0.0));

        let pos_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: [f64; 3]| {
                let _ = cmd.0.send(EditorCommand::SetLightPosition {
                    light_id: lid,
                    position: glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32),
                });
            }
        });

        let pos_editor = Vec3Editor {
            x: pos_x, y: pos_y, z: pos_z,
            on_change: Some(pos_cb.clone()),
            on_commit: Some(pos_cb),
            step: 0.01, min: -500.0, max: 500.0, decimals: 2, suffix: String::new(),
        };
        let pos_node = rinch::core::untracked(|| pos_editor.render(__scope, &[]));

        let int_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: f64| { let _ = cmd.0.send(EditorCommand::SetLightIntensity { light_id: lid, intensity: v as f32 }); }
        });
        let int_dv = DragValue {
            value: Memo::new(move || ui.lights.get().iter().find(|l| l.id == lid).map(|l| l.intensity as f64).unwrap_or(1.0)),
            on_change: Some(int_cb.clone()),
            on_commit: Some(int_cb),
            step: 0.1, min: 0.0, max: 50.0, decimals: 1,
            ..Default::default()
        };
        let int_node = rinch::core::untracked(|| int_dv.render(__scope, &[]));

        let range_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: f64| { let _ = cmd.0.send(EditorCommand::SetLightRange { light_id: lid, range: v as f32 }); }
        });
        let range_dv = DragValue {
            value: Memo::new(move || ui.lights.get().iter().find(|l| l.id == lid).map(|l| l.range as f64).unwrap_or(10.0)),
            on_change: Some(range_cb.clone()),
            on_commit: Some(range_cb),
            step: 0.5, min: 0.1, max: 100.0, decimals: 1, suffix: "m".to_string(),
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
