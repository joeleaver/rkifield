//! TransformEditor — Position, Rotation, Scale sections using Vec3Editor.

use rinch::prelude::*;

use crate::editor_state::SliderSignals;
use crate::CommandSender;

use super::Vec3Editor;

/// Edits a full transform: Position, Rotation, Scale, each as a Vec3Editor row.
#[derive(Debug)]
pub struct TransformEditor {
    pub pos_x: Signal<f64>,
    pub pos_y: Signal<f64>,
    pub pos_z: Signal<f64>,
    pub rot_x: Signal<f64>,
    pub rot_y: Signal<f64>,
    pub rot_z: Signal<f64>,
    pub scale_x: Signal<f64>,
    pub scale_y: Signal<f64>,
    pub scale_z: Signal<f64>,
}

impl Default for TransformEditor {
    fn default() -> Self {
        Self {
            pos_x: Signal::new(0.0),
            pos_y: Signal::new(0.0),
            pos_z: Signal::new(0.0),
            rot_x: Signal::new(0.0),
            rot_y: Signal::new(0.0),
            rot_z: Signal::new(0.0),
            scale_x: Signal::new(1.0),
            scale_y: Signal::new(1.0),
            scale_z: Signal::new(1.0),
        }
    }
}

impl Component for TransformEditor {
    fn render(&self, __scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let sliders = use_context::<SliderSignals>();
        let cmd = use_context::<CommandSender>();

        let on_change = ValueCallback::new(move |_v: f64| {
            sliders.send_object_transform_commands(&cmd);
        });

        #[allow(clippy::type_complexity)]
        let sections: [(&str, Signal<f64>, Signal<f64>, Signal<f64>, f64, u32, &str, f64, f64); 3] = [
            ("Position", self.pos_x, self.pos_y, self.pos_z, 0.01, 2, "", -1e9, 1e9),
            ("Rotation", self.rot_x, self.rot_y, self.rot_z, 0.5, 1, "\u{00b0}", -180.0, 180.0),
            ("Scale", self.scale_x, self.scale_y, self.scale_z, 0.01, 2, "", 0.01, 1e9),
        ];

        let container = rsx! {
            div { style: "display: flex; flex-direction: column; gap: 4px;" }
        };

        for (label, x, y, z, step, decimals, suffix, min, max) in sections {
            let editor = Vec3Editor {
                x,
                y,
                z,
                step,
                min,
                max,
                decimals,
                suffix: suffix.into(),
                on_change: Some(on_change.clone()),
            };

            let row = rsx! {
                div {
                    style: "display: flex; align-items: center; padding: 2px 12px; gap: 8px;",
                    span {
                        style: "font-size: 11px; color: #999; min-width: 56px; user-select: none;",
                        {label}
                    }
                }
            };

            row.append_child(&editor.render(__scope, &[]));
            container.append_child(&row);
        }

        container
    }
}
