//! Vec3Editor — three DragValues in a row with colored X/Y/Z labels.

use rinch::prelude::*;

use super::DragValue;

/// Three DragValues for editing a 3D vector, with colored X (red), Y (green), Z (blue) labels.
#[derive(Debug)]
pub struct Vec3Editor {
    pub x: Signal<f64>,
    pub y: Signal<f64>,
    pub z: Signal<f64>,
    pub step: f64,
    pub min: f64,
    pub max: f64,
    pub decimals: u32,
    pub suffix: String,
    /// Called when any axis value changes (drag or text entry).
    pub on_change: Option<ValueCallback<f64>>,
}

impl Default for Vec3Editor {
    fn default() -> Self {
        Self {
            x: Signal::new(0.0),
            y: Signal::new(0.0),
            z: Signal::new(0.0),
            step: 0.01,
            min: -1e9,
            max: 1e9,
            decimals: 2,
            suffix: String::new(),
            on_change: None,
        }
    }
}

impl Component for Vec3Editor {
    fn render(&self, scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let container = scope.create_element("div");
        container.set_attribute(
            "style",
            "display: flex; gap: 4px; align-items: center;",
        );

        let axes: [(Signal<f64>, &str, &str); 3] = [
            (self.x, "X", "#e05050"),
            (self.y, "Y", "#50c050"),
            (self.z, "Z", "#5080e0"),
        ];

        for (sig, label, color) in axes {
            let dv = DragValue {
                value: sig,
                step: self.step,
                min: self.min,
                max: self.max,
                decimals: self.decimals,
                label: label.into(),
                label_color: color.into(),
                suffix: self.suffix.clone(),
                on_change: self.on_change.clone(),
            };
            let node = dv.render(scope, &[]);
            container.append_child(&node);
        }

        container
    }
}
