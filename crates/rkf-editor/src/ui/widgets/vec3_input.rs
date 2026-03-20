//! Vec3Input — pure Layer 1 widget. Takes a [f64; 3] value, renders three
//! labeled float sliders (X/Y/Z), calls back on change with the full vector.

use rinch::prelude::*;

use super::float_slider::FloatSlider;

/// A row of three labeled float sliders for editing a 3D vector.
///
/// Each axis gets its own `FloatSlider`. When any axis changes, the full
/// `[f64; 3]` is passed to `on_change`. Uses manual DOM construction to
/// avoid rsx! closure depth issues with three child components.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn Vec3Input(
    value: [f64; 3],
    labels: [String; 3],
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
    on_change: Option<ValueCallback<[f64; 3]>>,
) -> NodeHandle {
    let container = __scope.create_element("div");
    container.set_attribute(
        "style",
        "display:flex;gap:2px;",
    );

    for axis in 0..3 {
        let current = value;
        let on_change = on_change.clone();
        let slider = FloatSlider {
            value: current[axis],
            min,
            max,
            step,
            decimals,
            label: labels[axis].clone(),
            suffix: String::new(),
            on_change: Some(ValueCallback::new(move |v: f64| {
                let mut new_val = current;
                new_val[axis] = v;
                if let Some(cb) = &on_change {
                    cb.0(new_val);
                }
            })),
        };
        let axis_container = __scope.create_element("div");
        axis_container.set_attribute("style", "flex:1;min-width:0;");
        let slider_node = rinch::core::untracked(|| slider.render(__scope, &[]));
        axis_container.append_child(&slider_node);
        container.append_child(&axis_container);
    }

    container
}
