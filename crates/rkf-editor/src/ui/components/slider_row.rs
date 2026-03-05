//! SliderRow — labeled slider with reactive value display.

use rinch::prelude::*;

/// A labeled row containing a Slider with reactive value readout.
///
/// Wraps the rinch `Slider` component with a label + value display on top.
/// The `on_change` callback fires when the user drags the slider.
#[component]
pub fn SliderRow(
    label: String,
    suffix: String,
    signal: Option<Signal<f64>>,
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
    on_change: Option<ValueCallback<f64>>,
) -> NodeHandle {
    let signal = signal.expect("SliderRow requires signal");

    let row = __scope.create_element("div");
    row.set_attribute("style", "padding:3px 12px;");

    // Label + reactive value display.
    let label_row = __scope.create_element("div");
    label_row.set_attribute(
        "style",
        "display:flex;justify-content:space-between;\
         font-size:11px;color:var(--rinch-color-dimmed);margin-bottom:2px;",
    );
    label_row.append_child(&__scope.create_text(&label));

    let val_span = __scope.create_element("span");
    val_span.set_attribute(
        "style",
        "font-family:var(--rinch-font-family-monospace);",
    );
    {
        let val_h = val_span.clone();
        let suffix = suffix.clone();
        Effect::new(move || {
            let v = signal.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix}"),
                1 => format!("{v:.1}{suffix}"),
                _ => format!("{v:.2}{suffix}"),
            };
            val_h.set_text(&text);
        });
    }
    label_row.append_child(&val_span);
    row.append_child(&label_row);

    // Slider component — rendered in untracked scope to avoid subscribing
    // parent reactive_component_dom to slider signal changes.
    let slider = Slider {
        min: Some(min),
        max: Some(max),
        step: Some(step),
        value_signal: Some(signal),
        size: "xs".to_string(),
        onchange: Some(ValueCallback::new(move |v: f64| {
            signal.set(v);
            if let Some(cb) = &on_change {
                cb.0(v);
            }
        })),
        ..Default::default()
    };
    let slider_node = rinch::core::untracked(|| slider.render(__scope, &[]));
    row.append_child(&slider_node);

    row
}

/// Build a SliderRow that only sets the signal (no external callback).
///
/// Used when batch sync writes all signal values to EditorState via an Effect.
#[allow(clippy::too_many_arguments)]
pub fn build_synced_slider_row(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
) {
    let row = SliderRow {
        label: label.to_string(),
        suffix: suffix.to_string(),
        signal: Some(signal),
        min,
        max,
        step,
        decimals,
        ..Default::default()
    };
    let node = rinch::core::untracked(|| row.render(scope, &[]));
    container.append_child(&node);
}
