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

    // Build Slider imperatively — needs untracked to avoid subscribing
    // parent scope to slider signal changes.
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

    // Build reactive value text imperatively — rsx! reactive closures are FnMut,
    // and String can't be moved out of FnMut. Use create_effect for surgical updates.
    let val_text = __scope.create_text("");
    let suffix2 = suffix;
    {
        let val_text = val_text.clone();
        __scope.create_effect(move || {
            let v = signal.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix2}"),
                1 => format!("{v:.1}{suffix2}"),
                _ => format!("{v:.2}{suffix2}"),
            };
            val_text.set_text(&text);
        });
    }
    let val_span = __scope.create_element("span");
    val_span.set_attribute("style", "font-family:var(--rinch-font-family-monospace);");
    val_span.append_child(&val_text);

    rsx! {
        div { style: "padding:3px 12px;",
            div { style: "display:flex;justify-content:space-between;font-size:11px;color:var(--rinch-color-dimmed);margin-bottom:2px;",
                {label.clone()}
                {val_span}
            }
            {slider_node}
        }
    }
}

