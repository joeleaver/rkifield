//! FloatSlider — pure Layer 1 widget. Takes a value, renders a labeled slider,
//! calls back on change. No store/ECS/command knowledge.

use rinch::prelude::*;

/// A labeled slider row with value display.
///
/// Unlike `SliderRow` (which owns a `Signal`), this widget receives a plain
/// `f64` value from the parent and calls `on_change` when the user drags.
/// An internal signal is used to drive the rinch `Slider`, with a guard to
/// suppress the spurious `onchange` that rinch fires during initial render.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn FloatSlider(
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
    label: String,
    suffix: String,
    on_change: Option<ValueCallback<f64>>,
) -> NodeHandle {
    // Internal signal — initialized from the prop value.
    let sig = Signal::new(value);

    // Guard: track the initial value so we can skip the onchange that rinch
    // fires during render when it sets the slider to the initial value.
    let initial = std::cell::Cell::new(value);

    // When the parent passes a new value, update the internal signal.
    // This runs each re-render, keeping the slider in sync with external state.
    sig.set(value);
    initial.set(value);

    // Build the Slider imperatively to avoid subscribing the parent scope.
    let slider = Slider {
        min: Some(min),
        max: Some(max),
        step: Some(step),
        value_signal: Some(sig),
        size: "xs".to_string(),
        onchange: Some(ValueCallback::new(move |v: f64| {
            // Guard: skip if value hasn't actually changed from what we set.
            if (v - initial.get()).abs() < 1e-12 {
                return;
            }
            sig.set(v);
            if let Some(cb) = &on_change {
                cb.0(v);
            }
        })),
        ..Default::default()
    };
    let slider_node = rinch::core::untracked(|| slider.render(__scope, &[]));

    // Reactive value text via create_effect (same pattern as existing SliderRow).
    let val_text = __scope.create_text("");
    {
        let val_text = val_text.clone();
        __scope.create_effect(move || {
            let v = sig.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix}"),
                1 => format!("{v:.1}{suffix}"),
                2 => format!("{v:.2}{suffix}"),
                _ => format!("{v:.3}{suffix}"),
            };
            val_text.set_text(&text);
        });
    }
    let val_span = __scope.create_element("span");
    val_span.set_attribute("style", "font-family:var(--rinch-font-family-monospace);");
    val_span.append_child(&val_text);

    rsx! {
        div { style: "padding:3px 12px;",
            div {
                style: "display:flex;justify-content:space-between;\
                        font-size:11px;color:var(--rinch-color-dimmed);margin-bottom:2px;",
                {label.clone()}
                {val_span}
            }
            {slider_node}
        }
    }
}
