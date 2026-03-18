//! Slider helper components shared by the right panel.

use rinch::prelude::*;

// ── SliderRow component ────────────────────────────────────────────────────

/// A labeled slider row with reactive value display.
///
/// Shows label + reactive value text on top and a Slider beneath.
/// The slider updates `signal` on drag and calls `on_change` with the new value.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn SliderRow(
    label: String,
    suffix: String,
    signal: Option<Signal<f64>>,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    decimals: Option<u32>,
    on_change: Option<ValueCallback<f64>>,
) -> NodeHandle {
    let signal = signal.expect("SliderRow requires signal");
    let min = min.unwrap_or(0.0);
    let max = max.unwrap_or(1.0);
    let step = step.unwrap_or(0.01);
    let decimals = decimals.unwrap_or(2);

    // Reactive value text — use create_effect for surgical updates because
    // rsx reactive closures are FnMut and String can't be moved out of FnMut.
    let val_text = __scope.create_text("");
    {
        let val_text = val_text.clone();
        __scope.create_effect(move || {
            let v = signal.get();
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
                        font-size:11px;color:rgba(255,255,255,0.7);margin-bottom:2px;",
                {label.clone()}
                {val_span}
            }
            Slider {
                min: Some(min),
                max: Some(max),
                step: Some(step),
                value_signal: Some(signal),
                size: "xs",
                onchange: move |v: f64| {
                    signal.set(v);
                    if let Some(cb) = &on_change {
                        cb.0(v);
                    }
                },
            }
        }
    }
}

// ── LogSliderRow component ─────────────────────────────────────────────────

/// A logarithmic slider row — fine control at small values, reaches large values.
///
/// The displayed value and `signal` are in real units (e.g. 0.01-10.0).
/// Internally the slider moves on a linear 0-1 scale mapped through ln/exp.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn LogSliderRow(
    label: String,
    suffix: String,
    signal: Option<Signal<f64>>,
    min: Option<f64>,
    max: Option<f64>,
    decimals: Option<u32>,
    on_change: Option<ValueCallback<f64>>,
) -> NodeHandle {
    let signal = signal.expect("LogSliderRow requires signal");
    let min = min.unwrap_or(0.01);
    let max = max.unwrap_or(10.0);
    let decimals = decimals.unwrap_or(2);

    let ln_min = min.max(1e-6).ln();
    let ln_max = max.ln();

    // Internal linear 0-1 knob signal, initialized from current value.
    let knob = Signal::new((signal.get().max(min).ln() - ln_min) / (ln_max - ln_min));

    // Reactive value text — use create_effect (same reason as SliderRow).
    let val_text = __scope.create_text("");
    {
        let val_text = val_text.clone();
        __scope.create_effect(move || {
            let v = signal.get();
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
                        font-size:11px;color:rgba(255,255,255,0.7);margin-bottom:2px;",
                {label.clone()}
                {val_span}
            }
            Slider {
                min: 0.0,
                max: 1.0,
                step: 0.001,
                value_signal: Some(knob),
                size: "xs",
                onchange: move |t: f64| {
                    knob.set(t);
                    let real = (ln_min + t * (ln_max - ln_min)).exp();
                    signal.set(real);
                    if let Some(cb) = &on_change {
                        cb.0(real);
                    }
                },
            }
        }
    }
}

// ── SyncedSliderRow component ──────────────────────────────────────────────

/// A slider row that sends all commands on change via SliderSignals.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn SyncedSliderRow(
    label: String,
    suffix: String,
    signal: Option<Signal<f64>>,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    decimals: Option<u32>,
    sliders: Option<crate::editor_state::SliderSignals>,
    cmd: Option<crate::CommandSender>,
    ui: Option<crate::editor_state::UiSignals>,
) -> NodeHandle {
    let sliders = sliders.expect("SyncedSliderRow requires sliders");
    let cmd = cmd.expect("SyncedSliderRow requires cmd");
    let ui = ui.expect("SyncedSliderRow requires ui");

    rsx! {
        SliderRow {
            label: label,
            suffix: suffix,
            signal: signal,
            min: min,
            max: max,
            step: step,
            decimals: decimals,
            on_change: move |_v: f64| {
                sliders.send_all_commands(&cmd, &ui);
            },
        }
    }
}
