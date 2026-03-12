//! Slider helper functions shared by the right panel.

use rinch::prelude::*;

// ── Slider helper ───────────────────────────────────────────────────────────

/// Build a labeled slider row and append it to the container.
///
/// Creates a row with label + reactive value display on top and a Slider beneath.
/// The slider updates `signal` on drag (fine-grained, no revision bump) and
/// calls `on_update` to write back to editor state.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_slider_row(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    decimals: usize,
    on_update: impl Fn(f64) + 'static,
) {
    let row = scope.create_element("div");
    row.set_attribute("style", "padding:3px 12px;");

    // Label + reactive value display.
    let label_row = scope.create_element("div");
    label_row.set_attribute(
        "style",
        "display:flex;justify-content:space-between;\
         font-size:11px;color:rgba(255,255,255,0.7);margin-bottom:2px;",
    );
    label_row.append_child(&scope.create_text(label));

    // Reactive value text — create_effect surgically updates the text node
    // instead of tearing down and rebuilding.
    let val_text = scope.create_text("");
    let suffix = suffix.to_string();
    {
        let val_text = val_text.clone();
        scope.create_effect(move || {
            let v = signal.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix}"),
                1 => format!("{v:.1}{suffix}"),
                _ => format!("{v:.2}{suffix}"),
            };
            val_text.set_text(&text);
        });
    }
    let val_span = scope.create_element("span");
    val_span.set_attribute(
        "style",
        "font-family:var(--rinch-font-family-monospace);",
    );
    val_span.append_child(&val_text);
    label_row.append_child(&val_span);
    row.append_child(&label_row);

    // Slider component.
    let slider = Slider {
        min: Some(min),
        max: Some(max),
        step: Some(step),
        value_signal: Some(signal),
        size: "xs".to_string(),
        onchange: Some(ValueCallback::new(move |v: f64| {
            signal.set(v);
            on_update(v);
        })),
        ..Default::default()
    };
    // Render the slider inside `untracked` so the Slider's internal
    // `value_signal.get()` (for initial value) doesn't subscribe the
    // parent scope to every slider signal.
    let slider_node = rinch::core::untracked(|| slider.render(scope, &[]));
    row.append_child(&slider_node);
    container.append_child(&row);
}

/// Build a logarithmic slider row — gives fine control at small values
/// while still reaching large values.
///
/// The displayed value and `signal` are in real units (e.g. 0.01–10.0).
/// Internally the slider knob moves on a linear 0–1 scale mapped through
/// `ln`/`exp` so that ~half the slider travel covers the lower decade.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_log_slider_row(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    decimals: usize,
    on_update: impl Fn(f64) + 'static,
) {
    let ln_min = min.max(1e-6).ln();
    let ln_max = max.ln();

    // Internal linear 0–1 knob signal, initialized from current value.
    let knob = Signal::new((signal.get().max(min).ln() - ln_min) / (ln_max - ln_min));

    let row = scope.create_element("div");
    row.set_attribute("style", "padding:3px 12px;");

    // Label + reactive value display (shows real value, not knob position).
    let label_row = scope.create_element("div");
    label_row.set_attribute(
        "style",
        "display:flex;justify-content:space-between;\
         font-size:11px;color:rgba(255,255,255,0.7);margin-bottom:2px;",
    );
    label_row.append_child(&scope.create_text(label));

    let val_text = scope.create_text("");
    let suffix = suffix.to_string();
    {
        let val_text = val_text.clone();
        scope.create_effect(move || {
            let v = signal.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix}"),
                1 => format!("{v:.1}{suffix}"),
                _ => format!("{v:.2}{suffix}"),
            };
            val_text.set_text(&text);
        });
    }
    let val_span = scope.create_element("span");
    val_span.set_attribute(
        "style",
        "font-family:var(--rinch-font-family-monospace);",
    );
    val_span.append_child(&val_text);
    label_row.append_child(&val_span);
    row.append_child(&label_row);

    // Slider operates on linear knob signal (0–1).
    let slider = Slider {
        min: Some(0.0),
        max: Some(1.0),
        step: Some(0.001),
        value_signal: Some(knob),
        size: "xs".to_string(),
        onchange: Some(ValueCallback::new(move |t: f64| {
            knob.set(t);
            let real = (ln_min + t * (ln_max - ln_min)).exp();
            signal.set(real);
            on_update(real);
        })),
        ..Default::default()
    };
    let slider_node = rinch::core::untracked(|| slider.render(scope, &[]));
    row.append_child(&slider_node);
    container.append_child(&row);
}

/// Build a slider row bound to a `SliderSignals` signal.
///
/// The slider updates `signal` on drag and calls `send_all_commands()`
/// on the SliderSignals store to push changes to the engine.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_synced_slider(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    decimals: usize,
    sliders: crate::editor_state::SliderSignals,
    cmd: crate::CommandSender,
    ui: crate::editor_state::UiSignals,
) {
    build_slider_row(
        scope, container, label, suffix, signal,
        min, max, step, decimals,
        move |_v| {
            sliders.send_all_commands(&cmd, &ui);
        },
    );
}
