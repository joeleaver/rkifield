//! BoundLogSlider — Layer 2 widget that binds a logarithmic slider to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::slider_helpers::LogSliderRow;

/// A logarithmic slider bound to a store path.
///
/// Reads the current value from `store.read(path)` and writes back on change
/// via `store.set(path, UiValue::Float(v))`. Uses log scale for fine control
/// at small values (e.g. brush radius 0.01–10.0).
#[component]
#[allow(clippy::too_many_arguments)]
pub fn BoundLogSlider(
    path: String,
    label: String,
    min: f64,
    max: f64,
    decimals: u32,
    suffix: String,
) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal.get().as_float().unwrap_or(min);

    // Create a local signal seeded from the store value.
    let local_signal = Signal::new(value);

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();
    LogSliderRow {
        label,
        suffix,
        signal: Some(local_signal),
        min: Some(min),
        max: Some(max),
        decimals: Some(decimals),
        on_change: Some(ValueCallback::new(move |v: f64| {
            store_for_cb.set(&path_for_cb, UiValue::Float(v));
        })),
    }
    .render(__scope, &[])
}
