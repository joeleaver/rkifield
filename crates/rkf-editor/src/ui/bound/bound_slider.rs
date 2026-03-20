//! BoundSlider — Layer 2 widget that binds a FloatSlider to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::float_slider::FloatSlider;

/// A float slider bound to a store path.
///
/// Reads the current value from `store.read(path)` and writes back on change
/// via `store.set(path, UiValue::Float(v))`.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn BoundSlider(
    path: String,
    label: String,
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
    suffix: String,
) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal.get().as_float().unwrap_or(min);

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();
    FloatSlider {
        value,
        min,
        max,
        step,
        decimals,
        label,
        suffix,
        on_change: Some(ValueCallback::new(move |v: f64| {
            store_for_cb.set(&path_for_cb, UiValue::Float(v));
        })),
    }
    .render(__scope, &[])
}
