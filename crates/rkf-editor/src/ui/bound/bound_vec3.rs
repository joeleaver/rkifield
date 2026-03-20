//! BoundVec3 — Layer 2 widget that binds a Vec3Input to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::vec3_input::Vec3Input;

/// A 3D vector input bound to a store path.
///
/// Reads `store.read(path)` as `UiValue::Vec3`, renders a `Vec3Input`,
/// and writes back via `store.set(path, UiValue::Vec3(...))` on change.
#[component]
#[allow(clippy::too_many_arguments)]
pub fn BoundVec3(
    path: String,
    label: String,
    min: f64,
    max: f64,
    step: f64,
    decimals: u32,
) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal.get().as_vec3().unwrap_or([0.0; 3]);

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();

    let labels = [
        format!("{label} X"),
        format!("{label} Y"),
        format!("{label} Z"),
    ];

    Vec3Input {
        value,
        labels,
        min,
        max,
        step,
        decimals,
        on_change: Some(ValueCallback::new(move |v: [f64; 3]| {
            store_for_cb.set(&path_for_cb, UiValue::Vec3(v));
        })),
    }
    .render(__scope, &[])
}
