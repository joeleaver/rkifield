//! BoundColor — Layer 2 widget that binds a ColorSwatch to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::color_swatch::ColorSwatch;

/// A color picker bound to a store path.
///
/// Reads the current hex string from `store.read(path)` and writes back on
/// change via `store.set(path, UiValue::String(hex))`. The store's routing
/// layer handles hex-to-RGB conversion.
#[component]
pub fn BoundColor(path: String, label: String) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal
        .get()
        .as_string()
        .unwrap_or("#000000")
        .to_string();

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();
    ColorSwatch {
        value,
        label,
        on_change: Some(InputCallback::new(move |hex: String| {
            store_for_cb.set(&path_for_cb, UiValue::String(hex));
        })),
    }
    .render(__scope, &[])
}
