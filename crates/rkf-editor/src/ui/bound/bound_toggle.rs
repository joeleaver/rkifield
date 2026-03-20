//! BoundToggle — Layer 2 widget that binds a BoolToggle to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::bool_toggle::BoolToggle;

/// A boolean toggle bound to a store path.
///
/// Reads the current value from `store.read(path)` and writes back on change
/// via `store.set(path, UiValue::Bool(v))`.
#[component]
pub fn BoundToggle(path: String, label: String) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal.get().as_bool().unwrap_or(false);

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();
    BoolToggle {
        value,
        label,
        on_change: Some(ValueCallback::new(move |v: bool| {
            store_for_cb.set(&path_for_cb, UiValue::Bool(v));
        })),
    }
    .render(__scope, &[])
}
