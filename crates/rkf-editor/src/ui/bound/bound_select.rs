//! BoundSelect — Layer 2 widget that binds an EnumSelect to a UiStore path.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::enum_select::EnumSelect;

/// A select dropdown bound to a store path.
///
/// Reads the current string value from `store.read(path)` and writes back on
/// change via `store.set(path, UiValue::String(v))`.
#[component]
pub fn BoundSelect(
    path: String,
    label: String,
    options: Vec<(String, String)>,
) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let value = signal
        .get()
        .as_string()
        .unwrap_or("")
        .to_string();

    let store_for_cb = store.clone();
    let path_for_cb = path.clone();
    EnumSelect {
        value,
        options,
        label,
        on_change: Some(InputCallback::new(move |v: String| {
            store_for_cb.set(&path_for_cb, UiValue::String(v));
        })),
    }
    .render(__scope, &[])
}
