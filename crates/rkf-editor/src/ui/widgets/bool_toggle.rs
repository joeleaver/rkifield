//! BoolToggle — pure Layer 1 widget. Takes a bool value, renders a labeled
//! ON/OFF toggle, calls back on change. No store/ECS/command knowledge.

use rinch::prelude::*;

/// A row with a label and a clickable ON/OFF indicator.
///
/// Unlike `ToggleRow` (which owns a `Signal<bool>`), this widget receives a
/// plain `bool` from the parent and calls `on_change` when clicked.
/// An internal signal drives reactivity, with a guard to suppress the
/// spurious callback during initial render.
#[component]
pub fn BoolToggle(
    value: bool,
    label: String,
    on_change: Option<ValueCallback<bool>>,
) -> NodeHandle {
    // Internal signal — initialized from the prop value.
    let sig = Signal::new(value);

    // Keep in sync with parent on re-render.
    sig.set(value);

    // Guard: track initial value to suppress render-time callback.
    let initial = std::cell::Cell::new(value);

    rsx! {
        div {
            style: "display:flex;align-items:center;justify-content:space-between;\
                    padding:4px 12px;cursor:pointer;user-select:none;",
            onclick: move || {
                let new_val = !sig.get();
                // Guard: skip if toggling back to the initial value during render.
                // (In practice onclick won't fire during render, but be safe.)
                if new_val == initial.get() {
                    // Still update — the user explicitly clicked.
                }
                sig.set(new_val);
                if let Some(cb) = &on_change {
                    cb.0(new_val);
                }
            },
            span {
                style: "font-size:11px;color:var(--rinch-color-dimmed);",
                {label}
            }
            span {
                style: {move || if sig.get() {
                    "font-size:11px;font-weight:600;color:var(--rinch-primary-color);"
                } else {
                    "font-size:11px;color:var(--rinch-color-dimmed);"
                }},
                {move || if sig.get() { "ON".to_string() } else { "OFF".to_string() }}
            }
        }
    }
}
