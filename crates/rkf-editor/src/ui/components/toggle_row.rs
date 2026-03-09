//! ToggleRow — clickable ON/OFF toggle with label.

use rinch::prelude::*;

/// A row with a label and a clickable ON/OFF indicator.
/// Clicking toggles the `enabled` signal and fires `on_change` if provided.
#[component]
pub fn ToggleRow(
    label: String,
    enabled: Option<Signal<bool>>,
    on_change: Option<Callback>,
) -> NodeHandle {
    let enabled = enabled.expect("ToggleRow requires enabled signal");

    rsx! {
        div {
            style: "display:flex;align-items:center;justify-content:space-between;\
                    padding:4px 12px;cursor:pointer;user-select:none;",
            onclick: move || {
                enabled.update(|v| *v = !*v);
                if let Some(cb) = &on_change {
                    cb.0();
                }
            },
            span {
                style: "font-size:11px;color:var(--rinch-color-dimmed);",
                {label}
            }
            span {
                style: {|| if enabled.get() {
                    "font-size:11px;font-weight:600;color:var(--rinch-primary-color);"
                } else {
                    "font-size:11px;color:var(--rinch-color-dimmed);"
                }},
                {|| if enabled.get() { "ON".to_string() } else { "OFF".to_string() }}
            }
        }
    }
}
