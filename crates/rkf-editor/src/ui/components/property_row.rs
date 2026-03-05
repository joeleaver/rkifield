//! PropertyRow — label on left, widget(s) on right.

use rinch::prelude::*;

/// A row with a label on the left and children (widgets) on the right.
#[component]
pub fn PropertyRow(label: String, children: &[NodeHandle]) -> NodeHandle {
    rsx! {
        div {
            style: "display:flex;align-items:center;padding:2px 12px;gap:8px;",
            span {
                style: "font-size:11px;color:var(--rinch-color-dimmed);min-width:72px;",
                {label}
            }
            div {
                style: "flex:1;display:flex;justify-content:flex-end;",
            }
        }
    }
}
