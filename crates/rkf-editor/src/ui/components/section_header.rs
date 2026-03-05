//! SectionHeader — collapsible section with triangle indicator.

use rinch::prelude::*;

/// A collapsible section header. Click toggles `collapsed` signal.
/// Children are hidden when collapsed.
#[component]
pub fn SectionHeader(title: String, collapsed: Option<Signal<bool>>, children: &[NodeHandle]) -> NodeHandle {
    let collapsed = collapsed.expect("SectionHeader requires collapsed signal");

    let header_style = "display:flex;align-items:center;padding:6px 12px;cursor:pointer;\
        font-size:12px;font-weight:600;color:var(--rinch-color-text);\
        user-select:none;gap:6px;";

    rsx! {
        div {
            div {
                style: {header_style},
                onclick: move || collapsed.update(|v| *v = !*v),
                span {
                    style: {|| if collapsed.get() {
                        "display:inline-block;font-size:8px;transition:transform 0.15s;"
                    } else {
                        "display:inline-block;font-size:8px;transition:transform 0.15s;\
                         transform:rotate(90deg);"
                    }},
                    "\u{25B6}"
                }
                span { {title} }
            }
            div {
                style: {|| if collapsed.get() { "display:none;" } else { "" }},
            }
        }
    }
}
