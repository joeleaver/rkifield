//! Loading modal — blocks interaction while a long-running task is in progress.
//!
//! Driven by `ui.loading_status: Signal<Option<String>>`. When `Some(message)`,
//! renders a centered semi-transparent overlay with the message.
//! When `None`, hidden via `display:none`.

use rinch::prelude::*;

use crate::editor_state::UiSignals;

/// Full-screen blocking modal shown during long tasks (e.g. game plugin build).
#[component]
pub fn LoadingModal() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            style: {
                let ui = ui;
                move || {
                    if ui.loading_status.get().is_some() {
                        "position:absolute;z-index:200;top:0;left:0;right:0;bottom:0;\
                         background:rgba(0,0,0,0.5);\
                         display:flex;align-items:center;justify-content:center;"
                            .to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }
            },

            // Center card
            div {
                style: "padding:24px 32px;background:var(--rinch-color-dark-8);\
                        border:1px solid var(--rinch-color-border);border-radius:8px;\
                        display:flex;align-items:center;gap:12px;\
                        box-shadow:0 4px 24px rgba(0,0,0,0.4);",

                // Indicator dot
                div {
                    style: "width:10px;height:10px;border-radius:50%;\
                            background:var(--rinch-color-yellow-5, #fcc419);flex-shrink:0;",
                }

                // Message text
                div {
                    style: "display:flex;flex-direction:column;gap:4px;",
                    div {
                        style: "font-size:14px;color:var(--rinch-color-text);",
                        {move || ui.loading_status.get().unwrap_or_default()}
                    }
                    div {
                        style: "font-size:12px;color:var(--rinch-color-dimmed, #868e96);",
                        "This may take a few minutes."
                    }
                }
            }
        }
    }
}
