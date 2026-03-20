//! ActionButton — renders a button for a registered action from the action registry.

use rinch::prelude::*;

use crate::store::UiStore;

/// A button that renders from a registered action's metadata and executes it on click.
#[component]
pub fn ActionButton(action_id: String) -> NodeHandle {
    let store = use_context::<UiStore>();

    let meta = store.get_action(&action_id);
    let (label, _shortcut, is_checked): (&str, Option<&str>, bool) = match meta {
        Some(m) => (m.label, m.shortcut, m.checked.map_or(false, |f| f())),
        None => ("???", None, false),
    };

    let style: &str = if is_checked {
        "display:flex;align-items:center;gap:6px;padding:4px 10px;\
         cursor:pointer;user-select:none;border-radius:3px;\
         font-size:11px;color:var(--rinch-color-text);\
         background:var(--rinch-primary-color);"
    } else {
        "display:flex;align-items:center;gap:6px;padding:4px 10px;\
         cursor:pointer;user-select:none;border-radius:3px;\
         font-size:11px;color:var(--rinch-color-text);\
         background:var(--rinch-color-dark-6);"
    };

    rsx! {
        div {
            style: {style},
            onclick: {
                let store = store.clone();
                let id = action_id.clone();
                move || {
                    store.execute_action(&id);
                }
            },
            {label.to_string()}
        }
    }
}
