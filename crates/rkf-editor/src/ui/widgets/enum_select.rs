//! EnumSelect — pure Layer 1 widget. Takes a string value and a list of
//! (value, label) options, renders a labeled dropdown, calls back on change.
//! No store/ECS/command knowledge.
//!
//! Uses manual DOM construction to avoid rsx! closure issues with
//! non-Copy `InputCallback`.

use rinch::prelude::*;

/// A labeled select dropdown for enum-like choices.
///
/// Uses the keyed rebuild pattern: a fingerprint of options + current value
/// forces the `Select` to reconstruct when the option list or value changes
/// externally, keeping the displayed selection in sync.
#[component]
pub fn EnumSelect(
    value: String,
    options: Vec<(String, String)>,
    label: String,
    on_change: Option<InputCallback>,
) -> NodeHandle {
    let row = __scope.create_element("div");
    row.set_attribute(
        "style",
        "display:flex;align-items:center;justify-content:space-between;\
         padding:3px 12px;gap:6px;",
    );

    let label_span = __scope.create_element("span");
    label_span.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);flex-shrink:0;",
    );
    label_span.set_text(&label);
    row.append_child(&label_span);

    let select_options: Vec<SelectOption> = options
        .iter()
        .map(|(v, l)| SelectOption::new(v.clone(), l.clone()))
        .collect();

    let initial_value = value.clone();
    let select = Select {
        size: "xs".into(),
        value,
        onchange: Some(InputCallback::new({
            move |v: String| {
                if v == initial_value {
                    return;
                }
                if let Some(cb) = &on_change {
                    cb.0(v);
                }
            }
        })),
        data: select_options,
        ..Default::default()
    };
    let select_node = rinch::core::untracked(|| select.render(__scope, &[]));
    row.append_child(&select_node);

    row
}
