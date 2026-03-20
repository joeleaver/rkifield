//! ColorSwatch — pure Layer 1 widget. Takes a hex color string, renders a
//! labeled color picker, calls back on change. No store/ECS/command knowledge.

use rinch::prelude::*;

fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.strip_prefix('#')?;
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some((r, g, b))
    } else {
        None
    }
}

/// A labeled color picker row.
///
/// Uses manual DOM construction with `__scope.create_element` to avoid
/// closure depth issues that occur when `ColorPicker` is nested inside `rsx!`.
/// The `onchange` callback is guarded against the initial-value echo that
/// rinch fires during render.
#[component]
pub fn ColorSwatch(
    value: String,
    label: String,
    on_change: Option<InputCallback>,
) -> NodeHandle {
    let row = __scope.create_element("div");
    row.set_attribute(
        "style",
        "display:flex;align-items:center;gap:6px;padding:3px 12px;\
         font-size:11px;color:var(--rinch-color-dimmed);",
    );

    let label_span = __scope.create_element("span");
    label_span.set_text(&label);
    row.append_child(&label_span);

    let initial_hex = value.clone();
    let picker = ColorPicker {
        format: "hex".into(),
        value,
        alpha: false,
        with_input: false,
        size: "xs".into(),
        onchange: Some(InputCallback::new({
            move |hex: String| {
                // Guard: skip the initial-value echo fired during render.
                if hex == initial_hex {
                    return;
                }
                // Validate before forwarding.
                if parse_hex_color(&hex).is_some() {
                    if let Some(cb) = &on_change {
                        cb.0(hex);
                    }
                }
            }
        })),
        ..Default::default()
    };
    let picker_node = rinch::core::untracked(|| picker.render(__scope, &[]));
    row.append_child(&picker_node);

    row
}
