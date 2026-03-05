//! DragValue — Unity-style number field with drag-to-adjust and double-click-to-edit.

use std::cell::Cell;

use rinch::prelude::*;

/// Unity-style number field: displays value as text, click-drag left/right to adjust,
/// double-click to type a new value.
#[derive(Debug)]
pub struct DragValue {
    /// Two-way binding for the numeric value.
    pub value: Signal<f64>,
    /// Increment per pixel of drag (default 0.01).
    pub step: f64,
    /// Minimum clamp value.
    pub min: f64,
    /// Maximum clamp value.
    pub max: f64,
    /// Display decimal places.
    pub decimals: u32,
    /// Axis label text ("X", "Y", "Z").
    pub label: String,
    /// CSS color for the label ("#e05050" for X-red).
    pub label_color: String,
    /// Unit suffix ("°", "m").
    pub suffix: String,
}

impl Default for DragValue {
    fn default() -> Self {
        Self {
            value: Signal::new(0.0),
            step: 0.01,
            min: -1e9,
            max: 1e9,
            decimals: 2,
            label: String::new(),
            label_color: String::new(),
            suffix: String::new(),
        }
    }
}

impl Component for DragValue {
    fn render(&self, scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let value = self.value;
        let step = self.step;
        let min = self.min;
        let max = self.max;
        let decimals = self.decimals;
        let suffix = self.suffix.clone();
        let suffix2 = self.suffix.clone();

        // Internal state
        let editing = Signal::new(false);
        let start_val = Signal::new(0.0_f64);
        let start_mouse_x = Signal::new(0.0_f32);

        // Double-click detection: track last click time
        thread_local! {
            static LAST_CLICK: Cell<Option<std::time::Instant>> = const { Cell::new(None) };
        }

        // --- Container ---
        let container = scope.create_element("div");
        container.set_attribute(
            "style",
            "display: flex; align-items: center; gap: 2px; min-width: 60px; height: 20px;",
        );

        // --- Label span ---
        if !self.label.is_empty() {
            let label_span = scope.create_element("span");
            let label_style = format!(
                "color: {}; font-weight: 600; font-size: 11px; width: 12px; text-align: center; \
                 user-select: none; flex-shrink: 0;",
                if self.label_color.is_empty() {
                    "#aaa"
                } else {
                    &self.label_color
                }
            );
            label_span.set_attribute("style", &label_style);
            let label_text = scope.create_text(&self.label);
            label_span.append_child(&label_text);
            container.append_child(&label_span);
        }

        // --- Value display (non-editing mode) ---
        let display_div = scope.create_element("div");
        display_div.set_attribute(
            "style",
            "flex: 1; background: rgba(255,255,255,0.06); border-radius: 3px; \
             padding: 1px 4px; font-size: 11px; color: #ddd; cursor: ew-resize; \
             user-select: none; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; \
             height: 18px; line-height: 18px;",
        );

        let format_value = {
            let suffix = suffix.clone();
            move |v: f64| -> String {
                format!("{:.prec$}{}", v, suffix, prec = decimals as usize)
            }
        };

        let value_text = scope.create_text(&format_value(untracked(|| value.get())));
        display_div.append_child(&value_text);

        // Click handler for display_div: start drag, detect double-click
        let click_handler = scope.register_handler(move || {
            let ctx = get_click_context();

            // Double-click detection (300ms window)
            let is_double = LAST_CLICK.with(|lc| {
                let now = std::time::Instant::now();
                let prev = lc.get();
                lc.set(Some(now));
                if let Some(prev_time) = prev {
                    now.duration_since(prev_time).as_millis() < 300
                } else {
                    false
                }
            });

            if is_double {
                // Enter edit mode
                editing.set(true);
                LAST_CLICK.with(|lc| lc.set(None));
                return;
            }

            // Start drag
            let sv = untracked(|| value.get());
            start_val.set(sv);
            start_mouse_x.set(ctx.mouse_x);

            start_drag_absolute(move |mx, _my| {
                let mods = rinch::core::get_modifier_state();
                let speed = if mods.shift {
                    0.1
                } else if mods.ctrl {
                    10.0
                } else {
                    1.0
                };
                let dx = (mx - start_mouse_x.get()) as f64;
                let new_val = (sv + dx * step * speed).clamp(min, max);
                value.set(new_val);
            });
        });
        display_div.set_attribute("data-rid", &click_handler.to_string());

        container.append_child(&display_div);

        // --- Input field (editing mode) ---
        let input_el = scope.create_element("input");
        input_el.set_attribute("type", "text");
        input_el.set_attribute(
            "style",
            "display: none; flex: 1; background: rgba(255,255,255,0.12); \
             border: 1px solid rgba(100,160,255,0.5); border-radius: 3px; \
             padding: 1px 4px; font-size: 11px; color: #fff; outline: none; \
             height: 18px; line-height: 18px; width: 100%; box-sizing: border-box;",
        );
        input_el.set_attribute(
            "value",
            &format!(
                "{:.prec$}",
                untracked(|| value.get()),
                prec = decimals as usize
            ),
        );

        // Input handler — update value on Enter/change
        let input_handler = scope.register_input_handler(move |text: String| {
            if let Ok(v) = text.parse::<f64>() {
                if v.is_finite() {
                    value.set(v.clamp(min, max));
                }
            }
            editing.set(false);
        });
        input_el.set_attribute("data-oninput", &input_handler.to_string());

        container.append_child(&input_el);

        // --- Reactive effects ---

        // Update display text when value changes
        let value_text_clone = value_text.clone();
        let format_fn = {
            let suffix = suffix2;
            move |v: f64| -> String {
                format!("{:.prec$}{}", v, suffix, prec = decimals as usize)
            }
        };
        scope.create_effect(move || {
            let v = value.get();
            value_text_clone.set_text(&format_fn(v));
        });

        // Toggle display/input visibility when editing changes
        let display_clone = display_div.clone();
        let input_clone = input_el.clone();
        scope.create_effect(move || {
            let is_editing = editing.get();
            if is_editing {
                display_clone.set_style("display", "none");
                input_clone.set_style("display", "block");
                // Pre-fill current value
                input_clone.set_attribute(
                    "value",
                    &format!(
                        "{:.prec$}",
                        untracked(|| value.get()),
                        prec = decimals as usize
                    ),
                );
            } else {
                display_clone.set_style("display", "block");
                input_clone.set_style("display", "none");
            }
        });

        container
    }
}
