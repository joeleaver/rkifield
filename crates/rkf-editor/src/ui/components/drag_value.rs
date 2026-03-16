//! DragValue — Unity-style number field with drag-to-adjust and double-click-to-edit.
//!
//! Single source of truth: reads from a Memo<f64> (derived, read-only).
//! Writes via on_change/on_commit callbacks. Never calls .set() on the value.

use std::cell::Cell;
use rinch::prelude::*;

/// Unity-style number field: displays value as text, click-drag left/right to adjust,
/// double-click to type a new value.
///
/// # Data flow
///
/// - **Read**: `value.get()` returns the current authoritative value (Memo from UiSignals).
/// - **During drag**: a local signal holds the live preview value for display.
/// - **on_change**: fires every drag tick (for live viewport feedback).
/// - **on_commit**: fires on drag end / Enter key (for final command).
pub struct DragValue {
    /// Read-only reactive value. Use Memo for engine-derived data,
    /// Signal for UI-owned data. DragValue never calls .set() on this.
    pub value: Memo<f64>,
    /// Called on every drag tick with the new value (live preview).
    pub on_change: Option<ValueCallback<f64>>,
    /// Called when editing is done (drag end, Enter key).
    pub on_commit: Option<ValueCallback<f64>>,
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

impl std::fmt::Debug for DragValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DragValue")
            .field("step", &self.step)
            .field("decimals", &self.decimals)
            .field("label", &self.label)
            .finish()
    }
}

impl Default for DragValue {
    fn default() -> Self {
        Self {
            value: Memo::new(|| 0.0),
            on_change: None,
            on_commit: None,
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

impl DragValue {
    /// Create a DragValue backed by a Signal (for environment/camera properties
    /// where the UI IS the authority). The on_change callback sets the signal
    /// and optionally fires an additional callback.
    pub fn from_signal(signal: Signal<f64>, on_change: Option<ValueCallback<f64>>) -> Self {
        Self {
            value: Memo::new(move || signal.get()),
            on_change: {
                let on_change = on_change.clone();
                Some(ValueCallback::new(move |v: f64| {
                    signal.set(v);
                    if let Some(cb) = &on_change {
                        cb.0(v);
                    }
                }))
            },
            on_commit: None,
            ..Default::default()
        }
    }
}

impl Component for DragValue {
    fn render(&self, __scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let value = self.value;  // Memo is Copy
        let step = self.step;
        let min = self.min;
        let max = self.max;
        let decimals = self.decimals;
        let on_change = self.on_change.clone();
        let on_commit = self.on_commit.clone();
        let on_commit2 = on_commit.clone();
        let suffix = self.suffix.clone();

        // Internal state
        let editing = Signal::new(false);
        let dragging = Signal::new(false);
        let drag_value = Signal::new(0.0_f64);
        let start_mouse_x = Signal::new(0.0_f32);

        // Double-click detection
        thread_local! {
            static LAST_CLICK: Cell<Option<std::time::Instant>> = const { Cell::new(None) };
        }

        // --- Container ---
        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display: flex; align-items: center; gap: 2px; min-width: 60px; height: 20px;",
        );

        // --- Label ---
        if !self.label.is_empty() {
            let label_span = __scope.create_element("span");
            label_span.set_attribute("style", &format!(
                "color: {}; font-weight: 600; font-size: 11px; width: 12px; text-align: center; \
                 user-select: none; flex-shrink: 0;",
                if self.label_color.is_empty() { "#aaa" } else { &self.label_color }
            ));
            label_span.append_child(&__scope.create_text(&self.label));
            container.append_child(&label_span);
        }

        // --- Display div (imperative to avoid rsx! move conflicts) ---
        let display_style_base = "flex: 1; background: rgba(255,255,255,0.06); border-radius: 3px; \
             padding: 1px 4px; font-size: 11px; color: #ddd; cursor: ew-resize; \
             user-select: none; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; \
             height: 18px; line-height: 18px;";

        let display_div = __scope.create_element("div");
        display_div.set_attribute("style", display_style_base);

        // Reactive style: hide when editing.
        {
            let dd = display_div.clone();
            let base = display_style_base.to_string();
            __scope.create_effect(move || {
                if editing.get() {
                    dd.set_attribute("style", &format!("display: none; {base}"));
                } else {
                    dd.set_attribute("style", &format!("display: block; {base}"));
                }
            });
        }

        // Reactive text: show drag value during drag, authoritative value otherwise.
        {
            let text_node = __scope.create_text("");
            let tn = text_node.clone();
            let suffix = suffix.clone();
            __scope.create_effect(move || {
                let v = if dragging.get() { drag_value.get() } else { value.get() };
                tn.set_text(&format!("{:.prec$}{suffix}", v, prec = decimals as usize));
            });
            display_div.append_child(&text_node);
        }

        // Click handler: double-click to edit, single-click to drag.
        let click_handler = __scope.register_handler(move || {
            let ctx = get_click_context();

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
                editing.set(true);
                LAST_CLICK.with(|lc| lc.set(None));
                return;
            }

            // Start drag
            let sv = value.get();
            drag_value.set(sv);
            dragging.set(true);
            start_mouse_x.set(ctx.mouse_x);

            Drag::absolute()
                .on_move({
                    let on_change = on_change.clone();
                    move |mx, _my| {
                        let mods = rinch::core::get_modifier_state();
                        let speed = if mods.shift { 0.1 } else if mods.ctrl { 10.0 } else { 1.0 };
                        let dx = (mx - start_mouse_x.get()) as f64;
                        let new_val = (sv + dx * step * speed).clamp(min, max);
                        drag_value.set(new_val);
                        if let Some(cb) = &on_change { cb.0(new_val); }
                    }
                })
                .on_end({
                    let on_commit = on_commit.clone();
                    move |_mx, _my| {
                        let final_val = drag_value.get();
                        dragging.set(false);
                        if let Some(cb) = &on_commit { cb.0(final_val); }
                    }
                })
                .start();
        });
        display_div.set_attribute("data-rid", &click_handler.to_string());
        container.append_child(&display_div);

        // --- Input field (editing mode) ---
        let input_el = __scope.create_element("input");
        input_el.set_attribute("type", "text");
        input_el.set_attribute("style",
            "display: none; flex: 1; background: rgba(255,255,255,0.12); \
             border: 1px solid rgba(100,160,255,0.5); border-radius: 3px; \
             padding: 1px 4px; font-size: 11px; color: #fff; outline: none; \
             height: 18px; line-height: 18px; width: 100%; box-sizing: border-box;");

        let draft = Signal::new(String::new());

        let input_handler = __scope.register_input_handler(move |text: String| {
            draft.set(text);
        });
        input_el.set_attribute("data-oninput", &input_handler.to_string());

        let submit_handler = __scope.register_handler(move || {
            let text = draft.get();
            if let Ok(v) = text.parse::<f64>() {
                if v.is_finite() {
                    let clamped = v.clamp(min, max);
                    if let Some(cb) = &on_commit2 { cb.0(clamped); }
                }
            }
            editing.set(false);
        });
        input_el.set_attribute("data-onsubmit", &submit_handler.to_string());

        // Toggle input visibility on editing change.
        {
            let input_clone = input_el.clone();
            __scope.create_effect(move || {
                if editing.get() {
                    let current = format!("{:.prec$}", value.get(), prec = decimals as usize);
                    draft.set(String::new());
                    input_clone.set_attribute("value", "");
                    input_clone.set_attribute("placeholder", &current);
                    input_clone.set_style("display", "block");
                    rinch::core::request_focus(input_clone.node_id().0);
                } else {
                    input_clone.set_style("display", "none");
                }
            });
        }

        container.append_child(&input_el);
        container
    }
}
