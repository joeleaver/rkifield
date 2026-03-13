//! Systems panel — shows registered behavior systems grouped by phase.
//!
//! Displays all systems from the GameplayRegistry with their execution order,
//! faulted status, and per-system frame timing. Systems are grouped by phase
//! (Update, LateUpdate) with section headers.
//!
//! Data is pushed from the engine thread into `UiSignals::systems` via
//! `run_on_main_thread` whenever play mode is active.

use rinch::prelude::*;

use crate::editor_state::UiSignals;
use crate::ui_snapshot::SystemSummary;

// ── Style constants ────────────────────────────────────────────────────────

const PHASE_HEADER_STYLE: &str = "\
    font-size:10px;color:var(--rinch-color-dimmed);text-transform:uppercase;\
    letter-spacing:1px;padding:6px 12px 2px 12px;\
    border-bottom:1px solid var(--rinch-color-border);";

const SYSTEM_ROW_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:3px 12px;font-size:11px;\
    font-family:var(--rinch-font-family-monospace);";

const SYSTEM_NAME_STYLE: &str = "\
    flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;\
    color:var(--rinch-color-text);";

const SYSTEM_NAME_FAULTED_STYLE: &str = "\
    flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;\
    color:var(--rinch-color-red-5, #ff6b6b);";

const ORDER_STYLE: &str = "\
    width:20px;text-align:right;color:var(--rinch-color-placeholder);\
    font-size:10px;flex-shrink:0;";

const TIMING_STYLE: &str = "\
    width:55px;text-align:right;color:var(--rinch-color-dimmed);\
    font-size:10px;flex-shrink:0;";

const FAULT_DOT_STYLE: &str = "\
    width:6px;height:6px;border-radius:50%;flex-shrink:0;";

const EMPTY_MSG_STYLE: &str = "\
    font-size:11px;color:var(--rinch-color-placeholder);padding:12px;";

// ── Component ──────────────────────────────────────────────────────────────

/// Systems panel — lists all registered behavior systems with phase grouping.
#[component]
pub fn SystemsPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let root = rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;overflow-y:auto;" }
    };

    // Header with system count.
    let header = rsx! {
        div {
            style: "font-size:10px;color:var(--rinch-color-placeholder);\
                    font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                    border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
            {move || {
                let systems = ui.systems.get();
                if systems.is_empty() {
                    "No systems registered".to_string()
                } else {
                    let faulted = systems.iter().filter(|s| s.faulted).count();
                    if faulted > 0 {
                        format!("{} systems ({} faulted)", systems.len(), faulted)
                    } else {
                        format!("{} systems", systems.len())
                    }
                }
            }}
        }
    };
    root.append_child(&header);

    // Play state hint — shown when not playing.
    let hint = rsx! {
        div {
            style: {
                let ui2 = ui;
                move || {
                    if ui2.systems.get().is_empty() && !ui2.play_state.get() {
                        EMPTY_MSG_STYLE.to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }
            },
            "Press Play (F5) to see system timings."
        }
    };
    root.append_child(&hint);

    // Systems list — rebuilt when the systems signal changes.
    let list_container = rsx! {
        div { style: "display:flex;flex-direction:column;" }
    };

    rinch::core::for_each_dom_typed(
        __scope,
        &list_container,
        move || build_display_items(ui.systems.get()),
        |item| item.key(),
        move |item, scope| {
            render_display_item(scope, &item)
        },
    );

    root.append_child(&list_container);
    root
}

// ── Display model ──────────────────────────────────────────────────────────

/// A display item in the systems list — either a phase header or a system row.
#[derive(Debug, Clone, PartialEq)]
enum DisplayItem {
    PhaseHeader(String),
    SystemRow(SystemSummary),
}

impl DisplayItem {
    fn key(&self) -> String {
        match self {
            DisplayItem::PhaseHeader(phase) => format!("hdr-{phase}"),
            DisplayItem::SystemRow(sys) => format!("sys-{}-{}", sys.phase, sys.order),
        }
    }
}

/// Build the flat list of display items from the systems signal value.
///
/// Groups systems by phase and inserts phase headers before each group.
fn build_display_items(systems: Vec<SystemSummary>) -> Vec<DisplayItem> {
    if systems.is_empty() {
        return Vec::new();
    }

    let mut items = Vec::with_capacity(systems.len() + 2);
    let mut current_phase: Option<String> = None;

    for sys in systems {
        if current_phase.as_ref() != Some(&sys.phase) {
            current_phase = Some(sys.phase.clone());
            items.push(DisplayItem::PhaseHeader(sys.phase.clone()));
        }
        items.push(DisplayItem::SystemRow(sys));
    }

    items
}

/// Render a single display item (phase header or system row).
fn render_display_item(scope: &mut RenderScope, item: &DisplayItem) -> NodeHandle {
    match item {
        DisplayItem::PhaseHeader(phase) => {
            let el = scope.create_element("div");
            el.set_attribute("style", PHASE_HEADER_STYLE);
            el.append_child(&scope.create_text(phase));
            el.into()
        }
        DisplayItem::SystemRow(sys) => {
            let row = scope.create_element("div");
            row.set_attribute("style", SYSTEM_ROW_STYLE);

            // Order index.
            let order_el = scope.create_element("div");
            order_el.set_attribute("style", ORDER_STYLE);
            order_el.append_child(&scope.create_text(&format!("{}", sys.order)));
            row.append_child(&order_el);

            // Fault indicator dot.
            let dot = scope.create_element("div");
            let dot_bg = if sys.faulted {
                "background:var(--rinch-color-red-5, #ff6b6b);"
            } else {
                "background:var(--rinch-color-green-5, #51cf66);"
            };
            dot.set_attribute("style", &format!("{FAULT_DOT_STYLE}{dot_bg}"));
            row.append_child(&dot);

            // System name.
            let name_el = scope.create_element("div");
            let name_style = if sys.faulted {
                SYSTEM_NAME_FAULTED_STYLE
            } else {
                SYSTEM_NAME_STYLE
            };
            name_el.set_attribute("style", name_style);
            name_el.append_child(&scope.create_text(&sys.name));
            row.append_child(&name_el);

            // Frame timing.
            let timing_el = scope.create_element("div");
            timing_el.set_attribute("style", TIMING_STYLE);
            let timing_text = match sys.last_frame_us {
                Some(us) if us >= 1000 => format!("{:.1}ms", us as f64 / 1000.0),
                Some(us) => format!("{}us", us),
                None => "--".to_string(),
            };
            timing_el.append_child(&scope.create_text(&timing_text));
            row.append_child(&timing_el);

            row.into()
        }
    }
}
