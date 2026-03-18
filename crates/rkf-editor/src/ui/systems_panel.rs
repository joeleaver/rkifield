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
                if !ui.dylib_ready.get() {
                    return "Building scripts...".to_string();
                }
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

    // Play state hint — shown when not playing and dylib is ready.
    let hint = rsx! {
        div {
            style: {
                let ui2 = ui;
                move || {
                    if ui2.dylib_ready.get() && ui2.systems.get().is_empty() && !ui2.play_state.get() {
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
    // Build a flat list of renderable items from the systems signal.
    let list_container = rsx! {
        div { style: "display:flex;flex-direction:column;",
            for item in build_display_items(ui.systems.get()) {
                div {
                    key: item.key(),
                    DisplayItemRow { item: item }
                }
            }
        }
    };

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

impl Default for DisplayItem {
    fn default() -> Self {
        DisplayItem::PhaseHeader(String::new())
    }
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
#[component]
fn DisplayItemRow(item: DisplayItem) -> NodeHandle {
    match &item {
        DisplayItem::PhaseHeader(phase) => {
            let phase = phase.clone();
            rsx! {
                div { style: {PHASE_HEADER_STYLE}, {phase} }
            }
        }
        DisplayItem::SystemRow(sys) => {
            let order_text = format!("{}", sys.order);
            let dot_style: &'static str = if sys.faulted {
                const FAULTED: &str = concat!(
                    "width:6px;height:6px;border-radius:50%;flex-shrink:0;",
                    "background:var(--rinch-color-red-5, #ff6b6b);"
                );
                FAULTED
            } else {
                const HEALTHY: &str = concat!(
                    "width:6px;height:6px;border-radius:50%;flex-shrink:0;",
                    "background:var(--rinch-color-green-5, #51cf66);"
                );
                HEALTHY
            };
            let name_style = if sys.faulted {
                SYSTEM_NAME_FAULTED_STYLE
            } else {
                SYSTEM_NAME_STYLE
            };
            let name = sys.name.clone();
            let timing_text = match sys.last_frame_us {
                Some(us) if us >= 1000 => format!("{:.1}ms", us as f64 / 1000.0),
                Some(us) => format!("{}us", us),
                None => "--".to_string(),
            };

            rsx! {
                div { style: {SYSTEM_ROW_STYLE},
                    div { style: {ORDER_STYLE}, {order_text} }
                    div { style: {dot_style} }
                    div { style: {name_style}, {name} }
                    div { style: {TIMING_STYLE}, {timing_text} }
                }
            }
        }
    }
}
