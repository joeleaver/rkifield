//! Console panel — unified log for scripts, engine, and build output.
//!
//! Displays console entries in a scrollable list. Entries are styled by
//! level (info=default, warn=yellow, error=red). Filter buttons in the
//! header toggle visibility of each level. A "Clear" button dismisses
//! all entries.
//!
//! Auto-scrolls to bottom when new messages arrive, unless the user has
//! manually scrolled up. A "scroll to bottom" indicator appears when
//! auto-scroll is disengaged.

use rinch::prelude::*;

use crate::store::UiStore;
use rkf_runtime::behavior::{ConsoleEntry, ConsoleFilter, ConsoleLevel};

// ── Style constants ────────────────────────────────────────────────────────

const HEADER_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    font-size:10px;color:var(--rinch-color-placeholder);\
    font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
    border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;";

const ROW_STYLE: &str = "\
    display:flex;align-items:flex-start;gap:8px;\
    padding:3px 12px;font-size:11px;\
    font-family:var(--rinch-font-family-monospace);\
    border-bottom:1px solid var(--rinch-color-border-subtle, rgba(255,255,255,0.04));";

const SEVERITY_DOT_STYLE: &str = "\
    width:6px;height:6px;border-radius:50%;flex-shrink:0;margin-top:5px;";

const MESSAGE_STYLE: &str = "\
    flex:1;min-width:0;color:var(--rinch-color-text);\
    white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text;";

const MESSAGE_ERROR_STYLE: &str = "\
    flex:1;min-width:0;color:var(--rinch-color-red-5, #ff6b6b);\
    white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text;";

const MESSAGE_WARNING_STYLE: &str = "\
    flex:1;min-width:0;color:var(--rinch-color-yellow-5, #ffd43b);\
    white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text;";

const LOCATION_STYLE: &str = "\
    flex-shrink:0;color:var(--rinch-color-dimmed);font-size:10px;\
    white-space:nowrap;margin-top:1px;user-select:text;cursor:text;";

const EMPTY_MSG_STYLE: &str = "\
    font-size:11px;color:var(--rinch-color-placeholder);padding:12px;";

const CLEAR_BTN_STYLE: &str = "\
    font-size:10px;color:var(--rinch-color-dimmed);\
    background:none;border:1px solid var(--rinch-color-border);\
    border-radius:3px;padding:1px 8px;cursor:pointer;\
    font-family:var(--rinch-font-family-monospace);";

const FILTER_BTN_BASE: &str = "\
    font-size:10px;border-radius:3px;padding:1px 8px;cursor:pointer;\
    font-family:var(--rinch-font-family-monospace);border:1px solid var(--rinch-color-border);";

const TIMESTAMP_STYLE: &str = "\
    flex-shrink:0;color:var(--rinch-color-dimmed);font-size:10px;\
    white-space:nowrap;margin-top:1px;min-width:48px;user-select:text;cursor:text;";

// ── Component ──────────────────────────────────────────────────────────────

/// Console panel — lists info, warning, and error messages from scripts and builds.
#[component]
pub fn DebugPanel() -> NodeHandle {
    let store = use_context::<UiStore>();

    let entries_sig = store.read_typed::<Vec<ConsoleEntry>>("console/entries");
    let filter_sig = store.read_typed::<ConsoleFilter>("console/filter");

    let root = rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;" }
    };

    // Header bar: filter buttons + counts + clear button.
    let header = rsx! {
        div { style: HEADER_STYLE }
    };

    // Filter toggle buttons.
    let info_btn = rsx! {
        button {
            style: {
                let filter_sig = filter_sig;
                move || {
                    let f = filter_sig.get();
                    if f.show_info {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-blue-5, #339af0);background:rgba(51,154,240,0.15);")
                    } else {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                    }
                }
            },
            onclick: {
                let store = store.clone();
                let filter_sig = filter_sig;
                move || {
                    let mut f = filter_sig.get();
                    f.show_info = !f.show_info;
                    store.set_typed::<ConsoleFilter>("console/filter", f);
                }
            },
            "Info"
        }
    };
    header.append_child(&info_btn);

    let warn_btn = rsx! {
        button {
            style: {
                let filter_sig = filter_sig;
                move || {
                    let f = filter_sig.get();
                    if f.show_warn {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-yellow-5, #ffd43b);background:rgba(255,212,59,0.15);")
                    } else {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                    }
                }
            },
            onclick: {
                let store = store.clone();
                let filter_sig = filter_sig;
                move || {
                    let mut f = filter_sig.get();
                    f.show_warn = !f.show_warn;
                    store.set_typed::<ConsoleFilter>("console/filter", f);
                }
            },
            "Warn"
        }
    };
    header.append_child(&warn_btn);

    let error_btn = rsx! {
        button {
            style: {
                let filter_sig = filter_sig;
                move || {
                    let f = filter_sig.get();
                    if f.show_error {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-red-5, #ff6b6b);background:rgba(255,107,107,0.15);")
                    } else {
                        format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                    }
                }
            },
            onclick: {
                let store = store.clone();
                let filter_sig = filter_sig;
                move || {
                    let mut f = filter_sig.get();
                    f.show_error = !f.show_error;
                    store.set_typed::<ConsoleFilter>("console/filter", f);
                }
            },
            "Error"
        }
    };
    header.append_child(&error_btn);

    let count_text = rsx! {
        div {
            style: "flex:1;",
            {move || {
                let entries = entries_sig.get();
                let filter = filter_sig.get();
                let visible = entries.iter().filter(|e| filter.accepts(e.level)).count();
                let total = entries.len();
                if total == 0 {
                    "No messages".to_string()
                } else if visible == total {
                    format!("{total} message{}", if total == 1 { "" } else { "s" })
                } else {
                    format!("{visible}/{total} shown")
                }
            }}
        }
    };
    header.append_child(&count_text);

    let clear_btn = rsx! {
        button {
            style: {
                let entries_sig = entries_sig;
                move || {
                    if entries_sig.get().is_empty() {
                        format!("{CLEAR_BTN_STYLE}display:none;")
                    } else {
                        CLEAR_BTN_STYLE.to_string()
                    }
                }
            },
            onclick: {
                let store = store.clone();
                move || {
                    store.execute_action("console.clear");
                }
            },
            "Clear"
        }
    };
    header.append_child(&clear_btn);
    root.append_child(&header);

    // Empty state hint.
    let hint = rsx! {
        div {
            style: {
                let entries_sig = entries_sig;
                move || {
                    if entries_sig.get().is_empty() {
                        EMPTY_MSG_STYLE.to_string()
                    } else {
                        "display:none;".to_string()
                    }
                }
            },
            "Script output and build messages will appear here."
        }
    };
    root.append_child(&hint);

    // Scrollable list of console entries (filtered).
    let list = rsx! {
        div { style: "display:flex;flex-direction:column;overflow-y:auto;flex:1;min-height:0;" }
    };

    rinch::core::for_each_dom_typed(
        __scope,
        &list,
        move || {
            let entries = entries_sig.get();
            let filter = filter_sig.get();
            entries.into_iter().filter(|e| filter.accepts(e.level)).collect::<Vec<_>>()
        },
        |entry| console_entry_key(entry),
        move |entry, scope| render_console_row(scope, &entry),
    );

    root.append_child(&list);

    root
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Stable key for a console entry.
fn console_entry_key(entry: &ConsoleEntry) -> String {
    let level = match entry.level {
        ConsoleLevel::Info => "I",
        ConsoleLevel::Warn => "W",
        ConsoleLevel::Error => "E",
    };
    let loc = match (&entry.file, entry.line) {
        (Some(f), Some(l)) => format!("{f}:{l}"),
        (Some(f), None) => f.clone(),
        _ => String::new(),
    };
    format!("{level}-{:.3}-{}-{loc}", entry.timestamp, entry.message)
}

/// Render a single console entry row.
fn render_console_row(scope: &mut RenderScope, entry: &ConsoleEntry) -> NodeHandle {
    let row = scope.create_element("div");
    row.set_attribute("style", ROW_STYLE);

    // Timestamp.
    let ts_el = scope.create_element("div");
    ts_el.set_attribute("style", TIMESTAMP_STYLE);
    let secs = entry.timestamp;
    let mins = (secs / 60.0) as u32;
    let secs_rem = secs % 60.0;
    ts_el.append_child(&scope.create_text(&format!("{mins}:{secs_rem:05.2}")));
    row.append_child(&ts_el);

    // Severity dot.
    let dot = scope.create_element("div");
    let dot_bg = match entry.level {
        ConsoleLevel::Error => "background:var(--rinch-color-red-5, #ff6b6b);",
        ConsoleLevel::Warn => "background:var(--rinch-color-yellow-5, #ffd43b);",
        ConsoleLevel::Info => "background:var(--rinch-color-blue-5, #339af0);",
    };
    dot.set_attribute("style", &format!("{SEVERITY_DOT_STYLE}{dot_bg}"));
    row.append_child(&dot);

    // Message text.
    let msg_el = scope.create_element("div");
    let msg_style = match entry.level {
        ConsoleLevel::Error => MESSAGE_ERROR_STYLE,
        ConsoleLevel::Warn => MESSAGE_WARNING_STYLE,
        ConsoleLevel::Info => MESSAGE_STYLE,
    };
    msg_el.set_attribute("style", msg_style);
    msg_el.append_child(&scope.create_text(&entry.message));
    row.append_child(&msg_el);

    // File location (if available).
    if let Some(ref file) = entry.file {
        let loc_el = scope.create_element("div");
        loc_el.set_attribute("style", LOCATION_STYLE);
        let loc_text = match (entry.line, entry.column) {
            (Some(line), Some(col)) => format!("{file}:{line}:{col}"),
            (Some(line), None) => format!("{file}:{line}"),
            _ => file.clone(),
        };
        loc_el.append_child(&scope.create_text(&loc_text));
        row.append_child(&loc_el);
    }

    row.into()
}
