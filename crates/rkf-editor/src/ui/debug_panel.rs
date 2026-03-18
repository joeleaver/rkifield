//! Console panel — unified log for scripts, engine, and build output.
//!
//! Displays console entries in a scrollable list. Entries are styled by
//! level (info=default, warn=yellow, error=red). Filter buttons in the
//! header toggle visibility of each level. A "Clear" button dismisses
//! all entries.

use rinch::prelude::*;

use crate::editor_state::UiSignals;
use rkf_runtime::behavior::{ConsoleEntry, ConsoleLevel};

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
    white-space:pre-wrap;word-break:break-word;";

const MESSAGE_ERROR_STYLE: &str = "\
    flex:1;min-width:0;color:var(--rinch-color-red-5, #ff6b6b);\
    white-space:pre-wrap;word-break:break-word;";

const MESSAGE_WARNING_STYLE: &str = "\
    flex:1;min-width:0;color:var(--rinch-color-yellow-5, #ffd43b);\
    white-space:pre-wrap;word-break:break-word;";

const LOCATION_STYLE: &str = "\
    flex-shrink:0;color:var(--rinch-color-dimmed);font-size:10px;\
    white-space:nowrap;margin-top:1px;";

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
    white-space:nowrap;margin-top:1px;min-width:48px;";

// ── Component ──────────────────────────────────────────────────────────────

/// Console panel — lists info, warning, and error messages from scripts and builds.
#[component]
pub fn DebugPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let filtered_entries = Memo::new(move || {
        let entries = ui.console_entries.get();
        let filter = ui.console_filter.get();
        entries.into_iter().filter(|e| filter.accepts(e.level)).collect::<Vec<_>>()
    });

    rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;",

            // Header bar: filter buttons + counts + clear button.
            div { style: {HEADER_STYLE},

                // Info filter button.
                button {
                    style: {
                        let ui = ui;
                        move || {
                            let f = ui.console_filter.get();
                            if f.show_info {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-blue-5, #339af0);background:rgba(51,154,240,0.15);")
                            } else {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                            }
                        }
                    },
                    onclick: {
                        let ui = ui;
                        move || {
                            let mut f = ui.console_filter.get();
                            f.show_info = !f.show_info;
                            ui.console_filter.set(f);
                        }
                    },
                    "Info"
                }

                // Warn filter button.
                button {
                    style: {
                        let ui = ui;
                        move || {
                            let f = ui.console_filter.get();
                            if f.show_warn {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-yellow-5, #ffd43b);background:rgba(255,212,59,0.15);")
                            } else {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                            }
                        }
                    },
                    onclick: {
                        let ui = ui;
                        move || {
                            let mut f = ui.console_filter.get();
                            f.show_warn = !f.show_warn;
                            ui.console_filter.set(f);
                        }
                    },
                    "Warn"
                }

                // Error filter button.
                button {
                    style: {
                        let ui = ui;
                        move || {
                            let f = ui.console_filter.get();
                            if f.show_error {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-red-5, #ff6b6b);background:rgba(255,107,107,0.15);")
                            } else {
                                format!("{FILTER_BTN_BASE}color:var(--rinch-color-dimmed);background:none;")
                            }
                        }
                    },
                    onclick: {
                        let ui = ui;
                        move || {
                            let mut f = ui.console_filter.get();
                            f.show_error = !f.show_error;
                            ui.console_filter.set(f);
                        }
                    },
                    "Error"
                }

                // Count text.
                div {
                    style: "flex:1;",
                    {move || {
                        let entries = ui.console_entries.get();
                        let filter = ui.console_filter.get();
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

                // Clear button.
                button {
                    style: {
                        let ui = ui;
                        move || {
                            if ui.console_entries.get().is_empty() {
                                format!("{CLEAR_BTN_STYLE}display:none;")
                            } else {
                                CLEAR_BTN_STYLE.to_string()
                            }
                        }
                    },
                    onclick: {
                        let ui = ui;
                        move || {
                            ui.console_entries.set(Vec::new());
                        }
                    },
                    "Clear"
                }
            }

            // Empty state hint.
            div {
                style: {
                    let ui = ui;
                    move || {
                        if ui.console_entries.get().is_empty() {
                            EMPTY_MSG_STYLE.to_string()
                        } else {
                            "display:none;".to_string()
                        }
                    }
                },
                "Script output and build messages will appear here."
            }

            // Scrollable list of console entries (filtered).
            div { style: "display:flex;flex-direction:column;overflow-y:auto;flex:1;min-height:0;",
                for entry in filtered_entries.get() {
                    ConsoleRow {
                        key: console_entry_key(&entry),
                        timestamp: entry.timestamp,
                        level: match entry.level {
                            ConsoleLevel::Info => 0,
                            ConsoleLevel::Warn => 1,
                            ConsoleLevel::Error => 2,
                        },
                        message: entry.message.clone(),
                        location: entry.file.as_ref().map(|file| {
                            match (entry.line, entry.column) {
                                (Some(line), Some(col)) => format!("{file}:{line}:{col}"),
                                (Some(line), None) => format!("{file}:{line}"),
                                _ => file.clone(),
                            }
                        }).unwrap_or_default(),
                    }
                }
            }
        }
    }
}

/// Single console entry row component.
///
/// `level`: 0=Info, 1=Warn, 2=Error.
#[component]
fn ConsoleRow(
    timestamp: f64,
    level: u8,
    message: String,
    location: String,
) -> NodeHandle {
    let timestamp_text = format!("{}:{:05.2}", (timestamp / 60.0) as u32, timestamp % 60.0);
    let dot_style = format!("{SEVERITY_DOT_STYLE}{}", match level {
        2 => "background:var(--rinch-color-red-5, #ff6b6b);",
        1 => "background:var(--rinch-color-yellow-5, #ffd43b);",
        _ => "background:var(--rinch-color-blue-5, #339af0);",
    });
    let msg_style = match level {
        2 => MESSAGE_ERROR_STYLE,
        1 => MESSAGE_WARNING_STYLE,
        _ => MESSAGE_STYLE,
    };
    let loc_style = if location.is_empty() {
        "display:none;".to_string()
    } else {
        LOCATION_STYLE.to_string()
    };

    rsx! {
        div {
            style: {ROW_STYLE},

            // Timestamp.
            div { style: {TIMESTAMP_STYLE}, {timestamp_text} }

            // Severity dot.
            div { style: {dot_style.as_str()} }

            // Message text.
            div { style: {msg_style}, {message} }

            // File location (if available).
            div { style: {loc_style.as_str()}, {location} }
        }
    }
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
