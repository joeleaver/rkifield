//! Welcome screen overlay — shown when no project is loaded.
//!
//! Displays recent projects with validation, plus New Project / Open Project buttons.
//! Hides reactively when `ui.project_loaded` becomes true.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_config::RecentProject;
use crate::editor_state::UiSignals;

/// Welcome screen overlay component.
///
/// Renders as an absolute overlay covering the entire editor. Conditionally
/// visible based on `ui.project_loaded` — when a project loads, the overlay
/// hides automatically via reactive signal.
#[component]
pub fn WelcomeScreen() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd_tx = use_context::<crate::CommandSender>().0.clone();

    rsx! {
        div {
            style: {
                let ui = ui;
                move || {
                    let loaded = ui.project_loaded.get();
                    if loaded {
                        "display:none;".to_string()
                    } else {
                        "position:absolute;z-index:100;top:36px;left:0;right:0;bottom:0;\
                         background:var(--rinch-color-dark-9);\
                         display:flex;align-items:center;justify-content:center;".to_string()
                    }
                }
            },

            // Center card
            div {
                style: "width:480px;max-height:80vh;display:flex;flex-direction:column;\
                        background:var(--rinch-color-dark-8);border:1px solid var(--rinch-color-border);\
                        border-radius:8px;overflow:hidden;",

                // Header
                div {
                    style: "padding:24px 28px 16px;",
                    div {
                        style: "font-size:18px;font-weight:600;color:var(--rinch-color-text);\
                                margin-bottom:4px;",
                        "RKIField"
                    }
                    div {
                        style: "font-size:12px;color:var(--rinch-color-dimmed);",
                        "Open a recent project or create a new one"
                    }
                }

                // Divider
                div { style: "height:1px;background:var(--rinch-color-border);" }

                // Recent projects list
                div {
                    style: "flex:1;overflow-y:auto;padding:8px 0;",

                    for project in ui.recent_projects.get() {
                        RecentProjectRow {
                            key: project.path.clone(),
                            project: project.clone(),
                        }
                    }

                    // Empty state
                    if ui.recent_projects.get().is_empty() {
                        div {
                            style: "padding:24px 28px;text-align:center;\
                                    font-size:12px;color:var(--rinch-color-dimmed);",
                            "No recent projects"
                        }
                    }
                }

                // Divider
                div { style: "height:1px;background:var(--rinch-color-border);" }

                // Action buttons
                div {
                    style: "display:flex;gap:8px;padding:16px 28px;",

                    Button {
                        variant: "filled",
                        onclick: {
                            let cmd_tx = cmd_tx.clone();
                            move || {
                                let _ = cmd_tx.send(EditorCommand::NewProject);
                            }
                        },
                        "New Project"
                    }

                    Button {
                        variant: "light",
                        onclick: {
                            let cmd_tx = cmd_tx.clone();
                            move || {
                                let _ = cmd_tx.send(EditorCommand::OpenProject {
                                    path: String::new(),
                                });
                            }
                        },
                        "Open Project..."
                    }
                }
            }
        }
    }
}

/// A single row in the recent projects list.
#[component]
fn RecentProjectRow(project: RecentProject) -> NodeHandle {
    let cmd_tx = use_context::<crate::CommandSender>().0.clone();

    // Check if the .rkproject file exists.
    let valid = std::path::Path::new(&project.path).exists();

    let path_for_open = project.path.clone();
    let path_for_remove = project.path.clone();

    let opacity = if valid { "1.0" } else { "0.5" };
    let cursor = if valid { "pointer" } else { "default" };
    let dot_color = if valid { "#4caf50" } else { "#f44336" };
    let last_opened = project.last_opened.clone();

    rsx! {
        div {
            style: format!(
                "display:flex;align-items:center;padding:8px 28px;gap:12px;\
                 opacity:{opacity};cursor:{cursor};"
            ),

            // Click to open
            onclick: {
                let cmd_tx = cmd_tx.clone();
                let path = path_for_open;
                move || {
                    if valid {
                        let _ = cmd_tx.send(EditorCommand::OpenProject {
                            path: path.clone(),
                        });
                    }
                }
            },

            // Validity dot
            div {
                style: format!(
                    "width:8px;height:8px;border-radius:50%;flex-shrink:0;\
                     background:{dot_color};"
                ),
            }

            // Name and path
            div {
                style: "flex:1;min-width:0;",
                div {
                    style: "font-size:13px;color:var(--rinch-color-text);\
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;",
                    {project.name.clone()}
                }
                div {
                    style: "font-size:11px;color:var(--rinch-color-dimmed);\
                            white-space:nowrap;overflow:hidden;text-overflow:ellipsis;\
                            font-family:var(--rinch-font-family-monospace);",
                    {project.path.clone()}
                }
            }

            // Date
            div {
                style: "font-size:11px;color:var(--rinch-color-dimmed);flex-shrink:0;",
                {last_opened}
            }

            // Remove button
            Button {
                variant: "subtle",
                size: "xs",
                onclick: {
                    let cmd_tx = cmd_tx.clone();
                    let path = path_for_remove;
                    move || {
                        let _ = cmd_tx.send(EditorCommand::RemoveRecentProject {
                            path: path.clone(),
                        });
                    }
                },
                "x"
            }
        }
    }
}
