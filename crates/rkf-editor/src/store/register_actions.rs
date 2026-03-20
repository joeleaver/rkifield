//! Core action registration — called once at startup to populate the action registry.

use crate::editor_command::EditorCommand;
use crate::store::actions::Action;
use crate::store::UiStore;

/// Register all core editor actions into the store's action registry.
pub fn register_core_actions(store: &UiStore) {
    // ── Edit ──────────────────────────────────────────────────────────
    store.register_action(Action {
        id: "edit.undo",
        label: "Undo",
        shortcut: Some("Ctrl+Z"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::Undo),
    });

    store.register_action(Action {
        id: "edit.redo",
        label: "Redo",
        shortcut: Some("Ctrl+Shift+Z"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::Redo),
    });

    store.register_action(Action {
        id: "edit.delete",
        label: "Delete",
        shortcut: Some("Delete"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::DeleteSelected),
    });

    store.register_action(Action {
        id: "edit.duplicate",
        label: "Duplicate",
        shortcut: Some("Ctrl+D"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::DuplicateSelected),
    });

    // ── File ──────────────────────────────────────────────────────────
    store.register_action(Action {
        id: "file.save",
        label: "Save",
        shortcut: Some("Ctrl+S"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::SaveProject),
    });

    store.register_action(Action {
        id: "file.new_project",
        label: "New Project",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::NewProject),
    });

    store.register_action(Action {
        id: "file.open_project",
        label: "Open Project",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::OpenProject { path: String::new() }),
    });

    store.register_action(Action {
        id: "file.open_scene",
        label: "Open Scene",
        shortcut: Some("Ctrl+O"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::OpenScene { path: String::new() }),
    });

    store.register_action(Action {
        id: "file.quit",
        label: "Quit",
        shortcut: Some("Esc"),
        enabled: None,
        checked: None,
        execute: |_| rinch::prelude::close_current_window(),
    });

    // ── View ──────────────────────────────────────────────────────────
    store.register_action(Action {
        id: "view.toggle_grid",
        label: "Toggle Grid",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::ToggleGrid),
    });

    store.register_action(Action {
        id: "view.toggle_shortcuts",
        label: "Toggle Shortcuts",
        shortcut: Some("F1"),
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::ToggleShortcuts),
    });

    // ── Editor mode ──────────────────────────────────────────────────
    store.register_action(Action {
        id: "mode.select",
        label: "Select",
        shortcut: Some("Escape"),
        enabled: None,
        checked: Some(|s| s.read_string("editor/mode") == "default"),
        execute: |s| s.dispatch(EditorCommand::SetEditorMode { mode: crate::editor_state::EditorMode::Default }),
    });

    store.register_action(Action {
        id: "mode.sculpt",
        label: "Sculpt",
        shortcut: None,
        enabled: None,
        checked: Some(|s| s.read_string("editor/mode") == "sculpt"),
        execute: |s| {
            let current = s.read_string("editor/mode");
            let mode = if current == "sculpt" {
                crate::editor_state::EditorMode::Default
            } else {
                crate::editor_state::EditorMode::Sculpt
            };
            s.dispatch(EditorCommand::SetEditorMode { mode });
        },
    });

    store.register_action(Action {
        id: "mode.paint",
        label: "Paint",
        shortcut: None,
        enabled: None,
        checked: Some(|s| s.read_string("editor/mode") == "paint"),
        execute: |s| {
            let current = s.read_string("editor/mode");
            let mode = if current == "paint" {
                crate::editor_state::EditorMode::Default
            } else {
                crate::editor_state::EditorMode::Paint
            };
            s.dispatch(EditorCommand::SetEditorMode { mode });
        },
    });

    // ── Gizmo mode ──────────────────────────────────────────────────
    store.register_action(Action {
        id: "gizmo.translate",
        label: "Move",
        shortcut: Some("G"),
        enabled: None,
        checked: Some(|s| s.read_string("gizmo/mode") == "translate"),
        execute: |s| s.dispatch(EditorCommand::SetGizmoMode { mode: crate::gizmo::GizmoMode::Translate }),
    });

    store.register_action(Action {
        id: "gizmo.rotate",
        label: "Rotate",
        shortcut: Some("R"),
        enabled: None,
        checked: Some(|s| s.read_string("gizmo/mode") == "rotate"),
        execute: |s| s.dispatch(EditorCommand::SetGizmoMode { mode: crate::gizmo::GizmoMode::Rotate }),
    });

    store.register_action(Action {
        id: "gizmo.scale",
        label: "Scale",
        shortcut: Some("L"),
        enabled: None,
        checked: Some(|s| s.read_string("gizmo/mode") == "scale"),
        execute: |s| s.dispatch(EditorCommand::SetGizmoMode { mode: crate::gizmo::GizmoMode::Scale }),
    });

    // ── Play mode ─────────────────────────────────────────────────────
    store.register_action(Action {
        id: "play.start",
        label: "Play",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::PlayStart),
    });

    store.register_action(Action {
        id: "play.stop",
        label: "Stop",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::PlayStop),
    });

    // ── Console ──────────────────────────────────────────────────────────
    store.register_action(Action {
        id: "console.clear",
        label: "Clear Console",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |_| {
            if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
                ui.console_entries.set(Vec::new());
            }
        },
    });
}
