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
        execute: |s| s.dispatch(EditorCommand::SaveScene { path: None }),
    });

    store.register_action(Action {
        id: "file.new_project",
        label: "New Project",
        shortcut: None,
        enabled: None,
        checked: None,
        execute: |s| s.dispatch(EditorCommand::NewProject),
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
}
