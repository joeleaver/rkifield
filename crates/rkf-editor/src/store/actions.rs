//! Action registry — named editor actions with labels, shortcuts, and execute callbacks.
//!
//! Actions are the abstraction behind menu items, toolbar buttons, and keyboard
//! shortcuts. Each action has a unique string ID, a human-readable label, an
//! optional shortcut hint, and an execute callback that dispatches through the
//! `UiStore`.

use std::collections::HashMap;

/// A registered editor action — menu item, toolbar button, or shortcut.
pub struct Action {
    pub id: &'static str,
    pub label: &'static str,
    pub shortcut: Option<&'static str>,
    pub enabled: Option<fn() -> bool>,
    pub checked: Option<fn() -> bool>,
    pub execute: fn(&crate::store::UiStore),
}

/// Registry of all editor actions, keyed by string ID.
pub struct ActionRegistry {
    actions: Vec<Action>,
    id_to_index: HashMap<&'static str, usize>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// Register an action. Panics if an action with the same ID already exists.
    pub fn register(&mut self, action: Action) {
        if self.id_to_index.contains_key(action.id) {
            panic!("Duplicate action ID: {}", action.id);
        }
        let index = self.actions.len();
        self.id_to_index.insert(action.id, index);
        self.actions.push(action);
    }

    /// Look up an action by ID.
    pub fn get(&self, id: &str) -> Option<&Action> {
        self.id_to_index.get(id).map(|&i| &self.actions[i])
    }

    /// Execute an action by ID. Returns `false` if the action was not found.
    pub fn execute(&self, id: &str, store: &crate::store::UiStore) -> bool {
        if let Some(action) = self.get(id) {
            (action.execute)(store);
            true
        } else {
            false
        }
    }

    /// All registered actions, in registration order.
    pub fn all(&self) -> &[Action] {
        &self.actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_store() -> crate::store::UiStore {
        let (tx, _rx) = crossbeam::channel::unbounded();
        crate::store::UiStore::new(tx)
    }

    #[test]
    fn register_and_get() {
        let mut reg = ActionRegistry::new();
        reg.register(Action {
            id: "test.action",
            label: "Test",
            shortcut: Some("Ctrl+T"),
            enabled: None,
            checked: None,
            execute: |_| {},
        });
        let action = reg.get("test.action").unwrap();
        assert_eq!(action.id, "test.action");
        assert_eq!(action.label, "Test");
        assert_eq!(action.shortcut, Some("Ctrl+T"));
    }

    #[test]
    fn get_missing_returns_none() {
        let reg = ActionRegistry::new();
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn execute_returns_true_on_found() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static CALLED: AtomicBool = AtomicBool::new(false);

        let mut reg = ActionRegistry::new();
        reg.register(Action {
            id: "test.exec",
            label: "Exec",
            shortcut: None,
            enabled: None,
            checked: None,
            execute: |_| {
                CALLED.store(true, Ordering::SeqCst);
            },
        });

        let store = dummy_store();
        assert!(reg.execute("test.exec", &store));
        assert!(CALLED.load(Ordering::SeqCst));
    }

    #[test]
    fn execute_returns_false_on_missing() {
        let reg = ActionRegistry::new();
        let store = dummy_store();
        assert!(!reg.execute("nonexistent", &store));
    }

    #[test]
    #[should_panic(expected = "Duplicate action ID")]
    fn duplicate_detection() {
        let mut reg = ActionRegistry::new();
        let make = || Action {
            id: "dup",
            label: "Dup",
            shortcut: None,
            enabled: None,
            checked: None,
            execute: |_| {},
        };
        reg.register(make());
        reg.register(make());
    }

    #[test]
    fn all_returns_registered_actions() {
        let mut reg = ActionRegistry::new();
        reg.register(Action {
            id: "a",
            label: "A",
            shortcut: None,
            enabled: None,
            checked: None,
            execute: |_| {},
        });
        reg.register(Action {
            id: "b",
            label: "B",
            shortcut: None,
            enabled: None,
            checked: None,
            execute: |_| {},
        });
        assert_eq!(reg.all().len(), 2);
        assert_eq!(reg.all()[0].id, "a");
        assert_eq!(reg.all()[1].id, "b");
    }
}
