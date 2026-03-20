//! UI Store — central reactive state management for the editor UI.
//!
//! All editor-visible state flows through the store. Widgets read via
//! `store.read(path)` and write via `store.set(path, value)`. The engine
//! thread pushes updates via the shared `PushBuffer`.
//!
//! See `docs/UI_STORE_ARCHITECTURE.md` for the full design.

pub mod actions;
pub mod path;
pub mod register_actions;
pub mod routing;
pub mod signals;
pub mod types;

use std::sync::Arc;

use crossbeam::channel::Sender;
use rinch::prelude::Signal;

use crate::editor_command::EditorCommand;
use actions::{Action, ActionRegistry};
use path::{PathRegistry, PathRoute};
use signals::{PushBuffer, SignalCache};
use types::UiValue;

/// Central UI store — owns reactive signals, path registry, and command routing.
///
/// Created on the main (UI) thread. Registered as a rinch context so any
/// component can `use_context::<UiStore>()`.
///
/// Thread safety: the store itself is NOT `Send`/`Sync` (signals are
/// thread-local). The engine thread communicates via the `PushBuffer`,
/// which is `Arc<Mutex<...>>`.
#[derive(Clone)]
pub struct UiStore {
    inner: std::rc::Rc<std::cell::RefCell<StoreInner>>,
    push_buffer: PushBuffer,
    cmd_tx: Sender<EditorCommand>,
    /// UUID of the active camera entity (for resolving `env/` paths).
    /// Updated by the engine loop push.
    active_camera: Signal<Option<uuid::Uuid>>,
    /// Registered editor actions (menu items, toolbar buttons, shortcuts).
    actions: std::rc::Rc<std::cell::RefCell<ActionRegistry>>,
}

struct StoreInner {
    registry: PathRegistry,
    cache: SignalCache,
}

impl UiStore {
    /// Create a new store with a shared push buffer.
    ///
    /// The `push_buffer` is shared with the engine thread — the engine writes
    /// to it, the store drains it on the main thread via `drain_pushes()`.
    /// Must be called on the main thread.
    pub fn new(cmd_tx: Sender<EditorCommand>, push_buffer: PushBuffer) -> Self {
        Self {
            inner: std::rc::Rc::new(std::cell::RefCell::new(StoreInner {
                registry: PathRegistry::new(),
                cache: SignalCache::new(),
            })),
            push_buffer,
            cmd_tx,
            active_camera: Signal::new(None),
            actions: std::rc::Rc::new(std::cell::RefCell::new(ActionRegistry::new())),
        }
    }

    /// Get the push buffer handle for the engine thread.
    pub fn push_buffer(&self) -> PushBuffer {
        self.push_buffer.clone()
    }

    /// Drain pending pushes from the engine thread and update signals.
    /// Must be called on the main thread (e.g. via `run_on_main_thread`).
    pub fn drain_pushes(&self) {
        let mut inner = self.inner.borrow_mut();
        let StoreInner { ref mut registry, ref mut cache } = *inner;
        signals::drain_push_buffer(&self.push_buffer, registry, cache);
    }

    /// Read a reactive signal for a store path.
    /// Creates the signal lazily on first access (initialized to `UiValue::None`).
    pub fn read(&self, path: &str) -> Signal<UiValue> {
        let mut inner = self.inner.borrow_mut();
        let id = match inner.registry.intern(path) {
            Ok(id) => id,
            Err(_) => return Signal::new(UiValue::None),
        };
        inner.cache.get_or_create(id)
    }

    /// Read a float value from a store path (convenience).
    pub fn read_float(&self, path: &str) -> f64 {
        self.read(path).get().as_float().unwrap_or(0.0)
    }

    /// Read a bool value from a store path (convenience).
    pub fn read_bool(&self, path: &str) -> bool {
        self.read(path).get().as_bool().unwrap_or(false)
    }

    /// Read a string value from a store path (convenience).
    pub fn read_string(&self, path: &str) -> String {
        self.read(path).get().as_string().unwrap_or_default().to_string()
    }

    /// Set the active camera UUID (called from engine loop push).
    pub fn set_active_camera(&self, uuid: Option<uuid::Uuid>) {
        self.active_camera.set(uuid);
    }

    /// Write a value to a store path.
    ///
    /// This does three things:
    /// 1. Updates the local signal immediately (responsive UI)
    /// 2. Converts the value to the appropriate type
    /// 3. Routes to an EditorCommand and sends it to the engine thread
    pub fn set(&self, path: &str, value: UiValue) {
        // Update local signal immediately.
        {
            let mut inner = self.inner.borrow_mut();
            if let Ok(id) = inner.registry.intern(path) {
                inner.cache.update(id, value.clone());
            }
        }

        // Parse path and route to command.
        let route = match path::parse_path(path) {
            Ok(r) => r,
            Err(_) => return,
        };

        // Resolve env/ shortcut to EcsField.
        let resolved = match route {
            PathRoute::EnvField { ref field } => {
                if let Some(cam_uuid) = self.active_camera.get() {
                    PathRoute::EcsField {
                        entity_id: cam_uuid,
                        component: "EnvironmentSettings".to_string(),
                        field: field.clone(),
                    }
                } else {
                    return; // No active camera — can't resolve
                }
            }
            other => other,
        };

        if let Some(cmd) = routing::route_write(&resolved, value) {
            let _ = self.cmd_tx.send(cmd);
        }
    }

    /// Dispatch an EditorCommand directly (for actions).
    pub fn dispatch(&self, cmd: EditorCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    // ── Action registry ──────────────────────────────────────────────

    /// Register an action in the action registry.
    pub fn register_action(&self, action: Action) {
        self.actions.borrow_mut().register(action);
    }

    /// Execute a registered action by ID. Returns `false` if not found.
    pub fn execute_action(&self, id: &str) -> bool {
        // Clone the Rc so the borrow is released before execute calls back into store.
        let registry = self.actions.clone();
        let reg = registry.borrow();
        if reg.get(id).is_some() {
            reg.execute(id, self);
            true
        } else {
            false
        }
    }

    /// Get action metadata by ID. Returns label, shortcut, checked fn, enabled fn.
    ///
    /// The returned struct borrows from the registry, so call within a
    /// short-lived scope (e.g. a render closure).
    pub fn get_action(&self, id: &str) -> Option<ActionMeta> {
        let reg = self.actions.borrow();
        reg.get(id).map(|a| ActionMeta {
            label: a.label,
            shortcut: a.shortcut,
            enabled: a.enabled,
            checked: a.checked,
        })
    }
}

/// Lightweight snapshot of action metadata, safe to hold after the registry
/// borrow is released.
pub struct ActionMeta {
    pub label: &'static str,
    pub shortcut: Option<&'static str>,
    pub enabled: Option<fn() -> bool>,
    pub checked: Option<fn() -> bool>,
}
