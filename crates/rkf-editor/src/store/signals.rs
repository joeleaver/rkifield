//! Signal cache and push batching for the UI store.
//!
//! `SignalCache` lazily creates `Signal<UiValue>` instances on first read,
//! keyed by path ID. `PushBuffer` is a thread-safe staging area for
//! engine→UI value pushes, drained on the main thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rinch::prelude::Signal;

use super::path::PathRegistry;
use super::types::UiValue;

/// Lazy signal cache — creates signals on first read, caches them by path ID.
pub struct SignalCache {
    signals: HashMap<u32, Signal<UiValue>>,
}

impl SignalCache {
    pub fn new() -> Self {
        Self {
            signals: HashMap::new(),
        }
    }

    /// Get or create a signal for the given path ID.
    /// First access creates a `Signal<UiValue>` initialized to `UiValue::None`.
    pub fn get_or_create(&mut self, path_id: u32) -> Signal<UiValue> {
        *self.signals.entry(path_id).or_insert_with(|| Signal::new(UiValue::None))
    }

    /// Update the signal for a path ID with a new value.
    /// No-op if the path ID has no signal (nothing is reading it).
    pub fn update(&mut self, path_id: u32, value: UiValue) {
        if let Some(signal) = self.signals.get(&path_id) {
            signal.set_if_changed(value);
        }
    }
}

/// Thread-safe staging buffer for engine→UI pushes.
pub type PushBuffer = Arc<Mutex<Vec<(String, UiValue)>>>;

/// Create a new push buffer.
pub fn new_push_buffer() -> PushBuffer {
    Arc::new(Mutex::new(Vec::new()))
}

/// Drain the push buffer and update signals.
/// Called on the UI thread via `run_on_main_thread`.
pub fn drain_push_buffer(
    buffer: &PushBuffer,
    registry: &mut PathRegistry,
    cache: &mut SignalCache,
) {
    let entries: Vec<(String, UiValue)> = {
        let mut buf = buffer.lock().expect("push buffer poisoned");
        buf.drain(..).collect()
    };

    for (path, value) in entries {
        if let Ok(id) = registry.intern(&path) {
            cache.update(id, value);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_or_create_returns_same_signal_for_same_id() {
        let mut cache = SignalCache::new();
        let s1 = cache.get_or_create(0);
        let s2 = cache.get_or_create(0);
        assert_eq!(s1.debug_id(), s2.debug_id());
    }

    #[test]
    fn get_or_create_initializes_to_none() {
        let mut cache = SignalCache::new();
        let signal = cache.get_or_create(42);
        assert_eq!(signal.get(), UiValue::None);
    }

    #[test]
    fn update_changes_signal_value() {
        let mut cache = SignalCache::new();
        let signal = cache.get_or_create(0);
        cache.update(0, UiValue::Float(3.14));
        assert_eq!(signal.get(), UiValue::Float(3.14));
    }

    #[test]
    fn update_unknown_id_is_noop() {
        let mut cache = SignalCache::new();
        // Should not panic.
        cache.update(999, UiValue::Bool(true));
    }

    #[test]
    fn drain_push_buffer_processes_all_entries() {
        let mut registry = PathRegistry::new();
        let mut cache = SignalCache::new();

        // Pre-create signals so update has something to write to.
        let id_fov = registry.intern("camera/fov").unwrap();
        let id_mode = registry.intern("editor/mode").unwrap();
        let sig_fov = cache.get_or_create(id_fov);
        let sig_mode = cache.get_or_create(id_mode);

        let buffer = new_push_buffer();
        {
            let mut buf = buffer.lock().unwrap();
            buf.push(("camera/fov".into(), UiValue::Float(90.0)));
            buf.push(("editor/mode".into(), UiValue::String("sculpt".into())));
        }

        drain_push_buffer(&buffer, &mut registry, &mut cache);

        assert_eq!(sig_fov.get(), UiValue::Float(90.0));
        assert_eq!(sig_mode.get(), UiValue::String("sculpt".into()));

        // Buffer should be empty after drain.
        assert!(buffer.lock().unwrap().is_empty());
    }

    #[test]
    fn drain_push_buffer_empty_is_noop() {
        let mut registry = PathRegistry::new();
        let mut cache = SignalCache::new();
        let buffer = new_push_buffer();

        // Should not panic.
        drain_push_buffer(&buffer, &mut registry, &mut cache);
    }

    #[test]
    fn drain_push_buffer_skips_invalid_paths() {
        let mut registry = PathRegistry::new();
        let mut cache = SignalCache::new();
        let buffer = new_push_buffer();

        {
            let mut buf = buffer.lock().unwrap();
            buf.push(("bogus_path".into(), UiValue::Float(1.0)));
            buf.push(("camera/fov".into(), UiValue::Float(60.0)));
        }

        // Pre-create signal for the valid path.
        let id = registry.intern("camera/fov").unwrap();
        let sig = cache.get_or_create(id);

        drain_push_buffer(&buffer, &mut registry, &mut cache);

        // Invalid path was skipped, valid path was updated.
        assert_eq!(sig.get(), UiValue::Float(60.0));
    }

    #[test]
    fn drain_push_buffer_interns_new_paths() {
        let mut registry = PathRegistry::new();
        let mut cache = SignalCache::new();
        let buffer = new_push_buffer();

        {
            let mut buf = buffer.lock().unwrap();
            buf.push(("camera/fov".into(), UiValue::Float(75.0)));
        }

        // Path not pre-interned — drain should intern it.
        drain_push_buffer(&buffer, &mut registry, &mut cache);

        // Path should now be interned.
        assert!(registry.get_id("camera/fov").is_some());
    }
}
