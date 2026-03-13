//! Reload timing and queue for hot-reload.
//!
//! [`ReloadQueue`] buffers a pending dylib path for the next reload cycle.
//! Integration with play/edit mode happens in Phase 12.

use std::path::PathBuf;

// ─── ReloadQueue ────────────────────────────────────────────────────────────

/// A simple queue for pending dylib reloads.
///
/// Holds at most one pending reload path. Queueing a new path overwrites any
/// previous pending path (only the latest build matters).
#[derive(Debug, Default)]
pub struct ReloadQueue {
    /// Queued dylib path, if any.
    pending_reload: Option<PathBuf>,
}

impl ReloadQueue {
    /// Create an empty reload queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a dylib path for reload. Overwrites any previous pending path.
    pub fn queue_reload(&mut self, path: PathBuf) {
        self.pending_reload = Some(path);
    }

    /// Consume and return the pending reload path, if any.
    pub fn take_pending(&mut self) -> Option<PathBuf> {
        self.pending_reload.take()
    }

    /// Whether a reload is pending.
    pub fn has_pending(&self) -> bool {
        self.pending_reload.is_some()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_and_take() {
        let mut q = ReloadQueue::new();
        assert!(!q.has_pending());
        assert!(q.take_pending().is_none());

        q.queue_reload(PathBuf::from("/tmp/libgame.so"));
        assert!(q.has_pending());

        let path = q.take_pending();
        assert_eq!(path, Some(PathBuf::from("/tmp/libgame.so")));

        // After take, nothing pending.
        assert!(!q.has_pending());
        assert!(q.take_pending().is_none());
    }

    #[test]
    fn has_pending() {
        let mut q = ReloadQueue::new();
        assert!(!q.has_pending());

        q.queue_reload(PathBuf::from("/tmp/a.so"));
        assert!(q.has_pending());

        q.take_pending();
        assert!(!q.has_pending());
    }

    #[test]
    fn overwrite_pending() {
        let mut q = ReloadQueue::new();
        q.queue_reload(PathBuf::from("/tmp/first.so"));
        q.queue_reload(PathBuf::from("/tmp/second.so"));

        // Only the latest path is kept.
        assert_eq!(q.take_pending(), Some(PathBuf::from("/tmp/second.so")));
        assert!(q.take_pending().is_none());
    }
}
