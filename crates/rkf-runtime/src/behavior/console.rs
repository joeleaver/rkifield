//! Console system — unified logging for scripts, engine, and build output.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Log level for console entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub enum ConsoleLevel {
    Info,
    Warn,
    Error,
}

/// A single console entry.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct ConsoleEntry {
    pub level: ConsoleLevel,
    pub message: String,
    /// Source file path (for compile errors).
    pub file: Option<String>,
    /// Line number (for compile errors).
    pub line: Option<u32>,
    /// Column number (for compile errors).
    pub column: Option<u32>,
    /// Seconds since engine start.
    pub timestamp: f64,
}

const MAX_ENTRIES: usize = 1000;

/// Thread-safe console buffer. Clone to share between threads.
#[derive(Clone)]
pub struct ConsoleBuffer {
    inner: Arc<Mutex<VecDeque<ConsoleEntry>>>,
    start: Instant,
    /// Monotonic revision counter, incremented on each push.
    revision: Arc<std::sync::atomic::AtomicU64>,
    /// Optional callback fired after each push (any thread).
    on_push: Arc<Mutex<Option<Box<dyn Fn() + Send>>>>,
}

impl ConsoleBuffer {
    /// Create a new empty console buffer.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_ENTRIES))),
            start: Instant::now(),
            revision: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            on_push: Arc::new(Mutex::new(None)),
        }
    }

    /// Set a callback that fires after each push.
    ///
    /// The callback runs on the pushing thread — use it to schedule
    /// a main-thread UI update (e.g. `run_on_main_thread`).
    pub fn set_on_push(&self, f: impl Fn() + Send + 'static) {
        if let Ok(mut cb) = self.on_push.lock() {
            *cb = Some(Box::new(f));
        }
    }

    /// Push a message at the given level.
    pub fn push(&self, level: ConsoleLevel, message: impl Into<String>) {
        let entry = ConsoleEntry {
            level,
            message: message.into(),
            file: None,
            line: None,
            column: None,
            timestamp: self.start.elapsed().as_secs_f64(),
        };
        self.push_entry(entry);
    }

    /// Push a pre-built entry (for compile errors with file:line).
    pub fn push_entry(&self, entry: ConsoleEntry) {
        if let Ok(mut buf) = self.inner.lock() {
            if buf.len() >= MAX_ENTRIES {
                buf.pop_front();
            }
            buf.push_back(entry);
        }
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(cb) = self.on_push.lock() {
            if let Some(ref f) = *cb {
                f();
            }
        }
    }

    /// Convenience: log an info message.
    pub fn info(&self, message: impl Into<String>) {
        self.push(ConsoleLevel::Info, message);
    }

    /// Convenience: log a warning.
    pub fn warn(&self, message: impl Into<String>) {
        self.push(ConsoleLevel::Warn, message);
    }

    /// Convenience: log an error.
    pub fn error(&self, message: impl Into<String>) {
        self.push(ConsoleLevel::Error, message);
    }

    /// Snapshot all entries (for UI sync).
    pub fn snapshot(&self) -> Vec<ConsoleEntry> {
        self.inner
            .lock()
            .map(|buf| buf.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Current revision number. Changes on every push.
    pub fn revision(&self) -> u64 {
        self.revision
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Clear all entries.
    pub fn clear(&self) {
        if let Ok(mut buf) = self.inner.lock() {
            buf.clear();
        }
        self.revision
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|buf| buf.len()).unwrap_or(0)
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ConsoleBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter state for the console panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct ConsoleFilter {
    pub show_info: bool,
    pub show_warn: bool,
    pub show_error: bool,
}

impl Default for ConsoleFilter {
    fn default() -> Self {
        Self {
            show_info: true,
            show_warn: true,
            show_error: true,
        }
    }
}

impl ConsoleFilter {
    /// Test whether an entry passes this filter.
    pub fn accepts(&self, level: ConsoleLevel) -> bool {
        match level {
            ConsoleLevel::Info => self.show_info,
            ConsoleLevel::Warn => self.show_warn,
            ConsoleLevel::Error => self.show_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_snapshot() {
        let buf = ConsoleBuffer::new();
        buf.info("hello");
        buf.warn("careful");
        buf.error("oops");
        let snap = buf.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].level, ConsoleLevel::Info);
        assert_eq!(snap[0].message, "hello");
        assert_eq!(snap[1].level, ConsoleLevel::Warn);
        assert_eq!(snap[2].level, ConsoleLevel::Error);
    }

    #[test]
    fn ring_buffer_eviction() {
        let buf = ConsoleBuffer::new();
        for i in 0..MAX_ENTRIES + 50 {
            buf.info(format!("msg {i}"));
        }
        assert_eq!(buf.len(), MAX_ENTRIES);
        let snap = buf.snapshot();
        assert_eq!(snap[0].message, "msg 50");
    }

    #[test]
    fn clear_empties() {
        let buf = ConsoleBuffer::new();
        buf.info("a");
        buf.info("b");
        assert_eq!(buf.len(), 2);
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn revision_increments() {
        let buf = ConsoleBuffer::new();
        let r0 = buf.revision();
        buf.info("x");
        assert!(buf.revision() > r0);
    }

    #[test]
    fn filter_accepts() {
        let f = ConsoleFilter {
            show_info: false,
            show_warn: true,
            show_error: true,
        };
        assert!(!f.accepts(ConsoleLevel::Info));
        assert!(f.accepts(ConsoleLevel::Warn));
        assert!(f.accepts(ConsoleLevel::Error));
    }

    #[test]
    fn clone_shares_buffer() {
        let a = ConsoleBuffer::new();
        let b = a.clone();
        a.info("from a");
        assert_eq!(b.len(), 1);
        assert_eq!(b.snapshot()[0].message, "from a");
    }
}
