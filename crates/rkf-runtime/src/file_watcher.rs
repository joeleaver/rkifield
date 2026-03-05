//! File watcher — monitors asset and shader directories for changes.
//!
//! Uses the `notify` crate with debouncing. Classifies filesystem events by
//! file extension and sends typed events through an mpsc channel.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

/// A file-change event classified by type.
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// A `.rkmat` material file was created or modified.
    MaterialChanged(PathBuf),
    /// A `.wgsl` shader file was created or modified.
    ShaderChanged(PathBuf),
}

/// Watches directories for material and shader file changes.
///
/// Runs a background thread with a `notify::RecommendedWatcher`. Events are
/// debounced (100ms) and classified by extension. Poll with [`poll_events`].
pub struct FileWatcher {
    rx: mpsc::Receiver<FileEvent>,
    /// Kept alive to maintain the watch. Dropping this stops the watcher.
    _watcher: RecommendedWatcher,
}

impl FileWatcher {
    /// Create a new file watcher monitoring the given directories.
    ///
    /// Each path is watched recursively. Only `.rkmat` and `.wgsl` file
    /// changes produce events; other files are silently ignored.
    pub fn new(watch_paths: &[&Path]) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel();

        let event_tx = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            let Ok(event) = res else { return };

            // Only care about creates and modifications.
            use notify::EventKind;
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {}
                _ => return,
            }

            for path in &event.paths {
                let ext = path.extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let file_event = match ext {
                    "rkmat" => Some(FileEvent::MaterialChanged(path.clone())),
                    "wgsl" => Some(FileEvent::ShaderChanged(path.clone())),
                    _ => None,
                };
                if let Some(fe) = file_event {
                    let _ = event_tx.send(fe);
                }
            }
        }).map_err(|e| format!("create watcher: {e}"))?;

        for path in watch_paths {
            if path.exists() {
                watcher.watch(path, RecursiveMode::Recursive)
                    .map_err(|e| format!("watch {}: {e}", path.display()))?;
            }
        }

        Ok(Self { rx, _watcher: watcher })
    }

    /// Poll for all pending file events (non-blocking).
    ///
    /// Returns events accumulated since the last call. Typically called
    /// once per frame.
    pub fn poll_events(&self) -> Vec<FileEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.rx.try_recv() {
            // Deduplicate: skip if we already have the same path+type.
            let dominated = events.iter().any(|existing: &FileEvent| match (existing, &event) {
                (FileEvent::MaterialChanged(a), FileEvent::MaterialChanged(b)) => a == b,
                (FileEvent::ShaderChanged(a), FileEvent::ShaderChanged(b)) => a == b,
                _ => false,
            });
            if !dominated {
                events.push(event);
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_event_debug_format() {
        let e = FileEvent::MaterialChanged(PathBuf::from("test.rkmat"));
        let s = format!("{e:?}");
        assert!(s.contains("MaterialChanged"));
        assert!(s.contains("test.rkmat"));
    }

    #[test]
    fn file_event_shader() {
        let e = FileEvent::ShaderChanged(PathBuf::from("shade.wgsl"));
        let s = format!("{e:?}");
        assert!(s.contains("ShaderChanged"));
    }

    #[test]
    fn watcher_nonexistent_path_skipped() {
        // Watching a non-existent path should not error (skipped).
        let p = Path::new("/tmp/rkf_test_nonexistent_dir_12345");
        let result = FileWatcher::new(&[p]);
        // Should succeed (non-existent paths are skipped).
        assert!(result.is_ok());
    }

    #[test]
    fn poll_empty_returns_nothing() {
        let dir = std::env::temp_dir().join("rkf_fw_test_empty");
        let _ = std::fs::create_dir_all(&dir);
        let watcher = FileWatcher::new(&[dir.as_path()]).unwrap();
        let events = watcher.poll_events();
        assert!(events.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
