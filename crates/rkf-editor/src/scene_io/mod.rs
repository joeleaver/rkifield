//! Scene I/O utilities for the RKIField editor.
//!
//! Provides recent files tracking, unsaved-changes state, and .rkf asset
//! export functions. Scene serialization uses the v3 entity-centric format
//! from `rkf_runtime::scene_file_v3`.

#![allow(dead_code)]

mod save;

pub use save::*;

/// An entry in the recent files list.
#[derive(Debug, Clone)]
pub struct RecentFileEntry {
    pub path: String,
    pub name: String,
    pub timestamp_ms: u64,
}

/// Tracks recently opened scene files (max 10).
#[derive(Debug)]
pub struct RecentFiles {
    entries: Vec<RecentFileEntry>,
}

const MAX_RECENT: usize = 10;

impl Default for RecentFiles {
    fn default() -> Self {
        Self::new()
    }
}

impl RecentFiles {
    /// Create an empty recent files list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a file to the recent list.
    pub fn add(&mut self, path: &str, name: &str, timestamp_ms: u64) {
        self.entries.retain(|e| e.path != path);
        self.entries.insert(
            0,
            RecentFileEntry {
                path: path.to_string(),
                name: name.to_string(),
                timestamp_ms,
            },
        );
        if self.entries.len() > MAX_RECENT {
            self.entries.truncate(MAX_RECENT);
        }
    }

    /// Remove a file from the recent list by path.
    pub fn remove(&mut self, path: &str) {
        self.entries.retain(|e| e.path != path);
    }

    /// Get the recent files list (most recent first).
    pub fn entries(&self) -> &[RecentFileEntry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Tracks whether the current scene has unsaved modifications.
#[derive(Debug, Clone)]
pub struct UnsavedChangesState {
    pub has_unsaved: bool,
}

impl Default for UnsavedChangesState {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsavedChangesState {
    pub fn new() -> Self {
        Self { has_unsaved: false }
    }

    pub fn mark_changed(&mut self) {
        self.has_unsaved = true;
    }

    pub fn mark_saved(&mut self) {
        self.has_unsaved = false;
    }

    pub fn needs_save(&self) -> bool {
        self.has_unsaved
    }
}
