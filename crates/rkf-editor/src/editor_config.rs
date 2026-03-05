//! Persistent editor configuration — survives across sessions.
//!
//! Stored at `~/.config/rkifield/editor.ron`. Currently tracks only the
//! last-opened project path for auto-open on launch.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Persisted editor preferences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditorConfig {
    /// Path to the last-opened `.rkproject` file. If set, the editor will
    /// attempt to re-open this project on next launch.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_project_path: Option<String>,
}

/// Default config directory: `~/.config/rkifield/`.
fn config_dir() -> Option<PathBuf> {
    // Use $XDG_CONFIG_HOME if set, otherwise $HOME/.config.
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".config"))
        })?;
    Some(base.join("rkifield"))
}

/// Full path to the config file.
fn config_file_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join("editor.ron"))
}

/// Load the editor config from disk. Returns `Default` if the file doesn't
/// exist or can't be parsed.
pub fn load_editor_config() -> EditorConfig {
    let Some(path) = config_file_path() else {
        return EditorConfig::default();
    };
    match std::fs::read_to_string(&path) {
        Ok(text) => ron::from_str(&text).unwrap_or_default(),
        Err(_) => EditorConfig::default(),
    }
}

/// Save the editor config to disk. Creates the config directory if needed.
pub fn save_editor_config(config: &EditorConfig) {
    let Some(dir) = config_dir() else { return };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("Failed to create config dir: {e}");
        return;
    }
    let Some(path) = config_file_path() else { return };
    let pretty = ron::ser::PrettyConfig::default();
    match ron::ser::to_string_pretty(config, pretty) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&path, text) {
                log::warn!("Failed to write editor config: {e}");
            }
        }
        Err(e) => log::warn!("Failed to serialize editor config: {e}"),
    }
}

/// Update just the last-project field and save.
pub fn set_last_project(project_path: Option<&str>) {
    let mut config = load_editor_config();
    config.last_project_path = project_path.map(|s| s.to_string());
    save_editor_config(&config);
}
