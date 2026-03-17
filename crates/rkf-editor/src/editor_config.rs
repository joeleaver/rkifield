//! Persistent editor configuration — survives across sessions.
//!
//! Stored at `~/.config/rkifield/editor.ron`. Tracks recent projects for
//! the welcome screen and other editor preferences.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Maximum number of recent projects to keep.
const MAX_RECENT_PROJECTS: usize = 10;

/// A recent project entry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RecentProject {
    /// Display name (usually the directory name).
    pub name: String,
    /// Full path to the `.rkproject` file.
    pub path: String,
    /// ISO 8601 date string when last opened (e.g. "2026-03-15").
    #[serde(default)]
    pub last_opened: String,
}

/// Persisted editor preferences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditorConfig {
    /// Deprecated: path to the last-opened `.rkproject` file.
    /// Migrated to `recent_projects` on load.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_project_path: Option<String>,

    /// Recently opened projects, most-recent first.
    #[serde(default)]
    pub recent_projects: Vec<RecentProject>,
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
///
/// Performs migration from `last_project_path` to `recent_projects` if needed.
pub fn load_editor_config() -> EditorConfig {
    let Some(path) = config_file_path() else {
        return EditorConfig::default();
    };
    let mut config: EditorConfig = match std::fs::read_to_string(&path) {
        Ok(text) => ron::from_str(&text).unwrap_or_default(),
        Err(_) => EditorConfig::default(),
    };

    // Migrate: if recent_projects is empty but last_project_path is set,
    // create a single entry from it.
    if config.recent_projects.is_empty() {
        if let Some(ref lpp) = config.last_project_path {
            let name = std::path::Path::new(lpp)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("Project")
                .to_string();
            config.recent_projects.push(RecentProject {
                name,
                path: lpp.clone(),
                last_opened: String::new(),
            });
        }
    }

    config
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

/// Add a project to the recent list (or move it to the front if already present).
///
/// Also updates `last_project_path` for backwards compatibility.
pub fn add_recent_project(project_path: &str, name: &str) {
    let mut config = load_editor_config();

    // Remove any existing entry with the same path.
    config.recent_projects.retain(|rp| rp.path != project_path);

    // Prepend new entry.
    let today = today_string();
    config.recent_projects.insert(0, RecentProject {
        name: name.to_string(),
        path: project_path.to_string(),
        last_opened: today,
    });

    // Truncate.
    config.recent_projects.truncate(MAX_RECENT_PROJECTS);

    // Keep last_project_path in sync.
    config.last_project_path = Some(project_path.to_string());

    save_editor_config(&config);
}

/// Remove a project from the recent list.
pub fn remove_recent_project(project_path: &str) {
    let mut config = load_editor_config();
    config.recent_projects.retain(|rp| rp.path != project_path);
    if config.last_project_path.as_deref() == Some(project_path) {
        config.last_project_path = config.recent_projects.first().map(|rp| rp.path.clone());
    }
    save_editor_config(&config);
}

/// Update just the last-project field and save.
pub fn set_last_project(project_path: Option<&str>) {
    let mut config = load_editor_config();
    config.last_project_path = project_path.map(|s| s.to_string());
    save_editor_config(&config);
}

/// Get today's date as an ISO 8601 string (YYYY-MM-DD).
fn today_string() -> String {
    // Use SystemTime to avoid pulling in chrono.
    let now = std::time::SystemTime::now();
    let secs = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple days-since-epoch calculation.
    let days = secs / 86400;
    // Approximate date from days since 1970-01-01.
    let (y, m, d) = days_to_date(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn today_string_format() {
        let s = today_string();
        assert_eq!(s.len(), 10);
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
    }

    #[test]
    fn days_to_date_epoch() {
        assert_eq!(days_to_date(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_date_known() {
        // 2026-03-15 = day 20527 since epoch
        assert_eq!(days_to_date(20527), (2026, 3, 15));
    }

    #[test]
    fn migration_from_last_project_path() {
        let config: EditorConfig = ron::from_str(
            "(last_project_path: Some(\"/home/user/proj/proj.rkproject\"))"
        ).unwrap();
        assert!(config.recent_projects.is_empty());
        // Simulate migration
        let mut c = config;
        if c.recent_projects.is_empty() {
            if let Some(ref lpp) = c.last_project_path {
                c.recent_projects.push(RecentProject {
                    name: "proj".to_string(),
                    path: lpp.clone(),
                    last_opened: String::new(),
                });
            }
        }
        assert_eq!(c.recent_projects.len(), 1);
        assert_eq!(c.recent_projects[0].path, "/home/user/proj/proj.rkproject");
    }

    #[test]
    fn recent_projects_dedup_and_order() {
        // Simulate adding the same project twice — should end up once, at front.
        let mut config = EditorConfig::default();
        config.recent_projects.push(RecentProject {
            name: "A".into(),
            path: "/a.rkproject".into(),
            last_opened: "2026-01-01".into(),
        });
        config.recent_projects.push(RecentProject {
            name: "B".into(),
            path: "/b.rkproject".into(),
            last_opened: "2026-01-02".into(),
        });

        // Simulate add_recent_project logic for "A":
        config.recent_projects.retain(|rp| rp.path != "/a.rkproject");
        config.recent_projects.insert(0, RecentProject {
            name: "A".into(),
            path: "/a.rkproject".into(),
            last_opened: "2026-03-15".into(),
        });
        config.recent_projects.truncate(MAX_RECENT_PROJECTS);

        assert_eq!(config.recent_projects.len(), 2);
        assert_eq!(config.recent_projects[0].path, "/a.rkproject");
        assert_eq!(config.recent_projects[1].path, "/b.rkproject");
    }
}
