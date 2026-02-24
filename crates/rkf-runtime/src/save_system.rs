//! Save/load system (.rksave) for the v2 RKIField engine.
//!
//! Save files capture the full game state for resumption — camera position,
//! environment settings, entity overrides, and an arbitrary key-value store
//! for game-specific data.  Files are RON-serialized (human-readable and
//! diff-friendly) with a `.rksave` extension.

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ── Data types ─────────────────────────────────────────────────────────────────

/// Camera state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CameraSnapshot {
    /// World position in f64 metres `[x, y, z]`.
    pub position: [f64; 3],
    /// Orientation as a unit quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Vertical field of view in degrees.
    pub fov: f32,
}

/// Environment state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentSnapshot {
    /// Index of the currently active environment profile.
    pub active_profile: usize,
    /// Index of the blend target profile, if a transition is in progress.
    pub target_profile: Option<usize>,
    /// Blend progress in `[0.0, 1.0]`.  `0.0` means fully on `active_profile`.
    pub blend_t: f32,
    /// Any active per-parameter overrides.
    pub overrides: std::collections::HashMap<String, f32>,
}

/// Per-entity override — captures modifications made to a placed entity at
/// runtime (moved, hidden, custom script state, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityOverride {
    /// Display name of the entity this override belongs to.
    pub entity_name: String,
    /// Overridden world position in f64 metres, if changed.
    pub position: Option<[f64; 3]>,
    /// Overridden orientation as a unit quaternion `[x, y, z, w]`, if changed.
    pub rotation: Option<[f32; 4]>,
    /// Overridden uniform scale factor, if changed.
    pub scale: Option<f32>,
    /// Overridden visibility, if changed.
    pub visible: Option<bool>,
    /// Arbitrary key-value pairs for game-specific per-entity state.
    pub custom_data: std::collections::HashMap<String, String>,
}

/// Complete save file.  Written to disk as pretty-printed RON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveFile {
    /// Save format version.  Currently `1`.
    pub version: u32,
    /// Timestamp as an ISO 8601 string (e.g. `"2026-02-24T15:30:00"`).
    pub timestamp: String,
    /// Human-readable display name chosen by the player / auto-save system.
    pub save_name: String,
    /// Paths of scenes that were loaded when this save was created.
    pub loaded_scenes: Vec<String>,
    /// Camera state at save time.
    pub camera: CameraSnapshot,
    /// Environment state at save time.
    pub environment: EnvironmentSnapshot,
    /// Game state key-value store (serialized as a map).
    pub game_state: std::collections::HashMap<String, String>,
    /// Per-entity overrides (modified positions, visibility, custom data, etc.).
    pub entity_overrides: Vec<EntityOverride>,
    /// Accumulated play time in seconds at save time.
    pub play_time: f64,
}

// ── Brief save info (for UI display) ──────────────────────────────────────────

/// Brief summary of a save file used for UI listings.
///
/// Populated by [`list_saves`] without reading the full file payload.
#[derive(Debug, Clone)]
pub struct SaveInfo {
    /// Absolute (or caller-relative) path to the `.rksave` file.
    pub path: String,
    /// Display name stored in the save file.
    pub save_name: String,
    /// Timestamp string stored in the save file.
    pub timestamp: String,
    /// Accumulated play time in seconds.
    pub play_time: f64,
}

// ── Timestamp helper ───────────────────────────────────────────────────────────

/// Format a [`std::time::SystemTime`] as `"YYYY-MM-DDTHH:MM:SS"` without any
/// external date/time dependency.
fn format_system_time(t: std::time::SystemTime) -> String {
    // Seconds since the Unix epoch (1970-01-01T00:00:00 UTC).
    let secs = t
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Gregorian calendar decomposition (handles leap years correctly for the
    // range of dates we care about — well past 2100).
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400; // days since 1970-01-01

    // Shift epoch to 1 Mar 2000 to simplify leap-year arithmetic.
    // Algorithm: http://howardhinnant.github.io/date_algorithms.html
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        y, mo, d, h, m, s
    )
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Create a new [`SaveFile`] with the given display name and sensible defaults.
///
/// The timestamp is set to the current wall-clock time.  All other fields are
/// empty or zero — the caller fills in the actual game state before persisting.
pub fn create_save(name: &str) -> SaveFile {
    let timestamp = format_system_time(std::time::SystemTime::now());

    SaveFile {
        version: 1,
        timestamp,
        save_name: name.to_string(),
        loaded_scenes: Vec::new(),
        camera: CameraSnapshot {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            fov: 60.0,
        },
        environment: EnvironmentSnapshot {
            active_profile: 0,
            target_profile: None,
            blend_t: 0.0,
            overrides: std::collections::HashMap::new(),
        },
        game_state: std::collections::HashMap::new(),
        entity_overrides: Vec::new(),
        play_time: 0.0,
    }
}

/// Serialize a [`SaveFile`] to a `.rksave` file at `path` (pretty-printed RON).
pub fn save_game(path: &str, save: &SaveFile) -> Result<()> {
    let config = ron::ser::PrettyConfig::default();
    let text = ron::ser::to_string_pretty(save, config)?;
    std::fs::write(path, text)?;
    Ok(())
}

/// Deserialize a [`SaveFile`] from a `.rksave` file at `path`.
pub fn load_game(path: &str) -> Result<SaveFile> {
    let text = std::fs::read_to_string(path)?;
    let save: SaveFile = ron::from_str(&text)?;
    Ok(save)
}

/// List all `.rksave` files in `directory`, sorted by timestamp (newest first).
///
/// Corrupt or unreadable files are silently skipped.  Returns an empty `Vec`
/// if the directory contains no valid save files.
pub fn list_saves(directory: &str) -> Result<Vec<SaveInfo>> {
    let dir = std::path::Path::new(directory);
    let mut infos: Vec<SaveInfo> = Vec::new();

    let entries = std::fs::read_dir(dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if extension != "rksave" {
            continue;
        }

        // Deserialize the full file to extract name/timestamp/play_time.
        // Files that fail to parse are skipped gracefully.
        let path_str = match path.to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };

        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let save: SaveFile = match ron::from_str(&text) {
            Ok(s) => s,
            Err(_) => continue,
        };

        infos.push(SaveInfo {
            path: path_str,
            save_name: save.save_name,
            timestamp: save.timestamp,
            play_time: save.play_time,
        });
    }

    // Sort newest-first by timestamp string.  ISO 8601 strings sort
    // lexicographically in the same order as chronologically.
    infos.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(infos)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────────

    fn temp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(name)
            .to_str()
            .unwrap()
            .to_string()
    }

    fn sample_save() -> SaveFile {
        let mut save = create_save("Test Save");

        save.loaded_scenes = vec!["scenes/level1.rkscene".to_string()];

        save.camera = CameraSnapshot {
            position: [10.0, 5.0, -3.0],
            rotation: [0.0, 0.707, 0.0, 0.707],
            fov: 75.0,
        };

        save.environment = EnvironmentSnapshot {
            active_profile: 2,
            target_profile: Some(3),
            blend_t: 0.4,
            overrides: {
                let mut m = std::collections::HashMap::new();
                m.insert("fog_density".to_string(), 0.05);
                m
            },
        };

        save.game_state.insert("player_health".to_string(), "80".to_string());
        save.game_state.insert("score".to_string(), "12500".to_string());

        save.entity_overrides.push(EntityOverride {
            entity_name: "Crate_01".to_string(),
            position: Some([5.0, 0.0, 2.0]),
            rotation: None,
            scale: Some(1.5),
            visible: Some(false),
            custom_data: {
                let mut m = std::collections::HashMap::new();
                m.insert("opened".to_string(), "true".to_string());
                m
            },
        });

        save.play_time = 3600.0;
        save
    }

    // ── 1. create_save_defaults ────────────────────────────────────────────────

    #[test]
    fn create_save_defaults() {
        let save = create_save("Quick Save");

        assert_eq!(save.save_name, "Quick Save");
        assert_eq!(save.version, 1);
        assert!(save.loaded_scenes.is_empty());
        assert!(save.game_state.is_empty());
        assert!(save.entity_overrides.is_empty());
        assert_eq!(save.play_time, 0.0);
        assert_eq!(save.camera.position, [0.0, 0.0, 0.0]);
        assert_eq!(save.camera.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(save.environment.active_profile, 0);
        assert!(save.environment.target_profile.is_none());
        assert_eq!(save.environment.blend_t, 0.0);
        assert!(save.environment.overrides.is_empty());
    }

    // ── 2. save_and_load_roundtrip ────────────────────────────────────────────

    #[test]
    fn save_and_load_roundtrip() {
        let save = sample_save();
        let path = temp_path("rkf_test_roundtrip.rksave");

        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");

        assert_eq!(loaded.save_name, save.save_name);
        assert_eq!(loaded.version, save.version);
        assert_eq!(loaded.loaded_scenes, save.loaded_scenes);
        assert_eq!(loaded.play_time, save.play_time);

        let _ = std::fs::remove_file(&path);
    }

    // ── 3. camera_snapshot_preserved ─────────────────────────────────────────

    #[test]
    fn camera_snapshot_preserved() {
        let save = sample_save();
        let path = temp_path("rkf_test_camera.rksave");

        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");

        assert_eq!(loaded.camera.position, [10.0, 5.0, -3.0]);
        assert!((loaded.camera.rotation[1] - 0.707).abs() < 1e-4);
        assert!((loaded.camera.fov - 75.0).abs() < 1e-4);

        let _ = std::fs::remove_file(&path);
    }

    // ── 4. environment_snapshot_preserved ────────────────────────────────────

    #[test]
    fn environment_snapshot_preserved() {
        let save = sample_save();
        let path = temp_path("rkf_test_environment.rksave");

        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");

        assert_eq!(loaded.environment.active_profile, 2);
        assert_eq!(loaded.environment.target_profile, Some(3));
        assert!((loaded.environment.blend_t - 0.4).abs() < 1e-6);

        let fog = loaded.environment.overrides.get("fog_density").copied();
        assert!(fog.is_some());
        assert!((fog.unwrap() - 0.05).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    // ── 5. game_state_preserved ───────────────────────────────────────────────

    #[test]
    fn game_state_preserved() {
        let save = sample_save();
        let path = temp_path("rkf_test_game_state.rksave");

        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");

        assert_eq!(
            loaded.game_state.get("player_health").map(String::as_str),
            Some("80")
        );
        assert_eq!(
            loaded.game_state.get("score").map(String::as_str),
            Some("12500")
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── 6. entity_overrides_preserved ────────────────────────────────────────

    #[test]
    fn entity_overrides_preserved() {
        let save = sample_save();
        let path = temp_path("rkf_test_overrides.rksave");

        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");

        assert_eq!(loaded.entity_overrides.len(), 1);
        let ov = &loaded.entity_overrides[0];
        assert_eq!(ov.entity_name, "Crate_01");
        assert_eq!(ov.position, Some([5.0, 0.0, 2.0]));
        assert!(ov.rotation.is_none());
        assert_eq!(ov.scale, Some(1.5));
        assert_eq!(ov.visible, Some(false));
        assert_eq!(
            ov.custom_data.get("opened").map(String::as_str),
            Some("true")
        );

        let _ = std::fs::remove_file(&path);
    }

    // ── 7. save_file_version ──────────────────────────────────────────────────

    #[test]
    fn save_file_version() {
        let save = create_save("Version Check");
        assert_eq!(save.version, 1);

        // Version must survive a round-trip.
        let path = temp_path("rkf_test_version.rksave");
        save_game(&path, &save).expect("save_game");
        let loaded = load_game(&path).expect("load_game");
        assert_eq!(loaded.version, 1);

        let _ = std::fs::remove_file(&path);
    }

    // ── 8. timestamp_format ───────────────────────────────────────────────────

    #[test]
    fn timestamp_format() {
        let save = create_save("Timestamp Test");

        // Must be non-empty and look like an ISO 8601 datetime.
        assert!(!save.timestamp.is_empty());
        // Basic shape: "YYYY-MM-DDTHH:MM:SS" — 19 characters.
        assert_eq!(
            save.timestamp.len(),
            19,
            "unexpected timestamp length: {:?}",
            save.timestamp
        );
        // Contains the 'T' separator between date and time.
        assert!(save.timestamp.contains('T'));
    }

    // ── 9. list_saves_empty_dir ───────────────────────────────────────────────

    #[test]
    fn list_saves_empty_dir() {
        // Use a scratch subdirectory that contains no .rksave files.
        let dir = std::env::temp_dir().join("rkf_test_list_empty");
        std::fs::create_dir_all(&dir).expect("create dir");

        // Remove any stale .rksave files from previous runs.
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }

        let result = list_saves(dir.to_str().unwrap()).expect("list_saves");
        assert!(result.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 10. list_saves_finds_files ────────────────────────────────────────────

    #[test]
    fn list_saves_finds_files() {
        let dir = std::env::temp_dir().join("rkf_test_list_saves");
        std::fs::create_dir_all(&dir).expect("create dir");

        // Write two save files with different names.
        let mut save_a = create_save("Alpha");
        save_a.play_time = 100.0;
        // Force a later timestamp so sort order is predictable.
        save_a.timestamp = "2026-06-01T10:00:00".to_string();

        let mut save_b = create_save("Beta");
        save_b.play_time = 200.0;
        save_b.timestamp = "2026-06-02T10:00:00".to_string();

        let path_a = dir.join("save_a.rksave");
        let path_b = dir.join("save_b.rksave");

        save_game(path_a.to_str().unwrap(), &save_a).expect("save a");
        save_game(path_b.to_str().unwrap(), &save_b).expect("save b");

        let infos = list_saves(dir.to_str().unwrap()).expect("list_saves");
        assert_eq!(infos.len(), 2);

        // Newest first: Beta (2026-06-02) should come before Alpha (2026-06-01).
        assert_eq!(infos[0].save_name, "Beta");
        assert_eq!(infos[1].save_name, "Alpha");
        assert!((infos[0].play_time - 200.0).abs() < 1e-6);
        assert!((infos[1].play_time - 100.0).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
