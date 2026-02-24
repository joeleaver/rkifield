//! Project file format (.rkproject) — RON-serialized project descriptor.
//!
//! A project file is the top-level entry point for an RKIField project. It lists
//! every scene the project contains, configures asset search paths, and records the
//! engine version the project was last saved with.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Top-level project descriptor serialized to `.rkproject` (RON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    /// Human-readable project name.
    pub name: String,
    /// Engine version string this project was last saved with (e.g. `"0.1.0"`).
    pub engine_version: String,
    /// All scenes known to this project (relative paths from the project root).
    pub scenes: Vec<SceneRef>,
    /// Name of the scene to load automatically when the project is opened.
    /// Must match one of the `name` fields in `scenes`.
    #[serde(default)]
    pub default_scene: Option<String>,
    /// Directories to search for asset files (relative to the project root).
    #[serde(default)]
    pub asset_paths: Vec<String>,
    /// Default render-quality preset name (e.g. `"medium"`).
    #[serde(default = "default_quality")]
    pub default_quality: String,
}

fn default_quality() -> String {
    "medium".to_string()
}

/// A reference to a scene file within a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneRef {
    /// Display name used in the editor scene list.
    pub name: String,
    /// Path to the `.rkscene` file, relative to the project root.
    pub path: String,
    /// If `true` this scene stays loaded whenever the project is open (e.g. a
    /// shared lighting / environment scene).
    #[serde(default)]
    pub persistent: bool,
}

impl ProjectFile {
    /// Create a new project with sensible defaults and no scenes yet.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            scenes: Vec::new(),
            default_scene: None,
            asset_paths: vec!["assets".to_string()],
            default_quality: "medium".to_string(),
        }
    }
}

/// Deserialize a [`ProjectFile`] from a `.rkproject` file on disk.
pub fn load_project(path: &str) -> Result<ProjectFile> {
    let text = std::fs::read_to_string(path)?;
    let project: ProjectFile = ron::from_str(&text)?;
    Ok(project)
}

/// Serialize a [`ProjectFile`] to a `.rkproject` file on disk (pretty-printed RON).
pub fn save_project(path: &str, project: &ProjectFile) -> Result<()> {
    let config = ron::ser::PrettyConfig::default();
    let text = ron::ser::to_string_pretty(project, config)?;
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_project() {
        let p = ProjectFile::new("MyGame");
        assert_eq!(p.name, "MyGame");
        assert_eq!(p.default_quality, "medium");
        assert!(p.default_scene.is_none());
        assert!(p.scenes.is_empty());
        assert_eq!(p.asset_paths, vec!["assets"]);
        // engine_version is set from the crate version
        assert!(!p.engine_version.is_empty());
    }

    #[test]
    fn roundtrip_ron() {
        let mut p = ProjectFile::new("RoundtripProject");
        p.scenes.push(SceneRef {
            name: "Main".to_string(),
            path: "scenes/main.rkscene".to_string(),
            persistent: false,
        });
        p.scenes.push(SceneRef {
            name: "Persistent".to_string(),
            path: "scenes/shared.rkscene".to_string(),
            persistent: true,
        });
        p.default_scene = Some("Main".to_string());
        p.default_quality = "high".to_string();

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.name, p.name);
        assert_eq!(decoded.engine_version, p.engine_version);
        assert_eq!(decoded.scenes.len(), 2);
        assert_eq!(decoded.scenes[0].name, "Main");
        assert_eq!(decoded.scenes[1].persistent, true);
        assert_eq!(decoded.default_scene, Some("Main".to_string()));
        assert_eq!(decoded.default_quality, "high");
    }

    #[test]
    fn save_and_load() {
        let mut p = ProjectFile::new("SaveLoadProject");
        p.scenes.push(SceneRef {
            name: "Level1".to_string(),
            path: "scenes/level1.rkscene".to_string(),
            persistent: false,
        });
        p.asset_paths = vec!["assets".to_string(), "shared_assets".to_string()];

        let path = std::env::temp_dir()
            .join("rkf_project_save_and_load_test.rkproject");
        let path = path.to_str().expect("path").to_string();

        save_project(&path, &p).expect("save");
        let loaded = load_project(&path).expect("load");

        assert_eq!(loaded.name, "SaveLoadProject");
        assert_eq!(loaded.scenes.len(), 1);
        assert_eq!(loaded.scenes[0].path, "scenes/level1.rkscene");
        assert_eq!(loaded.asset_paths.len(), 2);
    }

    #[test]
    fn persistent_scene_flag() {
        let scene = SceneRef {
            name: "Environment".to_string(),
            path: "scenes/env.rkscene".to_string(),
            persistent: true,
        };

        let ron_text =
            ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default()).expect("ser");
        let decoded: SceneRef = ron::from_str(&ron_text).expect("deser");

        assert!(decoded.persistent);
        assert_eq!(decoded.name, "Environment");
    }
}
