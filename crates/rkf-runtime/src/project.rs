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
    /// Path to a default `.rkmatlib` material palette for the project.
    /// Relative to the project root.
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material_palette: Option<String>,
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
            material_palette: None,
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

    #[test]
    fn malformed_ron_input() {
        let malformed_ron = r#"
            (
                name: "BadProject",
                engine_version: "0.1.0",
                scenes: [
                    (name: "Scene1", path: "scenes/1.rkscene", persistent: false,
                    // Missing closing paren on this struct
                ],
                default_scene: None,
                asset_paths: ["assets"],
                default_quality: "medium"
            )
        "#;

        let result: std::result::Result<ProjectFile, _> = ron::from_str(malformed_ron);
        assert!(result.is_err(), "malformed RON should fail to deserialize");
    }

    #[test]
    fn empty_project_no_scenes() {
        let p = ProjectFile::new("EmptyProject");
        assert!(p.scenes.is_empty());
        assert_eq!(p.default_scene, None);

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.name, "EmptyProject");
        assert!(decoded.scenes.is_empty());
        assert_eq!(decoded.default_scene, None);
    }

    #[test]
    fn project_with_multiple_scenes_one_persistent() {
        let mut p = ProjectFile::new("MultiSceneProject");
        p.scenes.push(SceneRef {
            name: "Level1".to_string(),
            path: "scenes/level1.rkscene".to_string(),
            persistent: false,
        });
        p.scenes.push(SceneRef {
            name: "Level2".to_string(),
            path: "scenes/level2.rkscene".to_string(),
            persistent: false,
        });
        p.scenes.push(SceneRef {
            name: "Lighting".to_string(),
            path: "scenes/lighting.rkscene".to_string(),
            persistent: true,
        });
        p.default_scene = Some("Level1".to_string());

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.scenes.len(), 3);
        let persistent_count = decoded
            .scenes
            .iter()
            .filter(|s| s.persistent)
            .count();
        assert_eq!(persistent_count, 1);
        assert_eq!(decoded.scenes[2].name, "Lighting");
        assert!(decoded.scenes[2].persistent);
        assert_eq!(decoded.default_scene, Some("Level1".to_string()));
    }

    #[test]
    fn default_scene_nonexistent_ref_roundtrips() {
        let mut p = ProjectFile::new("NonexistentDefaultProject");
        p.scenes.push(SceneRef {
            name: "Scene1".to_string(),
            path: "scenes/scene1.rkscene".to_string(),
            persistent: false,
        });
        // Set default_scene to a name that doesn't exist in scenes list
        p.default_scene = Some("NonexistentScene".to_string());

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        // The project should still deserialize successfully; default_scene is just a string
        assert_eq!(decoded.default_scene, Some("NonexistentScene".to_string()));
        assert_eq!(decoded.scenes.len(), 1);
        // Verify that the mismatch doesn't prevent serialization
        let ron_text2 = ron::ser::to_string_pretty(&decoded, ron::ser::PrettyConfig::default())
            .expect("re-serialize");
        let decoded2: ProjectFile = ron::from_str(&ron_text2).expect("re-deserialize");
        assert_eq!(decoded2.default_scene, Some("NonexistentScene".to_string()));
    }

    #[test]
    fn asset_paths_with_special_characters() {
        let mut p = ProjectFile::new("SpecialCharsProject");
        p.asset_paths = vec![
            "assets/models-v2".to_string(),
            "assets/textures (backup)".to_string(),
            "assets/shared_materials".to_string(),
            "assets/高清素材".to_string(), // UTF-8 Chinese characters
        ];
        p.scenes.push(SceneRef {
            name: "Scene with spaces".to_string(),
            path: "scenes/main scene.rkscene".to_string(),
            persistent: false,
        });

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.asset_paths.len(), 4);
        assert_eq!(decoded.asset_paths[1], "assets/textures (backup)");
        assert_eq!(decoded.asset_paths[3], "assets/高清素材");
        assert_eq!(decoded.scenes[0].name, "Scene with spaces");
        assert_eq!(decoded.scenes[0].path, "scenes/main scene.rkscene");
    }

    #[test]
    fn large_project_many_scenes() {
        let mut p = ProjectFile::new("LargeProject");
        p.asset_paths = vec![
            "assets".to_string(),
            "shared_assets".to_string(),
            "user_mods".to_string(),
        ];

        // Add 100 scenes
        for i in 0..100 {
            let persistent = (i + 1) % 10 == 0; // Every 10th scene (10, 20, 30, ..., 100) is persistent
            p.scenes.push(SceneRef {
                name: format!("Level_{:03}", i),
                path: format!("scenes/level_{:03}.rkscene", i),
                persistent,
            });
        }

        // Set a default scene in the middle
        p.default_scene = Some("Level_050".to_string());
        p.default_quality = "high".to_string();

        let ron_text = ron::ser::to_string_pretty(&p, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: ProjectFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.scenes.len(), 100);
        let persistent_count = decoded
            .scenes
            .iter()
            .filter(|s| s.persistent)
            .count();
        assert_eq!(persistent_count, 10); // Every 10th scene
        assert_eq!(decoded.default_scene, Some("Level_050".to_string()));
        assert_eq!(decoded.default_quality, "high");

        // Verify that specific scenes are accessible
        assert_eq!(decoded.scenes[0].name, "Level_000");
        assert!(!decoded.scenes[0].persistent);
        assert_eq!(decoded.scenes[9].name, "Level_009");
        assert!(decoded.scenes[9].persistent); // (9+1) % 10 == 0
        assert_eq!(decoded.scenes[50].name, "Level_050");
        assert!(!decoded.scenes[50].persistent);
        assert_eq!(decoded.scenes[99].name, "Level_099");
        assert!(decoded.scenes[99].persistent); // (99+1) % 10 == 0
    }

    #[test]
    fn save_and_load_with_special_characters() {
        let mut p = ProjectFile::new("ProjectWithSpecialChars");
        p.scenes.push(SceneRef {
            name: "Scene-with-dashes".to_string(),
            path: "scenes/scene_with_underscores.rkscene".to_string(),
            persistent: false,
        });
        p.scenes.push(SceneRef {
            name: "Szenę (Polish)".to_string(),
            path: "scenes/сцена (Russian).rkscene".to_string(),
            persistent: true,
        });
        p.asset_paths = vec!["assets/v2.0".to_string(), "assets (local)".to_string()];

        let path = std::env::temp_dir()
            .join("rkf_project_special_chars_test.rkproject");
        let path_str = path.to_str().expect("path").to_string();

        save_project(&path_str, &p).expect("save");
        let loaded = load_project(&path_str).expect("load");

        assert_eq!(loaded.name, "ProjectWithSpecialChars");
        assert_eq!(loaded.scenes.len(), 2);
        assert_eq!(loaded.scenes[0].name, "Scene-with-dashes");
        assert_eq!(loaded.scenes[1].name, "Szenę (Polish)");
        assert_eq!(loaded.scenes[1].path, "scenes/сцена (Russian).rkscene");
        assert_eq!(loaded.asset_paths.len(), 2);
        assert_eq!(loaded.asset_paths[1], "assets (local)");

        // Clean up
        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn default_quality_fallback() {
        // Test that default_quality field uses the default_quality() function
        let minimal_ron = r#"
            (
                name: "MinimalProject",
                engine_version: "0.1.0",
                scenes: [],
            )
        "#;

        let decoded: ProjectFile = ron::from_str(minimal_ron).expect("deserialize");
        assert_eq!(decoded.default_quality, "medium");
    }
}
