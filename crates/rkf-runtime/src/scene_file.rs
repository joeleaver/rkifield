//! Scene file format (.rkscene v2) — RON-serialized scene descriptor.
//!
//! A scene file captures all user-authored content for a single scene: SDF
//! objects (either asset-backed or analytical), camera spawn points, lights,
//! and a reference to the environment profile.  It is intentionally
//! human-readable and diff-friendly so scenes can live under version control.

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Root scene descriptor, serialized to `.rkscene` (RON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFile {
    /// Human-readable scene name.
    pub name: String,
    /// All SDF objects in the scene.
    #[serde(default)]
    pub objects: Vec<ObjectEntry>,
    /// Camera spawn points / editor camera states.
    #[serde(default)]
    pub cameras: Vec<CameraEntry>,
    /// Lights placed in the scene.
    #[serde(default)]
    pub lights: Vec<LightEntry>,
    /// Path or name of the environment profile (sky, fog, global illumination
    /// seed) to load alongside this scene.
    #[serde(default)]
    pub environment: Option<String>,
    /// Generic property bag for subsystem state.
    ///
    /// Each subsystem serializes its state to a RON string under a key.
    /// This is the preferred way to persist editor/runtime state — avoids
    /// adding new typed fields for every subsystem.
    #[serde(default)]
    pub properties: HashMap<String, String>,
}

/// Serde helper: deserialize scale as either a single `f32` (uniform →
/// `[s, s, s]`) or a 3-element array `[x, y, z]`.  Always serializes as an
/// array.  This provides backward compatibility with scene files that used a
/// single scalar scale value.
mod scale_serde {
    use super::*;

    pub fn serialize<S: Serializer>(v: &[f32; 3], s: S) -> Result<S::Ok, S::Error> {
        v.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 3], D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ScaleValue {
            Vec3([f32; 3]),
            Uniform(f32),
        }
        match ScaleValue::deserialize(d)? {
            ScaleValue::Vec3(v) => Ok(v),
            ScaleValue::Uniform(f) => Ok([f, f, f]),
        }
    }
}

/// A single SDF object placed in a scene.
///
/// An object is either asset-backed (has an `asset_path`) or analytical (has
/// an `analytical` type string and optional `analytical_params`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectEntry {
    /// Editor display name.
    pub name: String,
    /// Path to a `.rkf` asset file, relative to the project root.
    #[serde(default)]
    pub asset_path: Option<String>,
    /// World position encoded as `[x, y, z]` in f64 meters.
    /// High-precision; converted to `WorldPosition` at runtime.
    #[serde(default)]
    pub position: [f64; 3],
    /// Orientation as a unit quaternion `[x, y, z, w]`.
    #[serde(default = "default_rotation")]
    pub rotation: [f32; 4],
    /// Per-axis scale `[x, y, z]`.  Backward-compatible: old files with a
    /// single `f32` are deserialized as uniform `[s, s, s]`.
    #[serde(default = "default_scale", with = "scale_serde")]
    pub scale: [f32; 3],
    /// Optional material ID override applied after asset load.
    #[serde(default)]
    pub material_id: Option<u16>,
    /// Analytical SDF type name (e.g. `"sphere"`, `"box"`, `"capsule"`).
    /// Only used when `asset_path` is `None`.
    #[serde(default)]
    pub analytical: Option<String>,
    /// Parameters for the analytical SDF (e.g. `[radius]` for sphere,
    /// `[half_x, half_y, half_z]` for box).
    #[serde(default)]
    pub analytical_params: Option<Vec<f32>>,
    /// LOD importance bias.  Values above 1.0 increase detail, below 1.0 decrease it.
    #[serde(default = "default_importance_bias")]
    pub importance_bias: f32,
}

fn default_rotation() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0] // identity quaternion
}

fn default_scale() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_importance_bias() -> f32 {
    1.0
}

/// A named camera placement / spawn point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraEntry {
    /// Display name (e.g. `"PlayerStart"`, `"OverviewCam"`).
    pub name: String,
    /// World position in f64 meters `[x, y, z]`.
    #[serde(default)]
    pub position: [f64; 3],
    /// Horizontal rotation in degrees.
    #[serde(default)]
    pub yaw: f32,
    /// Vertical rotation in degrees (positive = looking up).
    #[serde(default)]
    pub pitch: f32,
    /// Vertical field of view in degrees.
    #[serde(default = "default_fov")]
    pub fov: f32,
}

fn default_fov() -> f32 {
    60.0
}

/// A light source placed in a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightEntry {
    /// Display name.
    pub name: String,
    /// Light classification.
    pub light_type: LightType,
    /// Position in world space (f32 is sufficient for most scenes).
    #[serde(default)]
    pub position: [f32; 3],
    /// Normalised direction vector (used by directional and spot lights).
    #[serde(default = "default_light_direction")]
    pub direction: [f32; 3],
    /// Linear RGB colour (1.0 = full channel).
    #[serde(default = "default_white")]
    pub color: [f32; 3],
    /// Luminous intensity / power multiplier.
    #[serde(default = "default_intensity")]
    pub intensity: f32,
    /// Maximum influence radius in metres (0 = unlimited for directional).
    #[serde(default)]
    pub range: f32,
}

fn default_light_direction() -> [f32; 3] {
    [0.0, -1.0, 0.0]
}

fn default_white() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_intensity() -> f32 {
    1.0
}

/// Discriminated union of supported light types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightType {
    /// Parallel light with no position (sun / moon).
    Directional,
    /// Omnidirectional point emitter.
    Point,
    /// Cone-shaped spotlight.
    Spot {
        /// Half-angle of the fully-lit inner cone, in degrees.
        inner_angle: f32,
        /// Half-angle of the penumbra outer cone, in degrees.
        outer_angle: f32,
    },
}

// ── SceneFile helpers ──────────────────────────────────────────────────────────

impl SceneFile {
    /// Create a new, empty scene with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            objects: Vec::new(),
            cameras: Vec::new(),
            lights: Vec::new(),
            environment: None,
            properties: HashMap::new(),
        }
    }
}

// ── I/O functions ─────────────────────────────────────────────────────────────

/// Deserialize a [`SceneFile`] from a `.rkscene` file on disk.
pub fn load_scene_file(path: &str) -> Result<SceneFile> {
    let text = std::fs::read_to_string(path)?;
    let scene: SceneFile = ron::from_str(&text)?;
    Ok(scene)
}

/// Serialize a [`SceneFile`] to a `.rkscene` file on disk (pretty-printed RON).
pub fn save_scene_file(path: &str, scene: &SceneFile) -> Result<()> {
    let config = ron::ser::PrettyConfig::default();
    let text = ron::ser::to_string_pretty(scene, config)?;
    std::fs::write(path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scene() -> SceneFile {
        let mut scene = SceneFile::new("TestScene");

        scene.objects.push(ObjectEntry {
            name: "Ground".to_string(),
            asset_path: Some("assets/ground.rkf".to_string()),
            position: [0.0, -1.0, 0.0],
            rotation: default_rotation(),
            scale: [1.0, 1.0, 1.0],
            material_id: Some(3),
            analytical: None,
            analytical_params: None,
            importance_bias: 1.0,
        });

        scene.objects.push(ObjectEntry {
            name: "AnalyticalSphere".to_string(),
            asset_path: None,
            position: [5.0, 2.0, 0.0],
            rotation: default_rotation(),
            scale: [2.0, 2.0, 2.0],
            material_id: None,
            analytical: Some("sphere".to_string()),
            analytical_params: Some(vec![1.5]),
            importance_bias: 1.5,
        });

        scene.cameras.push(CameraEntry {
            name: "PlayerStart".to_string(),
            position: [0.0, 1.8, 5.0],
            yaw: 180.0,
            pitch: -5.0,
            fov: 75.0,
        });

        scene.lights.push(LightEntry {
            name: "Sun".to_string(),
            light_type: LightType::Directional,
            position: [0.0; 3],
            direction: [-0.5, -0.8, -0.3],
            color: [1.0, 0.95, 0.88],
            intensity: 3.0,
            range: 0.0,
        });

        scene.lights.push(LightEntry {
            name: "Lamp".to_string(),
            light_type: LightType::Point,
            position: [2.0, 3.0, 1.0],
            direction: default_light_direction(),
            color: default_white(),
            intensity: 5.0,
            range: 8.0,
        });

        scene.lights.push(LightEntry {
            name: "Torch".to_string(),
            light_type: LightType::Spot {
                inner_angle: 15.0,
                outer_angle: 30.0,
            },
            position: [0.0, 4.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 0.7, 0.3],
            intensity: 10.0,
            range: 12.0,
        });

        scene.environment = Some("sunset_hdr".to_string());
        scene
    }

    #[test]
    fn new_scene() {
        let s = SceneFile::new("Empty");
        assert_eq!(s.name, "Empty");
        assert!(s.objects.is_empty());
        assert!(s.cameras.is_empty());
        assert!(s.lights.is_empty());
        assert!(s.environment.is_none());
    }

    #[test]
    fn roundtrip_ron() {
        let scene = sample_scene();

        let ron_text = ron::ser::to_string_pretty(&scene, ron::ser::PrettyConfig::default())
            .expect("serialize");
        let decoded: SceneFile = ron::from_str(&ron_text).expect("deserialize");

        assert_eq!(decoded.name, "TestScene");
        assert_eq!(decoded.objects.len(), 2);
        assert_eq!(decoded.cameras.len(), 1);
        assert_eq!(decoded.lights.len(), 3);
        assert_eq!(decoded.environment, Some("sunset_hdr".to_string()));

        let ground = &decoded.objects[0];
        assert_eq!(ground.name, "Ground");
        assert_eq!(ground.asset_path, Some("assets/ground.rkf".to_string()));
        assert_eq!(ground.material_id, Some(3));
        assert_eq!(ground.position, [0.0, -1.0, 0.0]);

        let sphere = &decoded.objects[1];
        assert_eq!(sphere.analytical, Some("sphere".to_string()));
        assert_eq!(sphere.analytical_params, Some(vec![1.5]));
        assert!((sphere.importance_bias - 1.5).abs() < 1e-6);
    }

    #[test]
    fn save_and_load() {
        let scene = sample_scene();

        let path = std::env::temp_dir()
            .join("rkf_scene_file_save_and_load_test.rkscene");
        let path = path.to_str().expect("path").to_string();

        save_scene_file(&path, &scene).expect("save");
        let loaded = load_scene_file(&path).expect("load");

        assert_eq!(loaded.name, scene.name);
        assert_eq!(loaded.objects.len(), scene.objects.len());
        assert_eq!(loaded.cameras.len(), scene.cameras.len());
        assert_eq!(loaded.lights.len(), scene.lights.len());
        assert_eq!(loaded.environment, scene.environment);

        let cam = &loaded.cameras[0];
        assert_eq!(cam.name, "PlayerStart");
        assert!((cam.fov - 75.0).abs() < 1e-6);
        assert!((cam.yaw - 180.0).abs() < 1e-6);
    }

    #[test]
    fn light_types() {
        // Directional
        let dir = LightType::Directional;
        let s = ron::ser::to_string_pretty(&dir, ron::ser::PrettyConfig::default()).unwrap();
        let d: LightType = ron::from_str(&s).unwrap();
        assert!(matches!(d, LightType::Directional));

        // Point
        let pt = LightType::Point;
        let s = ron::ser::to_string_pretty(&pt, ron::ser::PrettyConfig::default()).unwrap();
        let d: LightType = ron::from_str(&s).unwrap();
        assert!(matches!(d, LightType::Point));

        // Spot
        let spot = LightType::Spot {
            inner_angle: 12.5,
            outer_angle: 25.0,
        };
        let s = ron::ser::to_string_pretty(&spot, ron::ser::PrettyConfig::default()).unwrap();
        let d: LightType = ron::from_str(&s).unwrap();
        match d {
            LightType::Spot {
                inner_angle,
                outer_angle,
            } => {
                assert!((inner_angle - 12.5).abs() < 1e-6);
                assert!((outer_angle - 25.0).abs() < 1e-6);
            }
            _ => panic!("expected Spot"),
        }
    }

    #[test]
    fn object_with_analytical() {
        let obj = ObjectEntry {
            name: "Box".to_string(),
            asset_path: None,
            position: [1.0, 0.0, -2.0],
            rotation: default_rotation(),
            scale: [1.0, 1.0, 1.0],
            material_id: None,
            analytical: Some("box".to_string()),
            analytical_params: Some(vec![2.0, 1.0, 3.0]),
            importance_bias: default_importance_bias(),
        };

        let s = ron::ser::to_string_pretty(&obj, ron::ser::PrettyConfig::default()).unwrap();
        let d: ObjectEntry = ron::from_str(&s).unwrap();

        assert_eq!(d.analytical, Some("box".to_string()));
        assert_eq!(d.analytical_params, Some(vec![2.0, 1.0, 3.0]));
        assert!(d.asset_path.is_none());
        assert!(d.material_id.is_none());
        assert!((d.scale[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn object_with_asset() {
        let obj = ObjectEntry {
            name: "HeroMesh".to_string(),
            asset_path: Some("characters/hero.rkf".to_string()),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.707, 0.0, 0.707],
            scale: [1.5, 1.5, 1.5],
            material_id: Some(7),
            analytical: None,
            analytical_params: None,
            importance_bias: 2.0,
        };

        let s = ron::ser::to_string_pretty(&obj, ron::ser::PrettyConfig::default()).unwrap();
        let d: ObjectEntry = ron::from_str(&s).unwrap();

        assert_eq!(d.asset_path, Some("characters/hero.rkf".to_string()));
        assert_eq!(d.material_id, Some(7));
        assert!((d.scale[0] - 1.5).abs() < 1e-6);
        assert!((d.scale[1] - 1.5).abs() < 1e-6);
        assert!((d.scale[2] - 1.5).abs() < 1e-6);
        assert!((d.importance_bias - 2.0).abs() < 1e-6);
        assert!(d.analytical.is_none());
        assert!(d.analytical_params.is_none());
        // rotation round-trips correctly
        assert!((d.rotation[1] - 0.707).abs() < 1e-4);
        assert!((d.rotation[3] - 0.707).abs() < 1e-4);
    }
}
