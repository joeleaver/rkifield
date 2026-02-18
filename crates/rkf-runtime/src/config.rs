//! Engine configuration system with quality presets and RON serialization.
//!
//! The configuration is split into per-system settings structs that can be
//! individually tweaked. [`QualityPreset`] provides sane named defaults and
//! [`EngineConfig::from_preset`] applies them in bulk. Configs round-trip
//! through RON so they can be stored on disk and reloaded at startup.

use serde::{Deserialize, Serialize};

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur when loading or saving an [`EngineConfig`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// An I/O error while reading or writing the config file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// A RON deserialization error.
    #[error("RON deserialize error: {0}")]
    Deserialize(#[from] ron::error::SpannedError),
    /// A RON serialization error.
    #[error("RON serialize error: {0}")]
    Serialize(#[from] ron::Error),
}

// ── Quality preset ────────────────────────────────────────────────────────────

/// Named quality tiers that map to a bundle of per-system settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityPreset {
    /// Minimum quality: 1/4 resolution, 16 volumetric steps, GI off, clouds off.
    Low,
    /// Medium quality: 1/3 resolution, 32 volumetric steps, GI 32³.
    Medium,
    /// High quality (default): 1/2 resolution, 48 volumetric steps, GI 64³.
    High,
    /// Maximum quality: 3/4 resolution, 64 volumetric steps, GI 128³, brick clouds.
    Ultra,
    /// All per-system settings controlled manually; preset has no effect.
    Custom,
}

impl Default for QualityPreset {
    fn default() -> Self {
        Self::High
    }
}

impl QualityPreset {
    /// Human-readable name for UI display.
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Ultra => "Ultra",
            Self::Custom => "Custom",
        }
    }
}

// ── Per-system settings ───────────────────────────────────────────────────────

/// Settings for the primary ray march pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayMarchSettings {
    /// Maximum ray march steps per pixel.
    pub max_steps: u32,
    /// Step size multiplier (1.0 = default).
    pub step_multiplier: f32,
}

impl Default for RayMarchSettings {
    fn default() -> Self {
        // High preset defaults.
        Self {
            max_steps: 512,
            step_multiplier: 1.0,
        }
    }
}

/// Settings for the shading pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadingSettings {
    /// Maximum shadow-casting lights evaluated per pixel.
    pub shadow_budget: u32,
    /// Enable subsurface scattering.
    pub sss_enabled: bool,
    /// Debug visualization mode (0 = normal shading).
    pub debug_mode: u32,
}

impl Default for ShadingSettings {
    fn default() -> Self {
        Self {
            shadow_budget: 4,
            sss_enabled: true,
            debug_mode: 0,
        }
    }
}

/// Settings for the volumetric fog and cloud passes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumetricSettings {
    /// Enable volumetric fog/march.
    pub enabled: bool,
    /// Maximum volumetric ray march steps.
    pub max_steps: u32,
    /// Step size in world units.
    pub step_size: f32,
    /// Maximum march distance in world units.
    pub max_distance: f32,
    /// Enable temporal reprojection for the volumetric buffer.
    pub temporal_enabled: bool,
    /// Temporal blend factor (higher = more history weight).
    pub temporal_blend: f32,
    /// Enable cloud shadow map pass.
    pub cloud_shadows: bool,
}

impl Default for VolumetricSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_steps: 48,
            step_size: 0.2,
            max_distance: 200.0,
            temporal_enabled: true,
            temporal_blend: 0.9,
            cloud_shadows: true,
        }
    }
}

/// Settings for the upscale pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpscaleSettings {
    /// Enable edge-aware sharpening after upscale.
    pub sharpen_enabled: bool,
    /// Sharpen strength (0.0 = off, 1.0 = maximum).
    pub sharpen_strength: f32,
}

impl Default for UpscaleSettings {
    fn default() -> Self {
        Self {
            sharpen_enabled: true,
            sharpen_strength: 0.4,
        }
    }
}

/// Settings for post-processing effects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessSettings {
    /// Enable bloom.
    pub bloom_enabled: bool,
    /// Bloom intensity (0.0 to 1.0).
    pub bloom_intensity: f32,
    /// Bloom threshold — pixels brighter than this contribute to bloom.
    pub bloom_threshold: f32,
    /// Enable depth of field.
    pub dof_enabled: bool,
    /// DoF focus distance in world units.
    pub dof_focus_distance: f32,
    /// DoF focus range — objects within ±this distance are in focus.
    pub dof_focus_range: f32,
    /// Enable motion blur.
    pub motion_blur_enabled: bool,
    /// Motion blur intensity.
    pub motion_blur_intensity: f32,
    /// Enable automatic exposure adaptation.
    pub auto_exposure_enabled: bool,
    /// Enable color grading LUT pass.
    pub color_grade_enabled: bool,
    /// Color grade LUT mix intensity.
    pub color_grade_intensity: f32,
    /// Enable vignette.
    pub vignette_enabled: bool,
    /// Vignette intensity.
    pub vignette_intensity: f32,
    /// Enable film grain.
    pub grain_enabled: bool,
    /// Film grain intensity.
    pub grain_intensity: f32,
    /// Enable chromatic aberration.
    pub chromatic_aberration_enabled: bool,
    /// Chromatic aberration intensity.
    pub chromatic_aberration_intensity: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            dof_enabled: true,
            dof_focus_distance: 10.0,
            dof_focus_range: 5.0,
            motion_blur_enabled: true,
            motion_blur_intensity: 0.5,
            auto_exposure_enabled: true,
            color_grade_enabled: true,
            color_grade_intensity: 1.0,
            vignette_enabled: true,
            vignette_intensity: 0.3,
            grain_enabled: true,
            grain_intensity: 0.05,
            chromatic_aberration_enabled: false,
            chromatic_aberration_intensity: 0.5,
        }
    }
}

/// Settings for the global illumination (radiance volume) system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GiSettings {
    /// Enable global illumination.
    pub enabled: bool,
    /// Radiance volume dimension along each axis (32, 64, or 128).
    pub volume_dim: u32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            volume_dim: 64,
        }
    }
}

// ── Engine config ─────────────────────────────────────────────────────────────

/// Top-level engine configuration.
///
/// Create via [`EngineConfig::from_preset`] for preset-based defaults, or
/// load from disk with [`EngineConfig::load`]. Individual per-system fields
/// can be overridden after construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Output display resolution in pixels `[width, height]`.
    pub display_resolution: [u32; 2],
    /// Internal render scale relative to `display_resolution` (0.25 to 1.0).
    pub render_scale: f32,
    /// Enable vertical synchronisation.
    pub vsync: bool,
    /// Active quality preset (informational when `Custom`).
    pub quality_preset: QualityPreset,
    /// Ray march pass settings.
    pub ray_march: RayMarchSettings,
    /// Shading pass settings.
    pub shading: ShadingSettings,
    /// Volumetric fog/cloud settings.
    pub volumetrics: VolumetricSettings,
    /// Upscale pass settings.
    pub upscale: UpscaleSettings,
    /// Post-processing effect settings.
    pub post_process: PostProcessSettings,
    /// Global illumination settings.
    pub gi: GiSettings,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::from_preset(QualityPreset::High)
    }
}

impl EngineConfig {
    /// Build a config with all per-system settings driven by `preset`.
    ///
    /// For [`QualityPreset::Custom`] the High preset values are used as a
    /// starting point — callers then modify individual fields.
    pub fn from_preset(preset: QualityPreset) -> Self {
        match preset {
            QualityPreset::Low => Self {
                display_resolution: [1920, 1080],
                render_scale: 0.25,
                vsync: true,
                quality_preset: preset,
                ray_march: RayMarchSettings {
                    max_steps: 128,
                    step_multiplier: 1.0,
                },
                shading: ShadingSettings {
                    shadow_budget: 2,
                    sss_enabled: false,
                    debug_mode: 0,
                },
                volumetrics: VolumetricSettings {
                    enabled: false,
                    max_steps: 16,
                    step_size: 0.5,
                    max_distance: 50.0,
                    temporal_enabled: false,
                    temporal_blend: 0.9,
                    cloud_shadows: false,
                },
                upscale: UpscaleSettings {
                    sharpen_enabled: true,
                    sharpen_strength: 0.5,
                },
                post_process: PostProcessSettings {
                    bloom_enabled: false,
                    bloom_intensity: 0.3,
                    bloom_threshold: 1.0,
                    dof_enabled: false,
                    dof_focus_distance: 10.0,
                    dof_focus_range: 5.0,
                    motion_blur_enabled: false,
                    motion_blur_intensity: 0.5,
                    auto_exposure_enabled: false,
                    color_grade_enabled: false,
                    color_grade_intensity: 1.0,
                    vignette_enabled: false,
                    vignette_intensity: 0.3,
                    grain_enabled: false,
                    grain_intensity: 0.05,
                    chromatic_aberration_enabled: false,
                    chromatic_aberration_intensity: 0.5,
                },
                gi: GiSettings {
                    enabled: false,
                    volume_dim: 32,
                },
            },

            QualityPreset::Medium => Self {
                display_resolution: [1920, 1080],
                render_scale: 0.33,
                vsync: true,
                quality_preset: preset,
                ray_march: RayMarchSettings {
                    max_steps: 256,
                    step_multiplier: 1.0,
                },
                shading: ShadingSettings {
                    shadow_budget: 3,
                    sss_enabled: true,
                    debug_mode: 0,
                },
                volumetrics: VolumetricSettings {
                    enabled: true,
                    max_steps: 32,
                    step_size: 0.3,
                    max_distance: 100.0,
                    temporal_enabled: true,
                    temporal_blend: 0.9,
                    cloud_shadows: false,
                },
                upscale: UpscaleSettings {
                    sharpen_enabled: true,
                    sharpen_strength: 0.5,
                },
                post_process: PostProcessSettings {
                    bloom_enabled: true,
                    bloom_intensity: 0.3,
                    bloom_threshold: 1.0,
                    dof_enabled: false,
                    dof_focus_distance: 10.0,
                    dof_focus_range: 5.0,
                    motion_blur_enabled: false,
                    motion_blur_intensity: 0.5,
                    auto_exposure_enabled: true,
                    color_grade_enabled: false,
                    color_grade_intensity: 1.0,
                    vignette_enabled: true,
                    vignette_intensity: 0.3,
                    grain_enabled: false,
                    grain_intensity: 0.05,
                    chromatic_aberration_enabled: false,
                    chromatic_aberration_intensity: 0.5,
                },
                gi: GiSettings {
                    enabled: true,
                    volume_dim: 32,
                },
            },

            QualityPreset::High | QualityPreset::Custom => Self {
                display_resolution: [1920, 1080],
                render_scale: 0.5,
                vsync: true,
                quality_preset: preset,
                ray_march: RayMarchSettings {
                    max_steps: 512,
                    step_multiplier: 1.0,
                },
                shading: ShadingSettings {
                    shadow_budget: 4,
                    sss_enabled: true,
                    debug_mode: 0,
                },
                volumetrics: VolumetricSettings {
                    enabled: true,
                    max_steps: 48,
                    step_size: 0.2,
                    max_distance: 200.0,
                    temporal_enabled: true,
                    temporal_blend: 0.9,
                    cloud_shadows: true,
                },
                upscale: UpscaleSettings {
                    sharpen_enabled: true,
                    sharpen_strength: 0.4,
                },
                post_process: PostProcessSettings {
                    bloom_enabled: true,
                    bloom_intensity: 0.3,
                    bloom_threshold: 1.0,
                    dof_enabled: true,
                    dof_focus_distance: 10.0,
                    dof_focus_range: 5.0,
                    motion_blur_enabled: true,
                    motion_blur_intensity: 0.5,
                    auto_exposure_enabled: true,
                    color_grade_enabled: true,
                    color_grade_intensity: 1.0,
                    vignette_enabled: true,
                    vignette_intensity: 0.3,
                    grain_enabled: true,
                    grain_intensity: 0.05,
                    chromatic_aberration_enabled: false,
                    chromatic_aberration_intensity: 0.5,
                },
                gi: GiSettings {
                    enabled: true,
                    volume_dim: 64,
                },
            },

            QualityPreset::Ultra => Self {
                display_resolution: [1920, 1080],
                render_scale: 0.75,
                vsync: true,
                quality_preset: preset,
                ray_march: RayMarchSettings {
                    max_steps: 1024,
                    step_multiplier: 1.0,
                },
                shading: ShadingSettings {
                    shadow_budget: 6,
                    sss_enabled: true,
                    debug_mode: 0,
                },
                volumetrics: VolumetricSettings {
                    enabled: true,
                    max_steps: 64,
                    step_size: 0.15,
                    max_distance: 400.0,
                    temporal_enabled: true,
                    temporal_blend: 0.9,
                    cloud_shadows: true,
                },
                upscale: UpscaleSettings {
                    sharpen_enabled: true,
                    sharpen_strength: 0.3,
                },
                post_process: PostProcessSettings {
                    bloom_enabled: true,
                    bloom_intensity: 0.3,
                    bloom_threshold: 1.0,
                    dof_enabled: true,
                    dof_focus_distance: 10.0,
                    dof_focus_range: 5.0,
                    motion_blur_enabled: true,
                    motion_blur_intensity: 0.5,
                    auto_exposure_enabled: true,
                    color_grade_enabled: true,
                    color_grade_intensity: 1.0,
                    vignette_enabled: true,
                    vignette_intensity: 0.3,
                    grain_enabled: true,
                    grain_intensity: 0.05,
                    chromatic_aberration_enabled: true,
                    chromatic_aberration_intensity: 0.5,
                },
                gi: GiSettings {
                    enabled: true,
                    volume_dim: 128,
                },
            },
        }
    }

    /// Convert to [`crate::frame::FrameSettings`] by mapping config booleans
    /// to the frame-level pass-enable flags.
    pub fn to_frame_settings(&self) -> crate::frame::FrameSettings {
        crate::frame::FrameSettings {
            volumetrics_enabled: self.volumetrics.enabled,
            cloud_shadows_enabled: self.volumetrics.cloud_shadows,
            gi_enabled: self.gi.enabled,
            dof_enabled: self.post_process.dof_enabled,
            motion_blur_enabled: self.post_process.motion_blur_enabled,
            bloom_enabled: self.post_process.bloom_enabled,
            auto_exposure_enabled: self.post_process.auto_exposure_enabled,
            color_grade_enabled: self.post_process.color_grade_enabled,
            cosmetics_enabled: self.post_process.vignette_enabled
                || self.post_process.grain_enabled
                || self.post_process.chromatic_aberration_enabled,
            sharpen_enabled: self.upscale.sharpen_enabled,
        }
    }

    /// Load a config from a RON file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        let config = ron::from_str(&text)?;
        Ok(config)
    }

    /// Serialize this config to a RON file at `path`.
    ///
    /// The file is created (or overwritten) with pretty-printed RON output.
    pub fn save(&self, path: &std::path::Path) -> Result<(), ConfigError> {
        let pretty = ron::ser::PrettyConfig::new();
        let text = ron::ser::to_string_pretty(self, pretty)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_low() {
        let cfg = EngineConfig::from_preset(QualityPreset::Low);
        assert_eq!(cfg.render_scale, 0.25);
        assert_eq!(cfg.ray_march.max_steps, 128);
        assert_eq!(cfg.shading.shadow_budget, 2);
        assert!(!cfg.shading.sss_enabled);
        assert!(!cfg.volumetrics.enabled);
        assert_eq!(cfg.volumetrics.max_steps, 16);
        assert_eq!(cfg.volumetrics.step_size, 0.5);
        assert_eq!(cfg.volumetrics.max_distance, 50.0);
        assert!(!cfg.volumetrics.temporal_enabled);
        assert!(!cfg.volumetrics.cloud_shadows);
        assert!(!cfg.gi.enabled);
        assert_eq!(cfg.gi.volume_dim, 32);
        assert!(!cfg.post_process.bloom_enabled);
        assert!(!cfg.post_process.dof_enabled);
        assert!(!cfg.post_process.motion_blur_enabled);
        assert!(!cfg.post_process.auto_exposure_enabled);
        assert!(!cfg.post_process.color_grade_enabled);
        assert!(!cfg.post_process.vignette_enabled);
        assert!(!cfg.post_process.grain_enabled);
        assert!(!cfg.post_process.chromatic_aberration_enabled);
    }

    #[test]
    fn preset_high() {
        let cfg = EngineConfig::from_preset(QualityPreset::High);
        assert_eq!(cfg.render_scale, 0.5);
        assert_eq!(cfg.ray_march.max_steps, 512);
        assert_eq!(cfg.shading.shadow_budget, 4);
        assert!(cfg.shading.sss_enabled);
        assert!(cfg.volumetrics.enabled);
        assert_eq!(cfg.volumetrics.max_steps, 48);
        assert_eq!(cfg.volumetrics.step_size, 0.2);
        assert_eq!(cfg.volumetrics.max_distance, 200.0);
        assert!(cfg.volumetrics.temporal_enabled);
        assert!(cfg.volumetrics.cloud_shadows);
        assert!(cfg.gi.enabled);
        assert_eq!(cfg.gi.volume_dim, 64);
        assert!(cfg.post_process.bloom_enabled);
        assert!(cfg.post_process.dof_enabled);
        assert!(cfg.post_process.motion_blur_enabled);
        assert!(cfg.post_process.auto_exposure_enabled);
        assert!(cfg.post_process.color_grade_enabled);
        assert!(cfg.post_process.vignette_enabled);
        assert!(cfg.post_process.grain_enabled);
        assert!(!cfg.post_process.chromatic_aberration_enabled);
    }

    #[test]
    fn preset_ultra() {
        let cfg = EngineConfig::from_preset(QualityPreset::Ultra);
        assert_eq!(cfg.render_scale, 0.75);
        assert_eq!(cfg.ray_march.max_steps, 1024);
        assert_eq!(cfg.shading.shadow_budget, 6);
        assert!(cfg.shading.sss_enabled);
        assert!(cfg.volumetrics.enabled);
        assert_eq!(cfg.volumetrics.max_steps, 64);
        assert_eq!(cfg.volumetrics.step_size, 0.15);
        assert_eq!(cfg.volumetrics.max_distance, 400.0);
        assert!(cfg.volumetrics.cloud_shadows);
        assert!(cfg.gi.enabled);
        assert_eq!(cfg.gi.volume_dim, 128);
        assert!(cfg.post_process.chromatic_aberration_enabled);
    }

    #[test]
    fn default_is_high() {
        let default = EngineConfig::default();
        let high = EngineConfig::from_preset(QualityPreset::High);
        // Compare key fields (not direct PartialEq since we don't derive it).
        assert_eq!(default.render_scale, high.render_scale);
        assert_eq!(default.ray_march.max_steps, high.ray_march.max_steps);
        assert_eq!(default.gi.volume_dim, high.gi.volume_dim);
        assert_eq!(default.quality_preset, QualityPreset::High);
    }

    #[test]
    fn to_frame_settings_low() {
        let cfg = EngineConfig::from_preset(QualityPreset::Low);
        let fs = cfg.to_frame_settings();
        assert!(!fs.volumetrics_enabled);
        assert!(!fs.cloud_shadows_enabled);
        assert!(!fs.gi_enabled);
        assert!(!fs.dof_enabled);
        assert!(!fs.motion_blur_enabled);
        assert!(!fs.bloom_enabled);
        assert!(!fs.auto_exposure_enabled);
        assert!(!fs.color_grade_enabled);
        assert!(!fs.cosmetics_enabled);
        assert!(fs.sharpen_enabled);
    }

    #[test]
    fn to_frame_settings_ultra() {
        let cfg = EngineConfig::from_preset(QualityPreset::Ultra);
        let fs = cfg.to_frame_settings();
        assert!(fs.volumetrics_enabled);
        assert!(fs.cloud_shadows_enabled);
        assert!(fs.gi_enabled);
        assert!(fs.dof_enabled);
        assert!(fs.motion_blur_enabled);
        assert!(fs.bloom_enabled);
        assert!(fs.auto_exposure_enabled);
        assert!(fs.color_grade_enabled);
        // vignette | grain | chromatic_aberration — ultra enables all three.
        assert!(fs.cosmetics_enabled);
        assert!(fs.sharpen_enabled);
    }

    #[test]
    fn ron_roundtrip() {
        let original = EngineConfig::from_preset(QualityPreset::Medium);
        let ron_str = ron::ser::to_string_pretty(&original, ron::ser::PrettyConfig::new())
            .expect("serialize");
        let restored: EngineConfig = ron::from_str(&ron_str).expect("deserialize");
        // Spot-check a selection of fields.
        assert_eq!(restored.render_scale, original.render_scale);
        assert_eq!(restored.quality_preset, original.quality_preset);
        assert_eq!(restored.ray_march.max_steps, original.ray_march.max_steps);
        assert_eq!(restored.gi.volume_dim, original.gi.volume_dim);
        assert_eq!(
            restored.post_process.bloom_enabled,
            original.post_process.bloom_enabled
        );
        assert_eq!(
            restored.volumetrics.max_steps,
            original.volumetrics.max_steps
        );
    }

    #[test]
    fn quality_preset_name() {
        assert_eq!(QualityPreset::Low.name(), "Low");
        assert_eq!(QualityPreset::Medium.name(), "Medium");
        assert_eq!(QualityPreset::High.name(), "High");
        assert_eq!(QualityPreset::Ultra.name(), "Ultra");
        assert_eq!(QualityPreset::Custom.name(), "Custom");
    }

    #[test]
    fn render_scale_range() {
        for preset in [
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
            QualityPreset::Ultra,
            QualityPreset::Custom,
        ] {
            let cfg = EngineConfig::from_preset(preset);
            assert!(
                cfg.render_scale >= 0.25 && cfg.render_scale <= 1.0,
                "{} preset render_scale {} out of [0.25, 1.0]",
                preset.name(),
                cfg.render_scale
            );
        }
    }

    #[test]
    fn file_roundtrip() {
        let original = EngineConfig::from_preset(QualityPreset::Ultra);
        let dir = std::env::temp_dir();
        let path = dir.join("rkifield_config_test.ron");
        original.save(&path).expect("save");
        let loaded = EngineConfig::load(&path).expect("load");
        assert_eq!(loaded.render_scale, original.render_scale);
        assert_eq!(loaded.quality_preset, original.quality_preset);
        assert_eq!(loaded.ray_march.max_steps, original.ray_march.max_steps);
        assert_eq!(loaded.gi.volume_dim, original.gi.volume_dim);
        // Clean up.
        let _ = std::fs::remove_file(&path);
    }
}
