//! Quality preset validation — checks that engine configs have sane values.
//!
//! [`validate_config`] inspects an [`EngineConfig`] and returns a
//! [`PresetValidation`] listing any out-of-range or suspicious settings.
//! [`compare_presets`] diffs two configs field-by-field.

#![allow(dead_code)]

use crate::config::{EngineConfig, QualityPreset};

// ── Types ────────────────────────────────────────────────────────────────────

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Hard constraint violation — config will not work correctly.
    Error,
    /// Likely unintentional or performance-problematic setting.
    Warning,
    /// Informational note — not necessarily wrong.
    Info,
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Dot-path of the offending field (e.g. `"ray_march.max_steps"`).
    pub field: String,
    /// How severe this finding is.
    pub severity: Severity,
    /// Human-readable explanation.
    pub message: String,
}

/// Result of validating one config / preset.
#[derive(Debug, Clone)]
pub struct PresetValidation {
    /// Name of the preset (or `"Custom"`).
    pub preset_name: String,
    /// All findings, ordered by field name.
    pub issues: Vec<ValidationIssue>,
    /// `true` when there are zero `Error`-level issues.
    pub passed: bool,
}

/// A single field that differs between two configs.
#[derive(Debug, Clone)]
pub struct SettingDiff {
    /// Dot-path of the field.
    pub field: String,
    /// Stringified value in config A.
    pub value_a: String,
    /// Stringified value in config B.
    pub value_b: String,
}

/// Side-by-side comparison of two configs.
#[derive(Debug, Clone)]
pub struct PresetComparison {
    /// Name / label for config A.
    pub preset_a: String,
    /// Name / label for config B.
    pub preset_b: String,
    /// Fields that differ.
    pub differences: Vec<SettingDiff>,
}

// ── Validation ───────────────────────────────────────────────────────────────

/// Validate an [`EngineConfig`] and return all findings.
pub fn validate_config(config: &EngineConfig) -> PresetValidation {
    let mut issues = Vec::new();

    // ── render_scale ─────────────────────────────────────────────────────
    if config.render_scale < 0.1 || config.render_scale > 1.0 {
        issues.push(ValidationIssue {
            field: "render_scale".into(),
            severity: Severity::Error,
            message: format!(
                "render_scale {} is outside valid range [0.1, 1.0]",
                config.render_scale
            ),
        });
    }

    // ── ray_march ────────────────────────────────────────────────────────
    if config.ray_march.max_steps < 64 || config.ray_march.max_steps > 2048 {
        issues.push(ValidationIssue {
            field: "ray_march.max_steps".into(),
            severity: Severity::Error,
            message: format!(
                "max_steps {} is outside valid range [64, 2048]",
                config.ray_march.max_steps
            ),
        });
    }

    if config.ray_march.step_multiplier < 0.5 || config.ray_march.step_multiplier > 2.0 {
        issues.push(ValidationIssue {
            field: "ray_march.step_multiplier".into(),
            severity: Severity::Error,
            message: format!(
                "step_multiplier {} is outside valid range [0.5, 2.0]",
                config.ray_march.step_multiplier
            ),
        });
    }

    // ── volumetrics (only when enabled) ──────────────────────────────────
    if config.volumetrics.enabled {
        if config.volumetrics.max_steps < 8 || config.volumetrics.max_steps > 128 {
            issues.push(ValidationIssue {
                field: "volumetrics.max_steps".into(),
                severity: Severity::Error,
                message: format!(
                    "volumetric max_steps {} is outside valid range [8, 128]",
                    config.volumetrics.max_steps
                ),
            });
        }

        if config.volumetrics.step_size < 0.05 || config.volumetrics.step_size > 1.0 {
            issues.push(ValidationIssue {
                field: "volumetrics.step_size".into(),
                severity: Severity::Error,
                message: format!(
                    "volumetric step_size {} is outside valid range [0.05, 1.0]",
                    config.volumetrics.step_size
                ),
            });
        }
    }

    // ── GI ───────────────────────────────────────────────────────────────
    let dim = config.gi.volume_dim;
    let is_power_of_2 = dim > 0 && (dim & (dim - 1)) == 0;
    if !is_power_of_2 || !(16..=256).contains(&dim) {
        issues.push(ValidationIssue {
            field: "gi.volume_dim".into(),
            severity: Severity::Error,
            message: format!(
                "volume_dim {} must be a power of 2 in [16, 256]",
                dim
            ),
        });
    }

    // ── shading ──────────────────────────────────────────────────────────
    if config.shading.shadow_budget > 16 {
        issues.push(ValidationIssue {
            field: "shading.shadow_budget".into(),
            severity: Severity::Error,
            message: format!(
                "shadow_budget {} exceeds maximum of 16",
                config.shading.shadow_budget
            ),
        });
    }

    // ── post_process ─────────────────────────────────────────────────────
    if config.post_process.bloom_enabled && config.post_process.bloom_threshold <= 0.0 {
        issues.push(ValidationIssue {
            field: "post_process.bloom_threshold".into(),
            severity: Severity::Error,
            message: format!(
                "bloom_threshold {} must be > 0 when bloom is enabled",
                config.post_process.bloom_threshold
            ),
        });
    }

    // Check intensities in [0.0, 2.0].
    let intensity_fields: &[(&str, f32)] = &[
        ("post_process.bloom_intensity", config.post_process.bloom_intensity),
        ("post_process.motion_blur_intensity", config.post_process.motion_blur_intensity),
        ("post_process.color_grade_intensity", config.post_process.color_grade_intensity),
        ("post_process.vignette_intensity", config.post_process.vignette_intensity),
        ("post_process.grain_intensity", config.post_process.grain_intensity),
        (
            "post_process.chromatic_aberration_intensity",
            config.post_process.chromatic_aberration_intensity,
        ),
    ];
    for &(field, value) in intensity_fields {
        if !(0.0..=2.0).contains(&value) {
            issues.push(ValidationIssue {
                field: field.into(),
                severity: Severity::Error,
                message: format!(
                    "{} intensity {} is outside valid range [0.0, 2.0]",
                    field, value
                ),
            });
        }
    }

    // ── Warnings ─────────────────────────────────────────────────────────

    // GI on Low preset is a performance concern.
    if config.quality_preset == QualityPreset::Low && config.gi.enabled {
        issues.push(ValidationIssue {
            field: "gi.enabled".into(),
            severity: Severity::Warning,
            message: "GI enabled on Low preset — may cause performance issues".into(),
        });
    }

    // High render_scale without Ultra preset.
    if config.render_scale > 0.75 && config.quality_preset != QualityPreset::Ultra {
        issues.push(ValidationIssue {
            field: "render_scale".into(),
            severity: Severity::Warning,
            message: format!(
                "render_scale {} > 0.75 without Ultra preset — may be too expensive",
                config.render_scale
            ),
        });
    }

    let passed = !issues.iter().any(|i| i.severity == Severity::Error);

    PresetValidation {
        preset_name: config.quality_preset.name().to_string(),
        issues,
        passed,
    }
}

/// Validate all four built-in presets (Low, Medium, High, Ultra).
pub fn validate_all_presets() -> Vec<PresetValidation> {
    [
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Ultra,
    ]
    .iter()
    .map(|&p| validate_config(&EngineConfig::from_preset(p)))
    .collect()
}

// ── Comparison ───────────────────────────────────────────────────────────────

/// Compare two configs field-by-field and return all differences.
pub fn compare_presets(a: &EngineConfig, b: &EngineConfig) -> PresetComparison {
    let mut differences = Vec::new();

    macro_rules! cmp {
        ($field:expr, $va:expr, $vb:expr) => {
            let sa = format!("{:?}", $va);
            let sb = format!("{:?}", $vb);
            if sa != sb {
                differences.push(SettingDiff {
                    field: $field.into(),
                    value_a: sa,
                    value_b: sb,
                });
            }
        };
    }

    // Top-level.
    cmp!("render_scale", a.render_scale, b.render_scale);
    cmp!("vsync", a.vsync, b.vsync);
    cmp!("quality_preset", a.quality_preset, b.quality_preset);
    cmp!(
        "display_resolution",
        a.display_resolution,
        b.display_resolution
    );

    // Ray march.
    cmp!("ray_march.max_steps", a.ray_march.max_steps, b.ray_march.max_steps);
    cmp!(
        "ray_march.step_multiplier",
        a.ray_march.step_multiplier,
        b.ray_march.step_multiplier
    );

    // Shading.
    cmp!("shading.shadow_budget", a.shading.shadow_budget, b.shading.shadow_budget);
    cmp!("shading.sss_enabled", a.shading.sss_enabled, b.shading.sss_enabled);
    cmp!("shading.debug_mode", a.shading.debug_mode, b.shading.debug_mode);

    // Volumetrics.
    cmp!("volumetrics.enabled", a.volumetrics.enabled, b.volumetrics.enabled);
    cmp!("volumetrics.max_steps", a.volumetrics.max_steps, b.volumetrics.max_steps);
    cmp!("volumetrics.step_size", a.volumetrics.step_size, b.volumetrics.step_size);
    cmp!("volumetrics.max_distance", a.volumetrics.max_distance, b.volumetrics.max_distance);
    cmp!(
        "volumetrics.temporal_enabled",
        a.volumetrics.temporal_enabled,
        b.volumetrics.temporal_enabled
    );
    cmp!(
        "volumetrics.temporal_blend",
        a.volumetrics.temporal_blend,
        b.volumetrics.temporal_blend
    );
    cmp!(
        "volumetrics.cloud_shadows",
        a.volumetrics.cloud_shadows,
        b.volumetrics.cloud_shadows
    );

    // Upscale.
    cmp!(
        "upscale.sharpen_enabled",
        a.upscale.sharpen_enabled,
        b.upscale.sharpen_enabled
    );
    cmp!(
        "upscale.sharpen_strength",
        a.upscale.sharpen_strength,
        b.upscale.sharpen_strength
    );

    // Post-process.
    cmp!("post_process.bloom_enabled", a.post_process.bloom_enabled, b.post_process.bloom_enabled);
    cmp!(
        "post_process.bloom_intensity",
        a.post_process.bloom_intensity,
        b.post_process.bloom_intensity
    );
    cmp!(
        "post_process.bloom_threshold",
        a.post_process.bloom_threshold,
        b.post_process.bloom_threshold
    );
    cmp!("post_process.dof_enabled", a.post_process.dof_enabled, b.post_process.dof_enabled);
    cmp!(
        "post_process.dof_focus_distance",
        a.post_process.dof_focus_distance,
        b.post_process.dof_focus_distance
    );
    cmp!(
        "post_process.dof_focus_range",
        a.post_process.dof_focus_range,
        b.post_process.dof_focus_range
    );
    cmp!(
        "post_process.motion_blur_enabled",
        a.post_process.motion_blur_enabled,
        b.post_process.motion_blur_enabled
    );
    cmp!(
        "post_process.motion_blur_intensity",
        a.post_process.motion_blur_intensity,
        b.post_process.motion_blur_intensity
    );
    cmp!(
        "post_process.auto_exposure_enabled",
        a.post_process.auto_exposure_enabled,
        b.post_process.auto_exposure_enabled
    );
    cmp!(
        "post_process.color_grade_enabled",
        a.post_process.color_grade_enabled,
        b.post_process.color_grade_enabled
    );
    cmp!(
        "post_process.color_grade_intensity",
        a.post_process.color_grade_intensity,
        b.post_process.color_grade_intensity
    );
    cmp!(
        "post_process.vignette_enabled",
        a.post_process.vignette_enabled,
        b.post_process.vignette_enabled
    );
    cmp!(
        "post_process.vignette_intensity",
        a.post_process.vignette_intensity,
        b.post_process.vignette_intensity
    );
    cmp!("post_process.grain_enabled", a.post_process.grain_enabled, b.post_process.grain_enabled);
    cmp!(
        "post_process.grain_intensity",
        a.post_process.grain_intensity,
        b.post_process.grain_intensity
    );
    cmp!(
        "post_process.chromatic_aberration_enabled",
        a.post_process.chromatic_aberration_enabled,
        b.post_process.chromatic_aberration_enabled
    );
    cmp!(
        "post_process.chromatic_aberration_intensity",
        a.post_process.chromatic_aberration_intensity,
        b.post_process.chromatic_aberration_intensity
    );

    // GI.
    cmp!("gi.enabled", a.gi.enabled, b.gi.enabled);
    cmp!("gi.volume_dim", a.gi.volume_dim, b.gi.volume_dim);

    PresetComparison {
        preset_a: a.quality_preset.name().to_string(),
        preset_b: b.quality_preset.name().to_string(),
        differences,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EngineConfig, QualityPreset, RayMarchSettings};

    #[test]
    fn all_builtin_presets_pass() {
        let results = validate_all_presets();
        assert_eq!(results.len(), 4);
        for v in &results {
            assert!(
                v.passed,
                "preset {} failed validation: {:?}",
                v.preset_name,
                v.issues
                    .iter()
                    .filter(|i| i.severity == Severity::Error)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn low_preset_name() {
        let v = validate_config(&EngineConfig::from_preset(QualityPreset::Low));
        assert_eq!(v.preset_name, "Low");
    }

    #[test]
    fn render_scale_too_low() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.render_scale = 0.05;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "render_scale" && i.severity == Severity::Error));
    }

    #[test]
    fn render_scale_too_high() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.render_scale = 1.5;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "render_scale" && i.severity == Severity::Error));
    }

    #[test]
    fn ray_march_steps_too_low() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.ray_march = RayMarchSettings {
            max_steps: 10,
            step_multiplier: 1.0,
        };
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "ray_march.max_steps" && i.severity == Severity::Error));
    }

    #[test]
    fn ray_march_steps_too_high() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.ray_march.max_steps = 5000;
        let v = validate_config(&cfg);
        assert!(!v.passed);
    }

    #[test]
    fn step_multiplier_out_of_range() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.ray_march.step_multiplier = 3.0;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "ray_march.step_multiplier"));
    }

    #[test]
    fn volumetric_steps_out_of_range_when_enabled() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.volumetrics.enabled = true;
        cfg.volumetrics.max_steps = 200;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "volumetrics.max_steps"));
    }

    #[test]
    fn volumetric_steps_ignored_when_disabled() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.volumetrics.enabled = false;
        cfg.volumetrics.max_steps = 200; // out of range but disabled
        let v = validate_config(&cfg);
        assert!(!v
            .issues
            .iter()
            .any(|i| i.field == "volumetrics.max_steps"));
    }

    #[test]
    fn volumetric_step_size_out_of_range() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.volumetrics.step_size = 2.0;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v.issues.iter().any(|i| i.field == "volumetrics.step_size"));
    }

    #[test]
    fn gi_volume_dim_not_power_of_2() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.gi.volume_dim = 48;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v.issues.iter().any(|i| i.field == "gi.volume_dim"));
    }

    #[test]
    fn gi_volume_dim_too_small() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.gi.volume_dim = 8;
        let v = validate_config(&cfg);
        assert!(!v.passed);
    }

    #[test]
    fn gi_volume_dim_too_large() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.gi.volume_dim = 512;
        let v = validate_config(&cfg);
        assert!(!v.passed);
    }

    #[test]
    fn shadow_budget_too_high() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.shading.shadow_budget = 20;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "shading.shadow_budget"));
    }

    #[test]
    fn bloom_threshold_zero_when_enabled() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.post_process.bloom_enabled = true;
        cfg.post_process.bloom_threshold = 0.0;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "post_process.bloom_threshold"));
    }

    #[test]
    fn intensity_out_of_range() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.post_process.bloom_intensity = 3.0;
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "post_process.bloom_intensity"));
    }

    #[test]
    fn warning_gi_on_low() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::Low);
        cfg.gi.enabled = true;
        let v = validate_config(&cfg);
        // Should still pass (warning, not error).
        assert!(v.passed);
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "gi.enabled" && i.severity == Severity::Warning));
    }

    #[test]
    fn warning_high_render_scale_non_ultra() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::High);
        cfg.render_scale = 0.9;
        let v = validate_config(&cfg);
        assert!(v.passed); // warning only
        assert!(v
            .issues
            .iter()
            .any(|i| i.field == "render_scale" && i.severity == Severity::Warning));
    }

    #[test]
    fn no_warning_high_render_scale_on_ultra() {
        let mut cfg = EngineConfig::from_preset(QualityPreset::Ultra);
        cfg.render_scale = 0.9;
        let v = validate_config(&cfg);
        // Ultra preset at 0.9 should not warn.
        assert!(!v
            .issues
            .iter()
            .any(|i| i.field == "render_scale" && i.severity == Severity::Warning));
    }

    #[test]
    fn compare_identical_configs() {
        let a = EngineConfig::from_preset(QualityPreset::High);
        let b = EngineConfig::from_preset(QualityPreset::High);
        let cmp = compare_presets(&a, &b);
        assert!(cmp.differences.is_empty());
    }

    #[test]
    fn compare_low_vs_ultra_detects_differences() {
        let a = EngineConfig::from_preset(QualityPreset::Low);
        let b = EngineConfig::from_preset(QualityPreset::Ultra);
        let cmp = compare_presets(&a, &b);
        assert!(!cmp.differences.is_empty());
        assert_eq!(cmp.preset_a, "Low");
        assert_eq!(cmp.preset_b, "Ultra");
        // render_scale must differ.
        assert!(cmp
            .differences
            .iter()
            .any(|d| d.field == "render_scale"));
        // ray_march.max_steps must differ.
        assert!(cmp
            .differences
            .iter()
            .any(|d| d.field == "ray_march.max_steps"));
        // gi.volume_dim must differ.
        assert!(cmp
            .differences
            .iter()
            .any(|d| d.field == "gi.volume_dim"));
    }

    #[test]
    fn compare_single_field_diff() {
        let a = EngineConfig::from_preset(QualityPreset::High);
        let mut b = EngineConfig::from_preset(QualityPreset::High);
        b.render_scale = 0.6;
        // Ensure the preset tag matches so it is not counted as a diff.
        let cmp = compare_presets(&a, &b);
        assert!(cmp.differences.iter().any(|d| d.field == "render_scale"));
        // All other fields should be identical.
        assert!(cmp
            .differences
            .iter()
            .all(|d| d.field == "render_scale"));
    }

    #[test]
    fn validate_all_returns_four() {
        let results = validate_all_presets();
        assert_eq!(results.len(), 4);
        let names: Vec<&str> = results.iter().map(|r| r.preset_name.as_str()).collect();
        assert!(names.contains(&"Low"));
        assert!(names.contains(&"Medium"));
        assert!(names.contains(&"High"));
        assert!(names.contains(&"Ultra"));
    }

    #[test]
    fn severity_levels_present() {
        // Construct a config with both error and warning issues.
        let mut cfg = EngineConfig::from_preset(QualityPreset::Low);
        cfg.gi.enabled = true; // Warning: GI on Low.
        cfg.ray_march.max_steps = 10; // Error: out of range.
        let v = validate_config(&cfg);
        assert!(!v.passed);
        assert!(v.issues.iter().any(|i| i.severity == Severity::Error));
        assert!(v.issues.iter().any(|i| i.severity == Severity::Warning));
    }
}
