//! Frame scheduling for the RKIField render pipeline — stub pending v2 rewrite.
//!
//! Retains [`FrameSettings`] for downstream references.
//! `FrameContext` and `execute_frame` will be rebuilt when the v2 GPU scene
//! and ray marcher are operational (Phase 5+).

/// Controls which optional render passes are enabled for a frame.
///
/// All fields default to `true` — every pass is active unless explicitly
/// disabled. Pass individual fields to `false` to skip that subsystem.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSettings {
    /// Enable volumetric fog passes (shadow map, march, temporal, upscale, composite).
    pub volumetrics_enabled: bool,
    /// Enable cloud shadow map pass.
    pub cloud_shadows_enabled: bool,
    /// Enable global illumination (radiance inject + mip gen).
    pub gi_enabled: bool,
    /// Enable depth of field pass.
    pub dof_enabled: bool,
    /// Enable motion blur pass.
    pub motion_blur_enabled: bool,
    /// Enable bloom passes (extract/blur pre-upscale + composite post-upscale).
    pub bloom_enabled: bool,
    /// Enable automatic exposure adaptation.
    pub auto_exposure_enabled: bool,
    /// Enable color grading LUT pass.
    pub color_grade_enabled: bool,
    /// Enable cosmetic effects (vignette, grain, chromatic aberration).
    pub cosmetics_enabled: bool,
    /// Enable edge-aware sharpening (only meaningful when using the custom upscaler).
    pub sharpen_enabled: bool,
}

impl Default for FrameSettings {
    fn default() -> Self {
        Self {
            volumetrics_enabled: true,
            cloud_shadows_enabled: true,
            gi_enabled: true,
            dof_enabled: true,
            motion_blur_enabled: true,
            bloom_enabled: true,
            auto_exposure_enabled: true,
            color_grade_enabled: true,
            cosmetics_enabled: true,
            sharpen_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_settings_default_all_enabled() {
        let s = FrameSettings::default();
        assert!(s.volumetrics_enabled);
        assert!(s.cloud_shadows_enabled);
        assert!(s.gi_enabled);
        assert!(s.dof_enabled);
        assert!(s.motion_blur_enabled);
        assert!(s.bloom_enabled);
        assert!(s.auto_exposure_enabled);
        assert!(s.color_grade_enabled);
        assert!(s.cosmetics_enabled);
        assert!(s.sharpen_enabled);
    }

    #[test]
    fn frame_settings_field_count() {
        let s = FrameSettings::default();
        let enabled: usize = [
            s.volumetrics_enabled,
            s.cloud_shadows_enabled,
            s.gi_enabled,
            s.dof_enabled,
            s.motion_blur_enabled,
            s.bloom_enabled,
            s.auto_exposure_enabled,
            s.color_grade_enabled,
            s.cosmetics_enabled,
            s.sharpen_enabled,
        ]
        .iter()
        .filter(|&&v| v)
        .count();
        assert_eq!(enabled, 10, "all 10 settings should be enabled by default");
    }

    #[test]
    fn frame_settings_individual_toggle() {
        let mut s = FrameSettings::default();

        s.volumetrics_enabled = false;
        assert!(!s.volumetrics_enabled);
        assert!(s.gi_enabled, "other settings must be unaffected");

        s.gi_enabled = false;
        assert!(!s.gi_enabled);

        s.bloom_enabled = false;
        assert!(!s.bloom_enabled);

        s.sharpen_enabled = false;
        assert!(!s.sharpen_enabled);
    }

    #[test]
    fn frame_settings_minimal() {
        let s = FrameSettings {
            volumetrics_enabled: false,
            cloud_shadows_enabled: false,
            gi_enabled: false,
            dof_enabled: false,
            motion_blur_enabled: false,
            bloom_enabled: false,
            auto_exposure_enabled: false,
            color_grade_enabled: false,
            cosmetics_enabled: false,
            sharpen_enabled: false,
        };
        let enabled: usize = [
            s.volumetrics_enabled,
            s.cloud_shadows_enabled,
            s.gi_enabled,
            s.dof_enabled,
            s.motion_blur_enabled,
            s.bloom_enabled,
            s.auto_exposure_enabled,
            s.color_grade_enabled,
            s.cosmetics_enabled,
            s.sharpen_enabled,
        ]
        .iter()
        .filter(|&&v| v)
        .count();
        assert_eq!(enabled, 0, "all settings disabled — only mandatory passes run");
    }

    #[test]
    fn frame_settings_clone_eq() {
        let a = FrameSettings::default();
        let b = a.clone();
        assert_eq!(a, b);

        let mut c = a.clone();
        c.gi_enabled = false;
        assert_ne!(a, c);
    }
}
