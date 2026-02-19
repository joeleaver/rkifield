//! Environment editing data model for the RKIField editor.
//!
//! Provides settings structs for fog, atmosphere, clouds, and post-processing,
//! plus an `EnvironmentState` container with dirty tracking, reset, and
//! RON serialization. This is a pure data model independent of the GUI framework.

#![allow(dead_code)]

use glam::Vec3;

/// Fog settings for the scene environment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FogSettings {
    /// Whether fog is active.
    pub enabled: bool,
    /// Fog density (higher = thicker).
    pub density: f32,
    /// Fog color in linear RGB (0..1 per channel).
    pub color: Vec3,
    /// Distance from the camera where fog begins.
    pub start_distance: f32,
    /// Distance from the camera where fog reaches full density.
    pub end_distance: f32,
    /// Height-based falloff exponent (higher = fog thins faster with altitude).
    pub height_falloff: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.02,
            color: Vec3::new(0.7, 0.75, 0.8),
            start_distance: 10.0,
            end_distance: 500.0,
            height_falloff: 0.1,
        }
    }
}

/// Atmosphere / sky settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AtmosphereSettings {
    /// Whether atmospheric scattering is active.
    pub enabled: bool,
    /// Scale factor for Rayleigh scattering.
    pub rayleigh_scale: f32,
    /// Scale factor for Mie scattering.
    pub mie_scale: f32,
    /// Direction toward the sun (normalized).
    pub sun_direction: Vec3,
    /// Sun luminous intensity multiplier.
    pub sun_intensity: f32,
    /// Sun disk / scattering tint in linear RGB.
    pub sun_color: Vec3,
}

impl Default for AtmosphereSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            sun_direction: Vec3::new(0.0, -1.0, 0.3).normalize(),
            sun_intensity: 1.0,
            sun_color: Vec3::new(1.0, 0.95, 0.9),
        }
    }
}

/// Cloud layer settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CloudSettings {
    /// Whether clouds are rendered.
    pub enabled: bool,
    /// Cloud coverage fraction (0 = clear, 1 = overcast).
    pub coverage: f32,
    /// Cloud density multiplier.
    pub density: f32,
    /// Cloud layer base altitude in world units.
    pub altitude: f32,
    /// Vertical thickness of the cloud layer.
    pub thickness: f32,
    /// Horizontal wind direction (normalized XZ).
    pub wind_direction: Vec3,
    /// Wind speed in world units per second.
    pub wind_speed: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            coverage: 0.5,
            density: 1.0,
            altitude: 200.0,
            thickness: 50.0,
            wind_direction: Vec3::new(1.0, 0.0, 0.0),
            wind_speed: 5.0,
        }
    }
}

/// Post-processing settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PostProcessSettings {
    /// Whether bloom is enabled.
    pub bloom_enabled: bool,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// Luminance threshold above which bloom activates.
    pub bloom_threshold: f32,
    /// Exposure value (EV adjustment).
    pub exposure: f32,
    /// Contrast adjustment (1.0 = neutral).
    pub contrast: f32,
    /// Saturation adjustment (1.0 = neutral, 0.0 = grayscale).
    pub saturation: f32,
    /// Vignette darkening intensity (0.0 = none).
    pub vignette_intensity: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            vignette_intensity: 0.0,
        }
    }
}

/// Combined environment state for the editor.
///
/// Groups all environment sub-settings and tracks a dirty flag so the
/// runtime knows when to re-upload parameters to the GPU.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnvironmentState {
    /// Fog parameters.
    pub fog: FogSettings,
    /// Atmosphere / sky parameters.
    pub atmosphere: AtmosphereSettings,
    /// Cloud layer parameters.
    pub clouds: CloudSettings,
    /// Post-processing parameters.
    pub post_process: PostProcessSettings,
    /// Whether any setting has changed since the last `clear_dirty()`.
    #[serde(skip)]
    dirty: bool,
}

impl EnvironmentState {
    /// Create a new environment state with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark the state as dirty (needs GPU upload).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear the dirty flag after the runtime has consumed the changes.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Whether any setting has been modified since the last clear.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Reset fog to defaults and mark dirty.
    pub fn reset_fog(&mut self) {
        self.fog = FogSettings::default();
        self.dirty = true;
    }

    /// Reset atmosphere to defaults and mark dirty.
    pub fn reset_atmosphere(&mut self) {
        self.atmosphere = AtmosphereSettings::default();
        self.dirty = true;
    }

    /// Reset clouds to defaults and mark dirty.
    pub fn reset_clouds(&mut self) {
        self.clouds = CloudSettings::default();
        self.dirty = true;
    }

    /// Reset post-processing to defaults and mark dirty.
    pub fn reset_post_process(&mut self) {
        self.post_process = PostProcessSettings::default();
        self.dirty = true;
    }

    /// Reset all settings to defaults and mark dirty.
    pub fn reset_all(&mut self) {
        self.fog = FogSettings::default();
        self.atmosphere = AtmosphereSettings::default();
        self.clouds = CloudSettings::default();
        self.post_process = PostProcessSettings::default();
        self.dirty = true;
    }

    /// Serialize the environment state to a RON string.
    pub fn serialize_to_ron(&self) -> Result<String, String> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
            .map_err(|e| e.to_string())
    }

    /// Deserialize an environment state from a RON string.
    pub fn deserialize_from_ron(s: &str) -> Result<Self, String> {
        ron::from_str(s).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // --- Default values ---

    #[test]
    fn test_fog_defaults() {
        let fog = FogSettings::default();
        assert!(!fog.enabled);
        assert!(approx_eq(fog.density, 0.02));
        assert!(approx_eq(fog.start_distance, 10.0));
        assert!(approx_eq(fog.end_distance, 500.0));
        assert!(approx_eq(fog.height_falloff, 0.1));
    }

    #[test]
    fn test_atmosphere_defaults() {
        let atmo = AtmosphereSettings::default();
        assert!(atmo.enabled);
        assert!(approx_eq(atmo.rayleigh_scale, 1.0));
        assert!(approx_eq(atmo.mie_scale, 1.0));
        assert!(approx_eq(atmo.sun_intensity, 1.0));
    }

    #[test]
    fn test_cloud_defaults() {
        let clouds = CloudSettings::default();
        assert!(!clouds.enabled);
        assert!(approx_eq(clouds.coverage, 0.5));
        assert!(approx_eq(clouds.density, 1.0));
        assert!(approx_eq(clouds.altitude, 200.0));
        assert!(approx_eq(clouds.thickness, 50.0));
        assert!(approx_eq(clouds.wind_speed, 5.0));
    }

    #[test]
    fn test_post_process_defaults() {
        let pp = PostProcessSettings::default();
        assert!(pp.bloom_enabled);
        assert!(approx_eq(pp.bloom_intensity, 0.3));
        assert!(approx_eq(pp.bloom_threshold, 1.0));
        assert!(approx_eq(pp.exposure, 1.0));
        assert!(approx_eq(pp.contrast, 1.0));
        assert!(approx_eq(pp.saturation, 1.0));
        assert!(approx_eq(pp.vignette_intensity, 0.0));
    }

    #[test]
    fn test_environment_state_defaults() {
        let state = EnvironmentState::new();
        assert!(!state.is_dirty());
        assert!(!state.fog.enabled);
        assert!(state.atmosphere.enabled);
        assert!(!state.clouds.enabled);
        assert!(state.post_process.bloom_enabled);
    }

    // --- Dirty flag ---

    #[test]
    fn test_dirty_flag_initial() {
        let state = EnvironmentState::new();
        assert!(!state.is_dirty());
    }

    #[test]
    fn test_mark_dirty() {
        let mut state = EnvironmentState::new();
        state.mark_dirty();
        assert!(state.is_dirty());
    }

    #[test]
    fn test_clear_dirty() {
        let mut state = EnvironmentState::new();
        state.mark_dirty();
        state.clear_dirty();
        assert!(!state.is_dirty());
    }

    // --- Reset methods ---

    #[test]
    fn test_reset_fog() {
        let mut state = EnvironmentState::new();
        state.fog.density = 99.0;
        state.fog.enabled = true;
        state.reset_fog();
        assert!(state.is_dirty());
        assert!(!state.fog.enabled);
        assert!(approx_eq(state.fog.density, 0.02));
    }

    #[test]
    fn test_reset_atmosphere() {
        let mut state = EnvironmentState::new();
        state.atmosphere.rayleigh_scale = 99.0;
        state.reset_atmosphere();
        assert!(state.is_dirty());
        assert!(approx_eq(state.atmosphere.rayleigh_scale, 1.0));
    }

    #[test]
    fn test_reset_clouds() {
        let mut state = EnvironmentState::new();
        state.clouds.coverage = 0.99;
        state.reset_clouds();
        assert!(state.is_dirty());
        assert!(approx_eq(state.clouds.coverage, 0.5));
    }

    #[test]
    fn test_reset_post_process() {
        let mut state = EnvironmentState::new();
        state.post_process.exposure = 5.0;
        state.reset_post_process();
        assert!(state.is_dirty());
        assert!(approx_eq(state.post_process.exposure, 1.0));
    }

    #[test]
    fn test_reset_all() {
        let mut state = EnvironmentState::new();
        state.fog.density = 99.0;
        state.atmosphere.rayleigh_scale = 99.0;
        state.clouds.coverage = 0.99;
        state.post_process.exposure = 5.0;
        state.reset_all();
        assert!(state.is_dirty());
        assert!(approx_eq(state.fog.density, 0.02));
        assert!(approx_eq(state.atmosphere.rayleigh_scale, 1.0));
        assert!(approx_eq(state.clouds.coverage, 0.5));
        assert!(approx_eq(state.post_process.exposure, 1.0));
    }

    // --- Serialization ---

    #[test]
    fn test_serialize_to_ron() {
        let state = EnvironmentState::new();
        let ron_str = state.serialize_to_ron().expect("serialization failed");
        assert!(!ron_str.is_empty());
        assert!(ron_str.contains("fog"));
        assert!(ron_str.contains("atmosphere"));
        assert!(ron_str.contains("clouds"));
        assert!(ron_str.contains("post_process"));
    }

    #[test]
    fn test_deserialize_from_ron() {
        let original = EnvironmentState::new();
        let ron_str = original.serialize_to_ron().unwrap();
        let restored = EnvironmentState::deserialize_from_ron(&ron_str)
            .expect("deserialization failed");
        assert!(approx_eq(restored.fog.density, original.fog.density));
        assert!(approx_eq(
            restored.atmosphere.rayleigh_scale,
            original.atmosphere.rayleigh_scale
        ));
        assert!(approx_eq(restored.clouds.coverage, original.clouds.coverage));
        assert!(approx_eq(
            restored.post_process.exposure,
            original.post_process.exposure
        ));
    }

    #[test]
    fn test_serialize_roundtrip_modified() {
        let mut state = EnvironmentState::new();
        state.fog.enabled = true;
        state.fog.density = 0.05;
        state.atmosphere.sun_intensity = 2.5;
        state.clouds.enabled = true;
        state.clouds.coverage = 0.8;
        state.post_process.bloom_intensity = 0.7;
        state.post_process.vignette_intensity = 0.3;

        let ron_str = state.serialize_to_ron().unwrap();
        let restored = EnvironmentState::deserialize_from_ron(&ron_str).unwrap();

        assert!(restored.fog.enabled);
        assert!(approx_eq(restored.fog.density, 0.05));
        assert!(approx_eq(restored.atmosphere.sun_intensity, 2.5));
        assert!(restored.clouds.enabled);
        assert!(approx_eq(restored.clouds.coverage, 0.8));
        assert!(approx_eq(restored.post_process.bloom_intensity, 0.7));
        assert!(approx_eq(restored.post_process.vignette_intensity, 0.3));
    }

    #[test]
    fn test_deserialize_dirty_flag_not_persisted() {
        let mut state = EnvironmentState::new();
        state.mark_dirty();
        let ron_str = state.serialize_to_ron().unwrap();
        let restored = EnvironmentState::deserialize_from_ron(&ron_str).unwrap();
        // dirty is #[serde(skip)] so it should be false after deserialization
        assert!(!restored.is_dirty());
    }

    #[test]
    fn test_deserialize_invalid_ron() {
        let result = EnvironmentState::deserialize_from_ron("not valid ron {{{");
        assert!(result.is_err());
    }
}
