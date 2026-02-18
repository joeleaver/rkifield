//! Analytic fog types — Phase 11 task 11.4.
//!
//! Provides [`FogSettings`] (CPU-side configuration) and [`FogParams`] (GPU-uploadable
//! packed representation) for height fog and distance fog.
//!
//! Height fog uses an exponential density falloff above a base height. Distance fog
//! increases monotonically with camera distance. Both are accumulated in the volumetric
//! march pass alongside ambient dust.

use bytemuck::{Pod, Zeroable};

// ---------- Constants ----------

/// Default height-fog base density (0.0 = off by default).
pub const DEFAULT_FOG_BASE_DENSITY: f32 = 0.0;

/// Default fog base height — sea level (0.0 m).
pub const DEFAULT_FOG_BASE_HEIGHT: f32 = 0.0;

/// Default height-fog falloff exponent — gentle exponential decay above base.
pub const DEFAULT_FOG_HEIGHT_FALLOFF: f32 = 0.1;

/// Default fog scattering color — bluish-white haze.
pub const DEFAULT_FOG_COLOR: [f32; 3] = [0.7, 0.8, 0.9];

/// Default distance-fog density coefficient (0.0 = off by default).
pub const DEFAULT_DISTANCE_FOG_DENSITY: f32 = 0.0;

/// Default distance-fog falloff exponent.
pub const DEFAULT_DISTANCE_FOG_FALLOFF: f32 = 0.01;

// ---------- CPU-side settings ----------

/// CPU-side fog configuration.
///
/// This is the user-facing struct for configuring analytic fog. It gets packed
/// into [`FogParams`] for GPU upload via [`FogParams::from_settings`].
///
/// Both height fog and distance fog are **off by default** (`enabled = false`).
/// Setting the density fields to non-zero values while the enable flag is `false`
/// has no effect on rendering.
#[derive(Debug, Clone)]
pub struct FogSettings {
    /// Enable height-fog (exponential density falloff above `fog_base_height`).
    pub height_fog_enabled: bool,
    /// Peak volumetric extinction coefficient at `fog_base_height` (m⁻¹).
    pub fog_base_density: f32,
    /// World-space height (m) below which fog is at maximum density.
    pub fog_base_height: f32,
    /// Exponential falloff rate above `fog_base_height` (larger = thinner at altitude).
    pub fog_height_falloff: f32,
    /// Fog scattering color (linear RGB). Used as the single-scattering albedo.
    pub fog_color: [f32; 3],
    /// Enable distance fog (density accumulates with camera distance).
    pub distance_fog_enabled: bool,
    /// Distance-fog extinction coefficient scale (m⁻¹).
    pub fog_distance_density: f32,
    /// Distance-fog falloff: how quickly density rises with distance.
    pub fog_distance_falloff: f32,
    /// Uniform ambient dust density (m⁻¹). Replaces the old scalar field in vol_march.
    pub ambient_dust_density: f32,
    /// Henyey-Greenstein asymmetry for ambient dust (−1..1; 0 = isotropic).
    pub ambient_dust_g: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            height_fog_enabled: false,
            fog_base_density: DEFAULT_FOG_BASE_DENSITY,
            fog_base_height: DEFAULT_FOG_BASE_HEIGHT,
            fog_height_falloff: DEFAULT_FOG_HEIGHT_FALLOFF,
            fog_color: DEFAULT_FOG_COLOR,
            distance_fog_enabled: false,
            fog_distance_density: DEFAULT_DISTANCE_FOG_DENSITY,
            fog_distance_falloff: DEFAULT_DISTANCE_FOG_FALLOFF,
            ambient_dust_density: 0.0,
            ambient_dust_g: 0.3,
        }
    }
}

// ---------- GPU-uploadable params ----------

/// GPU-uploadable fog parameters (64 bytes, 16-byte aligned).
///
/// Packed for uniform buffer upload. Boolean enable flags are stored as `f32`
/// (0.0 = disabled, 1.0 = enabled) for GPU compatibility.
///
/// Memory layout:
/// ```text
/// offset  0 — fog_color    [f32; 4]  xyz = RGB color, w = height_fog_enable
/// offset 16 — fog_height   [f32; 4]  x = base_density, y = base_height,
///                                     z = height_falloff, w = distance_fog_enable
/// offset 32 — fog_distance [f32; 4]  x = distance_density, y = distance_falloff,
///                                     z = ambient_dust_density, w = ambient_dust_g
/// offset 48 — _pad         [f32; 4]  (reserved)
/// total: 64 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct FogParams {
    /// Fog scattering color (linear RGB). `w` = height fog enable (0.0 / 1.0).
    pub fog_color: [f32; 4],
    /// `x` = base density, `y` = base height, `z` = height falloff.
    /// `w` = distance fog enable (0.0 / 1.0).
    pub fog_height: [f32; 4],
    /// `x` = distance density, `y` = distance falloff,
    /// `z` = ambient dust density, `w` = ambient dust g.
    pub fog_distance: [f32; 4],
    /// Padding to 64 bytes (reserved for future use).
    pub _pad: [f32; 4],
}

impl FogParams {
    /// Pack CPU [`FogSettings`] into GPU-uploadable params.
    pub fn from_settings(settings: &FogSettings) -> Self {
        Self {
            fog_color: [
                settings.fog_color[0],
                settings.fog_color[1],
                settings.fog_color[2],
                if settings.height_fog_enabled { 1.0 } else { 0.0 },
            ],
            fog_height: [
                settings.fog_base_density,
                settings.fog_base_height,
                settings.fog_height_falloff,
                if settings.distance_fog_enabled { 1.0 } else { 0.0 },
            ],
            fog_distance: [
                settings.fog_distance_density,
                settings.fog_distance_falloff,
                settings.ambient_dust_density,
                settings.ambient_dust_g,
            ],
            _pad: [0.0; 4],
        }
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fog_params_size_is_64() {
        assert_eq!(std::mem::size_of::<FogParams>(), 64);
    }

    #[test]
    fn fog_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(FogParams, fog_color), 0);
        assert_eq!(std::mem::offset_of!(FogParams, fog_height), 16);
        assert_eq!(std::mem::offset_of!(FogParams, fog_distance), 32);
        assert_eq!(std::mem::offset_of!(FogParams, _pad), 48);
    }

    #[test]
    fn fog_params_pod_roundtrip() {
        let s = FogSettings {
            height_fog_enabled: true,
            fog_base_density: 0.5,
            fog_base_height: 10.0,
            fog_height_falloff: 0.2,
            fog_color: [0.8, 0.85, 0.9],
            distance_fog_enabled: true,
            fog_distance_density: 0.01,
            fog_distance_falloff: 0.005,
            ambient_dust_density: 0.003,
            ambient_dust_g: 0.4,
        };
        let p = FogParams::from_settings(&s);
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 64);
        let p2: &FogParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.fog_color[3], 1.0); // height fog enabled
        assert_eq!(p.fog_height[3], 1.0); // distance fog enabled
        assert!((p2.fog_distance[2] - 0.003).abs() < 1e-7); // dust density preserved
        assert!((p2.fog_height[0] - 0.5).abs() < 1e-7); // base density preserved
    }

    #[test]
    fn fog_settings_default_all_off() {
        let s = FogSettings::default();
        assert!(!s.height_fog_enabled);
        assert!(!s.distance_fog_enabled);
        assert_eq!(s.fog_base_density, 0.0);
        assert_eq!(s.fog_distance_density, 0.0);
        assert_eq!(s.ambient_dust_density, 0.0);
    }

    #[test]
    fn fog_params_enable_flags() {
        let mut s = FogSettings::default();
        let p = FogParams::from_settings(&s);
        assert_eq!(p.fog_color[3], 0.0); // height fog disabled
        assert_eq!(p.fog_height[3], 0.0); // distance fog disabled

        s.height_fog_enabled = true;
        s.distance_fog_enabled = true;
        let p = FogParams::from_settings(&s);
        assert_eq!(p.fog_color[3], 1.0); // height fog enabled
        assert_eq!(p.fog_height[3], 1.0); // distance fog enabled
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_FOG_BASE_DENSITY, 0.0);
        assert_eq!(DEFAULT_DISTANCE_FOG_DENSITY, 0.0);
        assert!(DEFAULT_FOG_HEIGHT_FALLOFF > 0.0);
        assert!(DEFAULT_DISTANCE_FOG_FALLOFF > 0.0);
    }

    #[test]
    fn fog_color_default_is_bluish_white() {
        // Ensure the default color is in a valid range.
        for c in DEFAULT_FOG_COLOR {
            assert!(c >= 0.0 && c <= 1.0, "color component {c} out of [0,1]");
        }
    }

    #[test]
    fn fog_params_dust_packed_in_distance() {
        let s = FogSettings {
            ambient_dust_density: 0.007,
            ambient_dust_g: 0.6,
            ..FogSettings::default()
        };
        let p = FogParams::from_settings(&s);
        assert!((p.fog_distance[2] - 0.007).abs() < 1e-7);
        assert!((p.fog_distance[3] - 0.6).abs() < 1e-7);
    }
}
