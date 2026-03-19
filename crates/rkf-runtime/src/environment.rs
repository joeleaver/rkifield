//! Environment profiles (.rkenv) for the v2 RKIField engine.
//!
//! An [`EnvironmentProfile`] captures sky rendering mode, fog, ambient lighting,
//! volumetric hints, and post-processing hints. Profiles can be blended with
//! [`lerp_profiles`]. Serialisation uses RON (`.rkenv` files).
//!
//! [`EnvironmentSettings`] is the full ECS component that lives on each camera
//! entity — the single source of truth for environment parameters per camera.

use serde::{Deserialize, Serialize};

// ─── Sky ──────────────────────────────────────────────────────────────────────

/// Sky rendering mode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SkyMode {
    /// Solid color fill.
    SolidColor {
        /// Linear RGB sky color.
        color: [f32; 3],
    },
    /// Gradient from horizon to zenith.
    Gradient {
        /// Linear RGB color at the horizon.
        horizon: [f32; 3],
        /// Linear RGB color at the zenith.
        zenith: [f32; 3],
    },
    /// Procedural atmosphere with Rayleigh/Mie scattering.
    Atmosphere {
        /// Normalized world-space direction toward the sun.
        sun_direction: [f32; 3],
        /// Sun disk intensity multiplier.
        sun_intensity: f32,
        /// Rayleigh scattering coefficient (controls sky blue-ness).
        rayleigh_coefficient: f32,
        /// Mie scattering coefficient (controls haze/glow around the sun).
        mie_coefficient: f32,
    },
}

impl Default for SkyMode {
    fn default() -> Self {
        SkyMode::Gradient {
            horizon: [0.8, 0.85, 0.9],
            zenith: [0.4, 0.6, 0.9],
        }
    }
}

// ─── Fog ──────────────────────────────────────────────────────────────────────

/// Fog configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FogConfig {
    /// Whether fog is active.
    pub enabled: bool,
    /// Linear RGB fog color.
    pub color: [f32; 3],
    /// Exponential fog density coefficient.
    pub density: f32,
    /// Distance at which fog begins (world units).
    pub start_distance: f32,
    /// Distance at which fog reaches full opacity (world units).
    pub end_distance: f32,
    /// Vertical height falloff — higher values thin the fog faster with altitude.
    pub height_falloff: f32,
}

impl Default for FogConfig {
    fn default() -> Self {
        FogConfig {
            enabled: false,
            color: [0.7, 0.7, 0.8],
            density: 0.01,
            start_distance: 50.0,
            end_distance: 500.0,
            height_falloff: 0.1,
        }
    }
}

// ─── Ambient ──────────────────────────────────────────────────────────────────

/// Ambient lighting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AmbientConfig {
    /// Linear RGB tint of the ambient term.
    pub color: [f32; 3],
    /// Overall ambient intensity multiplier.
    pub intensity: f32,
    /// Sky light contribution (indirect sky irradiance weight).
    pub sky_light_intensity: f32,
}

impl Default for AmbientConfig {
    fn default() -> Self {
        AmbientConfig {
            color: [0.5, 0.5, 0.6],
            intensity: 0.3,
            sky_light_intensity: 0.5,
        }
    }
}

// ─── Volumetrics ──────────────────────────────────────────────────────────────

/// Volumetric effects hints passed to the volumetric march pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VolumetricHints {
    /// Whether the volumetric march pass is active.
    pub enabled: bool,
    /// In-scatter coefficient (σ_s) in m⁻¹.
    pub scattering_coefficient: f32,
    /// Absorption coefficient (σ_a) in m⁻¹.
    pub absorption_coefficient: f32,
    /// Henyey-Greenstein asymmetry parameter g ∈ (−1, 1).
    pub phase_g: f32,
}

impl Default for VolumetricHints {
    fn default() -> Self {
        VolumetricHints {
            enabled: false,
            scattering_coefficient: 0.01,
            absorption_coefficient: 0.001,
            phase_g: 0.8,
        }
    }
}

// ─── Post-process hints ───────────────────────────────────────────────────────

/// Post-processing hints — an environment can suggest post-FX settings.
///
/// The renderer may ignore or blend these with user-configured values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostProcessHints {
    /// EV offset applied on top of auto-exposure (positive = brighter).
    pub exposure_compensation: f32,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// Color temperature in Kelvin (6500 = neutral daylight).
    pub color_temperature: f32,
    /// Saturation multiplier (1.0 = unchanged, 0.0 = greyscale).
    pub saturation: f32,
}

impl Default for PostProcessHints {
    fn default() -> Self {
        PostProcessHints {
            exposure_compensation: 0.0,
            bloom_intensity: 0.5,
            color_temperature: 6500.0,
            saturation: 1.0,
        }
    }
}

// ─── Complete profile ─────────────────────────────────────────────────────────

/// Complete environment profile — the root asset for `.rkenv` files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvironmentProfile {
    /// Human-readable profile name (used for display and blend labelling).
    pub name: String,
    /// Sky rendering configuration.
    pub sky: SkyMode,
    /// Fog configuration.
    pub fog: FogConfig,
    /// Ambient lighting configuration.
    pub ambient: AmbientConfig,
    /// Volumetric effects hints.
    pub volumetric: VolumetricHints,
    /// Post-processing hints.
    pub post_hints: PostProcessHints,
}

impl Default for EnvironmentProfile {
    fn default() -> Self {
        EnvironmentProfile {
            name: String::from("Default"),
            sky: SkyMode::default(),
            fog: FogConfig::default(),
            ambient: AmbientConfig::default(),
            volumetric: VolumetricHints::default(),
            post_hints: PostProcessHints::default(),
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp_f32(a[0], b[0], t),
        lerp_f32(a[1], b[1], t),
        lerp_f32(a[2], b[2], t),
    ]
}

/// Lerp between two [`SkyMode`] values.
///
/// If both variants are the same, the float parameters are interpolated.
/// If the variants differ, `b` is returned when `t > 0.5`, otherwise `a`.
fn lerp_sky(a: &SkyMode, b: &SkyMode, t: f32) -> SkyMode {
    match (a, b) {
        (SkyMode::SolidColor { color: ca }, SkyMode::SolidColor { color: cb }) => {
            SkyMode::SolidColor { color: lerp_color(*ca, *cb, t) }
        }
        (
            SkyMode::Gradient { horizon: ha, zenith: za },
            SkyMode::Gradient { horizon: hb, zenith: zb },
        ) => SkyMode::Gradient {
            horizon: lerp_color(*ha, *hb, t),
            zenith: lerp_color(*za, *zb, t),
        },
        (
            SkyMode::Atmosphere {
                sun_direction: sda,
                sun_intensity: sia,
                rayleigh_coefficient: rca,
                mie_coefficient: mca,
            },
            SkyMode::Atmosphere {
                sun_direction: sdb,
                sun_intensity: sib,
                rayleigh_coefficient: rcb,
                mie_coefficient: mcb,
            },
        ) => SkyMode::Atmosphere {
            sun_direction: lerp_color(*sda, *sdb, t),
            sun_intensity: lerp_f32(*sia, *sib, t),
            rayleigh_coefficient: lerp_f32(*rca, *rcb, t),
            mie_coefficient: lerp_f32(*mca, *mcb, t),
        },
        // Different variants: cross-fade at t == 0.5.
        _ => {
            if t > 0.5 {
                b.clone()
            } else {
                a.clone()
            }
        }
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Linearly interpolate between two environment profiles.
///
/// - `f32` fields: linear lerp.
/// - `[f32; 3]` color fields: component-wise lerp.
/// - `bool` fields: use `b` value when `t > 0.5`.
/// - [`SkyMode`]: if both share the same variant, lerp parameters; otherwise
///   use `b` when `t > 0.5`.
/// - `name`: use `b.name` when `t > 0.5`, otherwise `a.name`.
///
/// `t` is clamped to `[0.0, 1.0]`. At exactly `0.0` the original `a` is
/// returned unchanged; at exactly `1.0` the original `b` is returned
/// unchanged (avoiding floating-point rounding at the endpoints).
pub fn lerp_profiles(a: &EnvironmentProfile, b: &EnvironmentProfile, t: f32) -> EnvironmentProfile {
    let t = t.clamp(0.0, 1.0);
    // Fast path: avoid floating-point rounding at the exact endpoints.
    if t == 0.0 {
        return a.clone();
    }
    if t == 1.0 {
        return b.clone();
    }

    let name = if t > 0.5 { b.name.clone() } else { a.name.clone() };

    let sky = lerp_sky(&a.sky, &b.sky, t);

    let fog = FogConfig {
        enabled: if t > 0.5 { b.fog.enabled } else { a.fog.enabled },
        color: lerp_color(a.fog.color, b.fog.color, t),
        density: lerp_f32(a.fog.density, b.fog.density, t),
        start_distance: lerp_f32(a.fog.start_distance, b.fog.start_distance, t),
        end_distance: lerp_f32(a.fog.end_distance, b.fog.end_distance, t),
        height_falloff: lerp_f32(a.fog.height_falloff, b.fog.height_falloff, t),
    };

    let ambient = AmbientConfig {
        color: lerp_color(a.ambient.color, b.ambient.color, t),
        intensity: lerp_f32(a.ambient.intensity, b.ambient.intensity, t),
        sky_light_intensity: lerp_f32(
            a.ambient.sky_light_intensity,
            b.ambient.sky_light_intensity,
            t,
        ),
    };

    let volumetric = VolumetricHints {
        enabled: if t > 0.5 { b.volumetric.enabled } else { a.volumetric.enabled },
        scattering_coefficient: lerp_f32(
            a.volumetric.scattering_coefficient,
            b.volumetric.scattering_coefficient,
            t,
        ),
        absorption_coefficient: lerp_f32(
            a.volumetric.absorption_coefficient,
            b.volumetric.absorption_coefficient,
            t,
        ),
        phase_g: lerp_f32(a.volumetric.phase_g, b.volumetric.phase_g, t),
    };

    let post_hints = PostProcessHints {
        exposure_compensation: lerp_f32(
            a.post_hints.exposure_compensation,
            b.post_hints.exposure_compensation,
            t,
        ),
        bloom_intensity: lerp_f32(a.post_hints.bloom_intensity, b.post_hints.bloom_intensity, t),
        color_temperature: lerp_f32(
            a.post_hints.color_temperature,
            b.post_hints.color_temperature,
            t,
        ),
        saturation: lerp_f32(a.post_hints.saturation, b.post_hints.saturation, t),
    };

    EnvironmentProfile { name, sky, fog, ambient, volumetric, post_hints }
}

/// Load an environment profile from a `.rkenv` file (RON format).
pub fn load_environment(path: &str) -> anyhow::Result<EnvironmentProfile> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path, e))?;
    let profile: EnvironmentProfile = ron::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse {}: {}", path, e))?;
    Ok(profile)
}

/// Save an environment profile to a `.rkenv` file (RON format).
pub fn save_environment(path: &str, env: &EnvironmentProfile) -> anyhow::Result<()> {
    let text = ron::ser::to_string_pretty(env, ron::ser::PrettyConfig::default())
        .map_err(|e| anyhow::anyhow!("failed to serialize environment: {}", e))?;
    std::fs::write(path, text)
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", path, e))?;
    Ok(())
}

// ─── ECS environment component ───────────────────────────────────────────────

/// Fog settings for `EnvironmentSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FogSettings {
    pub enabled: bool,
    pub density: f32,
    pub color: [f32; 3],
    pub start_distance: f32,
    pub end_distance: f32,
    pub height_falloff: f32,
    pub ambient_dust_density: f32,
    pub dust_asymmetry: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 0.02,
            color: [0.7, 0.75, 0.8],
            start_distance: 10.0,
            end_distance: 500.0,
            height_falloff: 0.1,
            ambient_dust_density: 0.005,
            dust_asymmetry: 0.3,
        }
    }
}

/// Atmosphere settings for `EnvironmentSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtmosphereSettings {
    pub enabled: bool,
    pub rayleigh_scale: f32,
    pub mie_scale: f32,
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
}

impl Default for AtmosphereSettings {
    fn default() -> Self {
        let dir = {
            let v = [0.5f32, 1.0, 0.3];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / len, v[1] / len, v[2] / len]
        };
        Self {
            enabled: true,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            sun_direction: dir,
            sun_intensity: 3.0,
            sun_color: [1.0, 0.95, 0.85],
        }
    }
}

/// Cloud layer settings for `EnvironmentSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CloudSettings {
    pub enabled: bool,
    pub coverage: f32,
    pub density: f32,
    pub altitude: f32,
    pub thickness: f32,
    pub wind_direction: [f32; 3],
    pub wind_speed: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            coverage: 0.5,
            density: 1.0,
            altitude: 200.0,
            thickness: 1000.0,
            wind_direction: [1.0, 0.0, 0.0],
            wind_speed: 5.0,
        }
    }
}

/// Post-processing settings for `EnvironmentSettings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostProcessSettings {
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub vignette_intensity: f32,
    pub tone_map_mode: u32,
    pub sharpen_strength: f32,
    pub dof_enabled: bool,
    pub dof_focus_distance: f32,
    pub dof_focus_range: f32,
    pub dof_max_coc: f32,
    pub motion_blur_intensity: f32,
    pub god_rays_intensity: f32,
    pub grain_intensity: f32,
    pub chromatic_aberration: f32,
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
            tone_map_mode: 0,
            sharpen_strength: 0.5,
            dof_enabled: false,
            dof_focus_distance: 2.0,
            dof_focus_range: 3.0,
            dof_max_coc: 8.0,
            motion_blur_intensity: 1.0,
            god_rays_intensity: 0.5,
            grain_intensity: 0.0,
            chromatic_aberration: 0.0,
        }
    }
}

/// Full environment settings — the ECS single source of truth.
///
/// Lives on each camera entity. Matches the renderer's full parameter set.
/// The editor environment panel and linked camera resolution both read and
/// write this component on the active camera.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EnvironmentSettings {
    pub fog: FogSettings,
    pub atmosphere: AtmosphereSettings,
    pub clouds: CloudSettings,
    pub post_process: PostProcessSettings,
}

impl EnvironmentSettings {
    /// Create settings from an `EnvironmentProfile` (.rkenv file).
    ///
    /// Maps profile fields to the full settings struct. Fields not present
    /// in the profile (clouds, most post-process) use defaults.
    pub fn from_profile(profile: &EnvironmentProfile) -> Self {
        let mut s = Self::default();

        // Fog
        s.fog.enabled = profile.fog.enabled;
        s.fog.density = profile.fog.density;
        s.fog.color = profile.fog.color;
        s.fog.start_distance = profile.fog.start_distance;
        s.fog.end_distance = profile.fog.end_distance;
        s.fog.height_falloff = profile.fog.height_falloff;

        // Atmosphere (from SkyMode if Atmosphere variant)
        if let SkyMode::Atmosphere {
            sun_direction,
            sun_intensity,
            rayleigh_coefficient,
            mie_coefficient,
        } = &profile.sky
        {
            s.atmosphere.enabled = true;
            s.atmosphere.sun_direction = *sun_direction;
            s.atmosphere.sun_intensity = *sun_intensity;
            s.atmosphere.rayleigh_scale = *rayleigh_coefficient;
            s.atmosphere.mie_scale = *mie_coefficient;
        }

        // Post-process hints
        s.post_process.exposure = profile.post_hints.exposure_compensation;
        s.post_process.bloom_intensity = profile.post_hints.bloom_intensity;
        s.post_process.saturation = profile.post_hints.saturation;

        s
    }

    /// Convert to an `EnvironmentProfile` for .rkenv export.
    pub fn to_profile(&self, name: &str) -> EnvironmentProfile {
        let sky = if self.atmosphere.enabled {
            SkyMode::Atmosphere {
                sun_direction: self.atmosphere.sun_direction,
                sun_intensity: self.atmosphere.sun_intensity,
                rayleigh_coefficient: self.atmosphere.rayleigh_scale,
                mie_coefficient: self.atmosphere.mie_scale,
            }
        } else {
            SkyMode::default()
        };
        EnvironmentProfile {
            name: name.to_string(),
            sky,
            fog: FogConfig {
                enabled: self.fog.enabled,
                color: self.fog.color,
                density: self.fog.density,
                start_distance: self.fog.start_distance,
                end_distance: self.fog.end_distance,
                height_falloff: self.fog.height_falloff,
            },
            ambient: AmbientConfig::default(),
            volumetric: VolumetricHints::default(),
            post_hints: PostProcessHints {
                exposure_compensation: self.post_process.exposure,
                bloom_intensity: self.post_process.bloom_intensity,
                color_temperature: 6500.0,
                saturation: self.post_process.saturation,
            },
        }
    }
}


// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn color_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    /// Build a non-default profile to use as a blend target.
    fn profile_b() -> EnvironmentProfile {
        EnvironmentProfile {
            name: String::from("Night"),
            sky: SkyMode::Gradient {
                horizon: [0.1, 0.1, 0.15],
                zenith: [0.0, 0.0, 0.05],
            },
            fog: FogConfig {
                enabled: true,
                color: [0.2, 0.2, 0.3],
                density: 0.05,
                start_distance: 10.0,
                end_distance: 100.0,
                height_falloff: 0.5,
            },
            ambient: AmbientConfig {
                color: [0.1, 0.1, 0.2],
                intensity: 0.05,
                sky_light_intensity: 0.1,
            },
            volumetric: VolumetricHints {
                enabled: true,
                scattering_coefficient: 0.1,
                absorption_coefficient: 0.01,
                phase_g: 0.5,
            },
            post_hints: PostProcessHints {
                exposure_compensation: -1.0,
                bloom_intensity: 1.5,
                color_temperature: 4000.0,
                saturation: 0.8,
            },
        }
    }

    // 1. Default profile — sensible defaults.
    #[test]
    fn default_profile() {
        let p = EnvironmentProfile::default();
        assert_eq!(p.name, "Default");
        assert!(!p.fog.enabled);
        assert!(approx_eq(p.fog.density, 0.01));
        assert!(approx_eq(p.ambient.intensity, 0.3));
        assert!(approx_eq(p.post_hints.bloom_intensity, 0.5));
        assert!(approx_eq(p.post_hints.color_temperature, 6500.0));
        assert!(approx_eq(p.post_hints.saturation, 1.0));
        assert!(!p.volumetric.enabled);
        assert!(approx_eq(p.volumetric.phase_g, 0.8));
    }

    // 2. lerp at t=0 returns first profile.
    #[test]
    fn lerp_at_zero() {
        let a = EnvironmentProfile::default();
        let b = profile_b();
        let result = lerp_profiles(&a, &b, 0.0);
        assert_eq!(result, a);
    }

    // 3. lerp at t=1 returns second profile.
    #[test]
    fn lerp_at_one() {
        let a = EnvironmentProfile::default();
        let b = profile_b();
        let result = lerp_profiles(&a, &b, 1.0);
        assert_eq!(result, b);
    }

    // 4. lerp at midpoint interpolates scalar values.
    #[test]
    fn lerp_midpoint() {
        let a = EnvironmentProfile::default();
        let b = profile_b();
        let r = lerp_profiles(&a, &b, 0.5);
        // Ambient intensity: 0.3 and 0.05 → midpoint 0.175
        assert!(approx_eq(r.ambient.intensity, (0.3 + 0.05) * 0.5));
        // Fog density: 0.01 and 0.05 → midpoint 0.03
        assert!(approx_eq(r.fog.density, (0.01 + 0.05) * 0.5));
        // Post bloom: 0.5 and 1.5 → midpoint 1.0
        assert!(approx_eq(r.post_hints.bloom_intensity, (0.5 + 1.5) * 0.5));
    }

    // 5. lerp interpolates fog color correctly.
    #[test]
    fn lerp_fog_color() {
        let a = EnvironmentProfile::default(); // [0.7, 0.7, 0.8]
        let b = profile_b();                  // [0.2, 0.2, 0.3]
        let r = lerp_profiles(&a, &b, 0.5);
        let expected = lerp_color([0.7, 0.7, 0.8], [0.2, 0.2, 0.3], 0.5);
        assert!(color_approx_eq(r.fog.color, expected));
    }

    // 6. save and load roundtrip via temp file.
    #[test]
    fn save_and_load_roundtrip() {
        let original = profile_b();
        let tmp = std::env::temp_dir().join("test_env_roundtrip.rkenv");
        let path = tmp.to_str().unwrap();

        save_environment(path, &original).expect("save_environment failed");
        let loaded = load_environment(path).expect("load_environment failed");
        assert_eq!(loaded, original);

        let _ = std::fs::remove_file(tmp);
    }

    // ── EnvironmentSettings tests ──────────────────────────────────

    #[test]
    fn env_settings_defaults() {
        let s = EnvironmentSettings::default();
        assert!(!s.fog.enabled);
        assert!(approx_eq(s.fog.density, 0.02));
        assert!(s.atmosphere.enabled);
        assert!(approx_eq(s.atmosphere.sun_intensity, 3.0));
        assert!(!s.clouds.enabled);
        assert!(approx_eq(s.clouds.coverage, 0.5));
        assert!(s.post_process.bloom_enabled);
        assert!(approx_eq(s.post_process.exposure, 1.0));
    }

    #[test]
    fn env_settings_from_profile_fog() {
        let mut p = EnvironmentProfile::default();
        p.fog.enabled = true;
        p.fog.density = 0.07;
        p.fog.color = [0.1, 0.2, 0.3];
        let s = EnvironmentSettings::from_profile(&p);
        assert!(s.fog.enabled);
        assert!(approx_eq(s.fog.density, 0.07));
        assert!(color_approx_eq(s.fog.color, [0.1, 0.2, 0.3]));
    }

    #[test]
    fn env_settings_from_profile_atmosphere() {
        let p = EnvironmentProfile {
            sky: SkyMode::Atmosphere {
                sun_direction: [0.0, 1.0, 0.0],
                sun_intensity: 5.0,
                rayleigh_coefficient: 2.0,
                mie_coefficient: 0.5,
            },
            ..Default::default()
        };
        let s = EnvironmentSettings::from_profile(&p);
        assert!(s.atmosphere.enabled);
        assert!(approx_eq(s.atmosphere.sun_intensity, 5.0));
        assert!(approx_eq(s.atmosphere.rayleigh_scale, 2.0));
        assert!(approx_eq(s.atmosphere.mie_scale, 0.5));
    }

    #[test]
    fn env_settings_from_profile_non_atmosphere_sky() {
        // Gradient sky should leave atmosphere at defaults (enabled=true from default)
        let p = EnvironmentProfile::default(); // Gradient sky
        let s = EnvironmentSettings::from_profile(&p);
        // Atmosphere fields stay at defaults since sky isn't Atmosphere variant
        assert!(s.atmosphere.enabled); // default is true
        assert!(approx_eq(s.atmosphere.rayleigh_scale, 1.0));
    }

    #[test]
    fn env_settings_to_profile_roundtrip() {
        let mut s = EnvironmentSettings::default();
        s.fog.enabled = true;
        s.fog.density = 0.05;
        s.atmosphere.sun_intensity = 4.0;
        s.post_process.bloom_intensity = 0.8;

        let p = s.to_profile("Test");
        assert_eq!(p.name, "Test");
        assert!(p.fog.enabled);
        assert!(approx_eq(p.fog.density, 0.05));
        assert!(approx_eq(p.post_hints.bloom_intensity, 0.8));
        // Atmosphere enabled → SkyMode::Atmosphere
        if let SkyMode::Atmosphere { sun_intensity, .. } = &p.sky {
            assert!(approx_eq(*sun_intensity, 4.0));
        } else {
            panic!("expected SkyMode::Atmosphere");
        }
    }

    #[test]
    fn env_settings_serialize_roundtrip() {
        let mut s = EnvironmentSettings::default();
        s.fog.density = 0.05;
        s.atmosphere.sun_intensity = 4.0;
        let ron_str = ron::to_string(&s).unwrap();
        let restored: EnvironmentSettings = ron::from_str(&ron_str).unwrap();
        assert_eq!(s, restored);
    }

}
