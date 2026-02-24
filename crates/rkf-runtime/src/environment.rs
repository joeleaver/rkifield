//! Environment profiles (.rkenv) for the v2 RKIField engine.
//!
//! An [`EnvironmentProfile`] captures sky rendering mode, fog, ambient lighting,
//! volumetric hints, and post-processing hints. Profiles can be blended with
//! [`lerp_profiles`], augmented with per-property [`EnvironmentOverrides`] via
//! [`apply_overrides`], and resolved through the full chain with
//! [`resolve_environment`]. Serialisation uses RON (`.rkenv` files).

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

// ─── Overrides ────────────────────────────────────────────────────────────────

/// Per-property overrides that can be layered on top of a resolved profile.
///
/// Only `Some` fields are applied; `None` fields leave the base value unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentOverrides {
    /// Override for [`FogConfig::density`].
    pub fog_density: Option<f32>,
    /// Override for [`FogConfig::color`].
    pub fog_color: Option<[f32; 3]>,
    /// Override for [`AmbientConfig::intensity`].
    pub ambient_intensity: Option<f32>,
    /// Override for [`PostProcessHints::exposure_compensation`].
    pub exposure_compensation: Option<f32>,
    /// Override for [`PostProcessHints::bloom_intensity`].
    pub bloom_intensity: Option<f32>,
    /// Override for [`VolumetricHints::scattering_coefficient`].
    pub volumetric_scattering: Option<f32>,
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

/// Apply per-property overrides to a profile, returning a new profile.
///
/// Only fields that are `Some(...)` in `overrides` are modified; the rest are
/// copied unchanged from `base`.
pub fn apply_overrides(
    base: &EnvironmentProfile,
    overrides: &EnvironmentOverrides,
) -> EnvironmentProfile {
    let mut out = base.clone();
    if let Some(d) = overrides.fog_density {
        out.fog.density = d;
    }
    if let Some(c) = overrides.fog_color {
        out.fog.color = c;
    }
    if let Some(i) = overrides.ambient_intensity {
        out.ambient.intensity = i;
    }
    if let Some(e) = overrides.exposure_compensation {
        out.post_hints.exposure_compensation = e;
    }
    if let Some(bl) = overrides.bloom_intensity {
        out.post_hints.bloom_intensity = bl;
    }
    if let Some(s) = overrides.volumetric_scattering {
        out.volumetric.scattering_coefficient = s;
    }
    out
}

/// Resolve the final environment: `base` → blend toward `target` at `blend_t` →
/// apply `overrides`.
///
/// If `target` is `None`, `blend_t` is ignored and `base` is used directly.
pub fn resolve_environment(
    base: &EnvironmentProfile,
    target: Option<&EnvironmentProfile>,
    blend_t: f32,
    overrides: &EnvironmentOverrides,
) -> EnvironmentProfile {
    let blended = match target {
        Some(tgt) => lerp_profiles(base, tgt, blend_t),
        None => base.clone(),
    };
    apply_overrides(&blended, overrides)
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

    // 6. Override replaces fog density.
    #[test]
    fn override_fog_density() {
        let base = EnvironmentProfile::default();
        let overrides = EnvironmentOverrides {
            fog_density: Some(0.99),
            ..Default::default()
        };
        let result = apply_overrides(&base, &overrides);
        assert!(approx_eq(result.fog.density, 0.99));
        // Other fields unchanged.
        assert_eq!(result.fog.color, base.fog.color);
        assert!(approx_eq(result.ambient.intensity, base.ambient.intensity));
    }

    // 7. None overrides preserve base values.
    #[test]
    fn override_none_preserves() {
        let base = EnvironmentProfile::default();
        let overrides = EnvironmentOverrides::default(); // all None
        let result = apply_overrides(&base, &overrides);
        assert_eq!(result, base);
    }

    // 8. resolve with no target and no overrides returns base unchanged.
    #[test]
    fn resolve_no_target_no_overrides() {
        let base = EnvironmentProfile::default();
        let result = resolve_environment(&base, None, 0.7, &EnvironmentOverrides::default());
        assert_eq!(result, base);
    }

    // 9. resolve with blend target and overrides applies full chain.
    #[test]
    fn resolve_with_blend_and_overrides() {
        let a = EnvironmentProfile::default();
        let b = profile_b();
        let overrides = EnvironmentOverrides {
            ambient_intensity: Some(0.42),
            exposure_compensation: Some(2.0),
            ..Default::default()
        };
        let result = resolve_environment(&a, Some(&b), 0.5, &overrides);
        // Override wins over blend.
        assert!(approx_eq(result.ambient.intensity, 0.42));
        assert!(approx_eq(result.post_hints.exposure_compensation, 2.0));
        // Non-overridden value should be blended.
        let expected_bloom = (0.5 + 1.5) * 0.5;
        assert!(approx_eq(result.post_hints.bloom_intensity, expected_bloom));
    }

    // 10. save and load roundtrip via temp file.
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
}
