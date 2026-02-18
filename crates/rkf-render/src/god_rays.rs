//! God ray configuration and parameter tuning.
//!
//! God rays are an emergent effect from the volumetric pipeline. They appear
//! when participating media (ambient dust) is present and sunlight is
//! partially occluded by geometry. The volumetric shadow map creates visible
//! light shafts automatically.
//!
//! This module provides recommended constants and a configuration struct for
//! tuning god ray appearance.

/// Minimum dust density for visible god rays.
pub const MIN_GOD_RAY_DUST: f32 = 0.001;

/// Maximum recommended dust density before fog dominates.
pub const MAX_GOD_RAY_DUST: f32 = 0.01;

/// Default dust density for subtle god rays.
pub const DEFAULT_GOD_RAY_DUST: f32 = 0.003;

/// Default phase function asymmetry for god rays (strong forward scatter).
/// Higher g = brighter forward glow when looking toward the sun.
pub const DEFAULT_GOD_RAY_G: f32 = 0.6;

/// Minimum recommended phase g for visible light shafts.
pub const MIN_GOD_RAY_G: f32 = 0.3;

/// Maximum recommended phase g (very strong forward scatter).
pub const MAX_GOD_RAY_G: f32 = 0.85;

/// God ray configuration.
///
/// These settings control the appearance of volumetric light shafts.
/// God rays emerge from the existing volumetric system — no separate
/// pass is needed. Adjusting these values modifies the ambient dust
/// and phase function parameters in the volumetric march.
#[derive(Debug, Clone)]
pub struct GodRaySettings {
    /// Enable god rays (sets ambient dust density > 0).
    pub enabled: bool,
    /// Dust density (0.001–0.01 range). Higher = more visible rays but also more haze.
    pub dust_density: f32,
    /// Phase function asymmetry (0.3–0.85). Higher = brighter forward glow toward sun.
    pub phase_g: f32,
    /// Sun intensity multiplier for god ray contribution.
    pub intensity: f32,
}

impl Default for GodRaySettings {
    fn default() -> Self {
        Self {
            enabled: false,
            dust_density: DEFAULT_GOD_RAY_DUST,
            phase_g: DEFAULT_GOD_RAY_G,
            intensity: 1.0,
        }
    }
}

impl GodRaySettings {
    /// Create god ray settings with default subtle appearance.
    pub fn subtle() -> Self {
        Self {
            enabled: true,
            dust_density: 0.002,
            phase_g: 0.5,
            intensity: 1.0,
        }
    }

    /// Create god ray settings with strong, dramatic shafts.
    pub fn dramatic() -> Self {
        Self {
            enabled: true,
            dust_density: 0.008,
            phase_g: 0.75,
            intensity: 1.5,
        }
    }

    /// Apply these settings to a FogSettings struct.
    ///
    /// Updates the ambient dust density and phase asymmetry fields.
    pub fn apply_to_fog(&self, fog: &mut crate::fog::FogSettings) {
        if self.enabled {
            fog.ambient_dust_density = self.dust_density;
            fog.ambient_dust_g = self.phase_g;
        } else {
            fog.ambient_dust_density = 0.0;
        }
    }

    /// Clamp parameters to recommended ranges.
    pub fn clamped(&self) -> Self {
        Self {
            enabled: self.enabled,
            dust_density: self.dust_density.clamp(MIN_GOD_RAY_DUST, MAX_GOD_RAY_DUST),
            phase_g: self.phase_g.clamp(MIN_GOD_RAY_G, MAX_GOD_RAY_G),
            intensity: self.intensity.max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fog::FogSettings;

    #[test]
    fn default_disabled() {
        let s = GodRaySettings::default();
        assert!(!s.enabled);
        assert_eq!(s.dust_density, DEFAULT_GOD_RAY_DUST);
        assert_eq!(s.phase_g, DEFAULT_GOD_RAY_G);
    }

    #[test]
    fn subtle_preset() {
        let s = GodRaySettings::subtle();
        assert!(s.enabled);
        assert!(s.dust_density >= MIN_GOD_RAY_DUST);
        assert!(s.dust_density <= MAX_GOD_RAY_DUST);
        assert!(s.phase_g >= MIN_GOD_RAY_G);
    }

    #[test]
    fn dramatic_preset() {
        let s = GodRaySettings::dramatic();
        assert!(s.enabled);
        assert!(s.dust_density > GodRaySettings::subtle().dust_density);
        assert!(s.phase_g > GodRaySettings::subtle().phase_g);
    }

    #[test]
    fn apply_to_fog_enabled() {
        let gr = GodRaySettings {
            enabled: true,
            dust_density: 0.005,
            phase_g: 0.6,
            intensity: 1.0,
        };
        let mut fog = FogSettings::default();
        assert_eq!(fog.ambient_dust_density, 0.0);
        gr.apply_to_fog(&mut fog);
        assert_eq!(fog.ambient_dust_density, 0.005);
        assert_eq!(fog.ambient_dust_g, 0.6);
    }

    #[test]
    fn apply_to_fog_disabled() {
        let gr = GodRaySettings::default(); // disabled
        let mut fog = FogSettings::default();
        fog.ambient_dust_density = 0.01; // was set
        gr.apply_to_fog(&mut fog);
        assert_eq!(fog.ambient_dust_density, 0.0); // cleared
    }

    #[test]
    fn clamp_to_range() {
        let s = GodRaySettings {
            enabled: true,
            dust_density: 0.1, // way too high
            phase_g: 0.0,      // too low
            intensity: -1.0,   // negative
        };
        let c = s.clamped();
        assert_eq!(c.dust_density, MAX_GOD_RAY_DUST);
        assert_eq!(c.phase_g, MIN_GOD_RAY_G);
        assert_eq!(c.intensity, 0.0);
    }

    #[test]
    fn dust_density_range_valid() {
        assert!(MIN_GOD_RAY_DUST > 0.0);
        assert!(MAX_GOD_RAY_DUST > MIN_GOD_RAY_DUST);
        assert!(DEFAULT_GOD_RAY_DUST >= MIN_GOD_RAY_DUST);
        assert!(DEFAULT_GOD_RAY_DUST <= MAX_GOD_RAY_DUST);
    }

    #[test]
    fn phase_g_range_valid() {
        assert!(MIN_GOD_RAY_G > 0.0);
        assert!(MAX_GOD_RAY_G < 1.0); // g must be < 1 for HG
        assert!(DEFAULT_GOD_RAY_G >= MIN_GOD_RAY_G);
        assert!(DEFAULT_GOD_RAY_G <= MAX_GOD_RAY_G);
    }
}
