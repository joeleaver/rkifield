//! Procedural cloud configuration and GPU parameters.
//!
//! High-altitude clouds are evaluated analytically using FBM noise,
//! weather map modulation, and height gradient shaping. Wind scrolling
//! provides animation over time.
//!
//! `CloudSettings` is the CPU-side configuration; `CloudParams` is the
//! 64-byte GPU-uploadable uniform buffer struct.

use bytemuck::{Pod, Zeroable};

// ── Default constants ────────────────────────────────────────────────────────

/// Lower cloud altitude in meters.
pub const DEFAULT_CLOUD_MIN: f32 = 1000.0;
/// Upper cloud altitude in meters.
pub const DEFAULT_CLOUD_MAX: f32 = 3000.0;
/// Coverage threshold — higher values mean fewer clouds.
pub const DEFAULT_CLOUD_THRESHOLD: f32 = 0.4;
/// Global density scale multiplier.
pub const DEFAULT_CLOUD_DENSITY_SCALE: f32 = 1.0;
/// Shape FBM frequency (world-space).
pub const DEFAULT_CLOUD_SHAPE_FREQ: f32 = 0.0003;
/// Detail FBM frequency (world-space).
pub const DEFAULT_CLOUD_DETAIL_FREQ: f32 = 0.002;
/// Weight of subtractive detail noise.
pub const DEFAULT_CLOUD_DETAIL_WEIGHT: f32 = 0.3;
/// World-space size of the weather map tile.
pub const DEFAULT_CLOUD_WEATHER_SCALE: f32 = 10_000.0;
/// Default wind speed in metres per second.
pub const DEFAULT_CLOUD_WIND_SPEED: f32 = 5.0;
/// Default cloud shadow map resolution (square).
pub const DEFAULT_CLOUD_SHADOW_RESOLUTION: u32 = 1024;
/// World-space extent of the cloud shadow map in metres.
pub const DEFAULT_CLOUD_SHADOW_COVERAGE: f32 = 4000.0;

// ── CPU-side configuration ───────────────────────────────────────────────────

/// CPU-side cloud configuration.
///
/// Drives both procedural (analytically evaluated) and brick-backed cloud
/// volumes. Pass to [`CloudParams::from_settings`] to obtain the GPU struct.
#[derive(Debug, Clone)]
pub struct CloudSettings {
    /// Enable analytic procedural cloud evaluation.
    pub procedural_enabled: bool,
    /// Lower altitude of the cloud layer in metres.
    pub cloud_min: f32,
    /// Upper altitude of the cloud layer in metres.
    pub cloud_max: f32,
    /// Coverage threshold: fraction of density below which clouds are cut off.
    pub cloud_threshold: f32,
    /// Global density scale applied after thresholding.
    pub cloud_density_scale: f32,
    /// World-space frequency of the shape FBM noise.
    pub shape_frequency: f32,
    /// World-space frequency of the detail FBM noise.
    pub detail_frequency: f32,
    /// Blend weight for subtractive detail noise.
    pub detail_weight: f32,
    /// World-space size of the tiling weather map.
    pub weather_scale: f32,
    /// Horizontal wind direction (XZ plane), not required to be normalised.
    pub wind_direction: [f32; 2],
    /// Wind speed in metres per second.
    pub wind_speed: f32,
    /// Enable volumetric brick-backed cloud rendering (Phase 12+).
    pub brick_clouds_enabled: bool,
    /// Enable cloud shadow map generation.
    pub shadow_enabled: bool,
    /// Resolution of the cloud shadow map texture (square).
    pub shadow_map_resolution: u32,
    /// World-space extent covered by the cloud shadow map.
    pub shadow_coverage: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            procedural_enabled: false,
            cloud_min: DEFAULT_CLOUD_MIN,
            cloud_max: DEFAULT_CLOUD_MAX,
            cloud_threshold: DEFAULT_CLOUD_THRESHOLD,
            cloud_density_scale: DEFAULT_CLOUD_DENSITY_SCALE,
            shape_frequency: DEFAULT_CLOUD_SHAPE_FREQ,
            detail_frequency: DEFAULT_CLOUD_DETAIL_FREQ,
            detail_weight: DEFAULT_CLOUD_DETAIL_WEIGHT,
            weather_scale: DEFAULT_CLOUD_WEATHER_SCALE,
            wind_direction: [1.0, 0.0],
            wind_speed: DEFAULT_CLOUD_WIND_SPEED,
            brick_clouds_enabled: false,
            shadow_enabled: false,
            shadow_map_resolution: DEFAULT_CLOUD_SHADOW_RESOLUTION,
            shadow_coverage: DEFAULT_CLOUD_SHADOW_COVERAGE,
        }
    }
}

// ── GPU parameters ───────────────────────────────────────────────────────────

/// GPU-uploadable cloud parameters (64 bytes).
///
/// Matches the `CloudParams` struct in `clouds.wgsl`. Upload as a uniform
/// buffer or as part of a larger scene uniform.
///
/// Field layout:
/// - bytes  0–15: `altitude`  — `[cloud_min, cloud_max, threshold, density_scale]`
/// - bytes 16–31: `noise`     — `[shape_freq, detail_freq, detail_weight, weather_scale]`
/// - bytes 32–47: `wind`      — `[wind_dir.x, wind_dir.y, wind_speed, time]`
/// - bytes 48–63: `flags`     — `[procedural_enable, shadow_coverage, shadow_res, brick_clouds_enable]`
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CloudParams {
    /// `x = cloud_min`, `y = cloud_max`, `z = threshold`, `w = density_scale`.
    pub altitude: [f32; 4],
    /// `x = shape_freq`, `y = detail_freq`, `z = detail_weight`, `w = weather_scale`.
    pub noise: [f32; 4],
    /// `x = wind_dir.x`, `y = wind_dir.y`, `z = wind_speed`, `w = time` (set per frame).
    pub wind: [f32; 4],
    /// `x = procedural_enable (0/1)`, `y = shadow_coverage`, `z = shadow_resolution`, `w = brick_clouds_enable (0/1)`.
    pub flags: [f32; 4],
}

impl CloudParams {
    /// Build GPU parameters from CPU [`CloudSettings`] and the current time.
    ///
    /// `time` is accumulated engine time in seconds; it drives wind scrolling.
    pub fn from_settings(settings: &CloudSettings, time: f32) -> Self {
        Self {
            altitude: [
                settings.cloud_min,
                settings.cloud_max,
                settings.cloud_threshold,
                settings.cloud_density_scale,
            ],
            noise: [
                settings.shape_frequency,
                settings.detail_frequency,
                settings.detail_weight,
                settings.weather_scale,
            ],
            wind: [
                settings.wind_direction[0],
                settings.wind_direction[1],
                settings.wind_speed,
                time,
            ],
            flags: [
                if settings.procedural_enabled { 1.0 } else { 0.0 },
                settings.shadow_coverage,
                settings.shadow_map_resolution as f32,
                if settings.brick_clouds_enabled { 1.0 } else { 0.0 },
            ],
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_params_size_is_64() {
        assert_eq!(std::mem::size_of::<CloudParams>(), 64);
    }

    #[test]
    fn cloud_params_pod_roundtrip() {
        let s = CloudSettings::default();
        let p = CloudParams::from_settings(&s, 0.0);
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 64);
        let p2: &CloudParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.altitude[0], p2.altitude[0]); // cloud_min
        assert_eq!(p.noise[0], p2.noise[0]); // shape_freq
    }

    #[test]
    fn cloud_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(CloudParams, altitude), 0);
        assert_eq!(std::mem::offset_of!(CloudParams, noise), 16);
        assert_eq!(std::mem::offset_of!(CloudParams, wind), 32);
        assert_eq!(std::mem::offset_of!(CloudParams, flags), 48);
    }

    #[test]
    fn cloud_settings_default() {
        let s = CloudSettings::default();
        assert!(!s.procedural_enabled);
        assert!(!s.brick_clouds_enabled);
        assert!(!s.shadow_enabled);
        assert_eq!(s.cloud_min, DEFAULT_CLOUD_MIN);
        assert_eq!(s.cloud_max, DEFAULT_CLOUD_MAX);
        assert!(s.cloud_max > s.cloud_min);
    }

    #[test]
    fn cloud_params_enable_flag() {
        let mut s = CloudSettings::default();
        let p = CloudParams::from_settings(&s, 0.0);
        assert_eq!(p.flags[0], 0.0); // disabled

        s.procedural_enabled = true;
        let p = CloudParams::from_settings(&s, 0.0);
        assert_eq!(p.flags[0], 1.0); // enabled
    }

    #[test]
    fn cloud_params_time_propagates() {
        let s = CloudSettings::default();
        let p = CloudParams::from_settings(&s, 42.5);
        assert_eq!(p.wind[3], 42.5);
    }

    #[test]
    fn cloud_params_wind_direction() {
        let mut s = CloudSettings::default();
        s.wind_direction = [0.707, 0.707];
        s.wind_speed = 10.0;
        let p = CloudParams::from_settings(&s, 1.0);
        assert_eq!(p.wind[0], 0.707);
        assert_eq!(p.wind[1], 0.707);
        assert_eq!(p.wind[2], 10.0);
    }

    #[test]
    fn altitude_range_constants() {
        assert!(DEFAULT_CLOUD_MIN > 0.0);
        assert!(DEFAULT_CLOUD_MAX > DEFAULT_CLOUD_MIN);
        assert!(DEFAULT_CLOUD_THRESHOLD > 0.0);
        assert!(DEFAULT_CLOUD_THRESHOLD < 1.0);
    }
}
