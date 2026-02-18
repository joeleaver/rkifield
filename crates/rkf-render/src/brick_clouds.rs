//! Brick-backed clouds (low-altitude) configuration.
//!
//! Low-altitude clouds stored in the volumetric companion brick pool for
//! terrain-interacting effects: fog banks, valley mist, mountain wrapping.
//! Brick-backed clouds use the same density/emission format as local fog
//! volumes and are sampled identically in the volumetric march shader.
//!
//! These clouds complement the high-altitude procedural clouds (see
//! [`crate::clouds`]). Both feed the same volumetric march.

use crate::fog_volume::GpuFogVolume;

/// Default low-cloud altitude band minimum (meters).
pub const DEFAULT_BRICK_CLOUD_MIN: f32 = 0.0;

/// Default low-cloud altitude band maximum (meters).
pub const DEFAULT_BRICK_CLOUD_MAX: f32 = 200.0;

/// Default brick cloud density scale.
pub const DEFAULT_BRICK_CLOUD_DENSITY: f32 = 0.5;

/// Default phase function asymmetry for low clouds.
pub const DEFAULT_BRICK_CLOUD_G: f32 = 0.5;

/// Default scattering color for low clouds (white-ish).
pub const DEFAULT_BRICK_CLOUD_COLOR: [f32; 3] = [0.95, 0.95, 0.97];

/// Brick cloud region type.
///
/// Describes a low-altitude cloud region that will be backed by volumetric
/// bricks. Each region maps to a set of fog volume bricks in the companion pool.
#[derive(Debug, Clone)]
pub enum BrickCloudType {
    /// Static fog bank (e.g., cave fog, swamp mist).
    FogBank,
    /// Valley mist that pools in low terrain.
    ValleyMist,
    /// Mountain-wrapping cloud that flows around peaks.
    MountainWrap,
    /// Custom configuration with user-specified parameters.
    Custom,
}

/// CPU-side brick cloud configuration.
///
/// Each `BrickCloudRegion` defines a volume that will be populated with
/// volumetric bricks. The actual brick data is managed by the brick pool
/// system — this struct describes the region and its visual properties.
#[derive(Debug, Clone)]
pub struct BrickCloudRegion {
    /// Type of cloud region (affects procedural generation).
    pub cloud_type: BrickCloudType,
    /// AABB minimum corner (world space).
    pub aabb_min: [f32; 3],
    /// AABB maximum corner (world space).
    pub aabb_max: [f32; 3],
    /// Density scale multiplier.
    pub density_scale: f32,
    /// Scattering color (linear RGB).
    pub color: [f32; 3],
    /// Phase function asymmetry (Henyey-Greenstein g parameter).
    pub phase_g: f32,
    /// Edge falloff distance (meters) — soft edges.
    pub edge_falloff: f32,
    /// Whether this region is active.
    pub active: bool,
}

impl Default for BrickCloudRegion {
    fn default() -> Self {
        Self {
            cloud_type: BrickCloudType::FogBank,
            aabb_min: [0.0, DEFAULT_BRICK_CLOUD_MIN, 0.0],
            aabb_max: [100.0, DEFAULT_BRICK_CLOUD_MAX, 100.0],
            density_scale: DEFAULT_BRICK_CLOUD_DENSITY,
            color: DEFAULT_BRICK_CLOUD_COLOR,
            phase_g: DEFAULT_BRICK_CLOUD_G,
            edge_falloff: 5.0,
            active: true,
        }
    }
}

impl BrickCloudRegion {
    /// Create a fog bank cloud region.
    pub fn fog_bank(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            cloud_type: BrickCloudType::FogBank,
            aabb_min,
            aabb_max,
            density_scale: 0.3,
            color: [0.9, 0.92, 0.95],
            phase_g: 0.3,
            edge_falloff: 3.0,
            active: true,
        }
    }

    /// Create a valley mist cloud region.
    pub fn valley_mist(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            cloud_type: BrickCloudType::ValleyMist,
            aabb_min,
            aabb_max,
            density_scale: 0.15,
            color: [0.85, 0.88, 0.92],
            phase_g: 0.2,
            edge_falloff: 10.0,
            active: true,
        }
    }

    /// Create a mountain-wrapping cloud region.
    pub fn mountain_wrap(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> Self {
        Self {
            cloud_type: BrickCloudType::MountainWrap,
            aabb_min,
            aabb_max,
            density_scale: 0.6,
            color: DEFAULT_BRICK_CLOUD_COLOR,
            phase_g: 0.6,
            edge_falloff: 8.0,
            active: true,
        }
    }

    /// Convert this brick cloud region to a GpuFogVolume for GPU upload.
    ///
    /// Brick-backed clouds use the same GPU representation as local fog volumes.
    pub fn to_gpu_fog_volume(&self) -> GpuFogVolume {
        GpuFogVolume::new(
            self.aabb_min,
            self.aabb_max,
            self.density_scale,
            self.color,
            self.phase_g,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_region() {
        let r = BrickCloudRegion::default();
        assert!(r.active);
        assert_eq!(r.density_scale, DEFAULT_BRICK_CLOUD_DENSITY);
        assert_eq!(r.color, DEFAULT_BRICK_CLOUD_COLOR);
        assert_eq!(r.phase_g, DEFAULT_BRICK_CLOUD_G);
    }

    #[test]
    fn fog_bank_preset() {
        let r = BrickCloudRegion::fog_bank([0.0, 0.0, 0.0], [50.0, 10.0, 50.0]);
        assert!(r.active);
        assert!(r.density_scale > 0.0);
        assert!(r.density_scale < 1.0);
        assert!(matches!(r.cloud_type, BrickCloudType::FogBank));
    }

    #[test]
    fn valley_mist_preset() {
        let r = BrickCloudRegion::valley_mist([-100.0, -5.0, -100.0], [100.0, 20.0, 100.0]);
        assert!(r.active);
        assert!(r.density_scale < BrickCloudRegion::fog_bank([0.0; 3], [1.0; 3]).density_scale);
        assert!(matches!(r.cloud_type, BrickCloudType::ValleyMist));
    }

    #[test]
    fn mountain_wrap_preset() {
        let r = BrickCloudRegion::mountain_wrap([0.0, 50.0, 0.0], [200.0, 150.0, 200.0]);
        assert!(r.active);
        assert!(r.density_scale > BrickCloudRegion::fog_bank([0.0; 3], [1.0; 3]).density_scale);
        assert!(matches!(r.cloud_type, BrickCloudType::MountainWrap));
    }

    #[test]
    fn to_gpu_fog_volume() {
        let r = BrickCloudRegion::fog_bank([-10.0, 0.0, -10.0], [10.0, 5.0, 10.0]);
        let gpu = r.to_gpu_fog_volume();
        // aabb_min.w = density_scale
        assert_eq!(gpu.aabb_min[3], r.density_scale);
        // color.w = phase_g
        assert_eq!(gpu.color[3], r.phase_g);
        // AABB corners preserved
        assert_eq!(gpu.aabb_min[0], -10.0);
        assert_eq!(gpu.aabb_max[1], 5.0);
    }

    #[test]
    fn altitude_defaults() {
        assert_eq!(DEFAULT_BRICK_CLOUD_MIN, 0.0);
        assert!(DEFAULT_BRICK_CLOUD_MAX > DEFAULT_BRICK_CLOUD_MIN);
    }

    #[test]
    fn gpu_volume_size_matches() {
        // Brick clouds reuse GpuFogVolume, which is 64 bytes
        assert_eq!(std::mem::size_of::<crate::fog_volume::GpuFogVolume>(), 64);
    }
}
