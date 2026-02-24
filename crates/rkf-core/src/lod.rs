//! Per-object LOD (Level of Detail) selection.
//!
//! Each voxelized object can have multiple LOD levels with decreasing voxel
//! resolution. [`ObjectLod`] describes the available levels, and
//! [`select_lod`] picks the appropriate one based on screen-space coverage.
//!
//! # Strategy
//!
//! Screen-space driven: pick the LOD where one voxel projects to approximately
//! one pixel. When the object is far enough that even the coarsest voxel level
//! covers less than one pixel, fall back to the analytical bound (zero-cost
//! rendering via the primitive SDF, no bricks needed).

use crate::aabb::Aabb;
use crate::scene_node::SdfPrimitive;

/// A single LOD level for a voxelized object.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// Voxel size in object-local meters for this level.
    pub voxel_size: f32,
    /// Number of bricks at this level.
    pub brick_count: u32,
}

/// LOD configuration for a voxelized scene object.
#[derive(Debug, Clone)]
pub struct ObjectLod {
    /// Available LOD levels, sorted from finest (index 0) to coarsest.
    pub levels: Vec<LodLevel>,
    /// Analytical SDF bound used when no voxelized level is suitable
    /// (object too far away). `None` means the object simply disappears.
    pub analytical_bound: Option<SdfPrimitive>,
    /// Importance bias multiplier (default 1.0). Higher values cause the
    /// object to stay at finer LOD for longer. Useful for hero objects.
    pub importance_bias: f32,
}

impl ObjectLod {
    /// Create an LOD config with a single level (no LOD switching).
    pub fn single(voxel_size: f32, brick_count: u32) -> Self {
        Self {
            levels: vec![LodLevel {
                voxel_size,
                brick_count,
            }],
            analytical_bound: None,
            importance_bias: 1.0,
        }
    }

    /// Create an LOD config with multiple levels.
    ///
    /// Levels should be provided finest-first. If not sorted, they will be
    /// sorted by voxel_size ascending.
    pub fn multi(mut levels: Vec<LodLevel>, analytical_bound: Option<SdfPrimitive>) -> Self {
        levels.sort_by(|a, b| a.voxel_size.partial_cmp(&b.voxel_size).unwrap());
        Self {
            levels,
            analytical_bound,
            importance_bias: 1.0,
        }
    }
}

/// The result of LOD selection for a single object.
#[derive(Debug, Clone, PartialEq)]
pub enum LodSelection {
    /// Use the voxelized level at this index (into `ObjectLod::levels`).
    Voxelized(usize),
    /// Object is far enough to use the analytical SDF bound instead of bricks.
    Analytical,
    /// Object should not be rendered (too far, no analytical fallback).
    Hidden,
}

/// Select the appropriate LOD level for an object.
///
/// Uses screen-space projection: picks the coarsest level where one voxel
/// still projects to at most `max_voxel_pixels` screen pixels.
///
/// # Parameters
/// - `lod` — The object's LOD configuration
/// - `object_aabb` — The object's world-space AABB (for size estimation)
/// - `camera_distance` — Distance from camera to object center
/// - `viewport_height` — Viewport height in pixels
/// - `fov_y` — Vertical field of view in radians
/// - `max_voxel_pixels` — Maximum screen pixels per voxel (typically 1.0-2.0)
pub fn select_lod(
    lod: &ObjectLod,
    object_aabb: &Aabb,
    camera_distance: f32,
    viewport_height: f32,
    fov_y: f32,
    max_voxel_pixels: f32,
) -> LodSelection {
    if lod.levels.is_empty() {
        return if lod.analytical_bound.is_some() {
            LodSelection::Analytical
        } else {
            LodSelection::Hidden
        };
    }

    // Apply importance bias: a bias of 2.0 halves the effective distance,
    // keeping finer LOD longer.
    let effective_distance = camera_distance / lod.importance_bias.max(0.01);

    // Screen-space size of one world-unit at the given distance.
    // pixel_per_meter = viewport_height / (2 * distance * tan(fov_y/2))
    let half_tan = (fov_y * 0.5).tan();
    let pixel_per_meter = if effective_distance > 0.001 {
        viewport_height / (2.0 * effective_distance * half_tan)
    } else {
        // Extremely close — use finest LOD
        return LodSelection::Voxelized(0);
    };

    // Find the coarsest level where voxel screen size <= max_voxel_pixels.
    // Iterate from coarsest to finest, pick the first that's fine enough.
    let _ = object_aabb; // AABB reserved for future use (screen coverage)
    for i in (0..lod.levels.len()).rev() {
        let voxel_pixels = lod.levels[i].voxel_size * pixel_per_meter;
        if voxel_pixels <= max_voxel_pixels {
            return LodSelection::Voxelized(i);
        }
    }

    // Even the finest level is too coarse at this distance — shouldn't
    // normally happen (means object is very close). Use finest.
    // But wait — we iterate coarsest-to-finest checking <= threshold.
    // If we fall through, ALL levels have voxel_pixels > max. That means
    // the finest voxel projects larger than max_voxel_pixels, which only
    // happens very close to the object. Use finest level.
    LodSelection::Voxelized(0)
}

/// Determine if an object is far enough to use analytical fallback.
///
/// Returns true if the object's AABB projects to fewer than `min_pixels`
/// on screen.
pub fn should_use_analytical(
    object_aabb: &Aabb,
    camera_distance: f32,
    viewport_height: f32,
    fov_y: f32,
    min_pixels: f32,
) -> bool {
    let half_tan = (fov_y * 0.5).tan();
    if camera_distance < 0.001 {
        return false;
    }
    let pixel_per_meter = viewport_height / (2.0 * camera_distance * half_tan);
    let object_size = object_aabb.size().max_element();
    let screen_pixels = object_size * pixel_per_meter;
    screen_pixels < min_pixels
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use std::f32::consts::FRAC_PI_4;

    fn test_aabb() -> Aabb {
        Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0))
    }

    fn test_lod() -> ObjectLod {
        ObjectLod::multi(
            vec![
                LodLevel {
                    voxel_size: 0.005,
                    brick_count: 1000,
                }, // finest
                LodLevel {
                    voxel_size: 0.02,
                    brick_count: 100,
                }, // medium
                LodLevel {
                    voxel_size: 0.08,
                    brick_count: 20,
                }, // coarsest
            ],
            Some(SdfPrimitive::Sphere { radius: 1.0 }),
        )
    }

    #[test]
    fn single_level() {
        let lod = ObjectLod::single(0.02, 50);
        assert_eq!(lod.levels.len(), 1);
        assert_eq!(lod.levels[0].voxel_size, 0.02);
        assert!(lod.analytical_bound.is_none());
    }

    #[test]
    fn multi_sorts_by_voxel_size() {
        let lod = ObjectLod::multi(
            vec![
                LodLevel {
                    voxel_size: 0.08,
                    brick_count: 20,
                },
                LodLevel {
                    voxel_size: 0.005,
                    brick_count: 1000,
                },
                LodLevel {
                    voxel_size: 0.02,
                    brick_count: 100,
                },
            ],
            None,
        );
        assert_eq!(lod.levels[0].voxel_size, 0.005);
        assert_eq!(lod.levels[1].voxel_size, 0.02);
        assert_eq!(lod.levels[2].voxel_size, 0.08);
    }

    #[test]
    fn select_lod_close_uses_finest() {
        let lod = test_lod();
        let result = select_lod(&lod, &test_aabb(), 0.5, 1080.0, FRAC_PI_4, 1.0);
        // Very close — finest level
        assert_eq!(result, LodSelection::Voxelized(0));
    }

    #[test]
    fn select_lod_medium_distance() {
        let lod = test_lod();
        // At medium distance, medium LOD should be selected
        // pixel_per_meter = 1080 / (2 * 10.0 * tan(PI/8)) ≈ 1080 / (2 * 10 * 0.414) ≈ 130
        // finest: 0.005 * 130 = 0.65 px ✓
        // medium: 0.02 * 130 = 2.6 px ✗
        // coarsest: 0.08 * 130 = 10.4 px ✗
        // → coarsest that's ≤1.0 is finest (index 0)
        let result = select_lod(&lod, &test_aabb(), 10.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Voxelized(0));
    }

    #[test]
    fn select_lod_far_uses_coarser() {
        let lod = test_lod();
        // At far distance:
        // pixel_per_meter = 1080 / (2 * 100.0 * 0.414) ≈ 13.0
        // finest: 0.005 * 13 = 0.065 px
        // medium: 0.02 * 13 = 0.26 px
        // coarsest: 0.08 * 13 = 1.04 px  ← barely over 1.0
        // → medium (index 1) is the coarsest that's ≤ 1.0
        let result = select_lod(&lod, &test_aabb(), 100.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Voxelized(1));
    }

    #[test]
    fn select_lod_very_far_uses_coarsest() {
        let lod = test_lod();
        // pixel_per_meter = 1080 / (2 * 500 * 0.414) ≈ 2.61
        // finest: 0.005 * 2.61 = 0.013 px
        // medium: 0.02 * 2.61 = 0.052 px
        // coarsest: 0.08 * 2.61 = 0.209 px
        // → all under 1.0, so coarsest (index 2)
        let result = select_lod(&lod, &test_aabb(), 500.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Voxelized(2));
    }

    #[test]
    fn select_lod_empty_levels_analytical() {
        let lod = ObjectLod {
            levels: vec![],
            analytical_bound: Some(SdfPrimitive::Sphere { radius: 1.0 }),
            importance_bias: 1.0,
        };
        let result = select_lod(&lod, &test_aabb(), 10.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Analytical);
    }

    #[test]
    fn select_lod_empty_levels_no_analytical() {
        let lod = ObjectLod {
            levels: vec![],
            analytical_bound: None,
            importance_bias: 1.0,
        };
        let result = select_lod(&lod, &test_aabb(), 10.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Hidden);
    }

    #[test]
    fn select_lod_importance_bias() {
        let mut lod = test_lod();
        // Without bias at distance 100: picks medium (index 1)
        let r1 = select_lod(&lod, &test_aabb(), 100.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(r1, LodSelection::Voxelized(1));

        // With high bias (4.0): effective distance quartered → finer LOD
        lod.importance_bias = 4.0;
        let r2 = select_lod(&lod, &test_aabb(), 100.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(r2, LodSelection::Voxelized(0));
    }

    #[test]
    fn select_lod_zero_distance() {
        let lod = test_lod();
        let result = select_lod(&lod, &test_aabb(), 0.0, 1080.0, FRAC_PI_4, 1.0);
        assert_eq!(result, LodSelection::Voxelized(0));
    }

    #[test]
    fn should_use_analytical_close() {
        assert!(!should_use_analytical(
            &test_aabb(),
            5.0,
            1080.0,
            FRAC_PI_4,
            4.0
        ));
    }

    #[test]
    fn should_use_analytical_far() {
        // At 10000m, a 2m object projects very small
        assert!(should_use_analytical(
            &test_aabb(),
            10000.0,
            1080.0,
            FRAC_PI_4,
            4.0
        ));
    }
}
