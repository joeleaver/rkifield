//! CPU reference implementation for trilinear sampling within a brick.
//!
//! [`sample_brick_trilinear`] interpolates the SDF distance field at an
//! arbitrary position within a brick's local coordinate space (0.0–1.0 per axis).
//!
//! [`sample_brick_nearest_material`] returns the material ID of the nearest
//! voxel center — no interpolation, since material IDs are discrete.

use glam::Vec3;

use crate::brick::Brick;
use crate::constants::BRICK_DIM;

/// Trilinearly interpolate the SDF distance at a local position within a brick.
///
/// `local_pos` is in brick-local coordinates where each axis ranges from
/// 0.0 (first voxel center at 0.5/BRICK_DIM) to 1.0. The position is clamped
/// to the valid sampling range.
///
/// Voxel centers are at `(i + 0.5) / BRICK_DIM` for i in 0..BRICK_DIM.
///
/// Returns the trilinearly interpolated f32 distance value.
pub fn sample_brick_trilinear(brick: &Brick, local_pos: Vec3) -> f32 {
    let dim = BRICK_DIM as f32;

    // Map local_pos (0..1) to voxel-center-relative coordinates.
    // Voxel centers are at (i + 0.5) / dim. We want fractional indices
    // such that the center of voxel i maps to exactly i.
    let fx = (local_pos.x * dim - 0.5).clamp(0.0, dim - 1.0001);
    let fy = (local_pos.y * dim - 0.5).clamp(0.0, dim - 1.0001);
    let fz = (local_pos.z * dim - 0.5).clamp(0.0, dim - 1.0001);

    let ix = fx.floor() as u32;
    let iy = fy.floor() as u32;
    let iz = fz.floor() as u32;

    // Next voxel, clamped to brick bounds
    let ix1 = (ix + 1).min(BRICK_DIM - 1);
    let iy1 = (iy + 1).min(BRICK_DIM - 1);
    let iz1 = (iz + 1).min(BRICK_DIM - 1);

    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let tz = fz - iz as f32;

    // Fetch 8 corner distances
    let c000 = brick.sample(ix, iy, iz).distance_f32();
    let c100 = brick.sample(ix1, iy, iz).distance_f32();
    let c010 = brick.sample(ix, iy1, iz).distance_f32();
    let c110 = brick.sample(ix1, iy1, iz).distance_f32();
    let c001 = brick.sample(ix, iy, iz1).distance_f32();
    let c101 = brick.sample(ix1, iy, iz1).distance_f32();
    let c011 = brick.sample(ix, iy1, iz1).distance_f32();
    let c111 = brick.sample(ix1, iy1, iz1).distance_f32();

    // Trilinear interpolation
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    c0 * (1.0 - tz) + c1 * tz
}

/// Return the material ID of the nearest voxel to `local_pos`.
///
/// `local_pos` is in brick-local coordinates (0.0–1.0 per axis).
/// Uses nearest-neighbor lookup — material IDs are discrete and cannot
/// be interpolated.
pub fn sample_brick_nearest_material(brick: &Brick, local_pos: Vec3) -> u16 {
    let dim = BRICK_DIM as f32;

    let ix = ((local_pos.x * dim - 0.5).round() as u32).min(BRICK_DIM - 1);
    let iy = ((local_pos.y * dim - 0.5).round() as u32).min(BRICK_DIM - 1);
    let iz = ((local_pos.z * dim - 0.5).round() as u32).min(BRICK_DIM - 1);

    brick.sample(ix, iy, iz).material_id()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::VoxelSample;

    /// Helper: create a brick filled with a known distance gradient.
    /// Distance at voxel (x,y,z) = x + y*0.1 + z*0.01 (unique per voxel).
    fn gradient_brick() -> Brick {
        let mut brick = Brick::default();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let dist = x as f32 + y as f32 * 0.1 + z as f32 * 0.01;
                    // Material id is 6-bit (0-63), VoxelSample::new masks to 0x3F.
                    let mat = ((x + y * 8 + z * 64) % 256) as u16;
                    brick.set(x, y, z, VoxelSample::new(dist, mat, [255, 255, 255, 255]));
                }
            }
        }
        brick
    }

    /// Helper: create a brick with uniform distance.
    fn uniform_brick(dist: f32, mat: u16) -> Brick {
        let mut brick = Brick::default();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    brick.set(x, y, z, VoxelSample::new(dist, mat, [255, 255, 255, 255]));
                }
            }
        }
        brick
    }

    /// Voxel center position in local coords for voxel index i.
    fn voxel_center(ix: u32, iy: u32, iz: u32) -> Vec3 {
        let dim = BRICK_DIM as f32;
        Vec3::new(
            (ix as f32 + 0.5) / dim,
            (iy as f32 + 0.5) / dim,
            (iz as f32 + 0.5) / dim,
        )
    }

    // ------ Trilinear sampling at voxel centers ------

    #[test]
    fn trilinear_at_voxel_centers_matches_stored_value() {
        let brick = gradient_brick();
        // Sample at each voxel center — should get back the stored distance
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let pos = voxel_center(x, y, z);
                    let sampled = sample_brick_trilinear(&brick, pos);
                    let stored = brick.sample(x, y, z).distance_f32();
                    assert!(
                        (sampled - stored).abs() < 0.02,
                        "mismatch at ({x},{y},{z}): sampled={sampled}, stored={stored}"
                    );
                }
            }
        }
    }

    #[test]
    fn trilinear_uniform_brick_returns_constant() {
        let brick = uniform_brick(3.5, 1);
        // Sampling anywhere should return ~3.5
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.25, 0.75, 0.1),
        ];
        for pos in &positions {
            let sampled = sample_brick_trilinear(&brick, *pos);
            assert!(
                (sampled - 3.5).abs() < 0.1,
                "uniform brick at {pos}: sampled={sampled}, expected ~3.5"
            );
        }
    }

    // ------ Interpolation smoothness ------

    #[test]
    fn trilinear_midpoint_interpolation() {
        // Two-value brick: left half = 0.0, right half = 2.0
        let mut brick = Brick::default();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let dist = if x < 4 { 0.0 } else { 2.0 };
                    brick.set(x, y, z, VoxelSample::new(dist, 1, [255, 255, 255, 255]));
                }
            }
        }

        // Sample at the midpoint between voxel 3 and 4 on x-axis
        let mid_x = (3.5 + 0.5) / BRICK_DIM as f32; // between centers of voxel 3 and 4
        let sampled = sample_brick_trilinear(&brick, Vec3::new(mid_x, 0.5, 0.5));
        // Should be approximately 1.0 (midpoint of 0.0 and 2.0)
        assert!(
            (sampled - 1.0).abs() < 0.15,
            "midpoint interpolation: sampled={sampled}, expected ~1.0"
        );
    }

    #[test]
    fn trilinear_is_smooth() {
        // Sample along a line and verify monotonicity for a monotonic input
        let mut brick = Brick::default();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    // Linearly increasing distance along x
                    let dist = x as f32;
                    brick.set(x, y, z, VoxelSample::new(dist, 1, [255, 255, 255, 255]));
                }
            }
        }

        let mut prev = sample_brick_trilinear(&brick, Vec3::new(0.0, 0.5, 0.5));
        for i in 1..=20 {
            let t = i as f32 / 20.0;
            let cur = sample_brick_trilinear(&brick, Vec3::new(t, 0.5, 0.5));
            assert!(
                cur >= prev - 0.01,
                "not monotonic at t={t}: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    // ------ Clamping at boundaries ------

    #[test]
    fn trilinear_clamps_at_boundaries() {
        let brick = gradient_brick();
        // Sample outside valid range — should clamp, not panic
        let _d1 = sample_brick_trilinear(&brick, Vec3::new(-0.5, 0.5, 0.5));
        let _d2 = sample_brick_trilinear(&brick, Vec3::new(1.5, 0.5, 0.5));
        let _d3 = sample_brick_trilinear(&brick, Vec3::new(0.5, -0.1, 1.5));
    }

    // ------ Nearest material ------

    #[test]
    fn nearest_material_at_voxel_centers() {
        let brick = gradient_brick();
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let pos = voxel_center(x, y, z);
                    let mat = sample_brick_nearest_material(&brick, pos);
                    // material_id is 6-bit (0-63), gradient_brick uses (flat_index % 256)
                    // which gets masked to 6 bits by VoxelSample::new
                    let expected = (((x + y * 8 + z * 64) % 256) & 0x3F) as u16;
                    assert_eq!(
                        mat, expected,
                        "material mismatch at ({x},{y},{z}): got {mat}, expected {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn nearest_material_snaps_to_closest() {
        let mut brick = Brick::default();
        // Set voxel (3,3,3) to material 10, neighbors to material 20
        for z in 0..BRICK_DIM {
            for y in 0..BRICK_DIM {
                for x in 0..BRICK_DIM {
                    let mat = if x == 3 && y == 3 && z == 3 { 10 } else { 20 };
                    brick.set(x, y, z, VoxelSample::new(0.0, mat, [255, 255, 255, 255]));
                }
            }
        }

        // At the center of voxel (3,3,3)
        let pos = voxel_center(3, 3, 3);
        assert_eq!(sample_brick_nearest_material(&brick, pos), 10);

        // Slightly offset — should still snap to (3,3,3) if closer
        let slightly_off = Vec3::new(
            (3.0 + 0.6) / BRICK_DIM as f32,
            (3.0 + 0.5) / BRICK_DIM as f32,
            (3.0 + 0.5) / BRICK_DIM as f32,
        );
        assert_eq!(sample_brick_nearest_material(&brick, slightly_off), 10);
    }

    #[test]
    fn nearest_material_boundary_clamp() {
        let brick = uniform_brick(0.0, 42);
        // Out of range should clamp, not panic
        let mat = sample_brick_nearest_material(&brick, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(mat, 42);
        let mat = sample_brick_nearest_material(&brick, Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(mat, 42);
    }

}
