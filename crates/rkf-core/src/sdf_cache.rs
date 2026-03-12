//! Cached signed distance field for one 8×8×8 brick.
//!
//! [`SdfCache`] stores f16 distances at each voxel center. This is derived data —
//! computed from [`super::brick_geometry::BrickGeometry`] and cached for ray marching
//! and physics queries. It is never the source of truth for shape.

use half::f16;

use crate::constants::BRICK_DIM;

/// Cached signed distance field for one 8×8×8 brick.
///
/// 1024 bytes (512 × f16). Derived from geometry, not a source of truth.
/// Can always be recomputed from [`super::brick_geometry::BrickGeometry`].
#[derive(Clone)]
// Note: Debug intentionally not derived — [u16; 512] would produce excessive output.
// Use `SdfCache::get_distance()` for inspection.
pub struct SdfCache {
    /// f16 signed distance at each voxel center, stored as u16 bits.
    /// Layout: `x + y*8 + z*64` (z-major), matching brick_index.
    pub distances: [u16; 512],
}

impl Default for SdfCache {
    /// All distances set to +infinity (empty space).
    fn default() -> Self {
        Self {
            distances: [f16::INFINITY.to_bits(); 512],
        }
    }
}

impl std::fmt::Debug for SdfCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SdfCache")
            .field("distances", &format_args!("[f16; 512]"))
            .finish()
    }
}

impl SdfCache {
    /// Create a cache where all distances are +MAX (empty space).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a cache where all distances are -MAX (deep interior).
    pub fn interior() -> Self {
        Self {
            distances: [f16::NEG_INFINITY.to_bits(); 512],
        }
    }

    /// Flat index from 3D coordinates.
    #[inline]
    fn index(x: u8, y: u8, z: u8) -> usize {
        debug_assert!(x < 8 && y < 8 && z < 8);
        x as usize + y as usize * 8 + z as usize * 64
    }

    /// Get the signed distance at voxel (x, y, z) as f32.
    #[inline]
    pub fn get_distance(&self, x: u8, y: u8, z: u8) -> f32 {
        f16::from_bits(self.distances[Self::index(x, y, z)]).to_f32()
    }

    /// Set the signed distance at voxel (x, y, z) from f32.
    #[inline]
    pub fn set_distance(&mut self, x: u8, y: u8, z: u8, d: f32) {
        self.distances[Self::index(x, y, z)] = f16::from_f32(d).to_bits();
    }

    /// Trilinearly interpolate the SDF distance at a local position within the brick.
    ///
    /// `local_pos` is in brick-local coordinates where each axis ranges from
    /// 0.0 to 1.0. Voxel centers are at `(i + 0.5) / 8` for i in 0..8.
    pub fn sample_trilinear(&self, local_pos: glam::Vec3) -> f32 {
        let dim = BRICK_DIM as f32;

        let fx = (local_pos.x * dim - 0.5).clamp(0.0, dim - 1.0001);
        let fy = (local_pos.y * dim - 0.5).clamp(0.0, dim - 1.0001);
        let fz = (local_pos.z * dim - 0.5).clamp(0.0, dim - 1.0001);

        let ix = fx.floor() as u8;
        let iy = fy.floor() as u8;
        let iz = fz.floor() as u8;

        let ix1 = (ix + 1).min(7);
        let iy1 = (iy + 1).min(7);
        let iz1 = (iz + 1).min(7);

        let tx = fx - ix as f32;
        let ty = fy - iy as f32;
        let tz = fz - iz as f32;

        let c000 = self.get_distance(ix, iy, iz);
        let c100 = self.get_distance(ix1, iy, iz);
        let c010 = self.get_distance(ix, iy1, iz);
        let c110 = self.get_distance(ix1, iy1, iz);
        let c001 = self.get_distance(ix, iy, iz1);
        let c101 = self.get_distance(ix1, iy, iz1);
        let c011 = self.get_distance(ix, iy1, iz1);
        let c111 = self.get_distance(ix1, iy1, iz1);

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Return the raw f16 bits as a byte slice (1024 bytes) for GPU upload or file I/O.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.distances)
    }

    /// Create from raw bytes (1024 bytes of f16 distance data).
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 1024 {
            return None;
        }
        let mut distances = [0u16; 512];
        for (i, chunk) in data[..1024].chunks_exact(2).enumerate() {
            distances[i] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Some(Self { distances })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn default_is_positive_infinity() {
        let cache = SdfCache::default();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let d = cache.get_distance(x, y, z);
                    assert!(d.is_infinite() && d > 0.0, "expected +inf at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn interior_is_negative_infinity() {
        let cache = SdfCache::interior();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let d = cache.get_distance(x, y, z);
                    assert!(d.is_infinite() && d < 0.0, "expected -inf at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn set_get_roundtrip() {
        let mut cache = SdfCache::empty();
        cache.set_distance(3, 4, 5, -1.5);
        let d = cache.get_distance(3, 4, 5);
        assert!((d - (-1.5)).abs() < 0.01, "got {d}, expected -1.5");

        cache.set_distance(0, 0, 0, 0.25);
        assert!((cache.get_distance(0, 0, 0) - 0.25).abs() < 0.01);
    }

    #[test]
    fn trilinear_at_voxel_centers() {
        let mut cache = SdfCache::empty();
        // Set a gradient: distance = x
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    cache.set_distance(x, y, z, x as f32);
                }
            }
        }

        // At voxel center, should recover stored value
        for x in 0..8u8 {
            let pos = Vec3::new((x as f32 + 0.5) / 8.0, 0.5, 0.5);
            let sampled = cache.sample_trilinear(pos);
            assert!(
                (sampled - x as f32).abs() < 0.1,
                "at x={x}: sampled={sampled}, expected {x}"
            );
        }
    }

    #[test]
    fn trilinear_interpolation_midpoint() {
        let mut cache = SdfCache::empty();
        // Left half = 0, right half = 2
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    cache.set_distance(x, y, z, if x < 4 { 0.0 } else { 2.0 });
                }
            }
        }

        let mid_x = (3.5 + 0.5) / 8.0;
        let sampled = cache.sample_trilinear(Vec3::new(mid_x, 0.5, 0.5));
        assert!(
            (sampled - 1.0).abs() < 0.15,
            "midpoint: sampled={sampled}, expected ~1.0"
        );
    }

    #[test]
    fn trilinear_clamps_out_of_range() {
        let cache = SdfCache::empty();
        // Should not panic
        let _ = cache.sample_trilinear(Vec3::new(-0.5, 0.5, 0.5));
        let _ = cache.sample_trilinear(Vec3::new(1.5, 0.5, 0.5));
        let _ = cache.sample_trilinear(Vec3::new(0.5, -0.1, 1.5));
    }

    #[test]
    fn bytes_roundtrip() {
        let mut cache = SdfCache::empty();
        cache.set_distance(0, 0, 0, 1.5);
        cache.set_distance(7, 7, 7, -2.0);

        let bytes = cache.as_bytes();
        assert_eq!(bytes.len(), 1024);

        let cache2 = SdfCache::from_bytes(bytes).unwrap();
        assert!((cache2.get_distance(0, 0, 0) - 1.5).abs() < 0.01);
        assert!((cache2.get_distance(7, 7, 7) - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(SdfCache::from_bytes(&[0; 1023]).is_none());
    }

    #[test]
    fn uniform_trilinear_returns_constant() {
        let mut cache = SdfCache::empty();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    cache.set_distance(x, y, z, 3.5);
                }
            }
        }
        let positions = [
            Vec3::ZERO,
            Vec3::splat(0.5),
            Vec3::ONE,
            Vec3::new(0.25, 0.75, 0.1),
        ];
        for pos in &positions {
            let sampled = cache.sample_trilinear(*pos);
            assert!(
                (sampled - 3.5).abs() < 0.1,
                "uniform at {pos}: sampled={sampled}"
            );
        }
    }
}
