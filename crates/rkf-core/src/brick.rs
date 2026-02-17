//! The primary [`Brick`] type for the RKIField SDF engine.
//!
//! A [`Brick`] is an 8×8×8 grid of [`VoxelSample`]s — 512 voxels totalling 4 096 bytes.
//! Bricks are the fundamental unit of voxel storage in the brick pool.  All geometry,
//! materials, and SDF distances live in bricks.
//!
//! Use [`brick_index`] to convert 3D coordinates to a flat array index, then
//! [`Brick::sample`] / [`Brick::set`] for typed access.

use crate::constants::BRICK_DIM;
use crate::voxel::VoxelSample;
use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Index helper
// ---------------------------------------------------------------------------

/// Compute a flat voxel index from 3D coordinates within a brick.
///
/// Layout: `x + y * 8 + z * 64` (z-major order) — matches companion brick memory layout.
///
/// # Panics
///
/// Panics in debug builds if any coordinate is >= [`BRICK_DIM`] (8).
#[inline]
pub fn brick_index(x: u32, y: u32, z: u32) -> usize {
    debug_assert!(
        x < BRICK_DIM,
        "x={x} out of brick bounds (BRICK_DIM={BRICK_DIM})"
    );
    debug_assert!(
        y < BRICK_DIM,
        "y={y} out of brick bounds (BRICK_DIM={BRICK_DIM})"
    );
    debug_assert!(
        z < BRICK_DIM,
        "z={z} out of brick bounds (BRICK_DIM={BRICK_DIM})"
    );
    (x + y * BRICK_DIM + z * BRICK_DIM * BRICK_DIM) as usize
}

// ---------------------------------------------------------------------------
// Brick
// ---------------------------------------------------------------------------

/// A primary SDF brick — 512 [`VoxelSample`]s = 4 096 bytes.
///
/// Each brick covers an 8×8×8 region of voxel space.  The flat voxel array is
/// indexed in z-major order via [`brick_index`].
///
/// The type is [`bytemuck::Pod`] and [`bytemuck::Zeroable`] for direct GPU upload.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct Brick {
    /// Voxel array, indexed via [`brick_index`].
    pub voxels: [VoxelSample; 512],
}

// SAFETY: VoxelSample is Pod/Zeroable (both u32 fields, repr(C)).
// [VoxelSample; 512] inherits those properties.  repr(C,align(8)) adds no
// padding beyond VoxelSample's own 8-byte size.
unsafe impl Zeroable for Brick {}
unsafe impl Pod for Brick {}

impl Default for Brick {
    /// Returns a brick where every voxel has distance = `f16::INFINITY` and
    /// all other fields zeroed — the canonical "empty / outside surface" state.
    fn default() -> Self {
        let v = VoxelSample::default();
        Self { voxels: [v; 512] }
    }
}

impl std::fmt::Debug for Brick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Brick")
            .field("voxels", &format_args!("[VoxelSample; 512]"))
            .finish()
    }
}

impl PartialEq for Brick {
    fn eq(&self, other: &Self) -> bool {
        self.voxels
            .iter()
            .zip(other.voxels.iter())
            .all(|(a, b)| a == b)
    }
}

impl Brick {
    /// Flat index from 3D brick coordinates.
    #[inline]
    pub fn index(x: u32, y: u32, z: u32) -> usize {
        brick_index(x, y, z)
    }

    /// Return a copy of the [`VoxelSample`] at `(x, y, z)`.
    #[inline]
    pub fn sample(&self, x: u32, y: u32, z: u32) -> VoxelSample {
        self.voxels[Self::index(x, y, z)]
    }

    /// Write `sample` to `(x, y, z)`.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, z: u32, sample: VoxelSample) {
        self.voxels[Self::index(x, y, z)] = sample;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::mem::size_of;

    #[test]
    fn size_is_4096_bytes() {
        assert_eq!(size_of::<Brick>(), 4096);
    }

    #[test]
    fn default_all_voxels_infinity() {
        let brick = Brick::default();
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let v = brick.sample(x, y, z);
                    assert_eq!(
                        v.distance(),
                        f16::INFINITY,
                        "voxel ({x},{y},{z}) distance is not f16::INFINITY"
                    );
                    assert!(
                        v.distance_f32() > 0.0,
                        "voxel ({x},{y},{z}) distance is not positive infinity"
                    );
                }
            }
        }
    }

    #[test]
    fn brick_index_corners() {
        assert_eq!(brick_index(0, 0, 0), 0);
        assert_eq!(brick_index(7, 0, 0), 7);
        assert_eq!(brick_index(0, 7, 0), 56);
        assert_eq!(brick_index(0, 0, 7), 448);
        assert_eq!(brick_index(7, 7, 7), 511);
    }

    #[test]
    fn brick_index_all_unique() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let idx = brick_index(x, y, z);
                    assert!(idx < 512, "index {idx} out of range for ({x},{y},{z})");
                    assert!(seen.insert(idx), "duplicate index {idx} for ({x},{y},{z})");
                }
            }
        }
        assert_eq!(seen.len(), 512);
    }

    #[test]
    fn sample_set_roundtrip() {
        let mut brick = Brick::default();
        let v = VoxelSample::new(1.5, 42, 128, 7, 0b001);
        brick.set(3, 5, 6, v);
        assert_eq!(brick.sample(3, 5, 6), v);
        // Other voxels remain at default (infinity distance)
        let other = brick.sample(0, 0, 0);
        assert!(other.distance().is_infinite());
    }

    #[test]
    fn pod_zeroable() {
        let brick = Brick::default();
        let bytes = bytemuck::bytes_of(&brick);
        assert_eq!(bytes.len(), 4096);
    }

    #[test]
    fn pod_zeroed_brick_gives_zero_bytes_for_zero_distance_voxels() {
        // bytemuck::zeroed() gives all-zero bytes — different from Default
        // (which fills with infinity). Confirm it doesn't panic and has right length.
        let brick: Brick = bytemuck::Zeroable::zeroed();
        let bytes: &[u8] = bytemuck::bytes_of(&brick);
        assert_eq!(bytes.len(), 4096);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic]
    fn brick_index_out_of_bounds_x() {
        brick_index(8, 0, 0);
    }

    #[test]
    #[should_panic]
    fn brick_index_out_of_bounds_y() {
        brick_index(0, 8, 0);
    }

    #[test]
    #[should_panic]
    fn brick_index_out_of_bounds_z() {
        brick_index(0, 0, 8);
    }
}
