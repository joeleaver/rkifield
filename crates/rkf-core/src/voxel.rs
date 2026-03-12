use bytemuck::{Pod, Zeroable};
use half::f16;

/// A single voxel sample — 8 bytes, tightly packed for GPU upload.
///
/// Layout:
/// ```text
/// Word 0 (u32): [ f16 distance (bits 0–15) | material_id (bits 16–21) | secondary_material_id (bits 22–27) | reserved (bits 28–31) ]
/// Word 1 (u32): reserved (bytes 0–2) | blend_weight (byte 3, 0=primary only, 255=secondary only)
/// ```
///
/// Per-voxel color is stored in a separate `ColorBrick` companion pool, not inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct VoxelSample {
    pub word0: u32,
    pub word1: u32,
}

// SAFETY: VoxelSample is repr(C), all fields are u32 which are Pod
unsafe impl Zeroable for VoxelSample {}
unsafe impl Pod for VoxelSample {}

impl VoxelSample {
    /// Construct a new voxel sample with distance, material, and color.
    ///
    /// `material_id` is masked to 6 bits (0–63).
    /// `color` is `[R, G, B, blend_weight]`.
    pub fn new(distance: f32, material_id: u16, color: [u8; 4]) -> Self {
        let word0 = (f16::from_f32(distance).to_bits() as u32)
            | (((material_id as u32) & 0x3F) << 16);
        let word1 = u32::from_le_bytes(color);
        Self { word0, word1 }
    }

    /// Extract the f16 signed distance stored in the lower 16 bits of word0.
    #[inline]
    pub fn distance(&self) -> f16 {
        f16::from_bits((self.word0 & 0xFFFF) as u16)
    }

    /// Convenience: return distance as f32.
    #[inline]
    pub fn distance_f32(&self) -> f32 {
        self.distance().to_f32()
    }

    /// Extract the primary material id stored in bits 16–21 of word0 (6 bits, 0–63).
    #[inline]
    pub fn material_id(&self) -> u16 {
        ((self.word0 >> 16) & 0x3F) as u16
    }

    /// Replace the primary material id (bits 16–21), preserving all other fields.
    #[inline]
    pub fn set_material_id(&mut self, id: u16) {
        self.word0 = (self.word0 & !(0x3F << 16)) | (((id as u32) & 0x3F) << 16);
    }

    /// Extract the secondary material id stored in bits 22–27 of word0 (6 bits, 0–63).
    #[inline]
    pub fn secondary_material_id(&self) -> u8 {
        ((self.word0 >> 22) & 0x3F) as u8
    }

    /// Extract the blend weight from byte 3 of word1 (0=primary, 255=secondary).
    #[inline]
    pub fn blend_weight(&self) -> u8 {
        (self.word1 >> 24) as u8
    }

    /// Construct a voxel from geometry-first data (single material, no flags).
    ///
    /// `distance_f16_bits` is the raw f16 bits from [`SdfCache`].
    /// `material_id` is masked to 6 bits.
    /// `color` is `[R, G, B, blend_weight]`.
    pub fn from_geometry_data(distance_f16_bits: u16, material_id: u8, color: [u8; 4]) -> Self {
        let word0 = (distance_f16_bits as u32)
            | (((material_id as u32) & 0x3F) << 16);
        let word1 = u32::from_le_bytes(color);
        Self { word0, word1 }
    }

    /// Construct from geometry-first data with secondary material and blend weight.
    ///
    /// `blend_weight` is 0–255 (0 = primary only, 255 = fully secondary).
    /// Bits 28–31 of word0 are reserved (zero).
    pub fn from_geometry_data_blended(
        distance_f16_bits: u16,
        material_id: u8,
        secondary_material_id: u8,
        blend_weight: u8,
    ) -> Self {
        let word0 = (distance_f16_bits as u32)
            | (((material_id as u32) & 0x3F) << 16)
            | (((secondary_material_id as u32) & 0x3F) << 22);
        // word1: bytes 0-2 reserved (0), byte 3 = blend_weight
        let word1 = (blend_weight as u32) << 24;
        Self { word0, word1 }
    }
}

impl Default for VoxelSample {
    /// Returns a voxel far from any surface: distance = f16::INFINITY, material_id = 0, white color.
    fn default() -> Self {
        Self::new(f32::INFINITY, 0, [255, 255, 255, 255])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    const WHITE: [u8; 4] = [255, 255, 255, 255];

    #[test]
    fn size_is_8_bytes() {
        assert_eq!(mem::size_of::<VoxelSample>(), 8);
    }

    #[test]
    fn pod_zeroable_bytes_of_works() {
        let sample = VoxelSample::default();
        let bytes = bytemuck::bytes_of(&sample);
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn zero_sample_all_zeros() {
        let sample: VoxelSample = bytemuck::Zeroable::zeroed();
        assert_eq!(sample.word0, 0);
        assert_eq!(sample.word1, 0);
        assert_eq!(sample.material_id(), 0);
        assert_eq!(sample.distance(), f16::from_f32(0.0));
    }

    #[test]
    fn default_has_infinity_distance() {
        let sample = VoxelSample::default();
        assert!(sample.distance().is_infinite());
        assert!(sample.distance_f32().is_infinite());
        assert!(sample.distance_f32() > 0.0);
    }

    #[test]
    fn roundtrip_typical_values() {
        let dist = 1.5_f32;
        let mat = 42_u16;
        let color = [128, 64, 32, 255];

        let sample = VoxelSample::new(dist, mat, color);

        assert_eq!(sample.distance_f32(), 1.5_f32);
        assert_eq!(sample.material_id(), mat);
    }

    #[test]
    fn roundtrip_negative_distance() {
        let sample = VoxelSample::new(-0.5, 1, WHITE);
        assert_eq!(sample.distance_f32(), -0.5_f32);
        assert_eq!(sample.material_id(), 1);
    }

    #[test]
    fn edge_case_max_material_id() {
        // Material id is 6 bits, max 63.
        let sample = VoxelSample::new(0.0, 63, WHITE);
        assert_eq!(sample.material_id(), 63);
    }

    #[test]
    fn material_id_masked_to_6_bits() {
        // Values above 63 are masked.
        let sample = VoxelSample::new(0.0, 0xFF, WHITE);
        assert_eq!(sample.material_id(), 63); // 0xFF & 0x3F = 63
    }

    #[test]
    fn set_material_id_preserves_distance() {
        let mut sample = VoxelSample::new(-0.25, 5, WHITE);
        assert_eq!(sample.material_id(), 5);
        sample.set_material_id(42);
        assert_eq!(sample.material_id(), 42);
        assert!((sample.distance_f32() - (-0.25)).abs() < 0.01);
    }

    #[test]
    fn edge_case_f16_max_distance() {
        let dist = f16::MAX.to_f32();
        let sample = VoxelSample::new(dist, 0, WHITE);
        assert_eq!(sample.distance(), f16::MAX);
    }

    #[test]
    fn from_geometry_data_roundtrip() {
        let dist_bits = f16::from_f32(1.5).to_bits();
        let color = [100, 200, 50, 255];
        let sample = VoxelSample::from_geometry_data(dist_bits, 42, color);
        assert_eq!(sample.distance_f32(), 1.5);
        assert_eq!(sample.material_id(), 42);
    }

    #[test]
    fn fields_do_not_bleed_into_each_other() {
        // material_id is 6 bits (16-21), secondary is 6 bits (22-27), bits 28-31 reserved.
        let sample = VoxelSample::new(0.0, 0x3F, [0, 0, 0, 0]);
        assert_eq!(sample.distance(), f16::from_bits(0));
        assert_eq!(sample.material_id(), 0x3F);
        assert_eq!(sample.secondary_material_id(), 0); // must not bleed
    }

    #[test]
    fn new_with_large_material_id_does_not_overflow_into_secondary() {
        // material_id > 63 should be masked to 6-bit range, not bleed into
        // secondary_material_id (bits 22-27).
        let sample = VoxelSample::new(0.0, 0x7F, [0, 0, 0, 0]);
        assert_eq!(sample.material_id(), 63); // masked to 6 bits
        assert_eq!(sample.secondary_material_id(), 0); // must NOT bleed
    }

    #[test]
    fn blended_material_roundtrip() {
        let dist_bits = f16::from_f32(-0.5).to_bits();
        let sample = VoxelSample::from_geometry_data_blended(dist_bits, 3, 7, 128);
        assert_eq!(sample.material_id(), 3);
        assert_eq!(sample.secondary_material_id(), 7);
        assert_eq!(sample.blend_weight(), 128);
        assert!((sample.distance_f32() - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn set_material_id_preserves_secondary() {
        let dist_bits = f16::from_f32(1.0).to_bits();
        let mut sample = VoxelSample::from_geometry_data_blended(dist_bits, 3, 7, 0);
        assert_eq!(sample.secondary_material_id(), 7);
        sample.set_material_id(10);
        assert_eq!(sample.material_id(), 10);
        assert_eq!(sample.secondary_material_id(), 7); // preserved
    }

    #[test]
    fn all_fields_pack_without_overlap() {
        // Set all fields to their max values and verify no overlap.
        let dist_bits = 0xFFFF_u16; // all distance bits set
        let sample = VoxelSample::from_geometry_data_blended(dist_bits, 63, 63, 0xFF);
        // word0: bits 0-15 = dist, 16-21 = mat(63), 22-27 = sec(63), 28-31 = 0
        assert_eq!(sample.word0 & 0x0FFF_FFFF, 0x0FFF_FFFF);
        assert_eq!(sample.word0 >> 28, 0); // reserved bits are zero
        assert_eq!(sample.material_id(), 63);
        assert_eq!(sample.secondary_material_id(), 63);
        assert_eq!(sample.blend_weight(), 0xFF);
    }

    #[test]
    fn bytemuck_cast_slice_works() {
        let samples = vec![
            VoxelSample::new(1.0, 1, WHITE),
            VoxelSample::new(2.0, 2, WHITE),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&samples);
        assert_eq!(bytes.len(), 16);
    }
}
