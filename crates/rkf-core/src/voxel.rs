use bytemuck::{Pod, Zeroable};
use half::f16;

/// Flag bit: companion brick exists in bone data pool
pub const FLAG_HAS_BONE_DATA: u8 = 1 << 0;
/// Flag bit: companion brick exists in volumetric pool
pub const FLAG_HAS_VOLUMETRIC_DATA: u8 = 1 << 1;
/// Flag bit: companion brick exists in color pool
pub const FLAG_HAS_COLOR_DATA: u8 = 1 << 2;

/// A single voxel sample — 8 bytes, tightly packed for GPU upload.
///
/// Word 0: f16 distance | u16 material_id
/// Word 1: u8 blend_weight | u8 secondary_id | u8 flags | u8 reserved
///
/// Layout:
/// ```text
/// Word 0 (u32): [ f16 distance (bits 0–15) | u16 material_id (bits 16–31) ]
/// Word 1 (u32): [ u8 blend_weight (bits 0–7) | u8 secondary_id (bits 8–15) | u8 flags (bits 16–23) | u8 reserved (bits 24–31) ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct VoxelSample {
    /// Packed: lower 16 bits = f16 distance bits, upper 16 bits = u16 material_id
    pub word0: u32,
    /// Packed: byte0 = blend_weight, byte1 = secondary_id, byte2 = flags, byte3 = reserved
    pub word1: u32,
}

// SAFETY: VoxelSample is repr(C), all fields are u32 which are Pod
unsafe impl Zeroable for VoxelSample {}
unsafe impl Pod for VoxelSample {}

impl VoxelSample {
    /// Construct a new voxel sample with the given field values.
    ///
    /// `distance` is converted from f32 to f16 during packing.
    pub fn new(
        distance: f32,
        material_id: u16,
        blend_weight: u8,
        secondary_id: u8,
        flags: u8,
    ) -> Self {
        let reserved: u8 = 0;
        let word0 = (material_id as u32) << 16 | (f16::from_f32(distance).to_bits() as u32);
        let word1 = (reserved as u32) << 24
            | (flags as u32) << 16
            | (secondary_id as u32) << 8
            | (blend_weight as u32);
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

    /// Extract the material id stored in the upper 16 bits of word0.
    #[inline]
    pub fn material_id(&self) -> u16 {
        (self.word0 >> 16) as u16
    }

    /// Replace the material id in the upper 16 bits of word0, preserving distance.
    #[inline]
    pub fn set_material_id(&mut self, id: u16) {
        self.word0 = (self.word0 & 0xFFFF) | ((id as u32) << 16);
    }

    /// Extract blend_weight from byte 0 of word1.
    #[inline]
    pub fn blend_weight(&self) -> u8 {
        (self.word1 & 0xFF) as u8
    }

    /// Extract secondary_id from byte 1 of word1.
    #[inline]
    pub fn secondary_id(&self) -> u8 {
        ((self.word1 >> 8) & 0xFF) as u8
    }

    /// Extract flags from byte 2 of word1.
    #[inline]
    pub fn flags(&self) -> u8 {
        ((self.word1 >> 16) & 0xFF) as u8
    }

    /// Extract reserved byte from byte 3 of word1.
    #[inline]
    pub fn reserved(&self) -> u8 {
        ((self.word1 >> 24) & 0xFF) as u8
    }

    /// Returns true if the bone data companion brick flag is set.
    #[inline]
    pub fn has_bone_data(&self) -> bool {
        self.flags() & FLAG_HAS_BONE_DATA != 0
    }

    /// Returns true if the volumetric companion brick flag is set.
    #[inline]
    pub fn has_volumetric_data(&self) -> bool {
        self.flags() & FLAG_HAS_VOLUMETRIC_DATA != 0
    }

    /// Returns true if the color companion brick flag is set.
    #[inline]
    pub fn has_color_data(&self) -> bool {
        self.flags() & FLAG_HAS_COLOR_DATA != 0
    }

    /// Construct a voxel from geometry-first data.
    ///
    /// `distance_f16_bits` is the raw f16 bits from [`SdfCache`].
    /// `material_id` is a u8 (geometry-first uses 256 materials max).
    /// `color` is RGBA8 from the surface voxel (white = no tint).
    ///
    /// word0: f16 distance | (material_id << 16)
    /// word1: RGBA8 packed (R=byte0, G=byte1, B=byte2, A=byte3)
    pub fn from_geometry_data(distance_f16_bits: u16, material_id: u8, color: [u8; 4]) -> Self {
        let word0 = (distance_f16_bits as u32) | ((material_id as u32) << 16);
        let word1 = u32::from_le_bytes(color);
        Self { word0, word1 }
    }

    /// Extract per-voxel RGBA8 color from word1 (geometry-first format).
    #[inline]
    pub fn color(&self) -> [u8; 4] {
        self.word1.to_le_bytes()
    }
}

impl Default for VoxelSample {
    /// Returns a voxel far from any surface: distance = f16::INFINITY, material_id = 0, all other fields 0.
    fn default() -> Self {
        Self::new(f32::INFINITY, 0, 0, 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

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
        assert_eq!(sample.blend_weight(), 0);
        assert_eq!(sample.secondary_id(), 0);
        assert_eq!(sample.flags(), 0);
        assert_eq!(sample.reserved(), 0);
        // f16 bits = 0 => positive zero
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
        let blend = 128_u8;
        let sec = 7_u8;
        let flags = FLAG_HAS_BONE_DATA | FLAG_HAS_COLOR_DATA;

        let sample = VoxelSample::new(dist, mat, blend, sec, flags);

        // f16 has ~3 decimal digit precision; 1.5 is exactly representable
        assert_eq!(sample.distance_f32(), 1.5_f32);
        assert_eq!(sample.material_id(), mat);
        assert_eq!(sample.blend_weight(), blend);
        assert_eq!(sample.secondary_id(), sec);
        assert_eq!(sample.flags(), flags);
        assert_eq!(sample.reserved(), 0);
    }

    #[test]
    fn roundtrip_negative_distance() {
        // Negative distance = inside a surface
        let sample = VoxelSample::new(-0.5, 1, 0, 0, 0);
        assert_eq!(sample.distance_f32(), -0.5_f32);
        assert_eq!(sample.material_id(), 1);
    }

    #[test]
    fn edge_case_max_material_id() {
        let sample = VoxelSample::new(0.0, u16::MAX, 0, 0, 0);
        assert_eq!(sample.material_id(), u16::MAX);
    }

    #[test]
    fn set_material_id_preserves_distance() {
        let mut sample = VoxelSample::new(-0.25, 5, 128, 3, 0);
        assert_eq!(sample.material_id(), 5);
        sample.set_material_id(42);
        assert_eq!(sample.material_id(), 42);
        // Distance and other fields preserved.
        assert!((sample.distance_f32() - (-0.25)).abs() < 0.01);
        assert_eq!(sample.blend_weight(), 128);
        assert_eq!(sample.secondary_id(), 3);
    }

    #[test]
    fn edge_case_max_blend_weight() {
        let sample = VoxelSample::new(0.0, 0, u8::MAX, 0, 0);
        assert_eq!(sample.blend_weight(), u8::MAX);
        assert_eq!(sample.secondary_id(), 0);
        assert_eq!(sample.flags(), 0);
    }

    #[test]
    fn edge_case_max_secondary_id() {
        let sample = VoxelSample::new(0.0, 0, 0, u8::MAX, 0);
        assert_eq!(sample.secondary_id(), u8::MAX);
        assert_eq!(sample.blend_weight(), 0);
    }

    #[test]
    fn edge_case_f16_max_distance() {
        let dist = f16::MAX.to_f32();
        let sample = VoxelSample::new(dist, 0, 0, 0, 0);
        assert_eq!(sample.distance(), f16::MAX);
    }

    #[test]
    fn flag_has_bone_data() {
        let sample = VoxelSample::new(0.0, 0, 0, 0, FLAG_HAS_BONE_DATA);
        assert!(sample.has_bone_data());
        assert!(!sample.has_volumetric_data());
        assert!(!sample.has_color_data());
    }

    #[test]
    fn flag_has_volumetric_data() {
        let sample = VoxelSample::new(0.0, 0, 0, 0, FLAG_HAS_VOLUMETRIC_DATA);
        assert!(!sample.has_bone_data());
        assert!(sample.has_volumetric_data());
        assert!(!sample.has_color_data());
    }

    #[test]
    fn flag_has_color_data() {
        let sample = VoxelSample::new(0.0, 0, 0, 0, FLAG_HAS_COLOR_DATA);
        assert!(!sample.has_bone_data());
        assert!(!sample.has_volumetric_data());
        assert!(sample.has_color_data());
    }

    #[test]
    fn flag_all_companion_bits() {
        let all = FLAG_HAS_BONE_DATA | FLAG_HAS_VOLUMETRIC_DATA | FLAG_HAS_COLOR_DATA;
        let sample = VoxelSample::new(0.0, 0, 0, 0, all);
        assert!(sample.has_bone_data());
        assert!(sample.has_volumetric_data());
        assert!(sample.has_color_data());
        assert_eq!(sample.flags(), all);
    }

    #[test]
    fn fields_do_not_bleed_into_each_other() {
        // Set only material_id high bits — distance should stay 0
        let sample = VoxelSample::new(0.0, 0xABCD, 0, 0, 0);
        assert_eq!(sample.distance(), f16::from_bits(0));
        assert_eq!(sample.material_id(), 0xABCD);

        // Set only blend_weight — other word1 fields should be 0
        let sample2 = VoxelSample::new(0.0, 0, 0xFF, 0, 0);
        assert_eq!(sample2.secondary_id(), 0);
        assert_eq!(sample2.flags(), 0);
        assert_eq!(sample2.reserved(), 0);

        // Set only secondary_id
        let sample3 = VoxelSample::new(0.0, 0, 0, 0xFF, 0);
        assert_eq!(sample3.blend_weight(), 0);
        assert_eq!(sample3.flags(), 0);
        assert_eq!(sample3.reserved(), 0);
    }

    #[test]
    fn bytemuck_cast_slice_works() {
        let samples = vec![
            VoxelSample::new(1.0, 1, 0, 0, 0),
            VoxelSample::new(2.0, 2, 0, 0, 0),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&samples);
        assert_eq!(bytes.len(), 16);
    }
}
