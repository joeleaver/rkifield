//! GPU material table entry for the RKIField SDF engine.
//!
//! [`Material`] is a 96-byte, `repr(C)` struct stored in a global GPU storage
//! buffer as `array<Material>`, indexed by the voxel's `material_id` (u16).
//! Up to 65 536 materials are supported.
//!
//! All color values are **linear RGB** (not sRGB). The struct is [`Pod`] so it
//! can be uploaded to the GPU directly via `bytemuck::bytes_of`.
//!
//! # Noise channel flags
//!
//! [`NOISE_CHANNEL_ALBEDO`], [`NOISE_CHANNEL_ROUGHNESS`], and
//! [`NOISE_CHANNEL_NORMAL`] are bit-field constants for [`Material::noise_channels`].
//!
//! # Layout (96 bytes total)
//!
//! | Offset | Field              | Type     | Bytes |
//! |-------:|--------------------|----------|------:|
//! |      0 | albedo             | [f32; 3] |    12 |
//! |     12 | roughness          | f32      |     4 |
//! |     16 | metallic           | f32      |     4 |
//! |     20 | emission_color     | [f32; 3] |    12 |
//! |     32 | emission_strength  | f32      |     4 |
//! |     36 | subsurface         | f32      |     4 |
//! |     40 | subsurface_color   | [f32; 3] |    12 |
//! |     52 | opacity            | f32      |     4 |
//! |     56 | ior                | f32      |     4 |
//! |     60 | noise_scale        | f32      |     4 |
//! |     64 | noise_strength     | f32      |     4 |
//! |     68 | noise_channels     | u32      |     4 |
//! |     72 | shader_id          | u32      |     4 |
//! |     76 | _padding           | [f32; 5] |    20 |
//! |     96 | (end)              |          |       |
//!
//! # Example
//! ```
//! use rkf_core::material::{Material, NOISE_CHANNEL_ALBEDO, NOISE_CHANNEL_ROUGHNESS};
//! use std::mem;
//!
//! assert_eq!(mem::size_of::<Material>(), 96);
//!
//! let mut m = Material::default();
//! m.albedo = [1.0, 0.0, 0.0]; // red
//! m.roughness = 0.3;
//! m.noise_channels = NOISE_CHANNEL_ALBEDO | NOISE_CHANNEL_ROUGHNESS;
//! ```

use bytemuck::{Pod, Zeroable};

// --- noise channel constants -------------------------------------------------

/// Noise affects albedo color.
pub const NOISE_CHANNEL_ALBEDO: u32 = 1 << 0;
/// Noise affects roughness.
pub const NOISE_CHANNEL_ROUGHNESS: u32 = 1 << 1;
/// Noise perturbs surface normal (bump).
pub const NOISE_CHANNEL_NORMAL: u32 = 1 << 2;

// --- Material ----------------------------------------------------------------

/// PBR material with subsurface scattering and procedural noise -- 96 bytes.
///
/// Stored in a global GPU material table as `array<Material>`, indexed by
/// the voxel's `material_id` (u16). Maximum 65536 materials.
///
/// All color values are linear RGB (not sRGB).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Material {
    // PBR Baseline
    /// Base color (linear RGB), each component 0.0-1.0
    pub albedo: [f32; 3],
    /// Surface roughness: 0.0 = mirror, 1.0 = fully rough
    pub roughness: f32,
    /// Metallic factor: 0.0 = dielectric, 1.0 = metal
    pub metallic: f32,
    /// Emissive color (linear RGB)
    pub emission_color: [f32; 3],
    /// Emissive intensity (HDR, can exceed 1.0)
    pub emission_strength: f32,

    // Subsurface and Translucency
    /// Subsurface scattering strength: 0.0 = none, 1.0 = full SSS
    pub subsurface: f32,
    /// Color of scattered light (skin, wax, leaves)
    pub subsurface_color: [f32; 3],
    /// Opacity: 1.0 = solid, 0.0 = fully transparent
    pub opacity: f32,
    /// Index of refraction (glass ~1.5, water ~1.33)
    pub ior: f32,

    // Procedural Variation
    /// Spatial frequency of noise
    pub noise_scale: f32,
    /// Amplitude of noise perturbation
    pub noise_strength: f32,
    /// Bitfield: bit 0 = albedo, bit 1 = roughness, bit 2 = normal perturbation
    pub noise_channels: u32,

    // Shader selection
    /// Index into the shader registry (0 = default PBR). Set via `resolve_shader_ids()`.
    pub shader_id: u32,

    // Padding to 96 bytes for GPU alignment.
    // Fields above sum to 76 bytes; 5*f32 = 20 bytes brings total to 96.
    /// Reserved padding -- must be zero.
    pub _padding: [f32; 5],
}

// SAFETY: Material is repr(C), all fields are f32/u32/[f32;N] which are Pod
unsafe impl Zeroable for Material {}
unsafe impl Pod for Material {}

impl Default for Material {
    fn default() -> Self {
        Self {
            albedo: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
            emission_color: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0, 0.8, 0.6],
            opacity: 1.0,
            ior: 1.5,
            noise_scale: 0.0,
            noise_strength: 0.0,
            noise_channels: 0,
            shader_id: 0,
            _padding: [0.0; 5],
        }
    }
}

// --- tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    // -- size -----------------------------------------------------------------

    /// CRITICAL: Material must be exactly 96 bytes so the GPU array<Material>
    /// layout matches the Rust layout.
    #[test]
    fn size_is_96_bytes() {
        assert_eq!(
            mem::size_of::<Material>(),
            96,
            "Material size changed -- GPU layout will be wrong"
        );
    }

    // -- default values -------------------------------------------------------

    #[test]
    fn default_albedo_is_medium_gray() {
        let m = Material::default();
        assert_eq!(m.albedo, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn default_roughness_is_half() {
        let m = Material::default();
        assert!((m.roughness - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn default_metallic_is_zero() {
        let m = Material::default();
        assert_eq!(m.metallic, 0.0);
    }

    #[test]
    fn default_emission_is_black_zero_strength() {
        let m = Material::default();
        assert_eq!(m.emission_color, [0.0, 0.0, 0.0]);
        assert_eq!(m.emission_strength, 0.0);
    }

    #[test]
    fn default_subsurface_off() {
        let m = Material::default();
        assert_eq!(m.subsurface, 0.0);
    }

    #[test]
    fn default_subsurface_color_is_warm_skin_tone() {
        let m = Material::default();
        assert_eq!(m.subsurface_color, [1.0, 0.8, 0.6]);
    }

    #[test]
    fn default_opacity_is_solid() {
        let m = Material::default();
        assert_eq!(m.opacity, 1.0);
    }

    #[test]
    fn default_ior_is_glass() {
        let m = Material::default();
        assert!((m.ior - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn default_noise_all_zero() {
        let m = Material::default();
        assert_eq!(m.noise_scale, 0.0);
        assert_eq!(m.noise_strength, 0.0);
        assert_eq!(m.noise_channels, 0);
    }

    #[test]
    fn default_padding_zero() {
        let m = Material::default();
        assert_eq!(m._padding, [0.0_f32; 5]);
    }

    // -- bytemuck / Pod -------------------------------------------------------

    #[test]
    fn bytes_of_works() {
        let m = Material::default();
        let bytes = bytemuck::bytes_of(&m);
        assert_eq!(bytes.len(), 96);
    }

    #[test]
    fn zeroed_produces_all_zeros() {
        let m: Material = Zeroable::zeroed();
        let bytes = bytemuck::bytes_of(&m);
        assert!(
            bytes.iter().all(|&b| b == 0),
            "zeroed Material must be all-zero bytes"
        );
    }

    #[test]
    fn cast_slice_roundtrip() {
        let materials = vec![Material::default(); 4];
        let bytes: &[u8] = bytemuck::cast_slice(&materials);
        assert_eq!(bytes.len(), 4 * 96);
        let recovered: &[Material] = bytemuck::cast_slice(bytes);
        assert_eq!(recovered.len(), 4);
        assert_eq!(recovered[0], materials[0]);
        assert_eq!(recovered[3], materials[3]);
    }

    // -- field alignment / offsets --------------------------------------------

    #[test]
    fn first_field_at_offset_zero() {
        let m: Material = Zeroable::zeroed();
        let base = &m as *const Material as usize;
        let albedo_ptr = &m.albedo as *const [f32; 3] as usize;
        assert_eq!(albedo_ptr - base, 0);
    }

    #[test]
    fn field_offsets_are_correct() {
        let m: Material = Zeroable::zeroed();
        let base = &m as *const Material as usize;

        macro_rules! offset {
            ($field:expr) => {
                $field as *const _ as usize - base
            };
        }

        assert_eq!(offset!(&m.albedo),            0);
        assert_eq!(offset!(&m.roughness),         12);
        assert_eq!(offset!(&m.metallic),          16);
        assert_eq!(offset!(&m.emission_color),    20);
        assert_eq!(offset!(&m.emission_strength), 32);
        assert_eq!(offset!(&m.subsurface),        36);
        assert_eq!(offset!(&m.subsurface_color),  40);
        assert_eq!(offset!(&m.opacity),           52);
        assert_eq!(offset!(&m.ior),               56);
        assert_eq!(offset!(&m.noise_scale),       60);
        assert_eq!(offset!(&m.noise_strength),    64);
        assert_eq!(offset!(&m.noise_channels),    68);
        assert_eq!(offset!(&m.shader_id),         72);
        assert_eq!(offset!(&m._padding),          76);
    }

    // -- noise channel bitfield -----------------------------------------------

    #[test]
    fn noise_channel_constants_are_distinct_bits() {
        assert_eq!(NOISE_CHANNEL_ALBEDO,    0b001);
        assert_eq!(NOISE_CHANNEL_ROUGHNESS, 0b010);
        assert_eq!(NOISE_CHANNEL_NORMAL,    0b100);
    }

    #[test]
    fn noise_channel_no_overlap() {
        assert_eq!(NOISE_CHANNEL_ALBEDO & NOISE_CHANNEL_ROUGHNESS, 0);
        assert_eq!(NOISE_CHANNEL_ALBEDO & NOISE_CHANNEL_NORMAL,    0);
        assert_eq!(NOISE_CHANNEL_ROUGHNESS & NOISE_CHANNEL_NORMAL, 0);
    }

    #[test]
    fn set_single_noise_channel_albedo() {
        let mut m = Material::default();
        m.noise_channels |= NOISE_CHANNEL_ALBEDO;
        assert!(m.noise_channels & NOISE_CHANNEL_ALBEDO    != 0);
        assert!(m.noise_channels & NOISE_CHANNEL_ROUGHNESS == 0);
        assert!(m.noise_channels & NOISE_CHANNEL_NORMAL    == 0);
    }

    #[test]
    fn set_single_noise_channel_roughness() {
        let mut m = Material::default();
        m.noise_channels |= NOISE_CHANNEL_ROUGHNESS;
        assert!(m.noise_channels & NOISE_CHANNEL_ALBEDO    == 0);
        assert!(m.noise_channels & NOISE_CHANNEL_ROUGHNESS != 0);
        assert!(m.noise_channels & NOISE_CHANNEL_NORMAL    == 0);
    }

    #[test]
    fn set_single_noise_channel_normal() {
        let mut m = Material::default();
        m.noise_channels |= NOISE_CHANNEL_NORMAL;
        assert!(m.noise_channels & NOISE_CHANNEL_ALBEDO    == 0);
        assert!(m.noise_channels & NOISE_CHANNEL_ROUGHNESS == 0);
        assert!(m.noise_channels & NOISE_CHANNEL_NORMAL    != 0);
    }

    #[test]
    fn combine_all_noise_channels() {
        let all = NOISE_CHANNEL_ALBEDO | NOISE_CHANNEL_ROUGHNESS | NOISE_CHANNEL_NORMAL;
        assert_eq!(all, 0b111);

        let mut m = Material::default();
        m.noise_channels = all;
        assert!(m.noise_channels & NOISE_CHANNEL_ALBEDO    != 0);
        assert!(m.noise_channels & NOISE_CHANNEL_ROUGHNESS != 0);
        assert!(m.noise_channels & NOISE_CHANNEL_NORMAL    != 0);
    }

    #[test]
    fn clear_noise_channel() {
        let mut m = Material::default();
        m.noise_channels = NOISE_CHANNEL_ALBEDO | NOISE_CHANNEL_ROUGHNESS;
        m.noise_channels &= !NOISE_CHANNEL_ROUGHNESS;
        assert!(m.noise_channels & NOISE_CHANNEL_ALBEDO    != 0);
        assert!(m.noise_channels & NOISE_CHANNEL_ROUGHNESS == 0);
    }
}
