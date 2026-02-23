//! Shading compute pass — stub pending v2 rewrite.
//!
//! Retains ShadeUniforms and HDR_FORMAT for downstream references.
//! The ShadingPass will be rebuilt in Phase 7 using v2 object-centric
//! bind groups (coarse field + BVH + object metadata).

use bytemuck::{Pod, Zeroable};

/// GPU-uploadable shade uniforms (32 bytes).
///
/// Contains debug mode selector and camera world-space position.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ShadeUniforms {
    /// Debug visualization mode (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular).
    pub debug_mode: u32,
    /// Number of active lights.
    pub num_lights: u32,
    /// Number of tiles horizontally (for indexing tile_light_counts/indices).
    pub num_tiles_x: u32,
    /// Shadow budget: max shadow-casting lights per pixel (0 = unlimited).
    pub shadow_budget_k: u32,
    /// Camera world-space position (xyz) + unused (w).
    pub camera_pos: [f32; 4],
}

/// Format of the HDR output texture.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Shading compute pass — will be rebuilt in Phase 7 for v2 object-centric SDF.
pub struct ShadingPass {
    _private: (),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_uniforms_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<ShadeUniforms>(), 32);
    }

    #[test]
    fn shade_uniforms_pod_roundtrip() {
        let u = ShadeUniforms {
            debug_mode: 3,
            num_lights: 5,
            num_tiles_x: 60,
            shadow_budget_k: 4,
            camera_pos: [1.0, 2.0, 3.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 32);
        let u2: &ShadeUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.debug_mode, u2.debug_mode);
        assert_eq!(u.camera_pos, u2.camera_pos);
    }

    #[test]
    fn shade_uniforms_camera_pos_offset_is_16() {
        let offset = std::mem::offset_of!(ShadeUniforms, camera_pos);
        assert_eq!(offset, 16);
    }
}
