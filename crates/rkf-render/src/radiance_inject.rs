//! Radiance injection compute pass for GI — stub pending v2 rewrite.

use bytemuck::{Pod, Zeroable};

/// Uniforms for the radiance injection pass (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct InjectUniforms {
    /// Total number of lights in the light buffer.
    pub num_lights: u32,
    /// Max lights that get shadow evaluation in injection (default: 1).
    pub max_shadow_lights: u32,
    /// Padding.
    pub _pad: [u32; 2],
}

/// Radiance injection compute pass — will be rewritten in Phase 8 for v2
/// coarse-field + BVH injection.
pub struct RadianceInjectPass {
    _private: (),
}
