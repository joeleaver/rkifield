//! Light types and GPU light buffer.
//!
//! Defines the [`Light`] struct (64 bytes, GPU-aligned) matching the architecture
//! spec, and [`LightBuffer`] for uploading light arrays as GPU storage buffers.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximum number of lights supported.
pub const MAX_LIGHTS: usize = 256;

/// Maximum lights per 16×16 tile for tiled culling.
pub const MAX_LIGHTS_PER_TILE: u32 = 64;

/// Tile size in pixels for tiled light culling.
pub const TILE_SIZE: u32 = 16;

/// Light type constants.
pub const LIGHT_TYPE_DIRECTIONAL: u32 = 0;
/// Point light type.
pub const LIGHT_TYPE_POINT: u32 = 1;
/// Spot light type.
pub const LIGHT_TYPE_SPOT: u32 = 2;

/// GPU-aligned light struct (64 bytes).
///
/// Uses flat f32 fields instead of vec3 to avoid WGSL vec3 alignment issues.
///
/// | Field | Offset | Description |
/// |-------|--------|-------------|
/// | light_type | 0 | 0=directional, 1=point, 2=spot |
/// | pos_{xyz} | 4 | World-space position |
/// | dir_{xyz} | 16 | Direction (for directional and spot) |
/// | color_{rgb} | 28 | Linear RGB color |
/// | intensity | 40 | Multiplier |
/// | range | 44 | Attenuation cutoff distance |
/// | inner_angle | 48 | Spot inner cone half-angle (radians) |
/// | outer_angle | 52 | Spot outer cone half-angle (radians) |
/// | cookie_index | 56 | -1 = none, else cookie texture array index |
/// | shadow_caster | 60 | 1 = casts SDF soft shadows |
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Light {
    /// Light type: 0=directional, 1=point, 2=spot.
    pub light_type: u32,
    /// World-space position X.
    pub pos_x: f32,
    /// World-space position Y.
    pub pos_y: f32,
    /// World-space position Z.
    pub pos_z: f32,
    /// Direction X (for directional and spot).
    pub dir_x: f32,
    /// Direction Y.
    pub dir_y: f32,
    /// Direction Z.
    pub dir_z: f32,
    /// Linear RGB color R.
    pub color_r: f32,
    /// Linear RGB color G.
    pub color_g: f32,
    /// Linear RGB color B.
    pub color_b: f32,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Attenuation cutoff distance (point/spot only).
    pub range: f32,
    /// Spot inner cone half-angle in radians.
    pub inner_angle: f32,
    /// Spot outer cone half-angle in radians.
    pub outer_angle: f32,
    /// Cookie texture index (-1 = none).
    pub cookie_index: i32,
    /// Whether this light casts SDF soft shadows (0 or 1).
    pub shadow_caster: u32,
}

impl Light {
    /// Create a directional light.
    pub fn directional(
        dir: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        shadow_caster: bool,
    ) -> Self {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        Self {
            light_type: LIGHT_TYPE_DIRECTIONAL,
            pos_x: 0.0,
            pos_y: 0.0,
            pos_z: 0.0,
            dir_x: dir[0] / len,
            dir_y: dir[1] / len,
            dir_z: dir[2] / len,
            color_r: color[0],
            color_g: color[1],
            color_b: color[2],
            intensity,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: shadow_caster as u32,
        }
    }

    /// Create a point light.
    pub fn point(
        pos: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        shadow_caster: bool,
    ) -> Self {
        Self {
            light_type: LIGHT_TYPE_POINT,
            pos_x: pos[0],
            pos_y: pos[1],
            pos_z: pos[2],
            dir_x: 0.0,
            dir_y: 0.0,
            dir_z: 0.0,
            color_r: color[0],
            color_g: color[1],
            color_b: color[2],
            intensity,
            range,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: shadow_caster as u32,
        }
    }

    /// Create a spot light.
    pub fn spot(
        pos: [f32; 3],
        dir: [f32; 3],
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        shadow_caster: bool,
    ) -> Self {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        Self {
            light_type: LIGHT_TYPE_SPOT,
            pos_x: pos[0],
            pos_y: pos[1],
            pos_z: pos[2],
            dir_x: dir[0] / len,
            dir_y: dir[1] / len,
            dir_z: dir[2] / len,
            color_r: color[0],
            color_g: color[1],
            color_b: color[2],
            intensity,
            range,
            inner_angle,
            outer_angle,
            cookie_index: -1,
            shadow_caster: shadow_caster as u32,
        }
    }
}

/// GPU-resident light buffer with bind group for the shading pass.
pub struct LightBuffer {
    /// Storage buffer containing the light array.
    pub buffer: wgpu::Buffer,
    /// Number of lights.
    pub count: u32,
}

impl LightBuffer {
    /// Upload a slice of lights to the GPU.
    ///
    /// If the slice is empty, uploads a single zeroed light to avoid zero-size buffers.
    pub fn upload(device: &wgpu::Device, lights: &[Light]) -> Self {
        // wgpu doesn't allow zero-size buffers — use a zeroed light as fallback.
        let zero = [Light::zeroed()];
        let (bytes, count): (&[u8], u32) = if lights.is_empty() {
            (bytemuck::cast_slice(&zero), 0)
        } else {
            (bytemuck::cast_slice(lights), lights.len() as u32)
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light buffer"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self { buffer, count }
    }

    /// Re-upload light data to an existing buffer.
    pub fn update(&self, queue: &wgpu::Queue, lights: &[Light]) {
        if !lights.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(lights));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<Light>(), 64);
    }

    #[test]
    fn light_pod_roundtrip() {
        let l = Light::point([1.0, 2.0, 3.0], [1.0, 0.5, 0.0], 10.0, 5.0, true);
        let bytes = bytemuck::bytes_of(&l);
        assert_eq!(bytes.len(), 64);
        let l2: &Light = bytemuck::from_bytes(bytes);
        assert_eq!(l.light_type, l2.light_type);
        assert_eq!(l.pos_x, l2.pos_x);
        assert_eq!(l.intensity, l2.intensity);
        assert_eq!(l.shadow_caster, l2.shadow_caster);
    }

    #[test]
    fn directional_normalizes_direction() {
        let l = Light::directional([2.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0, false);
        let len = (l.dir_x * l.dir_x + l.dir_y * l.dir_y + l.dir_z * l.dir_z).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn spot_normalizes_direction() {
        let l = Light::spot(
            [0.0, 5.0, 0.0],
            [0.0, -3.0, 0.0],
            [1.0, 1.0, 1.0],
            10.0,
            8.0,
            0.3,
            0.5,
            true,
        );
        let len = (l.dir_x * l.dir_x + l.dir_y * l.dir_y + l.dir_z * l.dir_z).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn point_light_has_correct_type() {
        let l = Light::point([0.0; 3], [1.0; 3], 1.0, 5.0, false);
        assert_eq!(l.light_type, LIGHT_TYPE_POINT);
    }

    #[test]
    fn directional_has_correct_type() {
        let l = Light::directional([0.0, 1.0, 0.0], [1.0; 3], 1.0, false);
        assert_eq!(l.light_type, LIGHT_TYPE_DIRECTIONAL);
    }

    #[test]
    fn spot_has_correct_type() {
        let l = Light::spot([0.0; 3], [0.0, -1.0, 0.0], [1.0; 3], 1.0, 5.0, 0.3, 0.5, false);
        assert_eq!(l.light_type, LIGHT_TYPE_SPOT);
    }

    #[test]
    fn max_lights_per_tile_is_64() {
        assert_eq!(MAX_LIGHTS_PER_TILE, 64);
    }

    #[test]
    fn tile_size_is_16() {
        assert_eq!(TILE_SIZE, 16);
    }
}
