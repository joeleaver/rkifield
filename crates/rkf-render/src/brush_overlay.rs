//! GPU brush overlay — uploads geodesic distance data for cursor visualization.
//!
//! The shading pass uses this to draw a pixel-perfect brush ring on the object's
//! surface. The CPU computes geodesic flood fill distances and uploads them here.
//! The shader reads the distance per-pixel and draws a ring at the brush radius.
//!
//! Buffers are bound as part of the shading pass's group 3 (bindings 1-3).
//! After calling `update()`, the caller must call `ShadingPass::rebuild_brush_overlay()`
//! to point the bind group at the new buffers.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-uploadable brush overlay uniforms (48 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BrushOverlayUniforms {
    /// Brush radius in world units.
    pub brush_radius: f32,
    /// Brush falloff fraction (0.0 = hard edge, 1.0 = full falloff).
    pub brush_falloff: f32,
    /// Object ID the brush is targeting (0 = none).
    pub brush_object_id: u32,
    /// Whether the overlay is active (0 = hidden, 1 = visible).
    pub brush_active: u32,
    /// RGBA color for the brush ring visualization.
    pub brush_color: [f32; 4],
    /// Brush center in object-local space (xyz) + padding (w).
    pub brush_center_local: [f32; 4],
}

/// GPU-resident brush overlay buffers for the shading pass.
///
/// Provides geodesic distance data per-voxel so the shader can draw a
/// pixel-perfect brush cursor ring on the object's surface.
///
/// These buffers are bound in the shading pass's group 3 (bindings 1-3).
pub struct BrushOverlay {
    /// Storage buffer containing geodesic distances (array<f32>),
    /// indexed by overlay_slot * 512 + voxel_index.
    pub data_buffer: wgpu::Buffer,
    /// Storage buffer mapping brick_slot -> overlay_slot (or EMPTY_SLOT = 0xFFFFFFFF).
    pub map_buffer: wgpu::Buffer,
    /// Uniform buffer for brush parameters.
    pub uniform_buffer: wgpu::Buffer,
}

impl BrushOverlay {
    /// Create an empty brush overlay (placeholder when no brush is active).
    ///
    /// All buffers contain minimal data to satisfy wgpu validation.
    pub fn empty(device: &wgpu::Device) -> Self {
        let empty_data: [f32; 1] = [0.0];
        let empty_map: [u32; 1] = [0xFFFFFFFF]; // EMPTY_SLOT

        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush overlay data"),
            contents: bytemuck::cast_slice(&empty_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush overlay map"),
            contents: bytemuck::cast_slice(&empty_map),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let uniforms = BrushOverlayUniforms {
            brush_radius: 0.0,
            brush_falloff: 0.0,
            brush_object_id: 0,
            brush_active: 0,
            brush_color: [1.0, 1.0, 1.0, 1.0],
            brush_center_local: [0.0; 4],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush overlay uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            data_buffer,
            map_buffer,
            uniform_buffer,
        }
    }

    /// Update the brush overlay with new geodesic distance data and uniforms.
    ///
    /// `distances_data` is the flat geodesic distance array (overlay_slot * 512 + voxel_index).
    /// `map` maps brick pool slot index -> overlay slot (EMPTY_SLOT = no overlay data).
    /// `uniforms` contains brush radius, falloff, target object, and color.
    ///
    /// Data and map buffers are recreated each call (they change size).
    /// The uniform buffer is updated in-place via `write_buffer`.
    ///
    /// **After calling this, you must call `ShadingPass::rebuild_brush_overlay()`**
    /// to update the group 3 bind group with the new buffer references.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        distances_data: &[f32],
        map: &[u32],
        uniforms: &BrushOverlayUniforms,
    ) {
        // Recreate data buffer (size changes).
        self.data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush overlay data"),
            contents: bytemuck::cast_slice(distances_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Recreate map buffer (size changes).
        self.map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush overlay map"),
            contents: bytemuck::cast_slice(map),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Update uniform buffer in-place (fixed size).
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brush_overlay_uniforms_size_is_48_bytes() {
        assert_eq!(std::mem::size_of::<BrushOverlayUniforms>(), 48);
    }

    #[test]
    fn brush_overlay_uniforms_pod_roundtrip() {
        let u = BrushOverlayUniforms {
            brush_radius: 1.5,
            brush_falloff: 0.3,
            brush_object_id: 7,
            brush_active: 1,
            brush_color: [0.0, 1.0, 0.5, 0.8],
            brush_center_local: [1.0, 2.0, 3.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 48);
        let u2: &BrushOverlayUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.brush_radius, u2.brush_radius);
        assert_eq!(u.brush_object_id, u2.brush_object_id);
        assert_eq!(u.brush_color, u2.brush_color);
    }

    #[test]
    fn brush_overlay_uniforms_default_inactive() {
        let u = BrushOverlayUniforms {
            brush_radius: 0.0,
            brush_falloff: 0.0,
            brush_object_id: 0,
            brush_active: 0,
            brush_color: [1.0, 1.0, 1.0, 1.0],
            brush_center_local: [0.0; 4],
        };
        assert_eq!(u.brush_active, 0);
    }
}
