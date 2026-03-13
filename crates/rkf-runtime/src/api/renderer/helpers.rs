//! Free-standing helper functions for the renderer.

use glam::Vec3;

use rkf_core::aabb::Aabb;
use rkf_core::transform_bake;

use super::OFFSCREEN_FORMAT;

/// Transform a local-space AABB to world-space using a baked world transform.
pub fn transform_aabb(aabb: &Aabb, wt: &transform_bake::WorldTransform) -> Aabb {
    let smin = aabb.min * wt.scale;
    let smax = aabb.max * wt.scale;
    let corners = [
        Vec3::new(smin.x, smin.y, smin.z),
        Vec3::new(smax.x, smin.y, smin.z),
        Vec3::new(smin.x, smax.y, smin.z),
        Vec3::new(smax.x, smax.y, smin.z),
        Vec3::new(smin.x, smin.y, smax.z),
        Vec3::new(smax.x, smin.y, smax.z),
        Vec3::new(smin.x, smax.y, smax.z),
        Vec3::new(smax.x, smax.y, smax.z),
    ];
    let mut wmin = Vec3::splat(f32::MAX);
    let mut wmax = Vec3::splat(f32::MIN);
    for c in &corners {
        let r = wt.rotation * *c + wt.position;
        wmin = wmin.min(r);
        wmax = wmax.max(r);
    }
    Aabb::new(wmin, wmax)
}

/// Create the offscreen render target texture at the given resolution.
pub(super) fn create_offscreen_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_target"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Create a readback buffer sized for the given dimensions.
pub(super) fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    let bytes_per_pixel = 4u32;
    let unpadded_row = width * bytes_per_pixel;
    let padded_row = (unpadded_row + 255) & !255;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}
