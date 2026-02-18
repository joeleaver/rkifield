//! Bilateral upscale for volumetric scattering buffer — Phase 11 task 11.7.
//!
//! Upsamples the half-resolution volumetric scattering buffer to full internal
//! resolution using edge-aware bilateral filtering guided by the full-resolution
//! depth buffer.
//!
//! The bilateral filter weights each low-res neighbor by both its bilinear
//! proximity and depth similarity to the full-res pixel, preventing fog from
//! bleeding across sharp depth discontinuities (character silhouettes, object
//! edges).
//!
//! Input: half-res [`wgpu::TextureFormat::Rgba16Float`] (480×270)
//! Depth: full-res G-buffer position texture (960×540, `.w` channel = depth)
//! Output: full-res [`wgpu::TextureFormat::Rgba16Float`] (960×540)

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------- Public constants ----------

/// Default depth similarity sigma for bilateral weighting.
///
/// Smaller values preserve edges more aggressively; larger values behave more
/// like plain bilinear upscaling. 0.1 = 10% relative depth difference halves
/// the bilateral weight.
pub const DEFAULT_DEPTH_SIGMA: f32 = 0.1;

/// Output format for the upscaled volumetric buffer.
pub const VOL_UPSCALE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// ---------- GPU struct ----------

/// GPU-uploadable bilateral upscale parameters (32 bytes).
///
/// Memory layout:
/// ```text
/// offset  0 — out_width    u32   (4 bytes)
/// offset  4 — out_height   u32   (4 bytes)
/// offset  8 — in_width     u32   (4 bytes)
/// offset 12 — in_height    u32   (4 bytes)
/// offset 16 — depth_sigma  f32   (4 bytes)
/// offset 20 — _pad0        u32   (4 bytes)
/// offset 24 — _pad1        u32   (4 bytes)
/// offset 28 — _pad2        u32   (4 bytes)
/// total: 32 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct VolUpscaleParams {
    /// Full internal resolution width (e.g. 960).
    pub out_width: u32,
    /// Full internal resolution height (e.g. 540).
    pub out_height: u32,
    /// Half-resolution input width (e.g. 480).
    pub in_width: u32,
    /// Half-resolution input height (e.g. 270).
    pub in_height: u32,
    /// Depth similarity sigma; controls edge sharpness of the bilateral filter.
    pub depth_sigma: f32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
    /// Padding.
    pub _pad2: u32,
}

// ---------- Pass ----------

/// Bilateral upscale compute pass.
///
/// Owns the full-resolution output texture. Takes external views of the
/// half-res scatter buffer (produced by [`vol_march`] or [`vol_temporal`]) and
/// the full-res depth buffer (G-buffer position `.w`).
///
/// [`vol_march`]: crate::vol_march
/// [`vol_temporal`]: crate::vol_temporal
#[allow(dead_code)]
pub struct VolUpscalePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// Full-resolution output scatter texture.
    pub output_texture: wgpu::Texture,
    /// View into [`output_texture`](VolUpscalePass::output_texture).
    pub output_view: wgpu::TextureView,
    out_width: u32,
    out_height: u32,
}

impl VolUpscalePass {
    /// Create a new bilateral upscale pass.
    ///
    /// # Parameters
    /// - `device` — wgpu device.
    /// - `half_res_scatter_view` — view of the half-res volumetric scatter
    ///   texture (`Rgba16Float`, `in_width × in_height`).
    /// - `full_res_depth_view` — view of the full-res G-buffer position texture
    ///   (`.w` channel is used as depth; `out_width × out_height`).
    /// - `out_width`, `out_height` — full internal render resolution.
    /// - `in_width`, `in_height` — half render resolution.
    pub fn new(
        device: &wgpu::Device,
        half_res_scatter_view: &wgpu::TextureView,
        full_res_depth_view: &wgpu::TextureView,
        out_width: u32,
        out_height: u32,
        in_width: u32,
        in_height: u32,
    ) -> Self {
        // ---- Shader ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vol_upscale"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vol_upscale.wgsl").into(),
            ),
        });

        // ---- Params buffer ----
        let params = VolUpscaleParams {
            out_width,
            out_height,
            in_width,
            in_height,
            depth_sigma: DEFAULT_DEPTH_SIGMA,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol_upscale_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Output texture (full-res) ----
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vol_upscale_output"),
            size: wgpu::Extent3d {
                width: out_width,
                height: out_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VOL_UPSCALE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- Bind group layout ----
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol_upscale_bgl"),
            entries: &[
                // 0: uniform params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: half-res scatter (texture_2d<f32>, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: full-res depth (texture_2d<f32>, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: output scatter (storage texture, write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: VOL_UPSCALE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ---- Pipeline ----
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol_upscale_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol_upscale_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ---- Bind group ----
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol_upscale_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(half_res_scatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(full_res_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            output_texture,
            output_view,
            out_width,
            out_height,
        }
    }

    /// Dispatch the bilateral upscale at full internal resolution.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = 8u32;
        let dispatch_x = self.out_width.div_ceil(wg);
        let dispatch_y = self.out_height.div_ceil(wg);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol_upscale"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    /// Override the depth similarity sigma at runtime.
    ///
    /// Smaller values sharpen edges; larger values approach plain bilinear.
    pub fn set_depth_sigma(&self, queue: &wgpu::Queue, sigma: f32) {
        queue.write_buffer(
            &self.params_buffer,
            std::mem::offset_of!(VolUpscaleParams, depth_sigma) as u64,
            bytemuck::bytes_of(&sigma),
        );
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_upscale_params_size_is_32() {
        assert_eq!(std::mem::size_of::<VolUpscaleParams>(), 32);
    }

    #[test]
    fn vol_upscale_params_pod_roundtrip() {
        let p = VolUpscaleParams {
            out_width: 960,
            out_height: 540,
            in_width: 480,
            in_height: 270,
            depth_sigma: 0.1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
        let p2: &VolUpscaleParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.out_width, p2.out_width);
        assert_eq!(p.depth_sigma, p2.depth_sigma);
    }

    #[test]
    fn vol_upscale_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(VolUpscaleParams, out_width), 0);
        assert_eq!(std::mem::offset_of!(VolUpscaleParams, out_height), 4);
        assert_eq!(std::mem::offset_of!(VolUpscaleParams, in_width), 8);
        assert_eq!(std::mem::offset_of!(VolUpscaleParams, in_height), 12);
        assert_eq!(std::mem::offset_of!(VolUpscaleParams, depth_sigma), 16);
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_DEPTH_SIGMA, 0.1);
        assert_eq!(VOL_UPSCALE_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn upscale_ratio() {
        let out_w = 960u32;
        let out_h = 540u32;
        let in_w = 480u32;
        let in_h = 270u32;
        assert_eq!(out_w / in_w, 2);
        assert_eq!(out_h / in_h, 2);
    }
}
