//! Screen-space radial blur god rays compute pass (pre-upscale).
//!
//! Radially blurs bright sky/sun pixels toward the projected sun position,
//! creating visible light shaft streaks. Based on GPU Gems 3 / Crytek
//! screen-space light shaft technique.
//!
//! Only **sky** samples contribute — the G-buffer position.w is checked
//! to distinguish sky (MAX_FLOAT) from geometry.  This prevents emissive
//! materials from producing erroneous radial streaks.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default god rays blur intensity multiplier.
pub const DEFAULT_GOD_RAYS_INTENSITY: f32 = 0.5;
/// Default number of radial blur samples per pixel.
pub const DEFAULT_GOD_RAYS_SAMPLES: u32 = 64;
/// Default per-sample decay factor.
pub const DEFAULT_GOD_RAYS_DECAY: f32 = 0.97;
/// Default sample density weight.
pub const DEFAULT_GOD_RAYS_DENSITY: f32 = 1.0;
/// Default luminance threshold for shaft contribution.
pub const DEFAULT_GOD_RAYS_THRESHOLD: f32 = 0.8;

/// GPU-uploadable god rays blur parameters (48 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GodRaysBlurParams {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Number of radial blur samples (default 64).
    pub num_samples: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    /// Sun screen position UV (0..1).
    pub sun_uv: [f32; 2],
    /// Overall intensity multiplier (default 0.5).
    pub intensity: f32,
    /// Per-sample falloff (default 0.97).
    pub decay: f32,
    /// Sample weight (default 1.0).
    pub density: f32,
    /// Luminance threshold for shaft contribution (default 0.8).
    pub threshold: f32,
    /// dot(sun_dir, cam_forward) for fade control.
    pub sun_dot: f32,
    /// Padding for 16-byte alignment.
    pub _pad1: f32,
}

/// Screen-space radial blur god rays compute pass.
///
/// Reads the HDR color buffer (post-volumetric composite) and writes
/// a radially blurred HDR image with light shaft streaks to
/// [`GodRaysBlurPass::output_view`].
#[allow(dead_code)]
pub struct GodRaysBlurPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// HDR output texture with radial god ray streaks (Rgba16Float, internal resolution).
    pub output_texture: wgpu::Texture,
    /// View of the HDR output texture.
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl GodRaysBlurPass {
    /// Create the god rays blur pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `hdr_view`: HDR texture view from volumetric composite (internal resolution).
    /// - `position_view`: G-buffer position texture view (Rgba32Float, .w = MAX_FLOAT for sky).
    /// - `width`: internal resolution width.
    /// - `height`: internal resolution height.
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        position_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("god_rays_blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/god_rays_blur.wgsl").into(),
            ),
        });

        // --- Output texture (Rgba16Float, STORAGE_BINDING | TEXTURE_BINDING) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("god rays blur output texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- Bind group layout ---
        // binding 0: texture_2d<f32> (HDR input, non-filterable)
        // binding 1: texture_storage_2d<rgba16float, write> (output)
        // binding 2: uniform buffer (GodRaysBlurParams)
        // binding 3: texture_2d<f32> (G-buffer position, .w = sky sentinel)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("god rays blur bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // --- Params buffer ---
        let default_params = GodRaysBlurParams {
            width,
            height,
            num_samples: DEFAULT_GOD_RAYS_SAMPLES,
            _pad0: 0,
            sun_uv: [0.5, 0.5],
            intensity: DEFAULT_GOD_RAYS_INTENSITY,
            decay: DEFAULT_GOD_RAYS_DECAY,
            density: DEFAULT_GOD_RAYS_DENSITY,
            threshold: DEFAULT_GOD_RAYS_THRESHOLD,
            sun_dot: 0.0,
            _pad1: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("god rays blur params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("god rays blur bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("god rays blur pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("god rays blur pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Dispatch the god rays blur pass into the command encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("god rays blur"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Update the sun screen position and visibility each frame.
    ///
    /// - `sun_uv_x`, `sun_uv_y`: projected sun position in [0,1] screen UV.
    /// - `sun_dot`: dot(sun_dir, cam_forward) — positive when sun is in front.
    pub fn update_sun(&self, queue: &wgpu::Queue, sun_uv_x: f32, sun_uv_y: f32, sun_dot: f32) {
        // sun_uv is at byte offset 16 (after width, height, num_samples, _pad0)
        let data: [f32; 2] = [sun_uv_x, sun_uv_y];
        queue.write_buffer(&self.params_buffer, 16, bytemuck::bytes_of(&data));
        // sun_dot is at byte offset 40 (after sun_uv[8] + intensity[4] + decay[4] + density[4] + threshold[4])
        queue.write_buffer(&self.params_buffer, 40, bytemuck::bytes_of(&sun_dot));
    }

    /// Update the god rays intensity at runtime.
    ///
    /// Writes only the `intensity` field (byte offset 24).
    pub fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 24, bytemuck::bytes_of(&intensity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn god_rays_blur_params_size_is_48() {
        assert_eq!(std::mem::size_of::<GodRaysBlurParams>(), 48);
    }

    #[test]
    fn god_rays_blur_params_pod_roundtrip() {
        let p = GodRaysBlurParams {
            width: 960,
            height: 540,
            num_samples: 64,
            _pad0: 0,
            sun_uv: [0.5, 0.3],
            intensity: 0.5,
            decay: 0.97,
            density: 1.0,
            threshold: 0.8,
            sun_dot: 0.7,
            _pad1: 0.0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 48);
        let p2: &GodRaysBlurParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.intensity, p2.intensity);
        assert_eq!(p.num_samples, p2.num_samples);
        assert_eq!(p.sun_uv[0], p2.sun_uv[0]);
        assert_eq!(p.sun_dot, p2.sun_dot);
    }

    #[test]
    fn default_god_rays_values() {
        assert_eq!(DEFAULT_GOD_RAYS_INTENSITY, 0.5);
        assert_eq!(DEFAULT_GOD_RAYS_SAMPLES, 64);
        assert!((DEFAULT_GOD_RAYS_DECAY - 0.97).abs() < 1e-6);
    }

    #[test]
    fn byte_offsets_match_struct_layout() {
        // Verify the byte offsets used in update_sun and set_intensity
        let p = GodRaysBlurParams {
            width: 0, height: 0, num_samples: 0, _pad0: 0,
            sun_uv: [0.0; 2], intensity: 0.0, decay: 0.0,
            density: 0.0, threshold: 0.0, sun_dot: 0.0, _pad1: 0.0,
        };
        let base = &p as *const _ as usize;
        let sun_uv_offset = &p.sun_uv as *const _ as usize - base;
        let intensity_offset = &p.intensity as *const _ as usize - base;
        let sun_dot_offset = &p.sun_dot as *const _ as usize - base;
        assert_eq!(sun_uv_offset, 16);
        assert_eq!(intensity_offset, 24);
        assert_eq!(sun_dot_offset, 40);
    }
}
