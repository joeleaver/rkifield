//! Motion blur compute pass (pre-upscale).
//!
//! Per-pixel directional blur along the G-buffer motion vectors.
//! Sample count is proportional to motion magnitude (max 16 samples).
//! Depth-aware weighting uses motion vector similarity to avoid
//! bleeding between objects moving in different directions.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default motion blur intensity multiplier.
pub const DEFAULT_MOTION_BLUR_INTENSITY: f32 = 1.0;
/// Default maximum number of blur samples per pixel.
pub const DEFAULT_MAX_MOTION_SAMPLES: u32 = 16;

/// GPU-uploadable motion blur parameters (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct MotionBlurParams {
    /// Render target width in pixels.
    pub width: u32,
    /// Render target height in pixels.
    pub height: u32,
    /// Motion blur strength multiplier (default 1.0).
    pub intensity: f32,
    /// Maximum number of blur samples per pixel (default 16).
    pub max_samples: u32,
}

/// Motion blur compute pass.
///
/// Reads the HDR color buffer and the G-buffer motion vectors, then writes a
/// motion-blurred HDR image to [`MotionBlurPass::output_view`].
///
/// Samples are taken symmetrically along the per-pixel motion vector direction.
/// Sample count scales with motion magnitude (clamped to `max_samples`).
/// A dot-product similarity weight prevents colour bleeding between pixels
/// moving in opposite directions.
///
/// After [`MotionBlurPass::dispatch`], the result is in [`MotionBlurPass::output_view`].
#[allow(dead_code)]
pub struct MotionBlurPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// Motion-blurred HDR output texture (Rgba16Float, internal resolution).
    pub output_texture: wgpu::Texture,
    /// View of the motion-blurred HDR output texture.
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl MotionBlurPass {
    /// Create the motion blur pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `hdr_view`: HDR texture view from the shading/tone-map pass (internal resolution).
    /// - `motion_view`: G-buffer motion vector texture view (`Rg32Float`, UV-space velocity).
    /// - `width`: internal resolution width (e.g. 960).
    /// - `height`: internal resolution height (e.g. 540).
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        motion_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/motion_blur.wgsl").into(),
            ),
        });

        // --- Output texture (Rgba16Float, STORAGE_BINDING | TEXTURE_BINDING) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion blur output texture"),
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
        // binding 1: texture_2d<f32> (motion vectors, non-filterable)
        // binding 2: texture_storage_2d<rgba16float, write> (output)
        // binding 3: uniform buffer (MotionBlurParams)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion blur bind group layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // --- Params buffer ---
        let default_params = MotionBlurParams {
            width,
            height,
            intensity: DEFAULT_MOTION_BLUR_INTENSITY,
            max_samples: DEFAULT_MAX_MOTION_SAMPLES,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion blur params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion blur bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("motion blur pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motion blur pipeline"),
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

    /// Dispatch the motion blur pass into the command encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("motion blur"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Update the motion blur intensity at runtime.
    ///
    /// Writes only the `intensity` field (byte offset 8) so the GPU params
    /// buffer stays consistent.
    pub fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 8, bytemuck::bytes_of(&intensity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_blur_params_size_is_16() {
        assert_eq!(std::mem::size_of::<MotionBlurParams>(), 16);
    }

    #[test]
    fn motion_blur_params_pod_roundtrip() {
        let p = MotionBlurParams {
            width: 960,
            height: 540,
            intensity: 1.0,
            max_samples: 16,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &MotionBlurParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.intensity, p2.intensity);
        assert_eq!(p.max_samples, p2.max_samples);
    }

    #[test]
    fn default_motion_blur_values() {
        assert_eq!(DEFAULT_MOTION_BLUR_INTENSITY, 1.0);
        assert_eq!(DEFAULT_MAX_MOTION_SAMPLES, 16);
    }
}
