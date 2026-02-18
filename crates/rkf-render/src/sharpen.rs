//! Edge-aware sharpening compute pass.
//!
//! 5×5 cross kernel with material + depth similarity weighting.
//! Unsharp mask: `result = center + (center - blur) * strength`.
//! Material ID boundaries create hard sharpening edges that prevent
//! halos at object silhouettes.

use crate::gbuffer::GBuffer;
use crate::history::HISTORY_COLOR_FORMAT;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-uploadable sharpen uniforms (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SharpenUniforms {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Sharpening strength (0.0 = no sharpening, 1.0 = strong).
    pub strength: f32,
    /// Padding.
    pub _pad: u32,
}

/// Default sharpening strength.
pub const DEFAULT_SHARPEN_STRENGTH: f32 = 0.5;

/// Edge-aware sharpening pass.
#[allow(dead_code)]
pub struct SharpenPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Input bind group layout (group 0).
    input_layout: wgpu::BindGroupLayout,
    /// Input bind group.
    input_bind_group: wgpu::BindGroup,
    /// Output bind group layout (group 1).
    output_layout: wgpu::BindGroupLayout,
    /// Output bind group.
    output_bind_group: wgpu::BindGroup,
    /// Uniforms bind group layout (group 2).
    uniforms_layout: wgpu::BindGroupLayout,
    /// Uniforms bind group.
    uniforms_bind_group: wgpu::BindGroup,
    /// Uniforms buffer.
    uniforms_buffer: wgpu::Buffer,
    /// Sharpened output texture (display resolution, HDR).
    pub output_texture: wgpu::Texture,
    /// View for the sharpened output.
    pub output_view: wgpu::TextureView,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
}

impl SharpenPass {
    /// Create the sharpening pass.
    ///
    /// `input_view` is the upscaled HDR texture view (display resolution).
    /// `gbuffer` provides position (depth) and material for edge detection.
    pub fn new(
        device: &wgpu::Device,
        input_view: &wgpu::TextureView,
        gbuffer: &GBuffer,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sharpen.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sharpen.wgsl").into()),
        });

        // Output texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sharpened output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HISTORY_COLOR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // Group 0: input (upscaled color + G-buffer for edge detection)
        let input_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharpen input layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharpen input bind group"),
            layout: &input_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.material_view),
                },
            ],
        });

        // Group 1: output
        let output_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharpen output layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: HISTORY_COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharpen output bind group"),
            layout: &output_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output_view),
            }],
        });

        // Group 2: uniforms
        let uniforms = SharpenUniforms {
            width,
            height,
            strength: DEFAULT_SHARPEN_STRENGTH,
            _pad: 0,
        };
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sharpen uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sharpen uniforms layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sharpen uniforms bind group"),
            layout: &uniforms_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sharpen pipeline layout"),
            bind_group_layouts: &[&input_layout, &output_layout, &uniforms_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sharpen pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            input_layout,
            input_bind_group,
            output_layout,
            output_bind_group,
            uniforms_layout,
            uniforms_bind_group,
            uniforms_buffer,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Dispatch the sharpening pass.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sharpen"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.input_bind_group, &[]);
        pass.set_bind_group(1, &self.output_bind_group, &[]);
        pass.set_bind_group(2, &self.uniforms_bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Update sharpening strength at runtime.
    pub fn set_strength(&self, queue: &wgpu::Queue, strength: f32) {
        queue.write_buffer(&self.uniforms_buffer, 8, bytemuck::bytes_of(&strength));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharpen_uniforms_size_is_16() {
        assert_eq!(std::mem::size_of::<SharpenUniforms>(), 16);
    }

    #[test]
    fn sharpen_uniforms_pod_roundtrip() {
        let u = SharpenUniforms {
            width: 1920,
            height: 1080,
            strength: 0.5,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 16);
        let u2: &SharpenUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.width, u2.width);
        assert!((u.strength - u2.strength).abs() < 1e-6);
    }

    #[test]
    fn default_strength() {
        assert!((DEFAULT_SHARPEN_STRENGTH - 0.5).abs() < 1e-6);
    }

    #[test]
    fn strength_offset_is_8() {
        assert_eq!(std::mem::offset_of!(SharpenUniforms, strength), 8);
    }
}
