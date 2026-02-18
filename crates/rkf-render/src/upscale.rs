//! Upscaling backend selection and custom temporal upscaler.
//!
//! Provides `UpscaleBackend` for choosing between DLSS and the custom temporal
//! upscaler, plus the `UpscalePass` compute shader implementation.
//!
//! The custom upscaler runs at display resolution: bilinear-samples the
//! internal-resolution HDR frame, reprojects history via motion vectors,
//! applies 3×3 neighborhood clipping in YCoCg color space, and blends for
//! temporal super-sampling.
//!
//! Both backends produce display-resolution HDR output in the same format.

use crate::dlss::DlssContext;
use crate::gbuffer::GBuffer;
use crate::history::{HistoryBuffers, HISTORY_COLOR_FORMAT, HISTORY_METADATA_FORMAT};
use crate::shading::ShadingPass;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Upscaling backend selection.
///
/// Auto-detected at startup based on hardware capabilities, but can be
/// overridden via configuration. Both backends consume the same inputs
/// (G-buffer + HDR color at internal resolution) and produce the same output
/// (display-resolution HDR color).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleBackend {
    /// NVIDIA DLSS — preferred on RTX hardware. AI-based temporal upscaling
    /// with detail hallucination. Cannot use SDF-specific material IDs.
    Dlss,
    /// Custom temporal upscaler — cross-platform fallback. Uses material ID
    /// rejection for perfect SDF edges and SDF normal awareness.
    Custom,
}

impl UpscaleBackend {
    /// Auto-select the best available upscaling backend.
    ///
    /// Prefers DLSS on NVIDIA hardware with driver/SDK support. Falls back
    /// to the custom temporal upscaler on all other hardware.
    pub fn auto_select(dlss_context: &DlssContext) -> Self {
        if dlss_context.is_available() {
            UpscaleBackend::Dlss
        } else {
            UpscaleBackend::Custom
        }
    }

    /// Human-readable name for logging.
    pub fn name(self) -> &'static str {
        match self {
            UpscaleBackend::Dlss => "DLSS",
            UpscaleBackend::Custom => "Custom Temporal",
        }
    }
}

/// GPU-uploadable upscale uniforms (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct UpscaleUniforms {
    /// Display resolution width.
    pub display_width: u32,
    /// Display resolution height.
    pub display_height: u32,
    /// Internal resolution width.
    pub internal_width: u32,
    /// Internal resolution height.
    pub internal_height: u32,
}

/// Custom temporal upscaler pass.
#[allow(dead_code)]
pub struct UpscalePass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for current frame inputs (group 0).
    current_frame_layout: wgpu::BindGroupLayout,
    /// Bind group for current frame inputs.
    current_frame_bind_group: wgpu::BindGroup,
    /// Bind group layout for history read (group 1).
    history_read_layout: wgpu::BindGroupLayout,
    /// Bind groups for history read [A-reads, B-reads] — swap each frame.
    history_read_bind_groups: [wgpu::BindGroup; 2],
    /// Bind group layout for outputs (group 2).
    output_layout: wgpu::BindGroupLayout,
    /// Bind groups for outputs [A-writes, B-writes] — swap each frame.
    output_bind_groups: [wgpu::BindGroup; 2],
    /// Bind group layout for uniforms (group 3).
    uniforms_layout: wgpu::BindGroupLayout,
    /// Bind group for uniforms.
    uniforms_bind_group: wgpu::BindGroup,
    /// Uniforms buffer.
    uniforms_buffer: wgpu::Buffer,
    /// Upscaled HDR output texture at display resolution.
    pub output_texture: wgpu::Texture,
    /// View for the upscaled HDR output.
    pub output_view: wgpu::TextureView,
    /// Bilinear sampler for HDR upsampling.
    sampler: wgpu::Sampler,
    /// Display resolution width.
    pub display_width: u32,
    /// Display resolution height.
    pub display_height: u32,
}

impl UpscalePass {
    /// Create the temporal upscale pass.
    pub fn new(
        device: &wgpu::Device,
        shading: &ShadingPass,
        gbuffer: &GBuffer,
        history: &HistoryBuffers,
        display_width: u32,
        display_height: u32,
        internal_width: u32,
        internal_height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("temporal_upscale.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/temporal_upscale.wgsl").into(),
            ),
        });

        // Bilinear sampler for HDR color upsampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("upscale bilinear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Output texture at display resolution (HDR, same format as history color)
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("upscaled hdr output"),
            size: wgpu::Extent3d {
                width: display_width,
                height: display_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HISTORY_COLOR_FORMAT, // Rgba16Float
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- Group 0: Current frame inputs ---
        let current_frame_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("upscale current frame layout"),
                entries: &[
                    // binding 0: HDR color (filterable for bilinear)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: G-buffer position
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
                    // binding 2: G-buffer normal
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
                    // binding 3: G-buffer material (uint)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4: G-buffer motion
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 5: bilinear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let current_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale current frame bind group"),
            layout: &current_frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shading.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.motion_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // --- Group 1: History read (two bind groups for ping-pong) ---
        let history_read_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("upscale history read layout"),
                entries: &[
                    // binding 0: history color (non-filterable)
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
                    // binding 1: history metadata (uint)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        // A reads from buffer 0, B reads from buffer 1
        let history_read_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscale history read A"),
                layout: &history_read_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&history.color_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history.metadata_views[0]),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscale history read B"),
                layout: &history_read_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&history.color_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history.metadata_views[1]),
                    },
                ],
            }),
        ];

        // --- Group 2: Outputs (two bind groups for ping-pong) ---
        let output_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("upscale output layout"),
                entries: &[
                    // binding 0: output color (write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: HISTORY_COLOR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 1: history color write
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: HISTORY_COLOR_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // binding 2: history metadata write
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: HISTORY_METADATA_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        // A writes to buffer 1 (opposite of read), B writes to buffer 0
        let output_bind_groups = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscale output A (writes to B)"),
                layout: &output_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history.color_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&history.metadata_views[1]),
                    },
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("upscale output B (writes to A)"),
                layout: &output_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&history.color_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&history.metadata_views[0]),
                    },
                ],
            }),
        ];

        // --- Group 3: Uniforms ---
        let uniforms = UpscaleUniforms {
            display_width,
            display_height,
            internal_width,
            internal_height,
        };
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("upscale uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let uniforms_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("upscale uniforms layout"),
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
            label: Some("upscale uniforms bind group"),
            layout: &uniforms_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            }],
        });

        // --- Pipeline ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("temporal upscale pipeline layout"),
            bind_group_layouts: &[
                &current_frame_layout,
                &history_read_layout,
                &output_layout,
                &uniforms_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("temporal upscale pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            current_frame_layout,
            current_frame_bind_group,
            history_read_layout,
            history_read_bind_groups,
            output_layout,
            output_bind_groups,
            uniforms_layout,
            uniforms_bind_group,
            uniforms_buffer,
            output_texture,
            output_view,
            sampler,
            display_width,
            display_height,
        }
    }

    /// Dispatch the temporal upscaler.
    ///
    /// `history_read_idx` is the current read index from `HistoryBuffers::read_index()`.
    /// When read_idx=0, we read from buffer 0 and write to buffer 1 (and vice versa).
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, history_read_idx: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("temporal upscale"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.current_frame_bind_group, &[]);
        pass.set_bind_group(1, &self.history_read_bind_groups[history_read_idx], &[]);
        pass.set_bind_group(2, &self.output_bind_groups[history_read_idx], &[]);
        pass.set_bind_group(3, &self.uniforms_bind_group, &[]);

        let wg_x = self.display_width.div_ceil(8);
        let wg_y = self.display_height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upscale_uniforms_size_is_16() {
        assert_eq!(std::mem::size_of::<UpscaleUniforms>(), 16);
    }

    #[test]
    fn upscale_uniforms_pod_roundtrip() {
        let u = UpscaleUniforms {
            display_width: 1920,
            display_height: 1080,
            internal_width: 960,
            internal_height: 540,
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 16);
        let u2: &UpscaleUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.display_width, u2.display_width);
        assert_eq!(u.internal_height, u2.internal_height);
    }

    #[test]
    fn backend_name() {
        assert_eq!(UpscaleBackend::Dlss.name(), "DLSS");
        assert_eq!(UpscaleBackend::Custom.name(), "Custom Temporal");
    }

    #[test]
    fn backend_equality() {
        assert_eq!(UpscaleBackend::Custom, UpscaleBackend::Custom);
        assert_ne!(UpscaleBackend::Dlss, UpscaleBackend::Custom);
    }
}
