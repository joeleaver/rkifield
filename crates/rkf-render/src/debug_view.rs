//! Debug visualization pass — reads G-buffer and writes displayable colors.
//!
//! [`DebugViewPass`] is a compute pass that converts G-buffer data into
//! human-viewable colors for development and debugging. Supports multiple
//! visualization modes (Lambert shading, normals, positions, material IDs).
//!
//! This is the minimal visualization layer for the "first pixels" milestone.

use crate::gbuffer::GBuffer;

/// Debug visualization mode.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    /// Basic Lambert shading with ambient (default).
    Lambert = 0,
    /// Surface normals mapped to colors.
    Normals = 1,
    /// World positions (fractional part).
    Positions = 2,
    /// Material IDs (hash-based distinct colors).
    MaterialIds = 3,
}

/// Uniform data for the debug view pass (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugUniforms {
    /// Visualization mode (see [`DebugMode`]).
    pub mode: u32,
    /// Padding to 16 bytes.
    pub _pad: [u32; 3],
}

/// Output texture format for the debug view (matches what BlitPass expects).
pub const DEBUG_VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Debug visualization compute pass.
pub struct DebugViewPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group for G-buffer read + uniforms (group 0).
    gbuffer_bind_group: wgpu::BindGroup,
    /// Bind group layout for G-buffer read + uniforms (retained for resize).
    #[allow(dead_code)]
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for output texture (group 1).
    output_bind_group: wgpu::BindGroup,
    /// Bind group layout for output texture (retained for resize).
    #[allow(dead_code)]
    output_bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for debug mode.
    uniform_buffer: wgpu::Buffer,
    /// The output display texture.
    pub output_texture: wgpu::Texture,
    /// View for the output texture (used by BlitPass).
    pub output_view: wgpu::TextureView,
}

impl DebugViewPass {
    /// Create the debug view pass.
    pub fn new(device: &wgpu::Device, gbuffer: &GBuffer) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug_view.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/debug_view.wgsl").into(),
            ),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_view_uniforms"),
            size: std::mem::size_of::<DebugUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Group 0: G-buffer read textures + uniforms
        let gbuffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("debug_view gbuffer layout"),
                entries: &[
                    // 0: position texture
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
                    // 1: normal texture
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
                    // 2: material texture (uint)
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
                    // 3: debug uniforms
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

        let gbuffer_bind_group = Self::create_gbuffer_bind_group(
            device,
            &gbuffer_bind_group_layout,
            gbuffer,
            &uniform_buffer,
        );

        // Group 1: output storage texture
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("debug_view output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: DEBUG_VIEW_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let (output_texture, output_view) =
            Self::create_output_texture(device, gbuffer.width, gbuffer.height);
        let output_bind_group = Self::create_output_bind_group(
            device,
            &output_bind_group_layout,
            &output_view,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("debug_view_pipeline_layout"),
            bind_group_layouts: &[&gbuffer_bind_group_layout, &output_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("debug_view_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            gbuffer_bind_group,
            gbuffer_bind_group_layout,
            output_bind_group,
            output_bind_group_layout,
            uniform_buffer,
            output_texture,
            output_view,
        }
    }

    /// Update the debug visualization mode.
    pub fn set_mode(&self, queue: &wgpu::Queue, mode: DebugMode) {
        let uniforms = DebugUniforms {
            mode: mode as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Dispatch the debug view compute pass.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
    ) {
        let workgroups_x = (width + 7) / 8;
        let workgroups_y = (height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("debug_view"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.gbuffer_bind_group, &[]);
        pass.set_bind_group(1, &self.output_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn create_gbuffer_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &GBuffer,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_view gbuffer"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&gbuffer.material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_output_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("debug_view_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEBUG_VIEW_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_output_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        output_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug_view output"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output_view),
            }],
        })
    }
}
