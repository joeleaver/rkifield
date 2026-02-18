//! Tone mapping compute pass — HDR to LDR conversion.
//!
//! Reads the HDR `Rgba16Float` output from the shading pass, applies ACES
//! tone mapping and sRGB gamma correction, and writes to an `Rgba8Unorm`
//! texture ready for display via the blit pass.

/// LDR output format.
pub const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Tone mapping compute pass.
#[allow(dead_code)]
pub struct ToneMapPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// LDR output texture.
    pub ldr_texture: wgpu::Texture,
    /// View for the LDR output (used by blit pass).
    pub ldr_view: wgpu::TextureView,
    /// Bind group layout for HDR input (sampled texture).
    hdr_input_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for HDR input.
    hdr_input_bind_group: wgpu::BindGroup,
    /// Bind group layout for LDR output (storage texture).
    ldr_output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for LDR output.
    ldr_output_bind_group: wgpu::BindGroup,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl ToneMapPass {
    /// Create the tone mapping pass.
    ///
    /// `hdr_view` is the HDR texture view from the shading pass.
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tone_map.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tone_map.wgsl").into()),
        });

        // LDR output texture
        let ldr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ldr output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ldr_view = ldr_texture.create_view(&Default::default());

        // Group 0: HDR input (sampled texture)
        let hdr_input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tone map hdr input layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let hdr_input_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone map hdr input bind group"),
            layout: &hdr_input_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(hdr_view),
            }],
        });

        // Group 1: LDR output (storage texture)
        let ldr_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tone map ldr output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: LDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let ldr_output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tone map ldr output bind group"),
            layout: &ldr_output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ldr_view),
            }],
        });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tone map pipeline layout"),
            bind_group_layouts: &[&hdr_input_bind_group_layout, &ldr_output_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tone map pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            ldr_texture,
            ldr_view,
            hdr_input_bind_group_layout,
            hdr_input_bind_group,
            ldr_output_bind_group_layout,
            ldr_output_bind_group,
            width,
            height,
        }
    }

    /// Dispatch the tone mapping compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tone map"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.hdr_input_bind_group, &[]);
        pass.set_bind_group(1, &self.ldr_output_bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// The LDR output texture (for screenshot staging copy).
    pub fn ldr_texture(&self) -> &wgpu::Texture {
        &self.ldr_texture
    }
}
