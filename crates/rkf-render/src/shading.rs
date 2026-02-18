//! Shading compute pass — Cook-Torrance GGX BRDF.
//!
//! Reads the G-buffer and material table, evaluates PBR shading with a
//! directional sun light, and writes HDR color to an `Rgba16Float` texture.

use crate::gbuffer::GBuffer;
use crate::material_table::MaterialTable;
use wgpu::util::DeviceExt;

/// Format of the HDR output texture.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Shading compute pass.
#[allow(dead_code)]
pub struct ShadingPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// HDR output texture.
    pub hdr_texture: wgpu::Texture,
    /// View for the HDR output texture.
    pub hdr_view: wgpu::TextureView,
    /// Bind group layout for the HDR output (storage texture, write-only).
    hdr_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the HDR output.
    hdr_bind_group: wgpu::BindGroup,
    /// Debug uniform buffer (mode selector).
    debug_buffer: wgpu::Buffer,
    /// Bind group layout for debug uniforms.
    debug_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for debug uniforms.
    debug_bind_group: wgpu::BindGroup,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl ShadingPass {
    /// Create the shading pass.
    pub fn new(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        material_table: &MaterialTable,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shade.wgsl").into()),
        });

        // HDR output texture
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&Default::default());

        // Bind group layout for HDR output (group 2)
        let hdr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hdr output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let hdr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdr output bind group"),
            layout: &hdr_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            }],
        });

        // Debug uniform buffer (16 bytes: u32 mode + 12 bytes padding)
        let debug_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shading debug uniforms"),
            contents: &[0u8; 16],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let debug_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shading debug layout"),
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

        let debug_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shading debug bind group"),
            layout: &debug_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: debug_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout: group 0 = G-buffer read, group 1 = material table, group 2 = HDR output, group 3 = debug
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shading pipeline layout"),
            bind_group_layouts: &[
                &gbuffer.read_bind_group_layout,
                &material_table.bind_group_layout,
                &hdr_bind_group_layout,
                &debug_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shading pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            hdr_texture,
            hdr_view,
            hdr_bind_group_layout,
            hdr_bind_group,
            debug_buffer,
            debug_bind_group_layout,
            debug_bind_group,
            width,
            height,
        }
    }

    /// Dispatch the shading compute shader.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        material_table: &MaterialTable,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shading"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &gbuffer.read_bind_group, &[]);
        pass.set_bind_group(1, &material_table.bind_group, &[]);
        pass.set_bind_group(2, &self.hdr_bind_group, &[]);
        pass.set_bind_group(3, &self.debug_bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Set the debug visualization mode.
    ///
    /// - 0: Normal shading
    /// - 1: Surface normals
    /// - 2: World positions
    /// - 3: Material IDs (false-color)
    /// - 4: Diffuse only
    /// - 5: Specular only
    pub fn set_debug_mode(&self, queue: &wgpu::Queue, mode: u32) {
        let data = [mode, 0u32, 0u32, 0u32];
        queue.write_buffer(&self.debug_buffer, 0, bytemuck::cast_slice(&data));
    }
}
