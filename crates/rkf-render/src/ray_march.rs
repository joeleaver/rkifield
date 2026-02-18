//! Ray march compute pass.
//!
//! [`RayMarchPass`] manages the compute pipeline, output texture, and dispatch
//! for the basic sphere-tracing ray marcher (Phase 4, no DDA).

use crate::gpu_scene::GpuScene;

/// Default internal rendering resolution (width).
pub const INTERNAL_WIDTH: u32 = 960;
/// Default internal rendering resolution (height).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Ray march compute pass — dispatches the sphere-tracing shader and writes
/// to an internal-resolution output texture.
#[allow(dead_code)]
pub struct RayMarchPass {
    /// The compute pipeline for ray marching.
    pipeline: wgpu::ComputePipeline,
    /// Output texture at internal resolution.
    output_texture: wgpu::Texture,
    /// View for the output texture (used by blit pass for reading).
    pub output_view: wgpu::TextureView,
    /// Bind group layout for the output texture (group 1).
    output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the output texture (group 1).
    output_bind_group: wgpu::BindGroup,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl RayMarchPass {
    /// Create the ray march pass with the given internal resolution.
    pub fn new(device: &wgpu::Device, scene: &GpuScene, width: u32, height: u32) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ray_march.wgsl").into(),
            ),
        });

        // Output texture at internal resolution
        let (output_texture, output_view) = Self::create_output_texture(device, width, height);

        // Bind group layout for output texture (group 1)
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ray march output layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray march output bind group"),
            layout: &output_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&output_view),
            }],
        });

        // Pipeline layout: group 0 = scene data, group 1 = output texture
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ray march pipeline layout"),
            bind_group_layouts: &[&scene.bind_group_layout, &output_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ray march pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            output_texture,
            output_view,
            output_bind_group_layout,
            output_bind_group,
            width,
            height,
        }
    }

    /// Dispatch the ray march compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, scene: &GpuScene) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &self.output_bind_group, &[]);

        // Dispatch ceil(width/8) x ceil(height/8) workgroups
        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// The output texture reference (for creating blit pass bindings).
    pub fn output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    fn create_output_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ray march output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }
}
