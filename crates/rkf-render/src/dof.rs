//! Depth of field compute pass (pre-upscale).
//!
//! Computes per-pixel circle of confusion from the G-buffer depth and
//! focus settings, then applies a disc-kernel gather blur weighted by CoC.
//! Near-field objects can bleed over in-focus regions (bokeh effect).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default focus distance in world units (2 metres).
pub const DEFAULT_FOCUS_DISTANCE: f32 = 2.0;
/// Default focus range — depth range over which focus transitions.
pub const DEFAULT_FOCUS_RANGE: f32 = 3.0;
/// Default maximum CoC radius in pixels.
pub const DEFAULT_MAX_COC: f32 = 8.0;
/// CoC texture format (signed float for near/far distinction).
pub const COC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// GPU-uploadable depth-of-field parameters (32 bytes).
///
/// Signed CoC convention: negative = near field (in front of focus plane),
/// positive = far field (behind focus plane).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct DofParams {
    /// World-space distance to the focus plane.
    pub focus_distance: f32,
    /// Depth range over which focus transitions (controls aperture width).
    pub focus_range: f32,
    /// Maximum circle of confusion radius in pixels.
    pub max_coc: f32,
    /// Dispatch width in pixels.
    pub width: u32,
    /// Dispatch height in pixels.
    pub height: u32,
    /// Depth at which near-field blurring starts.
    pub near_start: f32,
    /// Depth at which near-field is fully blurred.
    pub near_end: f32,
    /// Padding to reach 32 bytes.
    pub _pad: u32,
}

/// Depth of field compute pass.
///
/// Runs two compute dispatches per frame:
/// 1. CoC compute — reads G-buffer position (w = depth) and writes signed CoC
///    into an R32Float texture. Negative CoC = near field, positive = far field.
/// 2. DoF blur — disc-kernel gather over the HDR input, weighted by CoC, writes
///    blurred HDR into the output texture.
///
/// After `dispatch`, the blurred result is in [`DofPass::output_view`].
#[allow(dead_code)]
pub struct DofPass {
    coc_pipeline: wgpu::ComputePipeline,
    blur_pipeline: wgpu::ComputePipeline,

    /// CoC texture (R32Float, signed: negative=near, positive=far).
    coc_texture: wgpu::Texture,
    /// View of the CoC texture.
    pub coc_view: wgpu::TextureView,

    coc_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,

    coc_bind_group: wgpu::BindGroup,
    blur_bind_group: wgpu::BindGroup,

    coc_params_buffer: wgpu::Buffer,
    blur_params_buffer: wgpu::Buffer,

    /// Blurred HDR output texture (Rgba16Float, same resolution as input).
    pub output_texture: wgpu::Texture,
    /// View of the blurred HDR output texture.
    pub output_view: wgpu::TextureView,

    width: u32,
    height: u32,
}

impl DofPass {
    /// Create the DoF pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `hdr_view`: HDR texture view from the shading pass (internal resolution).
    /// - `gbuffer_position_view`: G-buffer position texture view (xyz = world pos, w = depth).
    /// - `width`: internal resolution width (e.g. 960).
    /// - `height`: internal resolution height (e.g. 540).
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        gbuffer_position_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader module (both entry points) ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dof.wgsl").into()),
        });

        // --- CoC texture ---
        let coc_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dof coc texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COC_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let coc_view = coc_texture.create_view(&Default::default());

        // --- Output texture (blurred HDR) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dof output texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- CoC bind group layout ---
        // binding 0: texture_2d<f32> (G-buffer position, non-filterable)
        // binding 1: texture_storage_2d<r16float, write> (CoC output)
        // binding 2: uniform buffer (DofParams)
        let coc_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dof coc bind group layout"),
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
                            format: COC_FORMAT,
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
                ],
            });

        // --- Blur bind group layout ---
        // binding 0: texture_2d<f32> (HDR input, non-filterable)
        // binding 1: texture_2d<f32> (CoC read, non-filterable)
        // binding 2: texture_storage_2d<rgba16float, write> (blurred output)
        // binding 3: uniform buffer (DofParams)
        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dof blur bind group layout"),
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

        // --- Default params ---
        let default_params = DofParams {
            focus_distance: DEFAULT_FOCUS_DISTANCE,
            focus_range: DEFAULT_FOCUS_RANGE,
            max_coc: DEFAULT_MAX_COC,
            width,
            height,
            near_start: 0.5,
            near_end: 1.0,
            _pad: 0,
        };

        // --- Params buffers ---
        let coc_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof coc params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof blur params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- CoC bind group ---
        let coc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof coc bind group"),
            layout: &coc_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(gbuffer_position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: coc_params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Blur bind group ---
        let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof blur bind group"),
            layout: &blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&coc_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: blur_params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline layouts ---
        let coc_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dof coc pipeline layout"),
                bind_group_layouts: &[&coc_bind_group_layout],
                push_constant_ranges: &[],
            });
        let blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dof blur pipeline layout"),
                bind_group_layouts: &[&blur_bind_group_layout],
                push_constant_ranges: &[],
            });

        // --- Pipelines ---
        let coc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dof coc_compute pipeline"),
            layout: Some(&coc_pipeline_layout),
            module: &shader,
            entry_point: Some("coc_compute"),
            compilation_options: Default::default(),
            cache: None,
        });
        let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dof dof_blur pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &shader,
            entry_point: Some("dof_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            coc_pipeline,
            blur_pipeline,
            coc_texture,
            coc_view,
            coc_bind_group_layout,
            blur_bind_group_layout,
            coc_bind_group,
            blur_bind_group,
            coc_params_buffer,
            blur_params_buffer,
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Dispatch both DoF passes into the command encoder.
    ///
    /// Order:
    /// 1. CoC compute — G-buffer position → CoC texture (signed R32Float).
    /// 2. DoF blur    — HDR + CoC → blurred HDR output.
    ///
    /// Each dispatch is its own compute pass to ensure a pipeline barrier
    /// between the CoC write and the subsequent blur read.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        // --- Pass 1: CoC compute ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dof coc_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.coc_pipeline);
            pass.set_bind_group(0, &self.coc_bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }

        // --- Pass 2: DoF blur ---
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dof dof_blur"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        }
    }

    /// Update focus parameters at runtime.
    ///
    /// Writes to both params buffers so CoC and blur use consistent values.
    pub fn update_focus(
        &self,
        queue: &wgpu::Queue,
        focus_distance: f32,
        focus_range: f32,
        max_coc: f32,
    ) {
        let params = DofParams {
            focus_distance,
            focus_range,
            max_coc,
            width: self.width,
            height: self.height,
            near_start: 0.5,
            near_end: 1.0,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&params);
        queue.write_buffer(&self.coc_params_buffer, 0, bytes);
        queue.write_buffer(&self.blur_params_buffer, 0, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dof_params_size_is_32() {
        assert_eq!(std::mem::size_of::<DofParams>(), 32);
    }

    #[test]
    fn dof_params_pod_roundtrip() {
        let p = DofParams {
            focus_distance: 2.0,
            focus_range: 3.0,
            max_coc: 8.0,
            width: 960,
            height: 540,
            near_start: 0.5,
            near_end: 1.0,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
        let p2: &DofParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.focus_distance, p2.focus_distance);
        assert_eq!(p.width, p2.width);
    }

    #[test]
    fn default_focus_values() {
        assert_eq!(DEFAULT_FOCUS_DISTANCE, 2.0);
        assert_eq!(DEFAULT_FOCUS_RANGE, 3.0);
        assert_eq!(DEFAULT_MAX_COC, 8.0);
    }
}
