//! CSG edit compute pipeline.
//!
//! [`CsgEditPipeline`] manages the GPU compute pipeline that applies CSG
//! operations to individual bricks in the brick pool. The pipeline reads
//! [`EditParams`](crate::types::EditParams) from a uniform buffer and
//! modifies voxels in-place via a `read_write` storage binding.
//!
//! # Usage
//!
//! ```ignore
//! let pipeline = CsgEditPipeline::new(&device);
//!
//! // For each affected brick:
//! let params = edit_params.with_brick_info(base_index, world_min, voxel_size);
//! pipeline.dispatch(&mut encoder, &device, &brick_pool_buffer, &params);
//! ```

use crate::types::EditParams;
use wgpu::util::DeviceExt;

/// GPU compute pipeline for CSG edit operations.
///
/// Dispatches the `csg_edit.wgsl` shader with one workgroup (8x8x8) per brick.
/// The pipeline uses its own bind group layout separate from the render pipeline,
/// with `read_write` access to the brick pool storage buffer.
pub struct CsgEditPipeline {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout: binding 0 = rw storage (brick pool), binding 1 = uniform (EditParams).
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CsgEditPipeline {
    /// Create the CSG edit pipeline.
    ///
    /// This compiles the `csg_edit.wgsl` shader and creates the pipeline layout.
    /// The brick pool buffer is NOT bound here — it is bound per-dispatch via
    /// [`dispatch`](Self::dispatch).
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("csg_edit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/csg_edit.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("csg edit bind group layout"),
                entries: &[
                    // Binding 0: brick pool — read_write storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: edit params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("csg edit pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("csg edit pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Dispatch the CSG edit shader for a single brick.
    ///
    /// `brick_pool_buffer` must have been created with `STORAGE` usage.
    /// The buffer needs `read_write` access — if the render pipeline created it
    /// as read-only, a separate buffer with `STORAGE` usage (both read and write)
    /// must be provided.
    ///
    /// `params` should have its per-brick fields set via
    /// [`EditParams::with_brick_info`].
    ///
    /// Dispatches exactly 1 workgroup (8x8x8 = 512 threads).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        brick_pool_buffer: &wgpu::Buffer,
        params: &EditParams,
    ) {
        // Create a temporary uniform buffer with the edit params.
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("csg edit params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("csg edit bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brick_pool_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csg edit"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // 1 workgroup = 1 brick (8x8x8)
    }

    /// Dispatch the CSG edit shader for multiple bricks in a single compute pass.
    ///
    /// More efficient than calling [`dispatch`](Self::dispatch) in a loop because
    /// it reuses the same compute pass. Each brick gets its own bind group with
    /// a unique uniform buffer (different `brick_base_index` / `brick_world_min`).
    pub fn dispatch_multi(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        brick_pool_buffer: &wgpu::Buffer,
        params_list: &[EditParams],
    ) {
        if params_list.is_empty() {
            return;
        }

        // Pre-create all uniform buffers and bind groups
        let bind_groups: Vec<wgpu::BindGroup> = params_list
            .iter()
            .enumerate()
            .map(|(i, params)| {
                let params_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("csg edit params [{i}]")),
                        contents: bytemuck::bytes_of(params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("csg edit bind group [{i}]")),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: brick_pool_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csg edit multi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);

        for bind_group in &bind_groups {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
