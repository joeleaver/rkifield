//! Ray march compute pass — v2 object-centric ray marcher.
//!
//! [`RayMarchPass`] compiles the `ray_march.wgsl` shader and dispatches a
//! compute pass that sphere-traces through per-object SDFs (analytical or
//! voxelized) using BVH acceleration. Each thread processes one pixel at
//! internal resolution, writing G-buffer targets (position, normal, material,
//! motion vectors).
//!
//! # Bind Groups
//!
//! | Group | Content |
//! |-------|---------|
//! | 0 | GpuSceneV2 (brick pool, brick maps, objects, camera, scene, BVH) |
//! | 1 | G-buffer write targets (position, normal, material, motion) |
//! | 2 | Per-tile object lists from [`TileObjectCullPass`] (indices + counts) |
//! | 3 | Coarse acceleration field from [`CoarseField`] (3D texture + sampler + uniforms) |

use crate::coarse_field::CoarseField;
use crate::gbuffer::GBuffer;
use crate::gpu_scene::GpuSceneV2;
use crate::tile_object_cull::TileObjectCullPass;

/// Default internal rendering resolution (width).
pub const INTERNAL_WIDTH: u32 = 960;
/// Default internal rendering resolution (height).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Ray march compute pass — v2 object-centric sphere tracing.
pub struct RayMarchPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
}

impl RayMarchPass {
    /// Create the ray march pass by compiling the shader and building the pipeline.
    ///
    /// `tile_cull` provides the read-only bind group layout for per-tile object lists
    /// (group 2). `coarse_field` provides the 3D distance texture for empty-space
    /// skipping (group 3).
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuSceneV2,
        gbuffer: &GBuffer,
        tile_cull: &TileObjectCullPass,
        coarse_field: &CoarseField,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ray_march.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ray_march_pipeline_layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,
                &gbuffer.write_bind_group_layout,
                &tile_cull.read_bind_group_layout,
                &coarse_field.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ray_march_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline }
    }

    /// Record the ray march dispatch into a command encoder.
    ///
    /// Dispatches one thread per pixel at internal resolution using 8x8 workgroups.
    /// `tile_cull` provides per-tile object lists. `coarse_field` provides the
    /// 3D distance field for empty-space skipping.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuSceneV2,
        gbuffer: &GBuffer,
        tile_cull: &TileObjectCullPass,
        coarse_field: &CoarseField,
    ) {
        let workgroups_x = (gbuffer.width + 7) / 8;
        let workgroups_y = (gbuffer.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray_march"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &gbuffer.write_bind_group, &[]);
        pass.set_bind_group(2, &tile_cull.read_bind_group, &[]);
        pass.set_bind_group(3, &coarse_field.bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}
