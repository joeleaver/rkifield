//! Ray march compute pass.
//!
//! [`RayMarchPass`] manages the compute pipeline and dispatch for the
//! DDA ray marcher that writes G-buffer output (Phase 6+).
//! Supports optional clipmap LOD traversal via bind group 2.

use crate::clipmap_gpu::ClipmapGpuData;
use crate::gbuffer::GBuffer;
use crate::gpu_scene::GpuScene;

/// Default internal rendering resolution (width).
pub const INTERNAL_WIDTH: u32 = 960;
/// Default internal rendering resolution (height).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Ray march compute pass — dispatches the DDA ray marcher and writes
/// to the G-buffer textures at internal resolution.
///
/// The pipeline layout includes three bind groups:
/// - Group 0: scene data (brick pool, occupancy, slots, camera, scene uniforms)
/// - Group 1: G-buffer output textures
/// - Group 2: clipmap LOD data (occupancy, slots, uniforms)
///
/// When `clipmap.num_levels == 0` (e.g. from [`ClipmapGpuData::empty`]),
/// the shader falls back to the single-grid DDA path.
pub struct RayMarchPass {
    /// The compute pipeline for ray marching.
    pipeline: wgpu::ComputePipeline,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl RayMarchPass {
    /// Create the ray march pass that writes to the given G-buffer.
    ///
    /// The `clipmap` parameter provides the bind group layout for group 2.
    /// Use [`ClipmapGpuData::empty`] when no clipmap is active.
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuScene,
        gbuffer: &GBuffer,
        clipmap: &ClipmapGpuData,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ray_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ray_march.wgsl").into(),
            ),
        });

        // Pipeline layout: group 0 = scene, group 1 = G-buffer, group 2 = clipmap
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ray march pipeline layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,
                &gbuffer.write_bind_group_layout,
                &clipmap.bind_group_layout,
            ],
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
            width: gbuffer.width,
            height: gbuffer.height,
        }
    }

    /// Dispatch the ray march compute shader, writing to the G-buffer.
    ///
    /// The `clipmap` parameter provides the bind group for group 2.
    /// Use [`ClipmapGpuData::empty`] when no clipmap is active.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        gbuffer: &GBuffer,
        clipmap: &ClipmapGpuData,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &gbuffer.write_bind_group, &[]);
        pass.set_bind_group(2, &clipmap.bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
