//! Volumetric temporal reprojection compute pass — Phase 11 task 11.6.
//!
//! Reprojects the previous frame's half-res volumetric scatter buffer using
//! motion vectors from the G-buffer, then blends it with the current frame
//! result for temporal stability.
//!
//! Algorithm per frame:
//! 1. Reproject previous volumetric buffer using motion vectors.
//! 2. Validate (depth/transmittance consistency, bounds check).
//! 3. Valid: `result = lerp(current, history, 0.9)` — 90% history weight.
//! 4. Invalid: `result = current` — disocclusion detected, fall back to current.
//!
//! The pass owns two ping-pong [`wgpu::TextureFormat::Rgba16Float`] textures at
//! half the internal render resolution. Each [`VolTemporalPass::dispatch`] call
//! reads from one and writes to the other, then swaps indices.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------- Public constants ----------

/// Default temporal blend factor — 90% history, 10% current.
pub const DEFAULT_VOL_TEMPORAL_BLEND: f32 = 0.9;

/// Default depth threshold for reprojection validity (transmittance delta).
pub const DEFAULT_VOL_DEPTH_THRESHOLD: f32 = 0.3;

// ---------- GPU struct ----------

/// GPU-uploadable temporal reprojection parameters (32 bytes).
///
/// Memory layout:
/// ```text
/// offset  0 — width            u32   (4 bytes)
/// offset  4 — height           u32   (4 bytes)
/// offset  8 — blend_factor     f32   (4 bytes)
/// offset 12 — depth_threshold  f32   (4 bytes)
/// offset 16 — frame_index      u32   (4 bytes)
/// offset 20 — _pad0            u32   (4 bytes)
/// offset 24 — _pad1            u32   (4 bytes)
/// offset 28 — _pad2            u32   (4 bytes)
/// total: 32 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct VolTemporalParams {
    /// Half-resolution width in pixels.
    pub width: u32,
    /// Half-resolution height in pixels.
    pub height: u32,
    /// History blend weight in [0, 1]. 0.9 = 90% history.
    pub blend_factor: f32,
    /// Maximum transmittance delta before rejecting reprojection.
    pub depth_threshold: f32,
    /// Current frame index (monotonically increasing).
    pub frame_index: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
    /// Padding.
    pub _pad2: u32,
}

// ---------- Pass ----------

/// Volumetric temporal reprojection compute pass.
///
/// Owns two ping-pong scatter textures for history accumulation. Each
/// [`dispatch`](VolTemporalPass::dispatch) reads from the previous frame's
/// texture and writes the blended result to the current frame's texture.
#[allow(dead_code)]
pub struct VolTemporalPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pre-built bind groups for both ping-pong directions.
    ///
    /// `bind_groups[i]` reads from `scatter_textures[1 - i]` (history) and
    /// writes to `scatter_textures[i]` (output).
    bind_groups: [wgpu::BindGroup; 2],
    params_buffer: wgpu::Buffer,
    scatter_textures: [wgpu::Texture; 2],
    scatter_views: [wgpu::TextureView; 2],
    /// Index of the texture we will write to on the next dispatch.
    current_idx: usize,
    width: u32,
    height: u32,
}

impl VolTemporalPass {
    /// Create a new temporal reprojection pass.
    ///
    /// # Parameters
    /// - `device` — wgpu device.
    /// - `current_scatter_view` — view of the vol_march output texture
    ///   (external; used as the "current frame" input for *both* bind groups
    ///   since vol_march always writes to the same texture each frame).
    /// - `motion_vector_view` — view of the G-buffer motion vector texture
    ///   (`Rg32Float`, full internal resolution).
    /// - `width`, `height` — half-resolution dimensions.
    pub fn new(
        device: &wgpu::Device,
        current_scatter_view: &wgpu::TextureView,
        motion_vector_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // ---- Shader ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vol_temporal"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vol_temporal.wgsl").into(),
            ),
        });

        // ---- Params buffer ----
        let params = VolTemporalParams {
            width,
            height,
            blend_factor: DEFAULT_VOL_TEMPORAL_BLEND,
            depth_threshold: DEFAULT_VOL_DEPTH_THRESHOLD,
            frame_index: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol_temporal_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Ping-pong scatter textures ----
        let make_scatter = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let scatter_textures = [
            make_scatter("vol_temporal_scatter_0"),
            make_scatter("vol_temporal_scatter_1"),
        ];

        let scatter_views = [
            scatter_textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            scatter_textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];

        // ---- Bind group layout ----
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol_temporal_bgl"),
            entries: &[
                // 0: uniform params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: current scatter (texture_2d<f32>, non-filterable)
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
                // 2: history scatter (texture_2d<f32>, non-filterable)
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
                // 3: motion vectors (texture_2d<f32>, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: output scatter (storage texture, write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ---- Pipeline ----
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol_temporal_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol_temporal_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ---- Bind groups (pre-built for both ping-pong directions) ----
        //
        // bind_groups[0]: history = scatter_views[1], output = scatter_views[0]
        // bind_groups[1]: history = scatter_views[0], output = scatter_views[1]
        let make_bind_group = |history_view: &wgpu::TextureView,
                                output_view: &wgpu::TextureView,
                                label: &str|
         -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(current_scatter_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(history_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(motion_vector_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(output_view),
                    },
                ],
            })
        };

        let bind_groups = [
            make_bind_group(&scatter_views[1], &scatter_views[0], "vol_temporal_bg_0"),
            make_bind_group(&scatter_views[0], &scatter_views[1], "vol_temporal_bg_1"),
        ];

        Self {
            pipeline,
            bind_group_layout,
            bind_groups,
            params_buffer,
            scatter_textures,
            scatter_views,
            current_idx: 0,
            width,
            height,
        }
    }

    /// Dispatch temporal reprojection for the current frame.
    ///
    /// Uses `bind_groups[current_idx]`, then swaps `current_idx`.
    pub fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        frame_index: u32,
    ) {
        // Update frame_index in the params buffer
        queue.write_buffer(
            &self.params_buffer,
            std::mem::offset_of!(VolTemporalParams, frame_index) as u64,
            bytemuck::bytes_of(&frame_index),
        );

        let wg = 8u32;
        let dispatch_x = self.width.div_ceil(wg);
        let dispatch_y = self.height.div_ceil(wg);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vol_temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.current_idx], &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Swap ping-pong index
        self.current_idx = 1 - self.current_idx;
    }

    /// Get a view of the latest blended result.
    ///
    /// After [`dispatch`](VolTemporalPass::dispatch) swaps the index, the
    /// result lives in `scatter_views[1 - current_idx]`.
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.scatter_views[1 - self.current_idx]
    }

    /// Override the temporal blend factor at runtime.
    ///
    /// `factor` is clamped to [0, 1] by the caller — no GPU-side clamping.
    pub fn set_blend_factor(&self, queue: &wgpu::Queue, factor: f32) {
        queue.write_buffer(
            &self.params_buffer,
            std::mem::offset_of!(VolTemporalParams, blend_factor) as u64,
            bytemuck::bytes_of(&factor),
        );
    }

    /// Override the depth threshold at runtime.
    pub fn set_depth_threshold(&self, queue: &wgpu::Queue, threshold: f32) {
        queue.write_buffer(
            &self.params_buffer,
            std::mem::offset_of!(VolTemporalParams, depth_threshold) as u64,
            bytemuck::bytes_of(&threshold),
        );
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_temporal_params_size_is_32() {
        assert_eq!(std::mem::size_of::<VolTemporalParams>(), 32);
    }

    #[test]
    fn vol_temporal_params_pod_roundtrip() {
        let p = VolTemporalParams {
            width: 480,
            height: 270,
            blend_factor: 0.9,
            depth_threshold: 0.3,
            frame_index: 42,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
        let p2: &VolTemporalParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.blend_factor, p2.blend_factor);
        assert_eq!(p.frame_index, p2.frame_index);
    }

    #[test]
    fn vol_temporal_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(VolTemporalParams, width), 0);
        assert_eq!(std::mem::offset_of!(VolTemporalParams, height), 4);
        assert_eq!(std::mem::offset_of!(VolTemporalParams, blend_factor), 8);
        assert_eq!(std::mem::offset_of!(VolTemporalParams, depth_threshold), 12);
        assert_eq!(std::mem::offset_of!(VolTemporalParams, frame_index), 16);
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_VOL_TEMPORAL_BLEND, 0.9);
        assert_eq!(DEFAULT_VOL_DEPTH_THRESHOLD, 0.3);
    }

    #[test]
    fn ping_pong_index_alternates() {
        let mut idx: usize = 0;
        for _ in 0..4 {
            let write_to = idx;
            let read_from = 1 - idx;
            assert_ne!(write_to, read_from);
            idx = 1 - idx;
        }
        assert_eq!(idx, 0); // back to start after even number of swaps
    }
}
