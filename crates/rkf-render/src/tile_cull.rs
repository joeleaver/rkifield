//! Tiled light culling compute pass.
//!
//! Divides the screen into 16×16 pixel tiles. For each tile, computes
//! min/max depth from the G-buffer position texture, then tests each light
//! against the tile's depth range. Writes per-tile light index lists that
//! the shading pass reads to evaluate only relevant lights.

use crate::gbuffer::GBuffer;
use crate::light::{LightBuffer, MAX_LIGHTS_PER_TILE, TILE_SIZE};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-uploadable cull uniforms (64 bytes).
///
/// Must match the `CullUniforms` struct in `tile_cull.wgsl`.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CullUniforms {
    /// Number of tiles horizontally.
    pub num_tiles_x: u32,
    /// Number of tiles vertically.
    pub num_tiles_y: u32,
    /// Number of active lights.
    pub num_lights: u32,
    /// Screen width in pixels.
    pub screen_width: u32,
    /// Screen height in pixels.
    pub screen_height: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
    /// Padding.
    pub _pad2: u32,
    /// Camera world-space position (xyz, w unused).
    pub camera_pos: [f32; 4],
    /// Camera forward direction (xyz, w unused).
    pub camera_forward: [f32; 4],
}

/// Tiled light culling compute pass.
#[allow(dead_code)]
pub struct TileCullPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Cull uniforms buffer.
    cull_uniforms_buffer: wgpu::Buffer,
    /// Per-tile light index lists (num_tiles * MAX_LIGHTS_PER_TILE u32s).
    pub tile_light_indices_buffer: wgpu::Buffer,
    /// Per-tile light count (num_tiles u32s).
    pub tile_light_counts_buffer: wgpu::Buffer,
    /// Bind group layout for group 1 (lights + cull uniforms).
    lights_cull_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for group 1.
    lights_cull_bind_group: wgpu::BindGroup,
    /// Bind group layout for group 2 (tile output, read_write).
    tile_output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for group 2.
    tile_output_bind_group: wgpu::BindGroup,
    /// Bind group layout for the shade pass to read light/tile data (read-only).
    pub shade_light_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the shade pass.
    pub shade_light_bind_group: wgpu::BindGroup,
    /// Number of tiles horizontally.
    pub num_tiles_x: u32,
    /// Number of tiles vertically.
    pub num_tiles_y: u32,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl TileCullPass {
    /// Create the tiled light culling pass.
    pub fn new(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        light_buffer: &LightBuffer,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tile_cull.wgsl").into()),
        });

        let num_tiles_x = width.div_ceil(TILE_SIZE);
        let num_tiles_y = height.div_ceil(TILE_SIZE);
        let num_tiles = num_tiles_x * num_tiles_y;

        // Tile output buffers
        let tile_light_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile light indices"),
            size: (num_tiles * MAX_LIGHTS_PER_TILE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tile_light_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile light counts"),
            size: (num_tiles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Cull uniforms
        let cull_uniforms = CullUniforms {
            num_tiles_x,
            num_tiles_y,
            num_lights: light_buffer.count,
            screen_width: width,
            screen_height: height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_pos: [0.0; 4],
            camera_forward: [0.0; 4],
        };

        let cull_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cull uniforms"),
                contents: bytemuck::bytes_of(&cull_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Group 1: lights (storage, read) + cull uniforms
        let lights_cull_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile cull lights+uniforms layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        let lights_cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile cull lights+uniforms"),
            layout: &lights_cull_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cull_uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // Group 2: tile output (read_write storage)
        let tile_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile output layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let tile_output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile output"),
            layout: &tile_output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_light_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_light_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Shade pass read-only bind group for light/tile data (group 5 in shade pipeline)
        let shade_light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade light data layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shade_light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade light data"),
            layout: &shade_light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_light_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_light_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layout: group 0 = G-buffer read, group 1 = lights+cull, group 2 = tile output
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile cull pipeline layout"),
            bind_group_layouts: &[
                &gbuffer.read_bind_group_layout,
                &lights_cull_bind_group_layout,
                &tile_output_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile cull pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            cull_uniforms_buffer,
            tile_light_indices_buffer,
            tile_light_counts_buffer,
            lights_cull_bind_group_layout,
            lights_cull_bind_group,
            tile_output_bind_group_layout,
            tile_output_bind_group,
            shade_light_bind_group_layout,
            shade_light_bind_group,
            num_tiles_x,
            num_tiles_y,
            width,
            height,
        }
    }

    /// Update the cull uniforms (call each frame with current camera data).
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        num_lights: u32,
        camera_pos: [f32; 3],
        camera_forward: [f32; 3],
    ) {
        let uniforms = CullUniforms {
            num_tiles_x: self.num_tiles_x,
            num_tiles_y: self.num_tiles_y,
            num_lights,
            screen_width: self.width,
            screen_height: self.height,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            camera_forward: [camera_forward[0], camera_forward[1], camera_forward[2], 0.0],
        };
        queue.write_buffer(
            &self.cull_uniforms_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    /// Dispatch the tile culling compute shader.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, gbuffer: &GBuffer) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &gbuffer.read_bind_group, &[]);
        pass.set_bind_group(1, &self.lights_cull_bind_group, &[]);
        pass.set_bind_group(2, &self.tile_output_bind_group, &[]);

        // One workgroup per tile, each workgroup is 16×16 threads
        pass.dispatch_workgroups(self.num_tiles_x, self.num_tiles_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cull_uniforms_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<CullUniforms>(), 64);
    }

    #[test]
    fn cull_uniforms_pod_roundtrip() {
        let u = CullUniforms {
            num_tiles_x: 60,
            num_tiles_y: 34,
            num_lights: 5,
            screen_width: 960,
            screen_height: 540,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            camera_pos: [1.0, 2.0, 3.0, 0.0],
            camera_forward: [0.0, 0.0, -1.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 64);
        let u2: &CullUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.num_tiles_x, u2.num_tiles_x);
        assert_eq!(u.num_lights, u2.num_lights);
        assert_eq!(u.camera_pos, u2.camera_pos);
    }

    #[test]
    fn tile_count_calculation() {
        let nx = 960u32.div_ceil(16);
        let ny = 540u32.div_ceil(16);
        assert_eq!(nx, 60);
        assert_eq!(ny, 34); // 540/16 = 33.75 → ceil = 34
    }
}
