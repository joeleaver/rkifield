//! Tile-based object culling compute pass — Phase 6.
//!
//! Projects object AABBs to screen space and builds per-tile object lists.
//! Each tile (16×16 pixels) gets a list of which objects potentially overlap it.
//! The ray marcher reads these lists to only evaluate relevant objects per pixel,
//! replacing brute-force or BVH evaluation of all objects.
//!
//! Dispatch: ceil(width/16) × ceil(height/16) × 1
//! Workgroup: 16×16 = 256 threads (cooperative object testing)

use crate::gpu_scene::GpuScene;

/// Tile size in pixels (must match the shader's TILE_SIZE constant).
pub const OBJECT_TILE_SIZE: u32 = 16;

/// Maximum objects per tile (must match the shader's MAX_OBJECTS_PER_TILE constant).
pub const MAX_OBJECTS_PER_TILE: u32 = 32;

/// Tile-based object culling pass.
///
/// Projects object AABBs to screen space and builds per-tile object index lists.
/// The ray marcher reads these lists to evaluate only objects that overlap each tile.
pub struct TileObjectCullPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Per-tile object index lists (num_tiles * MAX_OBJECTS_PER_TILE u32s).
    pub tile_object_indices_buffer: wgpu::Buffer,
    /// Per-tile object count (num_tiles u32s).
    pub tile_object_counts_buffer: wgpu::Buffer,
    /// Bind group layout for tile output (group 1, retained for resize).
    #[allow(dead_code)]
    tile_output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for tile output (group 1, read_write).
    tile_output_bind_group: wgpu::BindGroup,
    /// Read-only bind group layout for the ray marcher to consume tile data.
    pub read_bind_group_layout: wgpu::BindGroupLayout,
    /// Read-only bind group for the ray marcher.
    pub read_bind_group: wgpu::BindGroup,
    /// Number of tiles horizontally.
    pub num_tiles_x: u32,
    /// Number of tiles vertically.
    pub num_tiles_y: u32,
}

impl TileObjectCullPass {
    /// Create the tile object culling pass.
    ///
    /// `gpu_scene` provides the bind group layout for group 0 (objects, camera, scene).
    /// `width` and `height` are the internal render resolution.
    pub fn new(device: &wgpu::Device, gpu_scene: &GpuScene, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_object_cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tile_object_cull.wgsl").into(),
            ),
        });

        let num_tiles_x = width.div_ceil(OBJECT_TILE_SIZE);
        let num_tiles_y = height.div_ceil(OBJECT_TILE_SIZE);
        let num_tiles = num_tiles_x * num_tiles_y;

        // Tile output buffers.
        let tile_object_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_object_indices"),
            size: (num_tiles * MAX_OBJECTS_PER_TILE * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let tile_object_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_object_counts"),
            size: (num_tiles * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Group 1: tile output (read_write for the cull pass).
        let tile_output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile_object_cull output layout"),
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
            label: Some("tile_object_cull output"),
            layout: &tile_output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_object_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_object_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Read-only bind group for the ray marcher to consume tile data.
        let read_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tile_object_cull read layout"),
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
                ],
            });

        let read_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile_object_cull read"),
            layout: &read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_object_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_object_counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline: group 0 = GpuScene, group 1 = tile output (read_write).
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile_object_cull pipeline layout"),
            bind_group_layouts: &[&gpu_scene.bind_group_layout, &tile_output_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile_object_cull pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            tile_object_indices_buffer,
            tile_object_counts_buffer,
            tile_output_bind_group_layout,
            tile_output_bind_group,
            read_bind_group_layout,
            read_bind_group,
            num_tiles_x,
            num_tiles_y,
        }
    }

    /// Dispatch the tile object culling compute shader.
    ///
    /// Must be called after `gpu_scene.upload_objects()` and `gpu_scene.update_camera()`
    /// so that object AABBs and camera data are current.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, gpu_scene: &GpuScene) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_object_cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &gpu_scene.bind_group, &[]);
        pass.set_bind_group(1, &self.tile_output_bind_group, &[]);
        pass.dispatch_workgroups(self.num_tiles_x, self.num_tiles_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_count_calculation() {
        let nx = 960u32.div_ceil(OBJECT_TILE_SIZE);
        let ny = 540u32.div_ceil(OBJECT_TILE_SIZE);
        assert_eq!(nx, 60);
        assert_eq!(ny, 34);
        assert_eq!(nx * ny * MAX_OBJECTS_PER_TILE, 65_280);
    }

    #[test]
    fn constants_match_shader() {
        assert_eq!(OBJECT_TILE_SIZE, 16);
        assert_eq!(MAX_OBJECTS_PER_TILE, 32);
    }
}
