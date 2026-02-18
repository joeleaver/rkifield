//! GPU-side data for the clipmap LOD system.
//!
//! Provides uniform buffer types ([`GpuClipmapLevel`], [`GpuClipmapUniforms`])
//! and the [`ClipmapGpuData`] manager that uploads all clipmap levels into
//! concatenated GPU storage buffers with a bind group ready for the ray marcher.

use bytemuck::{Pod, Zeroable};
use rkf_core::clipmap::{ClipmapConfig, ClipmapGridSet};
use wgpu::util::DeviceExt;

/// Maximum clipmap levels on the GPU (must match `MAX_CLIPMAP_LEVELS` in rkf-core).
pub const GPU_MAX_CLIPMAP_LEVELS: usize = 5;

/// Per-level LOD uniforms for the GPU ray marcher.
///
/// Layout (64 bytes, 4 × `[f32; 4]` or `[u32; 4]`):
/// - `params`:      `[voxel_size, brick_extent, radius, 0.0]`
/// - `grid_dims`:   `[dim_x, dim_y, dim_z, total_cells]`
/// - `grid_origin`: `[origin_x, origin_y, origin_z, 0.0]`
/// - `offsets`:     `[occupancy_offset, slot_offset, 0, 0]` — element offsets into combined buffers
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuClipmapLevel {
    /// `[voxel_size, brick_extent, radius, 0.0]`
    pub params: [f32; 4],
    /// `[dim_x, dim_y, dim_z, total_cells]`
    pub grid_dims: [u32; 4],
    /// `[origin_x, origin_y, origin_z, 0.0]`
    pub grid_origin: [f32; 4],
    /// `[occupancy_offset, slot_offset, 0, 0]` — element (u32) offsets into combined buffers
    pub offsets: [u32; 4],
}

/// Clipmap uniform buffer for the GPU.
///
/// Total size: 16 + 5 × 64 = **336 bytes**.
///
/// Layout:
/// - `num_levels` (4 bytes) + 12 bytes padding to reach 16-byte alignment
/// - `levels[5]` × 64 bytes each
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuClipmapUniforms {
    /// Number of active clipmap levels.
    pub num_levels: u32,
    /// Padding to 16-byte alignment.
    pub _pad: [u32; 3],
    /// Per-level parameters.
    pub levels: [GpuClipmapLevel; GPU_MAX_CLIPMAP_LEVELS],
}

/// GPU-side data for the clipmap LOD system.
///
/// Holds:
/// - Combined occupancy buffer (all levels concatenated)
/// - Combined slot buffer (all levels concatenated)
/// - Clipmap uniform buffer with per-level metadata
/// - Bind group for the ray marcher
pub struct ClipmapGpuData {
    /// Combined occupancy buffer (all levels).
    pub occupancy_buffer: wgpu::Buffer,
    /// Combined slot buffer (all levels).
    pub slot_buffer: wgpu::Buffer,
    /// Clipmap uniform buffer.
    pub uniform_buffer: wgpu::Buffer,
    /// Bind group layout for clipmap data.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group binding all clipmap resources.
    pub bind_group: wgpu::BindGroup,
    /// Local copy of uniforms (kept for incremental updates).
    pub uniforms: GpuClipmapUniforms,
}

impl ClipmapGpuData {
    /// Upload clipmap grid data to the GPU.
    ///
    /// Concatenates all levels' occupancy and slot data into combined buffers.
    /// Computes per-level element offsets stored in the uniform buffer.
    pub fn upload(
        device: &wgpu::Device,
        grid_set: &ClipmapGridSet,
        camera_pos: [f32; 3],
    ) -> Self {
        let config = grid_set.config();
        let mut uniforms = GpuClipmapUniforms {
            num_levels: config.num_levels() as u32,
            _pad: [0; 3],
            levels: [GpuClipmapLevel::zeroed(); GPU_MAX_CLIPMAP_LEVELS],
        };

        // Concatenate occupancy and slot data from all levels.
        let mut all_occupancy: Vec<u32> = Vec::new();
        let mut all_slots: Vec<u32> = Vec::new();

        for (i, level_config) in config.levels().iter().enumerate() {
            let grid = grid_set.grid(i);
            let dims = grid.dimensions();

            // Record element offsets before appending.
            let occ_offset = all_occupancy.len() as u32;
            let slot_offset = all_slots.len() as u32;

            all_occupancy.extend_from_slice(grid.occupancy_data());
            all_slots.extend_from_slice(grid.slot_data());

            // Compute grid origin centred on camera position.
            let half_extent = level_config.radius;
            let origin = [
                camera_pos[0] - half_extent,
                camera_pos[1] - half_extent,
                camera_pos[2] - half_extent,
            ];

            uniforms.levels[i] = GpuClipmapLevel {
                params: [
                    level_config.voxel_size,
                    level_config.brick_extent(),
                    level_config.radius,
                    0.0,
                ],
                grid_dims: [dims.x, dims.y, dims.z, grid.total_cells()],
                grid_origin: [origin[0], origin[1], origin[2], 0.0],
                offsets: [occ_offset, slot_offset, 0, 0],
            };
        }

        // Guard against zero-size buffers (invalid on some backends).
        if all_occupancy.is_empty() {
            all_occupancy.push(0);
        }
        if all_slots.is_empty() {
            all_slots.push(0xFFFF_FFFF);
        }

        let occupancy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clipmap occupancy"),
            contents: bytemuck::cast_slice(&all_occupancy),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let slot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clipmap slots"),
            contents: bytemuck::cast_slice(&all_slots),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clipmap uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("clipmap bind group layout"),
                entries: &[
                    // binding 0: combined occupancy (storage, read-only)
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
                    // binding 1: combined slots (storage, read-only)
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
                    // binding 2: clipmap uniforms
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clipmap bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            occupancy_buffer,
            slot_buffer,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            uniforms,
        }
    }

    /// Update grid origins in the uniform buffer when the camera moves.
    ///
    /// Re-centres each level's grid on `camera_pos` and writes the updated
    /// uniform buffer to the GPU queue.
    pub fn update_origins(
        &mut self,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        config: &ClipmapConfig,
    ) {
        for (i, level) in config.levels().iter().enumerate() {
            if i >= self.uniforms.num_levels as usize {
                break;
            }
            let half_extent = level.radius;
            self.uniforms.levels[i].grid_origin = [
                camera_pos[0] - half_extent,
                camera_pos[1] - half_extent,
                camera_pos[2] - half_extent,
                0.0,
            ];
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn gpu_clipmap_level_size() {
        // 4 × [f32;4] = 4 × 16 = 64 bytes
        assert_eq!(mem::size_of::<GpuClipmapLevel>(), 64);
    }

    #[test]
    fn gpu_clipmap_uniforms_size() {
        // 16 (header) + 5 × 64 (levels) = 336 bytes
        assert_eq!(mem::size_of::<GpuClipmapUniforms>(), 336);
    }

    #[test]
    fn gpu_clipmap_uniforms_pod_roundtrip() {
        let mut uniforms = GpuClipmapUniforms::zeroed();
        uniforms.num_levels = 3;
        uniforms.levels[0].params = [0.02, 0.16, 128.0, 0.0];
        uniforms.levels[0].grid_dims = [1600, 1600, 1600, 1600 * 1600 * 1600];
        uniforms.levels[0].grid_origin = [-128.0, -128.0, -128.0, 0.0];
        uniforms.levels[0].offsets = [0, 0, 0, 0];

        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 336);

        // Cast back and verify round-trip
        let recovered: &GpuClipmapUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.num_levels, 3);
        assert_eq!(recovered.levels[0].params, [0.02, 0.16, 128.0, 0.0]);
        assert_eq!(
            recovered.levels[0].grid_origin,
            [-128.0, -128.0, -128.0, 0.0]
        );
    }

    #[test]
    fn gpu_clipmap_level_is_pod() {
        // Verify Pod + Zeroable derive work (compile-time check via zeroed())
        let level = GpuClipmapLevel::zeroed();
        assert_eq!(level.params, [0.0; 4]);
        assert_eq!(level.grid_dims, [0; 4]);
        assert_eq!(level.offsets, [0; 4]);
    }

    #[test]
    fn gpu_max_clipmap_levels_matches_core() {
        assert_eq!(GPU_MAX_CLIPMAP_LEVELS, rkf_core::clipmap::MAX_CLIPMAP_LEVELS);
    }
}
