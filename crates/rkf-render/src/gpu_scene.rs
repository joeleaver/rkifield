//! GPU buffer upload for brick pool, sparse grid, and scene uniforms.
//!
//! [`GpuScene`] uploads CPU-side voxel data to GPU storage/uniform buffers
//! and creates the bind group layout + bind group used by the ray march shader.

use bytemuck::{Pod, Zeroable};
use rkf_core::brick_pool::BrickPool;
use rkf_core::sparse_grid::SparseGrid;
use wgpu::util::DeviceExt;

/// Scene-level uniforms describing the spatial grid layout (48 bytes, vec4-packed).
///
/// Uses `[u32; 4]` / `[f32; 4]` to match WGSL `vec4` alignment and avoid
/// `vec3` padding issues.
///
/// - `grid_dims`: `[x, y, z, 0]`
/// - `grid_origin`: `[x, y, z, brick_extent]`
/// - `params`: `[voxel_size, 0, 0, 0]`
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct SceneUniforms {
    /// Grid dimensions in cells per axis. `[x, y, z, unused]`.
    pub grid_dims: [u32; 4],
    /// Grid world-space origin + brick extent. `[ox, oy, oz, brick_extent]`.
    pub grid_origin: [f32; 4],
    /// `[voxel_size, 0, 0, 0]`.
    pub params: [f32; 4],
}

/// GPU-resident scene data: brick pool + sparse grid + scene uniforms.
///
/// Created from CPU-side data via [`GpuScene::upload`]. Provides the
/// bind group layout and bind group for the ray march shader.
pub struct GpuScene {
    /// Storage buffer containing all brick voxel data.
    pub brick_pool_buffer: wgpu::Buffer,
    /// Storage buffer containing the occupancy bitfield (Level 2).
    pub occupancy_buffer: wgpu::Buffer,
    /// Storage buffer containing the slot array (Level 1).
    pub slot_buffer: wgpu::Buffer,
    /// Uniform buffer for camera data.
    pub camera_buffer: wgpu::Buffer,
    /// Uniform buffer for scene/grid layout data.
    pub scene_buffer: wgpu::Buffer,
    /// Bind group layout shared with the ray march pipeline.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group binding all scene resources.
    pub bind_group: wgpu::BindGroup,
}

impl GpuScene {
    /// Upload CPU-side brick pool and sparse grid to the GPU.
    ///
    /// `camera_uniforms` should be the initial camera uniforms (updated each frame).
    pub fn upload(
        device: &wgpu::Device,
        pool: &BrickPool,
        grid: &SparseGrid,
        camera_data: &[u8],
        scene_uniforms: &SceneUniforms,
    ) -> Self {
        // Brick pool: each Brick is 4096 bytes (512 VoxelSamples * 8 bytes each).
        // Upload the entire pool backing slice as raw bytes.
        let brick_bytes: &[u8] = bytemuck::cast_slice(pool.as_slice());
        let brick_pool_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick pool"),
            contents: brick_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Occupancy bitfield (Level 2): array of u32, 2 bits per cell.
        let occupancy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occupancy"),
            contents: bytemuck::cast_slice(grid.occupancy_data()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Slot array (Level 1): array of u32, one per cell.
        let slot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slots"),
            contents: bytemuck::cast_slice(grid.slot_data()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Camera uniforms — updated each frame via queue.write_buffer.
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniforms"),
            contents: camera_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Scene uniforms — static for a given grid configuration.
        let scene_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene uniforms"),
            contents: bytemuck::bytes_of(scene_uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = Self::create_bind_group_layout(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brick_pool_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slot_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            brick_pool_buffer,
            occupancy_buffer,
            slot_buffer,
            camera_buffer,
            scene_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Write only the dirty bricks to the GPU brick pool buffer.
    ///
    /// Each brick is 4096 bytes (512 voxels * 8 bytes). Only the specified
    /// slots are uploaded, avoiding a full buffer rewrite.
    pub fn write_dirty_bricks(&self, queue: &wgpu::Queue, pool: &BrickPool, dirty_slots: &[u32]) {
        let brick_bytes: &[u8] = bytemuck::cast_slice(pool.as_slice());
        let brick_size = 4096u64; // 512 VoxelSamples * 8 bytes
        for &slot in dirty_slots {
            let offset = slot as u64 * brick_size;
            let start = offset as usize;
            let end = start + brick_size as usize;
            if end <= brick_bytes.len() {
                queue.write_buffer(&self.brick_pool_buffer, offset, &brick_bytes[start..end]);
            }
        }
    }

    /// Write only the dirty occupancy words to the GPU occupancy buffer.
    ///
    /// Each word is 4 bytes (u32), encoding 2 bits per cell for 16 cells per word.
    pub fn write_dirty_occupancy(&self, queue: &wgpu::Queue, grid: &SparseGrid, dirty_word_indices: &[u32]) {
        let occ_data = grid.occupancy_data();
        for &word_idx in dirty_word_indices {
            if (word_idx as usize) < occ_data.len() {
                let offset = word_idx as u64 * 4;
                queue.write_buffer(
                    &self.occupancy_buffer,
                    offset,
                    bytemuck::bytes_of(&occ_data[word_idx as usize]),
                );
            }
        }
    }

    /// Write only the dirty slot entries to the GPU slot buffer.
    ///
    /// Each slot entry is 4 bytes (u32), one per cell.
    pub fn write_dirty_slots(&self, queue: &wgpu::Queue, grid: &SparseGrid, dirty_cell_indices: &[u32]) {
        let slot_data = grid.slot_data();
        for &cell_idx in dirty_cell_indices {
            if (cell_idx as usize) < slot_data.len() {
                let offset = cell_idx as u64 * 4;
                queue.write_buffer(
                    &self.slot_buffer,
                    offset,
                    bytemuck::bytes_of(&slot_data[cell_idx as usize]),
                );
            }
        }
    }

    /// Create the bind group layout for scene data (bindings 0–4).
    ///
    /// This is separate so the ray march pipeline can reference the layout
    /// before the scene data exists.
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene bind group layout"),
            entries: &[
                // binding 0: brick pool (read-only storage)
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
                // binding 1: occupancy bitfield (read-only storage)
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
                // binding 2: slot array (read-only storage)
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
                // binding 3: camera uniforms
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
                // binding 4: scene uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_uniforms_size() {
        assert_eq!(std::mem::size_of::<SceneUniforms>(), 48);
    }

    #[test]
    fn scene_uniforms_pod_roundtrip() {
        let u = SceneUniforms {
            grid_dims: [10, 10, 10, 0],
            grid_origin: [-0.8, -0.8, -0.8, 0.16],
            params: [0.02, 0.0, 0.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 48);
        let u2: &SceneUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.grid_dims, u2.grid_dims);
        assert_eq!(u.grid_origin, u2.grid_origin);
        assert_eq!(u.params, u2.params);
    }
}
