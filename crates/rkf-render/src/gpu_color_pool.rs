//! GPU color pool — uploads color brick data and companion slot mapping.
//!
//! The shading pass uses this to look up per-voxel color for voxelized objects.

use wgpu::util::DeviceExt;

/// GPU-resident color pool with bind group for the shading pass.
pub struct GpuColorPool {
    /// Storage buffer containing color brick data (array of packed u32 RGBA8 voxels).
    pub color_data_buffer: wgpu::Buffer,
    /// Storage buffer mapping SDF brick slot → color brick slot (or EMPTY_SLOT).
    pub companion_map_buffer: wgpu::Buffer,
    /// Bind group layout (2 storage buffers).
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group binding both buffers.
    pub bind_group: wgpu::BindGroup,
}

impl GpuColorPool {
    /// Upload color pool data and companion slot mapping to the GPU.
    ///
    /// `color_data` is the flat color voxel data (cast from `Pool<ColorBrick>::as_slice()`).
    /// `companion_slots` maps SDF brick slot index → color brick slot (EMPTY_SLOT = no color).
    pub fn upload(
        device: &wgpu::Device,
        color_data: &[u8],
        companion_slots: &[u32],
    ) -> Self {
        let color_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color pool data"),
            contents: color_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let companion_map_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color companion map"),
            contents: bytemuck::cast_slice(companion_slots),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = Self::create_bind_group_layout(device);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color pool bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: companion_map_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            color_data_buffer,
            companion_map_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Create an empty color pool (placeholder when no objects use per-voxel color).
    ///
    /// Both buffers contain a single u32 element to satisfy wgpu validation.
    pub fn empty(device: &wgpu::Device) -> Self {
        let empty_data: [u32; 1] = [0];
        let empty_map: [u32; 1] = [0xFFFFFFFF]; // EMPTY_SLOT
        Self::upload(device, bytemuck::cast_slice(&empty_data), &empty_map)
    }

    /// Write a single color brick's data to the GPU buffer.
    ///
    /// `color_slot` is the color brick index. `data` is 512 packed u32 ColorVoxels.
    pub fn write_color_brick(&self, queue: &wgpu::Queue, color_slot: u32, data: &[u32; 512]) {
        let offset = color_slot as u64 * 512 * 4;
        if offset + 2048 <= self.color_data_buffer.size() {
            queue.write_buffer(&self.color_data_buffer, offset, bytemuck::cast_slice(data));
        }
    }

    /// Write a single companion map entry.
    pub fn write_companion_entry(&self, queue: &wgpu::Queue, brick_slot: u32, color_slot: u32) {
        let offset = brick_slot as u64 * 4;
        if offset + 4 <= self.companion_map_buffer.size() {
            queue.write_buffer(&self.companion_map_buffer, offset, bytemuck::bytes_of(&color_slot));
        }
    }

    /// Get the current color data buffer capacity in number of color bricks.
    pub fn color_brick_capacity(&self) -> u32 {
        (self.color_data_buffer.size() / (512 * 4)) as u32
    }

    /// Get the current companion map capacity in number of entries.
    pub fn companion_map_capacity(&self) -> u32 {
        (self.companion_map_buffer.size() / 4) as u32
    }

    /// Create the bind group layout (2 read-only storage buffers).
    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("color pool layout"),
            entries: &[
                // binding 0: color brick data (array<u32>)
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
                // binding 1: companion slot map (array<u32>)
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
        })
    }
}

#[cfg(test)]
mod tests {
    use rkf_core::companion::ColorVoxel;

    #[test]
    fn color_voxel_gpu_packing() {
        let v = ColorVoxel::new(255, 128, 64, 200);
        // Shader unpacking: r = packed & 0xFF, g = (packed >> 8) & 0xFF, etc.
        assert_eq!(v.packed & 0xFF, 255);
        assert_eq!((v.packed >> 8) & 0xFF, 128);
        assert_eq!((v.packed >> 16) & 0xFF, 64);
        assert_eq!((v.packed >> 24) & 0xFF, 200);
    }

    #[test]
    fn color_voxel_white_is_identity() {
        let white = ColorVoxel::new(255, 255, 255, 255);
        // In multiply mode, white * albedo = albedo (identity)
        let r = (white.packed & 0xFF) as f32 / 255.0;
        let g = ((white.packed >> 8) & 0xFF) as f32 / 255.0;
        let b = ((white.packed >> 16) & 0xFF) as f32 / 255.0;
        assert!((r - 1.0).abs() < f32::EPSILON);
        assert!((g - 1.0).abs() < f32::EPSILON);
        assert!((b - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn zero_intensity_preserves_albedo() {
        // When intensity=0, mix(albedo, albedo*color, 0) = albedo
        let v = ColorVoxel::new(0, 0, 0, 0);
        let intensity = ((v.packed >> 24) & 0xFF) as f32 / 255.0;
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn companion_empty_slot_is_max_u32() {
        let empty: u32 = 0xFFFFFFFF;
        assert_eq!(empty, u32::MAX);
    }

    #[test]
    fn color_pool_data_alignment() {
        // ColorVoxel is 4 bytes, same as u32 — no alignment issues for GPU upload
        assert_eq!(std::mem::size_of::<ColorVoxel>(), 4);
        assert_eq!(std::mem::align_of::<ColorVoxel>(), 4);
    }
}
