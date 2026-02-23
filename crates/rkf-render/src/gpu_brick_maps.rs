//! GPU brick maps buffer — uploads packed brick map entries to a storage buffer.
//!
//! All per-object brick maps are packed into a single `Vec<u32>` by
//! [`BrickMapAllocator`](rkf_core::BrickMapAllocator) on the CPU. This module
//! manages the corresponding wgpu storage buffer for GPU access.
//!
//! The ray march shader indexes into this buffer using:
//! ```text
//! map_index = object.brick_map_offset + flatten(brick_coord, object.brick_dims)
//! brick_slot = brick_maps[map_index]
//! ```

/// GPU brick maps storage buffer.
///
/// Holds a single wgpu buffer containing all packed brick map entries as `u32`s.
/// Updated via [`upload`](GpuBrickMaps::upload) (full replace) or
/// [`update_region`](GpuBrickMaps::update_region) (partial write).
pub struct GpuBrickMaps {
    /// The wgpu storage buffer.
    pub buffer: wgpu::Buffer,
    /// Current buffer capacity in u32 entries.
    capacity: u32,
    /// Current number of valid entries.
    len: u32,
}

impl GpuBrickMaps {
    /// Create a new GPU brick maps buffer with the given initial capacity.
    pub fn new(device: &wgpu::Device, initial_capacity: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_maps"),
            size: (initial_capacity as u64) * 4, // u32 = 4 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity: initial_capacity,
            len: 0,
        }
    }

    /// Upload the entire brick maps buffer from CPU data.
    ///
    /// If the data exceeds current capacity, a new buffer is created.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32]) {
        let needed = data.len() as u32;

        if needed > self.capacity {
            // Grow buffer (round up to power of 2 or 2x current).
            let new_cap = (needed).max(self.capacity * 2).max(1024);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brick_maps"),
                size: (new_cap as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_cap;
        }

        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        }
        self.len = needed;
    }

    /// Update a sub-region of the buffer without full re-upload.
    ///
    /// `offset` is in u32 entries (not bytes).
    pub fn update_region(&self, queue: &wgpu::Queue, offset: u32, data: &[u32]) {
        if !data.is_empty() {
            queue.write_buffer(
                &self.buffer,
                (offset as u64) * 4,
                bytemuck::cast_slice(data),
            );
        }
    }

    /// Current number of valid entries.
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether the buffer has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Current buffer capacity in u32 entries.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}
