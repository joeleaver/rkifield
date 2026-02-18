//! Double-buffered GPU uniform buffers.
//!
//! Provides [`DoubleBuffer`] and [`DoubleBufferSet`] for staging per-frame
//! uniform data without CPU/GPU stalls. While the GPU reads from the "read"
//! buffer, the CPU writes next-frame data into the "write" buffer. Call
//! [`DoubleBuffer::swap`] (or [`DoubleBufferSet::swap_all`]) after
//! `queue.submit()` to flip the roles.

/// Double-buffered GPU uniform storage.
///
/// Holds two copies of a uniform buffer. While the GPU reads from one (the
/// "read" buffer), the CPU writes the next frame's data to the other (the
/// "write" buffer). Call [`swap`](Self::swap) after `queue.submit()` to flip.
pub struct DoubleBuffer {
    buffers: [wgpu::Buffer; 2],
    /// Current write index (0 or 1). The read index is `1 - current`.
    current: usize,
    /// Size in bytes of each buffer.
    size: u64,
    /// Debug label.
    label: String,
}

impl DoubleBuffer {
    /// Create a new double buffer pair with the given byte size.
    pub fn new(device: &wgpu::Device, size: u64, label: &str) -> Self {
        let buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label} [0]")),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label} [1]")),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        ];
        Self {
            buffers,
            current: 0,
            size,
            label: label.to_string(),
        }
    }

    /// Write data to the current "write" buffer.
    ///
    /// Panics if `data.len()` exceeds the buffer size.
    pub fn write(&self, queue: &wgpu::Queue, data: &[u8]) {
        assert!(
            data.len() as u64 <= self.size,
            "data ({} bytes) exceeds buffer size ({} bytes)",
            data.len(),
            self.size
        );
        queue.write_buffer(&self.buffers[self.current], 0, data);
    }

    /// Write data at an offset in the current "write" buffer.
    ///
    /// Panics if `offset + data.len()` exceeds the buffer size.
    pub fn write_at(&self, queue: &wgpu::Queue, offset: u64, data: &[u8]) {
        assert!(
            offset + data.len() as u64 <= self.size,
            "write at offset {} with {} bytes exceeds buffer size {}",
            offset,
            data.len(),
            self.size
        );
        queue.write_buffer(&self.buffers[self.current], offset, data);
    }

    /// Get the "read" buffer (the one the GPU should read from this frame).
    ///
    /// This is the buffer that was written to *last* frame.
    pub fn read_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[1 - self.current]
    }

    /// Get the "write" buffer (the one the CPU writes to this frame).
    pub fn write_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[self.current]
    }

    /// Swap read/write roles after `queue.submit()`.
    ///
    /// Must be called once per frame, after submitting GPU work, so the buffer
    /// the GPU just read becomes the new write target.
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    /// Get the current write index (0 or 1).
    pub fn write_index(&self) -> usize {
        self.current
    }

    /// Get the current read index (0 or 1).
    pub fn read_index(&self) -> usize {
        1 - self.current
    }

    /// Get the byte size of each buffer.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the debug label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// A set of double-buffered uniforms for the main per-frame data.
///
/// Bundles camera, light settings, and engine settings uniform buffers into a
/// single swap-able unit. Call [`swap_all`](Self::swap_all) after
/// `queue.submit()` to flip all three buffers simultaneously.
pub struct DoubleBufferSet {
    /// Camera uniform double buffer (112 bytes — `CameraUniforms`).
    pub camera: DoubleBuffer,
    /// Light settings double buffer (configurable size).
    pub light_settings: DoubleBuffer,
    /// Engine settings double buffer (configurable size).
    pub engine_settings: DoubleBuffer,
}

impl DoubleBufferSet {
    /// Create a new set with the given buffer sizes in bytes.
    pub fn new(
        device: &wgpu::Device,
        camera_size: u64,
        light_settings_size: u64,
        engine_settings_size: u64,
    ) -> Self {
        Self {
            camera: DoubleBuffer::new(device, camera_size, "camera uniforms"),
            light_settings: DoubleBuffer::new(device, light_settings_size, "light settings"),
            engine_settings: DoubleBuffer::new(device, engine_settings_size, "engine settings"),
        }
    }

    /// Swap all buffers after `queue.submit()`.
    ///
    /// Must be called once per frame, in sync with each individual buffer's
    /// usage pattern.
    pub fn swap_all(&mut self) {
        self.camera.swap();
        self.light_settings.swap();
        self.engine_settings.swap();
    }
}

#[cfg(test)]
mod tests {
    // Tests for index-swapping logic that do not require a GPU device.

    /// Index state mirrors DoubleBuffer.current to let us test logic without wgpu.
    struct IndexTracker {
        current: usize,
    }

    impl IndexTracker {
        fn new() -> Self {
            Self { current: 0 }
        }
        fn write_index(&self) -> usize {
            self.current
        }
        fn read_index(&self) -> usize {
            1 - self.current
        }
        fn swap(&mut self) {
            self.current = 1 - self.current;
        }
    }

    #[test]
    fn initial_indices() {
        let t = IndexTracker::new();
        assert_eq!(t.write_index(), 0);
        assert_eq!(t.read_index(), 1);
    }

    #[test]
    fn swap_alternates_indices() {
        let mut t = IndexTracker::new();
        // Start: write=0, read=1
        assert_eq!(t.write_index(), 0);
        assert_eq!(t.read_index(), 1);

        t.swap();
        // After 1 swap: write=1, read=0
        assert_eq!(t.write_index(), 1);
        assert_eq!(t.read_index(), 0);

        t.swap();
        // After 2 swaps: back to write=0, read=1
        assert_eq!(t.write_index(), 0);
        assert_eq!(t.read_index(), 1);
    }

    #[test]
    fn double_swap_returns_to_start() {
        let mut t = IndexTracker::new();
        let w0 = t.write_index();
        let r0 = t.read_index();
        t.swap();
        t.swap();
        assert_eq!(t.write_index(), w0);
        assert_eq!(t.read_index(), r0);
    }

    #[test]
    fn write_and_read_are_always_different() {
        let mut t = IndexTracker::new();
        for _ in 0..10 {
            assert_ne!(t.write_index(), t.read_index());
            t.swap();
        }
    }

    #[test]
    fn indices_are_always_zero_or_one() {
        let mut t = IndexTracker::new();
        for _ in 0..8 {
            assert!(t.write_index() <= 1);
            assert!(t.read_index() <= 1);
            t.swap();
        }
    }

    /// Verify that three independent trackers stay in sync when swapped together.
    #[test]
    fn triple_swap_stays_in_sync() {
        let mut a = IndexTracker::new();
        let mut b = IndexTracker::new();
        let mut c = IndexTracker::new();

        for _ in 0..6 {
            assert_eq!(a.write_index(), b.write_index());
            assert_eq!(b.write_index(), c.write_index());
            a.swap();
            b.swap();
            c.swap();
        }
    }
}
