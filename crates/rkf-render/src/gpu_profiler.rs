//! GPU timestamp profiler — measures per-pass GPU execution time.
//!
//! Uses wgpu timestamp queries to bracket each render pass. Results are
//! resolved and read back once per frame. Requires `Features::TIMESTAMP`.

/// Maximum number of timestamp pairs (one per pass).
const MAX_PASSES: usize = 32;
/// Total timestamps: 2 per pass (begin + end).
const MAX_TIMESTAMPS: usize = MAX_PASSES * 2;

/// GPU timestamp profiler for measuring per-pass execution time.
#[allow(missing_docs)]
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    /// Current write index (incremented by 2 per pass).
    next_index: u32,
    /// Pass names for the current frame, in order.
    pass_names: Vec<&'static str>,
    /// Results from the previous frame (name, ms).
    pub results: Vec<(&'static str, f64)>,
    /// Nanoseconds per timestamp tick (from adapter).
    timestamp_period: f32,
    /// Whether the device supports timestamps.
    pub enabled: bool,
}

impl GpuProfiler {
    /// Create a new GPU profiler.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let enabled = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profiler_queries"),
            ty: if enabled {
                wgpu::QueryType::Timestamp
            } else {
                // Fallback — won't be used but QuerySet must be created.
                wgpu::QueryType::Timestamp
            },
            count: MAX_TIMESTAMPS as u32,
        });

        let buf_size = (MAX_TIMESTAMPS * std::mem::size_of::<u64>()) as u64;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_resolve"),
            size: buf_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profiler_readback"),
            size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let timestamp_period = queue.get_timestamp_period();

        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            next_index: 0,
            pass_names: Vec::with_capacity(MAX_PASSES),
            results: Vec::new(),
            timestamp_period,
            enabled,
        }
    }

    /// Reset for a new frame. Call at the start of command encoding.
    pub fn begin_frame(&mut self) {
        self.next_index = 0;
        self.pass_names.clear();
    }

    /// Write a begin-timestamp before a pass.
    pub fn begin_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &'static str) {
        if !self.enabled || self.next_index as usize >= MAX_TIMESTAMPS - 1 {
            return;
        }
        encoder.write_timestamp(&self.query_set, self.next_index);
        self.pass_names.push(name);
        self.next_index += 1;
    }

    /// Write an end-timestamp after a pass.
    pub fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.next_index == 0 {
            return;
        }
        encoder.write_timestamp(&self.query_set, self.next_index);
        self.next_index += 1;
    }

    /// Resolve timestamps and copy to readback buffer. Call after all passes,
    /// before `queue.submit()`.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled || self.next_index == 0 {
            return;
        }
        encoder.resolve_query_set(
            &self.query_set,
            0..self.next_index,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            (self.next_index as u64) * std::mem::size_of::<u64>() as u64,
        );
    }

    /// Read back resolved timestamps. Call after `device.poll()` has
    /// completed (i.e. after the readback from the main frame).
    /// Updates `self.results` with (pass_name, duration_ms) pairs.
    pub fn read_results(&mut self, device: &wgpu::Device) {
        if !self.enabled || self.next_index == 0 {
            return;
        }

        let n_timestamps = self.next_index as usize;
        let byte_size = n_timestamps * std::mem::size_of::<u64>();
        let slice = self.readback_buffer.slice(..byte_size as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // The device was already polled by the caller (map_readback), so
        // GPU work is done. Poll to process the map_async callback.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        self.results.clear();
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let timestamps: &[u64] =
                bytemuck::cast_slice(&data);

            let period_ns = self.timestamp_period as f64;
            for (i, name) in self.pass_names.iter().enumerate() {
                let begin = timestamps[i * 2];
                let end = timestamps[i * 2 + 1];
                let ns = (end.wrapping_sub(begin)) as f64 * period_ns;
                self.results.push((name, ns / 1_000_000.0));
            }
            drop(data);
            self.readback_buffer.unmap();
        } else {
            self.readback_buffer.unmap();
        }
    }
}
