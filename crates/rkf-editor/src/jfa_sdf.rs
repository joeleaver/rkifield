//! GPU Jump Flooding Algorithm (JFA) for SDF distance repair.
//!
//! After each sculpt stroke the signed-distance field stored in the voxel bricks
//! can drift: repeated `smin` applications distort distance magnitudes far from
//! the surface, causing normal artifacts at brick seams and stroke junctions.
//!
//! [`JfaSdfPass`] fixes this by running a 3D JFA on the GPU immediately after
//! the CPU sculpt writes.  It:
//!
//! 1. Receives a dense boolean solid/empty grid (built from the CPU brick pool).
//! 2. Uploads that grid to a GPU storage buffer.
//! 3. Dispatches `jfa_init → jfa_pass × ceil(log2(N)) → jfa_writeback`.
//! 4. Reads back a dense `f32` distance grid from the GPU.
//! 5. Returns the repaired distances to the caller, which writes them back into
//!    `cpu_brick_pool` (preserving material/blend/flag metadata).
//!
//! The readback is synchronous (blocking `device.poll`).  For typical sculpted
//! objects (≤ 128³ voxels ≈ 8 MB) this adds < 2 ms latency.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// GPU uniform struct — must match WGSL JfaUniforms exactly (32 bytes)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JfaUniforms {
    gw:         u32,
    gh:         u32,
    gd:         u32,
    step:       u32,   // bit 31 = parity; bits 0-30 = step size
    voxel_size: f32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

// ---------------------------------------------------------------------------
// JfaSdfPass
// ---------------------------------------------------------------------------

/// GPU JFA pipeline for per-object SDF repair.
///
/// Create one instance per device; reuse across frames and sculpt strokes.
pub struct JfaSdfPass {
    init_pipeline:      wgpu::ComputePipeline,
    pass_pipeline:      wgpu::ComputePipeline,
    writeback_pipeline: wgpu::ComputePipeline,
    bind_group_layout:  wgpu::BindGroupLayout,
}

impl JfaSdfPass {
    /// Create the JFA pipelines.  Compiles the WGSL shader once.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("jfa_sdf"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("jfa_sdf.wgsl").into(),
            ),
        });

        // All three entry points share one bind group layout.
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("jfa_sdf_bgl"),
                entries: &[
                    // binding 0: uniforms (dynamic — swapped per dispatch)
                    bgl_entry(0, wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    }),
                    // binding 1: solid (read-only)
                    bgl_entry(1, wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    }),
                    // binding 2: seeds_a (read_write)
                    bgl_entry(2, wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    }),
                    // binding 3: seeds_b (read_write)
                    bgl_entry(3, wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    }),
                    // binding 4: dist_out (read_write)
                    bgl_entry(4, wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    }),
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:                Some("jfa_sdf_layout"),
                bind_group_layouts:   &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        let make_pipeline = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:               Some(entry),
                layout:              Some(&pipeline_layout),
                module:              &shader,
                entry_point:         Some(entry),
                compilation_options: Default::default(),
                cache:               None,
            })
        };

        Self {
            init_pipeline:      make_pipeline("jfa_init"),
            pass_pipeline:      make_pipeline("jfa_pass"),
            writeback_pipeline: make_pipeline("jfa_writeback"),
            bind_group_layout,
        }
    }

    /// Run the full JFA repair pipeline and return a dense grid of signed distances.
    ///
    /// # Parameters
    /// - `solid`:      flat dense boolean grid (true = solid voxel).
    ///                 Length must be `gw * gh * gd`.
    /// - `gw/gh/gd`:  grid dimensions in voxels.
    /// - `voxel_size`: physical size of one voxel (metres).
    ///
    /// # Returns
    /// A `Vec<f32>` of length `gw * gh * gd` with signed Euclidean distances.
    /// Exterior voxels are positive, interior voxels are negative.
    ///
    /// Returns `None` if the grid is empty.
    pub fn repair(
        &self,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        solid:      &[bool],
        gw:         usize,
        gh:         usize,
        gd:         usize,
        voxel_size: f32,
    ) -> Option<Vec<f32>> {
        let total = gw * gh * gd;
        if total == 0 { return None; }
        assert_eq!(solid.len(), total, "solid grid length mismatch");

        let total_u64 = total as u64;

        // ---- Upload solid grid ----

        let solid_u32: Vec<u32> = solid.iter().map(|&s| s as u32).collect();
        let solid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("jfa_solid"),
            size:               total_u64 * 4,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&solid_buf, 0, bytemuck::cast_slice(&solid_u32));

        // ---- Seed ping-pong buffers ----

        let make_seed_buf = |label| device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some(label),
            size:               total_u64 * 4,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seeds_a_buf = make_seed_buf("jfa_seeds_a");
        let seeds_b_buf = make_seed_buf("jfa_seeds_b");

        // ---- Distance output ----

        let dist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("jfa_dist_out"),
            size:               total_u64 * 4,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ---- Staging for CPU readback ----

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("jfa_staging"),
            size:               total_u64 * 4,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Number of JFA passes ----

        let max_dim = gw.max(gh).max(gd);
        let num_passes = if max_dim <= 1 { 0 } else {
            usize::BITS as usize - (max_dim - 1).leading_zeros() as usize
        };

        // ---- Compute all per-dispatch uniforms up front ----
        //
        // Each dispatch needs its own GPU buffer because they're all recorded
        // into one encoder and submitted together — we can't reuse one buffer.

        let make_ubuf = |u: JfaUniforms| -> wgpu::Buffer {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("jfa_u"),
                size:               std::mem::size_of::<JfaUniforms>() as u64,
                usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
            buf
        };

        // Determine step sequence and parity for each pass.
        // parity 0 → jfa_pass reads seeds_a, writes seeds_b.
        // parity 1 → jfa_pass reads seeds_b, writes seeds_a.
        // After init, seeds_a has the seeds (parity = 0 for first pass).
        let mut pass_ubufs: Vec<wgpu::Buffer> = Vec::with_capacity(num_passes);
        let mut parity: u32 = 0;
        for k in 0..num_passes {
            let step_size = (max_dim >> (k + 1)).max(1) as u32;
            pass_ubufs.push(make_ubuf(JfaUniforms {
                gw: gw as u32, gh: gh as u32, gd: gd as u32,
                step: step_size | (parity << 31),
                voxel_size, _pad0: 0, _pad1: 0, _pad2: 0,
            }));
            parity ^= 1;
        }

        let init_ubuf = make_ubuf(JfaUniforms {
            gw: gw as u32, gh: gh as u32, gd: gd as u32,
            step: 0, voxel_size, _pad0: 0, _pad1: 0, _pad2: 0,
        });
        // For writeback, parity tells us which buffer holds the final seeds.
        let wb_ubuf = make_ubuf(JfaUniforms {
            gw: gw as u32, gh: gh as u32, gd: gd as u32,
            step: parity << 31,   // bits 0-30 unused in writeback
            voxel_size, _pad0: 0, _pad1: 0, _pad2: 0,
        });

        // ---- Helper: make a bind group for this dispatch's uniform buffer ----

        let make_bg = |ubuf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("jfa_bg"),
                layout:  &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: solid_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: seeds_a_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: seeds_b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: dist_buf.as_entire_binding() },
                ],
            })
        };

        let (wgx, wgy, wgz) = (
            gw.div_ceil(8) as u32,
            gh.div_ceil(8) as u32,
            gd.div_ceil(4) as u32,
        );

        // ---- Record all compute passes into one encoder ----

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("jfa_encoder") },
        );

        // jfa_init — seeds seeds_a
        {
            let bg = make_bg(&init_ubuf);
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("jfa_init"), timestamp_writes: None },
            );
            cp.set_pipeline(&self.init_pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(wgx, wgy, wgz);
        }

        // jfa_pass × num_passes
        for ubuf in &pass_ubufs {
            let bg = make_bg(ubuf);
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("jfa_pass"), timestamp_writes: None },
            );
            cp.set_pipeline(&self.pass_pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(wgx, wgy, wgz);
        }

        // jfa_writeback — writes to dist_buf
        {
            let bg = make_bg(&wb_ubuf);
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("jfa_writeback"),
                    timestamp_writes: None,
                },
            );
            cp.set_pipeline(&self.writeback_pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(wgx, wgy, wgz);
        }

        // Copy dist_buf → staging for CPU readback
        encoder.copy_buffer_to_buffer(&dist_buf, 0, &staging_buf, 0, total_u64 * 4);

        queue.submit([encoder.finish()]);

        // ---- Synchronous readback ----

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let data = slice.get_mapped_range();
        let distances: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        log::debug!(
            "jfa_sdf::repair: {}×{}×{} voxels, {} JFA passes, {} distances",
            gw, gh, gd, num_passes, distances.len(),
        );

        Some(distances)
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn bgl_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}
