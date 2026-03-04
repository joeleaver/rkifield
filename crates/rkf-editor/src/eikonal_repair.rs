//! GPU eikonal PDE re-initialization for SDF repair.
//!
//! Implements the Sussman-Smereka-Osher (1994) level-set re-initialization:
//!
//!   ∂d/∂τ = S(d₀)(1 − |∇d|)
//!
//! This iteratively drives |∇d| → 1 everywhere while preserving zero-crossings,
//! correcting the distance magnitudes distorted by CSG/smin without changing the
//! surface geometry.
//!
//! Pattern matches `jfa_sdf.rs`: single bind group layout, per-dispatch uniform
//! buffers, synchronous readback.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// GPU uniform struct — must match WGSL Uniforms exactly (32 bytes)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EikonalUniforms {
    gw:          u32,
    gh:          u32,
    gd:          u32,
    parity:      u32,   // 0 = read A / write B; 1 = read B / write A
    voxel_size:  f32,
    dt:          f32,   // 0.45 * voxel_size
    narrow_band: f32,   // 6.0 * voxel_size
    _pad:        u32,
}

// ---------------------------------------------------------------------------
// EikonalRepairPass
// ---------------------------------------------------------------------------

/// GPU eikonal PDE pipeline for per-stroke SDF repair.
///
/// Create one instance per device; reuse across frames and sculpt strokes.
pub struct EikonalRepairPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl EikonalRepairPass {
    /// Create the eikonal pipeline. Compiles the WGSL shader once.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("eikonal_repair"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("eikonal_repair.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("eikonal_bgl"),
                entries: &[
                    // binding 0: uniforms
                    bgl_entry(0, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }),
                    // binding 1: dist_a (read_write)
                    bgl_entry(1, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }),
                    // binding 2: dist_b (read_write)
                    bgl_entry(2, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }),
                    // binding 3: d0 (read-only initial distances)
                    bgl_entry(3, wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }),
                ],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("eikonal_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("eikonal_step"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("eikonal_step"),
                compilation_options: Default::default(),
                cache: None,
            },
        );

        Self { pipeline, bind_group_layout }
    }

    /// Run the eikonal PDE re-initialization and return corrected distances.
    ///
    /// # Parameters
    /// - `distances`: initial signed distances (dense grid, length = gw*gh*gd).
    /// - `gw/gh/gd`: grid dimensions in voxels.
    /// - `voxel_size`: physical size of one voxel.
    /// - `iterations`: number of PDE iterations (typically 10).
    ///
    /// # Returns
    /// Corrected `Vec<f32>` with |∇d| ≈ 1 and preserved zero-crossings.
    pub fn repair(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        distances: &[f32],
        gw: usize,
        gh: usize,
        gd: usize,
        voxel_size: f32,
        iterations: u32,
    ) -> Vec<f32> {
        let total = gw * gh * gd;
        assert_eq!(distances.len(), total, "distance grid length mismatch");
        if total == 0 { return Vec::new(); }

        let total_bytes = (total * 4) as u64;
        let dt = 0.45 * voxel_size;
        let narrow_band = 6.0 * voxel_size;

        // ---- Create GPU buffers ----

        let d0_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eikonal_d0"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&d0_buf, 0, bytemuck::cast_slice(distances));

        let dist_a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eikonal_dist_a"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dist_a_buf, 0, bytemuck::cast_slice(distances));

        let dist_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eikonal_dist_b"),
            size: total_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eikonal_staging"),
            size: total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ---- Per-iteration uniform buffers ----

        let make_ubuf = |parity: u32| -> wgpu::Buffer {
            let uniforms = EikonalUniforms {
                gw: gw as u32,
                gh: gh as u32,
                gd: gd as u32,
                parity,
                voxel_size,
                dt,
                narrow_band,
                _pad: 0,
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("eikonal_u"),
                size: std::mem::size_of::<EikonalUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&uniforms));
            buf
        };

        let mut iter_ubufs: Vec<wgpu::Buffer> = Vec::with_capacity(iterations as usize);
        for i in 0..iterations {
            iter_ubufs.push(make_ubuf(i % 2));
        }

        // ---- Bind group helper ----

        let make_bg = |ubuf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("eikonal_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dist_a_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: dist_b_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: d0_buf.as_entire_binding() },
                ],
            })
        };

        let (wgx, wgy, wgz) = (
            gw.div_ceil(8) as u32,
            gh.div_ceil(8) as u32,
            gd.div_ceil(4) as u32,
        );

        // ---- Record compute passes ----

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("eikonal_encoder") },
        );

        for ubuf in &iter_ubufs {
            let bg = make_bg(ubuf);
            let mut cp = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("eikonal_step"),
                    timestamp_writes: None,
                },
            );
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(wgx, wgy, wgz);
        }

        // Copy final result to staging. After N iterations with alternating parity,
        // the result is in dist_a if N is even, dist_b if N is odd.
        let final_buf = if iterations % 2 == 0 { &dist_a_buf } else { &dist_b_buf };
        encoder.copy_buffer_to_buffer(final_buf, 0, &staging_buf, 0, total_bytes);

        queue.submit([encoder.finish()]);

        // ---- Synchronous readback ----

        let slice = staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        log::debug!(
            "eikonal_repair: {}×{}×{}, {} iterations, {} distances",
            gw, gh, gd, iterations, result.len(),
        );

        result
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
