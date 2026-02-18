//! Radiance mip generation pass.
//!
//! Downsamples radiance from Level N to Level N+1 using a 2×2×2 box filter.
//! Called 3 times per frame: L0→L1, L1→L2, L2→L3.
//! Radiance: average of 8 children. Opacity: max of 8 (conservative).

use wgpu::util::DeviceExt;

use crate::radiance_volume::{RadianceVolume, RADIANCE_DIM, RADIANCE_FORMAT};

/// Labels for the three mip generation compute passes.
static MIP_PASS_LABELS: [&str; 3] = [
    "radiance mip L0→L1",
    "radiance mip L1→L2",
    "radiance mip L2→L3",
];

/// Radiance mip generation pass.
///
/// Dispatches a 3-D compute shader three times per frame to downsample each
/// consecutive level pair of the radiance volume clipmap:
///
/// | Step | Source | Destination |
/// |------|--------|-------------|
/// | 0    | L0     | L1          |
/// | 1    | L1     | L2          |
/// | 2    | L2     | L3          |
///
/// Each dispatch uses a 4×4×4 workgroup over the 128³ destination volume,
/// reading a 2×2×2 neighbourhood from the source level (clamped at borders).
#[allow(dead_code)]
pub struct RadianceMipPass {
    /// The compute pipeline, shared across all three dispatches.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout used by all level-pair bind groups.
    bind_group_layout: wgpu::BindGroupLayout,
    /// One bind group per level transition: `[L0→L1, L1→L2, L2→L3]`.
    bind_groups: [wgpu::BindGroup; 3],
    /// Uniform buffer holding `mip_params`: `[dim, 0, 0, 0]` as `[u32; 4]`.
    params_buffer: wgpu::Buffer,
}

impl RadianceMipPass {
    /// Create the radiance mip generation pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `radiance_volume`: provides the per-level texture views for both
    ///   source reads and destination writes.
    pub fn new(device: &wgpu::Device, radiance_volume: &RadianceVolume) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radiance_mip.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/radiance_mip.wgsl").into(),
            ),
        });

        // --- Bind group layout ---
        //
        // binding 0: sampled 3-D texture (source level, read via textureLoad)
        // binding 1: write-only storage texture (destination level)
        // binding 2: uniform buffer (mip_params: vec4<u32>, x = dim)

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radiance mip layout"),
                entries: &[
                    // binding 0: src texture (non-filterable sampled 3-D)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: dst storage texture (write-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: RADIANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    // binding 2: mip_params uniform buffer
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

        // --- Params buffer: [dim, 0, 0, 0] ---

        let mip_params: [u32; 4] = [RADIANCE_DIM, 0, 0, 0];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radiance mip params"),
            contents: bytemuck::cast_slice(&mip_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // --- One bind group per level transition ---
        //
        // The same texture view is used as both a sampled texture (src) and a
        // storage texture (dst) in different bind groups, which is valid because
        // the textures were created with TEXTURE_BINDING | STORAGE_BINDING and
        // consecutive steps operate on distinct texture objects.

        let make_bind_group = |src: usize, dst: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("radiance mip L{src}→L{dst}")),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&radiance_volume.views[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&radiance_volume.views[dst]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            })
        };

        let bind_groups = [
            make_bind_group(0, 1),
            make_bind_group(1, 2),
            make_bind_group(2, 3),
        ];

        // --- Pipeline ---

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radiance mip pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radiance mip pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_groups,
            params_buffer,
        }
    }

    /// Run all three mip generation steps: L0→L1, L1→L2, L2→L3.
    ///
    /// Each step is a separate compute pass to ensure a pipeline barrier
    /// between the write to level N+1 and the subsequent read of N+1 as
    /// source for the next step.
    ///
    /// Dispatch: 32×32×32 workgroups of size 4×4×4 → covers the full 128³
    /// output volume.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = RADIANCE_DIM / 4;
        for i in 0..3 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(MIP_PASS_LABELS[i]),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[i], &[]);
            pass.dispatch_workgroups(wg, wg, wg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_params_size() {
        // params is [u32; 4] = 16 bytes
        assert_eq!(std::mem::size_of::<[u32; 4]>(), 16);
    }

    #[test]
    fn dispatch_count_covers_volume() {
        // 32 workgroups × 4 threads = 128 = RADIANCE_DIM
        let wg = RADIANCE_DIM / 4;
        assert_eq!(wg * 4, RADIANCE_DIM);
        assert_eq!(wg, 32);
    }
}
