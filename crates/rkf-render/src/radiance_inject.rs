//! Radiance injection compute pass for GI.
//!
//! Iterates over every texel of the Level 0 radiance volume (128³) and writes
//! direct lighting into it.  The shader evaluates the SDF at each voxel centre,
//! samples up to `max_shadow_lights` shadow rays toward the nearest lights, and
//! accumulates radiance into the `Rgba16Float` write-only storage texture.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu_scene::GpuScene;
use crate::light::LightBuffer;
use crate::material_table::MaterialTable;
use crate::radiance_volume::{RadianceVolume, RADIANCE_DIM, RADIANCE_FORMAT};

/// Uniforms for the radiance injection pass (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct InjectUniforms {
    /// Total number of lights in the light buffer.
    pub num_lights: u32,
    /// Max lights that get shadow evaluation in injection (default: 1).
    pub max_shadow_lights: u32,
    /// Padding.
    pub _pad: [u32; 2],
}

/// Radiance injection compute pass.
///
/// Dispatches a 3-D compute shader over the Level 0 radiance volume to inject
/// direct-lighting contributions each frame.  Group layout:
///
/// | Group | Contents |
/// |-------|----------|
/// | 0     | Scene SDF data ([`GpuScene`]) |
/// | 1     | Material table ([`MaterialTable`]) |
/// | 2     | Lights storage buffer + [`InjectUniforms`] |
/// | 3     | Radiance L0 write view + radiance volume uniforms |
#[allow(dead_code)]
pub struct RadianceInjectPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for group 2 (lights + inject uniforms).
    light_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for group 2.
    light_bind_group: wgpu::BindGroup,
    /// Bind group layout for group 3 (radiance write + volume uniforms).
    write_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for group 3.
    write_bind_group: wgpu::BindGroup,
    /// Uniform buffer for [`InjectUniforms`].
    inject_uniforms_buffer: wgpu::Buffer,
}

impl RadianceInjectPass {
    /// Create the radiance injection pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `scene`: provides the group 0 bind group layout (SDF data).
    /// - `material_table`: provides the group 1 bind group layout.
    /// - `light_buffer`: lights storage buffer (group 2 binding 0).
    /// - `radiance_volume`: Level 0 write view and volume uniforms (group 3).
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuScene,
        material_table: &MaterialTable,
        light_buffer: &LightBuffer,
        radiance_volume: &RadianceVolume,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radiance_inject.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/radiance_inject.wgsl").into(),
            ),
        });

        // --- Group 2: lights storage buffer + InjectUniforms uniform ---

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radiance inject light layout"),
                entries: &[
                    // binding 0: lights array (storage, read-only)
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
                    // binding 1: InjectUniforms (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let inject_uniforms = InjectUniforms {
            num_lights: 0,
            max_shadow_lights: 1,
            _pad: [0; 2],
        };
        let inject_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("inject uniforms"),
                contents: bytemuck::bytes_of(&inject_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radiance inject light bind group"),
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inject_uniforms_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Group 3: radiance L0 write view + volume uniforms ---

        let write_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("radiance inject write layout"),
                entries: &[
                    // binding 0: Level 0 radiance volume (storage texture, write-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: RADIANCE_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    // binding 1: radiance volume uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let write_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radiance inject write bind group"),
            layout: &write_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&radiance_volume.views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: radiance_volume.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline layout ---

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("radiance inject pipeline layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,          // group 0
                &material_table.bind_group_layout, // group 1
                &light_bind_group_layout,          // group 2
                &write_bind_group_layout,          // group 3
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("radiance inject pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            light_bind_group_layout,
            light_bind_group,
            write_bind_group_layout,
            write_bind_group,
            inject_uniforms_buffer,
        }
    }

    /// Upload updated inject uniforms to the GPU.
    ///
    /// Call once per frame before [`dispatch`](Self::dispatch) whenever the
    /// light count changes.
    pub fn update_inject_uniforms(
        &self,
        queue: &wgpu::Queue,
        num_lights: u32,
        max_shadow_lights: u32,
    ) {
        let uniforms = InjectUniforms {
            num_lights,
            max_shadow_lights,
            _pad: [0; 2],
        };
        queue.write_buffer(
            &self.inject_uniforms_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    /// Dispatch the radiance injection compute shader.
    ///
    /// Workgroups: `RADIANCE_DIM / 4` per axis (32 × 32 × 32 = 32 768 groups).
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        material_table: &MaterialTable,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("radiance inject"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &material_table.bind_group, &[]);
        pass.set_bind_group(2, &self.light_bind_group, &[]);
        pass.set_bind_group(3, &self.write_bind_group, &[]);

        let wg = RADIANCE_DIM / 4;
        pass.dispatch_workgroups(wg, wg, wg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_uniforms_size() {
        assert_eq!(std::mem::size_of::<InjectUniforms>(), 16);
    }
}
