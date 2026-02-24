//! Radiance injection compute pass for GI — v2 object-centric.
//!
//! Fills Level 0 of the radiance volume with direct-lit radiance at surface
//! voxels. Interior voxels get opacity 1.0 (opaque), exterior voxels get 0.0.
//! Surface voxels compute Lambertian diffuse lighting via SDF normals, shadows
//! (coarse field + BVH), and the material table.
//!
//! # Bind Groups
//!
//! | Group | Content |
//! |-------|---------|
//! | 0 | GpuSceneV2 (brick pool, brick maps, objects, camera, scene, BVH) |
//! | 1 | Material table (storage buffer) |
//! | 2 | Lights (storage buffer) + InjectUniforms (uniform) |
//! | 3 | Radiance volume L0 write (storage texture) + volume uniforms (uniform) |
//! | 4 | Coarse acceleration field (3D texture + sampler + uniforms) |

use bytemuck::{Pod, Zeroable};

use crate::coarse_field::CoarseField;
use crate::gpu_scene::GpuSceneV2;
use crate::light::LightBuffer;
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

/// Radiance injection compute pass — v2 object-centric SDF evaluation.
pub struct RadianceInjectPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,

    /// Material table bind group layout (group 1).
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// Material table bind group.
    material_bind_group: wgpu::BindGroup,

    /// Lights + inject uniforms bind group layout (group 2).
    lights_inject_bind_group_layout: wgpu::BindGroupLayout,
    /// Lights + inject uniforms bind group.
    lights_inject_bind_group: wgpu::BindGroup,

    /// Inject uniforms buffer.
    inject_uniform_buffer: wgpu::Buffer,

    /// Radiance volume L0 write + volume uniforms bind group layout (group 3).
    #[allow(dead_code)]
    radiance_write_bind_group_layout: wgpu::BindGroupLayout,
    /// Radiance volume L0 write + volume uniforms bind group.
    radiance_write_bind_group: wgpu::BindGroup,
}

impl RadianceInjectPass {
    /// Create the radiance injection pass.
    pub fn new(
        device: &wgpu::Device,
        scene: &GpuSceneV2,
        material_buffer: &wgpu::Buffer,
        lights: &LightBuffer,
        radiance_volume: &RadianceVolume,
        coarse_field: &CoarseField,
    ) -> Self {
        // Group 1: Material table (storage buffer, read-only)
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inject_material_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inject_material"),
            layout: &material_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });

        // Group 2: Lights (storage, read) + InjectUniforms (uniform)
        let inject_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inject_uniforms"),
            size: std::mem::size_of::<InjectUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lights_inject_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inject_lights_layout"),
                entries: &[
                    // binding 0: lights (storage, read)
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
                    // binding 1: inject uniforms (uniform)
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
        let lights_inject_bind_group = Self::create_lights_bind_group(
            device,
            &lights_inject_bind_group_layout,
            &lights.buffer,
            &inject_uniform_buffer,
        );

        // Group 3: Radiance volume L0 write (storage texture) + volume uniforms (uniform)
        let radiance_write_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("inject_radiance_write_layout"),
                entries: &[
                    // binding 0: L0 storage texture (write-only)
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
                    // binding 1: volume uniforms (uniform)
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
        let radiance_write_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inject_radiance_write"),
            layout: &radiance_write_bind_group_layout,
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

        // Compile shader and create pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("radiance_inject.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/radiance_inject.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("inject_pipeline_layout"),
            bind_group_layouts: &[
                &scene.bind_group_layout,          // group 0
                &material_bind_group_layout,        // group 1
                &lights_inject_bind_group_layout,   // group 2
                &radiance_write_bind_group_layout,  // group 3
                &coarse_field.bind_group_layout,    // group 4
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("inject_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            material_bind_group_layout,
            material_bind_group,
            lights_inject_bind_group_layout,
            lights_inject_bind_group,
            inject_uniform_buffer,
            radiance_write_bind_group_layout,
            radiance_write_bind_group,
        }
    }

    /// Update inject uniforms.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &InjectUniforms) {
        queue.write_buffer(&self.inject_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Rebuild the lights bind group when the light buffer changes.
    pub fn update_lights(&mut self, device: &wgpu::Device, lights: &LightBuffer) {
        self.lights_inject_bind_group = Self::create_lights_bind_group(
            device,
            &self.lights_inject_bind_group_layout,
            &lights.buffer,
            &self.inject_uniform_buffer,
        );
    }

    /// Rebuild the material bind group when the material buffer changes.
    pub fn update_materials(&mut self, device: &wgpu::Device, material_buffer: &wgpu::Buffer) {
        self.material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inject_material"),
            layout: &self.material_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });
    }

    /// Record the radiance injection dispatch into a command encoder.
    ///
    /// Dispatches 32×32×32 workgroups (4×4×4 each) to cover the 128³ L0 volume.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuSceneV2,
        coarse_field: &CoarseField,
    ) {
        let wg = RADIANCE_DIM / 4; // 32

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("radiance_inject"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.set_bind_group(1, &self.material_bind_group, &[]);
        pass.set_bind_group(2, &self.lights_inject_bind_group, &[]);
        pass.set_bind_group(3, &self.radiance_write_bind_group, &[]);
        pass.set_bind_group(4, &coarse_field.bind_group, &[]);
        pass.dispatch_workgroups(wg, wg, wg);
    }

    fn create_lights_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        light_buffer: &wgpu::Buffer,
        inject_uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("inject_lights"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inject_uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_uniforms_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<InjectUniforms>(), 16);
    }

    #[test]
    fn inject_uniforms_pod_roundtrip() {
        let u = InjectUniforms {
            num_lights: 3,
            max_shadow_lights: 1,
            _pad: [0; 2],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 16);
        let u2: &InjectUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.num_lights, u2.num_lights);
        assert_eq!(u.max_shadow_lights, u2.max_shadow_lights);
    }
}
