//! PBR shading compute pass — v2 object-centric.
//!
//! Reads the G-buffer, evaluates PBR lighting with SDF soft shadows and ambient
//! occlusion via coarse field + BVH + per-object evaluation, and writes HDR output.
//!
//! # Bind Groups
//!
//! | Group | Content |
//! |-------|---------|
//! | 0 | G-buffer read (position, normal, material, motion) |
//! | 1 | Material table (storage buffer) |
//! | 2 | HDR output (storage texture, write) |
//! | 3 | Shade uniforms + brush overlay (uniforms, data, map, brush params) |
//! | 4 | GpuScene (brick pool, brick maps, objects, camera, scene, BVH) |
//! | 5 | Lights (storage buffer) |
//! | 6 | Coarse field (3D texture + sampler + uniforms) |
//! | 7 | Radiance volume (4 clipmap levels + sampler + uniforms) |

use bytemuck::{Pod, Zeroable};

use crate::brush_overlay::BrushOverlay;
use crate::coarse_field::CoarseField;
use crate::gbuffer::GBuffer;
use crate::gpu_color_pool::GpuColorPool;
use crate::gpu_scene::GpuScene;
use crate::light::LightBuffer;
use crate::radiance_volume::RadianceVolume;

/// GPU-uploadable shade uniforms (128 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ShadeUniforms {
    /// Debug visualization mode (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular).
    pub debug_mode: u32,
    /// Number of active lights.
    pub num_lights: u32,
    /// Padding (was num_tiles_x in v1).
    pub _pad0: u32,
    /// Shadow budget: max shadow-casting lights per pixel (0 = unlimited).
    pub shadow_budget_k: u32,
    /// Camera world-space position (xyz) + unused (w).
    /// Must match the camera position used by `flatten_object()` and the ray marcher.
    /// G-buffer positions are world-space, so this is needed for world→camera-relative
    /// conversion in shadow/AO/GI queries and for correct view direction computation.
    pub camera_pos: [f32; 4],
    /// Sun direction toward the sun (xyz) + sun intensity (w).
    pub sun_dir: [f32; 4],
    /// Sun color in linear RGB (xyz) + unused (w).
    pub sun_color: [f32; 4],
    /// Atmosphere params: x=rayleigh_scale, y=mie_scale, z=atmosphere_enabled (1.0/0.0), w=unused.
    pub sky_params: [f32; 4],
    /// Camera forward unit vector (xyz) + unused (w).
    pub cam_forward: [f32; 4],
    /// Camera right vector pre-scaled by tan(fov/2)*aspect (xyz) + unused (w).
    pub cam_right: [f32; 4],
    /// Camera up vector pre-scaled by tan(fov/2) (xyz) + unused (w).
    pub cam_up: [f32; 4],
}

/// Format of the HDR output texture.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// PBR shading compute pass — v2 object-centric SDF shadows and AO.
pub struct ShadingPass {
    /// The compute pipeline (None until shaders are loaded from a project).
    pipeline: Option<wgpu::ComputePipeline>,
    /// Pipeline layout (stored for shader hot-reload).
    pipeline_layout: wgpu::PipelineLayout,

    /// HDR output texture (Rgba16Float at internal resolution).
    pub hdr_texture: wgpu::Texture,
    /// HDR output texture view.
    pub hdr_view: wgpu::TextureView,

    /// Material table bind group layout (group 1).
    #[allow(dead_code)]
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// Material table bind group.
    material_bind_group: wgpu::BindGroup,

    /// HDR output bind group layout (group 2).
    #[allow(dead_code)]
    hdr_bind_group_layout: wgpu::BindGroupLayout,
    /// HDR output bind group.
    hdr_bind_group: wgpu::BindGroup,

    /// Shade uniforms buffer.
    uniform_buffer: wgpu::Buffer,
    /// Combined bind group layout for group 3 (shade uniforms + brush overlay).
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    /// Combined bind group for group 3 (shade uniforms + brush overlay buffers).
    uniform_bind_group: wgpu::BindGroup,

    /// Light buffer bind group layout (group 5).
    light_bind_group_layout: wgpu::BindGroupLayout,
    /// Light buffer bind group.
    light_bind_group: wgpu::BindGroup,

    /// Internal resolution width.
    width: u32,
    /// Internal resolution height.
    height: u32,
}

impl ShadingPass {
    /// Create the shading pass.
    ///
    /// `shader_source` is the composed uber-shader WGSL string produced by
    /// [`ShaderComposer::compose()`]. If `None`, falls back to the built-in
    /// default (PBR-only) composition.
    pub fn new(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        scene: &GpuScene,
        lights: &LightBuffer,
        coarse_field: &CoarseField,
        radiance_volume: &RadianceVolume,
        material_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
        shader_source: Option<&str>,
        brush_overlay: &BrushOverlay,
        color_pool: &GpuColorPool,
    ) -> Self {
        // HDR output texture
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&Default::default());

        // Group 1: Material table
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade_material_layout"),
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
            label: Some("shade_material"),
            layout: &material_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });

        // Group 2: HDR output
        let hdr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade_hdr_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });
        let hdr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade_hdr"),
            layout: &hdr_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            }],
        });

        // Group 3: Shade uniforms
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shade_uniforms"),
            size: std::mem::size_of::<ShadeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Group 3: Shade uniforms (binding 0) + brush overlay (bindings 1-3).
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade_uniform_layout"),
                entries: &[
                    // binding 0: shade uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: brush overlay geodesic distance data (array<f32>)
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
                    // binding 2: brush overlay brick slot → overlay slot map (array<u32>)
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
                    // binding 3: brush overlay uniforms
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
                    // binding 4: color pool data (array<u32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 5: color companion map (array<u32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let uniform_bind_group = Self::build_group3_bind_group(
            device, &uniform_bind_group_layout, &uniform_buffer, brush_overlay, color_pool,
        );

        // Group 5: Lights
        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade_light_layout"),
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
        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade_lights"),
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights.buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shade_pipeline_layout"),
            bind_group_layouts: &[
                &gbuffer.read_bind_group_layout,         // group 0
                &material_bind_group_layout,              // group 1
                &hdr_bind_group_layout,                   // group 2
                &uniform_bind_group_layout,               // group 3 (uniforms + brush overlay)
                &scene.bind_group_layout,                 // group 4
                &light_bind_group_layout,                 // group 5
                &coarse_field.bind_group_layout,          // group 6
                &radiance_volume.read_bind_group_layout,  // group 7
            ],
            push_constant_ranges: &[],
        });

        // Compile pipeline now if shader source provided, otherwise defer
        // until recompile() is called when a project loads its shaders.
        let pipeline = shader_source.map(|src| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shade_composed"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(src)),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shade_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        Self {
            pipeline,
            pipeline_layout,
            hdr_texture,
            hdr_view,
            material_bind_group_layout,
            material_bind_group,
            hdr_bind_group_layout,
            hdr_bind_group,
            uniform_buffer,
            uniform_bind_group_layout,
            uniform_bind_group,
            light_bind_group_layout,
            light_bind_group,
            width,
            height,
        }
    }

    /// Build the combined group 3 bind group (shade uniforms + brush overlay + color pool).
    fn build_group3_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        brush_overlay: &BrushOverlay,
        color_pool: &GpuColorPool,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade_uniforms_and_brush"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brush_overlay.data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: brush_overlay.map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: brush_overlay.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_pool.color_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: color_pool.companion_map_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Rebuild group 3 bind group after brush overlay or color pool buffers change.
    ///
    /// Call this after `BrushOverlay::update()` since it recreates data/map buffers,
    /// or after the color pool buffers are recreated.
    pub fn rebuild_group3(
        &mut self,
        device: &wgpu::Device,
        brush_overlay: &BrushOverlay,
        color_pool: &GpuColorPool,
    ) {
        self.uniform_bind_group = Self::build_group3_bind_group(
            device, &self.uniform_bind_group_layout, &self.uniform_buffer,
            brush_overlay, color_pool,
        );
    }

    /// Update shade uniforms.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &ShadeUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Rebuild the light bind group when the light buffer changes.
    pub fn update_lights(&mut self, device: &wgpu::Device, lights: &LightBuffer) {
        self.light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade_lights"),
            layout: &self.light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights.buffer.as_entire_binding(),
            }],
        });
    }

    /// Recompile the shading pipeline from composed WGSL source.
    ///
    /// Creates a new shader module and pipeline from the given source string.
    /// Used by the shader composer hot-reload path.
    pub fn recompile(&mut self, device: &wgpu::Device, source: &str) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade_composed"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        });
        self.recreate_pipeline(device, &module);
    }

    /// Recreate the compute pipeline with a new shader module (hot-reload).
    pub fn recreate_pipeline(&mut self, device: &wgpu::Device, module: &wgpu::ShaderModule) {
        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shade_pipeline"),
            layout: Some(&self.pipeline_layout),
            module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
    }

    /// Rebuild the material bind group when the material buffer changes.
    pub fn update_materials(&mut self, device: &wgpu::Device, material_buffer: &wgpu::Buffer) {
        self.material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade_material"),
            layout: &self.material_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buffer.as_entire_binding(),
            }],
        });
    }

    /// Record the shading dispatch into a command encoder.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        scene: &GpuScene,
        coarse_field: &CoarseField,
        radiance_volume: &RadianceVolume,
    ) {
        let workgroups_x = (self.width + 7) / 8;
        let workgroups_y = (self.height + 7) / 8;

        let pipeline = match &self.pipeline {
            Some(p) => p,
            None => return, // No shader compiled yet (no project loaded).
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shade"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &gbuffer.read_bind_group, &[]);
        pass.set_bind_group(1, &self.material_bind_group, &[]);
        pass.set_bind_group(2, &self.hdr_bind_group, &[]);
        pass.set_bind_group(3, &self.uniform_bind_group, &[]);
        pass.set_bind_group(4, &scene.bind_group, &[]);
        pass.set_bind_group(5, &self.light_bind_group, &[]);
        pass.set_bind_group(6, &coarse_field.bind_group, &[]);
        pass.set_bind_group(7, &radiance_volume.read_bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_uniforms_size_is_128_bytes() {
        assert_eq!(std::mem::size_of::<ShadeUniforms>(), 128);
    }

    #[test]
    fn shade_uniforms_pod_roundtrip() {
        let u = ShadeUniforms {
            debug_mode: 3,
            num_lights: 5,
            _pad0: 0,
            shadow_budget_k: 4,
            camera_pos: [1.0, 2.0, 3.0, 0.0],
            sun_dir: [0.5, 1.0, 0.3, 3.0],
            sun_color: [1.0, 0.95, 0.85, 0.0],
            sky_params: [1.0, 1.0, 1.0, 0.0],
            cam_forward: [0.0, 0.0, -1.0, 0.0],
            cam_right: [1.0, 0.0, 0.0, 0.0],
            cam_up: [0.0, 1.0, 0.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 128);
        let u2: &ShadeUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.debug_mode, u2.debug_mode);
        assert_eq!(u.camera_pos, u2.camera_pos);
        assert_eq!(u.sun_dir, u2.sun_dir);
    }

    #[test]
    fn shade_uniforms_camera_pos_offset_is_16() {
        let offset = std::mem::offset_of!(ShadeUniforms, camera_pos);
        assert_eq!(offset, 16);
    }

    #[test]
    fn shade_uniforms_sun_dir_offset_is_32() {
        let offset = std::mem::offset_of!(ShadeUniforms, sun_dir);
        assert_eq!(offset, 32);
    }

    #[test]
    fn hdr_format_is_rgba16float() {
        assert_eq!(HDR_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }
}
