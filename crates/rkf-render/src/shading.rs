//! Shading compute pass — Cook-Torrance GGX BRDF.
//!
//! Reads the G-buffer and material table, evaluates PBR shading with a
//! directional sun light, and writes HDR color to an `Rgba16Float` texture.

use crate::gbuffer::GBuffer;
use crate::gpu_color_pool::GpuColorPool;
use crate::gpu_scene::GpuScene;
use crate::material_table::MaterialTable;
use crate::radiance_volume::RadianceVolume;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// GPU-uploadable shade uniforms (32 bytes).
///
/// Contains debug mode selector and camera world-space position.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ShadeUniforms {
    /// Debug visualization mode (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular).
    pub debug_mode: u32,
    /// Number of active lights.
    pub num_lights: u32,
    /// Number of tiles horizontally (for indexing tile_light_counts/indices).
    pub num_tiles_x: u32,
    /// Shadow budget: max shadow-casting lights per pixel (0 = unlimited).
    pub shadow_budget_k: u32,
    /// Camera world-space position (xyz) + unused (w).
    pub camera_pos: [f32; 4],
}

/// Format of the HDR output texture.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Shading compute pass.
#[allow(dead_code)]
pub struct ShadingPass {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// HDR output texture.
    pub hdr_texture: wgpu::Texture,
    /// View for the HDR output texture.
    pub hdr_view: wgpu::TextureView,
    /// Bind group layout for the HDR output (storage texture, write-only).
    hdr_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the HDR output.
    hdr_bind_group: wgpu::BindGroup,
    /// Shade uniforms buffer (debug mode + camera position, 32 bytes).
    shade_uniforms_buffer: wgpu::Buffer,
    /// Bind group layout for shade uniforms.
    shade_uniforms_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for shade uniforms.
    shade_uniforms_bind_group: wgpu::BindGroup,
    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

impl ShadingPass {
    /// Create the shading pass.
    pub fn new(
        device: &wgpu::Device,
        gbuffer: &GBuffer,
        material_table: &MaterialTable,
        scene: &GpuScene,
        light_bind_group_layout: &wgpu::BindGroupLayout,
        radiance_volume: &RadianceVolume,
        color_pool: &GpuColorPool,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shade.wgsl").into()),
        });

        // HDR output texture
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr output"),
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

        // Bind group layout for HDR output (group 2)
        let hdr_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hdr output layout"),
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
            label: Some("hdr output bind group"),
            layout: &hdr_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_view),
            }],
        });

        // Shade uniforms buffer (32 bytes: debug mode + padding + camera_pos)
        let shade_uniforms = ShadeUniforms {
            debug_mode: 0,
            num_lights: 0,
            num_tiles_x: 0,
            shadow_budget_k: 4,
            camera_pos: [0.0; 4],
        };
        let shade_uniforms_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shade uniforms"),
                contents: bytemuck::bytes_of(&shade_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shade_uniforms_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shade uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shade_uniforms_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade uniforms bind group"),
            layout: &shade_uniforms_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shade_uniforms_buffer.as_entire_binding(),
            }],
        });

        // Pipeline layout: group 0 = G-buffer, group 1 = materials, group 2 = HDR output,
        // group 3 = shade uniforms, group 4 = scene SDF data, group 5 = light/tile data,
        // group 6 = radiance volume (GI cone tracing), group 7 = color pool
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shading pipeline layout"),
            bind_group_layouts: &[
                &gbuffer.read_bind_group_layout,
                &material_table.bind_group_layout,
                &hdr_bind_group_layout,
                &shade_uniforms_bind_group_layout,
                &scene.bind_group_layout,
                light_bind_group_layout,
                &radiance_volume.read_bind_group_layout,
                &color_pool.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shading pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            hdr_texture,
            hdr_view,
            hdr_bind_group_layout,
            hdr_bind_group,
            shade_uniforms_buffer,
            shade_uniforms_bind_group_layout,
            shade_uniforms_bind_group,
            width,
            height,
        }
    }

    /// Dispatch the shading compute shader.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        material_table: &MaterialTable,
        scene: &GpuScene,
        light_bind_group: &wgpu::BindGroup,
        radiance_volume: &RadianceVolume,
        color_pool: &GpuColorPool,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shading"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &gbuffer.read_bind_group, &[]);
        pass.set_bind_group(1, &material_table.bind_group, &[]);
        pass.set_bind_group(2, &self.hdr_bind_group, &[]);
        pass.set_bind_group(3, &self.shade_uniforms_bind_group, &[]);
        pass.set_bind_group(4, &scene.bind_group, &[]);
        pass.set_bind_group(5, light_bind_group, &[]);
        pass.set_bind_group(6, &radiance_volume.read_bind_group, &[]);
        pass.set_bind_group(7, &color_pool.bind_group, &[]);

        let wg_x = self.width.div_ceil(8);
        let wg_y = self.height.div_ceil(8);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Set the debug visualization mode.
    ///
    /// - 0: Normal shading
    /// - 1: Surface normals
    /// - 2: World positions
    /// - 3: Material IDs (false-color)
    /// - 4: Diffuse only
    /// - 5: Specular only
    pub fn set_debug_mode(&self, queue: &wgpu::Queue, mode: u32) {
        // Write only the first 4 bytes (debug_mode), preserving other fields.
        queue.write_buffer(&self.shade_uniforms_buffer, 0, bytemuck::bytes_of(&mode));
    }

    /// Update the light info in shade uniforms.
    ///
    /// - `num_lights`: total active lights
    /// - `num_tiles_x`: horizontal tile count for indexing
    /// - `shadow_budget_k`: max shadow-casting lights per pixel (0 = unlimited)
    pub fn update_light_info(
        &self,
        queue: &wgpu::Queue,
        num_lights: u32,
        num_tiles_x: u32,
        shadow_budget_k: u32,
    ) {
        let data = [num_lights, num_tiles_x, shadow_budget_k];
        queue.write_buffer(&self.shade_uniforms_buffer, 4, bytemuck::cast_slice(&data));
    }

    /// Update the camera world-space position for correct view direction computation.
    pub fn update_camera_pos(&self, queue: &wgpu::Queue, pos: [f32; 3]) {
        let data = [pos[0], pos[1], pos[2], 0.0f32];
        // camera_pos starts at byte offset 16 (after debug_mode + padding).
        queue.write_buffer(&self.shade_uniforms_buffer, 16, bytemuck::cast_slice(&data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_uniforms_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<ShadeUniforms>(), 32);
    }

    #[test]
    fn shade_uniforms_pod_roundtrip() {
        let u = ShadeUniforms {
            debug_mode: 3,
            num_lights: 5,
            num_tiles_x: 60,
            shadow_budget_k: 4,
            camera_pos: [1.0, 2.0, 3.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 32);
        let u2: &ShadeUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.debug_mode, u2.debug_mode);
        assert_eq!(u.camera_pos, u2.camera_pos);
    }

    #[test]
    fn shade_uniforms_camera_pos_offset_is_16() {
        // camera_pos field starts at byte 16 (after debug_mode u32 + 3*u32 padding)
        let offset = std::mem::offset_of!(ShadeUniforms, camera_pos);
        assert_eq!(offset, 16);
    }
}
