//! Material preview renderer — self-contained GPU pipeline for 128x128 material thumbnails.
//!
//! Uses a simple analytical SDF ray marcher to write G-buffer data, then runs
//! a dedicated preview shade shader with basic PBR lighting (no shadows/GI/AO).

use bytemuck::{Pod, Zeroable};
use rkf_core::material::Material;
use wgpu::util::DeviceExt;

use crate::gbuffer::{
    GBUFFER_MATERIAL_FORMAT, GBUFFER_MOTION_FORMAT, GBUFFER_NORMAL_FORMAT, GBUFFER_POSITION_FORMAT,
};
use crate::shading::HDR_FORMAT;

/// Preview thumbnail size in pixels (square).
const PREVIEW_SIZE: u32 = 128;

/// Uniforms for the material preview march pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct PreviewUniforms {
    camera_pos: [f32; 4],
    camera_forward: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    resolution: [f32; 2],
    primitive_type: u32,
    material_id: u32,
}

/// Uniforms for the preview shade pass.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct PreviewShadeUniforms {
    camera_pos: [f32; 4],
    resolution: [f32; 2],
    _pad: [f32; 2],
}

/// Self-contained GPU renderer for material preview thumbnails.
pub struct MaterialPreviewRenderer {
    // March pass
    march_pipeline: wgpu::ComputePipeline,
    march_uniform_buffer: wgpu::Buffer,
    march_uniform_bg_layout: wgpu::BindGroupLayout,
    march_gbuf_bg_layout: wgpu::BindGroupLayout,

    // G-buffer textures (128x128)
    gbuf_position_view: wgpu::TextureView,
    gbuf_normal_view: wgpu::TextureView,
    gbuf_material_view: wgpu::TextureView,
    gbuf_motion_view: wgpu::TextureView,

    // Shade pass (simplified: 3 bind groups)
    shade_pipeline: wgpu::ComputePipeline,
    shade_bg0: wgpu::BindGroup,             // G-buffer read
    shade_bg1_layout: wgpu::BindGroupLayout, // Material table (rebuilt per render)
    shade_bg2: wgpu::BindGroup,             // HDR output + uniforms

    // HDR output
    hdr_texture: wgpu::Texture,
    /// HDR output texture view for display.
    pub hdr_view: wgpu::TextureView,

    // Buffers
    shade_uniform_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    material_capacity: usize,

    // CPU readback
    readback_buffer: wgpu::Buffer,
    readback_pending: bool,

    /// Preview size in pixels.
    pub size: u32,
}

impl MaterialPreviewRenderer {
    /// Create the material preview renderer.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let size = PREVIEW_SIZE;

        // ── G-buffer textures ──
        let tex_size = wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 };
        let usage_rw = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

        let mk_gbuf = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: tex_size,
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format, usage: usage_rw, view_formats: &[],
            })
        };

        let gbuf_position = mk_gbuf("preview_gbuf_pos", GBUFFER_POSITION_FORMAT);
        let gbuf_normal = mk_gbuf("preview_gbuf_norm", GBUFFER_NORMAL_FORMAT);
        let gbuf_material = mk_gbuf("preview_gbuf_mat", GBUFFER_MATERIAL_FORMAT);
        let gbuf_motion = mk_gbuf("preview_gbuf_mot", GBUFFER_MOTION_FORMAT);

        let gbuf_position_view = gbuf_position.create_view(&Default::default());
        let gbuf_normal_view = gbuf_normal.create_view(&Default::default());
        let gbuf_material_view = gbuf_material.create_view(&Default::default());
        let gbuf_motion_view = gbuf_motion.create_view(&Default::default());

        // ── HDR output ──
        let hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview_hdr"),
            size: tex_size,
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: usage_rw | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let hdr_view = hdr_texture.create_view(&Default::default());

        // ── March pass pipeline ──
        let march_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview_march_uni"),
            size: std::mem::size_of::<PreviewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let march_uniform_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_march_uni_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });

        let march_gbuf_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_march_gbuf_layout"),
            entries: &[
                storage_tex_entry(0, GBUFFER_POSITION_FORMAT),
                storage_tex_entry(1, GBUFFER_NORMAL_FORMAT),
                storage_tex_entry(2, GBUFFER_MATERIAL_FORMAT),
                storage_tex_entry(3, GBUFFER_MOTION_FORMAT),
            ],
        });

        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material_preview_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/material_preview_march.wgsl").into()),
        });

        let march_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview_march_layout"),
            bind_group_layouts: &[&march_uniform_bg_layout, &march_gbuf_bg_layout],
            push_constant_ranges: &[],
        });

        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preview_march"),
            layout: Some(&march_pipeline_layout),
            module: &march_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Shade pass pipeline (simplified: 3 bind groups) ──

        // Group 0: G-buffer read (3 sampled textures — position, normal, material)
        let shade_bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_shade_bg0"),
            entries: &[
                sampled_tex_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                sampled_tex_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                sampled_tex_entry(2, wgpu::TextureSampleType::Uint),
            ],
        });

        // Group 1: Material table (storage buffer)
        let shade_bg1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_shade_bg1"),
            entries: &[storage_buf_entry(0)],
        });

        // Group 2: HDR output (storage texture) + uniforms
        let shade_bg2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preview_shade_bg2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HDR_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preview_shade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/preview_shade.wgsl").into()),
        });

        let shade_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preview_shade_layout"),
            bind_group_layouts: &[&shade_bg0_layout, &shade_bg1_layout, &shade_bg2_layout],
            push_constant_ranges: &[],
        });

        let shade_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preview_shade"),
            layout: Some(&shade_pipeline_layout),
            module: &shade_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Pre-built bind groups ──

        // BG0: G-buffer read
        let shade_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_shade_bg0"),
            layout: &shade_bg0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gbuf_position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gbuf_normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuf_material_view) },
            ],
        });

        // Shade uniforms
        let shade_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview_shade_uni"),
            size: std::mem::size_of::<PreviewShadeUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // BG2: HDR output + uniforms
        let shade_bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_shade_bg2"),
            layout: &shade_bg2_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_view) },
                wgpu::BindGroupEntry { binding: 1, resource: shade_uniform_buffer.as_entire_binding() },
            ],
        });

        // ── Material buffer ──
        let default_materials = vec![Material::default(); rkf_core::constants::MAX_MATERIALS as usize];
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("preview_materials"),
            contents: bytemuck::cast_slice(&default_materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Readback buffer ──
        let bytes_per_pixel = 8u32; // Rgba16Float
        let row_bytes = size * bytes_per_pixel;
        let padded_row = (row_bytes + 255) & !255;
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preview_readback"),
            size: (padded_row * size) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let _ = (queue, &gbuf_position, &gbuf_normal, &gbuf_material, &gbuf_motion);

        Self {
            march_pipeline,
            march_uniform_buffer,
            march_uniform_bg_layout,
            march_gbuf_bg_layout,
            gbuf_position_view,
            gbuf_normal_view,
            gbuf_material_view,
            gbuf_motion_view,
            shade_pipeline,
            shade_bg0,
            shade_bg1_layout,
            shade_bg2,
            hdr_texture,
            hdr_view,
            shade_uniform_buffer,
            material_buffer,
            material_capacity: rkf_core::constants::MAX_MATERIALS as usize,
            readback_buffer,
            readback_pending: false,
            size,
        }
    }

    /// Submit GPU work for preview rendering (march + shade + readback copy).
    /// Does NOT poll the device — the caller's main frame poll will flush this.
    pub fn dispatch_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[Material],
        material_slot: u16,
        primitive_type: u32,
    ) {
        let size = self.size;
        let bytes_per_pixel = 8u32;
        let row_bytes = size * bytes_per_pixel;
        let padded_row = (row_bytes + 255) & !255;

        // Update material buffer
        if materials.len() > self.material_capacity {
            self.material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("preview_materials"),
                size: (materials.len() * std::mem::size_of::<Material>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.material_capacity = materials.len();
        }
        queue.write_buffer(&self.material_buffer, 0, bytemuck::cast_slice(materials));

        // Rebuild material bind group
        let shade_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_shade_bg1"),
            layout: &self.shade_bg1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.material_buffer.as_entire_binding(),
            }],
        });

        // Update march uniforms
        let cam_pos = [2.5_f32, 1.8, 2.5];
        let target = [0.0_f32, 0.0, 0.0];
        let (fwd, right, up) = compute_preview_camera_vectors(&cam_pos, &target);

        let march_unis = PreviewUniforms {
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            camera_forward: [fwd[0], fwd[1], fwd[2], 0.0],
            camera_right: [right[0], right[1], right[2], 0.0],
            camera_up: [up[0], up[1], up[2], 0.0],
            resolution: [size as f32, size as f32],
            primitive_type,
            material_id: material_slot as u32,
        };
        queue.write_buffer(&self.march_uniform_buffer, 0, bytemuck::bytes_of(&march_unis));

        // Update shade uniforms
        let shade_unis = PreviewShadeUniforms {
            camera_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 0.0],
            resolution: [size as f32, size as f32],
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.shade_uniform_buffer, 0, bytemuck::bytes_of(&shade_unis));

        // Single command encoder: march + shade + readback copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("preview"),
        });

        let march_uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_march_uni_bg"),
            layout: &self.march_uniform_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.march_uniform_buffer.as_entire_binding(),
            }],
        });
        let march_gbuf_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preview_march_gbuf_bg"),
            layout: &self.march_gbuf_bg_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.gbuf_position_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.gbuf_normal_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.gbuf_material_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.gbuf_motion_view) },
            ],
        });

        let workgroups = (size + 7) / 8;

        // March pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preview_march"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.march_pipeline);
            pass.set_bind_group(0, &march_uniform_bg, &[]);
            pass.set_bind_group(1, &march_gbuf_bg, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // Shade pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("preview_shade"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &self.shade_bg0, &[]);
            pass.set_bind_group(1, &shade_bg1, &[]);
            pass.set_bind_group(2, &self.shade_bg2, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // Readback copy
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.hdr_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));
        self.readback_pending = true;
    }

    /// Read back preview pixels. Call AFTER the main frame's readback buffer
    /// has been unmapped (i.e., after `map_readback()` returns).
    pub fn read_pixels(&mut self, device: &wgpu::Device) -> Option<(Vec<u8>, u32, u32)> {
        if !self.readback_pending {
            return None;
        }
        self.readback_pending = false;

        let size = self.size;
        let bytes_per_pixel = 8u32;
        let row_bytes = size * bytes_per_pixel;
        let padded_row = (row_bytes + 255) & !255;

        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let Ok(Ok(())) = rx.recv() else {
            self.readback_buffer.unmap();
            return None;
        };

        let data = buffer_slice.get_mapped_range();
        let mut rgba8 = vec![0u8; (size * size * 4) as usize];
        for y in 0..size as usize {
            let src_offset = y * padded_row as usize;
            for x in 0..size as usize {
                let px_offset = src_offset + x * bytes_per_pixel as usize;
                let r_f16 = u16::from_le_bytes([data[px_offset], data[px_offset + 1]]);
                let g_f16 = u16::from_le_bytes([data[px_offset + 2], data[px_offset + 3]]);
                let b_f16 = u16::from_le_bytes([data[px_offset + 4], data[px_offset + 5]]);
                let a_f16 = u16::from_le_bytes([data[px_offset + 6], data[px_offset + 7]]);
                let r = half::f16::from_bits(r_f16).to_f32().clamp(0.0, 1.0);
                let g = half::f16::from_bits(g_f16).to_f32().clamp(0.0, 1.0);
                let b = half::f16::from_bits(b_f16).to_f32().clamp(0.0, 1.0);
                let a = half::f16::from_bits(a_f16).to_f32().clamp(0.0, 1.0);
                let dst = (y * size as usize + x) * 4;
                rgba8[dst] = (r * 255.0 + 0.5) as u8;
                rgba8[dst + 1] = (g * 255.0 + 0.5) as u8;
                rgba8[dst + 2] = (b * 255.0 + 0.5) as u8;
                rgba8[dst + 3] = (a * 255.0 + 0.5) as u8;
            }
        }
        drop(data);
        self.readback_buffer.unmap();

        Some((rgba8, size, size))
    }
}

// ── Helper functions ──

fn compute_preview_camera_vectors(
    cam_pos: &[f32; 3],
    target: &[f32; 3],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let fwd = [
        target[0] - cam_pos[0],
        target[1] - cam_pos[1],
        target[2] - cam_pos[2],
    ];
    let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
    let fwd = [fwd[0] / len, fwd[1] / len, fwd[2] / len];

    let world_up = [0.0_f32, 1.0, 0.0];
    let right = [
        fwd[1] * world_up[2] - fwd[2] * world_up[1],
        fwd[2] * world_up[0] - fwd[0] * world_up[2],
        fwd[0] * world_up[1] - fwd[1] * world_up[0],
    ];
    let rlen = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    let right = [right[0] / rlen, right[1] / rlen, right[2] / rlen];

    let up = [
        right[1] * fwd[2] - right[2] * fwd[1],
        right[2] * fwd[0] - right[0] * fwd[2],
        right[0] * fwd[1] - right[1] * fwd[0],
    ];

    let fov_y = 45.0_f32;
    let half_tan = (fov_y.to_radians() / 2.0).tan();
    let right_scaled = [right[0] * half_tan, right[1] * half_tan, right[2] * half_tan];
    let up_scaled = [up[0] * half_tan, up[1] * half_tan, up[2] * half_tan];

    (fwd, right_scaled, up_scaled)
}

fn storage_tex_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampled_tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_buf_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
