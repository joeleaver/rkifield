//! Color grading compute pass (post-upscale).
//!
//! Applies a 3D lookup table (LUT) as a final color transform. The LUT
//! maps input RGB to output RGB via trilinear sampling of a 3D texture.
//! Supports 32³ and 64³ LUT sizes. Intensity controls blend between
//! the original and graded colors.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Default LUT dimension (32³ = 32768 voxels).
pub const DEFAULT_LUT_SIZE: u32 = 32;

/// Default color grade intensity (1.0 = full LUT, 0.0 = no grading).
pub const DEFAULT_COLOR_GRADE_INTENSITY: f32 = 1.0;

// ---------------------------------------------------------------------------
// GPU uniform struct
// ---------------------------------------------------------------------------

/// GPU-uploadable color grade parameters (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ColorGradeParams {
    /// Display width in pixels.
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// LUT dimension (e.g. 32 or 64).
    pub lut_size: u32,
    /// Blend factor: 0.0 = no grading, 1.0 = full LUT.
    pub intensity: f32,
}

// ---------------------------------------------------------------------------
// ColorGradePass
// ---------------------------------------------------------------------------

/// Color grading compute pass (post-upscale).
///
/// Reads the LDR texture from tone mapping, applies a 3D LUT via trilinear
/// sampling, and writes the result to an Rgba8Unorm output texture.
///
/// The default LUT is the identity transform — no visible change until a
/// custom LUT is loaded via [`ColorGradePass::upload_lut`].
///
/// After [`ColorGradePass::dispatch`], the result is in
/// [`ColorGradePass::output_view`].
#[allow(dead_code)]
pub struct ColorGradePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
    /// Stored reference to the input LDR texture view (needed for bind group rebuild).
    input_ldr_view: wgpu::TextureView,
    /// Color-graded LDR output texture (Rgba8Unorm, display resolution).
    pub output_texture: wgpu::Texture,
    /// View of the color-graded LDR output texture.
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl ColorGradePass {
    /// Create the color grading pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `queue`: wgpu queue (used to upload the identity LUT).
    /// - `input_ldr_view`: LDR input texture view from tone mapping (non-filterable, Rgba8Unorm).
    /// - `width`: display width in pixels.
    /// - `height`: display height in pixels.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ldr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("color_grade.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/color_grade.wgsl").into(),
            ),
        });

        // --- Identity LUT ---
        let (lut_texture, lut_view) =
            Self::create_identity_lut(device, queue, DEFAULT_LUT_SIZE);

        // --- Trilinear clamp sampler ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("color grade LUT sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // --- Output texture (Rgba8Unorm, display resolution) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color grade output texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- Bind group layout ---
        // binding 0: texture_2d<f32>          — LDR input (non-filterable, textureLoad)
        // binding 1: texture_3d<f32>          — 3D LUT (filterable, textureSampleLevel)
        // binding 2: sampler                  — trilinear filtering
        // binding 3: texture_storage_2d       — Rgba8Unorm write output
        // binding 4: uniform buffer           — ColorGradeParams
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("color grade bind group layout"),
                entries: &[
                    // 0: LDR input (non-filterable — loaded by pixel coord)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: 3D LUT (filterable — trilinear via textureSampleLevel)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: trilinear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 3: storage texture output (Rgba8Unorm write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 4: uniform params
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // --- Params buffer ---
        let default_params = ColorGradeParams {
            width,
            height,
            lut_size: DEFAULT_LUT_SIZE,
            intensity: DEFAULT_COLOR_GRADE_INTENSITY,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("color grade params buffer"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color grade bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_ldr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("color grade pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("color grade pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            params_buffer,
            sampler,
            lut_texture,
            lut_view,
            input_ldr_view: input_ldr_view.clone(),
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Dispatch the color grading pass into the command encoder.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("color grade"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Update the grading intensity at runtime.
    ///
    /// Writes only the `intensity` field (byte offset 12).
    pub fn set_intensity(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 12, bytemuck::bytes_of(&intensity));
    }

    /// Upload a new LUT from raw RGBA8 data.
    ///
    /// `data` must be exactly `size^3 * 4` bytes. The LUT is organized
    /// in B-major order (outer loop B, then G, then R) matching
    /// [`create_identity_lut`].
    ///
    /// Recreates the LUT texture and view, then rebuilds the bind group
    /// with the new texture. Also updates `lut_size` in the params buffer.
    pub fn upload_lut(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        size: u32,
    ) {
        assert_eq!(
            data.len(),
            (size * size * size * 4) as usize,
            "LUT data length must be size^3 * 4"
        );

        // Recreate LUT texture and view.
        let (new_texture, new_view) = Self::upload_lut_texture(device, queue, data, size);
        self.lut_texture = new_texture;
        self.lut_view = new_view;

        // Update lut_size in params buffer (byte offset 8).
        queue.write_buffer(&self.params_buffer, 8, bytemuck::bytes_of(&size));

        // Rebuild bind group to use new LUT view (all 5 bindings required).
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color grade bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.input_ldr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Create an identity 3D LUT texture.
    ///
    /// Each voxel at (r, g, b) stores its own normalized coordinate as RGB,
    /// so the LUT produces no color change when applied.
    fn create_identity_lut(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let mut data = Vec::with_capacity((size * size * size * 4) as usize);
        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    data.push((r as f32 / (size - 1) as f32 * 255.0) as u8);
                    data.push((g as f32 / (size - 1) as f32 * 255.0) as u8);
                    data.push((b as f32 / (size - 1) as f32 * 255.0) as u8);
                    data.push(255u8);
                }
            }
        }
        Self::upload_lut_texture(device, queue, &data, size)
    }

    /// Upload raw RGBA8 data into a new 3D LUT texture.
    fn upload_lut_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        size: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color grade LUT"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
        );
        let view = texture.create_view(&Default::default());
        (texture, view)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_grade_params_size_is_16() {
        assert_eq!(std::mem::size_of::<ColorGradeParams>(), 16);
    }

    #[test]
    fn color_grade_params_pod_roundtrip() {
        let p = ColorGradeParams {
            width: 1920,
            height: 1080,
            lut_size: 32,
            intensity: 1.0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &ColorGradeParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.lut_size, p2.lut_size);
        assert_eq!(p.intensity, p2.intensity);
    }

    #[test]
    fn identity_lut_data_size() {
        let size = DEFAULT_LUT_SIZE;
        let expected = (size * size * size * 4) as usize;
        assert_eq!(expected, 131072); // 32^3 * 4 = 131072 bytes
    }

    #[test]
    fn default_values() {
        assert_eq!(DEFAULT_LUT_SIZE, 32);
        assert_eq!(DEFAULT_COLOR_GRADE_INTENSITY, 1.0);
    }
}
