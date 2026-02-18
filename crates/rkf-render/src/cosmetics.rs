//! Cosmetic post-processing effects (post-upscale).
//!
//! Three optional effects in a single compute pass:
//! - **Vignette**: radial darkening toward screen edges
//! - **Film grain**: luminance-weighted temporal noise
//! - **Chromatic aberration**: radial RGB channel offset
//!
//! All effects are off by default (intensity = 0.0).

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Default vignette intensity (0.0 = off).
pub const DEFAULT_VIGNETTE_INTENSITY: f32 = 0.0;

/// Default film grain intensity (0.0 = off).
pub const DEFAULT_GRAIN_INTENSITY: f32 = 0.0;

/// Default chromatic aberration intensity (0.0 = off).
pub const DEFAULT_CHROMATIC_ABERRATION: f32 = 0.0;

// ---------------------------------------------------------------------------
// GPU uniform struct
// ---------------------------------------------------------------------------

/// GPU-uploadable cosmetics parameters (32 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CosmeticsParams {
    /// Display width in pixels.
    pub width: u32,
    /// Display height in pixels.
    pub height: u32,
    /// Vignette intensity: 0.0 = off, ~0.5 = subtle, ~1.0 = strong.
    pub vignette_intensity: f32,
    /// Film grain intensity: 0.0 = off, ~0.05 = subtle, ~0.2 = strong.
    pub grain_intensity: f32,
    /// Chromatic aberration: 0.0 = off, ~0.002 = subtle, ~0.01 = strong.
    pub chromatic_aberration: f32,
    /// Frame index for temporal grain variation.
    pub frame_index: u32,
    #[doc(hidden)]
    pub _pad0: u32,
    #[doc(hidden)]
    pub _pad1: u32,
}

// ---------------------------------------------------------------------------
// CosmeticsPass
// ---------------------------------------------------------------------------

/// Cosmetic post-processing compute pass (post-upscale).
///
/// Applies up to three optional effects in a single dispatch:
/// vignette, film grain, and chromatic aberration. All are off by default.
///
/// After [`CosmeticsPass::dispatch`], the result is in
/// [`CosmeticsPass::output_view`].
#[allow(dead_code)]
pub struct CosmeticsPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// Cosmetics output texture (Rgba8Unorm, display resolution).
    pub output_texture: wgpu::Texture,
    /// View of the cosmetics output texture.
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl CosmeticsPass {
    /// Create the cosmetics pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `input_ldr_view`: LDR input texture view (non-filterable, Rgba8Unorm from color grading).
    /// - `width`: display width in pixels.
    /// - `height`: display height in pixels.
    pub fn new(
        device: &wgpu::Device,
        input_ldr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cosmetics.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cosmetics.wgsl").into(),
            ),
        });

        // --- Output texture (Rgba8Unorm, display resolution) ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cosmetics output texture"),
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
        // binding 0: texture_2d<f32>            — LDR input (non-filterable, textureLoad)
        // binding 1: texture_storage_2d          — Rgba8Unorm write output
        // binding 2: uniform buffer              — CosmeticsParams
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cosmetics bind group layout"),
                entries: &[
                    // 0: LDR input (non-filterable)
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
                    // 1: storage texture output (Rgba8Unorm write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // 2: uniform params
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

        // --- Params buffer (all effects off by default) ---
        let default_params = CosmeticsParams {
            width,
            height,
            vignette_intensity: DEFAULT_VIGNETTE_INTENSITY,
            grain_intensity: DEFAULT_GRAIN_INTENSITY,
            chromatic_aberration: DEFAULT_CHROMATIC_ABERRATION,
            frame_index: 0,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cosmetics params buffer"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cosmetics bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_ldr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cosmetics pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cosmetics pipeline"),
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
            output_texture,
            output_view,
            width,
            height,
        }
    }

    /// Dispatch the cosmetics pass into the command encoder.
    ///
    /// Updates `frame_index` in the params buffer before dispatching.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        frame_index: u32,
    ) {
        // Update frame_index at byte offset 20.
        queue.write_buffer(&self.params_buffer, 20, bytemuck::bytes_of(&frame_index));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cosmetics"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// Set the vignette intensity at runtime.
    ///
    /// Writes only the `vignette_intensity` field (byte offset 8).
    pub fn set_vignette(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 8, bytemuck::bytes_of(&intensity));
    }

    /// Set the film grain intensity at runtime.
    ///
    /// Writes only the `grain_intensity` field (byte offset 12).
    pub fn set_grain(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 12, bytemuck::bytes_of(&intensity));
    }

    /// Set the chromatic aberration intensity at runtime.
    ///
    /// Writes only the `chromatic_aberration` field (byte offset 16).
    pub fn set_chromatic_aberration(&self, queue: &wgpu::Queue, intensity: f32) {
        queue.write_buffer(&self.params_buffer, 16, bytemuck::bytes_of(&intensity));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosmetics_params_size_is_32() {
        assert_eq!(std::mem::size_of::<CosmeticsParams>(), 32);
    }

    #[test]
    fn cosmetics_params_pod_roundtrip() {
        let p = CosmeticsParams {
            width: 1920,
            height: 1080,
            vignette_intensity: 0.5,
            grain_intensity: 0.05,
            chromatic_aberration: 0.002,
            frame_index: 42,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
        let p2: &CosmeticsParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.vignette_intensity, p2.vignette_intensity);
        assert_eq!(p.frame_index, p2.frame_index);
    }

    #[test]
    fn defaults_are_off() {
        assert_eq!(DEFAULT_VIGNETTE_INTENSITY, 0.0);
        assert_eq!(DEFAULT_GRAIN_INTENSITY, 0.0);
        assert_eq!(DEFAULT_CHROMATIC_ABERRATION, 0.0);
    }

    #[test]
    fn params_field_offsets() {
        assert_eq!(std::mem::offset_of!(CosmeticsParams, vignette_intensity), 8);
        assert_eq!(std::mem::offset_of!(CosmeticsParams, grain_intensity), 12);
        assert_eq!(std::mem::offset_of!(CosmeticsParams, chromatic_aberration), 16);
        assert_eq!(std::mem::offset_of!(CosmeticsParams, frame_index), 20);
    }
}
