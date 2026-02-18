//! Volumetric compositing compute pass — Phase 11 task 11.8.
//!
//! Composites the upscaled volumetric scattering buffer over the shaded scene
//! color using the formula:
//!
//! ```text
//! final_rgb = shaded_rgb * transmittance + scatter_rgb
//! ```
//!
//! Where `transmittance` (the `.a` channel of the volumetric buffer) ranges
//! from 1.0 (fully clear) to 0.0 (medium is fully opaque). `scatter_rgb` (the
//! `.rgb` channels) is the light accumulated along the view ray through the
//! participating medium (fog, clouds, dust).
//!
//! This pass is inserted into the frame pipeline immediately after shading and
//! before pre-upscale post-processing.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------- Public constants ----------

/// Output format for the composited result.
///
/// Matches the shading pass output format (`Rgba16Float`) so the result can
/// flow directly into the tone-mapping or post-processing passes.
pub const VOL_COMPOSITE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// ---------- GPU struct ----------

/// GPU-uploadable volumetric compositing parameters (16 bytes).
///
/// Memory layout:
/// ```text
/// offset  0 — width   u32   (4 bytes)
/// offset  4 — height  u32   (4 bytes)
/// offset  8 — _pad0   u32   (4 bytes)
/// offset 12 — _pad1   u32   (4 bytes)
/// total: 16 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct VolCompositeParams {
    /// Internal render resolution width (e.g. 960).
    pub width: u32,
    /// Internal render resolution height (e.g. 540).
    pub height: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
}

// ---------- Pass ----------

/// Volumetric compositing compute pass.
///
/// Owns the full-resolution composited output texture. Takes external views of:
/// - the shaded scene color (from [`ShadingPass`])
/// - the upscaled volumetric scatter buffer (from [`VolUpscalePass`])
///
/// [`ShadingPass`]: crate::shading::ShadingPass
/// [`VolUpscalePass`]: crate::vol_upscale::VolUpscalePass
#[allow(dead_code)]
pub struct VolCompositePass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// Composited output texture at internal resolution.
    pub output_texture: wgpu::Texture,
    /// View into [`output_texture`](VolCompositePass::output_texture).
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl VolCompositePass {
    /// Create a new volumetric compositing pass.
    ///
    /// # Parameters
    /// - `device` — wgpu device.
    /// - `shaded_color_view` — view of the shaded scene color texture
    ///   (`Rgba16Float`, `width × height`).
    /// - `vol_scatter_view` — view of the upscaled volumetric scatter texture
    ///   (`Rgba16Float`, `width × height`; `.rgb` = scatter, `.a` = transmittance).
    /// - `width`, `height` — internal render resolution.
    pub fn new(
        device: &wgpu::Device,
        shaded_color_view: &wgpu::TextureView,
        vol_scatter_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // ---- Shader ----
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vol_composite"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vol_composite.wgsl").into(),
            ),
        });

        // ---- Params buffer ----
        let params = VolCompositeParams {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol_composite_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- Output texture (full internal resolution) ----
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vol_composite_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VOL_COMPOSITE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- Bind group layout ----
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vol_composite_bgl"),
            entries: &[
                // 0: uniform params
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
                // 1: shaded scene color (texture_2d<f32>, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: upscaled volumetric scatter (texture_2d<f32>, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 3: composited output (storage texture, write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: VOL_COMPOSITE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ---- Pipeline ----
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vol_composite_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol_composite_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ---- Bind group ----
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol_composite_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(shaded_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(vol_scatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
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

    /// Dispatch the compositing pass at internal resolution.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg = 8u32;
        let dispatch_x = self.width.div_ceil(wg);
        let dispatch_y = self.height.div_ceil(wg);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol_composite"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_composite_params_size_is_16() {
        assert_eq!(std::mem::size_of::<VolCompositeParams>(), 16);
    }

    #[test]
    fn vol_composite_params_pod_roundtrip() {
        let p = VolCompositeParams {
            width: 960,
            height: 540,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 16);
        let p2: &VolCompositeParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.width, p2.width);
        assert_eq!(p.height, p2.height);
    }

    #[test]
    fn vol_composite_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(VolCompositeParams, width), 0);
        assert_eq!(std::mem::offset_of!(VolCompositeParams, height), 4);
    }

    #[test]
    fn compositing_formula() {
        // Verify: final = scene * transmittance + scatter
        let scene = [1.0f32, 0.5, 0.2];
        let scatter = [0.1, 0.05, 0.02];
        let transmittance = 0.8;
        let result = [
            scene[0] * transmittance + scatter[0],
            scene[1] * transmittance + scatter[1],
            scene[2] * transmittance + scatter[2],
        ];
        assert!((result[0] - 0.9).abs() < 1e-6);
        assert!((result[1] - 0.45).abs() < 1e-6);
        assert!((result[2] - 0.18).abs() < 1e-6);
    }

    #[test]
    fn format_matches_shading() {
        assert_eq!(VOL_COMPOSITE_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn full_transmittance_passes_scene_unchanged() {
        // When transmittance = 1.0 and scatter = 0, output equals scene.
        let scene = [0.8f32, 0.4, 0.2];
        let scatter = [0.0f32, 0.0, 0.0];
        let transmittance = 1.0f32;
        let result = [
            scene[0] * transmittance + scatter[0],
            scene[1] * transmittance + scatter[1],
            scene[2] * transmittance + scatter[2],
        ];
        assert!((result[0] - scene[0]).abs() < 1e-6);
        assert!((result[1] - scene[1]).abs() < 1e-6);
        assert!((result[2] - scene[2]).abs() < 1e-6);
    }

    #[test]
    fn zero_transmittance_passes_scatter_only() {
        // When transmittance = 0, only scattered light is visible (fully opaque medium).
        let scene = [1.0f32, 1.0, 1.0];
        let scatter = [0.3f32, 0.6, 0.9];
        let transmittance = 0.0f32;
        let result = [
            scene[0] * transmittance + scatter[0],
            scene[1] * transmittance + scatter[1],
            scene[2] * transmittance + scatter[2],
        ];
        assert!((result[0] - scatter[0]).abs() < 1e-6);
        assert!((result[1] - scatter[1]).abs() < 1e-6);
        assert!((result[2] - scatter[2]).abs() < 1e-6);
    }

    #[test]
    fn dispatch_workgroup_count() {
        // 960×540 with 8×8 workgroups → ceil(960/8)=120, ceil(540/8)=68
        let width = 960u32;
        let height = 540u32;
        let wg = 8u32;
        assert_eq!(width.div_ceil(wg), 120);
        assert_eq!(height.div_ceil(wg), 68);
    }
}
