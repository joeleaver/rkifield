//! Auto-exposure compute pass.
//!
//! Builds a 256-bin luminance histogram from the HDR image, computes
//! weighted average scene luminance (excluding near-black pixels), and
//! smoothly adapts an exposure value between configurable EV bounds.
//!
//! # Two-pass design
//! 1. **histogram** — 16×16 workgroups over all pixels. Each thread maps its
//!    pixel's luminance to a log2-scaled bin and atomically increments it.
//! 2. **average** — single 256-thread workgroup. Parallel reduction over the
//!    256 bins produces weighted average log luminance → target EV → smooth
//!    adaptation. The histogram is cleared at the end of this pass.
//!
//! Exposure data persists in a GPU buffer. Read it from other passes via
//! [`AutoExposurePass::get_exposure_buffer`].

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Number of bins in the luminance histogram.
pub const HISTOGRAM_BINS: u32 = 256;

/// Default minimum exposure value (EV).
pub const DEFAULT_MIN_EV: f32 = -4.0;

/// Default maximum exposure value (EV).
pub const DEFAULT_MAX_EV: f32 = 16.0;

/// Default adaptation speed (EV units per second, exponential approach).
pub const DEFAULT_ADAPT_SPEED: f32 = 2.0;

// ---------------------------------------------------------------------------
// GPU uniform struct
// ---------------------------------------------------------------------------

/// GPU-uploadable exposure parameters (32 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ExposureParams {
    /// Render width in pixels.
    pub width: u32,
    /// Render height in pixels.
    pub height: u32,
    /// Minimum exposure value (e.g. -4.0).
    pub min_ev: f32,
    /// Maximum exposure value (e.g. 16.0).
    pub max_ev: f32,
    /// Adaptation rate per second (e.g. 2.0).
    pub adapt_speed: f32,
    /// Frame delta time in seconds.
    pub dt: f32,
    /// Total pixel count (width × height).
    pub num_pixels: u32,
    /// Padding to 32 bytes.
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// AutoExposurePass
// ---------------------------------------------------------------------------

/// Auto-exposure compute pass.
///
/// Reads the HDR render target, builds a luminance histogram, and adapts
/// a per-frame exposure value toward the scene average. The resulting
/// exposure (in EV) lives in a persistent GPU buffer accessible to
/// downstream passes (e.g. tone mapping) via [`get_exposure_buffer`].
///
/// [`get_exposure_buffer`]: AutoExposurePass::get_exposure_buffer
#[allow(dead_code)]
pub struct AutoExposurePass {
    histogram_pipeline: wgpu::ComputePipeline,
    average_pipeline: wgpu::ComputePipeline,

    /// 256-bin histogram buffer (256 × u32 = 1024 bytes).
    histogram_buffer: wgpu::Buffer,

    /// Exposure data buffer: `[current_exposure: f32, target_exposure: f32]` (8 bytes).
    exposure_buffer: wgpu::Buffer,

    histogram_bind_group_layout: wgpu::BindGroupLayout,
    average_bind_group_layout: wgpu::BindGroupLayout,

    histogram_bind_group: wgpu::BindGroup,
    average_bind_group: wgpu::BindGroup,

    params_buffer: wgpu::Buffer,

    width: u32,
    height: u32,
}

impl AutoExposurePass {
    /// Create the auto-exposure pass.
    ///
    /// # Parameters
    /// - `device`: wgpu device.
    /// - `hdr_view`: HDR render target view (non-filterable, used in histogram pass).
    /// - `width`: render width in pixels.
    /// - `height`: render height in pixels.
    pub fn new(
        device: &wgpu::Device,
        hdr_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        // --- Shader module ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("auto_exposure.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/auto_exposure.wgsl").into(),
            ),
        });

        // --- Histogram buffer: 256 × u32 = 1024 bytes ---
        let histogram_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auto exposure histogram buffer"),
            contents: &vec![0u8; (HISTOGRAM_BINS * 4) as usize],
            usage: wgpu::BufferUsages::STORAGE,
        });

        // --- Exposure buffer: [current_exposure f32, target_exposure f32] = 8 bytes ---
        let exposure_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auto exposure data buffer"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0f32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // --- Params buffer ---
        let default_params = ExposureParams {
            width,
            height,
            min_ev: DEFAULT_MIN_EV,
            max_ev: DEFAULT_MAX_EV,
            adapt_speed: DEFAULT_ADAPT_SPEED,
            dt: 0.016,
            num_pixels: width * height,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("auto exposure params buffer"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -----------------------------------------------------------------------
        // Histogram bind group layout
        // binding 0: texture_2d<f32> — HDR input (non-filterable)
        // binding 1: storage buffer read_write — histogram bins
        // binding 2: uniform buffer — ExposureParams
        // -----------------------------------------------------------------------
        let histogram_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auto exposure histogram bind group layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        // -----------------------------------------------------------------------
        // Average bind group layout
        // binding 0: storage buffer read_write — histogram bins
        // binding 1: storage buffer read_write — exposure data
        // binding 2: uniform buffer — ExposureParams
        // -----------------------------------------------------------------------
        let average_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auto exposure average bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
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

        // --- Bind groups ---
        let histogram_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("auto exposure histogram bind group"),
            layout: &histogram_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let average_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("auto exposure average bind group"),
            layout: &average_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: histogram_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exposure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipelines ---
        let histogram_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("auto exposure histogram pipeline layout"),
                bind_group_layouts: &[&histogram_bind_group_layout],
                push_constant_ranges: &[],
            });
        let histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("auto exposure histogram pipeline"),
                layout: Some(&histogram_pipeline_layout),
                module: &shader,
                entry_point: Some("histogram_build"),
                compilation_options: Default::default(),
                cache: None,
            });

        let average_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("auto exposure average pipeline layout"),
                bind_group_layouts: &[&average_bind_group_layout],
                push_constant_ranges: &[],
            });
        let average_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("auto exposure average pipeline"),
            layout: Some(&average_pipeline_layout),
            module: &shader,
            entry_point: Some("average"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            histogram_pipeline,
            average_pipeline,
            histogram_buffer,
            exposure_buffer,
            histogram_bind_group_layout,
            average_bind_group_layout,
            histogram_bind_group,
            average_bind_group,
            params_buffer,
            width,
            height,
        }
    }

    /// Dispatch both compute passes into `encoder`.
    ///
    /// Must be called once per frame. `dt` is the frame delta time in seconds.
    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, queue: &wgpu::Queue, dt: f32) {
        // Update dt in params buffer at byte offset 20 (field 5: after width, height, min_ev, max_ev, adapt_speed).
        queue.write_buffer(&self.params_buffer, 20, bytemuck::bytes_of(&dt));

        // Pass 1: build histogram (16×16 workgroups over all pixels).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("auto exposure histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &self.histogram_bind_group, &[]);
            pass.dispatch_workgroups(
                self.width.div_ceil(16),
                self.height.div_ceil(16),
                1,
            );
        }

        // Pass 2: parallel reduction, exposure update, histogram clear.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("auto exposure average"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.average_pipeline);
            pass.set_bind_group(0, &self.average_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Returns a reference to the exposure data buffer.
    ///
    /// Layout: `[current_exposure: f32, target_exposure: f32]` (8 bytes).
    /// Bind this buffer in tone-mapping or other downstream passes to read
    /// the current adapted exposure value at `offset 0`.
    pub fn get_exposure_buffer(&self) -> &wgpu::Buffer {
        &self.exposure_buffer
    }

    /// Update exposure settings at runtime.
    ///
    /// Writes `min_ev` (offset 8), `max_ev` (offset 12), and `adapt_speed`
    /// (offset 16) into the params uniform buffer.
    pub fn update_settings(
        &self,
        queue: &wgpu::Queue,
        min_ev: f32,
        max_ev: f32,
        adapt_speed: f32,
    ) {
        queue.write_buffer(&self.params_buffer, 8, bytemuck::bytes_of(&min_ev));
        queue.write_buffer(&self.params_buffer, 12, bytemuck::bytes_of(&max_ev));
        queue.write_buffer(&self.params_buffer, 16, bytemuck::bytes_of(&adapt_speed));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_params_size_is_32() {
        assert_eq!(std::mem::size_of::<ExposureParams>(), 32);
    }

    #[test]
    fn exposure_params_pod_roundtrip() {
        let p = ExposureParams {
            width: 960,
            height: 540,
            min_ev: -4.0,
            max_ev: 16.0,
            adapt_speed: 2.0,
            dt: 0.016,
            num_pixels: 960 * 540,
            _pad: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
        let p2: &ExposureParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.min_ev, p2.min_ev);
        assert_eq!(p.adapt_speed, p2.adapt_speed);
    }

    #[test]
    fn histogram_buffer_size() {
        assert_eq!(HISTOGRAM_BINS * 4, 1024); // 256 bins × 4 bytes each
    }

    #[test]
    fn default_exposure_values() {
        assert_eq!(DEFAULT_MIN_EV, -4.0);
        assert_eq!(DEFAULT_MAX_EV, 16.0);
        assert_eq!(DEFAULT_ADAPT_SPEED, 2.0);
    }
}
