//! Cloud shadow map compute pass — Phase 11 task 11.11.
//!
//! Generates a 2D texture (1024×1024, R32Float) where each texel stores the
//! transmittance from the column of air above that world-XZ position down
//! through the cloud layer. The map is camera-centered and updated once per
//! frame.
//!
//! Transmittance 1.0 means fully lit; 0.0 means fully shadowed.
//!
//! Usage in shading (WGSL):
//! ```wgsl
//! let cloud_shadow = textureSample(cloud_shadow_map, shadow_sampler,
//!                                  world_pos.xz / shadow_scale).r;
//! direct_light *= cloud_shadow;
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::clouds::{DEFAULT_CLOUD_MIN, DEFAULT_CLOUD_MAX};

// ── Public constants ──────────────────────────────────────────────────────────

/// Default cloud shadow map resolution (square, 1024×1024).
pub const DEFAULT_CLOUD_SHADOW_RES: u32 = 1024;

/// Default world-space coverage of the shadow map in metres.
pub const DEFAULT_CLOUD_SHADOW_COVERAGE: f32 = 4000.0;

/// Default number of march steps through the cloud layer for the shadow map.
pub const DEFAULT_CLOUD_SHADOW_STEPS: u32 = 16;

/// Default extinction coefficient for cloud shadows.
pub const DEFAULT_CLOUD_SHADOW_EXTINCTION: f32 = 5.0;

/// Texture format for the cloud shadow map (single-channel half-float).
pub const CLOUD_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

// ── GPU struct ────────────────────────────────────────────────────────────────

/// GPU-uploadable cloud shadow map parameters (64 bytes).
///
/// Memory layout:
///
/// ```text
/// offset  0 — center     [f32; 4]    (16 bytes) xyz = camera XZ center, w = unused
/// offset 16 — sun_dir    [f32; 4]    (16 bytes) xyz = toward sun (normalized), w = unused
/// offset 32 — cloud_min  f32         ( 4 bytes)
/// offset 36 — cloud_max  f32         ( 4 bytes)
/// offset 40 — resolution u32         ( 4 bytes)
/// offset 44 — coverage   f32         ( 4 bytes)
/// offset 48 — march_steps u32        ( 4 bytes)
/// offset 52 — extinction f32         ( 4 bytes)
/// offset 56 — _pad0      u32         ( 4 bytes)
/// offset 60 — _pad1      u32         ( 4 bytes)
/// total: 64 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CloudShadowParams {
    /// `xyz` = camera XZ center position (world space), `w` = unused.
    pub center: [f32; 4],
    /// `xyz` = direction toward the sun (normalised), `w` = unused.
    pub sun_dir: [f32; 4],
    /// Lower altitude of the cloud layer in metres.
    pub cloud_min: f32,
    /// Upper altitude of the cloud layer in metres.
    pub cloud_max: f32,
    /// Shadow map resolution (texels per side).
    pub resolution: u32,
    /// World-space extent (metres) of the shadow map area.
    pub coverage: f32,
    /// Number of march steps through the cloud layer.
    pub march_steps: u32,
    /// Extinction coefficient (opacity per unit density per unit distance).
    pub extinction: f32,
    #[doc(hidden)]
    pub _pad0: u32,
    #[doc(hidden)]
    pub _pad1: u32,
}

// ── Pass ──────────────────────────────────────────────────────────────────────

/// Cloud shadow map compute pass.
///
/// Dispatches an 8×8 compute shader that fills [`CloudShadowPass::shadow_texture`]
/// with per-texel transmittance values. Workgroup dispatch is
/// `(resolution.div_ceil(8), resolution.div_ceil(8), 1)`.
#[allow(dead_code)]
pub struct CloudShadowPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    cloud_params_buffer: wgpu::Buffer,
    /// The 2D shadow map texture (R32Float, STORAGE_BINDING | TEXTURE_BINDING).
    pub shadow_texture: wgpu::Texture,
    /// View over the full 2D shadow map texture.
    pub shadow_view: wgpu::TextureView,
    resolution: u32,
}

impl CloudShadowPass {
    /// Create the cloud shadow map pass.
    ///
    /// The map is initially centered on the world origin with the sun
    /// direction set to `(0, 1, 0.5)` (pointing upward toward sun).
    pub fn new(device: &wgpu::Device) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud_shadow.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cloud_shadow.wgsl").into(),
            ),
        });

        let resolution = DEFAULT_CLOUD_SHADOW_RES;

        // --- 2D shadow map texture ---
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud shadow map"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CLOUD_SHADOW_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&Default::default());

        // --- Default params (centered on origin, sun roughly overhead) ---
        // Normalize the default sun direction (0, 1, 0.5)
        let raw = [0.0f32, 1.0, 0.5];
        let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let sun_dir = [raw[0] / len, raw[1] / len, raw[2] / len];

        let default_params = CloudShadowParams {
            center: [0.0, 0.0, 0.0, 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            cloud_min: DEFAULT_CLOUD_MIN,
            cloud_max: DEFAULT_CLOUD_MAX,
            resolution,
            coverage: DEFAULT_CLOUD_SHADOW_COVERAGE,
            march_steps: DEFAULT_CLOUD_SHADOW_STEPS,
            extinction: DEFAULT_CLOUD_SHADOW_EXTINCTION,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud shadow params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Default cloud noise params — disabled (flags.x = 0.0).
        let default_cloud_params = crate::clouds::CloudParams {
            altitude: [1000.0, 3000.0, 0.4, 1.0],
            noise: [0.0003, 0.002, 0.3, 10000.0],
            wind: [1.0, 0.0, 5.0, 0.0],
            flags: [0.0, 4000.0, 1024.0, 0.0],
        };
        let cloud_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cloud shadow noise params"),
            contents: bytemuck::bytes_of(&default_cloud_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group layout ---
        // binding 0: uniform buffer (CloudShadowParams)
        // binding 1: texture_storage_2d<r32float, write>
        // binding 2: uniform buffer (CloudParams — noise settings)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cloud shadow bind group layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: CLOUD_SHADOW_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Cloud noise parameters (CloudParams, 64 bytes)
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

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud shadow bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cloud_params_buffer.as_entire_binding(),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud shadow pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cloud shadow pipeline"),
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
            cloud_params_buffer,
            shadow_texture,
            shadow_view,
            resolution,
        }
    }

    /// Update procedural cloud noise parameters.
    ///
    /// Writes the same [`crate::clouds::CloudParams`] used by [`crate::vol_march::VolMarchPass`]
    /// so the cloud shadow map noise matches the visible clouds exactly.
    pub fn set_cloud_params(&self, queue: &wgpu::Queue, cloud: &crate::clouds::CloudParams) {
        queue.write_buffer(&self.cloud_params_buffer, 0, bytemuck::bytes_of(cloud));
    }

    /// Update the camera center and sun direction, then write params to GPU.
    ///
    /// `camera_pos` — world-space camera position (XZ used; Y ignored for the
    /// 2D shadow map).
    ///
    /// `sun_dir` — world-space direction **toward** the sun (will be
    /// normalised internally).
    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sun_dir: [f32; 3],
    ) {
        self.update_params_ex(queue, camera_pos, sun_dir, DEFAULT_CLOUD_MIN, DEFAULT_CLOUD_MAX,
            DEFAULT_CLOUD_SHADOW_COVERAGE, DEFAULT_CLOUD_SHADOW_EXTINCTION);
    }

    /// Update params with custom cloud altitude, coverage and extinction.
    pub fn update_params_ex(
        &self,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sun_dir: [f32; 3],
        cloud_min: f32,
        cloud_max: f32,
        coverage: f32,
        extinction: f32,
    ) {
        // Normalise sun direction
        let len = (sun_dir[0] * sun_dir[0]
            + sun_dir[1] * sun_dir[1]
            + sun_dir[2] * sun_dir[2])
            .sqrt()
            .max(1e-10);
        let sun_norm = [sun_dir[0] / len, sun_dir[1] / len, sun_dir[2] / len];

        let params = CloudShadowParams {
            center: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
            sun_dir: [sun_norm[0], sun_norm[1], sun_norm[2], 0.0],
            cloud_min,
            cloud_max,
            resolution: self.resolution,
            coverage,
            march_steps: DEFAULT_CLOUD_SHADOW_STEPS,
            extinction,
            _pad0: 0,
            _pad1: 0,
        };

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// World-space coverage of this shadow map (for sampling in other passes).
    pub fn coverage(&self) -> f32 {
        DEFAULT_CLOUD_SHADOW_COVERAGE
    }

    /// Dispatch the cloud shadow map compute shader.
    ///
    /// Updates params then dispatches
    /// `(resolution.div_ceil(8), resolution.div_ceil(8), 1)` workgroups.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sun_dir: [f32; 3],
    ) {
        self.update_params(queue, camera_pos, sun_dir);
        self.dispatch_only(encoder);
    }

    /// Dispatch without updating params.
    ///
    /// Use when params have already been written via [`Self::update_params_ex`]
    /// and [`Self::set_cloud_params`].
    pub fn dispatch_only(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloud shadow"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.resolution.div_ceil(8),
            self.resolution.div_ceil(8),
            1,
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_shadow_params_size_is_64() {
        assert_eq!(std::mem::size_of::<CloudShadowParams>(), 64);
    }

    #[test]
    fn cloud_shadow_params_pod_roundtrip() {
        let p = CloudShadowParams {
            center: [100.0, 0.0, -50.0, 0.0],
            sun_dir: [0.0, -0.894, 0.447, 0.0],
            cloud_min: 1000.0,
            cloud_max: 3000.0,
            resolution: 1024,
            coverage: 4000.0,
            march_steps: 16,
            extinction: 5.0,
            _pad0: 0,
            _pad1: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 64);
        let p2: &CloudShadowParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.coverage, p2.coverage);
        assert_eq!(p.march_steps, p2.march_steps);
    }

    #[test]
    fn cloud_shadow_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(CloudShadowParams, center), 0);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, sun_dir), 16);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, cloud_min), 32);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, cloud_max), 36);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, resolution), 40);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, coverage), 44);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, march_steps), 48);
        assert_eq!(std::mem::offset_of!(CloudShadowParams, extinction), 52);
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_CLOUD_SHADOW_RES, 1024);
        assert!(DEFAULT_CLOUD_SHADOW_COVERAGE > 0.0);
        assert!(DEFAULT_CLOUD_SHADOW_STEPS > 0);
        assert!(DEFAULT_CLOUD_SHADOW_EXTINCTION > 0.0);
    }

    #[test]
    fn format_is_r16float() {
        assert_eq!(CLOUD_SHADOW_FORMAT, wgpu::TextureFormat::R32Float);
    }

    #[test]
    fn dispatch_workgroups() {
        let res = DEFAULT_CLOUD_SHADOW_RES;
        assert_eq!(res.div_ceil(8), 128);
    }

    #[test]
    fn sun_direction_normalisation() {
        let raw = [1.0f32, 2.0, 0.5];
        let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let normalized = [raw[0] / len, raw[1] / len, raw[2] / len];
        let magnitude = (normalized[0] * normalized[0]
            + normalized[1] * normalized[1]
            + normalized[2] * normalized[2])
            .sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }
}
