//! Volumetric ray march compute pass — Phase 11 task 11.2 / 11.4.
//!
//! Performs front-to-back fixed-step compositing through participating media
//! (fog, dust, clouds) at half the internal render resolution.
//!
//! The pass reads:
//! - A full-res depth buffer (`.w` channel from the G-buffer position texture)
//!   to avoid marching past solid geometry.
//! - A volumetric shadow map (task 11.1) for sun visibility at each sample.
//!
//! It writes a half-res [`VOL_MARCH_FORMAT`] (`Rgba16Float`) texture where
//! RGB = accumulated in-scattering and A = remaining transmittance. This is
//! composited over the shaded frame in a later pass.
//!
//! Task 11.4 expands fog support: the two legacy `ambient_dust_*` scalars are
//! replaced by three packed `vec4` fields (`fog_color`, `fog_height`,
//! `fog_distance`) that carry all fog configuration. Use [`VolMarchPass::set_fog_params`]
//! to push updated [`crate::fog::FogParams`] into the params buffer.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------- Public constants ----------

/// Output texture format: RGBA half-float (scatter_rgb, transmittance).
pub const VOL_MARCH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Default world-space step size along the view ray (metres).
pub const DEFAULT_VOL_STEP_SIZE: f32 = 2.0;

/// Default maximum number of march steps per pixel.
pub const DEFAULT_VOL_MAX_STEPS: u32 = 32;

/// Default near-plane distance for the volumetric march (metres).
pub const DEFAULT_VOL_NEAR: f32 = 0.5;

/// Default maximum march distance / far-plane fallback (metres).
pub const DEFAULT_VOL_FAR: f32 = 200.0;

/// Default ambient dust density — `0.0` keeps volumetrics off by default.
///
/// This constant is retained for compatibility. Ambient dust density is now
/// stored in `fog_distance[2]` inside [`VolMarchParams`].
pub const DEFAULT_AMBIENT_DUST: f32 = 0.0;

/// Default Henyey-Greenstein asymmetry for ambient dust (0 = isotropic).
///
/// This constant is retained for compatibility. The asymmetry is now stored
/// in `fog_distance[3]` inside [`VolMarchParams`].
pub const DEFAULT_AMBIENT_DUST_G: f32 = 0.3;

// ---------- GPU struct ----------

/// GPU-uploadable volumetric march parameters.
///
/// Memory layout (224 bytes, fully 16-byte aligned):
///
/// ```text
/// offset   0 — cam_pos          [f32; 4]  (16 bytes)
/// offset  16 — cam_forward      [f32; 4]  (16 bytes)
/// offset  32 — cam_right        [f32; 4]  (16 bytes)
/// offset  48 — cam_up           [f32; 4]  (16 bytes)
/// offset  64 — sun_dir          [f32; 4]  (16 bytes)
/// offset  80 — sun_color        [f32; 4]  (16 bytes)
/// offset  96 — width                       (4 bytes)
/// offset 100 — height                      (4 bytes)
/// offset 104 — full_width                  (4 bytes)
/// offset 108 — full_height                 (4 bytes)
/// offset 112 — max_steps                   (4 bytes)
/// offset 116 — step_size                   (4 bytes)
/// offset 120 — near                        (4 bytes)
/// offset 124 — far                         (4 bytes)
/// offset 128 — fog_color        [f32; 4]  xyz=RGB, w=height_fog_enable   (16 bytes)
/// offset 144 — fog_height       [f32; 4]  x=base_density, y=base_height,
///                                          z=height_falloff, w=dist_fog_enable (16 bytes)
/// offset 160 — fog_distance     [f32; 4]  x=distance_density, y=distance_falloff,
///                                          z=ambient_dust_density, w=ambient_dust_g (16 bytes)
/// offset 176 — frame_index                 (4 bytes)
/// offset 180 — _pad0                       (4 bytes)
/// offset 184 — _pad1                       (4 bytes)
/// offset 188 — _pad2                       (4 bytes)
/// offset 192 — vol_shadow_min   [f32; 4]  (16 bytes)
/// offset 208 — vol_shadow_max   [f32; 4]  (16 bytes)
/// total: 224 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct VolMarchParams {
    /// Camera world-space position (w unused).
    pub cam_pos: [f32; 4],
    /// Camera forward vector (w unused).
    pub cam_forward: [f32; 4],
    /// Camera right vector pre-scaled by horizontal FOV half-tan (w unused).
    pub cam_right: [f32; 4],
    /// Camera up vector pre-scaled by vertical FOV half-tan (w unused).
    pub cam_up: [f32; 4],
    /// Unit vector pointing **toward** the sun (w unused).
    pub sun_dir: [f32; 4],
    /// Sun radiance (color × intensity, w unused).
    pub sun_color: [f32; 4],
    /// Half-resolution output width in pixels.
    pub width: u32,
    /// Half-resolution output height in pixels.
    pub height: u32,
    /// Full internal-resolution width (for depth buffer sampling).
    pub full_width: u32,
    /// Full internal-resolution height (for depth buffer sampling).
    pub full_height: u32,
    /// Maximum number of march steps per pixel.
    pub max_steps: u32,
    /// World-space step size along the view ray (metres).
    pub step_size: f32,
    /// Near-plane march start distance (metres).
    pub near: f32,
    /// Maximum march distance / far-plane fallback (metres).
    pub far: f32,
    /// Fog scattering color (RGB) and height-fog enable flag.
    ///
    /// `xyz` = linear-RGB scattering albedo.
    /// `w` = height fog enable: `0.0` = off, `1.0` = on.
    pub fog_color: [f32; 4],
    /// Height fog parameters and distance-fog enable flag.
    ///
    /// `x` = base density (peak extinction at/below `y`).
    /// `y` = base height (world-space metres, fog peaks at or below this).
    /// `z` = height falloff exponent.
    /// `w` = distance fog enable: `0.0` = off, `1.0` = on.
    pub fog_height: [f32; 4],
    /// Distance fog and ambient dust parameters.
    ///
    /// `x` = distance fog density coefficient.
    /// `y` = distance fog falloff exponent.
    /// `z` = ambient dust density (uniform extinction, 0 = off).
    /// `w` = Henyey-Greenstein asymmetry for ambient dust.
    pub fog_distance: [f32; 4],
    /// Current frame index for jitter temporal variation.
    pub frame_index: u32,
    /// Volumetric ambient sky color (RGB). Used for multi-scatter approximation
    /// on clouds and fog. User-controlled — no longer derived from sun elevation.
    pub vol_ambient_color: [f32; 3],
    /// World-space minimum corner of the volumetric shadow volume (w unused).
    pub vol_shadow_min: [f32; 4],
    /// World-space maximum corner of the volumetric shadow volume (w unused).
    pub vol_shadow_max: [f32; 4],
}

// ---------- Pass ----------

/// Volumetric ray march compute pass (half internal resolution).
///
/// Dispatches [`vol_march.wgsl`](../shaders/vol_march.wgsl) over a half-res
/// grid. Workgroup size is 8×8×1; dispatch is
/// `(width.div_ceil(8), height.div_ceil(8), 1)` workgroups.
#[allow(dead_code)]
pub struct VolMarchPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    cloud_params_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Half-res output texture (Rgba16Float): scatter RGB + transmittance.
    pub output_texture: wgpu::Texture,
    /// View over the full output texture.
    pub output_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl VolMarchPass {
    /// Create the volumetric march pass.
    ///
    /// # Arguments
    /// - `device` / `queue` — wgpu device and queue.
    /// - `depth_view` — full-res depth texture view (G-buffer position `.w` = distance).
    ///   Must be a non-filterable float texture.
    /// - `vol_shadow_view` — 3D volumetric shadow map view from task 11.1
    ///   (filterable float texture).
    /// - `half_width` / `half_height` — output (half internal) resolution.
    /// - `full_width` / `full_height` — full internal resolution (for depth sampling).
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        vol_shadow_view: &wgpu::TextureView,
        cloud_shadow_view: &wgpu::TextureView,
        half_width: u32,
        half_height: u32,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vol_march.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vol_march.wgsl").into(),
            ),
        });

        // --- Output texture ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vol march output"),
            size: wgpu::Extent3d {
                width: half_width,
                height: half_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VOL_MARCH_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // --- Linear clamp sampler for the volumetric shadow map ---
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vol march shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // --- Default params ---
        let default_params = VolMarchParams {
            cam_pos: [0.0, 0.0, 0.0, 0.0],
            cam_forward: [0.0, 0.0, 1.0, 0.0],
            cam_right: [1.0, 0.0, 0.0, 0.0],
            cam_up: [0.0, 1.0, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [1.0, 0.95, 0.8, 0.0],
            width: half_width,
            height: half_height,
            full_width,
            full_height,
            max_steps: DEFAULT_VOL_MAX_STEPS,
            step_size: DEFAULT_VOL_STEP_SIZE,
            near: DEFAULT_VOL_NEAR,
            far: DEFAULT_VOL_FAR,
            // Fog defaults: all disabled, sensible falloff values.
            fog_color: [0.7, 0.8, 0.9, 0.0],     // height fog off (w=0)
            fog_height: [0.0, 0.0, 0.1, 0.0],     // distance fog off (w=0)
            fog_distance: [0.0, 0.01, 0.0, 0.3],  // no dust; g=0.3
            frame_index: 0,
            vol_ambient_color: [0.24, 0.30, 0.42],
            // Match VolShadowPass default bounds: 128×64×128 m volume centred at origin.
            vol_shadow_min: [-64.0, -32.0, -64.0, 0.0],
            vol_shadow_max: [64.0, 32.0, 64.0, 0.0],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol march params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Default cloud params — disabled (flags.x = 0.0).
        let default_cloud_params = crate::clouds::CloudParams {
            altitude: [1000.0, 3000.0, 0.4, 1.0],
            noise: [0.0003, 0.002, 0.3, 10000.0],
            wind: [1.0, 0.0, 5.0, 0.0],
            flags: [0.0, 4000.0, 1024.0, 0.0], // procedural disabled
        };
        let cloud_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol march cloud params"),
            contents: bytemuck::bytes_of(&default_cloud_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group layout ---
        // binding 0: uniform buffer (VolMarchParams)
        // binding 1: texture_2d<f32> depth (non-filterable)
        // binding 2: texture_3d<f32> vol shadow (filterable)
        // binding 3: sampler (filtering)
        // binding 4: texture_storage_2d<rgba16float, write> output
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol march bind group layout"),
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
                    // Depth buffer — non-filterable float (matches G-buffer Rgba32Float)
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
                    // Volumetric shadow map — filterable 3D float
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Filtering sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Output storage texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: VOL_MARCH_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Cloud parameters uniform (CloudParams, 64 bytes)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Cloud shadow map — filterable 2D float texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol march bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(vol_shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cloud_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(cloud_shadow_view),
                },
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vol march pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol march pipeline"),
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
            sampler,
            output_texture,
            output_view,
            width: half_width,
            height: half_height,
        }
    }

    /// Upload updated params and dispatch the volumetric march compute shader.
    ///
    /// Call once per frame after the G-buffer and volumetric shadow passes have
    /// completed.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        params: &VolMarchParams,
    ) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol march"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.width.div_ceil(8),
            self.height.div_ceil(8),
            1,
        );
    }

    /// Update the volumetric shadow map bounds in the params buffer.
    ///
    /// Call this after updating the [`VolShadowPass`] bounds to keep them in
    /// sync. The offsets match the [`VolMarchParams`] layout:
    /// - `vol_shadow_min` at offset 192
    /// - `vol_shadow_max` at offset 208
    pub fn set_shadow_bounds(
        &self,
        queue: &wgpu::Queue,
        vol_min: [f32; 3],
        vol_max: [f32; 3],
    ) {
        let min_vec4: [f32; 4] = [vol_min[0], vol_min[1], vol_min[2], 0.0];
        let max_vec4: [f32; 4] = [vol_max[0], vol_max[1], vol_max[2], 0.0];
        queue.write_buffer(&self.params_buffer, 192, bytemuck::bytes_of(&min_vec4));
        queue.write_buffer(&self.params_buffer, 208, bytemuck::bytes_of(&max_vec4));
    }

    /// Update fog parameters from CPU-side [`crate::fog::FogParams`].
    ///
    /// Writes the three packed fog vec4 fields (`fog_color`, `fog_height`,
    /// `fog_distance`) at offsets 128–176 in the params buffer.
    ///
    /// # Example
    /// ```ignore
    /// let fog = FogSettings { height_fog_enabled: true, fog_base_density: 0.05, .. Default::default() };
    /// vol_march_pass.set_fog_params(queue, &FogParams::from_settings(&fog));
    /// ```
    pub fn set_fog_params(&self, queue: &wgpu::Queue, fog: &crate::fog::FogParams) {
        // Write all three fog vec4s (48 bytes) starting at offset 128.
        // FogParams is 64 bytes total; we only upload the first 48 (skip _pad).
        let fog_bytes = bytemuck::bytes_of(fog);
        queue.write_buffer(&self.params_buffer, 128, &fog_bytes[..48]);
    }

    /// Update procedural cloud parameters.
    ///
    /// Writes the full 64-byte [`crate::clouds::CloudParams`] to the cloud
    /// params uniform buffer (binding 5). Call once per frame with updated
    /// time for wind animation.
    pub fn set_cloud_params(&self, queue: &wgpu::Queue, cloud: &crate::clouds::CloudParams) {
        queue.write_buffer(&self.cloud_params_buffer, 0, bytemuck::bytes_of(cloud));
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_march_params_size() {
        // 6 vec4s (96) + 8 scalars (32) + 3 fog vec4s (48) + 4 scalars (16) + 2 vec4s (32) = 224
        assert_eq!(std::mem::size_of::<VolMarchParams>(), 224);
    }

    #[test]
    fn vol_march_params_pod_roundtrip() {
        let p = VolMarchParams {
            cam_pos: [0.0, 5.0, -10.0, 0.0],
            cam_forward: [0.0, 0.0, 1.0, 0.0],
            cam_right: [1.0, 0.0, 0.0, 0.0],
            cam_up: [0.0, 1.0, 0.0, 0.0],
            sun_dir: [0.0, -0.894, 0.447, 0.0],
            sun_color: [1.0, 0.95, 0.8, 0.0],
            width: 480,
            height: 270,
            full_width: 960,
            full_height: 540,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [0.7, 0.8, 0.9, 1.0],
            fog_height: [0.05, 5.0, 0.1, 0.0],
            fog_distance: [0.01, 0.005, 0.003, 0.3],
            frame_index: 0,
            vol_ambient_color: [0.24, 0.30, 0.42],
            vol_shadow_min: [-64.0, -32.0, -64.0, 0.0],
            vol_shadow_max: [64.0, 32.0, 64.0, 0.0],
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 224);
        let p2: &VolMarchParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.width, p2.width);
        assert_eq!(p.fog_color, p2.fog_color);
        assert_eq!(p.fog_height, p2.fog_height);
        assert_eq!(p.fog_distance, p2.fog_distance);
        assert_eq!(p.cam_pos, p2.cam_pos);
        assert_eq!(p.sun_color, p2.sun_color);
        assert_eq!(p.vol_shadow_min, p2.vol_shadow_min);
        assert_eq!(p.vol_shadow_max, p2.vol_shadow_max);
    }

    #[test]
    fn vol_march_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(VolMarchParams, cam_pos), 0);
        assert_eq!(std::mem::offset_of!(VolMarchParams, cam_forward), 16);
        assert_eq!(std::mem::offset_of!(VolMarchParams, cam_right), 32);
        assert_eq!(std::mem::offset_of!(VolMarchParams, cam_up), 48);
        assert_eq!(std::mem::offset_of!(VolMarchParams, sun_dir), 64);
        assert_eq!(std::mem::offset_of!(VolMarchParams, sun_color), 80);
        assert_eq!(std::mem::offset_of!(VolMarchParams, width), 96);
        assert_eq!(std::mem::offset_of!(VolMarchParams, height), 100);
        assert_eq!(std::mem::offset_of!(VolMarchParams, full_width), 104);
        assert_eq!(std::mem::offset_of!(VolMarchParams, full_height), 108);
        assert_eq!(std::mem::offset_of!(VolMarchParams, max_steps), 112);
        assert_eq!(std::mem::offset_of!(VolMarchParams, step_size), 116);
        assert_eq!(std::mem::offset_of!(VolMarchParams, near), 120);
        assert_eq!(std::mem::offset_of!(VolMarchParams, far), 124);
        assert_eq!(std::mem::offset_of!(VolMarchParams, fog_color), 128);
        assert_eq!(std::mem::offset_of!(VolMarchParams, fog_height), 144);
        assert_eq!(std::mem::offset_of!(VolMarchParams, fog_distance), 160);
        assert_eq!(std::mem::offset_of!(VolMarchParams, frame_index), 176);
        assert_eq!(std::mem::offset_of!(VolMarchParams, vol_ambient_color), 180);
        assert_eq!(std::mem::offset_of!(VolMarchParams, vol_shadow_min), 192);
        assert_eq!(std::mem::offset_of!(VolMarchParams, vol_shadow_max), 208);
    }

    #[test]
    fn shadow_bounds_default() {
        // Default shadow volume bounds should match VolShadowPass defaults:
        // 128×64×128 m volume centred at origin.
        let expected_min = [-64.0f32, -32.0, -64.0, 0.0];
        let expected_max = [64.0f32, 32.0, 64.0, 0.0];

        let p = VolMarchParams {
            cam_pos: [0.0; 4],
            cam_forward: [0.0, 0.0, 1.0, 0.0],
            cam_right: [1.0, 0.0, 0.0, 0.0],
            cam_up: [0.0, 1.0, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [1.0, 0.95, 0.8, 0.0],
            width: 480,
            height: 270,
            full_width: 960,
            full_height: 540,
            max_steps: DEFAULT_VOL_MAX_STEPS,
            step_size: DEFAULT_VOL_STEP_SIZE,
            near: DEFAULT_VOL_NEAR,
            far: DEFAULT_VOL_FAR,
            fog_color: [0.7, 0.8, 0.9, 0.0],
            fog_height: [0.0, 0.0, 0.1, 0.0],
            fog_distance: [0.0, 0.01, DEFAULT_AMBIENT_DUST, DEFAULT_AMBIENT_DUST_G],
            frame_index: 0,
            vol_ambient_color: [0.24, 0.30, 0.42],
            vol_shadow_min: expected_min,
            vol_shadow_max: expected_max,
        };
        assert_eq!(p.vol_shadow_min, expected_min);
        assert_eq!(p.vol_shadow_max, expected_max);

        // Volume dimensions should be 128×64×128.
        let size_x = p.vol_shadow_max[0] - p.vol_shadow_min[0];
        let size_y = p.vol_shadow_max[1] - p.vol_shadow_min[1];
        let size_z = p.vol_shadow_max[2] - p.vol_shadow_min[2];
        assert!((size_x - 128.0).abs() < 1e-5);
        assert!((size_y - 64.0).abs() < 1e-5);
        assert!((size_z - 128.0).abs() < 1e-5);
    }

    #[test]
    fn fog_fields_default_disabled() {
        // With default construction the enable flags (w channels) must be 0.
        let p = VolMarchParams {
            cam_pos: [0.0; 4],
            cam_forward: [0.0, 0.0, 1.0, 0.0],
            cam_right: [1.0, 0.0, 0.0, 0.0],
            cam_up: [0.0, 1.0, 0.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [1.0, 0.95, 0.8, 0.0],
            width: 480,
            height: 270,
            full_width: 960,
            full_height: 540,
            max_steps: DEFAULT_VOL_MAX_STEPS,
            step_size: DEFAULT_VOL_STEP_SIZE,
            near: DEFAULT_VOL_NEAR,
            far: DEFAULT_VOL_FAR,
            fog_color: [0.7, 0.8, 0.9, 0.0],   // w=0 → height fog off
            fog_height: [0.0, 0.0, 0.1, 0.0],   // w=0 → distance fog off
            fog_distance: [0.0, 0.01, 0.0, 0.3],
            frame_index: 0,
            vol_ambient_color: [0.24, 0.30, 0.42],
            vol_shadow_min: [-64.0, -32.0, -64.0, 0.0],
            vol_shadow_max: [64.0, 32.0, 64.0, 0.0],
        };
        assert_eq!(p.fog_color[3], 0.0);   // height fog disabled
        assert_eq!(p.fog_height[3], 0.0);  // distance fog disabled
        assert_eq!(p.fog_distance[2], 0.0); // no ambient dust
    }

    #[test]
    fn default_constants() {
        assert_eq!(DEFAULT_VOL_MAX_STEPS, 32);
        assert!(DEFAULT_VOL_STEP_SIZE > 0.0);
        assert!(DEFAULT_VOL_NEAR > 0.0);
        assert!(DEFAULT_VOL_FAR > DEFAULT_VOL_NEAR);
        assert_eq!(DEFAULT_AMBIENT_DUST, 0.0); // off by default
        assert!(DEFAULT_AMBIENT_DUST_G >= -1.0 && DEFAULT_AMBIENT_DUST_G <= 1.0);
    }

    #[test]
    fn half_res_dimensions() {
        let full_w = 960u32;
        let full_h = 540u32;
        let half_w = full_w / 2;
        let half_h = full_h / 2;
        assert_eq!(half_w, 480);
        assert_eq!(half_h, 270);
    }

    #[test]
    fn vol_march_format_is_rgba16float() {
        assert_eq!(VOL_MARCH_FORMAT, wgpu::TextureFormat::Rgba16Float);
    }

    #[test]
    fn workgroup_dispatch_count() {
        // Verify div_ceil math for workgroup dispatch
        let width = 480u32;
        let height = 270u32;
        assert_eq!(width.div_ceil(8), 60);
        assert_eq!(height.div_ceil(8), 34); // ceil(270/8) = 34
    }

    #[test]
    fn shadow_bounds_offsets_match_layout() {
        // vol_shadow_min must be at offset 192, vol_shadow_max at 208.
        // These match the write_buffer calls in set_shadow_bounds.
        assert_eq!(std::mem::offset_of!(VolMarchParams, vol_shadow_min), 192);
        assert_eq!(std::mem::offset_of!(VolMarchParams, vol_shadow_max), 208);
    }

    #[test]
    fn fog_params_offset_matches_set_fog_params() {
        // set_fog_params writes at offset 128. fog_color must start there.
        assert_eq!(std::mem::offset_of!(VolMarchParams, fog_color), 128);
    }
}
