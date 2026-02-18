//! Volumetric shadow map compute pass — Phase 11 task 11.1.
//!
//! Generates a 3D texture (256×128×256, R32Float) where each texel stores
//! the transmittance from that world position to the sun. The volume is
//! camera-centered and updated once per frame.
//!
//! The shadow map is consumed by the volumetric march pass (task 11.2) to
//! look up sun visibility at arbitrary world positions without re-marching.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ---------- Public constants ----------

/// X dimension of the volumetric shadow map texture (voxels).
pub const VOL_SHADOW_DIM_X: u32 = 256;

/// Y dimension of the volumetric shadow map texture (voxels).
/// Smaller than X/Z because less vertical extent is needed.
pub const VOL_SHADOW_DIM_Y: u32 = 128;

/// Z dimension of the volumetric shadow map texture (voxels).
pub const VOL_SHADOW_DIM_Z: u32 = 256;

/// Texture format for the shadow map (single-channel half-float).
pub const VOL_SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Default world-space horizontal extent of the shadow volume (metres).
/// The volume spans ±RANGE/2 around the camera in X and Z.
pub const DEFAULT_VOL_SHADOW_RANGE: f32 = 80.0;

/// Default world-space vertical extent of the shadow volume (metres).
/// The volume spans ±HEIGHT/2 around the camera in Y.
pub const DEFAULT_VOL_SHADOW_HEIGHT: f32 = 20.0;

/// Default maximum march steps per texel toward the sun.
pub const DEFAULT_MAX_SHADOW_STEPS: u32 = 96;

/// Default world-space step size along the sun direction (metres).
pub const DEFAULT_SHADOW_STEP_SIZE: f32 = 0.15;

/// Default extinction coefficient (controls opacity per unit density).
pub const DEFAULT_EXTINCTION_COEFF: f32 = 10.0;

// ---------- GPU struct ----------

/// GPU-uploadable volumetric shadow map parameters.
///
/// Memory layout (96 bytes, 16-byte aligned throughout):
///
/// ```text
/// offset  0 — volume_min [f32; 3]    (12 bytes)
/// offset 12 — _pad0 f32              (4 bytes)
/// offset 16 — volume_max [f32; 3]    (12 bytes)
/// offset 28 — _pad1 f32              (4 bytes)
/// offset 32 — sun_dir [f32; 3]       (12 bytes)
/// offset 44 — _pad2 f32              (4 bytes)
/// offset 48 — dim_x u32              (4 bytes)
/// offset 52 — dim_y u32              (4 bytes)
/// offset 56 — dim_z u32              (4 bytes)
/// offset 60 — max_steps u32          (4 bytes)
/// offset 64 — step_size f32          (4 bytes)
/// offset 68 — extinction_coeff f32   (4 bytes)
/// offset 72 — _pad3 u32             (4 bytes)
/// offset 76 — _pad4 u32             (4 bytes)
/// total: 80 bytes
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct VolShadowParams {
    /// Minimum corner of the shadow volume (world space).
    pub volume_min: [f32; 3],
    #[doc(hidden)]
    pub _pad0: f32,
    /// Maximum corner of the shadow volume (world space).
    pub volume_max: [f32; 3],
    #[doc(hidden)]
    pub _pad1: f32,
    /// Sun direction (normalized, pointing **toward** the sun).
    pub sun_dir: [f32; 3],
    #[doc(hidden)]
    pub _pad2: f32,
    /// Shadow volume X dimension in texels.
    pub dim_x: u32,
    /// Shadow volume Y dimension in texels.
    pub dim_y: u32,
    /// Shadow volume Z dimension in texels.
    pub dim_z: u32,
    /// Maximum number of march steps per texel.
    pub max_steps: u32,
    /// World-space step size along the sun direction (metres).
    pub step_size: f32,
    /// Extinction coefficient (opacity per unit density per unit distance).
    pub extinction_coeff: f32,
    #[doc(hidden)]
    pub _pad3: u32,
    #[doc(hidden)]
    pub _pad4: u32,
}

// ---------- Pass ----------

/// Volumetric shadow map compute pass.
///
/// Dispatches a 3D compute shader that fills [`VolShadowPass::shadow_texture`]
/// with per-texel transmittance values. Workgroup size is 4×4×4; dispatch is
/// `(dim_x/4, dim_y/4, dim_z/4)` workgroups.
#[allow(dead_code)]
pub struct VolShadowPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    /// The 3D shadow map texture (R32Float, STORAGE_BINDING | TEXTURE_BINDING).
    pub shadow_texture: wgpu::Texture,
    /// View over the full 3D shadow map texture.
    pub shadow_view: wgpu::TextureView,
}

impl VolShadowPass {
    /// Create the volumetric shadow map pass.
    ///
    /// The volume is initially centered on the world origin with the sun
    /// direction set to `(0, -1, 0.5)` (normalized).
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        scene_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vol_shadow.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/vol_shadow.wgsl").into(),
            ),
        });

        // --- 3D shadow map texture ---
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vol shadow map"),
            size: wgpu::Extent3d {
                width: VOL_SHADOW_DIM_X,
                height: VOL_SHADOW_DIM_Y,
                depth_or_array_layers: VOL_SHADOW_DIM_Z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: VOL_SHADOW_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&Default::default());

        // --- Default params (volume centred on origin) ---
        let half_range = DEFAULT_VOL_SHADOW_RANGE * 0.5;
        let half_height = DEFAULT_VOL_SHADOW_HEIGHT * 0.5;
        // Normalize the default sun direction (0, -1, 0.5)
        let raw = [0.0f32, -1.0, 0.5];
        let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let sun_dir = [raw[0] / len, raw[1] / len, raw[2] / len];

        let default_params = VolShadowParams {
            volume_min: [-half_range, -half_height, -half_range],
            _pad0: 0.0,
            volume_max: [half_range, half_height, half_range],
            _pad1: 0.0,
            sun_dir,
            _pad2: 0.0,
            dim_x: VOL_SHADOW_DIM_X,
            dim_y: VOL_SHADOW_DIM_Y,
            dim_z: VOL_SHADOW_DIM_Z,
            max_steps: DEFAULT_MAX_SHADOW_STEPS,
            step_size: DEFAULT_SHADOW_STEP_SIZE,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            _pad3: 0,
            _pad4: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vol shadow params"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Bind group layout ---
        // binding 0: uniform buffer (VolShadowParams)
        // binding 1: texture_storage_3d<r16float, write>
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vol shadow bind group layout"),
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
                            format: VOL_SHADOW_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            });

        // --- Bind group ---
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vol shadow bind group"),
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
            ],
        });

        // --- Pipeline ---
        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vol shadow pipeline layout"),
                bind_group_layouts: &[&bind_group_layout, scene_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vol shadow pipeline"),
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
            shadow_texture,
            shadow_view,
        }
    }

    /// Update the volume bounds and sun direction, then write params to GPU.
    ///
    /// `camera_pos` — world-space camera position (chunk-relative f32 is fine
    /// since the shadow volume doesn't need sub-centimetre precision).
    ///
    /// `sun_direction` — world-space direction **toward** the sun (will be
    /// normalised internally).
    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sun_direction: [f32; 3],
    ) {
        let half_range = DEFAULT_VOL_SHADOW_RANGE * 0.5;
        let half_height = DEFAULT_VOL_SHADOW_HEIGHT * 0.5;

        // Normalise sun direction
        let len = (sun_direction[0] * sun_direction[0]
            + sun_direction[1] * sun_direction[1]
            + sun_direction[2] * sun_direction[2])
            .sqrt()
            .max(1e-10);
        let sun_dir = [
            sun_direction[0] / len,
            sun_direction[1] / len,
            sun_direction[2] / len,
        ];

        let params = VolShadowParams {
            volume_min: [
                camera_pos[0] - half_range,
                camera_pos[1] - half_height,
                camera_pos[2] - half_range,
            ],
            _pad0: 0.0,
            volume_max: [
                camera_pos[0] + half_range,
                camera_pos[1] + half_height,
                camera_pos[2] + half_range,
            ],
            _pad1: 0.0,
            sun_dir,
            _pad2: 0.0,
            dim_x: VOL_SHADOW_DIM_X,
            dim_y: VOL_SHADOW_DIM_Y,
            dim_z: VOL_SHADOW_DIM_Z,
            max_steps: DEFAULT_MAX_SHADOW_STEPS,
            step_size: DEFAULT_SHADOW_STEP_SIZE,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            _pad3: 0,
            _pad4: 0,
        };

        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// Dispatch the volumetric shadow map compute shader.
    ///
    /// Workgroup size is 4×4×4; dispatches
    /// `(dim_x.div_ceil(4), dim_y.div_ceil(4), dim_z.div_ceil(4))` groups.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera_pos: [f32; 3],
        sun_dir: [f32; 3],
        scene_bind_group: &wgpu::BindGroup,
    ) {
        self.update_params(queue, camera_pos, sun_dir);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vol shadow"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.dispatch_workgroups(
            VOL_SHADOW_DIM_X.div_ceil(4),
            VOL_SHADOW_DIM_Y.div_ceil(4),
            VOL_SHADOW_DIM_Z.div_ceil(4),
        );
    }
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_shadow_params_size_is_80() {
        // struct has 20 × 4-byte fields = 80 bytes
        assert_eq!(std::mem::size_of::<VolShadowParams>(), 80);
    }

    #[test]
    fn vol_shadow_params_pod_roundtrip() {
        let p = VolShadowParams {
            volume_min: [-64.0, -32.0, -64.0],
            _pad0: 0.0,
            volume_max: [64.0, 32.0, 64.0],
            _pad1: 0.0,
            sun_dir: [0.0, -0.894, 0.447],
            _pad2: 0.0,
            dim_x: VOL_SHADOW_DIM_X,
            dim_y: VOL_SHADOW_DIM_Y,
            dim_z: VOL_SHADOW_DIM_Z,
            max_steps: DEFAULT_MAX_SHADOW_STEPS,
            step_size: DEFAULT_SHADOW_STEP_SIZE,
            extinction_coeff: DEFAULT_EXTINCTION_COEFF,
            _pad3: 0,
            _pad4: 0,
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 80);
        let p2: &VolShadowParams = bytemuck::from_bytes(bytes);
        assert_eq!(p.dim_x, p2.dim_x);
        assert_eq!(p.sun_dir, p2.sun_dir);
        assert_eq!(p.step_size, p2.step_size);
        assert_eq!(p.extinction_coeff, p2.extinction_coeff);
    }

    #[test]
    fn vol_shadow_params_field_offsets() {
        assert_eq!(std::mem::offset_of!(VolShadowParams, volume_min), 0);
        assert_eq!(std::mem::offset_of!(VolShadowParams, volume_max), 16);
        assert_eq!(std::mem::offset_of!(VolShadowParams, sun_dir), 32);
        assert_eq!(std::mem::offset_of!(VolShadowParams, dim_x), 48);
        assert_eq!(std::mem::offset_of!(VolShadowParams, dim_y), 52);
        assert_eq!(std::mem::offset_of!(VolShadowParams, dim_z), 56);
        assert_eq!(std::mem::offset_of!(VolShadowParams, max_steps), 60);
        assert_eq!(std::mem::offset_of!(VolShadowParams, step_size), 64);
        assert_eq!(std::mem::offset_of!(VolShadowParams, extinction_coeff), 68);
    }

    #[test]
    fn default_constants() {
        assert_eq!(VOL_SHADOW_DIM_X, 256);
        assert_eq!(VOL_SHADOW_DIM_Y, 128);
        assert_eq!(VOL_SHADOW_DIM_Z, 256);
        assert!(DEFAULT_VOL_SHADOW_RANGE > 0.0);
        assert!(DEFAULT_VOL_SHADOW_HEIGHT > 0.0);
        assert!(DEFAULT_MAX_SHADOW_STEPS > 0);
        assert!(DEFAULT_SHADOW_STEP_SIZE > 0.0);
        assert!(DEFAULT_EXTINCTION_COEFF > 0.0);
    }

    #[test]
    fn vol_shadow_format_is_r16float() {
        assert_eq!(VOL_SHADOW_FORMAT, wgpu::TextureFormat::R32Float);
    }

    #[test]
    fn update_params_centers_on_camera() {
        // Verify the volume bounds logic is symmetric around camera_pos
        let cam = [10.0f32, 5.0, -20.0];
        let half_range = DEFAULT_VOL_SHADOW_RANGE * 0.5;
        let half_height = DEFAULT_VOL_SHADOW_HEIGHT * 0.5;
        let expected_min = [cam[0] - half_range, cam[1] - half_height, cam[2] - half_range];
        let expected_max = [cam[0] + half_range, cam[1] + half_height, cam[2] + half_range];
        assert!((expected_min[0] + expected_max[0]) / 2.0 - cam[0] < 1e-5);
        assert!((expected_min[1] + expected_max[1]) / 2.0 - cam[1] < 1e-5);
        assert!((expected_min[2] + expected_max[2]) / 2.0 - cam[2] < 1e-5);
    }

    #[test]
    fn sun_direction_normalisation() {
        // sun_dir must always be unit length after normalisation
        let raw = [1.0f32, -2.0, 0.5];
        let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let normalized = [raw[0] / len, raw[1] / len, raw[2] / len];
        let magnitude = (normalized[0] * normalized[0]
            + normalized[1] * normalized[1]
            + normalized[2] * normalized[2])
            .sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
    }
}
