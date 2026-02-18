//! Radiance volume for voxel cone tracing global illumination.
//!
//! A 4-level clipmap of 3D textures stores pre-integrated radiance (RGB) and
//! opacity (A) at progressively coarser resolutions, enabling efficient cone
//! tracing at any distance from the camera.
//!
//! | Level | Voxel Size | Coverage  | Memory |
//! |-------|-----------|-----------|--------|
//! | 0     | ~4 cm     | ~5 m      | 16 MB  |
//! | 1     | ~16 cm    | ~20 m     | 16 MB  |
//! | 2     | ~64 cm    | ~80 m     | 16 MB  |
//! | 3     | ~256 cm   | ~320 m    | 16 MB  |

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Number of clipmap levels in the radiance volume.
pub const RADIANCE_LEVELS: usize = 4;

/// Dimension (width = height = depth) of each radiance volume level.
pub const RADIANCE_DIM: u32 = 128;

/// Texture format for all radiance volume levels (RGBA, half-precision float).
pub const RADIANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Voxel sizes (in metres) for each radiance volume level \[L0, L1, L2, L3\].
pub const RADIANCE_VOXEL_SIZES: [f32; 4] = [0.04, 0.16, 0.64, 2.56];

/// Uniform data for the radiance volume, uploaded once per frame when the
/// camera moves significantly.
///
/// Layout is `repr(C)` and exactly 64 bytes so it can be bound as a uniform
/// buffer and read from WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RadianceVolumeUniforms {
    /// Centre of the radiance volume in world space (xyz) + unused (w).
    pub center: [f32; 4],
    /// Voxel size per level \[L0, L1, L2, L3\].
    pub voxel_sizes: [f32; 4],
    /// Inverse full extent per level: `1.0 / (voxel_size * RADIANCE_DIM)`.
    pub inv_extents: [f32; 4],
    /// `[volume_dim as u32, num_levels as u32, 0, 0]`.
    pub params: [u32; 4],
}

/// 4-level clipmap of 3D radiance textures for voxel cone tracing GI.
///
/// Each level is an independent 128³ `Rgba16Float` texture. The shade pass
/// samples from all four levels simultaneously using the `read_bind_group`.
pub struct RadianceVolume {
    /// One 128³ `Rgba16Float` texture per clipmap level.
    pub textures: [wgpu::Texture; RADIANCE_LEVELS],
    /// One view per level (used for both storage writes and sampled reads).
    pub views: [wgpu::TextureView; RADIANCE_LEVELS],
    /// Trilinear sampler shared across all levels.
    pub sampler: wgpu::Sampler,
    /// Uniform buffer holding [`RadianceVolumeUniforms`].
    pub uniform_buffer: wgpu::Buffer,
    /// Bind group layout for the shade pass (group 6) — 4 sampled textures,
    /// 1 sampler, 1 uniform buffer.
    pub read_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the shade pass.
    pub read_bind_group: wgpu::BindGroup,
}

impl RadianceVolume {
    /// Create the radiance volume: allocate all GPU resources and build bind
    /// groups.  No data is written to the textures yet — the radiance inject
    /// pass will fill them each frame.
    pub fn new(device: &wgpu::Device) -> Self {
        let extent = wgpu::Extent3d {
            width: RADIANCE_DIM,
            height: RADIANCE_DIM,
            depth_or_array_layers: RADIANCE_DIM,
        };

        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

        let textures: [wgpu::Texture; RADIANCE_LEVELS] = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("radiance L{i}")),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: RADIANCE_FORMAT,
                usage,
                view_formats: &[],
            })
        });

        let views: [wgpu::TextureView; RADIANCE_LEVELS] =
            std::array::from_fn(|i| textures[i].create_view(&Default::default()));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("radiance sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = Self::default_uniforms();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("radiance uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let read_bind_group_layout = Self::create_read_bind_group_layout(device);
        let read_bind_group = Self::create_read_bind_group(
            device,
            &read_bind_group_layout,
            &views,
            &sampler,
            &uniform_buffer,
        );

        Self {
            textures,
            views,
            sampler,
            uniform_buffer,
            read_bind_group_layout,
            read_bind_group,
        }
    }

    /// Update the centre position stored in the uniform buffer.
    ///
    /// Call this whenever the camera moves far enough to require a clipmap
    /// re-centre.  The radiance inject pass reads this to know where to write.
    pub fn update_center(&self, queue: &wgpu::Queue, center: [f32; 3]) {
        // Only the first three components (xyz) change; w stays 0.
        let offset = 0u64;
        let data: [f32; 4] = [center[0], center[1], center[2], 0.0];
        queue.write_buffer(
            &self.uniform_buffer,
            offset,
            bytemuck::bytes_of(&data),
        );
    }

    /// Build the default [`RadianceVolumeUniforms`] with a zero centre.
    fn default_uniforms() -> RadianceVolumeUniforms {
        let voxel_sizes = RADIANCE_VOXEL_SIZES;
        let inv_extents = std::array::from_fn::<f32, 4, _>(|i| {
            1.0 / (voxel_sizes[i] * RADIANCE_DIM as f32)
        });

        RadianceVolumeUniforms {
            center: [0.0; 4],
            voxel_sizes,
            inv_extents,
            params: [RADIANCE_DIM, RADIANCE_LEVELS as u32, 0, 0],
        }
    }

    /// Create the bind group layout for the shade pass (group 6).
    ///
    /// Bindings:
    /// - 0–3: `texture_3d<f32>` filterable (one per clipmap level)
    /// - 4: `sampler(filtering)` trilinear
    /// - 5: `uniform` buffer ([`RadianceVolumeUniforms`])
    fn create_read_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("radiance read layout"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                // binding 4: trilinear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 5: uniform buffer
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
            ],
        })
    }

    /// Assemble the read bind group from the four level views, sampler, and
    /// uniform buffer.
    fn create_read_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        views: &[wgpu::TextureView; RADIANCE_LEVELS],
        sampler: &wgpu::Sampler,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("radiance read bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_size() {
        assert_eq!(std::mem::size_of::<RadianceVolumeUniforms>(), 64);
    }

    #[test]
    fn test_uniforms_default_values() {
        let u = RadianceVolume::default_uniforms();

        // voxel_sizes must match the public constant exactly
        assert_eq!(u.voxel_sizes, RADIANCE_VOXEL_SIZES);

        // inv_extents: 1 / (voxel_size * 128)
        for i in 0..4 {
            let expected = 1.0 / (RADIANCE_VOXEL_SIZES[i] * RADIANCE_DIM as f32);
            assert!(
                (u.inv_extents[i] - expected).abs() < 1e-10,
                "inv_extents[{i}] mismatch: got {}, expected {expected}",
                u.inv_extents[i]
            );
        }

        // params: [128, 4, 0, 0]
        assert_eq!(u.params[0], RADIANCE_DIM);
        assert_eq!(u.params[1], RADIANCE_LEVELS as u32);
        assert_eq!(u.params[2], 0);
        assert_eq!(u.params[3], 0);

        // centre starts at zero
        assert_eq!(u.center, [0.0; 4]);
    }

    #[test]
    fn test_constants() {
        assert_eq!(RADIANCE_DIM, 128);
        assert_eq!(RADIANCE_LEVELS, 4);
        assert_eq!(
            RADIANCE_VOXEL_SIZES,
            [0.04_f32, 0.16_f32, 0.64_f32, 2.56_f32]
        );
    }
}
