//! G-buffer textures for deferred shading.
//!
//! The ray march pass writes per-pixel data to 4 render targets at internal
//! resolution. The shading pass reads these to compute final lighting.
//!
//! | Target | Format       | Content                                       |
//! |--------|-------------|-----------------------------------------------|
//! | 0      | Rgba32Float | position.xyz + hit_distance                   |
//! | 1      | Rgba16Float | normal.xyz + material_blend_weight            |
//! | 2      | R32Uint     | packed: material_id(lo16) + sec_id_flags(hi16)|
//! | 3      | Rg32Float   | motion_vector.xy (zeros for now)              |

/// G-buffer: 4 render targets at internal resolution for deferred shading.
pub struct GBuffer {
    /// Target 0: world position (xyz) + hit distance (w). `Rgba32Float`.
    pub position_texture: wgpu::Texture,
    /// View for target 0.
    pub position_view: wgpu::TextureView,

    /// Target 1: normal (xyz) + blend weight (w). `Rgba16Float`.
    pub normal_texture: wgpu::Texture,
    /// View for target 1.
    pub normal_view: wgpu::TextureView,

    /// Target 2: packed material_id (lo16) + secondary_id_and_flags (hi16). `R32Uint`.
    pub material_texture: wgpu::Texture,
    /// View for target 2.
    pub material_view: wgpu::TextureView,

    /// Target 3: motion vector (xy). `Rg16Float`.
    pub motion_texture: wgpu::Texture,
    /// View for target 3.
    pub motion_view: wgpu::TextureView,

    /// Bind group layout for writing (storage textures, used by ray march).
    pub write_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for writing.
    pub write_bind_group: wgpu::BindGroup,

    /// Bind group layout for reading (sampled textures, used by shading pass).
    pub read_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for reading.
    pub read_bind_group: wgpu::BindGroup,

    /// Internal resolution width.
    pub width: u32,
    /// Internal resolution height.
    pub height: u32,
}

/// Texture format for G-buffer target 0 (position + hit distance).
pub const GBUFFER_POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Texture format for G-buffer target 1 (normal + blend weight).
pub const GBUFFER_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Texture format for G-buffer target 2 (packed material IDs).
pub const GBUFFER_MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// Texture format for G-buffer target 3 (motion vectors).
pub const GBUFFER_MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

impl GBuffer {
    /// Create the G-buffer with 4 render targets at the given resolution.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

        // Target 0: position + hit_distance
        let position_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer position"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GBUFFER_POSITION_FORMAT,
            usage,
            view_formats: &[],
        });
        let position_view = position_texture.create_view(&Default::default());

        // Target 1: normal + blend_weight
        let normal_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer normal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GBUFFER_NORMAL_FORMAT,
            usage,
            view_formats: &[],
        });
        let normal_view = normal_texture.create_view(&Default::default());

        // Target 2: material_id + secondary_id_and_flags
        let material_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer material"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GBUFFER_MATERIAL_FORMAT,
            usage,
            view_formats: &[],
        });
        let material_view = material_texture.create_view(&Default::default());

        // Target 3: motion vectors
        let motion_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer motion"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GBUFFER_MOTION_FORMAT,
            usage,
            view_formats: &[],
        });
        let motion_view = motion_texture.create_view(&Default::default());

        // Write bind group layout (4 storage textures, write-only)
        let write_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gbuffer write layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: GBUFFER_POSITION_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: GBUFFER_NORMAL_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: GBUFFER_MATERIAL_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: GBUFFER_MOTION_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let write_bind_group = Self::create_write_bind_group(
            device,
            &write_bind_group_layout,
            &position_view,
            &normal_view,
            &material_view,
            &motion_view,
        );

        // Read bind group layout (4 sampled textures for shading pass)
        let read_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gbuffer read layout"),
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
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let read_bind_group = Self::create_read_bind_group(
            device,
            &read_bind_group_layout,
            &position_view,
            &normal_view,
            &material_view,
            &motion_view,
        );

        Self {
            position_texture,
            position_view,
            normal_texture,
            normal_view,
            material_texture,
            material_view,
            motion_texture,
            motion_view,
            write_bind_group_layout,
            write_bind_group,
            read_bind_group_layout,
            read_bind_group,
            width,
            height,
        }
    }

    fn create_write_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        motion_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer write bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(motion_view),
                },
            ],
        })
    }

    fn create_read_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        position_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        material_view: &wgpu::TextureView,
        motion_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer read bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(position_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(motion_view),
                },
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbuffer_format_constants() {
        assert_eq!(GBUFFER_POSITION_FORMAT, wgpu::TextureFormat::Rgba32Float);
        assert_eq!(GBUFFER_NORMAL_FORMAT, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(GBUFFER_MATERIAL_FORMAT, wgpu::TextureFormat::R32Uint);
        assert_eq!(GBUFFER_MOTION_FORMAT, wgpu::TextureFormat::Rg32Float);
    }
}
