//! GPU material table — uploads `Material` array as a storage buffer.
//!
//! The shading pass indexes into this table using the `material_id` from the
//! G-buffer to resolve PBR properties per pixel.

use rkf_core::material::Material;
use wgpu::util::DeviceExt;

/// GPU-resident material table with bind group for the shading pass.
pub struct MaterialTable {
    /// Storage buffer containing the material array.
    pub buffer: wgpu::Buffer,
    /// Bind group layout (single storage buffer).
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group binding the material buffer.
    pub bind_group: wgpu::BindGroup,
    /// Number of materials in the table.
    pub count: u32,
}

impl MaterialTable {
    /// Upload a slice of materials to the GPU.
    pub fn upload(device: &wgpu::Device, materials: &[Material]) -> Self {
        let bytes: &[u8] = bytemuck::cast_slice(materials);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material table"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material table layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material table bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            buffer,
            bind_group_layout,
            bind_group,
            count: materials.len() as u32,
        }
    }

    /// Update the GPU material table from a new slice of materials.
    ///
    /// If the material count hasn't changed, this uses `queue.write_buffer()` (fast path).
    /// If the count has changed, it recreates the buffer and bind group.
    ///
    /// Returns `true` if the buffer was recreated (callers must rebind affected passes).
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, materials: &[Material]) -> bool {
        let new_count = materials.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(materials);

        if new_count == self.count {
            // Fast path: same size, just write data.
            queue.write_buffer(&self.buffer, 0, bytes);
            false
        } else {
            // Buffer size changed — recreate buffer + bind group.
            self.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material table"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material table bind group"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                }],
            });
            self.count = new_count;
            true
        }
    }
}

#[deprecated(note = "Use MaterialLibrary::load_palette() to load materials from .rkmat files instead")]
/// Create a set of test materials for the Phase 6 materials showcase and Cornell box GI scene.
///
/// | Index | Name           | Description                                    |
/// |------:|----------------|------------------------------------------------|
/// |     0 | Default        | Medium gray, fallback                          |
/// |     1 | Stone          | Gray, rough, dielectric                        |
/// |     2 | Metal          | Silver, low roughness, metallic                |
/// |     3 | Wood           | Warm brown, moderate roughness                 |
/// |     4 | Emissive       | Bright cyan glow                               |
/// |     5 | Skin           | Warm tone, subsurface scattering               |
/// |     6 | White Diffuse  | High albedo for good GI bouncing               |
/// |     7 | Red Diffuse    | Cornell box left wall                          |
/// |     8 | Green Diffuse  | Cornell box right wall                         |
/// |     9 | Ceiling Light  | Warm white emissive ceiling panel              |
pub fn create_test_materials() -> Vec<Material> {
    vec![
        // 0: Default (fallback)
        Material::default(),
        // 1: Stone — gray, rough dielectric
        Material {
            albedo: [0.45, 0.43, 0.40],
            roughness: 0.85,
            metallic: 0.0,
            ..Default::default()
        },
        // 2: Metal — brushed silver
        Material {
            albedo: [0.9, 0.9, 0.92],
            roughness: 0.35,
            metallic: 1.0,
            ..Default::default()
        },
        // 3: Wood — warm brown
        Material {
            albedo: [0.55, 0.35, 0.18],
            roughness: 0.65,
            metallic: 0.0,
            ..Default::default()
        },
        // 4: Emissive — cyan glow
        Material {
            albedo: [0.05, 0.05, 0.05],
            roughness: 0.5,
            metallic: 0.0,
            emission_color: [0.2, 0.8, 1.0],
            emission_strength: 5.0,
            ..Default::default()
        },
        // 5: Skin — warm with SSS
        Material {
            albedo: [0.8, 0.6, 0.5],
            roughness: 0.45,
            metallic: 0.0,
            subsurface: 0.6,
            subsurface_color: [1.0, 0.4, 0.25],
            ..Default::default()
        },
        // 6: White diffuse — high albedo for good GI bouncing
        Material {
            albedo: [0.85, 0.85, 0.85],
            roughness: 0.9,
            metallic: 0.0,
            ..Default::default()
        },
        // 7: Red diffuse — Cornell box left wall
        Material {
            albedo: [0.75, 0.1, 0.1],
            roughness: 0.9,
            metallic: 0.0,
            ..Default::default()
        },
        // 8: Green diffuse — Cornell box right wall
        Material {
            albedo: [0.1, 0.75, 0.1],
            roughness: 0.9,
            metallic: 0.0,
            ..Default::default()
        },
        // 9: Ceiling light emissive — warm white
        Material {
            albedo: [0.1, 0.1, 0.1],
            roughness: 0.5,
            metallic: 0.0,
            emission_color: [1.0, 0.95, 0.85],
            emission_strength: 8.0,
            ..Default::default()
        },
        // 10: Dirt — brownish, rough (blend target for stone)
        Material {
            albedo: [0.35, 0.25, 0.15],
            roughness: 0.95,
            metallic: 0.0,
            ..Default::default()
        },
        // 11: Gold — warm metallic (blend target for silver metal)
        Material {
            albedo: [1.0, 0.84, 0.0],
            roughness: 0.25,
            metallic: 1.0,
            ..Default::default()
        },
        // 12: Noisy Stone — stone with albedo+roughness noise
        Material {
            albedo: [0.45, 0.43, 0.40],
            roughness: 0.85,
            metallic: 0.0,
            noise_scale: 5.0,
            noise_strength: 0.3,
            noise_channels: rkf_core::material::NOISE_CHANNEL_ALBEDO
                | rkf_core::material::NOISE_CHANNEL_ROUGHNESS,
            ..Default::default()
        },
        // 13: Bumpy Metal — metal with normal perturbation
        Material {
            albedo: [0.8, 0.75, 0.7],
            roughness: 0.3,
            metallic: 1.0,
            noise_scale: 10.0,
            noise_strength: 0.15,
            noise_channels: rkf_core::material::NOISE_CHANNEL_NORMAL,
            ..Default::default()
        },
    ]
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_materials_count() {
        let mats = create_test_materials();
        assert_eq!(mats.len(), 14);
    }

    #[test]
    fn test_materials_sizes() {
        let mats = create_test_materials();
        let bytes: &[u8] = bytemuck::cast_slice(&mats);
        assert_eq!(bytes.len(), 14 * 96);
    }

    #[test]
    fn stone_is_rough_dielectric() {
        let mats = create_test_materials();
        let stone = &mats[1];
        assert!(stone.roughness > 0.5);
        assert_eq!(stone.metallic, 0.0);
    }

    #[test]
    fn metal_is_shiny_conductor() {
        let mats = create_test_materials();
        let metal = &mats[2];
        assert!(metal.roughness < 0.5);
        assert_eq!(metal.metallic, 1.0);
    }

    #[test]
    fn emissive_has_emission() {
        let mats = create_test_materials();
        let emissive = &mats[4];
        assert!(emissive.emission_strength > 0.0);
    }

    #[test]
    fn skin_has_subsurface() {
        let mats = create_test_materials();
        let skin = &mats[5];
        assert!(skin.subsurface > 0.0);
    }

    #[test]
    fn blend_target_materials_exist() {
        let mats = create_test_materials();
        // Dirt (10) is rough dielectric
        assert!(mats[10].roughness > 0.9);
        assert_eq!(mats[10].metallic, 0.0);
        // Gold (11) is metallic
        assert_eq!(mats[11].metallic, 1.0);
    }

    #[test]
    fn material_blend_cpu_side() {
        let mats = create_test_materials();
        let stone = &mats[1]; // roughness 0.85
        let dirt = &mats[10]; // roughness 0.95
        // At weight=0.5, blended roughness should be midpoint
        let blended_roughness = stone.roughness * (1.0 - 0.5) + dirt.roughness * 0.5;
        assert!((blended_roughness - 0.9).abs() < 0.01);
    }

    #[test]
    fn material_blend_weight_zero_returns_primary() {
        let mats = create_test_materials();
        let stone = &mats[1];
        let weight = 0.0_f32;
        let blended_roughness = stone.roughness * (1.0 - weight) + 0.0 * weight;
        assert!((blended_roughness - stone.roughness).abs() < f32::EPSILON);
    }

    #[test]
    fn material_blend_weight_one_returns_secondary() {
        let mats = create_test_materials();
        let dirt = &mats[10];
        let stone = &mats[1];
        let weight = 1.0_f32;
        let blended_roughness = stone.roughness * (1.0 - weight) + dirt.roughness * weight;
        assert!((blended_roughness - dirt.roughness).abs() < f32::EPSILON);
    }

    #[test]
    fn noisy_stone_has_noise_channels() {
        let mats = create_test_materials();
        let noisy = &mats[12];
        assert!(noisy.noise_scale > 0.0);
        assert!(noisy.noise_strength > 0.0);
        assert!(noisy.noise_channels & rkf_core::material::NOISE_CHANNEL_ALBEDO != 0);
        assert!(noisy.noise_channels & rkf_core::material::NOISE_CHANNEL_ROUGHNESS != 0);
        assert!(noisy.noise_channels & rkf_core::material::NOISE_CHANNEL_NORMAL == 0);
    }

    #[test]
    fn bumpy_metal_has_normal_noise() {
        let mats = create_test_materials();
        let bumpy = &mats[13];
        assert!(bumpy.noise_channels & rkf_core::material::NOISE_CHANNEL_NORMAL != 0);
        assert!(bumpy.noise_channels & rkf_core::material::NOISE_CHANNEL_ALBEDO == 0);
    }

    #[test]
    fn non_noisy_materials_have_zero_noise() {
        let mats = create_test_materials();
        // Materials 0-9 should all have noise_channels = 0
        for i in 0..10 {
            assert_eq!(mats[i].noise_channels, 0, "material {i} should have no noise");
        }
    }
}
