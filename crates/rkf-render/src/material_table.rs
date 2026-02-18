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
            usage: wgpu::BufferUsages::STORAGE,
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
}

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
        // 2: Metal — polished silver
        Material {
            albedo: [0.9, 0.9, 0.92],
            roughness: 0.15,
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
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_materials_count() {
        let mats = create_test_materials();
        assert_eq!(mats.len(), 10);
    }

    #[test]
    fn test_materials_sizes() {
        let mats = create_test_materials();
        let bytes: &[u8] = bytemuck::cast_slice(&mats);
        assert_eq!(bytes.len(), 10 * 96);
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
        assert!(metal.roughness < 0.3);
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
}
