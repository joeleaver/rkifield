//! Material transfer: per-voxel color from mesh textures via BVH lookup.
//!
//! During voxelization, each voxel near the mesh surface needs a material ID
//! and optionally a per-voxel color sampled from the source mesh's textures.
//! This module provides:
//!
//! - [`sample_material`]: find nearest triangle, extract material + texture color
//! - [`sample_texture`]: bilinear-filtered texture lookup at UV coordinates
//! - [`map_material_properties`]: extract PBR properties from [`ImportMaterial`]

use glam::Vec3;

use crate::bvh::TriangleBvh;
use crate::mesh::{ImportMaterial, MeshData, TextureData};

/// Per-voxel color data (RGBA8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxelColor {
    /// Red channel [0, 255].
    pub r: u8,
    /// Green channel [0, 255].
    pub g: u8,
    /// Blue channel [0, 255].
    pub b: u8,
    /// Alpha channel [0, 255].
    pub a: u8,
}

/// Result of material transfer for a single voxel.
#[derive(Debug, Clone, Copy)]
pub struct MaterialSample {
    /// Engine material ID (index into material table).
    pub material_id: u16,
    /// Per-voxel color from texture sampling (if available).
    pub color: Option<VoxelColor>,
}

/// Sample material and color for a voxel at the given world position.
///
/// 1. Find nearest triangle via BVH
/// 2. Get material index from triangle
/// 3. Interpolate UV from barycentric coordinates
/// 4. Sample albedo texture at UV (if present), or use base color as fallback
pub fn sample_material(mesh: &MeshData, bvh: &TriangleBvh, world_pos: Vec3) -> MaterialSample {
    let nearest = bvh.nearest(world_pos);

    // Material ID from triangle's material index
    let material_id = if nearest.triangle_index < mesh.material_indices.len() {
        mesh.material_indices[nearest.triangle_index] as u16
    } else {
        0
    };

    // Sample texture color at the nearest point on the triangle
    let color = sample_texture_at_triangle(mesh, nearest.triangle_index, &nearest.barycentric);

    MaterialSample { material_id, color }
}

/// Sample texture color at a point on a triangle using barycentric interpolation.
///
/// Returns `None` only if there are no UVs AND no materials. Otherwise returns
/// the texture sample or the material's base color as fallback.
///
/// This is useful when you already have BVH query results (triangle index +
/// barycentric coords) and want to avoid a redundant BVH lookup.
pub fn sample_texture_at_triangle(
    mesh: &MeshData,
    tri_idx: usize,
    barycentric: &[f32; 3],
) -> Option<VoxelColor> {
    // Get triangle UVs (returns [[0,0]; 3] if no UVs)
    let uvs = mesh.triangle_uvs(tri_idx);

    // Interpolate UV using barycentric coordinates
    let mut u =
        uvs[0][0] * barycentric[0] + uvs[1][0] * barycentric[1] + uvs[2][0] * barycentric[2];
    let mut v =
        uvs[0][1] * barycentric[0] + uvs[1][1] * barycentric[1] + uvs[2][1] * barycentric[2];

    // Get material for this triangle
    let mat_idx = if tri_idx < mesh.material_indices.len() {
        mesh.material_indices[tri_idx] as usize
    } else {
        0
    };

    if mat_idx >= mesh.materials.len() {
        return None;
    }
    let material = &mesh.materials[mat_idx];

    // Apply KHR_texture_transform: final_uv = uv * scale + offset
    let xf = material.uv_transform;
    u = u * xf[2] + xf[0];
    v = v * xf[3] + xf[1];

    // If material has albedo texture and mesh has UVs, sample it
    if let Some(ref tex) = material.albedo_texture {
        if !mesh.uvs.is_empty() {
            return Some(sample_texture(tex, u, v));
        }
    }

    // Use base color as fallback
    Some(VoxelColor {
        r: (material.base_color[0] * 255.0).clamp(0.0, 255.0) as u8,
        g: (material.base_color[1] * 255.0).clamp(0.0, 255.0) as u8,
        b: (material.base_color[2] * 255.0).clamp(0.0, 255.0) as u8,
        a: 255,
    })
}

/// Sample a texture at UV coordinates with nearest-neighbor filtering.
///
/// UVs are wrapped to `[0, 1)` via `rem_euclid` for seamless tiling.
/// Bilinear filtering is an upgrade path.
pub fn sample_texture(tex: &TextureData, u: f32, v: f32) -> VoxelColor {
    if tex.width == 0 || tex.height == 0 || tex.data.is_empty() {
        return VoxelColor {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };
    }

    // Wrap UVs to [0, 1)
    let u = u.rem_euclid(1.0);
    let v = v.rem_euclid(1.0);

    // Pixel coordinates (nearest-neighbor)
    let ix = ((u * tex.width as f32).floor() as u32).min(tex.width - 1);
    let iy = ((v * tex.height as f32).floor() as u32).min(tex.height - 1);
    let idx = ((iy * tex.width + ix) * 4) as usize;

    if idx + 3 < tex.data.len() {
        VoxelColor {
            r: tex.data[idx],
            g: tex.data[idx + 1],
            b: tex.data[idx + 2],
            a: tex.data[idx + 3],
        }
    } else {
        VoxelColor {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        }
    }
}

/// Map an [`ImportMaterial`] to engine material properties.
///
/// Returns `(roughness, metallic, emission_strength)` for creating an engine Material.
pub fn map_material_properties(mat: &ImportMaterial) -> (f32, f32, f32) {
    (mat.roughness, mat.metallic, 0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::TriangleBvh;
    use crate::mesh::{ImportMaterial, MeshData, TextureData};
    use glam::Vec3;

    /// Helper: 2x2 RGBA8 texture with known pixels.
    fn make_test_texture() -> TextureData {
        TextureData {
            width: 2,
            height: 2,
            // Row 0: red, green
            // Row 1: blue, white
            data: vec![
                255, 0, 0, 255, // (0,0) red
                0, 255, 0, 255, // (1,0) green
                0, 0, 255, 255, // (0,1) blue
                255, 255, 255, 255, // (1,1) white
            ],
        }
    }

    /// Helper: single-triangle mesh with UVs and a textured material.
    fn make_textured_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "textured".to_string(),
                base_color: [0.5, 0.5, 0.5],
                metallic: 0.3,
                roughness: 0.7,
                albedo_texture: Some(make_test_texture()),
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    /// Helper: single-triangle mesh without UVs.
    fn make_untextured_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "plain".to_string(),
                base_color: [1.0, 0.0, 0.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    #[test]
    fn voxel_color_default_is_all_zeros() {
        let c = VoxelColor::default();
        assert_eq!(c.r, 0);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 0);
    }

    #[test]
    fn sample_texture_at_origin_returns_top_left_pixel() {
        let tex = make_test_texture();
        let color = sample_texture(&tex, 0.0, 0.0);
        // (0,0) = red pixel
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn sample_texture_wraps_uv_greater_than_one() {
        let tex = make_test_texture();
        // u=1.25 wraps to 0.25, v=0.0 -> still in row 0, column 0
        let color = sample_texture(&tex, 1.25, 0.0);
        // 0.25 * 2 = 0.5 -> floor -> ix=0 -> red pixel
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn sample_texture_wraps_negative_uv() {
        let tex = make_test_texture();
        // u=-0.25 wraps to 0.75, v=0.0
        let color = sample_texture(&tex, -0.25, 0.0);
        // 0.75 * 2 = 1.5 -> floor -> ix=1 -> green pixel
        assert_eq!(color.r, 0);
        assert_eq!(color.g, 255);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn map_material_properties_extracts_roughness_metallic() {
        let mat = ImportMaterial {
            name: "test".to_string(),
            base_color: [1.0, 1.0, 1.0],
            metallic: 0.8,
            roughness: 0.2,
            albedo_texture: None,
        };
        let (roughness, metallic, emission) = map_material_properties(&mat);
        assert!((roughness - 0.2).abs() < 1e-6);
        assert!((metallic - 0.8).abs() < 1e-6);
        assert!((emission - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sample_texture_at_triangle_no_uvs_returns_base_color() {
        let mesh = make_untextured_mesh();
        let barycentric = [1.0, 0.0, 0.0]; // at vertex 0
        let color = sample_texture_at_triangle(&mesh, 0, &barycentric);
        // base_color is [1.0, 0.0, 0.0] -> (255, 0, 0)
        let c = color.expect("should return base color");
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn sample_material_returns_correct_material_id() {
        let mesh = make_textured_mesh();
        let bvh = TriangleBvh::build(&mesh);
        // Query on the triangle surface
        let result = sample_material(&mesh, &bvh, Vec3::new(0.25, 0.25, 0.0));
        assert_eq!(result.material_id, 0);
        assert!(result.color.is_some());
    }

    #[test]
    fn sample_material_with_no_uvs_uses_base_color() {
        let mesh = make_untextured_mesh();
        let bvh = TriangleBvh::build(&mesh);
        let result = sample_material(&mesh, &bvh, Vec3::new(0.25, 0.25, 0.0));
        assert_eq!(result.material_id, 0);
        let c = result.color.expect("should have base color");
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn sample_texture_empty_texture_returns_gray() {
        let tex = TextureData {
            width: 0,
            height: 0,
            data: Vec::new(),
        };
        let color = sample_texture(&tex, 0.5, 0.5);
        assert_eq!(color.r, 128);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 128);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn sample_texture_bottom_right_pixel() {
        let tex = make_test_texture();
        // u=0.75, v=0.75 -> ix=1, iy=1 -> white pixel
        let color = sample_texture(&tex, 0.75, 0.75);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 255);
        assert_eq!(color.b, 255);
        assert_eq!(color.a, 255);
    }

    #[test]
    fn voxel_color_equality() {
        let a = VoxelColor {
            r: 10,
            g: 20,
            b: 30,
            a: 40,
        };
        let b = VoxelColor {
            r: 10,
            g: 20,
            b: 30,
            a: 40,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn sample_texture_at_triangle_with_texture_samples_it() {
        let mesh = make_textured_mesh();
        // Barycentric (1,0,0) = vertex 0 = UV (0,0) = red pixel
        let color = sample_texture_at_triangle(&mesh, 0, &[1.0, 0.0, 0.0]);
        let c = color.expect("should sample texture");
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 0);
        assert_eq!(c.b, 0);
    }

    #[test]
    fn sample_texture_at_triangle_invalid_material_returns_none() {
        let mesh = MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![5], // out of range
            materials: Vec::new(),     // no materials
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        };
        let color = sample_texture_at_triangle(&mesh, 0, &[1.0, 0.0, 0.0]);
        assert!(color.is_none());
    }
}
