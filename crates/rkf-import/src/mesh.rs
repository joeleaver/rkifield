//! Mesh loading and triangle data for the import pipeline.
//!
//! Loads polygon meshes from glTF (.gltf, .glb) files into a unified
//! [`MeshData`] representation ready for voxelization.

use anyhow::{Context, Result};
use glam::Vec3;

/// A triangle with vertex indices.
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    /// First vertex index.
    pub i0: u32,
    /// Second vertex index.
    pub i1: u32,
    /// Third vertex index.
    pub i2: u32,
}

/// Material properties extracted from a source mesh.
#[derive(Debug, Clone)]
pub struct ImportMaterial {
    /// Name of the material.
    pub name: String,
    /// Base color (linear RGB).
    pub base_color: [f32; 3],
    /// Metallic factor [0, 1].
    pub metallic: f32,
    /// Roughness factor [0, 1].
    pub roughness: f32,
    /// Albedo texture data (RGBA8, if present).
    pub albedo_texture: Option<TextureData>,
}

/// Texture data loaded from a mesh file.
#[derive(Debug, Clone)]
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 pixel data (row-major, 4 bytes per pixel).
    pub data: Vec<u8>,
}

/// Loaded mesh data ready for voxelization.
#[derive(Debug)]
pub struct MeshData {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals (same length as positions, or empty).
    pub normals: Vec<Vec3>,
    /// Vertex UVs (same length as positions, or empty).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices (length = num_triangles * 3).
    pub indices: Vec<u32>,
    /// Per-triangle material index (length = num_triangles).
    pub material_indices: Vec<u32>,
    /// Materials from the source file.
    pub materials: Vec<ImportMaterial>,
    /// Mesh bounding box minimum.
    pub bounds_min: Vec3,
    /// Mesh bounding box maximum.
    pub bounds_max: Vec3,
}

impl MeshData {
    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get a triangle's three vertex positions.
    pub fn triangle_positions(&self, tri_idx: usize) -> [Vec3; 3] {
        let base = tri_idx * 3;
        [
            self.positions[self.indices[base] as usize],
            self.positions[self.indices[base + 1] as usize],
            self.positions[self.indices[base + 2] as usize],
        ]
    }

    /// Get a triangle's three UVs (returns [[0,0]; 3] if no UVs).
    pub fn triangle_uvs(&self, tri_idx: usize) -> [[f32; 2]; 3] {
        if self.uvs.is_empty() {
            return [[0.0, 0.0]; 3];
        }
        let base = tri_idx * 3;
        [
            self.uvs[self.indices[base] as usize],
            self.uvs[self.indices[base + 1] as usize],
            self.uvs[self.indices[base + 2] as usize],
        ]
    }

    /// Compute average triangle edge length.
    pub fn average_edge_length(&self) -> f32 {
        if self.triangle_count() == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        let mut count = 0u32;
        for i in 0..self.triangle_count() {
            let [v0, v1, v2] = self.triangle_positions(i);
            total += (v1 - v0).length();
            total += (v2 - v1).length();
            total += (v0 - v2).length();
            count += 3;
        }
        total / count as f32
    }
}

/// Load a mesh from a file path. Supports glTF (.gltf, .glb).
pub fn load_mesh(path: &str) -> Result<MeshData> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gltf" | "glb" => load_gltf(path),
        other => anyhow::bail!("Unsupported format: .{other}. Supported: .gltf, .glb"),
    }
}

/// Load mesh data from a glTF/GLB file.
fn load_gltf(path: &str) -> Result<MeshData> {
    let (document, buffers, images) =
        gltf::import(path).with_context(|| format!("Failed to load glTF: {path}"))?;

    let mut all_positions = Vec::new();
    let mut all_normals = Vec::new();
    let mut all_uvs = Vec::new();
    let mut all_indices = Vec::new();
    let mut all_material_indices = Vec::new();
    let mut bounds_min = Vec3::splat(f32::MAX);
    let mut bounds_max = Vec3::splat(f32::MIN);

    // Collect materials
    let materials: Vec<ImportMaterial> = document
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            let bc = pbr.base_color_factor();

            // Try to load albedo texture
            let albedo_texture = pbr.base_color_texture().and_then(|info| {
                let tex = info.texture();
                let source = tex.source();
                let img_index = source.index();
                if img_index < images.len() {
                    let img = &images[img_index];
                    // Convert to RGBA8
                    let rgba_data = match img.format {
                        gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
                        gltf::image::Format::R8G8B8 => {
                            let mut rgba = Vec::with_capacity(img.pixels.len() / 3 * 4);
                            for chunk in img.pixels.chunks(3) {
                                rgba.extend_from_slice(chunk);
                                rgba.push(255);
                            }
                            rgba
                        }
                        _ => return None, // Skip unsupported formats
                    };
                    Some(TextureData {
                        width: img.width,
                        height: img.height,
                        data: rgba_data,
                    })
                } else {
                    None
                }
            });

            ImportMaterial {
                name: mat.name().unwrap_or("unnamed").to_string(),
                base_color: [bc[0], bc[1], bc[2]],
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                albedo_texture,
            }
        })
        .collect();

    // Process all meshes in the scene
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));
            let mat_idx = primitive.material().index().unwrap_or(0) as u32;

            let vertex_offset = all_positions.len() as u32;

            // Positions (required)
            if let Some(positions) = reader.read_positions() {
                for p in positions {
                    let v = Vec3::new(p[0], p[1], p[2]);
                    bounds_min = bounds_min.min(v);
                    bounds_max = bounds_max.max(v);
                    all_positions.push(v);
                }
            } else {
                continue; // Skip primitives without positions
            }

            // Normals (optional)
            if let Some(normals) = reader.read_normals() {
                for n in normals {
                    all_normals.push(Vec3::new(n[0], n[1], n[2]));
                }
            }

            // UVs (optional, first set)
            if let Some(uvs) = reader.read_tex_coords(0) {
                for uv in uvs.into_f32() {
                    all_uvs.push(uv);
                }
            }

            // Indices
            if let Some(indices) = reader.read_indices() {
                let tri_start = all_indices.len() / 3;
                for idx in indices.into_u32() {
                    all_indices.push(vertex_offset + idx);
                }
                let tri_end = all_indices.len() / 3;
                for _ in tri_start..tri_end {
                    all_material_indices.push(mat_idx);
                }
            }
        }
    }

    // If no materials in the file, add a default
    let materials = if materials.is_empty() {
        vec![ImportMaterial {
            name: "default".to_string(),
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
        }]
    } else {
        materials
    };

    // Pad normals if needed (they should match positions count)
    while all_normals.len() < all_positions.len() {
        all_normals.push(Vec3::Y); // default up normal
    }
    // Don't pad UVs -- empty is OK (means no UV data)

    Ok(MeshData {
        positions: all_positions,
        normals: all_normals,
        uvs: all_uvs,
        indices: all_indices,
        material_indices: all_material_indices,
        materials,
        bounds_min,
        bounds_max,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a simple MeshData for testing.
    fn make_test_mesh() -> MeshData {
        // A single triangle: (0,0,0), (1,0,0), (0,1,0)
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
                name: "test".to_string(),
                base_color: [1.0, 0.0, 0.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    /// Helper: build an empty MeshData.
    fn make_empty_mesh() -> MeshData {
        MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_indices: Vec::new(),
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    #[test]
    fn triangle_count_empty() {
        let mesh = make_empty_mesh();
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn triangle_count_single() {
        let mesh = make_test_mesh();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn average_edge_length_empty() {
        let mesh = make_empty_mesh();
        assert_eq!(mesh.average_edge_length(), 0.0);
    }

    #[test]
    fn average_edge_length_unit_triangle() {
        let mesh = make_test_mesh();
        let avg = mesh.average_edge_length();
        // Edges: (0,0,0)->(1,0,0) = 1.0, (1,0,0)->(0,1,0) = sqrt(2), (0,1,0)->(0,0,0) = 1.0
        let expected = (1.0 + std::f32::consts::SQRT_2 + 1.0) / 3.0;
        assert!((avg - expected).abs() < 1e-5, "expected {expected}, got {avg}");
    }

    #[test]
    fn triangle_positions_correct() {
        let mesh = make_test_mesh();
        let [v0, v1, v2] = mesh.triangle_positions(0);
        assert_eq!(v0, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(v1, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(v2, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn triangle_uvs_no_uvs_returns_zeros() {
        let mut mesh = make_test_mesh();
        mesh.uvs.clear();
        let uvs = mesh.triangle_uvs(0);
        assert_eq!(uvs, [[0.0, 0.0]; 3]);
    }

    #[test]
    fn triangle_uvs_with_data() {
        let mesh = make_test_mesh();
        let uvs = mesh.triangle_uvs(0);
        assert_eq!(uvs[0], [0.0, 0.0]);
        assert_eq!(uvs[1], [1.0, 0.0]);
        assert_eq!(uvs[2], [0.0, 1.0]);
    }

    #[test]
    fn import_material_defaults() {
        let mat = ImportMaterial {
            name: "default".to_string(),
            base_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
        };
        assert_eq!(mat.name, "default");
        assert_eq!(mat.base_color, [0.8, 0.8, 0.8]);
        assert_eq!(mat.metallic, 0.0);
        assert_eq!(mat.roughness, 0.5);
        assert!(mat.albedo_texture.is_none());
    }

    #[test]
    fn texture_data_construction() {
        let tex = TextureData {
            width: 2,
            height: 2,
            data: vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255],
        };
        assert_eq!(tex.width, 2);
        assert_eq!(tex.height, 2);
        assert_eq!(tex.data.len(), 16); // 2x2 RGBA8
    }

    #[test]
    fn load_mesh_unsupported_format() {
        let result = load_mesh("model.fbx");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Unsupported format"),
            "Expected unsupported format error, got: {err}"
        );
    }

    #[test]
    fn load_mesh_no_extension() {
        let result = load_mesh("noext");
        assert!(result.is_err());
    }

    #[test]
    fn triangle_count_multiple() {
        // Two triangles sharing an edge
        let mesh = MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: Vec::new(),
            indices: vec![0, 1, 2, 1, 3, 2],
            material_indices: vec![0, 0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ONE,
        };
        assert_eq!(mesh.triangle_count(), 2);
    }
}
