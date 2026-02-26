//! GPU object metadata — per-object data uploaded to the GPU for ray marching.
//!
//! [`GpuObject`] is a 256-byte `bytemuck::Pod` struct that carries everything the
//! GPU needs to evaluate an SDF object: inverse world transform, AABB, brick map
//! reference, SDF type and parameters, blend mode, material, and scale.
//!
//! [`ObjectMetadataBuffer`] manages a wgpu storage buffer containing an array of
//! `GpuObject`s, with dirty tracking for incremental updates.

use bytemuck::{Pod, Zeroable};

/// GPU-uploadable per-object metadata (256 bytes, bytemuck Pod).
///
/// This struct is uploaded to a storage buffer and read by the ray march shader
/// to evaluate each object. Fields are laid out for WGSL `array<GpuObject>` access.
///
/// # Layout (256 bytes)
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0      | 64   | inverse_world (mat4x4<f32>) |
/// | 64     | 16   | aabb_min (vec4) |
/// | 80     | 16   | aabb_max (vec4) |
/// | 96     | 4    | brick_map_offset (u32) |
/// | 100    | 12   | brick_map_dims (uvec3 as [u32;3]) |
/// | 112    | 4    | voxel_size (f32) |
/// | 116    | 4    | material_id (u32) |
/// | 120    | 4    | sdf_type (u32) |
/// | 124    | 4    | blend_mode (u32) |
/// | 128    | 4    | blend_radius (f32) |
/// | 132    | 16   | sdf_params (vec4) |
/// | 148    | 12   | accumulated_scale ([f32; 3]) |
/// | 160    | 4    | lod_level (u32) |
/// | 164    | 4    | object_id (u32) |
/// | 168    | 4    | primitive_type (u32) |
/// | 172    | 84   | _padding |
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuObject {
    /// Pre-computed inverse world transform (camera-relative).
    /// Used to transform rays into object-local space.
    /// Stored as column-major [col0, col1, col2, col3] = 16 floats.
    pub inverse_world: [[f32; 4]; 4],

    /// Object AABB minimum (camera-relative, xyz + padding).
    pub aabb_min: [f32; 4],
    /// Object AABB maximum (camera-relative, xyz + padding).
    pub aabb_max: [f32; 4],

    /// Offset into the packed brick maps storage buffer.
    pub brick_map_offset: u32,
    /// Dimensions of the 3D brick map grid [x, y, z].
    pub brick_map_dims: [u32; 3],

    /// World-space size of one voxel edge (0 for analytical objects).
    pub voxel_size: f32,
    /// Primary material ID for analytical objects.
    pub material_id: u32,
    /// SDF type: 0 = None, 1 = Analytical, 2 = Voxelized.
    pub sdf_type: u32,
    /// Blend mode: 0 = SmoothUnion, 1 = Union, 2 = Subtract, 3 = Intersect.
    pub blend_mode: u32,

    /// Blend radius for SmoothUnion (0 for other modes).
    pub blend_radius: f32,
    /// SDF primitive parameters (interpretation depends on sdf_type).
    /// Sphere: [radius, 0, 0, 0]
    /// Box: [half_x, half_y, half_z, 0]
    /// Capsule: [radius, half_height, 0, 0]
    /// Torus: [major_radius, minor_radius, 0, 0]
    /// Cylinder: [radius, half_height, 0, 0]
    /// Plane: [normal_x, normal_y, normal_z, distance]
    pub sdf_params: [f32; 4],

    /// Product of all per-axis scales from root to this node.
    pub accumulated_scale: [f32; 3],
    /// Current LOD level (0 = finest).
    pub lod_level: u32,
    /// Unique object ID (matches SceneObject::id).
    pub object_id: u32,
    /// Primitive type for analytical objects (see [`primitive_type`] module).
    pub primitive_type: u32,

    /// Padding to 256 bytes.
    pub _padding: [f32; 21],
}

/// SDF type constants for [`GpuObject::sdf_type`].
pub mod sdf_type {
    /// No SDF source (group node).
    pub const NONE: u32 = 0;
    /// Analytical SDF primitive.
    pub const ANALYTICAL: u32 = 1;
    /// Voxelized SDF (brick map lookup).
    pub const VOXELIZED: u32 = 2;
}

/// Primitive type constants for [`GpuObject::primitive_type`].
pub mod primitive_type {
    /// Sphere (params: radius).
    pub const SPHERE: u32 = 0;
    /// Box (params: half_x, half_y, half_z).
    pub const BOX: u32 = 1;
    /// Capsule (params: radius, half_height).
    pub const CAPSULE: u32 = 2;
    /// Torus (params: major_radius, minor_radius).
    pub const TORUS: u32 = 3;
    /// Cylinder (params: radius, half_height).
    pub const CYLINDER: u32 = 4;
    /// Plane (params: normal_xyz, distance).
    pub const PLANE: u32 = 5;
}

/// Blend mode constants for [`GpuObject::blend_mode`].
pub mod blend_mode {
    /// Smooth union (polynomial smooth-min).
    pub const SMOOTH_UNION: u32 = 0;
    /// Hard union (min).
    pub const UNION: u32 = 1;
    /// Subtraction.
    pub const SUBTRACT: u32 = 2;
    /// Intersection (max).
    pub const INTERSECT: u32 = 3;
}

impl GpuObject {
    /// Build a `GpuObject` from a flattened node and its parent object data.
    pub fn from_flat_node(
        flat: &rkf_core::transform_flatten::FlatNode,
        object_id: u32,
        aabb_min: [f32; 4],
        aabb_max: [f32; 4],
    ) -> Self {
        let inv = flat.inverse_world.to_cols_array_2d();

        let (sdf_type, material_id, sdf_params, brick_map_offset, brick_map_dims, voxel_size, prim_type) =
            match &flat.sdf_source {
                rkf_core::SdfSource::None => (
                    sdf_type::NONE,
                    0,
                    [0.0; 4],
                    0,
                    [0u32; 3],
                    0.0,
                    0,
                ),
                rkf_core::SdfSource::Analytical {
                    primitive,
                    material_id,
                } => {
                    let params = primitive_params(primitive);
                    (
                        sdf_type::ANALYTICAL,
                        *material_id as u32,
                        params,
                        0,
                        [0u32; 3],
                        0.0,
                        primitive_type_id(primitive),
                    )
                }
                rkf_core::SdfSource::Voxelized {
                    brick_map_handle,
                    voxel_size,
                    ..
                } => (
                    sdf_type::VOXELIZED,
                    0,
                    [0.0; 4],
                    brick_map_handle.offset,
                    [
                        brick_map_handle.dims.x,
                        brick_map_handle.dims.y,
                        brick_map_handle.dims.z,
                    ],
                    *voxel_size,
                    0,
                ),
            };

        let (bm, br) = blend_mode_encode(&flat.blend_mode);

        Self {
            inverse_world: inv,
            aabb_min,
            aabb_max,
            brick_map_offset,
            brick_map_dims,
            voxel_size,
            material_id,
            sdf_type,
            blend_mode: bm,
            blend_radius: br,
            sdf_params,
            accumulated_scale: flat.accumulated_scale.to_array(),
            lod_level: 0,
            object_id,
            primitive_type: prim_type,
            _padding: [0.0; 21],
        }
    }
}

/// Extract SDF primitive parameters into a `[f32; 4]` array.
fn primitive_type_id(primitive: &rkf_core::SdfPrimitive) -> u32 {
    match *primitive {
        rkf_core::SdfPrimitive::Sphere { .. } => primitive_type::SPHERE,
        rkf_core::SdfPrimitive::Box { .. } => primitive_type::BOX,
        rkf_core::SdfPrimitive::Capsule { .. } => primitive_type::CAPSULE,
        rkf_core::SdfPrimitive::Torus { .. } => primitive_type::TORUS,
        rkf_core::SdfPrimitive::Cylinder { .. } => primitive_type::CYLINDER,
        rkf_core::SdfPrimitive::Plane { .. } => primitive_type::PLANE,
    }
}

fn primitive_params(primitive: &rkf_core::SdfPrimitive) -> [f32; 4] {
    match *primitive {
        rkf_core::SdfPrimitive::Sphere { radius } => [radius, 0.0, 0.0, 0.0],
        rkf_core::SdfPrimitive::Box { half_extents } => {
            [half_extents.x, half_extents.y, half_extents.z, 0.0]
        }
        rkf_core::SdfPrimitive::Capsule {
            radius,
            half_height,
        } => [radius, half_height, 0.0, 0.0],
        rkf_core::SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => [major_radius, minor_radius, 0.0, 0.0],
        rkf_core::SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => [radius, half_height, 0.0, 0.0],
        rkf_core::SdfPrimitive::Plane { normal, distance } => {
            [normal.x, normal.y, normal.z, distance]
        }
    }
}

/// Encode a `BlendMode` into (mode_id, radius).
fn blend_mode_encode(mode: &rkf_core::BlendMode) -> (u32, f32) {
    match *mode {
        rkf_core::BlendMode::SmoothUnion(r) => (blend_mode::SMOOTH_UNION, r),
        rkf_core::BlendMode::Union => (blend_mode::UNION, 0.0),
        rkf_core::BlendMode::Subtract => (blend_mode::SUBTRACT, 0.0),
        rkf_core::BlendMode::Intersect => (blend_mode::INTERSECT, 0.0),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};
    use rkf_core::transform_flatten::FlatNode;
    use rkf_core::{BlendMode, SdfPrimitive, SdfSource};
    use std::mem;

    #[test]
    fn gpu_object_size_is_256_bytes() {
        assert_eq!(mem::size_of::<GpuObject>(), 256);
    }

    #[test]
    fn gpu_object_alignment() {
        assert!(mem::align_of::<GpuObject>() <= 16);
    }

    #[test]
    fn gpu_object_pod_roundtrip() {
        let obj = GpuObject {
            inverse_world: Mat4::IDENTITY.to_cols_array_2d(),
            aabb_min: [-1.0, -1.0, -1.0, 0.0],
            aabb_max: [1.0, 1.0, 1.0, 0.0],
            brick_map_offset: 42,
            brick_map_dims: [4, 4, 4],
            voxel_size: 0.02,
            material_id: 7,
            sdf_type: sdf_type::ANALYTICAL,
            blend_mode: blend_mode::SMOOTH_UNION,
            blend_radius: 0.1,
            sdf_params: [0.5, 0.0, 0.0, 0.0],
            accumulated_scale: [2.0, 2.0, 2.0],
            lod_level: 0,
            object_id: 99,
            primitive_type: primitive_type::SPHERE,
            _padding: [0.0; 21],
        };

        let bytes = bytemuck::bytes_of(&obj);
        assert_eq!(bytes.len(), 256);

        let restored: &GpuObject = bytemuck::from_bytes(bytes);
        assert_eq!(restored.object_id, 99);
        assert_eq!(restored.material_id, 7);
        assert_eq!(restored.brick_map_offset, 42);
        assert!((restored.accumulated_scale[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn from_flat_node_analytical() {
        let flat = FlatNode {
            inverse_world: Mat4::IDENTITY,
            accumulated_scale: Vec3::splat(1.5),
            sdf_source: SdfSource::Analytical {
                primitive: SdfPrimitive::Sphere { radius: 0.5 },
                material_id: 3,
            },
            blend_mode: BlendMode::SmoothUnion(0.1),
            depth: 0,
            parent_index: u32::MAX,
            name: "sphere".into(),
        };

        let gpu = GpuObject::from_flat_node(
            &flat,
            42,
            [-0.5, -0.5, -0.5, 0.0],
            [0.5, 0.5, 0.5, 0.0],
        );

        assert_eq!(gpu.object_id, 42);
        assert_eq!(gpu.sdf_type, sdf_type::ANALYTICAL);
        assert_eq!(gpu.material_id, 3);
        assert!((gpu.sdf_params[0] - 0.5).abs() < 1e-6);
        assert!((gpu.blend_radius - 0.1).abs() < 1e-6);
        assert_eq!(gpu.blend_mode, blend_mode::SMOOTH_UNION);
        assert!((gpu.accumulated_scale[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn from_flat_node_voxelized() {
        let flat = FlatNode {
            inverse_world: Mat4::IDENTITY,
            accumulated_scale: Vec3::ONE,
            sdf_source: SdfSource::Voxelized {
                brick_map_handle: rkf_core::BrickMapHandle {
                    offset: 100,
                    dims: glam::UVec3::new(8, 8, 8),
                },
                voxel_size: 0.02,
                aabb: rkf_core::Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
            },
            blend_mode: BlendMode::Union,
            depth: 1,
            parent_index: 0,
            name: "mesh".into(),
        };

        let gpu = GpuObject::from_flat_node(
            &flat,
            7,
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        );

        assert_eq!(gpu.sdf_type, sdf_type::VOXELIZED);
        assert_eq!(gpu.brick_map_offset, 100);
        assert_eq!(gpu.brick_map_dims, [8, 8, 8]);
        assert!((gpu.voxel_size - 0.02).abs() < 1e-6);
        assert_eq!(gpu.blend_mode, blend_mode::UNION);
    }

    #[test]
    fn from_flat_node_none() {
        let flat = FlatNode {
            inverse_world: Mat4::IDENTITY,
            accumulated_scale: Vec3::ONE,
            sdf_source: SdfSource::None,
            blend_mode: BlendMode::Subtract,
            depth: 0,
            parent_index: u32::MAX,
            name: "group".into(),
        };

        let gpu = GpuObject::from_flat_node(&flat, 1, [0.0; 4], [0.0; 4]);
        assert_eq!(gpu.sdf_type, sdf_type::NONE);
        assert_eq!(gpu.blend_mode, blend_mode::SUBTRACT);
    }

    #[test]
    fn primitive_params_all_types() {
        use rkf_core::SdfPrimitive;

        let s = primitive_params(&SdfPrimitive::Sphere { radius: 1.5 });
        assert!((s[0] - 1.5).abs() < 1e-6);

        let b = primitive_params(&SdfPrimitive::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        });
        assert!((b[0] - 1.0).abs() < 1e-6);
        assert!((b[1] - 2.0).abs() < 1e-6);
        assert!((b[2] - 3.0).abs() < 1e-6);

        let c = primitive_params(&SdfPrimitive::Capsule {
            radius: 0.5,
            half_height: 1.0,
        });
        assert!((c[0] - 0.5).abs() < 1e-6);
        assert!((c[1] - 1.0).abs() < 1e-6);

        let t = primitive_params(&SdfPrimitive::Torus {
            major_radius: 1.0,
            minor_radius: 0.3,
        });
        assert!((t[0] - 1.0).abs() < 1e-6);
        assert!((t[1] - 0.3).abs() < 1e-6);

        let cy = primitive_params(&SdfPrimitive::Cylinder {
            radius: 0.5,
            half_height: 2.0,
        });
        assert!((cy[0] - 0.5).abs() < 1e-6);
        assert!((cy[1] - 2.0).abs() < 1e-6);

        let p = primitive_params(&SdfPrimitive::Plane {
            normal: Vec3::Y,
            distance: 5.0,
        });
        assert!((p[1] - 1.0).abs() < 1e-6);
        assert!((p[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn blend_mode_encoding() {
        let (m, r) = blend_mode_encode(&BlendMode::SmoothUnion(0.2));
        assert_eq!(m, blend_mode::SMOOTH_UNION);
        assert!((r - 0.2).abs() < 1e-6);

        let (m, _) = blend_mode_encode(&BlendMode::Union);
        assert_eq!(m, blend_mode::UNION);

        let (m, _) = blend_mode_encode(&BlendMode::Subtract);
        assert_eq!(m, blend_mode::SUBTRACT);

        let (m, _) = blend_mode_encode(&BlendMode::Intersect);
        assert_eq!(m, blend_mode::INTERSECT);
    }

    #[test]
    fn zeroed_gpu_object_is_valid() {
        let obj: GpuObject = bytemuck::Zeroable::zeroed();
        assert_eq!(obj.object_id, 0);
        assert_eq!(obj.sdf_type, 0);
        assert_eq!(mem::size_of_val(&obj), 256);
    }
}
