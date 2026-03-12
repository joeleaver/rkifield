//! Edit operation types for CSG and sculpting.
//!
//! Defines the Rust-side representations of edit operations, brush shapes,
//! falloff curves, and the GPU-compatible [`EditParams`] struct that is
//! uploaded as a uniform buffer for the `csg_edit.wgsl` compute shader.

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Edit type — what operation to apply
// ---------------------------------------------------------------------------

/// The type of edit operation to perform on the SDF voxel data.
///
/// CSG operations modify the distance field. Paint operations only modify
/// material/blend/color data without changing geometry.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditType {
    /// Additive CSG: `min(existing, shape)` — adds material.
    CsgUnion = 0,
    /// Subtractive CSG: `max(existing, -shape)` — removes material.
    CsgSubtract = 1,
    /// Intersection CSG: `max(existing, shape)` — keeps overlap only.
    CsgIntersect = 2,
    /// Smooth additive: `smooth_min(existing, shape, k)`.
    SmoothUnion = 3,
    /// Smooth subtractive: `-smooth_min(-existing, shape, k)`.
    SmoothSubtract = 4,
    /// Smooth brush: weighted average of neighboring SDF values.
    Smooth = 5,
    /// Flatten brush: pull SDF toward a reference plane.
    Flatten = 6,
    /// Paint brush: sets `material_id` and per-voxel color on near-surface voxels (no geometry change).
    Paint = 7,
    /// Color paint: writes to the companion color pool (no geometry change).
    ColorPaint = 8,
}

impl EditType {
    /// Convert from a `u8` discriminant. Returns `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::CsgUnion),
            1 => Some(Self::CsgSubtract),
            2 => Some(Self::CsgIntersect),
            3 => Some(Self::SmoothUnion),
            4 => Some(Self::SmoothSubtract),
            5 => Some(Self::Smooth),
            6 => Some(Self::Flatten),
            7 => Some(Self::Paint),
            8 => Some(Self::ColorPaint),
            _ => None,
        }
    }

    /// Convert to `u32` for GPU uniform upload.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Shape type — analytic SDF primitive
// ---------------------------------------------------------------------------

/// The analytic SDF primitive shape used by an edit operation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShapeType {
    /// Sphere: `dimensions.x` = radius.
    Sphere = 0,
    /// Box (axis-aligned before rotation): `dimensions.xyz` = half-extents.
    Box = 1,
    /// Capsule: `dimensions.x` = radius, `dimensions.y` = half-height along local Y.
    Capsule = 2,
    /// Cylinder: `dimensions.x` = radius, `dimensions.y` = half-height along local Y.
    Cylinder = 3,
    /// Torus: `dimensions.x` = major radius, `dimensions.y` = minor (tube) radius.
    Torus = 4,
    /// Infinite plane: normal is local +Y, `dimensions.x` = offset along normal.
    Plane = 5,
}

impl ShapeType {
    /// Convert from a `u8` discriminant. Returns `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Sphere),
            1 => Some(Self::Box),
            2 => Some(Self::Capsule),
            3 => Some(Self::Cylinder),
            4 => Some(Self::Torus),
            5 => Some(Self::Plane),
            _ => None,
        }
    }

    /// Convert to `u32` for GPU uniform upload.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// Falloff curve
// ---------------------------------------------------------------------------

/// Falloff curve applied from brush center to edge.
///
/// Modulates the CSG blend strength per-voxel based on distance from the
/// edit center.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FalloffCurve {
    /// Linear falloff: `1 - d/r`.
    Linear = 0,
    /// Smooth (cubic Hermite) falloff: `smoothstep(r, 0, d)`.
    Smooth = 1,
    /// Sharp falloff: `(1 - d/r)^3`.
    Sharp = 2,
}

impl FalloffCurve {
    /// Convert from a `u8` discriminant. Returns `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Linear),
            1 => Some(Self::Smooth),
            2 => Some(Self::Sharp),
            _ => None,
        }
    }

    /// Convert to `u32` for GPU uniform upload.
    #[inline]
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

// ---------------------------------------------------------------------------
// EditParams — GPU uniform struct (128 bytes)
// ---------------------------------------------------------------------------

/// GPU-compatible edit parameters uploaded as a uniform buffer.
///
/// Matches the WGSL `EditParams` struct in `csg_edit.wgsl`. All fields are
/// vec4-aligned for correct GPU layout. Total size: 128 bytes.
///
/// # Layout (8 x vec4 = 128 bytes)
///
/// | Offset | Size | Field | Description |
/// |--------|------|-------|-------------|
/// | 0 | 16 | `position` | xyz = object-local pos, w = unused |
/// | 16 | 16 | `rotation` | quaternion xyzw |
/// | 32 | 16 | `dimensions` | xyz = half-extents/radius, w = unused |
/// | 48 | 16 | `strength`, `blend_k`, `falloff`, `material_id` | |
/// | 64 | 16 | `edit_type`, `shape_type`, `color_packed`, `_pad` | |
/// | 80 | 16 | `brick_base_index`, `brick_local_min[3]` | |
/// | 96 | 16 | `voxel_size`, `_pad[3]` | |
/// | 112 | 16 | `_pad2[4]` | padding to 128 bytes |
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct EditParams {
    // -- Spatial (48 bytes, 3 x vec4) --
    /// Object-local position of the edit center. `[x, y, z, 0.0]`.
    pub position: [f32; 4],
    /// Rotation quaternion (edit-local to world). `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Shape dimensions / half-extents. `[x, y, z, 0.0]`.
    /// Interpretation depends on [`ShapeType`].
    pub dimensions: [f32; 4],

    // -- Parameters (16 bytes, 1 x vec4) --
    /// Brush strength / opacity (0.0 .. 1.0).
    pub strength: f32,
    /// Smooth CSG blend radius (k parameter for smooth_min).
    pub blend_k: f32,
    /// Falloff curve type (see [`FalloffCurve`]).
    pub falloff: u32,
    /// Primary material ID to apply.
    pub material_id: u32,

    // -- Type info (16 bytes, 1 x vec4) --
    /// Edit operation type (see [`EditType`]).
    pub edit_type: u32,
    /// SDF shape primitive type (see [`ShapeType`]).
    pub shape_type: u32,
    /// Packed RGBA8 color (for color paint). `[R | G<<8 | B<<16 | A<<24]`.
    pub color_packed: u32,
    /// Padding (was secondary_id, now unused).
    pub _pad_type: u32,

    // -- Brick info (16 bytes, 1 x vec4) --
    /// Base index into the brick pool storage buffer for the target brick.
    pub brick_base_index: u32,
    /// Object-local minimum corner of the target brick. `[x, y, z]`.
    pub brick_local_min: [f32; 3],

    // -- Grid info (16 bytes, 1 x vec4) --
    /// Voxel size in world units for the target brick's resolution tier.
    pub voxel_size: f32,
    /// Padding within the grid info vec4.
    pub _pad: [f32; 3],

    // -- Padding to 128 bytes (16 bytes, 1 x vec4) --
    /// Reserved padding to bring total size to 128 bytes (8 x vec4).
    pub _pad2: [f32; 4],
}

impl EditParams {
    /// Create a zeroed `EditParams` (all fields zero / identity).
    pub fn zeroed() -> Self {
        bytemuck::Zeroable::zeroed()
    }

    /// Create an `EditParams` for a CSG operation with the given shape.
    ///
    /// This is a convenience builder — sets common fields and leaves
    /// brick-specific fields (brick_base_index, brick_local_min, voxel_size)
    /// to be filled per-brick at dispatch time.
    #[allow(clippy::too_many_arguments)]
    pub fn csg(
        edit_type: EditType,
        shape_type: ShapeType,
        position: [f32; 3],
        rotation: [f32; 4],
        dimensions: [f32; 3],
        strength: f32,
        blend_k: f32,
        falloff: FalloffCurve,
        material_id: u16,
    ) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            rotation,
            dimensions: [dimensions[0], dimensions[1], dimensions[2], 0.0],
            strength,
            blend_k,
            falloff: falloff.as_u32(),
            material_id: material_id as u32,
            edit_type: edit_type.as_u32(),
            shape_type: shape_type.as_u32(),
            color_packed: 0,
            _pad_type: 0,
            brick_base_index: 0,
            brick_local_min: [0.0; 3],
            voxel_size: 0.0,
            _pad: [0.0; 3],
            _pad2: [0.0; 4],
        }
    }

    /// Create an `EditParams` for a paint operation (material only, no geometry change).
    pub fn paint(
        position: [f32; 3],
        dimensions: [f32; 3],
        material_id: u16,
        strength: f32,
    ) -> Self {
        Self {
            position: [position[0], position[1], position[2], 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
            dimensions: [dimensions[0], dimensions[1], dimensions[2], 0.0],
            strength,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth.as_u32(),
            material_id: material_id as u32,
            edit_type: EditType::Paint.as_u32(),
            shape_type: ShapeType::Sphere.as_u32(),
            color_packed: 0,
            _pad_type: 0,
            brick_base_index: 0,
            brick_local_min: [0.0; 3],
            voxel_size: 0.0,
            _pad: [0.0; 3],
            _pad2: [0.0; 4],
        }
    }

    /// Set the per-brick fields for dispatch.
    ///
    /// Called once per affected brick before uploading to the GPU.
    pub fn with_brick_info(
        mut self,
        brick_base_index: u32,
        brick_local_min: [f32; 3],
        voxel_size: f32,
    ) -> Self {
        self.brick_base_index = brick_base_index;
        self.brick_local_min = brick_local_min;
        self.voxel_size = voxel_size;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem;

    #[test]
    fn edit_params_size_is_128_bytes() {
        assert_eq!(mem::size_of::<EditParams>(), 128);
    }

    #[test]
    fn edit_params_alignment_is_4() {
        assert_eq!(mem::align_of::<EditParams>(), 4);
    }

    #[test]
    fn edit_type_round_trip() {
        for v in 0u8..=8 {
            let et = EditType::from_u8(v).unwrap();
            assert_eq!(et as u8, v);
            assert_eq!(et.as_u32(), v as u32);
        }
        assert!(EditType::from_u8(9).is_none());
        assert!(EditType::from_u8(255).is_none());
    }

    #[test]
    fn shape_type_round_trip() {
        for v in 0u8..=5 {
            let st = ShapeType::from_u8(v).unwrap();
            assert_eq!(st as u8, v);
            assert_eq!(st.as_u32(), v as u32);
        }
        assert!(ShapeType::from_u8(6).is_none());
        assert!(ShapeType::from_u8(255).is_none());
    }

    #[test]
    fn falloff_curve_round_trip() {
        for v in 0u8..=2 {
            let fc = FalloffCurve::from_u8(v).unwrap();
            assert_eq!(fc as u8, v);
            assert_eq!(fc.as_u32(), v as u32);
        }
        assert!(FalloffCurve::from_u8(3).is_none());
    }

    #[test]
    fn edit_params_is_pod_and_zeroable() {
        // Pod and Zeroable are compile-time guarantees via derive,
        // but verify zeroed() produces a valid instance.
        let params = EditParams::zeroed();
        assert_eq!(params.edit_type, 0);
        assert_eq!(params.shape_type, 0);
        assert_eq!(params.strength, 0.0);
        assert_eq!(params.voxel_size, 0.0);

        // Verify we can cast to/from bytes (Pod contract).
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 128);
        let _round_trip: &EditParams = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn csg_builder_sets_fields() {
        let params = EditParams::csg(
            EditType::SmoothUnion,
            ShapeType::Sphere,
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            0.8,
            0.1,
            FalloffCurve::Smooth,
            42,
        );
        assert_eq!(params.edit_type, EditType::SmoothUnion.as_u32());
        assert_eq!(params.shape_type, ShapeType::Sphere.as_u32());
        assert_eq!(params.position[0], 1.0);
        assert_eq!(params.position[1], 2.0);
        assert_eq!(params.position[2], 3.0);
        assert_eq!(params.dimensions[0], 0.5);
        assert_eq!(params.strength, 0.8);
        assert_eq!(params.blend_k, 0.1);
        assert_eq!(params.falloff, FalloffCurve::Smooth.as_u32());
        assert_eq!(params.material_id, 42);
    }

    #[test]
    fn paint_builder_sets_fields() {
        let params = EditParams::paint([5.0, 6.0, 7.0], [1.0, 1.0, 1.0], 99, 0.5);
        assert_eq!(params.edit_type, EditType::Paint.as_u32());
        assert_eq!(params.shape_type, ShapeType::Sphere.as_u32());
        assert_eq!(params.material_id, 99);
        assert_eq!(params.strength, 0.5);
        // Identity quaternion
        assert_eq!(params.rotation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn with_brick_info_sets_per_brick_fields() {
        let params = EditParams::zeroed().with_brick_info(42, [10.0, 20.0, 30.0], 0.02);
        assert_eq!(params.brick_base_index, 42);
        assert_eq!(params.brick_local_min, [10.0, 20.0, 30.0]);
        assert_eq!(params.voxel_size, 0.02);
    }
}
