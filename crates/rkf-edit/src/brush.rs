//! Sculpting brush definitions for the edit system.
//!
//! A [`Brush`] combines a [`BrushType`] (what operation to perform), a
//! [`BrushShape`] (what SDF primitive to use), and tuning parameters like
//! radius, strength, falloff, and material IDs.
//!
//! Brushes are the user-facing abstraction. In v2, brushes operate in
//! object-local space. Use [`Brush::to_edit_op`] to convert a world-space
//! brush stroke into an object-local [`EditOp`](crate::edit_op::EditOp).

use glam::{Quat, Vec3};

use crate::edit_op::EditOp;
use crate::types::{EditType, FalloffCurve, ShapeType};

// ---------------------------------------------------------------------------
// BrushType — what the brush does
// ---------------------------------------------------------------------------

/// Brush type determines the edit operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrushType {
    /// CSG smooth union — adds material.
    Add,
    /// CSG smooth subtraction — removes material.
    Subtract,
    /// Weighted average of neighboring SDF — smooths geometry.
    Smooth,
    /// Pull SDF toward a reference plane — flattens geometry.
    Flatten,
    /// Set `material_id` and per-voxel color on near-surface voxels (no geometry change).
    Paint,
    /// Write to the companion color pool (no geometry change).
    ColorPaint,
}

// ---------------------------------------------------------------------------
// BrushShape — what primitive to use
// ---------------------------------------------------------------------------

/// Brush shape (what SDF primitive to use).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrushShape {
    /// Sphere: `radius` is the sphere radius.
    Sphere,
    /// Cube (axis-aligned before rotation): `radius` is the half-extent.
    Cube,
    /// Capsule: `radius` is the tube radius. Height is `2 * radius` by default.
    Capsule,
    /// Cylinder: `radius` is the tube radius. Height is `2 * radius` by default.
    Cylinder,
}

// ---------------------------------------------------------------------------
// Brush
// ---------------------------------------------------------------------------

/// A sculpting brush with all parameters needed to produce an edit operation.
#[derive(Debug, Clone)]
pub struct Brush {
    /// What operation the brush performs.
    pub brush_type: BrushType,
    /// What SDF primitive shape to use.
    pub shape: BrushShape,
    /// Brush radius in world units (meters).
    pub radius: f32,
    /// Brush strength / opacity (0.0 .. 1.0).
    pub strength: f32,
    /// Falloff curve from brush center to edge.
    pub falloff: FalloffCurve,
    /// Primary material ID to apply.
    pub material_id: u16,
    /// Smooth CSG blend radius (k parameter for smooth_min).
    pub blend_k: f32,
}

impl Brush {
    /// Create a default Add brush (sphere, smooth falloff).
    pub fn add_sphere(radius: f32, material_id: u16) -> Self {
        Self {
            brush_type: BrushType::Add,
            shape: BrushShape::Sphere,
            radius,
            strength: 1.0,
            falloff: FalloffCurve::Smooth,
            material_id,
            blend_k: radius * 0.3, // default blend radius = 30% of brush radius
        }
    }

    /// Create a Subtract brush (sphere, smooth falloff).
    pub fn subtract_sphere(radius: f32) -> Self {
        Self {
            brush_type: BrushType::Subtract,
            shape: BrushShape::Sphere,
            radius,
            strength: 1.0,
            falloff: FalloffCurve::Smooth,
            material_id: 0,
            blend_k: radius * 0.3,
        }
    }

    /// Create a Smooth brush (sphere, smooth falloff).
    pub fn smooth_sphere(radius: f32, strength: f32) -> Self {
        Self {
            brush_type: BrushType::Smooth,
            shape: BrushShape::Sphere,
            radius,
            strength: strength.clamp(0.0, 1.0),
            falloff: FalloffCurve::Smooth,
            material_id: 0,
            blend_k: 0.0, // smooth brush doesn't use blend_k
        }
    }

    /// Create a Paint brush (sphere, smooth falloff).
    pub fn paint_sphere(radius: f32, material_id: u16) -> Self {
        Self {
            brush_type: BrushType::Paint,
            shape: BrushShape::Sphere,
            radius,
            strength: 1.0,
            falloff: FalloffCurve::Smooth,
            material_id,
            blend_k: 0.0, // paint doesn't use blend_k
        }
    }

    /// Convert [`BrushType`] to the corresponding [`EditType`].
    pub fn edit_type(&self) -> EditType {
        match self.brush_type {
            BrushType::Add => EditType::SmoothUnion,
            BrushType::Subtract => EditType::SmoothSubtract,
            BrushType::Smooth => EditType::Smooth,
            BrushType::Flatten => EditType::Flatten,
            BrushType::Paint => EditType::Paint,
            BrushType::ColorPaint => EditType::ColorPaint,
        }
    }

    /// Convert [`BrushShape`] to the corresponding [`ShapeType`].
    pub fn shape_type(&self) -> ShapeType {
        match self.shape {
            BrushShape::Sphere => ShapeType::Sphere,
            BrushShape::Cube => ShapeType::Box,
            BrushShape::Capsule => ShapeType::Capsule,
            BrushShape::Cylinder => ShapeType::Cylinder,
        }
    }

    /// Convert a world-space brush application into an object-local [`EditOp`].
    ///
    /// Transforms the brush position and dimensions from world space into the
    /// target object's local coordinate space.
    ///
    /// # Parameters
    /// - `object_id` — Which object to edit
    /// - `world_pos` — Brush hit position in world space
    /// - `object_world_pos` — Object's world-space origin
    /// - `object_rotation` — Object's world rotation
    /// - `object_scale` — Object's uniform scale
    pub fn to_edit_op(
        &self,
        object_id: u32,
        world_pos: Vec3,
        object_world_pos: Vec3,
        object_rotation: Quat,
        object_scale: f32,
    ) -> EditOp {
        let local_pos =
            world_to_object_local(world_pos, object_world_pos, object_rotation, object_scale);
        let inv_scale = 1.0 / object_scale.max(1e-6);

        // Scale brush dimensions into object-local space
        let local_dims = match self.shape {
            BrushShape::Sphere => Vec3::new(self.radius * inv_scale, 0.0, 0.0),
            BrushShape::Cube => Vec3::splat(self.radius * inv_scale),
            BrushShape::Capsule => {
                Vec3::new(self.radius * inv_scale, self.radius * inv_scale, 0.0)
            }
            BrushShape::Cylinder => {
                Vec3::new(self.radius * inv_scale, self.radius * inv_scale, 0.0)
            }
        };

        EditOp {
            object_id,
            position: local_pos,
            rotation: Quat::IDENTITY, // brush rotation stays identity for now
            edit_type: self.edit_type(),
            shape_type: self.shape_type(),
            dimensions: local_dims,
            strength: self.strength,
            blend_k: self.blend_k * inv_scale,
            falloff: self.falloff,
            material_id: self.material_id,
            color_packed: 0,
        }
    }
}

/// Transform a world-space position into an object's local coordinate space.
///
/// Applies inverse rotation then inverse scale to `(world_pos - object_origin)`.
pub fn world_to_object_local(
    world_pos: Vec3,
    object_origin: Vec3,
    object_rotation: Quat,
    object_scale: f32,
) -> Vec3 {
    let relative = world_pos - object_origin;
    let unrotated = object_rotation.inverse() * relative;
    unrotated / object_scale.max(1e-6)
}

impl Default for Brush {
    /// Default brush: Add sphere, radius 0.5m, material 1.
    fn default() -> Self {
        Self::add_sphere(0.5, 1)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    // -- Default --

    #[test]
    fn default_brush_is_add_sphere() {
        let b = Brush::default();
        assert_eq!(b.brush_type, BrushType::Add);
        assert_eq!(b.shape, BrushShape::Sphere);
        assert_eq!(b.radius, 0.5);
        assert_eq!(b.strength, 1.0);
        assert_eq!(b.material_id, 1);
        assert_eq!(b.falloff, FalloffCurve::Smooth);
    }

    // -- Factory methods --

    #[test]
    fn add_sphere_factory() {
        let b = Brush::add_sphere(1.0, 42);
        assert_eq!(b.brush_type, BrushType::Add);
        assert_eq!(b.shape, BrushShape::Sphere);
        assert_eq!(b.radius, 1.0);
        assert_eq!(b.strength, 1.0);
        assert_eq!(b.material_id, 42);
        assert!((b.blend_k - 0.3).abs() < 1e-6, "blend_k = {}", b.blend_k);
    }

    #[test]
    fn subtract_sphere_factory() {
        let b = Brush::subtract_sphere(2.0);
        assert_eq!(b.brush_type, BrushType::Subtract);
        assert_eq!(b.shape, BrushShape::Sphere);
        assert_eq!(b.radius, 2.0);
        assert_eq!(b.material_id, 0);
        assert!((b.blend_k - 0.6).abs() < 1e-6);
    }

    #[test]
    fn smooth_sphere_factory() {
        let b = Brush::smooth_sphere(0.5, 0.7);
        assert_eq!(b.brush_type, BrushType::Smooth);
        assert_eq!(b.shape, BrushShape::Sphere);
        assert_eq!(b.radius, 0.5);
        assert!((b.strength - 0.7).abs() < 1e-6);
        assert_eq!(b.blend_k, 0.0);
    }

    #[test]
    fn smooth_sphere_clamps_strength() {
        let b = Brush::smooth_sphere(0.5, 1.5);
        assert!((b.strength - 1.0).abs() < 1e-6);
        let b2 = Brush::smooth_sphere(0.5, -0.5);
        assert!((b2.strength - 0.0).abs() < 1e-6);
    }

    #[test]
    fn paint_sphere_factory() {
        let b = Brush::paint_sphere(1.0, 99);
        assert_eq!(b.brush_type, BrushType::Paint);
        assert_eq!(b.shape, BrushShape::Sphere);
        assert_eq!(b.radius, 1.0);
        assert_eq!(b.material_id, 99);
        assert_eq!(b.blend_k, 0.0);
    }

    // -- edit_type conversion --

    #[test]
    fn edit_type_add() {
        let b = Brush::add_sphere(1.0, 1);
        assert_eq!(b.edit_type(), EditType::SmoothUnion);
    }

    #[test]
    fn edit_type_subtract() {
        let b = Brush::subtract_sphere(1.0);
        assert_eq!(b.edit_type(), EditType::SmoothSubtract);
    }

    #[test]
    fn edit_type_smooth() {
        let b = Brush::smooth_sphere(1.0, 0.5);
        assert_eq!(b.edit_type(), EditType::Smooth);
    }

    #[test]
    fn edit_type_paint() {
        let b = Brush::paint_sphere(1.0, 1);
        assert_eq!(b.edit_type(), EditType::Paint);
    }

    #[test]
    fn edit_type_all_variants() {
        let cases = [
            (BrushType::Add, EditType::SmoothUnion),
            (BrushType::Subtract, EditType::SmoothSubtract),
            (BrushType::Smooth, EditType::Smooth),
            (BrushType::Flatten, EditType::Flatten),
            (BrushType::Paint, EditType::Paint),
            (BrushType::ColorPaint, EditType::ColorPaint),
        ];
        for (bt, expected_et) in cases {
            let mut b = Brush::default();
            b.brush_type = bt;
            assert_eq!(b.edit_type(), expected_et, "BrushType::{bt:?} -> {expected_et:?}");
        }
    }

    // -- shape_type conversion --

    #[test]
    fn shape_type_all_variants() {
        let cases = [
            (BrushShape::Sphere, ShapeType::Sphere),
            (BrushShape::Cube, ShapeType::Box),
            (BrushShape::Capsule, ShapeType::Capsule),
            (BrushShape::Cylinder, ShapeType::Cylinder),
        ];
        for (bs, expected_st) in cases {
            let mut b = Brush::default();
            b.shape = bs;
            assert_eq!(b.shape_type(), expected_st, "BrushShape::{bs:?} -> {expected_st:?}");
        }
    }

    // -- Clone + Debug --

    #[test]
    fn brush_clone() {
        let b = Brush::add_sphere(1.0, 5);
        let b2 = b.clone();
        assert_eq!(b.brush_type, b2.brush_type);
        assert_eq!(b.radius, b2.radius);
        assert_eq!(b.material_id, b2.material_id);
    }

    #[test]
    fn brush_debug() {
        let b = Brush::default();
        let s = format!("{b:?}");
        assert!(s.contains("Add"), "debug string: {s}");
        assert!(s.contains("Sphere"), "debug string: {s}");
    }

    // -- world_to_object_local --

    #[test]
    fn world_to_local_identity() {
        let local = world_to_object_local(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
        );
        assert!((local - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn world_to_local_with_offset() {
        let local = world_to_object_local(
            Vec3::new(5.0, 3.0, 1.0),
            Vec3::new(2.0, 1.0, 0.0),
            Quat::IDENTITY,
            1.0,
        );
        assert!((local - Vec3::new(3.0, 2.0, 1.0)).length() < 1e-5);
    }

    #[test]
    fn world_to_local_with_scale() {
        let local = world_to_object_local(
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            2.0,
        );
        assert!((local - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn world_to_local_with_rotation() {
        // Object rotated 90° around Y: local +X maps to world -Z,
        // so inverse maps world +X to local +Z.
        let rot = Quat::from_rotation_y(FRAC_PI_2);
        let local = world_to_object_local(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            rot,
            1.0,
        );
        assert!((local - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-4);
    }

    // -- to_edit_op --

    #[test]
    fn to_edit_op_identity_transform() {
        let brush = Brush::add_sphere(0.5, 3);
        let op = brush.to_edit_op(
            42,
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
        );
        assert_eq!(op.object_id, 42);
        assert!((op.position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert_eq!(op.dimensions.x, 0.5);
        assert_eq!(op.edit_type, EditType::SmoothUnion);
        assert_eq!(op.material_id, 3);
    }

    #[test]
    fn to_edit_op_scales_radius() {
        let brush = Brush::add_sphere(1.0, 1);
        let op = brush.to_edit_op(
            1,
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            2.0, // object at 2x scale
        );
        // Position divided by scale
        assert!((op.position.x - 1.0).abs() < 1e-5);
        // Radius divided by scale
        assert!((op.dimensions.x - 0.5).abs() < 1e-5);
        // blend_k divided by scale (1.0 * 0.3 / 2.0 = 0.15)
        assert!((op.blend_k - 0.15).abs() < 1e-5);
    }
}
