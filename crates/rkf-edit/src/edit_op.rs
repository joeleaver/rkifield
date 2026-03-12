//! Edit operation orchestration for v2 object-local editing.
//!
//! An [`EditOp`] describes a CSG or sculpt operation in an object's local
//! coordinate space. The orchestration functions find affected bricks in
//! the object's brick map and prepare GPU-ready [`EditParams`].

use glam::{Quat, Vec3};

use crate::types::{EditParams, EditType, FalloffCurve, ShapeType};
use rkf_core::brick_map::{BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::scene_node::BrickMapHandle;

/// A CSG/sculpt edit operation targeting a specific object in local space.
#[derive(Debug, Clone)]
pub struct EditOp {
    /// Object ID within the scene.
    pub object_id: u32,
    /// Edit position in object-local space.
    pub position: Vec3,
    /// Edit rotation in object-local space.
    pub rotation: Quat,
    /// Edit type (CSG union, subtract, smooth, paint, etc.).
    pub edit_type: EditType,
    /// SDF shape primitive.
    pub shape_type: ShapeType,
    /// Shape dimensions (half-extents or radius, depends on shape_type).
    pub dimensions: Vec3,
    /// Brush strength (0.0 .. 1.0).
    pub strength: f32,
    /// Smooth blend radius.
    pub blend_k: f32,
    /// Falloff curve.
    pub falloff: FalloffCurve,
    /// Primary material ID.
    pub material_id: u16,
    /// Packed RGBA color (for paint operations).
    pub color_packed: u32,
}

/// Info about a single affected brick for GPU dispatch.
#[derive(Debug, Clone, Copy)]
pub struct AffectedBrick {
    /// Base index into the brick pool (slot * 512).
    pub brick_base_index: u32,
    /// Object-local minimum corner position of this brick.
    pub brick_local_min: [f32; 3],
    /// Voxel size at this brick's resolution.
    pub voxel_size: f32,
}

impl EditOp {
    /// Convert this edit op into GPU-ready EditParams for a specific brick.
    pub fn to_edit_params(&self, brick: &AffectedBrick) -> EditParams {
        let mut params = EditParams::csg(
            self.edit_type,
            self.shape_type,
            self.position.into(),
            [self.rotation.x, self.rotation.y, self.rotation.z, self.rotation.w],
            self.dimensions.into(),
            self.strength,
            self.blend_k,
            self.falloff,
            self.material_id,
        );
        params.color_packed = self.color_packed;
        params = params.with_brick_info(brick.brick_base_index, brick.brick_local_min, brick.voxel_size);
        params
    }

    /// Compute the AABB of the edit shape in object-local space.
    ///
    /// This is a conservative bound — the actual affected region may be smaller
    /// for rotated shapes.
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        // Use the maximum dimension as radius for conservative bound
        let max_dim = self.dimensions.x.max(self.dimensions.y).max(self.dimensions.z);
        // Add blend_k to account for smooth blend overflow
        let extent = max_dim + self.blend_k;
        (
            self.position - Vec3::splat(extent),
            self.position + Vec3::splat(extent),
        )
    }
}

/// Find all bricks in an object's brick map that overlap with an edit operation.
///
/// Returns a list of [`AffectedBrick`] entries for each non-empty brick
/// that overlaps the edit's AABB.
///
/// # Parameters
/// - `op` — The edit operation (in object-local space)
/// - `handle` — The object's brick map handle
/// - `allocator` — The brick map allocator (to look up slot values)
/// - `voxel_size` — The voxel size for this object
/// - `object_aabb_min` — The min corner of the object's local AABB (brick grid origin)
pub fn find_affected_bricks(
    op: &EditOp,
    handle: &BrickMapHandle,
    allocator: &BrickMapAllocator,
    voxel_size: f32,
    object_aabb_min: Vec3,
) -> Vec<AffectedBrick> {
    let brick_size = voxel_size * 8.0; // 8 voxels per brick axis
    let (edit_min, edit_max) = op.local_aabb();

    // Convert edit AABB to brick coordinates
    let bmin = ((edit_min - object_aabb_min) / brick_size).floor();
    let bmax = ((edit_max - object_aabb_min) / brick_size - Vec3::splat(0.001)).ceil();

    let bmin_x = (bmin.x as i32).max(0) as u32;
    let bmin_y = (bmin.y as i32).max(0) as u32;
    let bmin_z = (bmin.z as i32).max(0) as u32;
    let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
    let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
    let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

    let mut affected = Vec::new();

    for bz in bmin_z..=bmax_z {
        for by in bmin_y..=bmax_y {
            for bx in bmin_x..=bmax_x {
                if let Some(slot) = allocator.get_entry(handle, bx, by, bz) {
                    if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
                        let brick_local_min = object_aabb_min
                            + Vec3::new(
                                bx as f32 * brick_size,
                                by as f32 * brick_size,
                                bz as f32 * brick_size,
                            );
                        affected.push(AffectedBrick {
                            brick_base_index: slot * 512, // 8*8*8 voxels per brick
                            brick_local_min: brick_local_min.into(),
                            voxel_size,
                        });
                    }
                }
            }
        }
    }

    affected
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, UVec3, Vec3};
    use rkf_core::brick_map::{BrickMap, BrickMapAllocator};

    fn test_edit_op() -> EditOp {
        EditOp {
            object_id: 1,
            position: Vec3::new(0.5, 0.5, 0.5),
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothUnion,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.3, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.05,
            falloff: FalloffCurve::Smooth,
            material_id: 1,
            color_packed: 0,
        }
    }

    #[test]
    fn edit_op_local_aabb() {
        let op = test_edit_op();
        let (min, max) = op.local_aabb();
        // extent = max(0.3, 0.0, 0.0) + 0.05 = 0.35
        let expected_extent = 0.35;
        assert!((min.x - (0.5 - expected_extent)).abs() < 1e-5);
        assert!((max.x - (0.5 + expected_extent)).abs() < 1e-5);
        assert!((min.y - (0.5 - expected_extent)).abs() < 1e-5);
        assert!((max.y - (0.5 + expected_extent)).abs() < 1e-5);
    }

    #[test]
    fn edit_op_local_aabb_includes_blend_k() {
        let mut op = test_edit_op();
        op.blend_k = 0.2;
        op.dimensions = Vec3::new(0.5, 0.5, 0.5);
        let (min, max) = op.local_aabb();
        // extent = 0.5 + 0.2 = 0.7
        assert!((max.x - min.x - 1.4).abs() < 1e-5);
    }

    #[test]
    fn edit_op_to_edit_params() {
        let op = test_edit_op();
        let brick = AffectedBrick {
            brick_base_index: 1024,
            brick_local_min: [1.0, 2.0, 3.0],
            voxel_size: 0.02,
        };
        let params = op.to_edit_params(&brick);
        assert_eq!(params.edit_type, EditType::SmoothUnion.as_u32());
        assert_eq!(params.shape_type, ShapeType::Sphere.as_u32());
        assert_eq!(params.position[0], 0.5);
        assert_eq!(params.position[1], 0.5);
        assert_eq!(params.brick_base_index, 1024);
        assert_eq!(params.brick_local_min, [1.0, 2.0, 3.0]);
        assert_eq!(params.voxel_size, 0.02);
        assert_eq!(params.material_id, 1);
    }

    #[test]
    fn find_affected_bricks_empty_map() {
        let op = test_edit_op();
        let mut alloc = BrickMapAllocator::new();
        let map = BrickMap::new(UVec3::new(4, 4, 4));
        let handle = alloc.allocate(&map);

        let bricks = find_affected_bricks(&op, &handle, &alloc, 0.02, Vec3::ZERO);
        assert!(bricks.is_empty(), "empty map should have no affected bricks");
    }

    #[test]
    fn find_affected_bricks_finds_overlapping() {
        let op = EditOp {
            object_id: 1,
            position: Vec3::new(0.08, 0.08, 0.08), // center of brick (0,0,0) at voxel_size=0.02
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothUnion,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.05, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth,
            material_id: 1,
            color_packed: 0,
        };

        let mut alloc = BrickMapAllocator::new();
        let mut map = BrickMap::new(UVec3::new(4, 4, 4));
        // Allocate brick at (0,0,0)
        map.set(0, 0, 0, 42);
        let handle = alloc.allocate(&map);

        let bricks = find_affected_bricks(&op, &handle, &alloc, 0.02, Vec3::ZERO);
        assert_eq!(bricks.len(), 1);
        assert_eq!(bricks[0].brick_base_index, 42 * 512);
        assert_eq!(bricks[0].voxel_size, 0.02);
    }

    #[test]
    fn find_affected_bricks_skips_empty_slots() {
        let op = EditOp {
            object_id: 1,
            position: Vec3::new(0.24, 0.08, 0.08), // spans bricks (0,0,0) and (1,0,0)
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothUnion,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.1, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth,
            material_id: 1,
            color_packed: 0,
        };

        let mut alloc = BrickMapAllocator::new();
        let mut map = BrickMap::new(UVec3::new(4, 4, 4));
        // Only brick (1,0,0) is allocated, (0,0,0) is empty
        map.set(1, 0, 0, 10);
        let handle = alloc.allocate(&map);

        let bricks = find_affected_bricks(&op, &handle, &alloc, 0.02, Vec3::ZERO);
        assert_eq!(bricks.len(), 1);
        assert_eq!(bricks[0].brick_base_index, 10 * 512);
    }

    #[test]
    fn find_affected_bricks_out_of_range() {
        let op = EditOp {
            object_id: 1,
            position: Vec3::new(10.0, 10.0, 10.0), // way outside the brick map
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothUnion,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.1, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth,
            material_id: 1,
            color_packed: 0,
        };

        let mut alloc = BrickMapAllocator::new();
        let mut map = BrickMap::new(UVec3::new(4, 4, 4));
        map.set(0, 0, 0, 1);
        let handle = alloc.allocate(&map);

        let bricks = find_affected_bricks(&op, &handle, &alloc, 0.02, Vec3::ZERO);
        assert!(bricks.is_empty());
    }
}
