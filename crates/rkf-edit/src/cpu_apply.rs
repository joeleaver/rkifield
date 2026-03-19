//! CPU-side CSG evaluation for real-time sculpting.
//!
//! Applies [`EditOp`] operations directly to CPU brick pool data, returning
//! the list of modified brick pool slot indices for targeted GPU upload.
//!
//! This avoids GPU readback complexity — edits are computed on CPU (sub-ms for
//! typical brush sizes of 10–50 bricks) and uploaded via `queue.write_buffer()`.

use glam::Vec3;

use crate::edit_op::{AffectedBrick, EditOp};
use crate::types::{EditType, FalloffCurve, ShapeType};
use rkf_core::brick::Brick;
use rkf_core::brick_pool::Pool;
use rkf_core::sdf::smin;
use rkf_core::voxel::VoxelSample;

/// Apply a CSG/sculpt edit operation to the CPU brick pool.
///
/// For each affected brick, iterates all 512 voxels (8×8×8), evaluates the
/// edit shape SDF, and applies the CSG operation (union, subtract, smooth, etc.)
/// to modify the distance field and/or material data in-place.
///
/// # Returns
///
/// A deduplicated list of brick pool slot indices that were modified.
/// Use these to perform targeted `queue.write_buffer()` uploads to the GPU.
pub fn apply_edit_cpu(
    pool: &mut Pool<Brick>,
    affected: &[AffectedBrick],
    op: &EditOp,
) -> Vec<u32> {
    let inv_rot = op.rotation.inverse();
    let mut modified_slots = Vec::with_capacity(affected.len());

    // Maximum extent of the brush shape: the deepest point inside any brush has
    // |shape_d| ≤ max_dim.  Used to build the "effective brush at strength t":
    //   effective_shape_d = shape_d + (1 − strength) × max_extent
    // This replaces the old lerp(existing, csg_result, strength) pattern.
    // At strength=1: effective_shape_d = shape_d  → full brush stamped.
    // At strength=0: effective_shape_d very positive → brush has no effect.
    // Because smin/min of two valid SDFs is a valid SDF, and the offset just
    // shifts the brush surface inward, the result is always an approximately
    // valid SDF — gradient error is bounded to the smin blend zone (width k),
    // not spread across the entire brush volume as the lerp formula did.
    let max_extent = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);

    for ab in affected {
        let slot = ab.brick_base_index / 512;
        let vs = ab.voxel_size;
        let brick_min = Vec3::from(ab.brick_local_min);
        let brick = pool.get_mut(slot);
        let mut any_changed = false;

        for vz in 0u32..8 {
            for vy in 0u32..8 {
                for vx in 0u32..8 {
                    // Voxel center in object-local space.
                    let voxel_pos = brick_min
                        + Vec3::new(vx as f32, vy as f32, vz as f32) * vs
                        + Vec3::splat(vs * 0.5);

                    // Transform to edit-local space.
                    let edit_local = inv_rot * (voxel_pos - op.position);

                    // Evaluate the edit shape SDF at this position.
                    let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);

                    // Compute distance from voxel to edit center for falloff.
                    let dist_to_center = (voxel_pos - op.position).length();
                    let falloff_weight = evaluate_falloff(
                        op.falloff,
                        dist_to_center,
                        op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z)
                            + op.blend_k,
                    );

                    if falloff_weight <= 0.0 {
                        continue;
                    }

                    let sample = brick.sample(vx, vy, vz);
                    let existing_d = half::f16::to_f32(sample.distance());
                    let strength = op.strength * falloff_weight;

                    // Threshold for considering a voxel "meaningfully changed".
                    // Peripheral bricks within the falloff sphere but far from actual
                    // geometry will have smin(vs*2.0, large_effective_d, k) ≈ vs*2.0
                    // — the CSG result is indistinguishable from the initial fill.
                    // Using this threshold prevents those bricks from being counted
                    // as "modified", so the revert-to-EMPTY_SLOT step can clean them up.
                    let change_threshold = ab.voxel_size * 0.1;

                    match EditType::from_u8(op.edit_type as u8).unwrap_or(EditType::CsgUnion) {
                        EditType::CsgUnion => {
                            let effective_shape_d = shape_d + (1.0 - strength) * max_extent;
                            let new_d = existing_d.min(effective_shape_d);
                            let mat = if effective_shape_d < existing_d {
                                op.material_id
                            } else if new_d < 0.0 && sample.material_id() == 0 {
                                op.material_id
                            } else {
                                sample.material_id()
                            };
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, mat, sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::CsgSubtract => {
                            let effective_shape_d = shape_d + (1.0 - strength) * max_extent;
                            let new_d = existing_d.max(-effective_shape_d);
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, sample.material_id(), sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::CsgIntersect => {
                            let effective_shape_d = shape_d - (1.0 - strength) * max_extent;
                            let new_d = existing_d.max(effective_shape_d);
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, sample.material_id(), sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::SmoothUnion => {
                            let k = op.blend_k.max(0.001);
                            let effective_shape_d = shape_d + (1.0 - strength) * max_extent;
                            let new_d = smin(existing_d, effective_shape_d, k);
                            let mat = if effective_shape_d < existing_d {
                                op.material_id
                            } else if new_d < 0.0 && sample.material_id() == 0 {
                                op.material_id
                            } else {
                                sample.material_id()
                            };
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, mat, sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::SmoothSubtract => {
                            let k = op.blend_k.max(0.001);
                            let effective_shape_d = shape_d + (1.0 - strength) * max_extent;
                            let new_d = -smin(-existing_d, effective_shape_d, k);
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, sample.material_id(), sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::Smooth => {
                            let new_d = existing_d * (1.0 - strength * 0.3);
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, sample.material_id(), sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::Flatten => {
                            let plane_d = edit_local.y;
                            let new_d = lerp(existing_d, plane_d, strength * 0.5);
                            brick.set(vx, vy, vz, VoxelSample::new_blended(
                                new_d, sample.material_id(), sample.secondary_material_id(), sample.blend_weight(),
                            ));
                            any_changed |= (new_d - existing_d).abs() > change_threshold;
                        }
                        EditType::Paint => {
                            if existing_d.abs() < op.dimensions.x * 2.0 {
                                brick.set(vx, vy, vz, VoxelSample::new(
                                    existing_d, op.material_id, 0,
                                ));
                                any_changed = true;
                            }
                        }
                        EditType::ColorPaint => {
                            // Color paint writes to companion color pool, not handled here.
                        }
                    }
                }
            }
        }

        if any_changed {
            modified_slots.push(slot);
        }
    }

    modified_slots
}

/// Fill a brick with per-voxel brush SDF values.
///
/// Replaces the old "constant fill" pattern. Instead of filling every voxel
/// with a single arbitrary value (e.g. `vs * 2.0`), this function evaluates
/// the actual brush SDF at each voxel center and stores that as the initial
/// distance. This means:
///
/// - Voxels inside the brush start with correct negative distances.
/// - Voxels outside the brush start with the true distance to the brush surface.
/// - No gradient discontinuity between CSG-written voxels and "background" voxels.
/// - Conservative ray marching: fill IS the SDF, so it never over-estimates.
///
/// Values are clamped to `voxel_size * 8.0` (one brick extent) to cap step sizes
/// for voxels far outside the brush — matching what EMPTY_SLOT returns for stepping.
pub fn fill_brick_with_brush_sdf(
    brick: &mut rkf_core::brick::Brick,
    brick_local_min: glam::Vec3,
    voxel_size: f32,
    op: &crate::edit_op::EditOp,
) {
    let inv_rot = op.rotation.inverse();
    let max_d = voxel_size * 8.0;

    for vz in 0u32..8 {
        for vy in 0u32..8 {
            for vx in 0u32..8 {
                let voxel_center = brick_local_min
                    + glam::Vec3::new(vx as f32, vy as f32, vz as f32) * voxel_size
                    + glam::Vec3::splat(voxel_size * 0.5);
                let edit_local = inv_rot * (voxel_center - op.position);
                let d = evaluate_shape(op.shape_type, &op.dimensions, edit_local).min(max_d);
                let mat = if d < 0.0 { op.material_id } else { 0 };
                brick.set(vx, vy, vz, VoxelSample::new(d, mat, 0));
            }
        }
    }
}

/// Evaluate an analytic SDF shape at a local-space position.
///
/// The position is already in edit-local space (edit rotation applied).
pub fn evaluate_shape(shape: ShapeType, dims: &Vec3, pos: Vec3) -> f32 {
    match shape {
        ShapeType::Sphere => pos.length() - dims.x,
        ShapeType::Box => {
            let q = pos.abs() - *dims;
            let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
            let inside = q.x.max(q.y.max(q.z)).min(0.0);
            outside + inside
        }
        ShapeType::Capsule => {
            let a = Vec3::new(0.0, -dims.y, 0.0);
            let b = Vec3::new(0.0, dims.y, 0.0);
            let pa = pos - a;
            let ba = b - a;
            let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
            (pa - ba * h).length() - dims.x
        }
        ShapeType::Cylinder => {
            let d_radial = Vec3::new(pos.x, 0.0, pos.z).length() - dims.x;
            let d_height = pos.y.abs() - dims.y;
            let outside = glam::Vec2::new(d_radial.max(0.0), d_height.max(0.0)).length();
            let inside = d_radial.max(d_height).min(0.0);
            outside + inside
        }
        ShapeType::Torus => {
            let q_x = Vec3::new(pos.x, 0.0, pos.z).length() - dims.x;
            let q = glam::Vec2::new(q_x, pos.y);
            q.length() - dims.y
        }
        ShapeType::Plane => pos.y - dims.x,
    }
}

/// Evaluate falloff from brush center to edge.
pub fn evaluate_falloff(curve: FalloffCurve, distance: f32, radius: f32) -> f32 {
    if radius <= 0.0 || distance >= radius {
        return 0.0;
    }
    let t = distance / radius;
    match curve {
        FalloffCurve::Linear => 1.0 - t,
        FalloffCurve::Smooth => {
            // Smoothstep: 1 - (3t² - 2t³)
            1.0 - t * t * (3.0 - 2.0 * t)
        }
        FalloffCurve::Sharp => {
            let f = 1.0 - t;
            f * f * f
        }
    }
}

/// Linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edit_op::AffectedBrick;
    use crate::types::{EditType, FalloffCurve, ShapeType};
    use glam::{Quat, Vec3};
    use rkf_core::brick::Brick;
    use rkf_core::brick_pool::Pool;
    use rkf_core::voxel::VoxelSample;

    /// Helper: create a pool with one brick filled with positive SDF (empty space).
    fn setup_pool_with_empty_brick() -> (Pool<Brick>, u32) {
        let mut pool = Pool::<Brick>::new(16);
        let slot = pool.allocate().unwrap();
        // Fill with positive distance (far from surface).
        let brick = pool.get_mut(slot);
        for z in 0u32..8 {
            for y in 0u32..8 {
                for x in 0u32..8 {
                    brick.set(x, y, z, VoxelSample::new(1.0, 0, 0));
                }
            }
        }
        (pool, slot)
    }

    fn make_affected(slot: u32, voxel_size: f32) -> AffectedBrick {
        AffectedBrick {
            brick_base_index: slot * 512,
            brick_local_min: [0.0, 0.0, 0.0],
            voxel_size,
        }
    }

    fn make_union_op(pos: Vec3, radius: f32, material_id: u16) -> EditOp {
        EditOp {
            object_id: 1,
            position: pos,
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothUnion,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(radius, 0.0, 0.0),
            strength: 1.0,
            blend_k: radius * 0.3,
            falloff: FalloffCurve::Smooth,
            material_id,
            color_packed: 0,
        }
    }

    #[test]
    fn sphere_union_adds_material() {
        let (mut pool, slot) = setup_pool_with_empty_brick();
        let voxel_size = 0.05;
        let affected = vec![make_affected(slot, voxel_size)];

        // Place a sphere centered at brick center: (0.2, 0.2, 0.2)
        let op = make_union_op(Vec3::new(0.2, 0.2, 0.2), 0.15, 5);
        let modified = apply_edit_cpu(&mut pool, &affected, &op);

        assert_eq!(modified.len(), 1);
        assert_eq!(modified[0], slot);

        // Check that at least some voxels near the center have negative distance.
        let brick = pool.get(slot);
        let center_sample = brick.sample(4, 4, 4); // near (0.2, 0.2, 0.2) at vs=0.05
        let d = half::f16::to_f32(center_sample.distance());
        assert!(d < 1.0, "center voxel distance should be reduced: {d}");
    }

    #[test]
    fn sphere_subtract_carves() {
        let (mut pool, slot) = setup_pool_with_empty_brick();
        // First, fill with negative SDF (solid material).
        let brick = pool.get_mut(slot);
        for z in 0u32..8 {
            for y in 0u32..8 {
                for x in 0u32..8 {
                    brick.set(x, y, z, VoxelSample::new(-0.5, 2, 0));
                }
            }
        }

        let voxel_size = 0.05;
        let affected = vec![make_affected(slot, voxel_size)];

        let op = EditOp {
            object_id: 1,
            position: Vec3::new(0.2, 0.2, 0.2),
            rotation: Quat::IDENTITY,
            edit_type: EditType::SmoothSubtract,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.15, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.05,
            falloff: FalloffCurve::Smooth,
            material_id: 0,
            color_packed: 0,
        };

        let modified = apply_edit_cpu(&mut pool, &affected, &op);
        assert_eq!(modified.len(), 1);

        // Voxels near the subtraction center should have increased distance.
        let brick = pool.get(slot);
        let center_sample = brick.sample(4, 4, 4);
        let d = half::f16::to_f32(center_sample.distance());
        assert!(d > -0.5, "center distance should be increased by subtract: {d}");
    }

    #[test]
    fn paint_sets_material_near_surface() {
        let (mut pool, slot) = setup_pool_with_empty_brick();
        // Place some voxels near the surface (distance ~0).
        let brick = pool.get_mut(slot);
        for x in 0u32..8 {
            brick.set(x, 4, 4, VoxelSample::new(0.01, 1, 0));
        }

        let voxel_size = 0.05;
        let affected = vec![make_affected(slot, voxel_size)];

        let op = EditOp {
            object_id: 1,
            position: Vec3::new(0.2, 0.2, 0.2),
            rotation: Quat::IDENTITY,
            edit_type: EditType::Paint,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.5, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth,
            material_id: 42,
            color_packed: 0,
        };

        let modified = apply_edit_cpu(&mut pool, &affected, &op);
        assert_eq!(modified.len(), 1);

        // Near-surface voxels should have material 42.
        let brick = pool.get(slot);
        let sample = brick.sample(4, 4, 4);
        assert_eq!(sample.material_id(), 42);
    }

    #[test]
    fn no_affected_returns_empty() {
        let (mut pool, _slot) = setup_pool_with_empty_brick();
        let op = make_union_op(Vec3::ZERO, 0.1, 1);
        let modified = apply_edit_cpu(&mut pool, &[], &op);
        assert!(modified.is_empty());
    }

    #[test]
    fn evaluate_shape_sphere() {
        let d = evaluate_shape(ShapeType::Sphere, &Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO);
        assert!((d - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn evaluate_shape_box() {
        let d = evaluate_shape(ShapeType::Box, &Vec3::splat(0.5), Vec3::ZERO);
        assert!((d - (-0.5)).abs() < 1e-5);
    }

    #[test]
    fn evaluate_falloff_smooth_center() {
        let f = evaluate_falloff(FalloffCurve::Smooth, 0.0, 1.0);
        assert!((f - 1.0).abs() < 1e-5);
    }

    #[test]
    fn evaluate_falloff_smooth_edge() {
        let f = evaluate_falloff(FalloffCurve::Smooth, 1.0, 1.0);
        assert!(f.abs() < 1e-5);
    }

    #[test]
    fn evaluate_falloff_beyond_radius() {
        let f = evaluate_falloff(FalloffCurve::Linear, 1.5, 1.0);
        assert!(f.abs() < 1e-5);
    }

    #[test]
    fn smooth_operation_reduces_distance() {
        let (mut pool, slot) = setup_pool_with_empty_brick();
        // Set varying distances.
        let brick = pool.get_mut(slot);
        for z in 0u32..8 {
            for y in 0u32..8 {
                for x in 0u32..8 {
                    let d = (x as f32 - 3.5) * 0.1; // -0.35 to 0.35
                    brick.set(x, y, z, VoxelSample::new(d, 1, 0));
                }
            }
        }

        let voxel_size = 0.05;
        let affected = vec![make_affected(slot, voxel_size)];

        let op = EditOp {
            object_id: 1,
            position: Vec3::new(0.2, 0.2, 0.2),
            rotation: Quat::IDENTITY,
            edit_type: EditType::Smooth,
            shape_type: ShapeType::Sphere,
            dimensions: Vec3::new(0.5, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: FalloffCurve::Smooth,
            material_id: 0,
            color_packed: 0,
        };

        let modified = apply_edit_cpu(&mut pool, &affected, &op);
        assert!(!modified.is_empty());
    }
}
