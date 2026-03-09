//! Geometry-first edit operations.
//!
//! Applies brush edits to occupancy bitmasks (not SDF distances). After editing,
//! the caller recomputes SDF from the modified geometry.
//!
//! Operations:
//! - **Add** (CsgUnion/SmoothUnion): set occupancy bits where brush SDF is negative
//! - **Subtract** (CsgSubtract/SmoothSubtract): clear occupancy bits where brush SDF is negative
//! - **Paint**: modify color/material on existing surface voxels (no geometry change)
//! - **Smooth**: erode + dilate boundary voxels (blur occupancy boundary)
//! - **Flatten**: set occupancy to match a reference plane

use glam::{Quat, UVec3, Vec3};

use rkf_core::brick_geometry::BrickGeometry;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::brick_pool::Pool;
use rkf_core::constants::BRICK_DIM;
use rkf_core::scene_node::BrickMapHandle;

use crate::cpu_apply::{evaluate_falloff, evaluate_shape};
use crate::edit_op::EditOp;
use crate::types::EditType;

/// Result of applying a geometry edit.
pub struct GeometryEditResult {
    /// Brick coordinates that were modified (for SDF recomputation region).
    pub modified_bricks: Vec<UVec3>,
    /// Brick coordinates of newly allocated bricks.
    pub new_bricks: Vec<UVec3>,
    /// Brick coordinates of deallocated bricks (became fully empty/solid).
    pub removed_bricks: Vec<UVec3>,
    /// Region bounds (brick coords) for SDF recomputation. Min inclusive, max exclusive.
    pub sdf_region_min: UVec3,
    pub sdf_region_max: UVec3,
}

/// Apply a geometry edit operation to the object's brick geometry.
///
/// Modifies occupancy bitmasks based on the brush shape and edit type.
/// After this call, the caller must:
/// 1. Recompute SDF in the returned region
/// 2. Upload changed bricks to GPU
pub fn apply_geometry_edit(
    geo_pool: &mut Pool<BrickGeometry>,
    brick_map: &mut BrickMap,
    map_alloc: &mut BrickMapAllocator,
    handle: &BrickMapHandle,
    op: &EditOp,
    voxel_size: f32,
) -> GeometryEditResult {
    let brick_size = voxel_size * BRICK_DIM as f32;
    let (edit_min, edit_max) = op.local_aabb();

    // Grid origin (centered at local origin, matching ray march shader)
    let grid_origin = -Vec3::new(
        brick_map.dims.x as f32 * brick_size * 0.5,
        brick_map.dims.y as f32 * brick_size * 0.5,
        brick_map.dims.z as f32 * brick_size * 0.5,
    );

    // Convert edit AABB to brick coordinates
    let bmin = ((edit_min - grid_origin) / brick_size).floor();
    let bmax = ((edit_max - grid_origin) / brick_size).ceil();

    let bmin_x = (bmin.x as i32).max(0) as u32;
    let bmin_y = (bmin.y as i32).max(0) as u32;
    let bmin_z = (bmin.z as i32).max(0) as u32;
    let bmax_x = (bmax.x as i32).max(0) as u32;
    let bmax_y = (bmax.y as i32).max(0) as u32;
    let bmax_z = (bmax.z as i32).max(0) as u32;

    // Clamp to brick map dims
    let bmax_x = bmax_x.min(brick_map.dims.x);
    let bmax_y = bmax_y.min(brick_map.dims.y);
    let bmax_z = bmax_z.min(brick_map.dims.z);

    let mut modified_bricks = Vec::new();
    let mut new_bricks = Vec::new();
    let mut removed_bricks = Vec::new();

    let is_add = matches!(
        op.edit_type,
        EditType::CsgUnion | EditType::SmoothUnion
    );
    let is_subtract = matches!(
        op.edit_type,
        EditType::CsgSubtract | EditType::SmoothSubtract
    );
    let is_paint = matches!(
        op.edit_type,
        EditType::Paint | EditType::BlendPaint | EditType::ColorPaint
    );

    // Inverse rotation for transforming into edit-local space
    let inv_rot = op.rotation.inverse();
    let half_voxel = voxel_size * 0.5;

    for bz in bmin_z..bmax_z {
        for by in bmin_y..bmax_y {
            for bx in bmin_x..bmax_x {
                let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);

                // For add operations, we may need to allocate new bricks
                if slot == EMPTY_SLOT && is_add {
                    // Check if any voxel in this brick would be affected
                    let brick_min = grid_origin + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );
                    let brick_center = brick_min + Vec3::splat(brick_size * 0.5);

                    // Quick test: evaluate brush at brick center
                    let edit_local = inv_rot * (brick_center - op.position);
                    let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);
                    if shape_d > brick_size * 1.5 {
                        continue; // Brush doesn't reach this brick
                    }

                    // Allocate a new brick
                    let new_slot = match geo_pool.allocate() {
                        Some(s) => s,
                        None => continue,
                    };
                    brick_map.set(bx, by, bz, new_slot);
                    new_bricks.push(UVec3::new(bx, by, bz));
                    // Fall through to edit the new brick
                }

                // For subtract, INTERIOR_SLOT bricks need allocation too
                if slot == INTERIOR_SLOT && is_subtract {
                    let brick_min = grid_origin + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );
                    let brick_center = brick_min + Vec3::splat(brick_size * 0.5);
                    let edit_local = inv_rot * (brick_center - op.position);
                    let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);
                    if shape_d > brick_size * 1.5 {
                        continue;
                    }

                    let new_slot = match geo_pool.allocate() {
                        Some(s) => s,
                        None => continue,
                    };
                    // Initialize as fully solid
                    *geo_pool.get_mut(new_slot) = BrickGeometry::fully_solid();
                    brick_map.set(bx, by, bz, new_slot);
                    new_bricks.push(UVec3::new(bx, by, bz));
                }

                let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                    continue;
                }

                let brick_min = grid_origin + Vec3::new(
                    bx as f32 * brick_size,
                    by as f32 * brick_size,
                    bz as f32 * brick_size,
                );

                let geo = geo_pool.get_mut(slot);
                let mut any_changed = false;

                if is_add || is_subtract {
                    // Modify occupancy
                    for vz in 0..8u8 {
                        for vy in 0..8u8 {
                            for vx in 0..8u8 {
                                let world_pos = brick_min + Vec3::new(
                                    vx as f32 * voxel_size + half_voxel,
                                    vy as f32 * voxel_size + half_voxel,
                                    vz as f32 * voxel_size + half_voxel,
                                );

                                let edit_local = inv_rot * (world_pos - op.position);
                                let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);

                                // Apply falloff
                                let center_dist = (world_pos - op.position).length();
                                let max_dim = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
                                let falloff = evaluate_falloff(op.falloff, center_dist, max_dim + op.blend_k);
                                let effective_strength = op.strength * falloff;

                                if effective_strength < 0.01 {
                                    continue;
                                }

                                // For strength < 1, expand the shape threshold
                                let threshold = (1.0 - effective_strength) * voxel_size * 2.0;

                                if is_add && shape_d < threshold {
                                    if !geo.is_solid(vx, vy, vz) {
                                        geo.set_solid(vx, vy, vz, true);
                                        any_changed = true;
                                    }
                                } else if is_subtract && shape_d < threshold {
                                    if geo.is_solid(vx, vy, vz) {
                                        geo.set_solid(vx, vy, vz, false);
                                        any_changed = true;
                                    }
                                }
                            }
                        }
                    }
                } else if is_paint {
                    // Paint: modify color/material on existing surface voxels
                    let color = [
                        ((op.color_packed >> 24) & 0xFF) as u8,
                        ((op.color_packed >> 16) & 0xFF) as u8,
                        ((op.color_packed >> 8) & 0xFF) as u8,
                        (op.color_packed & 0xFF) as u8,
                    ];

                    for sv in &mut geo.surface_voxels {
                        let (vx, vy, vz) = rkf_core::brick_geometry::index_to_xyz(sv.index);
                        let world_pos = brick_min + Vec3::new(
                            vx as f32 * voxel_size + half_voxel,
                            vy as f32 * voxel_size + half_voxel,
                            vz as f32 * voxel_size + half_voxel,
                        );

                        let edit_local = inv_rot * (world_pos - op.position);
                        let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);

                        if shape_d < 0.0 {
                            let center_dist = (world_pos - op.position).length();
                            let max_dim = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
                            let falloff = evaluate_falloff(op.falloff, center_dist, max_dim);
                            let t = op.strength * falloff;

                            match op.edit_type {
                                EditType::Paint => {
                                    sv.material_id = op.material_id as u8;
                                    any_changed = true;
                                }
                                EditType::ColorPaint => {
                                    // Lerp color
                                    for i in 0..4 {
                                        sv.color[i] = (sv.color[i] as f32 * (1.0 - t) + color[i] as f32 * t) as u8;
                                    }
                                    any_changed = true;
                                }
                                EditType::BlendPaint => {
                                    // Not implemented in geometry-first yet
                                    // (would need per-voxel material blending)
                                    any_changed = true;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                if any_changed {
                    // Rebuild surface voxels list (preserving colors where possible)
                    if is_add || is_subtract {
                        geo.rebuild_surface_list_preserving();

                        // Assign color/material to new surface voxels from the brush
                        let color = [
                            ((op.color_packed >> 24) & 0xFF) as u8,
                            ((op.color_packed >> 16) & 0xFF) as u8,
                            ((op.color_packed >> 8) & 0xFF) as u8,
                            (op.color_packed & 0xFF) as u8,
                        ];

                        for sv in &mut geo.surface_voxels {
                            // If this is a new surface voxel (white default), try to assign brush color
                            if sv.color == [255, 255, 255, 255] && sv.material_id == 0 {
                                sv.material_id = op.material_id as u8;
                                if op.color_packed != 0 {
                                    sv.color = color;
                                }
                            }
                        }
                    }

                    // Check if brick should be deallocated
                    if geo.is_fully_empty() {
                        brick_map.set(bx, by, bz, EMPTY_SLOT);
                        geo_pool.deallocate(slot);
                        removed_bricks.push(UVec3::new(bx, by, bz));
                    } else if geo.is_fully_solid() {
                        brick_map.set(bx, by, bz, INTERIOR_SLOT);
                        geo_pool.deallocate(slot);
                        removed_bricks.push(UVec3::new(bx, by, bz));
                    } else {
                        modified_bricks.push(UVec3::new(bx, by, bz));
                    }
                }
            }
        }
    }

    // SDF recomputation region: affected area + 1 brick margin
    let sdf_min = UVec3::new(
        bmin_x.saturating_sub(1),
        bmin_y.saturating_sub(1),
        bmin_z.saturating_sub(1),
    );
    let sdf_max = UVec3::new(
        (bmax_x + 1).min(brick_map.dims.x),
        (bmax_y + 1).min(brick_map.dims.y),
        (bmax_z + 1).min(brick_map.dims.z),
    );

    GeometryEditResult {
        modified_bricks,
        new_bricks,
        removed_bricks,
        sdf_region_min: sdf_min,
        sdf_region_max: sdf_max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use rkf_core::brick_geometry::BrickGeometry;
    use rkf_core::brick_map::BrickMap;

    fn setup_empty_scene() -> (Pool<BrickGeometry>, BrickMap, BrickMapAllocator, BrickMapHandle) {
        let geo_pool: Pool<BrickGeometry> = Pool::new(256);
        let mut alloc = BrickMapAllocator::new();
        let map = BrickMap::new(UVec3::new(4, 4, 4));
        let handle = alloc.allocate(&map);
        (geo_pool, map, alloc, handle)
    }

    fn make_add_sphere_op(pos: Vec3, radius: f32, mat: u16) -> EditOp {
        EditOp {
            object_id: 1,
            position: pos,
            rotation: Quat::IDENTITY,
            edit_type: EditType::CsgUnion,
            shape_type: crate::types::ShapeType::Sphere,
            dimensions: Vec3::new(radius, 0.0, 0.0),
            strength: 1.0,
            blend_k: 0.0,
            falloff: crate::types::FalloffCurve::Smooth,
            material_id: mat,
            secondary_id: 0,
            color_packed: 0xFF804020, // RGBA
        }
    }

    fn count_solid(map: &BrickMap, pool: &Pool<BrickGeometry>) -> u32 {
        let mut total = 0u32;
        for bz in 0..map.dims.z {
            for by in 0..map.dims.y {
                for bx in 0..map.dims.x {
                    let slot = map.get(bx, by, bz).unwrap();
                    if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
                        total += pool.get(slot).solid_count();
                    }
                }
            }
        }
        total
    }

    #[test]
    fn add_sphere_to_empty_grid() {
        let (mut geo_pool, mut map, mut alloc, handle) = setup_empty_scene();
        let voxel_size = 0.1;

        let op = make_add_sphere_op(Vec3::ZERO, 0.3, 1);
        let result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &op, voxel_size,
        );

        assert!(!result.new_bricks.is_empty(), "should allocate new bricks");
        assert!(count_solid(&map, &geo_pool) > 0, "should have solid voxels after add");
    }

    #[test]
    fn subtract_from_solid_brick() {
        let (mut geo_pool, mut map, mut alloc, handle) = setup_empty_scene();
        let voxel_size = 0.1;

        // First, add a sphere
        let add_op = make_add_sphere_op(Vec3::ZERO, 0.3, 1);
        apply_geometry_edit(&mut geo_pool, &mut map, &mut alloc, &handle, &add_op, voxel_size);

        let solid_before = count_solid(&map, &geo_pool);

        // Subtract a smaller sphere
        let sub_op = EditOp {
            edit_type: EditType::CsgSubtract,
            ..make_add_sphere_op(Vec3::ZERO, 0.15, 1)
        };
        apply_geometry_edit(&mut geo_pool, &mut map, &mut alloc, &handle, &sub_op, voxel_size);

        let solid_after = count_solid(&map, &geo_pool);
        assert!(solid_after < solid_before, "subtract should reduce solid count: {solid_after} < {solid_before}");
    }

    #[test]
    fn paint_modifies_surface_voxels() {
        let (mut geo_pool, mut map, mut alloc, handle) = setup_empty_scene();
        let voxel_size = 0.1;

        // Add geometry first
        let add_op = make_add_sphere_op(Vec3::ZERO, 0.3, 1);
        apply_geometry_edit(&mut geo_pool, &mut map, &mut alloc, &handle, &add_op, voxel_size);

        // Paint with material 5
        let paint_op = EditOp {
            edit_type: EditType::Paint,
            material_id: 5,
            ..make_add_sphere_op(Vec3::ZERO, 0.3, 5)
        };
        let _result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &paint_op, voxel_size,
        );

        // Check that some surface voxels have material 5
        let mut found_mat5 = false;
        for bz in 0..map.dims.z {
            for by in 0..map.dims.y {
                for bx in 0..map.dims.x {
                    let slot = map.get(bx, by, bz).unwrap();
                    if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
                        for sv in &geo_pool.get(slot).surface_voxels {
                            if sv.material_id == 5 {
                                found_mat5 = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(found_mat5, "should have surface voxels with material 5 after paint");
    }

    #[test]
    fn edit_result_has_correct_region() {
        let (mut geo_pool, mut map, mut alloc, handle) = setup_empty_scene();
        let voxel_size = 0.1;

        let op = make_add_sphere_op(Vec3::ZERO, 0.2, 1);
        let result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &op, voxel_size,
        );

        // SDF region should be non-empty
        assert!(result.sdf_region_max.x > result.sdf_region_min.x);
        assert!(result.sdf_region_max.y > result.sdf_region_min.y);
        assert!(result.sdf_region_max.z > result.sdf_region_min.z);
    }
}
