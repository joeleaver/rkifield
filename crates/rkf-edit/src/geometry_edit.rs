//! Geometry-first edit operations.
//!
//! Applies brush edits to occupancy bitmasks (not SDF distances). After editing,
//! the caller recomputes SDF from the modified geometry.
//!
//! Sculpt operations use morphological dilation/erosion:
//! - **Add**: iteratively dilate the surface outward within the brush footprint
//! - **Subtract**: iteratively erode the surface inward within the brush footprint
//! - **Paint**: modify color/material on existing surface voxels (no geometry change)
//!
//! The brush radius defines the footprint on the surface. The strength controls
//! how many voxel layers to add/remove (1–8). No brush shape SDF is involved
//! in add/subtract — the new material conforms to the existing surface contour.

use glam::{UVec3, Vec3};

use rkf_core::brick_geometry::BrickGeometry;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT, INTERIOR_SLOT};
use rkf_core::brick_pool::Pool;
use rkf_core::constants::BRICK_DIM;
use rkf_core::scene_node::BrickMapHandle;

use crate::cpu_apply::{evaluate_falloff, evaluate_shape};
use crate::edit_op::EditOp;
use crate::types::EditType;

/// Maximum voxel layers per stroke.
const MAX_LAYERS: u32 = 8;

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
/// For Add/Subtract: morphological dilation/erosion within the brush footprint.
/// For Paint: modify surface voxel color/material within the brush.
///
/// After this call, the caller must:
/// 1. Recompute SDF in the returned region
/// 2. Upload changed bricks to GPU
pub fn apply_geometry_edit(
    geo_pool: &mut Pool<BrickGeometry>,
    brick_map: &mut BrickMap,
    _map_alloc: &mut BrickMapAllocator,
    _handle: &BrickMapHandle,
    op: &EditOp,
    voxel_size: f32,
) -> GeometryEditResult {
    let brick_size = voxel_size * BRICK_DIM as f32;

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

    // Grid origin (centered at local origin, matching ray march shader)
    let grid_origin = -Vec3::new(
        brick_map.dims.x as f32 * brick_size * 0.5,
        brick_map.dims.y as f32 * brick_size * 0.5,
        brick_map.dims.z as f32 * brick_size * 0.5,
    );

    let half_voxel = voxel_size * 0.5;

    // Brush footprint: which voxels are "under" the brush.
    // We use the brush radius to define the footprint on the surface.
    let brush_radius_sq = {
        let max_dim = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let r = max_dim + op.blend_k;
        r * r
    };

    // Number of layers for add/subtract
    let num_layers = (op.strength * MAX_LAYERS as f32).ceil().max(1.0) as u32;

    // Convert brush AABB to brick coordinates (expanded by num_layers for dilation)
    let layer_expand = num_layers as f32 * voxel_size;
    let (edit_min, edit_max) = op.local_aabb();
    let edit_min = edit_min - Vec3::splat(layer_expand);
    let edit_max = edit_max + Vec3::splat(layer_expand);

    let bmin = ((edit_min - grid_origin) / brick_size).floor();
    let bmax = ((edit_max - grid_origin) / brick_size).ceil();

    let bmin_x = (bmin.x as i32).max(0) as u32;
    let bmin_y = (bmin.y as i32).max(0) as u32;
    let bmin_z = (bmin.z as i32).max(0) as u32;
    let bmax_x = bmax.x.max(0.0) as u32;
    let bmax_y = bmax.y.max(0.0) as u32;
    let bmax_z = bmax.z.max(0.0) as u32;
    let bmax_x = bmax_x.min(brick_map.dims.x);
    let bmax_y = bmax_y.min(brick_map.dims.y);
    let bmax_z = bmax_z.min(brick_map.dims.z);

    let mut modified_bricks = Vec::new();
    let mut new_bricks = Vec::new();
    let mut removed_bricks = Vec::new();

    // ── Paint path (unchanged — operates on surface voxels) ─────────────
    if is_paint {
        let inv_rot = op.rotation.inverse();
        let color = [
            ((op.color_packed >> 24) & 0xFF) as u8,
            ((op.color_packed >> 16) & 0xFF) as u8,
            ((op.color_packed >> 8) & 0xFF) as u8,
            (op.color_packed & 0xFF) as u8,
        ];

        for bz in bmin_z..bmax_z {
            for by in bmin_y..bmax_y {
                for bx in bmin_x..bmax_x {
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
                                    for i in 0..4 {
                                        sv.color[i] = (sv.color[i] as f32 * (1.0 - t)
                                            + color[i] as f32 * t) as u8;
                                    }
                                    any_changed = true;
                                }
                                EditType::BlendPaint => {
                                    any_changed = true;
                                }
                                _ => {}
                            }
                        }
                    }

                    if any_changed {
                        modified_bricks.push(UVec3::new(bx, by, bz));
                    }
                }
            }
        }

        return build_result(modified_bricks, new_bricks, removed_bricks,
            bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z, brick_map);
    }

    // ── Add/Subtract: morphological dilation/erosion ────────────────────

    if !is_add && !is_subtract {
        return build_result(modified_bricks, new_bricks, removed_bricks,
            bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z, brick_map);
    }

    // Helper: check if a voxel is within the brush footprint.
    // Uses distance from brush center (projected), not a shape SDF.
    let in_brush = |world_pos: Vec3| -> bool {
        let d_sq = (world_pos - op.position).length_squared();
        d_sq <= brush_radius_sq
    };

    // Helper: world position of a voxel
    let voxel_world_pos = |bx: u32, by: u32, bz: u32, vx: u8, vy: u8, vz: u8| -> Vec3 {
        grid_origin + Vec3::new(
            bx as f32 * brick_size + vx as f32 * voxel_size + half_voxel,
            by as f32 * brick_size + vy as f32 * voxel_size + half_voxel,
            bz as f32 * brick_size + vz as f32 * voxel_size + half_voxel,
        )
    };

    // Pre-allocate empty bricks in the affected region so dilation can expand into them.
    if is_add {
        for bz in bmin_z..bmax_z {
            for by in bmin_y..bmax_y {
                for bx in bmin_x..bmax_x {
                    let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                    if slot == EMPTY_SLOT {
                        // Check if brush reaches this brick
                        let brick_center = grid_origin + Vec3::new(
                            (bx as f32 + 0.5) * brick_size,
                            (by as f32 + 0.5) * brick_size,
                            (bz as f32 + 0.5) * brick_size,
                        );
                        let dist_sq = (brick_center - op.position).length_squared();
                        let reach = brush_radius_sq.sqrt() + layer_expand + brick_size;
                        if dist_sq <= reach * reach {
                            if let Some(new_slot) = geo_pool.allocate() {
                                brick_map.set(bx, by, bz, new_slot);
                                new_bricks.push(UVec3::new(bx, by, bz));
                            }
                        }
                    }
                }
            }
        }
    }

    // For subtract, allocate INTERIOR_SLOT bricks that are in the brush region.
    if is_subtract {
        for bz in bmin_z..bmax_z {
            for by in bmin_y..bmax_y {
                for bx in bmin_x..bmax_x {
                    let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                    if slot == INTERIOR_SLOT {
                        let brick_center = grid_origin + Vec3::new(
                            (bx as f32 + 0.5) * brick_size,
                            (by as f32 + 0.5) * brick_size,
                            (bz as f32 + 0.5) * brick_size,
                        );
                        let dist_sq = (brick_center - op.position).length_squared();
                        let reach = brush_radius_sq.sqrt() + layer_expand + brick_size;
                        if dist_sq <= reach * reach {
                            if let Some(new_slot) = geo_pool.allocate() {
                                *geo_pool.get_mut(new_slot) = BrickGeometry::fully_solid();
                                brick_map.set(bx, by, bz, new_slot);
                                new_bricks.push(UVec3::new(bx, by, bz));
                            }
                        }
                    }
                }
            }
        }
    }

    // Track which bricks were modified in any layer.
    let mut changed_bricks = std::collections::HashSet::new();

    // ── Iterative dilation / erosion ────────────────────────────────────
    for _layer in 0..num_layers {
        let mut layer_changes: Vec<(u32, u32, u32, u8, u8, u8, bool)> = Vec::new();

        for bz in bmin_z..bmax_z {
            for by in bmin_y..bmax_y {
                for bx in bmin_x..bmax_x {
                    let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                        continue;
                    }

                    let geo = geo_pool.get(slot);

                    for vz in 0..8u8 {
                        for vy in 0..8u8 {
                            for vx in 0..8u8 {
                                let solid = geo.is_solid(vx, vy, vz);

                                if is_add && solid {
                                    continue; // Already solid, nothing to dilate
                                }
                                if is_subtract && !solid {
                                    continue; // Already empty, nothing to erode
                                }

                                // Check brush footprint
                                let pos = voxel_world_pos(bx, by, bz, vx, vy, vz);
                                if !in_brush(pos) {
                                    continue;
                                }

                                // Check 6-connected neighbors for opposite occupancy.
                                // For add: need at least one solid neighbor (dilate from surface).
                                // For subtract: need at least one empty neighbor (erode from surface).
                                let has_opposite_neighbor = has_neighbor_with_occupancy(
                                    geo_pool, brick_map, bx, by, bz, vx, vy, vz,
                                    if is_add { true } else { false }, // looking for solid (add) or empty (subtract)
                                );

                                if has_opposite_neighbor {
                                    // Record change: set solid (add) or clear solid (subtract)
                                    layer_changes.push((bx, by, bz, vx, vy, vz, is_add));
                                }
                            }
                        }
                    }
                }
            }
        }

        if layer_changes.is_empty() {
            break; // No more changes possible
        }

        // Apply changes
        for &(bx, by, bz, vx, vy, vz, set_solid) in &layer_changes {
            let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
            if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                continue;
            }
            let geo = geo_pool.get_mut(slot);
            geo.set_solid(vx, vy, vz, set_solid);
            changed_bricks.insert(UVec3::new(bx, by, bz));
        }
    }

    // ── Post-process: rebuild surface lists, assign materials, compact ───

    let color = [
        ((op.color_packed >> 24) & 0xFF) as u8,
        ((op.color_packed >> 16) & 0xFF) as u8,
        ((op.color_packed >> 8) & 0xFF) as u8,
        (op.color_packed & 0xFF) as u8,
    ];

    for brick_coord in &changed_bricks {
        let (bx, by, bz) = (brick_coord.x, brick_coord.y, brick_coord.z);
        let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
        if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
            continue;
        }

        let geo = geo_pool.get_mut(slot);

        // Rebuild surface voxel list preserving existing colors
        geo.rebuild_surface_list_preserving();

        // Assign color/material to new surface voxels
        for sv in &mut geo.surface_voxels {
            if sv.color == [255, 255, 255, 255] && sv.material_id == 0 {
                sv.material_id = op.material_id as u8;
                if op.color_packed != 0 {
                    sv.color = color;
                }
            }
        }

        // Compact: deallocate fully empty or fully solid bricks
        if geo.is_fully_empty() {
            brick_map.set(bx, by, bz, EMPTY_SLOT);
            geo_pool.deallocate(slot);
            removed_bricks.push(*brick_coord);
        } else if geo.is_fully_solid() {
            brick_map.set(bx, by, bz, INTERIOR_SLOT);
            geo_pool.deallocate(slot);
            removed_bricks.push(*brick_coord);
        } else {
            modified_bricks.push(*brick_coord);
        }
    }

    // Also mark newly allocated but unchanged bricks as modified (they need SDF)
    for bc in &new_bricks {
        if !changed_bricks.contains(bc) {
            let slot = brick_map.get(bc.x, bc.y, bc.z).unwrap_or(EMPTY_SLOT);
            if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
                // Empty brick that was allocated but not touched — deallocate
                geo_pool.deallocate(slot);
                brick_map.set(bc.x, bc.y, bc.z, EMPTY_SLOT);
            }
        }
    }

    // Filter new_bricks to only include those that survived
    new_bricks.retain(|bc| {
        let slot = brick_map.get(bc.x, bc.y, bc.z).unwrap_or(EMPTY_SLOT);
        slot != EMPTY_SLOT && slot != INTERIOR_SLOT
    });

    build_result(modified_bricks, new_bricks, removed_bricks,
        bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z, brick_map)
}

/// Check if voxel (vx, vy, vz) in brick (bx, by, bz) has a 6-connected neighbor
/// with the specified occupancy state, considering cross-brick boundaries.
fn has_neighbor_with_occupancy(
    geo_pool: &Pool<BrickGeometry>,
    brick_map: &BrickMap,
    bx: u32, by: u32, bz: u32,
    vx: u8, vy: u8, vz: u8,
    looking_for_solid: bool,
) -> bool {
    const OFFSETS: [(i8, i8, i8); 6] = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ];

    for &(dx, dy, dz) in &OFFSETS {
        let nx = vx as i8 + dx;
        let ny = vy as i8 + dy;
        let nz = vz as i8 + dz;

        // Determine which brick the neighbor is in
        let (nbx, nvy) = if nx < 0 {
            if bx == 0 { continue; }
            (bx - 1, 7u8)
        } else if nx >= 8 {
            if bx + 1 >= brick_map.dims.x { continue; }
            (bx + 1, 0u8)
        } else {
            (bx, nx as u8)
        };

        let (nby, nvy_y) = if ny < 0 {
            if by == 0 { continue; }
            (by - 1, 7u8)
        } else if ny >= 8 {
            if by + 1 >= brick_map.dims.y { continue; }
            (by + 1, 0u8)
        } else {
            (by, ny as u8)
        };

        let (nbz, nvz) = if nz < 0 {
            if bz == 0 { continue; }
            (bz - 1, 7u8)
        } else if nz >= 8 {
            if bz + 1 >= brick_map.dims.z { continue; }
            (bz + 1, 0u8)
        } else {
            (bz, nz as u8)
        };

        // Resolve the actual voxel coordinates in the neighbor brick
        let final_vx = if dx != 0 { nvy } else { vx };
        let final_vy = if dy != 0 { nvy_y } else { vy };
        let final_vz = if dz != 0 { nvz } else { vz };
        let final_bx = if dx != 0 { nbx } else { bx };
        let final_by = if dy != 0 { nby } else { by };
        let final_bz = if dz != 0 { nbz } else { bz };

        let neighbor_solid = voxel_is_solid(geo_pool, brick_map, final_bx, final_by, final_bz, final_vx, final_vy, final_vz);

        if neighbor_solid == looking_for_solid {
            return true;
        }
    }

    false
}

/// Check if a specific voxel is solid, handling EMPTY_SLOT and INTERIOR_SLOT.
fn voxel_is_solid(
    geo_pool: &Pool<BrickGeometry>,
    brick_map: &BrickMap,
    bx: u32, by: u32, bz: u32,
    vx: u8, vy: u8, vz: u8,
) -> bool {
    let slot = match brick_map.get(bx, by, bz) {
        Some(s) => s,
        None => return false,
    };
    if slot == EMPTY_SLOT {
        return false;
    }
    if slot == INTERIOR_SLOT {
        return true;
    }
    geo_pool.get(slot).is_solid(vx, vy, vz)
}

/// Build the GeometryEditResult with SDF region bounds.
fn build_result(
    modified_bricks: Vec<UVec3>,
    new_bricks: Vec<UVec3>,
    removed_bricks: Vec<UVec3>,
    bmin_x: u32, bmin_y: u32, bmin_z: u32,
    bmax_x: u32, bmax_y: u32, bmax_z: u32,
    brick_map: &BrickMap,
) -> GeometryEditResult {
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

    fn make_add_op(pos: Vec3, radius: f32, strength: f32, mat: u16) -> EditOp {
        EditOp {
            object_id: 1,
            position: pos,
            rotation: Quat::IDENTITY,
            edit_type: EditType::CsgUnion,
            shape_type: crate::types::ShapeType::Sphere,
            dimensions: Vec3::new(radius, 0.0, 0.0),
            strength,
            blend_k: 0.0,
            falloff: crate::types::FalloffCurve::Smooth,
            material_id: mat,
            secondary_id: 0,
            color_packed: 0xFF804020,
        }
    }

    fn count_solid(map: &BrickMap, pool: &Pool<BrickGeometry>) -> u32 {
        let mut total = 0u32;
        for bz in 0..map.dims.z {
            for by in 0..map.dims.y {
                for bx in 0..map.dims.x {
                    let slot = map.get(bx, by, bz).unwrap();
                    if slot == INTERIOR_SLOT {
                        total += 512; // 8*8*8
                    } else if slot != EMPTY_SLOT {
                        total += pool.get(slot).solid_count();
                    }
                }
            }
        }
        total
    }

    /// Set up a scene with a solid sphere for testing dilation/erosion.
    fn setup_sphere_scene(voxel_size: f32, sphere_radius: f32) -> (Pool<BrickGeometry>, BrickMap, BrickMapAllocator, BrickMapHandle) {
        let brick_size = voxel_size * 8.0;
        let margin = voxel_size * 4.0;
        let half = sphere_radius + margin;
        let dims = UVec3::new(
            ((half * 2.0 / brick_size).ceil() as u32).max(1),
            ((half * 2.0 / brick_size).ceil() as u32).max(1),
            ((half * 2.0 / brick_size).ceil() as u32).max(1),
        );

        let mut geo_pool: Pool<BrickGeometry> = Pool::new(512);
        let mut alloc = BrickMapAllocator::new();
        let mut map = BrickMap::new(dims);

        let grid_origin = -Vec3::new(
            dims.x as f32 * brick_size * 0.5,
            dims.y as f32 * brick_size * 0.5,
            dims.z as f32 * brick_size * 0.5,
        );

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let brick_min = grid_origin + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );

                    let slot = match geo_pool.allocate() {
                        Some(s) => s,
                        None => continue,
                    };
                    let geo = geo_pool.get_mut(slot);
                    let mut any_solid = false;
                    let mut all_solid = true;

                    for vz in 0..8u8 {
                        for vy in 0..8u8 {
                            for vx in 0..8u8 {
                                let pos = brick_min + Vec3::new(
                                    vx as f32 * voxel_size + voxel_size * 0.5,
                                    vy as f32 * voxel_size + voxel_size * 0.5,
                                    vz as f32 * voxel_size + voxel_size * 0.5,
                                );
                                if pos.length() <= sphere_radius {
                                    geo.set_solid(vx, vy, vz, true);
                                    any_solid = true;
                                } else {
                                    all_solid = false;
                                }
                            }
                        }
                    }

                    if !any_solid {
                        geo_pool.deallocate(slot);
                        map.set(bx, by, bz, EMPTY_SLOT);
                    } else if all_solid {
                        geo_pool.deallocate(slot);
                        map.set(bx, by, bz, INTERIOR_SLOT);
                    } else {
                        map.set(bx, by, bz, slot);
                    }
                }
            }
        }

        let handle = alloc.allocate(&map);
        (geo_pool, map, alloc, handle)
    }

    #[test]
    fn dilate_adds_layers_to_surface() {
        let voxel_size = 0.02;
        let sphere_radius = 0.15;
        let (mut geo_pool, mut map, mut alloc, handle) = setup_sphere_scene(voxel_size, sphere_radius);

        let solid_before = count_solid(&map, &geo_pool);
        assert!(solid_before > 0, "sphere should have solid voxels");

        // Add 2 layers at the surface near origin
        let op = make_add_op(Vec3::new(0.0, 0.0, sphere_radius), 0.1, 0.25, 1); // 0.25 * 8 = 2 layers
        let result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &op, voxel_size,
        );

        let solid_after = count_solid(&map, &geo_pool);
        assert!(solid_after > solid_before,
            "dilation should add voxels: {solid_after} > {solid_before}");
        assert!(!result.modified_bricks.is_empty() || !result.new_bricks.is_empty(),
            "should report modified or new bricks");
    }

    #[test]
    fn erode_removes_layers_from_surface() {
        let voxel_size = 0.02;
        let sphere_radius = 0.15;
        let (mut geo_pool, mut map, mut alloc, handle) = setup_sphere_scene(voxel_size, sphere_radius);

        let solid_before = count_solid(&map, &geo_pool);

        // Subtract 2 layers from surface near origin
        let mut op = make_add_op(Vec3::new(0.0, 0.0, sphere_radius), 0.1, 0.25, 1);
        op.edit_type = EditType::CsgSubtract;
        let _result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &op, voxel_size,
        );

        let solid_after = count_solid(&map, &geo_pool);
        assert!(solid_after < solid_before,
            "erosion should remove voxels: {solid_after} < {solid_before}");
    }

    #[test]
    fn strength_controls_layer_count() {
        let voxel_size = 0.02;
        let sphere_radius = 0.15;

        // 1 layer (strength=0.125)
        let (mut pool1, mut map1, mut alloc1, handle1) = setup_sphere_scene(voxel_size, sphere_radius);
        let before1 = count_solid(&map1, &pool1);
        let op1 = make_add_op(Vec3::new(0.0, 0.0, sphere_radius), 0.1, 0.125, 1);
        apply_geometry_edit(&mut pool1, &mut map1, &mut alloc1, &handle1, &op1, voxel_size);
        let added1 = count_solid(&map1, &pool1) as i64 - before1 as i64;

        // 4 layers (strength=0.5)
        let (mut pool4, mut map4, mut alloc4, handle4) = setup_sphere_scene(voxel_size, sphere_radius);
        let before4 = count_solid(&map4, &pool4);
        let op4 = make_add_op(Vec3::new(0.0, 0.0, sphere_radius), 0.1, 0.5, 1);
        apply_geometry_edit(&mut pool4, &mut map4, &mut alloc4, &handle4, &op4, voxel_size);
        let added4 = count_solid(&map4, &pool4) as i64 - before4 as i64;

        assert!(added4 > added1,
            "more layers should add more voxels: {added4} > {added1}");
    }

    #[test]
    fn paint_modifies_surface_voxels() {
        let voxel_size = 0.02;
        let sphere_radius = 0.15;
        let (mut geo_pool, mut map, mut alloc, handle) = setup_sphere_scene(voxel_size, sphere_radius);

        // Rebuild surface lists
        for bz in 0..map.dims.z {
            for by in 0..map.dims.y {
                for bx in 0..map.dims.x {
                    let slot = map.get(bx, by, bz).unwrap();
                    if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
                        geo_pool.get_mut(slot).rebuild_surface_list();
                    }
                }
            }
        }

        let paint_op = EditOp {
            edit_type: EditType::Paint,
            material_id: 5,
            ..make_add_op(Vec3::ZERO, 0.3, 1.0, 5)
        };
        apply_geometry_edit(&mut geo_pool, &mut map, &mut alloc, &handle, &paint_op, voxel_size);

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
        let voxel_size = 0.02;
        let sphere_radius = 0.15;
        let (mut geo_pool, mut map, mut alloc, handle) = setup_sphere_scene(voxel_size, sphere_radius);

        let op = make_add_op(Vec3::new(0.0, 0.0, sphere_radius), 0.1, 0.25, 1);
        let result = apply_geometry_edit(
            &mut geo_pool, &mut map, &mut alloc, &handle, &op, voxel_size,
        );

        assert!(result.sdf_region_max.x > result.sdf_region_min.x);
        assert!(result.sdf_region_max.y > result.sdf_region_min.y);
        assert!(result.sdf_region_max.z > result.sdf_region_min.z);
    }
}
