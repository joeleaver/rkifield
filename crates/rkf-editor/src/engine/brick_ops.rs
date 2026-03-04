//! Brick map and pool operations for the editor engine.

use glam::Vec3;
use rkf_core::{Aabb, Scene, SdfSource};
use super::EditorEngine;

impl EditorEngine {
    /// Grow the brick map if the edit extends beyond the current grid.
    ///
    /// The grid is always centered at the object's local origin. When a sculpt
    /// edit extends past the current grid boundary, this method:
    /// 1. Computes new (larger) grid dimensions that contain the edit
    /// 2. Creates a new BrickMap with expanded dims
    /// 3. Copies existing slot entries (with coordinate offset for centering)
    /// 4. Deallocates old map, allocates new map
    /// 5. Updates SdfSource with new handle/AABB
    /// 6. Re-uploads brick maps to GPU
    ///
    /// Returns true if the map was grown.
    pub(super) fn grow_brick_map_if_needed(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
    ) -> bool {
        use rkf_core::brick_map::BrickMap;

        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return false,
        };

        let (handle, voxel_size) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return false,
        };

        let brick_size = voxel_size * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let grid_origin = -Vec3::new(
            handle.dims.x as f32 * brick_size * 0.5,
            handle.dims.y as f32 * brick_size * 0.5,
            handle.dims.z as f32 * brick_size * 0.5,
        );
        let grid_end = -grid_origin;

        // Check if edit fits within current grid (with padding margin).
        // ensure_sdf_consistency BFS needs ~2 bricks of padding plus 1 for
        // gradient sampling at the boundary = 3 bricks total.
        let margin = brick_size * 3.0;
        if edit_min.x >= grid_origin.x + margin
            && edit_min.y >= grid_origin.y + margin
            && edit_min.z >= grid_origin.z + margin
            && edit_max.x <= grid_end.x - margin
            && edit_max.y <= grid_end.y - margin
            && edit_max.z <= grid_end.z - margin
        {
            return false; // Fits fine, no growth needed.
        }

        // Compute required extent: union of current grid and edit AABB, plus margin.
        // 4 bricks: 2 for consistency BFS padding + 1 for Catmull-Rom gradient + 1 safety.
        // The brick MAP (flat u32 index array) is cheap; actual GPU cost is only allocated
        // brick pool slots (4KB each), which are proportional to surface area, not grid dims.
        let growth_margin = brick_size * 4.0;
        let required_min = edit_min.min(grid_origin) - Vec3::splat(growth_margin);
        let required_max = edit_max.max(grid_end) + Vec3::splat(growth_margin);

        // New grid must be symmetric about origin (shader assumes centered).
        // The dimension change must be even so that integer pad = (new - old) / 2
        // exactly matches the floating-point centering offset. An odd delta
        // would shift old data by half a brick, scrambling the SDF.
        let half_extent = required_min.abs().max(required_max.abs());
        let raw_dims = glam::UVec3::new(
            ((half_extent.x * 2.0 / brick_size).ceil() as u32).max(handle.dims.x),
            ((half_extent.y * 2.0 / brick_size).ceil() as u32).max(handle.dims.y),
            ((half_extent.z * 2.0 / brick_size).ceil() as u32).max(handle.dims.z),
        );
        // Ensure (new_dims - old_dims) is even on each axis.
        let fix_parity = |new: u32, old: u32| -> u32 {
            if (new.wrapping_sub(old)) % 2 != 0 { new + 1 } else { new }
        };
        let new_dims = glam::UVec3::new(
            fix_parity(raw_dims.x, handle.dims.x),
            fix_parity(raw_dims.y, handle.dims.y),
            fix_parity(raw_dims.z, handle.dims.z),
        );

        if new_dims == handle.dims {
            return false; // Already big enough.
        }

        // Compute offset: old brick (0,0,0) maps to new brick at this offset.
        let pad_x = (new_dims.x - handle.dims.x) / 2;
        let pad_y = (new_dims.y - handle.dims.y) / 2;
        let pad_z = (new_dims.z - handle.dims.z) / 2;

        // Build new BrickMap and copy existing entries.
        // IMPORTANT: copy both allocated bricks AND INTERIOR_SLOT markers.
        // is_unallocated() returns true for INTERIOR_SLOT, but interior markers
        // must be preserved — they tell the GPU that these bricks are deep inside
        // the object (returns -vs*2.0). If lost during growth, they'd become
        // EMPTY_SLOT (returns +vs*2.0), flipping the SDF sign and creating
        // concentric ring artifacts.
        let mut new_map = BrickMap::new(new_dims);
        for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if slot != rkf_core::brick_map::EMPTY_SLOT {
                            new_map.set(bx + pad_x, by + pad_y, bz + pad_z, slot);
                        }
                    }
                }
            }
        }

        // Deallocate old, allocate new.
        self.cpu_brick_map_alloc.deallocate(handle);
        let new_handle = self.cpu_brick_map_alloc.allocate(&new_map);

        // Update AABB to match new grid extent (centered at origin).
        let new_half = Vec3::new(
            new_dims.x as f32 * brick_size * 0.5,
            new_dims.y as f32 * brick_size * 0.5,
            new_dims.z as f32 * brick_size * 0.5,
        );
        let new_aabb = Aabb::new(-new_half, new_half);

        // Update the scene object.
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id).unwrap();
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: new_aabb,
        };
        obj.aabb = new_aabb;

        // Re-upload brick maps to GPU.
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }

        true
    }

    /// Sample the dominant material_id from the CPU brick pool near `local_pos`.
    ///
    /// Checks a small neighborhood of voxels (3x3x3 around the nearest voxel)
    /// and returns the most common non-zero material_id. Falls back to 1 if
    /// no material is found (all voxels are empty or material 0).
    pub(super) fn sample_dominant_material(
        &self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        _aabb_min: Vec3,
        local_pos: Vec3,
    ) -> u16 {
        let vs = voxel_size;
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(
            dims.x as f32 * brick_extent,
            dims.y as f32 * brick_extent,
            dims.z as f32 * brick_extent,
        );

        // Convert local_pos to grid-space (grid is centered at origin).
        let grid_pos = local_pos + grid_size * 0.5;
        let center_voxel = (grid_pos / vs).floor();
        let cx = center_voxel.x as i32;
        let cy = center_voxel.y as i32;
        let cz = center_voxel.z as i32;

        let total_x = (dims.x * 8) as i32;
        let total_y = (dims.y * 8) as i32;
        let total_z = (dims.z * 8) as i32;

        // Sample a 3x3x3 neighborhood and count material occurrences.
        // Only count voxels near the surface (negative SDF, i.e. inside).
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let vx = (cx + dx).clamp(0, total_x - 1);
                    let vy = (cy + dy).clamp(0, total_y - 1);
                    let vz = (cz + dz).clamp(0, total_z - 1);

                    let bx = (vx / 8) as u32;
                    let by = (vy / 8) as u32;
                    let bz = (vz / 8) as u32;
                    let lx = (vx % 8) as u32;
                    let ly = (vy % 8) as u32;
                    let lz = (vz % 8) as u32;

                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let sample = self.cpu_brick_pool.get(slot).sample(lx, ly, lz);
                            let mid = sample.material_id();
                            // Only count non-zero materials from voxels near/on surface.
                            if mid != 0 && sample.distance_f32() < voxel_size {
                                *counts.entry(mid).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Return the most common material, or 1 as fallback.
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(mid, _)| mid)
            .unwrap_or(1)
    }

    /// For positions in EMPTY_SLOT bricks, determines the sign (inside/outside)
    /// by using a precomputed interior/exterior classification for EMPTY_SLOT
    /// bricks. Exterior empty bricks return `+vs*4.0`, interior ones return
    /// `-vs*4.0`. This prevents false surfaces in object interiors and
    /// ensures correct normals at brick boundaries.
    pub(super) fn sample_sdf_cpu(
        pool: &rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        local_pos: Vec3,
        _exterior_bricks: &std::collections::HashSet<(u32, u32, u32)>,
    ) -> f32 {
        let vs = voxel_size;
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(
            dims.x as f32 * brick_extent,
            dims.y as f32 * brick_extent,
            dims.z as f32 * brick_extent,
        );
        let grid_pos = local_pos + grid_size * 0.5;

        // Clamp to valid range.
        let eps = vs * 0.01;
        let clamped = grid_pos.clamp(Vec3::splat(eps), grid_size - Vec3::splat(eps));

        // Convert to continuous voxel coordinates (voxel centers at integers, -0.5 shift).
        let voxel_coord = clamped / vs - Vec3::splat(0.5);
        let v0x = voxel_coord.x.floor() as i32;
        let v0y = voxel_coord.y.floor() as i32;
        let v0z = voxel_coord.z.floor() as i32;
        let tx = voxel_coord.x - v0x as f32;
        let ty = voxel_coord.y - v0y as f32;
        let tz = voxel_coord.z - v0z as f32;

        let total_x = (dims.x * 8) as i32;
        let total_y = (dims.y * 8) as i32;
        let total_z = (dims.z * 8) as i32;

        let read_voxel = |vx: i32, vy: i32, vz: i32| -> f32 {
            let cx = vx.clamp(0, total_x - 1);
            let cy = vy.clamp(0, total_y - 1);
            let cz = vz.clamp(0, total_z - 1);
            let bx = (cx / 8) as u32;
            let by = (cy / 8) as u32;
            let bz = (cz / 8) as u32;
            let lx = (cx % 8) as u32;
            let ly = (cy % 8) as u32;
            let lz = (cz % 8) as u32;

            match alloc.get_entry(handle, bx, by, bz) {
                Some(slot) if !Self::is_unallocated(slot) => {
                    pool.get(slot).sample(lx, ly, lz).distance_f32()
                }
                _ => {
                    // Unallocated bricks always return positive distance,
                    // matching GPU EMPTY_SLOT behavior (vs * 2.0).
                    // Using the flood-fill exterior/interior distinction here
                    // caused false surfaces between disconnected sculpt bodies
                    // (the flood fill can't reach enclosed empty bricks from
                    // the boundary, so they'd get classified as "interior" and
                    // return negative distance, creating concentric ring artifacts).
                    // The BFS in ensure_sdf_consistency naturally propagates
                    // correct negative values from allocated neighbor bricks.
                    vs * 2.0
                }
            }
        };

        // 8-corner trilinear interpolation.
        let c000 = read_voxel(v0x, v0y, v0z);
        let c100 = read_voxel(v0x + 1, v0y, v0z);
        let c010 = read_voxel(v0x, v0y + 1, v0z);
        let c110 = read_voxel(v0x + 1, v0y + 1, v0z);
        let c001 = read_voxel(v0x, v0y, v0z + 1);
        let c101 = read_voxel(v0x + 1, v0y, v0z + 1);
        let c011 = read_voxel(v0x, v0y + 1, v0z + 1);
        let c111 = read_voxel(v0x + 1, v0y + 1, v0z + 1);

        let c00 = c000 + (c100 - c000) * tx;
        let c10 = c010 + (c110 - c010) * tx;
        let c01 = c001 + (c101 - c001) * tx;
        let c11 = c011 + (c111 - c011) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        c0 + (c1 - c0) * tz
    }

    /// Classify all EMPTY_SLOT bricks as exterior or interior via flood-fill.
    ///
    /// BFS from all EMPTY_SLOT bricks on the grid boundary (faces of the 3D
    /// grid), expanding through EMPTY_SLOT only — allocated bricks act as
    /// walls. Any EMPTY_SLOT reachable from the boundary is exterior (should
    /// return `+vs*4.0`). Any EMPTY_SLOT NOT reachable is interior (should
    /// return `-vs*4.0`).
    ///
    /// Returns true if a brick map slot is unallocated (no pool data).
    pub(super) fn classify_exterior_bricks(
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
    ) -> std::collections::HashSet<(u32, u32, u32)> {
        use std::collections::{HashSet, VecDeque};

        let dims = handle.dims;
        let mut exterior: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut queue: VecDeque<(u32, u32, u32)> = VecDeque::new();

        // Seed: all unallocated bricks on any face of the brick map grid.
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let is_boundary = bx == 0 || bx == dims.x - 1
                        || by == 0 || by == dims.y - 1
                        || bz == 0 || bz == dims.z - 1;
                    if !is_boundary {
                        continue;
                    }
                    if let Some(slot) = alloc.get_entry(handle, bx, by, bz) {
                        if Self::is_unallocated(slot) {
                            if exterior.insert((bx, by, bz)) {
                                queue.push_back((bx, by, bz));
                            }
                        }
                    }
                }
            }
        }

        // BFS through unallocated neighbors. Allocated bricks block traversal.
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        while let Some((bx, by, bz)) = queue.pop_front() {
            for &(dx, dy, dz) in &deltas {
                let nx = bx as i32 + dx;
                let ny = by as i32 + dy;
                let nz = bz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0
                    || nx >= dims.x as i32
                    || ny >= dims.y as i32
                    || nz >= dims.z as i32
                {
                    continue;
                }
                let coord = (nx as u32, ny as u32, nz as u32);
                if let Some(slot) = alloc.get_entry(handle, coord.0, coord.1, coord.2) {
                    if Self::is_unallocated(slot) && exterior.insert(coord) {
                        queue.push_back(coord);
                    }
                }
            }
        }

        exterior
    }

    /// Allocate new bricks in the edit region for additive operations.
    ///
    /// When sculpting "add" material, the brush may extend into brick map cells
    /// that are currently unallocated. This method allocates new bricks and
    /// fills them with a constant SDF based on interior/exterior classification.
    ///
    /// The CSG edit (`apply_edit_cpu`) then creates the correct zero crossings
    /// from the brush shape SDF blended with these constants. This is simpler
    /// and more correct than trilinear interpolation from fallback values, which
    /// produces wrong SDF magnitudes that cause gradient artifacts.
    ///
    /// Also grows the brick map if the edit extends beyond the current grid.
    ///
    /// Returns the list of newly allocated pool slot indices.
    ///
    /// **Important:** After allocation, `prefill_new_brick_faces()` should be
    /// called to replace constant fill with SDF extrapolated from existing
    /// neighbors before CSG is applied. Without this, constant fill creates
    /// seam artifacts at the transition between old and new bricks.
    pub(super) fn allocate_bricks_in_region(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
        voxel_size: f32,
    ) -> Vec<u32> {
        self.grow_brick_map_if_needed(scene, object_id, op);

        let obj = match scene.objects.iter_mut().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return Vec::new(),
        };

        let (handle, _aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, aabb, .. } => (*brick_map_handle, aabb.min),
            _ => return Vec::new(),
        };

        let brick_size = voxel_size * 8.0;
        let (edit_min, edit_max) = op.local_aabb();

        let bmin = ((edit_min - _aabb_min) / brick_size).floor();
        let bmax = ((edit_max - _aabb_min) / brick_size - Vec3::splat(0.001)).ceil();

        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

        // The CSG falloff is spherical (Euclidean distance from brush center),
        // but the AABB is a cube. Bricks at the AABB corners are outside the
        // falloff sphere and won't be modified by CSG. If we allocate them with
        // constant-fill values (±vs*2.0 based on interior/exterior), the sign
        // boundary between interior and exterior creates a false zero-crossing —
        // a phantom surface that renders as "stalactite" artifacts.
        //
        // Fix: only allocate bricks whose centers are within the falloff sphere
        // (plus half a brick diagonal for overlap margin). Bricks outside stay
        // as EMPTY_SLOT and get properly filled by ensure_sdf_consistency's BFS.
        let max_dim = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let falloff_radius = max_dim + op.blend_k;
        let brick_half_diag = brick_size * 0.5 * (3.0f32).sqrt();
        let alloc_radius = falloff_radius + brick_half_diag;
        let alloc_radius_sq = alloc_radius * alloc_radius;

        let mut new_slots = Vec::new();

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if !Self::is_unallocated(s) {
                            continue; // Already allocated — skip.
                        }
                    }

                    // Check if this brick's center is within the falloff sphere.
                    // Bricks outside the sphere won't be modified by CSG and
                    // should stay EMPTY_SLOT for the consistency BFS to handle.
                    let brick_center = _aabb_min + Vec3::new(
                        (bx as f32 + 0.5) * brick_size,
                        (by as f32 + 0.5) * brick_size,
                        (bz as f32 + 0.5) * brick_size,
                    );
                    let dist_sq = (brick_center - op.position).length_squared();
                    if dist_sq > alloc_radius_sq {
                        continue; // Outside falloff sphere — skip.
                    }

                    if let Some(new_slot) = self.cpu_brick_pool.allocate() {
                        // Initialize as exterior. Newly-allocated bricks represent space
                        // the brush may reach into, but they have no geometry yet.
                        // vs*2.0 = "exterior, close to the surface" — small enough that
                        // trilinear interpolation stays conservative near newly-sculpted
                        // surfaces (no overshoot), large enough that the zero-crossing
                        // only appears when CSG actually writes interior voxels.
                        //
                        // Bricks that CSG leaves with no zero-crossing are reverted to
                        // EMPTY_SLOT by Step 6c, so no false-shadow contribution.
                        let fill = voxel_size * 2.0;
                        {
                            use rkf_core::voxel::VoxelSample;
                            let brick = self.cpu_brick_pool.get_mut(new_slot);
                            for z in 0u32..8 {
                                for y in 0u32..8 {
                                    for x in 0u32..8 {
                                        brick.set(x, y, z, VoxelSample::new(fill, 0, 0, 0, 0));
                                    }
                                }
                            }
                        }

                        self.cpu_brick_map_alloc.set_entry(&handle, bx, by, bz, new_slot);
                        new_slots.push(new_slot);
                    }
                }
            }
        }

        if !new_slots.is_empty() {
            log::info!(
                "  allocate_bricks_in_region: allocated {} new bricks (brush SDF fill)",
                new_slots.len(),
            );
            let map_data = self.cpu_brick_map_alloc.as_slice();
            if !map_data.is_empty() {
                self.gpu_scene.upload_brick_maps(
                    &self.ctx.device, &self.ctx.queue, map_data,
                );
            }
        }

        new_slots
    }
}

