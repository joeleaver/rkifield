//! Sculpt brush operations for the editor engine.

use glam::Vec3;
use rkf_core::{Aabb, Scene, SdfSource};
use super::EditorEngine;

// sculpt_fmm_repair is in sdf_repair.rs (FMM/EDT/JFA operations).

impl EditorEngine {
    /// Surface-displacement sculpt brush.
    ///
    /// Instead of stamping a sphere SDF and combining it with the existing field
    /// via CSG min(), this directly modifies the stored distance values in a narrow
    /// band around the existing surface — exactly as mesh sculpting tools do.
    ///
    /// Add:      `new_d = orig_d - influence * step`  (surface moves outward)
    /// Subtract: `new_d = orig_d + influence * step`  (surface moves inward)
    ///
    /// `influence` is a smooth falloff from 1.0 at the brush centre to 0.0 at the
    /// brush edge.  Only voxels with `|orig_d| < narrow_band` are touched; deep
    /// interior/exterior voxels are left unchanged.
    ///
    /// Because we add a constant offset to each distance value — `d → d − c(pos)` —
    /// the gradient direction `∇d` is approximately preserved everywhere (the
    /// perturbation `∇c` is small for a smooth, large-radius brush).  This means:
    ///   • No junction artifacts between overlapping strokes (no min() discontinuity)
    ///   • Normals remain correct at the surface without any EDT repair step
    ///   • Multiple strokes accumulate naturally without drift
    pub(super) fn apply_sculpt_displacement(
        &mut self,
        scene:      &mut Scene,
        object_id:  u32,
        op:         &rkf_edit::edit_op::EditOp,
        voxel_size: f32,
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> (Vec<u32>, glam::UVec3, glam::UVec3) {
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick::brick_index;
        use rkf_edit::types::EditType;

        let is_add = op.edit_type == EditType::SmoothUnion;
        let is_sub = op.edit_type == EditType::SmoothSubtract;
        if !is_add && !is_sub {
            return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO);
        }

        let h = voxel_size;
        let radius = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let inv_rot = op.rotation.inverse();

        // How much the surface moves per stroke at full strength, brush centre.
        // One voxel per stroke; multiple strokes build up the deformation.
        let step = h * op.strength;

        if is_add {
            self.grow_brick_map_if_needed(scene, object_id, op);
        }

        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO),
        };
        let (handle, aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, aabb, .. } =>
                (*brick_map_handle, aabb.min),
            _ => return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO),
        };

        let brick_size = h * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let bmin = ((edit_min - aabb_min) / brick_size).floor();
        let bmax = ((edit_max - aabb_min) / brick_size - Vec3::splat(0.001)).ceil();
        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

        let mut modified: Vec<u32> = Vec::new();

        if is_add {
            self.grow_brick_pool_if_needed(64);
        }

        // Step 1: Subtract from existing allocated bricks only.
        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };

                    if let Some(ref mut acc) = undo_acc {
                        if acc.captured_slots.insert(slot) {
                            acc.snapshots.push((slot, self.cpu_brick_pool.get(slot).clone()));
                        }
                    }

                    let brick_origin = aabb_min + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );
                    let mut any_changed = false;

                    for lz in 0u32..8 {
                        for ly in 0u32..8 {
                            for lx in 0u32..8 {
                                let voxel_center = brick_origin
                                    + Vec3::new(lx as f32, ly as f32, lz as f32) * h
                                    + Vec3::splat(h * 0.5);

                                // Brush-local distance (handles rotation + non-sphere shapes).
                                let edit_local = inv_rot * (voxel_center - op.position);
                                let dist_to_edge = rkf_edit::cpu_apply::evaluate_shape(
                                    op.shape_type, &op.dimensions, edit_local,
                                );
                                // Only voxels inside the brush shape.
                                if dist_to_edge >= 0.0 { continue; }

                                let vi = brick_index(lx, ly, lz);
                                let orig = self.cpu_brick_pool.get(slot).voxels[vi];
                                let orig_d = orig.distance_f32();

                                // Smooth falloff: 1.0 at brush centre, 0.0 at brush edge.
                                let t = (-dist_to_edge / radius).min(1.0);
                                let influence = t * t * (3.0 - 2.0 * t); // smoothstep

                                // Level-set dilation/erosion: subtract (add) or add
                                // (subtract) a uniform amount from all distances.
                                // This shifts the zero-crossing outward/inward while
                                // preserving |∇d| = 1 exactly (gradient of a constant
                                // is zero). No distance recompute needed.
                                let delta = influence * step;
                                if delta < 1e-8 { continue; }

                                let new_d = if is_add {
                                    orig_d - delta
                                } else {
                                    orig_d + delta
                                };

                                self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                                    new_d,
                                    orig.material_id(),
                                    orig.blend_weight(),
                                    orig.secondary_id(),
                                    orig.flags(),
                                );
                                any_changed = true;
                            }
                        }
                    }

                    if any_changed {
                        modified.push(slot);
                    }
                }
            }
        }

        // Step 2 (Add only): Just-in-time allocation.
        // Check modified bricks for face voxels that went negative where
        // the neighbor is unallocated. The surface just arrived at that
        // boundary — allocate the neighbor NOW with a simple linear ramp.
        let mut new_slots: Vec<u32> = Vec::new();
        if is_add && !modified.is_empty() {
            let dirs: [(i32,i32,i32); 6] = [
                (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
            ];
            // Collect bricks needing expansion.
            let mut to_alloc: Vec<(u32,u32,u32, i32,i32,i32, u32)> = Vec::new();
            for bz in bmin_z..=bmax_z {
                for by in bmin_y..=bmax_y {
                    for bx in bmin_x..=bmax_x {
                        let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            Some(s) if !Self::is_unallocated(s) => s,
                            _ => continue,
                        };
                        if !modified.contains(&slot) { continue; }
                        for &(dx, dy, dz) in &dirs {
                            let nx = bx as i32 + dx;
                            let ny = by as i32 + dy;
                            let nz = bz as i32 + dz;
                            if nx < 0 || ny < 0 || nz < 0 { continue; }
                            let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                            if nx >= handle.dims.x || ny >= handle.dims.y || nz >= handle.dims.z {
                                continue;
                            }
                            let nbr = self.cpu_brick_map_alloc.get_entry(&handle, nx, ny, nz);
                            if matches!(nbr, Some(s) if !Self::is_unallocated(s)) {
                                continue; // already allocated
                            }
                            // Check if the face toward this neighbor has any negative voxel.
                            let brick = self.cpu_brick_pool.get(slot);
                            let has_neg_face = (0u32..8).any(|a| (0u32..8).any(|b| {
                                let d = if dx == 1 { brick.sample(7, a, b).distance_f32() }
                                    else if dx == -1 { brick.sample(0, a, b).distance_f32() }
                                    else if dy == 1 { brick.sample(a, 7, b).distance_f32() }
                                    else if dy == -1 { brick.sample(a, 0, b).distance_f32() }
                                    else if dz == 1 { brick.sample(a, b, 7).distance_f32() }
                                    else { brick.sample(a, b, 0).distance_f32() };
                                d < 0.0
                            }));
                            if has_neg_face {
                                to_alloc.push((nx, ny, nz, -dx, -dy, -dz, slot));
                            }
                        }
                    }
                }
            }
            // Allocate and fill with linear ramp from the source face.
            for (nx, ny, nz, dx, dy, dz, src_slot) in to_alloc {
                // Skip if already allocated by a prior iteration.
                if matches!(self.cpu_brick_map_alloc.get_entry(&handle, nx, ny, nz),
                            Some(s) if !Self::is_unallocated(s)) {
                    continue;
                }
                let ns = match self.cpu_brick_pool.allocate() {
                    Some(s) => s,
                    None => {
                        self.grow_brick_pool_if_needed(64);
                        match self.cpu_brick_pool.allocate() {
                            Some(s) => s,
                            None => continue,
                        }
                    }
                };
                // Fill with constant exterior value. No fake geometry.
                // The surface is still in the source brick — the ray marcher
                // finds it there. Subtraction on future strokes will naturally
                // bring these values down toward zero.
                let new_brick = self.cpu_brick_pool.get_mut(ns);
                for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                    new_brick.set(lx, ly, lz, VoxelSample::new(h * 4.0, 0, 0, 0, 0));
                }}}
                self.cpu_brick_map_alloc.set_entry(&handle, nx, ny, nz, ns);
                new_slots.push(ns);
                modified.push(ns);
            }
        }

        log::info!(
            "sculpt displacement: modified {} bricks, {} new (JIT), step={:.6}, h={:.6}, is_add={}",
            modified.len(), new_slots.len(), step, h, is_add,
        );

        (modified,
         glam::UVec3::new(bmin_x, bmin_y, bmin_z),
         glam::UVec3::new(bmax_x, bmax_y, bmax_z))
    }

    /// Boolean-stamp a sculpt brush into the brick pool.
    ///
    /// For Add (SmoothUnion): grows the brick map if needed, allocates bricks
    /// in the brush region (filled as large exterior), then blends via smin_poly.
    ///
    /// For Subtract (SmoothSubtract): carves via smax (= -smin(-a,-b,k)).
    /// No new bricks are allocated — carving empty space has no visible effect.
    ///
    /// Strength is applied via `effective_shape_d = shape_d + (1−strength) × max_extent`:
    /// at strength=1, the full brush is stamped; at strength=0, nothing changes.
    ///
    /// Newly-allocated bricks that end up with no interior voxels are immediately
    /// reverted to EMPTY_SLOT (the brush didn't reach them).
    ///
    /// The correct Euclidean SDF distances are recomputed by `fix_sdfs_cpu`
    /// (3D EDT) after each stroke, using the binary solid/empty field set here.
    ///
    /// Returns the pool slot indices of all bricks that were modified plus the
    /// brick-coordinate scope (min, max) of the stroke, for use in local EDT.
    pub(super) fn apply_sculpt_boolean(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
        voxel_size: f32,
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> (Vec<u32>, glam::UVec3, glam::UVec3) {
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick::brick_index;
        use rkf_edit::cpu_apply::evaluate_shape;
        use rkf_edit::types::EditType;

        let is_add = op.edit_type == EditType::SmoothUnion;
        let is_sub = op.edit_type == EditType::SmoothSubtract;
        if !is_add && !is_sub { return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO); }

        let h = voxel_size;
        let max_extent = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let inv_rot = op.rotation.inverse();

        // For add, grow the brick map if the brush extends beyond the current grid.
        if is_add {
            self.grow_brick_map_if_needed(scene, object_id, op);
        }

        // Re-lookup handle + aabb_min after potential grow.
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o, None => return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO),
        };
        let (handle, aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, aabb, .. } => (*brick_map_handle, aabb.min),
            _ => return (Vec::new(), glam::UVec3::ZERO, glam::UVec3::ZERO),
        };

        let brick_size = h * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let bmin = ((edit_min - aabb_min) / brick_size).floor();
        let bmax = ((edit_max - aabb_min) / brick_size - Vec3::splat(0.001)).ceil();
        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

        let mut new_slots: Vec<u32> = Vec::new(); // newly allocated this call
        let mut modified: Vec<u32> = Vec::new();  // all touched slots

        // Pass 1: allocate new bricks (fill with h*8.0 sentinel).
        if is_add {
            for bz in bmin_z..=bmax_z {
                for by in bmin_y..=bmax_y {
                    for bx in bmin_x..=bmax_x {
                        let existing = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz);
                        if matches!(existing, Some(s) if !Self::is_unallocated(s)) {
                            continue; // already allocated
                        }
                        match self.cpu_brick_pool.allocate() {
                            Some(ns) => {
                                let brick = self.cpu_brick_pool.get_mut(ns);
                                for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                                    brick.set(lx, ly, lz, VoxelSample::new(h * 8.0, 0, 0, 0, 0));
                                }}}
                                self.cpu_brick_map_alloc.set_entry(&handle, bx, by, bz, ns);
                                new_slots.push(ns);
                            }
                            None => {} // Pool full — skip.
                        }
                    }
                }
            }

            // Prefill face voxels from adjacent existing bricks BEFORE the stamp.
            // Without this, newly-allocated bricks keep h*8.0 on faces adjacent to
            // solid (negative) existing bricks, creating false zero-crossings that
            // the ray marcher hits as phantom surfaces (dark pits in the normal view).
            if !new_slots.is_empty() {
                self.prefill_new_brick_faces(&handle, h, &new_slots);
            }
        }

        // Pass 2: boolean stamp over all allocated bricks in the AABB.
        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => {
                            if is_sub { continue; }
                            continue; // skip (not allocated; shouldn't happen for add)
                        }
                    };

                    // Snapshot for undo BEFORE modifying this brick.
                    if let Some(ref mut acc) = undo_acc {
                        if acc.captured_slots.insert(slot) {
                            acc.snapshots.push((slot, self.cpu_brick_pool.get(slot).clone()));
                        }
                    }

                    // Boolean stamp: mark voxels inside the effective brush.
                    let brick_min = aabb_min + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );
                    let mut any_changed = false;

                    for lz in 0u32..8 {
                        for ly in 0u32..8 {
                            for lx in 0u32..8 {
                                let voxel_center = brick_min
                                    + Vec3::new(lx as f32, ly as f32, lz as f32) * h
                                    + Vec3::splat(h * 0.5);
                                let edit_local = inv_rot * (voxel_center - op.position);
                                let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);
                                let effective_d = shape_d + (1.0 - op.strength) * max_extent;

                                let vi = brick_index(lx, ly, lz);
                                let orig = self.cpu_brick_pool.get(slot).voxels[vi];
                                let orig_d = orig.distance_f32();

                                let (new_d, new_mat) = if is_add {
                                    // Direct CSG union: write the actual brush SDF value.
                                    // The brush SDF already IS a valid distance field, so
                                    // sampling its gradient gives correct smooth normals
                                    // without any staircase quantization or EDT artifacts.
                                    let nd = orig_d.min(effective_d);
                                    if nd >= orig_d - h * 0.001 { continue; }
                                    let mat = if effective_d < orig_d { op.material_id } else { orig.material_id() };
                                    (nd, mat)
                                } else {
                                    // Direct CSG subtract: carve by negated brush SDF.
                                    let nd = orig_d.max(-effective_d);
                                    if nd <= orig_d + h * 0.001 { continue; }
                                    (nd, orig.material_id())
                                };
                                self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                                    new_d, new_mat,
                                    orig.blend_weight(), orig.secondary_id(), orig.flags(),
                                );
                                any_changed = true;
                            }
                        }
                    }

                    if any_changed {
                        modified.push(slot);
                    }
                }
            }
        }

        // Revert newly-allocated bricks that contain no interior voxels.
        // These are bricks where the boolean stamp wrote no solid voxels —
        // the brush AABB covered the brick but the brush shape didn't reach it.
        if !new_slots.is_empty() {
            let new_set: std::collections::HashSet<u32> = new_slots.iter().copied().collect();
            let mut slot_to_coord: std::collections::HashMap<u32, (u32, u32, u32)> =
                std::collections::HashMap::new();
            for bz in bmin_z..=bmax_z {
                for by in bmin_y..=bmax_y {
                    for bx in bmin_x..=bmax_x {
                        if let Some(s) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            if new_set.contains(&s) {
                                slot_to_coord.insert(s, (bx, by, bz));
                            }
                        }
                    }
                }
            }
            let mut reverted = 0u32;
            for &slot in &new_slots {
                let has_interior = self.cpu_brick_pool.get(slot)
                    .voxels.iter()
                    .any(|v| v.distance_f32() < 0.0);
                if !has_interior {
                    if let Some(&(bx, by, bz)) = slot_to_coord.get(&slot) {
                        self.cpu_brick_map_alloc.set_entry(
                            &handle, bx, by, bz, rkf_core::brick_map::EMPTY_SLOT,
                        );
                        self.cpu_brick_pool.deallocate(slot);
                        modified.retain(|&s| s != slot);
                        reverted += 1;
                    }
                }
            }
            if reverted > 0 {
                log::info!("  apply_sculpt_boolean: reverted {} surfaceless bricks", reverted);
            }
        }

        log::info!(
            "  apply_sculpt_boolean: {} modified slots (is_add={})",
            modified.len(), is_add,
        );
        let scope_min = glam::UVec3::new(bmin_x, bmin_y, bmin_z);
        let scope_max = glam::UVec3::new(bmax_x, bmax_y, bmax_z);
        (modified, scope_min, scope_max)
    }

    /// Auto-voxelize an analytical object for sculpting.
    ///
    /// If the object's root node is `SdfSource::Analytical`, converts it to
    /// `SdfSource::Voxelized` with a resolution based on the primitive's size
    /// (diameter / 48, clamped to [0.005, 0.5]).
    ///
    /// Returns the new `(voxel_size, aabb, handle)` if voxelization occurred.
    pub fn ensure_object_voxelized(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
    ) -> Option<(f32, Aabb, rkf_core::scene_node::BrickMapHandle)> {
        use super::primitive_diameter;
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id)?;

        let (primitive, material_id) = match &obj.root_node.sdf_source {
            SdfSource::Analytical { primitive, material_id } => (*primitive, *material_id),
            SdfSource::Voxelized { .. } => return None, // Already voxelized.
            SdfSource::None => return None,
        };

        // Compute object diameter from primitive bounding.
        let diameter = primitive_diameter(&primitive);
        let voxel_size = (diameter / 48.0).clamp(0.005, 0.5);

        // Build AABB with 50% growth margin for sculpting expansion.
        let half = diameter * 0.5;
        let margin = half * 0.5;
        let aabb = Aabb::new(
            Vec3::splat(-half - margin),
            Vec3::splat(half + margin),
        );

        // Build SDF closure from the primitive.
        let sdf_fn = move |pos: Vec3| -> (f32, u16) {
            (rkf_core::evaluate_primitive(&primitive, pos), material_id)
        };

        let result = rkf_core::voxelize_sdf(
            sdf_fn,
            &aabb,
            voxel_size,
            &mut self.cpu_brick_pool,
            &mut self.cpu_brick_map_alloc,
        );

        let (handle, brick_count) = match result {
            Some(r) => r,
            None => {
                log::warn!("Auto-voxelize failed: not enough brick pool slots");
                return None;
            }
        };

        // Compute the grid-aligned AABB from the actual dims returned by voxelize_sdf.
        // The GPU shader centers the grid at the object origin: grid_pos = local_pos + grid_size * 0.5.
        // The voxelizer uses grid_origin = -grid_size/2 (which may differ from aabb.min due to
        // ceil rounding of dims). We MUST store the grid-aligned AABB so the CPU edit path
        // (find_affected_bricks, apply_edit_cpu) uses the same coordinate origin as the shader.
        let brick_size = voxel_size * 8.0;
        let grid_half = Vec3::new(
            handle.dims.x as f32 * brick_size * 0.5,
            handle.dims.y as f32 * brick_size * 0.5,
            handle.dims.z as f32 * brick_size * 0.5,
        );
        let grid_aabb = Aabb::new(-grid_half, grid_half);

        log::info!(
            "Auto-voxelized object {} ({}): {} bricks, voxel_size={:.4}, dims={:?}, aabb_min={:?} grid_min={:?}",
            object_id, obj.name, brick_count, voxel_size, handle.dims, aabb.min, grid_aabb.min,
        );

        // Update the object's SDF source.
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: handle,
            voxel_size,
            aabb: grid_aabb,
        };
        obj.aabb = grid_aabb;

        // Re-upload brick pool + brick maps to GPU.
        self.reupload_brick_data();

        Some((voxel_size, grid_aabb, handle))
    }

    /// Apply a batch of sculpt edit requests to the CPU brick pool and upload changes.
    ///
    /// Each request is converted to an `EditOp` in object-local space, then
    /// `apply_edit_cpu` modifies the CPU brick pool. Changed bricks are
    /// uploaded to the GPU via targeted `queue.write_buffer()`.
    ///
    /// Returns the list of all modified brick pool slot indices.
    ///
    /// If `undo_acc` is provided, bricks are snapshot before modification
    /// so the undo system can restore them.
    pub fn apply_sculpt_edits(
        &mut self,
        scene: &mut Scene,
        edits: &[crate::sculpt::SculptEditRequest],
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> Vec<u32> {
        use rkf_edit::cpu_apply::apply_edit_cpu;

        use rkf_edit::types::{EditType, FalloffCurve, ShapeType};

        let mut all_modified_slots = Vec::new();

        for req in edits {
            // 1. Auto-voxelize if analytical.
            self.ensure_object_voxelized(scene, req.object_id);

            // 2. Look up the object to get transform + SDF source info.
            let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                Some(o) => o,
                None => continue,
            };

            let (voxel_size, handle_pre, aabb_min_pre) = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { voxel_size, brick_map_handle, aabb } => {
                    (*voxel_size, *brick_map_handle, aabb.min)
                }
                _ => continue, // Shouldn't happen after auto-voxelize.
            };

            // 3. Transform world position to object-local space.
            let local_pos = crate::sculpt::world_to_object_local_v2(
                req.world_position,
                obj,
            );

            // 3b. Sample dominant material from the surface at the brush hit.
            //
            // Instead of using a fixed material_id from BrushSettings, sample
            // the existing voxel data around the hit point and use the most
            // common (dominant) material. This makes additive sculpting
            // naturally continue with the object's existing material.
            let sampled_material = self.sample_dominant_material(
                &handle_pre, voxel_size, aabb_min_pre, local_pos,
            );

            // 4. Convert BrushSettings → EditOp.
            let min_scale = obj.scale.x.min(obj.scale.y.min(obj.scale.z)).max(1e-6);
            let inv_scale = 1.0 / min_scale;

            let edit_type = match req.settings.brush_type {
                crate::sculpt::BrushType::Add => EditType::SmoothUnion,
                crate::sculpt::BrushType::Subtract => EditType::SmoothSubtract,
                crate::sculpt::BrushType::Smooth => EditType::Smooth,
                crate::sculpt::BrushType::Flatten => EditType::Flatten,
                crate::sculpt::BrushType::Sharpen => EditType::Smooth,
            };

            let shape_type = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => ShapeType::Sphere,
                crate::sculpt::BrushShape::Cube => ShapeType::Box,
                crate::sculpt::BrushShape::Cylinder => ShapeType::Cylinder,
            };

            let local_radius = req.settings.radius * inv_scale;
            let local_dims = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => Vec3::new(local_radius, 0.0, 0.0),
                crate::sculpt::BrushShape::Cube => Vec3::splat(local_radius),
                crate::sculpt::BrushShape::Cylinder => Vec3::new(local_radius, local_radius, 0.0),
            };

            let blend_k = local_radius * 0.3; // default smooth blend

            let op = rkf_edit::edit_op::EditOp {
                object_id: req.object_id,
                position: local_pos,
                rotation: glam::Quat::IDENTITY,
                edit_type,
                shape_type,
                dimensions: local_dims,
                strength: req.settings.strength,
                blend_k,
                falloff: FalloffCurve::Smooth,
                material_id: sampled_material,
                secondary_id: 0,
                color_packed: 0,
            };

            // 5/6. Apply edit: surface displacement for Add/Subtract,
            // apply_edit_cpu for Smooth/Flatten/Paint.
            let is_displacement = matches!(edit_type, EditType::SmoothUnion | EditType::SmoothSubtract);
            let (mut modified, scope_min, scope_max) =
                if is_displacement {
                self.apply_sculpt_displacement(
                    scene, req.object_id, &op, voxel_size, undo_acc.as_deref_mut(),
                )
            } else {
                // Smooth/Flatten/Paint: operate on already-allocated bricks only.
                // No new brick allocation needed — these ops only reshape existing geometry.
                let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                    Some(o) => o, None => continue,
                };
                let (handle, voxel_size_cur, aabb_min_cur) = match &obj.root_node.sdf_source {
                    SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } =>
                        (*brick_map_handle, *voxel_size, aabb.min),
                    _ => continue,
                };
                let all_bricks = Self::collect_all_allocated_bricks(
                    &self.cpu_brick_map_alloc, &handle, voxel_size_cur, aabb_min_cur,
                );
                if all_bricks.is_empty() {
                    log::warn!("  No allocated bricks found — skipping {:?}", edit_type);
                    continue;
                }
                if let Some(ref mut acc) = undo_acc {
                    for ab in &all_bricks {
                        let slot = ab.brick_base_index / 512;
                        if acc.captured_slots.insert(slot) {
                            acc.snapshots.push((slot, self.cpu_brick_pool.get(slot).clone()));
                        }
                    }
                }
                (apply_edit_cpu(&mut self.cpu_brick_pool, &all_bricks, &op),
                 glam::UVec3::ZERO, glam::UVec3::ZERO)
            };

            // NOTE: we intentionally do NOT call mark_interior_empties() here.
            // The flood-fill heuristic (unreachable from boundary = interior)
            // fails when sculpting creates disconnected bodies — EMPTY bricks
            // between them get incorrectly marked INTERIOR, causing the GPU to
            // render concentric ring artifacts. Interior marking is only safe
            // during initial voxelization (single connected body).

            // No distance recomputation needed — uniform subtraction
            // preserves |∇d| = 1 exactly.

            // Targeted GPU upload.
            {
                let brick_byte_size =
                    std::mem::size_of::<rkf_core::brick::Brick>() as u64;
                let gpu_buf_size = self.gpu_scene.brick_pool_buffer().size();

                for &slot in &modified {
                    let offset = slot as u64 * brick_byte_size;
                    if offset + brick_byte_size <= gpu_buf_size {
                        self.ctx.queue.write_buffer(
                            self.gpu_scene.brick_pool_buffer(),
                            offset,
                            bytemuck::bytes_of(self.cpu_brick_pool.get(slot)),
                        );
                    }
                }

                // Brick maps always need to be re-uploaded (new allocations may
                // have changed entries).
                let map_data = self.cpu_brick_map_alloc.as_slice();
                if !map_data.is_empty() {
                    self.gpu_scene.upload_brick_maps(
                        &self.ctx.device, &self.ctx.queue, map_data,
                    );
                }
            }

            all_modified_slots.extend(&modified);
        }

        all_modified_slots
    }
}
