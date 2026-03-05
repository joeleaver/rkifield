//! Brick repair, SDF consistency, and re-voxelize operations for the editor engine.

use glam::Vec3;
use rkf_core::{Aabb, Scene, SdfSource};
use super::EditorEngine;

impl EditorEngine {
    /// Narrow-band SDF repair after a boolean stamp.
    ///
    /// Recomputes correct signed Euclidean distances for all voxels in the
    /// modified scope using Dijkstra propagation from sub-voxel zero-crossing
    /// seeds.  This ensures |∇d| ≈ 1 near the surface regardless of how many
    /// overlapping strokes have accumulated.
    pub(super) fn narrow_band_sdf_repair(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        scope_min: glam::UVec3,
        scope_max: glam::UVec3,
    ) {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick::brick_index;
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

        let h = voxel_size;
        let dims = handle.dims;
        // Expand scope by 1 brick in each direction for margin.
        let bx0 = scope_min.x.saturating_sub(1);
        let by0 = scope_min.y.saturating_sub(1);
        let bz0 = scope_min.z.saturating_sub(1);
        let bx1 = (scope_max.x + 1).min(dims.x.saturating_sub(1));
        let by1 = (scope_max.y + 1).min(dims.y.saturating_sub(1));
        let bz1 = (scope_max.z + 1).min(dims.z.saturating_sub(1));
        // Voxel coordinate ranges for the repair region.
        let gx0 = bx0 * 8;
        let gy0 = by0 * 8;
        let gz0 = bz0 * 8;
        let gx1 = (bx1 + 1) * 8; // exclusive
        let gy1 = (by1 + 1) * 8;
        let gz1 = (bz1 + 1) * 8;
        let rw = (gx1 - gx0) as usize;
        let rh = (gy1 - gy0) as usize;
        let li = |gx: u32, gy: u32, gz: u32| -> usize {
            (gx - gx0) as usize
                + (gy - gy0) as usize * rw
                + (gz - gz0) as usize * rw * rh
        };

        let total = rw * rh * (gz1 - gz0) as usize;
        let inf: f32 = h * 1024.0;
        // Per-voxel arrays over the repair region.
        let mut vdist  = vec![inf;   total]; // Dijkstra unsigned distance
        let mut vsign  = vec![true;  total]; // true = exterior (d ≥ 0)
        let mut vactive = vec![false; total]; // voxel belongs to an allocated brick

        // Phase 1: snapshot stored distances + initial sign estimate.
        let mut stored_d_grid = vec![0.0f32; total]; // for sign BFS transition detection
        for bz in bz0..=bz1 {
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let slot = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(s) => s,
                        None    => continue,
                    };
                    for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                        let gx = bx * 8 + lx;
                        let gy = by * 8 + ly;
                        let gz = bz * 8 + lz;
                        let d = if slot == EMPTY_SLOT    {  h * 8.0 }
                                else if slot == INTERIOR_SLOT { -h * 2.0 }
                                else {
                                    let vi = brick_index(lx, ly, lz);
                                    self.cpu_brick_pool.get(slot).voxels[vi].distance_f32()
                                };
                        let i = li(gx, gy, gz);
                        stored_d_grid[i] = d;
                        vsign[i]   = d >= 0.0;
                        vactive[i] = true;
                    }}}
                }
            }
        }

        let dirs: [(i32,i32,i32); 6] = [
            (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
        ];

        // Phase 1b: sign BFS — correct interior/exterior classification.
        //
        // classify_exterior_bricks flood-fills from the actual grid boundary so
        // enclosed EMPTY_SLOT bricks (former gap between merged blobs) get seeded
        // as interior, not exterior.
        {
            use std::collections::VecDeque;

            let ext_bricks = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, handle);

            // 0 = unvisited, 1 = exterior, 2 = interior
            let mut bfs_mark = vec![0u8; total];
            let mut bfs: VecDeque<u32> = VecDeque::new();

            for bz in bz0..=bz1 { for by in by0..=by1 { for bx in bx0..=bx1 {
                let slot = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                    Some(s) => s, None => continue,
                };
                if slot != EMPTY_SLOT && slot != INTERIOR_SLOT { continue; }
                let mark = if ext_bricks.contains(&(bx, by, bz)) {
                    1u8  // truly exterior (grid-boundary-connected)
                } else {
                    2u8  // enclosed empty or INTERIOR_SLOT → interior
                };
                for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                    let gx = bx * 8 + lx;
                    let gy = by * 8 + ly;
                    let gz = bz * 8 + lz;
                    let i = li(gx, gy, gz) as u32;
                    bfs_mark[i as usize] = mark;
                    bfs.push_back(i);
                }}}
            }}}

            let rw_u = rw as u32;
            let rh_u = rh as u32;
            let rd_u = (gz1 - gz0) as u32;
            let li_u = |gx: u32, gy: u32, gz: u32| -> u32 {
                (gx - gx0) + (gy - gy0) * rw_u + (gz - gz0) * rw_u * rh_u
            };

            while let Some(fi) = bfs.pop_front() {
                let mark = bfs_mark[fi as usize];
                let gx = gx0 + (fi % rw_u);
                let gy = gy0 + (fi / rw_u) % rh_u;
                let gz = gz0 + fi / (rw_u * rh_u);

                for &(dx, dy, dz) in &dirs {
                    let nx = gx as i32 + dx;
                    let ny = gy as i32 + dy;
                    let nz = gz as i32 + dz;
                    if nx < gx0 as i32 || ny < gy0 as i32 || nz < gz0 as i32 { continue; }
                    if nx >= gx1 as i32 || ny >= gy1 as i32 || nz >= gz1 as i32 { continue; }
                    let nfi = li_u(nx as u32, ny as u32, nz as u32);
                    if bfs_mark[nfi as usize] != 0 { continue; }

                    let d_nc = stored_d_grid[nfi as usize];
                    // Stop at stored zero-crossings using propagation mark, not d_here.
                    // Using d_here * d_nc < 0 is wrong when d_here is h*8.0 (EMPTY_SLOT)
                    // but mark is interior (enclosed EMPTY_SLOT), causing it to propagate
                    // through stale-positive gap voxels.
                    if (mark == 1) != (d_nc >= 0.0) {
                        continue;
                    }
                    bfs_mark[nfi as usize] = mark;
                    bfs.push_back(nfi);
                }
            }

            // Apply BFS results to vsign.
            // Unreachable voxels (bfs_mark=0): use stored_d sign.
            // Enclosed exterior voids (cavities, gaps) have stored_d > 0 → correctly exterior.
            // Deep interior with no nearby surface has stored_d < 0 → correctly interior.
            for i in 0..total {
                match bfs_mark[i] {
                    1 => vsign[i] = true,                          // exterior
                    2 => vsign[i] = false,                         // interior
                    _ => vsign[i] = stored_d_grid[i] >= 0.0,      // enclosed void or deep interior
                }
            }
        }

        // Phase 2: find zero-crossing seeds.

        // Heap entries: (dist_bits, gx, gy, gz) — f32 bits for ordering.
        let mut heap: BinaryHeap<Reverse<(u32, u32, u32, u32)>> = BinaryHeap::new();
        for bz in bz0..=bz1 {
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let slot = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(s) if s != EMPTY_SLOT && s != INTERIOR_SLOT => s,
                        _ => continue,
                    };
                    for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                        let gx = bx * 8 + lx;
                        let gy = by * 8 + ly;
                        let gz = bz * 8 + lz;
                        let vi = brick_index(lx, ly, lz);
                        let d_self = self.cpu_brick_pool.get(slot).voxels[vi].distance_f32();
                        let sign_self = d_self >= 0.0;
                        let abs_self  = d_self.abs().max(1e-6);

                        let mut best_seed = inf;
                        for &(dx, dy, dz) in &dirs {
                            let nx = gx as i32 + dx;
                            let ny = gy as i32 + dy;
                            let nz = gz as i32 + dz;
                            if nx < gx0 as i32 || ny < gy0 as i32 || nz < gz0 as i32 { continue; }
                            if nx >= gx1 as i32 || ny >= gy1 as i32 || nz >= gz1 as i32 { continue; }
                            let ni = li(nx as u32, ny as u32, nz as u32);
                            if !vactive[ni] { continue; }

                            // Look up neighbor distance from the snapshot sign
                            // (we already have vsign; reconstruct magnitude from brick).
                            let (nbx, nby, nbz) = (nx as u32 / 8, ny as u32 / 8, nz as u32 / 8);
                            let (nlx, nly, nlz) = (nx as u32 % 8, ny as u32 % 8, nz as u32 % 8);
                            let nslot = match self.cpu_brick_map_alloc.get_entry(handle, nbx, nby, nbz) {
                                Some(s) => s, None => continue,
                            };
                            let d_n = if nslot == EMPTY_SLOT    {  h * 8.0 }
                                      else if nslot == INTERIOR_SLOT { -h * 2.0 }
                                      else {
                                          let nvi = brick_index(nlx, nly, nlz);
                                          self.cpu_brick_pool.get(nslot).voxels[nvi].distance_f32()
                                      };
                            let sign_n = d_n >= 0.0;

                            if sign_self != sign_n {
                                let abs_n = d_n.abs().max(1e-6);
                                let t    = abs_self / (abs_self + abs_n);
                                let seed = t * h;
                                if seed < best_seed { best_seed = seed; }
                            }
                        }

                        if best_seed < inf {
                            let i = li(gx, gy, gz);
                            vdist[i] = best_seed;
                            heap.push(Reverse((best_seed.to_bits(), gx, gy, gz)));
                        }
                    }}}
                }
            }
        }

        // Phase 3: Dijkstra propagation.
        while let Some(Reverse((d_bits, gx, gy, gz))) = heap.pop() {
            let i   = li(gx, gy, gz);
            let cur = f32::from_bits(d_bits);
            if cur > vdist[i] { continue; } // stale entry

            for &(dx, dy, dz) in &dirs {
                let nx = gx as i32 + dx;
                let ny = gy as i32 + dy;
                let nz = gz as i32 + dz;
                if nx < gx0 as i32 || ny < gy0 as i32 || nz < gz0 as i32 { continue; }
                if nx >= gx1 as i32 || ny >= gy1 as i32 || nz >= gz1 as i32 { continue; }
                let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                let ni = li(nx, ny, nz);
                if !vactive[ni] { continue; }

                let new_d = cur + h;
                if new_d < vdist[ni] {
                    vdist[ni] = new_d;
                    heap.push(Reverse((new_d.to_bits(), nx, ny, nz)));
                }
            }
        }

        // Phase 4: write corrected distances back (core scope only).
        for bz in scope_min.z..=scope_max.z {
            for by in scope_min.y..=scope_max.y {
                for bx in scope_min.x..=scope_max.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(s) if s != EMPTY_SLOT && s != INTERIOR_SLOT => s,
                        _ => continue,
                    };
                    for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                        let gx = bx * 8 + lx;
                        let gy = by * 8 + ly;
                        let gz = bz * 8 + lz;
                        let i  = li(gx, gy, gz);
                        let d  = vdist[i];
                        if d >= inf { continue; } // unreachable — leave stored value
                        let signed_d = if vsign[i] { d } else { -d };
                        let vi   = brick_index(lx, ly, lz);
                        let orig = self.cpu_brick_pool.get(slot).voxels[vi];
                        self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                            signed_d, orig.material_id(),
                            orig.blend_weight(), orig.secondary_id(), orig.flags(),
                        );
                    }}}
                }
            }
        }
    }

    /// Pre-fill new brick voxels by extrapolating from adjacent filled bricks.
    ///
    /// Before CSG runs, this ensures the boundary between old and new bricks
    /// has consistent background SDF. Since `smooth_min(same_bg, brush, k)`
    /// produces identical results on both sides, there's no gradient discontinuity
    /// and no visible shading seam.
    pub(super) fn prefill_new_brick_faces(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        newly_allocated: &[u32],
    ) {
        use std::collections::HashSet;
        use rkf_core::voxel::VoxelSample;

        if newly_allocated.is_empty() {
            return;
        }

        let vs = voxel_size;
        let dims = handle.dims;
        let new_set: HashSet<u32> = newly_allocated.iter().copied().collect();

        // Build (bx, by, bz, slot) for new bricks.
        let mut new_coords: Vec<(u32, u32, u32, u32)> = Vec::new();
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if new_set.contains(&s) {
                            new_coords.push((bx, by, bz, s));
                        }
                    }
                }
            }
        }

        let face_dirs: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        let fill = vs * 2.0; // constant fill value
        let depth = 8u32; // fill entire brick

        // Multi-pass flood fill: propagate from old bricks through layers of
        // new bricks. Each pass fills unfilled new bricks from any filled
        // neighbor (old or previously-filled new). This ensures ALL new bricks
        // the brush touches have a smooth, continuous SDF background.
        let mut filled_set: HashSet<u32> = HashSet::new();
        // All non-new allocated bricks are "filled" (they have real SDF data).
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(s) && !new_set.contains(&s) {
                            filled_set.insert(s);
                        }
                    }
                }
            }
        }

        let mut total_copied = 0u32;
        let mut pass = 0u32;
        let max_passes = 10; // safety limit

        loop {
            pass += 1;
            if pass > max_passes { break; }

            let mut pass_copied = 0u32;
            let mut newly_filled: Vec<u32> = Vec::new();

            for &(nbx, nby, nbz, new_slot) in &new_coords {
                if filled_set.contains(&new_slot) { continue; }

                let mut brick_got_data = false;

                for &(dx, dy, dz) in &face_dirs {
                    let ax = nbx as i32 + dx;
                    let ay = nby as i32 + dy;
                    let az = nbz as i32 + dz;
                    if ax < 0 || ay < 0 || az < 0 { continue; }
                    let (abx, aby, abz) = (ax as u32, ay as u32, az as u32);
                    if abx >= dims.x || aby >= dims.y || abz >= dims.z { continue; }

                    // Copy from any FILLED neighbor (old or already-filled new).
                    let adj_slot = match self.cpu_brick_map_alloc.get_entry(handle, abx, aby, abz) {
                        Some(s) if filled_set.contains(&s) => s,
                        _ => continue,
                    };

                    // Pre-read face data to avoid borrow conflict with get_mut.
                    let mut face_data = [[(0.0f32, 0.0f32, 0u16); 8]; 8];
                    {
                        let adj_brick = self.cpu_brick_pool.get(adj_slot);
                        for a in 0u32..8 {
                            for b in 0u32..8 {
                                let (fv, pv, mat) = if dx == -1 {
                                    (adj_brick.sample(7, a, b).distance_f32(),
                                     adj_brick.sample(6, a, b).distance_f32(),
                                     adj_brick.sample(7, a, b).material_id())
                                } else if dx == 1 {
                                    (adj_brick.sample(0, a, b).distance_f32(),
                                     adj_brick.sample(1, a, b).distance_f32(),
                                     adj_brick.sample(0, a, b).material_id())
                                } else if dy == -1 {
                                    (adj_brick.sample(a, 7, b).distance_f32(),
                                     adj_brick.sample(a, 6, b).distance_f32(),
                                     adj_brick.sample(a, 7, b).material_id())
                                } else if dy == 1 {
                                    (adj_brick.sample(a, 0, b).distance_f32(),
                                     adj_brick.sample(a, 1, b).distance_f32(),
                                     adj_brick.sample(a, 0, b).material_id())
                                } else if dz == -1 {
                                    (adj_brick.sample(a, b, 7).distance_f32(),
                                     adj_brick.sample(a, b, 6).distance_f32(),
                                     adj_brick.sample(a, b, 7).material_id())
                                } else {
                                    (adj_brick.sample(a, b, 0).distance_f32(),
                                     adj_brick.sample(a, b, 1).distance_f32(),
                                     adj_brick.sample(a, b, 0).material_id())
                                };
                                face_data[a as usize][b as usize] = (fv, pv, mat);
                            }
                        }
                    }

                    // Extrapolate `depth` voxels from neighbor's face into new brick.
                    for a in 0u32..8 {
                        for b in 0u32..8 {
                            let (face_val, prev_val, mat) = face_data[a as usize][b as usize];
                            // Extrapolate even when the adjacent face voxel is interior
                            // (negative). The .max(vs * 0.5) clamp below prevents false
                            // surfaces from negative extrapolation in concave regions.
                            // Without this, once the surface grows all the way to a brick's
                            // outer face (making it negative), the next shell of bricks can
                            // never be prefilled and growth stalls permanently.
                            let grad = (face_val - prev_val).clamp(-vs, vs);

                            // When the adjacent face is interior (face_val < 0) the
                            // surface lies between that brick and this new brick.
                            // Force the gradient to be positive so distances increase
                            // outward into the new brick. abs().max(vs) guarantees
                            // |∇d| = 1 (correct SDF gradient) into the new brick.
                            let grad = if face_val <= 0.0 {
                                grad.abs().max(vs)
                            } else {
                                grad
                            };

                            for d in 0..depth {
                                let val = face_val + grad * ((d + 1) as f32);
                                // Exterior faces: allow values to reach near-zero so the
                                // surface can naturally advance through subtraction.
                                // Small positive floor prevents accidental sign flip.
                                let val = if face_val > 0.0 { val.max(vs * 0.1) } else { val };

                                let (dst_x, dst_y, dst_z) = if dx == -1 {
                                    (d, a, b)
                                } else if dx == 1 {
                                    (7 - d, a, b)
                                } else if dy == -1 {
                                    (a, d, b)
                                } else if dy == 1 {
                                    (a, 7 - d, b)
                                } else if dz == -1 {
                                    (a, b, d)
                                } else {
                                    (a, b, 7 - d)
                                };

                                let cur = self.cpu_brick_pool.get(new_slot)
                                    .sample(dst_x, dst_y, dst_z).distance_f32();
                                if (cur - fill).abs() < 0.01 || val.abs() < cur.abs() {
                                    self.cpu_brick_pool.get_mut(new_slot).set(
                                        dst_x, dst_y, dst_z,
                                        VoxelSample::new(val, mat, 0, 0, 0),
                                    );
                                }
                            }
                        }
                    }
                    brick_got_data = true;
                    pass_copied += 1;
                }

                if brick_got_data {
                    newly_filled.push(new_slot);
                }
            }

            for s in &newly_filled {
                filled_set.insert(*s);
            }
            total_copied += pass_copied;

            if pass_copied == 0 { break; }
        }

        if total_copied > 0 {
            log::info!(
                "  prefill_new_brick_faces: extrapolated {} face boundaries over {} passes",
                total_copied, pass - 1,
            );
        }
    }

    /// Ensure SDF consistency in the region around a sculpt edit.
    ///
    /// After a CSG edit, the GPU may see gradient discontinuities at boundaries
    /// between allocated bricks (with real SDF) and EMPTY_SLOT bricks (which
    /// the GPU shader treats as `+vs*4.0`). This produces flipped normals on
    /// the interior side of sculpted geometry.
    ///
    /// Returns newly allocated pool slot indices.
    pub(super) fn ensure_sdf_consistency(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        aabb_min: Vec3,
        edit_op: &rkf_edit::edit_op::EditOp,
    ) -> Vec<u32> {
        use std::collections::{HashSet, VecDeque};
        use rkf_core::voxel::VoxelSample;

        let dims = handle.dims;
        let brick_size = voxel_size * 8.0;

        // Compute a padded region: the edit AABB expanded by 4 bricks on each side.
        let (edit_min, edit_max) = edit_op.local_aabb();
        let pad = brick_size * 4.0;
        let region_min = edit_min - Vec3::splat(pad);
        let region_max = edit_max + Vec3::splat(pad);

        // Convert to brick coordinates, clamped to grid bounds.
        let bmin = ((region_min - aabb_min) / brick_size).floor();
        let bmax = ((region_max - aabb_min) / brick_size - Vec3::splat(0.001)).ceil();
        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(dims.z.saturating_sub(1));

        // Collect all EMPTY_SLOT bricks in the region and seed the BFS
        // from all already-allocated bricks.
        let mut empty_set: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut bfs_seeds: VecDeque<(u32, u32, u32)> = VecDeque::new();

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => {
                            bfs_seeds.push_back((bx, by, bz));
                        }
                        _ => {
                            empty_set.insert((bx, by, bz));
                        }
                    }
                }
            }
        }

        if empty_set.is_empty() {
            return Vec::new();
        }

        log::info!(
            "  ensure_sdf_consistency: region [{},{}]→[{},{}], {} empty bricks to fill",
            bmin_x, bmin_y, bmax_x, bmax_y, empty_set.len(),
        );

        // BFS from allocated bricks outward.
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        let mut visited: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut new_slots: Vec<(u32, u32, u32, u32)> = Vec::new(); // (bx, by, bz, slot)

        // Classify exterior vs interior EMPTY_SLOT bricks via flood-fill.
        let exterior = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, handle);

        // Mark all seeds as visited (they're already allocated).
        for &coord in bfs_seeds.iter() {
            visited.insert(coord);
        }

        // BFS layer by layer.
        while !bfs_seeds.is_empty() {
            // Collect the next wave: all EMPTY_SLOT neighbors of the current
            // frontier that haven't been visited yet.
            let mut next_wave: Vec<(u32, u32, u32)> = Vec::new();
            for &(bx, by, bz) in bfs_seeds.iter() {
                for &(dx, dy, dz) in &deltas {
                    let nx = bx as i32 + dx;
                    let ny = by as i32 + dy;
                    let nz = bz as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let coord = (nx as u32, ny as u32, nz as u32);
                    if empty_set.contains(&coord) && visited.insert(coord) {
                        next_wave.push(coord);
                    }
                }
            }

            if next_wave.is_empty() {
                break;
            }

            // Allocate + fill every brick in this wave.
            for &(nbx, nby, nbz) in &next_wave {
                // Sample SDF into temp buffer (immutable borrows of pool).
                let mut sdf_buf = [0.0f32; 512];
                let mut min_val = f32::MAX;
                for z in 0u32..8 {
                    for y in 0u32..8 {
                        for x in 0u32..8 {
                            let voxel_pos = aabb_min
                                + Vec3::new(
                                    (nbx * 8 + x) as f32 * voxel_size + voxel_size * 0.5,
                                    (nby * 8 + y) as f32 * voxel_size + voxel_size * 0.5,
                                    (nbz * 8 + z) as f32 * voxel_size + voxel_size * 0.5,
                                );
                            let d = Self::sample_sdf_cpu(
                                &self.cpu_brick_pool,
                                &self.cpu_brick_map_alloc,
                                handle,
                                voxel_size,
                                voxel_pos,
                                &exterior,
                            );
                            sdf_buf[(x + y * 8 + z * 64) as usize] = d;
                            if d < min_val { min_val = d; }
                        }
                    }
                }

                // Skip bricks that are entirely outside any surface.
                if min_val > voxel_size {
                    continue;
                }

                // Allocate and fill (mutable borrows).
                if let Some(new_slot) = self.cpu_brick_pool.allocate() {
                    let nbrick = self.cpu_brick_pool.get_mut(new_slot);
                    for z in 0u32..8 {
                        for y in 0u32..8 {
                            for x in 0u32..8 {
                                let d = sdf_buf[(x + y * 8 + z * 64) as usize];
                                nbrick.set(x, y, z, VoxelSample::new(d, 0, 0, 0, 0));
                            }
                        }
                    }
                    self.cpu_brick_map_alloc.set_entry(handle, nbx, nby, nbz, new_slot);
                    new_slots.push((nbx, nby, nbz, new_slot));
                }
            }

            // Remove filled bricks from the empty set so they're not
            // re-processed, and use this wave as the next frontier.
            for &coord in &next_wave {
                empty_set.remove(&coord);
            }
            bfs_seeds.clear();
            bfs_seeds.extend(next_wave);
        }

        let slot_ids: Vec<u32> = new_slots.iter().map(|&(_, _, _, s)| s).collect();

        if !slot_ids.is_empty() {
            log::info!("  ensure_sdf_consistency: allocated {} bricks", slot_ids.len());
            let map_data = self.cpu_brick_map_alloc.as_slice();
            if !map_data.is_empty() {
                self.gpu_scene.upload_brick_maps(
                    &self.ctx.device, &self.ctx.queue, map_data,
                );
            }
        }

        slot_ids
    }

    /// Compute the tight local-space AABB of all non-EMPTY_SLOT bricks for a
    /// voxelized FlatNode.
    ///
    /// Used to populate `GpuObject::geometry_aabb_min/max` so the shader can
    /// skip the empty expanded region that `grow_brick_map_if_needed` adds.
    pub(super) fn compute_geometry_aabb_for_flat_node(
        &self,
        flat: &rkf_core::transform_flatten::FlatNode,
    ) -> ([f32; 3], [f32; 3]) {
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
        let (handle, vs) = match &flat.sdf_source {
            rkf_core::SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return ([0.0; 3], [0.0; 3]),
        };
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(dims.x as f32, dims.y as f32, dims.z as f32) * brick_extent;
        let mut gmin = Vec3::splat(f32::MAX);
        let mut gmax = Vec3::splat(f32::MIN);
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) => s,
                        None => continue,
                    };
                    // Skip sentinel values — only real pool slots contribute geometry.
                    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                        continue;
                    }
                    let brick_min = Vec3::new(bx as f32, by as f32, bz as f32) * brick_extent - grid_size * 0.5;
                    let brick_max = Vec3::new((bx + 1) as f32, (by + 1) as f32, (bz + 1) as f32) * brick_extent - grid_size * 0.5;
                    gmin = gmin.min(brick_min);
                    gmax = gmax.max(brick_max);
                }
            }
        }
        if gmin.x > gmax.x {
            return ([0.0; 3], [0.0; 3]);
        }
        (gmin.to_array(), gmax.to_array())
    }

    /// Mark all interior EMPTY_SLOT bricks with INTERIOR_SLOT in the brick map.
    ///
    /// The GPU shader returns `+vs*4.0` for EMPTY_SLOT and `-vs*4.0` for
    /// INTERIOR_SLOT. This method classifies unallocated bricks via flood-fill
    /// from the grid boundary and sets INTERIOR_SLOT for any EMPTY_SLOT brick
    /// that isn't reachable from the boundary (i.e. enclosed by allocated bricks).
    ///
    /// Returns `true` if any entries were changed (brick maps need re-upload).
    pub(super) fn mark_interior_empties(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
    ) -> bool {
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

        let exterior = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, handle);
        let dims = handle.dims;
        let mut changed = false;

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if slot == EMPTY_SLOT && !exterior.contains(&(bx, by, bz)) {
                            self.cpu_brick_map_alloc.set_entry(handle, bx, by, bz, INTERIOR_SLOT);
                            changed = true;
                        }
                    }
                }
            }
        }

        if changed {
            log::info!("  mark_interior_empties: marked interior bricks as INTERIOR_SLOT");
        }

        changed
    }

    /// Grow the CPU brick pool (and matching GPU buffer) if free slots are low.
    ///
    /// Called before any sculpt allocation pass. Doubles capacity whenever fewer
    /// than `min_free` slots remain. The GPU buffer is recreated at the new size
    /// and the existing brick data is re-uploaded so slot indices stay valid.
    pub(super) fn grow_brick_pool_if_needed(&mut self, min_free: u32) {
        if self.cpu_brick_pool.free_count() >= min_free {
            return;
        }
        let old_cap = self.cpu_brick_pool.capacity();
        let new_cap = (old_cap * 2).max(old_cap + min_free);
        log::info!(
            "grow_brick_pool_if_needed: {} → {} slots ({} free, needed {})",
            old_cap, new_cap, self.cpu_brick_pool.free_count(), min_free,
        );
        self.cpu_brick_pool.grow(new_cap);

        // Recreate the GPU buffer at the new size and re-upload all brick data.
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: pool_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
        self.gpu_scene.set_brick_pool(&self.ctx.device, new_buf);
    }

    /// Re-upload the entire CPU brick pool and brick maps to the GPU.
    ///
    /// Called after voxelization or when the GPU buffer needs to grow.
    pub(crate) fn reupload_brick_data(&mut self) {
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        let gpu_buf = self.gpu_scene.brick_pool_buffer();

        if pool_data.len() as u64 > gpu_buf.size() {
            // GPU buffer too small — recreate it.
            let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brick_pool"),
                size: pool_data.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
            self.gpu_scene.set_brick_pool(&self.ctx.device, new_buf);
        } else {
            self.ctx.queue.write_buffer(gpu_buf, 0, pool_data);
        }

        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }
    }

    /// Collect ALL allocated bricks in an object's brick map.
    ///
    /// Returns an `AffectedBrick` for every non-empty brick in the grid.
    /// Used to apply CSG to the entire object (not just the edit AABB),
    /// ensuring perfectly consistent SDF with no missed bricks.
    pub(super) fn collect_all_allocated_bricks(
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        aabb_min: Vec3,
    ) -> Vec<rkf_edit::edit_op::AffectedBrick> {
        let brick_size = voxel_size * 8.0;
        let mut bricks = Vec::new();

        for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let brick_local_min = aabb_min
                                + Vec3::new(
                                    bx as f32 * brick_size,
                                    by as f32 * brick_size,
                                    bz as f32 * brick_size,
                                );
                            bricks.push(rkf_edit::edit_op::AffectedBrick {
                                brick_base_index: slot * 512,
                                brick_local_min: brick_local_min.into(),
                                voxel_size,
                            });
                        }
                    }
                }
            }
        }

        bricks
    }

    /// Process re-voxelization of an object with non-uniform scale.
    ///
    /// Resamples the brick volume at the current stretched dimensions and
    /// resets scale to `(1,1,1)`, eliminating the extra march-step overhead.
    pub fn process_revoxelize(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        use rkf_core::brick::Brick;
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::sampling::{sample_brick_trilinear, sample_brick_nearest_material};

        // Find the object.
        let obj = match scene.objects.iter_mut().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return false,
        };

        // Must be voxelized with non-uniform scale.
        let (old_handle, voxel_size, old_aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (*brick_map_handle, *voxel_size, *aabb)
            }
            _ => return false,
        };

        let scale = obj.scale;
        if (scale - Vec3::ONE).length() < 1e-4 {
            return false; // Already uniform — nothing to do.
        }

        // 1. Copy old brick data into temporary storage.
        let old_dims = old_handle.dims;
        let mut old_bricks: std::collections::HashMap<(u32, u32, u32), Brick> =
            std::collections::HashMap::new();

        for bz in 0..old_dims.z {
            for by in 0..old_dims.y {
                for bx in 0..old_dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(
                        &old_handle, bx, by, bz,
                    ) {
                        if !Self::is_unallocated(slot) {
                            old_bricks.insert(
                                (bx, by, bz),
                                self.cpu_brick_pool.get(slot).clone(),
                            );
                        }
                    }
                }
            }
        }

        // 2. Deallocate old bricks from the pool.
        for bz in 0..old_dims.z {
            for by in 0..old_dims.y {
                for bx in 0..old_dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(
                        &old_handle, bx, by, bz,
                    ) {
                        if !Self::is_unallocated(slot) {
                            self.cpu_brick_pool.deallocate(slot);
                        }
                    }
                }
            }
        }
        self.cpu_brick_map_alloc.deallocate(old_handle);

        // 3. Build a sampling closure that reads from the old brick data.
        let brick_world_size = voxel_size * BRICK_DIM as f32;
        let old_grid_size = Vec3::new(
            old_dims.x as f32 * brick_world_size,
            old_dims.y as f32 * brick_world_size,
            old_dims.z as f32 * brick_world_size,
        );
        let old_grid_origin = -old_grid_size * 0.5;
        let min_scale = scale.x.min(scale.y.min(scale.z));

        let sample_fn = |new_pos: Vec3| -> (f32, u16) {
            // Map from new (scaled) space to old (unscaled) space.
            let old_pos = new_pos / scale;

            // Convert to grid-relative coordinates.
            let grid_rel = (old_pos - old_grid_origin) / brick_world_size;

            // Brick coordinates.
            let bx = grid_rel.x.floor() as i32;
            let by = grid_rel.y.floor() as i32;
            let bz = grid_rel.z.floor() as i32;

            // Check bounds.
            if bx < 0 || by < 0 || bz < 0
                || bx >= old_dims.x as i32
                || by >= old_dims.y as i32
                || bz >= old_dims.z as i32
            {
                return (f32::MAX, 0);
            }

            let bx = bx as u32;
            let by = by as u32;
            let bz = bz as u32;

            if let Some(brick) = old_bricks.get(&(bx, by, bz)) {
                // Local position within the brick (0..1).
                let local = Vec3::new(
                    grid_rel.x - bx as f32,
                    grid_rel.y - by as f32,
                    grid_rel.z - bz as f32,
                );
                let dist = sample_brick_trilinear(brick, local) * min_scale;
                let mat = sample_brick_nearest_material(brick, local);
                (dist, mat)
            } else {
                // Empty brick — far from surface.
                (f32::MAX, 0)
            }
        };

        // 4. Compute new AABB (scaled version of old AABB).
        let new_aabb = Aabb::new(old_aabb.min * scale, old_aabb.max * scale);

        // 5. Run voxelize_sdf with the sampling closure.
        let result = rkf_core::voxelize_sdf(
            sample_fn,
            &new_aabb,
            voxel_size,
            &mut self.cpu_brick_pool,
            &mut self.cpu_brick_map_alloc,
        );

        let (new_handle, _brick_count) = match result {
            Some(r) => r,
            None => {
                log::warn!("Re-voxelize failed: not enough brick pool slots");
                return false;
            }
        };

        // 6. Update the object. Use grid-aligned AABB from actual dims.
        let revox_brick_size = voxel_size * 8.0;
        let revox_grid_half = Vec3::new(
            new_handle.dims.x as f32 * revox_brick_size * 0.5,
            new_handle.dims.y as f32 * revox_brick_size * 0.5,
            new_handle.dims.z as f32 * revox_brick_size * 0.5,
        );
        let revox_grid_aabb = Aabb::new(-revox_grid_half, revox_grid_half);
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id).unwrap();
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: revox_grid_aabb,
        };
        obj.aabb = revox_grid_aabb;
        obj.scale = Vec3::ONE;

        // 7. Re-upload brick pool and brick maps to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        self.ctx.queue.write_buffer(
            self.gpu_scene.brick_pool_buffer(), 0, pool_data,
        );
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }

        log::info!(
            "Re-voxelized object {} — scale reset to (1,1,1)",
            object_id,
        );
        true
    }
}
