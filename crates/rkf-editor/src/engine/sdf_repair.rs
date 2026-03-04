//! SDF repair operations: FMM, EDT, JFA for the editor engine.

use rkf_core::{Scene, SdfSource};
use super::EditorEngine;

impl EditorEngine {
    /// Recompute correct SDF magnitudes from zero-crossings via BFS (Dijkstra).
    ///
    /// After iterative sculpting, the CSG formula leaves surrounding voxels with
    /// incorrect distance magnitudes — each stroke compounds the error. This method
    /// repairs the field without changing geometry:
    ///
    /// 1. Collect all allocated voxels.
    /// 2. Seed a min-heap with voxels adjacent to sign changes (|original dist|).
    /// 3. Dijkstra: propagate distance + voxel_size to 6-face neighbors.
    /// 4. Sign flood-fill from boundary voxels (always exterior) to determine
    ///    interior vs exterior; propagate through positive-sign voxels, stop at
    ///    sign changes.
    /// 5. Write back: ±bfs_distance, preserving material_id/blend_weight/flags.
    /// 6. Re-run mark_interior_empties and full GPU reupload.
    ///
    /// Returns `true` if a fix was performed.
    pub fn process_fix_sdfs(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        let (handle, voxel_size) = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => match &o.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } =>
                    (*brick_map_handle, *voxel_size),
                _ => return false,
            },
            None => return false,
        };

        // Run narrow-band Dijkstra repair over the entire brick map.
        let scope_min = glam::UVec3::ZERO;
        let scope_max = handle.dims.saturating_sub(glam::UVec3::ONE);
        self.narrow_band_sdf_repair(&handle, voxel_size, scope_min, scope_max);

        // Full GPU reupload.
        self.reupload_brick_data();
        true
    }

    /// Recompute correct SDF distances for `object_id` using GPU JFA.
    ///
    /// 1. Builds a dense solid/empty grid from the CPU brick pool.
    /// 2. Runs GPU JFA (init → log2(N) passes → writeback) to get Euclidean distances.
    /// 3. Reads back the result and writes distances into `cpu_brick_pool`
    ///    (preserving material_id, blend_weight, flags).
    ///
    /// Does NOT call `reupload_brick_data` — the caller is responsible.
    /// Returns `false` if the object is not found or has no voxelized data.
    pub(super) fn run_jfa_repair(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick_map::INTERIOR_SLOT;

        let (handle, voxel_size) = {
            let obj = match scene.objects.iter().find(|o| o.id == object_id) {
                Some(o) => o,
                None => return false,
            };
            match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                    (*brick_map_handle, *voxel_size)
                }
                _ => return false,
            }
        };

        let dims = handle.dims;
        let bd = BRICK_DIM as u32;
        let gw = (dims.x * bd) as usize;
        let gh = (dims.y * bd) as usize;
        let gd = (dims.z * bd) as usize;
        let total = gw * gh * gd;
        if total == 0 { return false; }

        let idx3 = |x: usize, y: usize, z: usize| -> usize { z * gh * gw + y * gw + x };

        // Build dense solid grid + per-voxel metadata for allocated bricks.
        let mut solid = vec![false; total];
        let mut is_allocated = vec![false; total];
        let mut mat_grid: Vec<u16>  = vec![0; total];
        let mut blend_grid: Vec<u8> = vec![0; total];
        let mut sec_id_grid: Vec<u8> = vec![0; total];
        let mut flags_grid: Vec<u8>  = vec![0; total];

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if s == INTERIOR_SLOT => {
                            // Treat entire INTERIOR_SLOT brick as solid.
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                solid[idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                )] = true;
                            }}}
                        }
                        Some(s) if !Self::is_unallocated(s) => {
                            let brick = self.cpu_brick_pool.get(s);
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                let vi = brick_index(lx, ly, lz);
                                let vs = brick.voxels[vi];
                                let i = idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                );
                                solid[i]      = vs.distance_f32() < 0.0;
                                is_allocated[i] = true;
                                mat_grid[i]    = vs.material_id();
                                blend_grid[i]  = vs.blend_weight();
                                sec_id_grid[i] = vs.secondary_id();
                                flags_grid[i]  = vs.flags();
                            }}}
                        }
                        _ => {} // EMPTY_SLOT or None → stays empty/false
                    }
                }
            }
        }

        // Run GPU JFA.
        let mut distances = match self.jfa_sdf.repair(
            &self.ctx.device, &self.ctx.queue,
            &solid, gw, gh, gd, voxel_size,
        ) {
            Some(d) => d,
            None => return false,
        };

        // Smooth the JFA distances to remove Voronoi cell boundary artifacts.
        //
        // JFA gives correct Euclidean distances, but the gradient is a Voronoi
        // diagram: discontinuous at cell boundaries between different nearest seeds.
        // Without smoothing this produces "squarish" normals (brick-sized faceting).
        //
        // 3 iterations of sign-preserving 6-neighbour box blur restores smooth
        // gradients while keeping the zero-crossing in place (no geometry change).
        // Only allocated voxels participate — avoids pulling surface distances
        // toward the large empty-brick fallback values.
        Self::smooth_sdf_distances(
            &mut distances, &is_allocated, gw, gh, gd, voxel_size, 3,
        );

        // Write corrected distances back to allocated bricks only.
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };
                    let brick = self.cpu_brick_pool.get_mut(slot);
                    for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                        let i = idx3(
                            (bx*bd+lx) as usize,
                            (by*bd+ly) as usize,
                            (bz*bd+lz) as usize,
                        );
                        if !is_allocated[i] { continue; }
                        let vi = brick_index(lx, ly, lz);
                        brick.voxels[vi] = VoxelSample::new(
                            distances[i],
                            mat_grid[i],
                            blend_grid[i],
                            sec_id_grid[i],
                            flags_grid[i],
                        );
                    }}}
                }
            }
        }

        log::info!(
            "run_jfa_repair: repaired SDF for object {} ({}×{}×{} voxels)",
            object_id, gw, gh, gd,
        );
        true
    }

    /// Smooth a dense flat SDF distance grid to remove Voronoi cell artifacts.
    ///
    /// Applies `iters` iterations of sign-preserving 6-neighbour box blur.
    /// Only allocated voxels contribute to (and receive) averages, so empty-brick
    /// fallback values don't bleed into the surface region.
    ///
    /// Sign preservation: each voxel keeps its original sign after blurring, so
    /// the zero-crossing (and therefore the geometry) does not move.
    pub(super) fn smooth_sdf_distances(
        distances:    &mut [f32],
        is_allocated: &[bool],
        gw: usize, gh: usize, gd: usize,
        voxel_size: f32,
        iters: u32,
    ) {
        let total = gw * gh * gd;
        let idx3 = |x: usize, y: usize, z: usize| -> usize { z * gh * gw + y * gw + x };
        let min_mag = voxel_size * 0.01;

        let mut tmp = vec![0.0f32; total];

        for _ in 0..iters {
            for gz in 0..gd {
                for gy in 0..gh {
                    for gx in 0..gw {
                        let i = idx3(gx, gy, gz);

                        if !is_allocated[i] {
                            tmp[i] = distances[i];
                            continue;
                        }

                        let d0    = distances[i];
                        let mut sum   = d0;
                        let mut count = 1.0f32;

                        macro_rules! add_neighbor {
                            ($nx:expr, $ny:expr, $nz:expr) => {{
                                let ni = idx3($nx, $ny, $nz);
                                if is_allocated[ni] {
                                    sum   += distances[ni];
                                    count += 1.0;
                                }
                            }};
                        }

                        if gx > 0       { add_neighbor!(gx - 1, gy,     gz    ); }
                        if gx + 1 < gw  { add_neighbor!(gx + 1, gy,     gz    ); }
                        if gy > 0       { add_neighbor!(gx,     gy - 1, gz    ); }
                        if gy + 1 < gh  { add_neighbor!(gx,     gy + 1, gz    ); }
                        if gz > 0       { add_neighbor!(gx,     gy,     gz - 1); }
                        if gz + 1 < gd  { add_neighbor!(gx,     gy,     gz + 1); }

                        // Preserve the sign of the centre voxel — zero-crossing stays put.
                        let avg = sum / count;
                        tmp[i] = if d0 < 0.0 { avg.min(-min_mag) } else { avg.max(min_mag) };
                    }
                }
            }
            distances.copy_from_slice(&tmp);
        }
    }

    /// Recompute SDF distance magnitudes via 3D EDT for a local brick neighbourhood.
    /// CPU-only — does NOT upload to GPU.
    ///
    /// Only processes bricks in [`stroke_bmin`, `stroke_bmax`] ± 1-brick border.
    /// This keeps the EDT grid tiny (stroke extent + normal-kernel padding) rather
    /// than iterating the entire object.
    ///
    /// Seeds:
    ///   - Voxels in `modified_slots` (freshly written by the boolean stamp with exact
    ///     sphere SDF values): use stored `|d|` as seed distance → sub-voxel accuracy,
    ///     no accumulated stale distances.
    ///   - All other zero-crossing voxels: binary `0.5h` seed (safe, deterministic).
    ///
    /// Signs are never changed — only magnitudes are recomputed.
    /// Returns the pool slot indices of all bricks in the local scope that were
    /// rewritten, for use in targeted GPU upload.
    pub(super) fn fix_sdfs_cpu(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        modified_slots: &[u32],
        stroke_bmin: glam::UVec3,
        stroke_bmax: glam::UVec3,
    ) -> Vec<u32> {
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick_map::INTERIOR_SLOT;

        // 1. Get handle + voxel_size from object.
        let (handle, voxel_size) = {
            let obj = match scene.objects.iter().find(|o| o.id == object_id) {
                Some(o) => o,
                None => return Vec::new(),
            };
            match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } =>
                    (*brick_map_handle, *voxel_size),
                _ => return Vec::new(),
            }
        };

        let h = voxel_size;
        let bd = BRICK_DIM;          // u32, for brick_index()
        let bdu = BRICK_DIM as usize; // usize, for index arithmetic

        // Build a HashSet for O(1) modified-brick lookup.
        let modified_set: std::collections::HashSet<u32> =
            modified_slots.iter().copied().collect();

        // 2. Compute local brick scope: stroke extent + 1-brick border on each side.
        //    The 1-brick border (= 8 voxels) covers the Catmull-Rom normal kernel
        //    (±2 voxels from surface) plus a small propagation margin.
        let dims = handle.dims;
        let s0x = stroke_bmin.x.saturating_sub(1);
        let s0y = stroke_bmin.y.saturating_sub(1);
        let s0z = stroke_bmin.z.saturating_sub(1);
        let s1x = (stroke_bmax.x + 1).min(dims.x.saturating_sub(1));
        let s1y = (stroke_bmax.y + 1).min(dims.y.saturating_sub(1));
        let s1z = (stroke_bmax.z + 1).min(dims.z.saturating_sub(1));

        let sbx = (s1x - s0x + 1) as usize;
        let sby = (s1y - s0y + 1) as usize;
        let sbz = (s1z - s0z + 1) as usize;
        let lw = sbx * bdu;
        let lh = sby * bdu;
        let ld = sbz * bdu;
        let total = lw * lh * ld;
        if total == 0 { return Vec::new(); }

        let lidx = |x: usize, y: usize, z: usize| -> usize { z * lh * lw + y * lw + x };

        // 3. Populate local grid from brick pool.
        let mut stored_sign  = vec![1i8;         total];
        let mut stored_mag   = vec![h * 4.0f32;  total];
        let mut is_allocated = vec![false;        total];
        let mut is_modified  = vec![false;        total];
        let mut mat_grid:    Vec<u16> = vec![0;   total];
        let mut blend_grid:  Vec<u8>  = vec![0;   total];
        let mut sec_id_grid: Vec<u8>  = vec![0;   total];
        let mut flags_grid:  Vec<u8>  = vec![0;   total];

        for bz in s0z..=s1z {
            for by in s0y..=s1y {
                for bx in s0x..=s1x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if s == INTERIOR_SLOT => {
                            // Interior slot: all voxels solid, large negative dist.
                            let base_lx = ((bx - s0x) as usize) * bdu;
                            let base_ly = ((by - s0y) as usize) * bdu;
                            let base_lz = ((bz - s0z) as usize) * bdu;
                            for lz in 0..bdu { for ly in 0..bdu { for lx in 0..bdu {
                                let i = lidx(base_lx + lx, base_ly + ly, base_lz + lz);
                                stored_sign[i] = -1;
                                stored_mag[i]  = h * 4.0;
                            }}}
                            continue;
                        }
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };
                    let in_modified = modified_set.contains(&slot);
                    let brick = self.cpu_brick_pool.get(slot);
                    let base_lx = ((bx - s0x) as usize) * bdu;
                    let base_ly = ((by - s0y) as usize) * bdu;
                    let base_lz = ((bz - s0z) as usize) * bdu;
                    for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                        let vi = brick_index(lx, ly, lz);
                        let vs = brick.voxels[vi];
                        let d  = vs.distance_f32();
                        let i  = lidx(
                            base_lx + lx as usize,
                            base_ly + ly as usize,
                            base_lz + lz as usize,
                        );
                        is_allocated[i] = true;
                        is_modified[i]  = in_modified;
                        stored_sign[i]  = if d < 0.0 { -1 } else { 1 };
                        stored_mag[i]   = d.abs();
                        mat_grid[i]     = vs.material_id();
                        blend_grid[i]   = vs.blend_weight();
                        sec_id_grid[i]  = vs.secondary_id();
                        flags_grid[i]   = vs.flags();
                    }}}
                }
            }
        }

        // 4. Build EDT seed grid.
        //
        //    For each allocated voxel adjacent to a sign change (a zero-crossing),
        //    we place a seed at 0.5h (half-voxel from the face — binary, derived
        //    purely from solid/empty topology, never from stored SDF magnitudes).
        //
        //    Only SOLID voxels adjacent to exterior/unallocated regions at the local
        //    grid boundary are seeded — exterior boundary voxels are left unseeded
        //    to avoid propagating false small distances through the exterior.
        let mut edt_grid = vec![f32::MAX; total];

        for gz in 0..ld {
            for gy in 0..lh {
                for gx in 0..lw {
                    let i = lidx(gx, gy, gz);
                    if !is_allocated[i] { continue; }
                    let s = stored_sign[i];

                    let nbrs: [(usize, usize, usize, bool); 6] = [
                        (gx.wrapping_sub(1), gy, gz, gx > 0),
                        (gx + 1,             gy, gz, gx + 1 < lw),
                        (gx, gy.wrapping_sub(1), gz, gy > 0),
                        (gx, gy + 1,             gz, gy + 1 < lh),
                        (gx, gy, gz.wrapping_sub(1), gz > 0),
                        (gx, gy, gz + 1,             gz + 1 < ld),
                    ];

                    for (nx, ny, nz, valid) in nbrs {
                        // Determine whether a zero-crossing exists on this face.
                        let nbr_opposite: bool;
                        if !valid {
                            // Local grid boundary = exterior.  Seed solid side only.
                            if s >= 0 { continue; }
                            nbr_opposite = true;
                        } else {
                            let ni = lidx(nx, ny, nz);
                            if !is_allocated[ni] {
                                // Use stored_sign to distinguish INTERIOR_SLOT (solid,
                                // stored_sign=-1) from EMPTY_SLOT (exterior, stored_sign=+1).
                                // Without this, INTERIOR_SLOT neighbors trigger false seeds
                                // deep inside the solid, corrupting EDT distances near baked
                                // interior bricks (e.g. from initial mark_interior_empties).
                                if stored_sign[ni] == s { continue; } // same side, no crossing
                                if s >= 0 { continue; }               // only seed solid side
                                nbr_opposite = true;
                            } else if stored_sign[ni] != s {
                                nbr_opposite = true;
                            } else {
                                continue; // Same sign — no zero-crossing.
                            }
                        }
                        if !nbr_opposite { continue; }

                        // Seed distance: 0.5h from face (binary topology only —
                        // never reads stored SDF magnitudes, which may be corrupted).
                        let half = h * 0.5;
                        let sq = half * half;
                        if sq < edt_grid[i] { edt_grid[i] = sq; }

                        // Seed the NEIGHBOUR too (if allocated in local grid).
                        if valid {
                            let ni = lidx(nx, ny, nz);
                            if is_allocated[ni] {
                                if sq < edt_grid[ni] { edt_grid[ni] = sq; }
                            }
                        }
                    }
                }
            }
        }

        // 5. Felzenszwalb 3D EDT (X → Y → Z separable passes).
        {
            let max_dim = lw.max(lh).max(ld);
            let mut col = vec![0.0f32; max_dim];
            let mut tmp = vec![0.0f32; max_dim];

            // X pass
            for gz in 0..ld {
                for gy in 0..lh {
                    for gx in 0..lw { col[gx] = edt_grid[lidx(gx, gy, gz)]; }
                    Self::edt_1d_pass(&col[..lw], h, &mut tmp[..lw]);
                    for gx in 0..lw { edt_grid[lidx(gx, gy, gz)] = tmp[gx]; }
                }
            }
            // Y pass
            for gz in 0..ld {
                for gx in 0..lw {
                    for gy in 0..lh { col[gy] = edt_grid[lidx(gx, gy, gz)]; }
                    Self::edt_1d_pass(&col[..lh], h, &mut tmp[..lh]);
                    for gy in 0..lh { edt_grid[lidx(gx, gy, gz)] = tmp[gy]; }
                }
            }
            // Z pass
            for gy in 0..lh {
                for gx in 0..lw {
                    for gz in 0..ld { col[gz] = edt_grid[lidx(gx, gy, gz)]; }
                    Self::edt_1d_pass(&col[..ld], h, &mut tmp[..ld]);
                    for gz in 0..ld { edt_grid[lidx(gx, gy, gz)] = tmp[gz]; }
                }
            }
        }

        // 6. Write corrected distances back to local bricks.
        let mut out_slots: Vec<u32> = Vec::new();
        for bz in s0z..=s1z {
            for by in s0y..=s1y {
                for bx in s0x..=s1x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };
                    let base_lx = ((bx - s0x) as usize) * bdu;
                    let base_ly = ((by - s0y) as usize) * bdu;
                    let base_lz = ((bz - s0z) as usize) * bdu;
                    let brick = self.cpu_brick_pool.get_mut(slot);
                    for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                        let i = lidx(
                            base_lx + lx as usize,
                            base_ly + ly as usize,
                            base_lz + lz as usize,
                        );
                        if !is_allocated[i] { continue; }
                        let vi = brick_index(lx, ly, lz);
                        let mag = if edt_grid[i] < f32::MAX { edt_grid[i].sqrt() } else { h * 4.0 };
                        brick.voxels[vi] = VoxelSample::new(
                            mag * stored_sign[i] as f32,
                            mat_grid[i], blend_grid[i], sec_id_grid[i], flags_grid[i],
                        );
                    }}}
                    out_slots.push(slot);
                }
            }
        }

        log::debug!("fix_sdfs_cpu: local EDT object={} scope=[{},{},{}]-[{},{},{}] {}×{}×{} vox {} slots",
            object_id, s0x, s0y, s0z, s1x, s1y, s1z, lw, lh, ld, out_slots.len());
        let _ = is_modified; // suppress unused warning
        out_slots
    }

    /// Felzenszwalb 1D EDT (lower envelope of parabolas).
    ///
    /// `col[i]` = 0.0 at seed positions, f32::MAX otherwise (first pass), or
    /// the accumulated squared distance from previous passes (second/third pass).
    /// `h` = voxel spacing.
    /// Writes squared Euclidean distances to nearest seed into `out`.
    pub(super) fn edt_1d_pass(col: &[f32], h: f32, out: &mut [f32]) {
        let n = col.len();
        for x in out.iter_mut() { *x = f32::MAX; }
        if n == 0 { return; }

        // v[k] = index of k-th parabola center.
        // z[k] = left boundary of k-th parabola's valid domain; z[v.len()] = +inf.
        // Invariant: z.len() == v.len() + 1.
        let mut v: Vec<usize> = Vec::with_capacity(n);
        let mut z: Vec<f32> = Vec::with_capacity(n + 1);

        for q in 0..n {
            if col[q] >= f32::MAX { continue; }
            let qf = q as f32 * h;

            loop {
                if v.is_empty() {
                    v.push(q);
                    z.push(f32::NEG_INFINITY);
                    z.push(f32::INFINITY);
                    break;
                }
                let r = *v.last().unwrap();
                let rf = r as f32 * h;
                // Intersection of parabola at q and at r:
                // (x−qf)² + col[q] = (x−rf)² + col[r]  →  solve for x.
                let s = ((col[q] + qf * qf) - (col[r] + rf * rf)) / (2.0 * (qf - rf));
                // z[z.len()-2] is the left boundary of parabola r's domain.
                if s <= z[z.len() - 2] {
                    // Parabola r is completely dominated by q — remove it.
                    v.pop();
                    z.pop(); // remove +inf
                    z.pop(); // remove left boundary of r
                    z.push(f32::INFINITY); // restore right boundary
                } else {
                    // Parabola q starts at s.
                    *z.last_mut().unwrap() = s;
                    v.push(q);
                    z.push(f32::INFINITY);
                    break;
                }
            }
        }

        if v.is_empty() { return; }

        let mut j = 0usize;
        for q in 0..n {
            let qf = q as f32 * h;
            while j + 1 < v.len() && z[j + 1] <= qf { j += 1; }
            let r = v[j];
            if col[r] < f32::MAX {
                let diff = qf - r as f32 * h;
                out[q] = diff * diff + col[r];
            }
        }
    }

    /// Separable 3D EDT via three 1D Felzenszwalb passes (X→Y→Z).
    ///
    /// `seeds[i]` = true if voxel i is a seed (source).
    /// Returns squared Euclidean distances to nearest seed, in world units.
    pub(super) fn edt_3d_squared(seeds: &[bool], gw: usize, gh: usize, gd: usize, h: f32) -> Vec<f32> {
        let idx3 = |x: usize, y: usize, z: usize| z * gh * gw + y * gw + x;
        let total = gw * gh * gd;

        // Initialize: 0 at seeds, MAX elsewhere.
        let mut grid: Vec<f32> = (0..total).map(|i| if seeds[i] { 0.0 } else { f32::MAX }).collect();

        let max_dim = gw.max(gh).max(gd);
        let mut col = vec![0.0f32; max_dim];
        let mut tmp = vec![0.0f32; max_dim];

        // Pass 1: X direction
        for z in 0..gd {
            for y in 0..gh {
                for x in 0..gw { col[x] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gw], h, &mut tmp[..gw]);
                for x in 0..gw { grid[idx3(x, y, z)] = tmp[x]; }
            }
        }

        // Pass 2: Y direction
        for z in 0..gd {
            for x in 0..gw {
                for y in 0..gh { col[y] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gh], h, &mut tmp[..gh]);
                for y in 0..gh { grid[idx3(x, y, z)] = tmp[y]; }
            }
        }

        // Pass 3: Z direction
        for y in 0..gh {
            for x in 0..gw {
                for z in 0..gd { col[z] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gd], h, &mut tmp[..gd]);
                for z in 0..gd { grid[idx3(x, y, z)] = tmp[z]; }
            }
        }

        grid
    }

    // sculpt_fmm_repair is in sdf_fmm.rs.
}
