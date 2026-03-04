//! Local CPU distance recomputation for sculpt strokes.
//!
//! After an occupancy flip (geometry change), recomputes correct distance
//! magnitudes from the zero-crossings using a local CPU sweep.
//! Geometry drives distances: signs are ground truth, magnitudes are derived.

use rkf_core::{Scene, SdfSource};
use rkf_core::constants::BRICK_DIM;
use rkf_core::brick::brick_index;
use rkf_core::voxel::VoxelSample;
use rkf_core::brick_map::INTERIOR_SLOT;
use super::EditorEngine;

impl EditorEngine {
    /// Recompute SDF distances from geometry after an occupancy change.
    ///
    /// Finds zero-crossings (sign changes between neighbors), computes
    /// exact distances at those boundaries, then propagates outward a
    /// few layers. Purely local — only touches voxels near the surface.
    pub(super) fn local_sdf_recompute(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        center_local: glam::Vec3,
        half_voxels: i32,
    ) -> Vec<u32> {
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return Vec::new(),
        };
        let (handle, voxel_size, aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } =>
                (brick_map_handle.clone(), *voxel_size, aabb.min),
            _ => return Vec::new(),
        };

        let dims = handle.dims;
        let bd = BRICK_DIM as i32;
        let gw = dims.x as i32 * bd;
        let gh = dims.y as i32 * bd;
        let gd = dims.z as i32 * bd;
        let h = voxel_size;

        // Region: half_voxels + 1-brick margin for boundary data.
        let margin = bd;
        let outer = half_voxels + margin;

        let cx = ((center_local.x - aabb_min.x) / h).round() as i32;
        let cy = ((center_local.y - aabb_min.y) / h).round() as i32;
        let cz = ((center_local.z - aabb_min.z) / h).round() as i32;

        let x0 = (cx - outer).clamp(0, gw - 1) as usize;
        let y0 = (cy - outer).clamp(0, gh - 1) as usize;
        let z0 = (cz - outer).clamp(0, gd - 1) as usize;
        let x1 = (cx + outer).clamp(0, gw - 1) as usize;
        let y1 = (cy + outer).clamp(0, gh - 1) as usize;
        let z1 = (cz + outer).clamp(0, gd - 1) as usize;

        let fw = x1 - x0 + 1;
        let fh = y1 - y0 + 1;
        let fd = z1 - z0 + 1;
        let n = fw * fh * fd;
        if n == 0 { return Vec::new(); }

        let fw_fh = fw * fh;
        let fidx = |gx: usize, gy: usize, gz: usize| -> usize {
            (gz - z0) * fw_fh + (gy - y0) * fw + (gx - x0)
        };
        let is_inner = |gx: i32, gy: i32, gz: i32| -> bool {
            (gx - cx).abs() <= half_voxels
                && (gy - cy).abs() <= half_voxels
                && (gz - cz).abs() <= half_voxels
        };

        // 1. Load from brick pool into dense grid.
        let mut dist = vec![h * 8.0_f32; n];
        let mut write_slot = vec![u32::MAX; n];
        let mut write_vi = vec![0u32; n];

        let bd_us = BRICK_DIM as usize;
        for gz in z0..=z1 {
            for gy in y0..=y1 {
                for gx in x0..=x1 {
                    let bx = (gx / bd_us) as u32;
                    let by = (gy / bd_us) as u32;
                    let bz = (gz / bd_us) as u32;
                    let lx = (gx % bd_us) as u32;
                    let ly = (gy % bd_us) as u32;
                    let lz = (gz % bd_us) as u32;
                    let fi = fidx(gx, gy, gz);

                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if slot == INTERIOR_SLOT {
                            dist[fi] = -h * 8.0;
                        } else if !Self::is_unallocated(slot) {
                            let vi = brick_index(lx, ly, lz);
                            let d = self.cpu_brick_pool.get(slot).voxels[vi].distance_f32();
                            dist[fi] = d;
                            write_slot[fi] = slot;
                            write_vi[fi] = vi as u32;
                        }
                    }
                }
            }
        }

        // 2. Find sign-change seeds: compute exact distance to zero-crossing.
        let mut corrected = vec![f32::MAX; n];
        let mut is_seed = vec![false; n];

        // Neighbor offsets in the flat grid.
        let offsets: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        for gz in z0..=z1 {
            for gy in y0..=y1 {
                for gx in x0..=x1 {
                    let fi = fidx(gx, gy, gz);
                    let d_this = dist[fi];
                    if d_this.abs() >= h * 7.0 { continue; } // sentinel

                    let sign_this = d_this >= 0.0;
                    let mut min_dist = f32::MAX;

                    for &(dx, dy, dz) in &offsets {
                        let nx = gx as i32 + dx;
                        let ny = gy as i32 + dy;
                        let nz = gz as i32 + dz;
                        if nx < x0 as i32 || nx > x1 as i32 { continue; }
                        if ny < y0 as i32 || ny > y1 as i32 { continue; }
                        if nz < z0 as i32 || nz > z1 as i32 { continue; }

                        let ni = fidx(nx as usize, ny as usize, nz as usize);
                        let d_nbr = dist[ni];
                        if d_nbr.abs() >= h * 7.0 { continue; } // sentinel

                        let sign_nbr = d_nbr >= 0.0;
                        if sign_this == sign_nbr { continue; }

                        // Sign change: zero-crossing between this and neighbor.
                        let abs_this = d_this.abs();
                        let abs_nbr = d_nbr.abs();
                        let sum = abs_this + abs_nbr;
                        let t = if sum > 1e-10 { abs_this / sum } else { 0.5 };
                        let seed_dist = t * h;
                        min_dist = min_dist.min(seed_dist);
                    }

                    if min_dist < f32::MAX {
                        let sign = if d_this < 0.0 { -1.0 } else { 1.0 };
                        corrected[fi] = sign * min_dist;
                        is_seed[fi] = true;
                    }
                }
            }
        }

        // 3. Propagate outward from seeds, 4 layers.
        for _iter in 0..4 {
            for gz in z0..=z1 {
                for gy in y0..=y1 {
                    for gx in x0..=x1 {
                        let fi = fidx(gx, gy, gz);
                        if is_seed[fi] { continue; }
                        let d_this = dist[fi];
                        if d_this.abs() >= h * 7.0 { continue; }

                        let sign = if d_this < 0.0 { -1.0 } else { 1.0 };
                        let mut best = corrected[fi].abs();

                        for &(dx, dy, dz) in &offsets {
                            let nx = gx as i32 + dx;
                            let ny = gy as i32 + dy;
                            let nz = gz as i32 + dz;
                            if nx < x0 as i32 || nx > x1 as i32 { continue; }
                            if ny < y0 as i32 || ny > y1 as i32 { continue; }
                            if nz < z0 as i32 || nz > z1 as i32 { continue; }

                            let ni = fidx(nx as usize, ny as usize, nz as usize);
                            if corrected[ni] == f32::MAX { continue; }

                            let candidate = corrected[ni].abs() + h;
                            if candidate < best {
                                best = candidate;
                            }
                        }

                        if best < f32::MAX && best < corrected[fi].abs() {
                            corrected[fi] = sign * best;
                        }
                    }
                }
            }
        }

        // 4. Write back corrected distances (inner region, near-surface only).
        let band = h * 5.0;
        let mut modified_set = std::collections::HashSet::new();
        for fi in 0..n {
            if corrected[fi] == f32::MAX { continue; }
            let slot = write_slot[fi];
            if slot == u32::MAX { continue; }

            let gx = x0 + (fi % fw);
            let gy = y0 + (fi / fw) % fh;
            let gz = z0 + (fi / fw_fh);
            if !is_inner(gx as i32, gy as i32, gz as i32) { continue; }

            let new_d = corrected[fi];
            if new_d.abs() >= band { continue; }

            let vi = write_vi[fi] as usize;
            let orig = self.cpu_brick_pool.get(slot).voxels[vi];

            self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                new_d,
                orig.material_id(),
                orig.blend_weight(),
                orig.secondary_id(),
                orig.flags(),
            );
            modified_set.insert(slot);
        }

        modified_set.into_iter().collect()
    }
}
