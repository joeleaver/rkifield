//! Per-stroke FMM (Fast Marching Method) SDF repair for the editor engine.

use rkf_core::{Scene, SdfSource};
use super::EditorEngine;

impl EditorEngine {
    /// Per-stroke SDF repair via flat-array FMM with restricted write-back.
    ///
    /// Called after every sculpt stroke to correct the distance magnitudes introduced
    /// by CSG in the brush region.  Correct magnitudes are critical because the
    /// Catmull-Rom normal kernel uses a 4×4×4 voxel neighbourhood — wrong magnitudes
    /// produce banding and incorrect normals.
    ///
    /// # Algorithm
    ///
    /// 1. Define outer box  = brush_center ± (inner_half + 1-brick margin) in voxel coords.
    /// 2. Load the outer box into flat f32 arrays (post-CSG for inner, original for margin).
    /// 3. Detect zero-crossings, seed FMM at |d|.min(h).
    /// 4. Run heap-based FMM on the flat array — O(N log N) with O(1) index arithmetic
    ///    (≈50–100× faster than the HashMap-based full-object FMM).
    /// 5. Determine sign: seed BFS from all margin voxels (valid sign, untouched by CSG),
    ///    flood into the inner box stopping at sign changes.
    /// 6. Write back ONLY the inner voxels.  The margin is read-only — no discontinuity
    ///    is created at the boundary, so the next stroke's zero-crossing detection sees
    ///    clean SDF values in the surrounding region.
    /// Returns the pool slot indices of all bricks that were rewritten (for GPU upload).
    pub(super) fn sculpt_fmm_repair(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        center_local: glam::Vec3,
        inner_half_voxels: i32,
    ) -> Vec<u32> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        use std::collections::VecDeque;
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick_map::INTERIOR_SLOT;

        // 1. Look up object's voxel data.
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o, None => return Vec::new(),
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
        let gd_ = dims.z as i32 * bd;
        let h = voxel_size;

        // 2. Define outer box (inner + 1-brick margin), clamped to grid.
        let margin_v = bd;
        let outer_half = inner_half_voxels + margin_v;

        let cx = ((center_local.x - aabb_min.x) / h).round() as i32;
        let cy = ((center_local.y - aabb_min.y) / h).round() as i32;
        let cz = ((center_local.z - aabb_min.z) / h).round() as i32;

        let ox0 = (cx - outer_half).clamp(0, gw - 1) as usize;
        let oy0 = (cy - outer_half).clamp(0, gh - 1) as usize;
        let oz0 = (cz - outer_half).clamp(0, gd_ - 1) as usize;
        let ox1 = (cx + outer_half).clamp(0, gw - 1) as usize;
        let oy1 = (cy + outer_half).clamp(0, gh - 1) as usize;
        let oz1 = (cz + outer_half).clamp(0, gd_ - 1) as usize;

        let fw  = ox1 - ox0 + 1;
        let fh  = oy1 - oy0 + 1;
        let _fd = oz1 - oz0 + 1;
        let fw_fh = fw * fh;
        let fn_count = fw * fh * (oz1 - oz0 + 1);
        if fn_count == 0 { return Vec::new(); }

        // Flat index: (gx, gy, gz) → usize.  All coords in absolute voxel space.
        let fidx = |gx: usize, gy: usize, gz: usize| -> usize {
            (gz - oz0) * fw_fh + (gy - oy0) * fw + (gx - ox0)
        };
        // Is global voxel (gx,gy,gz) in the inner box?
        let is_inner = |gx: i32, gy: i32, gz: i32| -> bool {
            (gx - cx).abs() <= inner_half_voxels
                && (gy - cy).abs() <= inner_half_voxels
                && (gz - cz).abs() <= inner_half_voxels
        };

        // 3. Allocate flat buffers.
        let large_dist = h * gw.max(gh).max(gd_) as f32;
        let mut stored_d    = vec![large_dist; fn_count]; // post-CSG signed distance
        let mut is_alloc    = vec![false; fn_count];       // in brick pool?
        let mut fmm_d       = vec![f32::MAX; fn_count];   // unsigned FMM distance
        let mut fmm_settled = vec![false; fn_count];
        // For write-back: store (slot, brick_index) for allocated inner voxels.
        let mut write_slot = vec![u32::MAX; fn_count];
        let mut write_vi   = vec![0u32; fn_count];

        // 4. Load from brick pool.
        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let bd_us = BRICK_DIM as usize;
                    let bx = (gx / bd_us) as u32;
                    let by = (gy / bd_us) as u32;
                    let bz = (gz / bd_us) as u32;
                    let lx = (gx % bd_us) as u32;
                    let ly = (gy % bd_us) as u32;
                    let lz = (gz % bd_us) as u32;
                    let fi = fidx(gx, gy, gz);
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if slot == INTERIOR_SLOT {
                            // Mark INTERIOR_SLOT bricks as traversable interior voxels.
                            // Without this, FMM cannot propagate through interior gaps:
                            // allocated bricks adjacent to INTERIOR_SLOT remain unsettled
                            // and fall back to ±large_dist, creating fake zero-crossings
                            // that compound with each sculpt stroke.
                            stored_d[fi] = -h * 4.0; // interior, large magnitude for sign BFS
                            is_alloc[fi] = true;      // FMM will propagate through here
                            // fmm_d stays MAX — FMM computes the correct distance from neighbors
                            // write_slot stays MAX  — no pool slot, skip write-back
                        } else if !Self::is_unallocated(slot) {
                            let vi = brick_index(lx, ly, lz);
                            let d = self.cpu_brick_pool.get(slot).voxels[vi].distance_f32();
                            stored_d[fi] = d;
                            is_alloc[fi] = true;
                            write_slot[fi] = slot;
                            write_vi[fi]   = vi as u32;
                        }
                    }
                }
            }
        }

        // 5. Detect zero-crossings, seed FMM.
        let face_dx: [i32; 6] = [-1, 1,  0, 0,  0, 0];
        let face_dy: [i32; 6] = [ 0, 0, -1, 1,  0, 0];
        let face_dz: [i32; 6] = [ 0, 0,  0, 0, -1, 1];

        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let fi = fidx(gx, gy, gz);
                    if !is_alloc[fi] { continue; }
                    let d = stored_d[fi];
                    if !d.is_finite() { continue; }
                    for dir in 0..6usize {
                        let nx = gx as i32 + face_dx[dir];
                        let ny = gy as i32 + face_dy[dir];
                        let nz = gz as i32 + face_dz[dir];
                        if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                        if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                        let nfi = fidx(nx as usize, ny as usize, nz as usize);
                        if !is_alloc[nfi] { continue; }
                        let nd = stored_d[nfi];
                        if !nd.is_finite() || d * nd >= 0.0 { continue; }
                        let sd  = d.abs().min(h);
                        let snd = nd.abs().min(h);
                        if sd  < fmm_d[fi]  { fmm_d[fi]  = sd;  }
                        if snd < fmm_d[nfi] { fmm_d[nfi] = snd; }
                    }
                }
            }
        }

        // 6. Heap-based FMM on flat array — O(1) neighbour lookup via index arithmetic.
        let eikonal = |a: f32, b: f32, c: f32| -> f32 {
            if a == f32::MAX { return f32::MAX; }
            let u1 = a + h;
            if b == f32::MAX || u1 <= b { return u1; }
            let disc2 = 2.0 * h * h - (b - a) * (b - a);
            let u2 = if disc2 >= 0.0 { (a + b + disc2.sqrt()) * 0.5 } else { a + h };
            if c == f32::MAX || u2 <= c { return u2; }
            let s = a + b + c;
            let disc3 = s * s - 3.0 * (a * a + b * b + c * c - h * h);
            if disc3 >= 0.0 { (s + disc3.sqrt()) / 3.0 } else { u2 }
        };

        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for fi in 0..fn_count {
            if fmm_d[fi] < f32::MAX {
                heap.push(Reverse((fmm_d[fi].to_bits(), fi as u32)));
            }
        }

        while let Some(Reverse((bits, fi_u32))) = heap.pop() {
            let fi = fi_u32 as usize;
            if fmm_settled[fi] { continue; }
            let dist = f32::from_bits(bits);
            if dist > fmm_d[fi] + h * 0.01 { continue; }
            fmm_settled[fi] = true;

            // Recover absolute coords from flat index.
            let gx = ox0 + (fi % fw);
            let gy = oy0 + (fi / fw) % fh;
            let gz = oz0 + (fi / fw_fh);

            for dir in 0..6usize {
                let nx = gx as i32 + face_dx[dir];
                let ny = gy as i32 + face_dy[dir];
                let nz = gz as i32 + face_dz[dir];
                if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                let nfi = fidx(nx as usize, ny as usize, nz as usize);
                if !is_alloc[nfi] || fmm_settled[nfi] { continue; }

                let nfx = nx as usize; let nfy = ny as usize; let nfz = nz as usize;
                // Per-axis minimum settled distance for Eikonal update.
                let mut da = [f32::MAX; 3];
                macro_rules! check_nbr {
                    ($i:expr, $ax:expr, $ay:expr, $az:expr) => {
                        let ax = nfx as i32 + $ax; let ay = nfy as i32 + $ay; let az = nfz as i32 + $az;
                        if ax >= ox0 as i32 && ay >= oy0 as i32 && az >= oz0 as i32
                            && ax <= ox1 as i32 && ay <= oy1 as i32 && az <= oz1 as i32 {
                            let afi = fidx(ax as usize, ay as usize, az as usize);
                            if fmm_settled[afi] { da[$i] = da[$i].min(fmm_d[afi]); }
                        }
                    }
                }
                check_nbr!(0, -1, 0, 0); check_nbr!(0, 1, 0, 0);
                check_nbr!(1, 0, -1, 0); check_nbr!(1, 0, 1, 0);
                check_nbr!(2, 0, 0, -1); check_nbr!(2, 0, 0, 1);
                da.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let new_d = eikonal(da[0], da[1], da[2]);
                if new_d < fmm_d[nfi] {
                    fmm_d[nfi] = new_d;
                    heap.push(Reverse((new_d.to_bits(), nfi as u32)));
                }
            }
        }
        // Unsettled allocated voxels: fall back to large_dist.
        for fi in 0..fn_count {
            if is_alloc[fi] && !fmm_settled[fi] { fmm_d[fi] = large_dist; }
        }

        // 7. Sign BFS: determine exterior/interior using globally-correct seeding.
        //
        // Two seed sources:
        //   (a) All unallocated voxels: classified by classify_exterior_bricks (grid-boundary
        //       flood-fill), so enclosed EMPTY_SLOT bricks are treated as interior.
        //   (b) All allocated MARGIN voxels (outer box minus inner box): use stored_d sign.
        //       These were NOT modified by CSG this stroke, so their signs are trustworthy
        //       anchors. This ensures seeds exist even when the repair box is fully surrounded
        //       by allocated bricks (deep inside a large sculpted object).
        let ext_bricks = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, &handle);

        let mut ext_sign = vec![0i8; fn_count]; // 1=ext, -1=int, 0=unknown
        let mut bfs: VecDeque<usize> = VecDeque::new();

        let bd_us = BRICK_DIM as usize;
        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let fi = fidx(gx, gy, gz);
                    let in_margin = !is_inner(gx as i32, gy as i32, gz as i32);

                    if !is_alloc[fi] {
                        // Unallocated voxel: use grid-topology classification.
                        let bbx = (gx / bd_us) as u32;
                        let bby = (gy / bd_us) as u32;
                        let bbz = (gz / bd_us) as u32;
                        let sign = if ext_bricks.contains(&(bbx, bby, bbz)) {
                            1i8   // truly exterior (grid-boundary-connected)
                        } else {
                            -1i8  // enclosed empty or INTERIOR_SLOT → interior
                        };
                        ext_sign[fi] = sign;
                        bfs.push_back(fi);
                    } else if in_margin {
                        // Allocated margin voxel: trust stored_d sign.
                        // Margin is outside the inner (CSG-modified) region.
                        let sign = if stored_d[fi] >= 0.0 { 1i8 } else { -1i8 };
                        ext_sign[fi] = sign;
                        bfs.push_back(fi);
                    }
                }
            }
        }

        while let Some(fi) = bfs.pop_front() {
            let sign = ext_sign[fi];
            let gx = ox0 + (fi % fw);
            let gy = oy0 + (fi / fw) % fh;
            let gz = oz0 + (fi / fw_fh);
            for dir in 0..6usize {
                let nx = gx as i32 + face_dx[dir];
                let ny = gy as i32 + face_dy[dir];
                let nz = gz as i32 + face_dz[dir];
                if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                let nfi = fidx(nx as usize, ny as usize, nz as usize);
                if ext_sign[nfi] != 0 { continue; }
                if !is_alloc[nfi] {
                    // Already seeded in the seed pass; skip.
                    continue;
                }
                let d_nc = stored_d[nfi];
                // Stop at stored zero-crossings: sign disagrees with stored_d sign.
                if (sign > 0) != (d_nc >= 0.0) { continue; }
                ext_sign[nfi] = sign;
                bfs.push_back(nfi);
            }
        }
        // Fallback: voxels unreachable from all BFS seeds.
        // Use stored_d sign directly.  Unreachable voxels are one of:
        //   - Enclosed exterior voids (cavities from subtract, enclosed gaps from merge):
        //     stored_d > 0 (fresh from CSG or stale) → correctly exterior (+1).
        //   - Deep interior with no nearby surface: stored_d < 0 → correctly interior (-1).
        // Note: "enclosed gap from partial merge" is topologically EXTERIOR (empty space
        // enclosed by solid), so stored_d > 0 → +1 is correct.
        for fi in 0..fn_count {
            if ext_sign[fi] == 0 && is_alloc[fi] {
                ext_sign[fi] = if stored_d[fi] >= 0.0 { 1i8 } else { -1i8 };
            }
        }

        // 8. Write back ONLY the inner voxels.
        //
        // The margin is read-only: it provides anchor sign/distance values for the
        // FMM propagation, but must NOT be overwritten.  Writing the margin would
        // corrupt parts of the object outside the brush stroke region.
        let mut modified_set = std::collections::HashSet::new();
        for fi in 0..fn_count {
            let slot = write_slot[fi];
            if slot == u32::MAX { continue; }

            // Recover absolute coords to check inner/margin membership.
            let gx = ox0 + (fi % fw);
            let gy = oy0 + (fi / fw) % fh;
            let gz = oz0 + (fi / fw_fh);
            if !is_inner(gx as i32, gy as i32, gz as i32) { continue; }

            let vi = write_vi[fi] as usize;
            let orig = self.cpu_brick_pool.get(slot).voxels[vi];
            let sign = ext_sign[fi] as f32;
            let mag = fmm_d[fi];
            self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                sign * mag,
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
