//! Geodesic surface flood fill over voxel brick geometry.
//!
//! Given a seed point on an object's surface, expands outward along
//! face-adjacent (6-connected) surface voxels using Dijkstra's algorithm,
//! accumulating geodesic distance. Returns all reachable surface voxels
//! within a maximum distance.
//!
//! This is a reusable spatial query — callers decide what to do with the
//! visited set (paint, sculpt falloff, selection highlight, etc.).

use std::collections::{BinaryHeap, HashMap};
use std::cmp::{Ordering, Reverse};

use glam::{UVec3, Vec3};

use crate::brick_geometry::{self, BrickGeometry};
use crate::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};
use crate::brick_pool::Pool;
use crate::constants::BRICK_DIM;

/// A visited surface voxel with its geodesic distance from the seed.
#[derive(Debug, Clone, Copy)]
pub struct FloodEntry {
    /// Brick coordinate in the object's brick map.
    pub brick_coord: UVec3,
    /// Pool slot for the geometry brick.
    pub geo_slot: u32,
    /// Index of this voxel in the brick's `surface_voxels` vec.
    pub surface_index: usize,
    /// Voxel index within the 8×8×8 brick (0..511).
    pub voxel_index: u16,
    /// Geodesic distance from the seed point along the surface (world-space units).
    pub geodesic_distance: f32,
}

/// Configuration for a surface flood fill.
pub struct FloodFillParams {
    /// Maximum geodesic distance from the seed (world-space units).
    pub max_distance: f32,
    /// Per-axis cost of one voxel step (typically `voxel_size * object.scale`).
    /// Accounts for non-uniform object scale.
    pub step_cost: Vec3,
}

/// Internal key for the visited set: (geo_slot, voxel_index_within_brick).
type VoxelKey = (u32, u16);

/// f32 wrapper with total ordering (for BinaryHeap). NaN sorts last.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// 26-connected neighbor offsets (dx, dy, dz) with step cost multipliers.
/// Face neighbors: cost = 1.0, edge neighbors: cost = sqrt(2), corner: cost = sqrt(3).
const NEIGHBORS_26: [(i8, i8, i8, f32); 26] = [
    // 6 face neighbors (1 axis changes)
    (-1,  0,  0, 1.0), ( 1,  0,  0, 1.0),
    ( 0, -1,  0, 1.0), ( 0,  1,  0, 1.0),
    ( 0,  0, -1, 1.0), ( 0,  0,  1, 1.0),
    // 12 edge neighbors (2 axes change)
    (-1, -1,  0, 1.414), (-1,  1,  0, 1.414), ( 1, -1,  0, 1.414), ( 1,  1,  0, 1.414),
    (-1,  0, -1, 1.414), (-1,  0,  1, 1.414), ( 1,  0, -1, 1.414), ( 1,  0,  1, 1.414),
    ( 0, -1, -1, 1.414), ( 0, -1,  1, 1.414), ( 0,  1, -1, 1.414), ( 0,  1,  1, 1.414),
    // 8 corner neighbors (3 axes change)
    (-1, -1, -1, 1.732), (-1, -1,  1, 1.732), (-1,  1, -1, 1.732), (-1,  1,  1, 1.732),
    ( 1, -1, -1, 1.732), ( 1, -1,  1, 1.732), ( 1,  1, -1, 1.732), ( 1,  1,  1, 1.732),
];

/// Per-brick lookup: voxel_index → index into surface_voxels vec, or NONE.
/// Built lazily as the BFS expands into new bricks.
const SV_NONE: u16 = u16::MAX;

fn build_surface_lookup(geo: &BrickGeometry) -> [u16; 512] {
    let mut lookup = [SV_NONE; 512];
    for (i, sv) in geo.surface_voxels.iter().enumerate() {
        lookup[sv.index() as usize] = i as u16;
    }
    lookup
}

/// Compute the grid origin for an object's brick map (local space, centered).
pub fn grid_origin(brick_map: &BrickMap, voxel_size: f32) -> Vec3 {
    let brick_size = voxel_size * BRICK_DIM as f32;
    -Vec3::new(
        brick_map.dims.x as f32 * brick_size * 0.5,
        brick_map.dims.y as f32 * brick_size * 0.5,
        brick_map.dims.z as f32 * brick_size * 0.5,
    )
}

/// Compute the local-space position of a voxel center.
pub fn voxel_local_pos(
    brick_coord: UVec3,
    vx: u8, vy: u8, vz: u8,
    origin: Vec3,
    voxel_size: f32,
) -> Vec3 {
    let brick_size = voxel_size * BRICK_DIM as f32;
    let half = voxel_size * 0.5;
    origin + Vec3::new(
        brick_coord.x as f32 * brick_size + vx as f32 * voxel_size + half,
        brick_coord.y as f32 * brick_size + vy as f32 * voxel_size + half,
        brick_coord.z as f32 * brick_size + vz as f32 * voxel_size + half,
    )
}

/// Find the nearest surface voxel to `seed_local` in the brick map.
///
/// Searches the target brick and its 26 neighbors. Returns
/// `(geo_slot, brick_coord, surface_voxel_index_in_vec, voxel_index, distance)`.
fn find_seed_voxel(
    seed_local: Vec3,
    brick_map: &BrickMap,
    geo_pool: &Pool<BrickGeometry>,
    voxel_size: f32,
) -> Option<(u32, UVec3, usize, u16, f32)> {
    let origin = grid_origin(brick_map, voxel_size);
    let brick_size = voxel_size * BRICK_DIM as f32;

    // Target brick coordinate.
    let raw = (seed_local - origin) / brick_size;
    let center_bx = raw.x.floor() as i32;
    let center_by = raw.y.floor() as i32;
    let center_bz = raw.z.floor() as i32;

    let mut best: Option<(u32, UVec3, usize, u16, f32)> = None;

    // Search target brick + 26 neighbors.
    for dz in -1..=1_i32 {
        for dy in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let bx = center_bx + dx;
                let by = center_by + dy;
                let bz = center_bz + dz;
                if bx < 0 || by < 0 || bz < 0 {
                    continue;
                }
                let bx = bx as u32;
                let by = by as u32;
                let bz = bz as u32;
                if bx >= brick_map.dims.x || by >= brick_map.dims.y || bz >= brick_map.dims.z {
                    continue;
                }
                let slot = brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                    continue;
                }
                let geo = geo_pool.get(slot);
                let bc = UVec3::new(bx, by, bz);
                for (si, sv) in geo.surface_voxels.iter().enumerate() {
                    let (vx, vy, vz) = brick_geometry::index_to_xyz(sv.index());
                    let pos = voxel_local_pos(bc, vx, vy, vz, origin, voxel_size);
                    let dist = (pos - seed_local).length_squared();
                    if best.is_none() || dist < best.unwrap().4 {
                        best = Some((slot, bc, si, sv.index(), dist));
                    }
                }
            }
        }
    }

    best.map(|(slot, bc, si, vi, d)| (slot, bc, si, vi, d.sqrt()))
}

/// Perform a geodesic surface flood fill from a seed point.
///
/// Expands along face-adjacent surface voxels using Dijkstra's algorithm.
/// Returns all reachable surface voxels within `params.max_distance`,
/// sorted by ascending geodesic distance.
pub fn surface_flood_fill(
    seed_local: Vec3,
    brick_map: &BrickMap,
    geo_pool: &Pool<BrickGeometry>,
    voxel_size: f32,
    params: &FloodFillParams,
) -> Vec<FloodEntry> {
    // Find the nearest surface voxel to the seed.
    let (seed_slot, seed_bc, _seed_si, seed_vi, _) =
        match find_seed_voxel(seed_local, brick_map, geo_pool, voxel_size) {
            Some(s) => s,
            None => return Vec::new(),
        };

    // Lazy per-brick surface lookup cache.
    let mut lookup_cache: HashMap<u32, [u16; 512]> = HashMap::new();

    // Visited set: VoxelKey → best known geodesic distance.
    let mut visited: HashMap<VoxelKey, f32> = HashMap::new();
    // Also track brick_coord per geo_slot for output.
    let mut slot_to_brick: HashMap<u32, UVec3> = HashMap::new();

    // Min-heap: (distance, geo_slot, voxel_index).
    // Use u32 bits for f32 to make it Ord-compatible.
    let mut heap: BinaryHeap<Reverse<(OrdF32, u32, u16)>> = BinaryHeap::new();

    // Seed.
    let seed_key: VoxelKey = (seed_slot, seed_vi);
    visited.insert(seed_key, 0.0);
    slot_to_brick.insert(seed_slot, seed_bc);
    heap.push(Reverse((OrdF32(0.0), seed_slot, seed_vi)));

    // Pre-build seed brick lookup.
    lookup_cache.insert(seed_slot, build_surface_lookup(geo_pool.get(seed_slot)));

    while let Some(Reverse((dist_of, slot, vi))) = heap.pop() {
        let dist = dist_of.0;
        let key: VoxelKey = (slot, vi);

        // Skip if we've already found a shorter path.
        if let Some(&best) = visited.get(&key) {
            if dist > best + 1e-6 {
                continue;
            }
        }

        let (vx, vy, vz) = brick_geometry::index_to_xyz(vi);
        let brick_coord = *slot_to_brick.get(&slot).unwrap();

        // Expand to 26-connected neighbors (face + edge + corner).
        for &(dx, dy, dz, dist_mult) in &NEIGHBORS_26 {
            let nx = vx as i8 + dx;
            let ny = vy as i8 + dy;
            let nz = vz as i8 + dz;

            // Determine target brick and voxel coordinates.
            let (nslot, nbrick_coord, nvx, nvy, nvz);
            if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 && nz >= 0 && nz < 8 {
                // Same brick.
                nslot = slot;
                nbrick_coord = brick_coord;
                nvx = nx as u8;
                nvy = ny as u8;
                nvz = nz as u8;
            } else {
                // Cross-brick boundary.
                let mut nbx = brick_coord.x as i32;
                let mut nby = brick_coord.y as i32;
                let mut nbz = brick_coord.z as i32;
                let mut cvx = nx;
                let mut cvy = ny;
                let mut cvz = nz;
                if cvx < 0 { nbx -= 1; cvx = 7; }
                if cvx >= 8 { nbx += 1; cvx = 0; }
                if cvy < 0 { nby -= 1; cvy = 7; }
                if cvy >= 8 { nby += 1; cvy = 0; }
                if cvz < 0 { nbz -= 1; cvz = 7; }
                if cvz >= 8 { nbz += 1; cvz = 0; }
                if nbx < 0 || nby < 0 || nbz < 0 {
                    continue;
                }
                let ubx = nbx as u32;
                let uby = nby as u32;
                let ubz = nbz as u32;
                if ubx >= brick_map.dims.x || uby >= brick_map.dims.y || ubz >= brick_map.dims.z {
                    continue;
                }
                let s = brick_map.get(ubx, uby, ubz).unwrap_or(EMPTY_SLOT);
                if s == EMPTY_SLOT || s == INTERIOR_SLOT {
                    continue;
                }
                nslot = s;
                nbrick_coord = UVec3::new(ubx, uby, ubz);
                nvx = cvx as u8;
                nvy = cvy as u8;
                nvz = cvz as u8;
            }

            // Check if neighbor is a surface voxel.
            if !lookup_cache.contains_key(&nslot) {
                lookup_cache.insert(nslot, build_surface_lookup(geo_pool.get(nslot)));
            }
            let lookup = lookup_cache.get(&nslot).unwrap();
            let nvi = brick_geometry::voxel_index(nvx, nvy, nvz);
            if lookup[nvi as usize] == SV_NONE {
                continue;
            }

            // Compute step cost: average of the axis costs that changed,
            // scaled by the Euclidean distance multiplier (1, sqrt2, sqrt3).
            let avg_axis_cost = {
                let mut sum = 0.0_f32;
                let mut count = 0;
                if dx != 0 { sum += params.step_cost.x; count += 1; }
                if dy != 0 { sum += params.step_cost.y; count += 1; }
                if dz != 0 { sum += params.step_cost.z; count += 1; }
                sum / count as f32
            };
            let new_dist = dist + avg_axis_cost * dist_mult;

            if new_dist > params.max_distance {
                continue;
            }

            let nkey: VoxelKey = (nslot, nvi);
            let is_better = match visited.get(&nkey) {
                Some(&best) => new_dist < best - 1e-6,
                None => true,
            };

            if is_better {
                visited.insert(nkey, new_dist);
                slot_to_brick.entry(nslot).or_insert(nbrick_coord);
                heap.push(Reverse((OrdF32(new_dist), nslot, nvi)));
            }
        }
    }

    // Collect results.
    let mut results: Vec<FloodEntry> = visited.iter().map(|(&(slot, vi), &dist)| {
        let bc = *slot_to_brick.get(&slot).unwrap();
        let lookup = lookup_cache.get(&slot).unwrap();
        let si = lookup[vi as usize] as usize;
        FloodEntry {
            brick_coord: bc,
            geo_slot: slot,
            surface_index: si,
            voxel_index: vi,
            geodesic_distance: dist,
        }
    }).collect();

    results.sort_by(|a, b| a.geodesic_distance.partial_cmp(&b.geodesic_distance).unwrap());
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick_geometry::{BrickGeometry, SurfaceVoxel, voxel_index};
    use crate::brick_map::BrickMap;
    use crate::brick_pool::Pool;

    /// Helper: create a BrickGeometry with surface voxels at specified positions.
    fn make_geo(positions: &[(u8, u8, u8)], material_id: u8) -> BrickGeometry {
        let mut geo = BrickGeometry::new();
        for &(x, y, z) in positions {
            let idx = voxel_index(x, y, z);
            // Set occupancy bit.
            let word = idx as usize / 64;
            let bit = idx as usize % 64;
            geo.occupancy[word] |= 1u64 << bit;
            geo.surface_voxels.push(SurfaceVoxel::new(idx, material_id));
        }
        geo
    }

    /// Helper: allocate a brick in the pool and assign to brick map.
    fn place_brick(
        pool: &mut Pool<BrickGeometry>,
        map: &mut BrickMap,
        bx: u32, by: u32, bz: u32,
        geo: BrickGeometry,
    ) -> u32 {
        let slot = pool.allocate().expect("pool full");
        *pool.get_mut(slot) = geo;
        map.set(bx, by, bz, slot);
        slot
    }

    #[test]
    fn test_seed_finding_exact() {
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(2, 2, 2));
        let geo = make_geo(&[(4, 4, 4)], 1);
        let slot = place_brick(&mut pool, &mut map, 1, 1, 1, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let expected_pos = voxel_local_pos(UVec3::new(1, 1, 1), 4, 4, 4, origin, vs);

        let result = find_seed_voxel(expected_pos, &map, &pool, vs);
        assert!(result.is_some());
        let (s, bc, _si, vi, dist) = result.unwrap();
        assert_eq!(s, slot);
        assert_eq!(bc, UVec3::new(1, 1, 1));
        assert_eq!(vi, voxel_index(4, 4, 4));
        assert!(dist < 0.001, "seed should be at exact voxel center, dist={dist}");
    }

    #[test]
    fn test_single_brick_flood() {
        // Line of 4 surface voxels along X axis in one brick.
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(1, 1, 1));
        let geo = make_geo(&[(2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4)], 1);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 2, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::splat(vs),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        assert_eq!(results.len(), 4, "should visit all 4 connected surface voxels");
        assert!(results[0].geodesic_distance < 0.001, "seed should be at distance 0");
        // Second voxel should be 1 step away.
        assert!((results[1].geodesic_distance - vs).abs() < 0.01);
    }

    #[test]
    fn test_cross_brick_flood() {
        // Two adjacent bricks along X. Surface voxel at (7,4,4) in brick (0,0,0)
        // and (0,4,4) in brick (1,0,0) — they're face neighbors.
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(2, 1, 1));
        let geo0 = make_geo(&[(7, 4, 4)], 1);
        let geo1 = make_geo(&[(0, 4, 4)], 2);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo0);
        place_brick(&mut pool, &mut map, 1, 0, 0, geo1);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 7, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::splat(vs),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        assert_eq!(results.len(), 2, "should cross brick boundary");
        assert_eq!(results[1].brick_coord, UVec3::new(1, 0, 0));
    }

    #[test]
    fn test_max_distance_cutoff() {
        // 5 voxels in a line. Max distance = 2.5 steps → should visit 3.
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(1, 1, 1));
        let geo = make_geo(&[(2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4), (6, 4, 4)], 1);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 2, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: vs * 2.5,
            step_cost: Vec3::splat(vs),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        assert_eq!(results.len(), 3, "max_distance should cut off at 3 voxels");
    }

    #[test]
    fn test_non_uniform_step_cost() {
        // Two voxels: one step in X, one step in Y from seed.
        // X step costs 0.1, Y step costs 0.5.
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(1, 1, 1));
        let geo = make_geo(&[(4, 4, 4), (5, 4, 4), (4, 5, 4)], 1);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 4, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::new(0.1, 0.5, 0.1),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        assert_eq!(results.len(), 3);
        // X neighbor should be closer than Y neighbor.
        let x_entry = results.iter().find(|e| {
            let lookup = build_surface_lookup(pool.get(e.geo_slot));
            lookup[voxel_index(5, 4, 4) as usize] != SV_NONE
                && e.surface_index == lookup[voxel_index(5, 4, 4) as usize] as usize
        }).unwrap();
        let y_entry = results.iter().find(|e| {
            let lookup = build_surface_lookup(pool.get(e.geo_slot));
            lookup[voxel_index(4, 5, 4) as usize] != SV_NONE
                && e.surface_index == lookup[voxel_index(4, 5, 4) as usize] as usize
        }).unwrap();
        assert!(x_entry.geodesic_distance < y_entry.geodesic_distance,
            "X step (0.1) should be cheaper than Y step (0.5): {} vs {}",
            x_entry.geodesic_distance, y_entry.geodesic_distance);
    }

    #[test]
    fn test_disconnected_surfaces() {
        // Two separate clusters with a gap between them.
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(1, 1, 1));
        let geo = make_geo(&[(1, 4, 4), (2, 4, 4), (6, 4, 4), (7, 4, 4)], 1);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 1, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::splat(vs),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        assert_eq!(results.len(), 2, "should only reach connected cluster");
    }

    #[test]
    fn test_empty_seed_returns_empty() {
        let pool = Pool::<BrickGeometry>::new(4);
        let map = BrickMap::new(UVec3::new(1, 1, 1));

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::splat(0.1),
        };
        let results = surface_flood_fill(Vec3::ZERO, &map, &pool, 0.1, &params);
        assert!(results.is_empty());
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let mut pool = Pool::new(4);
        let mut map = BrickMap::new(UVec3::new(1, 1, 1));
        let geo = make_geo(&[(2, 4, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4), (6, 4, 4)], 1);
        place_brick(&mut pool, &mut map, 0, 0, 0, geo);

        let vs = 0.1;
        let origin = grid_origin(&map, vs);
        let seed = voxel_local_pos(UVec3::ZERO, 2, 4, 4, origin, vs);

        let params = FloodFillParams {
            max_distance: 10.0,
            step_cost: Vec3::splat(vs),
        };
        let results = surface_flood_fill(seed, &map, &pool, vs, &params);

        for w in results.windows(2) {
            assert!(w[0].geodesic_distance <= w[1].geodesic_distance + 1e-6,
                "results should be sorted: {} <= {}",
                w[0].geodesic_distance, w[1].geodesic_distance);
        }
    }
}
