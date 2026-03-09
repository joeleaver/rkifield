//! Compute SDF distances from brick geometry (occupancy → signed distances).
//!
//! This is the core geometry-first algorithm: occupancy bitmasks go in, signed
//! distance fields come out. The SDF is derived data — always recomputable from
//! geometry.
//!
//! Two entry points:
//! - [`compute_sdf_from_geometry`] — full object, all allocated bricks
//! - [`compute_sdf_region`] — local region (e.g., after a sculpt edit)
//!
//! Both use Dijkstra propagation from identified surface voxels.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use glam::UVec3;

use crate::brick_geometry::{BrickGeometry, NeighborContext, index_to_xyz, voxel_index};
use crate::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};
use crate::sdf_cache::SdfCache;

/// A mapping from a brick map slot to geometry and SDF cache pool slots.
#[derive(Debug, Clone, Copy)]
pub struct SlotMapping {
    /// Index in the brick map entries (the value stored in the brick map).
    pub brick_slot: u32,
    /// Index into the geometry pool.
    pub geometry_slot: u32,
    /// Index into the SDF cache pool.
    pub sdf_slot: u32,
}

/// Compute SDF distances for all allocated bricks from their geometry.
///
/// Identifies surface voxels across the entire object (considering cross-brick
/// neighbors), then runs Dijkstra propagation to compute signed distances at
/// every voxel in every allocated brick.
pub fn compute_sdf_from_geometry(
    brick_map: &BrickMap,
    geometry: &[BrickGeometry],       // indexed by slot_mapping.geometry_slot
    sdf_caches: &mut [SdfCache],      // indexed by slot_mapping.sdf_slot
    slot_mappings: &[SlotMapping],    // one per allocated brick
    voxel_size: f32,
) {
    compute_sdf_region(
        brick_map,
        geometry,
        sdf_caches,
        slot_mappings,
        UVec3::ZERO,
        brick_map.dims,
        voxel_size,
    );
}

/// Compute SDF distances for a region of bricks.
///
/// `region_min` and `region_max` are in brick coordinates (inclusive min, exclusive max).
/// Only bricks within this region have their SDF updated, but neighbor geometry
/// outside the region is read for cross-brick surface identification.
pub fn compute_sdf_region(
    brick_map: &BrickMap,
    geometry: &[BrickGeometry],
    sdf_caches: &mut [SdfCache],
    slot_mappings: &[SlotMapping],
    region_min: UVec3,
    region_max: UVec3,
    voxel_size: f32,
) {
    let dims = brick_map.dims;

    // Build reverse lookup: brick_slot → index in slot_mappings
    let max_slot = slot_mappings.iter().map(|m| m.brick_slot).max().unwrap_or(0) as usize;
    let mut slot_to_mapping: Vec<Option<usize>> = vec![None; max_slot + 1];
    for (i, m) in slot_mappings.iter().enumerate() {
        slot_to_mapping[m.brick_slot as usize] = Some(i);
    }

    // Helper: get geometry for a brick map entry
    let get_geometry = |slot: u32| -> Option<&BrickGeometry> {
        if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
            return None;
        }
        let idx = slot_to_mapping.get(slot as usize)?.as_ref()?;
        Some(&geometry[slot_mappings[*idx].geometry_slot as usize])
    };

    // Helper: check if a brick coordinate is in the region
    let in_region = |bx: u32, by: u32, bz: u32| -> bool {
        bx >= region_min.x && bx < region_max.x
            && by >= region_min.y && by < region_max.y
            && bz >= region_min.z && bz < region_max.z
    };

    // Phase 1: Identify surface voxels and seed the priority queue
    // Global voxel coordinate = (bx * 8 + vx, by * 8 + vy, bz * 8 + vz)
    let mut heap: BinaryHeap<DijkstraEntry> = BinaryHeap::new();

    // Track best known distance for each voxel in region
    // Key: (brick_index_in_mappings, voxel_index_in_brick)
    // We use a flat map keyed by global voxel coordinate for simplicity
    let region_dims = region_max - region_min;
    let region_voxel_dims = region_dims * 8;
    let total_region_voxels = (region_voxel_dims.x * region_voxel_dims.y * region_voxel_dims.z) as usize;
    let mut best_dist: Vec<f32> = vec![f32::INFINITY; total_region_voxels];
    let mut is_solid: Vec<bool> = vec![false; total_region_voxels];

    let region_voxel_index = |bx: u32, by: u32, bz: u32, vx: u8, vy: u8, vz: u8| -> Option<usize> {
        let rbx = bx.checked_sub(region_min.x)?;
        let rby = by.checked_sub(region_min.y)?;
        let rbz = bz.checked_sub(region_min.z)?;
        if rbx >= region_dims.x || rby >= region_dims.y || rbz >= region_dims.z {
            return None;
        }
        let gx = rbx * 8 + vx as u32;
        let gy = rby * 8 + vy as u32;
        let gz = rbz * 8 + vz as u32;
        Some((gx + gy * region_voxel_dims.x + gz * region_voxel_dims.x * region_voxel_dims.y) as usize)
    };

    // Populate is_solid array and find surface voxels
    for bz in region_min.z..region_max.z {
        for by in region_min.y..region_max.y {
            for bx in region_min.x..region_max.x {
                let slot = match brick_map.get(bx, by, bz) {
                    Some(s) => s,
                    None => continue,
                };

                if slot == EMPTY_SLOT {
                    // All voxels empty (is_solid already false)
                    continue;
                }

                if slot == INTERIOR_SLOT {
                    // All voxels solid
                    for vz in 0..8u8 {
                        for vy in 0..8u8 {
                            for vx in 0..8u8 {
                                if let Some(idx) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                    is_solid[idx] = true;
                                }
                            }
                        }
                    }
                    continue;
                }

                let geo = match get_geometry(slot) {
                    Some(g) => g,
                    None => continue,
                };

                // Populate is_solid
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if geo.is_solid(vx, vy, vz) {
                                if let Some(idx) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                    is_solid[idx] = true;
                                }
                            }
                        }
                    }
                }

                // NeighborContext.neighbors: None = EMPTY, Some(None) = INTERIOR, Some(Some(geo)) = allocated
                // get_neighbor returns Some(outer) where outer: Option<Option<&BrickGeometry>>
                // For out-of-bounds, treat as EMPTY_SLOT (None)
                let neighbor = |nbx: i32, nby: i32, nbz: i32| -> Option<Option<&BrickGeometry>> {
                    if nbx < 0 || nby < 0 || nbz < 0 {
                        return None; // Out of bounds → EMPTY
                    }
                    let (nbx, nby, nbz) = (nbx as u32, nby as u32, nbz as u32);
                    if nbx >= dims.x || nby >= dims.y || nbz >= dims.z {
                        return None; // Out of bounds → EMPTY
                    }
                    match brick_map.get(nbx, nby, nbz)? {
                        EMPTY_SLOT => None,
                        INTERIOR_SLOT => Some(None),
                        s => Some(get_geometry(s)),
                    }
                };

                let ctx = NeighborContext {
                    center: geo,
                    neighbors: [
                        neighbor(bx as i32 - 1, by as i32, bz as i32),
                        neighbor(bx as i32 + 1, by as i32, bz as i32),
                        neighbor(bx as i32, by as i32 - 1, bz as i32),
                        neighbor(bx as i32, by as i32 + 1, bz as i32),
                        neighbor(bx as i32, by as i32, bz as i32 - 1),
                        neighbor(bx as i32, by as i32, bz as i32 + 1),
                    ],
                };

                // Seed surface voxels with distance ~0.5 voxel_size
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if ctx.is_surface_voxel(vx, vy, vz) {
                                if let Some(idx) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                    // Surface voxels seed with sub-voxel distance estimate
                                    let d = 0.5 * voxel_size;
                                    best_dist[idx] = d;
                                    heap.push(DijkstraEntry { dist: d, index: idx });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Phase 2: Dijkstra propagation (26-connected)
    while let Some(entry) = heap.pop() {
        if entry.dist > best_dist[entry.index] {
            continue; // Stale entry
        }

        // Decode global voxel position in region
        let gi = entry.index as u32;
        let gx = gi % region_voxel_dims.x;
        let gy = (gi / region_voxel_dims.x) % region_voxel_dims.y;
        let gz = gi / (region_voxel_dims.x * region_voxel_dims.y);

        // 26-connected neighbors
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }

                    let nx = gx as i32 + dx;
                    let ny = gy as i32 + dy;
                    let nz = gz as i32 + dz;

                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let (nx, ny, nz) = (nx as u32, ny as u32, nz as u32);
                    if nx >= region_voxel_dims.x || ny >= region_voxel_dims.y || nz >= region_voxel_dims.z {
                        continue;
                    }

                    let ni = (nx + ny * region_voxel_dims.x + nz * region_voxel_dims.x * region_voxel_dims.y) as usize;

                    // Check that neighbor is in an allocated brick
                    let nbx = nx / 8 + region_min.x;
                    let nby = ny / 8 + region_min.y;
                    let nbz = nz / 8 + region_min.z;

                    if let Some(nslot) = brick_map.get(nbx, nby, nbz) {
                        if nslot == EMPTY_SLOT || nslot == INTERIOR_SLOT {
                            // Skip — we only compute distances within allocated bricks
                            // (plus interior/empty get sentinel values)
                            continue;
                        }
                    } else {
                        continue;
                    }

                    // Euclidean distance between voxel centers
                    let step_dist = ((dx * dx + dy * dy + dz * dz) as f32).sqrt() * voxel_size;
                    let new_dist = entry.dist + step_dist;

                    if new_dist < best_dist[ni] {
                        best_dist[ni] = new_dist;
                        heap.push(DijkstraEntry { dist: new_dist, index: ni });
                    }
                }
            }
        }
    }

    // Phase 3: Write signed distances to SDF caches
    for bz in region_min.z..region_max.z {
        for by in region_min.y..region_max.y {
            for bx in region_min.x..region_max.x {
                let slot = match brick_map.get(bx, by, bz) {
                    Some(s) => s,
                    None => continue,
                };

                if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                    continue;
                }

                let mapping_idx = match slot_to_mapping.get(slot as usize).and_then(|v| *v) {
                    Some(i) => i,
                    None => continue,
                };

                if !in_region(bx, by, bz) {
                    continue;
                }

                let sdf_slot = slot_mappings[mapping_idx].sdf_slot as usize;
                let cache = &mut sdf_caches[sdf_slot];

                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if let Some(idx) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                let dist = best_dist[idx];
                                let signed = if is_solid[idx] { -dist } else { dist };
                                cache.set_distance(vx, vy, vz, signed);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Priority queue entry for Dijkstra propagation.
#[derive(Clone, Copy)]
struct DijkstraEntry {
    dist: f32,
    index: usize,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (BinaryHeap is a max-heap)
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple 1-brick object with the given geometry,
    /// compute SDF, and return the cache.
    fn compute_single_brick(geo: &BrickGeometry, voxel_size: f32) -> SdfCache {
        let mut map = BrickMap::new(UVec3::ONE);
        map.set(0, 0, 0, 0); // slot 0

        let mappings = vec![SlotMapping {
            brick_slot: 0,
            geometry_slot: 0,
            sdf_slot: 0,
        }];

        let geometry = vec![geo.clone()];
        let mut sdf_caches = vec![SdfCache::empty()];

        compute_sdf_from_geometry(&map, &geometry, &mut sdf_caches, &mappings, voxel_size);
        sdf_caches.into_iter().next().unwrap()
    }

    #[test]
    fn half_solid_plane_linear_distances() {
        // Lower half (z=0..3) solid, upper half (z=4..7) empty
        let mut geo = BrickGeometry::new();
        for z in 0..4u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    geo.set_solid(x, y, z, true);
                }
            }
        }

        let vs = 0.1;
        let cache = compute_single_brick(&geo, vs);

        // Surface is at z=3/z=4 boundary
        // Distance should increase with distance from surface
        let d_z2 = cache.get_distance(4, 4, 2); // 1 voxel below surface → negative (solid)
        let d_z3 = cache.get_distance(4, 4, 3); // surface solid → negative, small
        let d_z4 = cache.get_distance(4, 4, 4); // surface empty → positive, small
        let d_z5 = cache.get_distance(4, 4, 5); // 1 voxel above surface → positive

        // Signs
        assert!(d_z2 < 0.0, "z=2 should be negative (solid), got {d_z2}");
        assert!(d_z3 < 0.0, "z=3 should be negative (solid), got {d_z3}");
        assert!(d_z4 > 0.0, "z=4 should be positive (empty), got {d_z4}");
        assert!(d_z5 > 0.0, "z=5 should be positive (empty), got {d_z5}");

        // Monotonicity: further from surface → larger magnitude
        assert!(d_z2.abs() > d_z3.abs(), "|z2|={} should > |z3|={}", d_z2.abs(), d_z3.abs());
        assert!(d_z5.abs() > d_z4.abs(), "|z5|={} should > |z4|={}", d_z5.abs(), d_z4.abs());
    }

    #[test]
    fn sphere_occupancy_smooth_distances() {
        // Create a sphere of radius 3 voxels centered at (3.5, 3.5, 3.5)
        let mut geo = BrickGeometry::new();
        let center = 3.5f32;
        let radius = 3.0f32;

        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let dx = x as f32 + 0.5 - center;
                    let dy = y as f32 + 0.5 - center;
                    let dz = z as f32 + 0.5 - center;
                    if (dx * dx + dy * dy + dz * dz).sqrt() <= radius {
                        geo.set_solid(x, y, z, true);
                    }
                }
            }
        }

        let vs = 0.1;
        let cache = compute_single_brick(&geo, vs);

        // Center should be solid (negative)
        let d_center = cache.get_distance(3, 3, 3);
        assert!(d_center < 0.0, "center should be negative, got {d_center}");

        // Corner (0,0,0) should be empty (positive)
        let d_corner = cache.get_distance(0, 0, 0);
        assert!(d_corner > 0.0, "corner should be positive, got {d_corner}");

        // Distances should be approximately smooth (no huge jumps between neighbors)
        for z in 0..7u8 {
            let d0 = cache.get_distance(4, 4, z);
            let d1 = cache.get_distance(4, 4, z + 1);
            let jump = (d1 - d0).abs();
            assert!(
                jump < 2.0 * vs,
                "large jump between z={z} ({d0}) and z={} ({d1}): {jump}",
                z + 1
            );
        }
    }

    #[test]
    fn cross_brick_propagation() {
        // Two bricks side by side. Surface in the left brick, distances should
        // propagate into the right brick.
        let mut map = BrickMap::new(UVec3::new(2, 1, 1));
        map.set(0, 0, 0, 0); // left brick
        map.set(1, 0, 0, 1); // right brick

        let mut geo_left = BrickGeometry::new();
        // Single solid voxel at x=7 (right edge) of left brick
        geo_left.set_solid(7, 4, 4, true);

        let geo_right = BrickGeometry::new(); // all empty

        let geometry = vec![geo_left, geo_right];
        let mut sdf_caches = vec![SdfCache::empty(), SdfCache::empty()];

        let mappings = vec![
            SlotMapping { brick_slot: 0, geometry_slot: 0, sdf_slot: 0 },
            SlotMapping { brick_slot: 1, geometry_slot: 1, sdf_slot: 1 },
        ];

        let vs = 0.1;
        compute_sdf_from_geometry(&map, &geometry, &mut sdf_caches, &mappings, vs);

        // Left brick: voxel (7,4,4) should be negative (solid)
        let d_solid = sdf_caches[0].get_distance(7, 4, 4);
        assert!(d_solid < 0.0, "solid voxel should be negative, got {d_solid}");

        // Right brick: voxel (0,4,4) is adjacent to the solid in left brick → surface → small positive
        let d_adjacent = sdf_caches[1].get_distance(0, 4, 4);
        assert!(d_adjacent > 0.0, "adjacent empty should be positive, got {d_adjacent}");

        // Right brick: voxel (7,4,4) is far from surface → larger positive
        let d_far = sdf_caches[1].get_distance(7, 4, 4);
        assert!(d_far > d_adjacent, "far voxel dist {d_far} should > adjacent {d_adjacent}");
    }

    #[test]
    fn empty_brick_stays_infinity() {
        let geo = BrickGeometry::new();
        let cache = compute_single_brick(&geo, 0.1);

        // All distances should remain at infinity (no surface voxels to seed from)
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let d = cache.get_distance(x, y, z);
                    assert!(d.is_infinite() || d > 100.0, "expected large positive, got {d} at ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn region_limited_computation() {
        // 2x1x1 brick map, only update the first brick
        let mut map = BrickMap::new(UVec3::new(2, 1, 1));
        map.set(0, 0, 0, 0);
        map.set(1, 0, 0, 1);

        let mut geo0 = BrickGeometry::new();
        geo0.set_solid(4, 4, 4, true);

        let mut geo1 = BrickGeometry::new();
        geo1.set_solid(4, 4, 4, true);

        let geometry = vec![geo0, geo1];
        let mut sdf_caches = vec![SdfCache::empty(), SdfCache::empty()];

        let mappings = vec![
            SlotMapping { brick_slot: 0, geometry_slot: 0, sdf_slot: 0 },
            SlotMapping { brick_slot: 1, geometry_slot: 1, sdf_slot: 1 },
        ];

        // Only compute region for brick 0
        compute_sdf_region(
            &map, &geometry, &mut sdf_caches, &mappings,
            UVec3::ZERO, UVec3::new(1, 1, 1), 0.1,
        );

        // Brick 0 should have distances computed
        let d0 = sdf_caches[0].get_distance(4, 4, 4);
        assert!(d0 < 0.0, "brick 0 center should be negative, got {d0}");

        // Brick 1 should still be at default (not updated)
        let d1 = sdf_caches[1].get_distance(4, 4, 4);
        assert!(d1.is_infinite(), "brick 1 should be untouched, got {d1}");
    }

    #[test]
    fn interior_slot_boundary() {
        // Test that INTERIOR_SLOT is handled correctly:
        // A brick with some solid voxels next to an INTERIOR_SLOT brick
        let mut map = BrickMap::new(UVec3::new(2, 1, 1));
        map.set(0, 0, 0, 0);           // allocated
        map.set(1, 0, 0, INTERIOR_SLOT); // deep interior

        let mut geo = BrickGeometry::fully_solid();
        // Make voxel (7,4,4) empty — it's on the boundary with the interior brick
        geo.set_solid(7, 4, 4, false);

        let geometry = vec![geo];
        let mut sdf_caches = vec![SdfCache::empty()];

        let mappings = vec![
            SlotMapping { brick_slot: 0, geometry_slot: 0, sdf_slot: 0 },
        ];

        compute_sdf_from_geometry(&map, &geometry, &mut sdf_caches, &mappings, 0.1);

        // Voxel (7,4,4) is empty, has solid neighbors → surface → small positive distance
        let d = sdf_caches[0].get_distance(7, 4, 4);
        assert!(d > 0.0, "empty voxel should be positive, got {d}");

        // Voxel (6,4,4) is solid with an empty neighbor at (7,4,4) → surface → small negative
        let d2 = sdf_caches[0].get_distance(6, 4, 4);
        assert!(d2 < 0.0, "solid surface voxel should be negative, got {d2}");
    }
}
