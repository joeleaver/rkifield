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
//! Both use Fast Sweeping Method (Eikonal solver) for true Euclidean distances.

use glam::UVec3;

use crate::brick_geometry::{BrickGeometry, NeighborContext};
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
/// neighbors), then runs Fast Sweeping to compute signed distances at every
/// voxel in every allocated brick.
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
        None, // write all bricks
    );
}

/// Compute SDF distances for a region of bricks using the Fast Sweeping Method.
///
/// `region_min` and `region_max` are in brick coordinates (inclusive min, exclusive max).
/// Only bricks within this region have their SDF updated, but neighbor geometry
/// outside the region is read for cross-brick surface identification.
///
/// If `dirty_bricks` is provided, only those brick coordinates have their SDF cache
/// written back. Margin bricks participate in the Eikonal solve for context but
/// keep their existing SDF values.
pub fn compute_sdf_region(
    brick_map: &BrickMap,
    geometry: &[BrickGeometry],
    sdf_caches: &mut [SdfCache],
    slot_mappings: &[SlotMapping],
    region_min: UVec3,
    region_max: UVec3,
    voxel_size: f32,
    dirty_bricks: Option<&std::collections::HashSet<UVec3>>,
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

    // Region voxel grid
    let region_dims = region_max - region_min;
    let rvx = region_dims.x * 8;
    let rvy = region_dims.y * 8;
    let rvz = region_dims.z * 8;
    let total = (rvx * rvy * rvz) as usize;
    if total == 0 {
        return;
    }

    let mut dist: Vec<f32> = vec![f32::MAX; total];
    let mut is_solid: Vec<bool> = vec![false; total];
    // Track which voxels are in allocated bricks (vs empty/interior/out-of-bounds)
    let mut is_allocated: Vec<bool> = vec![false; total];

    let idx = |gx: u32, gy: u32, gz: u32| -> usize {
        (gx + gy * rvx + gz * rvx * rvy) as usize
    };

    let region_voxel_index = |bx: u32, by: u32, bz: u32, vx: u8, vy: u8, vz: u8| -> Option<usize> {
        let rbx = bx.checked_sub(region_min.x)?;
        let rby = by.checked_sub(region_min.y)?;
        let rbz = bz.checked_sub(region_min.z)?;
        if rbx >= region_dims.x || rby >= region_dims.y || rbz >= region_dims.z {
            return None;
        }
        Some(idx(rbx * 8 + vx as u32, rby * 8 + vy as u32, rbz * 8 + vz as u32))
    };

    // ── Phase 1: Populate is_solid, is_allocated, and seed surface voxels ───

    for bz in region_min.z..region_max.z {
        for by in region_min.y..region_max.y {
            for bx in region_min.x..region_max.x {
                let slot = match brick_map.get(bx, by, bz) {
                    Some(s) => s,
                    None => continue,
                };

                if slot == EMPTY_SLOT {
                    continue;
                }

                if slot == INTERIOR_SLOT {
                    for vz in 0..8u8 {
                        for vy in 0..8u8 {
                            for vx in 0..8u8 {
                                if let Some(i) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                    is_solid[i] = true;
                                    is_allocated[i] = true;
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

                // Mark solid and allocated
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if let Some(i) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                is_allocated[i] = true;
                                if geo.is_solid(vx, vy, vz) {
                                    is_solid[i] = true;
                                }
                            }
                        }
                    }
                }

                // Build cross-brick neighbor context for surface detection
                let neighbor = |nbx: i32, nby: i32, nbz: i32| -> Option<Option<&BrickGeometry>> {
                    if nbx < 0 || nby < 0 || nbz < 0 {
                        return None;
                    }
                    let (nbx, nby, nbz) = (nbx as u32, nby as u32, nbz as u32);
                    if nbx >= dims.x || nby >= dims.y || nbz >= dims.z {
                        return None;
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

                // Seed surface voxels with sub-voxel distance estimate
                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if ctx.is_surface_voxel(vx, vy, vz) {
                                if let Some(i) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                    let d = surface_seed_distance(&ctx, vx, vy, vz, voxel_size);
                                    dist[i] = d;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Phase 2: Fast Sweeping Method (Eikonal solver) ──────────────────
    //
    // Solve |∇d| = 1/h via iterative axis-aligned sweeps in all 8 octant
    // directions. Each voxel update solves the quadratic Eikonal equation
    // using its 6-connected neighbors' current distances.

    let h = voxel_size;

    // 8 sweep directions: all combinations of (±x, ±y, ±z)
    const SWEEPS: [(bool, bool, bool); 8] = [
        (false, false, false), (true,  false, false),
        (false, true,  false), (true,  true,  false),
        (false, false, true),  (true,  false, true),
        (false, true,  true),  (true,  true,  true),
    ];

    // 2 full iterations (typically converges in 1 for convex, 2 for concave)
    for _iter in 0..2 {
        for &(rev_x, rev_y, rev_z) in &SWEEPS {
            let x_range: Box<dyn Iterator<Item = u32>> = if rev_x {
                Box::new((0..rvx).rev())
            } else {
                Box::new(0..rvx)
            };

            for gz in sweep_range(rvz, rev_z) {
                for gy in sweep_range(rvy, rev_y) {
                    for gx in x_range_clone(rvx, rev_x) {
                        let i = idx(gx, gy, gz);
                        if !is_allocated[i] {
                            continue;
                        }

                        // Gather the minimum neighbor distance along each axis
                        let ax = axis_min(&dist, rvx, rvy, rvz, gx, gy, gz, 0);
                        let ay = axis_min(&dist, rvx, rvy, rvz, gx, gy, gz, 1);
                        let az = axis_min(&dist, rvx, rvy, rvz, gx, gy, gz, 2);

                        let new_d = eikonal_solve(ax, ay, az, h);
                        if new_d < dist[i] {
                            dist[i] = new_d;
                        }
                    }
                }
            }
            drop(x_range);
        }
    }

    // ── Phase 3: Write signed distances to SDF caches ───────────────────

    for bz in region_min.z..region_max.z {
        for by in region_min.y..region_max.y {
            for bx in region_min.x..region_max.x {
                // Skip bricks not in the dirty set.
                if let Some(dirty) = dirty_bricks {
                    if !dirty.contains(&UVec3::new(bx, by, bz)) {
                        continue;
                    }
                }

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

                let sdf_slot = slot_mappings[mapping_idx].sdf_slot as usize;
                let cache = &mut sdf_caches[sdf_slot];

                for vz in 0..8u8 {
                    for vy in 0..8u8 {
                        for vx in 0..8u8 {
                            if let Some(i) = region_voxel_index(bx, by, bz, vx, vy, vz) {
                                let d = dist[i];
                                let signed = if is_solid[i] { -d } else { d };
                                cache.set_distance(vx, vy, vz, signed);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Eikonal solver helpers ──────────────────────────────────────────────────

/// Solve the Eikonal equation `|∇d| = 1` at a voxel given the minimum
/// neighbor distances along each axis (a ≤ b ≤ c after sorting).
///
/// Returns the candidate distance. If no valid solution exists, returns f32::MAX.
fn eikonal_solve(ax: f32, ay: f32, az: f32, h: f32) -> f32 {
    // Sort so a ≤ b ≤ c
    let mut abc = [ax, ay, az];
    if abc[0] > abc[1] { abc.swap(0, 1); }
    if abc[1] > abc[2] { abc.swap(1, 2); }
    if abc[0] > abc[1] { abc.swap(0, 1); }
    let [a, b, c] = abc;

    // 1D: d = a + h
    let d1 = a + h;
    if d1 <= b {
        return d1;
    }

    // 2D: solve (d-a)² + (d-b)² = h²
    let sum_ab = a + b;
    let diff_ab = a - b;
    let disc2 = 2.0 * h * h - diff_ab * diff_ab;
    if disc2 >= 0.0 {
        let d2 = (sum_ab + disc2.sqrt()) * 0.5;
        if d2 <= c {
            return d2;
        }
    }

    // 3D: solve (d-a)² + (d-b)² + (d-c)² = h²
    let sum_abc = a + b + c;
    let sq_sum = a * a + b * b + c * c;
    let disc3 = sum_abc * sum_abc - 3.0 * (sq_sum - h * h);
    if disc3 >= 0.0 {
        let d3 = (sum_abc + disc3.sqrt()) / 3.0;
        return d3;
    }

    f32::MAX
}

/// Get the minimum distance from the two axis-aligned neighbors along `axis` (0=x, 1=y, 2=z).
#[inline]
fn axis_min(dist: &[f32], sx: u32, sy: u32, _sz: u32, gx: u32, gy: u32, gz: u32, axis: u8) -> f32 {
    let stride = match axis {
        0 => 1u32,
        1 => sx,
        _ => sx * sy,
    };
    let coord = match axis {
        0 => gx,
        1 => gy,
        _ => gz,
    };
    let max_coord = match axis {
        0 => sx,
        1 => sy,
        _ => _sz,
    };
    let base = gx + gy * sx + gz * sx * sy;

    let lo = if coord > 0 { dist[(base - stride) as usize] } else { f32::MAX };
    let hi = if coord + 1 < max_coord { dist[(base + stride) as usize] } else { f32::MAX };
    lo.min(hi)
}

/// Produce an iterator for a sweep direction along one axis.
fn sweep_range(size: u32, reverse: bool) -> Box<dyn Iterator<Item = u32>> {
    if reverse {
        Box::new((0..size).rev())
    } else {
        Box::new(0..size)
    }
}

/// Clone-friendly version of sweep_range for the inner x loop.
fn x_range_clone(size: u32, reverse: bool) -> Box<dyn Iterator<Item = u32>> {
    if reverse {
        Box::new((0..size).rev())
    } else {
        Box::new(0..size)
    }
}

/// Estimate sub-voxel distance for a surface voxel using the local occupancy gradient.
///
/// Instead of a flat 0.5h, count how many axes have a boundary crossing
/// (face neighbor with different occupancy). More crossing axes means the
/// surface passes closer to the voxel center:
/// - 1 axis:  ~0.50h (surface perpendicular to one axis)
/// - 2 axes:  ~0.35h (surface crosses an edge region)
/// - 3 axes:  ~0.29h (surface crosses a corner region)
fn surface_seed_distance(
    ctx: &NeighborContext<'_>,
    vx: u8,
    vy: u8,
    vz: u8,
    voxel_size: f32,
) -> f32 {
    let solid = ctx.center.is_solid(vx, vy, vz);
    let mut crossing_axes = 0u32;

    // Check each axis for a boundary crossing
    let offsets: [(i8, i8, i8); 6] = [
        (-1, 0, 0), (1, 0, 0),  // x axis
        (0, -1, 0), (0, 1, 0),  // y axis
        (0, 0, -1), (0, 0, 1),  // z axis
    ];

    let mut x_cross = false;
    let mut y_cross = false;
    let mut z_cross = false;

    for (i, &(dx, dy, dz)) in offsets.iter().enumerate() {
        let nx = vx as i8 + dx;
        let ny = vy as i8 + dy;
        let nz = vz as i8 + dz;
        let neighbor_solid = ctx.is_neighbor_solid(nx, ny, nz);
        if solid != neighbor_solid {
            match i / 2 {
                0 => x_cross = true,
                1 => y_cross = true,
                _ => z_cross = true,
            }
        }
    }

    crossing_axes = x_cross as u32 + y_cross as u32 + z_cross as u32;

    // Distance = 0.5h / sqrt(crossing_axes), clamped to at least 1 axis
    let axes = crossing_axes.max(1) as f32;
    0.5 * voxel_size / axes.sqrt()
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
    fn empty_brick_stays_large() {
        let geo = BrickGeometry::new();
        let cache = compute_single_brick(&geo, 0.1);

        // All distances should remain large (no surface voxels to seed from)
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let d = cache.get_distance(x, y, z);
                    assert!(d > 100.0 || d == f32::MAX || d.is_infinite(),
                        "expected large positive, got {d} at ({x},{y},{z})");
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
            None,
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

    #[test]
    fn eikonal_flat_plane_accuracy() {
        // Half-solid plane: exact distance should be (z_from_surface + 0.5) * voxel_size
        // The Eikonal solver should produce nearly linear distances.
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

        // At x=4, y=4, along z: distances should be approximately linear
        // z=3 is the last solid, z=4 is first empty. Surface between z=3 and z=4.
        // Expected: d(z) ≈ (|z - 3.5| + some_offset) * vs
        // The Eikonal should give smooth, approximately linear values.
        let d3 = cache.get_distance(4, 4, 3).abs();
        let d4 = cache.get_distance(4, 4, 4).abs();
        let d5 = cache.get_distance(4, 4, 5).abs();

        // d4 and d3 should be approximately equal (symmetric around surface)
        let asymmetry = (d4 - d3).abs();
        assert!(asymmetry < vs * 0.3, "asymmetry too large: {asymmetry} (d3={d3}, d4={d4})");

        // d5 should be approximately d4 + vs
        let step = d5 - d4;
        assert!((step - vs).abs() < vs * 0.5, "step from z4→z5 should be ~{vs}, got {step}");
    }
}
