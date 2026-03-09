//! Geometry-first brick representation.
//!
//! [`BrickGeometry`] is the source of truth for voxel shape. It stores a 512-bit
//! occupancy bitmask (1 = solid) and a list of [`SurfaceVoxel`]s — voxels on the
//! solid/empty boundary that carry color and material data.
//!
//! SDF distances are derived from geometry and cached separately in [`super::sdf_cache::SdfCache`].

use crate::constants::BRICK_DIM;

/// Compact geometry representation for one 8×8×8 brick.
///
/// Occupancy is the source of truth for shape. Surface voxels carry per-voxel
/// color and material data for voxels on the solid/empty boundary.
#[derive(Debug, Clone)]
pub struct BrickGeometry {
    /// 512-bit occupancy bitmask — bit N = voxel N is solid.
    /// Layout: `voxel_index(x,y,z) = x + y*8 + z*64`, same as [`super::brick::brick_index`].
    pub occupancy: [u64; 8],
    /// Data for voxels on the solid/empty boundary.
    pub surface_voxels: Vec<SurfaceVoxel>,
}

/// Per-surface-voxel data: position within brick, RGBA color, material.
///
/// 8 bytes, naturally aligned. Only stored for voxels on the solid/empty boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SurfaceVoxel {
    /// Index within the 8×8×8 brick (0–511). Same layout as [`voxel_index`].
    pub index: u16,
    /// RGBA diffuse color. Alpha available for custom shader control.
    pub color: [u8; 4],
    /// Material table index for PBR properties (roughness, metallic, etc.).
    pub material_id: u8,
    /// Reserved for future use.
    pub _reserved: u8,
}

/// Convert 3D coordinates to a flat voxel index within an 8×8×8 brick.
///
/// Layout: `x + y*8 + z*64` (z-major order).
#[inline]
pub fn voxel_index(x: u8, y: u8, z: u8) -> u16 {
    debug_assert!(x < 8 && y < 8 && z < 8, "voxel coords out of range: ({x},{y},{z})");
    x as u16 + y as u16 * 8 + z as u16 * 64
}

/// Convert a flat voxel index (0–511) back to 3D coordinates.
#[inline]
pub fn index_to_xyz(index: u16) -> (u8, u8, u8) {
    debug_assert!(index < 512, "voxel index out of range: {index}");
    let x = (index % 8) as u8;
    let y = ((index / 8) % 8) as u8;
    let z = (index / 64) as u8;
    (x, y, z)
}

impl SurfaceVoxel {
    /// Create a new surface voxel.
    pub fn new(index: u16, color: [u8; 4], material_id: u8) -> Self {
        Self {
            index,
            color,
            material_id,
            _reserved: 0,
        }
    }

    /// Byte serialization (8 bytes).
    pub fn to_bytes(&self) -> [u8; 8] {
        let idx = self.index.to_le_bytes();
        [
            idx[0], idx[1],
            self.color[0], self.color[1], self.color[2], self.color[3],
            self.material_id, self._reserved,
        ]
    }

    /// Deserialize from 8 bytes.
    pub fn from_bytes(bytes: &[u8; 8]) -> Self {
        Self {
            index: u16::from_le_bytes([bytes[0], bytes[1]]),
            color: [bytes[2], bytes[3], bytes[4], bytes[5]],
            material_id: bytes[6],
            _reserved: bytes[7],
        }
    }
}

impl Default for BrickGeometry {
    fn default() -> Self {
        Self {
            occupancy: [0u64; 8],
            surface_voxels: Vec::new(),
        }
    }
}

impl BrickGeometry {
    /// Create a new empty brick geometry (all voxels empty).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a fully solid brick (all 512 bits set).
    pub fn fully_solid() -> Self {
        Self {
            occupancy: [u64::MAX; 8],
            surface_voxels: Vec::new(),
        }
    }

    // ── Occupancy queries ──────────────────────────────────────────────

    /// Check if the voxel at (x, y, z) is solid.
    #[inline]
    pub fn is_solid(&self, x: u8, y: u8, z: u8) -> bool {
        let idx = voxel_index(x, y, z);
        let word = idx / 64;
        let bit = idx % 64;
        (self.occupancy[word as usize] >> bit) & 1 != 0
    }

    /// Set or clear the solid bit for the voxel at (x, y, z).
    #[inline]
    pub fn set_solid(&mut self, x: u8, y: u8, z: u8, solid: bool) {
        let idx = voxel_index(x, y, z);
        let word = idx / 64;
        let bit = idx % 64;
        if solid {
            self.occupancy[word as usize] |= 1u64 << bit;
        } else {
            self.occupancy[word as usize] &= !(1u64 << bit);
        }
    }

    /// Count of solid voxels in this brick.
    pub fn solid_count(&self) -> u32 {
        self.occupancy.iter().map(|w| w.count_ones()).sum()
    }

    /// Count of surface voxels.
    pub fn surface_count(&self) -> usize {
        self.surface_voxels.len()
    }

    /// Returns true if all 512 voxels are empty (EMPTY_SLOT candidate).
    pub fn is_fully_empty(&self) -> bool {
        self.occupancy.iter().all(|&w| w == 0)
    }

    /// Returns true if all 512 voxels are solid (INTERIOR_SLOT candidate).
    pub fn is_fully_solid(&self) -> bool {
        self.occupancy.iter().all(|&w| w == u64::MAX)
    }

    // ── Surface voxel identification ───────────────────────────────────

    /// Check if the voxel at (x, y, z) is a surface voxel within this brick only.
    ///
    /// A surface voxel is solid with at least one empty 6-neighbor, OR empty with
    /// at least one solid 6-neighbor. Neighbors outside the brick boundary are
    /// treated as empty (conservative — use [`is_surface_voxel_with_context`] for
    /// cross-brick accuracy).
    pub fn is_surface_voxel(&self, x: u8, y: u8, z: u8) -> bool {
        let solid = self.is_solid(x, y, z);

        // Check 6-connected neighbors
        let neighbors: [(i8, i8, i8); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        for (dx, dy, dz) in neighbors {
            let nx = x as i8 + dx;
            let ny = y as i8 + dy;
            let nz = z as i8 + dz;

            let neighbor_solid = if nx < 0 || nx >= 8 || ny < 0 || ny >= 8 || nz < 0 || nz >= 8 {
                false // out of brick = treat as empty
            } else {
                self.is_solid(nx as u8, ny as u8, nz as u8)
            };

            if solid && !neighbor_solid {
                return true; // solid with empty neighbor
            }
            if !solid && neighbor_solid {
                return true; // empty with solid neighbor
            }
        }

        false
    }

    /// Rebuild the surface voxel list from occupancy (brick-local only).
    ///
    /// Existing surface voxels are cleared. New surface voxels get default color
    /// (white, 255 alpha) and material_id 0.
    pub fn rebuild_surface_list(&mut self) {
        self.surface_voxels.clear();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    if self.is_surface_voxel(x, y, z) {
                        self.surface_voxels.push(SurfaceVoxel::new(
                            voxel_index(x, y, z),
                            [255, 255, 255, 255],
                            0,
                        ));
                    }
                }
            }
        }
    }

    /// Rebuild surface list, preserving color/material from existing surface voxels
    /// where possible.
    pub fn rebuild_surface_list_preserving(&mut self) {
        // Build a lookup from old surface voxels
        let old: std::collections::HashMap<u16, SurfaceVoxel> =
            self.surface_voxels.iter().map(|sv| (sv.index, *sv)).collect();

        self.surface_voxels.clear();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    if self.is_surface_voxel(x, y, z) {
                        let idx = voxel_index(x, y, z);
                        if let Some(old_sv) = old.get(&idx) {
                            self.surface_voxels.push(*old_sv);
                        } else {
                            self.surface_voxels.push(SurfaceVoxel::new(
                                idx,
                                [255, 255, 255, 255],
                                0,
                            ));
                        }
                    }
                }
            }
        }
    }

    /// Look up surface voxel data at the given index, if it exists.
    pub fn get_surface_voxel(&self, index: u16) -> Option<&SurfaceVoxel> {
        self.surface_voxels.iter().find(|sv| sv.index == index)
    }

    /// Look up mutable surface voxel data at the given index.
    pub fn get_surface_voxel_mut(&mut self, index: u16) -> Option<&mut SurfaceVoxel> {
        self.surface_voxels.iter_mut().find(|sv| sv.index == index)
    }

    // ── Serialization ──────────────────────────────────────────────────

    /// Serialize to bytes: 64 bytes occupancy + 2 bytes surface_count + 8 bytes per surface voxel.
    pub fn to_bytes(&self) -> Vec<u8> {
        let surface_count = self.surface_voxels.len() as u16;
        let mut buf = Vec::with_capacity(64 + 2 + self.surface_voxels.len() * 8);

        // Occupancy: 8 × u64 = 64 bytes
        for word in &self.occupancy {
            buf.extend_from_slice(&word.to_le_bytes());
        }

        // Surface count
        buf.extend_from_slice(&surface_count.to_le_bytes());

        // Surface voxels
        for sv in &self.surface_voxels {
            buf.extend_from_slice(&sv.to_bytes());
        }

        buf
    }

    /// Deserialize from bytes. Returns None if the data is too short.
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 66 {
            return None; // minimum: 64 (occupancy) + 2 (count)
        }

        let mut occupancy = [0u64; 8];
        for (i, word) in occupancy.iter_mut().enumerate() {
            let offset = i * 8;
            *word = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        }

        let surface_count = u16::from_le_bytes(data[64..66].try_into().ok()?) as usize;
        let total_size = 66 + surface_count * 8;

        if data.len() < total_size {
            return None;
        }

        let mut surface_voxels = Vec::with_capacity(surface_count);
        for i in 0..surface_count {
            let offset = 66 + i * 8;
            let bytes: [u8; 8] = data[offset..offset + 8].try_into().ok()?;
            surface_voxels.push(SurfaceVoxel::from_bytes(&bytes));
        }

        Some((
            Self {
                occupancy,
                surface_voxels,
            },
            total_size,
        ))
    }
}

// ── Cross-brick neighbor context ───────────────────────────────────────────

/// Context for cross-brick neighbor queries at brick boundaries.
///
/// The center brick's surface voxels can depend on neighbors in adjacent bricks.
/// At each of the 6 faces:
/// - `None` → EMPTY_SLOT (all neighbors on that face are empty)
/// - `Some(None)` → INTERIOR_SLOT (all neighbors on that face are solid)
/// - `Some(Some(geo))` → allocated brick (read occupancy from neighbor geometry)
pub struct NeighborContext<'a> {
    /// The brick being queried.
    pub center: &'a BrickGeometry,
    /// Neighbor bricks in order: -X, +X, -Y, +Y, -Z, +Z.
    pub neighbors: [Option<Option<&'a BrickGeometry>>; 6],
}

impl<'a> NeighborContext<'a> {
    /// Check if a neighbor voxel (possibly in an adjacent brick) is solid.
    fn is_neighbor_solid(&self, x: i8, y: i8, z: i8) -> bool {
        if x >= 0 && x < 8 && y >= 0 && y < 8 && z >= 0 && z < 8 {
            return self.center.is_solid(x as u8, y as u8, z as u8);
        }

        // Determine which face we crossed and the coordinate in the neighbor brick
        let (face_idx, nx, ny, nz) = if x < 0 {
            (0, 7u8, y as u8, z as u8)
        } else if x >= 8 {
            (1, 0u8, y as u8, z as u8)
        } else if y < 0 {
            (2, x as u8, 7u8, z as u8)
        } else if y >= 8 {
            (3, x as u8, 0u8, z as u8)
        } else if z < 0 {
            (4, x as u8, y as u8, 7u8)
        } else {
            // z >= 8
            (5, x as u8, y as u8, 0u8)
        };

        match &self.neighbors[face_idx] {
            None => false,                       // EMPTY_SLOT → empty
            Some(None) => true,                  // INTERIOR_SLOT → solid
            Some(Some(geo)) => geo.is_solid(nx, ny, nz),
        }
    }

    /// Check if the voxel at (x, y, z) in the center brick is a surface voxel,
    /// considering cross-brick neighbors.
    pub fn is_surface_voxel(&self, x: u8, y: u8, z: u8) -> bool {
        let solid = self.center.is_solid(x, y, z);

        let neighbors_offsets: [(i8, i8, i8); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        for (dx, dy, dz) in neighbors_offsets {
            let nx = x as i8 + dx;
            let ny = y as i8 + dy;
            let nz = z as i8 + dz;

            let neighbor_solid = self.is_neighbor_solid(nx, ny, nz);

            if solid && !neighbor_solid {
                return true;
            }
            if !solid && neighbor_solid {
                return true;
            }
        }

        false
    }

    /// Rebuild surface voxel list for the center brick using cross-brick context.
    pub fn rebuild_surface_list(&self) -> Vec<SurfaceVoxel> {
        // Preserve existing surface voxel data where possible
        let old: std::collections::HashMap<u16, SurfaceVoxel> =
            self.center.surface_voxels.iter().map(|sv| (sv.index, *sv)).collect();

        let mut result = Vec::new();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    if self.is_surface_voxel(x, y, z) {
                        let idx = voxel_index(x, y, z);
                        if let Some(old_sv) = old.get(&idx) {
                            result.push(*old_sv);
                        } else {
                            result.push(SurfaceVoxel::new(idx, [255, 255, 255, 255], 0));
                        }
                    }
                }
            }
        }
        result
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn surface_voxel_size_is_8_bytes() {
        assert_eq!(mem::size_of::<SurfaceVoxel>(), 8);
    }

    #[test]
    fn voxel_index_corners() {
        assert_eq!(voxel_index(0, 0, 0), 0);
        assert_eq!(voxel_index(7, 0, 0), 7);
        assert_eq!(voxel_index(0, 7, 0), 56);
        assert_eq!(voxel_index(0, 0, 7), 448);
        assert_eq!(voxel_index(7, 7, 7), 511);
    }

    #[test]
    fn index_to_xyz_roundtrip() {
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    let idx = voxel_index(x, y, z);
                    let (rx, ry, rz) = index_to_xyz(idx);
                    assert_eq!((rx, ry, rz), (x, y, z), "roundtrip failed for ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn all_indices_unique() {
        let mut seen = std::collections::HashSet::new();
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    assert!(seen.insert(voxel_index(x, y, z)));
                }
            }
        }
        assert_eq!(seen.len(), 512);
    }

    // ── Occupancy ──

    #[test]
    fn empty_brick_has_no_solids() {
        let geo = BrickGeometry::new();
        assert_eq!(geo.solid_count(), 0);
        assert!(geo.is_fully_empty());
        assert!(!geo.is_fully_solid());
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    assert!(!geo.is_solid(x, y, z));
                }
            }
        }
    }

    #[test]
    fn fully_solid_brick() {
        let geo = BrickGeometry::fully_solid();
        assert_eq!(geo.solid_count(), 512);
        assert!(geo.is_fully_solid());
        assert!(!geo.is_fully_empty());
        for z in 0..8u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    assert!(geo.is_solid(x, y, z));
                }
            }
        }
    }

    #[test]
    fn set_solid_and_clear() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(3, 4, 5, true);
        assert!(geo.is_solid(3, 4, 5));
        assert_eq!(geo.solid_count(), 1);

        geo.set_solid(3, 4, 5, false);
        assert!(!geo.is_solid(3, 4, 5));
        assert_eq!(geo.solid_count(), 0);
    }

    #[test]
    fn set_solid_multiple() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(0, 0, 0, true);
        geo.set_solid(7, 7, 7, true);
        geo.set_solid(3, 3, 3, true);
        assert_eq!(geo.solid_count(), 3);
        assert!(!geo.is_fully_empty());
        assert!(!geo.is_fully_solid());
    }

    // ── Surface identification ──

    #[test]
    fn single_solid_voxel_is_surface() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(4, 4, 4, true);
        assert!(geo.is_surface_voxel(4, 4, 4));
    }

    #[test]
    fn interior_voxel_not_surface() {
        let mut geo = BrickGeometry::fully_solid();
        // Voxel (4,4,4) is surrounded by solid on all 6 sides within the brick
        // But boundary voxels (face touching edge) ARE surface with the default
        // "out-of-brick = empty" convention.
        // So (4,4,4) with all neighbors solid should NOT be surface.
        assert!(!geo.is_surface_voxel(4, 4, 4));
    }

    #[test]
    fn edge_voxel_of_solid_brick_is_surface() {
        let geo = BrickGeometry::fully_solid();
        // Edge voxel has out-of-brick neighbor → treated as empty → is surface
        assert!(geo.is_surface_voxel(0, 4, 4));
        assert!(geo.is_surface_voxel(7, 4, 4));
        assert!(geo.is_surface_voxel(4, 0, 4));
        assert!(geo.is_surface_voxel(4, 7, 4));
        assert!(geo.is_surface_voxel(4, 4, 0));
        assert!(geo.is_surface_voxel(4, 4, 7));
    }

    #[test]
    fn empty_neighbor_of_solid_is_surface() {
        let mut geo = BrickGeometry::new();
        // 2x2x2 solid block at (3,3,3)-(4,4,4)
        for z in 3..=4u8 {
            for y in 3..=4u8 {
                for x in 3..=4u8 {
                    geo.set_solid(x, y, z, true);
                }
            }
        }
        // All 8 voxels should be surface (each has at least one empty neighbor)
        for z in 3..=4u8 {
            for y in 3..=4u8 {
                for x in 3..=4u8 {
                    assert!(geo.is_surface_voxel(x, y, z), "({x},{y},{z}) should be surface");
                }
            }
        }
        // Empty neighbors of solid voxels are also surface
        assert!(geo.is_surface_voxel(2, 3, 3)); // -X neighbor of (3,3,3)
        assert!(geo.is_surface_voxel(5, 4, 4)); // +X neighbor of (4,4,4)
    }

    #[test]
    fn empty_voxel_far_from_solid_not_surface() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(0, 0, 0, true);
        // (7,7,7) is far from (0,0,0) — all its neighbors are empty → not surface
        assert!(!geo.is_surface_voxel(7, 7, 7));
    }

    // ── Rebuild surface list ──

    #[test]
    fn rebuild_surface_list_empty_brick() {
        let mut geo = BrickGeometry::new();
        geo.rebuild_surface_list();
        assert_eq!(geo.surface_count(), 0);
    }

    #[test]
    fn rebuild_surface_list_single_voxel() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(4, 4, 4, true);
        geo.rebuild_surface_list();
        // The solid voxel + its 6 empty neighbors = 7 surface voxels
        assert_eq!(geo.surface_count(), 7);
        // The solid voxel itself should be in the list
        let solid_idx = voxel_index(4, 4, 4);
        assert!(geo.get_surface_voxel(solid_idx).is_some());
    }

    #[test]
    fn rebuild_surface_list_fully_solid() {
        let mut geo = BrickGeometry::fully_solid();
        geo.rebuild_surface_list();
        // Surface = shell of the 8³ cube (within-brick boundary detection treats
        // out-of-brick as empty)
        // Shell count = 512 - 6³ = 512 - 216 = 296
        assert_eq!(geo.surface_count(), 296);
    }

    #[test]
    fn rebuild_preserving_keeps_old_colors() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(4, 4, 4, true);
        geo.rebuild_surface_list();

        // Paint the solid voxel red
        let idx = voxel_index(4, 4, 4);
        geo.get_surface_voxel_mut(idx).unwrap().color = [255, 0, 0, 255];

        // Add another solid voxel and rebuild preserving
        geo.set_solid(4, 4, 5, true);
        geo.rebuild_surface_list_preserving();

        // Old voxel should keep its red color
        let sv = geo.get_surface_voxel(idx).unwrap();
        assert_eq!(sv.color, [255, 0, 0, 255]);

        // New voxel should have default white
        let new_idx = voxel_index(4, 4, 5);
        let new_sv = geo.get_surface_voxel(new_idx).unwrap();
        assert_eq!(new_sv.color, [255, 255, 255, 255]);
    }

    // ── Cross-brick neighbor context ──

    #[test]
    fn neighbor_context_empty_neighbors() {
        let geo = BrickGeometry::fully_solid();
        let ctx = NeighborContext {
            center: &geo,
            neighbors: [None; 6], // all EMPTY_SLOT
        };

        // Interior voxel should NOT be surface (all 6 neighbors within brick are solid)
        assert!(!ctx.is_surface_voxel(4, 4, 4));
        // Edge voxel has out-of-brick neighbor → EMPTY_SLOT → empty → surface
        assert!(ctx.is_surface_voxel(0, 4, 4));
    }

    #[test]
    fn neighbor_context_interior_neighbors() {
        let geo = BrickGeometry::fully_solid();
        let ctx = NeighborContext {
            center: &geo,
            neighbors: [Some(None); 6], // all INTERIOR_SLOT = all solid
        };

        // All voxels should have solid neighbors → no surface voxels
        assert!(!ctx.is_surface_voxel(0, 0, 0));
        assert!(!ctx.is_surface_voxel(4, 4, 4));
        assert!(!ctx.is_surface_voxel(7, 7, 7));
    }

    #[test]
    fn neighbor_context_allocated_neighbor() {
        let mut center = BrickGeometry::new();
        center.set_solid(0, 4, 4, true); // solid at x=0 face

        let mut neighbor_neg_x = BrickGeometry::new();
        neighbor_neg_x.set_solid(7, 4, 4, true); // solid at x=7 (touching center's x=0)

        let ctx = NeighborContext {
            center: &center,
            neighbors: [
                Some(Some(&neighbor_neg_x)), // -X
                None, None, None, None, None,
            ],
        };

        // Center (0,4,4) has a solid neighbor at -X → still surface because other neighbors empty
        assert!(ctx.is_surface_voxel(0, 4, 4));
    }

    #[test]
    fn neighbor_context_rebuild_surface_list() {
        let geo = BrickGeometry::fully_solid();
        let ctx = NeighborContext {
            center: &geo,
            neighbors: [Some(None); 6], // all INTERIOR_SLOT
        };

        // With all interior neighbors, a fully solid brick has NO surface voxels
        let surface = ctx.rebuild_surface_list();
        assert_eq!(surface.len(), 0);
    }

    #[test]
    fn neighbor_context_partial_neighbor() {
        let geo = BrickGeometry::fully_solid();
        let ctx = NeighborContext {
            center: &geo,
            neighbors: [
                Some(None),  // -X: INTERIOR
                Some(None),  // +X: INTERIOR
                Some(None),  // -Y: INTERIOR
                Some(None),  // +Y: INTERIOR
                Some(None),  // -Z: INTERIOR
                None,        // +Z: EMPTY
            ],
        };

        // Only the z=7 face should have surface voxels
        assert!(!ctx.is_surface_voxel(4, 4, 4)); // interior
        assert!(ctx.is_surface_voxel(4, 4, 7));  // z=7 face → empty +Z neighbor
        assert!(!ctx.is_surface_voxel(4, 4, 0)); // z=0 face → solid -Z neighbor
    }

    // ── Serialization ──

    #[test]
    fn surface_voxel_bytes_roundtrip() {
        let sv = SurfaceVoxel::new(42, [255, 128, 64, 200], 7);
        let bytes = sv.to_bytes();
        let sv2 = SurfaceVoxel::from_bytes(&bytes);
        assert_eq!(sv, sv2);
    }

    #[test]
    fn brick_geometry_bytes_roundtrip() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(3, 4, 5, true);
        geo.set_solid(0, 0, 0, true);
        geo.rebuild_surface_list();

        // Paint one surface voxel
        let idx = voxel_index(3, 4, 5);
        geo.get_surface_voxel_mut(idx).unwrap().color = [100, 200, 50, 255];
        geo.get_surface_voxel_mut(idx).unwrap().material_id = 3;

        let bytes = geo.to_bytes();
        let (geo2, consumed) = BrickGeometry::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(geo.occupancy, geo2.occupancy);
        assert_eq!(geo.surface_voxels.len(), geo2.surface_voxels.len());
        for (a, b) in geo.surface_voxels.iter().zip(geo2.surface_voxels.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn brick_geometry_bytes_empty() {
        let geo = BrickGeometry::new();
        let bytes = geo.to_bytes();
        assert_eq!(bytes.len(), 66); // 64 occupancy + 2 count
        let (geo2, consumed) = BrickGeometry::from_bytes(&bytes).unwrap();
        assert_eq!(consumed, 66);
        assert!(geo2.is_fully_empty());
        assert_eq!(geo2.surface_count(), 0);
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(BrickGeometry::from_bytes(&[0; 65]).is_none());
        assert!(BrickGeometry::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn from_bytes_truncated_surface() {
        let mut geo = BrickGeometry::new();
        geo.set_solid(4, 4, 4, true);
        geo.rebuild_surface_list();
        let bytes = geo.to_bytes();
        // Truncate the surface voxel data
        let truncated = &bytes[..bytes.len() - 1];
        assert!(BrickGeometry::from_bytes(truncated).is_none());
    }
}
