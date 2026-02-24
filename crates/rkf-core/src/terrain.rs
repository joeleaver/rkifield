//! Terrain node — grid-based SDF container for large ground surfaces.
//!
//! Unlike scene objects which use BVH traversal, [`TerrainNode`] uses a flat
//! HashMap keyed by grid coordinate for O(1) tile lookup by world position.
//! Tiles can be procedurally generated or sculpted.

use std::collections::HashMap;

use glam::{IVec2, Vec3};

use crate::aabb::Aabb;
use crate::scene_node::BrickMapHandle;

// ─── TerrainTile ─────────────────────────────────────────────────────────────

/// A single terrain tile — one grid cell of the terrain.
#[derive(Debug, Clone)]
pub struct TerrainTile {
    /// Grid coordinate of this tile.
    pub coord: IVec2,
    /// Brick map handle for this tile's voxel data.
    pub brick_map: Option<BrickMapHandle>,
    /// LOD level currently loaded (0 = finest).
    pub lod_level: usize,
    /// Whether this tile has been modified by sculpting.
    pub sculpted: bool,
    /// Height range for quick culling (min_y, max_y).
    pub height_range: (f32, f32),
}

// ─── TerrainLodTier ──────────────────────────────────────────────────────────

/// LOD tier definition for terrain.
#[derive(Debug, Clone)]
pub struct TerrainLodTier {
    /// Voxel size for this tier.
    pub voxel_size: f32,
    /// Maximum camera distance for this tier.
    pub max_distance: f32,
}

// ─── TerrainConfig ───────────────────────────────────────────────────────────

/// Configuration for terrain generation.
#[derive(Debug, Clone)]
pub struct TerrainConfig {
    /// World-space size of one tile edge (in meters).
    pub tile_size: f32,
    /// Voxel resolution at finest LOD.
    pub voxel_size: f32,
    /// Smooth-min blend radius between adjacent tiles (meters).
    pub blend_radius: f32,
    /// LOD tiers (finest first).
    pub lod_tiers: Vec<TerrainLodTier>,
    /// Material ID for terrain surface.
    pub material_id: u16,
    /// Maximum terrain height (for AABB computation).
    pub max_height: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            tile_size: 64.0,
            voxel_size: 0.08,
            blend_radius: 0.5,
            lod_tiers: vec![
                TerrainLodTier { voxel_size: 0.08, max_distance: 128.0 },
                TerrainLodTier { voxel_size: 0.32, max_distance: 512.0 },
                TerrainLodTier { voxel_size: 1.28, max_distance: 2048.0 },
            ],
            material_id: 1,
            max_height: 100.0,
        }
    }
}

// ─── TerrainNode ─────────────────────────────────────────────────────────────

/// Grid-based terrain container.
///
/// Tiles are stored in a `HashMap` keyed by (x, z) grid coordinates.
/// O(1) lookup by world position via [`TerrainNode::world_to_tile`].
///
/// When no brick data is loaded for a position, [`TerrainNode::evaluate_sdf`]
/// falls back to a procedural layered-sine-wave height field.
pub struct TerrainNode {
    /// Terrain configuration (tile size, LOD tiers, material).
    pub config: TerrainConfig,
    tiles: HashMap<IVec2, TerrainTile>,
}

impl TerrainNode {
    /// Create a new, empty terrain node with the given configuration.
    pub fn new(config: TerrainConfig) -> Self {
        Self {
            config,
            tiles: HashMap::new(),
        }
    }

    /// Convert a world-space XZ position to a tile grid coordinate.
    ///
    /// The Y component of `world_pos` is ignored — only X and Z determine the
    /// tile.  Negative positions floor toward negative infinity, so
    /// `(-0.1, 0, 0)` with `tile_size = 64` maps to tile `(-1, 0)`.
    pub fn world_to_tile(&self, world_pos: Vec3) -> IVec2 {
        let ts = self.config.tile_size;
        let tx = (world_pos.x / ts).floor() as i32;
        let tz = (world_pos.z / ts).floor() as i32;
        IVec2::new(tx, tz)
    }

    /// Convert a tile coordinate to its world-space axis-aligned bounding box.
    ///
    /// The Y axis spans `[0, max_height]`.  Tiles with negative heights are not
    /// supported by the current design — sculpting below zero requires explicit
    /// `height_range` tracking on the [`TerrainTile`].
    pub fn tile_aabb(&self, coord: IVec2) -> Aabb {
        let ts = self.config.tile_size;
        let min = Vec3::new(coord.x as f32 * ts, 0.0, coord.y as f32 * ts);
        let max = Vec3::new(
            (coord.x + 1) as f32 * ts,
            self.config.max_height,
            (coord.y + 1) as f32 * ts,
        );
        Aabb::new(min, max)
    }

    /// Get a reference to a tile by grid coordinate.
    pub fn get_tile(&self, coord: IVec2) -> Option<&TerrainTile> {
        self.tiles.get(&coord)
    }

    /// Get a mutable reference to a tile by grid coordinate.
    pub fn get_tile_mut(&mut self, coord: IVec2) -> Option<&mut TerrainTile> {
        self.tiles.get_mut(&coord)
    }

    /// Insert or replace a tile.  The tile's `coord` field is used as the key.
    pub fn set_tile(&mut self, tile: TerrainTile) {
        self.tiles.insert(tile.coord, tile);
    }

    /// Remove a tile by grid coordinate, returning it if it existed.
    pub fn remove_tile(&mut self, coord: IVec2) -> Option<TerrainTile> {
        self.tiles.remove(&coord)
    }

    /// Return the grid coordinates of all tiles whose world-space centers are
    /// within `range` metres of `center`.
    ///
    /// This is a conservative scan over the grid cells that overlap the
    /// bounding square of the circle, filtered by actual Euclidean distance on
    /// the XZ plane.
    pub fn tiles_in_range(&self, center: Vec3, range: f32) -> Vec<IVec2> {
        let ts = self.config.tile_size;
        // Candidate grid range (conservative bounding square).
        let min_tile = self.world_to_tile(Vec3::new(center.x - range, 0.0, center.z - range));
        let max_tile = self.world_to_tile(Vec3::new(center.x + range, 0.0, center.z + range));

        let mut result = Vec::new();
        for tx in min_tile.x..=max_tile.x {
            for tz in min_tile.y..=max_tile.y {
                let coord = IVec2::new(tx, tz);
                // Compute the world-space center of this tile.
                let tile_center_x = (tx as f32 + 0.5) * ts;
                let tile_center_z = (tz as f32 + 0.5) * ts;
                let dx = tile_center_x - center.x;
                let dz = tile_center_z - center.z;
                if dx * dx + dz * dz <= range * range {
                    result.push(coord);
                }
            }
        }
        result
    }

    /// Return `coord` and its 8 neighbors (up to 9 coordinates total), in
    /// row-major order (x varies fastest within each z row).
    ///
    /// All 9 cells are returned regardless of whether tiles are loaded in them;
    /// callers use this list to drive seam-blending queries.
    pub fn tile_with_neighbors(&self, coord: IVec2) -> Vec<IVec2> {
        let mut result = Vec::with_capacity(9);
        for dz in -1_i32..=1 {
            for dx in -1_i32..=1 {
                result.push(IVec2::new(coord.x + dx, coord.y + dz));
            }
        }
        result
    }

    /// Select the finest LOD tier whose `max_distance` exceeds
    /// `camera_distance`.
    ///
    /// Returns the index into `config.lod_tiers`.  If all tiers are exceeded
    /// (or if `lod_tiers` is empty) the last (coarsest) tier index is returned.
    pub fn select_lod(&self, camera_distance: f32) -> usize {
        let tiers = &self.config.lod_tiers;
        if tiers.is_empty() {
            return 0;
        }
        for (i, tier) in tiers.iter().enumerate() {
            if camera_distance <= tier.max_distance {
                return i;
            }
        }
        tiers.len() - 1
    }

    /// Number of currently loaded tiles.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// All loaded tile coordinates (order is unspecified).
    pub fn loaded_coords(&self) -> Vec<IVec2> {
        self.tiles.keys().copied().collect()
    }

    /// Evaluate terrain SDF at a world position (for collision / physics).
    ///
    /// Uses a procedural height-field fallback (layered sine waves) when no
    /// brick data is loaded for the queried position.
    ///
    /// Returns `(distance, material_id)` where distance is positive above the
    /// terrain surface and negative below it.
    pub fn evaluate_sdf(&self, world_pos: Vec3) -> (f32, u16) {
        let height = procedural_height(world_pos.x, world_pos.z);
        let distance = world_pos.y - height;
        (distance, self.config.material_id)
    }
}

// ─── Private helpers ─────────────────────────────────────────────────────────

/// Layered sine-wave procedural terrain height at (x, z).
///
/// Matches the legacy procgen terrain used in earlier engine phases so that
/// physics and collision queries produce consistent results with visual output.
fn procedural_height(x: f32, z: f32) -> f32 {
    let h1 = (x * 0.02).sin() * (z * 0.02).sin() * 20.0;
    let h2 = (x * 0.1).sin() * (z * 0.08).sin() * 3.0;
    h1 + h2
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_terrain() -> TerrainNode {
        TerrainNode::new(TerrainConfig::default())
    }

    fn make_tile(coord: IVec2) -> TerrainTile {
        TerrainTile {
            coord,
            brick_map: None,
            lod_level: 0,
            sculpted: false,
            height_range: (0.0, 100.0),
        }
    }

    // ── 1. default_config ─────────────────────────────────────────────────────

    #[test]
    fn default_config() {
        let cfg = TerrainConfig::default();
        assert!((cfg.tile_size - 64.0).abs() < 1e-6);
        assert!((cfg.voxel_size - 0.08).abs() < 1e-6);
        assert!((cfg.blend_radius - 0.5).abs() < 1e-6);
        assert_eq!(cfg.lod_tiers.len(), 3);
        assert_eq!(cfg.material_id, 1);
        assert!((cfg.max_height - 100.0).abs() < 1e-6);

        // Finest tier first
        assert!((cfg.lod_tiers[0].voxel_size - 0.08).abs() < 1e-6);
        assert!((cfg.lod_tiers[0].max_distance - 128.0).abs() < 1e-6);
        // Coarsest tier last
        assert!((cfg.lod_tiers[2].voxel_size - 1.28).abs() < 1e-6);
        assert!((cfg.lod_tiers[2].max_distance - 2048.0).abs() < 1e-6);
    }

    // ── 2. world_to_tile_origin ───────────────────────────────────────────────

    #[test]
    fn world_to_tile_origin() {
        let t = default_terrain();
        let coord = t.world_to_tile(Vec3::ZERO);
        assert_eq!(coord, IVec2::new(0, 0));
    }

    // ── 3. world_to_tile_positive ─────────────────────────────────────────────

    #[test]
    fn world_to_tile_positive() {
        let t = default_terrain(); // tile_size = 64
        // (65, 0, 130) → tile (1, 2)
        let coord = t.world_to_tile(Vec3::new(65.0, 0.0, 130.0));
        assert_eq!(coord, IVec2::new(1, 2));

        // Exactly at tile boundary (128.0 / 64.0 = 2.0) → tile 2
        let coord2 = t.world_to_tile(Vec3::new(128.0, 0.0, 0.0));
        assert_eq!(coord2, IVec2::new(2, 0));
    }

    // ── 4. world_to_tile_negative ─────────────────────────────────────────────

    #[test]
    fn world_to_tile_negative() {
        let t = default_terrain(); // tile_size = 64
        // (-0.1, 0, 0) → tile (-1, 0)
        let coord = t.world_to_tile(Vec3::new(-0.1, 0.0, 0.0));
        assert_eq!(coord, IVec2::new(-1, 0));

        // (-64.0, 0, -64.0) → tile (-1, -1)
        let coord2 = t.world_to_tile(Vec3::new(-64.0, 0.0, -64.0));
        assert_eq!(coord2, IVec2::new(-1, -1));

        // (-64.1, 0, 0) → tile (-2, 0)
        let coord3 = t.world_to_tile(Vec3::new(-64.1, 0.0, 0.0));
        assert_eq!(coord3, IVec2::new(-2, 0));
    }

    // ── 5. tile_aabb ─────────────────────────────────────────────────────────

    #[test]
    fn tile_aabb() {
        let t = default_terrain(); // tile_size=64, max_height=100
        let aabb = t.tile_aabb(IVec2::new(1, 2));
        // x: [64, 128], y: [0, 100], z: [128, 192]
        assert!((aabb.min.x - 64.0).abs() < 1e-5);
        assert!((aabb.min.y - 0.0).abs() < 1e-5);
        assert!((aabb.min.z - 128.0).abs() < 1e-5);
        assert!((aabb.max.x - 128.0).abs() < 1e-5);
        assert!((aabb.max.y - 100.0).abs() < 1e-5);
        assert!((aabb.max.z - 192.0).abs() < 1e-5);
    }

    // ── 6. set_and_get_tile ───────────────────────────────────────────────────

    #[test]
    fn set_and_get_tile() {
        let mut t = default_terrain();
        let tile = make_tile(IVec2::new(3, -2));
        t.set_tile(tile);

        let got = t.get_tile(IVec2::new(3, -2)).expect("tile should be present");
        assert_eq!(got.coord, IVec2::new(3, -2));
        assert!(!got.sculpted);
    }

    // ── 7. remove_tile ───────────────────────────────────────────────────────

    #[test]
    fn remove_tile() {
        let mut t = default_terrain();
        t.set_tile(make_tile(IVec2::new(0, 0)));
        assert!(t.get_tile(IVec2::new(0, 0)).is_some());

        let removed = t.remove_tile(IVec2::new(0, 0));
        assert!(removed.is_some());
        assert!(t.get_tile(IVec2::new(0, 0)).is_none());

        // Removing again should return None.
        assert!(t.remove_tile(IVec2::new(0, 0)).is_none());
    }

    // ── 8. tiles_in_range ────────────────────────────────────────────────────

    #[test]
    fn tiles_in_range() {
        let t = default_terrain(); // tile_size=64
        // Query from the world origin with a range of 40 m.
        // Tile (0,0) has centre at (32, 32); distance = sqrt(32²+32²) ≈ 45.25 > 40.
        // Only tiles whose *centres* are within 40 m qualify.
        // With tile_size=64 the nearest centre is at (32,32) ≈ 45.25 m → outside range.
        let result = t.tiles_in_range(Vec3::ZERO, 40.0);
        // The single tile (0,0) has its centre farther than 40 m — none qualify.
        assert!(result.is_empty(), "expected no tiles within 40 m, got {:?}", result);

        // Expand range to 50 m — tile (0,0) centre at ~45.25 m should be included.
        let result2 = t.tiles_in_range(Vec3::ZERO, 50.0);
        assert!(result2.contains(&IVec2::new(0, 0)), "tile (0,0) should be in range");
    }

    // ── 9. tile_with_neighbors ───────────────────────────────────────────────

    #[test]
    fn tile_with_neighbors() {
        let t = default_terrain();
        let neighbors = t.tile_with_neighbors(IVec2::new(0, 0));
        assert_eq!(neighbors.len(), 9);

        // All 9 cells of the 3×3 grid centred at (0,0) must be present.
        for dz in -1_i32..=1 {
            for dx in -1_i32..=1 {
                let expected = IVec2::new(dx, dz);
                assert!(
                    neighbors.contains(&expected),
                    "missing neighbor {:?}",
                    expected
                );
            }
        }
    }

    // ── 10. select_lod_close ─────────────────────────────────────────────────

    #[test]
    fn select_lod_close() {
        let t = default_terrain();
        // Distance 50 m < 128 m → finest tier (index 0).
        assert_eq!(t.select_lod(50.0), 0);
        // Distance exactly at tier 0 boundary.
        assert_eq!(t.select_lod(128.0), 0);
    }

    // ── 11. select_lod_far ───────────────────────────────────────────────────

    #[test]
    fn select_lod_far() {
        let t = default_terrain();
        // Distance 600 m: exceeds tier 0 (128) and tier 1 (512), within tier 2 (2048).
        assert_eq!(t.select_lod(600.0), 2);
        // Distance beyond all tiers → last index.
        assert_eq!(t.select_lod(10_000.0), 2);
    }

    // ── 12. evaluate_sdf_above_ground ─────────────────────────────────────────

    #[test]
    fn evaluate_sdf_above_ground() {
        let t = default_terrain();
        // At x=0, z=0: procedural_height(0,0) = sin(0)*sin(0)*20 + sin(0)*sin(0)*3 = 0.
        // Position (0, 5, 0) is 5 m above the surface → positive distance.
        let (dist, mat) = t.evaluate_sdf(Vec3::new(0.0, 5.0, 0.0));
        assert!(dist > 0.0, "expected positive distance above ground, got {dist}");
        assert!((dist - 5.0).abs() < 1e-5, "expected distance ~5, got {dist}");
        assert_eq!(mat, 1);
    }

    // ── 13. evaluate_sdf_below_ground ─────────────────────────────────────────

    #[test]
    fn evaluate_sdf_below_ground() {
        let t = default_terrain();
        // At x=0, z=0: height = 0. Position (0, -3, 0) is 3 m below the surface.
        let (dist, mat) = t.evaluate_sdf(Vec3::new(0.0, -3.0, 0.0));
        assert!(dist < 0.0, "expected negative distance below ground, got {dist}");
        assert!((dist + 3.0).abs() < 1e-5, "expected distance ~ -3, got {dist}");
        assert_eq!(mat, 1);
    }
}
