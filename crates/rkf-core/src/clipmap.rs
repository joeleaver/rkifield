//! Clipmap LOD system — multi-level sparse grids for camera-relative detail.
//!
//! A clipmap organises the world into concentric LOD shells around the camera.
//! Each shell (level) uses a coarser voxel resolution and covers a larger radius,
//! so fine detail is only stored close to the camera while distant geometry uses
//! fewer bricks at lower resolution.
//!
//! # Structure
//!
//! - [`ClipmapLevel`] — configuration for a single LOD level (voxel size + radius).
//! - [`ClipmapConfig`] — ordered collection of LOD levels with validation.
//! - [`ClipmapGridSet`] — one [`SparseGrid`] per LOD level, all sharing the same
//!   brick pool.

use glam::UVec3;

use crate::constants::BRICK_DIM;
use crate::sparse_grid::SparseGrid;

/// Maximum number of clipmap LOD levels.
pub const MAX_CLIPMAP_LEVELS: usize = 5;

/// Default clipmap LOD configuration.
///
/// Based on the architecture doc:
/// - LOD 0:  2cm voxels,  128m radius — near detail
/// - LOD 1:  4cm voxels,  256m radius
/// - LOD 2:  8cm voxels,  512m radius
/// - LOD 3: 16cm voxels, 1024m radius
/// - LOD 4: 32cm voxels, 2048m radius — horizon
pub const DEFAULT_CLIPMAP_LEVELS: [ClipmapLevel; MAX_CLIPMAP_LEVELS] = [
    ClipmapLevel { voxel_size: 0.02, radius: 128.0 },
    ClipmapLevel { voxel_size: 0.04, radius: 256.0 },
    ClipmapLevel { voxel_size: 0.08, radius: 512.0 },
    ClipmapLevel { voxel_size: 0.16, radius: 1024.0 },
    ClipmapLevel { voxel_size: 0.32, radius: 2048.0 },
];

/// Configuration for a single clipmap LOD level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipmapLevel {
    /// Voxel edge length in meters for this LOD level.
    pub voxel_size: f32,
    /// Maximum rendering radius from camera in meters.
    pub radius: f32,
}

impl ClipmapLevel {
    /// Brick spatial extent in meters (`voxel_size × BRICK_DIM`).
    pub fn brick_extent(&self) -> f32 {
        self.voxel_size * BRICK_DIM as f32
    }
}

/// Configuration for the entire clipmap LOD system.
#[derive(Debug, Clone)]
pub struct ClipmapConfig {
    /// Ordered LOD levels (finest to coarsest).
    levels: Vec<ClipmapLevel>,
}

impl ClipmapConfig {
    /// Create a clipmap config with the given levels.
    ///
    /// Levels must be sorted by `voxel_size` ascending (finest first).
    /// Radii must also be ascending (each coarser level covers more distance).
    ///
    /// # Panics
    ///
    /// Panics if levels are empty, exceed [`MAX_CLIPMAP_LEVELS`], or are not sorted.
    pub fn new(levels: Vec<ClipmapLevel>) -> Self {
        assert!(!levels.is_empty(), "clipmap needs at least one level");
        assert!(levels.len() <= MAX_CLIPMAP_LEVELS, "too many clipmap levels");
        for i in 1..levels.len() {
            assert!(
                levels[i].voxel_size > levels[i - 1].voxel_size,
                "clipmap levels must have increasing voxel sizes"
            );
            assert!(
                levels[i].radius > levels[i - 1].radius,
                "clipmap levels must have increasing radii"
            );
        }
        Self { levels }
    }

    /// Create the default 5-level clipmap configuration.
    pub fn default_config() -> Self {
        Self { levels: DEFAULT_CLIPMAP_LEVELS.to_vec() }
    }

    /// Number of LOD levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get a specific LOD level.
    pub fn level(&self, index: usize) -> &ClipmapLevel {
        &self.levels[index]
    }

    /// Iterate over all levels.
    pub fn levels(&self) -> &[ClipmapLevel] {
        &self.levels
    }

    /// Replace a single level's configuration.
    ///
    /// Does **not** validate sort order — the caller is responsible for
    /// ensuring the levels remain sorted by voxel size and radius.
    pub fn set_level(&mut self, index: usize, level: ClipmapLevel) {
        self.levels[index] = level;
    }

    /// Determine which LOD level a given distance from camera falls into.
    ///
    /// Returns the finest level whose radius encompasses the distance.
    /// Returns the coarsest level index if distance exceeds all radii.
    pub fn level_for_distance(&self, distance: f32) -> usize {
        for (i, level) in self.levels.iter().enumerate() {
            if distance <= level.radius {
                return i;
            }
        }
        self.levels.len() - 1
    }
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// A set of sparse grids, one per clipmap LOD level.
///
/// Each grid covers its LOD level's spatial volume at the level's voxel resolution.
/// All grids share the same brick pool (brick data is allocated from the global pool).
#[derive(Debug)]
pub struct ClipmapGridSet {
    /// The clipmap configuration.
    config: ClipmapConfig,
    /// One sparse grid per LOD level.
    grids: Vec<SparseGrid>,
}

impl ClipmapGridSet {
    /// Create a new grid set from a configuration and per-level grid dimensions.
    ///
    /// `dimensions` must have exactly `config.num_levels()` entries,
    /// each specifying the grid size in cells for that LOD level.
    pub fn new(config: ClipmapConfig, dimensions: &[UVec3]) -> Self {
        assert_eq!(
            config.num_levels(),
            dimensions.len(),
            "must provide dimensions for each LOD level"
        );
        let grids = dimensions.iter().map(|&dims| SparseGrid::new(dims)).collect();
        Self { config, grids }
    }

    /// Create a grid set with dimensions calculated from the config.
    ///
    /// Each level's grid dimensions are computed as:
    /// `min(ceil(2 × radius / brick_extent), max_dim)` per axis.
    ///
    /// `max_dim` caps the grid size per axis to prevent multi-gigabyte allocations.
    /// In production, clipmap grids are sized by the streaming memory budget, not
    /// the full rendering radius. A typical value is 256–512 cells per axis.
    pub fn from_config(config: ClipmapConfig, max_dim: u32) -> Self {
        let dimensions: Vec<UVec3> = config
            .levels()
            .iter()
            .map(|level| {
                let diameter = 2.0 * level.radius;
                let brick_ext = level.brick_extent();
                let dim = ((diameter / brick_ext).ceil() as u32).min(max_dim);
                UVec3::new(dim, dim, dim)
            })
            .collect();
        Self::new(config, &dimensions)
    }

    /// Number of LOD levels.
    pub fn num_levels(&self) -> usize {
        self.grids.len()
    }

    /// Get the clipmap configuration.
    pub fn config(&self) -> &ClipmapConfig {
        &self.config
    }

    /// Get a specific LOD level's grid.
    pub fn grid(&self, level: usize) -> &SparseGrid {
        &self.grids[level]
    }

    /// Get a mutable reference to a specific LOD level's grid.
    pub fn grid_mut(&mut self, level: usize) -> &mut SparseGrid {
        &mut self.grids[level]
    }

    /// Iterate over all `(level_index, grid)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &SparseGrid)> {
        self.grids.iter().enumerate()
    }

    /// Replace one level's config and grid (for dynamic expansion).
    ///
    /// Swaps the grid and updates the config entry. The caller is responsible
    /// for ensuring the new grid's dimensions match the new config's radius.
    pub fn replace_level(&mut self, level: usize, new_config: ClipmapLevel, new_grid: SparseGrid) {
        self.config.set_level(level, new_config);
        self.grids[level] = new_grid;
    }

    /// Total cells across all levels.
    pub fn total_cells(&self) -> u64 {
        self.grids.iter().map(|g| g.total_cells() as u64).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ClipmapConfig ---

    #[test]
    fn default_config_has_5_levels() {
        let cfg = ClipmapConfig::default_config();
        assert_eq!(cfg.num_levels(), 5);
    }

    #[test]
    fn level_voxel_sizes_increase() {
        let cfg = ClipmapConfig::default_config();
        for i in 1..cfg.num_levels() {
            assert!(
                cfg.level(i).voxel_size > cfg.level(i - 1).voxel_size,
                "voxel_size not increasing at level {i}"
            );
        }
    }

    #[test]
    fn level_radii_increase() {
        let cfg = ClipmapConfig::default_config();
        for i in 1..cfg.num_levels() {
            assert!(
                cfg.level(i).radius > cfg.level(i - 1).radius,
                "radius not increasing at level {i}"
            );
        }
    }

    #[test]
    fn brick_extent_calculation() {
        // LOD 0: 0.02 * 8 = 0.16m
        let level = ClipmapLevel { voxel_size: 0.02, radius: 128.0 };
        let expected = 0.02 * BRICK_DIM as f32;
        assert!((level.brick_extent() - expected).abs() < 1e-6);
        // LOD 4: 0.32 * 8 = 2.56m
        let level4 = ClipmapLevel { voxel_size: 0.32, radius: 2048.0 };
        let expected4 = 0.32 * BRICK_DIM as f32;
        assert!((level4.brick_extent() - expected4).abs() < 1e-6);
    }

    #[test]
    fn level_for_distance_maps_correctly() {
        let cfg = ClipmapConfig::default_config();
        // d=50  → inside LOD 0 radius (128m)
        assert_eq!(cfg.level_for_distance(50.0), 0);
        // d=200 → beyond LOD 0 (128m), inside LOD 1 (256m)
        assert_eq!(cfg.level_for_distance(200.0), 1);
        // d=300 → beyond LOD 1 (256m), inside LOD 2 (512m)
        assert_eq!(cfg.level_for_distance(300.0), 2);
        // d=5000 → beyond all radii → coarsest (LOD 4)
        assert_eq!(cfg.level_for_distance(5000.0), 4);
    }

    #[test]
    fn level_for_distance_at_exact_boundary() {
        let cfg = ClipmapConfig::default_config();
        // Exactly at LOD 0 radius boundary
        assert_eq!(cfg.level_for_distance(128.0), 0);
        // Just over LOD 0 radius
        assert_eq!(cfg.level_for_distance(128.001), 1);
    }

    /// Helper: small 2-level config for testing without OOM.
    fn small_test_config() -> ClipmapConfig {
        ClipmapConfig::new(vec![
            ClipmapLevel { voxel_size: 0.02, radius: 4.0 },
            ClipmapLevel { voxel_size: 0.08, radius: 16.0 },
        ])
    }

    #[test]
    fn grid_set_from_config_dimensions() {
        let cfg = small_test_config();
        let set = ClipmapGridSet::from_config(cfg.clone(), 1024);
        assert_eq!(set.num_levels(), 2);

        // LOD 0: diameter=8m, brick_extent=0.02*8=0.16m → ceil(8/0.16)=50
        let expected_lod0_dim = ((2.0 * 4.0_f32) / (0.02 * 8.0)).ceil() as u32;
        assert_eq!(set.grid(0).dimensions().x, expected_lod0_dim);
        assert_eq!(set.grid(0).dimensions().y, expected_lod0_dim);
        assert_eq!(set.grid(0).dimensions().z, expected_lod0_dim);

        // LOD 1: diameter=32m, brick_extent=0.08*8=0.64m → ceil(32/0.64)=50
        let expected_lod1_dim = ((2.0 * 16.0_f32) / (0.08 * 8.0)).ceil() as u32;
        assert_eq!(set.grid(1).dimensions().x, expected_lod1_dim);
    }

    #[test]
    fn grid_set_from_config_with_max_dim_cap() {
        let cfg = ClipmapConfig::default_config();
        let max_dim = 64;
        let set = ClipmapGridSet::from_config(cfg, max_dim);
        assert_eq!(set.num_levels(), 5);
        // All grids should be capped at max_dim
        for i in 0..5 {
            assert!(set.grid(i).dimensions().x <= max_dim);
            assert!(set.grid(i).dimensions().y <= max_dim);
            assert!(set.grid(i).dimensions().z <= max_dim);
        }
    }

    #[test]
    fn grid_set_total_cells() {
        let cfg = small_test_config();
        let set = ClipmapGridSet::from_config(cfg, 1024);
        let mut expected: u64 = 0;
        for i in 0..set.num_levels() {
            let d = set.grid(i).dimensions();
            expected += (d.x as u64) * (d.y as u64) * (d.z as u64);
        }
        assert_eq!(set.total_cells(), expected);
        assert!(set.total_cells() > 0);
    }

    #[test]
    fn grid_set_iter_covers_all_levels() {
        let cfg = small_test_config();
        let set = ClipmapGridSet::from_config(cfg, 1024);
        let indices: Vec<usize> = set.iter().map(|(i, _)| i).collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "too many clipmap levels")]
    fn config_validation_panics_on_too_many_levels() {
        let levels = vec![
            ClipmapLevel { voxel_size: 0.01, radius: 10.0 },
            ClipmapLevel { voxel_size: 0.02, radius: 20.0 },
            ClipmapLevel { voxel_size: 0.04, radius: 40.0 },
            ClipmapLevel { voxel_size: 0.08, radius: 80.0 },
            ClipmapLevel { voxel_size: 0.16, radius: 160.0 },
            ClipmapLevel { voxel_size: 0.32, radius: 320.0 }, // 6th level — too many
        ];
        ClipmapConfig::new(levels);
    }

    #[test]
    #[should_panic(expected = "clipmap levels must have increasing voxel sizes")]
    fn config_validation_panics_on_unsorted_voxel_sizes() {
        let levels = vec![
            ClipmapLevel { voxel_size: 0.08, radius: 128.0 },
            ClipmapLevel { voxel_size: 0.02, radius: 256.0 }, // smaller than previous — bad
        ];
        ClipmapConfig::new(levels);
    }

    #[test]
    #[should_panic(expected = "clipmap needs at least one level")]
    fn config_validation_panics_on_empty() {
        ClipmapConfig::new(vec![]);
    }

    #[test]
    fn single_level_config() {
        let cfg = ClipmapConfig::new(vec![ClipmapLevel { voxel_size: 0.02, radius: 100.0 }]);
        assert_eq!(cfg.num_levels(), 1);
        // Any distance maps to level 0
        assert_eq!(cfg.level_for_distance(0.0), 0);
        assert_eq!(cfg.level_for_distance(50.0), 0);
        assert_eq!(cfg.level_for_distance(999.0), 0);
    }

    #[test]
    fn grid_set_mut_access() {
        let cfg = small_test_config();
        let mut set = ClipmapGridSet::from_config(cfg, 1024);
        // Should be able to mutably access and set a cell state
        let grid = set.grid_mut(0);
        grid.set_cell_state(0, 0, 0, crate::cell_state::CellState::Surface);
        assert_eq!(set.grid(0).cell_state(0, 0, 0), crate::cell_state::CellState::Surface);
    }

    #[test]
    fn replace_level_swaps_grid_and_config() {
        let cfg = small_test_config();
        let mut set = ClipmapGridSet::from_config(cfg, 1024);
        let old_dims = set.grid(0).dimensions();

        // Replace level 0 with a larger grid and updated config
        let new_config = ClipmapLevel { voxel_size: 0.02, radius: 8.0 };
        let new_dims = UVec3::splat(100);
        let new_grid = SparseGrid::new(new_dims);
        set.replace_level(0, new_config, new_grid);

        assert_eq!(set.grid(0).dimensions(), new_dims);
        assert_ne!(set.grid(0).dimensions(), old_dims);
        assert!((set.config().level(0).radius - 8.0).abs() < 1e-6);
        // Level 1 should be untouched
        assert!((set.config().level(1).radius - 16.0).abs() < 1e-6);
    }

    #[test]
    fn multi_level_population() {
        use crate::aabb::Aabb;
        use crate::brick_pool::Pool;
        use crate::constants::RESOLUTION_TIERS;
        use crate::populate::populate_grid_with_material;
        use crate::sdf::sphere_sdf;
        use glam::Vec3;

        // 2-level clipmap: level 0 → tier 1 (2cm), level 1 → tier 2 (8cm)
        let config = ClipmapConfig::new(vec![
            ClipmapLevel { voxel_size: 0.02, radius: 2.0 },
            ClipmapLevel { voxel_size: 0.08, radius: 8.0 },
        ]);
        let mut grid_set = ClipmapGridSet::from_config(config.clone(), 64);
        let mut pool: crate::brick_pool::BrickPool = Pool::new(32768);

        let tiers = [1usize, 2usize];
        let mut level_bricks = vec![];

        for level_idx in 0..config.num_levels() {
            let level = config.level(level_idx);
            let tier = tiers[level_idx];
            let brick_ext = RESOLUTION_TIERS[tier].brick_extent;
            let dims = grid_set.grid(level_idx).dimensions();
            let half = level.radius;
            let aabb = Aabb::new(
                Vec3::splat(-half),
                Vec3::new(
                    -half + dims.x as f32 * brick_ext,
                    -half + dims.y as f32 * brick_ext,
                    -half + dims.z as f32 * brick_ext,
                ),
            );

            let grid = grid_set.grid_mut(level_idx);
            let count = populate_grid_with_material(
                &mut pool,
                grid,
                |p| sphere_sdf(Vec3::ZERO, 0.5, p),
                tier,
                &aabb,
                1,
            )
            .unwrap();
            level_bricks.push(count);
        }

        // Both levels should have bricks
        assert!(level_bricks[0] > 0, "level 0 should have bricks");
        assert!(level_bricks[1] > 0, "level 1 should have bricks");
        // Finer level should have more bricks for the same object
        assert!(
            level_bricks[0] > level_bricks[1],
            "finer level should have more bricks: {} vs {}",
            level_bricks[0],
            level_bricks[1]
        );
        // All bricks came from the shared pool
        let total: u32 = level_bricks.iter().sum();
        assert_eq!(pool.allocated_count(), total);
    }
}
