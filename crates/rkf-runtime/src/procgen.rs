//! Procedural chunk generation for testing the streaming system.
//!
//! Generates `.rkf` chunk files containing heightmap-based terrain SDF using
//! layered sine waves. No external noise crate is needed — the terrain is
//! built from simple trigonometric functions that produce gentle rolling hills.
//!
//! # Example
//! ```no_run
//! use rkf_runtime::procgen::{ProcgenConfig, generate_chunk, generate_world};
//! use glam::IVec3;
//! use std::path::Path;
//!
//! // Generate a single chunk
//! let chunk = generate_chunk(IVec3::ZERO, &ProcgenConfig::default());
//! assert!(chunk.brick_count > 0);
//!
//! // Generate a 4x4 grid of chunks
//! let results = generate_world(Path::new("/tmp/terrain"), 4, 4, &ProcgenConfig::default())
//!     .expect("failed to generate world");
//! assert_eq!(results.len(), 16);
//! ```

use std::path::{Path, PathBuf};

use glam::{IVec3, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::chunk::{save_chunk_file, Chunk, TierGrid};
use rkf_core::sdf::voxelize_sdf;
use rkf_core::world_position::CHUNK_SIZE;

/// Configuration for procedural chunk generation.
#[derive(Debug, Clone)]
pub struct ProcgenConfig {
    /// Resolution tier for generated chunks (0-3).
    pub tier: usize,
    /// Amplitude of terrain height variation in metres.
    pub amplitude: f32,
    /// Frequency of terrain undulation (higher = more hills).
    pub frequency: f32,
    /// Base height offset (Y-level of the "ground plane").
    pub base_height: f32,
}

impl Default for ProcgenConfig {
    fn default() -> Self {
        Self {
            tier: 2,        // 8cm voxels — reasonable for terrain
            amplitude: 2.0, // +/-2m hills
            frequency: 0.3, // gentle rolling hills
            base_height: 4.0, // ground at Y=4m (middle of chunk)
        }
    }
}

/// Procedural terrain SDF: heightmap using layered sine waves.
///
/// Returns signed distance from `point` to the terrain surface.
/// Negative below surface, positive above.
fn terrain_sdf(point: Vec3, config: &ProcgenConfig) -> f32 {
    // Simple multi-octave sine heightmap
    let height = config.base_height
        + config.amplitude * (point.x * config.frequency).sin()
        + config.amplitude * 0.5 * (point.z * config.frequency * 1.7).sin()
        + config.amplitude * 0.25 * ((point.x + point.z) * config.frequency * 2.3).sin();

    // SDF: distance above the heightmap surface
    point.y - height
}

/// Generate a single procedural chunk at the given coordinates.
///
/// Returns the chunk data (not saved to disk). The chunk contains a single
/// tier grid at the resolution specified by `config.tier`.
pub fn generate_chunk(coords: IVec3, config: &ProcgenConfig) -> Chunk {
    // Chunk world-space AABB
    let chunk_min = Vec3::new(
        coords.x as f32 * CHUNK_SIZE,
        coords.y as f32 * CHUNK_SIZE,
        coords.z as f32 * CHUNK_SIZE,
    );
    let chunk_max = chunk_min + Vec3::splat(CHUNK_SIZE);
    let aabb = Aabb::new(chunk_min, chunk_max);

    // Voxelize terrain SDF
    let (grid, bricks) = voxelize_sdf(|p| terrain_sdf(p, config), config.tier, &aabb);

    let brick_count = bricks.len() as u32;

    Chunk {
        coords,
        grids: vec![TierGrid {
            tier: config.tier as u8,
            grid,
            bricks,
        }],
        brick_count,
    }
}

/// Generate a grid of procedural chunks and save them to disk.
///
/// Creates an `extent_x` x `extent_z` grid of chunks at Y=0,
/// centered on the origin. Returns the paths of generated files.
///
/// # Parameters
/// - `output_dir`: directory to write `.rkf` files
/// - `extent_x`: number of chunks along X axis
/// - `extent_z`: number of chunks along Z axis
/// - `config`: procedural generation config
///
/// # Errors
/// Returns `std::io::Error` if directory creation or file writing fails.
pub fn generate_world(
    output_dir: &Path,
    extent_x: u32,
    extent_z: u32,
    config: &ProcgenConfig,
) -> Result<Vec<(IVec3, PathBuf)>, std::io::Error> {
    std::fs::create_dir_all(output_dir)?;

    let half_x = (extent_x / 2) as i32;
    let half_z = (extent_z / 2) as i32;

    let mut results = Vec::new();

    for x in -half_x..(extent_x as i32 - half_x) {
        for z in -half_z..(extent_z as i32 - half_z) {
            let coords = IVec3::new(x, 0, z);
            let chunk = generate_chunk(coords, config);

            let filename = format!("chunk_{}_{}_{}.rkf", coords.x, coords.y, coords.z);
            let path = output_dir.join(&filename);

            save_chunk_file(&chunk, &path)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            results.push((coords, path));
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::chunk::load_chunk_file;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create a unique temporary directory for each test to avoid collisions.
    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("rkf_procgen_{}_{pid}_{id}", prefix));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Clean up a temporary directory.
    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    // ------ ProcgenConfig ------

    #[test]
    fn default_config() {
        let config = ProcgenConfig::default();
        assert_eq!(config.tier, 2);
        assert!((config.amplitude - 2.0).abs() < f32::EPSILON);
        assert!((config.frequency - 0.3).abs() < f32::EPSILON);
        assert!((config.base_height - 4.0).abs() < f32::EPSILON);
    }

    // ------ terrain_sdf ------

    #[test]
    fn terrain_sdf_below_surface() {
        let config = ProcgenConfig::default();
        // Well below the base height — should be negative
        let point = Vec3::new(0.0, -10.0, 0.0);
        let d = terrain_sdf(point, &config);
        assert!(d < 0.0, "point well below surface should be negative, got {d}");
    }

    #[test]
    fn terrain_sdf_above_surface() {
        let config = ProcgenConfig::default();
        // Well above any possible terrain height — should be positive
        // Max height = base_height + amplitude + 0.5*amplitude + 0.25*amplitude = 4 + 3.5 = 7.5
        let point = Vec3::new(0.0, 20.0, 0.0);
        let d = terrain_sdf(point, &config);
        assert!(d > 0.0, "point well above surface should be positive, got {d}");
    }

    #[test]
    fn terrain_sdf_near_surface() {
        let config = ProcgenConfig::default();
        // At x=0, z=0: sin(0)=0 for all terms, so height = base_height = 4.0
        // Point at base_height should be zero distance
        let point = Vec3::new(0.0, config.base_height, 0.0);
        let d = terrain_sdf(point, &config);
        assert!(
            d.abs() < 0.01,
            "point at base height with x=z=0 should be near zero, got {d}"
        );
    }

    // ------ generate_chunk ------

    #[test]
    fn generate_single_chunk() {
        let config = ProcgenConfig::default();
        let chunk = generate_chunk(IVec3::ZERO, &config);
        assert_eq!(chunk.coords, IVec3::ZERO);
        assert!(
            chunk.brick_count > 0,
            "chunk at origin should have bricks (terrain passes through)"
        );
    }

    #[test]
    fn generate_chunk_at_offset() {
        let config = ProcgenConfig::default();
        let coords = IVec3::new(5, 0, 3);
        let chunk = generate_chunk(coords, &config);
        assert_eq!(chunk.coords, coords);
    }

    #[test]
    fn generate_chunk_has_tier_grid() {
        let config = ProcgenConfig::default();
        let chunk = generate_chunk(IVec3::ZERO, &config);
        assert_eq!(chunk.grids.len(), 1, "should have exactly one tier grid");
        assert_eq!(
            chunk.grids[0].tier, config.tier as u8,
            "tier should match config"
        );
    }

    // ------ generate_world ------

    #[test]
    fn generate_world_creates_files() {
        let dir = unique_temp_dir("creates_files");
        let config = ProcgenConfig::default();

        let results = generate_world(&dir, 3, 3, &config).expect("generate_world failed");

        assert_eq!(results.len(), 9, "3x3 grid should produce 9 chunks");

        // Verify all files exist on disk
        for (_, path) in &results {
            assert!(path.exists(), "file should exist: {}", path.display());
        }

        cleanup(&dir);
    }

    #[test]
    fn generate_world_returns_correct_coords() {
        let dir = unique_temp_dir("correct_coords");
        let config = ProcgenConfig::default();

        let results = generate_world(&dir, 3, 3, &config).expect("generate_world failed");

        // 3x3 centered: half_x=1, half_z=1 => x in -1..2, z in -1..2
        let coords: Vec<IVec3> = results.iter().map(|(c, _)| *c).collect();
        for x in -1..2_i32 {
            for z in -1..2_i32 {
                let expected = IVec3::new(x, 0, z);
                assert!(
                    coords.contains(&expected),
                    "missing expected coord {expected:?}, got: {coords:?}"
                );
            }
        }

        cleanup(&dir);
    }

    #[test]
    fn generated_chunk_saves_and_loads() {
        let dir = unique_temp_dir("save_load");
        let config = ProcgenConfig::default();

        let chunk = generate_chunk(IVec3::ZERO, &config);
        let path = dir.join("test_chunk.rkf");
        save_chunk_file(&chunk, &path)
            .expect("save failed");

        let loaded = load_chunk_file(&path).expect("load failed");

        assert_eq!(loaded.coords, chunk.coords);
        assert_eq!(loaded.brick_count, chunk.brick_count);
        assert_eq!(loaded.grids.len(), chunk.grids.len());
        assert_eq!(loaded.grids[0].tier, chunk.grids[0].tier);
        assert_eq!(loaded.grids[0].bricks.len(), chunk.grids[0].bricks.len());

        cleanup(&dir);
    }

    #[test]
    fn generate_world_centered_on_origin() {
        let dir = unique_temp_dir("centered");
        let config = ProcgenConfig::default();

        let results = generate_world(&dir, 4, 4, &config).expect("generate_world failed");

        assert_eq!(results.len(), 16, "4x4 grid should produce 16 chunks");

        // 4x4: half_x=2, half_z=2 => x in -2..2, z in -2..2
        let coords: Vec<IVec3> = results.iter().map(|(c, _)| *c).collect();
        for x in -2..2_i32 {
            for z in -2..2_i32 {
                let expected = IVec3::new(x, 0, z);
                assert!(
                    coords.contains(&expected),
                    "4x4 grid missing coord {expected:?}"
                );
            }
        }

        // Verify no coords outside expected range
        for (coord, _) in &results {
            assert!(coord.x >= -2 && coord.x < 2, "x out of range: {}", coord.x);
            assert!(coord.z >= -2 && coord.z < 2, "z out of range: {}", coord.z);
            assert_eq!(coord.y, 0, "y should always be 0");
        }

        cleanup(&dir);
    }
}
