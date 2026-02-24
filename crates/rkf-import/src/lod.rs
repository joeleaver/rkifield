//! Multi-LOD generation for the v2 import pipeline.
//!
//! Generates multiple LOD levels for a voxelized object by calling
//! [`voxelize_mesh`] with progressively coarser voxel sizes. Each level is
//! stored as a [`BrickMapHandle`] into the shared allocator, paired with its
//! brick pool slots.
//!
//! # Typical usage
//!
//! ```ignore
//! let config = LodConfig::default();
//! let result = generate_lods(&mesh, &config, material_id, &mut pool, &mut map_alloc)?;
//! let save_levels = to_save_lod_levels(&result, &pool, &map_alloc);
//! save_object(&mut writer, &result.aabb, None, &[material_id], &save_levels)?;
//! ```

use rkf_core::aabb::Aabb;
use rkf_core::asset_file::SaveLodLevel;
use rkf_core::brick::Brick;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT};
use rkf_core::scene_node::BrickMapHandle;
use rkf_core::brick_pool::Pool;
use rkf_core::voxel::VoxelSample;

use crate::mesh::MeshData;
use crate::voxelize::{voxelize_mesh, VoxelizeResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for multi-LOD generation.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Voxel size for the finest LOD level (metres).
    ///
    /// Set to 0.0 to use [`auto_voxel_size`] based on the mesh's average edge
    /// length.
    pub finest_voxel_size: f32,

    /// Number of LOD levels to generate.
    ///
    /// Must be at least 1.
    pub num_levels: usize,

    /// Voxel size multiplier between successive LOD levels.
    ///
    /// A value of 4.0 means each level is 4× coarser than the previous one.
    pub scale_factor: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            finest_voxel_size: 0.0, // auto from mesh edge length
            num_levels: 3,
            scale_factor: 4.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// One generated LOD level — the result of voxelizing a mesh at a particular
/// resolution.
#[derive(Debug)]
pub struct GeneratedLod {
    /// World-space voxel size for this level.
    pub voxel_size: f32,
    /// Number of allocated bricks.
    pub brick_count: u32,
    /// Handle into the shared [`BrickMapAllocator`] for this level's map.
    pub handle: BrickMapHandle,
}

/// Result of generating all LOD levels for a mesh.
#[derive(Debug)]
pub struct LodGenerationResult {
    /// LOD levels, sorted finest-first (ascending voxel_size).
    pub levels: Vec<GeneratedLod>,
    /// Axis-aligned bounding box of the original mesh.
    pub aabb: Aabb,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the recommended finest voxel size from the mesh's average edge
/// length.
///
/// Uses `average_edge_length / 4` as a heuristic that places roughly 4 voxels
/// across each average edge. The result is clamped to [0.001, 0.5] metres.
pub fn auto_voxel_size(mesh: &MeshData) -> f32 {
    let avg = mesh.average_edge_length();
    if avg <= 0.0 {
        // Empty or degenerate mesh — return minimum meaningful size.
        return 0.001;
    }
    (avg / 4.0).clamp(0.001, 0.5)
}

/// Generate all LOD levels for a mesh.
///
/// Computes `config.num_levels` voxel sizes starting from the finest:
/// `finest`, `finest * scale_factor`, `finest * scale_factor²`, …
///
/// For each level, calls [`voxelize_mesh`] and allocates a [`BrickMapHandle`]
/// in `map_alloc`. Returns [`None`] if any voxelization fails (e.g. pool full)
/// or if `config.num_levels` is 0.
///
/// On success the returned [`LodGenerationResult::levels`] are sorted
/// finest-first (ascending `voxel_size`).
pub fn generate_lods(
    mesh: &MeshData,
    config: &LodConfig,
    material_id: u16,
    pool: &mut Pool<Brick>,
    map_alloc: &mut BrickMapAllocator,
) -> Option<LodGenerationResult> {
    if config.num_levels == 0 {
        return None;
    }

    let finest = if config.finest_voxel_size > 0.0 {
        config.finest_voxel_size
    } else {
        auto_voxel_size(mesh)
    };

    // Compute voxel sizes for each level.
    let voxel_sizes: Vec<f32> = (0..config.num_levels)
        .map(|i| finest * config.scale_factor.powi(i as i32))
        .collect();

    let mut levels: Vec<GeneratedLod> = Vec::with_capacity(config.num_levels);
    let mut shared_aabb: Option<Aabb> = None;

    for &voxel_size in &voxel_sizes {
        let VoxelizeResult {
            handle,
            brick_count,
            aabb,
            ..
        } = voxelize_mesh(mesh, voxel_size, material_id, pool, map_alloc)?;

        // All LODs share the same AABB (mesh bounds never change).
        if shared_aabb.is_none() {
            shared_aabb = Some(aabb);
        }

        levels.push(GeneratedLod {
            voxel_size,
            brick_count,
            handle,
        });
    }

    // Levels are already finest-first since we iterated in ascending order.
    Some(LodGenerationResult {
        levels,
        aabb: shared_aabb.unwrap_or_else(|| Aabb::new(mesh.bounds_min, mesh.bounds_max)),
    })
}

/// Convert a [`LodGenerationResult`] into [`SaveLodLevel`]s ready for
/// [`rkf_core::asset_file::save_object`].
///
/// Extracts the [`BrickMap`] from `map_alloc` for each level and collects the
/// corresponding brick data from `pool`. The returned `Vec` is sorted
/// finest-first (matching `result.levels`).
///
/// # Extraction strategy
///
/// The [`BrickMapAllocator`] stores a packed flat buffer. We rebuild a local
/// [`BrickMap`] by iterating all `(bx, by, bz)` coordinates in the handle's
/// dimensions and calling [`BrickMapAllocator::get_entry`] for each. Non-empty
/// slots are used to fetch brick data from `pool`.
pub fn to_save_lod_levels(
    result: &LodGenerationResult,
    pool: &Pool<Brick>,
    map_alloc: &BrickMapAllocator,
) -> Vec<SaveLodLevel> {
    result
        .levels
        .iter()
        .map(|lod| {
            let handle = &lod.handle;
            let dims = handle.dims;

            // Rebuild the BrickMap from the allocator.
            let mut brick_map = BrickMap::new(dims);
            let mut brick_data: Vec<[VoxelSample; 512]> = Vec::new();

            // Map from original pool slot to its position in brick_data.
            // We insert bricks in the order we discover non-empty slots so the
            // brick_data ordering matches the brick_map entries (SaveLodLevel
            // requirement: entries that are non-EMPTY correspond to brick_data
            // in the order of discovery).
            let mut slot_to_local: std::collections::HashMap<u32, u32> =
                std::collections::HashMap::new();

            for bz in 0..dims.z {
                for by in 0..dims.y {
                    for bx in 0..dims.x {
                        let slot = map_alloc.get_entry(handle, bx, by, bz).unwrap_or(EMPTY_SLOT);
                        if slot != EMPTY_SLOT {
                            let local_idx = *slot_to_local.entry(slot).or_insert_with(|| {
                                let idx = brick_data.len() as u32;
                                brick_data.push(pool.get(slot).voxels);
                                idx
                            });
                            brick_map.set(bx, by, bz, local_idx);
                        }
                        // EMPTY_SLOT entries remain EMPTY_SLOT (BrickMap::new fills with EMPTY_SLOT)
                    }
                }
            }

            SaveLodLevel {
                voxel_size: lod.voxel_size,
                brick_map,
                brick_data,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use rkf_core::brick_map::EMPTY_SLOT;

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Build a simple single-triangle mesh for testing.
    ///
    /// Triangle: (0,0,0), (1,0,0), (0,1,0) — 1 metre scale.
    fn make_test_mesh() -> MeshData {
        use crate::mesh::ImportMaterial;
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [0.8, 0.8, 0.8],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    /// Build an empty mesh.
    #[allow(dead_code)]
    fn make_empty_mesh() -> MeshData {
        use crate::mesh::ImportMaterial;
        MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
            material_indices: Vec::new(),
            materials: vec![ImportMaterial {
                name: "default".to_string(),
                base_color: [0.8, 0.8, 0.8],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    // ── auto_voxel_size ──────────────────────────────────────────────────────

    #[test]
    fn auto_voxel_size_reasonable() {
        let mesh = make_test_mesh();
        let vs = auto_voxel_size(&mesh);
        assert!(vs >= 0.001, "voxel_size {vs} below minimum 0.001");
        assert!(vs <= 0.5, "voxel_size {vs} above maximum 0.5");
    }

    #[test]
    fn auto_voxel_size_clamps() {
        // Very small mesh — edges ~0.001 m → avg/4 = 0.00025 → clamp to 0.001
        let small_mesh = MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.001, 0.0, 0.0),
                Vec3::new(0.0, 0.001, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(0.001, 0.001, 0.0),
        };
        let vs = auto_voxel_size(&small_mesh);
        assert!(
            (vs - 0.001).abs() < 1e-6,
            "expected clamp to 0.001, got {vs}"
        );

        // Very large mesh — edges ~100 m → avg/4 = 25 → clamp to 0.5
        let large_mesh = MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(100.0, 0.0, 0.0),
                Vec3::new(0.0, 100.0, 0.0),
            ],
            normals: vec![Vec3::Z; 3],
            uvs: Vec::new(),
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(100.0, 100.0, 0.0),
        };
        let vs = auto_voxel_size(&large_mesh);
        assert!(
            (vs - 0.5).abs() < 1e-6,
            "expected clamp to 0.5, got {vs}"
        );
    }

    // ── LodConfig::default ───────────────────────────────────────────────────

    #[test]
    fn default_config() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.num_levels, 3, "default num_levels should be 3");
        assert!(
            (cfg.scale_factor - 4.0).abs() < 1e-6,
            "default scale_factor should be 4.0"
        );
        // finest_voxel_size == 0 triggers auto mode
        assert_eq!(cfg.finest_voxel_size, 0.0);
    }

    // ── generate_lods ────────────────────────────────────────────────────────

    #[test]
    fn lod_levels_decrease_resolution() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.05,
            num_levels: 3,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");

        assert_eq!(
            result.levels.len(),
            3,
            "should have 3 levels"
        );

        // Levels are sorted finest-first (ascending voxel_size).
        for i in 0..result.levels.len() - 1 {
            assert!(
                result.levels[i].voxel_size < result.levels[i + 1].voxel_size,
                "levels[{i}].voxel_size ({}) should be finer than levels[{}].voxel_size ({})",
                result.levels[i].voxel_size,
                i + 1,
                result.levels[i + 1].voxel_size
            );
        }

        // Check expected voxel sizes.
        assert!(
            (result.levels[0].voxel_size - 0.05).abs() < 1e-6,
            "finest should be 0.05, got {}",
            result.levels[0].voxel_size
        );
        assert!(
            (result.levels[1].voxel_size - 0.2).abs() < 1e-6,
            "level 1 should be 0.2, got {}",
            result.levels[1].voxel_size
        );
        assert!(
            (result.levels[2].voxel_size - 0.8).abs() < 1e-6,
            "level 2 should be 0.8, got {}",
            result.levels[2].voxel_size
        );
    }

    #[test]
    fn coarser_lod_fewer_bricks() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.02,
            num_levels: 2,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(65536);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");

        assert_eq!(result.levels.len(), 2);

        let fine_bricks = result.levels[0].brick_count;
        let coarse_bricks = result.levels[1].brick_count;

        // The coarser level must use no more bricks than the finer level.
        // (Typically strictly fewer, but allow equal for degenerate cases.)
        assert!(
            coarse_bricks <= fine_bricks,
            "coarser level ({coarse_bricks} bricks) should have <= bricks than finer ({fine_bricks})"
        );
    }

    #[test]
    fn single_level() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.1,
            num_levels: 1,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed with 1 level");

        assert_eq!(result.levels.len(), 1);
        assert!(
            (result.levels[0].voxel_size - 0.1).abs() < 1e-6,
            "single level should have the configured voxel_size"
        );
    }

    #[test]
    fn generate_returns_none_on_pool_full() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.05,
            num_levels: 1,
            scale_factor: 4.0,
        };
        // Pool with capacity 0 — cannot allocate any bricks.
        let mut pool = Pool::<Brick>::new(0);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc);
        assert!(
            result.is_none(),
            "generate_lods should return None when pool is full"
        );
    }

    // ── to_save_lod_levels ───────────────────────────────────────────────────

    #[test]
    fn to_save_lod_levels_produces_correct_count() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.1,
            num_levels: 2,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");

        let save_levels = to_save_lod_levels(&result, &pool, &map_alloc);
        assert_eq!(
            save_levels.len(),
            result.levels.len(),
            "to_save_lod_levels should produce one SaveLodLevel per GeneratedLod"
        );
    }

    #[test]
    fn to_save_lod_levels_voxel_sizes_match() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.05,
            num_levels: 2,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");
        let save_levels = to_save_lod_levels(&result, &pool, &map_alloc);

        for (r#gen, save) in result.levels.iter().zip(save_levels.iter()) {
            assert!(
                (r#gen.voxel_size - save.voxel_size).abs() < 1e-6,
                "voxel_size mismatch: generated={}, saved={}",
                r#gen.voxel_size,
                save.voxel_size
            );
        }
    }

    #[test]
    fn to_save_lod_levels_brick_data_non_empty_for_valid_mesh() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.1,
            num_levels: 1,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");
        let save_levels = to_save_lod_levels(&result, &pool, &map_alloc);

        assert_eq!(save_levels.len(), 1);
        let save = &save_levels[0];
        // brick_count should match brick_data length.
        assert_eq!(
            save.brick_data.len(),
            result.levels[0].brick_count as usize,
            "brick_data length should match brick_count"
        );
    }

    #[test]
    fn to_save_lod_levels_brick_map_entries_valid() {
        let mesh = make_test_mesh();
        let config = LodConfig {
            finest_voxel_size: 0.1,
            num_levels: 1,
            scale_factor: 4.0,
        };
        let mut pool = Pool::<Brick>::new(4096);
        let mut map_alloc = BrickMapAllocator::new();

        let result = generate_lods(&mesh, &config, 1, &mut pool, &mut map_alloc)
            .expect("generate_lods should succeed");
        let save_levels = to_save_lod_levels(&result, &pool, &map_alloc);

        let save = &save_levels[0];
        let brick_count = save.brick_data.len() as u32;

        // Every non-empty brick map entry should be a valid local index.
        for &entry in &save.brick_map.entries {
            if entry != EMPTY_SLOT {
                assert!(
                    entry < brick_count,
                    "brick map entry {entry} out of range [0, {brick_count})"
                );
            }
        }
    }
}
