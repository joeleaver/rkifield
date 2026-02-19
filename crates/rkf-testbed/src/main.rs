use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use glam::{UVec3, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use rkf_core::aabb::Aabb;
use rkf_core::brick_pool::Pool;
use rkf_core::cell_state::CellState;
use rkf_core::clipmap::{ClipmapConfig, ClipmapGridSet, ClipmapLevel};
use rkf_core::constants::{BRICK_DIM, RESOLUTION_TIERS};
use rkf_core::populate::populate_grid_with_material;
use rkf_core::sdf::{box_sdf, capsule_sdf, smin, sphere_sdf};
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::voxel::VoxelSample;
use rkf_core::BrickPool;

use rkf_render::auto_exposure::AutoExposurePass;
use rkf_render::blit::BlitPass;
use rkf_render::bloom::BloomPass;
use rkf_render::bloom_composite::BloomCompositePass;
use rkf_render::camera::Camera;
use rkf_render::clipmap_gpu::ClipmapGpuData;
use rkf_render::color_grade::ColorGradePass;
use rkf_render::cosmetics::CosmeticsPass;
use rkf_render::dof::DofPass;
use rkf_render::gbuffer::GBuffer;
use rkf_render::gpu_color_pool::GpuColorPool;
use rkf_render::gpu_scene::{GpuScene, SceneUniforms};
use rkf_render::history::HistoryBuffers;
use rkf_render::light::{Light, LightBuffer};
use rkf_render::material_table::{self, MaterialTable};
use rkf_render::motion_blur::MotionBlurPass;
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::radiance_volume::RadianceVolume;
use rkf_render::ray_march::{RayMarchPass, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use rkf_render::shading::ShadingPass;
use rkf_render::sharpen::SharpenPass;
use rkf_render::tile_cull::TileCullPass;
use rkf_render::tone_map::ToneMapPass;
use rkf_render::upscale::UpscalePass;
use rkf_render::vol_shadow::VolShadowPass;
use rkf_render::vol_march::{VolMarchPass, VolMarchParams};
use rkf_render::vol_upscale::VolUpscalePass;
use rkf_render::vol_composite::VolCompositePass;
use rkf_render::fog::{FogSettings, FogParams};
use rkf_render::clouds::{CloudSettings, CloudParams};
use rkf_render::cloud_shadow::{CloudShadowPass, DEFAULT_CLOUD_SHADOW_EXTINCTION};
use rkf_render::vol_temporal::VolTemporalPass;
use rkf_render::RenderContext;

use rkf_runtime::frame::{execute_frame, FrameContext, FrameSettings};

mod automation;
use automation::{SharedState, TestbedAutomationApi};

// ---------------------------------------------------------------------------
// Phase 7 scene — lighting showcase (7 objects, all materials, 22+ lights)
// ---------------------------------------------------------------------------

/// Resolution tier for the test scene (Tier 2 = 8cm voxels).
const TEST_TIER: usize = 2;

/// Display (output) resolution width — the window size.
const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height — the window size.
const DISPLAY_HEIGHT: u32 = 720;

/// Create a large outdoor scene for volumetric fog / god ray validation.
///
/// Tier 2 (8cm voxels, 0.64m bricks) allows a 30m scene with manageable grid size.
/// Features:
/// - Large ground plane (14m × 14m)
/// - 8 tall stone pillars in two rows (avenue) for god-ray occlusion
/// - 2 horizontal lintels connecting pillar pairs at the top
/// - A tall monolith wall to create broad shadow regions
/// - 2 boulders for variety
#[allow(dead_code)]
fn create_volumetric_scene() -> (BrickPool, SparseGrid, Aabb) {
    let aabb = Aabb::new(Vec3::new(-15.0, -1.0, -15.0), Vec3::new(15.0, 8.0, 15.0));
    let res = &RESOLUTION_TIERS[TEST_TIER];
    let size = aabb.size();
    let dims = UVec3::new(
        ((size.x / res.brick_extent).ceil() as u32).max(1),
        ((size.y / res.brick_extent).ceil() as u32).max(1),
        ((size.z / res.brick_extent).ceil() as u32).max(1),
    );
    let pool_cap = if TEST_TIER <= 1 { 262144 } else { 65536 }; // 256K bricks for fine tiers
    let mut pool: BrickPool = Pool::new(pool_cap);
    let mut grid = SparseGrid::new(dims);
    let mut total = 0u32;

    // Ground plane — large flat surface, stone (material 1)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(7.0, 0.1, 7.0), p - Vec3::new(0.0, -0.5, 0.0)),
        TEST_TIER, &aabb, 1,
    ).expect("ground");

    // --- Pillar avenue: two rows of 4 pillars each, aligned along X axis ---
    // Left row (z = -1.5), pillars at x = -4, -1.5, 1.5, 4
    let pillar_positions = [
        Vec3::new(-4.0, 0.0, -1.5),
        Vec3::new(-1.5, 0.0, -1.5),
        Vec3::new( 1.5, 0.0, -1.5),
        Vec3::new( 4.0, 0.0, -1.5),
        // Right row (z = 1.5)
        Vec3::new(-4.0, 0.0, 1.5),
        Vec3::new(-1.5, 0.0, 1.5),
        Vec3::new( 1.5, 0.0, 1.5),
        Vec3::new( 4.0, 0.0, 1.5),
    ];

    for (i, base) in pillar_positions.iter().enumerate() {
        let bottom = Vec3::new(base.x, -0.4, base.z);
        let top = Vec3::new(base.x, 3.5, base.z);
        total += populate_grid_with_material(
            &mut pool, &mut grid,
            |p| capsule_sdf(bottom, top, 0.25, p),
            TEST_TIER, &aabb, 1, // stone
        ).expect(&format!("pillar {i}"));
    }

    // --- Lintels: horizontal beams connecting pillar pairs across the avenue ---
    // Lintel 1: connects pillars at x=-1.5 across z=-1.5 to z=1.5
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(0.3, 0.2, 1.8), p - Vec3::new(-1.5, 3.6, 0.0)),
        TEST_TIER, &aabb, 1,
    ).expect("lintel 1");

    // Lintel 2: connects pillars at x=1.5
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(0.3, 0.2, 1.8), p - Vec3::new(1.5, 3.6, 0.0)),
        TEST_TIER, &aabb, 1,
    ).expect("lintel 2");

    // --- Tall monolith wall — creates broad shadow region ---
    // Positioned to one side, perpendicular to sun direction
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(0.3, 2.5, 3.0), p - Vec3::new(-7.0, 2.0, 0.0)),
        TEST_TIER, &aabb, 2, // metal material for contrast
    ).expect("monolith wall");

    // --- Boulders for variety ---
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(3.0, 0.0, -3.5), 0.8, p),
        TEST_TIER, &aabb, 1,
    ).expect("boulder 1");

    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(-2.5, -0.1, 4.0), 0.6, p),
        TEST_TIER, &aabb, 1,
    ).expect("boulder 2");

    log::info!(
        "Volumetric scene: {total} bricks, grid {}x{}x{}, tier {TEST_TIER}",
        dims.x, dims.y, dims.z,
    );
    (pool, grid, aabb)
}

/// Apply material blending to existing voxels based on a position-to-weight function.
///
/// For each surface voxel in the grid, if `selector(world_pos)` returns `Some(weight)`,
/// sets the secondary_id and blend_weight on that voxel. Weight is 0.0–1.0.
fn apply_material_blend(
    pool: &mut BrickPool,
    grid: &SparseGrid,
    tier: usize,
    aabb: &Aabb,
    secondary_id: u8,
    selector: impl Fn(Vec3) -> Option<f32>,
) {
    let voxel_size = RESOLUTION_TIERS[tier].voxel_size;
    let brick_ext = RESOLUTION_TIERS[tier].brick_extent;
    let dims = grid.dimensions();

    for cz in 0..dims.z {
        for cy in 0..dims.y {
            for cx in 0..dims.x {
                if grid.cell_state(cx, cy, cz) != CellState::Surface {
                    continue;
                }
                let slot = match grid.brick_slot(cx, cy, cz) {
                    Some(s) => s,
                    None => continue,
                };
                let brick_min = aabb.min + Vec3::new(
                    cx as f32 * brick_ext,
                    cy as f32 * brick_ext,
                    cz as f32 * brick_ext,
                );
                let brick = pool.get_mut(slot);
                for vz in 0..BRICK_DIM {
                    for vy in 0..BRICK_DIM {
                        for vx in 0..BRICK_DIM {
                            let existing = brick.sample(vx, vy, vz);
                            // Only modify surface-adjacent voxels (small distance)
                            let dist = existing.distance_f32();
                            if dist.abs() > voxel_size * 2.0 {
                                continue;
                            }
                            let world_pos = brick_min + Vec3::new(
                                (vx as f32 + 0.5) * voxel_size,
                                (vy as f32 + 0.5) * voxel_size,
                                (vz as f32 + 0.5) * voxel_size,
                            );
                            if let Some(weight) = selector(world_pos) {
                                let w = (weight.clamp(0.0, 1.0) * 255.0) as u8;
                                brick.set(vx, vy, vz, VoxelSample::new(
                                    dist,
                                    existing.material_id(),
                                    w,
                                    secondary_id,
                                    existing.flags(),
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Create a multi-LOD clipmap scene for Phase 13 validation.
///
/// Two LOD levels sharing the same brick pool:
/// - Level 0: Tier 2 (8cm voxels, 0.64m bricks), 8m radius — fine detail
/// - Level 1: Tier 3 (32cm voxels, 2.56m bricks), 32m radius — distant/coarse
///
/// The same objects are populated at both resolutions. Near objects render
/// at fine resolution, distant regions fall back to the coarse level.
fn create_clipmap_scene() -> (BrickPool, ClipmapGridSet, ClipmapConfig, Aabb) {
    let config = ClipmapConfig::new(vec![
        ClipmapLevel { voxel_size: 0.08, radius: 8.0 },   // Level 0 = Tier 2
        ClipmapLevel { voxel_size: 0.32, radius: 32.0 },   // Level 1 = Tier 3
    ]);
    let mut grid_set = ClipmapGridSet::from_config(config.clone(), 64);
    let mut pool: BrickPool = Pool::new(65536);

    let tiers = [2usize, 3usize];

    // Pre-compute AABBs for both levels (avoids borrow conflict with grid_mut).
    let aabbs: Vec<Aabb> = (0..config.num_levels()).map(|i| {
        let level = config.level(i);
        let tier = tiers[i];
        let brick_ext = RESOLUTION_TIERS[tier].brick_extent;
        let dims = grid_set.grid(i).dimensions();
        let half = level.radius;
        Aabb::new(
            Vec3::new(-half, -half, -half),
            Vec3::new(
                -half + dims.x as f32 * brick_ext,
                -half + dims.y as f32 * brick_ext,
                -half + dims.z as f32 * brick_ext,
            ),
        )
    }).collect();

    // Shared pillar layout for both levels.
    let pillar_positions = [
        Vec3::new(-4.0, 0.0, -1.5), Vec3::new(-1.5, 0.0, -1.5),
        Vec3::new( 1.5, 0.0, -1.5), Vec3::new( 4.0, 0.0, -1.5),
        Vec3::new(-4.0, 0.0,  1.5), Vec3::new(-1.5, 0.0,  1.5),
        Vec3::new( 1.5, 0.0,  1.5), Vec3::new( 4.0, 0.0,  1.5),
    ];

    // === Level 0 (fine, 8cm voxels, 8m radius) — detailed nearby objects ===
    {
        let aabb = &aabbs[0];
        let grid = grid_set.grid_mut(0);
        let tier = tiers[0];

        // Ground — grass green (material 8)
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| box_sdf(Vec3::new(8.0, 0.5, 8.0), p - Vec3::new(0.0, -0.5, 0.0)),
            tier, aabb, 8);

        // Pillars + lintels + boulder — warm sandstone (material 1)
        // Pillars embedded into ground (bottom at y=-0.3) for seamless contact.
        let pillars = pillar_positions.clone();
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| {
                let mut d = f32::MAX;
                for base in &pillars {
                    let bottom = Vec3::new(base.x, -0.3, base.z);
                    let top = Vec3::new(base.x, 3.5, base.z);
                    d = d.min(capsule_sdf(bottom, top, 0.25, p));
                }
                // Lintels
                d = d.min(box_sdf(Vec3::new(0.3, 0.2, 1.8), p - Vec3::new(-1.5, 3.6, 0.0)));
                d = d.min(box_sdf(Vec3::new(0.3, 0.2, 1.8), p - Vec3::new( 1.5, 3.6, 0.0)));
                // Boulder
                d = d.min(sphere_sdf(Vec3::new(3.0, 0.2, -3.5), 0.8, p));
                d
            },
            tier, aabb, 1);

        // Monolith wall (material 2 — metal) — separate pass, different material
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| box_sdf(Vec3::new(0.3, 2.5, 3.0), p - Vec3::new(-7.0, 2.2, 0.0)),
            tier, aabb, 2);

        // === Material Blending Demos ===

        // 1) Blend sphere: sandstone (mat 1) base → gold (mat 11) top, height gradient
        let blend_sphere_center = Vec3::new(0.0, 1.5, -4.5);
        let blend_sphere_radius = 1.2;
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| sphere_sdf(blend_sphere_center, blend_sphere_radius, p),
            tier, aabb, 1);
        apply_material_blend(&mut pool, grid, tier, aabb, 11, |pos| {
            // Only within the sphere's bounding region
            let d = (pos - blend_sphere_center).length();
            if d > blend_sphere_radius + 0.2 { return None; }
            // Blend from 0 at bottom (y=0.3) to 1 at top (y=2.7)
            let t = ((pos.y - 0.3) / 2.4).clamp(0.0, 1.0);
            Some(t)
        });

        // 2) Dirt (mat 10) blended onto pillar bases — height gradient near ground
        apply_material_blend(&mut pool, grid, tier, aabb, 10, |pos| {
            // Only near pillar positions, blend dirt at base (y < 0.8)
            let near_pillar = pillar_positions.iter().any(|b| {
                let dx = pos.x - b.x;
                let dz = pos.z - b.z;
                (dx * dx + dz * dz).sqrt() < 0.6
            });
            if !near_pillar { return None; }
            let t = 1.0 - (pos.y / 0.8).clamp(0.0, 1.0);
            if t < 0.01 { None } else { Some(t) }
        });

        // 3) Gold (mat 11) blend on monolith wall top
        apply_material_blend(&mut pool, grid, tier, aabb, 11, |pos| {
            // Only within monolith AABB
            let center = Vec3::new(-7.0, 2.2, 0.0);
            let half = Vec3::new(0.5, 2.7, 3.2);
            if (pos.x - center.x).abs() > half.x
                || (pos.z - center.z).abs() > half.z { return None; }
            // Blend from 0 at y=3.0 to 1 at y=4.7
            let t = ((pos.y - 3.0) / 1.7).clamp(0.0, 1.0);
            if t < 0.01 { None } else { Some(t) }
        });

        log::info!("Level 0 (tier 2, 8cm): {} bricks", pool.allocated_count());
    }

    // === Level 1 (coarse, 32cm voxels, 32m radius) — extended terrain + distant objects ===
    let level0_count = pool.allocated_count();
    {
        let aabb = &aabbs[1];
        let grid = grid_set.grid_mut(1);
        let tier = tiers[1];

        // Material 8 (grass): ground + hills as single combined SDF
        // Hills use smooth-min to blend into ground naturally
        let blend_k = 1.0; // larger blend for coarse geometry
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| {
                // Ground plane (top at y=0, 0.5m thick, 30m extent)
                let mut d = box_sdf(Vec3::new(30.0, 0.5, 30.0), p - Vec3::new(0.0, -0.5, 0.0));
                // Distant hills merge smoothly into ground
                d = smin(d, sphere_sdf(Vec3::new(16.0, -1.0, -14.0), 4.0, p), blend_k);
                d = smin(d, sphere_sdf(Vec3::new(-18.0, -1.0, 12.0), 5.0, p), blend_k);
                d = smin(d, sphere_sdf(Vec3::new(10.0, -1.0, 20.0), 3.5, p), blend_k);
                d
            },
            tier, aabb, 8);

        // Material 2 (metal): distant structures as single combined SDF
        let _ = populate_grid_with_material(&mut pool, grid,
            |p| {
                let monolith = box_sdf(Vec3::new(1.5, 8.0, 1.5), p - Vec3::new(22.0, 4.0, -5.0));
                let wall = box_sdf(Vec3::new(8.0, 3.0, 0.5), p - Vec3::new(0.0, 1.5, -20.0));
                monolith.min(wall)
            },
            tier, aabb, 2);

        log::info!("Level 1 (tier 3, 32cm): {} bricks ({} new)",
            pool.allocated_count(), pool.allocated_count() - level0_count);
    }

    // Use the finest level's AABB for GpuScene (single-grid fallback uses this)
    let level0 = config.level(0);
    let half0 = level0.radius;
    let brick_ext0 = RESOLUTION_TIERS[tiers[0]].brick_extent;
    let dims0 = grid_set.grid(0).dimensions();
    let scene_aabb = Aabb::new(
        Vec3::new(-half0, -half0, -half0),
        Vec3::new(
            -half0 + dims0.x as f32 * brick_ext0,
            -half0 + dims0.y as f32 * brick_ext0,
            -half0 + dims0.z as f32 * brick_ext0,
        ),
    );

    log::info!(
        "Clipmap scene: {} levels, {} total bricks in shared pool",
        config.num_levels(), pool.allocated_count(),
    );
    (pool, grid_set, config, scene_aabb)
}

/// Clean overhead lighting for clipmap LOD validation.
/// White sun from above-right for clear shadows, warm fill from opposite side.
fn create_clipmap_lights() -> Vec<Light> {
    vec![
        // 0: Main sun — overhead from right-front, white, shadow-casting
        Light::directional(
            [0.5, 0.7, 0.3],       // above-right-front
            [1.0, 0.98, 0.95],     // near-white, very slight warm
            3.0,                    // strong intensity
            true,
        ),
        // 1: Fill from opposite side (no shadows) — prevents pure black
        Light::directional(
            [-0.3, 0.6, -0.5],    // above-left-back
            [0.6, 0.65, 0.8],     // cool fill
            0.4,                    // visible fill, not dim
            false,
        ),
    ]
}

/// Create a lighting showcase scene with multiple objects and materials.
///
/// Objects and their material IDs:
/// - Ground plane:         material 1 (stone)
/// - Center sphere:        material 2 (metal)
/// - Left sphere:          material 5 (skin/SSS)
/// - Right sphere:         material 3 (wood)
/// - Back-left capsule:    material 4 (emissive)
/// - Back-right box:       material 2 (metal)
/// - Front-center sphere:  material 1 (stone)
#[allow(dead_code)]
fn create_test_scene() -> (BrickPool, SparseGrid, Aabb) {
    let aabb = Aabb::new(Vec3::new(-3.0, -0.6, -3.0), Vec3::new(3.0, 1.5, 3.0));
    let res = &RESOLUTION_TIERS[TEST_TIER];
    let size = aabb.size();
    let dims = UVec3::new(
        ((size.x / res.brick_extent).ceil() as u32).max(1),
        ((size.y / res.brick_extent).ceil() as u32).max(1),
        ((size.z / res.brick_extent).ceil() as u32).max(1),
    );
    let mut pool: BrickPool = Pool::new(32768);
    let mut grid = SparseGrid::new(dims);
    let mut total = 0u32;

    // Ground plane — thin flat box, material 1 (stone)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(2.8, 0.05, 2.8), p - Vec3::new(0.0, -0.45, 0.0)),
        TEST_TIER, &aabb, 1,
    ).expect("ground");

    // Center sphere — material 2 (metal)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(0.0, 0.0, 0.0), 0.35, p),
        TEST_TIER, &aabb, 2,
    ).expect("center sphere");

    // Left sphere — material 5 (skin/SSS)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(-1.0, 0.0, 0.0), 0.3, p),
        TEST_TIER, &aabb, 5,
    ).expect("left sphere");

    // Right sphere — material 3 (wood)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(1.0, 0.0, 0.0), 0.3, p),
        TEST_TIER, &aabb, 3,
    ).expect("right sphere");

    // Back-left capsule — material 4 (emissive cyan)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| capsule_sdf(Vec3::new(-0.7, -0.2, -0.8), Vec3::new(-0.7, 0.3, -0.8), 0.12, p),
        TEST_TIER, &aabb, 4,
    ).expect("emissive capsule");

    // Back-right box — material 2 (metal)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::splat(0.22), p - Vec3::new(0.7, 0.0, -0.8)),
        TEST_TIER, &aabb, 2,
    ).expect("metal box");

    // Front-center small sphere — material 1 (stone)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(0.0, -0.2, 0.7), 0.18, p),
        TEST_TIER, &aabb, 1,
    ).expect("front sphere");

    log::info!(
        "Lighting showcase: {total} bricks, grid {}x{}x{}, tier {TEST_TIER}",
        dims.x, dims.y, dims.z,
    );
    (pool, grid, aabb)
}

/// Create the lighting showcase light set (22+ lights).
///
/// Mix of directional, point, and spot lights at various positions and colors.
/// Some cast shadows, some don't, to exercise the shadow budget system.
/// Light index 0 is a camera headlight updated per-frame.
#[allow(dead_code)]
fn create_showcase_lights(cam_pos: Vec3) -> Vec<Light> {
    vec![
        // 0: Camera headlight — updated per-frame
        Light::point([cam_pos.x, cam_pos.y + 0.3, cam_pos.z], [1.0, 0.98, 0.95], 1.0, 25.0, true),

        // --- Directional lights (2) ---
        // 1: Main sun — warm, shadow-casting
        Light::directional([0.4, 0.8, 0.3], [1.0, 0.95, 0.85], 0.8, true),
        // 2: Fill sky — cool blue, no shadows
        Light::directional([-0.2, 0.6, -0.5], [0.4, 0.5, 0.7], 0.25, false),

        // --- Point lights (12) ---
        // 3: Warm key above center
        Light::point([0.0, 1.2, 0.5], [1.0, 0.9, 0.7], 3.0, 4.0, true),
        // 4: Red accent left
        Light::point([-1.5, 0.5, 0.3], [1.0, 0.2, 0.1], 2.0, 3.0, true),
        // 5: Blue accent right
        Light::point([1.5, 0.5, 0.3], [0.1, 0.3, 1.0], 2.0, 3.0, true),
        // 6: Green ground bounce back
        Light::point([0.0, -0.1, -1.2], [0.2, 0.8, 0.3], 1.5, 3.0, false),
        // 7: Purple rim light
        Light::point([0.0, 0.8, -1.5], [0.6, 0.1, 0.8], 2.5, 4.0, true),
        // 8: Orange low left
        Light::point([-0.8, -0.2, 0.6], [1.0, 0.5, 0.1], 1.2, 2.5, false),
        // 9: Cyan low right
        Light::point([0.8, -0.2, 0.6], [0.1, 0.8, 0.9], 1.2, 2.5, false),
        // 10: White overhead far
        Light::point([0.0, 1.5, -0.5], [1.0, 1.0, 1.0], 4.0, 5.0, true),
        // 11: Yellow near ground front-left
        Light::point([-1.2, 0.1, 1.0], [1.0, 0.9, 0.3], 1.2, 2.5, false),
        // 12: Pink near ground front-right
        Light::point([1.2, 0.1, 1.0], [1.0, 0.3, 0.5], 1.2, 2.5, false),
        // 13: Dim white fill from below
        Light::point([0.0, -0.35, 0.0], [1.0, 1.0, 1.0], 0.8, 2.0, false),
        // 14: White back-center high
        Light::point([0.0, 1.3, -1.0], [1.0, 1.0, 0.95], 3.0, 4.5, true),

        // --- Spot lights (8) ---
        // 15: Spotlight on center sphere from above-front
        Light::spot([0.0, 1.5, 1.5], [0.0, -0.8, -0.6], [1.0, 1.0, 1.0], 5.0, 5.0, 0.15, 0.35, true),
        // 16: Red spot on left sphere
        Light::spot([-1.0, 1.2, 0.8], [0.0, -0.7, -0.5], [1.0, 0.15, 0.1], 4.0, 4.0, 0.2, 0.4, true),
        // 17: Blue spot on right sphere
        Light::spot([1.0, 1.2, 0.8], [0.0, -0.7, -0.5], [0.1, 0.2, 1.0], 4.0, 4.0, 0.2, 0.4, true),
        // 18: Green spot from behind on ground
        Light::spot([0.0, 0.8, -2.0], [0.0, -0.3, 0.9], [0.2, 1.0, 0.3], 3.0, 5.0, 0.1, 0.3, false),
        // 19: Warm narrow spot on back-right box
        Light::spot([0.7, 1.0, -0.3], [0.0, -1.0, -0.2], [1.0, 0.8, 0.5], 3.0, 3.0, 0.1, 0.2, true),
        // 20: Cool narrow spot on back-left capsule
        Light::spot([-0.7, 1.0, -0.3], [0.0, -1.0, -0.2], [0.5, 0.7, 1.0], 3.0, 3.0, 0.1, 0.2, true),
        // 21: Wide purple wash from side
        Light::spot([-2.0, 0.5, 0.0], [1.0, -0.2, 0.0], [0.5, 0.1, 0.8], 2.0, 5.0, 0.3, 0.6, false),
        // 22: Wide orange wash from other side
        Light::spot([2.0, 0.5, 0.0], [-1.0, -0.2, 0.0], [1.0, 0.5, 0.1], 2.0, 5.0, 0.3, 0.6, false),
    ]
}

// ---------------------------------------------------------------------------
// Phase 8 scene — Cornell box for GI validation
// ---------------------------------------------------------------------------

/// Create a Cornell box scene for GI validation.
///
/// Closed box (~2m per side) with colored walls to make color bleeding obvious:
/// - Floor, ceiling, back wall: white diffuse (material 6)
/// - Left wall: red diffuse (material 7)
/// - Right wall: green diffuse (material 8)
/// - Ceiling light panel: emissive (material 9)
/// - Interior box: metal (material 2)
/// - Interior sphere: stone (material 1)
#[allow(dead_code)]
fn create_cornell_box() -> (BrickPool, SparseGrid, Aabb) {
    let aabb = Aabb::new(Vec3::new(-1.5, -1.5, -1.5), Vec3::new(1.5, 1.5, 1.5));
    let res = &RESOLUTION_TIERS[TEST_TIER];
    let size = aabb.size();
    let dims = UVec3::new(
        ((size.x / res.brick_extent).ceil() as u32).max(1),
        ((size.y / res.brick_extent).ceil() as u32).max(1),
        ((size.z / res.brick_extent).ceil() as u32).max(1),
    );
    let mut pool: BrickPool = Pool::new(32768);
    let mut grid = SparseGrid::new(dims);
    let mut total = 0u32;

    let wall_thickness = 0.05;

    // Floor — white diffuse (material 6)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(1.0, wall_thickness, 1.0), p - Vec3::new(0.0, -1.0, 0.0)),
        TEST_TIER, &aabb, 6,
    ).expect("floor");

    // Ceiling — white diffuse (material 6)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(1.0, wall_thickness, 1.0), p - Vec3::new(0.0, 1.0, 0.0)),
        TEST_TIER, &aabb, 6,
    ).expect("ceiling");

    // Back wall — white diffuse (material 6)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(1.0, 1.0, wall_thickness), p - Vec3::new(0.0, 0.0, -1.0)),
        TEST_TIER, &aabb, 6,
    ).expect("back wall");

    // Left wall — red diffuse (material 7)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(wall_thickness, 1.0, 1.0), p - Vec3::new(-1.0, 0.0, 0.0)),
        TEST_TIER, &aabb, 7,
    ).expect("left wall");

    // Right wall — green diffuse (material 8)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(wall_thickness, 1.0, 1.0), p - Vec3::new(1.0, 0.0, 0.0)),
        TEST_TIER, &aabb, 8,
    ).expect("right wall");

    // Ceiling light panel — emissive (material 9)
    // Large panel on the ceiling for strong GI illumination
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::new(0.5, 0.03, 0.5), p - Vec3::new(0.0, 0.92, 0.0)),
        TEST_TIER, &aabb, 9,
    ).expect("ceiling light");

    // Interior box — metal (material 2), slightly rotated feel via offset
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| box_sdf(Vec3::splat(0.25), p - Vec3::new(-0.35, -0.7, -0.3)),
        TEST_TIER, &aabb, 2,
    ).expect("metal box");

    // Interior sphere — stone (material 1)
    total += populate_grid_with_material(
        &mut pool, &mut grid,
        |p| sphere_sdf(Vec3::new(0.35, -0.7, 0.2), 0.28, p),
        TEST_TIER, &aabb, 1,
    ).expect("stone sphere");

    log::info!(
        "Cornell box: {total} bricks, grid {}x{}x{}, tier {TEST_TIER}",
        dims.x, dims.y, dims.z,
    );
    (pool, grid, aabb)
}

/// Cornell box lighting — single overhead light representing the emissive panel.
/// This is the standard Cornell box setup: one area light on the ceiling.
/// The point light approximates the panel's emission as direct illumination
/// so the inject pass can compute how the panel lights nearby surfaces.
#[allow(dead_code)]
fn create_cornell_lights(cam_pos: Vec3) -> Vec<Light> {
    vec![
        // 0: Camera headlight — very dim, updated per-frame
        Light::point([cam_pos.x, cam_pos.y, cam_pos.z], [1.0, 0.98, 0.95], 0.1, 8.0, false),
        // 1: Ceiling panel light — represents the emissive panel's illumination
        // Positioned just below the panel surface, warm white, wide range
        Light::point([0.0, 0.85, 0.0], [1.0, 0.98, 0.92], 5.0, 5.0, false),
    ]
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct GpuState {
    context: RenderContext,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    scene: GpuScene,
    clipmap: ClipmapGpuData,
    clipmap_config: ClipmapConfig,
    gbuffer: GBuffer,
    material_table: MaterialTable,
    lights: Vec<Light>,
    light_buffer: LightBuffer,
    tile_cull: TileCullPass,
    radiance_volume: RadianceVolume,
    radiance_inject: RadianceInjectPass,
    radiance_mip: RadianceMipPass,
    color_pool: GpuColorPool,
    ray_march: RayMarchPass,
    shading: ShadingPass,
    vol_shadow: VolShadowPass,
    vol_march: VolMarchPass,
    vol_temporal: VolTemporalPass,
    vol_upscale: VolUpscalePass,
    vol_composite: VolCompositePass,
    fog_settings: FogSettings,
    cloud_shadow: CloudShadowPass,
    cloud_settings: CloudSettings,
    start_time: Instant,
    dof: DofPass,
    motion_blur: MotionBlurPass,
    history: HistoryBuffers,
    upscale: UpscalePass,
    sharpen: SharpenPass,
    bloom: BloomPass,
    bloom_composite: BloomCompositePass,
    auto_exposure: AutoExposurePass,
    tone_map: ToneMapPass,
    color_grade: ColorGradePass,
    cosmetics: CosmeticsPass,
    blit: BlitPass,
    camera: Camera,
    staging_buffer: wgpu::Buffer,
    shared_state: Arc<Mutex<SharedState>>,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
}

impl GpuState {
    fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .expect("failed to create surface");
        let context = RenderContext::new(&instance, &surface);
        let display_width = size.width.max(1);
        let display_height = size.height.max(1);
        let surface_format =
            context.configure_surface(&surface, display_width, display_height);

        // Scene — multi-LOD clipmap for Phase 13 validation
        let (pool, grid_set, clipmap_config, aabb) = create_clipmap_scene();
        let grid = grid_set.grid(0); // Finest level grid for GpuScene (single-grid fallback)

        // Update shared state with pool info
        {
            let mut state = shared_state.lock().unwrap();
            state.pool_capacity = pool.capacity() as u64;
            state.pool_allocated = pool.allocated_count() as u64;
        }

        // Camera — inside the pillar avenue looking down the avenue.
        // Offset from grid center; level 1 provides ground/objects beyond 8m.
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.yaw = 0.0;  // facing -Z (toward the avenue)
        camera.pitch = -0.15;  // slightly downward to see ground + pillars
        camera.fov_degrees = 70.0;

        let camera_uniforms = camera.uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, 0, [[0.0; 4]; 4]);
        let camera_bytes = bytemuck::bytes_of(&camera_uniforms);

        // SceneUniforms from finest-level grid
        let dims = grid.dimensions();
        let scene_uniforms = SceneUniforms {
            grid_dims: [dims.x, dims.y, dims.z, 0],
            grid_origin: [
                aabb.min.x,
                aabb.min.y,
                aabb.min.z,
                RESOLUTION_TIERS[TEST_TIER].brick_extent,
            ],
            params: [RESOLUTION_TIERS[TEST_TIER].voxel_size, 0.0, 0.0, 0.0],
        };

        // Upload scene to GPU
        let scene = GpuScene::upload(&context.device, &pool, grid, camera_bytes, &scene_uniforms);

        // Materials — override colors for clipmap validation
        let mut materials = material_table::create_test_materials();
        // Material 1: warm sandstone for pillars/structures
        materials[1].albedo = [0.72, 0.52, 0.35];
        materials[1].roughness = 0.75;
        // Material 8: natural grass green for ground
        materials[8].albedo = [0.22, 0.42, 0.12];
        materials[8].roughness = 0.95;
        materials[8].metallic = 0.0;
        // Material 10: dirt (earthy brown) — blend target for pillar bases
        materials[10].albedo = [0.4, 0.28, 0.15];
        materials[10].roughness = 0.95;
        materials[10].metallic = 0.0;
        // Material 11: gold — blend target for sphere top and monolith cap
        materials[11].albedo = [1.0, 0.84, 0.0];
        materials[11].roughness = 0.25;
        materials[11].metallic = 1.0;
        let material_table = MaterialTable::upload(&context.device, &materials);

        // G-buffer
        let gbuffer = GBuffer::new(&context.device, INTERNAL_WIDTH, INTERNAL_HEIGHT);

        // Clean overhead lighting for clipmap LOD validation
        let lights = create_clipmap_lights();
        log::info!("Volumetric scene: {} lights", lights.len());
        let light_buffer = LightBuffer::upload(&context.device, &lights);

        // Tile cull
        let tile_cull = TileCullPass::new(
            &context.device,
            &gbuffer,
            &light_buffer,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        // Radiance volume + GI passes
        let radiance_volume = RadianceVolume::new(&context.device);
        let radiance_inject = RadianceInjectPass::new(
            &context.device, &scene, &material_table, &light_buffer, &radiance_volume,
        );
        let radiance_mip = RadianceMipPass::new(&context.device, &radiance_volume);

        // Initialize inject uniforms with light count
        radiance_inject.update_inject_uniforms(
            &context.queue, lights.len() as u32, 2,
        );

        let color_pool = GpuColorPool::empty(&context.device);
        let clipmap = ClipmapGpuData::upload(&context.device, &grid_set, [0.0, 0.0, 0.0]);

        // Render passes
        let ray_march = RayMarchPass::new(&context.device, &scene, &gbuffer, &clipmap);
        let shading = ShadingPass::new(
            &context.device,
            &gbuffer,
            &material_table,
            &scene,
            &tile_cull.shade_light_bind_group_layout,
            &radiance_volume,
            &color_pool,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        // Phase 12: Volumetric pipeline
        let vol_shadow = VolShadowPass::new(&context.device, &context.queue, &scene.bind_group_layout);
        let cloud_shadow = CloudShadowPass::new(&context.device);

        let half_w = INTERNAL_WIDTH / 2;
        let half_h = INTERNAL_HEIGHT / 2;
        let vol_march = VolMarchPass::new(
            &context.device, &context.queue,
            &gbuffer.position_view, &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w, half_h,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        let vol_temporal = VolTemporalPass::new(
            &context.device,
            &vol_march.output_view,
            &gbuffer.motion_view,
            half_w, half_h,
        );

        let vol_upscale_pass = VolUpscalePass::new(
            &context.device,
            &vol_march.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
            half_w, half_h,
        );

        let vol_composite = VolCompositePass::new(
            &context.device,
            &shading.hdr_view, &vol_upscale_pass.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Fog disabled for clean shading validation
        let fog_settings = FogSettings {
            height_fog_enabled: false,
            fog_base_density: 0.0,
            fog_base_height: -0.3,
            fog_height_falloff: 0.12,
            fog_color: [0.9, 0.7, 0.5],
            distance_fog_enabled: false,
            fog_distance_density: 0.0,
            fog_distance_falloff: 0.02,
            ambient_dust_density: 0.0,
            ambient_dust_g: 0.82,
        };

        // Clouds disabled for clean shading validation
        let cloud_settings = CloudSettings {
            procedural_enabled: false,
            cloud_density_scale: 0.0,
            shadow_enabled: false,
            ..Default::default()
        };

        // Phase 10: DoF (pre-upscale) — reads composited HDR + gbuffer depth
        let dof = DofPass::new(
            &context.device, &vol_composite.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        // DoF disabled for clean shading validation (max_coc=0 = no blur)
        dof.update_focus(&context.queue, 5.0, 100.0, 0.0);

        // Phase 10: Motion blur (pre-upscale) — reads DoF output + gbuffer motion vectors
        let motion_blur = MotionBlurPass::new(
            &context.device, &dof.output_view, &gbuffer.motion_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        // Motion blur disabled for clean shading validation
        motion_blur.set_intensity(&context.queue, 0.0);

        // Phase 10: Bloom (pre-upscale) — extract bright pixels from motion-blurred HDR
        let bloom = BloomPass::new(
            &context.device, &motion_blur.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Phase 10: Temporal upscaling pipeline — reads from end of pre-upscale chain
        log::info!("Upscale backend: Custom Temporal ({}×{} → {}×{})",
            INTERNAL_WIDTH, INTERNAL_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT);

        let history = HistoryBuffers::new(
            &context.device, DISPLAY_WIDTH, DISPLAY_HEIGHT, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let upscale = UpscalePass::new(
            &context.device, &motion_blur.output_view, &gbuffer, &history,
            DISPLAY_WIDTH, DISPLAY_HEIGHT, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let sharpen = SharpenPass::new(
            &context.device, &upscale.output_view, &gbuffer,
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
        );

        // Phase 10: Bloom composite (post-upscale) — blend bloom onto sharpened HDR
        let bloom_composite = BloomCompositePass::new(
            &context.device, &sharpen.output_view,
            bloom.mip_views(),
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
        );

        // Phase 10: Auto-exposure — histogram-based exposure computation
        let auto_exposure = AutoExposurePass::new(
            &context.device, &bloom_composite.output_view,
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
        );

        // Tone map reads from bloom composite output, with auto-exposure buffer bound
        let tone_map = ToneMapPass::new_with_exposure(
            &context.device,
            &bloom_composite.output_view,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            Some(auto_exposure.get_exposure_buffer()),
        );

        // Phase 10: Color grading (identity LUT by default — passthrough)
        let color_grade = ColorGradePass::new(
            &context.device, &context.queue,
            &tone_map.ldr_view,
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
        );

        // Phase 10: Cosmetics — vignette, grain, chromatic aberration
        let cosmetics = CosmeticsPass::new(
            &context.device,
            &color_grade.output_view,
            DISPLAY_WIDTH, DISPLAY_HEIGHT,
        );
        // Post-processing disabled for clean shading validation
        bloom.set_threshold(&context.queue, 0.8, 0.4);
        bloom_composite.set_intensity(&context.queue, 0.0);
        cosmetics.set_vignette(&context.queue, 0.0);
        cosmetics.set_chromatic_aberration(&context.queue, 0.0);

        // Blit reads from final cosmetics output
        let blit = BlitPass::new(&context.device, &cosmetics.output_view, surface_format);

        // Staging buffer for CPU readback at display resolution
        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot staging"),
            size: (DISPLAY_WIDTH * DISPLAY_HEIGHT * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            context,
            surface,
            surface_format,
            width: display_width,
            height: display_height,
            scene,
            clipmap,
            clipmap_config,
            gbuffer,
            material_table,
            lights,
            light_buffer,
            tile_cull,
            radiance_volume,
            radiance_inject,
            radiance_mip,
            color_pool,
            ray_march,
            shading,
            vol_shadow,
            vol_march,
            vol_temporal,
            vol_upscale: vol_upscale_pass,
            vol_composite,
            fog_settings,
            cloud_shadow,
            cloud_settings,
            start_time: Instant::now(),
            dof,
            motion_blur,
            history,
            upscale,
            sharpen,
            bloom,
            bloom_composite,
            auto_exposure,
            tone_map,
            color_grade,
            cosmetics,
            blit,
            camera,
            staging_buffer,
            shared_state,
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.context
            .configure_surface(&self.surface, width, height);
    }

    fn update_camera(&mut self) {
        let mut uniforms =
            self.camera
                .uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, self.frame_index, self.prev_vp);
        // Disable sub-pixel jitter — spatial-only upscaling (Phase 9 decision)
        uniforms.jitter = [0.0, 0.0];

        // Store current VP as prev for next frame
        let vp = self.camera.view_projection(INTERNAL_WIDTH, INTERNAL_HEIGHT);
        self.prev_vp = vp.to_cols_array_2d();

        self.context.queue.write_buffer(
            &self.scene.camera_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn render(&mut self, dt: f32) {
        // Check for pending debug mode change from MCP
        if let Ok(mut state) = self.shared_state.lock() {
            if let Some(mode) = state.pending_debug_mode.take() {
                self.shading.set_debug_mode(&self.context.queue, mode);
                log::info!("Debug mode set via MCP: {mode}");
            }
            if let Some(cam) = state.pending_camera.take() {
                self.camera.position = cam.position;
                self.camera.yaw = cam.yaw;
                self.camera.pitch = cam.pitch;
                log::info!("Camera set via MCP: pos={:?} yaw={:.2} pitch={:.2}", cam.position, cam.yaw, cam.pitch);
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);

        // Update camera uniforms on GPU
        self.update_camera();

        // Update camera position for shading pass (correct view direction)
        let cam_pos = self.camera.position;
        self.shading
            .update_camera_pos(&self.context.queue, [cam_pos.x, cam_pos.y, cam_pos.z]);

        // No per-frame light updates needed — only directional lights

        // Update tile cull uniforms with camera data
        let cam_fwd = self.camera.forward();
        self.tile_cull.update_uniforms(
            &self.context.queue,
            self.light_buffer.count,
            [cam_pos.x, cam_pos.y, cam_pos.z],
            [cam_fwd.x, cam_fwd.y, cam_fwd.z],
        );

        // Update shade uniforms with light/tile info
        self.shading.update_light_info(
            &self.context.queue,
            self.light_buffer.count,
            self.tile_cull.num_tiles_x,
            4,
        );

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(self.width, self.height);
                return;
            }
            Err(e) => {
                log::error!("surface error: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // Update GI inject uniforms
        self.radiance_inject.update_inject_uniforms(
            &self.context.queue,
            self.lights.len() as u32,
            2,
        );

        // Update radiance volume center to camera position
        self.radiance_volume.update_center(
            &self.context.queue,
            [cam_pos.x, cam_pos.y, cam_pos.z],
        );

        // ── Pre-frame param updates for execute_frame ────────────────────────

        // Sun direction (matches directional light)
        let sun_dir = [0.9f32, 0.26, 0.15];
        let sun_len = (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]).sqrt();
        let sun_dir_n = [sun_dir[0] / sun_len, sun_dir[1] / sun_len, sun_dir[2] / sun_len];

        // Cloud params (time drives wind scrolling)
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let cloud_gpu = CloudParams::from_settings(&self.cloud_settings, elapsed);
        self.vol_march.set_cloud_params(&self.context.queue, &cloud_gpu);

        // Cloud shadow — pre-configure params (execute_frame calls dispatch_only)
        self.cloud_shadow.update_params_ex(
            &self.context.queue,
            [cam_pos.x, cam_pos.y, cam_pos.z], sun_dir_n,
            self.cloud_settings.cloud_min, self.cloud_settings.cloud_max,
            self.cloud_settings.shadow_coverage,
            DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.set_cloud_params(&self.context.queue, &cloud_gpu);

        // Volumetric march params
        let cam_right = self.camera.right();
        let cam_up_vec = self.camera.up();
        let fov_half_tan = (self.camera.fov_degrees.to_radians() * 0.5).tan();
        let aspect = INTERNAL_WIDTH as f32 / INTERNAL_HEIGHT as f32;
        let fog_params = FogParams::from_settings(&self.fog_settings);
        let half_range = rkf_render::DEFAULT_VOL_SHADOW_RANGE * 0.5;
        let half_height = rkf_render::DEFAULT_VOL_SHADOW_HEIGHT * 0.5;

        let vol_params = VolMarchParams {
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 0.0],
            cam_forward: [cam_fwd.x, cam_fwd.y, cam_fwd.z, 0.0],
            cam_right: [
                cam_right.x * fov_half_tan * aspect,
                cam_right.y * fov_half_tan * aspect,
                cam_right.z * fov_half_tan * aspect,
                0.0,
            ],
            cam_up: [
                cam_up_vec.x * fov_half_tan,
                cam_up_vec.y * fov_half_tan,
                cam_up_vec.z * fov_half_tan,
                0.0,
            ],
            sun_dir: [sun_dir_n[0], sun_dir_n[1], sun_dir_n[2], 0.0],
            sun_color: [1.0, 0.85, 0.6, 0.0],
            width: INTERNAL_WIDTH / 2,
            height: INTERNAL_HEIGHT / 2,
            full_width: INTERNAL_WIDTH,
            full_height: INTERNAL_HEIGHT,
            max_steps: 96,
            step_size: 0.5,
            near: 0.3,
            far: 60.0,
            fog_color: fog_params.fog_color,
            fog_height: fog_params.fog_height,
            fog_distance: fog_params.fog_distance,
            frame_index: self.frame_index,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            vol_shadow_min: [
                cam_pos.x - half_range, cam_pos.y - half_height,
                cam_pos.z - half_range, 0.0,
            ],
            vol_shadow_max: [
                cam_pos.x + half_range, cam_pos.y + half_height,
                cam_pos.z + half_range, 0.0,
            ],
        };

        // Upscale jitter (disabled — spatial only, Phase 9 decision)
        self.upscale.update_jitter(&self.context.queue, [0.0, 0.0]);
        let history_read_idx = self.history.read_index();

        // ── Execute frame via rkf-runtime frame scheduler ────────────────────
        let settings = FrameSettings::default();
        let mut ctx = FrameContext {
            encoder: &mut encoder,
            queue: &self.context.queue,
            settings: &settings,
            scene: &self.scene,
            gbuffer: &self.gbuffer,
            material_table: &self.material_table,
            light_buffer: &self.light_buffer,
            clipmap: &self.clipmap,
            ray_march: &self.ray_march,
            tile_cull: &self.tile_cull,
            shading: &self.shading,
            tone_map: &self.tone_map,
            blit: &self.blit,
            shade_light_bind_group: &self.tile_cull.shade_light_bind_group,
            radiance_volume: &self.radiance_volume,
            radiance_inject: &self.radiance_inject,
            radiance_mip: &self.radiance_mip,
            color_pool: &self.color_pool,
            vol_shadow: &self.vol_shadow,
            vol_march: &self.vol_march,
            vol_march_params: &vol_params,
            vol_temporal: &mut self.vol_temporal,
            vol_upscale: &self.vol_upscale,
            vol_composite: &self.vol_composite,
            cloud_shadow: &self.cloud_shadow,
            bloom: &self.bloom,
            dof: &self.dof,
            motion_blur: &self.motion_blur,
            upscale: &self.upscale,
            sharpen: &self.sharpen,
            history_read_idx,
            bloom_composite: &self.bloom_composite,
            auto_exposure: &self.auto_exposure,
            color_grade: &self.color_grade,
            cosmetics: &self.cosmetics,
            frame_index: self.frame_index,
            dt,
            camera_pos: [cam_pos.x, cam_pos.y, cam_pos.z],
            sun_dir: sun_dir_n,
            swapchain_view: &view,
        };
        execute_frame(&mut ctx);

        // Copy final LDR texture -> staging buffer for screenshot readback
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.cosmetics.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(DISPLAY_WIDTH * 4),
                    rows_per_image: Some(DISPLAY_HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: DISPLAY_WIDTH,
                height: DISPLAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );

        self.context.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        // Read back the staging buffer into shared state for MCP screenshot tool
        self.capture_frame();

        // Swap history ping-pong for next frame
        self.history.swap();

        // Update shared state with camera and timing info
        if let Ok(mut state) = self.shared_state.lock() {
            state.camera_position = self.camera.position;
            state.camera_yaw = self.camera.yaw;
            state.camera_pitch = self.camera.pitch;
            state.camera_fov = self.camera.fov_degrees;
            state.frame_time_ms = dt as f64 * 1000.0;
        }
    }

    fn capture_frame(&mut self) {
        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.context.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            if let Ok(mut state) = self.shared_state.lock() {
                state.frame_pixels.copy_from_slice(&data);
            }
            drop(data);
            self.staging_buffer.unmap();
        }
    }
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_captured: bool,
}

impl InputState {
    fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            mouse_captured: false,
        }
    }

    fn process_key(&mut self, key: KeyCode, pressed: bool) -> Option<u32> {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            // Number keys set debug visualization mode (only on press)
            KeyCode::Digit0 if pressed => return Some(0),
            KeyCode::Digit1 if pressed => return Some(1),
            KeyCode::Digit2 if pressed => return Some(2),
            KeyCode::Digit3 if pressed => return Some(3),
            KeyCode::Digit4 if pressed => return Some(4),
            KeyCode::Digit5 if pressed => return Some(5),
            KeyCode::Digit6 if pressed => return Some(6),
            _ => {}
        }
        None
    }

    fn apply_to_camera(&self, camera: &mut Camera, dt: f32) {
        let speed = camera.move_speed * dt;
        if self.forward {
            camera.translate_forward(speed);
        }
        if self.backward {
            camera.translate_forward(-speed);
        }
        if self.right {
            camera.translate_right(speed);
        }
        if self.left {
            camera.translate_right(-speed);
        }
        if self.up {
            camera.translate_up(speed);
        }
        if self.down {
            camera.translate_up(-speed);
        }
    }
}

// ---------------------------------------------------------------------------
// IPC server
// ---------------------------------------------------------------------------

/// Spawn the IPC server on a background thread with its own tokio runtime.
///
/// Returns the socket path for cleanup.
fn spawn_ipc_server(api: Arc<dyn rkf_core::automation::AutomationApi>) -> String {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let path_clone = socket_path.clone();

    std::thread::Builder::new()
        .name("ipc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime for IPC server");

            rt.block_on(async {
                let mut registry = rkf_mcp::registry::ToolRegistry::new();
                rkf_mcp::tools::observation::register_observation_tools(&mut registry);
                let registry = Arc::new(registry);

                let config = rkf_mcp::ipc::IpcConfig {
                    socket_path: Some(path_clone),
                    tcp_port: 0,
                    mode: rkf_mcp::registry::ToolMode::Debug,
                };

                if let Err(e) = rkf_mcp::ipc::run_server(config, registry, api).await {
                    log::error!("IPC server error: {e}");
                }
            });
        })
        .expect("failed to spawn IPC server thread");

    socket_path
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    input: InputState,
    frame_count: u64,
    last_title_update: Instant,
    last_frame: Instant,
    socket_path: Option<String>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            input: InputState::new(),
            frame_count: 0,
            last_title_update: Instant::now(),
            last_frame: Instant::now(),
            socket_path: None,
        }
    }

    fn toggle_mouse_capture(&mut self) {
        if let Some(window) = &self.window {
            self.input.mouse_captured = !self.input.mouse_captured;
            if self.input.mouse_captured {
                let _ = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                window.set_cursor_visible(false);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("RKIField Testbed [Phase 13]")
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // Create shared state for automation API
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0, // will be updated by GpuState::new
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        )));

        self.gpu = Some(GpuState::new(window.clone(), Arc::clone(&shared_state)));
        self.window = Some(window);
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();

        // Spawn IPC server for MCP tool access
        let api: Arc<dyn rkf_core::automation::AutomationApi> =
            Arc::new(TestbedAutomationApi::new(shared_state));
        let socket_path = spawn_ipc_server(api);
        log::info!("IPC server listening on {socket_path}");
        self.socket_path = Some(socket_path);

        log::info!("Phase 13 validation — multi-LOD clipmap scene");
        log::info!("Click to capture mouse, WASD to move, mouse to look, Esc to exit");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if !self.input.mouse_captured {
                    self.toggle_mouse_capture();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && pressed {
                        if self.input.mouse_captured {
                            self.toggle_mouse_capture();
                        } else {
                            event_loop.exit();
                        }
                        return;
                    }
                    if let Some(debug_mode) = self.input.process_key(key, pressed) {
                        if let Some(gpu) = &self.gpu {
                            gpu.shading.set_debug_mode(&gpu.context.queue, debug_mode);
                            log::info!("Debug mode: {debug_mode}");
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                self.frame_count += 1;

                // Update camera from input
                if let Some(gpu) = &mut self.gpu {
                    self.input.apply_to_camera(&mut gpu.camera, dt);
                    gpu.render(dt);
                }

                // Update title bar with frame time every 500ms
                if now.duration_since(self.last_title_update).as_millis() > 500 {
                    if let Some(window) = &self.window {
                        let elapsed = now.duration_since(self.last_title_update).as_secs_f64();
                        let fps = self.frame_count as f64 / elapsed;
                        window.set_title(&format!(
                            "RKIField Testbed [Phase 13] — {fps:.0} fps ({:.2} ms)",
                            1000.0 / fps
                        ));
                        self.frame_count = 0;
                        self.last_title_update = now;
                    }
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.input.mouse_captured {
                if let Some(gpu) = &mut self.gpu {
                    gpu.camera.rotate(delta.0, delta.1);
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Clean up IPC socket file
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
            log::info!("Cleaned up IPC socket: {path}");
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
