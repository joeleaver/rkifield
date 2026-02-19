//! Engine rendering state — mirrors testbed's GpuState for the editor viewport.
//!
//! Creates a demo clipmap scene and all render passes. The editor's main loop
//! calls [`EngineState::render`] each frame to run the full SDF pipeline and
//! blit the result to the swapchain.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::Vec3;

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
use rkf_render::cloud_shadow::{CloudShadowPass, DEFAULT_CLOUD_SHADOW_EXTINCTION};
use rkf_render::clouds::{CloudParams, CloudSettings};
use rkf_render::color_grade::ColorGradePass;
use rkf_render::cosmetics::CosmeticsPass;
use rkf_render::dof::DofPass;
use rkf_render::fog::{FogParams, FogSettings};
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
use rkf_render::vol_composite::VolCompositePass;
use rkf_render::vol_march::{VolMarchPass, VolMarchParams};
use rkf_render::vol_shadow::VolShadowPass;
use rkf_render::vol_temporal::VolTemporalPass;
use rkf_render::vol_upscale::VolUpscalePass;

use rkf_runtime::frame::{execute_frame, FrameContext, FrameSettings};

use rkf_edit::transform_ops::{
    self, ObjectTransform, SdfObjectRegistry, SdfPrimitive, SdfRecipe,
};

use crate::automation::SharedState;
use crate::environment::EnvironmentState;
use crate::light_editor::{EditorLight, EditorLightType};

/// Display (output) resolution.
pub const DISPLAY_WIDTH: u32 = 1280;
pub const DISPLAY_HEIGHT: u32 = 720;

/// Resolution tier for the demo scene (Tier 2 = 8cm voxels).
const SCENE_TIER: usize = 2;

// ---------------------------------------------------------------------------
// Scene creation
// ---------------------------------------------------------------------------

/// Apply material blending to existing surface voxels.
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
                let brick_min = aabb.min
                    + Vec3::new(
                        cx as f32 * brick_ext,
                        cy as f32 * brick_ext,
                        cz as f32 * brick_ext,
                    );
                let brick = pool.get_mut(slot);
                for vz in 0..BRICK_DIM {
                    for vy in 0..BRICK_DIM {
                        for vx in 0..BRICK_DIM {
                            let existing = brick.sample(vx, vy, vz);
                            let dist = existing.distance_f32();
                            if dist.abs() > voxel_size * 2.0 {
                                continue;
                            }
                            let world_pos = brick_min
                                + Vec3::new(
                                    (vx as f32 + 0.5) * voxel_size,
                                    (vy as f32 + 0.5) * voxel_size,
                                    (vz as f32 + 0.5) * voxel_size,
                                );
                            if let Some(weight) = selector(world_pos) {
                                let w = (weight.clamp(0.0, 1.0) * 255.0) as u8;
                                brick.set(
                                    vx,
                                    vy,
                                    vz,
                                    VoxelSample::new(
                                        dist,
                                        existing.material_id(),
                                        w,
                                        secondary_id,
                                        existing.flags(),
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Create a multi-LOD clipmap scene.
///
/// Two LOD levels sharing the same brick pool:
/// - Level 0: Tier 2 (8cm voxels), 8m radius — fine detail
/// - Level 1: Tier 3 (32cm voxels), 32m radius — distant/coarse
fn create_clipmap_scene() -> (BrickPool, ClipmapGridSet, ClipmapConfig, Aabb, Vec<Aabb>, SdfObjectRegistry) {
    let config = ClipmapConfig::new(vec![
        ClipmapLevel {
            voxel_size: 0.08,
            radius: 8.0,
        },
        ClipmapLevel {
            voxel_size: 0.32,
            radius: 32.0,
        },
    ]);
    let mut grid_set = ClipmapGridSet::from_config(config.clone(), 64);
    let mut pool: BrickPool = Pool::new(65536);

    let tiers = [2usize, 3usize];

    let aabbs: Vec<Aabb> = (0..config.num_levels())
        .map(|i| {
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
        })
        .collect();

    let pillar_positions = [
        Vec3::new(-4.0, 0.0, -1.5),
        Vec3::new(-1.5, 0.0, -1.5),
        Vec3::new(1.5, 0.0, -1.5),
        Vec3::new(4.0, 0.0, -1.5),
        Vec3::new(-4.0, 0.0, 1.5),
        Vec3::new(-1.5, 0.0, 1.5),
        Vec3::new(1.5, 0.0, 1.5),
        Vec3::new(4.0, 0.0, 1.5),
    ];

    // === Level 0 (fine, 8cm voxels, 8m radius) ===
    {
        let aabb = &aabbs[0];
        let grid = grid_set.grid_mut(0);
        let tier = tiers[0];

        // Ground — grass green (material 8)
        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| box_sdf(Vec3::new(8.0, 0.5, 8.0), p - Vec3::new(0.0, -0.5, 0.0)),
            tier,
            aabb,
            8,
        );

        // Pillars + lintels + boulder — warm sandstone (material 1)
        let pillars = pillar_positions;
        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| {
                let mut d = f32::MAX;
                for base in &pillars {
                    let bottom = Vec3::new(base.x, -0.3, base.z);
                    let top = Vec3::new(base.x, 3.5, base.z);
                    d = d.min(capsule_sdf(bottom, top, 0.25, p));
                }
                d = d.min(box_sdf(
                    Vec3::new(0.3, 0.2, 1.8),
                    p - Vec3::new(-1.5, 3.6, 0.0),
                ));
                d = d.min(box_sdf(
                    Vec3::new(0.3, 0.2, 1.8),
                    p - Vec3::new(1.5, 3.6, 0.0),
                ));
                d = d.min(sphere_sdf(Vec3::new(3.0, 0.2, -3.5), 0.8, p));
                d
            },
            tier,
            aabb,
            1,
        );

        // Monolith wall — metal (material 2)
        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| box_sdf(Vec3::new(0.3, 2.5, 3.0), p - Vec3::new(-7.0, 2.2, 0.0)),
            tier,
            aabb,
            2,
        );

        // Blend sphere: sandstone (mat 1) → gold (mat 11) height gradient
        let blend_sphere_center = Vec3::new(0.0, 1.5, -4.5);
        let blend_sphere_radius = 1.2;
        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| sphere_sdf(blend_sphere_center, blend_sphere_radius, p),
            tier,
            aabb,
            1,
        );
        apply_material_blend(&mut pool, grid, tier, aabb, 11, |pos| {
            let d = (pos - blend_sphere_center).length();
            if d > blend_sphere_radius + 0.2 {
                return None;
            }
            let t = ((pos.y - 0.3) / 2.4).clamp(0.0, 1.0);
            Some(t)
        });

        // Dirt blended onto pillar bases
        apply_material_blend(&mut pool, grid, tier, aabb, 10, |pos| {
            let near_pillar = pillar_positions.iter().any(|b| {
                let dx = pos.x - b.x;
                let dz = pos.z - b.z;
                (dx * dx + dz * dz).sqrt() < 0.6
            });
            if !near_pillar {
                return None;
            }
            let t = 1.0 - (pos.y / 0.8).clamp(0.0, 1.0);
            if t < 0.01 {
                None
            } else {
                Some(t)
            }
        });

        // Gold blend on monolith wall top
        apply_material_blend(&mut pool, grid, tier, aabb, 11, |pos| {
            let center = Vec3::new(-7.0, 2.2, 0.0);
            let half = Vec3::new(0.5, 2.7, 3.2);
            if (pos.x - center.x).abs() > half.x || (pos.z - center.z).abs() > half.z {
                return None;
            }
            let t = ((pos.y - 3.0) / 1.7).clamp(0.0, 1.0);
            if t < 0.01 {
                None
            } else {
                Some(t)
            }
        });

        log::info!("Level 0 (tier 2, 8cm): {} bricks", pool.allocated_count());
    }

    // === Level 1 (coarse, 32cm voxels, 32m radius) ===
    let level0_count = pool.allocated_count();
    {
        let aabb = &aabbs[1];
        let grid = grid_set.grid_mut(1);
        let tier = tiers[1];

        let blend_k = 1.0;
        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| {
                let mut d =
                    box_sdf(Vec3::new(30.0, 0.5, 30.0), p - Vec3::new(0.0, -0.5, 0.0));
                d = smin(
                    d,
                    sphere_sdf(Vec3::new(16.0, -1.0, -14.0), 4.0, p),
                    blend_k,
                );
                d = smin(
                    d,
                    sphere_sdf(Vec3::new(-18.0, -1.0, 12.0), 5.0, p),
                    blend_k,
                );
                d = smin(
                    d,
                    sphere_sdf(Vec3::new(10.0, -1.0, 20.0), 3.5, p),
                    blend_k,
                );
                d
            },
            tier,
            aabb,
            8,
        );

        let _ = populate_grid_with_material(
            &mut pool,
            grid,
            |p| {
                let monolith =
                    box_sdf(Vec3::new(1.5, 8.0, 1.5), p - Vec3::new(22.0, 4.0, -5.0));
                let wall =
                    box_sdf(Vec3::new(8.0, 3.0, 0.5), p - Vec3::new(0.0, 1.5, -20.0));
                monolith.min(wall)
            },
            tier,
            aabb,
            2,
        );

        log::info!(
            "Level 1 (tier 3, 32cm): {} bricks ({} new)",
            pool.allocated_count(),
            pool.allocated_count() - level0_count
        );
    }

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

    // ── Register SDF objects for transform tracking ────────────────────
    let mut registry = SdfObjectRegistry::new();
    let pillar_positions_vec: Vec<(SdfPrimitive, Vec3)> = pillar_positions
        .iter()
        .map(|base| {
            let bottom = Vec3::new(base.x, -0.3, base.z);
            let top = Vec3::new(base.x, 3.5, base.z);
            (
                SdfPrimitive::Capsule {
                    a: bottom,
                    b: top,
                    radius: 0.25,
                },
                Vec3::ZERO,
            )
        })
        .chain([
            (
                SdfPrimitive::Box {
                    half_extents: Vec3::new(0.3, 0.2, 1.8),
                },
                Vec3::new(-1.5, 3.6, 0.0),
            ),
            (
                SdfPrimitive::Box {
                    half_extents: Vec3::new(0.3, 0.2, 1.8),
                },
                Vec3::new(1.5, 3.6, 0.0),
            ),
            (
                SdfPrimitive::Sphere { radius: 0.8 },
                Vec3::new(3.0, 0.2, -3.5),
            ),
        ])
        .collect();

    // Entity 1: Ground
    registry.register_with_id(
        1,
        SdfRecipe {
            primitive: SdfPrimitive::Box {
                half_extents: Vec3::new(8.0, 0.5, 8.0),
            },
            material_id: 8,
        },
        ObjectTransform {
            position: Vec3::new(0.0, -0.5, 0.0),
            ..Default::default()
        },
        0,
    );

    // Entity 2: Pillars + lintels + boulder
    registry.register_with_id(
        2,
        SdfRecipe {
            primitive: SdfPrimitive::Union(pillar_positions_vec),
            material_id: 1,
        },
        ObjectTransform::default(),
        0,
    );

    // Entity 3: Monolith wall
    registry.register_with_id(
        3,
        SdfRecipe {
            primitive: SdfPrimitive::Box {
                half_extents: Vec3::new(0.3, 2.5, 3.0),
            },
            material_id: 2,
        },
        ObjectTransform {
            position: Vec3::new(-7.0, 2.2, 0.0),
            ..Default::default()
        },
        0,
    );

    // Entity 4: Blend sphere
    registry.register_with_id(
        4,
        SdfRecipe {
            primitive: SdfPrimitive::Sphere { radius: 1.2 },
            material_id: 1,
        },
        ObjectTransform {
            position: Vec3::new(0.0, 1.5, -4.5),
            ..Default::default()
        },
        0,
    );

    // Entity 5: Distant terrain (level 1)
    registry.register_with_id(
        5,
        SdfRecipe {
            primitive: SdfPrimitive::SmoothUnion {
                children: vec![
                    (
                        SdfPrimitive::Box {
                            half_extents: Vec3::new(30.0, 0.5, 30.0),
                        },
                        Vec3::new(0.0, -0.5, 0.0),
                    ),
                    (SdfPrimitive::Sphere { radius: 4.0 }, Vec3::new(16.0, -1.0, -14.0)),
                    (SdfPrimitive::Sphere { radius: 5.0 }, Vec3::new(-18.0, -1.0, 12.0)),
                    (SdfPrimitive::Sphere { radius: 3.5 }, Vec3::new(10.0, -1.0, 20.0)),
                ],
                k: 1.0,
            },
            material_id: 8,
        },
        ObjectTransform::default(),
        1,
    );

    // Entity 6: Far monolith + wall (level 1)
    registry.register_with_id(
        6,
        SdfRecipe {
            primitive: SdfPrimitive::Union(vec![
                (
                    SdfPrimitive::Box {
                        half_extents: Vec3::new(1.5, 8.0, 1.5),
                    },
                    Vec3::new(22.0, 4.0, -5.0),
                ),
                (
                    SdfPrimitive::Box {
                        half_extents: Vec3::new(8.0, 3.0, 0.5),
                    },
                    Vec3::new(0.0, 1.5, -20.0),
                ),
            ]),
            material_id: 2,
        },
        ObjectTransform::default(),
        1,
    );

    log::info!(
        "Clipmap scene: {} levels, {} total bricks, {} registered objects",
        config.num_levels(),
        pool.allocated_count(),
        registry.len(),
    );
    (pool, grid_set, config, scene_aabb, aabbs, registry)
}

/// Overhead lighting: white sun + cool fill.
fn create_clipmap_lights() -> Vec<Light> {
    vec![
        Light::directional([0.5, 0.7, 0.3], [1.0, 0.98, 0.95], 3.0, true),
        Light::directional([-0.3, 0.6, -0.5], [0.6, 0.65, 0.8], 0.4, false),
    ]
}

// ---------------------------------------------------------------------------
// EngineState
// ---------------------------------------------------------------------------

/// Full SDF render pipeline state for the editor viewport.
#[allow(dead_code)]
pub struct EngineState {
    device: wgpu::Device,
    queue: wgpu::Queue,
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
    pub shading: ShadingPass,
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
    pub cosmetics: CosmeticsPass,
    blit: BlitPass,
    pub camera: Camera,
    staging_buffer: wgpu::Buffer,
    shared_state: Arc<Mutex<SharedState>>,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    // Line overlay pipeline (gizmos, wireframes)
    line_pipeline: wgpu::RenderPipeline,
    line_vp_buffer: wgpu::Buffer,
    line_vp_bind_group: wgpu::BindGroup,
    line_vertex_buffer: wgpu::Buffer,
    line_vertex_capacity: usize,
    surface_format: wgpu::TextureFormat,
    // CPU-side voxel data for transform operations
    cpu_pool: BrickPool,
    cpu_grid_set: ClipmapGridSet,
    level_aabbs: Vec<Aabb>,
    /// Registry of SDF objects for transform tracking.
    pub object_registry: SdfObjectRegistry,
}

impl EngineState {
    /// Create the engine state with all render passes.
    ///
    /// The `surface_format` is needed for the final blit pass.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        shared_state: Arc<Mutex<SharedState>>,
    ) -> Self {
        // Scene
        let (pool, grid_set, clipmap_config, aabb, level_aabbs, object_registry) =
            create_clipmap_scene();
        let grid = grid_set.grid(0);

        // Update shared state with pool info
        {
            let mut state = shared_state.lock().unwrap();
            state.pool_capacity = pool.capacity() as u64;
            state.pool_allocated = pool.allocated_count() as u64;
        }

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.yaw = 0.0;
        camera.pitch = -0.15;
        camera.fov_degrees = 70.0;

        let camera_uniforms =
            camera.uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, 0, [[0.0; 4]; 4]);
        let camera_bytes = bytemuck::bytes_of(&camera_uniforms);

        // SceneUniforms
        let dims = grid.dimensions();
        let scene_uniforms = SceneUniforms {
            grid_dims: [dims.x, dims.y, dims.z, 0],
            grid_origin: [
                aabb.min.x,
                aabb.min.y,
                aabb.min.z,
                RESOLUTION_TIERS[SCENE_TIER].brick_extent,
            ],
            params: [RESOLUTION_TIERS[SCENE_TIER].voxel_size, 0.0, 0.0, 0.0],
        };

        let scene = GpuScene::upload(device, &pool, grid, camera_bytes, &scene_uniforms);

        // Materials
        let mut materials = material_table::create_test_materials();
        materials[1].albedo = [0.72, 0.52, 0.35]; // sandstone
        materials[8].albedo = [0.22, 0.42, 0.12]; // grass
        materials[8].roughness = 0.95;
        materials[8].metallic = 0.0;
        materials[10].albedo = [0.4, 0.28, 0.15]; // dirt
        materials[10].roughness = 0.95;
        materials[10].metallic = 0.0;
        materials[11].albedo = [1.0, 0.84, 0.0]; // gold
        materials[11].roughness = 0.25;
        materials[11].metallic = 1.0;
        let material_table = MaterialTable::upload(device, &materials);

        let gbuffer = GBuffer::new(device, INTERNAL_WIDTH, INTERNAL_HEIGHT);

        let lights = create_clipmap_lights();
        let light_buffer = LightBuffer::upload(device, &lights);

        let tile_cull = TileCullPass::new(
            device,
            &gbuffer,
            &light_buffer,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        let radiance_volume = RadianceVolume::new(device);
        let radiance_inject =
            RadianceInjectPass::new(device, &scene, &material_table, &light_buffer, &radiance_volume);
        let radiance_mip = RadianceMipPass::new(device, &radiance_volume);
        radiance_inject.update_inject_uniforms(queue, lights.len() as u32, 2);

        let color_pool = GpuColorPool::empty(device);
        let clipmap = ClipmapGpuData::upload(device, &grid_set, [0.0, 0.0, 0.0]);

        let ray_march = RayMarchPass::new(device, &scene, &gbuffer, &clipmap);
        let shading = ShadingPass::new(
            device,
            &gbuffer,
            &material_table,
            &scene,
            &tile_cull.shade_light_bind_group_layout,
            &radiance_volume,
            &color_pool,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        let vol_shadow = VolShadowPass::new(device, queue, &scene.bind_group_layout);
        let cloud_shadow = CloudShadowPass::new(device);

        let half_w = INTERNAL_WIDTH / 2;
        let half_h = INTERNAL_HEIGHT / 2;
        let vol_march = VolMarchPass::new(
            device,
            queue,
            &gbuffer.position_view,
            &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w,
            half_h,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        let vol_temporal = VolTemporalPass::new(
            device,
            &vol_march.output_view,
            &gbuffer.motion_view,
            half_w,
            half_h,
        );

        let vol_upscale = VolUpscalePass::new(
            device,
            &vol_march.output_view,
            &gbuffer.position_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
            half_w,
            half_h,
        );

        let vol_composite = VolCompositePass::new(
            device,
            &shading.hdr_view,
            &vol_upscale.output_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

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

        let cloud_settings = CloudSettings {
            procedural_enabled: false,
            cloud_density_scale: 0.0,
            shadow_enabled: false,
            ..Default::default()
        };

        let dof = DofPass::new(
            device,
            &vol_composite.output_view,
            &gbuffer.position_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        dof.update_focus(queue, 5.0, 100.0, 0.0);

        let motion_blur = MotionBlurPass::new(
            device,
            &dof.output_view,
            &gbuffer.motion_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        motion_blur.set_intensity(queue, 0.0);

        let bloom = BloomPass::new(
            device,
            &motion_blur.output_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        let history = HistoryBuffers::new(
            device,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        let upscale = UpscalePass::new(
            device,
            &motion_blur.output_view,
            &gbuffer,
            &history,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        let sharpen = SharpenPass::new(
            device,
            &upscale.output_view,
            &gbuffer,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        );

        let bloom_composite = BloomCompositePass::new(
            device,
            &sharpen.output_view,
            bloom.mip_views(),
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        );

        let auto_exposure = AutoExposurePass::new(
            device,
            &bloom_composite.output_view,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
        );

        let tone_map = ToneMapPass::new_with_exposure(
            device,
            &bloom_composite.output_view,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT,
            Some(auto_exposure.get_exposure_buffer()),
        );

        let color_grade =
            ColorGradePass::new(device, queue, &tone_map.ldr_view, DISPLAY_WIDTH, DISPLAY_HEIGHT);

        let cosmetics =
            CosmeticsPass::new(device, &color_grade.output_view, DISPLAY_WIDTH, DISPLAY_HEIGHT);

        bloom.set_threshold(queue, 0.8, 0.4);
        bloom_composite.set_intensity(queue, 0.0);
        cosmetics.set_vignette(queue, 0.0);
        cosmetics.set_chromatic_aberration(queue, 0.0);

        let blit = BlitPass::new(device, &cosmetics.output_view, surface_format);

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("editor screenshot staging"),
            size: (DISPLAY_WIDTH * DISPLAY_HEIGHT * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ── Line overlay pipeline (gizmos, wireframes) ──────────────────
        let line_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("line shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("line.wgsl").into(),
                ),
            });

        let line_vp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line VP uniform"),
            size: 64, // mat4x4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let line_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("line bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let line_vp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line VP bind group"),
            layout: &line_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: line_vp_buffer.as_entire_binding(),
            }],
        });

        let line_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("line pipeline layout"),
                bind_group_layouts: &[&line_bind_group_layout],
                push_constant_ranges: &[],
            });

        let line_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("line pipeline"),
                layout: Some(&line_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &line_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 28, // vec3 (12) + vec4 (16)
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 12,
                                shader_location: 1,
                            },
                        ],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &line_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // Initial vertex buffer: 1024 vertices (enough for basic gizmo)
        let line_vertex_capacity = 1024;
        let line_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line vertex buffer"),
            size: (line_vertex_capacity * 28) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device: device.clone(),
            queue: queue.clone(),
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
            vol_upscale,
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
            line_pipeline,
            line_vp_buffer,
            line_vp_bind_group,
            line_vertex_buffer,
            line_vertex_capacity,
            surface_format,
            cpu_pool: pool,
            cpu_grid_set: grid_set,
            level_aabbs,
            object_registry,
        }
    }

    /// Apply editor environment settings to engine render state.
    ///
    /// Maps the editor's `EnvironmentState` (pure data model) to the engine's
    /// fog, cloud, and post-process parameters. Called when `is_dirty()` is true.
    pub fn apply_environment(&mut self, env: &EnvironmentState) {
        // Fog
        self.fog_settings.height_fog_enabled = env.fog.enabled;
        self.fog_settings.fog_base_density = env.fog.density;
        self.fog_settings.fog_height_falloff = env.fog.height_falloff;
        self.fog_settings.fog_color = [env.fog.color.x, env.fog.color.y, env.fog.color.z];
        self.fog_settings.distance_fog_enabled = env.fog.enabled;
        self.fog_settings.fog_distance_density = env.fog.density * 0.5;
        self.fog_settings.fog_distance_falloff = 1.0 / env.fog.end_distance.max(1.0);

        // Clouds
        self.cloud_settings.procedural_enabled = env.clouds.enabled;
        self.cloud_settings.cloud_density_scale = env.clouds.density;
        self.cloud_settings.shadow_enabled = env.clouds.enabled;
        self.cloud_settings.cloud_min = env.clouds.altitude;
        self.cloud_settings.cloud_max = env.clouds.altitude + env.clouds.thickness;
        self.cloud_settings.shadow_coverage = env.clouds.coverage;
        self.cloud_settings.wind_speed = env.clouds.wind_speed;

        // Post-process
        self.bloom_composite
            .set_intensity(&self.queue, if env.post_process.bloom_enabled { env.post_process.bloom_intensity } else { 0.0 });
        self.bloom
            .set_threshold(&self.queue, env.post_process.bloom_threshold, 0.4);
        self.cosmetics
            .set_vignette(&self.queue, env.post_process.vignette_intensity);

        log::debug!("Environment settings applied to engine");
    }

    /// Apply editor lights to the engine, merging with base scene lights.
    ///
    /// Converts `EditorLight` → `rkf_render::light::Light`, appends to the
    /// base scene lights, and writes the combined array to the GPU buffer.
    pub fn apply_lights(&mut self, editor_lights: &[EditorLight]) {
        // Start with the base scene lights
        let mut combined = create_clipmap_lights();

        // Append editor-defined lights
        for el in editor_lights {
            let pos = [el.position.x, el.position.y, el.position.z];
            let dir = [el.direction.x, el.direction.y, el.direction.z];
            let color = [el.color.x, el.color.y, el.color.z];
            let light = match el.light_type {
                EditorLightType::Point => {
                    Light::point(pos, color, el.intensity, el.range, el.cast_shadows)
                }
                EditorLightType::Spot => Light::spot(
                    pos,
                    dir,
                    color,
                    el.intensity,
                    el.range,
                    el.spot_inner_angle,
                    el.spot_outer_angle,
                    el.cast_shadows,
                ),
                EditorLightType::Directional => {
                    Light::directional(dir, color, el.intensity, el.cast_shadows)
                }
            };
            combined.push(light);
        }

        // Write to buffer (truncate to existing buffer capacity)
        let max_lights = self.light_buffer.buffer.size() as usize / std::mem::size_of::<Light>();
        let count = combined.len().min(max_lights);
        let data = &combined[..count];
        self.queue.write_buffer(
            &self.light_buffer.buffer,
            0,
            bytemuck::cast_slice(data),
        );
        self.light_buffer.count = count as u32;
        self.lights = combined;

        // Update radiance inject with new light count
        self.radiance_inject
            .update_inject_uniforms(&self.queue, count as u32, 2);

        log::debug!("Lights updated: {} total ({} editor)", count, editor_lights.len());
    }

    /// Apply a transform change to an SDF object and upload dirty voxels to GPU.
    ///
    /// Looks up the object's clipmap level, clears+re-voxelizes the affected
    /// region on the CPU-side pool/grid, then incrementally uploads only the
    /// changed bricks, occupancy words, and slot entries to the GPU.
    ///
    /// Returns `true` on success.
    pub fn apply_object_transform(&mut self, object_id: u64, new_transform: ObjectTransform) -> bool {
        let obj = match self.object_registry.get(object_id) {
            Some(o) => o,
            None => {
                log::warn!("apply_object_transform: unknown object {object_id}");
                return false;
            }
        };
        let level = obj.clipmap_level;

        // Determine the tier for this level (same mapping as create_clipmap_scene)
        let tiers = [2usize, 3usize];
        let tier = if level < tiers.len() { tiers[level] } else { return false };
        let grid_aabb = match self.level_aabbs.get(level) {
            Some(a) => *a,
            None => return false,
        };

        let grid = self.cpu_grid_set.grid_mut(level);
        let result = transform_ops::apply_transform_change(
            &mut self.cpu_pool,
            grid,
            &grid_aabb,
            tier,
            &self.object_registry,
            object_id,
            &new_transform,
        );

        let result = match result {
            Some(r) => r,
            None => return false,
        };

        // Upload dirty bricks
        self.scene.write_dirty_bricks(&self.queue, &self.cpu_pool, &result.dirty_brick_slots);

        // Upload dirty occupancy words and slot entries
        let grid = self.cpu_grid_set.grid(level);

        // Compute occupancy word indices from dirty cell indices
        // flat_index = x + y * dim_x + z * dim_x * dim_y
        // occupancy word = flat_index / 16 (2 bits per cell, 32 bits per word)
        let mut dirty_occ_words: Vec<u32> = result
            .dirty_cell_indices
            .iter()
            .map(|&idx| idx / 16)
            .collect();
        dirty_occ_words.sort_unstable();
        dirty_occ_words.dedup();

        self.scene.write_dirty_occupancy(&self.queue, grid, &dirty_occ_words);
        self.scene.write_dirty_slots(&self.queue, grid, &result.dirty_cell_indices);

        // Update the registry with the new transform
        if let Some(obj) = self.object_registry.get_mut(object_id) {
            obj.transform = new_transform;
        }

        // Update shared state pool info
        if let Ok(mut state) = self.shared_state.lock() {
            state.pool_allocated = self.cpu_pool.allocated_count() as u64;
        }

        log::info!(
            "Object {object_id} transform applied: {} dirty bricks, {} dirty cells",
            result.dirty_brick_slots.len(),
            result.dirty_cell_indices.len(),
        );
        true
    }

    /// Render one frame: full SDF pipeline + line overlay + blit to swapchain.
    ///
    /// `line_batch` provides optional gizmo/wireframe lines rendered on top.
    /// Submits GPU commands and blocks on staging readback for MCP screenshots.
    pub fn render(
        &mut self,
        swapchain_view: &wgpu::TextureView,
        dt: f32,
        line_batch: Option<&crate::overlay::LineBatch>,
    ) {
        // Check for pending MCP debug mode (camera is consumed in main loop)
        if let Ok(mut state) = self.shared_state.lock() {
            if let Some(mode) = state.pending_debug_mode.take() {
                self.shading.set_debug_mode(&self.queue, mode);
                log::info!("Debug mode set via MCP: {mode}");
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);

        // Update camera uniforms
        let mut uniforms =
            self.camera
                .uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, self.frame_index, self.prev_vp);
        uniforms.jitter = [0.0, 0.0]; // spatial-only upscaling

        let vp = self
            .camera
            .view_projection(INTERNAL_WIDTH, INTERNAL_HEIGHT);
        self.prev_vp = vp.to_cols_array_2d();

        self.queue.write_buffer(
            &self.scene.camera_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let cam_pos = self.camera.position;
        self.shading
            .update_camera_pos(&self.queue, [cam_pos.x, cam_pos.y, cam_pos.z]);

        let cam_fwd = self.camera.forward();
        self.tile_cull.update_uniforms(
            &self.queue,
            self.light_buffer.count,
            [cam_pos.x, cam_pos.y, cam_pos.z],
            [cam_fwd.x, cam_fwd.y, cam_fwd.z],
        );

        self.shading.update_light_info(
            &self.queue,
            self.light_buffer.count,
            self.tile_cull.num_tiles_x,
            4,
        );

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("editor frame encoder"),
                });

        self.radiance_inject
            .update_inject_uniforms(&self.queue, self.lights.len() as u32, 2);
        self.radiance_volume
            .update_center(&self.queue, [cam_pos.x, cam_pos.y, cam_pos.z]);

        // Sun direction
        let sun_dir = [0.9f32, 0.26, 0.15];
        let sun_len =
            (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]).sqrt();
        let sun_dir_n = [
            sun_dir[0] / sun_len,
            sun_dir[1] / sun_len,
            sun_dir[2] / sun_len,
        ];

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let cloud_gpu = CloudParams::from_settings(&self.cloud_settings, elapsed);
        self.vol_march.set_cloud_params(&self.queue, &cloud_gpu);

        self.cloud_shadow.update_params_ex(
            &self.queue,
            [cam_pos.x, cam_pos.y, cam_pos.z],
            sun_dir_n,
            self.cloud_settings.cloud_min,
            self.cloud_settings.cloud_max,
            self.cloud_settings.shadow_coverage,
            DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow
            .set_cloud_params(&self.queue, &cloud_gpu);

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
                cam_pos.x - half_range,
                cam_pos.y - half_height,
                cam_pos.z - half_range,
                0.0,
            ],
            vol_shadow_max: [
                cam_pos.x + half_range,
                cam_pos.y + half_height,
                cam_pos.z + half_range,
                0.0,
            ],
        };

        self.upscale.update_jitter(&self.queue, [0.0, 0.0]);
        let history_read_idx = self.history.read_index();

        // Execute the full render pipeline
        let settings = FrameSettings::default();
        let mut ctx = FrameContext {
            encoder: &mut encoder,
            queue: &self.queue,
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
            swapchain_view,
        };
        execute_frame(&mut ctx);

        // ── Line overlay pass (gizmos, wireframes) ──────────────────────
        if let Some(lines) = line_batch {
            if !lines.is_empty() {
                // Compute VP at display resolution for line overlay
                let line_vp = self
                    .camera
                    .view_projection(DISPLAY_WIDTH, DISPLAY_HEIGHT);
                self.queue.write_buffer(
                    &self.line_vp_buffer,
                    0,
                    bytemuck::bytes_of(&line_vp),
                );

                // Billboard quad generation: each segment → 6 GPU vertices
                let cam_pos = self.camera.position;
                let fov_half_tan =
                    (self.camera.fov_degrees.to_radians() / 2.0).tan();
                let vh = DISPLAY_HEIGHT as f32;

                let gpu_verts = lines.segments.len() * 6;
                let byte_size = gpu_verts * 28;
                if gpu_verts > self.line_vertex_capacity {
                    self.line_vertex_capacity = gpu_verts.next_power_of_two();
                    self.line_vertex_buffer =
                        self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("line vertex buffer"),
                            size: (self.line_vertex_capacity * 28) as u64,
                            usage: wgpu::BufferUsages::VERTEX
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                }

                // Generate camera-facing quads for each line segment
                let mut packed = Vec::with_capacity(byte_size);
                for seg in &lines.segments {
                    let p0 = seg.start;
                    let p1 = seg.end;
                    let mid = (p0 + p1) * 0.5;
                    let dist = (cam_pos - mid).length().max(0.01);
                    let whw = seg.width * dist * fov_half_tan / vh;

                    let ld = p1 - p0;
                    let ld_n = if ld.length_squared() > 1e-12 {
                        ld.normalize()
                    } else {
                        glam::Vec3::X
                    };
                    let to_cam = (cam_pos - mid).normalize();
                    let mut side = ld_n.cross(to_cam);
                    if side.length_squared() < 1e-6 {
                        side = if ld_n.dot(glam::Vec3::Y).abs() < 0.99 {
                            ld_n.cross(glam::Vec3::Y)
                        } else {
                            ld_n.cross(glam::Vec3::X)
                        };
                    }
                    side = side.normalize() * whw;

                    let c = [p0 - side, p0 + side, p1 + side, p1 - side];
                    for &idx in &[0u8, 1, 2, 0, 2, 3] {
                        packed.extend_from_slice(bytemuck::bytes_of(
                            &c[idx as usize],
                        ));
                        packed
                            .extend_from_slice(bytemuck::bytes_of(&seg.color));
                    }
                }
                self.queue
                    .write_buffer(&self.line_vertex_buffer, 0, &packed);

                // Render pass on top of swapchain (LoadOp::Load preserves content)
                {
                    let mut rpass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("line overlay"),
                            color_attachments: &[Some(
                                wgpu::RenderPassColorAttachment {
                                    view: swapchain_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    },
                                    depth_slice: None,
                                },
                            )],
                            depth_stencil_attachment: None,
                            ..Default::default()
                        });
                    rpass.set_pipeline(&self.line_pipeline);
                    rpass.set_bind_group(0, &self.line_vp_bind_group, &[]);
                    rpass.set_vertex_buffer(
                        0,
                        self.line_vertex_buffer.slice(..byte_size as u64),
                    );
                    rpass.draw(0..gpu_verts as u32, 0..1);
                }
            }
        }

        // Copy final texture to staging for MCP screenshot readback
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

        self.queue
            .submit(std::iter::once(encoder.finish()));

        // Readback for MCP screenshots
        self.capture_frame();

        // Swap history ping-pong
        self.history.swap();

        // Update shared state
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
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

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
