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
use rkf_core::clipmap::{ClipmapConfig, ClipmapGridSet, ClipmapLevel};
use rkf_core::constants::RESOLUTION_TIERS;
use rkf_core::sparse_grid::SparseGrid;
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
use rkf_render::ray_march::RayMarchPass;
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
    free_level_bricks, voxelize_all_objects,
};

use crate::automation::SharedState;
use crate::environment::EnvironmentState;
use crate::light_editor::{EditorLight, EditorLightType};

/// Display (output) resolution (default window size).
pub const DISPLAY_WIDTH: u32 = 1280;
pub const DISPLAY_HEIGHT: u32 = 720;

/// Align `value` up to the next multiple of `align`.
fn align_up(value: u32, align: u32) -> u32 {
    (value + align - 1) / align * align
}

/// Compute padded bytes-per-row for staging buffer copies (256-byte alignment).
fn padded_bytes_per_row(width: u32) -> u32 {
    align_up(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Resolution tier for the demo scene (Tier 2 = 8cm voxels).
const SCENE_TIER: usize = 2;

// ---------------------------------------------------------------------------
// Scene creation
// ---------------------------------------------------------------------------

/// Create a single-level clipmap scene with individually selectable objects.
///
/// One LOD level: Tier 2 (8cm voxels), 8m radius. Each piece of geometry
/// is registered as its own SDF object so it can be independently selected
/// and transformed in the editor.
fn create_clipmap_scene() -> (BrickPool, ClipmapGridSet, ClipmapConfig, Aabb, Vec<Aabb>, SdfObjectRegistry) {
    let config = ClipmapConfig::new(vec![
        ClipmapLevel {
            voxel_size: 0.08,
            radius: 8.0,
        },
    ]);
    let mut grid_set = ClipmapGridSet::from_config(config.clone(), 64);
    let mut pool: BrickPool = Pool::new(rkf_core::constants::DEFAULT_CORE_POOL_CAPACITY);

    let brick_ext = RESOLUTION_TIERS[SCENE_TIER].brick_extent;
    let dims = grid_set.grid(0).dimensions();
    let half = config.level(0).radius;
    let aabb = Aabb::new(
        Vec3::new(-half, -half, -half),
        Vec3::new(
            -half + dims.x as f32 * brick_ext,
            -half + dims.y as f32 * brick_ext,
            -half + dims.z as f32 * brick_ext,
        ),
    );
    let aabbs = vec![aabb];

    // ── Register all SDF objects ────────────────────────────────────────
    let mut registry = SdfObjectRegistry::new();

    // Entity 1: Ground — grass green (material 8)
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

    // Entities 2–9: Individual pillars — sandstone (material 1)
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
    for (i, base) in pillar_positions.iter().enumerate() {
        registry.register_with_id(
            (i + 2) as u64,
            SdfRecipe {
                primitive: SdfPrimitive::Capsule {
                    a: Vec3::new(0.0, -0.3, 0.0),
                    b: Vec3::new(0.0, 3.5, 0.0),
                    radius: 0.25,
                },
                material_id: 1,
            },
            ObjectTransform {
                position: *base,
                ..Default::default()
            },
            0,
        );
    }

    // Entity 10: Left lintel — sandstone (material 1)
    registry.register_with_id(
        10,
        SdfRecipe {
            primitive: SdfPrimitive::Box {
                half_extents: Vec3::new(0.3, 0.2, 1.8),
            },
            material_id: 1,
        },
        ObjectTransform {
            position: Vec3::new(-1.5, 3.6, 0.0),
            ..Default::default()
        },
        0,
    );

    // Entity 11: Right lintel — sandstone (material 1)
    registry.register_with_id(
        11,
        SdfRecipe {
            primitive: SdfPrimitive::Box {
                half_extents: Vec3::new(0.3, 0.2, 1.8),
            },
            material_id: 1,
        },
        ObjectTransform {
            position: Vec3::new(1.5, 3.6, 0.0),
            ..Default::default()
        },
        0,
    );

    // Entity 12: Boulder — sandstone (material 1)
    registry.register_with_id(
        12,
        SdfRecipe {
            primitive: SdfPrimitive::Sphere { radius: 0.8 },
            material_id: 1,
        },
        ObjectTransform {
            position: Vec3::new(3.0, 0.2, -3.5),
            ..Default::default()
        },
        0,
    );

    // Entity 13: Monolith wall — metal (material 2)
    registry.register_with_id(
        13,
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

    // Entity 14: Sphere — sandstone (material 1)
    registry.register_with_id(
        14,
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

    // ── Voxelize all objects into the grid ──────────────────────────────
    let grid = grid_set.grid_mut(0);
    voxelize_all_objects(&mut pool, grid, &aabb, SCENE_TIER, &registry, 0);

    log::info!(
        "Clipmap scene: {} levels, {} total bricks, {} registered objects",
        config.num_levels(),
        pool.allocated_count(),
        registry.len(),
    );
    (pool, grid_set, config, aabb, aabbs, registry)
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
    // Resolution (dynamically set from viewport)
    pub display_width: u32,
    pub display_height: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    /// Viewport rect (x, y, w, h) for blit and line overlay set_viewport.
    pub viewport_rect: Option<(u32, u32, u32, u32)>,

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
    /// Padded bytes per row for staging buffer (aligned to COPY_BYTES_PER_ROW_ALIGNMENT).
    staging_padded_bpr: u32,
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
    /// `display_width`/`display_height` set the output resolution; internal
    /// resolution is computed as 75% of display.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        shared_state: Arc<Mutex<SharedState>>,
        display_width: u32,
        display_height: u32,
    ) -> Self {
        let internal_width = (display_width as f32 * 0.75).round() as u32;
        let internal_height = (display_height as f32 * 0.75).round() as u32;
        // Scene
        let (pool, grid_set, clipmap_config, aabb, level_aabbs, object_registry) =
            create_clipmap_scene();
        let grid = grid_set.grid(0);

        // Update shared state with pool info and frame dimensions
        {
            let mut state = shared_state.lock().unwrap();
            state.pool_capacity = pool.capacity() as u64;
            state.pool_allocated = pool.allocated_count() as u64;
            state.frame_pixels = vec![0u8; (display_width * display_height * 4) as usize];
            state.frame_width = display_width;
            state.frame_height = display_height;
        }

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.yaw = 0.0;
        camera.pitch = -0.15;
        camera.fov_degrees = 70.0;

        let camera_uniforms =
            camera.uniforms(internal_width, internal_height, 0, [[0.0; 4]; 4]);
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

        let gbuffer = GBuffer::new(device, internal_width, internal_height);

        let lights = create_clipmap_lights();
        let light_buffer = LightBuffer::upload(device, &lights);

        let tile_cull = TileCullPass::new(
            device,
            &gbuffer,
            &light_buffer,
            internal_width,
            internal_height,
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
            internal_width,
            internal_height,
        );

        let vol_shadow = VolShadowPass::new(device, queue, &scene.bind_group_layout);
        let cloud_shadow = CloudShadowPass::new(device);

        let half_w = internal_width / 2;
        let half_h = internal_height / 2;
        let vol_march = VolMarchPass::new(
            device,
            queue,
            &gbuffer.position_view,
            &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w,
            half_h,
            internal_width,
            internal_height,
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
            internal_width,
            internal_height,
            half_w,
            half_h,
        );

        let vol_composite = VolCompositePass::new(
            device,
            &shading.hdr_view,
            &vol_upscale.output_view,
            internal_width,
            internal_height,
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
            internal_width,
            internal_height,
        );
        dof.update_focus(queue, 5.0, 100.0, 0.0);

        let motion_blur = MotionBlurPass::new(
            device,
            &dof.output_view,
            &gbuffer.motion_view,
            internal_width,
            internal_height,
        );
        motion_blur.set_intensity(queue, 0.0);

        let bloom = BloomPass::new(
            device,
            &motion_blur.output_view,
            internal_width,
            internal_height,
        );

        let history = HistoryBuffers::new(
            device,
            display_width,
            display_height,
            internal_width,
            internal_height,
        );
        let upscale = UpscalePass::new(
            device,
            &motion_blur.output_view,
            &gbuffer,
            &history,
            display_width,
            display_height,
            internal_width,
            internal_height,
        );
        let sharpen = SharpenPass::new(
            device,
            &upscale.output_view,
            &gbuffer,
            display_width,
            display_height,
        );

        let bloom_composite = BloomCompositePass::new(
            device,
            &sharpen.output_view,
            bloom.mip_views(),
            display_width,
            display_height,
        );

        let auto_exposure = AutoExposurePass::new(
            device,
            &bloom_composite.output_view,
            display_width,
            display_height,
        );

        let tone_map = ToneMapPass::new_with_exposure(
            device,
            &bloom_composite.output_view,
            display_width,
            display_height,
            Some(auto_exposure.get_exposure_buffer()),
        );

        let color_grade =
            ColorGradePass::new(device, queue, &tone_map.ldr_view, display_width, display_height);

        let cosmetics =
            CosmeticsPass::new(device, &color_grade.output_view, display_width, display_height);

        bloom.set_threshold(queue, 0.8, 0.4);
        bloom_composite.set_intensity(queue, 0.0);
        cosmetics.set_vignette(queue, 0.0);
        cosmetics.set_chromatic_aberration(queue, 0.0);

        let blit = BlitPass::new(device, &cosmetics.output_view, surface_format);

        let staging_padded_bpr = padded_bytes_per_row(display_width);
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("editor screenshot staging"),
            size: (staging_padded_bpr * display_height) as u64,
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
            display_width,
            display_height,
            internal_width,
            internal_height,
            viewport_rect: None,
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
            staging_padded_bpr,
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

    /// Rebuild all resolution-dependent GPU passes for a new viewport size.
    ///
    /// Preserves CPU-side state (pool, grids, objects, camera, lights) and only
    /// recreates GPU textures/passes that depend on display or internal resolution.
    pub fn rebuild_display_passes(&mut self, display_w: u32, display_h: u32) {
        self.display_width = display_w;
        self.display_height = display_h;
        self.internal_width = (display_w as f32 * 0.75).round() as u32;
        self.internal_height = (display_h as f32 * 0.75).round() as u32;

        let iw = self.internal_width;
        let ih = self.internal_height;
        let dw = self.display_width;
        let dh = self.display_height;

        // Internal-resolution passes
        self.gbuffer = GBuffer::new(&self.device, iw, ih);
        self.ray_march = RayMarchPass::new(&self.device, &self.scene, &self.gbuffer, &self.clipmap);
        self.tile_cull = TileCullPass::new(
            &self.device, &self.gbuffer, &self.light_buffer, iw, ih,
        );
        self.shading = ShadingPass::new(
            &self.device, &self.gbuffer, &self.material_table, &self.scene,
            &self.tile_cull.shade_light_bind_group_layout,
            &self.radiance_volume, &self.color_pool, iw, ih,
        );

        let half_w = iw / 2;
        let half_h = ih / 2;
        self.vol_march = VolMarchPass::new(
            &self.device, &self.queue,
            &self.gbuffer.position_view,
            &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w, half_h, iw, ih,
        );
        self.vol_temporal = VolTemporalPass::new(
            &self.device, &self.vol_march.output_view,
            &self.gbuffer.motion_view, half_w, half_h,
        );
        self.vol_upscale = VolUpscalePass::new(
            &self.device, &self.vol_march.output_view,
            &self.gbuffer.position_view, iw, ih, half_w, half_h,
        );
        self.vol_composite = VolCompositePass::new(
            &self.device, &self.shading.hdr_view,
            &self.vol_upscale.output_view, iw, ih,
        );
        self.dof = DofPass::new(
            &self.device, &self.vol_composite.output_view,
            &self.gbuffer.position_view, iw, ih,
        );
        self.dof.update_focus(&self.queue, 5.0, 100.0, 0.0);
        self.motion_blur = MotionBlurPass::new(
            &self.device, &self.dof.output_view,
            &self.gbuffer.motion_view, iw, ih,
        );
        self.motion_blur.set_intensity(&self.queue, 0.0);
        self.bloom = BloomPass::new(
            &self.device, &self.motion_blur.output_view, iw, ih,
        );

        // Display-resolution passes
        self.history = HistoryBuffers::new(&self.device, dw, dh, iw, ih);
        self.upscale = UpscalePass::new(
            &self.device, &self.motion_blur.output_view, &self.gbuffer,
            &self.history, dw, dh, iw, ih,
        );
        self.sharpen = SharpenPass::new(
            &self.device, &self.upscale.output_view, &self.gbuffer, dw, dh,
        );
        self.bloom_composite = BloomCompositePass::new(
            &self.device, &self.sharpen.output_view,
            self.bloom.mip_views(), dw, dh,
        );
        self.auto_exposure = AutoExposurePass::new(
            &self.device, &self.bloom_composite.output_view, dw, dh,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            &self.device, &self.bloom_composite.output_view, dw, dh,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            &self.device, &self.queue, &self.tone_map.ldr_view, dw, dh,
        );
        self.cosmetics = CosmeticsPass::new(
            &self.device, &self.color_grade.output_view, dw, dh,
        );

        // Re-apply post-process settings
        self.bloom.set_threshold(&self.queue, 0.8, 0.4);
        self.bloom_composite.set_intensity(&self.queue, 0.0);
        self.cosmetics.set_vignette(&self.queue, 0.0);
        self.cosmetics.set_chromatic_aberration(&self.queue, 0.0);

        self.blit = BlitPass::new(
            &self.device, &self.cosmetics.output_view, self.surface_format,
        );

        // Rebuild staging buffer for screenshots
        self.staging_padded_bpr = padded_bytes_per_row(dw);
        self.staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("editor screenshot staging"),
            size: (self.staging_padded_bpr * dh) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Update shared state frame buffer
        if let Ok(mut state) = self.shared_state.lock() {
            state.frame_pixels = vec![0u8; (dw * dh * 4) as usize];
            state.frame_width = dw;
            state.frame_height = dh;
        }

        log::info!(
            "Display passes rebuilt: {}x{} display, {}x{} internal",
            dw, dh, iw, ih,
        );
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

    /// Expand a clipmap level's grid to encompass the given required AABB.
    ///
    /// Frees all bricks for the level, creates a new larger grid, re-voxelizes
    /// all objects in the level, and recreates GPU resources (`scene` and `clipmap`).
    fn expand_grid_for_level(&mut self, level: usize, required: &Aabb) {
        let tier = SCENE_TIER;
        let res = &RESOLUTION_TIERS[tier];
        let brick_extent = res.brick_extent;

        // 1. Compute new radius: half the required span + 50% margin
        let required_half = required.half_extents();
        let max_half = required_half.x.max(required_half.y).max(required_half.z);
        let new_radius = max_half * 1.5; // 50% margin
        let new_radius = new_radius.max(self.clipmap_config.level(level).radius); // never shrink

        // 2. Compute new grid dimensions
        let diameter = 2.0 * new_radius;
        let dim = (diameter / brick_extent).ceil() as u32;
        let new_dims = glam::UVec3::splat(dim.min(128)); // cap to prevent huge grids

        // If dims were capped, clamp radius to what the grid can actually cover.
        // Without this, the AABB origin shifts far negative and objects near the
        // origin end up outside the grid.
        let actual_radius = (new_dims.x as f32 * brick_extent * 0.5).min(new_radius);

        // 3. Free old bricks
        let old_grid = self.cpu_grid_set.grid(level);
        free_level_bricks(&mut self.cpu_pool, old_grid);

        // 4. Create new grid and swap into the grid set
        let new_grid = SparseGrid::new(new_dims);
        let new_config = rkf_core::clipmap::ClipmapLevel {
            voxel_size: res.voxel_size,
            radius: actual_radius,
        };
        self.cpu_grid_set.replace_level(level, new_config, new_grid);
        self.clipmap_config.set_level(level, new_config);

        // 5. Update level AABB — always centered at origin
        let half = actual_radius;
        let new_aabb = Aabb::new(
            Vec3::new(-half, -half, -half),
            Vec3::new(half, half, half),
        );
        self.level_aabbs[level] = new_aabb;

        // 6. Re-voxelize all objects for this level
        let grid = self.cpu_grid_set.grid_mut(level);
        voxelize_all_objects(
            &mut self.cpu_pool, grid, &new_aabb, tier,
            &self.object_registry, level,
        );

        // 7. Recreate GpuScene from level-0 data
        let grid0 = self.cpu_grid_set.grid(0);
        let dims0 = grid0.dimensions();
        let aabb0 = &self.level_aabbs[0];
        let scene_uniforms = SceneUniforms {
            grid_dims: [dims0.x, dims0.y, dims0.z, 0],
            grid_origin: [
                aabb0.min.x,
                aabb0.min.y,
                aabb0.min.z,
                RESOLUTION_TIERS[SCENE_TIER].brick_extent,
            ],
            params: [RESOLUTION_TIERS[SCENE_TIER].voxel_size, 0.0, 0.0, 0.0],
        };
        let camera_uniforms = self.camera.uniforms(
            self.internal_width, self.internal_height, self.frame_index, self.prev_vp,
        );
        let camera_bytes = bytemuck::bytes_of(&camera_uniforms);
        self.scene = GpuScene::upload(&self.device, &self.cpu_pool, grid0, camera_bytes, &scene_uniforms);

        // 8. Recreate ClipmapGpuData from full grid set
        // Grid is world-centered (origin 0,0,0), not camera-centered
        self.clipmap = ClipmapGpuData::upload(
            &self.device, &self.cpu_grid_set,
            [0.0, 0.0, 0.0],
        );

        // 9. Rebuild render passes that hold bind groups referencing scene/clipmap
        self.ray_march = RayMarchPass::new(&self.device, &self.scene, &self.gbuffer, &self.clipmap);
        self.shading = ShadingPass::new(
            &self.device, &self.gbuffer, &self.material_table, &self.scene,
            &self.tile_cull.shade_light_bind_group_layout,
            &self.radiance_volume, &self.color_pool,
            self.internal_width, self.internal_height,
        );
        self.radiance_inject = RadianceInjectPass::new(
            &self.device, &self.scene, &self.material_table,
            &self.light_buffer, &self.radiance_volume,
        );
        self.radiance_inject.update_inject_uniforms(
            &self.queue, self.lights.len() as u32, 2,
        );
        self.vol_shadow = VolShadowPass::new(
            &self.device, &self.queue, &self.scene.bind_group_layout,
        );

        // Rebuild volumetric march (depends on vol_shadow views)
        let half_w = self.internal_width / 2;
        let half_h = self.internal_height / 2;
        self.vol_march = VolMarchPass::new(
            &self.device, &self.queue,
            &self.gbuffer.position_view,
            &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w, half_h,
            self.internal_width, self.internal_height,
        );
        self.vol_temporal = VolTemporalPass::new(
            &self.device, &self.vol_march.output_view,
            &self.gbuffer.motion_view, half_w, half_h,
        );
        self.vol_upscale = VolUpscalePass::new(
            &self.device, &self.vol_march.output_view,
            &self.gbuffer.position_view,
            self.internal_width, self.internal_height,
            half_w, half_h,
        );

        // 10. Rebuild downstream post-shading chain (these hold bind groups
        //     referencing the old shading.hdr_view / vol_upscale.output_view).
        let iw = self.internal_width;
        let ih = self.internal_height;
        let dw = self.display_width;
        let dh = self.display_height;

        self.vol_composite = VolCompositePass::new(
            &self.device, &self.shading.hdr_view,
            &self.vol_upscale.output_view, iw, ih,
        );
        self.dof = DofPass::new(
            &self.device, &self.vol_composite.output_view,
            &self.gbuffer.position_view, iw, ih,
        );
        self.dof.update_focus(&self.queue, 5.0, 100.0, 0.0);
        self.motion_blur = MotionBlurPass::new(
            &self.device, &self.dof.output_view,
            &self.gbuffer.motion_view, iw, ih,
        );
        self.motion_blur.set_intensity(&self.queue, 0.0);
        self.bloom = BloomPass::new(
            &self.device, &self.motion_blur.output_view, iw, ih,
        );

        self.history = HistoryBuffers::new(&self.device, dw, dh, iw, ih);
        self.upscale = UpscalePass::new(
            &self.device, &self.motion_blur.output_view, &self.gbuffer,
            &self.history, dw, dh, iw, ih,
        );
        self.sharpen = SharpenPass::new(
            &self.device, &self.upscale.output_view, &self.gbuffer, dw, dh,
        );
        self.bloom_composite = BloomCompositePass::new(
            &self.device, &self.sharpen.output_view,
            self.bloom.mip_views(), dw, dh,
        );
        self.auto_exposure = AutoExposurePass::new(
            &self.device, &self.bloom_composite.output_view, dw, dh,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            &self.device, &self.bloom_composite.output_view, dw, dh,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            &self.device, &self.queue, &self.tone_map.ldr_view, dw, dh,
        );
        self.cosmetics = CosmeticsPass::new(
            &self.device, &self.color_grade.output_view, dw, dh,
        );

        self.bloom.set_threshold(&self.queue, 0.8, 0.4);
        self.bloom_composite.set_intensity(&self.queue, 0.0);
        self.cosmetics.set_vignette(&self.queue, 0.0);
        self.cosmetics.set_chromatic_aberration(&self.queue, 0.0);

        self.blit = BlitPass::new(
            &self.device, &self.cosmetics.output_view, self.surface_format,
        );

        // Update shared state pool info
        if let Ok(mut state) = self.shared_state.lock() {
            state.pool_allocated = self.cpu_pool.allocated_count() as u64;
        }

        log::info!(
            "Grid expanded: level {level}, new radius {new_radius:.1}m, dims {dim}³, {} bricks",
            self.cpu_pool.allocated_count(),
        );
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

        // Only level 0 is supported in the current single-level scene
        if level != 0 {
            log::warn!("apply_object_transform: unsupported level {level}");
            return false;
        }
        let tier = SCENE_TIER;
        let grid_aabb = match self.level_aabbs.get(level) {
            Some(a) => *a,
            None => return false,
        };

        // Preview the object's AABB at the new transform to check grid bounds
        {
            let local = self.object_registry.get(object_id).unwrap().recipe.primitive.local_aabb();
            let corners = local.corners();
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            for corner in &corners {
                let world = new_transform.rotation * (*corner * new_transform.scale)
                    + new_transform.position;
                min = min.min(world);
                max = max.max(world);
            }
            let margin = Vec3::splat(0.64);
            let new_world_aabb = Aabb::new(min - margin, max + margin);

            if !grid_aabb.contains_aabb(&new_world_aabb) {
                // Update transform in registry first, then expand + re-voxelize all
                self.object_registry.get_mut(object_id).unwrap().transform = new_transform;
                let required = self.object_registry.level_bounds(level)
                    .unwrap_or(new_world_aabb);
                self.expand_grid_for_level(level, &required);
                return true;
            }
        }

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

        // Upload dirty bricks (brick pool is shared, lives in GpuScene)
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

        // Write to clipmap buffers (ray marcher reads from these)
        self.clipmap.write_dirty_occupancy(&self.queue, grid, level, &dirty_occ_words);
        self.clipmap.write_dirty_slots(&self.queue, grid, level, &result.dirty_cell_indices);

        // Also write to GpuScene buffers for level 0 (radiance inject, vol_shadow, etc. read these)
        if level == 0 {
            self.scene.write_dirty_occupancy(&self.queue, grid, &dirty_occ_words);
            self.scene.write_dirty_slots(&self.queue, grid, &result.dirty_cell_indices);
        }

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
                .uniforms(self.internal_width, self.internal_height, self.frame_index, self.prev_vp);
        uniforms.jitter = [0.0, 0.0]; // spatial-only upscaling

        let vp = self
            .camera
            .view_projection(self.internal_width, self.internal_height);
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
        let aspect = self.internal_width as f32 / self.internal_height as f32;
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
            width: self.internal_width / 2,
            height: self.internal_height / 2,
            full_width: self.internal_width,
            full_height: self.internal_height,
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
            viewport_rect: self.viewport_rect.map(|(x, y, w, h)| {
                (x as f32, y as f32, w as f32, h as f32)
            }),
        };
        execute_frame(&mut ctx);

        // ── Line overlay pass (gizmos, wireframes) ──────────────────────
        if let Some(lines) = line_batch {
            if !lines.is_empty() {
                // Compute VP at display resolution for line overlay
                let line_vp = self
                    .camera
                    .view_projection(self.display_width, self.display_height);
                self.queue.write_buffer(
                    &self.line_vp_buffer,
                    0,
                    bytemuck::bytes_of(&line_vp),
                );

                // Billboard quad generation: each segment → 6 GPU vertices
                let cam_pos = self.camera.position;
                let fov_half_tan =
                    (self.camera.fov_degrees.to_radians() / 2.0).tan();
                let vh = self.display_height as f32;

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
                    if let Some((vx, vy, vw, vh)) = self.viewport_rect {
                        rpass.set_viewport(
                            vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0,
                        );
                    }
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
                    bytes_per_row: Some(self.staging_padded_bpr),
                    rows_per_image: Some(self.display_height),
                },
            },
            wgpu::Extent3d {
                width: self.display_width,
                height: self.display_height,
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
            let unpadded_bpr = (self.display_width * 4) as usize;
            let padded_bpr = self.staging_padded_bpr as usize;
            if let Ok(mut state) = self.shared_state.lock() {
                if unpadded_bpr == padded_bpr {
                    // No padding — direct copy
                    state.frame_pixels.copy_from_slice(&data);
                } else {
                    // Strip row padding
                    for row in 0..self.display_height as usize {
                        let src_start = row * padded_bpr;
                        let dst_start = row * unpadded_bpr;
                        state.frame_pixels[dst_start..dst_start + unpadded_bpr]
                            .copy_from_slice(&data[src_start..src_start + unpadded_bpr]);
                    }
                }
            }
            drop(data);
            self.staging_buffer.unmap();
        }
    }
}
