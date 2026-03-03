//! Editor engine — v2 object-centric render pipeline.
//!
//! Extracted from the testbed's `EngineState`. Provides the full v2 compute-shader
//! render pipeline (ray march, shading, GI, volumetrics, post-processing) in a
//! reusable struct that the editor's event loop drives each frame.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::{Quat, Vec3};
use winit::window::Window;

use rkf_core::{
    Aabb, BrickMapAllocator, BrickPool, Scene, SceneNode, SceneObject,
    SdfPrimitive, SdfSource, voxelize_sdf,
    transform_flatten::flatten_object,
};
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuObject, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::{RadianceInjectPass, InjectUniforms};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::{MaterialTable, create_test_materials};
use rkf_animation::character::{
    AnimatedCharacter, build_humanoid_skeleton, build_humanoid_visuals, build_walk_clip,
};

use crate::automation::SharedState;
use crate::camera::EditorCamera;
use crate::engine_viewport::RENDER_SCALE;

/// Internal render resolution width (used by the legacy surface-based path).
pub const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height (used by the legacy surface-based path).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Offscreen render target format for the compositor path.
///
/// sRGB variant so the blit pass's linear→sRGB conversion happens in hardware,
/// matching the current swapchain behavior. The compositor samples this texture
/// (auto-decoding sRGB→linear) and writes to rinch's sRGB swapchain.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// ---------------------------------------------------------------------------
// Demo scene
// ---------------------------------------------------------------------------

/// Build result containing the scene and CPU brick data for GPU upload.
struct DemoScene {
    scene: Scene,
    brick_pool: BrickPool,
    brick_map_alloc: BrickMapAllocator,
    character: AnimatedCharacter,
    character_obj_index: usize,
}

/// Try to load a voxelized object from a .rkf file into the brick pool.
///
/// Returns `(BrickMapHandle, voxel_size, grid_aabb, brick_count)` on success.
fn load_rkf_into_pool(
    path: &str,
    pool: &mut rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
    alloc: &mut BrickMapAllocator,
) -> Result<(rkf_core::scene_node::BrickMapHandle, f32, Aabb, u32), String> {
    use rkf_core::asset_file::{load_object_header, load_object_lod};
    use rkf_core::brick_map::EMPTY_SLOT;
    use std::io::BufReader;

    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut reader = BufReader::new(file);

    let header = load_object_header(&mut reader).map_err(|e| format!("header: {e}"))?;
    if header.lod_entries.is_empty() {
        return Err("no LOD levels in .rkf".into());
    }

    // Load the finest LOD (last entry, since they're sorted coarsest-first).
    let finest_idx = header.lod_entries.len() - 1;
    let lod = load_object_lod(&mut reader, &header, finest_idx)
        .map_err(|e| format!("lod: {e}"))?;

    let voxel_size = header.lod_entries[finest_idx].voxel_size;
    let brick_count = lod.brick_data.len() as u32;

    // Allocate pool slots for all bricks.
    let slots = pool.allocate_range(brick_count)
        .ok_or_else(|| format!("pool full: need {brick_count} bricks"))?;

    // Build a new BrickMap with real pool slot indices, and copy brick data.
    let dims = lod.brick_map.dims;
    let mut brick_map = rkf_core::brick_map::BrickMap::new(dims);
    let mut slot_idx = 0usize;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let local_idx = lod.brick_map.get(bx, by, bz).unwrap_or(EMPTY_SLOT);
                if local_idx == EMPTY_SLOT {
                    continue;
                }

                let pool_slot = slots[slot_idx];
                slot_idx += 1;
                brick_map.set(bx, by, bz, pool_slot);

                // Copy voxel data into the pool brick.
                let src = &lod.brick_data[local_idx as usize];
                let dst = pool.get_mut(pool_slot);
                dst.voxels.copy_from_slice(src);
            }
        }
    }

    // Register the brick map in the allocator.
    let handle = alloc.allocate(&brick_map);

    // Compute grid-aligned AABB from dims.
    let brick_world_size = voxel_size * 8.0;
    let grid_half = Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );
    let grid_aabb = Aabb::new(-grid_half, grid_half);

    log::info!(
        "Loaded {path}: {brick_count} bricks, dims={dims:?}, voxel_size={voxel_size}"
    );

    Ok((handle, voxel_size, grid_aabb, brick_count))
}

/// Build the demo scene.
///
/// If `scenes/test_cross.rkf` exists, loads it as the primary voxelized object.
/// Otherwise falls back to an inline voxelized sphere.
fn build_demo_scene() -> DemoScene {
    let mut scene = Scene::new("editor_demo");

    // Ground plane
    let ground = SceneNode::analytical("ground", SdfPrimitive::Box {
        half_extents: Vec3::new(10.0, 0.1, 10.0),
    }, 1);
    let ground_obj = SceneObject {
        id: 0,
        name: "ground".into(),
        parent_id: None,
        position: Vec3::new(0.0, -0.8, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: ground,
        aabb: Aabb::new(Vec3::new(-10.0, -0.1, -10.0), Vec3::new(10.0, 0.1, 10.0)),
    };
    scene.add_object_full(ground_obj);

    let mut brick_pool = BrickPool::new(4096);
    let mut brick_map_alloc = BrickMapAllocator::new();

    // Try loading from .rkf file on disk.
    let rkf_path = "scenes/test_cross.rkf";
    match load_rkf_into_pool(rkf_path, &mut brick_pool, &mut brick_map_alloc) {
        Ok((handle, voxel_size, grid_aabb, _brick_count)) => {
            let mut vox_node = SceneNode::new("vox_cross");
            vox_node.sdf_source = SdfSource::Voxelized {
                brick_map_handle: handle,
                voxel_size,
                aabb: grid_aabb,
            };
            let vox_obj = SceneObject {
                id: 0,
                name: "vox_cross".into(),
                parent_id: None,
                position: Vec3::new(0.0, 0.0, -2.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: vox_node,
                aabb: grid_aabb,
            };
            scene.add_object_full(vox_obj);
        }
        Err(e) => {
            log::warn!("Failed to load {rkf_path}: {e} — falling back to inline sphere");

            // Fallback: inline voxelized sphere
            let vox_radius = 0.4;
            let voxel_size = 0.04;
            let margin = voxel_size * 2.0;
            let vox_aabb = Aabb::new(
                Vec3::splat(-vox_radius - margin),
                Vec3::splat(vox_radius + margin),
            );
            let sdf_fn = |pos: Vec3| -> (f32, u16) {
                (pos.length() - vox_radius, 6u16)
            };
            let (handle, brick_count) = voxelize_sdf(
                sdf_fn, &vox_aabb, voxel_size, &mut brick_pool, &mut brick_map_alloc,
            ).expect("voxelize sphere");

            let vox_brick_size = voxel_size * 8.0;
            let vox_grid_half = Vec3::new(
                handle.dims.x as f32 * vox_brick_size * 0.5,
                handle.dims.y as f32 * vox_brick_size * 0.5,
                handle.dims.z as f32 * vox_brick_size * 0.5,
            );
            let vox_grid_aabb = Aabb::new(-vox_grid_half, vox_grid_half);

            log::info!(
                "Voxelized sphere: {} bricks, handle offset={} dims={:?}",
                brick_count, handle.offset, handle.dims
            );

            let mut vox_node = SceneNode::new("vox_sphere");
            vox_node.sdf_source = SdfSource::Voxelized {
                brick_map_handle: handle,
                voxel_size,
                aabb: vox_grid_aabb,
            };
            let vox_obj = SceneObject {
                id: 0,
                name: "vox_sphere".into(),
                parent_id: None,
                position: Vec3::new(0.0, 0.0, -2.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                root_node: vox_node,
                aabb: vox_grid_aabb,
            };
            scene.add_object_full(vox_obj);
        }
    }

    // Animated humanoid character (required by DemoScene struct)
    let skeleton = build_humanoid_skeleton();
    let visuals = build_humanoid_visuals(5);
    let walk_clip = build_walk_clip();
    let character = AnimatedCharacter::new(skeleton, visuals, walk_clip, 0.08);
    let char_root = character.build_scene_node();
    let char_obj = SceneObject {
        id: 0,
        name: "humanoid".into(),
        parent_id: None,
        position: Vec3::new(-3.0, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: char_root,
        aabb: Aabb::new(Vec3::new(-0.6, -0.5, -0.5), Vec3::new(0.6, 2.0, 0.5)),
    };
    scene.add_object_full(char_obj);
    let character_obj_index = scene.objects.len() - 1;

    DemoScene {
        scene,
        brick_pool,
        brick_map_alloc,
        character,
        character_obj_index,
    }
}

/// Compute the diameter of an analytical SDF primitive's bounding sphere.
fn primitive_diameter(prim: &rkf_core::SdfPrimitive) -> f32 {
    use rkf_core::SdfPrimitive;
    match *prim {
        SdfPrimitive::Sphere { radius } => radius * 2.0,
        SdfPrimitive::Box { half_extents } => half_extents.length() * 2.0,
        SdfPrimitive::Capsule { radius, half_height } => {
            (radius + half_height) * 2.0
        }
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            (major_radius + minor_radius) * 2.0
        }
        SdfPrimitive::Cylinder { radius, half_height } => {
            Vec3::new(radius, half_height, radius).length() * 2.0
        }
        SdfPrimitive::Plane { .. } => 2.0, // planes get default size
    }
}

// ---------------------------------------------------------------------------
// Editor engine
// ---------------------------------------------------------------------------

/// GPU render engine for the editor window.
///
/// Contains the full v2 compute-shader pipeline: ray march, shading, GI,
/// volumetrics, and post-processing. Drives rendering each frame and provides
/// GPU readback for MCP screenshots.
pub struct EditorEngine {
    ctx: RenderContext,
    // Surface-based rendering (legacy path — used when engine owns the window).
    surface: Option<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    // Offscreen rendering (compositor path — used when rinch owns the window).
    offscreen_texture: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,
    offscreen_blit: Option<BlitPass>,
    /// Viewport size in physical pixels (set by RenderSurface layout).
    viewport_width: u32,
    viewport_height: u32,
    gpu_scene: GpuSceneV2,
    gbuffer: GBuffer,
    tile_cull: TileObjectCullPass,
    coarse_field: CoarseField,
    ray_march: RayMarchPass,
    #[allow(dead_code)]
    debug_view: DebugViewPass,
    shading_pass: ShadingPass,
    radiance_volume: RadianceVolume,
    radiance_inject: RadianceInjectPass,
    radiance_mip: RadianceMipPass,
    // Volumetric pipeline
    vol_shadow: VolShadowPass,
    cloud_shadow: CloudShadowPass,
    vol_march: VolMarchPass,
    vol_upscale: VolUpscalePass,
    vol_composite: VolCompositePass,
    // Post-processing pipeline
    god_rays_blur: GodRaysBlurPass,
    bloom: BloomPass,
    auto_exposure: AutoExposurePass,
    dof: DofPass,
    motion_blur: MotionBlurPass,
    bloom_composite: BloomCompositePass,
    tone_map: ToneMapPass,
    color_grade: ColorGradePass,
    cosmetics: CosmeticsPass,
    #[allow(dead_code)]
    sharpen: SharpenPass,
    blit: BlitPass,
    // State
    camera: Camera,
    pub world_lights: Vec<Light>,
    light_buffer: LightBuffer,
    material_buffer: wgpu::Buffer,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    shade_debug_mode: u32,
    // Cached environment vol params (updated by apply_environment).
    env_sun_dir: [f32; 3],
    env_sun_color: [f32; 3],
    env_fog_color: [f32; 3],
    env_fog_density: f32,
    env_fog_height_falloff: f32,
    // Atmosphere params for analytic sky.
    env_sun_intensity: f32,
    env_sun_color_raw: [f32; 3],
    env_rayleigh_scale: f32,
    env_mie_scale: f32,
    env_atmosphere_enabled: bool,
    // Dust params for god rays.
    env_ambient_dust: f32,
    env_dust_g: f32,
    // Cloud params from environment.
    env_cloud_settings: rkf_render::CloudSettings,
    accumulated_time: f32,
    // Render resolution (tracks viewport physical pixels)
    render_width: u32,
    render_height: u32,
    // GPU object picking (single-pixel readback from material G-buffer)
    pick_readback_buffer: wgpu::Buffer,
    // GPU brush hit readback (single-pixel from position G-buffer, Rgba32Float = 16 bytes)
    brush_readback_buffer: wgpu::Buffer,
    // Screenshot readback (window-resolution, captures composited output with UI)
    readback_buffer: wgpu::Buffer,
    window_width: u32,
    window_height: u32,
    shared_state: Arc<Mutex<SharedState>>,
    wireframe_pass: Option<crate::wireframe::WireframePass>,
    character: Option<AnimatedCharacter>,
    character_obj_index: Option<usize>,
    last_frame_time: Instant,
    // CPU-side brick data retained for re-voxelize operations.
    cpu_brick_pool: BrickPool,
    cpu_brick_map_alloc: BrickMapAllocator,
    // GPU JFA pipeline for per-stroke SDF repair.
    jfa_sdf: crate::jfa_sdf::JfaSdfPass,
}

impl EditorEngine {
    /// Initialize the render engine with wgpu, all passes, and a demo scene.
    ///
    /// Returns `(engine, scene)` — the caller stores the scene in EditorState.
    pub fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> (Self, Scene) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let ctx = RenderContext::new(&instance, &surface);

        let size = window.inner_size();
        // Configure surface with COPY_SRC so we can read back the composited
        // frame (engine + UI overlay) for MCP screenshots.
        let surface_format = Self::configure_surface_copy(
            &ctx, &surface, size.width, size.height,
        );

        // Build demo scene.
        let demo = build_demo_scene();
        let scene = demo.scene;

        // Upload brick pool to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(demo.brick_pool.as_slice());
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: pool_data.len().max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !pool_data.is_empty() {
            ctx.queue.write_buffer(&brick_pool_buffer, 0, pool_data);
        }

        let mut gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);

        // Upload brick maps.
        let brick_map_data = demo.brick_map_alloc.as_slice();
        if !brick_map_data.is_empty() {
            gpu_scene.upload_brick_maps(&ctx.device, &ctx.queue, brick_map_data);
        }

        let gbuffer = GBuffer::new(&ctx.device, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Coarse acceleration field — compute world-space AABBs from local AABBs + baked transforms.
        let init_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let init_default_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = init_transforms.get(&obj.id).unwrap_or(&init_default_wt);
                let smin = obj.aabb.min * wt.scale;
                let smax = obj.aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z), Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z), Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z), Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z), Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                (wmin, wmax)
            })
            .collect();
        let mut coarse_field = CoarseField::from_scene_aabbs(
            &ctx.device, &scene_aabbs, COARSE_VOXEL_SIZE, 1.0,
        );
        coarse_field.populate(&scene_aabbs);
        coarse_field.upload(&ctx.queue, Vec3::ZERO);
        log::info!(
            "Coarse field: {}x{}x{} cells, voxel_size={}m",
            coarse_field.dims.x, coarse_field.dims.y, coarse_field.dims.z,
            coarse_field.voxel_size,
        );

        let ray_march = RayMarchPass::new(
            &ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field,
        );
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material table.
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);
        log::info!("Material table: {} materials uploaded", material_table.count);

        // Lights (point/spot only — directional sun is synthesized from environment each frame).
        let world_lights = vec![
            Light::point([2.0, 1.5, -1.0], [1.0, 0.8, 0.5], 5.0, 8.0, true),
            Light::point([-2.0, 1.0, -3.0], [0.5, 0.7, 1.0], 3.0, 6.0, false),
        ];
        // Allocate light buffer with headroom (1 sun + up to 63 point/spot).
        // Slot 0 is reserved for the sun light synthesized from environment each frame.
        let mut init_lights = Vec::with_capacity(64);
        init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false)); // placeholder for sun
        init_lights.extend(&world_lights);
        // Pad to 64 lights so the buffer can accommodate runtime additions.
        while init_lights.len() < 64 {
            init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        }
        let light_buffer = LightBuffer::upload(&ctx.device, &init_lights);
        log::info!("Lights: {} point/spot + 1 env sun", world_lights.len());

        // Radiance volume for GI.
        let radiance_volume = RadianceVolume::new(&ctx.device);
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device, &gpu_scene, &material_table.buffer,
            &light_buffer, &radiance_volume, &coarse_field,
        );
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Shading pass.
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Volumetric pipeline.
        let vol_shadow = VolShadowPass::new(
            &ctx.device, &ctx.queue, &coarse_field.bind_group_layout,
        );
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = INTERNAL_WIDTH / 2;
        let half_h = INTERNAL_HEIGHT / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device, &ctx.queue,
            &gbuffer.position_view, &vol_shadow.shadow_view, &cloud_shadow.shadow_view,
            half_w, half_h, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device, &vol_march.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT, half_w, half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device, &shading_pass.hdr_view, &vol_upscale.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Post-processing pipeline.
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let bloom = BloomPass::new(
            &ctx.device, &god_rays_blur.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &god_rays_blur.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let dof = DofPass::new(
            &ctx.device, &god_rays_blur.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device, &dof.output_view, &gbuffer.motion_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device, &motion_blur.output_view, bloom.mip_views(),
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device, &bloom_composite.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device, &ctx.queue, &tone_map.ldr_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let cosmetics = CosmeticsPass::new(
            &ctx.device, &color_grade.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let sharpen = SharpenPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Blit to swapchain.
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, surface_format);

        // Render camera (synced from EditorCamera each frame).
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Pick readback buffer — 1 pixel from the R32Uint material texture.
        // bytes_per_row must be aligned to 256, so buffer is 256 bytes.
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback buffer — 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
        // bytes_per_row must be 256-aligned, so buffer is 256 bytes.
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Readback buffer for MCP screenshots — at window resolution so we
        // capture the composited output (engine viewport + rinch UI panels).
        let window_width = size.width.max(1);
        let window_height = size.height.max(1);
        let readback_buffer = Self::create_readback_buffer(
            &ctx.device, window_width, window_height,
        );

        let jfa_sdf = crate::jfa_sdf::JfaSdfPass::new(&ctx.device);

        let engine = Self {
            ctx,
            surface: Some(surface),
            surface_format,
            offscreen_texture: None,
            offscreen_view: None,
            offscreen_blit: None,
            viewport_width: 0,
            viewport_height: 0,
            gpu_scene,
            gbuffer,
            tile_cull,
            coarse_field,
            ray_march,
            debug_view,
            shading_pass,
            radiance_volume,
            radiance_inject,
            radiance_mip,
            vol_shadow,
            cloud_shadow,
            vol_march,
            vol_upscale,
            vol_composite,
            god_rays_blur,
            bloom,
            auto_exposure,
            dof,
            motion_blur,
            bloom_composite,
            tone_map,
            color_grade,
            cosmetics,
            sharpen,
            blit,
            camera,
            world_lights,
            light_buffer,
            material_buffer: material_table.buffer,
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            // Matches EnvironmentState defaults; first frame apply_environment
            // overwrites these with actual data model values.
            env_sun_dir: {
                let d = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
                [d.x, d.y, d.z]
            },
            env_sun_color: [3.0, 2.85, 2.55], // sun_color * sun_intensity ([1.0,0.95,0.85] * 3.0)
            env_fog_color: [0.7, 0.75, 0.8],
            env_fog_density: 0.0, // fog disabled by default
            env_fog_height_falloff: 0.1,
            env_sun_intensity: 3.0,
            env_sun_color_raw: [1.0, 0.95, 0.85],
            env_rayleigh_scale: 1.0,
            env_mie_scale: 1.0,
            env_atmosphere_enabled: true,
            env_ambient_dust: 0.005,
            env_dust_g: 0.3,
            env_cloud_settings: rkf_render::CloudSettings::default(),
            accumulated_time: 0.0,
            render_width: INTERNAL_WIDTH,
            render_height: INTERNAL_HEIGHT,
            pick_readback_buffer,
            brush_readback_buffer,
            readback_buffer,
            window_width,
            window_height,
            shared_state,
            wireframe_pass: None,
            character: Some(demo.character),
            character_obj_index: Some(demo.character_obj_index),
            last_frame_time: Instant::now(),
            cpu_brick_pool: demo.brick_pool,
            cpu_brick_map_alloc: demo.brick_map_alloc,
            jfa_sdf,
        };
        (engine, scene)
    }

    /// Fill 1 layer of SDF padding bricks around each voxelized object.
    ///
    /// At load time, narrow-band bricks go straight to EMPTY_SLOT with no
    /// transition. This causes gradient discontinuities (faceted normals) at
    /// the narrow-band boundary because `sample_voxel_at` returns a constant
    /// for EMPTY_SLOT bricks. This method allocates padding bricks and fills
    /// them with properly trilinear-sampled SDF values, making the gradient
    /// smooth across the boundary.
    pub fn init_sdf_padding(&mut self, scene: &mut Scene) {
        for obj in &mut scene.objects {
            let (handle, voxel_size, aabb_min) = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                    (*brick_map_handle, *voxel_size, aabb.min)
                }
                _ => continue,
            };

            // Mark interior empties first (needed for correct sign in padding).
            self.mark_interior_empties(&handle);

            // Create a fake EditOp covering the entire object so
            // ensure_sdf_consistency fills all boundary bricks.
            let half = Vec3::new(
                handle.dims.x as f32 * voxel_size * 4.0,
                handle.dims.y as f32 * voxel_size * 4.0,
                handle.dims.z as f32 * voxel_size * 4.0,
            );
            let fake_op = rkf_edit::edit_op::EditOp {
                object_id: obj.id,
                position: Vec3::ZERO,
                rotation: glam::Quat::IDENTITY,
                edit_type: rkf_edit::types::EditType::SmoothUnion,
                shape_type: rkf_edit::types::ShapeType::Sphere,
                dimensions: half,
                strength: 1.0,
                blend_k: 0.0,
                falloff: rkf_edit::types::FalloffCurve::Smooth,
                material_id: 0,
                secondary_id: 0,
                color_packed: 0,
            };

            let slots = self.ensure_sdf_consistency(&handle, voxel_size, aabb_min, &fake_op);
            if !slots.is_empty() {
                log::info!(
                    "init_sdf_padding: filled {} padding bricks for '{}'",
                    slots.len(), obj.name,
                );
            }
        }

        // Reupload all brick data + maps after padding.
        self.reupload_brick_data();
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }
    }

    /// Access the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.device
    }

    /// Access the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.ctx.queue
    }

    /// The swapchain texture format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Sync the render camera from the editor camera state.
    pub fn sync_camera(&mut self, editor_cam: &EditorCamera) {
        self.camera.position = editor_cam.position;
        self.camera.yaw = editor_cam.fly_yaw;
        self.camera.pitch = editor_cam.fly_pitch;
        self.camera.fov_degrees = editor_cam.fov_y.to_degrees();
    }

    /// Set the shading debug mode (0=normal, 1=normals, 2=positions, etc).
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.shade_debug_mode = mode;
    }

    /// Apply environment settings to the render pipeline.
    ///
    /// Updates volumetric params (sun, fog) and all post-processing passes
    /// from the editor's `EnvironmentState`. Only writes to the GPU when
    /// the environment is marked dirty, then clears the flag.
    pub fn apply_environment(&mut self, env: &mut crate::environment::EnvironmentState) {
        if !env.is_dirty() {
            return;
        }

        // Cache vol params for use in render_frame_inner.
        let atmo = &env.atmosphere;
        self.env_sun_dir = [atmo.sun_direction.x, atmo.sun_direction.y, atmo.sun_direction.z];

        // Tint sun color based on elevation: at low angles the light path
        // through the atmosphere is much longer, scattering away blue/green
        // and leaving orange/red (Rayleigh extinction approximation).
        // Uses fixed gentle coefficients — the user's rayleigh/mie_scale
        // control sky scattering, not direct sun color.
        let sun_elevation = atmo.sun_direction.y.asin(); // radians
        let base_color = atmo.sun_color;
        let tinted_color = {
            let path = (1.0 / sun_elevation.max(0.02).sin()).min(12.0);
            let tau = glam::Vec3::new(0.02, 0.06, 0.15);
            let extinction = glam::Vec3::new(
                (-tau.x * path).exp(),
                (-tau.y * path).exp(),
                (-tau.z * path).exp(),
            );
            base_color * extinction
        };
        let sc = tinted_color * atmo.sun_intensity;
        self.env_sun_color = [sc.x, sc.y, sc.z];
        self.env_sun_intensity = atmo.sun_intensity;
        self.env_sun_color_raw = [tinted_color.x, tinted_color.y, tinted_color.z];
        self.env_rayleigh_scale = atmo.rayleigh_scale;
        self.env_mie_scale = atmo.mie_scale;
        self.env_atmosphere_enabled = atmo.enabled;

        let fog = &env.fog;
        self.env_fog_color = [fog.color.x, fog.color.y, fog.color.z];
        self.env_fog_density = if fog.enabled { fog.density } else { 0.0 };
        self.env_fog_height_falloff = fog.height_falloff;
        self.env_ambient_dust = fog.ambient_dust_density;
        self.env_dust_g = fog.dust_asymmetry;

        // Cloud settings → GPU CloudParams.
        let clouds = &env.clouds;
        self.env_cloud_settings.procedural_enabled = clouds.enabled;
        self.env_cloud_settings.cloud_min = clouds.altitude;
        self.env_cloud_settings.cloud_max = clouds.altitude + clouds.thickness;
        self.env_cloud_settings.cloud_density_scale = clouds.density;
        // Coverage → threshold: the FBM product (shape * weather * height_gradient)
        // is concentrated near 0 (three [0,1] values multiplied), so threshold
        // must drop quickly to reveal clouds. A power curve (coverage^0.35)
        // makes the slider response perceptually linear instead of bunching
        // all visible change into the last 20% of the range.
        self.env_cloud_settings.cloud_threshold =
            (1.0 - clouds.coverage.clamp(0.0, 1.0).powf(0.35)) * 0.4;
        self.env_cloud_settings.wind_direction = [clouds.wind_direction.x, clouds.wind_direction.z];
        self.env_cloud_settings.wind_speed = clouds.wind_speed;
        self.env_cloud_settings.shadow_enabled = clouds.enabled;

        let queue = &self.ctx.queue;
        let pp = &env.post_process;

        // Bloom.
        let t = pp.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, pp.bloom_intensity);

        // Tone mapping.
        let mode = if pp.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, pp.exposure);

        // Depth of field — max_coc=0 disables the blur.
        if pp.dof_enabled {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, pp.dof_max_coc);
        } else {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, 0.0);
        }

        // Sharpen.
        self.sharpen.set_strength(queue, pp.sharpen_strength);

        // Motion blur.
        self.motion_blur.set_intensity(queue, pp.motion_blur_intensity);

        // God rays blur.
        self.god_rays_blur.set_intensity(queue, pp.god_rays_intensity);

        // Cosmetics (vignette, grain, chromatic aberration).
        self.cosmetics.set_vignette(queue, pp.vignette_intensity);
        self.cosmetics.set_grain(queue, pp.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, pp.chromatic_aberration);

        env.clear_dirty();
    }

    /// Apply environment settings from a snapshot (no dirty flag management).
    ///
    /// Used by the engine thread after snapshotting EnvironmentState under the
    /// editor_state lock and releasing the lock before doing GPU work.
    pub fn apply_environment_snapshot(&mut self, env: &crate::environment::EnvironmentState) {
        let atmo = &env.atmosphere;
        self.env_sun_dir = [atmo.sun_direction.x, atmo.sun_direction.y, atmo.sun_direction.z];

        let sun_elevation = atmo.sun_direction.y.asin();
        let base_color = atmo.sun_color;
        let tinted_color = {
            let path = (1.0 / sun_elevation.max(0.02).sin()).min(12.0);
            let tau = glam::Vec3::new(0.02, 0.06, 0.15);
            let extinction = glam::Vec3::new(
                (-tau.x * path).exp(),
                (-tau.y * path).exp(),
                (-tau.z * path).exp(),
            );
            base_color * extinction
        };
        let sc = tinted_color * atmo.sun_intensity;
        self.env_sun_color = [sc.x, sc.y, sc.z];
        self.env_sun_intensity = atmo.sun_intensity;
        self.env_sun_color_raw = [tinted_color.x, tinted_color.y, tinted_color.z];
        self.env_rayleigh_scale = atmo.rayleigh_scale;
        self.env_mie_scale = atmo.mie_scale;
        self.env_atmosphere_enabled = atmo.enabled;

        let fog = &env.fog;
        self.env_fog_color = [fog.color.x, fog.color.y, fog.color.z];
        self.env_fog_density = if fog.enabled { fog.density } else { 0.0 };
        self.env_fog_height_falloff = fog.height_falloff;
        self.env_ambient_dust = fog.ambient_dust_density;
        self.env_dust_g = fog.dust_asymmetry;

        let clouds = &env.clouds;
        self.env_cloud_settings.procedural_enabled = clouds.enabled;
        self.env_cloud_settings.cloud_min = clouds.altitude;
        self.env_cloud_settings.cloud_max = clouds.altitude + clouds.thickness;
        self.env_cloud_settings.cloud_density_scale = clouds.density;
        self.env_cloud_settings.cloud_threshold =
            (1.0 - clouds.coverage.clamp(0.0, 1.0).powf(0.35)) * 0.4;
        self.env_cloud_settings.wind_direction = [clouds.wind_direction.x, clouds.wind_direction.z];
        self.env_cloud_settings.wind_speed = clouds.wind_speed;
        self.env_cloud_settings.shadow_enabled = clouds.enabled;

        let queue = &self.ctx.queue;
        let pp = &env.post_process;

        let t = pp.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, pp.bloom_intensity);

        let mode = if pp.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, pp.exposure);

        if pp.dof_enabled {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, pp.dof_max_coc);
        } else {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, 0.0);
        }

        self.sharpen.set_strength(queue, pp.sharpen_strength);
        self.motion_blur.set_intensity(queue, pp.motion_blur_intensity);
        self.god_rays_blur.set_intensity(queue, pp.god_rays_intensity);
        self.cosmetics.set_vignette(queue, pp.vignette_intensity);
        self.cosmetics.set_grain(queue, pp.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, pp.chromatic_aberration);
    }

    /// Camera-relative view-projection matrix for overlay rendering.
    pub fn view_projection(&self) -> glam::Mat4 {
        self.camera.view_projection(self.render_width, self.render_height)
    }

    /// Current camera position in world space.
    pub fn camera_position(&self) -> Vec3 {
        self.camera.position
    }

    /// Draw wireframe lines onto the offscreen render target.
    ///
    /// Called after `render_frame_offscreen()`. Uses a separate encoder + submit
    /// so the wireframe composites on top of the final post-processed image.
    pub fn draw_wireframe(&mut self, vertices: &[crate::wireframe::LineVertex]) {
        if vertices.is_empty() {
            return;
        }
        let Some(ref wireframe) = self.wireframe_pass else { return };
        let Some(ref offscreen_view) = self.offscreen_view else { return };

        let vp_matrix = self.view_projection();
        let viewport = (
            0.0,
            0.0,
            self.viewport_width as f32,
            self.viewport_height as f32,
        );

        let mut encoder =
            self.ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wireframe"),
                });
        wireframe.draw(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            offscreen_view,
            vp_matrix,
            viewport,
            vertices,
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Refresh the coarse acceleration field from current scene AABBs.
    ///
    /// If the scene still fits within the existing coarse field bounds, this
    /// only repopulates the CPU data and re-uploads the texture — no GPU
    /// pipeline or bind group rebuilds needed.  If the scene has grown beyond
    /// the current bounds, the field is reallocated and passes are rebuilt.
    pub fn refresh_coarse_field(&mut self, scene: &Scene) {
        let wts = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let def_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = wts.get(&obj.id).unwrap_or(&def_wt);
                let smin = obj.aabb.min * wt.scale;
                let smax = obj.aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z), Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z), Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z), Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z), Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                (wmin, wmax)
            })
            .collect();

        // Check if current coarse field covers the scene (with margin).
        let margin = 1.0_f32;
        let (scene_min, scene_max) = if scene_aabbs.is_empty() {
            (Vec3::ZERO, Vec3::splat(self.coarse_field.voxel_size))
        } else {
            let mut lo = Vec3::splat(f32::MAX);
            let mut hi = Vec3::splat(f32::MIN);
            for (amin, amax) in &scene_aabbs {
                lo = lo.min(*amin);
                hi = hi.max(*amax);
            }
            (lo, hi)
        };

        let needed_min = scene_min - Vec3::splat(margin);
        let needed_max = scene_max + Vec3::splat(margin);
        let current_max = self.coarse_field.origin + Vec3::new(
            self.coarse_field.dims.x as f32,
            self.coarse_field.dims.y as f32,
            self.coarse_field.dims.z as f32,
        ) * self.coarse_field.voxel_size;

        let fits = needed_min.cmpge(self.coarse_field.origin).all()
            && needed_max.cmple(current_max).all();

        if fits {
            // Fast path: repopulate + upload, same texture — bind groups stay valid.
            self.coarse_field.populate(&scene_aabbs);
            self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);
        } else {
            // Scene outgrew the field — reallocate texture and rebuild passes.
            self.coarse_field = CoarseField::from_scene_aabbs(
                &self.ctx.device, &scene_aabbs, COARSE_VOXEL_SIZE, margin,
            );
            self.coarse_field.populate(&scene_aabbs);
            self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

            self.ray_march = RayMarchPass::new(
                &self.ctx.device, &self.gpu_scene, &self.gbuffer,
                &self.tile_cull, &self.coarse_field,
            );
            self.radiance_inject = RadianceInjectPass::new(
                &self.ctx.device, &self.gpu_scene, &self.material_buffer,
                &self.light_buffer, &self.radiance_volume, &self.coarse_field,
            );
            self.shading_pass = ShadingPass::new(
                &self.ctx.device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
                &self.coarse_field, &self.radiance_volume, &self.material_buffer,
                self.render_width, self.render_height,
            );
            self.vol_shadow = VolShadowPass::new(
                &self.ctx.device, &self.ctx.queue, &self.coarse_field.bind_group_layout,
            );
            log::info!("Coarse field resized for {} objects", scene.objects.len());
        }
    }

    /// Configure the surface with COPY_SRC for screenshot readback.
    fn configure_surface_copy(
        ctx: &RenderContext,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let adapter = ctx.adapter.as_ref()
            .expect("configure_surface_copy requires an adapter");
        let caps = surface.get_capabilities(adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctx.device, &config);

        log::info!("Surface configured: {width}x{height}, format={format:?} (COPY_SRC enabled)");
        format
    }

    /// Create a readback buffer sized for the given window dimensions.
    fn create_readback_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Reconfigure the surface for a new window size (surface-based path only).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            if let Some(ref surface) = self.surface {
                Self::configure_surface_copy(&self.ctx, surface, width, height);
            }
            self.window_width = width;
            self.window_height = height;
            self.readback_buffer = Self::create_readback_buffer(
                &self.ctx.device, width, height,
            );
            // Update SharedState dimensions for MCP screenshot encoding.
            if let Ok(mut state) = self.shared_state.lock() {
                state.frame_width = width;
                state.frame_height = height;
            }
        }
    }

    /// Current render resolution width.
    pub fn render_width(&self) -> u32 {
        self.render_width
    }

    /// Current render resolution height.
    pub fn render_height(&self) -> u32 {
        self.render_height
    }

    /// Resize the internal render resolution.
    ///
    /// Recreates all size-dependent GPU resources: GBuffer, tile cull, shading,
    /// volumetric pipeline, post-processing, and blit pass. Called when the
    /// viewport dimensions change (window resize or panel layout change).
    pub fn resize_render(&mut self, width: u32, height: u32) {
        let width = width.max(64);
        let height = height.max(64);
        if width == self.render_width && height == self.render_height {
            return;
        }
        log::info!("Render resolution: {}x{} → {}x{}", self.render_width, self.render_height, width, height);
        self.render_width = width;
        self.render_height = height;

        let device = &self.ctx.device;

        // --- Core passes ---
        self.gbuffer = GBuffer::new(device, width, height);
        self.tile_cull = TileObjectCullPass::new(device, &self.gpu_scene, width, height);
        // RayMarchPass pipeline is resolution-agnostic (dispatch from gbuffer dims).
        // DebugViewPass stores bind groups to gbuffer.
        self.debug_view = DebugViewPass::new(device, &self.gbuffer);
        self.shading_pass = ShadingPass::new(
            device, &self.gbuffer, &self.gpu_scene, &self.light_buffer,
            &self.coarse_field, &self.radiance_volume, &self.material_buffer,
            width, height,
        );

        // --- Volumetric pipeline (half-res march, full-res upscale) ---
        // vol_shadow and cloud_shadow are resolution-independent.
        let half_w = width / 2;
        let half_h = height / 2;
        self.vol_march = VolMarchPass::new(
            device, &self.ctx.queue,
            &self.gbuffer.position_view, &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w, half_h, width, height,
        );
        self.vol_upscale = VolUpscalePass::new(
            device, &self.vol_march.output_view, &self.gbuffer.position_view,
            width, height, half_w, half_h,
        );
        self.vol_composite = VolCompositePass::new(
            device, &self.shading_pass.hdr_view, &self.vol_upscale.output_view,
            width, height,
        );

        // --- Post-processing pipeline ---
        self.god_rays_blur = GodRaysBlurPass::new(
            device, &self.vol_composite.output_view, &self.gbuffer.position_view,
            width, height,
        );
        self.bloom = BloomPass::new(
            device, &self.god_rays_blur.output_view, width, height,
        );
        self.auto_exposure = AutoExposurePass::new(
            device, &self.god_rays_blur.output_view, width, height,
        );
        self.dof = DofPass::new(
            device, &self.god_rays_blur.output_view, &self.gbuffer.position_view,
            width, height,
        );
        self.motion_blur = MotionBlurPass::new(
            device, &self.dof.output_view, &self.gbuffer.motion_view,
            width, height,
        );
        self.bloom_composite = BloomCompositePass::new(
            device, &self.motion_blur.output_view, self.bloom.mip_views(),
            width, height,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            device, &self.bloom_composite.output_view,
            width, height,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            device, &self.ctx.queue, &self.tone_map.ldr_view,
            width, height,
        );
        self.cosmetics = CosmeticsPass::new(
            device, &self.color_grade.output_view, width, height,
        );
        self.sharpen = SharpenPass::new(
            device, &self.vol_composite.output_view, &self.gbuffer,
            width, height,
        );

        // --- Blit (update source view to new cosmetics output) ---
        self.blit.update_source(device, &self.cosmetics.output_view);
    }

    /// Advance character animation on the given scene (mutates bone transforms).
    ///
    /// Call this on a cloned scene before rendering so animation doesn't
    /// pollute the authoritative scene in EditorState.
    pub fn advance_character(&mut self, scene: &mut Scene) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        if let (Some(character), Some(idx)) = (&mut self.character, self.character_obj_index) {
            if idx < scene.objects.len() {
                character.advance_and_update(dt, &mut scene.objects[idx].root_node);
            }
        }
    }

    /// Process a pending re-voxelize request.
    ///
    /// When a voxelized object has non-uniform scale, this resamples its brick
    /// data at the stretched dimensions and resets scale to (1,1,1). The new
    /// brick data is uploaded to the GPU.
    ///
    /// Returns `true` if a re-voxelize was performed.
    pub fn process_revoxelize(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        use rkf_core::brick::Brick;
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::sampling::{sample_brick_trilinear, sample_brick_nearest_material};

        // Find the object.
        let obj = match scene.objects.iter_mut().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return false,
        };

        // Must be voxelized with non-uniform scale.
        let (old_handle, voxel_size, old_aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (*brick_map_handle, *voxel_size, *aabb)
            }
            _ => return false,
        };

        let scale = obj.scale;
        if (scale - Vec3::ONE).length() < 1e-4 {
            return false; // Already uniform — nothing to do.
        }

        // 1. Copy old brick data into temporary storage.
        let old_dims = old_handle.dims;
        let mut old_bricks: std::collections::HashMap<(u32, u32, u32), Brick> =
            std::collections::HashMap::new();

        for bz in 0..old_dims.z {
            for by in 0..old_dims.y {
                for bx in 0..old_dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(
                        &old_handle, bx, by, bz,
                    ) {
                        if !Self::is_unallocated(slot) {
                            old_bricks.insert(
                                (bx, by, bz),
                                self.cpu_brick_pool.get(slot).clone(),
                            );
                        }
                    }
                }
            }
        }

        // 2. Deallocate old bricks from the pool.
        for bz in 0..old_dims.z {
            for by in 0..old_dims.y {
                for bx in 0..old_dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(
                        &old_handle, bx, by, bz,
                    ) {
                        if !Self::is_unallocated(slot) {
                            self.cpu_brick_pool.deallocate(slot);
                        }
                    }
                }
            }
        }
        self.cpu_brick_map_alloc.deallocate(old_handle);

        // 3. Build a sampling closure that reads from the old brick data.
        //    The closure receives a local-space position in the NEW (scaled) grid,
        //    maps it back to the old grid, and trilinearly interpolates.
        let brick_world_size = voxel_size * BRICK_DIM as f32;
        let old_grid_size = Vec3::new(
            old_dims.x as f32 * brick_world_size,
            old_dims.y as f32 * brick_world_size,
            old_dims.z as f32 * brick_world_size,
        );
        let old_grid_origin = -old_grid_size * 0.5;
        let min_scale = scale.x.min(scale.y.min(scale.z));

        let sample_fn = |new_pos: Vec3| -> (f32, u16) {
            // Map from new (scaled) space to old (unscaled) space.
            let old_pos = new_pos / scale;

            // Convert to grid-relative coordinates.
            let grid_rel = (old_pos - old_grid_origin) / brick_world_size;

            // Brick coordinates.
            let bx = grid_rel.x.floor() as i32;
            let by = grid_rel.y.floor() as i32;
            let bz = grid_rel.z.floor() as i32;

            // Check bounds.
            if bx < 0 || by < 0 || bz < 0
                || bx >= old_dims.x as i32
                || by >= old_dims.y as i32
                || bz >= old_dims.z as i32
            {
                return (f32::MAX, 0);
            }

            let bx = bx as u32;
            let by = by as u32;
            let bz = bz as u32;

            if let Some(brick) = old_bricks.get(&(bx, by, bz)) {
                // Local position within the brick (0..1).
                let local = Vec3::new(
                    grid_rel.x - bx as f32,
                    grid_rel.y - by as f32,
                    grid_rel.z - bz as f32,
                );
                let dist = sample_brick_trilinear(brick, local) * min_scale;
                let mat = sample_brick_nearest_material(brick, local);
                (dist, mat)
            } else {
                // Empty brick — far from surface.
                (f32::MAX, 0)
            }
        };

        // 4. Compute new AABB (scaled version of old AABB).
        let new_aabb = Aabb::new(old_aabb.min * scale, old_aabb.max * scale);

        // 5. Run voxelize_sdf with the sampling closure.
        let result = rkf_core::voxelize_sdf(
            sample_fn,
            &new_aabb,
            voxel_size,
            &mut self.cpu_brick_pool,
            &mut self.cpu_brick_map_alloc,
        );

        let (new_handle, _brick_count) = match result {
            Some(r) => r,
            None => {
                log::warn!("Re-voxelize failed: not enough brick pool slots");
                return false;
            }
        };

        // 6. Update the object. Use grid-aligned AABB from actual dims.
        let revox_brick_size = voxel_size * 8.0;
        let revox_grid_half = Vec3::new(
            new_handle.dims.x as f32 * revox_brick_size * 0.5,
            new_handle.dims.y as f32 * revox_brick_size * 0.5,
            new_handle.dims.z as f32 * revox_brick_size * 0.5,
        );
        let revox_grid_aabb = Aabb::new(-revox_grid_half, revox_grid_half);
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: revox_grid_aabb,
        };
        obj.aabb = revox_grid_aabb;
        obj.scale = Vec3::ONE;

        // 7. Re-upload brick pool and brick maps to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        self.ctx.queue.write_buffer(
            self.gpu_scene.brick_pool_buffer(), 0, pool_data,
        );
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }

        log::info!(
            "Re-voxelized object {} — scale reset to (1,1,1)",
            object_id,
        );
        true
    }

    /// Per-stroke SDF repair via flat-array FMM with restricted write-back.
    ///
    /// Called after every sculpt stroke to correct the distance magnitudes introduced
    /// by CSG in the brush region.  Correct magnitudes are critical because the
    /// Catmull-Rom normal kernel uses a 4×4×4 voxel neighbourhood — wrong magnitudes
    /// produce banding and incorrect normals.
    ///
    /// # Algorithm
    ///
    /// 1. Define outer box  = brush_center ± (inner_half + 1-brick margin) in voxel coords.
    /// 2. Load the outer box into flat f32 arrays (post-CSG for inner, original for margin).
    /// 3. Detect zero-crossings, seed FMM at |d|.min(h).
    /// 4. Run heap-based FMM on the flat array — O(N log N) with O(1) index arithmetic
    ///    (≈50–100× faster than the HashMap-based full-object FMM).
    /// 5. Determine sign: seed BFS from all margin voxels (valid sign, untouched by CSG),
    ///    flood into the inner box stopping at sign changes.
    /// 6. Write back ONLY the inner voxels.  The margin is read-only — no discontinuity
    ///    is created at the boundary, so the next stroke's zero-crossing detection sees
    ///    clean SDF values in the surrounding region.
    fn sculpt_fmm_repair(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        center_local: glam::Vec3,
        inner_half_voxels: i32,
    ) -> bool {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        use std::collections::VecDeque;
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;

        // 1. Look up object's voxel data.
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o, None => return false,
        };
        let (handle, voxel_size, aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } =>
                (brick_map_handle.clone(), *voxel_size, aabb.min),
            _ => return false,
        };
        let dims = handle.dims;
        let bd = BRICK_DIM as i32;
        let gw = dims.x as i32 * bd;
        let gh = dims.y as i32 * bd;
        let gd_ = dims.z as i32 * bd;
        let h = voxel_size;

        // 2. Define outer box (inner + 1-brick margin), clamped to grid.
        let margin_v = bd;
        let outer_half = inner_half_voxels + margin_v;

        let cx = ((center_local.x - aabb_min.x) / h).round() as i32;
        let cy = ((center_local.y - aabb_min.y) / h).round() as i32;
        let cz = ((center_local.z - aabb_min.z) / h).round() as i32;

        let ox0 = (cx - outer_half).clamp(0, gw - 1) as usize;
        let oy0 = (cy - outer_half).clamp(0, gh - 1) as usize;
        let oz0 = (cz - outer_half).clamp(0, gd_ - 1) as usize;
        let ox1 = (cx + outer_half).clamp(0, gw - 1) as usize;
        let oy1 = (cy + outer_half).clamp(0, gh - 1) as usize;
        let oz1 = (cz + outer_half).clamp(0, gd_ - 1) as usize;

        let fw  = ox1 - ox0 + 1;
        let fh  = oy1 - oy0 + 1;
        let _fd = oz1 - oz0 + 1;
        let fw_fh = fw * fh;
        let fn_count = fw * fh * (oz1 - oz0 + 1);
        if fn_count == 0 { return false; }

        // Flat index: (gx, gy, gz) → usize.  All coords in absolute voxel space.
        let fidx = |gx: usize, gy: usize, gz: usize| -> usize {
            (gz - oz0) * fw_fh + (gy - oy0) * fw + (gx - ox0)
        };
        // Is global voxel (gx,gy,gz) in the inner box?
        let is_inner = |gx: i32, gy: i32, gz: i32| -> bool {
            (gx - cx).abs() <= inner_half_voxels
                && (gy - cy).abs() <= inner_half_voxels
                && (gz - cz).abs() <= inner_half_voxels
        };

        // 3. Allocate flat buffers.
        let large_dist = h * gw.max(gh).max(gd_) as f32;
        let mut stored_d    = vec![large_dist; fn_count]; // post-CSG signed distance
        let mut is_alloc    = vec![false; fn_count];       // in brick pool?
        let mut fmm_d       = vec![f32::MAX; fn_count];   // unsigned FMM distance
        let mut fmm_settled = vec![false; fn_count];
        // For write-back: store (slot, brick_index) for allocated inner voxels.
        let mut write_slot = vec![u32::MAX; fn_count];
        let mut write_vi   = vec![0u32; fn_count];

        // 4. Load from brick pool.
        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let bd_us = BRICK_DIM as usize;
                    let bx = (gx / bd_us) as u32;
                    let by = (gy / bd_us) as u32;
                    let bz = (gz / bd_us) as u32;
                    let lx = (gx % bd_us) as u32;
                    let ly = (gy % bd_us) as u32;
                    let lz = (gz % bd_us) as u32;
                    let fi = fidx(gx, gy, gz);
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let vi = brick_index(lx, ly, lz);
                            let d = self.cpu_brick_pool.get(slot).voxels[vi].distance_f32();
                            stored_d[fi] = d;
                            is_alloc[fi] = true;
                            // Write back all allocated outer-box voxels so the
                            // Catmull-Rom normal kernel never crosses into
                            // un-repaired h*8.0 margin voxels.
                            write_slot[fi] = slot;
                            write_vi[fi]   = vi as u32;
                        }
                    }
                }
            }
        }

        // 5. Detect zero-crossings, seed FMM.
        let face_dx: [i32; 6] = [-1, 1,  0, 0,  0, 0];
        let face_dy: [i32; 6] = [ 0, 0, -1, 1,  0, 0];
        let face_dz: [i32; 6] = [ 0, 0,  0, 0, -1, 1];

        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let fi = fidx(gx, gy, gz);
                    if !is_alloc[fi] { continue; }
                    let d = stored_d[fi];
                    if !d.is_finite() { continue; }
                    for dir in 0..6usize {
                        let nx = gx as i32 + face_dx[dir];
                        let ny = gy as i32 + face_dy[dir];
                        let nz = gz as i32 + face_dz[dir];
                        if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                        if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                        let nfi = fidx(nx as usize, ny as usize, nz as usize);
                        if !is_alloc[nfi] { continue; }
                        let nd = stored_d[nfi];
                        if !nd.is_finite() || d * nd >= 0.0 { continue; }
                        let sd  = d.abs().min(h);
                        let snd = nd.abs().min(h);
                        if sd  < fmm_d[fi]  { fmm_d[fi]  = sd;  }
                        if snd < fmm_d[nfi] { fmm_d[nfi] = snd; }
                    }
                }
            }
        }

        // 6. Heap-based FMM on flat array — O(1) neighbour lookup via index arithmetic.
        let eikonal = |a: f32, b: f32, c: f32| -> f32 {
            if a == f32::MAX { return f32::MAX; }
            let u1 = a + h;
            if b == f32::MAX || u1 <= b { return u1; }
            let disc2 = 2.0 * h * h - (b - a) * (b - a);
            let u2 = if disc2 >= 0.0 { (a + b + disc2.sqrt()) * 0.5 } else { a + h };
            if c == f32::MAX || u2 <= c { return u2; }
            let s = a + b + c;
            let disc3 = s * s - 3.0 * (a * a + b * b + c * c - h * h);
            if disc3 >= 0.0 { (s + disc3.sqrt()) / 3.0 } else { u2 }
        };

        let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        for fi in 0..fn_count {
            if fmm_d[fi] < f32::MAX {
                heap.push(Reverse((fmm_d[fi].to_bits(), fi as u32)));
            }
        }

        while let Some(Reverse((bits, fi_u32))) = heap.pop() {
            let fi = fi_u32 as usize;
            if fmm_settled[fi] { continue; }
            let dist = f32::from_bits(bits);
            if dist > fmm_d[fi] + h * 0.01 { continue; }
            fmm_settled[fi] = true;

            // Recover absolute coords from flat index.
            let gx = ox0 + (fi % fw);
            let gy = oy0 + (fi / fw) % fh;
            let gz = oz0 + (fi / fw_fh);

            for dir in 0..6usize {
                let nx = gx as i32 + face_dx[dir];
                let ny = gy as i32 + face_dy[dir];
                let nz = gz as i32 + face_dz[dir];
                if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                let nfi = fidx(nx as usize, ny as usize, nz as usize);
                if !is_alloc[nfi] || fmm_settled[nfi] { continue; }

                let nfx = nx as usize; let nfy = ny as usize; let nfz = nz as usize;
                // Per-axis minimum settled distance for Eikonal update.
                let mut da = [f32::MAX; 3];
                macro_rules! check_nbr {
                    ($i:expr, $ax:expr, $ay:expr, $az:expr) => {
                        let ax = nfx as i32 + $ax; let ay = nfy as i32 + $ay; let az = nfz as i32 + $az;
                        if ax >= ox0 as i32 && ay >= oy0 as i32 && az >= oz0 as i32
                            && ax <= ox1 as i32 && ay <= oy1 as i32 && az <= oz1 as i32 {
                            let afi = fidx(ax as usize, ay as usize, az as usize);
                            if fmm_settled[afi] { da[$i] = da[$i].min(fmm_d[afi]); }
                        }
                    }
                }
                check_nbr!(0, -1, 0, 0); check_nbr!(0, 1, 0, 0);
                check_nbr!(1, 0, -1, 0); check_nbr!(1, 0, 1, 0);
                check_nbr!(2, 0, 0, -1); check_nbr!(2, 0, 0, 1);
                da.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let new_d = eikonal(da[0], da[1], da[2]);
                if new_d < fmm_d[nfi] {
                    fmm_d[nfi] = new_d;
                    heap.push(Reverse((new_d.to_bits(), nfi as u32)));
                }
            }
        }
        // Unsettled allocated voxels: fall back to large_dist.
        for fi in 0..fn_count {
            if is_alloc[fi] && !fmm_settled[fi] { fmm_d[fi] = large_dist; }
        }

        // 7. Sign BFS: seed from margin voxels (valid sign, untouched by CSG),
        //    flood into the inner box stopping at zero-crossings.
        //    Margin sign: stored_d > 0 = exterior (valid because CSG didn't touch margin).
        //    Unallocated voxels = exterior (empty space).
        let mut ext_sign = vec![0i8; fn_count]; // 1=ext, -1=int, 0=unknown
        let mut bfs: VecDeque<usize> = VecDeque::new();

        for gz in oz0..=oz1 {
            for gy in oy0..=oy1 {
                for gx in ox0..=ox1 {
                    let fi = fidx(gx, gy, gz);
                    let in_inner = is_inner(gx as i32, gy as i32, gz as i32);
                    if !in_inner {
                        // Margin: seed with valid sign.
                        let sign = if !is_alloc[fi] || stored_d[fi] > 0.0 { 1i8 } else { -1i8 };
                        ext_sign[fi] = sign;
                        bfs.push_back(fi);
                    }
                }
            }
        }

        while let Some(fi) = bfs.pop_front() {
            let sign = ext_sign[fi];
            let d_here = stored_d[fi];
            let gx = ox0 + (fi % fw);
            let gy = oy0 + (fi / fw) % fh;
            let gz = oz0 + (fi / fw_fh);
            for dir in 0..6usize {
                let nx = gx as i32 + face_dx[dir];
                let ny = gy as i32 + face_dy[dir];
                let nz = gz as i32 + face_dz[dir];
                if nx < ox0 as i32 || ny < oy0 as i32 || nz < oz0 as i32 { continue; }
                if nx > ox1 as i32 || ny > oy1 as i32 || nz > oz1 as i32 { continue; }
                let nfi = fidx(nx as usize, ny as usize, nz as usize);
                if ext_sign[nfi] != 0 { continue; }
                if !is_alloc[nfi] { ext_sign[nfi] = 1; bfs.push_back(nfi); continue; }
                let d_nc = stored_d[nfi];
                if d_here.is_finite() && d_nc.is_finite() && d_here * d_nc < 0.0 { continue; }
                ext_sign[nfi] = sign;
                bfs.push_back(nfi);
            }
        }
        // Fallback for inner voxels BFS didn't reach: use post-CSG sign.
        for fi in 0..fn_count {
            if ext_sign[fi] == 0 && is_alloc[fi] {
                ext_sign[fi] = if stored_d[fi] > 0.0 { 1 } else { -1 };
            }
        }

        // 8. Write back: ALL allocated outer-box voxels (inner + margin).
        //    Writing the full outer box ensures the Catmull-Rom normal kernel
        //    (±1, ±2 voxels) never crosses into un-repaired h*8.0 margin
        //    voxels and picks up a gradient discontinuity → rough normals.
        //    FMM is seeded from boolean-stamp zero-crossings so distances are
        //    exact Euclidean — safe to write at any radius.
        for fi in 0..fn_count {
            let slot = write_slot[fi];
            if slot == u32::MAX { continue; } // not inner or not allocated
            let vi = write_vi[fi] as usize;
            let orig = self.cpu_brick_pool.get(slot).voxels[vi];
            let sign = ext_sign[fi] as f32;
            let mag = fmm_d[fi]; // FMM Euclidean distance for both signs
            self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                sign * mag,
                orig.material_id(),
                orig.blend_weight(),
                orig.secondary_id(),
                orig.flags(),
            );
        }

        true
    }

    /// Boolean-stamp a sculpt brush into the brick pool.
    ///
    /// For Add (SmoothUnion): grows the brick map if needed, allocates bricks
    /// in the brush region (filled as large exterior), then stamps every voxel
    /// inside the effective brush as solid (`-(h*0.5)`).
    ///
    /// For Subtract (SmoothSubtract): stamps every voxel inside the brush as
    /// exterior (`+(h*0.5)`). No new bricks are allocated — carving empty space
    /// has no visible effect.
    ///
    /// Strength is applied via `effective_shape_d = shape_d + (1−strength) × max_extent`:
    /// at strength=1, the full brush is stamped; at strength=0, nothing changes.
    ///
    /// Newly-allocated bricks that end up with no interior voxels are immediately
    /// reverted to EMPTY_SLOT (the brush didn't reach them).
    ///
    /// The correct Euclidean SDF distances are recomputed by `fix_sdfs_cpu`
    /// (3D EDT) after each stroke, using the binary solid/empty field set here.
    ///
    /// Returns the pool slot indices of all bricks that were modified.
    fn apply_sculpt_boolean(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
        voxel_size: f32,
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> Vec<u32> {
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick::brick_index;
        use rkf_edit::cpu_apply::evaluate_shape;
        use rkf_edit::types::EditType;

        let is_add = op.edit_type == EditType::SmoothUnion;
        let is_sub = op.edit_type == EditType::SmoothSubtract;
        if !is_add && !is_sub { return Vec::new(); }

        let h = voxel_size;
        let max_extent = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let inv_rot = op.rotation.inverse();

        // For add, grow the brick map if the brush extends beyond the current grid.
        if is_add {
            self.grow_brick_map_if_needed(scene, object_id, op);
        }

        // Re-lookup handle + aabb_min after potential grow.
        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o, None => return Vec::new(),
        };
        let (handle, aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, aabb, .. } => (*brick_map_handle, aabb.min),
            _ => return Vec::new(),
        };

        let brick_size = h * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let bmin = ((edit_min - aabb_min) / brick_size).floor();
        let bmax = ((edit_max - aabb_min) / brick_size - Vec3::splat(0.001)).ceil();
        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

        let mut new_slots: Vec<u32> = Vec::new(); // newly allocated this call
        let mut modified: Vec<u32> = Vec::new();  // all touched slots

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    let slot = {
                        let existing = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz);
                        match existing {
                            Some(s) if !Self::is_unallocated(s) => s,
                            _ => {
                                if is_sub {
                                    continue; // Don't allocate new bricks for subtract.
                                }
                                // Allocate a new brick for add, filled as large exterior.
                                // FMM will compute real distances; `h * 8.0` is just a
                                // safe step value so the ray marcher doesn't stall.
                                match self.cpu_brick_pool.allocate() {
                                    Some(ns) => {
                                        let brick = self.cpu_brick_pool.get_mut(ns);
                                        for lz in 0u32..8 { for ly in 0u32..8 { for lx in 0u32..8 {
                                            brick.set(lx, ly, lz, VoxelSample::new(h * 8.0, 0, 0, 0, 0));
                                        }}}
                                        self.cpu_brick_map_alloc.set_entry(&handle, bx, by, bz, ns);
                                        new_slots.push(ns);
                                        ns
                                    }
                                    None => continue, // Pool full.
                                }
                            }
                        }
                    };

                    // Snapshot for undo BEFORE modifying this brick.
                    if let Some(ref mut acc) = undo_acc {
                        if acc.captured_slots.insert(slot) {
                            acc.snapshots.push((slot, self.cpu_brick_pool.get(slot).clone()));
                        }
                    }

                    // Boolean stamp: mark voxels inside the effective brush.
                    let brick_min = aabb_min + Vec3::new(
                        bx as f32 * brick_size,
                        by as f32 * brick_size,
                        bz as f32 * brick_size,
                    );
                    let mut any_changed = false;

                    for lz in 0u32..8 {
                        for ly in 0u32..8 {
                            for lx in 0u32..8 {
                                // Voxel center in object-local space.
                                let voxel_center = brick_min
                                    + Vec3::new(lx as f32, ly as f32, lz as f32) * h
                                    + Vec3::splat(h * 0.5);
                                let edit_local = inv_rot * (voxel_center - op.position);
                                let shape_d = evaluate_shape(op.shape_type, &op.dimensions, edit_local);
                                // Strength shrinks the effective brush: at strength=1,
                                // effective_d = shape_d (full brush). At strength=0,
                                // effective_d is always positive (no change).
                                let effective_d = shape_d + (1.0 - op.strength) * max_extent;
                                // Smooth-min blend: creates smooth stroke junctions so
                                // the geometry between overlapping strokes is smooth.
                                // EDT recomputes distances from the resulting binary
                                // solid/empty field (sign of stored distance).
                                if effective_d >= op.blend_k { continue; }

                                let vi = brick_index(lx, ly, lz);
                                let orig = self.cpu_brick_pool.get(slot).voxels[vi];

                                let (new_d, new_mat) = if is_add {
                                    let d = rkf_core::sdf::smin(
                                        orig.distance_f32(), effective_d, op.blend_k,
                                    );
                                    if d >= orig.distance_f32() { continue; }
                                    (d, op.material_id)
                                } else {
                                    // Smooth subtract: smooth max of orig and the
                                    // complement of the brush (-effective_d).
                                    // smax(a,b) = -smin(-a,-b), so
                                    // smax(orig,-effective_d) = -smin(-orig, effective_d).
                                    let d = -rkf_core::sdf::smin(
                                        -orig.distance_f32(), effective_d, op.blend_k,
                                    );
                                    if d <= orig.distance_f32() { continue; }
                                    (d, orig.material_id())
                                };
                                self.cpu_brick_pool.get_mut(slot).voxels[vi] = VoxelSample::new(
                                    new_d, new_mat,
                                    orig.blend_weight(), orig.secondary_id(), orig.flags(),
                                );
                                any_changed = true;
                            }
                        }
                    }

                    if any_changed {
                        modified.push(slot);
                    }
                }
            }
        }

        // Revert newly-allocated bricks that contain no interior voxels.
        // These are bricks where the boolean stamp wrote no solid voxels —
        // the brush AABB covered the brick but the brush shape didn't reach it.
        if !new_slots.is_empty() {
            let new_set: std::collections::HashSet<u32> = new_slots.iter().copied().collect();
            let mut slot_to_coord: std::collections::HashMap<u32, (u32, u32, u32)> =
                std::collections::HashMap::new();
            for bz in bmin_z..=bmax_z {
                for by in bmin_y..=bmax_y {
                    for bx in bmin_x..=bmax_x {
                        if let Some(s) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            if new_set.contains(&s) {
                                slot_to_coord.insert(s, (bx, by, bz));
                            }
                        }
                    }
                }
            }
            let mut reverted = 0u32;
            for &slot in &new_slots {
                let has_interior = self.cpu_brick_pool.get(slot)
                    .voxels.iter()
                    .any(|v| v.distance_f32() < 0.0);
                if !has_interior {
                    if let Some(&(bx, by, bz)) = slot_to_coord.get(&slot) {
                        self.cpu_brick_map_alloc.set_entry(
                            &handle, bx, by, bz, rkf_core::brick_map::EMPTY_SLOT,
                        );
                        self.cpu_brick_pool.deallocate(slot);
                        modified.retain(|&s| s != slot);
                        reverted += 1;
                    }
                }
            }
            if reverted > 0 {
                log::info!("  apply_sculpt_boolean: reverted {} surfaceless bricks", reverted);
            }
        }

        log::info!(
            "  apply_sculpt_boolean: {} modified slots (is_add={})",
            modified.len(), is_add,
        );
        modified
    }

    /// Recompute correct SDF magnitudes from zero-crossings via BFS (Dijkstra).
    ///
    /// After iterative sculpting, the CSG formula leaves surrounding voxels with
    /// incorrect distance magnitudes — each stroke compounds the error. This method
    /// repairs the field without changing geometry:
    ///
    /// 1. Collect all allocated voxels.
    /// 2. Seed a min-heap with voxels adjacent to sign changes (|original dist|).
    /// 3. Dijkstra: propagate distance + voxel_size to 6-face neighbors.
    /// 4. Sign flood-fill from boundary voxels (always exterior) to determine
    ///    interior vs exterior; propagate through positive-sign voxels, stop at
    ///    sign changes.
    /// 5. Write back: ±bfs_distance, preserving material_id/blend_weight/flags.
    /// 6. Re-run mark_interior_empties and full GPU reupload.
    ///
    /// Returns `true` if a fix was performed.
    pub fn process_fix_sdfs(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        // Get handle before running fix.
        let handle = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => match &o.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, .. } => *brick_map_handle,
                _ => return false,
            },
            None => return false,
        };

        // Clear stale INTERIOR_SLOT markers before repair.
        //
        // mark_interior_empties is safe after initial voxelization (single connected
        // body), but NOT during/after sculpting. After sculpting, the brush expands
        // the brick map and allocates peripheral bricks (filled with vs*2.0). These
        // peripheral bricks form "walls" that enclose empty regions, causing
        // mark_interior_empties to incorrectly classify them as INTERIOR_SLOT.
        // INTERIOR_SLOT bricks return -(vs*2.0) from the shader — a negative SDF
        // that registers as a hit, producing false shadows across the entire AABB.
        //
        // Solution: clear all INTERIOR_SLOT → EMPTY_SLOT, run EDT repair, and do NOT
        // re-mark interior bricks. The correct SDF magnitudes from fix_sdfs_cpu make
        // EMPTY_SLOT bricks harmless (the shader returns MAX_FLOAT, skipped quickly).
        {
            use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
            let dims = handle.dims;
            for bz in 0..dims.z {
                for by in 0..dims.y {
                    for bx in 0..dims.x {
                        if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            if slot == INTERIOR_SLOT {
                                self.cpu_brick_map_alloc.set_entry(&handle, bx, by, bz, EMPTY_SLOT);
                            }
                        }
                    }
                }
            }
        }

        if !self.run_jfa_repair(scene, object_id) {
            return false;
        }

        // Do NOT call mark_interior_empties here — see comment above.
        self.reupload_brick_data();
        true
    }

    /// Recompute correct SDF distances for `object_id` using GPU JFA.
    ///
    /// 1. Builds a dense solid/empty grid from the CPU brick pool.
    /// 2. Runs GPU JFA (init → log2(N) passes → writeback) to get Euclidean distances.
    /// 3. Reads back the result and writes distances into `cpu_brick_pool`
    ///    (preserving material_id, blend_weight, flags).
    ///
    /// Does NOT call `reupload_brick_data` — the caller is responsible.
    /// Returns `false` if the object is not found or has no voxelized data.
    fn run_jfa_repair(&mut self, scene: &mut Scene, object_id: u32) -> bool {
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick_map::INTERIOR_SLOT;

        let (handle, voxel_size) = {
            let obj = match scene.objects.iter().find(|o| o.id == object_id) {
                Some(o) => o,
                None => return false,
            };
            match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                    (*brick_map_handle, *voxel_size)
                }
                _ => return false,
            }
        };

        let dims = handle.dims;
        let bd = BRICK_DIM as u32;
        let gw = (dims.x * bd) as usize;
        let gh = (dims.y * bd) as usize;
        let gd = (dims.z * bd) as usize;
        let total = gw * gh * gd;
        if total == 0 { return false; }

        let idx3 = |x: usize, y: usize, z: usize| -> usize { z * gh * gw + y * gw + x };

        // Build dense solid grid + per-voxel metadata for allocated bricks.
        let mut solid = vec![false; total];
        let mut is_allocated = vec![false; total];
        let mut mat_grid: Vec<u16>  = vec![0; total];
        let mut blend_grid: Vec<u8> = vec![0; total];
        let mut sec_id_grid: Vec<u8> = vec![0; total];
        let mut flags_grid: Vec<u8>  = vec![0; total];

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if s == INTERIOR_SLOT => {
                            // Treat entire INTERIOR_SLOT brick as solid.
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                solid[idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                )] = true;
                            }}}
                        }
                        Some(s) if !Self::is_unallocated(s) => {
                            let brick = self.cpu_brick_pool.get(s);
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                let vi = brick_index(lx, ly, lz);
                                let vs = brick.voxels[vi];
                                let i = idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                );
                                solid[i]      = vs.distance_f32() < 0.0;
                                is_allocated[i] = true;
                                mat_grid[i]    = vs.material_id();
                                blend_grid[i]  = vs.blend_weight();
                                sec_id_grid[i] = vs.secondary_id();
                                flags_grid[i]  = vs.flags();
                            }}}
                        }
                        _ => {} // EMPTY_SLOT or None → stays empty/false
                    }
                }
            }
        }

        // Run GPU JFA.
        let distances = match self.jfa_sdf.repair(
            &self.ctx.device, &self.ctx.queue,
            &solid, gw, gh, gd, voxel_size,
        ) {
            Some(d) => d,
            None => return false,
        };

        // Write corrected distances back to allocated bricks only.
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };
                    let brick = self.cpu_brick_pool.get_mut(slot);
                    for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                        let i = idx3(
                            (bx*bd+lx) as usize,
                            (by*bd+ly) as usize,
                            (bz*bd+lz) as usize,
                        );
                        if !is_allocated[i] { continue; }
                        let vi = brick_index(lx, ly, lz);
                        brick.voxels[vi] = VoxelSample::new(
                            distances[i],
                            mat_grid[i],
                            blend_grid[i],
                            sec_id_grid[i],
                            flags_grid[i],
                        );
                    }}}
                }
            }
        }

        log::info!(
            "run_jfa_repair: repaired SDF for object {} ({}×{}×{} voxels)",
            object_id, gw, gh, gd,
        );
        true
    }

    /// Recompute correct SDF distances via 3D Euclidean Distance Transform (EDT).
    /// CPU-only — does NOT upload to GPU.
    ///
    /// Algorithm:
    ///   1. Build a dense 3D solid/empty boolean grid from ALL brick data.
    ///      INTERIOR_SLOT bricks → solid. EMPTY_SLOT → empty. Allocated bricks → sign of stored value.
    ///   2. Run separable 3D EDT (Felzenszwalb algorithm) twice: once from solid seeds,
    ///      once from empty seeds. Gives exact Euclidean distance to nearest surface boundary.
    ///   3. Sign by flood-fill from exterior boundary.
    ///   4. Write SDF = sign * (sqrt(edt) - h/2) back to allocated bricks only.
    ///
    /// This avoids ALL the FMM problems: no seed quality issues, no INTERIOR_SLOT propagation
    /// gaps, no scope seams. The EDT always produces globally correct Euclidean distances.
    fn fix_sdfs_cpu(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        _scope: Option<(glam::Vec3, f32)>,
    ) -> bool {
        use rkf_core::constants::BRICK_DIM;
        use rkf_core::brick::brick_index;
        use rkf_core::voxel::VoxelSample;
        use rkf_core::brick_map::INTERIOR_SLOT;
        use std::collections::VecDeque;

        // 1. Get handle + voxel_size from object.
        let (handle, voxel_size) = {
            let obj = match scene.objects.iter().find(|o| o.id == object_id) {
                Some(o) => o,
                None => return false,
            };
            match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                    (*brick_map_handle, *voxel_size)
                }
                _ => return false,
            }
        };

        let dims = handle.dims;
        let bd = BRICK_DIM as u32;
        let gw = (dims.x * bd) as usize;
        let gh = (dims.y * bd) as usize;
        let gd = (dims.z * bd) as usize;
        let total = gw * gh * gd;
        if total == 0 { return false; }
        let h = voxel_size;

        let idx3 = |x: usize, y: usize, z: usize| -> usize { z * gh * gw + y * gw + x };

        // 2. Build dense solid/empty grid + per-voxel metadata for allocated bricks.
        //
        //    INTERIOR_SLOT bricks → all voxels treated as solid.
        //    EMPTY_SLOT/None bricks → all voxels empty.
        //    Allocated bricks → solid if stored distance < 0.
        //    is_allocated[] tracks which voxels have real data to write back.
        let mut solid = vec![false; total];
        let mut is_allocated = vec![false; total];
        let mut mat_grid: Vec<u16> = vec![0u16; total];
        let mut blend_grid: Vec<u8> = vec![0u8; total];
        let mut sec_id_grid: Vec<u8> = vec![0u8; total];
        let mut flags_grid: Vec<u8> = vec![0u8; total];

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot_opt = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz);
                    match slot_opt {
                        Some(s) if s == INTERIOR_SLOT => {
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                solid[idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                )] = true;
                            }}}
                        }
                        Some(s) if !Self::is_unallocated(s) => {
                            let brick = self.cpu_brick_pool.get(s);
                            for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                                let vi = brick_index(lx, ly, lz);
                                let vs = brick.voxels[vi];
                                let i = idx3(
                                    (bx*bd+lx) as usize,
                                    (by*bd+ly) as usize,
                                    (bz*bd+lz) as usize,
                                );
                                solid[i] = vs.distance_f32() < 0.0;
                                is_allocated[i] = true;
                                mat_grid[i] = vs.material_id();
                                blend_grid[i] = vs.blend_weight();
                                sec_id_grid[i] = vs.secondary_id();
                                flags_grid[i] = vs.flags();
                            }}}
                        }
                        _ => {} // EMPTY_SLOT or None — stays empty (false)
                    }
                }
            }
        }

        // 3. Run 3D EDT in both directions for exact Euclidean distances.
        //    edt_to_solid: squared dist from each voxel to nearest solid voxel.
        //    edt_to_empty: squared dist from each voxel to nearest empty voxel.
        let edt_to_solid = Self::edt_3d_squared(&solid, gw, gh, gd, h);
        let not_solid: Vec<bool> = solid.iter().map(|&s| !s).collect();
        let edt_to_empty = Self::edt_3d_squared(&not_solid, gw, gh, gd, h);

        // 4. Sign determination via BFS flood-fill from grid boundary.
        //    Empty voxels connected to the boundary are exterior (+).
        //    Everything else is interior (−): solid voxels or enclosed empty voids.
        let mut exterior = vec![false; total];
        let mut queue: VecDeque<(usize, usize, usize)> = VecDeque::new();

        for z in 0..gd {
            for y in 0..gh {
                for x in 0..gw {
                    let on_boundary = x == 0 || x == gw - 1
                        || y == 0 || y == gh - 1
                        || z == 0 || z == gd - 1;
                    if on_boundary {
                        let i = idx3(x, y, z);
                        if !solid[i] && !exterior[i] {
                            exterior[i] = true;
                            queue.push_back((x, y, z));
                        }
                    }
                }
            }
        }

        while let Some((x, y, z)) = queue.pop_front() {
            let nbrs: [(usize, usize, usize); 6] = [
                (x.wrapping_sub(1), y, z), (x + 1, y, z),
                (x, y.wrapping_sub(1), z), (x, y + 1, z),
                (x, y, z.wrapping_sub(1)), (x, y, z + 1),
            ];
            for (nx, ny, nz) in nbrs {
                if nx >= gw || ny >= gh || nz >= gd { continue; }
                let ni = idx3(nx, ny, nz);
                if !solid[ni] && !exterior[ni] {
                    exterior[ni] = true;
                    queue.push_back((nx, ny, nz));
                }
            }
        }

        // 5. Write back: sign * (sqrt(edt) − h/2) to allocated bricks only.
        //
        //    The zero-crossing is at the voxel boundary (midpoint between adjacent
        //    solid and empty voxel centers), so we subtract h/2 from the Euclidean
        //    distance between centers to get the distance to the surface.
        //    Clamp magnitude to at least h*0.01 to avoid numerical zero.
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => s,
                        _ => continue,
                    };
                    let brick = self.cpu_brick_pool.get_mut(slot);
                    for lz in 0..bd { for ly in 0..bd { for lx in 0..bd {
                        let i = idx3(
                            (bx*bd+lx) as usize,
                            (by*bd+ly) as usize,
                            (bz*bd+lz) as usize,
                        );
                        if !is_allocated[i] { continue; }
                        let is_ext = exterior[i];
                        let edt_sq = if is_ext { edt_to_solid[i] } else { edt_to_empty[i] };
                        let mag = (edt_sq.sqrt() - h * 0.5).max(h * 0.01);
                        let signed_dist = if is_ext { mag } else { -mag };
                        let vi = brick_index(lx, ly, lz);
                        brick.voxels[vi] = VoxelSample::new(
                            signed_dist,
                            mat_grid[i],
                            blend_grid[i],
                            sec_id_grid[i],
                            flags_grid[i],
                        );
                    }}}
                }
            }
        }

        log::info!("fix_sdfs_cpu: EDT repaired SDF for object {} ({}×{}×{} voxels)",
                   object_id, gw, gh, gd);
        true
    }

    /// Felzenszwalb 1D EDT (lower envelope of parabolas).
    ///
    /// `col[i]` = 0.0 at seed positions, f32::MAX otherwise (first pass), or
    /// the accumulated squared distance from previous passes (second/third pass).
    /// `h` = voxel spacing.
    /// Writes squared Euclidean distances to nearest seed into `out`.
    fn edt_1d_pass(col: &[f32], h: f32, out: &mut [f32]) {
        let n = col.len();
        for x in out.iter_mut() { *x = f32::MAX; }
        if n == 0 { return; }

        // v[k] = index of k-th parabola center.
        // z[k] = left boundary of k-th parabola's valid domain; z[v.len()] = +inf.
        // Invariant: z.len() == v.len() + 1.
        let mut v: Vec<usize> = Vec::with_capacity(n);
        let mut z: Vec<f32> = Vec::with_capacity(n + 1);

        for q in 0..n {
            if col[q] >= f32::MAX { continue; }
            let qf = q as f32 * h;

            loop {
                if v.is_empty() {
                    v.push(q);
                    z.push(f32::NEG_INFINITY);
                    z.push(f32::INFINITY);
                    break;
                }
                let r = *v.last().unwrap();
                let rf = r as f32 * h;
                // Intersection of parabola at q and at r:
                // (x−qf)² + col[q] = (x−rf)² + col[r]  →  solve for x.
                let s = ((col[q] + qf * qf) - (col[r] + rf * rf)) / (2.0 * (qf - rf));
                // z[z.len()-2] is the left boundary of parabola r's domain.
                if s <= z[z.len() - 2] {
                    // Parabola r is completely dominated by q — remove it.
                    v.pop();
                    z.pop(); // remove +inf
                    z.pop(); // remove left boundary of r
                    z.push(f32::INFINITY); // restore right boundary
                } else {
                    // Parabola q starts at s.
                    *z.last_mut().unwrap() = s;
                    v.push(q);
                    z.push(f32::INFINITY);
                    break;
                }
            }
        }

        if v.is_empty() { return; }

        let mut j = 0usize;
        for q in 0..n {
            let qf = q as f32 * h;
            while j + 1 < v.len() && z[j + 1] <= qf { j += 1; }
            let r = v[j];
            if col[r] < f32::MAX {
                let diff = qf - r as f32 * h;
                out[q] = diff * diff + col[r];
            }
        }
    }

    /// Separable 3D EDT via three 1D Felzenszwalb passes (X→Y→Z).
    ///
    /// `seeds[i]` = true if voxel i is a seed (source).
    /// Returns squared Euclidean distances to nearest seed, in world units.
    fn edt_3d_squared(seeds: &[bool], gw: usize, gh: usize, gd: usize, h: f32) -> Vec<f32> {
        let idx3 = |x: usize, y: usize, z: usize| z * gh * gw + y * gw + x;
        let total = gw * gh * gd;

        // Initialize: 0 at seeds, MAX elsewhere.
        let mut grid: Vec<f32> = (0..total).map(|i| if seeds[i] { 0.0 } else { f32::MAX }).collect();

        let max_dim = gw.max(gh).max(gd);
        let mut col = vec![0.0f32; max_dim];
        let mut tmp = vec![0.0f32; max_dim];

        // Pass 1: X direction
        for z in 0..gd {
            for y in 0..gh {
                for x in 0..gw { col[x] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gw], h, &mut tmp[..gw]);
                for x in 0..gw { grid[idx3(x, y, z)] = tmp[x]; }
            }
        }

        // Pass 2: Y direction
        for z in 0..gd {
            for x in 0..gw {
                for y in 0..gh { col[y] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gh], h, &mut tmp[..gh]);
                for y in 0..gh { grid[idx3(x, y, z)] = tmp[y]; }
            }
        }

        // Pass 3: Z direction
        for y in 0..gh {
            for x in 0..gw {
                for z in 0..gd { col[z] = grid[idx3(x, y, z)]; }
                Self::edt_1d_pass(&col[..gd], h, &mut tmp[..gd]);
                for z in 0..gd { grid[idx3(x, y, z)] = tmp[z]; }
            }
        }

        grid
    }

    /// Auto-voxelize an analytical object for sculpting.
    ///
    /// If the object's root node is `SdfSource::Analytical`, converts it to
    /// `SdfSource::Voxelized` with a resolution based on the primitive's size
    /// (diameter / 48, clamped to [0.005, 0.5]).
    ///
    /// Returns the new `(voxel_size, aabb, handle)` if voxelization occurred.
    pub fn ensure_object_voxelized(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
    ) -> Option<(f32, Aabb, rkf_core::scene_node::BrickMapHandle)> {
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id)?;

        let (primitive, material_id) = match &obj.root_node.sdf_source {
            SdfSource::Analytical { primitive, material_id } => (*primitive, *material_id),
            SdfSource::Voxelized { .. } => return None, // Already voxelized.
            SdfSource::None => return None,
        };

        // Compute object diameter from primitive bounding.
        let diameter = primitive_diameter(&primitive);
        let voxel_size = (diameter / 48.0).clamp(0.005, 0.5);

        // Build AABB with 50% growth margin for sculpting expansion.
        let half = diameter * 0.5;
        let margin = half * 0.5;
        let aabb = Aabb::new(
            Vec3::splat(-half - margin),
            Vec3::splat(half + margin),
        );

        // Build SDF closure from the primitive.
        let sdf_fn = move |pos: Vec3| -> (f32, u16) {
            (rkf_core::evaluate_primitive(&primitive, pos), material_id)
        };

        let result = rkf_core::voxelize_sdf(
            sdf_fn,
            &aabb,
            voxel_size,
            &mut self.cpu_brick_pool,
            &mut self.cpu_brick_map_alloc,
        );

        let (handle, brick_count) = match result {
            Some(r) => r,
            None => {
                log::warn!("Auto-voxelize failed: not enough brick pool slots");
                return None;
            }
        };

        // Compute the grid-aligned AABB from the actual dims returned by voxelize_sdf.
        // The GPU shader centers the grid at the object origin: grid_pos = local_pos + grid_size * 0.5.
        // The voxelizer uses grid_origin = -grid_size/2 (which may differ from aabb.min due to
        // ceil rounding of dims). We MUST store the grid-aligned AABB so the CPU edit path
        // (find_affected_bricks, apply_edit_cpu) uses the same coordinate origin as the shader.
        let brick_size = voxel_size * 8.0;
        let grid_half = Vec3::new(
            handle.dims.x as f32 * brick_size * 0.5,
            handle.dims.y as f32 * brick_size * 0.5,
            handle.dims.z as f32 * brick_size * 0.5,
        );
        let grid_aabb = Aabb::new(-grid_half, grid_half);

        log::info!(
            "Auto-voxelized object {} ({}): {} bricks, voxel_size={:.4}, dims={:?}, aabb_min={:?} grid_min={:?}",
            object_id, obj.name, brick_count, voxel_size, handle.dims, aabb.min, grid_aabb.min,
        );

        // Update the object's SDF source.
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: handle,
            voxel_size,
            aabb: grid_aabb,
        };
        obj.aabb = grid_aabb;

        // Re-upload brick pool + brick maps to GPU.
        self.reupload_brick_data();

        Some((voxel_size, grid_aabb, handle))
    }

    /// Apply a batch of sculpt edit requests to the CPU brick pool and upload changes.
    ///
    /// Each request is converted to an `EditOp` in object-local space, then
    /// `apply_edit_cpu` modifies the CPU brick pool. Changed bricks are
    /// uploaded to the GPU via targeted `queue.write_buffer()`.
    ///
    /// Returns the list of all modified brick pool slot indices.
    ///
    /// If `undo_acc` is provided, bricks are snapshot before modification
    /// so the undo system can restore them.
    pub fn apply_sculpt_edits(
        &mut self,
        scene: &mut Scene,
        edits: &[crate::sculpt::SculptEditRequest],
        mut undo_acc: Option<&mut crate::editor_state::SculptUndoAccumulator>,
    ) -> Vec<u32> {
        use rkf_edit::cpu_apply::apply_edit_cpu;

        use rkf_edit::types::{EditType, FalloffCurve, ShapeType};

        let mut all_modified_slots = Vec::new();

        for req in edits {
            // 1. Auto-voxelize if analytical.
            self.ensure_object_voxelized(scene, req.object_id);

            // 2. Look up the object to get transform + SDF source info.
            let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                Some(o) => o,
                None => continue,
            };

            let (voxel_size, handle_pre, aabb_min_pre) = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { voxel_size, brick_map_handle, aabb } => {
                    (*voxel_size, *brick_map_handle, aabb.min)
                }
                _ => continue, // Shouldn't happen after auto-voxelize.
            };

            // 3. Transform world position to object-local space.
            let local_pos = crate::sculpt::world_to_object_local_v2(
                req.world_position,
                obj,
            );

            // 3b. Sample dominant material from the surface at the brush hit.
            //
            // Instead of using a fixed material_id from BrushSettings, sample
            // the existing voxel data around the hit point and use the most
            // common (dominant) material. This makes additive sculpting
            // naturally continue with the object's existing material.
            let sampled_material = self.sample_dominant_material(
                &handle_pre, voxel_size, aabb_min_pre, local_pos,
            );

            // 4. Convert BrushSettings → EditOp.
            let min_scale = obj.scale.x.min(obj.scale.y.min(obj.scale.z)).max(1e-6);
            let inv_scale = 1.0 / min_scale;

            let edit_type = match req.settings.brush_type {
                crate::sculpt::BrushType::Add => EditType::SmoothUnion,
                crate::sculpt::BrushType::Subtract => EditType::SmoothSubtract,
                crate::sculpt::BrushType::Smooth => EditType::Smooth,
                crate::sculpt::BrushType::Flatten => EditType::Flatten,
                crate::sculpt::BrushType::Sharpen => EditType::Smooth,
            };

            let shape_type = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => ShapeType::Sphere,
                crate::sculpt::BrushShape::Cube => ShapeType::Box,
                crate::sculpt::BrushShape::Cylinder => ShapeType::Cylinder,
            };

            let local_radius = req.settings.radius * inv_scale;
            let local_dims = match req.settings.shape {
                crate::sculpt::BrushShape::Sphere => Vec3::new(local_radius, 0.0, 0.0),
                crate::sculpt::BrushShape::Cube => Vec3::splat(local_radius),
                crate::sculpt::BrushShape::Cylinder => Vec3::new(local_radius, local_radius, 0.0),
            };

            let blend_k = local_radius * 0.3; // default smooth blend

            let op = rkf_edit::edit_op::EditOp {
                object_id: req.object_id,
                position: local_pos,
                rotation: glam::Quat::IDENTITY,
                edit_type,
                shape_type,
                dimensions: local_dims,
                strength: req.settings.strength,
                blend_k,
                falloff: FalloffCurve::Smooth,
                material_id: sampled_material,
                secondary_id: 0,
                color_packed: 0,
            };

            // 5/6. Apply edit: boolean stamp for Add/Subtract, apply_edit_cpu for
            // Smooth/Flatten/Paint. FMM recomputes exact distances after any geometry op.
            let modified: Vec<u32> = if matches!(edit_type, EditType::SmoothUnion | EditType::SmoothSubtract) {
                // Boolean stamp: lazy-allocate + sign-stamp. No SDF blending — just
                // mark voxels solid (-(h*0.5)) or exterior (+(h*0.5)).
                // sculpt_fmm_repair below computes exact Euclidean distances from the
                // new zero-crossings.
                self.apply_sculpt_boolean(
                    scene, req.object_id, &op, voxel_size, undo_acc.as_deref_mut(),
                )
            } else {
                // Smooth/Flatten/Paint: operate on already-allocated bricks only.
                // No new brick allocation needed — these ops only reshape existing geometry.
                let obj = match scene.objects.iter().find(|o| o.id == req.object_id) {
                    Some(o) => o, None => continue,
                };
                let (handle, voxel_size_cur, aabb_min_cur) = match &obj.root_node.sdf_source {
                    SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } =>
                        (*brick_map_handle, *voxel_size, aabb.min),
                    _ => continue,
                };
                let all_bricks = Self::collect_all_allocated_bricks(
                    &self.cpu_brick_map_alloc, &handle, voxel_size_cur, aabb_min_cur,
                );
                if all_bricks.is_empty() {
                    log::warn!("  No allocated bricks found — skipping {:?}", edit_type);
                    continue;
                }
                if let Some(ref mut acc) = undo_acc {
                    for ab in &all_bricks {
                        let slot = ab.brick_base_index / 512;
                        if acc.captured_slots.insert(slot) {
                            acc.snapshots.push((slot, self.cpu_brick_pool.get(slot).clone()));
                        }
                    }
                }
                apply_edit_cpu(&mut self.cpu_brick_pool, &all_bricks, &op)
            };

            // NOTE: Per-stroke JFA repair disabled — JFA produces Euclidean
            // (Voronoi) distances whose gradient is discontinuous at Voronoi cell
            // boundaries, causing faceted brick-sized normal artifacts that are
            // worse than the smooth smin normals. JFA is only invoked via the
            // "Fix SDFs" button (process_fix_sdfs) where the user explicitly
            // trades smooth normals for corrected distance magnitudes.
            // TODO: Consider a near-surface smoothing pass after JFA to restore
            // gradient smoothness while keeping corrected far-field distances.

            // NOTE: we intentionally do NOT call mark_interior_empties() here.
            // The flood-fill heuristic (unreachable from boundary = interior)
            // fails when sculpting creates disconnected bodies — EMPTY bricks
            // between them get incorrectly marked INTERIOR, causing the GPU to
            // render concentric ring artifacts. Interior marking is only safe
            // during initial voxelization (single connected body).

            log::info!(
                "Sculpt pool stats AFTER all: {}/{} used ({} free)",
                self.cpu_brick_pool.allocated_count(),
                self.cpu_brick_pool.capacity(),
                self.cpu_brick_pool.free_count(),
            );

            // 7. Full GPU reupload — ensures CPU and GPU brick data are always
            //    perfectly in sync. This is the safe path; partial uploads can be
            //    re-enabled once the sculpt pipeline is validated.
            self.reupload_brick_data();

            // Always re-upload brick maps — new allocations, interior markers,
            // and consistency passes all modify brick map entries.
            {
                let map_data = self.cpu_brick_map_alloc.as_slice();
                if !map_data.is_empty() {
                    self.gpu_scene.upload_brick_maps(
                        &self.ctx.device, &self.ctx.queue, map_data,
                    );
                }
            }

            all_modified_slots.extend(&modified);
        }

        all_modified_slots
    }

    /// Grow the brick map if the edit extends beyond the current grid.
    ///
    /// The grid is always centered at the object's local origin. When a sculpt
    /// edit extends past the current grid boundary, this method:
    /// 1. Computes new (larger) grid dimensions that contain the edit
    /// 2. Creates a new BrickMap with expanded dims
    /// 3. Copies existing slot entries (with coordinate offset for centering)
    /// 4. Deallocates old map, allocates new map
    /// 5. Updates SdfSource with new handle/AABB
    /// 6. Re-uploads brick maps to GPU
    ///
    /// Returns true if the map was grown.
    fn grow_brick_map_if_needed(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
    ) -> bool {
        use rkf_core::brick_map::BrickMap;

        let obj = match scene.objects.iter().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return false,
        };

        let (handle, voxel_size) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return false,
        };

        let brick_size = voxel_size * 8.0;
        let (edit_min, edit_max) = op.local_aabb();
        let grid_origin = -Vec3::new(
            handle.dims.x as f32 * brick_size * 0.5,
            handle.dims.y as f32 * brick_size * 0.5,
            handle.dims.z as f32 * brick_size * 0.5,
        );
        let grid_end = -grid_origin;

        // Check if edit fits within current grid (with padding margin).
        // ensure_sdf_consistency BFS needs ~2 bricks of padding plus 1 for
        // gradient sampling at the boundary = 3 bricks total.
        let margin = brick_size * 3.0;
        if edit_min.x >= grid_origin.x + margin
            && edit_min.y >= grid_origin.y + margin
            && edit_min.z >= grid_origin.z + margin
            && edit_max.x <= grid_end.x - margin
            && edit_max.y <= grid_end.y - margin
            && edit_max.z <= grid_end.z - margin
        {
            return false; // Fits fine, no growth needed.
        }

        // Compute required extent: union of current grid and edit AABB, plus margin.
        // 4 bricks: 2 for consistency BFS padding + 1 for Catmull-Rom gradient + 1 safety.
        // The brick MAP (flat u32 index array) is cheap; actual GPU cost is only allocated
        // brick pool slots (4KB each), which are proportional to surface area, not grid dims.
        let growth_margin = brick_size * 4.0;
        let required_min = edit_min.min(grid_origin) - Vec3::splat(growth_margin);
        let required_max = edit_max.max(grid_end) + Vec3::splat(growth_margin);

        // New grid must be symmetric about origin (shader assumes centered).
        // The dimension change must be even so that integer pad = (new - old) / 2
        // exactly matches the floating-point centering offset. An odd delta
        // would shift old data by half a brick, scrambling the SDF.
        let half_extent = required_min.abs().max(required_max.abs());
        let raw_dims = glam::UVec3::new(
            ((half_extent.x * 2.0 / brick_size).ceil() as u32).max(handle.dims.x),
            ((half_extent.y * 2.0 / brick_size).ceil() as u32).max(handle.dims.y),
            ((half_extent.z * 2.0 / brick_size).ceil() as u32).max(handle.dims.z),
        );
        // Ensure (new_dims - old_dims) is even on each axis.
        let fix_parity = |new: u32, old: u32| -> u32 {
            if (new.wrapping_sub(old)) % 2 != 0 { new + 1 } else { new }
        };
        let new_dims = glam::UVec3::new(
            fix_parity(raw_dims.x, handle.dims.x),
            fix_parity(raw_dims.y, handle.dims.y),
            fix_parity(raw_dims.z, handle.dims.z),
        );

        if new_dims == handle.dims {
            return false; // Already big enough.
        }

        // Compute offset: old brick (0,0,0) maps to new brick at this offset.
        let pad_x = (new_dims.x - handle.dims.x) / 2;
        let pad_y = (new_dims.y - handle.dims.y) / 2;
        let pad_z = (new_dims.z - handle.dims.z) / 2;

        // Build new BrickMap and copy existing entries.
        // IMPORTANT: copy both allocated bricks AND INTERIOR_SLOT markers.
        // is_unallocated() returns true for INTERIOR_SLOT, but interior markers
        // must be preserved — they tell the GPU that these bricks are deep inside
        // the object (returns -vs*2.0). If lost during growth, they'd become
        // EMPTY_SLOT (returns +vs*2.0), flipping the SDF sign and creating
        // concentric ring artifacts.
        let mut new_map = BrickMap::new(new_dims);
        for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if slot != rkf_core::brick_map::EMPTY_SLOT {
                            new_map.set(bx + pad_x, by + pad_y, bz + pad_z, slot);
                        }
                    }
                }
            }
        }

        // Deallocate old, allocate new.
        self.cpu_brick_map_alloc.deallocate(handle);
        let new_handle = self.cpu_brick_map_alloc.allocate(&new_map);

        // Update AABB to match new grid extent (centered at origin).
        let new_half = Vec3::new(
            new_dims.x as f32 * brick_size * 0.5,
            new_dims.y as f32 * brick_size * 0.5,
            new_dims.z as f32 * brick_size * 0.5,
        );
        let new_aabb = Aabb::new(-new_half, new_half);

        // Update the scene object.
        let obj = scene.objects.iter_mut().find(|o| o.id == object_id).unwrap();
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: new_aabb,
        };
        obj.aabb = new_aabb;

        // Re-upload brick maps to GPU.
        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }

        true
    }

    /// Sample the CPU brick pool SDF at an object-local position using trilinear
    /// interpolation — matching the GPU shader's `sample_voxelized`.
    ///
    /// Sample the dominant material_id from the CPU brick pool near `local_pos`.
    ///
    /// Checks a small neighborhood of voxels (3x3x3 around the nearest voxel)
    /// and returns the most common non-zero material_id. Falls back to 1 if
    /// no material is found (all voxels are empty or material 0).
    fn sample_dominant_material(
        &self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        _aabb_min: Vec3,
        local_pos: Vec3,
    ) -> u16 {
        let vs = voxel_size;
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(
            dims.x as f32 * brick_extent,
            dims.y as f32 * brick_extent,
            dims.z as f32 * brick_extent,
        );

        // Convert local_pos to grid-space (grid is centered at origin).
        let grid_pos = local_pos + grid_size * 0.5;
        let center_voxel = (grid_pos / vs).floor();
        let cx = center_voxel.x as i32;
        let cy = center_voxel.y as i32;
        let cz = center_voxel.z as i32;

        let total_x = (dims.x * 8) as i32;
        let total_y = (dims.y * 8) as i32;
        let total_z = (dims.z * 8) as i32;

        // Sample a 3x3x3 neighborhood and count material occurrences.
        // Only count voxels near the surface (negative SDF, i.e. inside).
        let mut counts: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();

        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let vx = (cx + dx).clamp(0, total_x - 1);
                    let vy = (cy + dy).clamp(0, total_y - 1);
                    let vz = (cz + dz).clamp(0, total_z - 1);

                    let bx = (vx / 8) as u32;
                    let by = (vy / 8) as u32;
                    let bz = (vz / 8) as u32;
                    let lx = (vx % 8) as u32;
                    let ly = (vy % 8) as u32;
                    let lz = (vz % 8) as u32;

                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let sample = self.cpu_brick_pool.get(slot).sample(lx, ly, lz);
                            let mid = sample.material_id();
                            // Only count non-zero materials from voxels near/on surface.
                            if mid != 0 && sample.distance_f32() < voxel_size {
                                *counts.entry(mid).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }

        // Return the most common material, or 1 as fallback.
        counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(mid, _)| mid)
            .unwrap_or(1)
    }

    /// For positions in EMPTY_SLOT bricks, determines the sign (inside/outside)
    /// by using a precomputed interior/exterior classification for EMPTY_SLOT
    /// bricks. Exterior empty bricks return `+vs*4.0`, interior ones return
    /// `-vs*4.0`. This prevents false surfaces in object interiors and
    /// ensures correct normals at brick boundaries.
    fn sample_sdf_cpu(
        pool: &rkf_core::brick_pool::Pool<rkf_core::brick::Brick>,
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        local_pos: Vec3,
        _exterior_bricks: &std::collections::HashSet<(u32, u32, u32)>,
    ) -> f32 {
        let vs = voxel_size;
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(
            dims.x as f32 * brick_extent,
            dims.y as f32 * brick_extent,
            dims.z as f32 * brick_extent,
        );
        let grid_pos = local_pos + grid_size * 0.5;

        // Clamp to valid range.
        let eps = vs * 0.01;
        let clamped = grid_pos.clamp(Vec3::splat(eps), grid_size - Vec3::splat(eps));

        // Convert to continuous voxel coordinates (voxel centers at integers, -0.5 shift).
        let voxel_coord = clamped / vs - Vec3::splat(0.5);
        let v0x = voxel_coord.x.floor() as i32;
        let v0y = voxel_coord.y.floor() as i32;
        let v0z = voxel_coord.z.floor() as i32;
        let tx = voxel_coord.x - v0x as f32;
        let ty = voxel_coord.y - v0y as f32;
        let tz = voxel_coord.z - v0z as f32;

        let total_x = (dims.x * 8) as i32;
        let total_y = (dims.y * 8) as i32;
        let total_z = (dims.z * 8) as i32;

        let read_voxel = |vx: i32, vy: i32, vz: i32| -> f32 {
            let cx = vx.clamp(0, total_x - 1);
            let cy = vy.clamp(0, total_y - 1);
            let cz = vz.clamp(0, total_z - 1);
            let bx = (cx / 8) as u32;
            let by = (cy / 8) as u32;
            let bz = (cz / 8) as u32;
            let lx = (cx % 8) as u32;
            let ly = (cy % 8) as u32;
            let lz = (cz % 8) as u32;

            match alloc.get_entry(handle, bx, by, bz) {
                Some(slot) if !Self::is_unallocated(slot) => {
                    pool.get(slot).sample(lx, ly, lz).distance_f32()
                }
                _ => {
                    // Unallocated bricks always return positive distance,
                    // matching GPU EMPTY_SLOT behavior (vs * 2.0).
                    // Using the flood-fill exterior/interior distinction here
                    // caused false surfaces between disconnected sculpt bodies
                    // (the flood fill can't reach enclosed empty bricks from
                    // the boundary, so they'd get classified as "interior" and
                    // return negative distance, creating concentric ring artifacts).
                    // The BFS in ensure_sdf_consistency naturally propagates
                    // correct negative values from allocated neighbor bricks.
                    vs * 2.0
                }
            }
        };

        // 8-corner trilinear interpolation.
        let c000 = read_voxel(v0x, v0y, v0z);
        let c100 = read_voxel(v0x + 1, v0y, v0z);
        let c010 = read_voxel(v0x, v0y + 1, v0z);
        let c110 = read_voxel(v0x + 1, v0y + 1, v0z);
        let c001 = read_voxel(v0x, v0y, v0z + 1);
        let c101 = read_voxel(v0x + 1, v0y, v0z + 1);
        let c011 = read_voxel(v0x, v0y + 1, v0z + 1);
        let c111 = read_voxel(v0x + 1, v0y + 1, v0z + 1);

        let c00 = c000 + (c100 - c000) * tx;
        let c10 = c010 + (c110 - c010) * tx;
        let c01 = c001 + (c101 - c001) * tx;
        let c11 = c011 + (c111 - c011) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        c0 + (c1 - c0) * tz
    }

    /// Classify all EMPTY_SLOT bricks as exterior or interior via flood-fill.
    ///
    /// BFS from all EMPTY_SLOT bricks on the grid boundary (faces of the 3D
    /// grid), expanding through EMPTY_SLOT only — allocated bricks act as
    /// walls. Any EMPTY_SLOT reachable from the boundary is exterior (should
    /// return `+vs*4.0`). Any EMPTY_SLOT NOT reachable is interior (should
    /// return `-vs*4.0`).
    ///
    /// This replaces the unreliable face-neighbor heuristic that defaulted to
    /// positive for bricks with no allocated neighbors, causing false surfaces
    /// inside sculpted geometry.
    /// Returns true if a brick map slot is unallocated (no pool data).
    fn is_unallocated(slot: u32) -> bool {
        slot == rkf_core::brick_map::EMPTY_SLOT || slot == rkf_core::brick_map::INTERIOR_SLOT
    }

    /// Collect ALL allocated bricks in an object's brick map.
    ///
    /// Returns an `AffectedBrick` for every non-empty brick in the grid.
    /// Used to apply CSG to the entire object (not just the edit AABB),
    /// ensuring perfectly consistent SDF with no missed bricks.
    fn collect_all_allocated_bricks(
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        aabb_min: Vec3,
    ) -> Vec<rkf_edit::edit_op::AffectedBrick> {
        let brick_size = voxel_size * 8.0;
        let mut bricks = Vec::new();

        for bz in 0..handle.dims.z {
            for by in 0..handle.dims.y {
                for bx in 0..handle.dims.x {
                    if let Some(slot) = alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(slot) {
                            let brick_local_min = aabb_min
                                + Vec3::new(
                                    bx as f32 * brick_size,
                                    by as f32 * brick_size,
                                    bz as f32 * brick_size,
                                );
                            bricks.push(rkf_edit::edit_op::AffectedBrick {
                                brick_base_index: slot * 512,
                                brick_local_min: brick_local_min.into(),
                                voxel_size,
                            });
                        }
                    }
                }
            }
        }

        bricks
    }

    fn classify_exterior_bricks(
        alloc: &rkf_core::brick_map::BrickMapAllocator,
        handle: &rkf_core::scene_node::BrickMapHandle,
    ) -> std::collections::HashSet<(u32, u32, u32)> {
        use std::collections::{HashSet, VecDeque};

        let dims = handle.dims;
        let mut exterior: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut queue: VecDeque<(u32, u32, u32)> = VecDeque::new();

        // Seed: all unallocated bricks on any face of the brick map grid.
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let is_boundary = bx == 0 || bx == dims.x - 1
                        || by == 0 || by == dims.y - 1
                        || bz == 0 || bz == dims.z - 1;
                    if !is_boundary {
                        continue;
                    }
                    if let Some(slot) = alloc.get_entry(handle, bx, by, bz) {
                        if Self::is_unallocated(slot) {
                            if exterior.insert((bx, by, bz)) {
                                queue.push_back((bx, by, bz));
                            }
                        }
                    }
                }
            }
        }

        // BFS through unallocated neighbors. Allocated bricks block traversal.
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        while let Some((bx, by, bz)) = queue.pop_front() {
            for &(dx, dy, dz) in &deltas {
                let nx = bx as i32 + dx;
                let ny = by as i32 + dy;
                let nz = bz as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0
                    || nx >= dims.x as i32
                    || ny >= dims.y as i32
                    || nz >= dims.z as i32
                {
                    continue;
                }
                let coord = (nx as u32, ny as u32, nz as u32);
                if let Some(slot) = alloc.get_entry(handle, coord.0, coord.1, coord.2) {
                    if Self::is_unallocated(slot) && exterior.insert(coord) {
                        queue.push_back(coord);
                    }
                }
            }
        }

        exterior
    }

    /// Allocate new bricks in the edit region for additive operations.
    ///
    /// When sculpting "add" material, the brush may extend into brick map cells
    /// that are currently unallocated. This method allocates new bricks and
    /// fills them with a constant SDF based on interior/exterior classification.
    ///
    /// The CSG edit (`apply_edit_cpu`) then creates the correct zero crossings
    /// from the brush shape SDF blended with these constants. This is simpler
    /// and more correct than trilinear interpolation from fallback values, which
    /// produces wrong SDF magnitudes that cause gradient artifacts.
    ///
    /// Also grows the brick map if the edit extends beyond the current grid.
    ///
    /// Returns the list of newly allocated pool slot indices.
    ///
    /// **Important:** After allocation, `prefill_new_brick_faces()` should be
    /// called to replace constant fill with SDF extrapolated from existing
    /// neighbors before CSG is applied. Without this, constant fill creates
    /// seam artifacts at the transition between old and new bricks.
    fn allocate_bricks_in_region(
        &mut self,
        scene: &mut Scene,
        object_id: u32,
        op: &rkf_edit::edit_op::EditOp,
        voxel_size: f32,
    ) -> Vec<u32> {
        self.grow_brick_map_if_needed(scene, object_id, op);

        let obj = match scene.objects.iter_mut().find(|o| o.id == object_id) {
            Some(o) => o,
            None => return Vec::new(),
        };

        let (handle, _aabb_min) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, aabb, .. } => (*brick_map_handle, aabb.min),
            _ => return Vec::new(),
        };

        let brick_size = voxel_size * 8.0;
        let (edit_min, edit_max) = op.local_aabb();

        let bmin = ((edit_min - _aabb_min) / brick_size).floor();
        let bmax = ((edit_max - _aabb_min) / brick_size - Vec3::splat(0.001)).ceil();

        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(handle.dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(handle.dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(handle.dims.z.saturating_sub(1));

        // The CSG falloff is spherical (Euclidean distance from brush center),
        // but the AABB is a cube. Bricks at the AABB corners are outside the
        // falloff sphere and won't be modified by CSG. If we allocate them with
        // constant-fill values (±vs*2.0 based on interior/exterior), the sign
        // boundary between interior and exterior creates a false zero-crossing —
        // a phantom surface that renders as "stalactite" artifacts.
        //
        // Fix: only allocate bricks whose centers are within the falloff sphere
        // (plus half a brick diagonal for overlap margin). Bricks outside stay
        // as EMPTY_SLOT and get properly filled by ensure_sdf_consistency's BFS.
        let max_dim = op.dimensions.x.max(op.dimensions.y).max(op.dimensions.z);
        let falloff_radius = max_dim + op.blend_k;
        let brick_half_diag = brick_size * 0.5 * (3.0f32).sqrt();
        let alloc_radius = falloff_radius + brick_half_diag;
        let alloc_radius_sq = alloc_radius * alloc_radius;

        let mut new_slots = Vec::new();

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        if !Self::is_unallocated(s) {
                            continue; // Already allocated — skip.
                        }
                    }

                    // Check if this brick's center is within the falloff sphere.
                    // Bricks outside the sphere won't be modified by CSG and
                    // should stay EMPTY_SLOT for the consistency BFS to handle.
                    let brick_center = _aabb_min + Vec3::new(
                        (bx as f32 + 0.5) * brick_size,
                        (by as f32 + 0.5) * brick_size,
                        (bz as f32 + 0.5) * brick_size,
                    );
                    let dist_sq = (brick_center - op.position).length_squared();
                    if dist_sq > alloc_radius_sq {
                        continue; // Outside falloff sphere — skip.
                    }

                    if let Some(new_slot) = self.cpu_brick_pool.allocate() {
                        // Initialize as exterior. Newly-allocated bricks represent space
                        // the brush may reach into, but they have no geometry yet.
                        // vs*2.0 = "exterior, close to the surface" — small enough that
                        // trilinear interpolation stays conservative near newly-sculpted
                        // surfaces (no overshoot), large enough that the zero-crossing
                        // only appears when CSG actually writes interior voxels.
                        //
                        // Bricks that CSG leaves with no zero-crossing are reverted to
                        // EMPTY_SLOT by Step 6c, so no false-shadow contribution.
                        let fill = voxel_size * 2.0;
                        {
                            use rkf_core::voxel::VoxelSample;
                            let brick = self.cpu_brick_pool.get_mut(new_slot);
                            for z in 0u32..8 {
                                for y in 0u32..8 {
                                    for x in 0u32..8 {
                                        brick.set(x, y, z, VoxelSample::new(fill, 0, 0, 0, 0));
                                    }
                                }
                            }
                        }

                        self.cpu_brick_map_alloc.set_entry(&handle, bx, by, bz, new_slot);
                        new_slots.push(new_slot);
                    }
                }
            }
        }

        if !new_slots.is_empty() {
            log::info!(
                "  allocate_bricks_in_region: allocated {} new bricks (brush SDF fill)",
                new_slots.len(),
            );
            let map_data = self.cpu_brick_map_alloc.as_slice();
            if !map_data.is_empty() {
                self.gpu_scene.upload_brick_maps(
                    &self.ctx.device, &self.ctx.queue, map_data,
                );
            }
        }

        new_slots
    }

    /// Pre-fill new brick voxels by extrapolating from adjacent filled bricks.
    ///
    /// Before CSG runs, this ensures the boundary between old and new bricks
    /// has consistent background SDF. Since `smooth_min(same_bg, brush, k)`
    /// produces identical results on both sides, there's no gradient discontinuity
    /// and no visible shading seam.
    ///
    /// Uses multi-pass flood fill: each pass propagates from already-filled
    /// bricks (old or previously-filled new) into unfilled new bricks. Fills
    /// all 8 voxels deep (full brick depth) using linear extrapolation with
    /// (d+1) spatial offset correction to account for voxel center spacing
    /// across the brick boundary. Repeats until all reachable new bricks
    /// have smooth, continuous SDF background.
    fn prefill_new_brick_faces(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        newly_allocated: &[u32],
    ) {
        use std::collections::HashSet;
        use rkf_core::voxel::VoxelSample;

        if newly_allocated.is_empty() {
            return;
        }

        let vs = voxel_size;
        let dims = handle.dims;
        let new_set: HashSet<u32> = newly_allocated.iter().copied().collect();

        // Build (bx, by, bz, slot) for new bricks.
        let mut new_coords: Vec<(u32, u32, u32, u32)> = Vec::new();
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if new_set.contains(&s) {
                            new_coords.push((bx, by, bz, s));
                        }
                    }
                }
            }
        }

        let face_dirs: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        let fill = vs * 2.0; // constant fill value
        let depth = 8u32; // fill entire brick

        // Multi-pass flood fill: propagate from old bricks through layers of
        // new bricks. Each pass fills unfilled new bricks from any filled
        // neighbor (old or previously-filled new). This ensures ALL new bricks
        // the brush touches have a smooth, continuous SDF background.
        let mut filled_set: HashSet<u32> = HashSet::new();
        // All non-new allocated bricks are "filled" (they have real SDF data).
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(s) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if !Self::is_unallocated(s) && !new_set.contains(&s) {
                            filled_set.insert(s);
                        }
                    }
                }
            }
        }

        let mut total_copied = 0u32;
        let mut pass = 0u32;
        let max_passes = 10; // safety limit

        loop {
            pass += 1;
            if pass > max_passes { break; }

            let mut pass_copied = 0u32;
            let mut newly_filled: Vec<u32> = Vec::new();

            for &(nbx, nby, nbz, new_slot) in &new_coords {
                if filled_set.contains(&new_slot) { continue; }

                let mut brick_got_data = false;

                for &(dx, dy, dz) in &face_dirs {
                    let ax = nbx as i32 + dx;
                    let ay = nby as i32 + dy;
                    let az = nbz as i32 + dz;
                    if ax < 0 || ay < 0 || az < 0 { continue; }
                    let (abx, aby, abz) = (ax as u32, ay as u32, az as u32);
                    if abx >= dims.x || aby >= dims.y || abz >= dims.z { continue; }

                    // Copy from any FILLED neighbor (old or already-filled new).
                    let adj_slot = match self.cpu_brick_map_alloc.get_entry(handle, abx, aby, abz) {
                        Some(s) if filled_set.contains(&s) => s,
                        _ => continue,
                    };

                    // Pre-read face data to avoid borrow conflict with get_mut.
                    let mut face_data = [[(0.0f32, 0.0f32, 0u16); 8]; 8];
                    {
                        let adj_brick = self.cpu_brick_pool.get(adj_slot);
                        for a in 0u32..8 {
                            for b in 0u32..8 {
                                let (fv, pv, mat) = if dx == -1 {
                                    (adj_brick.sample(7, a, b).distance_f32(),
                                     adj_brick.sample(6, a, b).distance_f32(),
                                     adj_brick.sample(7, a, b).material_id())
                                } else if dx == 1 {
                                    (adj_brick.sample(0, a, b).distance_f32(),
                                     adj_brick.sample(1, a, b).distance_f32(),
                                     adj_brick.sample(0, a, b).material_id())
                                } else if dy == -1 {
                                    (adj_brick.sample(a, 7, b).distance_f32(),
                                     adj_brick.sample(a, 6, b).distance_f32(),
                                     adj_brick.sample(a, 7, b).material_id())
                                } else if dy == 1 {
                                    (adj_brick.sample(a, 0, b).distance_f32(),
                                     adj_brick.sample(a, 1, b).distance_f32(),
                                     adj_brick.sample(a, 0, b).material_id())
                                } else if dz == -1 {
                                    (adj_brick.sample(a, b, 7).distance_f32(),
                                     adj_brick.sample(a, b, 6).distance_f32(),
                                     adj_brick.sample(a, b, 7).material_id())
                                } else {
                                    (adj_brick.sample(a, b, 0).distance_f32(),
                                     adj_brick.sample(a, b, 1).distance_f32(),
                                     adj_brick.sample(a, b, 0).material_id())
                                };
                                face_data[a as usize][b as usize] = (fv, pv, mat);
                            }
                        }
                    }

                    // Extrapolate `depth` voxels from neighbor's face into new brick.
                    // Use (d+1) offset because voxel centers are spaced 1*vs apart:
                    // face voxel (A[7]) is at 7.5*vs, dest voxel B[0] is at 8.5*vs.
                    // So B[d] should have value at face_val + grad*(d+1), not face_val + grad*d.
                    for a in 0u32..8 {
                        for b in 0u32..8 {
                            let (face_val, prev_val, mat) = face_data[a as usize][b as usize];
                            let grad = (face_val - prev_val).clamp(-vs, vs);

                            for d in 0..depth {
                                // Clamp to positive: prevents false surfaces from
                                // negative extrapolation in concave/corner regions.
                                let val = (face_val + grad * ((d + 1) as f32)).max(vs * 0.5);

                                let (dst_x, dst_y, dst_z) = if dx == -1 {
                                    (d, a, b)
                                } else if dx == 1 {
                                    (7 - d, a, b)
                                } else if dy == -1 {
                                    (a, d, b)
                                } else if dy == 1 {
                                    (a, 7 - d, b)
                                } else if dz == -1 {
                                    (a, b, d)
                                } else {
                                    (a, b, 7 - d)
                                };

                                let cur = self.cpu_brick_pool.get(new_slot)
                                    .sample(dst_x, dst_y, dst_z).distance_f32();
                                if (cur - fill).abs() < 0.01 || val.abs() < cur.abs() {
                                    self.cpu_brick_pool.get_mut(new_slot).set(
                                        dst_x, dst_y, dst_z,
                                        VoxelSample::new(val, mat, 0, 0, 0),
                                    );
                                }
                            }
                        }
                    }
                    brick_got_data = true;
                    pass_copied += 1;
                }

                if brick_got_data {
                    newly_filled.push(new_slot);
                }
            }

            for s in &newly_filled {
                filled_set.insert(*s);
            }
            total_copied += pass_copied;

            if pass_copied == 0 { break; }
        }

        if total_copied > 0 {
            log::info!(
                "  prefill_new_brick_faces: extrapolated {} face boundaries over {} passes",
                total_copied, pass - 1,
            );
        }
    }

    /// Ensure SDF consistency in the region around a sculpt edit.
    ///
    /// After a CSG edit, the GPU may see gradient discontinuities at boundaries
    /// between allocated bricks (with real SDF) and EMPTY_SLOT bricks (which
    /// the GPU shader treats as `+vs*4.0`). This produces flipped normals on
    /// the interior side of sculpted geometry.
    ///
    /// This method fixes the problem by brute-force: it allocates and fills
    /// ALL EMPTY_SLOT bricks within a padded region around the edit. Bricks
    /// are processed in BFS layers outward from already-allocated bricks, so
    /// each layer can sample SDF from previously-filled neighbors. The region
    /// extends far enough past the brush influence that the boundary is
    /// entirely positive SDF, guaranteeing no false surfaces from EMPTY_SLOT.
    ///
    /// Returns newly allocated pool slot indices.
    fn ensure_sdf_consistency(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
        voxel_size: f32,
        aabb_min: Vec3,
        edit_op: &rkf_edit::edit_op::EditOp,
    ) -> Vec<u32> {
        use std::collections::{HashSet, VecDeque};
        use rkf_core::voxel::VoxelSample;

        let dims = handle.dims;
        let brick_size = voxel_size * 8.0;

        // Compute a padded region: the edit AABB expanded by 4 bricks on
        // each side. This ensures the allocated boundary is well past the
        // brush influence zone, where SDF is safely positive.
        let (edit_min, edit_max) = edit_op.local_aabb();
        let pad = brick_size * 4.0;
        let region_min = edit_min - Vec3::splat(pad);
        let region_max = edit_max + Vec3::splat(pad);

        // Convert to brick coordinates, clamped to grid bounds.
        let bmin = ((region_min - aabb_min) / brick_size).floor();
        let bmax = ((region_max - aabb_min) / brick_size - Vec3::splat(0.001)).ceil();
        let bmin_x = (bmin.x as i32).max(0) as u32;
        let bmin_y = (bmin.y as i32).max(0) as u32;
        let bmin_z = (bmin.z as i32).max(0) as u32;
        let bmax_x = ((bmax.x as i32).max(0) as u32).min(dims.x.saturating_sub(1));
        let bmax_y = ((bmax.y as i32).max(0) as u32).min(dims.y.saturating_sub(1));
        let bmax_z = ((bmax.z as i32).max(0) as u32).min(dims.z.saturating_sub(1));

        // Collect all EMPTY_SLOT bricks in the region and seed the BFS
        // from all already-allocated bricks.
        let mut empty_set: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut bfs_seeds: VecDeque<(u32, u32, u32)> = VecDeque::new();

        for bz in bmin_z..=bmax_z {
            for by in bmin_y..=bmax_y {
                for bx in bmin_x..=bmax_x {
                    match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(s) if !Self::is_unallocated(s) => {
                            bfs_seeds.push_back((bx, by, bz));
                        }
                        _ => {
                            empty_set.insert((bx, by, bz));
                        }
                    }
                }
            }
        }

        if empty_set.is_empty() {
            return Vec::new();
        }

        log::info!(
            "  ensure_sdf_consistency: region [{},{}]→[{},{}], {} empty bricks to fill",
            bmin_x, bmin_y, bmax_x, bmax_y, empty_set.len(),
        );

        // BFS from allocated bricks outward. Each "wave" processes all
        // EMPTY_SLOT bricks adjacent to the current frontier. Because each
        // wave is registered before the next starts, `sample_sdf_cpu` can
        // read from previously-filled bricks for correct sign propagation.
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];

        let mut visited: HashSet<(u32, u32, u32)> = HashSet::new();
        let mut new_slots: Vec<(u32, u32, u32, u32)> = Vec::new(); // (bx, by, bz, slot)

        // Classify exterior vs interior EMPTY_SLOT bricks via flood-fill.
        let exterior = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, handle);

        // Mark all seeds as visited (they're already allocated).
        for &coord in bfs_seeds.iter() {
            visited.insert(coord);
        }

        // BFS layer by layer.
        while !bfs_seeds.is_empty() {
            // Collect the next wave: all EMPTY_SLOT neighbors of the current
            // frontier that haven't been visited yet.
            let mut next_wave: Vec<(u32, u32, u32)> = Vec::new();
            for &(bx, by, bz) in bfs_seeds.iter() {
                for &(dx, dy, dz) in &deltas {
                    let nx = bx as i32 + dx;
                    let ny = by as i32 + dy;
                    let nz = bz as i32 + dz;
                    if nx < 0 || ny < 0 || nz < 0 {
                        continue;
                    }
                    let coord = (nx as u32, ny as u32, nz as u32);
                    if empty_set.contains(&coord) && visited.insert(coord) {
                        next_wave.push(coord);
                    }
                }
            }

            if next_wave.is_empty() {
                break;
            }

            // Allocate + fill every brick in this wave.
            for &(nbx, nby, nbz) in &next_wave {
                // Sample SDF into temp buffer (immutable borrows of pool).
                let mut sdf_buf = [0.0f32; 512];
                let mut min_val = f32::MAX;
                for z in 0u32..8 {
                    for y in 0u32..8 {
                        for x in 0u32..8 {
                            let voxel_pos = aabb_min
                                + Vec3::new(
                                    (nbx * 8 + x) as f32 * voxel_size + voxel_size * 0.5,
                                    (nby * 8 + y) as f32 * voxel_size + voxel_size * 0.5,
                                    (nbz * 8 + z) as f32 * voxel_size + voxel_size * 0.5,
                                );
                            let d = Self::sample_sdf_cpu(
                                &self.cpu_brick_pool,
                                &self.cpu_brick_map_alloc,
                                handle,
                                voxel_size,
                                voxel_pos,
                                &exterior,
                            );
                            sdf_buf[(x + y * 8 + z * 64) as usize] = d;
                            if d < min_val { min_val = d; }
                        }
                    }
                }

                // Skip bricks that are entirely outside any surface.
                // When the interpolated SDF is uniformly positive (min > vs),
                // this brick is in open air (e.g. between cross arms). Allocating
                // it would create false surfaces from interpolation artifacts.
                // The GPU returns vs*2.0 for EMPTY_SLOT, which is close enough
                // that there's no visible discontinuity.
                if min_val > voxel_size {
                    continue;
                }

                // Allocate and fill (mutable borrows).
                if let Some(new_slot) = self.cpu_brick_pool.allocate() {
                    let nbrick = self.cpu_brick_pool.get_mut(new_slot);
                    for z in 0u32..8 {
                        for y in 0u32..8 {
                            for x in 0u32..8 {
                                let d = sdf_buf[(x + y * 8 + z * 64) as usize];
                                nbrick.set(x, y, z, VoxelSample::new(d, 0, 0, 0, 0));
                            }
                        }
                    }
                    self.cpu_brick_map_alloc.set_entry(handle, nbx, nby, nbz, new_slot);
                    new_slots.push((nbx, nby, nbz, new_slot));
                }
            }

            // Remove filled bricks from the empty set so they're not
            // re-processed, and use this wave as the next frontier.
            for &coord in &next_wave {
                empty_set.remove(&coord);
            }
            bfs_seeds.clear();
            bfs_seeds.extend(next_wave);
        }

        let slot_ids: Vec<u32> = new_slots.iter().map(|&(_, _, _, s)| s).collect();

        if !slot_ids.is_empty() {
            log::info!("  ensure_sdf_consistency: allocated {} bricks", slot_ids.len());
            let map_data = self.cpu_brick_map_alloc.as_slice();
            if !map_data.is_empty() {
                self.gpu_scene.upload_brick_maps(
                    &self.ctx.device, &self.ctx.queue, map_data,
                );
            }
        }

        slot_ids
    }

    /// Mark all interior EMPTY_SLOT bricks with INTERIOR_SLOT in the brick map.
    ///
    /// The GPU shader returns `+vs*4.0` for EMPTY_SLOT and `-vs*4.0` for
    /// INTERIOR_SLOT. This method classifies unallocated bricks via flood-fill
    /// from the grid boundary and sets INTERIOR_SLOT for any EMPTY_SLOT brick
    /// that isn't reachable from the boundary (i.e. enclosed by allocated bricks).
    /// Compute the tight local-space AABB of all non-EMPTY_SLOT bricks for a
    /// voxelized FlatNode. Returns `([0.0;3], [0.0;3])` for non-voxelized nodes
    /// or when no bricks are allocated (disables the geometry AABB optimization).
    ///
    /// Used to populate `GpuObject::geometry_aabb_min/max` so the shader can
    /// skip the empty expanded region that `grow_brick_map_if_needed` adds.
    fn compute_geometry_aabb_for_flat_node(
        &self,
        flat: &rkf_core::transform_flatten::FlatNode,
    ) -> ([f32; 3], [f32; 3]) {
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
        let (handle, vs) = match &flat.sdf_source {
            rkf_core::SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                (*brick_map_handle, *voxel_size)
            }
            _ => return ([0.0; 3], [0.0; 3]),
        };
        let brick_extent = vs * 8.0;
        let dims = handle.dims;
        let grid_size = Vec3::new(dims.x as f32, dims.y as f32, dims.z as f32) * brick_extent;
        let mut gmin = Vec3::splat(f32::MAX);
        let mut gmax = Vec3::splat(f32::MIN);
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    let slot = match self.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                        Some(s) => s,
                        None => continue,
                    };
                    // Skip sentinel values — only real pool slots contribute geometry.
                    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                        continue;
                    }
                    let brick_min = Vec3::new(bx as f32, by as f32, bz as f32) * brick_extent - grid_size * 0.5;
                    let brick_max = Vec3::new((bx + 1) as f32, (by + 1) as f32, (bz + 1) as f32) * brick_extent - grid_size * 0.5;
                    gmin = gmin.min(brick_min);
                    gmax = gmax.max(brick_max);
                }
            }
        }
        if gmin.x > gmax.x {
            return ([0.0; 3], [0.0; 3]);
        }
        (gmin.to_array(), gmax.to_array())
    }

    ///
    /// Unlike `fill_interior_empties`, this does NOT allocate pool memory — it
    /// only updates the brick map entries. The GPU handles the sign correctly
    /// without needing constant-fill bricks that create gradient discontinuities.
    ///
    /// Returns `true` if any entries were changed (brick maps need re-upload).
    fn mark_interior_empties(
        &mut self,
        handle: &rkf_core::scene_node::BrickMapHandle,
    ) -> bool {
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};

        let exterior = Self::classify_exterior_bricks(&self.cpu_brick_map_alloc, handle);
        let dims = handle.dims;
        let mut changed = false;

        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(slot) = self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        if slot == EMPTY_SLOT && !exterior.contains(&(bx, by, bz)) {
                            self.cpu_brick_map_alloc.set_entry(handle, bx, by, bz, INTERIOR_SLOT);
                            changed = true;
                        }
                    }
                }
            }
        }

        if changed {
            log::info!("  mark_interior_empties: marked interior bricks as INTERIOR_SLOT");
        }

        changed
    }

    /// Restore brick pool data from undo snapshots.
    ///
    /// Writes each snapshot brick back to the CPU pool and uploads to GPU.
    pub fn apply_sculpt_undo(&mut self, snapshots: &[(u32, rkf_core::brick::Brick)]) {
        let brick_byte_size = std::mem::size_of::<rkf_core::brick::Brick>() as u64;
        for (slot, brick) in snapshots {
            *self.cpu_brick_pool.get_mut(*slot) = brick.clone();
            let offset = *slot as u64 * brick_byte_size;
            let brick_data: &[u8] = bytemuck::bytes_of(self.cpu_brick_pool.get(*slot));
            let gpu_buf_size = self.gpu_scene.brick_pool_buffer().size();
            if offset + brick_byte_size <= gpu_buf_size {
                self.ctx.queue.write_buffer(
                    self.gpu_scene.brick_pool_buffer(),
                    offset,
                    brick_data,
                );
            }
        }
    }

    /// Re-upload the entire CPU brick pool and brick maps to the GPU.
    ///
    /// Called after voxelization or when the GPU buffer needs to grow.
    fn reupload_brick_data(&mut self) {
        let pool_data: &[u8] = bytemuck::cast_slice(self.cpu_brick_pool.as_slice());
        let gpu_buf = self.gpu_scene.brick_pool_buffer();

        if pool_data.len() as u64 > gpu_buf.size() {
            // GPU buffer too small — recreate it.
            let new_buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brick_pool"),
                size: pool_data.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.ctx.queue.write_buffer(&new_buf, 0, pool_data);
            self.gpu_scene.set_brick_pool(&self.ctx.device, new_buf);
        } else {
            self.ctx.queue.write_buffer(gpu_buf, 0, pool_data);
        }

        let map_data = self.cpu_brick_map_alloc.as_slice();
        if !map_data.is_empty() {
            self.gpu_scene.upload_brick_maps(
                &self.ctx.device, &self.ctx.queue, map_data,
            );
        }
    }

    /// Render one frame without UI overlay (full-screen engine blit).
    pub fn render_frame(&mut self, scene: &Scene) {
        self.render_frame_composited(scene, |_, _, _| {});
    }

    /// Render one frame with an overlay compositing callback.
    ///
    /// The callback runs after the engine has submitted its compute passes
    /// and blit to the swapchain, but before present. The callback receives
    /// the device, queue, and swapchain target view so it can render an
    /// overlay (e.g. rinch UI) on top.
    ///
    /// The engine blit fills the full swapchain. Use `render_frame_viewport`
    /// for sub-region rendering with panels.
    pub fn render_frame_composited<F>(&mut self, scene: &Scene, post_engine: F)
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(scene, None, post_engine);
    }

    /// Render one frame with engine output constrained to a viewport sub-region.
    ///
    /// The engine blit is drawn into `viewport` (x, y, width, height) in pixels.
    /// Areas outside the viewport are cleared to black (covered by UI panels).
    /// The `post_engine` callback composites the UI overlay on top.
    pub fn render_frame_viewport<F>(
        &mut self,
        scene: &Scene,
        viewport: (f32, f32, f32, f32),
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(scene, Some(viewport), post_engine);
    }

    fn render_frame_inner<F>(
        &mut self,
        scene: &Scene,
        viewport: Option<(f32, f32, f32, f32)>,
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        let camera_pos_vec = self.camera.position;

        // Bake world transforms for parent-child hierarchy.
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        // Flatten all objects → GPU object list + BVH.
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        let mut world_aabbs_for_coarse: Vec<(Vec3, Vec3)> = Vec::new();
        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos_vec;
            // Transform local AABB to world space via baked transform.
            let local_aabb = obj.aabb;
            let world_aabb = {
                let smin = local_aabb.min * wt.scale;
                let smax = local_aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z),
                    Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z),
                    Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z),
                    Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z),
                    Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                Aabb::new(wmin, wmax)
            };
            world_aabbs_for_coarse.push((world_aabb.min, world_aabb.max));
            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - self.camera.position;
                let cam_rel_max = world_aabb.max - self.camera.position;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                    geom_min,
                    geom_max,
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        self.gpu_scene.upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);

        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Repopulate coarse field every frame so moved objects stay visible.
        self.coarse_field.populate(&world_aabbs_for_coarse);
        self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

        // Update camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        // Scene uniforms.
        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Synthesize directional light from environment sun settings.
        let sun_light = Light {
            light_type: 0, // directional
            pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: self.env_sun_dir[0],
            dir_y: self.env_sun_dir[1],
            dir_z: self.env_sun_dir[2],
            color_r: self.env_sun_color[0],
            color_g: self.env_sun_color[1],
            color_b: self.env_sun_color[2],
            intensity: 1.0, // already baked into env_sun_color (sun_color * sun_intensity)
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: 1,
        };

        // Camera-relative point/spot lights.
        let cam = self.camera.position;
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            if cl.light_type != 0 {
                cl.pos_x -= cam.x;
                cl.pos_y -= cam.y;
                cl.pos_z -= cam.z;
            }
            cl
        }).collect();

        // Sun (directional) + point/spot lights.
        let total_lights = 1 + cam_rel_lights.len() as u32;
        let mut all_lights = vec![sun_light];
        all_lights.extend(cam_rel_lights);
        self.light_buffer.update(&self.ctx.queue, &all_lights);

        // Shade uniforms — includes atmosphere + camera basis for sky rendering.
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = self.render_width as f32 / self.render_height as f32;
        let fwd = self.camera.forward();
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;

        self.shading_pass.update_uniforms(&self.ctx.queue, &ShadeUniforms {
            debug_mode: self.shade_debug_mode,
            num_lights: total_lights,
            _pad0: 0,
            shadow_budget_k: 0,
            camera_pos: [camera_pos_vec.x, camera_pos_vec.y, camera_pos_vec.z, 0.0],
            sun_dir: [self.env_sun_dir[0], self.env_sun_dir[1], self.env_sun_dir[2], self.env_sun_intensity],
            sun_color: [self.env_sun_color_raw[0], self.env_sun_color_raw[1], self.env_sun_color_raw[2], 0.0],
            sky_params: [self.env_rayleigh_scale, self.env_mie_scale, if self.env_atmosphere_enabled { 1.0 } else { 0.0 }, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
        });

        // Debug view (kept for fallback).
        let debug_mode = match self.shade_debug_mode {
            0 => DebugMode::Lambert,
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, debug_mode);

        // Store VP for next frame's motion vectors.
        self.prev_vp = self.camera.view_projection(self.render_width, self.render_height)
            .to_cols_array_2d();

        // Poll the GPU device to acknowledge completed work from previous frames.
        // Without this, heavy compute (cloud FBM) can fill the swapchain during
        // rapid slider drags, causing get_current_texture() to block permanently.
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        // Get swapchain texture (surface-based path).
        let surface = self.surface.as_ref()
            .expect("render_frame_inner requires a surface (use render_frame_offscreen for compositor path)");
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(
                    self.window_width.max(64),
                    self.window_height.max(64),
                );
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let target_view = frame.texture.create_view(&Default::default());

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // Update per-frame uniforms.
        self.coarse_field.update_uniforms(&self.ctx.queue, self.camera.position);
        self.radiance_volume.update_center(
            &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
        );
        self.radiance_inject.update_uniforms(&self.ctx.queue, &InjectUniforms {
            num_lights: total_lights,
            max_shadow_lights: 1,
            _pad: [0; 2],
        });

        // --- Core rendering ---
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        self.radiance_inject.dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.radiance_mip.dispatch(&mut encoder);
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );

        // --- Volumetric pipeline (uses cached env values) ---
        // Upload cloud parameters each frame (time advances for wind animation).
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

        // DEBUG: Save cloud params for post-frame diagnostic.
        let cloud_params_snapshot = [
            cloud_params.flags[0],     // enabled
            cloud_params.altitude[0],  // cloud_min
            cloud_params.altitude[1],  // cloud_max
            cloud_params.altitude[2],  // threshold
            cloud_params.altitude[3],  // density_scale
        ];
        let cam_snapshot = [cam.x, cam.y, cam.z];

        self.vol_march.set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow.set_cloud_params(&self.ctx.queue, &cloud_params);

        let sun_dir = self.env_sun_dir;
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
        );
        // Use the actual cloud altitude from cloud_params, not the defaults
        // (DEFAULT_CLOUD_MIN=1000, DEFAULT_CLOUD_MAX=3000).  The shadow map must
        // march through the same altitude band as the visible clouds.
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],  // cloud_min
            cloud_params.altitude[1],  // cloud_max
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);
        let sc = self.env_sun_color;
        let fc = self.env_fog_color;
        let fog_alpha = if self.env_fog_density > 0.0 { 1.0 } else { 0.0 };
        // Reuse the FOV-scaled camera basis from the shade pass so vol march
        // rays match the G-buffer exactly (same fwd/right/up from line ~990).
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [sc[0], sc[1], sc[2], 0.0],
            width: self.render_width / 2,
            height: self.render_height / 2,
            full_width: self.render_width,
            full_height: self.render_height,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [fc[0], fc[1], fc[2], fog_alpha],
            fog_height: [self.env_fog_density, -0.5, self.env_fog_height_falloff, 0.0],
            fog_distance: [0.0, 0.01, self.env_ambient_dust, self.env_dust_g],
            frame_index: self.frame_index,
            _pad0: 0, _pad1: 0, _pad2: 0,
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // --- Post-processing pipeline ---
        // Project sun to screen UV for radial blur god rays.
        {
            let sun_dir = glam::Vec3::from(self.env_sun_dir).normalize_or_zero();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir.dot(right) / sun_dot;
                let ndc_y = -sun_dir.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur.update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // --- Blit engine output to full swapchain ---
        // Always blit to the full window. The rinch overlay will paint opaque
        // panels over non-viewport areas (like the game-embed and video player
        // examples). This avoids sub-pixel alignment gaps between the viewport
        // blit region and the overlay's transparent hole.
        let _ = viewport; // viewport used only for resize_render, not blit positioning
        self.blit.draw(&mut encoder, &target_view);

        // Submit engine work.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Let the caller composite overlay (e.g. rinch UI) on top.
        // This runs after engine submit (Vello needs the queue), before present.
        post_engine(&self.ctx.device, &self.ctx.queue, &target_view);

        // --- GPU pick readback (single pixel from material G-buffer) ---
        let pending_pick = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_pick.take());

        if let Some((px, py)) = pending_pick {
            // Clamp to internal resolution bounds.
            let px = px.min(self.render_width.saturating_sub(1));
            let py = py.min(self.render_height.saturating_sub(1));

            let mut pick_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("pick_readback") },
            );
            pick_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256), // must be 256-aligned
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(pick_enc.finish()));

            // Synchronous readback — fast for 1 pixel.
            let slice = self.pick_readback_buffer.slice(..4);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let object_id = packed >> 24; // bits 24-31
                drop(data);
                self.pick_readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.pick_result = Some(object_id);
                }
            } else {
                self.pick_readback_buffer.unmap();
            }
        }

        // --- GPU brush hit readback (position + object_id from G-buffer) ---
        let pending_brush = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_brush_hit.take());

        if let Some((bx, by)) = pending_brush {
            let bx = bx.min(self.render_width.saturating_sub(1));
            let by = by.min(self.render_height.saturating_sub(1));

            let mut brush_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("brush_readback") },
            );

            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.position_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.brush_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            self.ctx.queue.submit(std::iter::once(brush_enc.finish()));

            let pos_slice = self.brush_readback_buffer.slice(..16);
            let (tx_pos, rx_pos) = std::sync::mpsc::channel();
            pos_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_pos.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut hit_pos = [0.0f32; 4];
            let pos_ok = if let Ok(Ok(())) = rx_pos.recv() {
                let data = pos_slice.get_mapped_range();
                hit_pos = [
                    f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                    f32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                    f32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                    f32::from_le_bytes([data[12], data[13], data[14], data[15]]),
                ];
                drop(data);
                self.brush_readback_buffer.unmap();
                true
            } else {
                self.brush_readback_buffer.unmap();
                false
            };

            let mat_slice = self.pick_readback_buffer.slice(..4);
            let (tx_mat, rx_mat) = std::sync::mpsc::channel();
            mat_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_mat.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut object_id = 0u32;
            if let Ok(Ok(())) = rx_mat.recv() {
                let data = mat_slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                object_id = packed >> 24;
                drop(data);
                self.pick_readback_buffer.unmap();
            } else {
                self.pick_readback_buffer.unmap();
            }

            if pos_ok && hit_pos[3] < 1e30 {
                let result = crate::automation::BrushHitResult {
                    position: Vec3::new(hit_pos[0], hit_pos[1], hit_pos[2]),
                    object_id,
                };
                if let Ok(mut state) = self.shared_state.lock() {
                    state.brush_hit_result = Some(result);
                }
            }
        }

        // --- Screenshot readback (after overlay composite, captures full UI) ---
        let do_readback = self.shared_state.lock()
            .map(|s| s.screenshot_requested)
            .unwrap_or(false);

        if do_readback {
            let w = self.window_width;
            let h = self.window_height;
            let bytes_per_pixel = 4u32;
            let unpadded_row = w * bytes_per_pixel;
            let padded_row = (unpadded_row + 255) & !255;

            // Copy from composited swapchain texture to readback buffer.
            let mut readback_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("screenshot_readback") },
            );
            readback_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            self.ctx.queue.submit(std::iter::once(readback_enc.finish()));

            // Map and read back the pixels.
            let buffer_slice = self.readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let pixel_count = (w * h) as usize;
                let mut rgba8 = vec![0u8; pixel_count * 4];
                let is_bgra = matches!(
                    self.surface_format,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                );

                for y in 0..h as usize {
                    let src_row_offset = y * padded_row as usize;
                    let dst_row_offset = y * w as usize * 4;
                    let row_bytes = w as usize * 4;
                    rgba8[dst_row_offset..dst_row_offset + row_bytes]
                        .copy_from_slice(&data[src_row_offset..src_row_offset + row_bytes]);
                }

                // Convert BGRA → RGBA if the surface format is BGRA.
                if is_bgra {
                    for pixel in rgba8.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                }

                drop(data);
                self.readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_pixels = rgba8;
                    state.frame_width = w;
                    state.frame_height = h;
                    state.screenshot_requested = false;
                }
            } else {
                self.readback_buffer.unmap();
                if let Ok(mut state) = self.shared_state.lock() {
                    state.screenshot_requested = false;
                }
            }
        }

        frame.present();

        // DEBUG: Mark frame completed (pair with pre-dispatch write above).
        {
            let diag = format!(
                "f={} en={} min={:.1} max={:.1} thr={:.4} dens={:.2} cam=[{:.1},{:.1},{:.1}] res={}x{} OK\n",
                self.frame_index,
                cloud_params_snapshot[0] > 0.5,
                cloud_params_snapshot[1],
                cloud_params_snapshot[2],
                cloud_params_snapshot[3],
                cloud_params_snapshot[4],
                cam_snapshot[0], cam_snapshot[1], cam_snapshot[2],
                self.render_width / 2, self.render_height / 2,
            );
            let _ = std::fs::write("/tmp/rkf-cloud-diag.txt", &diag);
        }

        self.frame_index += 1;
    }

    // -----------------------------------------------------------------------
    // Offscreen rendering (compositor path — rinch owns the window)
    // -----------------------------------------------------------------------

    /// Create an engine using a shared wgpu device (from rinch's `GpuHandle`).
    ///
    /// Renders to an offscreen texture instead of a swapchain surface. The
    /// compositor reads the offscreen texture directly via `TextureView` —
    /// zero-copy GPU compositing.
    ///
    /// Returns `(engine, scene)` — the caller stores the scene in EditorState.
    pub fn new_with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        viewport_width: u32,
        viewport_height: u32,
        shared_state: Arc<Mutex<SharedState>>,
    ) -> (Self, Scene) {
        let ctx = RenderContext::from_shared(device, queue);

        // Compute internal render resolution from viewport size.
        let vp_w = viewport_width.max(64);
        let vp_h = viewport_height.max(64);
        let internal_w = ((vp_w as f32 * RENDER_SCALE) as u32).max(64);
        let internal_h = ((vp_h as f32 * RENDER_SCALE) as u32).max(64);

        // Build demo scene.
        let demo = build_demo_scene();
        let scene = demo.scene;

        // Upload brick pool to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(demo.brick_pool.as_slice());
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: pool_data.len().max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !pool_data.is_empty() {
            ctx.queue.write_buffer(&brick_pool_buffer, 0, pool_data);
        }

        let mut gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);
        let brick_map_data = demo.brick_map_alloc.as_slice();
        if !brick_map_data.is_empty() {
            gpu_scene.upload_brick_maps(&ctx.device, &ctx.queue, brick_map_data);
        }

        let gbuffer = GBuffer::new(&ctx.device, internal_w, internal_h);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, internal_w, internal_h,
        );

        // Coarse acceleration field.
        let init_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let init_default_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = init_transforms.get(&obj.id).unwrap_or(&init_default_wt);
                let smin = obj.aabb.min * wt.scale;
                let smax = obj.aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z), Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z), Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z), Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z), Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                (wmin, wmax)
            })
            .collect();
        let mut coarse_field = CoarseField::from_scene_aabbs(
            &ctx.device, &scene_aabbs, COARSE_VOXEL_SIZE, 1.0,
        );
        coarse_field.populate(&scene_aabbs);
        coarse_field.upload(&ctx.queue, Vec3::ZERO);

        let ray_march = RayMarchPass::new(
            &ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field,
        );
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material table.
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights.
        let world_lights = vec![
            Light::point([2.0, 1.5, -1.0], [1.0, 0.8, 0.5], 5.0, 8.0, true),
            Light::point([-2.0, 1.0, -3.0], [0.5, 0.7, 1.0], 3.0, 6.0, false),
        ];
        let mut init_lights = Vec::with_capacity(64);
        init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        init_lights.extend(&world_lights);
        while init_lights.len() < 64 {
            init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
        }
        let light_buffer = LightBuffer::upload(&ctx.device, &init_lights);

        // GI radiance volume.
        let radiance_volume = RadianceVolume::new(&ctx.device);
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device, &gpu_scene, &material_table.buffer,
            &light_buffer, &radiance_volume, &coarse_field,
        );
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Shading pass.
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            internal_w, internal_h,
        );

        // Volumetric pipeline.
        let vol_shadow = VolShadowPass::new(
            &ctx.device, &ctx.queue, &coarse_field.bind_group_layout,
        );
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = internal_w / 2;
        let half_h = internal_h / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device, &ctx.queue,
            &gbuffer.position_view, &vol_shadow.shadow_view, &cloud_shadow.shadow_view,
            half_w, half_h, internal_w, internal_h,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device, &vol_march.output_view, &gbuffer.position_view,
            internal_w, internal_h, half_w, half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device, &shading_pass.hdr_view, &vol_upscale.output_view,
            internal_w, internal_h,
        );

        // Post-processing pipeline.
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
            internal_w, internal_h,
        );
        let bloom = BloomPass::new(
            &ctx.device, &god_rays_blur.output_view, internal_w, internal_h,
        );
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &god_rays_blur.output_view, internal_w, internal_h,
        );
        let dof = DofPass::new(
            &ctx.device, &god_rays_blur.output_view, &gbuffer.position_view,
            internal_w, internal_h,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device, &dof.output_view, &gbuffer.motion_view,
            internal_w, internal_h,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device, &motion_blur.output_view, bloom.mip_views(),
            internal_w, internal_h,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device, &bloom_composite.output_view,
            internal_w, internal_h,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device, &ctx.queue, &tone_map.ldr_view,
            internal_w, internal_h,
        );
        let cosmetics = CosmeticsPass::new(
            &ctx.device, &color_grade.output_view, internal_w, internal_h,
        );
        let sharpen = SharpenPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer,
            internal_w, internal_h,
        );

        // Offscreen render target at viewport resolution.
        let (offscreen_tex, offscreen_view) = Self::create_offscreen_target(
            &ctx.device, vp_w, vp_h,
        );

        // Blit from post-processed output (internal res) to offscreen target
        // (viewport res). The BlitPass uses bilinear sampling for upscale.
        let offscreen_blit = BlitPass::new(
            &ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT,
        );

        // Legacy blit (unused in offscreen path, but struct requires it).
        let blit = BlitPass::new(
            &ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT,
        );

        // Wireframe pass for selection highlights and gizmos (targets offscreen texture).
        let wireframe_pass = crate::wireframe::WireframePass::new(&ctx.device, OFFSCREEN_FORMAT);

        // Render camera.
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Pick readback buffer.
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback buffer (Rgba32Float position = 16 bytes, 256-aligned).
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Screenshot readback at viewport resolution.
        let readback_buffer = Self::create_readback_buffer(
            &ctx.device, vp_w, vp_h,
        );

        let jfa_sdf = crate::jfa_sdf::JfaSdfPass::new(&ctx.device);

        let engine = Self {
            ctx,
            surface: None,
            surface_format: OFFSCREEN_FORMAT,
            offscreen_texture: Some(offscreen_tex),
            offscreen_view: Some(offscreen_view),
            offscreen_blit: Some(offscreen_blit),
            viewport_width: vp_w,
            viewport_height: vp_h,
            gpu_scene,
            gbuffer,
            tile_cull,
            coarse_field,
            ray_march,
            debug_view,
            shading_pass,
            radiance_volume,
            radiance_inject,
            radiance_mip,
            vol_shadow,
            cloud_shadow,
            vol_march,
            vol_upscale,
            vol_composite,
            god_rays_blur,
            bloom,
            auto_exposure,
            dof,
            motion_blur,
            bloom_composite,
            tone_map,
            color_grade,
            cosmetics,
            sharpen,
            blit,
            camera,
            world_lights,
            light_buffer,
            material_buffer: material_table.buffer,
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            env_sun_dir: {
                let d = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
                [d.x, d.y, d.z]
            },
            env_sun_color: [3.0, 2.85, 2.55],
            env_fog_color: [0.7, 0.75, 0.8],
            env_fog_density: 0.0,
            env_fog_height_falloff: 0.1,
            env_sun_intensity: 3.0,
            env_sun_color_raw: [1.0, 0.95, 0.85],
            env_rayleigh_scale: 1.0,
            env_mie_scale: 1.0,
            env_atmosphere_enabled: true,
            env_ambient_dust: 0.005,
            env_dust_g: 0.3,
            env_cloud_settings: rkf_render::CloudSettings::default(),
            accumulated_time: 0.0,
            render_width: internal_w,
            render_height: internal_h,
            pick_readback_buffer,
            brush_readback_buffer,
            readback_buffer,
            window_width: vp_w,
            window_height: vp_h,
            shared_state,
            wireframe_pass: Some(wireframe_pass),
            character: Some(demo.character),
            character_obj_index: Some(demo.character_obj_index),
            last_frame_time: Instant::now(),
            cpu_brick_pool: demo.brick_pool,
            cpu_brick_map_alloc: demo.brick_map_alloc,
            jfa_sdf,
        };
        (engine, scene)
    }

    /// Create the offscreen render target texture at the given resolution.
    fn create_offscreen_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// Get the offscreen texture view for compositor integration.
    ///
    /// Returns `None` if the engine is using the surface-based path.
    /// The compositor calls `set_texture_source()` with this view.
    pub fn offscreen_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.offscreen_view.as_ref()
    }

    /// Current viewport dimensions (physical pixels).
    pub fn viewport_size(&self) -> (u32, u32) {
        (self.viewport_width, self.viewport_height)
    }

    /// Resize the viewport and recreate resolution-dependent resources.
    ///
    /// Called when the `RenderSurface` layout changes (e.g., panel resize).
    /// Recreates the offscreen texture, internal render resources, and
    /// readback buffer. Returns the new offscreen `TextureView` for the
    /// compositor to call `set_texture_source()`.
    pub fn resize_viewport(&mut self, viewport_w: u32, viewport_h: u32) -> Option<wgpu::TextureView> {
        let vp_w = viewport_w.max(64);
        let vp_h = viewport_h.max(64);
        if vp_w == self.viewport_width && vp_h == self.viewport_height {
            return None;
        }

        log::info!("Viewport resize: {}x{} → {}x{}", self.viewport_width, self.viewport_height, vp_w, vp_h);
        self.viewport_width = vp_w;
        self.viewport_height = vp_h;

        // Recreate offscreen target at new viewport resolution.
        let (tex, view) = Self::create_offscreen_target(&self.ctx.device, vp_w, vp_h);
        self.offscreen_texture = Some(tex);
        self.offscreen_view = Some(view);

        // Recreate readback buffer at new viewport resolution.
        self.readback_buffer = Self::create_readback_buffer(&self.ctx.device, vp_w, vp_h);
        self.window_width = vp_w;
        self.window_height = vp_h;

        // Resize internal render resolution.
        let internal_w = ((vp_w as f32 * RENDER_SCALE) as u32).max(64);
        let internal_h = ((vp_h as f32 * RENDER_SCALE) as u32).max(64);
        self.resize_render(internal_w, internal_h);

        // Update offscreen blit source (cosmetics output may have changed).
        if let Some(ref mut offscreen_blit) = self.offscreen_blit {
            offscreen_blit.update_source(&self.ctx.device, &self.cosmetics.output_view);
        }

        // Update SharedState.
        if let Ok(mut state) = self.shared_state.lock() {
            state.frame_width = vp_w;
            state.frame_height = vp_h;
        }

        // Return new view for set_texture_source().
        self.offscreen_view.clone()
    }

    /// Render one frame to the offscreen texture (compositor path).
    ///
    /// Runs the full compute pipeline at internal resolution, then blits to the
    /// offscreen render target at viewport resolution. The compositor reads
    /// the offscreen texture directly — no CPU readback needed per frame.
    ///
    /// Screenshot readback only happens when `SharedState.screenshot_requested`
    /// is true.
    pub fn render_frame_offscreen(&mut self, scene: &Scene) {
        let camera_pos_vec = self.camera.position;

        // Bake world transforms for parent-child hierarchy.
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        // Flatten all objects → GPU object list + BVH.
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        let mut world_aabbs_for_coarse: Vec<(Vec3, Vec3)> = Vec::new();
        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos_vec;
            let local_aabb = obj.aabb;
            let world_aabb = {
                let smin = local_aabb.min * wt.scale;
                let smax = local_aabb.max * wt.scale;
                let corners = [
                    Vec3::new(smin.x, smin.y, smin.z), Vec3::new(smax.x, smin.y, smin.z),
                    Vec3::new(smin.x, smax.y, smin.z), Vec3::new(smax.x, smax.y, smin.z),
                    Vec3::new(smin.x, smin.y, smax.z), Vec3::new(smax.x, smin.y, smax.z),
                    Vec3::new(smin.x, smax.y, smax.z), Vec3::new(smax.x, smax.y, smax.z),
                ];
                let mut wmin = Vec3::splat(f32::MAX);
                let mut wmax = Vec3::splat(f32::MIN);
                for c in &corners {
                    let r = wt.rotation * *c + wt.position;
                    wmin = wmin.min(r);
                    wmax = wmax.max(r);
                }
                Aabb::new(wmin, wmax)
            };
            world_aabbs_for_coarse.push((world_aabb.min, world_aabb.max));
            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - self.camera.position;
                let cam_rel_max = world_aabb.max - self.camera.position;
                let (geom_min, geom_max) = self.compute_geometry_aabb_for_flat_node(flat);
                gpu_objects.push(GpuObject::from_flat_node(
                    flat, obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                    geom_min,
                    geom_max,
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        self.gpu_scene.upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);
        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        self.coarse_field.populate(&world_aabbs_for_coarse);
        self.coarse_field.upload(&self.ctx.queue, Vec3::ZERO);

        // Camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Synthesize directional light from environment sun.
        let sun_light = Light {
            light_type: 0,
            pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: self.env_sun_dir[0],
            dir_y: self.env_sun_dir[1],
            dir_z: self.env_sun_dir[2],
            color_r: self.env_sun_color[0],
            color_g: self.env_sun_color[1],
            color_b: self.env_sun_color[2],
            intensity: 1.0,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: 1,
        };

        let cam = self.camera.position;
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            if cl.light_type != 0 {
                cl.pos_x -= cam.x;
                cl.pos_y -= cam.y;
                cl.pos_z -= cam.z;
            }
            cl
        }).collect();

        let total_lights = 1 + cam_rel_lights.len() as u32;
        let mut all_lights = vec![sun_light];
        all_lights.extend(cam_rel_lights);
        self.light_buffer.update(&self.ctx.queue, &all_lights);

        // Shade uniforms.
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = self.render_width as f32 / self.render_height as f32;
        let fwd = self.camera.forward();
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;

        self.shading_pass.update_uniforms(&self.ctx.queue, &ShadeUniforms {
            debug_mode: self.shade_debug_mode,
            num_lights: total_lights,
            _pad0: 0,
            shadow_budget_k: 0,
            camera_pos: [camera_pos_vec.x, camera_pos_vec.y, camera_pos_vec.z, 0.0],
            sun_dir: [self.env_sun_dir[0], self.env_sun_dir[1], self.env_sun_dir[2], self.env_sun_intensity],
            sun_color: [self.env_sun_color_raw[0], self.env_sun_color_raw[1], self.env_sun_color_raw[2], 0.0],
            sky_params: [self.env_rayleigh_scale, self.env_mie_scale, if self.env_atmosphere_enabled { 1.0 } else { 0.0 }, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
        });

        let debug_mode = match self.shade_debug_mode {
            0 => DebugMode::Lambert,
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, debug_mode);

        self.prev_vp = self.camera.view_projection(self.render_width, self.render_height)
            .to_cols_array_2d();

        // Poll GPU to prevent command buffer buildup.
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_frame"),
        });

        // Per-frame uniforms.
        self.coarse_field.update_uniforms(&self.ctx.queue, self.camera.position);
        self.radiance_volume.update_center(
            &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
        );
        self.radiance_inject.update_uniforms(&self.ctx.queue, &InjectUniforms {
            num_lights: total_lights,
            max_shadow_lights: 1,
            _pad: [0; 2],
        });

        // --- Core rendering ---
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        self.radiance_inject.dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.radiance_mip.dispatch(&mut encoder);
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );

        // --- Volumetric pipeline ---
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

        self.vol_march.set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow.set_cloud_params(&self.ctx.queue, &cloud_params);

        let sun_dir = self.env_sun_dir;
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
        );
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);

        let sc = self.env_sun_color;
        let fc = self.env_fog_color;
        let fog_alpha = if self.env_fog_density > 0.0 { 1.0 } else { 0.0 };
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [sc[0], sc[1], sc[2], 0.0],
            width: self.render_width / 2,
            height: self.render_height / 2,
            full_width: self.render_width,
            full_height: self.render_height,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [fc[0], fc[1], fc[2], fog_alpha],
            fog_height: [self.env_fog_density, -0.5, self.env_fog_height_falloff, 0.0],
            fog_distance: [0.0, 0.01, self.env_ambient_dust, self.env_dust_g],
            frame_index: self.frame_index,
            _pad0: 0, _pad1: 0, _pad2: 0,
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // --- Post-processing pipeline ---
        {
            let sun_dir_v = glam::Vec3::from(self.env_sun_dir).normalize_or_zero();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir_v.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir_v.dot(right) / sun_dot;
                let ndc_y = -sun_dir_v.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur.update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // --- Blit to offscreen render target ---
        let offscreen_view = self.offscreen_view.as_ref()
            .expect("offscreen_view must exist in offscreen mode");
        if let Some(ref offscreen_blit) = self.offscreen_blit {
            offscreen_blit.draw(&mut encoder, offscreen_view);
        }

        // Submit all GPU work. The compositor reads the offscreen texture
        // on the next paint — no readback needed per frame.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // --- GPU pick readback (same as surface path) ---
        let pending_pick = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_pick.take());

        if let Some((px, py)) = pending_pick {
            let px = px.min(self.render_width.saturating_sub(1));
            let py = py.min(self.render_height.saturating_sub(1));

            let mut pick_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("pick_readback") },
            );
            pick_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(pick_enc.finish()));

            let slice = self.pick_readback_buffer.slice(..4);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                let object_id = packed >> 24;
                drop(data);
                self.pick_readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.pick_result = Some(object_id);
                }
            } else {
                self.pick_readback_buffer.unmap();
            }
        }

        // --- GPU brush hit readback (position + object_id from G-buffer) ---
        let pending_brush = self.shared_state.lock()
            .ok()
            .and_then(|mut s| s.pending_brush_hit.take());

        if let Some((bx, by)) = pending_brush {
            let bx = bx.min(self.render_width.saturating_sub(1));
            let by = by.min(self.render_height.saturating_sub(1));

            let mut brush_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("brush_readback") },
            );

            // Copy 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.position_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.brush_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            // Also copy 1 pixel from material G-buffer for object_id (reuse pick buffer).
            brush_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.gbuffer.material_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.pick_readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );

            self.ctx.queue.submit(std::iter::once(brush_enc.finish()));

            // Read position (4×f32 = 16 bytes).
            let pos_slice = self.brush_readback_buffer.slice(..16);
            let (tx_pos, rx_pos) = std::sync::mpsc::channel();
            pos_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_pos.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut hit_pos = [0.0f32; 4];
            let pos_ok = if let Ok(Ok(())) = rx_pos.recv() {
                let data = pos_slice.get_mapped_range();
                hit_pos = [
                    f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                    f32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                    f32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                    f32::from_le_bytes([data[12], data[13], data[14], data[15]]),
                ];
                drop(data);
                self.brush_readback_buffer.unmap();
                true
            } else {
                self.brush_readback_buffer.unmap();
                false
            };

            // Read object_id from material G-buffer (bits 24-31).
            let mat_slice = self.pick_readback_buffer.slice(..4);
            let (tx_mat, rx_mat) = std::sync::mpsc::channel();
            mat_slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_mat.send(r); });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            let mut object_id = 0u32;
            if let Ok(Ok(())) = rx_mat.recv() {
                let data = mat_slice.get_mapped_range();
                let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                object_id = packed >> 24;
                drop(data);
                self.pick_readback_buffer.unmap();
            } else {
                self.pick_readback_buffer.unmap();
            }

            // Only produce a hit result if the ray actually hit geometry (not sky).
            if pos_ok && hit_pos[3] < 1e30 {
                let result = crate::automation::BrushHitResult {
                    position: Vec3::new(hit_pos[0], hit_pos[1], hit_pos[2]),
                    object_id,
                };
                if let Ok(mut state) = self.shared_state.lock() {
                    state.brush_hit_result = Some(result);
                }
            }
        }

        // --- Screenshot readback (on demand only) ---
        let do_readback = self.shared_state.lock()
            .map(|s| s.screenshot_requested)
            .unwrap_or(false);

        if do_readback {
            let w = self.viewport_width;
            let h = self.viewport_height;
            let bytes_per_pixel = 4u32;
            let unpadded_row = w * bytes_per_pixel;
            let padded_row = (unpadded_row + 255) & !255;

            let offscreen_tex = self.offscreen_texture.as_ref()
                .expect("offscreen_texture must exist");
            let mut readback_enc = self.ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("screenshot_readback") },
            );
            readback_enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: offscreen_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.ctx.queue.submit(std::iter::once(readback_enc.finish()));

            let buffer_slice = self.readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let pixel_count = (w * h) as usize;
                let mut rgba8 = vec![0u8; pixel_count * 4];

                for y in 0..h as usize {
                    let src_row_offset = y * padded_row as usize;
                    let dst_row_offset = y * w as usize * 4;
                    let row_bytes = w as usize * 4;
                    rgba8[dst_row_offset..dst_row_offset + row_bytes]
                        .copy_from_slice(&data[src_row_offset..src_row_offset + row_bytes]);
                }

                // Offscreen format is RGBA — no BGRA swap needed.
                drop(data);
                self.readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_pixels = rgba8;
                    state.frame_width = w;
                    state.frame_height = h;
                    state.screenshot_requested = false;
                }
            } else {
                self.readback_buffer.unmap();
                if let Ok(mut state) = self.shared_state.lock() {
                    state.screenshot_requested = false;
                }
            }
        }

        self.frame_index += 1;
    }

    /// Sample a 2D XZ slice of SDF distances from an object's voxel brick data.
    ///
    /// Walks the object-local XZ extent at the given Y coordinate, sampling
    /// the CPU brick pool at each voxel center. Returns a `VoxelSliceResult`
    /// with the distance grid and per-sample slot status.
    pub fn sample_voxel_slice(
        &self,
        scene: &rkf_core::Scene,
        object_id: u32,
        y_coord: f32,
    ) -> Result<rkf_core::automation::VoxelSliceResult, String> {
        use rkf_core::automation::VoxelSliceResult;
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
        use rkf_core::SdfSource;

        // Find the object in the scene.
        let obj = scene.objects.iter()
            .find(|o| o.id == object_id)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        // Must be voxelized.
        let (handle, voxel_size, _aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (brick_map_handle, *voxel_size, aabb)
            }
            _ => return Err(format!("object {object_id} is not voxelized")),
        };

        let dims = handle.dims;
        let brick_extent = voxel_size * 8.0;
        let grid_size_x = dims.x as f32 * brick_extent;
        let grid_size_z = dims.z as f32 * brick_extent;
        let x_min = -(grid_size_x * 0.5);
        let z_min = -(grid_size_z * 0.5);

        let total_voxels_x = dims.x * 8;
        let total_voxels_z = dims.z * 8;

        let mut distances = Vec::with_capacity((total_voxels_x * total_voxels_z) as usize);
        let mut slot_status = Vec::with_capacity((total_voxels_x * total_voxels_z) as usize);

        // Sample at voxel centers across the XZ extent.
        for vz in 0..total_voxels_z {
            for vx in 0..total_voxels_x {
                let _local_x = x_min + (vx as f32 + 0.5) * voxel_size;
                let _local_z = z_min + (vz as f32 + 0.5) * voxel_size;

                // Determine brick and local voxel coordinates.
                let bx = vx / 8;
                let bz = vz / 8;
                // Compute brick Y from y_coord.
                let grid_y = y_coord + (dims.y as f32 * brick_extent * 0.5);
                let vy_f = grid_y / voxel_size;
                let by = (vy_f / 8.0).floor() as u32;
                let ly = ((vy_f as u32) % 8).min(7);
                let lx = vx % 8;
                let lz = vz % 8;

                if by >= dims.y {
                    distances.push(f32::MAX);
                    slot_status.push(0); // EMPTY
                    continue;
                }

                match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                    Some(slot) if slot == EMPTY_SLOT => {
                        distances.push(voxel_size * 2.0);
                        slot_status.push(0);
                    }
                    Some(slot) if slot == INTERIOR_SLOT => {
                        distances.push(-(voxel_size * 2.0));
                        slot_status.push(1);
                    }
                    Some(slot) => {
                        let sample = self.cpu_brick_pool.get(slot).sample(lx, ly, lz);
                        distances.push(sample.distance_f32());
                        slot_status.push(2);
                    }
                    None => {
                        distances.push(f32::MAX);
                        slot_status.push(0);
                    }
                }
            }
        }

        Ok(VoxelSliceResult {
            origin: [x_min, z_min],
            spacing: voxel_size,
            width: total_voxels_x,
            height: total_voxels_z,
            y_coord,
            distances,
            slot_status,
        })
    }

    /// Sample the SDF distance at a world-space position by iterating scene objects.
    ///
    /// Transforms the query position into each object's local space and evaluates
    /// the SDF. Returns the closest result across all objects.
    pub fn sample_spatial_query(
        &self,
        scene: &rkf_core::Scene,
        world_pos: Vec3,
    ) -> rkf_core::automation::SpatialQueryResult {
        use rkf_core::automation::SpatialQueryResult;
        use rkf_core::SdfSource;

        let mut best_dist = f32::MAX;
        let mut best_mat: u16 = 0;

        for obj in &scene.objects {
            // Transform world_pos to object-local space.
            let inv = glam::Mat4::from_scale_rotation_translation(
                obj.scale, obj.rotation, obj.position,
            ).inverse();
            let local_pos = inv.transform_point3(world_pos);

            let dist = match &obj.root_node.sdf_source {
                SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
                    let exterior = Self::classify_exterior_bricks(
                        &self.cpu_brick_map_alloc, brick_map_handle,
                    );
                    Self::sample_sdf_cpu(
                        &self.cpu_brick_pool,
                        &self.cpu_brick_map_alloc,
                        brick_map_handle,
                        *voxel_size,
                        local_pos,
                        &exterior,
                    )
                }
                SdfSource::Analytical { primitive, .. } => {
                    rkf_core::evaluate_primitive(primitive, local_pos)
                }
                SdfSource::None => continue,
            };

            // Apply conservative scale correction.
            let scale_min = obj.scale.min_element();
            let scaled_dist = dist * scale_min;

            if scaled_dist.abs() < best_dist.abs() {
                best_dist = scaled_dist;
                best_mat = match &obj.root_node.sdf_source {
                    SdfSource::Analytical { material_id, .. } => *material_id,
                    _ => 0,
                };
            }
        }

        SpatialQueryResult {
            distance: best_dist,
            material_id: best_mat,
            inside: best_dist < 0.0,
        }
    }

    /// Sample a compact brick-level 3D shape overview of an object.
    ///
    /// Each brick is categorized as empty (`.`), interior (`#`), or
    /// surface/allocated (`+`). Returns per-Y-level ASCII slices.
    pub fn sample_object_shape(
        &self,
        scene: &rkf_core::Scene,
        object_id: u32,
    ) -> Result<rkf_core::automation::ObjectShapeResult, String> {
        use rkf_core::automation::ObjectShapeResult;
        use rkf_core::brick_map::{EMPTY_SLOT, INTERIOR_SLOT};
        use rkf_core::SdfSource;

        let obj = scene.objects.iter()
            .find(|o| o.id == object_id)
            .ok_or_else(|| format!("object {object_id} not found"))?;

        let (handle, voxel_size, aabb) = match &obj.root_node.sdf_source {
            SdfSource::Voxelized { brick_map_handle, voxel_size, aabb } => {
                (brick_map_handle, *voxel_size, aabb)
            }
            _ => return Err(format!("object {object_id} is not voxelized")),
        };

        let dims = handle.dims;
        let brick_extent = voxel_size * 8.0;
        let _half_x = dims.x as f32 * brick_extent * 0.5;
        let _half_y = dims.y as f32 * brick_extent * 0.5;
        let _half_z = dims.z as f32 * brick_extent * 0.5;

        let mut empty_count = 0u32;
        let mut interior_count = 0u32;
        let mut surface_count = 0u32;
        let mut y_slices = Vec::with_capacity(dims.y as usize);

        for by in 0..dims.y {
            let mut slice = String::new();
            for bz in 0..dims.z {
                for bx in 0..dims.x {
                    let ch = match self.cpu_brick_map_alloc.get_entry(handle, bx, by, bz) {
                        Some(slot) if slot == EMPTY_SLOT => {
                            empty_count += 1;
                            '.'
                        }
                        Some(slot) if slot == INTERIOR_SLOT => {
                            interior_count += 1;
                            '#'
                        }
                        Some(_) => {
                            surface_count += 1;
                            '+'
                        }
                        None => {
                            empty_count += 1;
                            '.'
                        }
                    };
                    slice.push(ch);
                }
                if bz + 1 < dims.z {
                    slice.push('\n');
                }
            }
            y_slices.push(slice);
        }

        Ok(ObjectShapeResult {
            object_id,
            dims: [dims.x, dims.y, dims.z],
            voxel_size,
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z],
            empty_count,
            interior_count,
            surface_count,
            y_slices,
        })
    }
}
