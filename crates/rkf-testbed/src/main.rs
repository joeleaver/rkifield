//! RKIField visual testbed — v2 object-centric render loop.
//!
//! Scene: 5 analytical + 1 voxelized SDF objects rendered via the v2
//! object-centric ray marcher with BVH acceleration. Debug visualization
//! shows Lambert-shaded normals by default.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use glam::{Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

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

mod automation;
use automation::{SharedState, TestbedAutomationApi};

/// Display (output) resolution width — the window size.
const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height — the window size.
const DISPLAY_HEIGHT: u32 = 720;

/// Internal render resolution width.
const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height.
const INTERNAL_HEIGHT: u32 = 540;

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

/// Build result containing the scene and CPU brick data for GPU upload.
struct DemoScene {
    scene: Scene,
    brick_pool: BrickPool,
    brick_map_alloc: BrickMapAllocator,
    /// Animated character (stored separately for per-frame pose updates).
    character: AnimatedCharacter,
    /// Index of the character's SceneObject in scene.objects.
    character_obj_index: usize,
}

/// Build the demo scene with 5 analytical + 1 voxelized object.
fn build_demo_scene() -> DemoScene {
    let mut scene = Scene::new("v2_testbed");

    // 1. Ground plane (large flat box)
    let ground = SceneNode::analytical("ground", SdfPrimitive::Box {
        half_extents: Vec3::new(10.0, 0.1, 10.0),
    }, 1);
    let ground_obj = SceneObject {
        id: 0,
        name: "ground".into(),
        parent_id: None,
        position: Vec3::new(0.0, -0.6, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: ground,
        aabb: Aabb::new(Vec3::new(-10.0, -0.1, -10.0), Vec3::new(10.0, 0.1, 10.0)),
    };
    scene.add_object_full(ground_obj);

    // 2. Sphere (analytical)
    let sphere = SceneNode::analytical("sphere", SdfPrimitive::Sphere { radius: 0.5 }, 2);
    let sphere_obj = SceneObject {
        id: 0,
        name: "sphere".into(),
        parent_id: None,
        position: Vec3::new(-1.5, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: sphere,
        aabb: Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
    };
    scene.add_object_full(sphere_obj);

    // 3. Box (analytical)
    let box_node = SceneNode::analytical("box", SdfPrimitive::Box {
        half_extents: Vec3::new(0.35, 0.35, 0.35),
    }, 3);
    let box_obj = SceneObject {
        id: 0,
        name: "box".into(),
        parent_id: None,
        position: Vec3::new(0.0, 0.0, -2.0),
        rotation: Quat::from_rotation_y(0.5),
        scale: Vec3::ONE,
        root_node: box_node,
        aabb: Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
    };
    scene.add_object_full(box_obj);

    // 4. Capsule (analytical)
    let capsule = SceneNode::analytical("capsule", SdfPrimitive::Capsule {
        radius: 0.2,
        half_height: 0.4,
    }, 4);
    let capsule_obj = SceneObject {
        id: 0,
        name: "capsule".into(),
        parent_id: None,
        position: Vec3::new(1.5, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: capsule,
        aabb: Aabb::new(Vec3::new(-0.5, -0.6, -0.5), Vec3::new(0.5, 0.6, 0.5)),
    };
    scene.add_object_full(capsule_obj);

    // 5. Torus (analytical)
    let torus = SceneNode::analytical("torus", SdfPrimitive::Torus {
        major_radius: 0.4,
        minor_radius: 0.12,
    }, 5);
    let torus_obj = SceneObject {
        id: 0,
        name: "torus".into(),
        parent_id: None,
        position: Vec3::new(0.0, 0.3, -3.5),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: torus,
        aabb: Aabb::new(Vec3::new(-0.6, -0.5, -0.6), Vec3::new(0.6, 0.5, 0.6)),
    };
    scene.add_object_full(torus_obj);

    // 6. Voxelized sphere — demonstrates voxelized SDF rendering path.
    let mut brick_pool = BrickPool::new(4096);
    let mut brick_map_alloc = BrickMapAllocator::new();

    let vox_radius = 0.4;
    let voxel_size = 0.04; // 4cm voxels
    let margin = voxel_size * 2.0;
    let vox_aabb = Aabb::new(
        Vec3::splat(-vox_radius - margin),
        Vec3::splat(vox_radius + margin),
    );

    let sdf_fn = |pos: Vec3| -> (f32, u16) {
        (pos.length() - vox_radius, 6u16) // material 6 = distinct color
    };

    let (handle, brick_count) = voxelize_sdf(
        sdf_fn, &vox_aabb, voxel_size, &mut brick_pool, &mut brick_map_alloc,
    ).expect("voxelize sphere");

    log::info!(
        "Voxelized sphere: {} bricks, handle offset={} dims={:?}",
        brick_count, handle.offset, handle.dims
    );

    let mut vox_node = SceneNode::new("vox_sphere");
    vox_node.sdf_source = SdfSource::Voxelized {
        brick_map_handle: handle,
        voxel_size,
        aabb: vox_aabb,
    };

    // Place the voxelized sphere to the right of the analytical objects.
    let vox_obj = SceneObject {
        id: 0,
        name: "vox_sphere".into(),
        parent_id: None,
        position: Vec3::new(3.0, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        root_node: vox_node,
        aabb: Aabb::new(
            Vec3::new(-vox_radius - margin, -vox_radius - margin, -vox_radius - margin),
            Vec3::new(vox_radius + margin, vox_radius + margin, vox_radius + margin),
        ),
    };
    scene.add_object_full(vox_obj);

    // 7. Animated humanoid character (14-bone skeleton as SceneNode tree).
    let skeleton = build_humanoid_skeleton();
    let visuals = build_humanoid_visuals(5); // material 5 = skin-like
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

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

struct EngineState {
    ctx: RenderContext,
    surface: wgpu::Surface<'static>,
    #[allow(dead_code)]
    surface_format: wgpu::TextureFormat,
    gpu_scene: GpuSceneV2,
    gbuffer: GBuffer,
    tile_cull: TileObjectCullPass,
    coarse_field: CoarseField,
    ray_march: RayMarchPass,
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
    camera: Camera,
    scene: Scene,
    /// World-space lights (converted to camera-relative each frame).
    world_lights: Vec<Light>,
    light_buffer: LightBuffer,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    debug_mode: DebugMode,
    /// Raw shade debug mode (u32) — supports modes 0-6 for the shade shader.
    shade_debug_mode: u32,
    /// Staging buffer for GPU readback (Rgba8Unorm → RGBA8 for MCP screenshots).
    readback_buffer: wgpu::Buffer,
    /// Shared state for MCP observation.
    shared_state: Arc<Mutex<SharedState>>,
    /// Animated character (updated each frame).
    character: AnimatedCharacter,
    /// Index of the character's SceneObject in scene.objects.
    character_obj_index: usize,
    /// Last frame time for animation advancement.
    last_frame_time: Instant,
}

impl EngineState {
    fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let ctx = RenderContext::new(&instance, &surface);

        let size = window.inner_size();
        let surface_format = ctx.configure_surface(&surface, size.width, size.height);

        // Build demo scene (includes voxelized objects).
        let demo = build_demo_scene();
        let scene = demo.scene;

        // Upload brick pool to GPU (array of VoxelSample, 8 bytes each).
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

        // Upload brick maps to GPU.
        let brick_map_data = demo.brick_map_alloc.as_slice();
        if !brick_map_data.is_empty() {
            gpu_scene.upload_brick_maps(&ctx.device, &ctx.queue, brick_map_data);
        }

        let gbuffer = GBuffer::new(&ctx.device, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Build coarse acceleration field from world-space AABBs.
        let init_wts = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let init_def_wt = rkf_core::transform_bake::WorldTransform::default();
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.objects.iter()
            .map(|obj| {
                let wt = init_wts.get(&obj.id).unwrap_or(&init_def_wt);
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

        // Material table — 14 test materials with distinct PBR properties.
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);
        log::info!("Material table: {} materials uploaded", material_table.count);

        // Lights — 1 directional (sun) + 2 point lights.
        let world_lights = vec![
            // Sun: warm white directional, from above-right-front, shadow casting
            Light::directional(
                [0.5, 1.0, 0.3],
                [1.0, 0.95, 0.85],
                3.0,
                true,
            ),
            // Warm point light (right side)
            Light::point(
                [2.0, 1.5, -1.0],
                [1.0, 0.8, 0.5],
                5.0,
                8.0,
                true,
            ),
            // Cool point light (left-back)
            Light::point(
                [-2.0, 1.0, -3.0],
                [0.5, 0.7, 1.0],
                3.0,
                6.0,
                false,
            ),
        ];
        let light_buffer = LightBuffer::upload(&ctx.device, &world_lights);
        log::info!("Lights: {} lights uploaded", world_lights.len());

        // Radiance volume — 4-level clipmap for voxel cone tracing GI.
        let radiance_volume = RadianceVolume::new(&ctx.device);
        log::info!("Radiance volume: 4 levels × 128³ Rgba16Float");

        // Radiance injection — fills L0 with direct-lit radiance at surface voxels.
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device, &gpu_scene, &material_table.buffer,
            &light_buffer, &radiance_volume, &coarse_field,
        );

        // Radiance mip generation — downsamples L0 → L1 → L2 → L3.
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Shading pass — full PBR with SDF shadows, AO, and GI cone tracing.
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // --- Volumetric pipeline ---

        // Volumetric shadow map (coarse field SDF density → 3D transmittance).
        let vol_shadow = VolShadowPass::new(
            &ctx.device, &ctx.queue, &coarse_field.bind_group_layout,
        );
        log::info!("Vol shadow map: 256×128×256 R32Float");

        // Cloud shadow map (analytical FBM clouds → 2D transmittance).
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        log::info!("Cloud shadow map: 1024×1024 R32Float");

        // Volumetric march (half-res fog/dust/clouds with shadow lookups).
        let half_w = INTERNAL_WIDTH / 2;
        let half_h = INTERNAL_HEIGHT / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device, &ctx.queue,
            &gbuffer.position_view,
            &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w, half_h, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        log::info!("Vol march: {}×{} half-res", half_w, half_h);

        // Bilateral upscale (half-res scatter → full-res).
        let vol_upscale = VolUpscalePass::new(
            &ctx.device, &vol_march.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT, half_w, half_h,
        );

        // Volumetric composite (shade HDR + upscaled scatter → composited HDR).
        let vol_composite = VolCompositePass::new(
            &ctx.device, &shading_pass.hdr_view, &vol_upscale.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        log::info!("Volumetric pipeline wired: vol_shadow → cloud_shadow → vol_march → vol_upscale → vol_composite");

        // --- Post-processing pipeline ---

        // Screen-space radial blur god rays (composited HDR → HDR + light shafts).
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Bloom extraction + downsample (from god rays HDR → 4-level mip pyramid).
        let bloom = BloomPass::new(
            &ctx.device, &god_rays_blur.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Auto-exposure (histogram from god rays HDR → adapted EV).
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &god_rays_blur.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Depth of field (god rays HDR + depth → blurred HDR).
        let dof = DofPass::new(
            &ctx.device, &god_rays_blur.output_view, &gbuffer.position_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Motion blur (DoF output + motion vectors → motion-blurred HDR).
        let motion_blur = MotionBlurPass::new(
            &ctx.device, &dof.output_view, &gbuffer.motion_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Bloom composite (motion-blurred HDR + bloom mips → HDR with bloom glow).
        let bloom_composite = BloomCompositePass::new(
            &ctx.device, &motion_blur.output_view, bloom.mip_views(),
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Tone mapping: HDR → LDR (with auto-exposure integration).
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device, &bloom_composite.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
            Some(auto_exposure.get_exposure_buffer()),
        );

        // Color grading (LUT-based color correction).
        let color_grade = ColorGradePass::new(
            &ctx.device, &ctx.queue, &tone_map.ldr_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Cosmetics (vignette, film grain, chromatic aberration).
        let cosmetics = CosmeticsPass::new(
            &ctx.device, &color_grade.output_view,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );

        // Edge-aware sharpen (applied in HDR space before tone map, optional).
        // For now, sharpen reads from vol_composite and is not in the main chain.
        let sharpen = SharpenPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer,
            INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        log::info!("Post-processing pipeline wired: god_rays_blur → bloom → dof → motion_blur → bloom_composite → tone_map → color_grade → cosmetics");

        // Blit: final LDR → swapchain.
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, surface_format);

        let mut camera = Camera::new(Vec3::new(0.0, 1.0, -0.5));
        camera.pitch = -0.15;
        camera.move_speed = 3.0;

        // Readback buffer for MCP screenshots.
        // Rgba8Unorm = 4 bytes per pixel, row must be aligned to 256 bytes.
        let bytes_per_pixel = 4u32;
        let unpadded_row = INTERNAL_WIDTH * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255; // align to 256
        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * INTERNAL_HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            surface,
            surface_format,
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
            scene,
            world_lights,
            light_buffer,
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            debug_mode: DebugMode::Lambert,
            shade_debug_mode: 0,
            readback_buffer,
            shared_state,
            character: demo.character,
            character_obj_index: demo.character_obj_index,
            last_frame_time: Instant::now(),
        }
    }

    fn render(&mut self) {
        // Advance character animation.
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1); // cap at 100ms
        self.last_frame_time = now;
        self.character.advance_and_update(
            dt,
            &mut self.scene.objects[self.character_obj_index].root_node,
        );
        let camera_pos_vec = self.camera.position;

        // Bake world transforms for parent-child hierarchy.
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&self.scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        // Flatten all objects and build GPU object list + BVH pairs.
        // BVH leaf indices must match the GPU objects[] array index (0-based),
        // NOT the scene object IDs (which start at 1).
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        for obj in &self.scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos_vec;
            // Transform local AABB to world space via baked transform.
            let world_aabb = {
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
                Aabb::new(wmin, wmax)
            };
            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - self.camera.position;
                let cam_rel_max = world_aabb.max - self.camera.position;
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        // Upload objects.
        self.gpu_scene.upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);

        // Build BVH using GPU array indices (not scene IDs).
        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Update camera.
        let cam_uniforms = self.camera.uniforms(
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
            self.frame_index,
            self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        // Update scene uniforms.
        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Update shade uniforms (debug_mode, lights, camera position).
        // G-buffer positions are camera-relative, so camera_pos = [0,0,0].
        let fwd = self.camera.forward();
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = INTERNAL_WIDTH as f32 / INTERNAL_HEIGHT as f32;
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;
        // Default sun direction for testbed (matches legacy directional light).
        let sun_dir = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
        self.shading_pass.update_uniforms(&self.ctx.queue, &ShadeUniforms {
            debug_mode: self.shade_debug_mode,
            num_lights: self.world_lights.len() as u32,
            _pad0: 0,
            shadow_budget_k: 0, // unlimited
            camera_pos: [0.0, 0.0, 0.0, 0.0], // camera-relative origin
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 3.0],
            sun_color: [1.0, 0.95, 0.85, 0.0],
            sky_params: [1.0, 1.0, 1.0, 0.0], // atmosphere enabled with default Rayleigh/Mie
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
        });

        // Upload camera-relative light positions each frame.
        let cam = self.camera.position;
        let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
            let mut cl = *l;
            // Only point/spot lights have meaningful positions.
            if cl.light_type != 0 { // not directional
                cl.pos_x -= cam.x;
                cl.pos_y -= cam.y;
                cl.pos_z -= cam.z;
            }
            cl
        }).collect();
        self.light_buffer.update(&self.ctx.queue, &cam_rel_lights);

        // Update debug view (kept for fallback).
        self.debug_view.set_mode(&self.ctx.queue, self.debug_mode);

        // Store current VP for next frame's motion vectors.
        self.prev_vp = self.camera.view_projection(INTERNAL_WIDTH, INTERNAL_HEIGHT)
            .to_cols_array_2d();

        // Poll the GPU device to acknowledge completed work from previous frames.
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        // Get swapchain texture.
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = PhysicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT);
                self.ctx.configure_surface(&self.surface, size.width, size.height);
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let target_view = frame.texture.create_view(&Default::default());

        // Record command buffer.
        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // Update coarse field uniforms (camera-relative origin changes each frame).
        self.coarse_field.update_uniforms(&self.ctx.queue, self.camera.position);

        // Update radiance volume center to camera world position.
        self.radiance_volume.update_center(
            &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
        );

        // Update inject uniforms.
        self.radiance_inject.update_uniforms(&self.ctx.queue, &InjectUniforms {
            num_lights: self.world_lights.len() as u32,
            max_shadow_lights: 1,
            _pad: [0; 2],
        });

        // --- Core rendering ---
        // 1. Tile object culling → per-tile object lists
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        // 2. Ray march → G-buffer (reads tile lists + coarse field)
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        // 3. Radiance injection → fill L0 with direct-lit surface radiance
        self.radiance_inject.dispatch(
            &mut encoder, &self.gpu_scene, &self.coarse_field,
        );
        // 4. Radiance mip gen → downsample L0 → L1 → L2 → L3
        self.radiance_mip.dispatch(&mut encoder);
        // 5. PBR shading → HDR output (with SDF shadows + AO + GI cone tracing)
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );

        // --- Volumetric pipeline ---
        // 6. Volumetric shadow map (coarse field → 3D transmittance)
        let sun_dir = [0.5f32, 1.0, 0.3]; // match scene sun direction
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
            sun_dir,
            &self.coarse_field.bind_group,
        );
        // 7. Cloud shadow map (analytical FBM → 2D transmittance)
        self.cloud_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
            sun_dir,
        );
        // 8. Volumetric march (half-res fog/dust/clouds)
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [self.camera.forward().x, self.camera.forward().y, self.camera.forward().z, 0.0],
            cam_right: [self.camera.right().x, self.camera.right().y, self.camera.right().z, 0.0],
            cam_up: [self.camera.up().x, self.camera.up().y, self.camera.up().z, 0.0],
            sun_dir: [sun_dir[0], sun_dir[1], sun_dir[2], 0.0],
            sun_color: [1.0, 0.95, 0.85, 0.0],
            width: INTERNAL_WIDTH / 2,
            height: INTERNAL_HEIGHT / 2,
            full_width: INTERNAL_WIDTH,
            full_height: INTERNAL_HEIGHT,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            // Light height fog for atmosphere
            fog_color: [0.7, 0.8, 0.9, 1.0],     // height fog ON (w=1)
            fog_height: [0.01, -0.5, 0.15, 0.0],   // gentle base density, below ground
            fog_distance: [0.0, 0.01, 0.001, 0.3], // very light ambient dust
            frame_index: self.frame_index,
            _pad0: 0, _pad1: 0, _pad2: 0,
            vol_shadow_min: [
                self.camera.position.x - 40.0,
                self.camera.position.y - 10.0,
                self.camera.position.z - 40.0, 0.0,
            ],
            vol_shadow_max: [
                self.camera.position.x + 40.0,
                self.camera.position.y + 10.0,
                self.camera.position.z + 40.0, 0.0,
            ],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        // 9. Bilateral upscale (half-res scatter → full-res)
        self.vol_upscale.dispatch(&mut encoder);
        // 10. Volumetric composite (shade HDR + scatter → composited HDR)
        self.vol_composite.dispatch(&mut encoder);

        // --- Post-processing pipeline ---
        // 10.5. Screen-space radial blur god rays
        {
            let sun_dir_vec = glam::Vec3::new(0.5, 1.0, 0.3).normalize();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir_vec.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir_vec.dot(right) / sun_dot;
                let ndc_y = -sun_dir_vec.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur.update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        // 11. Bloom extraction + downsample
        self.bloom.dispatch(&mut encoder);
        // 12. Auto-exposure (histogram + adapted EV)
        let dt = 1.0 / 60.0; // approximate dt for exposure adaptation
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, dt);
        // 13. Depth of field
        self.dof.dispatch(&mut encoder);
        // 14. Motion blur
        self.motion_blur.dispatch(&mut encoder);
        // 15. Bloom composite (add bloom glow to motion-blurred HDR)
        self.bloom_composite.dispatch(&mut encoder);
        // 16. Tone map (HDR → LDR with auto-exposure)
        self.tone_map.dispatch(&mut encoder);
        // 17. Color grading
        self.color_grade.dispatch(&mut encoder);
        // 18. Cosmetics (vignette, grain, chromatic aberration)
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);
        // 19. Blit → swapchain
        self.blit.draw(&mut encoder, &target_view);

        // 6. If MCP screenshot requested, copy LDR output to readback buffer.
        let do_readback = self.shared_state.lock()
            .map(|s| s.screenshot_requested)
            .unwrap_or(false);

        let bytes_per_pixel = 4u32; // Rgba8Unorm = 4 bytes
        let unpadded_row = INTERNAL_WIDTH * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;

        if do_readback {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.cosmetics.output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(INTERNAL_HEIGHT),
                    },
                },
                wgpu::Extent3d {
                    width: INTERNAL_WIDTH,
                    height: INTERNAL_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        // 7. Read back pixels for MCP screenshots (only when requested).
        if do_readback {
            let buffer_slice = self.readback_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                let data = buffer_slice.get_mapped_range();
                let pixel_count = (INTERNAL_WIDTH * INTERNAL_HEIGHT) as usize;
                let mut rgba8 = vec![0u8; pixel_count * 4];

                // Rgba8Unorm: direct byte copy (strip row padding).
                for y in 0..INTERNAL_HEIGHT as usize {
                    let src_row_offset = y * padded_row as usize;
                    let dst_row_offset = y * INTERNAL_WIDTH as usize * 4;
                    let row_bytes = INTERNAL_WIDTH as usize * 4;
                    rgba8[dst_row_offset..dst_row_offset + row_bytes]
                        .copy_from_slice(&data[src_row_offset..src_row_offset + row_bytes]);
                }
                drop(data);
                self.readback_buffer.unmap();

                if let Ok(mut state) = self.shared_state.lock() {
                    state.frame_pixels = rgba8;
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
}

// ---------------------------------------------------------------------------
// IPC server setup
// ---------------------------------------------------------------------------

fn start_ipc_server(
    rt: &tokio::runtime::Runtime,
    api: Arc<dyn rkf_core::automation::AutomationApi>,
) -> (tokio::task::JoinHandle<()>, String) {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let meta_path = socket_path.replace(".sock", ".json");
    let meta = serde_json::json!({
        "type": "testbed",
        "pid": std::process::id(),
        "socket": &socket_path,
        "name": "RKIField Testbed",
        "version": "0.1.0"
    });
    let _ = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap());

    let config = rkf_mcp::ipc::IpcConfig {
        socket_path: Some(socket_path.clone()),
        tcp_port: 0,
        mode: rkf_mcp::registry::ToolMode::Editor,
    };

    let handle = rt.spawn(async move {
        if let Err(e) = rkf_mcp::ipc::run_server(config, api).await {
            log::error!("IPC server error: {e}");
        }
    });

    (handle, socket_path)
}

fn cleanup_ipc(socket_path: &str) {
    let _ = std::fs::remove_file(socket_path);
    let meta_path = socket_path.replace(".sock", ".json");
    let _ = std::fs::remove_file(&meta_path);
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    engine: Option<EngineState>,
    shared_state: Arc<Mutex<SharedState>>,
    last_frame: Instant,
    _ipc_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: Option<String>,
    rt: tokio::runtime::Runtime,
    // Input state
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        )));
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        Self {
            window: None,
            engine: None,
            shared_state,
            last_frame: Instant::now(),
            _ipc_handle: None,
            socket_path: None,
            rt,
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
        }
    }

    fn process_movement(&mut self, dt: f32) {
        let engine = match self.engine.as_mut() {
            Some(e) => e,
            None => return,
        };

        let speed = engine.camera.move_speed * dt;
        if self.keys_held.contains(&KeyCode::KeyW) {
            engine.camera.translate_forward(speed);
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            engine.camera.translate_forward(-speed);
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            engine.camera.translate_right(-speed);
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            engine.camera.translate_right(speed);
        }
        if self.keys_held.contains(&KeyCode::Space) {
            engine.camera.translate_up(speed);
        }
        if self.keys_held.contains(&KeyCode::ShiftLeft) {
            engine.camera.translate_up(-speed);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("RKIField Testbed — v2 Object-Centric")
            .with_inner_size(PhysicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        // Initialize GPU engine.
        let engine = EngineState::new(window.clone(), Arc::clone(&self.shared_state));
        self.engine = Some(engine);

        // Start IPC server.
        let api = TestbedAutomationApi::new(Arc::clone(&self.shared_state));
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
        let (handle, path) = start_ipc_server(&self.rt, api);
        self._ipc_handle = Some(handle);
        self.socket_path = Some(path);

        log::info!("v2 testbed initialized — 5 analytical objects, BVH traversal");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured {
                if let Some(engine) = self.engine.as_mut() {
                    engine.camera.rotate(delta.0, delta.1);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref path) = self.socket_path {
                    cleanup_ipc(path);
                }
                event_loop.exit();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.mouse_captured = state == ElementState::Pressed;
                    if let Some(window) = &self.window {
                        let _ = window.set_cursor_visible(!self.mouse_captured);
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_held.insert(key);

                            match key {
                                KeyCode::Escape => {
                                    if let Some(ref path) = self.socket_path {
                                        cleanup_ipc(path);
                                    }
                                    event_loop.exit();
                                }
                                // Debug mode toggle: 1-4
                                KeyCode::Digit1 => {
                                    if let Some(e) = self.engine.as_mut() {
                                        e.debug_mode = DebugMode::Lambert;
                                        e.shade_debug_mode = 0;
                                        log::info!("Debug mode: Lambert");
                                    }
                                }
                                KeyCode::Digit2 => {
                                    if let Some(e) = self.engine.as_mut() {
                                        e.debug_mode = DebugMode::Normals;
                                        e.shade_debug_mode = 1;
                                        log::info!("Debug mode: Normals");
                                    }
                                }
                                KeyCode::Digit3 => {
                                    if let Some(e) = self.engine.as_mut() {
                                        e.debug_mode = DebugMode::Positions;
                                        e.shade_debug_mode = 2;
                                        log::info!("Debug mode: Positions");
                                    }
                                }
                                KeyCode::Digit4 => {
                                    if let Some(e) = self.engine.as_mut() {
                                        e.debug_mode = DebugMode::MaterialIds;
                                        e.shade_debug_mode = 3;
                                        log::info!("Debug mode: Material IDs");
                                    }
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.keys_held.remove(&key);
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Apply camera controls.
                self.process_movement(dt);

                // Apply pending MCP commands.
                if let Some(engine) = self.engine.as_mut() {
                    if let Ok(mut state) = self.shared_state.lock() {
                        if let Some(cam) = state.pending_camera.take() {
                            engine.camera.position = cam.position;
                            engine.camera.yaw = cam.yaw;
                            engine.camera.pitch = cam.pitch;
                        }
                        if let Some(mode) = state.pending_debug_mode.take() {
                            engine.shade_debug_mode = mode;
                            engine.debug_mode = match mode {
                                0 => DebugMode::Lambert,
                                1 => DebugMode::Normals,
                                2 => DebugMode::Positions,
                                3 => DebugMode::MaterialIds,
                                _ => DebugMode::Lambert,
                            };
                        }
                        // Update shared state for MCP observation.
                        state.camera_position = engine.camera.position;
                        state.camera_yaw = engine.camera.yaw;
                        state.camera_pitch = engine.camera.pitch;
                        state.frame_time_ms = dt as f64 * 1000.0;
                    }
                }

                // Render frame.
                if let Some(engine) = self.engine.as_mut() {
                    engine.render();
                }

                // Request next frame.
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    log::info!("RKIField Testbed v2 — object-centric ray marching");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
