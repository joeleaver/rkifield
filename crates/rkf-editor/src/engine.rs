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

/// Build the demo scene with 5 analytical + 1 voxelized + 1 animated character.
fn build_demo_scene() -> DemoScene {
    let mut scene = Scene::new("editor_demo");

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

    // 6. Voxelized sphere
    let mut brick_pool = BrickPool::new(4096);
    let mut brick_map_alloc = BrickMapAllocator::new();

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

    // 7. Animated humanoid character
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
        };
        (engine, scene)
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
        use rkf_core::brick_map::EMPTY_SLOT;
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
                        if slot != EMPTY_SLOT {
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
                        if slot != EMPTY_SLOT {
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

        // 6. Update the object.
        obj.root_node.sdf_source = SdfSource::Voxelized {
            brick_map_handle: new_handle,
            voxel_size,
            aabb: new_aabb,
        };
        obj.aabb = new_aabb;
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
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
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
                gpu_objects.push(GpuObject::from_flat_node(
                    flat, obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
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
}
