//! Editor engine — v2 object-centric render pipeline.
//!
//! Extracted from the testbed's `EngineState`. Provides the full v2 compute-shader
//! render pipeline (ray march, shading, GI, volumetrics, post-processing) in a
//! reusable struct that the editor's event loop drives each frame.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use glam::{IVec3, Quat, Vec3};
use winit::window::Window;

use rkf_core::{
    Aabb, BrickMapAllocator, BrickPool, Scene, SceneNode, SceneObject,
    SdfPrimitive, SdfSource, WorldPosition, voxelize_sdf,
    transform_flatten::flatten_object,
};
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass,
    GBuffer, GpuObject, GpuSceneV2, Light, LightBuffer, MotionBlurPass, RadianceVolume,
    RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms, ShadingPass, SharpenPass,
    TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass, VolShadowPass,
    VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::{RadianceInjectPass, InjectUniforms};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::{MaterialTable, create_test_materials};
use rkf_animation::character::{
    AnimatedCharacter, build_humanoid_skeleton, build_humanoid_visuals, build_walk_clip,
};

use crate::automation::SharedState;
use crate::camera::EditorCamera;

/// Internal render resolution width.
pub const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height.
pub const INTERNAL_HEIGHT: u32 = 540;

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
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(0.0, -0.6, 0.0)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: ground,
        aabb: Aabb::new(Vec3::new(-10.0, -0.7, -10.0), Vec3::new(10.0, -0.5, 10.0)),
    };
    scene.add_object_full(ground_obj);

    // 2. Sphere (analytical)
    let sphere = SceneNode::analytical("sphere", SdfPrimitive::Sphere { radius: 0.5 }, 2);
    let sphere_obj = SceneObject {
        id: 0,
        name: "sphere".into(),
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(-1.5, 0.0, -2.0)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: sphere,
        aabb: Aabb::new(Vec3::new(-2.0, -0.5, -2.5), Vec3::new(-1.0, 0.5, -1.5)),
    };
    scene.add_object_full(sphere_obj);

    // 3. Box (analytical)
    let box_node = SceneNode::analytical("box", SdfPrimitive::Box {
        half_extents: Vec3::new(0.35, 0.35, 0.35),
    }, 3);
    let box_obj = SceneObject {
        id: 0,
        name: "box".into(),
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(0.0, 0.0, -2.0)),
        rotation: Quat::from_rotation_y(0.5),
        scale: 1.0,
        root_node: box_node,
        aabb: Aabb::new(Vec3::new(-0.5, -0.5, -2.5), Vec3::new(0.5, 0.5, -1.5)),
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
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(1.5, 0.0, -2.0)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: capsule,
        aabb: Aabb::new(Vec3::new(1.0, -0.6, -2.5), Vec3::new(2.0, 0.6, -1.5)),
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
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(0.0, 0.3, -3.5)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: torus,
        aabb: Aabb::new(Vec3::new(-0.6, -0.2, -4.1), Vec3::new(0.6, 0.8, -2.9)),
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
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(3.0, 0.0, -2.0)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: vox_node,
        aabb: Aabb::new(
            Vec3::new(3.0 - vox_radius - margin, -vox_radius - margin, -2.0 - vox_radius - margin),
            Vec3::new(3.0 + vox_radius + margin, vox_radius + margin, -2.0 + vox_radius + margin),
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
        world_position: WorldPosition::new(IVec3::ZERO, Vec3::new(-3.0, 0.0, -2.0)),
        rotation: Quat::IDENTITY,
        scale: 1.0,
        root_node: char_root,
        aabb: Aabb::new(Vec3::new(-3.6, -0.5, -2.5), Vec3::new(-2.4, 2.0, -1.5)),
    };
    scene.add_object_full(char_obj);
    let character_obj_index = scene.root_objects.len() - 1;

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
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
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
    pub scene: Scene,
    world_lights: Vec<Light>,
    light_buffer: LightBuffer,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    shade_debug_mode: u32,
    // Screenshot readback (window-resolution, captures composited output with UI)
    readback_buffer: wgpu::Buffer,
    window_width: u32,
    window_height: u32,
    shared_state: Arc<Mutex<SharedState>>,
    character: AnimatedCharacter,
    character_obj_index: usize,
    last_frame_time: Instant,
}

impl EditorEngine {
    /// Initialize the render engine with wgpu, all passes, and a demo scene.
    pub fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> Self {
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

        // Coarse acceleration field.
        let scene_aabbs: Vec<(Vec3, Vec3)> = scene.root_objects.iter()
            .map(|obj| (obj.aabb.min, obj.aabb.max))
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

        // Lights.
        let world_lights = vec![
            Light::directional([0.5, 1.0, 0.3], [1.0, 0.95, 0.85], 3.0, true),
            Light::point([2.0, 1.5, -1.0], [1.0, 0.8, 0.5], 5.0, 8.0, true),
            Light::point([-2.0, 1.0, -3.0], [0.5, 0.7, 1.0], 3.0, 6.0, false),
        ];
        let light_buffer = LightBuffer::upload(&ctx.device, &world_lights);
        log::info!("Lights: {} lights uploaded", world_lights.len());

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
        let bloom = BloomPass::new(
            &ctx.device, &vol_composite.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let auto_exposure = AutoExposurePass::new(
            &ctx.device, &vol_composite.output_view, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        );
        let dof = DofPass::new(
            &ctx.device, &vol_composite.output_view, &gbuffer.position_view,
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

        // Readback buffer for MCP screenshots — at window resolution so we
        // capture the composited output (engine viewport + rinch UI panels).
        let window_width = size.width.max(1);
        let window_height = size.height.max(1);
        let readback_buffer = Self::create_readback_buffer(
            &ctx.device, window_width, window_height,
        );

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
            shade_debug_mode: 0,
            readback_buffer,
            window_width,
            window_height,
            shared_state,
            character: demo.character,
            character_obj_index: demo.character_obj_index,
            last_frame_time: Instant::now(),
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

    /// Configure the surface with COPY_SRC for screenshot readback.
    fn configure_surface_copy(
        ctx: &RenderContext,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let caps = surface.get_capabilities(&ctx.adapter);
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
            present_mode: wgpu::PresentMode::AutoVsync,
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

    /// Reconfigure the surface for a new window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            Self::configure_surface_copy(&self.ctx, &self.surface, width, height);
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

    /// Render one frame without UI overlay (full-screen engine blit).
    pub fn render_frame(&mut self) {
        self.render_frame_composited(|_, _, _| {});
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
    pub fn render_frame_composited<F>(&mut self, post_engine: F)
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(None, post_engine);
    }

    /// Render one frame with engine output constrained to a viewport sub-region.
    ///
    /// The engine blit is drawn into `viewport` (x, y, width, height) in pixels.
    /// Areas outside the viewport are cleared to black (covered by UI panels).
    /// The `post_engine` callback composites the UI overlay on top.
    pub fn render_frame_viewport<F>(
        &mut self,
        viewport: (f32, f32, f32, f32),
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        self.render_frame_inner(Some(viewport), post_engine);
    }

    fn render_frame_inner<F>(
        &mut self,
        viewport: Option<(f32, f32, f32, f32)>,
        post_engine: F,
    )
    where
        F: FnOnce(&wgpu::Device, &wgpu::Queue, &wgpu::TextureView),
    {
        // Advance character animation.
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        self.character.advance_and_update(
            dt,
            &mut self.scene.root_objects[self.character_obj_index].root_node,
        );

        let camera_pos = WorldPosition::new(IVec3::ZERO, self.camera.position);

        // Flatten all objects → GPU object list + BVH.
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        for obj in &self.scene.root_objects {
            let flat_nodes = flatten_object(obj, &camera_pos);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let aabb = obj.aabb;
                let cam_rel_min = aabb.min - self.camera.position;
                let cam_rel_max = aabb.max - self.camera.position;
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                ));
                bvh_pairs.push((gpu_idx, aabb));
            }
        }

        self.gpu_scene.upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);

        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene.upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Update camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            INTERNAL_WIDTH, INTERNAL_HEIGHT, self.frame_index, self.prev_vp,
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

        // Shade uniforms.
        self.shading_pass.update_uniforms(&self.ctx.queue, &ShadeUniforms {
            debug_mode: self.shade_debug_mode,
            num_lights: self.world_lights.len() as u32,
            _pad0: 0,
            shadow_budget_k: 0,
            camera_pos: [0.0, 0.0, 0.0, 0.0],
        });

        // Camera-relative lights.
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
        self.light_buffer.update(&self.ctx.queue, &cam_rel_lights);

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
        self.prev_vp = self.camera.view_projection(INTERNAL_WIDTH, INTERNAL_HEIGHT)
            .to_cols_array_2d();

        // Get swapchain texture.
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(
                    INTERNAL_WIDTH.max(64),
                    INTERNAL_HEIGHT.max(64),
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
            num_lights: self.world_lights.len() as u32,
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
        let sun_dir = [0.5f32, 1.0, 0.3];
        self.vol_shadow.dispatch(
            &mut encoder, &self.ctx.queue,
            [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
        );
        self.cloud_shadow.dispatch(
            &mut encoder, &self.ctx.queue, [cam.x, cam.y, cam.z], sun_dir,
        );
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
            fog_color: [0.7, 0.8, 0.9, 1.0],
            fog_height: [0.01, -0.5, 0.15, 0.0],
            fog_distance: [0.0, 0.01, 0.001, 0.3],
            frame_index: self.frame_index,
            _pad0: 0, _pad1: 0, _pad2: 0,
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // --- Post-processing pipeline ---
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // --- Blit engine output to swapchain ---
        if let Some(vp) = viewport {
            self.blit.draw_viewport(&mut encoder, &target_view, vp);
        } else {
            self.blit.draw(&mut encoder, &target_view);
        }

        // Submit engine work.
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Let the caller composite overlay (e.g. rinch UI) on top.
        // This runs after engine submit (Vello needs the queue), before present.
        post_engine(&self.ctx.device, &self.ctx.queue, &target_view);

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

        self.frame_index += 1;
    }
}
