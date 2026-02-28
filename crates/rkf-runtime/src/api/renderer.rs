//! Renderer — the GPU rendering pipeline.
//!
//! Owns all GPU resources (brick pool buffer, G-buffer, 21 compute passes,
//! readback buffers). Takes a [`&Scene`](rkf_core::scene::Scene) per frame via
//! the [`World`](super::world::World) reference — does NOT own the scene.
//!
//! # Usage
//!
//! ```ignore
//! let mut renderer = Renderer::new(&window, RendererConfig::default());
//! renderer.add_light(Light::directional([0.5, 1.0, 0.3], [1.0, 0.95, 0.85], 3.0, true));
//! // Per frame:
//! renderer.render(&world, &RenderTarget::Surface(&surface));
//! ```

use std::sync::Arc;

use glam::Vec3;
use winit::window::Window;

use rkf_core::aabb::Aabb;
use rkf_core::transform_bake;
use rkf_core::transform_flatten::flatten_object;
use rkf_render::material_table::{MaterialTable, create_test_materials};
use rkf_render::radiance_inject::{InjectUniforms, RadianceInjectPass};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass, GBuffer,
    GodRaysBlurPass, GpuObject, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms,
    ShadingPass, SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass,
    VolMarchPass, VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_core::material::Material;

use super::world::World;

/// Renderer configuration.
pub struct RendererConfig {
    /// Internal render resolution width (default 960).
    pub internal_width: u32,
    /// Internal render resolution height (default 540).
    pub internal_height: u32,
    /// Display/window resolution width (default 1280).
    pub display_width: u32,
    /// Display/window resolution height (default 720).
    pub display_height: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            internal_width: 960,
            internal_height: 540,
            display_width: 1280,
            display_height: 720,
        }
    }
}

/// Where to render the final image.
pub enum RenderTarget<'a> {
    /// Render to a window surface (swapchain).
    Surface(&'a wgpu::Surface<'a>),
    /// Render to an offscreen texture view.
    Texture(&'a wgpu::TextureView),
}

/// The GPU rendering pipeline.
///
/// Owns all GPU resources and dispatches 21 compute passes per frame.
/// Takes `&World` to read the scene — does NOT own scene data.
pub struct Renderer {
    // GPU context
    ctx: RenderContext,

    // Scene GPU data
    gpu_scene: GpuSceneV2,
    brick_pool_buffer: wgpu::Buffer,

    // Core rendering
    gbuffer: GBuffer,
    tile_cull: TileObjectCullPass,
    coarse_field: CoarseField,
    ray_march: RayMarchPass,
    debug_view: DebugViewPass,
    shading_pass: ShadingPass,
    material_table: MaterialTable,

    // Global illumination
    radiance_volume: RadianceVolume,
    radiance_inject: RadianceInjectPass,
    radiance_mip: RadianceMipPass,

    // Volumetrics
    vol_shadow: VolShadowPass,
    cloud_shadow: CloudShadowPass,
    vol_march: VolMarchPass,
    vol_upscale: VolUpscalePass,
    vol_composite: VolCompositePass,

    // Post-processing
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

    // Readback
    readback_buffer: wgpu::Buffer,

    // CPU state
    camera: Camera,
    world_lights: Vec<Light>,
    materials: Vec<Material>,
    light_buffer: LightBuffer,
    environment: crate::environment::EnvironmentProfile,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    shade_debug_mode: u32,

    // Config
    internal_width: u32,
    internal_height: u32,
    display_width: u32,
    display_height: u32,
    surface_format: wgpu::TextureFormat,
}

impl Renderer {
    /// Create a new Renderer with its own GPU device, attached to a window.
    pub fn new(window: Arc<Window>, config: RendererConfig) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let ctx = RenderContext::new(&instance, &surface);

        let surface_format =
            ctx.configure_surface(&surface, config.display_width, config.display_height);

        Self::build(ctx, surface_format, config)
    }

    /// Create a new Renderer using a shared GPU device (e.g. from an editor compositor).
    pub fn from_shared(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: RendererConfig,
    ) -> Self {
        let ctx = RenderContext::from_shared(device, queue);
        Self::build(ctx, surface_format, config)
    }

    fn build(ctx: RenderContext, surface_format: wgpu::TextureFormat, config: RendererConfig) -> Self {
        let iw = config.internal_width;
        let ih = config.internal_height;

        // Create empty brick pool buffer.
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: 8, // minimum — resized on first upload
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);

        // Re-create brick pool buffer as owned (gpu_scene took ownership of the first one)
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool_shadow"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gbuffer = GBuffer::new(&ctx.device, iw, ih);
        let tile_cull = TileObjectCullPass::new(&ctx.device, &gpu_scene, iw, ih);

        // Initial coarse field (will be recomputed from scene each frame)
        let coarse_field = CoarseField::from_scene_aabbs(
            &ctx.device,
            &[], // empty scene
            COARSE_VOXEL_SIZE,
            1.0,
        );

        let ray_march = RayMarchPass::new(&ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field);
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material table
        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights — start with a default sun
        let world_lights = vec![Light::directional(
            [0.5, 1.0, 0.3],
            [1.0, 0.95, 0.85],
            3.0,
            true,
        )];
        let light_buffer = LightBuffer::upload(&ctx.device, &world_lights);

        // GI
        let radiance_volume = RadianceVolume::new(&ctx.device);
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device,
            &gpu_scene,
            &material_table.buffer,
            &light_buffer,
            &radiance_volume,
            &coarse_field,
        );
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Shading
        let shading_pass = ShadingPass::new(
            &ctx.device,
            &gbuffer,
            &gpu_scene,
            &light_buffer,
            &coarse_field,
            &radiance_volume,
            &material_table.buffer,
            iw,
            ih,
        );

        // Volumetrics
        let vol_shadow =
            VolShadowPass::new(&ctx.device, &ctx.queue, &coarse_field.bind_group_layout);
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = iw / 2;
        let half_h = ih / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device,
            &ctx.queue,
            &gbuffer.position_view,
            &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w,
            half_h,
            iw,
            ih,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device,
            &vol_march.output_view,
            &gbuffer.position_view,
            iw,
            ih,
            half_w,
            half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device,
            &shading_pass.hdr_view,
            &vol_upscale.output_view,
            iw,
            ih,
        );

        // Post-processing
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device,
            &vol_composite.output_view,
            &gbuffer.position_view,
            iw,
            ih,
        );
        let bloom = BloomPass::new(&ctx.device, &god_rays_blur.output_view, iw, ih);
        let auto_exposure =
            AutoExposurePass::new(&ctx.device, &god_rays_blur.output_view, iw, ih);
        let dof = DofPass::new(
            &ctx.device,
            &god_rays_blur.output_view,
            &gbuffer.position_view,
            iw,
            ih,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device,
            &dof.output_view,
            &gbuffer.motion_view,
            iw,
            ih,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device,
            &motion_blur.output_view,
            bloom.mip_views(),
            iw,
            ih,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device,
            &bloom_composite.output_view,
            iw,
            ih,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device,
            &ctx.queue,
            &tone_map.ldr_view,
            iw,
            ih,
        );
        let cosmetics = CosmeticsPass::new(&ctx.device, &color_grade.output_view, iw, ih);
        let sharpen = SharpenPass::new(
            &ctx.device,
            &vol_composite.output_view,
            &gbuffer,
            iw,
            ih,
        );
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, surface_format);

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 3.0;

        // Readback buffer
        let bytes_per_pixel = 4u32;
        let unpadded_row = iw * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;
        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * ih) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            gpu_scene,
            brick_pool_buffer,
            gbuffer,
            tile_cull,
            coarse_field,
            ray_march,
            debug_view,
            shading_pass,
            material_table,
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
            readback_buffer,
            camera,
            world_lights,
            materials,
            light_buffer,
            environment: crate::environment::EnvironmentProfile::default(),
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            internal_width: iw,
            internal_height: ih,
            display_width: config.display_width,
            display_height: config.display_height,
            surface_format,
        }
    }

    // ── Per-frame rendering ────────────────────────────────────────────────

    /// Render a frame from the given world to a surface.
    pub fn render_to_surface(&mut self, world: &World, surface: &wgpu::Surface<'_>) {
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e}");
                return;
            }
        };
        let target_view = frame.texture.create_view(&Default::default());
        self.render_frame(world, Some(&target_view));
        frame.present();
    }

    /// Render a frame to the internal offscreen texture (for editor compositing).
    ///
    /// Returns a reference to the final LDR output texture view.
    pub fn render_offscreen(&mut self, world: &World) -> &wgpu::TextureView {
        self.render_frame(world, None);
        &self.cosmetics.output_view
    }

    /// Core render dispatch — executes all 21 passes.
    fn render_frame(&mut self, world: &World, blit_target: Option<&wgpu::TextureView>) {
        let scene = world.scene();
        let camera_pos = self.camera.position;
        let iw = self.internal_width;
        let ih = self.internal_height;

        // Bake world transforms for parent-child hierarchy
        let world_transforms = transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = transform_bake::WorldTransform::default();

        // Flatten all objects to GPU representation + build BVH pairs
        let mut gpu_objects = Vec::new();
        let mut bvh_pairs = Vec::new();
        let mut scene_aabbs = Vec::new();

        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let camera_rel = wt.position - camera_pos;

            // Transform local AABB to world space
            let world_aabb = transform_aabb(&obj.aabb, wt);
            scene_aabbs.push((world_aabb.min, world_aabb.max));

            let flat_nodes = flatten_object(obj, camera_rel);
            for flat in &flat_nodes {
                let gpu_idx = gpu_objects.len() as u32;
                let cam_rel_min = world_aabb.min - camera_pos;
                let cam_rel_max = world_aabb.max - camera_pos;
                gpu_objects.push(GpuObject::from_flat_node(
                    flat,
                    obj.id,
                    [cam_rel_min.x, cam_rel_min.y, cam_rel_min.z, 0.0],
                    [cam_rel_max.x, cam_rel_max.y, cam_rel_max.z, 0.0],
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        // Upload brick pool if needed
        let pool_data: &[u8] = bytemuck::cast_slice(world.brick_pool().as_slice());
        if !pool_data.is_empty() {
            // Check if we need to reallocate
            if pool_data.len() as u64 > self.brick_pool_buffer.size() {
                self.brick_pool_buffer =
                    self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("brick_pool"),
                        size: pool_data.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                // Create a second buffer for the GPU scene bind group
                let new_pool_buf =
                    self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("brick_pool_gpu"),
                        size: pool_data.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                self.gpu_scene
                    .set_brick_pool(&self.ctx.device, new_pool_buf);
            }
            self.ctx
                .queue
                .write_buffer(&self.brick_pool_buffer, 0, pool_data);
        }

        // Upload brick maps
        let brick_map_data = world.brick_map_alloc().as_slice();
        if !brick_map_data.is_empty() {
            self.gpu_scene
                .upload_brick_maps(&self.ctx.device, &self.ctx.queue, brick_map_data);
        }

        // Upload GPU objects
        self.gpu_scene
            .upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);

        // Build and upload BVH
        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene
            .upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Update camera uniforms
        let cam_uniforms = self.camera.uniforms(iw, ih, self.frame_index, self.prev_vp);
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        // Update scene uniforms
        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene
            .update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Update shade uniforms
        let fwd = self.camera.forward();
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = iw as f32 / ih as f32;
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;
        let sun_dir = Vec3::new(0.5, 1.0, 0.3).normalize();
        let cam = self.camera.position;

        self.shading_pass.update_uniforms(
            &self.ctx.queue,
            &ShadeUniforms {
                debug_mode: self.shade_debug_mode,
                num_lights: self.world_lights.len() as u32,
                _pad0: 0,
                shadow_budget_k: 0,
                camera_pos: [cam.x, cam.y, cam.z, 0.0],
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 3.0],
                sun_color: [1.0, 0.95, 0.85, 0.0],
                sky_params: [1.0, 1.0, 1.0, 0.0],
                cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
                cam_right: [right.x, right.y, right.z, 0.0],
                cam_up: [up.x, up.y, up.z, 0.0],
            },
        );

        // Upload camera-relative light positions
        let cam_rel_lights: Vec<Light> = self
            .world_lights
            .iter()
            .map(|l| {
                let mut cl = *l;
                if cl.light_type != 0 {
                    cl.pos_x -= cam.x;
                    cl.pos_y -= cam.y;
                    cl.pos_z -= cam.z;
                }
                cl
            })
            .collect();
        self.light_buffer.update(&self.ctx.queue, &cam_rel_lights);

        // Update debug view
        let dm = match self.shade_debug_mode {
            1 => DebugMode::Normals,
            2 => DebugMode::Positions,
            3 => DebugMode::MaterialIds,
            _ => DebugMode::Lambert,
        };
        self.debug_view.set_mode(&self.ctx.queue, dm);

        // Store VP for motion vectors
        self.prev_vp = self.camera.view_projection(iw, ih).to_cols_array_2d();

        // Poll device
        let _ = self.ctx.device.poll(wgpu::PollType::Poll);

        // Build command buffer
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Update coarse field
        if !scene_aabbs.is_empty() {
            self.coarse_field = CoarseField::from_scene_aabbs(
                &self.ctx.device,
                &scene_aabbs,
                COARSE_VOXEL_SIZE,
                1.0,
            );
            self.coarse_field.populate(&scene_aabbs);
        }
        self.coarse_field.upload(&self.ctx.queue, camera_pos);

        // Update radiance volume center
        self.radiance_volume
            .update_center(&self.ctx.queue, [cam.x, cam.y, cam.z]);

        // Update inject uniforms
        self.radiance_inject.update_uniforms(
            &self.ctx.queue,
            &InjectUniforms {
                num_lights: self.world_lights.len() as u32,
                max_shadow_lights: 1,
                _pad: [0; 2],
            },
        );

        // === Core rendering ===
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.ray_march.dispatch(
            &mut encoder,
            &self.gpu_scene,
            &self.gbuffer,
            &self.tile_cull,
            &self.coarse_field,
        );
        self.radiance_inject
            .dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.radiance_mip.dispatch(&mut encoder);
        self.shading_pass.dispatch(
            &mut encoder,
            &self.gbuffer,
            &self.gpu_scene,
            &self.coarse_field,
            &self.radiance_volume,
        );

        // === Volumetrics ===
        let sun_dir_arr = [0.5f32, 1.0, 0.3];
        self.vol_shadow.dispatch(
            &mut encoder,
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
            &self.coarse_field.bind_group,
        );
        self.cloud_shadow.dispatch(
            &mut encoder,
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
        );
        let half_w = iw / 2;
        let half_h = ih / 2;
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir_arr[0], sun_dir_arr[1], sun_dir_arr[2], 0.0],
            sun_color: [1.0, 0.95, 0.85, 0.0],
            width: half_w,
            height: half_h,
            full_width: iw,
            full_height: ih,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [0.7, 0.8, 0.9, 1.0],
            fog_height: [0.01, -0.5, 0.15, 0.0],
            fog_distance: [0.0, 0.01, 0.001, 0.3],
            frame_index: self.frame_index,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            vol_shadow_min: [cam.x - 40.0, cam.y - 10.0, cam.z - 40.0, 0.0],
            vol_shadow_max: [cam.x + 40.0, cam.y + 10.0, cam.z + 40.0, 0.0],
        };
        self.vol_march
            .dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.vol_upscale.dispatch(&mut encoder);
        self.vol_composite.dispatch(&mut encoder);

        // === Post-processing ===
        {
            let sun_dir_vec = Vec3::new(0.5, 1.0, 0.3).normalize();
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir_vec.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir_vec.dot(right) / sun_dot;
                let ndc_y = -sun_dir_vec.dot(up) / sun_dot;
                (ndc_x * 0.5 + 0.5, ndc_y * 0.5 + 0.5)
            } else {
                (0.5, 0.5)
            };
            self.god_rays_blur
                .update_sun(&self.ctx.queue, sun_uv_x, sun_uv_y, sun_dot);
        }
        self.god_rays_blur.dispatch(&mut encoder);
        self.bloom.dispatch(&mut encoder);
        self.auto_exposure
            .dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.dof.dispatch(&mut encoder);
        self.motion_blur.dispatch(&mut encoder);
        self.bloom_composite.dispatch(&mut encoder);
        self.tone_map.dispatch(&mut encoder);
        self.color_grade.dispatch(&mut encoder);
        self.cosmetics
            .dispatch(&mut encoder, &self.ctx.queue, self.frame_index);

        // Blit to target if provided
        if let Some(target) = blit_target {
            self.blit.draw(&mut encoder, target);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        self.frame_index += 1;
    }

    // ── Camera ─────────────────────────────────────────────────────────────

    /// Get a shared reference to the camera.
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get a mutable reference to the camera.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Set the camera world position.
    pub fn set_camera_position(&mut self, pos: Vec3) {
        self.camera.position = pos;
    }

    /// Set the camera orientation (yaw and pitch in radians).
    pub fn set_camera_orientation(&mut self, yaw: f32, pitch: f32) {
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
    }

    /// Set the camera field of view (in degrees).
    pub fn set_camera_fov(&mut self, fov_degrees: f32) {
        self.camera.fov_degrees = fov_degrees;
    }

    // ── Materials ──────────────────────────────────────────────────────────

    /// Set a material at the given index.
    pub fn set_material(&mut self, index: u16, material: Material) {
        let idx = index as usize;
        if idx >= self.materials.len() {
            self.materials.resize(idx + 1, Material::default());
        }
        self.materials[idx] = material;
        // Re-upload on next render
        self.material_table = MaterialTable::upload(&self.ctx.device, &self.materials);
    }

    /// Set all materials at once.
    pub fn set_materials(&mut self, materials: &[Material]) {
        self.materials = materials.to_vec();
        self.material_table = MaterialTable::upload(&self.ctx.device, &self.materials);
    }

    /// Get the current materials.
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    // ── Lights ─────────────────────────────────────────────────────────────

    /// Add a light and return its index.
    pub fn add_light(&mut self, light: Light) -> usize {
        let idx = self.world_lights.len();
        self.world_lights.push(light);
        self.light_buffer = LightBuffer::upload(&self.ctx.device, &self.world_lights);
        idx
    }

    /// Remove a light by index.
    pub fn remove_light(&mut self, index: usize) {
        if index < self.world_lights.len() {
            self.world_lights.remove(index);
            self.light_buffer = LightBuffer::upload(&self.ctx.device, &self.world_lights);
        }
    }

    /// Get a shared reference to a light.
    pub fn light(&self, index: usize) -> Option<&Light> {
        self.world_lights.get(index)
    }

    /// Get a mutable reference to a light.
    pub fn light_mut(&mut self, index: usize) -> Option<&mut Light> {
        self.world_lights.get_mut(index)
    }

    /// Replace all lights.
    pub fn set_lights(&mut self, lights: Vec<Light>) {
        self.world_lights = lights;
        self.light_buffer = LightBuffer::upload(&self.ctx.device, &self.world_lights);
    }

    /// Get all lights.
    pub fn lights(&self) -> &[Light] {
        &self.world_lights
    }

    // ── Environment ────────────────────────────────────────────────────────

    /// Set the environment profile.
    pub fn set_environment(&mut self, env: crate::environment::EnvironmentProfile) {
        self.environment = env;
    }

    /// Get the environment profile.
    pub fn environment(&self) -> &crate::environment::EnvironmentProfile {
        &self.environment
    }

    /// Get a mutable reference to the environment profile.
    pub fn environment_mut(&mut self) -> &mut crate::environment::EnvironmentProfile {
        &mut self.environment
    }

    // ── Debug / quality ────────────────────────────────────────────────────

    /// Set the debug visualization mode.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.shade_debug_mode = mode;
    }

    /// Get the current debug mode.
    pub fn debug_mode(&self) -> u32 {
        self.shade_debug_mode
    }

    // ── GPU readback ───────────────────────────────────────────────────────

    /// Capture a screenshot of the current frame as RGBA8 pixels.
    pub fn screenshot(&self) -> Vec<u8> {
        // This requires a frame to have been rendered. The pixels are in the
        // cosmetics output texture. We need to copy it to a readback buffer.
        let iw = self.internal_width;
        let ih = self.internal_height;
        let bytes_per_pixel = 4u32;
        let unpadded_row = iw * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screenshot"),
            });

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
                    rows_per_image: Some(ih),
                },
            },
            wgpu::Extent3d {
                width: iw,
                height: ih,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let pixel_count = (iw * ih) as usize;
        let mut rgba8 = vec![0u8; pixel_count * 4];

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..ih as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * iw as usize * 4;
                let row_bytes = iw as usize * 4;
                rgba8[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&data[src_offset..src_offset + row_bytes]);
            }
            drop(data);
            self.readback_buffer.unmap();
        } else {
            self.readback_buffer.unmap();
        }

        rgba8
    }

    // ── Stats ──────────────────────────────────────────────────────────────

    /// Get the current frame index.
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Get the internal render resolution.
    pub fn internal_resolution(&self) -> (u32, u32) {
        (self.internal_width, self.internal_height)
    }

    /// Get the display resolution.
    pub fn display_resolution(&self) -> (u32, u32) {
        (self.display_width, self.display_height)
    }

    // ── GPU access (advanced) ──────────────────────────────────────────────

    /// Get a reference to the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.device
    }

    /// Get a reference to the GPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.ctx.queue
    }

    /// Get the surface format used for presentation.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Get a reference to the final LDR output texture view (for compositor integration).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.cosmetics.output_view
    }

    /// Get a reference to the blit pass (for drawing to a target view).
    pub fn blit_pass(&self) -> &BlitPass {
        &self.blit
    }
}

/// Transform a local-space AABB to world-space using a baked world transform.
fn transform_aabb(aabb: &Aabb, wt: &transform_bake::WorldTransform) -> Aabb {
    let smin = aabb.min * wt.scale;
    let smax = aabb.max * wt.scale;
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
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // CPU-only tests — no GPU required

    #[test]
    fn renderer_config_default() {
        let config = RendererConfig::default();
        assert_eq!(config.internal_width, 960);
        assert_eq!(config.internal_height, 540);
        assert_eq!(config.display_width, 1280);
        assert_eq!(config.display_height, 720);
    }
}
