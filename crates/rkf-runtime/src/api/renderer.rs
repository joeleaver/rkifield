//! Renderer — the GPU rendering pipeline.
//!
//! Owns all GPU resources (brick pool buffer, G-buffer, 21 compute passes,
//! readback buffers). Takes a [`&World`](super::world::World) per frame — does
//! NOT own the scene.
//!
//! # Features
//!
//! - Full 21-pass compute pipeline (ray march, shading, GI, volumetrics, post-FX)
//! - Wireframe overlay rendering (selection highlights, gizmos, debug)
//! - GPU pick readback (object ID at pixel)
//! - GPU brush hit readback (world position + object ID at pixel)
//! - Environment-driven rendering (sun, fog, clouds, post-processing)
//! - Offscreen rendering with compositor integration
//!
//! # Usage
//!
//! ```ignore
//! let mut renderer = Renderer::new(&window, RendererConfig::default());
//! renderer.add_light(Light::directional([0.5, 1.0, 0.3], [1.0, 0.95, 0.85], 3.0, true));
//! // Per frame:
//! renderer.render_to_surface(&world, &surface);
//! ```

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::window::Window;

use rkf_core::aabb::Aabb;
use rkf_core::material::Material;
use rkf_core::transform_bake;
use rkf_core::transform_flatten::flatten_object;
use rkf_render::material_table::{create_test_materials, MaterialTable};
use rkf_render::radiance_inject::{InjectUniforms, RadianceInjectPass};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass, GBuffer,
    GodRaysBlurPass, GpuObject, GpuSceneV2, Light, LightBuffer, LineVertex, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass, VolShadowPass,
    VolUpscalePass, WireframePass, COARSE_VOXEL_SIZE,
};

use super::error::WorldError;
use super::world::World;
use crate::components::CameraComponent;

/// Offscreen render target format for the compositor path.
///
/// sRGB variant so the blit pass's linear-to-sRGB conversion happens in hardware.
const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// ── Configuration ────────────────────────────────────────────────────────────

/// Renderer configuration.
pub struct RendererConfig {
    /// Internal render resolution width (default 960).
    pub internal_width: u32,
    /// Internal render resolution height (default 540).
    pub internal_height: u32,
    /// Display/viewport resolution width (default 1280).
    pub display_width: u32,
    /// Display/viewport resolution height (default 720).
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

/// Rendering environment configuration.
///
/// Contains all parameters that drive the rendering pipeline: sun, atmosphere,
/// fog, clouds, and post-processing. Set via [`Renderer::set_render_environment`].
#[derive(Debug, Clone)]
pub struct RenderEnvironment {
    // Atmosphere
    /// Sun direction (normalized).
    pub sun_direction: Vec3,
    /// Sun base color (before tinting by elevation).
    pub sun_color: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Rayleigh scattering scale (atmosphere).
    pub rayleigh_scale: f32,
    /// Mie scattering scale (atmosphere).
    pub mie_scale: f32,
    /// Whether analytic atmosphere sky is enabled.
    pub atmosphere_enabled: bool,

    // Fog
    /// Fog base color.
    pub fog_color: Vec3,
    /// Fog density (0 = disabled).
    pub fog_density: f32,
    /// Fog height falloff rate.
    pub fog_height_falloff: f32,
    /// Ambient dust particle density for god rays.
    pub ambient_dust: f32,
    /// Dust asymmetry parameter for Henyey-Greenstein phase function.
    pub dust_asymmetry: f32,

    // Clouds
    /// Cloud rendering settings.
    pub cloud_settings: rkf_render::CloudSettings,

    // Post-processing
    /// Bloom threshold (HDR luminance).
    pub bloom_threshold: f32,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// Tone map mode (0 = ACES, 1 = AgX).
    pub tone_map_mode: u32,
    /// Exposure compensation (EV).
    pub exposure: f32,
    /// Whether depth of field is enabled.
    pub dof_enabled: bool,
    /// DoF focus distance.
    pub dof_focus_distance: f32,
    /// DoF focus range (transition zone).
    pub dof_focus_range: f32,
    /// DoF maximum circle of confusion.
    pub dof_max_coc: f32,
    /// Sharpen filter strength.
    pub sharpen_strength: f32,
    /// Motion blur intensity.
    pub motion_blur_intensity: f32,
    /// God rays radial blur intensity.
    pub god_rays_intensity: f32,
    /// Vignette darkening intensity.
    pub vignette_intensity: f32,
    /// Film grain intensity.
    pub grain_intensity: f32,
    /// Chromatic aberration strength.
    pub chromatic_aberration: f32,
}

impl Default for RenderEnvironment {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.5, 1.0, 0.3).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.85),
            sun_intensity: 3.0,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            atmosphere_enabled: true,
            fog_color: Vec3::new(0.7, 0.75, 0.8),
            fog_density: 0.0,
            fog_height_falloff: 0.1,
            ambient_dust: 0.005,
            dust_asymmetry: 0.3,
            cloud_settings: rkf_render::CloudSettings::default(),
            bloom_threshold: 1.0,
            bloom_intensity: 0.3,
            tone_map_mode: 0,
            exposure: 0.0,
            dof_enabled: false,
            dof_focus_distance: 5.0,
            dof_focus_range: 3.0,
            dof_max_coc: 0.02,
            sharpen_strength: 0.0,
            motion_blur_intensity: 0.0,
            god_rays_intensity: 0.0,
            vignette_intensity: 0.0,
            grain_intensity: 0.0,
            chromatic_aberration: 0.0,
        }
    }
}

/// Result from a GPU brush hit readback.
#[derive(Debug, Clone)]
pub struct BrushHitResult {
    /// World-space position of the hit.
    pub position: Vec3,
    /// Object ID at the hit point.
    pub object_id: u32,
}

// ── Renderer ─────────────────────────────────────────────────────────────────

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

    // Wireframe overlay
    wireframe_pass: WireframePass,

    // Offscreen rendering (compositor path)
    offscreen_texture: Option<wgpu::Texture>,
    offscreen_view: Option<wgpu::TextureView>,
    offscreen_blit: Option<BlitPass>,

    // GPU readback
    readback_buffer: wgpu::Buffer,
    pick_readback_buffer: wgpu::Buffer,
    brush_readback_buffer: wgpu::Buffer,

    // CPU state
    camera: Camera,
    world_lights: Vec<Light>,
    materials: Vec<Material>,
    light_buffer: LightBuffer,
    render_env: RenderEnvironment,
    frame_index: u32,
    prev_vp: [[f32; 4]; 4],
    shade_debug_mode: u32,
    accumulated_time: f32,

    // Config
    internal_width: u32,
    internal_height: u32,
    display_width: u32,
    display_height: u32,
    surface_format: wgpu::TextureFormat,
}

impl Renderer {
    /// Create a new Renderer with its own GPU device, attached to a window.
    ///
    /// Returns the Renderer and the configured `wgpu::Surface` for frame
    /// presentation. Pass the surface to [`render_to_surface`] each frame.
    pub fn new(window: Arc<Window>, config: RendererConfig) -> (Self, wgpu::Surface<'static>) {
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

        (Self::build(ctx, surface_format, config, false), surface)
    }

    /// Create a new Renderer using a shared GPU device (e.g. from an editor compositor).
    ///
    /// Sets up an offscreen render target at `display_width x display_height` for
    /// compositor integration. Use [`offscreen_view`] to get the texture view.
    pub fn from_shared(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: RendererConfig,
    ) -> Self {
        let ctx = RenderContext::from_shared(device, queue);
        Self::build(ctx, surface_format, config, true)
    }

    fn build(
        ctx: RenderContext,
        surface_format: wgpu::TextureFormat,
        config: RendererConfig,
        offscreen: bool,
    ) -> Self {
        let iw = config.internal_width;
        let ih = config.internal_height;
        let dw = config.display_width.max(64);
        let dh = config.display_height.max(64);

        // Create empty brick pool buffer.
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);

        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool_shadow"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gbuffer = GBuffer::new(&ctx.device, iw, ih);
        let tile_cull = TileObjectCullPass::new(&ctx.device, &gpu_scene, iw, ih);

        let coarse_field =
            CoarseField::from_scene_aabbs(&ctx.device, &[], COARSE_VOXEL_SIZE, 1.0);

        let ray_march =
            RayMarchPass::new(&ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field);
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights — start with a default sun.
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

        // Blit target format depends on mode.
        let blit_format = if offscreen { OFFSCREEN_FORMAT } else { surface_format };
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, blit_format);

        // Wireframe pass (targets offscreen or swapchain format).
        let wireframe_format = if offscreen { OFFSCREEN_FORMAT } else { surface_format };
        let wireframe_pass = WireframePass::new(&ctx.device, wireframe_format);

        // Offscreen render target (compositor path only).
        let (offscreen_texture, offscreen_view, offscreen_blit) = if offscreen {
            let (tex, view) = create_offscreen_target(&ctx.device, dw, dh);
            let os_blit = BlitPass::new(&ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT);
            (Some(tex), Some(view), Some(os_blit))
        } else {
            (None, None, None)
        };

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Readback buffers
        let readback_width = if offscreen { dw } else { iw };
        let readback_height = if offscreen { dh } else { ih };
        let readback_buffer = create_readback_buffer(&ctx.device, readback_width, readback_height);

        // Pick readback — 1 pixel from R32Uint material texture (256-byte aligned).
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback — 1 pixel from Rgba32Float position (16 bytes, 256-aligned).
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
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
            wireframe_pass,
            offscreen_texture,
            offscreen_view,
            offscreen_blit,
            readback_buffer,
            pick_readback_buffer,
            brush_readback_buffer,
            camera,
            world_lights,
            materials,
            light_buffer,
            render_env: RenderEnvironment::default(),
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            accumulated_time: 0.0,
            internal_width: iw,
            internal_height: ih,
            display_width: dw,
            display_height: dh,
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
    /// Runs the full 21-pass compute pipeline, then blits to the offscreen render
    /// target at viewport resolution. Returns the offscreen texture view for the
    /// compositor. If no offscreen target exists (surface-based mode), returns the
    /// cosmetics compute output directly.
    pub fn render_offscreen(&mut self, world: &World) -> &wgpu::TextureView {
        if self.offscreen_view.is_some() {
            // Offscreen path: render compute passes, then blit to offscreen target.
            self.render_frame(world, None);
            // Blit from cosmetics output to offscreen sRGB target.
            if let Some(ref offscreen_blit) = self.offscreen_blit {
                if let Some(ref offscreen_view) = self.offscreen_view {
                    let mut encoder = self.ctx.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("offscreen_blit"),
                        },
                    );
                    offscreen_blit.draw(&mut encoder, offscreen_view);
                    self.ctx.queue.submit(std::iter::once(encoder.finish()));
                }
            }
            self.offscreen_view.as_ref().unwrap()
        } else {
            // No offscreen target — return compute output directly.
            self.render_frame(world, None);
            &self.cosmetics.output_view
        }
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
                    [0.0; 3],
                    [0.0; 3],
                ));
                bvh_pairs.push((gpu_idx, world_aabb));
            }
        }

        // Upload brick pool if needed
        let pool_data: &[u8] = bytemuck::cast_slice(world.brick_pool().as_slice());
        if !pool_data.is_empty() {
            if pool_data.len() as u64 > self.brick_pool_buffer.size() {
                self.brick_pool_buffer =
                    self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("brick_pool"),
                        size: pool_data.len() as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
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

        // Upload GPU objects + BVH
        self.gpu_scene
            .upload_objects(&self.ctx.device, &self.ctx.queue, &gpu_objects);
        let bvh = rkf_core::Bvh::build(&bvh_pairs);
        self.gpu_scene
            .upload_bvh(&self.ctx.device, &self.ctx.queue, &bvh);

        // Camera uniforms
        let cam_uniforms = self.camera.uniforms(iw, ih, self.frame_index, self.prev_vp);
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        let scene_uniforms = SceneUniforms {
            num_objects: gpu_objects.len() as u32,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene
            .update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // ── Compute environment-derived values ───────────────────────────────

        let env = &self.render_env;
        let sun_dir = env.sun_direction.normalize();

        // Tint sun color based on elevation (Rayleigh extinction approximation).
        let sun_elevation = sun_dir.y.asin();
        let tinted_color = {
            let path = (1.0 / sun_elevation.max(0.02).sin()).min(12.0);
            let tau = Vec3::new(0.02, 0.06, 0.15);
            let extinction = Vec3::new(
                (-tau.x * path).exp(),
                (-tau.y * path).exp(),
                (-tau.z * path).exp(),
            );
            env.sun_color * extinction
        };
        let sun_color_tinted = tinted_color * env.sun_intensity;
        let sun_dir_arr = [sun_dir.x, sun_dir.y, sun_dir.z];

        let fog_density = if env.fog_density > 0.0 { env.fog_density } else { 0.0 };
        let fog_alpha = if fog_density > 0.0 { 1.0 } else { 0.0 };

        // Camera basis for shade + vol march
        let fwd = self.camera.forward();
        let fov_rad = self.camera.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = iw as f32 / ih as f32;
        let right = self.camera.right() * half_fov_tan * aspect;
        let up = self.camera.up() * half_fov_tan;
        let cam = self.camera.position;

        // Synthesize directional sun light.
        let sun_light = Light {
            light_type: 0,
            pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
            dir_x: sun_dir_arr[0],
            dir_y: sun_dir_arr[1],
            dir_z: sun_dir_arr[2],
            color_r: sun_color_tinted.x,
            color_g: sun_color_tinted.y,
            color_b: sun_color_tinted.z,
            intensity: 1.0,
            range: 0.0,
            inner_angle: 0.0,
            outer_angle: 0.0,
            cookie_index: -1,
            shadow_caster: 1,
        };

        // Camera-relative point/spot lights.
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

        // Shade uniforms
        self.shading_pass.update_uniforms(
            &self.ctx.queue,
            &ShadeUniforms {
                debug_mode: self.shade_debug_mode,
                num_lights: total_lights,
                _pad0: 0,
                shadow_budget_k: 0,
                camera_pos: [cam.x, cam.y, cam.z, 0.0],
                sun_dir: [sun_dir_arr[0], sun_dir_arr[1], sun_dir_arr[2], env.sun_intensity],
                sun_color: [tinted_color.x, tinted_color.y, tinted_color.z, 0.0],
                sky_params: [
                    env.rayleigh_scale,
                    env.mie_scale,
                    if env.atmosphere_enabled { 1.0 } else { 0.0 },
                    0.0,
                ],
                cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
                cam_right: [right.x, right.y, right.z, 0.0],
                cam_up: [up.x, up.y, up.z, 0.0],
            },
        );

        // Debug view
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

        self.radiance_volume
            .update_center(&self.ctx.queue, [cam.x, cam.y, cam.z]);

        self.radiance_inject.update_uniforms(
            &self.ctx.queue,
            &InjectUniforms {
                num_lights: total_lights,
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
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &env.cloud_settings,
            self.accumulated_time,
        );
        self.vol_march
            .set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow
            .set_cloud_params(&self.ctx.queue, &cloud_params);

        self.vol_shadow.dispatch(
            &mut encoder,
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
            &self.coarse_field.bind_group,
        );
        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir_arr,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.cloud_shadow.dispatch_only(&mut encoder);

        let sc = [sun_color_tinted.x, sun_color_tinted.y, sun_color_tinted.z];
        let fc = [env.fog_color.x, env.fog_color.y, env.fog_color.z];
        let half_w = iw / 2;
        let half_h = ih / 2;
        let vol_params = rkf_render::VolMarchParams {
            cam_pos: [cam.x, cam.y, cam.z, 0.0],
            cam_forward: [fwd.x, fwd.y, fwd.z, 0.0],
            cam_right: [right.x, right.y, right.z, 0.0],
            cam_up: [up.x, up.y, up.z, 0.0],
            sun_dir: [sun_dir_arr[0], sun_dir_arr[1], sun_dir_arr[2], 0.0],
            sun_color: [sc[0], sc[1], sc[2], 0.0],
            width: half_w,
            height: half_h,
            full_width: iw,
            full_height: ih,
            max_steps: 32,
            step_size: 2.0,
            near: 0.5,
            far: 200.0,
            fog_color: [fc[0], fc[1], fc[2], fog_alpha],
            fog_height: [fog_density, -0.5, env.fog_height_falloff, 0.0],
            fog_distance: [0.0, 0.01, env.ambient_dust, env.dust_asymmetry],
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
        // Project sun to screen UV for radial blur god rays.
        {
            let cam_fwd = self.camera.forward();
            let sun_dot = sun_dir.dot(cam_fwd);
            let (sun_uv_x, sun_uv_y) = if sun_dot > 0.0 {
                let ndc_x = sun_dir.dot(right) / sun_dot;
                let ndc_y = -sun_dir.dot(up) / sun_dot;
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

    /// Copy a camera entity's state to the viewport (rendering) camera.
    ///
    /// Reads the entity's position from `World` and its `CameraComponent`
    /// from ECS, then sets the renderer's camera position, orientation,
    /// and field of view to match.
    pub fn snap_camera_to(
        &mut self,
        world: &World,
        entity: super::entity::Entity,
    ) -> Result<(), WorldError> {
        let pos = world.position(entity)?;
        let cam = world
            .get::<CameraComponent>(entity)
            .map_err(|_| WorldError::MissingComponent(entity, "CameraComponent"))?;
        self.camera.position = pos.to_vec3();
        self.camera.yaw = cam.yaw.to_radians();
        self.camera.pitch = cam.pitch.to_radians();
        self.camera.fov_degrees = cam.fov_degrees;
        Ok(())
    }

    /// Camera-relative view-projection matrix for overlay rendering.
    pub fn view_projection(&self) -> Mat4 {
        self.camera
            .view_projection(self.internal_width, self.internal_height)
    }

    /// Current camera position in world space.
    pub fn camera_position(&self) -> Vec3 {
        self.camera.position
    }

    // ── Materials ──────────────────────────────────────────────────────────

    /// Set a material at the given index.
    pub fn set_material(&mut self, index: u16, material: Material) {
        let idx = index as usize;
        if idx >= self.materials.len() {
            self.materials.resize(idx + 1, Material::default());
        }
        self.materials[idx] = material;
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

    /// Set the render environment (sun, fog, clouds, post-processing).
    ///
    /// This also applies post-processing settings to the GPU passes immediately.
    pub fn set_render_environment(&mut self, env: RenderEnvironment) {
        self.apply_post_process_settings(&env);
        self.render_env = env;
    }

    /// Get the current render environment.
    pub fn render_environment(&self) -> &RenderEnvironment {
        &self.render_env
    }

    /// Get a mutable reference to the render environment.
    pub fn render_environment_mut(&mut self) -> &mut RenderEnvironment {
        &mut self.render_env
    }

    /// Apply post-processing settings from the render environment to GPU passes.
    fn apply_post_process_settings(&mut self, env: &RenderEnvironment) {
        let queue = &self.ctx.queue;

        // Bloom
        let t = env.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, env.bloom_intensity);

        // Tone mapping
        let mode = if env.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, env.exposure);

        // Depth of field
        if env.dof_enabled {
            self.dof.update_focus(
                queue,
                env.dof_focus_distance,
                env.dof_focus_range,
                env.dof_max_coc,
            );
        } else {
            self.dof.update_focus(
                queue,
                env.dof_focus_distance,
                env.dof_focus_range,
                0.0,
            );
        }

        // Other post-FX
        self.sharpen.set_strength(queue, env.sharpen_strength);
        self.motion_blur.set_intensity(queue, env.motion_blur_intensity);
        self.god_rays_blur.set_intensity(queue, env.god_rays_intensity);
        self.cosmetics.set_vignette(queue, env.vignette_intensity);
        self.cosmetics.set_grain(queue, env.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, env.chromatic_aberration);
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

    // ── Wireframe ─────────────────────────────────────────────────────────

    /// Draw wireframe lines onto the offscreen render target.
    ///
    /// Call after [`render_offscreen`]. Positions should be in camera-relative
    /// world space. Uses the camera's view-projection matrix for transformation.
    pub fn draw_wireframe(&mut self, vertices: &[LineVertex]) {
        if vertices.is_empty() {
            return;
        }
        let target = match self.offscreen_view.as_ref() {
            Some(v) => v,
            None => return,
        };
        let vp_matrix = self.view_projection();
        let viewport = (
            0.0,
            0.0,
            self.display_width as f32,
            self.display_height as f32,
        );
        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("wireframe"),
            },
        );
        self.wireframe_pass.draw(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            target,
            vp_matrix,
            viewport,
            vertices,
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    // ── GPU readback ───────────────────────────────────────────────────────

    /// Capture a screenshot of the current frame as RGBA8 pixels.
    ///
    /// Returns pixels at the offscreen viewport resolution (if offscreen) or
    /// at internal resolution (if surface-based).
    pub fn screenshot(&self) -> Vec<u8> {
        let (source_texture, w, h) = if let Some(ref tex) = self.offscreen_texture {
            (tex, self.display_width, self.display_height)
        } else {
            (&self.cosmetics.output_texture, self.internal_width, self.internal_height)
        };

        let bytes_per_pixel = 4u32;
        let unpadded_row = w * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("screenshot"),
            },
        );

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
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

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let pixel_count = (w * h) as usize;
        let mut rgba8 = vec![0u8; pixel_count * 4];

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..h as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * w as usize * 4;
                let row_bytes = w as usize * 4;
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

    /// GPU pick readback — returns the object ID at the given pixel coordinate.
    ///
    /// Coordinates are in internal render resolution space. Returns `None` if
    /// the pixel is background (no object hit).
    pub fn pick(&self, x: u32, y: u32) -> Option<u32> {
        let px = x.min(self.internal_width.saturating_sub(1));
        let py = y.min(self.internal_height.saturating_sub(1));

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("pick_readback"),
            },
        );
        encoder.copy_texture_to_buffer(
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
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.pick_readback_buffer.slice(..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let object_id = packed >> 24; // bits 24-31
            drop(data);
            self.pick_readback_buffer.unmap();
            if object_id > 0 {
                Some(object_id)
            } else {
                None
            }
        } else {
            self.pick_readback_buffer.unmap();
            None
        }
    }

    /// GPU brush hit readback — returns world position + object ID at pixel.
    ///
    /// Coordinates are in internal render resolution space. Returns `None` if
    /// the pixel is background (no geometry hit).
    pub fn brush_hit(&self, x: u32, y: u32) -> Option<BrushHitResult> {
        let bx = x.min(self.internal_width.saturating_sub(1));
        let by = y.min(self.internal_height.saturating_sub(1));

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("brush_readback"),
            },
        );

        // Copy 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
        encoder.copy_texture_to_buffer(
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
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        // Also copy 1 pixel from material G-buffer for object_id.
        encoder.copy_texture_to_buffer(
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
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read position (4xf32 = 16 bytes).
        let pos_slice = self.brush_readback_buffer.slice(..16);
        let (tx_pos, rx_pos) = std::sync::mpsc::channel();
        pos_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_pos.send(r);
        });
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
        mat_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_mat.send(r);
        });
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
            Some(BrushHitResult {
                position: Vec3::new(hit_pos[0], hit_pos[1], hit_pos[2]),
                object_id,
            })
        } else {
            None
        }
    }

    // ── Resize ────────────────────────────────────────────────────────────

    /// Resize the internal render resolution.
    ///
    /// Recreates all size-dependent GPU resources.
    pub fn resize_render(&mut self, width: u32, height: u32) {
        let width = width.max(64);
        let height = height.max(64);
        if width == self.internal_width && height == self.internal_height {
            return;
        }
        log::info!(
            "Render resolution: {}x{} -> {}x{}",
            self.internal_width,
            self.internal_height,
            width,
            height
        );
        self.internal_width = width;
        self.internal_height = height;

        let device = &self.ctx.device;

        // Core passes
        self.gbuffer = GBuffer::new(device, width, height);
        self.tile_cull = TileObjectCullPass::new(device, &self.gpu_scene, width, height);
        self.debug_view = DebugViewPass::new(device, &self.gbuffer);
        self.shading_pass = ShadingPass::new(
            device,
            &self.gbuffer,
            &self.gpu_scene,
            &self.light_buffer,
            &self.coarse_field,
            &self.radiance_volume,
            &self.material_table.buffer,
            width,
            height,
        );

        // Volumetric pipeline
        let half_w = width / 2;
        let half_h = height / 2;
        self.vol_march = VolMarchPass::new(
            device,
            &self.ctx.queue,
            &self.gbuffer.position_view,
            &self.vol_shadow.shadow_view,
            &self.cloud_shadow.shadow_view,
            half_w,
            half_h,
            width,
            height,
        );
        self.vol_upscale = VolUpscalePass::new(
            device,
            &self.vol_march.output_view,
            &self.gbuffer.position_view,
            width,
            height,
            half_w,
            half_h,
        );
        self.vol_composite = VolCompositePass::new(
            device,
            &self.shading_pass.hdr_view,
            &self.vol_upscale.output_view,
            width,
            height,
        );

        // Post-processing pipeline
        self.god_rays_blur = GodRaysBlurPass::new(
            device,
            &self.vol_composite.output_view,
            &self.gbuffer.position_view,
            width,
            height,
        );
        self.bloom = BloomPass::new(device, &self.god_rays_blur.output_view, width, height);
        self.auto_exposure =
            AutoExposurePass::new(device, &self.god_rays_blur.output_view, width, height);
        self.dof = DofPass::new(
            device,
            &self.god_rays_blur.output_view,
            &self.gbuffer.position_view,
            width,
            height,
        );
        self.motion_blur = MotionBlurPass::new(
            device,
            &self.dof.output_view,
            &self.gbuffer.motion_view,
            width,
            height,
        );
        self.bloom_composite = BloomCompositePass::new(
            device,
            &self.motion_blur.output_view,
            self.bloom.mip_views(),
            width,
            height,
        );
        self.tone_map = ToneMapPass::new_with_exposure(
            device,
            &self.bloom_composite.output_view,
            width,
            height,
            Some(self.auto_exposure.get_exposure_buffer()),
        );
        self.color_grade = ColorGradePass::new(
            device,
            &self.ctx.queue,
            &self.tone_map.ldr_view,
            width,
            height,
        );
        self.cosmetics = CosmeticsPass::new(device, &self.color_grade.output_view, width, height);
        self.sharpen = SharpenPass::new(
            device,
            &self.vol_composite.output_view,
            &self.gbuffer,
            width,
            height,
        );

        // Update blit source
        self.blit.update_source(device, &self.cosmetics.output_view);
        if let Some(ref mut os_blit) = self.offscreen_blit {
            os_blit.update_source(device, &self.cosmetics.output_view);
        }

        // Re-apply post-processing settings
        let env = self.render_env.clone();
        self.apply_post_process_settings(&env);
    }

    /// Resize the viewport (offscreen render target) and internal resolution.
    ///
    /// Called when the viewport dimensions change (e.g., panel resize).
    /// Returns the new offscreen `TextureView` for the compositor to use,
    /// or `None` if the viewport didn't actually change.
    pub fn resize_viewport(
        &mut self,
        viewport_w: u32,
        viewport_h: u32,
        render_scale: f32,
    ) -> Option<wgpu::TextureView> {
        let vp_w = viewport_w.max(64);
        let vp_h = viewport_h.max(64);
        if vp_w == self.display_width && vp_h == self.display_height {
            return None;
        }

        log::info!(
            "Viewport resize: {}x{} -> {}x{}",
            self.display_width,
            self.display_height,
            vp_w,
            vp_h
        );
        self.display_width = vp_w;
        self.display_height = vp_h;

        // Recreate offscreen target.
        let (tex, view) = create_offscreen_target(&self.ctx.device, vp_w, vp_h);
        self.offscreen_texture = Some(tex);
        self.offscreen_view = Some(view);

        // Recreate readback buffer at new viewport resolution.
        self.readback_buffer = create_readback_buffer(&self.ctx.device, vp_w, vp_h);

        // Resize internal render resolution.
        let internal_w = ((vp_w as f32 * render_scale) as u32).max(64);
        let internal_h = ((vp_h as f32 * render_scale) as u32).max(64);
        self.resize_render(internal_w, internal_h);

        // Return new view for compositor.
        self.offscreen_view.clone()
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

    /// Get the display/viewport resolution.
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

    /// Get the final LDR compute output texture view (before offscreen blit).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.cosmetics.output_view
    }

    /// Get the offscreen texture view for compositor integration.
    ///
    /// Returns `None` if the renderer is using the surface-based path.
    pub fn offscreen_view(&self) -> Option<&wgpu::TextureView> {
        self.offscreen_view.as_ref()
    }

    /// Get a reference to the blit pass (for custom target rendering).
    pub fn blit_pass(&self) -> &BlitPass {
        &self.blit
    }

    /// Access the GPU scene (for direct brick pool / brick map operations).
    pub fn gpu_scene(&self) -> &GpuSceneV2 {
        &self.gpu_scene
    }

    /// Access the GPU scene mutably.
    pub fn gpu_scene_mut(&mut self) -> &mut GpuSceneV2 {
        &mut self.gpu_scene
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

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

/// Create a readback buffer sized for the given dimensions.
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

    #[test]
    fn renderer_config_default() {
        let config = RendererConfig::default();
        assert_eq!(config.internal_width, 960);
        assert_eq!(config.internal_height, 540);
        assert_eq!(config.display_width, 1280);
        assert_eq!(config.display_height, 720);
    }

    #[test]
    fn render_environment_default() {
        let env = RenderEnvironment::default();
        assert_eq!(env.sun_intensity, 3.0);
        assert!(env.atmosphere_enabled);
        assert_eq!(env.fog_density, 0.0);
        assert_eq!(env.tone_map_mode, 0);
        assert!(!env.dof_enabled);
    }

    #[test]
    fn brush_hit_result_debug() {
        let hit = BrushHitResult {
            position: Vec3::new(1.0, 2.0, 3.0),
            object_id: 42,
        };
        let s = format!("{:?}", hit);
        assert!(s.contains("42"));
    }

    // snap_camera_to tests — verify World-side prerequisites.
    // Full integration (GPU + snap) is tested via rkf-testbed.

    #[test]
    fn snap_camera_non_camera_entity_would_error() {
        use rkf_core::scene_node::SdfPrimitive;
        let mut world = World::new("test");
        let e = world
            .spawn("cube")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(1)
            .build();
        // Attempting to snap to a non-camera entity fails on get::<CameraComponent>
        assert!(!world.has::<CameraComponent>(e));
    }

    #[test]
    fn camera_entity_has_component() {
        use rkf_core::WorldPosition;
        let mut world = World::new("test");
        let cam = world.spawn_camera("Main", WorldPosition::default(), 45.0, -15.0, 75.0);
        assert!(world.has::<CameraComponent>(cam));
        let c = world.get::<CameraComponent>(cam).unwrap();
        assert!((c.fov_degrees - 75.0).abs() < 1e-6);
        assert!((c.yaw - 45.0).abs() < 1e-6);
        assert!((c.pitch - -15.0).abs() < 1e-6);
    }

    #[test]
    fn camera_position_accessible() {
        use rkf_core::WorldPosition;
        let mut world = World::new("test");
        let pos = WorldPosition::new(glam::IVec3::new(1, 2, 3), Vec3::new(0.5, 0.5, 0.5));
        let cam = world.spawn_camera("Cam", pos, 0.0, 0.0, 60.0);
        assert_eq!(world.position(cam).unwrap(), pos);
    }

    #[test]
    fn snap_camera_requires_camera_component() {
        // Verify a non-camera entity lacks CameraComponent
        let mut world = World::new("test");
        let e = world.spawn("ecs_only").build();
        let result = world.get::<CameraComponent>(e);
        assert!(result.is_err());
    }
}
