//! Offscreen rendering path for the editor engine (compositor/rinch integration).

use std::sync::{Arc, Mutex};
use glam::Vec3;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, BrushOverlay, Camera,
    CloudShadowPass, CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass,
    DofPass, GBuffer, GodRaysBlurPass, GpuScene, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::MaterialTable;
use rkf_core::material_library::MaterialLibrary;
use rkf_core::brick_pool::BrickPool;
use rkf_core::brick_map::BrickMapAllocator;
use super::EditorEngine;
use super::OFFSCREEN_FORMAT;
use crate::engine_viewport::RENDER_SCALE;

/// Default brick pool capacity for an empty engine (resized on scene load).
const DEFAULT_BRICK_POOL_CAPACITY: u32 = 4096;
use crate::automation::SharedState;

impl EditorEngine {
    /// Create an engine with its own dedicated wgpu device.
    ///
    /// Renders to an offscreen texture. The caller reads back pixels via
    /// [`readback_frame`] and submits them to the compositor's CPU pixel
    /// path (`SurfaceWriter::submit_frame`). This eliminates GPU contention
    /// between the engine and the compositor.
    ///
    /// Returns the engine with an empty scene. The caller loads content via
    /// the project/scene loading path (hecs is the single source of truth).
    pub fn new_headless(
        viewport_width: u32,
        viewport_height: u32,
        shared_state: Arc<Mutex<SharedState>>,
    ) -> Self {
        let ctx = RenderContext::new_headless();

        // Compute internal render resolution from viewport size.
        let vp_w = viewport_width.max(64);
        let vp_h = viewport_height.max(64);
        let internal_w = ((vp_w as f32 * RENDER_SCALE) as u32).max(64);
        let internal_h = ((vp_h as f32 * RENDER_SCALE) as u32).max(64);

        // Empty brick pool — content loaded via project/scene path.
        let empty_pool = BrickPool::new(DEFAULT_BRICK_POOL_CAPACITY);
        let empty_alloc = BrickMapAllocator::new();

        // Upload empty brick pool to GPU.
        let pool_data: &[u8] = bytemuck::cast_slice(empty_pool.as_slice());
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: (DEFAULT_BRICK_POOL_CAPACITY as u64 * std::mem::size_of::<rkf_core::Brick>() as u64).max(8),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !pool_data.is_empty() {
            ctx.queue.write_buffer(&brick_pool_buffer, 0, pool_data);
        }

        let gpu_scene = GpuScene::new(&ctx.device, brick_pool_buffer);

        let gbuffer = GBuffer::new(&ctx.device, internal_w, internal_h);
        let tile_cull = TileObjectCullPass::new(
            &ctx.device, &gpu_scene, internal_w, internal_h,
        );

        // Coarse acceleration field — empty, populated when scene loads.
        let empty_aabbs: Vec<(Vec3, Vec3)> = Vec::new();
        let coarse_field = CoarseField::from_scene_aabbs(
            &ctx.device, &empty_aabbs, COARSE_VOXEL_SIZE, 1.0,
        );
        coarse_field.upload(&ctx.queue, Vec3::ZERO);

        let ray_march = RayMarchPass::new(
            &ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field,
        );
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        // Material library — load from engine library palette.
        let palette_path = rkf_runtime::project::engine_library_dir()
            .join("materials/default.rkmatlib");
        let material_library = match MaterialLibrary::load_palette(&palette_path) {
            Ok(lib) => {
                log::info!("Material library loaded from {}", palette_path.display());
                lib
            }
            Err(e) => {
                log::warn!("Failed to load material palette from {}: {e}", palette_path.display());
                MaterialLibrary::new(16)
            }
        };
        let materials = material_library.all_materials().to_vec();
        let material_library = Arc::new(Mutex::new(material_library));

        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights — empty, populated when scene loads.
        let world_lights = Vec::new();
        let mut init_lights = Vec::with_capacity(64);
        // Slot 0 = directional (driven by environment).
        init_lights.push(Light::point([0.0; 3], [0.0; 3], 0.0, 0.0, false));
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

        // Shader composer — user shaders are loaded when a project is opened
        // via scan_user_shaders(). At startup there's no project yet.
        let shader_composer = rkf_render::ShaderComposer::new();

        // Brush overlay (empty placeholder until brush is active).
        let brush_overlay = BrushOverlay::empty(&ctx.device);

        // Color pool (empty placeholder until paint is used).
        let gpu_color_pool = rkf_render::GpuColorPool::empty(&ctx.device);

        // Shading pass — no pipeline compiled yet (no project loaded).
        // Pipeline gets compiled when scan_user_shaders() triggers recompose_and_recompile().
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            internal_w, internal_h,
            None,
            &brush_overlay,
            &gpu_color_pool,
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

        // Double-buffered readback at viewport resolution.
        let readback_buffers = Self::create_readback_buffers(
            &ctx.device, vp_w, vp_h,
        );

        let gpu_profiler = rkf_render::gpu_profiler::GpuProfiler::new(
            &ctx.device, &ctx.queue,
        );

        let material_preview = rkf_render::material_preview::MaterialPreviewRenderer::new(
            &ctx.device, &ctx.queue,
        );

        let brick_pool_capacity = empty_pool.capacity();
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
            brush_overlay,
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
            material_buffer: material_table.buffer.clone(),
            material_table,
            material_library: material_library.clone(),
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
            env_vol_ambient_color: [0.24, 0.30, 0.42],
            env_cloud_settings: rkf_render::CloudSettings::default(),
            accumulated_time: 0.0,
            render_width: internal_w,
            render_height: internal_h,
            pick_readback_buffer,
            brush_readback_buffer,
            readback_buffers,
            readback_parity: 0,
            readback_pending: None,
            prev_frame_pixels: None,
            window_width: vp_w,
            window_height: vp_h,
            shared_state,
            wireframe_pass: Some(wireframe_pass),
            cpu_brick_pool: empty_pool,
            cpu_brick_map_alloc: empty_alloc,
            cpu_geometry_pool: rkf_core::brick_pool::GeometryPool::new(256),
            cpu_sdf_cache_pool: rkf_core::brick_pool::SdfCachePool::new(256),
            geometry_first_data: std::collections::HashMap::new(),
            // Incremental update cache — first frame triggers full rebuild.
            cached_gpu_objects: Vec::new(),
            cached_bvh: None,
            cached_bvh_pairs: Vec::new(),
            cached_world_aabbs: std::collections::HashMap::new(),
            object_gpu_ranges: std::collections::HashMap::new(),
            dirty_objects: std::collections::HashSet::new(),
            spawned_objects: Vec::new(),
            despawned_objects: Vec::new(),
            tombstone_count: 0,
            topology_changed: true,
            project_root: None,
            lights_dirty: true,
            last_light_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            pending_wireframe: Vec::new(),
            gpu_profiler,
            last_vol_shadow_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            last_vol_shadow_sun_dir: [f32::NAN; 3],
            file_watcher: {
                // Watch engine shaders and source. Project-specific paths (user shaders,
                // materials) are added when a project is opened via reinit_file_watcher().
                let engine_root = rkf_runtime::project::engine_library_dir()
                    .parent()
                    .unwrap_or(std::path::Path::new("."))
                    .to_path_buf();
                let shaders_dir = engine_root.join("crates/rkf-render/shaders");
                let source_dir = engine_root.join("crates");
                match rkf_runtime::FileWatcher::new(&[shaders_dir.as_path(), source_dir.as_path()]) {
                    Ok(w) => {
                        log::info!("File watcher started for engine shaders + source");
                        Some(w)
                    }
                    Err(e) => {
                        log::warn!("File watcher failed: {e}");
                        None
                    }
                }
            },
            build_watcher: None,
            shader_composer,
            shader_error: None,
            material_preview,
            cpu_color_bricks: Vec::new(),
            color_companion_map: vec![0xFFFFFFFF; brick_pool_capacity as usize],
            gpu_color_pool,
            console: rkf_runtime::behavior::ConsoleBuffer::new(),
        };
        engine
    }

    /// Create the offscreen render target texture at the given resolution.
    pub(super) fn create_offscreen_target(
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

        // Recreate readback buffers at new viewport resolution.
        self.readback_buffers = Self::create_readback_buffers(&self.ctx.device, vp_w, vp_h);
        self.readback_pending = None;
        self.prev_frame_pixels = None;
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
}
