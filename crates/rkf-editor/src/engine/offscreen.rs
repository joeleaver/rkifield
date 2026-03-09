//! Offscreen rendering path for the editor engine (compositor/rinch integration).

use std::sync::{Arc, Mutex};
use glam::Vec3;
use rkf_core::Scene;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugMode, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, SceneUniforms, ShadeUniforms, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass, COARSE_VOXEL_SIZE,
};
use rkf_render::radiance_inject::{RadianceInjectPass, InjectUniforms};
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::material_table::{MaterialTable, create_test_materials};
use rkf_core::material_library::MaterialLibrary;
use super::EditorEngine;
use super::{OFFSCREEN_FORMAT, build_demo_scene};
use crate::engine_viewport::RENDER_SCALE;
use crate::automation::SharedState;

impl EditorEngine {
    /// Create an engine with its own dedicated wgpu device.
    ///
    /// Renders to an offscreen texture. The caller reads back pixels via
    /// [`readback_frame`] and submits them to the compositor's CPU pixel
    /// path (`SurfaceWriter::submit_frame`). This eliminates GPU contention
    /// between the engine and the compositor.
    ///
    /// Returns `(engine, scene)` — the caller stores the scene in EditorState.
    pub fn new_headless(
        viewport_width: u32,
        viewport_height: u32,
        shared_state: Arc<Mutex<SharedState>>,
    ) -> (Self, Scene) {
        let ctx = RenderContext::new_headless();

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

        // Material library — load from palette file, fallback to hardcoded test materials.
        let palette_path = std::path::Path::new("assets/materials/default.rkmatlib");
        let material_library = match MaterialLibrary::load_palette(palette_path) {
            Ok(lib) => {
                log::info!("Material library loaded from {}", palette_path.display());
                lib
            }
            Err(e) => {
                log::warn!("Failed to load material palette: {e} — using test materials");
                let mut lib = MaterialLibrary::new(16);
                #[allow(deprecated)]
                for (i, mat) in create_test_materials().into_iter().enumerate() {
                    lib.set_material(i as u16, mat);
                }
                lib.mark_dirty();
                lib
            }
        };
        let materials = material_library.all_materials().to_vec();
        let material_library = Arc::new(Mutex::new(material_library));

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

        // Shader composer — compose the uber-shader from built-in + user shaders.
        let mut shader_composer = rkf_render::ShaderComposer::new();
        {
            let shader_dir = std::path::Path::new("assets/shaders");
            if shader_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(shader_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                            continue;
                        }
                        if let Ok(source) = std::fs::read_to_string(&path) {
                            let name = super::EditorEngine::extract_shader_name(&source)
                                .unwrap_or_else(|| {
                                    let fname = path.file_name().unwrap().to_str().unwrap_or("unknown");
                                    fname.strip_prefix("shade_")
                                        .and_then(|s| s.strip_suffix(".wgsl"))
                                        .unwrap_or(fname.trim_end_matches(".wgsl"))
                                        .to_string()
                                });
                            let file_path = path.display().to_string();
                            let id = shader_composer.register_with_path(&name, source, Some(file_path));
                            log::info!("Registered user shader: {name} (id={id}) from {}", path.display());
                        }
                    }
                }
            }
        }
        let composed_source = shader_composer.compose().to_string();

        // Resolve shader IDs in the material library.
        {
            let mut lib = material_library.lock().unwrap();
            let composer = &shader_composer;
            lib.resolve_shader_ids(|name| composer.shader_id(name));
        }

        // Shading pass.
        let shading_pass = ShadingPass::new(
            &ctx.device, &gbuffer, &gpu_scene, &light_buffer,
            &coarse_field, &radiance_volume, &material_table.buffer,
            internal_w, internal_h,
            Some(&composed_source),
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

        let jfa_sdf = crate::jfa_sdf::JfaSdfPass::new(&ctx.device);
        let eikonal_repair = crate::eikonal_repair::EikonalRepairPass::new(&ctx.device);
        let gpu_profiler = rkf_render::gpu_profiler::GpuProfiler::new(
            &ctx.device, &ctx.queue,
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
            cpu_brick_pool: demo.brick_pool,
            cpu_brick_map_alloc: demo.brick_map_alloc,
            cpu_geometry_pool: rkf_core::brick_pool::GeometryPool::new(256),
            cpu_sdf_cache_pool: rkf_core::brick_pool::SdfCachePool::new(256),
            geometry_first_data: std::collections::HashMap::new(),
            jfa_sdf,
            eikonal_repair,
            // Incremental update cache — first frame triggers full rebuild.
            cached_gpu_objects: Vec::new(),
            cached_bvh: None,
            cached_bvh_pairs: Vec::new(),
            cached_world_aabbs: Vec::new(),
            object_gpu_ranges: std::collections::HashMap::new(),
            dirty_objects: std::collections::HashSet::new(),
            topology_changed: true,
            lights_dirty: true,
            last_light_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            pending_wireframe: Vec::new(),
            gpu_profiler,
            last_vol_shadow_cam_pos: Vec3::new(f32::NAN, f32::NAN, f32::NAN),
            last_vol_shadow_sun_dir: [f32::NAN; 3],
            file_watcher: {
                let assets_dir = std::path::Path::new("assets/materials");
                let shaders_dir = std::path::Path::new("crates/rkf-render/shaders");
                let user_shaders_dir = std::path::Path::new("assets/shaders");
                match rkf_runtime::FileWatcher::new(&[assets_dir, shaders_dir, user_shaders_dir]) {
                    Ok(w) => {
                        log::info!("File watcher started for materials + shaders + user shaders");
                        Some(w)
                    }
                    Err(e) => {
                        log::warn!("File watcher failed: {e}");
                        None
                    }
                }
            },
            shader_composer,
            shader_error: None,
        };
        (engine, scene)
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

        // Incremental scene-to-GPU update: only re-flatten dirty objects,
        // skip entirely on static frames (camera-only movement).
        let num_objects = self.update_scene_gpu(scene);

        // Camera uniforms.
        let cam_uniforms = self.camera.uniforms(
            self.render_width, self.render_height, self.frame_index, self.prev_vp,
        );
        self.gpu_scene.update_camera(&self.ctx.queue, &cam_uniforms);

        let scene_uniforms = SceneUniforms {
            num_objects,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        };
        self.gpu_scene.update_scene_uniforms(&self.ctx.queue, &scene_uniforms);

        // Light buffer: only re-upload when lights or camera position changed.
        let total_lights = 1 + self.world_lights.len() as u32;
        let cam = self.camera.position;
        let cam_moved = cam != self.last_light_cam_pos;
        if self.lights_dirty || cam_moved {
            let sun_light = Light {
                light_type: 0, pos_x: 0.0, pos_y: 0.0, pos_z: 0.0,
                dir_x: self.env_sun_dir[0], dir_y: self.env_sun_dir[1], dir_z: self.env_sun_dir[2],
                color_r: self.env_sun_color[0], color_g: self.env_sun_color[1], color_b: self.env_sun_color[2],
                intensity: 1.0, range: 0.0, inner_angle: 0.0, outer_angle: 0.0,
                cookie_index: -1, shadow_caster: 1,
            };
            let cam_rel_lights: Vec<Light> = self.world_lights.iter().map(|l| {
                let mut cl = *l;
                if cl.light_type != 0 { cl.pos_x -= cam.x; cl.pos_y -= cam.y; cl.pos_z -= cam.z; }
                cl
            }).collect();
            let mut all_lights = vec![sun_light];
            all_lights.extend(cam_rel_lights);
            self.light_buffer.update(&self.ctx.queue, &all_lights);
            self.lights_dirty = false;
            self.last_light_cam_pos = cam;
        }

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

        let mut encoder = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_frame"),
        });

        self.gpu_profiler.begin_frame();

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
        self.gpu_profiler.begin_pass(&mut encoder, "tile_cull");
        self.tile_cull.dispatch(&mut encoder, &self.gpu_scene);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "ray_march");
        self.ray_march.dispatch(
            &mut encoder, &self.gpu_scene, &self.gbuffer,
            &self.tile_cull, &self.coarse_field,
        );
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "radiance_inject");
        self.radiance_inject.dispatch(&mut encoder, &self.gpu_scene, &self.coarse_field);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "radiance_mip");
        self.radiance_mip.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "shading");
        self.shading_pass.dispatch(
            &mut encoder, &self.gbuffer, &self.gpu_scene,
            &self.coarse_field, &self.radiance_volume,
        );
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Volumetric pipeline ---
        self.accumulated_time += 1.0 / 60.0;
        let cloud_params = rkf_render::CloudParams::from_settings(
            &self.env_cloud_settings, self.accumulated_time,
        );

        self.vol_march.set_cloud_params(&self.ctx.queue, &cloud_params);
        self.cloud_shadow.set_cloud_params(&self.ctx.queue, &cloud_params);

        let sun_dir = self.env_sun_dir;
        // Skip vol_shadow dispatch when camera and sun haven't changed.
        let vs_cam_delta = Vec3::new(
            cam.x - self.last_vol_shadow_cam_pos.x,
            cam.y - self.last_vol_shadow_cam_pos.y,
            cam.z - self.last_vol_shadow_cam_pos.z,
        );
        let vs_sun_changed = sun_dir != self.last_vol_shadow_sun_dir;
        if vs_cam_delta.length_squared() > 0.01 || vs_sun_changed {
            self.gpu_profiler.begin_pass(&mut encoder, "vol_shadow");
            self.vol_shadow.dispatch(
                &mut encoder, &self.ctx.queue,
                [cam.x, cam.y, cam.z], sun_dir, &self.coarse_field.bind_group,
            );
            self.gpu_profiler.end_pass(&mut encoder);
            self.last_vol_shadow_cam_pos = cam;
            self.last_vol_shadow_sun_dir = sun_dir;
        }

        self.cloud_shadow.update_params_ex(
            &self.ctx.queue,
            [cam.x, cam.y, cam.z],
            sun_dir,
            cloud_params.altitude[0],
            cloud_params.altitude[1],
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_COVERAGE,
            rkf_render::cloud_shadow::DEFAULT_CLOUD_SHADOW_EXTINCTION,
        );
        self.gpu_profiler.begin_pass(&mut encoder, "cloud_shadow");
        self.cloud_shadow.dispatch_only(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

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
        self.gpu_profiler.begin_pass(&mut encoder, "vol_march");
        self.vol_march.dispatch(&mut encoder, &self.ctx.queue, &vol_params);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "vol_upscale");
        self.vol_upscale.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "vol_composite");
        self.vol_composite.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

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
        self.gpu_profiler.begin_pass(&mut encoder, "god_rays");
        self.god_rays_blur.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "bloom");
        self.bloom.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "auto_exposure");
        self.auto_exposure.dispatch(&mut encoder, &self.ctx.queue, 1.0 / 60.0);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "dof");
        self.dof.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "motion_blur");
        self.motion_blur.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "bloom_composite");
        self.bloom_composite.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "tone_map");
        self.tone_map.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "color_grade");
        self.color_grade.dispatch(&mut encoder);
        self.gpu_profiler.end_pass(&mut encoder);

        self.gpu_profiler.begin_pass(&mut encoder, "cosmetics");
        self.cosmetics.dispatch(&mut encoder, &self.ctx.queue, self.frame_index);
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Blit to offscreen render target ---
        let offscreen_view = self.offscreen_view.as_ref()
            .expect("offscreen_view must exist in offscreen mode");
        self.gpu_profiler.begin_pass(&mut encoder, "blit");
        if let Some(ref offscreen_blit) = self.offscreen_blit {
            offscreen_blit.draw(&mut encoder, offscreen_view);
        }
        self.gpu_profiler.end_pass(&mut encoder);

        // --- Wireframe overlays (same encoder — no extra GPU submit) ---
        if !self.pending_wireframe.is_empty() {
            let vp_matrix = self.camera.view_projection(
                self.viewport_width, self.viewport_height,
            );
            let viewport = (
                0.0f32, 0.0f32,
                self.viewport_width as f32, self.viewport_height as f32,
            );
            if let Some(ref mut wf) = self.wireframe_pass {
                wf.draw(
                    &self.ctx.device, &self.ctx.queue, &mut encoder,
                    offscreen_view, vp_matrix, viewport, &self.pending_wireframe,
                );
            }
            self.pending_wireframe.clear();
        }

        // --- Frame readback (double-buffered CPU pixel path) ---
        // Copy to the CURRENT parity buffer. The previous parity buffer
        // holds last frame's data and can be mapped without waiting.
        {
            let w = self.viewport_width;
            let h = self.viewport_height;
            let padded_row = (w * 4 + 255) & !255;
            let offscreen_tex = self.offscreen_texture.as_ref()
                .expect("offscreen_texture must exist");
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: offscreen_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback_buffers[self.readback_parity],
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_row),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        // Resolve GPU timestamps before submit.
        self.gpu_profiler.resolve(&mut encoder);

        // Single submit for render + wireframe + readback copy.
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

        // Screenshot readback is now handled by the caller via map_readback(),
        // which reads the readback_buffer populated by the copy above.

        self.frame_index += 1;
    }

    /// Synchronous readback: wait for GPU to finish, map buffer, read pixels.
    ///
    /// Called after `render_frame_offscreen()` which submitted render + copy
    /// to `readback_buffers[0]`. This waits for the GPU, reads the pixels,
    /// and returns them. The engine loop runs at GPU frame rate.
    pub fn map_readback(&mut self) -> (Vec<u8>, u32, u32) {
        let w = self.viewport_width;
        let h = self.viewport_height;
        let padded_row = (w * 4 + 255) & !255;

        let t0 = std::time::Instant::now();

        let buffer_slice = self.readback_buffers[0].slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let t1 = std::time::Instant::now();

        let mut rgba8 = vec![0u8; (w * h * 4) as usize];
        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..h as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * w as usize * 4;
                let row_bytes = w as usize * 4;
                if src_offset + row_bytes <= data.len()
                    && dst_offset + row_bytes <= rgba8.len()
                {
                    rgba8[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&data[src_offset..src_offset + row_bytes]);
                }
            }
            drop(data);
            self.readback_buffers[0].unmap();
        } else {
            self.readback_buffers[0].unmap();
        }

        let t2 = std::time::Instant::now();

        // GPU timestamps are ready after poll — read them.
        self.gpu_profiler.read_results(&self.ctx.device);

        // Log timing every 60 frames.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % 60 == 0 {
            eprintln!(
                "[READBACK] poll_wait: {:.2}ms  memcpy: {:.2}ms  total: {:.2}ms  ({}x{})",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t2 - t0).as_secs_f64() * 1000.0,
                w, h,
            );
            // Per-pass GPU timing.
            if !self.gpu_profiler.results.is_empty() {
                let mut total_gpu = 0.0;
                eprint!("[GPU PASSES]");
                for (name, ms) in &self.gpu_profiler.results {
                    eprint!("  {}:{:.2}ms", name, ms);
                    total_gpu += ms;
                }
                eprintln!("  TOTAL:{:.2}ms", total_gpu);
            }
        }

        (rgba8, w, h)
    }
}
