//! Editor engine — v2 object-centric render pipeline.
//!
//! Extracted from the testbed's `EngineState`. Provides the full v2 compute-shader
//! render pipeline (ray march, shading, GI, volumetrics, post-processing) in a
//! reusable struct that the editor's event loop drives each frame.

use std::sync::{Arc, Mutex};

use glam::Vec3;

use rkf_core::{
    Aabb, BrickMapAllocator, BrickPool,
    SdfPrimitive,
};
use rkf_core::brick::Brick;
use rkf_core::brick_pool::{GeometryPool, SdfCachePool};
use rkf_core::material_library::MaterialLibrary;
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, BrushOverlay, Camera,
    CloudShadowPass, CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuObject, GpuScene, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShaderComposer, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass,
};
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;

use crate::automation::SharedState;
// SceneCamera removed — engine uses CameraSnapshot via sync_camera_snapshot()

mod init;
mod environment;
mod render;
mod sculpt;
mod paint;
mod brick_ops;
mod offscreen;
mod offscreen_render;
mod query;
pub(crate) mod file_loading;
pub(crate) use file_loading::*;

/// Internal render resolution width (used by the legacy surface-based path).
pub const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height (used by the legacy surface-based path).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Offscreen render target format for the compositor path.
///
/// sRGB variant so the blit pass's linear→sRGB conversion happens in hardware,
/// matching the current swapchain behavior. The compositor samples this texture
/// (auto-decoding sRGB→linear) and writes to rinch's sRGB swapchain.
pub(super) const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

// ---------------------------------------------------------------------------
// Demo scene
// ---------------------------------------------------------------------------

/// Per-object geometry-first slot mapping.
///
/// Tracks the correspondence between geometry pool, SDF cache pool, and brick pool
/// slots for objects that use the geometry-first data model.
///
/// The `geo_brick_map` is the geometry-layer spatial index — entries are geometry
/// pool slot indices (or EMPTY_SLOT / INTERIOR_SLOT). This is distinct from the
/// GPU brick map in the allocator, which maps to brick pool slots.
pub(super) struct GeometryFirstData {
    /// Geometry-layer brick map: entries are geometry pool slot indices.
    pub geo_brick_map: rkf_core::brick_map::BrickMap,
    /// Per-slot mapping: geometry pool slot → (SDF cache slot, brick pool slot).
    pub slot_map: std::collections::HashMap<u32, (u32, u32)>,
    /// Voxel size for this object.
    pub voxel_size: f32,
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
    pub(super) ctx: RenderContext,
    // Surface-based rendering (legacy path — used when engine owns the window).
    pub(super) surface: Option<wgpu::Surface<'static>>,
    pub(super) surface_format: wgpu::TextureFormat,
    // Offscreen rendering (compositor path — used when rinch owns the window).
    pub(super) offscreen_texture: Option<wgpu::Texture>,
    pub(super) offscreen_view: Option<wgpu::TextureView>,
    pub(super) offscreen_blit: Option<BlitPass>,
    /// Viewport size in physical pixels (set by RenderSurface layout).
    pub(super) viewport_width: u32,
    pub(super) viewport_height: u32,
    pub(super) gpu_scene: GpuScene,
    pub(super) gbuffer: GBuffer,
    pub(super) tile_cull: TileObjectCullPass,
    pub(super) coarse_field: CoarseField,
    pub(super) ray_march: RayMarchPass,
    #[allow(dead_code)]
    pub(super) debug_view: DebugViewPass,
    pub(super) shading_pass: ShadingPass,
    pub(super) brush_overlay: BrushOverlay,
    pub(super) radiance_volume: RadianceVolume,
    pub(super) radiance_inject: RadianceInjectPass,
    pub(super) radiance_mip: RadianceMipPass,
    // Volumetric pipeline
    pub(super) vol_shadow: VolShadowPass,
    pub(super) cloud_shadow: CloudShadowPass,
    pub(super) vol_march: VolMarchPass,
    pub(super) vol_upscale: VolUpscalePass,
    pub(super) vol_composite: VolCompositePass,
    // Post-processing pipeline
    pub(super) god_rays_blur: GodRaysBlurPass,
    pub(super) bloom: BloomPass,
    pub(super) auto_exposure: AutoExposurePass,
    pub(super) dof: DofPass,
    pub(super) motion_blur: MotionBlurPass,
    pub(super) bloom_composite: BloomCompositePass,
    pub(super) tone_map: ToneMapPass,
    pub(super) color_grade: ColorGradePass,
    pub(super) cosmetics: CosmeticsPass,
    #[allow(dead_code)]
    pub(super) sharpen: SharpenPass,
    pub(super) blit: BlitPass,
    // State
    pub(super) camera: Camera,
    pub world_lights: Vec<Light>,
    pub(super) light_buffer: LightBuffer,
    pub(super) material_table: rkf_render::material_table::MaterialTable,
    pub(super) material_buffer: wgpu::Buffer,
    /// Material library backing the GPU material table.
    pub(crate) material_library: Arc<Mutex<MaterialLibrary>>,
    pub(super) frame_index: u32,
    pub(super) prev_vp: [[f32; 4]; 4],
    pub(super) shade_debug_mode: u32,
    // Cached environment vol params (updated by apply_environment).
    pub(super) env_sun_dir: [f32; 3],
    pub(super) env_sun_color: [f32; 3],
    pub(super) env_fog_color: [f32; 3],
    pub(super) env_fog_density: f32,
    pub(super) env_fog_height_falloff: f32,
    // Atmosphere params for analytic sky.
    pub(super) env_sun_intensity: f32,
    pub(super) env_sun_color_raw: [f32; 3],
    pub(super) env_rayleigh_scale: f32,
    pub(super) env_mie_scale: f32,
    pub(super) env_atmosphere_enabled: bool,
    // Dust params for god rays.
    pub(super) env_ambient_dust: f32,
    pub(super) env_dust_g: f32,
    pub(super) env_vol_ambient_color: [f32; 3],
    pub(super) env_vol_ambient_intensity: f32,
    pub(super) env_gi_intensity: f32,
    // Cloud params from environment.
    pub(super) env_cloud_settings: rkf_render::CloudSettings,
    pub(super) accumulated_time: f32,
    // Render resolution (tracks viewport physical pixels)
    pub(super) render_width: u32,
    pub(super) render_height: u32,
    // GPU object picking (single-pixel readback from material G-buffer)
    pub(super) pick_readback_buffer: wgpu::Buffer,
    // GPU brush hit readback (single-pixel from position G-buffer, Rgba32Float = 16 bytes)
    pub(super) brush_readback_buffer: wgpu::Buffer,
    // Double-buffered frame readback (window-resolution).
    // While one buffer is mapped for CPU read, the other receives the GPU copy.
    pub(super) readback_buffers: [wgpu::Buffer; 2],
    pub(super) readback_parity: usize, // alternates 0/1 each frame
    /// Pending async map from the previous frame's readback buffer.
    /// Contains (buffer_index, receiver). Checked non-blocking at frame start.
    pub(super) readback_pending: Option<(usize, std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,
    /// Latest available pixels for the compositor.
    pub(super) prev_frame_pixels: Option<(Vec<u8>, u32, u32)>,
    pub(super) window_width: u32,
    pub(super) window_height: u32,
    pub(super) shared_state: Arc<Mutex<SharedState>>,
    pub(super) wireframe_pass: Option<crate::wireframe::WireframePass>,
    // CPU-side brick data retained for re-voxelize operations.
    pub(super) cpu_brick_pool: BrickPool,
    pub(super) cpu_brick_map_alloc: BrickMapAllocator,
    // Geometry-first pools (source of truth for geometry-first objects).
    pub(super) cpu_geometry_pool: GeometryPool,
    pub(super) cpu_sdf_cache_pool: SdfCachePool,
    /// Per-object geometry-first slot mappings: object_id → GeometryFirstData.
    pub(super) geometry_first_data: std::collections::HashMap<u32, GeometryFirstData>,
    // ── Cached GPU state for incremental updates ──
    /// Cached GPU objects from last full rebuild. Used for partial updates.
    pub(super) cached_gpu_objects: Vec<GpuObject>,
    /// Cached BVH from last full rebuild.
    pub(super) cached_bvh: Option<rkf_core::Bvh>,
    /// Cached BVH pairs (gpu_idx, world_aabb) for BVH refit.
    pub(super) cached_bvh_pairs: Vec<(u32, Aabb)>,
    /// Cached world-space AABBs keyed by object_id (for incremental coarse field updates).
    pub(super) cached_world_aabbs: std::collections::HashMap<u32, Aabb>,
    /// Map from object_id → (start_index, count) in cached_gpu_objects.
    pub(super) object_gpu_ranges: std::collections::HashMap<u32, (usize, usize)>,
    /// Object IDs that need re-flattening this frame (transform changes, sculpt edits).
    pub(super) dirty_objects: std::collections::HashSet<u32>,
    /// Object IDs spawned this frame (need GPU insert).
    pub(super) spawned_objects: Vec<u32>,
    /// Object IDs despawned this frame (need GPU tombstone).
    pub(super) despawned_objects: Vec<u32>,
    /// Number of tombstoned (dead) GPU object slots.
    pub(super) tombstone_count: usize,
    /// Full scene replaced (open/new project) — triggers complete rebuild.
    pub(super) topology_changed: bool,
    /// Current project root directory. Used to resolve project-relative paths
    /// (e.g. `assets/shaders/` for user shaders).
    pub(crate) project_root: Option<std::path::PathBuf>,
    /// Lights changed this frame (light editor dirty or world_lights replaced).
    pub(super) lights_dirty: bool,
    /// Last camera position used for light upload (detect camera movement).
    pub(super) last_light_cam_pos: Vec3,
    /// Wireframe vertices staged for the current frame (drawn in same encoder).
    pub(super) pending_wireframe: Vec<crate::wireframe::LineVertex>,
    /// GPU timestamp profiler for per-pass timing.
    pub(super) gpu_profiler: rkf_render::gpu_profiler::GpuProfiler,
    /// Last camera position used for vol_shadow dispatch (skip when static).
    pub(super) last_vol_shadow_cam_pos: Vec3,
    /// Last sun direction used for vol_shadow dispatch (skip when static).
    pub(super) last_vol_shadow_sun_dir: [f32; 3],
    /// File watcher for material, shader, and source hot-reload.
    pub(super) file_watcher: Option<rkf_runtime::FileWatcher>,
    /// Build watcher for auto-building on .rs source changes.
    /// Set by the engine loop after a non-blocking `build_and_load_game_dylib` call,
    /// polled each frame via `process_file_events()`.
    pub(crate) build_watcher: Option<rkf_runtime::BuildWatcher>,
    /// Shader composer — manages the uber-shader composition and shader registry.
    pub(crate) shader_composer: ShaderComposer,
    /// Last shader compile error for status bar display (cleared on success).
    pub(crate) shader_error: Option<String>,
    /// Material preview renderer — renders a small preview of a single material on a primitive.
    pub(crate) material_preview: rkf_render::material_preview::MaterialPreviewRenderer,
    // ── Color companion pool ──
    /// CPU-side color bricks (per-voxel paint data).
    pub(super) cpu_color_bricks: Vec<rkf_core::companion::ColorBrick>,
    /// Companion map: brick_pool_slot → color_brick_index (EMPTY_SLOT = no color).
    pub(super) color_companion_map: Vec<u32>,
    /// GPU color pool (buffers for shading pass).
    pub(super) gpu_color_pool: rkf_render::GpuColorPool,
    /// Unified console buffer for scripts, engine, and build output.
    pub(crate) console: rkf_runtime::behavior::ConsoleBuffer,
}

impl EditorEngine {
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

    /// Sync the render camera from a `CameraSnapshot`.
    pub fn sync_camera_snapshot(&mut self, snap: &crate::camera::CameraSnapshot) {
        self.camera.position = snap.position;
        self.camera.yaw = snap.yaw;
        self.camera.pitch = snap.pitch;
        self.camera.fov_degrees = snap.fov_degrees;
    }

    /// Set the shading debug mode (0=normal, 1=normals, 2=positions, etc).
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.shade_debug_mode = mode;
    }

    /// Camera-relative view-projection matrix for overlay rendering.
    pub fn view_projection(&self) -> glam::Mat4 {
        self.camera.view_projection(self.render_width, self.render_height)
    }

    /// Current camera position in world space.
    pub fn camera_position(&self) -> Vec3 {
        self.camera.position
    }

    /// Stage wireframe vertices for the next frame. The wireframe will be
    /// drawn into the same command encoder as the main render pass (no extra
    /// GPU submit).
    pub fn set_wireframe_vertices(&mut self, vertices: Vec<crate::wireframe::LineVertex>) {
        self.pending_wireframe = vertices;
    }

    /// Current render resolution width.
    pub fn render_width(&self) -> u32 {
        self.render_width
    }

    /// Current render resolution height.
    pub fn render_height(&self) -> u32 {
        self.render_height
    }


    /// Restore geometry-first data from undo snapshots and re-derive Bricks for GPU.
    ///
    /// Writes each snapshot's BrickGeometry + SdfCache back to the geometry and SDF
    /// cache pools, re-derives the Brick via `from_geometry`, and uploads to GPU.
    pub fn apply_sculpt_undo(&mut self, snapshots: &[crate::editor_state::GeometryUndoEntry]) {
        let brick_byte_size = std::mem::size_of::<rkf_core::brick::Brick>() as u64;
        let gpu_buf_size = self.gpu_scene.brick_pool_buffer().size();

        for entry in snapshots {
            // Restore geometry source of truth.
            *self.cpu_geometry_pool.get_mut(entry.geo_slot) = entry.geometry.clone();

            // Restore SDF cache (find the sdf_slot from any object's slot_map).
            // We need the sdf_slot — look it up from geometry_first_data.
            let sdf_slot = self.geometry_first_data.values()
                .find_map(|gfd| gfd.slot_map.get(&entry.geo_slot).map(|&(s, _)| s));

            if let Some(sdf_slot) = sdf_slot {
                *self.cpu_sdf_cache_pool.get_mut(sdf_slot) = entry.sdf_cache.clone();
            }

            // Re-derive Brick from restored geometry + SDF cache.
            let brick = Brick::from_geometry(&entry.geometry, &entry.sdf_cache);
            *self.cpu_brick_pool.get_mut(entry.brick_slot) = brick;

            // Upload to GPU.
            let offset = entry.brick_slot as u64 * brick_byte_size;
            if offset + brick_byte_size <= gpu_buf_size {
                self.ctx.queue.write_buffer(
                    self.gpu_scene.brick_pool_buffer(),
                    offset,
                    bytemuck::bytes_of(self.cpu_brick_pool.get(entry.brick_slot)),
                );
            }
        }
    }

    /// Returns true if a brick map slot is unallocated (no pool data).
    pub(super) fn is_unallocated(slot: u32) -> bool {
        slot == rkf_core::brick_map::EMPTY_SLOT || slot == rkf_core::brick_map::INTERIOR_SLOT
    }

    /// Check if the material library is dirty and re-upload to GPU if needed.
    ///
    /// When the material buffer is recreated (size change), rebinds affected
    /// passes (shading, radiance inject).
    pub fn sync_materials(&mut self) {
        let dirty = {
            let lib = self.material_library.lock().unwrap();
            lib.is_dirty()
        };
        if !dirty {
            return;
        }

        let materials = {
            let mut lib = self.material_library.lock().unwrap();
            let mats = lib.all_materials().to_vec();
            lib.clear_dirty();
            mats
        };

        let buffer_recreated = self.material_table.update(
            &self.ctx.device, &self.ctx.queue, &materials,
        );

        if buffer_recreated {
            // Update the cached buffer reference.
            self.material_buffer = self.material_table.buffer.clone();
            // Rebind passes that reference the material buffer.
            self.shading_pass.update_materials(&self.ctx.device, &self.material_buffer);
            self.radiance_inject.update_materials(&self.ctx.device, &self.material_buffer);
            log::info!("Material buffer recreated ({} materials), passes rebound", materials.len());
        } else {
            log::debug!("Material table updated ({} materials)", materials.len());
        }
    }

    /// Process file watcher events (material + shader + source hot-reload).
    ///
    /// Returns `Some(path)` when the build watcher reports a successful build,
    /// signaling that the caller should hot-reload the game dylib.
    pub fn process_file_events(&mut self) -> Option<std::path::PathBuf> {
        let events = match self.file_watcher {
            Some(ref watcher) => watcher.poll_events(),
            None => Vec::new(),
        };

        for event in events {
            match event {
                rkf_runtime::FileEvent::MaterialChanged(path) => {
                    let mut lib = self.material_library.lock().unwrap();
                    if let Err(e) = lib.reload_material(&path) {
                        log::warn!("Material reload failed: {e}");
                    }
                    // Resolve shader IDs after reload so shader_id in GPU Material is updated.
                    let composer = &self.shader_composer;
                    lib.resolve_shader_ids(|name| composer.shader_id(name));
                }
                rkf_runtime::FileEvent::ShaderChanged(path) => {
                    self.try_reload_shader(&path);
                }
                rkf_runtime::FileEvent::SourceChanged(path) => {
                    if let Some(ref root) = self.project_root {
                        // Re-scaffold to copy updated scripts into the generated crate.
                        let engine_root = rkf_runtime::project::engine_library_dir()
                            .parent()
                            .unwrap_or(std::path::Path::new("."))
                            .to_path_buf();
                        let _ = rkf_runtime::behavior::scaffold::generate_game_crate(root, &engine_root);
                    }
                    if let Some(ref mut bw) = self.build_watcher {
                        log::info!("Source changed: {} — triggering build", path.display());
                        self.console.info("Compiling scripts...");
                        bw.trigger_build(Some(self.console.clone()));
                    }
                }
            }
        }

        // Poll build watcher for completion (stderr streams directly to console via reader thread).
        if let Some(ref mut bw) = self.build_watcher {
            bw.poll();
            if let Some(path) = bw.take_success() {
                self.console.info("Build succeeded");
                return Some(path);
            }
            if let rkf_runtime::behavior::BuildState::Error(_) = bw.state() {
                if let rkf_runtime::behavior::BuildState::Error(ref err) = bw.state() {
                    log::error!("Game crate build failed: {err}");
                    self.console.error(format!("Build failed: {err}"));
                }
                bw.reset();
            }
        }

        None
    }

    /// Attempt to reload a shader from disk and recreate the affected pipeline.
    fn try_reload_shader(&mut self, path: &std::path::Path) {
        let filename = path.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("");

        // User custom shader in <project>/assets/shaders/ — register/update in composer and recompose.
        let is_user_shader = self.project_root.as_ref().map_or(false, |root| {
            let user_shader_dir = root.join("assets").join("shaders");
            path.starts_with(&user_shader_dir)
        });

        if is_user_shader && filename.ends_with(".wgsl") {
            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    let msg = format!("Shader read error: {}: {e}", path.display());
                    log::error!("{msg}");
                    self.shader_error = Some(msg);
                    return;
                }
            };

            // Extract shader name from function signature: fn shade_<name>(
            let shader_name = Self::extract_shader_name(&source).unwrap_or_else(|| {
                filename.strip_prefix("shade_")
                    .and_then(|s| s.strip_suffix(".wgsl"))
                    .unwrap_or(filename.trim_end_matches(".wgsl"))
                    .to_string()
            });

            log::info!("Custom shader changed: {filename} (name={shader_name}) — recomposing");
            let file_path = path.display().to_string();
            self.shader_composer.register_with_path(&shader_name, source, Some(file_path));
            self.recompose_and_recompile();
            return;
        }

        // Built-in shade composition files trigger a recompose + recompile.
        if filename.starts_with("shade_") && filename.ends_with(".wgsl") {
            log::info!("Shade component changed: {filename} — recomposing uber-shader");
            // Re-read the changed source and update in the composer.
            if let Ok(source) = std::fs::read_to_string(path) {
                let name = filename.strip_prefix("shade_")
                    .and_then(|s| s.strip_suffix(".wgsl"))
                    .unwrap_or("");
                match name {
                    "common" | "main" => {
                        // Common/main changed — rebuild composer from scratch (they're include_str'd).
                        self.shader_composer = ShaderComposer::new();
                        // Re-register any user shaders from disk.
                        self.scan_user_shaders();
                    }
                    _ => {
                        // A built-in shading model changed — update its source.
                        self.shader_composer.update_source(name, source);
                    }
                }
            }
            self.recompose_and_recompile();
            return;
        }

        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                let msg = format!("Shader read error: {}: {e}", path.display());
                log::error!("{msg}");
                self.shader_error = Some(msg);
                return;
            }
        };

        // Try to create the shader module — validation happens here.
        let module = self.ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(filename),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
        });

        // Map shader filename to the pass that uses it and recreate the pipeline.
        match filename {
            "radiance_inject.wgsl" => {
                self.radiance_inject.recreate_pipeline(&self.ctx.device, &module);
            }
            "ray_march.wgsl" => {
                self.ray_march.recreate_pipeline(&self.ctx.device, &module);
            }
            _ => {
                log::info!("Shader changed but no hot-reload mapping: {filename}");
                return;
            }
        }
        self.shader_error = None;
        log::info!("Shader hot-reloaded: {filename}");
    }

    /// Recompose the uber-shader from the current ShaderComposer state and recompile.
    fn recompose_and_recompile(&mut self) {
        let source = self.shader_composer.compose().to_string();
        // Dump composed shader for debugging.
        let _ = std::fs::write("/tmp/rkf_composed_shade.wgsl", &source);
        log::info!("Composed shade shader: {} bytes, dumped to /tmp/rkf_composed_shade.wgsl", source.len());
        self.shading_pass.recompile(&self.ctx.device, &source);
        self.shader_error = None;

        // Resolve shader IDs in the material library.
        {
            let mut lib = self.material_library.lock().unwrap();
            let composer = &self.shader_composer;
            lib.resolve_shader_ids(|name| composer.shader_id(name));
        }

        log::info!("Shade uber-shader recompiled ({} shaders registered)",
            self.shader_composer.shader_names().len());
    }

    /// Scan `assets/shaders/` for user shader files and register them.
    pub(crate) fn scan_user_shaders(&mut self) {
        let shader_dir = match &self.project_root {
            Some(root) => root.join("assets/shaders"),
            None => return,
        };
        if !shader_dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(&shader_dir) {
            Ok(e) => e,
            Err(e) => {
                log::warn!("Failed to read {}: {e}", shader_dir.display());
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("wgsl") {
                continue;
            }

            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("Failed to read {}: {e}", path.display());
                    continue;
                }
            };

            let shader_name = Self::extract_shader_name(&source).unwrap_or_else(|| {
                let filename = path.file_name().unwrap().to_str().unwrap_or("unknown");
                filename.strip_prefix("shade_")
                    .and_then(|s| s.strip_suffix(".wgsl"))
                    .unwrap_or(filename.trim_end_matches(".wgsl"))
                    .to_string()
            });

            let file_path = path.display().to_string();
            let id = self.shader_composer.register_with_path(&shader_name, source, Some(file_path));
            log::info!("Registered user shader: {shader_name} (id={id}) from {}", path.display());
        }

        // Recompose the uber-shader so new shaders take effect immediately.
        self.recompose_and_recompile();
    }

    /// Reinitialize the file watcher to include project-specific directories.
    pub(crate) fn reinit_file_watcher(&mut self) {
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        if let Some(ref root) = self.project_root {
            let assets = root.join("assets");
            if assets.exists() {
                dirs.push(assets);
            }
        }
        let dir_refs: Vec<&std::path::Path> = dirs.iter().map(|d| d.as_path()).collect();
        match rkf_runtime::FileWatcher::new(&dir_refs) {
            Ok(w) => {
                log::info!("File watcher reinitialized for {} dirs", dirs.len());
                self.file_watcher = Some(w);
            }
            Err(e) => {
                log::warn!("File watcher reinit failed: {e}");
            }
        }
    }

    /// Extract shader name from WGSL source by looking for `fn shade_<name>(`.
    fn extract_shader_name(source: &str) -> Option<String> {
        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("fn shade_") {
                if let Some(paren) = rest.find('(') {
                    let name = rest[..paren].trim();
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
        None
    }

    /// Sync geometry-first data to the brick pool for a specific object.
    ///
    /// For each allocated slot in the geometry brick map, converts
    /// BrickGeometry + SdfCache to a Brick and writes it to cpu_brick_pool.
    pub(super) fn sync_geometry_to_bricks(&mut self, object_id: u32) {
        let data = match self.geometry_first_data.get(&object_id) {
            Some(d) => d,
            None => return,
        };
        for (&geo_slot, &(sdf_slot, brick_slot)) in &data.slot_map {
            let geo = self.cpu_geometry_pool.get(geo_slot);
            let cache = self.cpu_sdf_cache_pool.get(sdf_slot);
            let brick = Brick::from_geometry(geo, cache);
            *self.cpu_brick_pool.get_mut(brick_slot) = brick;
        }
    }

    /// Convert an analytical primitive object to a geometry-first voxelized object.
    ///
    /// Populates `geometry_first_data` for the given `object_id`.
    ///
    /// If `bake_scale` is `Some(scale)`, the primitive is voxelized at scaled
    /// dimensions — the AABB is expanded by the per-axis scale factors and the
    /// SDF is evaluated in un-scaled space with conservative distance correction.
    /// The caller should reset the object's scale to `(1,1,1)` afterwards.
    ///
    /// Returns the new BrickMapHandle and voxel_size, or None if conversion fails.
    pub fn convert_to_geometry_first(
        &mut self,
        primitive: &SdfPrimitive,
        material_id: u8,
        voxel_size: f32,
        object_id: u32,
        bake_scale: Option<Vec3>,
    ) -> Option<(rkf_core::scene_node::BrickMapHandle, f32, Aabb, u32)> {
        use rkf_core::brick_map::{BrickMap, EMPTY_SLOT};
        use rkf_core::voxelize_object::voxelize_to_geometry;

        let scale = bake_scale.unwrap_or(Vec3::ONE);
        let half_extents = primitive_half_extents(primitive) * scale;
        let margin = voxel_size * 2.0;
        let aabb = Aabb::new(
            -half_extents - Vec3::splat(margin),
            half_extents + Vec3::splat(margin),
        );

        // Estimate brick count from AABB and voxel_size, grow pools if needed.
        let brick_size = voxel_size * 8.0;
        let dims_est = glam::UVec3::new(
            ((aabb.max.x - aabb.min.x) / brick_size).ceil() as u32,
            ((aabb.max.y - aabb.min.y) / brick_size).ceil() as u32,
            ((aabb.max.z - aabb.min.z) / brick_size).ceil() as u32,
        );
        let max_bricks = dims_est.x * dims_est.y * dims_est.z;
        if self.cpu_geometry_pool.free_count() < max_bricks {
            let new_cap = (self.cpu_geometry_pool.capacity() * 2).max(
                self.cpu_geometry_pool.capacity() + max_bricks,
            );
            self.cpu_geometry_pool.grow(new_cap);
        }
        if self.cpu_sdf_cache_pool.free_count() < max_bricks {
            let new_cap = (self.cpu_sdf_cache_pool.capacity() * 2).max(
                self.cpu_sdf_cache_pool.capacity() + max_bricks,
            );
            self.cpu_sdf_cache_pool.grow(new_cap);
        }

        // The voxelizer uses the SDF function for two things:
        // 1. Sign (inside/outside) → occupancy. Correct regardless of scale.
        // 2. Distance magnitude → narrow-band brick culling threshold.
        // SDF cache distances are computed from voxel geometry, not this function.
        //
        // For Box, evaluate with scaled half_extents — exact in scaled space.
        // For other primitives, evaluate in original space — sign is correct,
        // and conservative min_scale keeps narrow-band culling safe.
        let sdf_fn: Box<dyn Fn(Vec3) -> (f32, u8, [u8; 3])> =
            if let Some(scaled_prim) = scale_primitive(primitive, scale) {
                Box::new(move |pos: Vec3| {
                    let d = rkf_core::evaluate_primitive(&scaled_prim, pos);
                    (d, material_id, [255, 255, 255])
                })
            } else {
                let prim = primitive.clone();
                let min_scale = scale.x.min(scale.y).min(scale.z).max(1e-6);
                let inv_scale = Vec3::new(
                    1.0 / scale.x.max(1e-6),
                    1.0 / scale.y.max(1e-6),
                    1.0 / scale.z.max(1e-6),
                );
                Box::new(move |pos: Vec3| {
                    let d = rkf_core::evaluate_primitive(&prim, pos * inv_scale) * min_scale;
                    (d, material_id, [255, 255, 255])
                })
            };

        let result = voxelize_to_geometry(
            sdf_fn,
            &aabb,
            voxel_size,
            &mut self.cpu_geometry_pool,
            &mut self.cpu_sdf_cache_pool,
            &mut self.cpu_brick_map_alloc,
        )?;

        let brick_count = result.brick_count;
        let dims = result.handle.dims;

        // Build geometry-layer brick map (entries = geo pool slots).
        // The allocator's handle currently has geo pool slot indices in it
        // (that's what voxelize_to_geometry stores).
        let mut geo_brick_map = BrickMap::new(dims);
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(entry) = self.cpu_brick_map_alloc.get_entry(&result.handle, bx, by, bz) {
                        geo_brick_map.set(bx, by, bz, entry);
                    }
                }
            }
        }

        // Allocate brick pool slots for GPU staging
        let brick_slots = self.cpu_brick_pool.allocate_range(brick_count)
            .unwrap_or_else(|| {
                let new_cap = (self.cpu_brick_pool.capacity() * 2).max(
                    self.cpu_brick_pool.capacity() + brick_count,
                );
                self.cpu_brick_pool.grow(new_cap);
                self.cpu_brick_pool.allocate_range(brick_count)
                    .expect("allocation after grow")
            });

        // Build slot map and remap allocator entries from geo slots → brick pool slots
        let mut slot_map = std::collections::HashMap::new();
        for (i, (&geo_slot, &sdf_slot)) in result.geometry_slots.iter()
            .zip(result.sdf_slots.iter())
            .enumerate()
        {
            let brick_slot = brick_slots[i];
            slot_map.insert(geo_slot, (sdf_slot, brick_slot));

            // Convert geometry → brick and write to brick pool
            let geo = self.cpu_geometry_pool.get(geo_slot);
            let cache = self.cpu_sdf_cache_pool.get(sdf_slot);
            let brick = Brick::from_geometry(geo, cache);
            *self.cpu_brick_pool.get_mut(brick_slot) = brick;
        }

        // Remap entries in the packed allocator buffer to brick pool slots
        for bz in 0..dims.z {
            for by in 0..dims.y {
                for bx in 0..dims.x {
                    if let Some(entry) = self.cpu_brick_map_alloc.get_entry(&result.handle, bx, by, bz) {
                        if entry != EMPTY_SLOT {
                            if let Some(&(_, brick_slot)) = slot_map.get(&entry) {
                                self.cpu_brick_map_alloc.set_entry(&result.handle, bx, by, bz, brick_slot);
                            }
                        }
                    }
                }
            }
        }

        // Store geometry-first data
        self.geometry_first_data.insert(object_id, GeometryFirstData {
            geo_brick_map,
            slot_map,
            voxel_size,
        });

        // Compute grid AABB
        let brick_world = voxel_size * 8.0;
        let grid_half = Vec3::new(
            dims.x as f32 * brick_world * 0.5,
            dims.y as f32 * brick_world * 0.5,
            dims.z as f32 * brick_world * 0.5,
        );
        let grid_aabb = Aabb::new(-grid_half, grid_half);

        log::info!(
            "Converted to geometry-first: {} bricks, dims={:?}, voxel_size={}",
            brick_count, dims, voxel_size
        );

        Some((result.handle, voxel_size, grid_aabb, brick_count))
    }

    /// Clear all scene data from the engine, preparing for a fresh scene load.
    ///
    /// Frees all allocated brick pool slots, resets the brick map allocator,
    /// clears the coarse field, and marks topology as changed.
    pub fn clear_scene(&mut self) {
        // Deallocate all used brick pool slots by resetting to a fresh pool
        // with the same capacity.
        let capacity = self.cpu_brick_pool.capacity();
        self.cpu_brick_pool = BrickPool::new(capacity);
        self.cpu_brick_map_alloc = BrickMapAllocator::new();

        // Clear geometry-first pools.
        let geo_cap = self.cpu_geometry_pool.capacity();
        self.cpu_geometry_pool = GeometryPool::new(geo_cap.max(256));
        let sdf_cap = self.cpu_sdf_cache_pool.capacity();
        self.cpu_sdf_cache_pool = SdfCachePool::new(sdf_cap.max(256));
        self.geometry_first_data.clear();

        // Clear color companion pool.
        self.cpu_color_bricks.clear();
        self.color_companion_map = vec![0xFFFFFFFF; capacity as usize];

        // Clear cached GPU state.
        self.cached_gpu_objects.clear();
        self.cached_bvh = None;
        self.cached_bvh_pairs.clear();
        self.cached_world_aabbs.clear();
        self.object_gpu_ranges.clear();
        self.dirty_objects.clear();
        self.spawned_objects.clear();
        self.despawned_objects.clear();
        self.tombstone_count = 0;

        // Force full rebuild on next frame.
        self.topology_changed = true;
    }
}
