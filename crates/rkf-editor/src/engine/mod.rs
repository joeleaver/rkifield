//! Editor engine — v2 object-centric render pipeline.
//!
//! Extracted from the testbed's `EngineState`. Provides the full v2 compute-shader
//! render pipeline (ray march, shading, GI, volumetrics, post-processing) in a
//! reusable struct that the editor's event loop drives each frame.

use std::sync::{Arc, Mutex};

use glam::{Quat, Vec3};

use rkf_core::{
    Aabb, BrickMapAllocator, BrickPool, Scene, SceneNode, SceneObject,
    SdfPrimitive, SdfSource, voxelize_sdf,
};
use rkf_render::{
    AutoExposurePass, BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass,
    GBuffer, GodRaysBlurPass, GpuObject, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass,
    VolShadowPass, VolUpscalePass,
};
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;

use crate::automation::SharedState;
use crate::camera::EditorCamera;

mod init;
mod environment;
mod render;
mod sculpt;
mod sdf_repair;
mod sdf_fmm;
mod eikonal;
mod brick_ops;
mod brick_ops_repair;
mod offscreen;
mod query;

// ---------------------------------------------------------------------------
// Sculpt blend helpers
// ---------------------------------------------------------------------------

/// Polynomial smooth-minimum.  Returns the smooth union of two SDF distances.
///
/// `k` controls the blend radius: larger k → softer junction.
/// At k=0 this degenerates to `a.min(b)`.
#[inline]
pub(super) fn smin_poly(a: f32, b: f32, k: f32) -> f32 {
    if k <= 0.0 { return a.min(b); }
    let h = ((k - (a - b).abs()) / k).max(0.0);
    a.min(b) - h * h * k * 0.25
}

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

/// Build result containing the scene and CPU brick data for GPU upload.
pub(super) struct DemoScene {
    pub(super) scene: Scene,
    pub(super) brick_pool: BrickPool,
    pub(super) brick_map_alloc: BrickMapAllocator,
}

/// Try to load a voxelized object from a .rkf file into the brick pool.
///
/// Returns `(BrickMapHandle, voxel_size, grid_aabb, brick_count)` on success.
pub(super) fn load_rkf_into_pool(
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
pub(super) fn build_demo_scene() -> DemoScene {
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

    DemoScene {
        scene,
        brick_pool,
        brick_map_alloc,
    }
}

/// Compute the diameter of an analytical SDF primitive's bounding sphere.
pub(super) fn primitive_diameter(prim: &rkf_core::SdfPrimitive) -> f32 {
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
    pub(super) gpu_scene: GpuSceneV2,
    pub(super) gbuffer: GBuffer,
    pub(super) tile_cull: TileObjectCullPass,
    pub(super) coarse_field: CoarseField,
    pub(super) ray_march: RayMarchPass,
    #[allow(dead_code)]
    pub(super) debug_view: DebugViewPass,
    pub(super) shading_pass: ShadingPass,
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
    pub(super) material_buffer: wgpu::Buffer,
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
    // GPU JFA pipeline for per-stroke SDF repair.
    pub(super) jfa_sdf: crate::jfa_sdf::JfaSdfPass,
    // GPU eikonal PDE pipeline for per-stroke SDF repair.
    pub(super) eikonal_repair: crate::eikonal_repair::EikonalRepairPass,
    // ── Cached GPU state for incremental updates ──
    /// Cached GPU objects from last full rebuild. Used for partial updates.
    pub(super) cached_gpu_objects: Vec<GpuObject>,
    /// Cached BVH from last full rebuild.
    pub(super) cached_bvh: Option<rkf_core::Bvh>,
    /// Cached BVH pairs (gpu_idx, world_aabb) for BVH refit.
    pub(super) cached_bvh_pairs: Vec<(u32, Aabb)>,
    /// Cached world-space AABBs for coarse field population.
    pub(super) cached_world_aabbs: Vec<(Vec3, Vec3)>,
    /// Map from object_id → (start_index, count) in cached_gpu_objects.
    pub(super) object_gpu_ranges: std::collections::HashMap<u32, (usize, usize)>,
    /// Object IDs that need re-flattening this frame (transform changes, sculpt edits).
    pub(super) dirty_objects: std::collections::HashSet<u32>,
    /// Full scene topology changed (spawn/delete/open) — triggers complete rebuild.
    pub(super) topology_changed: bool,
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

    /// Returns true if a brick map slot is unallocated (no pool data).
    pub(super) fn is_unallocated(slot: u32) -> bool {
        slot == rkf_core::brick_map::EMPTY_SLOT || slot == rkf_core::brick_map::INTERIOR_SLOT
    }
}
