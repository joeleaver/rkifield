use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use glam::{UVec3, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

use rkf_core::aabb::Aabb;
use rkf_core::brick_pool::Pool;
use rkf_core::constants::RESOLUTION_TIERS;
use rkf_core::populate::populate_grid_with_material;
use rkf_core::sdf::sphere_sdf;
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::BrickPool;

use rkf_render::blit::BlitPass;
use rkf_render::camera::Camera;
use rkf_render::clipmap_gpu::ClipmapGpuData;
use rkf_render::gbuffer::GBuffer;
use rkf_render::gpu_color_pool::GpuColorPool;
use rkf_render::gpu_scene::{GpuScene, SceneUniforms};
use rkf_render::light::{Light, LightBuffer};
use rkf_render::material_table::{self, MaterialTable};
use rkf_render::radiance_volume::RadianceVolume;
use rkf_render::ray_march::{RayMarchPass, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use rkf_render::shading::ShadingPass;
use rkf_render::tile_cull::TileCullPass;
use rkf_render::tone_map::ToneMapPass;
use rkf_render::RenderContext;

mod automation;
use automation::{SharedState, TestbedAutomationApi};

// ---------------------------------------------------------------------------
// Phase 4 scene — single sphere
// ---------------------------------------------------------------------------

fn create_phase4_scene() -> (BrickPool, SparseGrid, Aabb) {
    let aabb = Aabb::new(Vec3::splat(-1.5), Vec3::splat(1.5));
    let res = &RESOLUTION_TIERS[1]; // Tier 1 = 2cm
    let size = aabb.size();
    let dims = UVec3::new(
        ((size.x / res.brick_extent).ceil() as u32).max(1),
        ((size.y / res.brick_extent).ceil() as u32).max(1),
        ((size.z / res.brick_extent).ceil() as u32).max(1),
    );
    let mut pool: BrickPool = Pool::new(4096);
    let mut grid = SparseGrid::new(dims);
    let count = populate_grid_with_material(
        &mut pool,
        &mut grid,
        |p| sphere_sdf(Vec3::ZERO, 0.5, p),
        1,
        &aabb,
        1,
    )
    .expect("sphere");
    log::info!(
        "Phase 4 scene: {count} bricks, grid {}x{}x{}",
        dims.x,
        dims.y,
        dims.z
    );
    (pool, grid, aabb)
}

// ---------------------------------------------------------------------------
// GPU state
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct GpuState {
    context: RenderContext,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    scene: GpuScene,
    clipmap: ClipmapGpuData,
    gbuffer: GBuffer,
    material_table: MaterialTable,
    light_buffer: LightBuffer,
    tile_cull: TileCullPass,
    radiance_volume: RadianceVolume,
    color_pool: GpuColorPool,
    ray_march: RayMarchPass,
    shading: ShadingPass,
    tone_map: ToneMapPass,
    blit: BlitPass,
    camera: Camera,
    staging_buffer: wgpu::Buffer,
    shared_state: Arc<Mutex<SharedState>>,
    frame_index: u32,
}

impl GpuState {
    fn new(window: Arc<Window>, shared_state: Arc<Mutex<SharedState>>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .expect("failed to create surface");
        let context = RenderContext::new(&instance, &surface);
        let display_width = size.width.max(1);
        let display_height = size.height.max(1);
        let surface_format =
            context.configure_surface(&surface, display_width, display_height);

        // Scene
        let (pool, grid, aabb) = create_phase4_scene();

        // Update shared state with pool info
        {
            let mut state = shared_state.lock().unwrap();
            state.pool_capacity = pool.capacity() as u64;
            state.pool_allocated = pool.allocated_count() as u64;
        }

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 0.0, 2.0));
        camera.fov_degrees = 60.0;

        let camera_uniforms = camera.uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, 0, [[0.0; 4]; 4]);
        let camera_bytes = bytemuck::bytes_of(&camera_uniforms);

        // SceneUniforms from grid
        let dims = grid.dimensions();
        let scene_uniforms = SceneUniforms {
            grid_dims: [dims.x, dims.y, dims.z, 0],
            grid_origin: [
                aabb.min.x,
                aabb.min.y,
                aabb.min.z,
                RESOLUTION_TIERS[1].brick_extent,
            ],
            params: [RESOLUTION_TIERS[1].voxel_size, 0.0, 0.0, 0.0],
        };

        // Upload scene to GPU
        let scene = GpuScene::upload(&context.device, &pool, &grid, camera_bytes, &scene_uniforms);

        // Materials
        let materials = material_table::create_test_materials();
        let material_table = MaterialTable::upload(&context.device, &materials);

        // G-buffer
        let gbuffer = GBuffer::new(&context.device, INTERNAL_WIDTH, INTERNAL_HEIGHT);

        // Single directional light
        let lights = vec![Light::directional(
            [0.4, 0.8, 0.3],
            [1.0, 0.95, 0.85],
            1.0,
            true,
        )];
        let light_buffer = LightBuffer::upload(&context.device, &lights);

        // Tile cull
        let tile_cull = TileCullPass::new(
            &context.device,
            &gbuffer,
            &light_buffer,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );

        // Dummy/minimal resources needed by ShadingPass constructor
        let radiance_volume = RadianceVolume::new(&context.device);
        let color_pool = GpuColorPool::empty(&context.device);
        let clipmap = ClipmapGpuData::empty(&context.device);

        // Render passes
        let ray_march = RayMarchPass::new(&context.device, &scene, &gbuffer, &clipmap);
        let shading = ShadingPass::new(
            &context.device,
            &gbuffer,
            &material_table,
            &scene,
            &tile_cull.shade_light_bind_group_layout,
            &radiance_volume,
            &color_pool,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        let tone_map = ToneMapPass::new(
            &context.device,
            &shading.hdr_view,
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        );
        let blit = BlitPass::new(&context.device, &tone_map.ldr_view, surface_format);

        // Staging buffer for CPU readback of rendered frames (screenshot support).
        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot staging"),
            size: (INTERNAL_WIDTH * INTERNAL_HEIGHT * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            context,
            surface,
            surface_format,
            width: display_width,
            height: display_height,
            scene,
            clipmap,
            gbuffer,
            material_table,
            light_buffer,
            tile_cull,
            radiance_volume,
            color_pool,
            ray_march,
            shading,
            tone_map,
            blit,
            camera,
            staging_buffer,
            shared_state,
            frame_index: 0,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.context
            .configure_surface(&self.surface, width, height);
    }

    fn update_camera(&mut self) {
        let uniforms =
            self.camera
                .uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT, self.frame_index, [[0.0; 4]; 4]);
        self.context.queue.write_buffer(
            &self.scene.camera_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn render(&mut self, dt: f32) {
        // Check for pending debug mode change from MCP
        if let Ok(mut state) = self.shared_state.lock() {
            if let Some(mode) = state.pending_debug_mode.take() {
                self.shading.set_debug_mode(&self.context.queue, mode);
                log::info!("Debug mode set via MCP: {mode}");
            }
        }

        self.frame_index = self.frame_index.wrapping_add(1);

        // Update camera uniforms on GPU
        self.update_camera();

        // Update camera position for shading pass (correct view direction)
        let cam_pos = self.camera.position;
        self.shading
            .update_camera_pos(&self.context.queue, [cam_pos.x, cam_pos.y, cam_pos.z]);

        // Update tile cull uniforms with camera data
        let cam_fwd = self.camera.forward();
        self.tile_cull.update_uniforms(
            &self.context.queue,
            self.light_buffer.count,
            [cam_pos.x, cam_pos.y, cam_pos.z],
            [cam_fwd.x, cam_fwd.y, cam_fwd.z],
        );

        // Update shade uniforms with light/tile info
        self.shading.update_light_info(
            &self.context.queue,
            self.light_buffer.count,
            self.tile_cull.num_tiles_x,
            4,
        );

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(self.width, self.height);
                return;
            }
            Err(e) => {
                log::error!("surface error: {e}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        // Phase 4 pipeline: ray march -> tile cull -> shade -> tone map -> blit
        self.ray_march
            .dispatch(&mut encoder, &self.scene, &self.gbuffer, &self.clipmap);
        self.tile_cull.dispatch(&mut encoder, &self.gbuffer);
        self.shading.dispatch(
            &mut encoder,
            &self.gbuffer,
            &self.material_table,
            &self.scene,
            &self.tile_cull.shade_light_bind_group,
            &self.radiance_volume,
            &self.color_pool,
        );
        self.tone_map.dispatch(&mut encoder);
        self.blit.draw(&mut encoder, &view);

        // Copy LDR texture -> staging buffer for screenshot readback
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.tone_map.ldr_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INTERNAL_WIDTH * 4),
                    rows_per_image: Some(INTERNAL_HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: INTERNAL_WIDTH,
                height: INTERNAL_HEIGHT,
                depth_or_array_layers: 1,
            },
        );

        self.context.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        // Read back the staging buffer into shared state for MCP screenshot tool
        self.capture_frame();

        // Update shared state with camera and timing info
        if let Ok(mut state) = self.shared_state.lock() {
            state.camera_position = self.camera.position;
            state.camera_yaw = self.camera.yaw;
            state.camera_pitch = self.camera.pitch;
            state.camera_fov = self.camera.fov_degrees;
            state.frame_time_ms = dt as f64 * 1000.0;
        }
    }

    fn capture_frame(&mut self) {
        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.context.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            if let Ok(mut state) = self.shared_state.lock() {
                state.frame_pixels.copy_from_slice(&data);
            }
            drop(data);
            self.staging_buffer.unmap();
        }
    }
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    mouse_captured: bool,
}

impl InputState {
    fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            mouse_captured: false,
        }
    }

    fn process_key(&mut self, key: KeyCode, pressed: bool) -> Option<u32> {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            // Number keys set debug visualization mode (only on press)
            KeyCode::Digit0 if pressed => return Some(0),
            KeyCode::Digit1 if pressed => return Some(1),
            KeyCode::Digit2 if pressed => return Some(2),
            KeyCode::Digit3 if pressed => return Some(3),
            KeyCode::Digit4 if pressed => return Some(4),
            KeyCode::Digit5 if pressed => return Some(5),
            _ => {}
        }
        None
    }

    fn apply_to_camera(&self, camera: &mut Camera, dt: f32) {
        let speed = camera.move_speed * dt;
        if self.forward {
            camera.translate_forward(speed);
        }
        if self.backward {
            camera.translate_forward(-speed);
        }
        if self.right {
            camera.translate_right(speed);
        }
        if self.left {
            camera.translate_right(-speed);
        }
        if self.up {
            camera.translate_up(speed);
        }
        if self.down {
            camera.translate_up(-speed);
        }
    }
}

// ---------------------------------------------------------------------------
// IPC server
// ---------------------------------------------------------------------------

/// Spawn the IPC server on a background thread with its own tokio runtime.
///
/// Returns the socket path for cleanup.
fn spawn_ipc_server(api: Arc<dyn rkf_core::automation::AutomationApi>) -> String {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let path_clone = socket_path.clone();

    std::thread::Builder::new()
        .name("ipc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime for IPC server");

            rt.block_on(async {
                let mut registry = rkf_mcp::registry::ToolRegistry::new();
                rkf_mcp::tools::observation::register_observation_tools(&mut registry);
                let registry = Arc::new(registry);

                let config = rkf_mcp::ipc::IpcConfig {
                    socket_path: Some(path_clone),
                    tcp_port: 0,
                    mode: rkf_mcp::registry::ToolMode::Debug,
                };

                if let Err(e) = rkf_mcp::ipc::run_server(config, registry, api).await {
                    log::error!("IPC server error: {e}");
                }
            });
        })
        .expect("failed to spawn IPC server thread");

    socket_path
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    input: InputState,
    frame_count: u64,
    last_title_update: Instant,
    last_frame: Instant,
    socket_path: Option<String>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            input: InputState::new(),
            frame_count: 0,
            last_title_update: Instant::now(),
            last_frame: Instant::now(),
            socket_path: None,
        }
    }

    fn toggle_mouse_capture(&mut self) {
        if let Some(window) = &self.window {
            self.input.mouse_captured = !self.input.mouse_captured;
            if self.input.mouse_captured {
                let _ = window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                window.set_cursor_visible(false);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("RKIField Testbed [Phase 4]")
            .with_inner_size(PhysicalSize::new(1280u32, 720u32));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // Create shared state for automation API
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0, // will be updated by GpuState::new
            INTERNAL_WIDTH,
            INTERNAL_HEIGHT,
        )));

        self.gpu = Some(GpuState::new(window.clone(), Arc::clone(&shared_state)));
        self.window = Some(window);
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();

        // Spawn IPC server for MCP tool access
        let api: Arc<dyn rkf_core::automation::AutomationApi> =
            Arc::new(TestbedAutomationApi::new(shared_state));
        let socket_path = spawn_ipc_server(api);
        log::info!("IPC server listening on {socket_path}");
        self.socket_path = Some(socket_path);

        log::info!("Phase 4 validation — single sphere, basic shading");
        log::info!("Click to capture mouse, WASD to move, mouse to look, Esc to exit");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if !self.input.mouse_captured {
                    self.toggle_mouse_capture();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(key) = event.physical_key {
                    if key == KeyCode::Escape && pressed {
                        if self.input.mouse_captured {
                            self.toggle_mouse_capture();
                        } else {
                            event_loop.exit();
                        }
                        return;
                    }
                    if let Some(debug_mode) = self.input.process_key(key, pressed) {
                        if let Some(gpu) = &self.gpu {
                            gpu.shading.set_debug_mode(&gpu.context.queue, debug_mode);
                            log::info!("Debug mode: {debug_mode}");
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32();
                self.last_frame = now;
                self.frame_count += 1;

                // Update camera from input
                if let Some(gpu) = &mut self.gpu {
                    self.input.apply_to_camera(&mut gpu.camera, dt);
                    gpu.render(dt);
                }

                // Update title bar with frame time every 500ms
                if now.duration_since(self.last_title_update).as_millis() > 500 {
                    if let Some(window) = &self.window {
                        let elapsed = now.duration_since(self.last_title_update).as_secs_f64();
                        let fps = self.frame_count as f64 / elapsed;
                        window.set_title(&format!(
                            "RKIField Testbed [Phase 4] — {fps:.0} fps ({:.2} ms)",
                            1000.0 / fps
                        ));
                        self.frame_count = 0;
                        self.last_title_update = now;
                    }
                }

                // Request next frame
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.input.mouse_captured {
                if let Some(gpu) = &mut self.gpu {
                    gpu.camera.rotate(delta.0, delta.1);
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Clean up IPC socket file
        if let Some(path) = &self.socket_path {
            let _ = std::fs::remove_file(path);
            log::info!("Cleaned up IPC socket: {path}");
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
