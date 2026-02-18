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
use rkf_core::populate::populate_grid;
use rkf_core::sdf::{box_sdf, capsule_sdf, sphere_sdf};
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::BrickPool;

use rkf_render::blit::BlitPass;
use rkf_render::camera::Camera;
use rkf_render::gpu_scene::{GpuScene, SceneUniforms};
use rkf_render::ray_march::{RayMarchPass, INTERNAL_HEIGHT, INTERNAL_WIDTH};
use rkf_render::RenderContext;

mod automation;
use automation::{SharedState, TestbedAutomationApi};

// ---------------------------------------------------------------------------
// Test scene creation
// ---------------------------------------------------------------------------

/// Resolution tier for the test scene (Tier 1 = 2cm voxels).
const TEST_TIER: usize = 1;

/// Create a multi-object test scene: sphere + box + capsule.
///
/// Returns `(BrickPool, SparseGrid, Aabb)`.
fn create_test_scene() -> (BrickPool, SparseGrid, Aabb) {
    // Object parameters
    let sphere_center = Vec3::ZERO;
    let sphere_radius = 0.35;

    let box_center = Vec3::new(0.8, 0.0, 0.0);
    let box_half = Vec3::splat(0.25);

    let capsule_a = Vec3::new(-0.8, -0.25, 0.0);
    let capsule_b = Vec3::new(-0.8, 0.25, 0.0);
    let capsule_radius = 0.15;

    // Combined AABB with margin
    let aabb = Aabb::new(Vec3::new(-1.5, -1.0, -1.0), Vec3::new(1.5, 1.0, 1.0));

    let res = &RESOLUTION_TIERS[TEST_TIER];
    let size = aabb.size();
    let dims = UVec3::new(
        ((size.x / res.brick_extent).ceil() as u32).max(1),
        ((size.y / res.brick_extent).ceil() as u32).max(1),
        ((size.z / res.brick_extent).ceil() as u32).max(1),
    );

    let mut pool: BrickPool = Pool::new(4096);
    let mut grid = SparseGrid::new(dims);

    // SDF union of all three objects
    let count = populate_grid(
        &mut pool,
        &mut grid,
        |p| {
            let s = sphere_sdf(sphere_center, sphere_radius, p);
            let b = box_sdf(box_half, p - box_center);
            let c = capsule_sdf(capsule_a, capsule_b, capsule_radius, p);
            s.min(b).min(c)
        },
        TEST_TIER,
        &aabb,
    )
    .expect("failed to populate grid");

    log::info!(
        "Test scene: {count} bricks, grid {}x{}x{}, tier {TEST_TIER}",
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
    ray_march: RayMarchPass,
    blit: BlitPass,
    camera: Camera,
    staging_buffer: wgpu::Buffer,
    shared_state: Arc<Mutex<SharedState>>,
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
        let surface_format =
            context.configure_surface(&surface, size.width.max(1), size.height.max(1));

        // Create test scene on CPU
        let (pool, grid, aabb) = create_test_scene();

        // Update shared state with pool info
        {
            let mut state = shared_state.lock().unwrap();
            state.pool_capacity = pool.capacity() as u64;
            state.pool_allocated = pool.allocated_count() as u64;
        }

        // Camera positioned to see all three objects (sphere, box, capsule)
        let mut camera = Camera::new(Vec3::new(0.0, 0.5, 3.0));
        camera.fov_degrees = 60.0;

        let camera_uniforms = camera.uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let camera_bytes = bytemuck::bytes_of(&camera_uniforms);

        let dims = grid.dimensions();
        let res = &RESOLUTION_TIERS[TEST_TIER];
        let scene_uniforms = SceneUniforms {
            grid_dims: [dims.x, dims.y, dims.z, 0],
            grid_origin: [aabb.min.x, aabb.min.y, aabb.min.z, res.brick_extent],
            params: [res.voxel_size, 0.0, 0.0, 0.0],
        };

        // Upload to GPU
        let scene = GpuScene::upload(
            &context.device,
            &pool,
            &grid,
            camera_bytes,
            &scene_uniforms,
        );

        // Create render passes
        let ray_march =
            RayMarchPass::new(&context.device, &scene, INTERNAL_WIDTH, INTERNAL_HEIGHT);
        let blit = BlitPass::new(&context.device, &ray_march.output_view, surface_format);

        // Staging buffer for CPU readback of rendered frames (screenshot support).
        // 960 * 4 = 3840 bytes per row, which is already 256-byte aligned.
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
            width: size.width.max(1),
            height: size.height.max(1),
            scene,
            ray_march,
            blit,
            camera,
            staging_buffer,
            shared_state,
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
        let uniforms = self.camera.uniforms(INTERNAL_WIDTH, INTERNAL_HEIGHT);
        self.context.queue.write_buffer(
            &self.scene.camera_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
    }

    fn render(&mut self, dt: f32) {
        // Update camera uniforms on GPU
        self.update_camera();

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

        // Pass 1: Ray march (compute) → output texture
        self.ray_march.dispatch(&mut encoder, &self.scene);

        // Pass 2: Blit output texture → swapchain
        self.blit.draw(&mut encoder, &view);

        // Pass 3: Copy output texture → staging buffer for screenshot readback
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: self.ray_march.output_texture(),
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

    fn process_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            _ => {}
        }
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
                let _ = window.set_cursor_grab(CursorGrabMode::Locked)
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
            .with_title("RKIField Testbed")
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

        log::info!("Window created — click to capture mouse, WASD to move, mouse to look, Esc to exit");
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
                    self.input.process_key(key, pressed);
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
                            "RKIField Testbed — {fps:.0} fps ({:.2} ms)",
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
