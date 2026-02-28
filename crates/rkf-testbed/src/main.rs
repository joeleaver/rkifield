//! RKIField visual testbed — v2 object-centric render loop.
//!
//! Scene: 5 analytical + 1 voxelized SDF objects rendered via the v2
//! object-centric ray marcher with BVH acceleration. Uses the public
//! World + Renderer API from `rkf_runtime::api`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use glam::{Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

use rkf_core::{Aabb, SdfPrimitive, SdfSource, SceneNode};
use rkf_render::Light;
use rkf_runtime::api::{Entity, Renderer, RendererConfig, World};
use rkf_animation::character::{
    AnimatedCharacter, build_humanoid_skeleton, build_humanoid_visuals, build_walk_clip,
};

mod automation;
use automation::{SharedState, TestbedAutomationApi};

/// Display (output) resolution width — the window size.
const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height — the window size.
const DISPLAY_HEIGHT: u32 = 720;

/// Internal render resolution width.
const INTERNAL_WIDTH: u32 = 960;
/// Internal render resolution height.
const INTERNAL_HEIGHT: u32 = 540;

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

/// Build the demo scene using the World API.
///
/// Returns the character and its entity handle for per-frame animation.
fn build_demo_scene(world: &mut World) -> (AnimatedCharacter, Entity) {
    // 1. Ground plane (large flat box)
    world.spawn("ground")
        .position_vec3(Vec3::new(0.0, -0.6, 0.0))
        .sdf(SdfPrimitive::Box {
            half_extents: Vec3::new(10.0, 0.1, 10.0),
        })
        .material(1)
        .build();

    // 2. Sphere (analytical)
    world.spawn("sphere")
        .position_vec3(Vec3::new(-1.5, 0.0, -2.0))
        .sdf(SdfPrimitive::Sphere { radius: 0.5 })
        .material(2)
        .build();

    // 3. Box (analytical)
    world.spawn("box")
        .position_vec3(Vec3::new(0.0, 0.0, -2.0))
        .rotation(Quat::from_rotation_y(0.5))
        .sdf(SdfPrimitive::Box {
            half_extents: Vec3::new(0.35, 0.35, 0.35),
        })
        .material(3)
        .build();

    // 4. Capsule (analytical)
    world.spawn("capsule")
        .position_vec3(Vec3::new(1.5, 0.0, -2.0))
        .sdf(SdfPrimitive::Capsule {
            radius: 0.2,
            half_height: 0.4,
        })
        .material(4)
        .build();

    // 5. Torus (analytical)
    world.spawn("torus")
        .position_vec3(Vec3::new(0.0, 0.3, -3.5))
        .sdf(SdfPrimitive::Torus {
            major_radius: 0.4,
            minor_radius: 0.12,
        })
        .material(5)
        .build();

    // 6. Voxelized sphere — demonstrates voxelized SDF rendering path.
    let vox_radius = 0.4;
    let voxel_size = 0.04; // 4cm voxels
    let margin = voxel_size * 2.0;
    let vox_aabb = Aabb::new(
        Vec3::splat(-vox_radius - margin),
        Vec3::splat(vox_radius + margin),
    );

    let sdf_fn = |pos: Vec3| -> (f32, u16) {
        (pos.length() - vox_radius, 6u16) // material 6 = distinct color
    };

    let (handle, brick_count) = world
        .voxelize(sdf_fn, &vox_aabb, voxel_size)
        .expect("voxelize sphere");

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

    world.spawn("vox_sphere")
        .position_vec3(Vec3::new(3.0, 0.0, -2.0))
        .sdf_tree(vox_node)
        .build();

    // 7. Animated humanoid character (14-bone skeleton as SceneNode tree).
    let skeleton = build_humanoid_skeleton();
    let visuals = build_humanoid_visuals(5); // material 5 = skin-like
    let walk_clip = build_walk_clip();
    let character = AnimatedCharacter::new(skeleton, visuals, walk_clip, 0.08);
    let char_root = character.build_scene_node();

    let char_entity = world.spawn("humanoid")
        .position_vec3(Vec3::new(-3.0, 0.0, -2.0))
        .sdf_tree(char_root)
        .build();

    (character, char_entity)
}

// ---------------------------------------------------------------------------
// IPC server setup
// ---------------------------------------------------------------------------

fn start_ipc_server(
    rt: &tokio::runtime::Runtime,
    api: Arc<dyn rkf_core::automation::AutomationApi>,
) -> (tokio::task::JoinHandle<()>, String) {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let meta_path = socket_path.replace(".sock", ".json");
    let meta = serde_json::json!({
        "type": "testbed",
        "pid": std::process::id(),
        "socket": &socket_path,
        "name": "RKIField Testbed",
        "version": "0.1.0"
    });
    let _ = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap());

    let config = rkf_mcp::ipc::IpcConfig {
        socket_path: Some(socket_path.clone()),
        tcp_port: 0,
        mode: rkf_mcp::registry::ToolMode::Editor,
    };

    let handle = rt.spawn(async move {
        if let Err(e) = rkf_mcp::ipc::run_server(config, api).await {
            log::error!("IPC server error: {e}");
        }
    });

    (handle, socket_path)
}

fn cleanup_ipc(socket_path: &str) {
    let _ = std::fs::remove_file(socket_path);
    let meta_path = socket_path.replace(".sock", ".json");
    let _ = std::fs::remove_file(&meta_path);
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    window: Option<Arc<Window>>,
    world: Option<World>,
    renderer: Option<Renderer>,
    surface: Option<wgpu::Surface<'static>>,
    shared_state: Arc<Mutex<SharedState>>,
    last_frame: Instant,
    _ipc_handle: Option<tokio::task::JoinHandle<()>>,
    socket_path: Option<String>,
    rt: tokio::runtime::Runtime,
    // Animation
    character: Option<AnimatedCharacter>,
    character_entity: Option<Entity>,
    last_anim_time: Instant,
    // Input state
    keys_held: std::collections::HashSet<KeyCode>,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        let shared_state = Arc::new(Mutex::new(SharedState::new(
            0, 0, INTERNAL_WIDTH, INTERNAL_HEIGHT,
        )));
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        Self {
            window: None,
            world: None,
            renderer: None,
            surface: None,
            shared_state,
            last_frame: Instant::now(),
            _ipc_handle: None,
            socket_path: None,
            rt,
            character: None,
            character_entity: None,
            last_anim_time: Instant::now(),
            keys_held: std::collections::HashSet::new(),
            mouse_captured: false,
        }
    }

    fn process_movement(&mut self, dt: f32) {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };

        let speed = renderer.camera().move_speed * dt;
        let cam = renderer.camera_mut();
        if self.keys_held.contains(&KeyCode::KeyW) {
            cam.translate_forward(speed);
        }
        if self.keys_held.contains(&KeyCode::KeyS) {
            cam.translate_forward(-speed);
        }
        if self.keys_held.contains(&KeyCode::KeyA) {
            cam.translate_right(-speed);
        }
        if self.keys_held.contains(&KeyCode::KeyD) {
            cam.translate_right(speed);
        }
        if self.keys_held.contains(&KeyCode::Space) {
            cam.translate_up(speed);
        }
        if self.keys_held.contains(&KeyCode::ShiftLeft) {
            cam.translate_up(-speed);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("RKIField Testbed — v2 Object-Centric")
            .with_inner_size(PhysicalSize::new(DISPLAY_WIDTH, DISPLAY_HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.window = Some(window.clone());

        // Initialize Renderer (creates GPU device + surface).
        let config = RendererConfig {
            internal_width: INTERNAL_WIDTH,
            internal_height: INTERNAL_HEIGHT,
            display_width: DISPLAY_WIDTH,
            display_height: DISPLAY_HEIGHT,
        };
        let (mut renderer, surface) = Renderer::new(window.clone(), config);

        // Configure camera.
        renderer.set_camera_position(Vec3::new(0.0, 1.0, -0.5));
        renderer.set_camera_orientation(0.0, -0.15);
        renderer.camera_mut().move_speed = 3.0;

        // Configure lights: 1 directional (sun) + 2 point lights.
        renderer.add_light(Light::directional(
            [0.5, 1.0, 0.3],
            [1.0, 0.95, 0.85],
            3.0,
            true,
        ));
        renderer.add_light(Light::point(
            [2.0, 1.5, -1.0],
            [1.0, 0.8, 0.5],
            5.0,
            8.0,
            true,
        ));
        renderer.add_light(Light::point(
            [-2.0, 1.0, -3.0],
            [0.5, 0.7, 1.0],
            3.0,
            6.0,
            false,
        ));

        // Build demo scene via World API.
        let mut world = World::new("testbed");
        let (character, char_entity) = build_demo_scene(&mut world);

        self.world = Some(world);
        self.renderer = Some(renderer);
        self.surface = Some(surface);
        self.character = Some(character);
        self.character_entity = Some(char_entity);
        self.last_anim_time = Instant::now();

        // Start IPC server.
        let api = TestbedAutomationApi::new(Arc::clone(&self.shared_state));
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(api);
        let (handle, path) = start_ipc_server(&self.rt, api);
        self._ipc_handle = Some(handle);
        self.socket_path = Some(path);

        log::info!("v2 testbed initialized — 5 analytical + 1 voxelized + 1 animated, using World + Renderer API");
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.mouse_captured {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.camera_mut().rotate(delta.0, delta.1);
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(ref path) = self.socket_path {
                    cleanup_ipc(path);
                }
                event_loop.exit();
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    self.mouse_captured = state == ElementState::Pressed;
                    if let Some(window) = &self.window {
                        window.set_cursor_visible(!self.mouse_captured);
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_held.insert(key);

                            match key {
                                KeyCode::Escape => {
                                    if let Some(ref path) = self.socket_path {
                                        cleanup_ipc(path);
                                    }
                                    event_loop.exit();
                                }
                                // Debug mode toggle: 1-4
                                KeyCode::Digit1 => {
                                    if let Some(r) = self.renderer.as_mut() {
                                        r.set_debug_mode(0);
                                        log::info!("Debug mode: Lambert");
                                    }
                                }
                                KeyCode::Digit2 => {
                                    if let Some(r) = self.renderer.as_mut() {
                                        r.set_debug_mode(1);
                                        log::info!("Debug mode: Normals");
                                    }
                                }
                                KeyCode::Digit3 => {
                                    if let Some(r) = self.renderer.as_mut() {
                                        r.set_debug_mode(2);
                                        log::info!("Debug mode: Positions");
                                    }
                                }
                                KeyCode::Digit4 => {
                                    if let Some(r) = self.renderer.as_mut() {
                                        r.set_debug_mode(3);
                                        log::info!("Debug mode: Material IDs");
                                    }
                                }
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.keys_held.remove(&key);
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // Apply camera controls.
                self.process_movement(dt);

                // Apply pending MCP commands.
                if let Some(renderer) = self.renderer.as_mut() {
                    if let Ok(mut state) = self.shared_state.lock() {
                        if let Some(cam) = state.pending_camera.take() {
                            renderer.set_camera_position(cam.position);
                            renderer.set_camera_orientation(cam.yaw, cam.pitch);
                        }
                        if let Some(mode) = state.pending_debug_mode.take() {
                            renderer.set_debug_mode(mode);
                        }
                        // Update shared state for MCP observation.
                        state.camera_position = renderer.camera().position;
                        state.camera_yaw = renderer.camera().yaw;
                        state.camera_pitch = renderer.camera().pitch;
                        state.frame_time_ms = dt as f64 * 1000.0;
                    }
                }

                // Advance character animation.
                let anim_now = Instant::now();
                let anim_dt = (anim_now - self.last_anim_time).as_secs_f32().min(0.1);
                self.last_anim_time = anim_now;
                if let (Some(character), Some(char_entity), Some(world)) = (
                    self.character.as_mut(),
                    self.character_entity,
                    self.world.as_mut(),
                ) {
                    if let Ok(root_node) = world.root_node_mut(char_entity) {
                        character.advance_and_update(anim_dt, root_node);
                    }
                }

                // Render frame.
                if let (Some(renderer), Some(world), Some(surface)) = (
                    self.renderer.as_mut(),
                    self.world.as_ref(),
                    self.surface.as_ref(),
                ) {
                    renderer.render_to_surface(world, surface);

                    // Handle MCP screenshot requests.
                    let do_readback = self.shared_state.lock()
                        .map(|s| s.screenshot_requested)
                        .unwrap_or(false);
                    if do_readback {
                        let pixels = renderer.screenshot();
                        if let Ok(mut state) = self.shared_state.lock() {
                            state.frame_pixels = pixels;
                            state.screenshot_requested = false;
                        }
                    }
                }

                // Request next frame.
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    log::info!("RKIField Testbed v2 — object-centric ray marching (World + Renderer API)");

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
