//! Editor camera with orbit, fly, and follow modes.
//!
//! The editor camera supports three interaction modes:
//! - **Orbit**: Rotate around a target point (right-drag), pan (middle-drag), zoom (scroll).
//! - **Fly**: Free-flight with WASD movement and mouse-look.
//! - **Follow**: Track a specific entity (future use).
//!
//! All math uses `glam` types. View and projection matrices are right-handed.

use crate::input::{InputState, KeyCode};
use glam::{Mat4, Vec3};

/// Camera interaction mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraMode {
    /// Orbit around a target point.
    Orbit,
    /// Free-flight camera.
    Fly,
    /// Follow a specific entity (entity ID stored for ECS lookup).
    Follow { target_entity: u64 },
}

/// Editor camera state and controls.
///
/// Supports orbit mode (rotate/pan/zoom around a target), fly mode (WASD + mouse-look),
/// and follow mode (track an entity). All angles are in radians.
#[derive(Clone, Copy)]
pub struct SceneCamera {
    /// Current world-space position.
    pub position: Vec3,
    /// Look-at target point (orbit center in orbit mode).
    pub target: Vec3,
    /// Up vector (typically Y-up).
    pub up: Vec3,
    /// Current interaction mode.
    pub mode: CameraMode,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,

    // Orbit mode state
    /// Distance from target in orbit mode.
    pub orbit_distance: f32,
    /// Horizontal angle in radians (yaw, around Y axis).
    pub orbit_yaw: f32,
    /// Vertical angle in radians (pitch, clamped to avoid gimbal lock).
    pub orbit_pitch: f32,

    // Fly mode state
    /// Fly-mode yaw (horizontal rotation).
    pub fly_yaw: f32,
    /// Fly-mode pitch (vertical rotation).
    pub fly_pitch: f32,

    // Speed / sensitivity
    /// Movement speed in meters per second (fly mode).
    pub fly_speed: f32,
    /// Orbit rotation sensitivity (radians per pixel of mouse delta).
    pub orbit_speed: f32,
    /// Zoom sensitivity (distance units per scroll tick).
    pub zoom_speed: f32,

    // Limits
    /// Minimum orbit distance (prevents clipping through target).
    pub min_orbit_distance: f32,
    /// Maximum orbit distance.
    pub max_orbit_distance: f32,
    /// Minimum pitch angle (radians). Prevents looking straight down.
    pub min_pitch: f32,
    /// Maximum pitch angle (radians). Prevents looking straight up.
    pub max_pitch: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCamera {
    /// Create a new editor camera with sensible defaults.
    ///
    /// Default position: (0, 5, 10) looking at the origin.
    /// FOV: 60 degrees, orbit distance: 10m.
    pub fn new() -> Self {
        let mut cam = Self {
            position: Vec3::new(0.0, 5.0, 10.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            mode: CameraMode::Orbit,
            fov_y: 60.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            orbit_distance: 10.0,
            orbit_yaw: 0.0,
            orbit_pitch: 0.3,
            fly_yaw: 0.0,
            fly_pitch: 0.0,
            fly_speed: 5.0,
            orbit_speed: 0.005,
            zoom_speed: 1.0,
            min_orbit_distance: 0.5,
            max_orbit_distance: 500.0,
            min_pitch: -1.4, // ~-80 degrees
            max_pitch: 1.4,  // ~+80 degrees
        };
        cam.update_from_orbit();
        cam
    }

    /// Rotate the orbit camera by a mouse delta (in pixels).
    ///
    /// `dx` rotates horizontally (yaw), `dy` rotates vertically (pitch).
    /// Pitch is clamped to `[min_pitch, max_pitch]` to prevent gimbal lock.
    pub fn orbit_rotate(&mut self, dx: f32, dy: f32) {
        self.orbit_yaw += dx * self.orbit_speed;
        self.orbit_pitch = (self.orbit_pitch + dy * self.orbit_speed)
            .clamp(self.min_pitch, self.max_pitch);
        self.update_from_orbit();
    }

    /// Zoom the orbit camera by a scroll delta.
    ///
    /// Positive delta zooms in (decreases distance), negative zooms out.
    /// Distance is clamped to `[min_orbit_distance, max_orbit_distance]`.
    pub fn orbit_zoom(&mut self, delta: f32) {
        self.orbit_distance = (self.orbit_distance - delta * self.zoom_speed)
            .clamp(self.min_orbit_distance, self.max_orbit_distance);
        self.update_from_orbit();
    }

    /// Pan the orbit camera in its local right/up plane.
    ///
    /// `dx` moves along the camera's right vector, `dy` moves along the camera's up vector.
    /// Pan speed scales with orbit distance for consistent feel at any zoom level.
    pub fn orbit_pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let cam_up = right.cross(forward).normalize();

        let scale = self.orbit_distance * 0.002;
        let offset = right * (-dx * scale) + cam_up * (dy * scale);

        self.target += offset;
        self.update_from_orbit();
    }

    /// Rotate the fly-mode camera by a mouse delta.
    ///
    /// `dx` rotates yaw, `dy` rotates pitch. Pitch is clamped.
    /// Yaw is subtracted so that dragging right → looking right (matches
    /// the render camera's forward convention where yaw=0 faces -Z).
    pub fn fly_rotate(&mut self, dx: f32, dy: f32) {
        self.fly_yaw -= dx * self.orbit_speed;
        self.fly_pitch = (self.fly_pitch - dy * self.orbit_speed)
            .clamp(self.min_pitch, self.max_pitch);

        // Update target from fly orientation
        let dir = fly_direction(self.fly_yaw, self.fly_pitch);
        self.target = self.position + dir;
    }

    /// Move the fly-mode camera along its local axes.
    ///
    /// `forward`: positive = forward, negative = backward.
    /// `right`: positive = right, negative = left.
    /// `up`: positive = up, negative = down.
    /// `dt`: delta time in seconds.
    pub fn fly_move(&mut self, forward: f32, right: f32, up: f32, dt: f32) {
        let dir = fly_direction(self.fly_yaw, self.fly_pitch);
        let right_vec = dir.cross(Vec3::Y).normalize();
        let up_vec = Vec3::Y;

        let velocity = (dir * forward + right_vec * right + up_vec * up) * self.fly_speed * dt;
        self.position += velocity;
        self.target = self.position + dir;
    }

    /// Recompute position from orbit parameters (target + distance + yaw + pitch).
    ///
    /// Uses spherical coordinates:
    /// ```text
    /// position = target + Vec3(cos(pitch)*sin(yaw), sin(pitch), cos(pitch)*cos(yaw)) * distance
    /// ```
    pub fn update_from_orbit(&mut self) {
        let offset = Vec3::new(
            self.orbit_pitch.cos() * self.orbit_yaw.sin(),
            self.orbit_pitch.sin(),
            self.orbit_pitch.cos() * self.orbit_yaw.cos(),
        ) * self.orbit_distance;

        self.position = self.target + offset;

        // Keep fly_yaw/fly_pitch in sync so sync_to_engine_camera
        // gives the render camera the correct view direction.
        let dir = (self.target - self.position).normalize();
        self.fly_pitch = dir.y.asin();
        self.fly_yaw = (-dir.x).atan2(-dir.z);
    }

    /// Compute the right-handed view matrix (world → camera space).
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// Compute the right-handed perspective projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Set orbit yaw/pitch and recompute position (for camera presets).
    pub fn set_orbit_angles(&mut self, yaw: f32, pitch: f32) {
        self.orbit_yaw = yaw;
        self.orbit_pitch = pitch.clamp(self.min_pitch, self.max_pitch);
        self.mode = CameraMode::Orbit;
        self.update_from_orbit();
    }

    /// Snap the orbit camera to look at a specific point from a given distance.
    pub fn focus_on(&mut self, point: Vec3, distance: f32) {
        self.target = point;
        self.orbit_distance = distance.clamp(self.min_orbit_distance, self.max_orbit_distance);
        self.update_from_orbit();
    }

    /// Main per-frame update method. Reads input state and updates the camera.
    ///
    /// In orbit mode:
    /// - Right mouse drag: orbit rotate
    /// - Middle mouse drag: orbit pan
    /// - Scroll wheel: orbit zoom
    ///
    /// In fly mode:
    /// - Mouse delta (when right button held): fly rotate
    /// - WASD: forward/back/left/right
    /// - Q/E: down/up
    pub fn update(&mut self, input: &InputState, dt: f32) {
        match self.mode {
            CameraMode::Orbit => {
                // Right mouse: orbit rotate
                if input.is_mouse_button_down(1) {
                    self.orbit_rotate(input.mouse_delta.x, input.mouse_delta.y);
                }
                // Middle mouse: orbit pan
                if input.is_mouse_button_down(2) {
                    self.orbit_pan(input.mouse_delta.x, input.mouse_delta.y);
                }
                // Scroll: zoom
                if input.scroll_delta.abs() > f32::EPSILON {
                    self.orbit_zoom(input.scroll_delta);
                }
            }
            CameraMode::Fly => {
                // Mouse look (always active in fly mode when right button held)
                if input.is_mouse_button_down(1) {
                    self.fly_rotate(input.mouse_delta.x, input.mouse_delta.y);
                }

                // WASD + Q/E movement
                let mut forward = 0.0_f32;
                let mut right = 0.0_f32;
                let mut up = 0.0_f32;

                if input.is_key_pressed(KeyCode::W) {
                    forward += 1.0;
                }
                if input.is_key_pressed(KeyCode::S) {
                    forward -= 1.0;
                }
                if input.is_key_pressed(KeyCode::D) {
                    right += 1.0;
                }
                if input.is_key_pressed(KeyCode::A) {
                    right -= 1.0;
                }
                if input.is_key_pressed(KeyCode::E) || input.is_key_pressed(KeyCode::Space) {
                    up += 1.0;
                }
                if input.is_key_pressed(KeyCode::Q) || input.is_key_pressed(KeyCode::ShiftLeft) {
                    up -= 1.0;
                }

                if forward != 0.0 || right != 0.0 || up != 0.0 {
                    self.fly_move(forward, right, up, dt);
                }

                // Scroll wheel: nudge camera forward/backward
                if input.scroll_delta.abs() > f32::EPSILON {
                    let dir = fly_direction(self.fly_yaw, self.fly_pitch);
                    self.position += dir * input.scroll_delta * self.zoom_speed;
                    self.target = self.position + dir;
                }
            }
            CameraMode::Follow { .. } => {
                // Follow mode: would track entity transform via ECS query.
                // Not yet implemented — requires runtime ECS integration.
            }
        }
    }
}

/// Generate a world-space ray from a screen pixel coordinate.
///
/// `pixel_x`, `pixel_y` are in physical (pixel) coordinates relative to the
/// viewport origin. `vp_width`, `vp_height` are the viewport dimensions in pixels.
///
/// Returns `(ray_origin, ray_direction)` where `ray_direction` is normalized.
pub fn screen_to_ray(
    cam: &SceneCamera,
    pixel_x: f32,
    pixel_y: f32,
    vp_width: f32,
    vp_height: f32,
) -> (Vec3, Vec3) {
    let aspect = vp_width / vp_height;
    let view = cam.view_matrix();
    let proj = cam.projection_matrix(aspect);
    let inv_vp = (proj * view).inverse();

    // Normalize pixel to NDC [-1, 1].
    let ndc_x = (pixel_x / vp_width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_y / vp_height) * 2.0; // Y flipped

    let near_clip = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
    let far_clip = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));

    let dir = (far_clip - near_clip).normalize();
    (cam.position, dir)
}

/// Compute a direction vector from yaw and pitch angles.
///
/// Matches the render camera convention: yaw=0, pitch=0 → facing -Z.
/// This is the standard OpenGL/wgpu convention where -Z is "into the screen."
fn fly_direction(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    )
    .normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    /// Helper to check approximate float equality.
    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    /// Helper to check approximate Vec3 equality.
    fn vec3_approx_eq(a: Vec3, b: Vec3, epsilon: f32) -> bool {
        approx_eq(a.x, b.x, epsilon)
            && approx_eq(a.y, b.y, epsilon)
            && approx_eq(a.z, b.z, epsilon)
    }

    #[test]
    fn test_default_camera() {
        let cam = SceneCamera::new();

        // Should be in orbit mode
        assert_eq!(cam.mode, CameraMode::Orbit);

        // Target at origin
        assert_eq!(cam.target, Vec3::ZERO);

        // Position should be above and behind origin (positive Y and Z)
        assert!(cam.position.y > 0.0, "camera should be above origin");
        assert!(cam.position.z > 0.0, "camera should be behind origin (positive Z)");

        // Distance should match orbit_distance
        let actual_dist = (cam.position - cam.target).length();
        assert!(
            approx_eq(actual_dist, cam.orbit_distance, 0.01),
            "position should be orbit_distance from target: got {actual_dist}, expected {}",
            cam.orbit_distance
        );

        // FOV should be ~60 degrees
        assert!(approx_eq(cam.fov_y, 60.0_f32.to_radians(), 0.001));
    }

    #[test]
    fn test_orbit_rotate_yaw() {
        let mut cam = SceneCamera::new();
        let initial_yaw = cam.orbit_yaw;
        let initial_z = cam.position.z;

        // Rotate right (positive dx)
        cam.orbit_rotate(100.0, 0.0);

        assert!(
            cam.orbit_yaw > initial_yaw,
            "yaw should increase with positive dx"
        );

        // Position x should change (camera moved around the target)
        // The exact x depends on the yaw, but it should differ from original
        // At yaw=0 the camera is on the +Z axis; rotating increases yaw so
        // position moves along the XZ circle.
        assert!(
            (cam.position.z - initial_z).abs() > 0.01 || cam.position.x.abs() > 0.01,
            "position should change when rotating"
        );

        // Distance from target should remain constant
        let dist = (cam.position - cam.target).length();
        assert!(
            approx_eq(dist, cam.orbit_distance, 0.01),
            "orbit distance should be preserved"
        );
    }

    #[test]
    fn test_orbit_rotate_pitch_clamp() {
        let mut cam = SceneCamera::new();

        // Try to rotate pitch far past the maximum
        cam.orbit_rotate(0.0, 100_000.0);
        assert!(
            cam.orbit_pitch <= cam.max_pitch,
            "pitch should be clamped to max: {} > {}",
            cam.orbit_pitch,
            cam.max_pitch
        );

        // Try to rotate pitch far past the minimum
        cam.orbit_rotate(0.0, -200_000.0);
        assert!(
            cam.orbit_pitch >= cam.min_pitch,
            "pitch should be clamped to min: {} < {}",
            cam.orbit_pitch,
            cam.min_pitch
        );
    }

    #[test]
    fn test_orbit_zoom() {
        let mut cam = SceneCamera::new();
        let initial_dist = cam.orbit_distance;

        // Scroll in (positive delta = closer)
        cam.orbit_zoom(2.0);
        assert!(
            cam.orbit_distance < initial_dist,
            "zooming in should decrease distance"
        );

        // Distance from target should match orbit_distance
        let actual_dist = (cam.position - cam.target).length();
        assert!(approx_eq(actual_dist, cam.orbit_distance, 0.01));
    }

    #[test]
    fn test_orbit_zoom_clamp() {
        let mut cam = SceneCamera::new();

        // Zoom way in
        cam.orbit_zoom(10_000.0);
        assert!(
            cam.orbit_distance >= cam.min_orbit_distance,
            "distance should not go below minimum"
        );

        // Zoom way out
        cam.orbit_zoom(-100_000.0);
        assert!(
            cam.orbit_distance <= cam.max_orbit_distance,
            "distance should not exceed maximum"
        );
    }

    #[test]
    fn test_orbit_pan() {
        let mut cam = SceneCamera::new();
        let initial_target = cam.target;

        // Pan to the right
        cam.orbit_pan(100.0, 0.0);
        assert!(
            !vec3_approx_eq(cam.target, initial_target, 0.001),
            "target should move when panning"
        );

        // Distance should remain the same
        let dist = (cam.position - cam.target).length();
        assert!(
            approx_eq(dist, cam.orbit_distance, 0.01),
            "orbit distance should be preserved after pan"
        );
    }

    #[test]
    fn test_orbit_pan_vertical() {
        let mut cam = SceneCamera::new();
        let initial_y = cam.target.y;

        // Pan upward (positive dy)
        cam.orbit_pan(0.0, 100.0);
        assert!(
            cam.target.y > initial_y,
            "panning up should increase target Y"
        );
    }

    #[test]
    fn test_fly_move_forward() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.fly_yaw = 0.0;
        cam.fly_pitch = 0.0;

        let initial_pos = cam.position;

        // Move forward (fly_yaw=0, fly_pitch=0 means looking along -Z)
        cam.fly_move(1.0, 0.0, 0.0, 1.0);

        let delta = cam.position - initial_pos;
        assert!(
            delta.z < 0.0,
            "moving forward at yaw=0 should decrease Z (camera faces -Z): delta = {delta:?}"
        );
    }

    #[test]
    fn test_fly_move_right() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.fly_yaw = 0.0;
        cam.fly_pitch = 0.0;

        let initial_pos = cam.position;

        // Move right
        cam.fly_move(0.0, 1.0, 0.0, 1.0);

        let delta = cam.position - initial_pos;
        // At yaw=0 looking along -Z, right should be +X direction
        assert!(
            delta.x.abs() > 0.01,
            "moving right should change X position: delta = {delta:?}"
        );
    }

    #[test]
    fn test_fly_move_up() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;

        let initial_y = cam.position.y;

        cam.fly_move(0.0, 0.0, 1.0, 1.0);
        assert!(
            cam.position.y > initial_y,
            "moving up should increase Y"
        );
    }

    #[test]
    fn test_fly_rotate() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.fly_yaw = 0.0;
        cam.fly_pitch = 0.0;

        cam.fly_rotate(100.0, 0.0);
        assert!(
            cam.fly_yaw.abs() > 0.01,
            "fly_rotate should change yaw"
        );

        cam.fly_rotate(0.0, 50.0);
        assert!(
            cam.fly_pitch.abs() > 0.01,
            "fly_rotate should change pitch"
        );
    }

    #[test]
    fn test_fly_rotate_pitch_clamp() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;

        cam.fly_rotate(0.0, -1_000_000.0);
        assert!(cam.fly_pitch <= cam.max_pitch);
        assert!(cam.fly_pitch >= cam.min_pitch);
    }

    #[test]
    fn test_view_matrix() {
        let cam = SceneCamera::new();
        let view = cam.view_matrix();

        // View matrix should be invertible (non-zero determinant)
        assert!(
            view.determinant().abs() > 1e-6,
            "view matrix should be invertible"
        );

        // The view matrix transforms position to origin-ish
        let cam_in_view = view.transform_point3(cam.position);
        assert!(
            vec3_approx_eq(cam_in_view, Vec3::ZERO, 0.01),
            "camera position in view space should be near origin: {cam_in_view:?}"
        );
    }

    #[test]
    fn test_projection_matrix() {
        let cam = SceneCamera::new();
        let proj = cam.projection_matrix(16.0 / 9.0);

        // Projection matrix should be invertible
        assert!(
            proj.determinant().abs() > 1e-6,
            "projection matrix should be invertible"
        );

        // Near plane points should map to z near 0 (wgpu/Vulkan RH convention, depth [0,1])
        // A point at (0, 0, -near) in view space should map to z = 0 in NDC
        let near_point = proj.project_point3(Vec3::new(0.0, 0.0, -cam.near));
        assert!(
            approx_eq(near_point.z, 0.0, 0.05),
            "near plane should map to z near 0: got {}",
            near_point.z
        );
    }

    #[test]
    fn test_focus_on() {
        let mut cam = SceneCamera::new();
        let focus_point = Vec3::new(10.0, 5.0, -3.0);
        let focus_dist = 8.0;

        cam.focus_on(focus_point, focus_dist);

        assert_eq!(cam.target, focus_point);
        assert!(
            approx_eq(cam.orbit_distance, focus_dist, 0.001),
            "orbit distance should match requested distance"
        );

        let actual_dist = (cam.position - cam.target).length();
        assert!(
            approx_eq(actual_dist, focus_dist, 0.01),
            "actual position-to-target distance should match"
        );
    }

    #[test]
    fn test_focus_on_clamps_distance() {
        let mut cam = SceneCamera::new();

        // Focus with distance below minimum
        cam.focus_on(Vec3::ZERO, 0.01);
        assert!(
            cam.orbit_distance >= cam.min_orbit_distance,
            "focus_on should clamp distance to minimum"
        );

        // Focus with distance above maximum
        cam.focus_on(Vec3::ZERO, 10_000.0);
        assert!(
            cam.orbit_distance <= cam.max_orbit_distance,
            "focus_on should clamp distance to maximum"
        );
    }

    #[test]
    fn test_update_orbit_mode_right_drag() {
        let mut cam = SceneCamera::new();
        let initial_yaw = cam.orbit_yaw;

        let mut input = InputState::new();
        input.mouse_buttons[1] = true; // right mouse
        input.mouse_delta = glam::Vec2::new(50.0, 0.0);

        cam.update(&input, 1.0 / 60.0);

        assert!(
            cam.orbit_yaw != initial_yaw,
            "orbit yaw should change from right-drag"
        );
    }

    #[test]
    fn test_update_orbit_mode_scroll() {
        let mut cam = SceneCamera::new();
        let initial_dist = cam.orbit_distance;

        let mut input = InputState::new();
        input.scroll_delta = 3.0;

        cam.update(&input, 1.0 / 60.0);

        assert!(
            cam.orbit_distance < initial_dist,
            "scroll should zoom in"
        );
    }

    #[test]
    fn test_update_fly_mode_wasd() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.fly_yaw = 0.0;
        cam.fly_pitch = 0.0;

        let initial_pos = cam.position;

        let mut input = InputState::new();
        input.keys_pressed.insert(KeyCode::W);

        cam.update(&input, 1.0);

        assert!(
            !vec3_approx_eq(cam.position, initial_pos, 0.01),
            "pressing W in fly mode should move the camera"
        );
    }

    #[test]
    fn test_orbit_preserves_distance_after_rotation() {
        let mut cam = SceneCamera::new();

        // Multiple rotations should preserve distance
        for i in 0..100 {
            cam.orbit_rotate(3.7, (i as f32) * 0.1 - 5.0);
        }

        let dist = (cam.position - cam.target).length();
        assert!(
            approx_eq(dist, cam.orbit_distance, 0.01),
            "distance should be preserved after many rotations: got {dist}, expected {}",
            cam.orbit_distance
        );
    }

    #[test]
    fn test_camera_mode_eq() {
        assert_eq!(CameraMode::Orbit, CameraMode::Orbit);
        assert_eq!(CameraMode::Fly, CameraMode::Fly);
        assert_ne!(CameraMode::Orbit, CameraMode::Fly);
        assert_eq!(
            CameraMode::Follow { target_entity: 42 },
            CameraMode::Follow { target_entity: 42 }
        );
        assert_ne!(
            CameraMode::Follow { target_entity: 1 },
            CameraMode::Follow { target_entity: 2 }
        );
    }

    #[test]
    fn test_fly_speed_scaling() {
        let mut cam = SceneCamera::new();
        cam.mode = CameraMode::Fly;
        cam.fly_yaw = FRAC_PI_4;
        cam.fly_pitch = 0.0;

        let pos1 = cam.position;
        cam.fly_move(1.0, 0.0, 0.0, 0.5);
        let half_step = (cam.position - pos1).length();

        cam.position = pos1;
        cam.fly_move(1.0, 0.0, 0.0, 1.0);
        let full_step = (cam.position - pos1).length();

        assert!(
            approx_eq(full_step, half_step * 2.0, 0.01),
            "movement should scale linearly with dt: half={half_step}, full={full_step}"
        );
    }
}
