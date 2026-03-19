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

/// Camera control state — owns only interaction parameters, not the camera transform.
///
/// Position, yaw, and pitch live on the ECS entity (`Transform` + `CameraComponent`).
/// This struct holds the orbit/fly control knobs and per-mode working state
/// (orbit target, distance, etc.) that don't belong on the entity.
#[derive(Debug, Clone, Copy)]
pub struct CameraControlState {
    /// Current interaction mode.
    pub mode: CameraMode,
    /// Look-at target point (orbit center in orbit mode).
    pub target: Vec3,
    /// Up vector (typically Y-up).
    pub up: Vec3,

    // Orbit mode state
    /// Distance from target in orbit mode.
    pub orbit_distance: f32,

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

impl Default for CameraControlState {
    fn default() -> Self {
        Self {
            mode: CameraMode::Orbit,
            target: Vec3::ZERO,
            up: Vec3::Y,
            orbit_distance: 10.0,
            fly_speed: 5.0,
            orbit_speed: 0.005,
            zoom_speed: 1.0,
            min_orbit_distance: 0.5,
            max_orbit_distance: 500.0,
            min_pitch: -1.4,
            max_pitch: 1.4,
        }
    }
}

impl CameraControlState {
    /// Main per-frame update. Reads input, mutates position/yaw/pitch via mutable refs.
    ///
    /// `position`, `yaw`, `pitch` are borrowed from the ECS entity's transform.
    /// This method applies orbit/fly controls and writes back through the refs.
    pub fn update(
        &mut self,
        input: &InputState,
        dt: f32,
        position: &mut Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
    ) {
        match self.mode {
            CameraMode::Orbit => {
                // Right mouse: orbit rotate
                if input.is_mouse_button_down(1) {
                    self.orbit_rotate(input.mouse_delta.x, input.mouse_delta.y, position, yaw, pitch);
                }
                // Middle mouse: orbit pan
                if input.is_mouse_button_down(2) {
                    self.orbit_pan(input.mouse_delta.x, input.mouse_delta.y, *position, yaw, pitch, position);
                }
                // Scroll: zoom
                if input.scroll_delta.abs() > f32::EPSILON {
                    self.orbit_zoom(input.scroll_delta, position, *yaw, *pitch);
                }
            }
            CameraMode::Fly => {
                // Mouse look (right button held)
                if input.is_mouse_button_down(1) {
                    self.fly_rotate(input.mouse_delta.x, input.mouse_delta.y, position, yaw, pitch);
                }

                // WASD + Q/E movement
                let mut forward = 0.0_f32;
                let mut right = 0.0_f32;
                let mut up = 0.0_f32;

                if input.is_key_pressed(KeyCode::W) { forward += 1.0; }
                if input.is_key_pressed(KeyCode::S) { forward -= 1.0; }
                if input.is_key_pressed(KeyCode::D) { right += 1.0; }
                if input.is_key_pressed(KeyCode::A) { right -= 1.0; }
                if input.is_key_pressed(KeyCode::E) || input.is_key_pressed(KeyCode::Space) { up += 1.0; }
                if input.is_key_pressed(KeyCode::Q) || input.is_key_pressed(KeyCode::ShiftLeft) { up -= 1.0; }

                if forward != 0.0 || right != 0.0 || up != 0.0 {
                    self.fly_move(forward, right, up, dt, position, *yaw, *pitch);
                }

                // Scroll wheel: nudge camera forward/backward
                if input.scroll_delta.abs() > f32::EPSILON {
                    let dir = fly_direction(*yaw, *pitch);
                    *position += dir * input.scroll_delta * self.zoom_speed;
                }
            }
            CameraMode::Follow { .. } => {}
        }
    }

    /// Orbit rotate by mouse delta. Updates yaw/pitch/position.
    pub fn orbit_rotate(
        &mut self,
        dx: f32,
        dy: f32,
        position: &mut Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
    ) {
        *yaw += dx * self.orbit_speed;
        *pitch = (*pitch + dy * self.orbit_speed).clamp(self.min_pitch, self.max_pitch);
        *position = self.position_from_orbit(*yaw, *pitch);
    }

    /// Zoom orbit camera. Updates position.
    pub fn orbit_zoom(
        &mut self,
        delta: f32,
        position: &mut Vec3,
        yaw: f32,
        pitch: f32,
    ) {
        self.orbit_distance = (self.orbit_distance - delta * self.zoom_speed)
            .clamp(self.min_orbit_distance, self.max_orbit_distance);
        *position = self.position_from_orbit(yaw, pitch);
    }

    /// Pan orbit camera in local right/up plane.
    pub fn orbit_pan(
        &mut self,
        dx: f32,
        dy: f32,
        cur_position: Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
        position: &mut Vec3,
    ) {
        let forward = (self.target - cur_position).normalize();
        let right = forward.cross(self.up).normalize();
        let cam_up = right.cross(forward).normalize();

        let scale = self.orbit_distance * 0.002;
        let offset = right * (-dx * scale) + cam_up * (dy * scale);

        self.target += offset;
        *position = self.position_from_orbit(*yaw, *pitch);
    }

    /// Fly-mode rotate by mouse delta.
    pub fn fly_rotate(
        &self,
        dx: f32,
        dy: f32,
        position: &mut Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
    ) {
        *yaw -= dx * self.orbit_speed;
        *pitch = (*pitch - dy * self.orbit_speed).clamp(self.min_pitch, self.max_pitch);
        let _ = position; // position unchanged during rotation
    }

    /// Fly-mode move along local axes.
    pub fn fly_move(
        &self,
        forward: f32,
        right: f32,
        up: f32,
        dt: f32,
        position: &mut Vec3,
        yaw: f32,
        pitch: f32,
    ) {
        let dir = fly_direction(yaw, pitch);
        let right_vec = dir.cross(Vec3::Y).normalize();
        let up_vec = Vec3::Y;

        let velocity = (dir * forward + right_vec * right + up_vec * up) * self.fly_speed * dt;
        *position += velocity;
    }

    /// Compute position from orbit parameters (target + distance + yaw + pitch).
    pub fn position_from_orbit(&self, yaw: f32, pitch: f32) -> Vec3 {
        let offset = Vec3::new(
            pitch.cos() * yaw.sin(),
            pitch.sin(),
            pitch.cos() * yaw.cos(),
        ) * self.orbit_distance;
        self.target + offset
    }

    /// Set orbit angles and recompute position.
    pub fn set_orbit_angles(
        &mut self,
        new_yaw: f32,
        new_pitch: f32,
        position: &mut Vec3,
        yaw: &mut f32,
        pitch: &mut f32,
    ) {
        *yaw = new_yaw;
        *pitch = new_pitch.clamp(self.min_pitch, self.max_pitch);
        self.mode = CameraMode::Orbit;
        *position = self.position_from_orbit(*yaw, *pitch);
    }

    /// Snap orbit to look at a specific point from a given distance.
    pub fn focus_on(
        &mut self,
        point: Vec3,
        distance: f32,
        position: &mut Vec3,
        yaw: f32,
        pitch: f32,
    ) {
        self.target = point;
        self.orbit_distance = distance.clamp(self.min_orbit_distance, self.max_orbit_distance);
        *position = self.position_from_orbit(yaw, pitch);
    }
}

/// Lightweight per-frame camera snapshot extracted from the ECS entity.
///
/// Contains everything needed to render a frame — position, orientation, and
/// projection parameters. Read-only after extraction.
#[derive(Debug, Clone, Copy)]
pub struct CameraSnapshot {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
}

impl CameraSnapshot {
    /// Compute the right-handed view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let dir = fly_direction(self.yaw, self.pitch);
        let target = self.position + dir;
        Mat4::look_at_rh(self.position, target, Vec3::Y)
    }

    /// Compute the right-handed perspective projection matrix.
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_degrees.to_radians(), aspect, self.near, self.far)
    }
}

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

/// Generate a world-space ray from a screen pixel using a `CameraSnapshot`.
pub fn screen_to_ray_snapshot(
    snap: &CameraSnapshot,
    pixel_x: f32,
    pixel_y: f32,
    vp_width: f32,
    vp_height: f32,
) -> (Vec3, Vec3) {
    let aspect = vp_width / vp_height;
    let view = snap.view_matrix();
    let proj = snap.projection_matrix(aspect);
    let inv_vp = (proj * view).inverse();

    let ndc_x = (pixel_x / vp_width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel_y / vp_height) * 2.0;

    let near_clip = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, -1.0));
    let far_clip = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));

    let dir = (far_clip - near_clip).normalize();
    (snap.position, dir)
}

/// Compute a direction vector from yaw and pitch angles (public).
///
/// Matches the render camera convention: yaw=0, pitch=0 → facing -Z.
pub fn fly_direction_pub(yaw: f32, pitch: f32) -> Vec3 {
    fly_direction(yaw, pitch)
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
}
