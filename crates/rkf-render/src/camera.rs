//! Camera system for ray generation.
//!
//! [`Camera`] provides a fly-camera with yaw/pitch rotation and position.
//! [`CameraUniforms`] is the GPU-uploadable struct with pre-scaled direction
//! vectors encoding FOV and aspect ratio for simple ray generation in shaders.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Compute the n-th element of the Halton sequence for the given base.
///
/// Returns a value in [0, 1) that is quasi-random and low-discrepancy,
/// providing good coverage of the unit interval over many samples.
pub fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let inv_base = 1.0 / base as f32;
    while index > 0 {
        f *= inv_base;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Compute sub-pixel jitter offset for a given frame index.
///
/// Uses Halton bases 2 and 3 with a 16-frame cycle. Returns offsets
/// in [-0.5, 0.5) pixel units suitable for adding to pixel center coordinates.
pub fn jitter_for_frame(frame_index: u32) -> [f32; 2] {
    let i = frame_index % 16;
    [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5]
}

/// Default vertical field of view in degrees.
pub const DEFAULT_FOV_DEGREES: f32 = 60.0;

/// Default movement speed in meters per second.
pub const DEFAULT_MOVE_SPEED: f32 = 2.0;

/// Default mouse sensitivity in radians per pixel.
pub const DEFAULT_MOUSE_SENSITIVITY: f32 = 0.003;

/// A fly-camera with position and yaw/pitch orientation.
#[derive(Debug, Clone)]
pub struct Camera {
    /// World-space position.
    pub position: Vec3,
    /// Horizontal rotation in radians (0 = -Z forward, positive = left).
    pub yaw: f32,
    /// Vertical rotation in radians (0 = horizontal, positive = up).
    pub pitch: f32,
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
    /// Movement speed in meters per second.
    pub move_speed: f32,
    /// Mouse sensitivity in radians per pixel.
    pub mouse_sensitivity: f32,
}

impl Camera {
    /// Create a new camera at the given position, looking along -Z.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov_degrees: DEFAULT_FOV_DEGREES,
            move_speed: DEFAULT_MOVE_SPEED,
            mouse_sensitivity: DEFAULT_MOUSE_SENSITIVITY,
        }
    }

    /// Forward direction vector (unit length).
    #[inline]
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    /// Right direction vector (unit length).
    #[inline]
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    }

    /// Up direction vector (unit length, camera-relative).
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    /// Move the camera along its forward axis.
    pub fn translate_forward(&mut self, amount: f32) {
        self.position += self.forward() * amount;
    }

    /// Move the camera along its right axis.
    pub fn translate_right(&mut self, amount: f32) {
        self.position += self.right() * amount;
    }

    /// Move the camera along world up (Y axis).
    pub fn translate_up(&mut self, amount: f32) {
        self.position += Vec3::Y * amount;
    }

    /// Rotate the camera by mouse delta (in pixels).
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx as f32 * self.mouse_sensitivity;
        self.pitch -= dy as f32 * self.mouse_sensitivity;
        // Clamp pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    /// Build GPU-uploadable camera uniforms.
    ///
    /// The `right` and `up` vectors are pre-scaled by FOV and aspect ratio
    /// so the shader can do simple ray generation:
    /// ```text
    /// ndc = uv * 2.0 - 1.0;
    /// ray_dir = normalize(forward + ndc.x * right + ndc.y * up);
    /// ```
    pub fn uniforms(&self, width: u32, height: u32, frame_index: u32) -> CameraUniforms {
        let fov_rad = self.fov_degrees.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = width as f32 / height as f32;

        let fwd = self.forward();
        let r = self.right() * half_fov_tan * aspect;
        let u = self.up() * half_fov_tan;

        CameraUniforms {
            position: [self.position.x, self.position.y, self.position.z, 0.0],
            forward: [fwd.x, fwd.y, fwd.z, 0.0],
            right: [r.x, r.y, r.z, 0.0],
            up: [u.x, u.y, u.z, 0.0],
            resolution: [width as f32, height as f32],
            jitter: jitter_for_frame(frame_index),
        }
    }
}

/// GPU-uploadable camera uniforms (96 bytes, 16-byte aligned).
///
/// Direction vectors are pre-scaled so the shader only needs:
/// ```text
/// ray_dir = normalize(forward + ndc.x * right + ndc.y * up);
/// ```
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniforms {
    /// Camera world-space position (xyz) + padding.
    pub position: [f32; 4],
    /// Forward direction (xyz) + padding.
    pub forward: [f32; 4],
    /// Right direction scaled by FOV * aspect (xyz) + padding.
    pub right: [f32; 4],
    /// Up direction scaled by FOV (xyz) + padding.
    pub up: [f32; 4],
    /// Output resolution in pixels `[width, height]`.
    pub resolution: [f32; 2],
    /// Sub-pixel jitter offset in pixel units `[jitter_x, jitter_y]`.
    /// Applied to pixel center before ray generation for temporal super-sampling.
    pub jitter: [f32; 2],
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        (a - b).length() < eps
    }

    // ------ Construction ------

    #[test]
    fn new_camera_at_origin() {
        let cam = Camera::new(Vec3::ZERO);
        assert_eq!(cam.position, Vec3::ZERO);
        assert_eq!(cam.yaw, 0.0);
        assert_eq!(cam.pitch, 0.0);
    }

    #[test]
    fn new_camera_at_position() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cam.position, Vec3::new(1.0, 2.0, 3.0));
    }

    // ------ Direction vectors ------

    #[test]
    fn forward_default_is_neg_z() {
        let cam = Camera::new(Vec3::ZERO);
        assert!(approx_eq(cam.forward(), Vec3::new(0.0, 0.0, -1.0), 1e-6));
    }

    #[test]
    fn right_default_is_pos_x() {
        let cam = Camera::new(Vec3::ZERO);
        assert!(approx_eq(cam.right(), Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn up_default_is_pos_y() {
        let cam = Camera::new(Vec3::ZERO);
        let up = cam.up();
        assert!(approx_eq(up, Vec3::new(0.0, 1.0, 0.0), 1e-6));
    }

    #[test]
    fn forward_after_yaw_90_is_pos_x() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.yaw = -FRAC_PI_2; // Turn right 90 degrees
        assert!(approx_eq(cam.forward(), Vec3::new(1.0, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn forward_after_pitch_up() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.pitch = FRAC_PI_2 * 0.99; // Look almost straight up
        let fwd = cam.forward();
        assert!(fwd.y > 0.9, "should be pointing up, got {fwd:?}");
    }

    #[test]
    fn directions_are_orthogonal() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.yaw = 0.7;
        cam.pitch = 0.3;
        let f = cam.forward();
        let r = cam.right();
        let u = cam.up();
        assert!(f.dot(r).abs() < 1e-5, "forward and right not orthogonal");
        assert!(f.dot(u).abs() < 1e-5, "forward and up not orthogonal");
        assert!(r.dot(u).abs() < 1e-5, "right and up not orthogonal");
    }

    #[test]
    fn directions_are_unit_length() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.yaw = 1.2;
        cam.pitch = -0.4;
        assert!((cam.forward().length() - 1.0).abs() < 1e-5);
        assert!((cam.right().length() - 1.0).abs() < 1e-5);
        assert!((cam.up().length() - 1.0).abs() < 1e-5);
    }

    // ------ Translation ------

    #[test]
    fn translate_forward_moves_along_forward() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.translate_forward(1.0);
        assert!(approx_eq(cam.position, Vec3::new(0.0, 0.0, -1.0), 1e-6));
    }

    #[test]
    fn translate_right_moves_along_right() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.translate_right(1.0);
        assert!(approx_eq(cam.position, Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn translate_up_moves_along_y() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.translate_up(1.0);
        assert!(approx_eq(cam.position, Vec3::new(0.0, 1.0, 0.0), 1e-6));
    }

    // ------ Rotation ------

    #[test]
    fn rotate_changes_yaw() {
        let mut cam = Camera::new(Vec3::ZERO);
        let old_yaw = cam.yaw;
        cam.rotate(100.0, 0.0);
        assert_ne!(cam.yaw, old_yaw);
    }

    #[test]
    fn rotate_changes_pitch() {
        let mut cam = Camera::new(Vec3::ZERO);
        let old_pitch = cam.pitch;
        cam.rotate(0.0, 100.0);
        assert_ne!(cam.pitch, old_pitch);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut cam = Camera::new(Vec3::ZERO);
        cam.rotate(0.0, -100000.0); // Huge upward look
        assert!(cam.pitch < FRAC_PI_2);
        assert!(cam.pitch > -FRAC_PI_2);
    }

    // ------ CameraUniforms ------

    #[test]
    fn uniforms_size_is_80_bytes() {
        assert_eq!(std::mem::size_of::<CameraUniforms>(), 80);
    }

    #[test]
    fn uniforms_resolution_matches() {
        let cam = Camera::new(Vec3::ZERO);
        let u = cam.uniforms(960, 540, 0);
        assert_eq!(u.resolution, [960.0, 540.0]);
    }

    #[test]
    fn uniforms_position_matches() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0));
        let u = cam.uniforms(960, 540, 0);
        assert_eq!(u.position[0], 1.0);
        assert_eq!(u.position[1], 2.0);
        assert_eq!(u.position[2], 3.0);
    }

    #[test]
    fn uniforms_right_is_scaled_by_fov_and_aspect() {
        let cam = Camera::new(Vec3::ZERO);
        let u = cam.uniforms(960, 540, 0);
        let fov_rad = DEFAULT_FOV_DEGREES.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let aspect = 960.0 / 540.0;
        let expected_length = half_fov_tan * aspect;
        let right_len =
            (u.right[0] * u.right[0] + u.right[1] * u.right[1] + u.right[2] * u.right[2]).sqrt();
        assert!(
            (right_len - expected_length).abs() < 1e-5,
            "right length {right_len} != expected {expected_length}"
        );
    }

    #[test]
    fn uniforms_up_is_scaled_by_fov() {
        let cam = Camera::new(Vec3::ZERO);
        let u = cam.uniforms(960, 540, 0);
        let fov_rad = DEFAULT_FOV_DEGREES.to_radians();
        let half_fov_tan = (fov_rad * 0.5).tan();
        let up_len = (u.up[0] * u.up[0] + u.up[1] * u.up[1] + u.up[2] * u.up[2]).sqrt();
        assert!(
            (up_len - half_fov_tan).abs() < 1e-5,
            "up length {up_len} != expected {half_fov_tan}"
        );
    }

    #[test]
    fn uniforms_pod_roundtrip() {
        let cam = Camera::new(Vec3::new(1.0, 2.0, 3.0));
        let u = cam.uniforms(1920, 1080, 0);
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 80);
        let u2: &CameraUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.position, u2.position);
        assert_eq!(u.resolution, u2.resolution);
    }

    // ------ Halton sequence ------

    #[test]
    fn halton_base2_first_values() {
        // Halton(1,2)=0.5, Halton(2,2)=0.25, Halton(3,2)=0.75, Halton(4,2)=0.125
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(4, 2) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn halton_base3_first_values() {
        // Halton(1,3)=1/3, Halton(2,3)=2/3, Halton(3,3)=1/9
        assert!((halton(1, 3) - 1.0/3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0/3.0).abs() < 1e-6);
        assert!((halton(3, 3) - 1.0/9.0).abs() < 1e-6);
    }

    #[test]
    fn halton_index_zero_is_zero() {
        assert_eq!(halton(0, 2), 0.0);
        assert_eq!(halton(0, 3), 0.0);
    }

    #[test]
    fn jitter_range_is_half_pixel() {
        for i in 0..16 {
            let j = jitter_for_frame(i);
            assert!(j[0] >= -0.5 && j[0] < 0.5, "jitter_x out of range at frame {i}: {}", j[0]);
            assert!(j[1] >= -0.5 && j[1] < 0.5, "jitter_y out of range at frame {i}: {}", j[1]);
        }
    }

    #[test]
    fn jitter_16_frame_cycle() {
        let j0 = jitter_for_frame(0);
        let j16 = jitter_for_frame(16);
        assert_eq!(j0, j16, "jitter should repeat after 16 frames");
    }

    #[test]
    fn uniforms_jitter_at_frame_zero() {
        let cam = Camera::new(Vec3::ZERO);
        let u = cam.uniforms(960, 540, 0);
        let expected = jitter_for_frame(0);
        assert_eq!(u.jitter, expected);
    }

    // ------ Full yaw sweep ------

    #[test]
    fn yaw_sweep_forward_is_continuous() {
        let mut cam = Camera::new(Vec3::ZERO);
        let mut prev = cam.forward();
        for i in 1..=36 {
            cam.yaw = (i as f32) * PI / 18.0; // 10-degree steps
            let fwd = cam.forward();
            let dist = (fwd - prev).length();
            assert!(
                dist < 0.35,
                "step {i}: forward jumped {dist} (prev={prev:?}, cur={fwd:?})"
            );
            prev = fwd;
        }
    }
}
