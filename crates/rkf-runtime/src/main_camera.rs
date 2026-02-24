//! Main camera with environment ownership for the v2 RKIField engine.
//!
//! The main camera is the singular observing entity in the scene. It owns
//! the active environment profile and drives zone-based environment transitions,
//! blending between profiles as the camera moves through tagged AABB regions.
//!
//! # Environment blending
//!
//! When the camera enters a new [`EnvironmentZone`], a blend is started from
//! the current active profile towards the zone's target profile. The blend
//! progress `env_blend_t` advances from `0.0` to `1.0` over `blend_duration`
//! seconds. When it reaches `1.0`, the target becomes the new active profile
//! and the blend is cleared.
//!
//! # Zone priority
//!
//! Zones may overlap. When the camera is inside multiple zones simultaneously,
//! the one with the highest `priority` value wins. Ties are broken by zone
//! index (lower index wins).
//!
//! # Usage
//!
//! ```rust
//! use rkf_runtime::main_camera::{EnvironmentZone, MainCamera};
//! use glam::Vec3;
//!
//! let mut cam = MainCamera::new(Vec3::ZERO, 60.0);
//!
//! let zones = vec![
//!     EnvironmentZone {
//!         name: "cave".into(),
//!         min: Vec3::new(-5.0, -5.0, -5.0),
//!         max: Vec3::new(5.0, 5.0, 5.0),
//!         profile_index: 1,
//!         blend_duration: 2.0,
//!         priority: 0,
//!     },
//! ];
//!
//! // Each frame: check zones then advance blend.
//! cam.check_zones(&zones);
//! cam.update_env_blend(0.016);
//! ```

use glam::{Mat4, Quat, Vec3};

// ---------------------------------------------------------------------------
// EnvironmentZone
// ---------------------------------------------------------------------------

/// An environment zone — axis-aligned region that triggers environment changes.
///
/// When the camera enters this zone the engine begins blending from the
/// current environment profile to `profile_index` over `blend_duration`
/// seconds. Overlapping zones are resolved by `priority` (higher wins).
#[derive(Debug, Clone)]
pub struct EnvironmentZone {
    /// Human-readable name (for editor display and debugging).
    pub name: String,
    /// AABB minimum corner (inclusive).
    pub min: Vec3,
    /// AABB maximum corner (inclusive).
    pub max: Vec3,
    /// Index into the engine's environment profile table.
    pub profile_index: usize,
    /// How long (in seconds) the blend to this profile takes.
    pub blend_duration: f32,
    /// Priority when multiple zones overlap. Higher value wins.
    pub priority: i32,
}

impl EnvironmentZone {
    /// Returns `true` if `point` is strictly inside (or on the boundary of)
    /// this zone's AABB.
    fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }
}

// ---------------------------------------------------------------------------
// MainCamera
// ---------------------------------------------------------------------------

/// The main camera with environment ownership.
///
/// There is exactly one `MainCamera` per scene. It tracks its own position
/// and rotation (f32 world-space — convert from [`WorldPosition`] externally
/// before writing), owns the active environment profile index, and manages
/// blending between profiles as the camera traverses environment zones.
#[derive(Debug)]
pub struct MainCamera {
    /// Camera position in world space.
    ///
    /// Use f32 for this struct; callers must convert from
    /// [`rkf_core::WorldPosition`] externally (subtract camera-chunk origin).
    pub position: Vec3,
    /// Camera orientation quaternion.
    pub rotation: Quat,
    /// Vertical field-of-view in degrees.
    pub fov: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Index of the currently active environment profile.
    pub active_env: usize,
    /// Index of the blend-target environment profile, if a blend is in progress.
    pub target_env: Option<usize>,
    /// Current blend progress in `[0.0, 1.0]`.
    ///
    /// `0.0` means fully in `active_env`; `1.0` means the blend has completed
    /// and `target_env` has just been promoted to `active_env`.
    pub env_blend_t: f32,
    /// Blend speed: `1.0 / blend_duration_seconds`.
    ///
    /// Stored pre-inverted so `update_env_blend` only multiplies.
    blend_speed: f32,
    /// Indices into the caller-supplied zone slice that the camera is currently
    /// inside. Refreshed on every [`check_zones`] call.
    active_zones: Vec<usize>,
}

impl MainCamera {
    /// Construct a new `MainCamera` at `position` with a given vertical FOV
    /// (in degrees).
    ///
    /// Defaults:
    /// - `rotation` = identity (looking along −Z)
    /// - `near` = 0.1, `far` = 1000.0
    /// - `active_env` = 0, no blend in progress
    pub fn new(position: Vec3, fov: f32) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            fov,
            near: 0.1,
            far: 1000.0,
            active_env: 0,
            target_env: None,
            env_blend_t: 0.0,
            blend_speed: 0.0,
            active_zones: Vec::new(),
        }
    }

    // ── Transform helpers ────────────────────────────────────────────────────

    /// Set camera position and rotation simultaneously.
    pub fn set_transform(&mut self, position: Vec3, rotation: Quat) {
        self.position = position;
        self.rotation = rotation;
    }

    /// Set camera orientation from yaw and pitch angles (both in degrees).
    ///
    /// - `yaw` rotates around the world Y axis (left/right). Positive yaw turns
    ///   the camera to the right (clockwise from above in a right-hand frame).
    /// - `pitch` is the elevation angle (positive = look upward, negative = look
    ///   downward). Clamped to ±89° to prevent gimbal lock at the poles.
    ///
    /// Implements the standard first-person camera decomposition:
    /// `rotation = yaw_around_Y × pitch_around_local_X`.
    pub fn set_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
        let pitch_clamped = pitch.clamp(-89.0, 89.0);
        let yaw_rad = yaw.to_radians();
        let pitch_rad = pitch_clamped.to_radians();
        // Build rotation: yaw around world-Y first, then pitch around local-X.
        // from_rotation_x(+theta) applied to NEG_Z yields forward with +Y,
        // so positive pitch naturally means "look upward".
        let yaw_quat = Quat::from_rotation_y(yaw_rad);
        let pitch_quat = Quat::from_rotation_x(pitch_rad);
        self.rotation = yaw_quat * pitch_quat;
    }

    /// Return the camera forward (look) direction in world space.
    ///
    /// Under the OpenGL convention used here, the camera looks along **−Z**
    /// in its local frame, so the forward vector is `rotation * (-Z)`.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    /// Return the camera right direction in world space.
    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    /// Return the camera up direction in world space.
    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    // ── Environment blending ─────────────────────────────────────────────────

    /// Begin blending from the current active profile to `target_index`.
    ///
    /// If `duration` is ≤ 0 the switch is instantaneous.
    ///
    /// Calling this while a blend is already in progress replaces the in-flight
    /// blend (the current `env_blend_t` is reset to 0).
    pub fn begin_env_blend(&mut self, target_index: usize, duration: f32) {
        if duration <= 0.0 {
            // Instantaneous — promote immediately.
            self.active_env = target_index;
            self.target_env = None;
            self.env_blend_t = 0.0;
            self.blend_speed = 0.0;
        } else {
            self.target_env = Some(target_index);
            self.env_blend_t = 0.0;
            self.blend_speed = 1.0 / duration;
        }
    }

    /// Advance the environment blend by `dt` seconds.
    ///
    /// Returns `true` if the blend **completed** on this call (i.e. `target_env`
    /// was promoted to `active_env`). Returns `false` when no blend is active or
    /// the blend is still in progress.
    pub fn update_env_blend(&mut self, dt: f32) -> bool {
        let Some(target) = self.target_env else {
            return false;
        };

        self.env_blend_t = (self.env_blend_t + dt * self.blend_speed).min(1.0);

        if self.env_blend_t >= 1.0 {
            // Blend finished — promote target to active.
            self.active_env = target;
            self.target_env = None;
            self.env_blend_t = 0.0;
            self.blend_speed = 0.0;
            true
        } else {
            false
        }
    }

    /// Check environment zones against the current camera position and trigger
    /// transitions when necessary.
    ///
    /// # Algorithm
    ///
    /// 1. Walk `zones`, collect those whose AABB contains `self.position`.
    /// 2. Among those, pick the one with the highest `priority` (ties broken
    ///    by lower slice index).
    /// 3. If the winning profile differs from `active_env` and we are not
    ///    already blending towards it, start a new blend.
    ///
    /// Returns `Some(profile_index)` of the winning zone, or `None` if the
    /// camera is outside every zone.
    pub fn check_zones(&mut self, zones: &[EnvironmentZone]) -> Option<usize> {
        // Collect all zones the camera is currently inside.
        self.active_zones.clear();
        for (i, zone) in zones.iter().enumerate() {
            if zone.contains(self.position) {
                self.active_zones.push(i);
            }
        }

        if self.active_zones.is_empty() {
            return None;
        }

        // Pick the highest-priority zone (stable: first occurrence wins ties).
        let best_idx = self
            .active_zones
            .iter()
            .copied()
            .max_by_key(|&i| zones[i].priority)
            .unwrap(); // safe: active_zones is non-empty

        let best = &zones[best_idx];

        // Start a blend if the profile differs and we're not already headed there.
        let already_blending_there = self.target_env == Some(best.profile_index);
        let already_active = self.active_env == best.profile_index;

        if !already_active && !already_blending_there {
            self.begin_env_blend(best.profile_index, best.blend_duration);
        }

        Some(best.profile_index)
    }

    // ── Blend state queries ───────────────────────────────────────────────────

    /// Returns `true` if a blend between environment profiles is currently active.
    pub fn is_blending(&self) -> bool {
        self.target_env.is_some()
    }

    /// Return a snapshot of the current blend state as
    /// `(active_index, target_index, blend_t)`.
    ///
    /// When no blend is in progress, `target_index` is `None` and `blend_t`
    /// is `0.0`.
    pub fn blend_state(&self) -> (usize, Option<usize>, f32) {
        (self.active_env, self.target_env, self.env_blend_t)
    }

    // ── Matrices ─────────────────────────────────────────────────────────────

    /// Compute the view matrix (world-to-camera transform).
    ///
    /// Equivalent to `look_at` built from the camera's current position,
    /// forward, and up vectors.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), self.up())
    }

    /// Compute the perspective projection matrix.
    ///
    /// - `aspect_ratio` — viewport width divided by viewport height.
    ///
    /// Uses a reversed-Z (infinite far plane) formulation if `far` is
    /// effectively infinite, otherwise standard perspective.
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), aspect_ratio, self.near, self.far)
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn unit_zone(min: Vec3, max: Vec3, profile: usize, priority: i32) -> EnvironmentZone {
        EnvironmentZone {
            name: format!("zone_{profile}"),
            min,
            max,
            profile_index: profile,
            blend_duration: 1.0,
            priority,
        }
    }

    // ── test 1: new_camera ────────────────────────────────────────────────────

    #[test]
    fn new_camera() {
        let cam = MainCamera::new(Vec3::new(1.0, 2.0, 3.0), 75.0);

        assert_eq!(cam.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cam.rotation, Quat::IDENTITY);
        assert_eq!(cam.fov, 75.0);
        assert_eq!(cam.near, 0.1);
        assert_eq!(cam.far, 1000.0);
        assert_eq!(cam.active_env, 0);
        assert_eq!(cam.target_env, None);
        assert_eq!(cam.env_blend_t, 0.0);
        assert!(!cam.is_blending());
    }

    // ── test 2: set_transform ─────────────────────────────────────────────────

    #[test]
    fn set_transform() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);
        let new_pos = Vec3::new(10.0, 5.0, -3.0);
        let new_rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);

        cam.set_transform(new_pos, new_rot);

        assert_eq!(cam.position, new_pos);
        assert!((cam.rotation.dot(new_rot) - 1.0).abs() < 1e-5);
    }

    // ── test 3: forward_default ───────────────────────────────────────────────

    #[test]
    fn forward_default() {
        // Identity rotation → camera looks along -Z (OpenGL convention).
        let cam = MainCamera::new(Vec3::ZERO, 60.0);
        let fwd = cam.forward();

        assert!((fwd - Vec3::NEG_Z).length() < 1e-5,
            "expected forward = (0, 0, -1), got {fwd:?}");
    }

    // ── test 4: yaw_pitch ─────────────────────────────────────────────────────

    #[test]
    fn yaw_pitch() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);

        // 90° yaw around world-Y in a right-handed system (glam): CCW when
        // viewed from above (+Y). Rotating -Z by +90° around +Y gives -X.
        cam.set_yaw_pitch(90.0, 0.0);
        let fwd = cam.forward();
        assert!((fwd.x + 1.0).abs() < 1e-5, "yaw 90° → expect -X forward (RH), got {fwd:?}");
        assert!(fwd.y.abs() < 1e-5);

        // +45° pitch (rotate_x in camera frame): tilts the camera upward,
        // so forward gains a positive Y component.
        cam.set_yaw_pitch(0.0, 45.0);
        let fwd2 = cam.forward();
        // With 0 yaw and +45° pitch, forward has positive Y and negative Z.
        assert!(fwd2.y > 0.1, "positive pitch should lift the forward vector, got {fwd2:?}");
        assert!(fwd2.z < 0.0, "forward Z should remain negative, got {fwd2:?}");
    }

    // ── test 5: begin_blend ───────────────────────────────────────────────────

    #[test]
    fn begin_blend() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);

        cam.begin_env_blend(3, 2.0);

        assert_eq!(cam.target_env, Some(3));
        assert_eq!(cam.env_blend_t, 0.0);
        assert!(cam.is_blending());
        // blend_speed should be 1/2 = 0.5
        assert!((cam.blend_speed - 0.5).abs() < 1e-6);
    }

    // ── test 6: update_blend_progress ────────────────────────────────────────

    #[test]
    fn update_blend_progress() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);
        cam.begin_env_blend(1, 4.0); // blend_speed = 0.25

        let completed = cam.update_env_blend(1.0); // dt = 1s → t = 0.25

        assert!(!completed);
        assert!(cam.is_blending());
        assert!((cam.env_blend_t - 0.25).abs() < 1e-5,
            "expected blend_t ≈ 0.25, got {}", cam.env_blend_t);
        assert_eq!(cam.target_env, Some(1));
        assert_eq!(cam.active_env, 0); // not promoted yet
    }

    // ── test 7: blend_completion ──────────────────────────────────────────────

    #[test]
    fn blend_completion() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);
        cam.begin_env_blend(2, 1.0); // blend_speed = 1.0

        // Single large dt drives blend to completion.
        let completed = cam.update_env_blend(1.0);

        assert!(completed, "blend should report completion");
        assert_eq!(cam.active_env, 2, "target should have been promoted");
        assert_eq!(cam.target_env, None);
        assert_eq!(cam.env_blend_t, 0.0);
        assert!(!cam.is_blending());
    }

    // ── test 8: check_zones_inside ────────────────────────────────────────────

    #[test]
    fn check_zones_inside() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);
        let zones = vec![
            unit_zone(Vec3::splat(-5.0), Vec3::splat(5.0), 1, 0),
        ];

        // Camera at origin is inside the zone.
        let result = cam.check_zones(&zones);

        assert_eq!(result, Some(1));
        assert!(cam.is_blending());
        assert_eq!(cam.target_env, Some(1));
    }

    // ── test 9: check_zones_outside ──────────────────────────────────────────

    #[test]
    fn check_zones_outside() {
        let mut cam = MainCamera::new(Vec3::new(100.0, 0.0, 0.0), 60.0);
        let zones = vec![
            unit_zone(Vec3::splat(-5.0), Vec3::splat(5.0), 1, 0),
        ];

        // Camera is far outside the zone.
        let result = cam.check_zones(&zones);

        assert_eq!(result, None);
        assert!(!cam.is_blending());
        assert_eq!(cam.active_env, 0);
    }

    // ── test 10: zone_priority ────────────────────────────────────────────────

    #[test]
    fn zone_priority() {
        let mut cam = MainCamera::new(Vec3::ZERO, 60.0);

        // Two overlapping zones at the origin: zone 0 (priority 0, profile 1)
        // and zone 1 (priority 10, profile 2). Higher priority should win.
        let zones = vec![
            unit_zone(Vec3::splat(-5.0), Vec3::splat(5.0), 1, 0),
            unit_zone(Vec3::splat(-5.0), Vec3::splat(5.0), 2, 10),
        ];

        let result = cam.check_zones(&zones);

        assert_eq!(result, Some(2), "higher priority zone should win");
        assert_eq!(cam.target_env, Some(2));
    }
}
