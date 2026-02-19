//! Transform gizmo math for the RKIField editor.
//!
//! Provides the geometric calculations for translate/rotate/scale gizmos:
//! ray-axis closest point, ray-plane intersection, axis picking, and
//! constrained delta computation. This is pure math — no rendering or input handling.

#![allow(dead_code)]

use glam::{Quat, Vec3};
use log::warn;

/// Which transform operation the gizmo performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which axis or plane the gizmo is constrained to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    XY,
    XZ,
    YZ,
    /// Screen-space (view-aligned) operation.
    View,
    None,
}

impl GizmoAxis {
    /// Return the unit direction vector for single-axis constraints.
    /// Returns `Vec3::ZERO` for plane/view/none axes.
    pub fn direction(&self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            _ => Vec3::ZERO,
        }
    }

    /// Return the plane normal for plane constraints.
    /// For single axes, returns the axis direction (used as rotation plane normal).
    /// Returns `Vec3::ZERO` for View/None.
    pub fn plane_normal(&self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
            GizmoAxis::XY => Vec3::Z,
            GizmoAxis::XZ => Vec3::Y,
            GizmoAxis::YZ => Vec3::X,
            _ => Vec3::ZERO,
        }
    }
}

/// State of an in-progress gizmo drag operation.
#[derive(Debug, Clone)]
pub struct GizmoState {
    /// Current gizmo mode.
    pub mode: GizmoMode,
    /// Which axis/plane is active.
    pub active_axis: GizmoAxis,
    /// Whether a drag is currently in progress.
    pub dragging: bool,
    /// World-space point where the drag started.
    pub drag_start: Vec3,
    /// Current world-space drag point.
    pub drag_current: Vec3,
    /// Position of the object when the drag began.
    pub initial_position: Vec3,
    /// Rotation of the object when the drag began.
    pub initial_rotation: Quat,
    /// Scale of the object when the drag began.
    pub initial_scale: f32,
}

impl Default for GizmoState {
    fn default() -> Self {
        Self::new()
    }
}

impl GizmoState {
    /// Create a new idle gizmo state.
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            active_axis: GizmoAxis::None,
            dragging: false,
            drag_start: Vec3::ZERO,
            drag_current: Vec3::ZERO,
            initial_position: Vec3::ZERO,
            initial_rotation: Quat::IDENTITY,
            initial_scale: 1.0,
        }
    }

    /// Begin a drag operation.
    pub fn begin_drag(
        &mut self,
        axis: GizmoAxis,
        start_point: Vec3,
        position: Vec3,
        rotation: Quat,
        scale: f32,
    ) {
        self.active_axis = axis;
        self.dragging = true;
        self.drag_start = start_point;
        self.drag_current = start_point;
        self.initial_position = position;
        self.initial_rotation = rotation;
        self.initial_scale = scale;
    }

    /// End the current drag operation.
    pub fn end_drag(&mut self) {
        self.dragging = false;
        self.active_axis = GizmoAxis::None;
    }
}

/// Find the parameter `t` along an axis line that is closest to a ray.
///
/// Given a ray (origin + direction) and an axis line (origin + direction),
/// returns the parameter `t` such that `axis_origin + axis_dir * t` is the
/// closest point on the axis to the ray.
///
/// Returns 0.0 if the ray and axis are parallel (degenerate case).
pub fn ray_axis_closest_point(
    ray_origin: Vec3,
    ray_dir: Vec3,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> f32 {
    // Closest approach between two lines:
    // Line 1: P = ray_origin + ray_dir * s
    // Line 2: Q = axis_origin + axis_dir * t
    // We want to find t that minimizes |P - Q|.
    //
    // Using the standard formula for closest points on two lines:
    let w0 = ray_origin - axis_origin;
    let a = ray_dir.dot(ray_dir); // always >= 0
    let b = ray_dir.dot(axis_dir);
    let c = axis_dir.dot(axis_dir); // always >= 0
    let d = ray_dir.dot(w0);
    let e = axis_dir.dot(w0);

    let denom = a * c - b * b;

    // If denom is ~0, the lines are parallel.
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    (a * e - b * d) / denom
}

/// Intersect a ray with a plane.
///
/// Returns the intersection point, or `None` if the ray is parallel to the plane
/// or the intersection is behind the ray origin.
pub fn project_to_plane(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_origin: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = ray_dir.dot(plane_normal);

    // Ray is parallel to or nearly parallel to the plane.
    if denom.abs() < 1e-7 {
        return None;
    }

    let t = (plane_origin - ray_origin).dot(plane_normal) / denom;

    // Intersection is behind the ray origin.
    if t < 0.0 {
        return None;
    }

    Some(ray_origin + ray_dir * t)
}

/// Hit-test the gizmo axes and return the closest axis within a threshold.
///
/// Tests the three primary axes (X, Y, Z) emanating from `gizmo_center`,
/// each with length `gizmo_size`. Returns the axis whose line is closest
/// to the pick ray, provided the distance is within a screen-space-like threshold.
pub fn pick_gizmo_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_center: Vec3,
    gizmo_size: f32,
) -> GizmoAxis {
    let threshold = gizmo_size * 0.15; // 15% of gizmo size as pick tolerance
    let ray_dir = ray_dir.normalize();

    let axes = [
        (GizmoAxis::X, Vec3::X),
        (GizmoAxis::Y, Vec3::Y),
        (GizmoAxis::Z, Vec3::Z),
    ];

    let mut best_axis = GizmoAxis::None;
    let mut best_dist = f32::MAX;

    for (axis_id, axis_dir) in &axes {
        let t = ray_axis_closest_point(ray_origin, ray_dir, gizmo_center, *axis_dir);

        // Clamp t to the visible portion of the axis [0, gizmo_size].
        let t_clamped = t.clamp(0.0, gizmo_size);
        let axis_point = gizmo_center + *axis_dir * t_clamped;

        // Find the closest point on the ray to this axis point.
        let ray_t = (axis_point - ray_origin).dot(ray_dir);
        if ray_t < 0.0 {
            continue; // Behind camera
        }
        let ray_point = ray_origin + ray_dir * ray_t;
        let dist = (ray_point - axis_point).length();

        if dist < threshold && dist < best_dist {
            best_dist = dist;
            best_axis = *axis_id;
        }
    }

    best_axis
}

/// Compute the translation delta constrained to the active axis.
///
/// Projects the current ray onto the active axis or plane and returns the
/// world-space translation offset from the drag start.
pub fn compute_translate_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> Vec3 {
    let ray_dir = ray_dir.normalize();

    match state.active_axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
            let axis_dir = state.active_axis.direction();
            let t_current =
                ray_axis_closest_point(ray_origin, ray_dir, state.initial_position, axis_dir);
            let t_start = ray_axis_closest_point(
                ray_origin,
                ray_dir,
                state.initial_position,
                axis_dir,
            );
            // We need to compute the delta relative to drag_start.
            // Project drag_start onto the axis to get reference t.
            let start_t = (state.drag_start - state.initial_position).dot(axis_dir);
            let _ = t_start; // The current ray projection gives us the new t.
            axis_dir * (t_current - start_t)
        }
        GizmoAxis::XY | GizmoAxis::XZ | GizmoAxis::YZ => {
            let normal = state.active_axis.plane_normal();
            let current = project_to_plane(ray_origin, ray_dir, state.initial_position, normal);
            match current {
                Some(point) => point - state.drag_start,
                None => Vec3::ZERO,
            }
        }
        GizmoAxis::View => {
            // View-plane translation: project onto a plane facing the camera.
            let view_normal = -ray_dir;
            let current =
                project_to_plane(ray_origin, ray_dir, state.initial_position, view_normal);
            match current {
                Some(point) => point - state.drag_start,
                None => Vec3::ZERO,
            }
        }
        GizmoAxis::None => Vec3::ZERO,
    }
}

/// Compute the rotation delta around the active axis.
///
/// Projects the ray onto the plane perpendicular to the active axis at the
/// gizmo center, then computes the angle between drag_start and current point.
pub fn compute_rotate_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_center: Vec3,
) -> Quat {
    let axis = match state.active_axis {
        GizmoAxis::X => Vec3::X,
        GizmoAxis::Y => Vec3::Y,
        GizmoAxis::Z => Vec3::Z,
        _ => return Quat::IDENTITY,
    };

    let ray_dir = ray_dir.normalize();

    // Project both drag_start and current ray onto the rotation plane.
    let Some(current_point) = project_to_plane(ray_origin, ray_dir, gizmo_center, axis) else {
        return Quat::IDENTITY;
    };

    // Vectors from center to start and current, projected onto the plane.
    let start_vec = (state.drag_start - gizmo_center).normalize();
    let current_vec = (current_point - gizmo_center).normalize();

    if start_vec.length_squared() < 1e-6 || current_vec.length_squared() < 1e-6 {
        return Quat::IDENTITY;
    }

    // Compute angle using atan2 for proper sign.
    let dot = start_vec.dot(current_vec).clamp(-1.0, 1.0);
    let cross = start_vec.cross(current_vec);
    let angle = cross.dot(axis).atan2(dot);

    Quat::from_axis_angle(axis, angle)
}

/// Compute the uniform scale factor from a drag operation.
///
/// Measures how much the ray has moved along the active axis relative to
/// the drag start, and converts that to a multiplicative scale factor.
///
/// Per engine rule 3 (uniform scale only), this always returns a uniform
/// factor regardless of axis. The axis merely determines the drag direction.
pub fn compute_scale_delta(
    state: &GizmoState,
    ray_origin: Vec3,
    ray_dir: Vec3,
) -> f32 {
    if state.active_axis == GizmoAxis::None {
        return 1.0;
    }

    // Warn if non-uniform scale is attempted (per rule 3).
    if matches!(
        state.active_axis,
        GizmoAxis::XY | GizmoAxis::XZ | GizmoAxis::YZ
    ) {
        warn!(
            "Non-uniform scale attempted via plane axis {:?}. \
             SDF engine requires uniform scale only (rule 3). Using uniform factor.",
            state.active_axis
        );
    }

    let ray_dir = ray_dir.normalize();

    // Use the primary axis direction for the drag, or camera direction for View.
    let drag_axis = match state.active_axis {
        GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => state.active_axis.direction(),
        GizmoAxis::View => -ray_dir,
        _ => Vec3::Y, // Fallback to Y for plane axes (uniform anyway)
    };

    let t_current =
        ray_axis_closest_point(ray_origin, ray_dir, state.initial_position, drag_axis);
    let start_offset = (state.drag_start - state.initial_position).dot(drag_axis);

    // Scale sensitivity: each gizmo_size unit of drag = 1x scale change.
    let delta = t_current - start_offset;
    let sensitivity = 0.01;

    // Convert linear delta to multiplicative factor, clamped to reasonable range.
    (1.0 + delta * sensitivity).clamp(0.01, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // --- ray_axis_closest_point tests ---

    #[test]
    fn test_ray_axis_perpendicular() {
        // Ray along +Z, axis along +X at origin
        let t = ray_axis_closest_point(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, Vec3::ZERO, Vec3::X);
        // Closest point on X axis to a ray along Z through origin is at t=0
        assert!(approx_eq(t, 0.0), "t = {t}");
    }

    #[test]
    fn test_ray_axis_offset() {
        // Ray along +Z passing through (3, 0, z), axis along +X at origin
        let t = ray_axis_closest_point(
            Vec3::new(3.0, 0.0, -5.0),
            Vec3::Z,
            Vec3::ZERO,
            Vec3::X,
        );
        // Closest point on X-axis is x=3
        assert!(approx_eq(t, 3.0), "t = {t}");
    }

    #[test]
    fn test_ray_axis_parallel() {
        // Ray parallel to axis — degenerate, should return 0
        let t = ray_axis_closest_point(Vec3::new(0.0, 1.0, 0.0), Vec3::X, Vec3::ZERO, Vec3::X);
        assert!(approx_eq(t, 0.0), "parallel case should return 0: t = {t}");
    }

    #[test]
    fn test_ray_axis_negative_t() {
        // Ray along +Z passing through (-2, 0, z), axis along +X at origin
        let t = ray_axis_closest_point(
            Vec3::new(-2.0, 0.0, -5.0),
            Vec3::Z,
            Vec3::ZERO,
            Vec3::X,
        );
        assert!(approx_eq(t, -2.0), "t = {t}");
    }

    // --- project_to_plane tests ---

    #[test]
    fn test_plane_intersection_basic() {
        // Ray from (0,5,0) downward onto XZ plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        assert!(vec3_approx_eq(hit.unwrap(), Vec3::ZERO), "hit = {:?}", hit);
    }

    #[test]
    fn test_plane_intersection_offset() {
        // Ray from (3, 10, 7) downward onto XZ plane at y=0
        let hit = project_to_plane(Vec3::new(3.0, 10.0, 7.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        let p = hit.unwrap();
        assert!(approx_eq(p.x, 3.0) && approx_eq(p.y, 0.0) && approx_eq(p.z, 7.0));
    }

    #[test]
    fn test_plane_parallel_returns_none() {
        // Ray parallel to the plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), Vec3::X, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_none());
    }

    #[test]
    fn test_plane_behind_ray_returns_none() {
        // Ray pointing away from the plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_none());
    }

    // --- pick_gizmo_axis tests ---

    #[test]
    fn test_pick_x_axis() {
        // Ray from front, aiming right along X axis handle
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray grazing the X axis at y=0, z offset close
        let axis = pick_gizmo_axis(
            Vec3::new(0.5, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::X);
    }

    #[test]
    fn test_pick_y_axis() {
        let center = Vec3::ZERO;
        let size = 1.0;
        let axis = pick_gizmo_axis(
            Vec3::new(0.0, 0.5, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::Y);
    }

    #[test]
    fn test_pick_z_axis() {
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray from the side, aimed to pass near the Z axis handle
        let axis = pick_gizmo_axis(
            Vec3::new(5.0, 0.0, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::Z);
    }

    #[test]
    fn test_pick_miss() {
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray far from any axis
        let axis = pick_gizmo_axis(
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::None);
    }

    // --- compute_translate_delta tests ---

    #[test]
    fn test_translate_delta_x_axis() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::X;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Ray from (3, 0, 5) looking along -Z, should project to x=3 on X axis
        let delta = compute_translate_delta(&state, Vec3::new(3.0, 0.0, 5.0), -Vec3::Z);
        // Delta should be purely along X
        assert!(approx_eq(delta.y, 0.0), "y should be 0: {:?}", delta);
        assert!(approx_eq(delta.z, 0.0), "z should be 0: {:?}", delta);
        assert!(approx_eq(delta.x, 3.0), "x should be ~3: {:?}", delta);
    }

    #[test]
    fn test_translate_delta_none_axis() {
        let state = GizmoState::new();
        let delta = compute_translate_delta(&state, Vec3::ZERO, Vec3::Z);
        assert!(vec3_approx_eq(delta, Vec3::ZERO));
    }

    #[test]
    fn test_translate_delta_plane_xz() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::XZ;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        // Drag started at (1, 0, 1) on the XZ plane
        state.drag_start = Vec3::new(1.0, 0.0, 1.0);

        // Ray from above hitting (3, 0, 2) on XZ plane
        let delta = compute_translate_delta(&state, Vec3::new(3.0, 5.0, 2.0), -Vec3::Y);
        // Delta should be (3-1, 0, 2-1) = (2, 0, 1)
        assert!(approx_eq(delta.x, 2.0), "x: {:?}", delta);
        assert!(approx_eq(delta.y, 0.0), "y: {:?}", delta);
        assert!(approx_eq(delta.z, 1.0), "z: {:?}", delta);
    }

    // --- compute_rotate_delta tests ---

    #[test]
    fn test_rotate_delta_y_axis_90_degrees() {
        let mut state = GizmoState::new();
        state.mode = GizmoMode::Rotate;
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        let center = Vec3::ZERO;
        // Start at +X on the XZ plane
        state.drag_start = Vec3::new(1.0, 0.0, 0.0);

        // Current ray hits +Z on the XZ plane (90 degrees CCW around Y)
        let rot = compute_rotate_delta(
            &state,
            Vec3::new(0.0, 5.0, 1.0),
            -Vec3::Y,
            center,
        );

        // Should be ~90 degrees around Y
        let (axis, angle) = rot.to_axis_angle();
        assert!(
            approx_eq(angle.abs(), PI / 2.0),
            "angle should be ~90 deg: {} (axis: {:?})",
            angle.to_degrees(),
            axis
        );
    }

    #[test]
    fn test_rotate_delta_no_axis() {
        let state = GizmoState::new();
        let rot = compute_rotate_delta(&state, Vec3::ZERO, Vec3::Z, Vec3::ZERO);
        // Should be identity when no axis selected
        assert!(approx_eq(rot.w, 1.0), "should be identity: {:?}", rot);
    }

    #[test]
    fn test_rotate_delta_zero_movement() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        state.drag_start = Vec3::new(1.0, 0.0, 0.0);

        // Ray hits the same point as drag_start
        let rot = compute_rotate_delta(
            &state,
            Vec3::new(1.0, 5.0, 0.0),
            -Vec3::Y,
            Vec3::ZERO,
        );

        let (_, angle) = rot.to_axis_angle();
        assert!(
            approx_eq(angle, 0.0),
            "no movement should give ~0 angle: {}",
            angle
        );
    }

    // --- compute_scale_delta tests ---

    #[test]
    fn test_scale_delta_no_movement() {
        let mut state = GizmoState::new();
        state.mode = GizmoMode::Scale;
        state.active_axis = GizmoAxis::X;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Ray at origin along X — t=0, start_offset=0, so delta=0 → scale=1.0
        let scale = compute_scale_delta(&state, Vec3::new(0.0, 0.0, 5.0), -Vec3::Z);
        // With no effective movement along X, scale should be ~1.0
        assert!(
            approx_eq(scale, 1.0),
            "no movement should give scale ~1.0: {scale}"
        );
    }

    #[test]
    fn test_scale_delta_none_axis() {
        let state = GizmoState::new();
        let scale = compute_scale_delta(&state, Vec3::ZERO, Vec3::Z);
        assert!(approx_eq(scale, 1.0));
    }

    #[test]
    fn test_scale_delta_clamped() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Extreme negative scale should be clamped to 0.01
        let scale = compute_scale_delta(&state, Vec3::new(0.0, -100000.0, 5.0), -Vec3::Z);
        assert!(scale >= 0.01, "scale should be clamped: {scale}");

        // Extreme positive scale should be clamped to 100.0
        let scale = compute_scale_delta(&state, Vec3::new(0.0, 100000.0, 5.0), -Vec3::Z);
        assert!(scale <= 100.0, "scale should be clamped: {scale}");
    }

    // --- GizmoState tests ---

    #[test]
    fn test_gizmo_state_default() {
        let state = GizmoState::new();
        assert_eq!(state.mode, GizmoMode::Translate);
        assert_eq!(state.active_axis, GizmoAxis::None);
        assert!(!state.dragging);
    }

    #[test]
    fn test_gizmo_begin_end_drag() {
        let mut state = GizmoState::new();
        state.begin_drag(
            GizmoAxis::X,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
        );

        assert!(state.dragging);
        assert_eq!(state.active_axis, GizmoAxis::X);
        assert_eq!(state.drag_start, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(state.initial_scale, 1.0);

        state.end_drag();
        assert!(!state.dragging);
        assert_eq!(state.active_axis, GizmoAxis::None);
    }

    // --- GizmoAxis tests ---

    #[test]
    fn test_axis_direction() {
        assert_eq!(GizmoAxis::X.direction(), Vec3::X);
        assert_eq!(GizmoAxis::Y.direction(), Vec3::Y);
        assert_eq!(GizmoAxis::Z.direction(), Vec3::Z);
        assert_eq!(GizmoAxis::XY.direction(), Vec3::ZERO);
        assert_eq!(GizmoAxis::None.direction(), Vec3::ZERO);
    }

    #[test]
    fn test_axis_plane_normal() {
        assert_eq!(GizmoAxis::XY.plane_normal(), Vec3::Z);
        assert_eq!(GizmoAxis::XZ.plane_normal(), Vec3::Y);
        assert_eq!(GizmoAxis::YZ.plane_normal(), Vec3::X);
    }

    // --- Degenerate cases ---

    #[test]
    fn test_ray_axis_zero_direction() {
        // Zero-length ray direction
        let t = ray_axis_closest_point(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::X);
        // Should handle gracefully (denom = 0)
        assert!(t.is_finite(), "should return finite value");
    }

    #[test]
    fn test_plane_intersection_at_origin() {
        let hit = project_to_plane(Vec3::new(0.0, 1.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        assert!(vec3_approx_eq(hit.unwrap(), Vec3::ZERO));
    }

    #[test]
    fn test_gizmo_mode_eq() {
        assert_eq!(GizmoMode::Translate, GizmoMode::Translate);
        assert_ne!(GizmoMode::Translate, GizmoMode::Rotate);
        assert_ne!(GizmoMode::Rotate, GizmoMode::Scale);
    }
}
