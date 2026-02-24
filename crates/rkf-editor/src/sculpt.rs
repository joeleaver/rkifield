//! Sculpt mode data model for the RKIField editor.
//!
//! Provides brush types, shapes, settings, stroke recording, and sculpt state
//! management. This is a pure data model that can be tested independently of
//! the GPU sculpting pipeline.

#![allow(dead_code)]

use glam::Vec3;

/// The type of sculpting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushType {
    /// Add material (inflate / build up).
    Add,
    /// Remove material (carve / dig).
    Subtract,
    /// Smooth SDF values (blur distance field).
    Smooth,
    /// Flatten to a reference plane.
    Flatten,
    /// Sharpen edges (inverse of smooth).
    Sharpen,
}

/// The shape of the sculpt brush.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrushShape {
    /// Spherical brush.
    Sphere,
    /// Cubic brush (axis-aligned).
    Cube,
    /// Cylindrical brush (Y-axis aligned).
    Cylinder,
}

/// Settings for a sculpt brush.
#[derive(Debug, Clone)]
pub struct BrushSettings {
    /// The sculpting operation type.
    pub brush_type: BrushType,
    /// The shape of the brush volume.
    pub shape: BrushShape,
    /// Radius of the brush in world units.
    pub radius: f32,
    /// Strength of the effect, 0.0 to 1.0.
    pub strength: f32,
    /// Material ID to apply when adding material.
    pub material_id: u16,
    /// Falloff curve parameter, 0.0 (hard cutoff) to 1.0 (smooth).
    pub falloff: f32,
}

impl Default for BrushSettings {
    fn default() -> Self {
        Self {
            brush_type: BrushType::Add,
            shape: BrushShape::Sphere,
            radius: 1.0,
            strength: 0.5,
            material_id: 0,
            falloff: 0.5,
        }
    }
}

/// A recorded sculpt stroke (sequence of brush applications along a path).
#[derive(Debug, Clone)]
pub struct SculptStroke {
    /// World-space points along the stroke path.
    pub points: Vec<Vec3>,
    /// Brush settings used for this stroke.
    pub settings: BrushSettings,
    /// Whether the stroke is currently in progress.
    started: bool,
}

impl SculptStroke {
    /// Create a new stroke with the given settings.
    pub fn new(settings: BrushSettings) -> Self {
        Self {
            points: Vec::new(),
            settings,
            started: true,
        }
    }

    /// Add a point to the stroke path.
    pub fn add_point(&mut self, pos: Vec3) {
        self.points.push(pos);
    }

    /// Return the number of recorded points.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Whether the stroke is currently in progress.
    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Finish the stroke, marking it as no longer active.
    pub fn finish(&mut self) {
        self.started = false;
    }
}

/// Manages sculpt mode state including the current brush, active stroke, and history.
#[derive(Debug, Clone)]
pub struct SculptState {
    /// Current brush settings (applied to new strokes).
    pub current_settings: BrushSettings,
    /// The currently active stroke, if any.
    pub active_stroke: Option<SculptStroke>,
    /// History of completed strokes (for undo).
    pub stroke_history: Vec<SculptStroke>,
}

impl Default for SculptState {
    fn default() -> Self {
        Self {
            current_settings: BrushSettings::default(),
            active_stroke: None,
            stroke_history: Vec::new(),
        }
    }
}

impl SculptState {
    /// Create a new sculpt state with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new stroke at the given position.
    ///
    /// If a stroke is already active, it is finished and added to history first.
    pub fn begin_stroke(&mut self, pos: Vec3) {
        // Finish any active stroke before starting a new one.
        if self.active_stroke.is_some() {
            self.end_stroke();
        }

        let mut stroke = SculptStroke::new(self.current_settings.clone());
        stroke.add_point(pos);
        self.active_stroke = Some(stroke);
    }

    /// Continue the active stroke by adding a new point.
    ///
    /// Does nothing if no stroke is active.
    pub fn continue_stroke(&mut self, pos: Vec3) {
        if let Some(ref mut stroke) = self.active_stroke {
            stroke.add_point(pos);
        }
    }

    /// End the active stroke and move it to the history.
    ///
    /// Does nothing if no stroke is active.
    pub fn end_stroke(&mut self) {
        if let Some(mut stroke) = self.active_stroke.take() {
            stroke.finish();
            self.stroke_history.push(stroke);
        }
    }

    /// Remove and return the last completed stroke from history (undo).
    pub fn undo_last_stroke(&mut self) -> Option<SculptStroke> {
        self.stroke_history.pop()
    }

    /// Set the brush type.
    pub fn set_brush_type(&mut self, brush_type: BrushType) {
        self.current_settings.brush_type = brush_type;
    }

    /// Set the brush radius.
    pub fn set_radius(&mut self, radius: f32) {
        self.current_settings.radius = radius.max(0.01);
    }

    /// Set the brush strength (clamped to 0.0-1.0).
    pub fn set_strength(&mut self, strength: f32) {
        self.current_settings.strength = strength.clamp(0.0, 1.0);
    }

    /// Set the brush material ID.
    pub fn set_material(&mut self, material_id: u16) {
        self.current_settings.material_id = material_id;
    }
}

/// Transform a world-space position into a v2 object's local coordinate space.
///
/// Converts `world_pos` from world space into the object's local space by
/// applying the inverse of the object's world transform. The object transform
/// is `T(world_position.local) * R(rotation) * S(scale)`.
///
/// This is used by the sculpt pipeline to convert brush hit positions (which
/// are in world space) into the per-object SDF space before evaluating or
/// writing voxels.
pub fn world_to_object_local_v2(
    world_pos: glam::Vec3,
    object: &rkf_core::scene::SceneObject,
) -> glam::Vec3 {
    use glam::Mat4;

    // Build the object's world-to-local matrix.
    let world_matrix = Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(object.scale),
        object.rotation,
        object.world_position.local,
    );
    let inv = world_matrix.inverse();
    inv.transform_point3(world_pos)
}

/// Compute brush falloff at a given distance from the brush center.
///
/// Returns a value from 1.0 at the center to 0.0 at the radius edge.
/// When `falloff` > 0, uses smoothstep for a gradual transition.
/// When `falloff` == 0, uses a hard cutoff (1.0 inside, 0.0 outside).
///
/// The `falloff` parameter controls the width of the transition zone:
/// - `falloff = 0.0`: hard edge (binary 1/0)
/// - `falloff = 1.0`: smooth transition across the full radius
pub fn brush_falloff(distance: f32, radius: f32, falloff: f32) -> f32 {
    if radius <= 0.0 || distance < 0.0 {
        return 0.0;
    }

    if distance >= radius {
        return 0.0;
    }

    if falloff <= 0.0 {
        // Hard cutoff: 1.0 inside the radius, 0.0 at or beyond.
        return 1.0;
    }

    // Normalized distance [0, 1].
    let t = distance / radius;

    // The falloff zone starts at (1 - falloff) of the radius.
    let edge_start = 1.0 - falloff;

    if t <= edge_start {
        1.0
    } else {
        // Map t from [edge_start, 1.0] to [0, 1] and apply smoothstep.
        let s = (t - edge_start) / falloff;
        // Smoothstep: 1 - (3s^2 - 2s^3)
        1.0 - s * s * (3.0 - 2.0 * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // --- BrushSettings tests ---

    #[test]
    fn test_brush_settings_default() {
        let settings = BrushSettings::default();
        assert_eq!(settings.brush_type, BrushType::Add);
        assert_eq!(settings.shape, BrushShape::Sphere);
        assert!(approx_eq(settings.radius, 1.0));
        assert!(approx_eq(settings.strength, 0.5));
        assert_eq!(settings.material_id, 0);
        assert!(approx_eq(settings.falloff, 0.5));
    }

    // --- SculptStroke tests ---

    #[test]
    fn test_stroke_new() {
        let stroke = SculptStroke::new(BrushSettings::default());
        assert!(stroke.is_started());
        assert_eq!(stroke.point_count(), 0);
    }

    #[test]
    fn test_stroke_add_points() {
        let mut stroke = SculptStroke::new(BrushSettings::default());
        stroke.add_point(Vec3::ZERO);
        stroke.add_point(Vec3::ONE);
        stroke.add_point(Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(stroke.point_count(), 3);
    }

    #[test]
    fn test_stroke_finish() {
        let mut stroke = SculptStroke::new(BrushSettings::default());
        assert!(stroke.is_started());
        stroke.finish();
        assert!(!stroke.is_started());
    }

    // --- SculptState tests ---

    #[test]
    fn test_sculpt_state_new() {
        let state = SculptState::new();
        assert!(state.active_stroke.is_none());
        assert!(state.stroke_history.is_empty());
    }

    #[test]
    fn test_begin_stroke() {
        let mut state = SculptState::new();
        state.begin_stroke(Vec3::new(1.0, 2.0, 3.0));

        assert!(state.active_stroke.is_some());
        let stroke = state.active_stroke.as_ref().unwrap();
        assert!(stroke.is_started());
        assert_eq!(stroke.point_count(), 1);
        assert_eq!(stroke.points[0], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_continue_stroke() {
        let mut state = SculptState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);
        state.continue_stroke(Vec3::new(2.0, 0.0, 0.0));

        let stroke = state.active_stroke.as_ref().unwrap();
        assert_eq!(stroke.point_count(), 3);
    }

    #[test]
    fn test_continue_stroke_no_active() {
        let mut state = SculptState::new();
        state.continue_stroke(Vec3::ONE); // should do nothing
        assert!(state.active_stroke.is_none());
    }

    #[test]
    fn test_end_stroke() {
        let mut state = SculptState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);
        state.end_stroke();

        assert!(state.active_stroke.is_none());
        assert_eq!(state.stroke_history.len(), 1);
        assert!(!state.stroke_history[0].is_started());
        assert_eq!(state.stroke_history[0].point_count(), 2);
    }

    #[test]
    fn test_end_stroke_no_active() {
        let mut state = SculptState::new();
        state.end_stroke(); // should do nothing
        assert!(state.stroke_history.is_empty());
    }

    #[test]
    fn test_begin_stroke_finishes_previous() {
        let mut state = SculptState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);

        // Starting a new stroke should finish the old one.
        state.begin_stroke(Vec3::new(5.0, 0.0, 0.0));

        // Old stroke in history
        assert_eq!(state.stroke_history.len(), 1);
        assert_eq!(state.stroke_history[0].point_count(), 2);
        assert!(!state.stroke_history[0].is_started());

        // New stroke active
        assert!(state.active_stroke.is_some());
        assert_eq!(state.active_stroke.as_ref().unwrap().point_count(), 1);
    }

    #[test]
    fn test_undo_last_stroke() {
        let mut state = SculptState::new();
        state.begin_stroke(Vec3::ZERO);
        state.end_stroke();
        state.begin_stroke(Vec3::ONE);
        state.end_stroke();

        assert_eq!(state.stroke_history.len(), 2);

        let undone = state.undo_last_stroke();
        assert!(undone.is_some());
        assert_eq!(state.stroke_history.len(), 1);

        let undone2 = state.undo_last_stroke();
        assert!(undone2.is_some());
        assert!(state.stroke_history.is_empty());

        let undone3 = state.undo_last_stroke();
        assert!(undone3.is_none());
    }

    #[test]
    fn test_set_brush_type() {
        let mut state = SculptState::new();
        state.set_brush_type(BrushType::Subtract);
        assert_eq!(state.current_settings.brush_type, BrushType::Subtract);
    }

    #[test]
    fn test_set_radius() {
        let mut state = SculptState::new();
        state.set_radius(3.5);
        assert!(approx_eq(state.current_settings.radius, 3.5));
    }

    #[test]
    fn test_set_radius_clamps_min() {
        let mut state = SculptState::new();
        state.set_radius(-1.0);
        assert!(state.current_settings.radius >= 0.01);
    }

    #[test]
    fn test_set_strength() {
        let mut state = SculptState::new();
        state.set_strength(0.8);
        assert!(approx_eq(state.current_settings.strength, 0.8));
    }

    #[test]
    fn test_set_strength_clamped() {
        let mut state = SculptState::new();
        state.set_strength(2.0);
        assert!(approx_eq(state.current_settings.strength, 1.0));
        state.set_strength(-0.5);
        assert!(approx_eq(state.current_settings.strength, 0.0));
    }

    #[test]
    fn test_set_material() {
        let mut state = SculptState::new();
        state.set_material(42);
        assert_eq!(state.current_settings.material_id, 42);
    }

    // --- brush_falloff tests ---

    #[test]
    fn test_falloff_at_center() {
        let f = brush_falloff(0.0, 1.0, 0.5);
        assert!(approx_eq(f, 1.0), "center should be 1.0: {f}");
    }

    #[test]
    fn test_falloff_at_edge() {
        let f = brush_falloff(1.0, 1.0, 0.5);
        assert!(approx_eq(f, 0.0), "at radius should be 0.0: {f}");
    }

    #[test]
    fn test_falloff_beyond_radius() {
        let f = brush_falloff(1.5, 1.0, 0.5);
        assert!(approx_eq(f, 0.0), "beyond radius should be 0.0: {f}");
    }

    #[test]
    fn test_falloff_hard_cutoff() {
        // falloff = 0 means hard edge
        assert!(approx_eq(brush_falloff(0.0, 1.0, 0.0), 1.0));
        assert!(approx_eq(brush_falloff(0.5, 1.0, 0.0), 1.0));
        assert!(approx_eq(brush_falloff(0.99, 1.0, 0.0), 1.0));
        assert!(approx_eq(brush_falloff(1.0, 1.0, 0.0), 0.0));
    }

    #[test]
    fn test_falloff_smooth_midpoint() {
        // With full falloff, the midpoint (0.5 of radius) should be ~0.5
        let f = brush_falloff(0.5, 1.0, 1.0);
        assert!(f > 0.3 && f < 0.7,
            "midpoint with full falloff should be near 0.5: {f}");
    }

    #[test]
    fn test_falloff_monotonic() {
        // Falloff should decrease monotonically with distance
        let r = 2.0;
        let fo = 0.8;
        let mut prev = brush_falloff(0.0, r, fo);
        for i in 1..20 {
            let d = (i as f32) * r / 20.0;
            let curr = brush_falloff(d, r, fo);
            assert!(curr <= prev + EPS,
                "falloff should be monotonically decreasing: d={d} curr={curr} prev={prev}");
            prev = curr;
        }
    }

    #[test]
    fn test_falloff_zero_radius() {
        let f = brush_falloff(0.0, 0.0, 0.5);
        assert!(approx_eq(f, 0.0), "zero radius should return 0: {f}");
    }

    #[test]
    fn test_brush_type_eq() {
        assert_eq!(BrushType::Add, BrushType::Add);
        assert_ne!(BrushType::Add, BrushType::Subtract);
        assert_ne!(BrushType::Smooth, BrushType::Flatten);
    }

    #[test]
    fn test_brush_shape_eq() {
        assert_eq!(BrushShape::Sphere, BrushShape::Sphere);
        assert_ne!(BrushShape::Sphere, BrushShape::Cube);
        assert_ne!(BrushShape::Cube, BrushShape::Cylinder);
    }

    // --- world_to_object_local_v2 tests ---

    fn make_test_object(local_pos: Vec3, scale: f32) -> rkf_core::scene::SceneObject {
        use rkf_core::{aabb::Aabb, scene::SceneObject, scene_node::SceneNode, WorldPosition};
        use glam::{IVec3, Quat};
        SceneObject {
            id: 1,
            name: "test".into(),
            world_position: WorldPosition::new(IVec3::ZERO, local_pos),
            rotation: Quat::IDENTITY,
            scale,
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        }
    }

    #[test]
    fn test_world_to_object_local_identity() {
        // Object at origin, scale 1 — world pos should equal local pos
        let obj = make_test_object(Vec3::ZERO, 1.0);
        let world = Vec3::new(1.0, 2.0, 3.0);
        let local = super::world_to_object_local_v2(world, &obj);
        assert!((local - world).length() < EPS, "identity: {local:?}");
    }

    #[test]
    fn test_world_to_object_local_translated() {
        // Object at (5, 0, 0), scale 1 — local(0,0,0) should give world(5,0,0)
        let obj = make_test_object(Vec3::new(5.0, 0.0, 0.0), 1.0);
        // World point at (5,0,0) should be at local (0,0,0)
        let local = super::world_to_object_local_v2(Vec3::new(5.0, 0.0, 0.0), &obj);
        assert!(local.length() < EPS, "translated: {local:?}");
    }

    #[test]
    fn test_world_to_object_local_scaled() {
        // Object at origin, scale 2 — world (2,0,0) should be local (1,0,0)
        let obj = make_test_object(Vec3::ZERO, 2.0);
        let local = super::world_to_object_local_v2(Vec3::new(2.0, 0.0, 0.0), &obj);
        assert!(approx_eq(local.x, 1.0), "scaled x: {local:?}");
        assert!(local.y.abs() < EPS);
        assert!(local.z.abs() < EPS);
    }
}
