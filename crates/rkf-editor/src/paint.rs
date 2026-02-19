//! Paint mode data model for the RKIField editor.
//!
//! Provides paint modes (material, color, blend), settings, stroke recording,
//! and paint state management. This is a pure data model that can be tested
//! independently of the GPU painting pipeline.

#![allow(dead_code)]

use glam::Vec3;

/// The type of painting operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintMode {
    /// Paint material IDs onto voxels.
    Material,
    /// Paint per-voxel color onto voxels.
    Color,
    /// Paint blend weight between two materials.
    Blend,
}

/// Settings for a paint brush.
#[derive(Debug, Clone)]
pub struct PaintSettings {
    /// The painting operation mode.
    pub mode: PaintMode,
    /// Primary material ID to paint.
    pub material_id: u16,
    /// RGB color to paint (each channel 0.0-1.0).
    pub color: Vec3,
    /// Secondary material ID (used in Blend mode).
    pub secondary_material_id: u16,
    /// Blend gradient parameter for smooth transitions (0.0-1.0).
    pub blend_gradient: f32,
    /// Radius of the paint brush in world units.
    pub radius: f32,
    /// Strength of the paint effect (0.0-1.0).
    pub strength: f32,
    /// Falloff curve parameter (0.0 = hard, 1.0 = smooth).
    pub falloff: f32,
}

impl Default for PaintSettings {
    fn default() -> Self {
        Self {
            mode: PaintMode::Material,
            material_id: 0,
            color: Vec3::ONE, // white
            secondary_material_id: 0,
            blend_gradient: 0.5,
            radius: 1.0,
            strength: 1.0,
            falloff: 0.5,
        }
    }
}

/// A recorded paint stroke (sequence of brush applications along a path).
#[derive(Debug, Clone)]
pub struct PaintStroke {
    /// World-space points along the stroke path.
    pub points: Vec<Vec3>,
    /// Paint settings used for this stroke.
    pub settings: PaintSettings,
    /// Whether the stroke is currently in progress.
    started: bool,
}

impl PaintStroke {
    /// Create a new paint stroke with the given settings.
    pub fn new(settings: PaintSettings) -> Self {
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

/// Manages paint mode state including the current brush, active stroke, and history.
#[derive(Debug, Clone)]
pub struct PaintState {
    /// Current paint settings (applied to new strokes).
    pub current_settings: PaintSettings,
    /// The currently active stroke, if any.
    pub active_stroke: Option<PaintStroke>,
    /// History of completed strokes (for undo).
    pub stroke_history: Vec<PaintStroke>,
}

impl Default for PaintState {
    fn default() -> Self {
        Self {
            current_settings: PaintSettings::default(),
            active_stroke: None,
            stroke_history: Vec::new(),
        }
    }
}

impl PaintState {
    /// Create a new paint state with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new paint stroke at the given position.
    ///
    /// If a stroke is already active, it is finished and added to history first.
    pub fn begin_stroke(&mut self, pos: Vec3) {
        if self.active_stroke.is_some() {
            self.end_stroke();
        }

        let mut stroke = PaintStroke::new(self.current_settings.clone());
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
    pub fn undo_last_stroke(&mut self) -> Option<PaintStroke> {
        self.stroke_history.pop()
    }

    /// Set the paint mode.
    pub fn set_mode(&mut self, mode: PaintMode) {
        self.current_settings.mode = mode;
    }

    /// Set the primary material ID.
    pub fn set_material(&mut self, material_id: u16) {
        self.current_settings.material_id = material_id;
    }

    /// Set the paint color (RGB, each channel 0.0-1.0).
    pub fn set_color(&mut self, color: Vec3) {
        self.current_settings.color = Vec3::new(
            color.x.clamp(0.0, 1.0),
            color.y.clamp(0.0, 1.0),
            color.z.clamp(0.0, 1.0),
        );
    }

    /// Set the blend material IDs (primary and secondary).
    pub fn set_blend_materials(&mut self, primary: u16, secondary: u16) {
        self.current_settings.material_id = primary;
        self.current_settings.secondary_material_id = secondary;
    }
}

/// Compute the paint weight at a given distance from the brush center.
///
/// Like `brush_falloff` but multiplied by `strength` to produce the final
/// paint application weight. Returns a value from `strength` at the center
/// to 0.0 at the radius edge.
///
/// - `distance`: distance from the brush center
/// - `radius`: brush radius
/// - `strength`: brush strength (0.0-1.0)
/// - `falloff`: falloff parameter (0.0 = hard, 1.0 = smooth)
pub fn paint_weight_at(distance: f32, radius: f32, strength: f32, falloff: f32) -> f32 {
    if radius <= 0.0 || distance < 0.0 {
        return 0.0;
    }

    if distance >= radius {
        return 0.0;
    }

    if falloff <= 0.0 {
        // Hard cutoff
        return strength;
    }

    // Normalized distance [0, 1]
    let t = distance / radius;

    // Falloff zone starts at (1 - falloff) of the radius.
    let edge_start = 1.0 - falloff;

    if t <= edge_start {
        strength
    } else {
        let s = (t - edge_start) / falloff;
        // Smoothstep falloff, scaled by strength
        strength * (1.0 - s * s * (3.0 - 2.0 * s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // --- PaintSettings tests ---

    #[test]
    fn test_paint_settings_default() {
        let settings = PaintSettings::default();
        assert_eq!(settings.mode, PaintMode::Material);
        assert_eq!(settings.material_id, 0);
        assert!(vec3_approx_eq(settings.color, Vec3::ONE));
        assert_eq!(settings.secondary_material_id, 0);
        assert!(approx_eq(settings.blend_gradient, 0.5));
        assert!(approx_eq(settings.radius, 1.0));
        assert!(approx_eq(settings.strength, 1.0));
        assert!(approx_eq(settings.falloff, 0.5));
    }

    // --- PaintStroke tests ---

    #[test]
    fn test_paint_stroke_new() {
        let stroke = PaintStroke::new(PaintSettings::default());
        assert!(stroke.is_started());
        assert_eq!(stroke.point_count(), 0);
    }

    #[test]
    fn test_paint_stroke_add_points() {
        let mut stroke = PaintStroke::new(PaintSettings::default());
        stroke.add_point(Vec3::ZERO);
        stroke.add_point(Vec3::ONE);
        assert_eq!(stroke.point_count(), 2);
    }

    #[test]
    fn test_paint_stroke_finish() {
        let mut stroke = PaintStroke::new(PaintSettings::default());
        assert!(stroke.is_started());
        stroke.finish();
        assert!(!stroke.is_started());
    }

    // --- PaintState tests ---

    #[test]
    fn test_paint_state_new() {
        let state = PaintState::new();
        assert!(state.active_stroke.is_none());
        assert!(state.stroke_history.is_empty());
    }

    #[test]
    fn test_begin_paint_stroke() {
        let mut state = PaintState::new();
        state.begin_stroke(Vec3::new(1.0, 2.0, 3.0));

        assert!(state.active_stroke.is_some());
        let stroke = state.active_stroke.as_ref().unwrap();
        assert!(stroke.is_started());
        assert_eq!(stroke.point_count(), 1);
        assert_eq!(stroke.points[0], Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_continue_paint_stroke() {
        let mut state = PaintState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);
        state.continue_stroke(Vec3::new(2.0, 0.0, 0.0));

        let stroke = state.active_stroke.as_ref().unwrap();
        assert_eq!(stroke.point_count(), 3);
    }

    #[test]
    fn test_continue_paint_stroke_no_active() {
        let mut state = PaintState::new();
        state.continue_stroke(Vec3::ONE);
        assert!(state.active_stroke.is_none());
    }

    #[test]
    fn test_end_paint_stroke() {
        let mut state = PaintState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);
        state.end_stroke();

        assert!(state.active_stroke.is_none());
        assert_eq!(state.stroke_history.len(), 1);
        assert!(!state.stroke_history[0].is_started());
        assert_eq!(state.stroke_history[0].point_count(), 2);
    }

    #[test]
    fn test_end_paint_stroke_no_active() {
        let mut state = PaintState::new();
        state.end_stroke();
        assert!(state.stroke_history.is_empty());
    }

    #[test]
    fn test_begin_paint_stroke_finishes_previous() {
        let mut state = PaintState::new();
        state.begin_stroke(Vec3::ZERO);
        state.continue_stroke(Vec3::ONE);

        state.begin_stroke(Vec3::new(5.0, 0.0, 0.0));

        assert_eq!(state.stroke_history.len(), 1);
        assert_eq!(state.stroke_history[0].point_count(), 2);
        assert!(!state.stroke_history[0].is_started());

        assert!(state.active_stroke.is_some());
        assert_eq!(state.active_stroke.as_ref().unwrap().point_count(), 1);
    }

    #[test]
    fn test_undo_paint_stroke() {
        let mut state = PaintState::new();
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
    fn test_set_paint_mode() {
        let mut state = PaintState::new();
        state.set_mode(PaintMode::Color);
        assert_eq!(state.current_settings.mode, PaintMode::Color);

        state.set_mode(PaintMode::Blend);
        assert_eq!(state.current_settings.mode, PaintMode::Blend);
    }

    #[test]
    fn test_set_paint_material() {
        let mut state = PaintState::new();
        state.set_material(42);
        assert_eq!(state.current_settings.material_id, 42);
    }

    #[test]
    fn test_set_paint_color() {
        let mut state = PaintState::new();
        state.set_color(Vec3::new(1.0, 0.0, 0.5));
        assert!(vec3_approx_eq(state.current_settings.color, Vec3::new(1.0, 0.0, 0.5)));
    }

    #[test]
    fn test_set_paint_color_clamped() {
        let mut state = PaintState::new();
        state.set_color(Vec3::new(2.0, -1.0, 0.5));
        assert!(vec3_approx_eq(state.current_settings.color, Vec3::new(1.0, 0.0, 0.5)));
    }

    #[test]
    fn test_set_blend_materials() {
        let mut state = PaintState::new();
        state.set_blend_materials(5, 10);
        assert_eq!(state.current_settings.material_id, 5);
        assert_eq!(state.current_settings.secondary_material_id, 10);
    }

    // --- paint_weight_at tests ---

    #[test]
    fn test_paint_weight_at_center() {
        let w = paint_weight_at(0.0, 1.0, 0.8, 0.5);
        assert!(approx_eq(w, 0.8), "center should equal strength: {w}");
    }

    #[test]
    fn test_paint_weight_at_edge() {
        let w = paint_weight_at(1.0, 1.0, 0.8, 0.5);
        assert!(approx_eq(w, 0.0), "at radius should be 0.0: {w}");
    }

    #[test]
    fn test_paint_weight_at_beyond() {
        let w = paint_weight_at(2.0, 1.0, 1.0, 0.5);
        assert!(approx_eq(w, 0.0), "beyond radius should be 0.0: {w}");
    }

    #[test]
    fn test_paint_weight_at_hard_cutoff() {
        assert!(approx_eq(paint_weight_at(0.0, 1.0, 0.7, 0.0), 0.7));
        assert!(approx_eq(paint_weight_at(0.5, 1.0, 0.7, 0.0), 0.7));
        assert!(approx_eq(paint_weight_at(0.99, 1.0, 0.7, 0.0), 0.7));
        assert!(approx_eq(paint_weight_at(1.0, 1.0, 0.7, 0.0), 0.0));
    }

    #[test]
    fn test_paint_weight_at_zero_strength() {
        let w = paint_weight_at(0.0, 1.0, 0.0, 0.5);
        assert!(approx_eq(w, 0.0), "zero strength should give zero weight: {w}");
    }

    #[test]
    fn test_paint_weight_at_full_strength_full_falloff() {
        // At midpoint with full falloff and full strength
        let w = paint_weight_at(0.5, 1.0, 1.0, 1.0);
        assert!(w > 0.3 && w < 0.7,
            "midpoint with full falloff should be near 0.5: {w}");
    }

    #[test]
    fn test_paint_weight_at_monotonic() {
        let r = 2.0;
        let s = 0.9;
        let fo = 0.6;
        let mut prev = paint_weight_at(0.0, r, s, fo);
        for i in 1..20 {
            let d = (i as f32) * r / 20.0;
            let curr = paint_weight_at(d, r, s, fo);
            assert!(curr <= prev + EPS,
                "weight should decrease: d={d} curr={curr} prev={prev}");
            prev = curr;
        }
    }

    #[test]
    fn test_paint_weight_at_zero_radius() {
        let w = paint_weight_at(0.0, 0.0, 1.0, 0.5);
        assert!(approx_eq(w, 0.0), "zero radius should return 0: {w}");
    }

    #[test]
    fn test_paint_mode_eq() {
        assert_eq!(PaintMode::Material, PaintMode::Material);
        assert_ne!(PaintMode::Material, PaintMode::Color);
        assert_ne!(PaintMode::Color, PaintMode::Blend);
    }
}
