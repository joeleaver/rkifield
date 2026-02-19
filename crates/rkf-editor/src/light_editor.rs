//! Light editing data model for the RKIField editor.
//!
//! Provides light type definitions, per-light property storage, selection tracking,
//! and a manager for adding/removing/editing lights in the scene. This is a pure
//! data model independent of the GUI framework.

#![allow(dead_code)]

use glam::Vec3;

/// The type of editor light.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorLightType {
    /// Omnidirectional light with position and range.
    Point,
    /// Cone-shaped light with inner/outer angles.
    Spot,
    /// Infinitely distant light affecting the entire scene.
    Directional,
}

/// An individual light in the editor scene.
#[derive(Debug, Clone)]
pub struct EditorLight {
    /// Unique identifier for this light.
    pub id: u64,
    /// The type of light (point, spot, directional).
    pub light_type: EditorLightType,
    /// World-space position (ignored for directional lights in shading, but
    /// kept for gizmo placement in the editor).
    pub position: Vec3,
    /// Direction the light points (relevant for spot and directional).
    pub direction: Vec3,
    /// Linear RGB color, each component in 0..1.
    pub color: Vec3,
    /// Luminous intensity multiplier.
    pub intensity: f32,
    /// Maximum influence range in world units.
    pub range: f32,
    /// Inner cone half-angle in radians (spot lights only).
    pub spot_inner_angle: f32,
    /// Outer cone half-angle in radians (spot lights only).
    pub spot_outer_angle: f32,
    /// Whether this light casts shadows.
    pub cast_shadows: bool,
    /// Optional path to a cookie/gobo texture.
    pub cookie_path: Option<String>,
}

/// Manages the set of lights in the editor scene.
#[derive(Debug, Clone)]
pub struct LightEditor {
    /// All lights in the scene.
    lights: Vec<EditorLight>,
    /// Currently selected light id, if any.
    selected_id: Option<u64>,
    /// Monotonically increasing id counter.
    next_id: u64,
    /// Whether lights have changed since last `clear_dirty()`.
    dirty: bool,
}

impl Default for LightEditor {
    fn default() -> Self {
        Self {
            lights: Vec::new(),
            selected_id: None,
            next_id: 1,
            dirty: false,
        }
    }
}

impl LightEditor {
    /// Create a new empty light editor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any light has been added, removed, or modified since the last `clear_dirty()`.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Mark the state as dirty (needs re-upload to engine).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear the dirty flag after the runtime has consumed the changes.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Add a light of the given type with sensible defaults. Returns its id.
    pub fn add_light(&mut self, light_type: EditorLightType) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let light = match light_type {
            EditorLightType::Point => EditorLight {
                id,
                light_type,
                position: Vec3::ZERO,
                direction: Vec3::NEG_Y,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_inner_angle: 0.0,
                spot_outer_angle: 0.0,
                cast_shadows: true,
                cookie_path: None,
            },
            EditorLightType::Spot => EditorLight {
                id,
                light_type,
                position: Vec3::ZERO,
                direction: Vec3::NEG_Y,
                color: Vec3::ONE,
                intensity: 1.0,
                range: 15.0,
                spot_inner_angle: 0.3,
                spot_outer_angle: 0.5,
                cast_shadows: true,
                cookie_path: None,
            },
            EditorLightType::Directional => EditorLight {
                id,
                light_type,
                position: Vec3::ZERO,
                direction: Vec3::new(0.0, -1.0, 0.0),
                color: Vec3::ONE,
                intensity: 0.8,
                range: f32::INFINITY,
                spot_inner_angle: 0.0,
                spot_outer_angle: 0.0,
                cast_shadows: true,
                cookie_path: None,
            },
        };

        self.lights.push(light);
        self.dirty = true;
        id
    }

    /// Remove a light by id. Returns true if the light was found and removed.
    pub fn remove_light(&mut self, id: u64) -> bool {
        let len_before = self.lights.len();
        self.lights.retain(|l| l.id != id);
        let removed = self.lights.len() < len_before;

        // Clear selection if the removed light was selected.
        if self.selected_id == Some(id) {
            self.selected_id = None;
        }

        if removed {
            self.dirty = true;
        }
        removed
    }

    /// Get an immutable reference to a light by id.
    pub fn get_light(&self, id: u64) -> Option<&EditorLight> {
        self.lights.iter().find(|l| l.id == id)
    }

    /// Get a mutable reference to a light by id.
    pub fn get_light_mut(&mut self, id: u64) -> Option<&mut EditorLight> {
        self.lights.iter_mut().find(|l| l.id == id)
    }

    /// Select a light by id.
    pub fn select(&mut self, id: u64) {
        self.selected_id = Some(id);
    }

    /// Clear the current selection.
    pub fn deselect(&mut self) {
        self.selected_id = None;
    }

    /// Return a reference to the currently selected light, if any.
    pub fn selected(&self) -> Option<&EditorLight> {
        self.selected_id.and_then(|id| self.get_light(id))
    }

    /// Return a slice of all lights.
    pub fn all_lights(&self) -> &[EditorLight] {
        &self.lights
    }

    /// Set the position of a light by id.
    pub fn set_position(&mut self, id: u64, pos: Vec3) {
        if let Some(light) = self.get_light_mut(id) {
            light.position = pos;
            self.dirty = true;
        }
    }

    /// Set the color of a light by id.
    pub fn set_color(&mut self, id: u64, color: Vec3) {
        if let Some(light) = self.get_light_mut(id) {
            light.color = color;
            self.dirty = true;
        }
    }

    /// Set the intensity of a light by id.
    pub fn set_intensity(&mut self, id: u64, intensity: f32) {
        if let Some(light) = self.get_light_mut(id) {
            light.intensity = intensity;
            self.dirty = true;
        }
    }

    /// Set the range of a light by id.
    pub fn set_range(&mut self, id: u64, range: f32) {
        if let Some(light) = self.get_light_mut(id) {
            light.range = range;
            self.dirty = true;
        }
    }

    /// Set spot cone angles for a light by id.
    ///
    /// Validates that `inner <= outer`. If `inner > outer`, inner is clamped
    /// to equal outer.
    pub fn set_spot_angles(&mut self, id: u64, inner: f32, outer: f32) {
        if let Some(light) = self.get_light_mut(id) {
            let clamped_inner = if inner > outer { outer } else { inner };
            light.spot_inner_angle = clamped_inner;
            light.spot_outer_angle = outer;
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // --- Add / remove ---

    #[test]
    fn test_add_point_light() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        assert_eq!(editor.all_lights().len(), 1);
        let light = editor.get_light(id).unwrap();
        assert_eq!(light.light_type, EditorLightType::Point);
        assert!(approx_eq(light.range, 10.0));
        assert!(approx_eq(light.intensity, 1.0));
    }

    #[test]
    fn test_add_spot_light() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Spot);
        let light = editor.get_light(id).unwrap();
        assert_eq!(light.light_type, EditorLightType::Spot);
        assert!(approx_eq(light.range, 15.0));
        assert!(approx_eq(light.spot_inner_angle, 0.3));
        assert!(approx_eq(light.spot_outer_angle, 0.5));
        assert!(approx_eq(light.intensity, 1.0));
    }

    #[test]
    fn test_add_directional_light() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Directional);
        let light = editor.get_light(id).unwrap();
        assert_eq!(light.light_type, EditorLightType::Directional);
        assert!(approx_eq(light.intensity, 0.8));
        assert!(approx_eq(light.direction.y, -1.0));
    }

    #[test]
    fn test_unique_ids() {
        let mut editor = LightEditor::new();
        let id1 = editor.add_light(EditorLightType::Point);
        let id2 = editor.add_light(EditorLightType::Spot);
        let id3 = editor.add_light(EditorLightType::Directional);
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_remove_light() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        assert!(editor.remove_light(id));
        assert!(editor.all_lights().is_empty());
        assert!(editor.get_light(id).is_none());
    }

    #[test]
    fn test_remove_nonexistent_light() {
        let mut editor = LightEditor::new();
        assert!(!editor.remove_light(999));
    }

    #[test]
    fn test_remove_clears_selection() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.select(id);
        assert!(editor.selected().is_some());
        editor.remove_light(id);
        assert!(editor.selected().is_none());
    }

    // --- Select / deselect ---

    #[test]
    fn test_select_light() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.select(id);
        let selected = editor.selected().unwrap();
        assert_eq!(selected.id, id);
    }

    #[test]
    fn test_deselect() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.select(id);
        editor.deselect();
        assert!(editor.selected().is_none());
    }

    #[test]
    fn test_select_nonexistent_returns_none() {
        let mut editor = LightEditor::new();
        editor.select(999);
        // selected_id is set, but selected() resolves to None
        assert!(editor.selected().is_none());
    }

    // --- Property setters ---

    #[test]
    fn test_set_position() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.set_position(id, Vec3::new(1.0, 2.0, 3.0));
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.position.x, 1.0));
        assert!(approx_eq(light.position.y, 2.0));
        assert!(approx_eq(light.position.z, 3.0));
    }

    #[test]
    fn test_set_color() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.set_color(id, Vec3::new(1.0, 0.5, 0.0));
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.color.x, 1.0));
        assert!(approx_eq(light.color.y, 0.5));
        assert!(approx_eq(light.color.z, 0.0));
    }

    #[test]
    fn test_set_intensity() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.set_intensity(id, 5.0);
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.intensity, 5.0));
    }

    #[test]
    fn test_set_range() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        editor.set_range(id, 25.0);
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.range, 25.0));
    }

    #[test]
    fn test_set_spot_angles_valid() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Spot);
        editor.set_spot_angles(id, 0.2, 0.6);
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.spot_inner_angle, 0.2));
        assert!(approx_eq(light.spot_outer_angle, 0.6));
    }

    #[test]
    fn test_set_spot_angles_inner_greater_than_outer() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Spot);
        editor.set_spot_angles(id, 0.8, 0.4);
        let light = editor.get_light(id).unwrap();
        // inner should be clamped to outer
        assert!(approx_eq(light.spot_inner_angle, 0.4));
        assert!(approx_eq(light.spot_outer_angle, 0.4));
    }

    #[test]
    fn test_set_spot_angles_equal() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Spot);
        editor.set_spot_angles(id, 0.5, 0.5);
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.spot_inner_angle, 0.5));
        assert!(approx_eq(light.spot_outer_angle, 0.5));
    }

    // --- Mutable access ---

    #[test]
    fn test_get_light_mut() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        {
            let light = editor.get_light_mut(id).unwrap();
            light.cast_shadows = false;
            light.cookie_path = Some("cookie.png".to_string());
        }
        let light = editor.get_light(id).unwrap();
        assert!(!light.cast_shadows);
        assert_eq!(light.cookie_path.as_deref(), Some("cookie.png"));
    }

    #[test]
    fn test_get_light_nonexistent() {
        let editor = LightEditor::new();
        assert!(editor.get_light(42).is_none());
    }

    // --- all_lights ---

    #[test]
    fn test_all_lights_empty() {
        let editor = LightEditor::new();
        assert!(editor.all_lights().is_empty());
    }

    #[test]
    fn test_all_lights_multiple() {
        let mut editor = LightEditor::new();
        editor.add_light(EditorLightType::Point);
        editor.add_light(EditorLightType::Spot);
        editor.add_light(EditorLightType::Directional);
        assert_eq!(editor.all_lights().len(), 3);
    }

    // --- Default values per type ---

    #[test]
    fn test_point_default_color_white() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        let light = editor.get_light(id).unwrap();
        assert!(approx_eq(light.color.x, 1.0));
        assert!(approx_eq(light.color.y, 1.0));
        assert!(approx_eq(light.color.z, 1.0));
    }

    #[test]
    fn test_default_cast_shadows() {
        let mut editor = LightEditor::new();
        let id = editor.add_light(EditorLightType::Point);
        let light = editor.get_light(id).unwrap();
        assert!(light.cast_shadows);
    }

    #[test]
    fn test_set_property_nonexistent_id_is_noop() {
        let mut editor = LightEditor::new();
        // None of these should panic.
        editor.set_position(999, Vec3::ONE);
        editor.set_color(999, Vec3::ONE);
        editor.set_intensity(999, 5.0);
        editor.set_range(999, 20.0);
        editor.set_spot_angles(999, 0.1, 0.5);
    }
}
