//! Input state and routing for the RKIField editor.
//!
//! Handles input state tracking and focus routing between the viewport and UI panels.
//! This is a pure data model that can be tested independently of the GUI framework.

#![allow(dead_code)] // Variants used by future editor panels

use std::collections::HashSet;

/// Which UI region currently has input focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusTarget {
    Viewport,
    SceneHierarchy,
    Properties,
    AssetBrowser,
    MenuBar,
    None,
}

/// Modifier key state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
}

/// Simplified key codes for common editor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    // Movement
    W,
    A,
    S,
    D,
    Q,
    E,
    // Gizmo mode (grab, rotate, scale)
    G,
    R,
    L,
    // Axis constraint
    X,
    Y,
    Z,
    // General
    Delete,
    Escape,
    Space,
    Tab,
    Return,
    // Focus selected
    F,
    // View presets
    Num1,
    Num2,
    Num3,
    // Modifiers (used as movement keys)
    ShiftLeft,
    // Function keys
    F5,
    F12,
}

/// Screen-space bounds of the viewport region.
#[derive(Debug, Clone, Copy)]
pub struct ViewportBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportBounds {
    /// Create new viewport bounds.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Check if a screen-space point is inside these bounds.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x
            && px < self.x + self.width
            && py >= self.y
            && py < self.y + self.height
    }
}

/// Tracks all input state for the editor, including mouse, keyboard, and focus.
pub struct InputState {
    pub focus: FocusTarget,
    pub mouse_pos: glam::Vec2,
    pub mouse_delta: glam::Vec2,
    /// Mouse button state: \[left, right, middle\].
    pub mouse_buttons: [bool; 3],
    pub scroll_delta: f32,
    pub keys_pressed: HashSet<KeyCode>,
    /// Keys pressed this frame (cleared each frame by reset_frame_deltas).
    pub keys_just_pressed: HashSet<KeyCode>,
    pub modifiers: Modifiers,
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

impl InputState {
    /// Create a new input state with everything zeroed / released.
    pub fn new() -> Self {
        Self {
            focus: FocusTarget::None,
            mouse_pos: glam::Vec2::ZERO,
            mouse_delta: glam::Vec2::ZERO,
            mouse_buttons: [false; 3],
            scroll_delta: 0.0,
            keys_pressed: HashSet::new(),
            keys_just_pressed: HashSet::new(),
            modifiers: Modifiers::default(),
        }
    }

    /// Update focus target based on current mouse position and viewport bounds.
    ///
    /// If the mouse is inside the viewport, focus is set to `Viewport`.
    /// Otherwise focus is set to `None` (other panels would set their own focus
    /// via their own hit-testing in the full implementation).
    pub fn update_focus(&mut self, viewport: &ViewportBounds) {
        if viewport.contains(self.mouse_pos.x, self.mouse_pos.y) {
            self.focus = FocusTarget::Viewport;
        } else {
            // In a full implementation, we'd check other panel bounds here.
            // For now, anything outside viewport gets None.
            self.focus = FocusTarget::None;
        }
    }

    /// Returns true if the viewport currently has input focus.
    pub fn is_viewport_focused(&self) -> bool {
        self.focus == FocusTarget::Viewport
    }

    /// Returns true if the given key is currently pressed.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Returns true if the given mouse button is currently held down.
    ///
    /// Button indices: 0 = left, 1 = right, 2 = middle.
    pub fn is_mouse_button_down(&self, button: usize) -> bool {
        self.mouse_buttons.get(button).copied().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_viewport() -> ViewportBounds {
        ViewportBounds::new(250.0, 32.0, 730.0, 664.0)
    }

    #[test]
    fn test_focus_in_viewport() {
        let mut input = InputState::new();
        let vp = default_viewport();

        // Mouse clearly inside viewport
        input.mouse_pos = glam::Vec2::new(500.0, 400.0);
        input.update_focus(&vp);
        assert_eq!(input.focus, FocusTarget::Viewport);
        assert!(input.is_viewport_focused());
    }

    #[test]
    fn test_focus_outside_viewport() {
        let mut input = InputState::new();
        let vp = default_viewport();

        // Mouse in the left panel area (x < 250)
        input.mouse_pos = glam::Vec2::new(100.0, 400.0);
        input.update_focus(&vp);
        assert_ne!(input.focus, FocusTarget::Viewport);
        assert!(!input.is_viewport_focused());

        // Mouse in the menu bar area (y < 32)
        input.mouse_pos = glam::Vec2::new(500.0, 10.0);
        input.update_focus(&vp);
        assert!(!input.is_viewport_focused());

        // Mouse in the right panel area (x >= 250 + 730 = 980)
        input.mouse_pos = glam::Vec2::new(990.0, 400.0);
        input.update_focus(&vp);
        assert!(!input.is_viewport_focused());
    }

    #[test]
    fn test_focus_at_viewport_boundary() {
        let mut input = InputState::new();
        let vp = default_viewport();

        // Exactly at top-left corner of viewport (inclusive)
        input.mouse_pos = glam::Vec2::new(250.0, 32.0);
        input.update_focus(&vp);
        assert!(input.is_viewport_focused());

        // Just past the bottom-right boundary (exclusive)
        input.mouse_pos = glam::Vec2::new(250.0 + 730.0, 32.0 + 664.0);
        input.update_focus(&vp);
        assert!(!input.is_viewport_focused());
    }

    #[test]
    fn test_key_pressed() {
        let mut input = InputState::new();
        assert!(!input.is_key_pressed(KeyCode::W));

        input.keys_pressed.insert(KeyCode::W);
        assert!(input.is_key_pressed(KeyCode::W));
        assert!(!input.is_key_pressed(KeyCode::S));

        input.keys_pressed.remove(&KeyCode::W);
        assert!(!input.is_key_pressed(KeyCode::W));
    }

    #[test]
    fn test_multiple_keys() {
        let mut input = InputState::new();
        input.keys_pressed.insert(KeyCode::W);
        input.keys_pressed.insert(KeyCode::A);
        input.keys_pressed.insert(KeyCode::Space);

        assert!(input.is_key_pressed(KeyCode::W));
        assert!(input.is_key_pressed(KeyCode::A));
        assert!(input.is_key_pressed(KeyCode::Space));
        assert!(!input.is_key_pressed(KeyCode::D));
    }

    #[test]
    fn test_modifiers() {
        let mut input = InputState::new();

        // Default: nothing held
        assert!(!input.modifiers.shift);
        assert!(!input.modifiers.ctrl);
        assert!(!input.modifiers.alt);

        input.modifiers.shift = true;
        input.modifiers.ctrl = true;
        assert!(input.modifiers.shift);
        assert!(input.modifiers.ctrl);
        assert!(!input.modifiers.alt);

        input.modifiers = Modifiers::default();
        assert!(!input.modifiers.shift);
        assert!(!input.modifiers.ctrl);
    }

    #[test]
    fn test_mouse_buttons() {
        let mut input = InputState::new();

        // All released by default
        assert!(!input.is_mouse_button_down(0));
        assert!(!input.is_mouse_button_down(1));
        assert!(!input.is_mouse_button_down(2));

        // Press left mouse
        input.mouse_buttons[0] = true;
        assert!(input.is_mouse_button_down(0));
        assert!(!input.is_mouse_button_down(1));

        // Press right mouse
        input.mouse_buttons[1] = true;
        assert!(input.is_mouse_button_down(1));

        // Invalid index returns false
        assert!(!input.is_mouse_button_down(5));
    }

    #[test]
    fn test_mouse_delta_tracking() {
        let mut input = InputState::new();
        assert_eq!(input.mouse_delta, glam::Vec2::ZERO);

        input.mouse_delta = glam::Vec2::new(5.0, -3.0);
        assert_eq!(input.mouse_delta.x, 5.0);
        assert_eq!(input.mouse_delta.y, -3.0);
    }

    #[test]
    fn test_scroll_delta() {
        let mut input = InputState::new();
        assert_eq!(input.scroll_delta, 0.0);

        input.scroll_delta = 2.5;
        assert_eq!(input.scroll_delta, 2.5);

        input.scroll_delta = -1.0;
        assert_eq!(input.scroll_delta, -1.0);
    }

    #[test]
    fn test_default_state() {
        let input = InputState::default();
        assert_eq!(input.focus, FocusTarget::None);
        assert_eq!(input.mouse_pos, glam::Vec2::ZERO);
        assert_eq!(input.mouse_delta, glam::Vec2::ZERO);
        assert_eq!(input.scroll_delta, 0.0);
        assert!(input.keys_pressed.is_empty());
        assert!(!input.is_mouse_button_down(0));
    }

    #[test]
    fn test_viewport_bounds_contains() {
        let vp = ViewportBounds::new(10.0, 20.0, 100.0, 200.0);

        assert!(vp.contains(10.0, 20.0)); // top-left inclusive
        assert!(vp.contains(50.0, 100.0)); // center
        assert!(vp.contains(109.9, 219.9)); // near bottom-right
        assert!(!vp.contains(110.0, 220.0)); // at exclusive boundary
        assert!(!vp.contains(9.0, 20.0)); // just left
        assert!(!vp.contains(10.0, 19.0)); // just above
    }
}
