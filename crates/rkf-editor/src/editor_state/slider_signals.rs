//! Slider signals for camera and brush properties.
//!
//! Environment slider signals have been migrated to the UI Store.
//! Only camera (FOV, speed, near/far) and brush (radius, strength,
//! falloff) remain here.

use rinch::prelude::Signal;

use super::EditorState;
use super::UiSignals;

/// Reactive signals for camera and brush slider properties.
#[derive(Clone, Copy)]
pub struct SliderSignals {
    // Camera
    pub fov: Signal<f64>,
    pub fly_speed: Signal<f64>,
    pub near: Signal<f64>,
    pub far: Signal<f64>,
    // Brush
    pub brush_radius: Signal<f64>,
    pub brush_strength: Signal<f64>,
    pub brush_falloff: Signal<f64>,
    // Environment fields removed — migrated to UI Store bound widgets.
}

// Old send_env_* functions removed — all environment field writes now go
// through the UI Store's set() method.

impl SliderSignals {
    /// Create slider signals initialized from the current `EditorState`.
    /// Must be called on the main thread (signals use thread-local reactive state).
    ///
    /// Camera and brush sliders init from EditorState.
    /// Environment sliders removed — all handled by UI Store bound widgets.
    pub fn new(es: &EditorState) -> Self {
        Self {
            fov: Signal::new(es.editor_camera_fov_degrees() as f64),
            fly_speed: Signal::new(es.camera_control.fly_speed as f64),
            near: Signal::new(es.editor_camera_near() as f64),
            far: Signal::new(es.editor_camera_far() as f64),
            brush_radius: Signal::new(es.sculpt.current_settings.radius as f64),
            brush_strength: Signal::new(es.sculpt.current_settings.strength as f64),
            brush_falloff: Signal::new(es.sculpt.current_settings.falloff as f64),
        }
    }

    /// Send camera-related commands (FOV, speed, near/far).
    pub fn send_camera_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetCameraFov {
            fov: self.fov.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetCameraSpeed {
            speed: self.fly_speed.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetCameraNearFar {
            near: self.near.get() as f32,
            far: self.far.get() as f32,
        });
    }

    // All environment send_*_commands removed — migrated to store-bound widgets.
    // Sun direction (derived from azimuth/elevation) will return as a
    // DirectionInput widget.

    /// Send brush/sculpt/paint settings commands.
    pub fn send_brush_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetSculptSettings {
            radius: self.brush_radius.get() as f32,
            strength: self.brush_strength.get() as f32,
            falloff: self.brush_falloff.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetPaintSettings {
            radius: self.brush_radius.get() as f32,
            strength: self.brush_strength.get() as f32,
            falloff: self.brush_falloff.get() as f32,
        });
    }

    /// Send all non-store slider values as EditorCommands.
    ///
    /// Environment fields are now handled by store-bound widgets.
    /// This only sends camera and brush.
    pub fn send_all_commands(&self, cmd: &crate::CommandSender, _ui: &UiSignals) {
        self.send_camera_commands(cmd);
        self.send_brush_commands(cmd);
    }

    /// Write camera + brush slider values back to `EditorState`.
    ///
    /// Environment values are no longer synced here — they flow through the
    /// ECS `EnvironmentSettings` component via `SetComponentField` commands.
    pub fn sync_to_state(&self, es: &mut EditorState) {
        // Camera
        es.set_editor_camera_component_field(|c| c.fov_degrees = self.fov.get() as f32);
        es.camera_control.fly_speed = self.fly_speed.get() as f32;
        es.set_editor_camera_component_field(|c| { c.near = self.near.get() as f32; c.far = self.far.get() as f32; });

        // Brush — sync to both sculpt and paint settings
        let radius = self.brush_radius.get() as f32;
        let strength = self.brush_strength.get() as f32;
        let falloff = self.brush_falloff.get() as f32;
        es.sculpt.set_radius(radius);
        es.sculpt.set_strength(strength);
        es.sculpt.current_settings.falloff = falloff;
        es.paint.current_settings.radius = radius;
        es.paint.current_settings.strength = strength;
        es.paint.current_settings.falloff = falloff;
    }
}
