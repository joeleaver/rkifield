//! Slider signals for brush properties.
//!
//! Camera and environment slider signals have been migrated to the UI Store.
//! Only brush (radius, strength, falloff) remains here.

use rinch::prelude::Signal;

use super::EditorState;
use super::UiSignals;

/// Reactive signals for brush slider properties.
#[derive(Clone, Copy)]
pub struct SliderSignals {
    // Brush
    pub brush_radius: Signal<f64>,
    pub brush_strength: Signal<f64>,
    pub brush_falloff: Signal<f64>,
    // Camera and environment fields removed — migrated to UI Store bound widgets.
}

impl SliderSignals {
    /// Create slider signals initialized from the current `EditorState`.
    /// Must be called on the main thread (signals use thread-local reactive state).
    ///
    /// Only brush sliders remain. Camera and environment are handled by UI Store.
    pub fn new(es: &EditorState) -> Self {
        Self {
            brush_radius: Signal::new(es.sculpt.current_settings.radius as f64),
            brush_strength: Signal::new(es.sculpt.current_settings.strength as f64),
            brush_falloff: Signal::new(es.sculpt.current_settings.falloff as f64),
        }
    }

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
    /// Camera and environment fields are now handled by store-bound widgets.
    /// This only sends brush commands.
    pub fn send_all_commands(&self, cmd: &crate::CommandSender, _ui: &UiSignals) {
        self.send_brush_commands(cmd);
    }

    /// Write brush slider values back to `EditorState`.
    ///
    /// Camera and environment values are no longer synced here — they flow
    /// through the UI Store.
    pub fn sync_to_state(&self, es: &mut EditorState) {
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
