//! Animation preview data model for the RKIField editor.
//!
//! Provides playback state tracking, timeline scrubbing, looping, speed control,
//! skeleton overlay toggling, and blend shape weight management. This is a pure
//! data model independent of the GUI framework and the GPU animation pipeline.

#![allow(dead_code)]

/// Playback state of the animation preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Animation is actively advancing each frame.
    Playing,
    /// Animation is frozen at the current time.
    Paused,
    /// Animation is inactive (time reset to zero).
    Stopped,
}

/// Animation preview controller for a single entity.
#[derive(Debug, Clone)]
pub struct AnimationPreview {
    /// The entity whose animation is being previewed, if any.
    pub entity_id: Option<u64>,
    /// Current playback state.
    pub playback_state: PlaybackState,
    /// Current playback time in seconds.
    pub current_time: f32,
    /// Total animation duration in seconds.
    pub duration: f32,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f32,
    /// Whether the animation loops back to the start on completion.
    pub looping: bool,
    /// Whether to render the skeleton overlay in the viewport.
    pub show_skeleton: bool,
    /// Named blend shape weights (name, weight 0..1).
    pub blend_shape_weights: Vec<(String, f32)>,
}

impl Default for AnimationPreview {
    fn default() -> Self {
        Self {
            entity_id: None,
            playback_state: PlaybackState::Stopped,
            current_time: 0.0,
            duration: 0.0,
            speed: 1.0,
            looping: false,
            show_skeleton: false,
            blend_shape_weights: Vec::new(),
        }
    }
}

impl AnimationPreview {
    /// Create a new animation preview in the stopped state with no entity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind an entity for preview, setting its duration and initializing
    /// blend shape weights (all at 0.0).
    pub fn set_entity(&mut self, id: u64, duration: f32, blend_shape_names: Vec<String>) {
        self.entity_id = Some(id);
        self.duration = duration;
        self.current_time = 0.0;
        self.playback_state = PlaybackState::Stopped;
        self.blend_shape_weights = blend_shape_names
            .into_iter()
            .map(|name| (name, 0.0))
            .collect();
    }

    /// Unbind the current entity and reset all state.
    pub fn clear_entity(&mut self) {
        self.entity_id = None;
        self.duration = 0.0;
        self.current_time = 0.0;
        self.playback_state = PlaybackState::Stopped;
        self.blend_shape_weights.clear();
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        self.playback_state = PlaybackState::Playing;
    }

    /// Pause playback at the current time.
    pub fn pause(&mut self) {
        self.playback_state = PlaybackState::Paused;
    }

    /// Stop playback and reset time to zero.
    pub fn stop(&mut self) {
        self.playback_state = PlaybackState::Stopped;
        self.current_time = 0.0;
    }

    /// Toggle between playing and paused. If stopped, starts playing.
    pub fn toggle_playback(&mut self) {
        match self.playback_state {
            PlaybackState::Playing => self.pause(),
            PlaybackState::Paused | PlaybackState::Stopped => self.play(),
        }
    }

    /// Scrub to a specific time, clamped to `[0, duration]`.
    pub fn scrub_to(&mut self, time: f32) {
        self.current_time = time.clamp(0.0, self.duration);
    }

    /// Advance the animation by `dt` seconds (scaled by `speed`).
    ///
    /// When looping is enabled and the end is reached, wraps around.
    /// When looping is disabled and the end is reached, clamps to duration
    /// and pauses.
    pub fn advance(&mut self, dt: f32) {
        if self.playback_state != PlaybackState::Playing {
            return;
        }
        if self.duration <= 0.0 {
            return;
        }

        self.current_time += dt * self.speed;

        if self.current_time >= self.duration {
            if self.looping {
                // Wrap around, preserving fractional overshoot.
                self.current_time %= self.duration;
            } else {
                self.current_time = self.duration;
                self.playback_state = PlaybackState::Paused;
            }
        } else if self.current_time < 0.0 {
            // Negative speed could cause this.
            if self.looping {
                self.current_time = self.duration + (self.current_time % self.duration);
            } else {
                self.current_time = 0.0;
                self.playback_state = PlaybackState::Paused;
            }
        }
    }

    /// Set the playback speed multiplier.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Toggle skeleton overlay visibility.
    pub fn toggle_skeleton(&mut self) {
        self.show_skeleton = !self.show_skeleton;
    }

    /// Set a blend shape weight by name, clamped to `[0, 1]`.
    ///
    /// Does nothing if the name is not found.
    pub fn set_blend_weight(&mut self, name: &str, weight: f32) {
        if let Some(entry) = self.blend_shape_weights.iter_mut().find(|(n, _)| n == name) {
            entry.1 = weight.clamp(0.0, 1.0);
        }
    }

    /// Get the current weight of a blend shape by name.
    pub fn get_blend_weight(&self, name: &str) -> Option<f32> {
        self.blend_shape_weights
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, w)| *w)
    }

    /// Return the normalized playback position in `[0, 1]`.
    ///
    /// Returns 0.0 when duration is zero.
    pub fn normalized_time(&self) -> f32 {
        if self.duration <= 0.0 {
            return 0.0;
        }
        self.current_time / self.duration
    }

    /// Whether the preview is currently playing.
    pub fn is_playing(&self) -> bool {
        self.playback_state == PlaybackState::Playing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // --- Construction ---

    #[test]
    fn test_new_default() {
        let preview = AnimationPreview::new();
        assert!(preview.entity_id.is_none());
        assert_eq!(preview.playback_state, PlaybackState::Stopped);
        assert!(approx_eq(preview.current_time, 0.0));
        assert!(approx_eq(preview.duration, 0.0));
        assert!(approx_eq(preview.speed, 1.0));
        assert!(!preview.looping);
        assert!(!preview.show_skeleton);
        assert!(preview.blend_shape_weights.is_empty());
    }

    // --- Entity binding ---

    #[test]
    fn test_set_entity() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(42, 3.0, vec!["smile".into(), "blink".into()]);
        assert_eq!(preview.entity_id, Some(42));
        assert!(approx_eq(preview.duration, 3.0));
        assert_eq!(preview.blend_shape_weights.len(), 2);
        assert_eq!(preview.blend_shape_weights[0].0, "smile");
        assert!(approx_eq(preview.blend_shape_weights[0].1, 0.0));
    }

    #[test]
    fn test_set_entity_resets_time() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.play();
        preview.advance(2.0);
        // Re-bind resets time.
        preview.set_entity(2, 10.0, vec!["jaw".into()]);
        assert!(approx_eq(preview.current_time, 0.0));
        assert_eq!(preview.playback_state, PlaybackState::Stopped);
    }

    #[test]
    fn test_clear_entity() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(42, 3.0, vec!["smile".into()]);
        preview.play();
        preview.advance(1.0);
        preview.clear_entity();
        assert!(preview.entity_id.is_none());
        assert!(approx_eq(preview.current_time, 0.0));
        assert!(approx_eq(preview.duration, 0.0));
        assert_eq!(preview.playback_state, PlaybackState::Stopped);
        assert!(preview.blend_shape_weights.is_empty());
    }

    // --- Playback lifecycle ---

    #[test]
    fn test_play_pause_stop() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        assert_eq!(preview.playback_state, PlaybackState::Stopped);

        preview.play();
        assert_eq!(preview.playback_state, PlaybackState::Playing);
        assert!(preview.is_playing());

        preview.pause();
        assert_eq!(preview.playback_state, PlaybackState::Paused);
        assert!(!preview.is_playing());

        preview.play();
        preview.advance(1.0);
        preview.stop();
        assert_eq!(preview.playback_state, PlaybackState::Stopped);
        assert!(approx_eq(preview.current_time, 0.0));
    }

    #[test]
    fn test_toggle_playback() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);

        // Stopped -> Playing
        preview.toggle_playback();
        assert_eq!(preview.playback_state, PlaybackState::Playing);

        // Playing -> Paused
        preview.toggle_playback();
        assert_eq!(preview.playback_state, PlaybackState::Paused);

        // Paused -> Playing
        preview.toggle_playback();
        assert_eq!(preview.playback_state, PlaybackState::Playing);
    }

    // --- Advance ---

    #[test]
    fn test_advance_basic() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.play();
        preview.advance(1.5);
        assert!(approx_eq(preview.current_time, 1.5));
    }

    #[test]
    fn test_advance_with_speed() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 10.0, vec![]);
        preview.set_speed(2.0);
        preview.play();
        preview.advance(1.0);
        assert!(approx_eq(preview.current_time, 2.0));
    }

    #[test]
    fn test_advance_no_loop_clamps_and_pauses() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 3.0, vec![]);
        preview.looping = false;
        preview.play();
        preview.advance(5.0);
        assert!(approx_eq(preview.current_time, 3.0));
        assert_eq!(preview.playback_state, PlaybackState::Paused);
    }

    #[test]
    fn test_advance_with_loop_wraps() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 3.0, vec![]);
        preview.looping = true;
        preview.play();
        preview.advance(5.0); // 5.0 mod 3.0 = 2.0
        assert!(approx_eq(preview.current_time, 2.0));
        assert!(preview.is_playing());
    }

    #[test]
    fn test_advance_while_paused_does_nothing() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.pause();
        preview.advance(2.0);
        assert!(approx_eq(preview.current_time, 0.0));
    }

    #[test]
    fn test_advance_while_stopped_does_nothing() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.advance(2.0);
        assert!(approx_eq(preview.current_time, 0.0));
    }

    #[test]
    fn test_advance_zero_duration() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 0.0, vec![]);
        preview.play();
        preview.advance(1.0);
        // Should not panic or produce NaN.
        assert!(approx_eq(preview.current_time, 0.0));
    }

    // --- Scrub ---

    #[test]
    fn test_scrub_within_range() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.scrub_to(2.5);
        assert!(approx_eq(preview.current_time, 2.5));
    }

    #[test]
    fn test_scrub_clamps_above() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.scrub_to(10.0);
        assert!(approx_eq(preview.current_time, 5.0));
    }

    #[test]
    fn test_scrub_clamps_below() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec![]);
        preview.scrub_to(-2.0);
        assert!(approx_eq(preview.current_time, 0.0));
    }

    // --- Blend shapes ---

    #[test]
    fn test_set_blend_weight() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec!["smile".into(), "blink".into()]);
        preview.set_blend_weight("smile", 0.7);
        assert!(approx_eq(preview.get_blend_weight("smile").unwrap(), 0.7));
        assert!(approx_eq(preview.get_blend_weight("blink").unwrap(), 0.0));
    }

    #[test]
    fn test_set_blend_weight_clamps() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec!["smile".into()]);
        preview.set_blend_weight("smile", 1.5);
        assert!(approx_eq(preview.get_blend_weight("smile").unwrap(), 1.0));
        preview.set_blend_weight("smile", -0.5);
        assert!(approx_eq(preview.get_blend_weight("smile").unwrap(), 0.0));
    }

    #[test]
    fn test_get_blend_weight_missing() {
        let preview = AnimationPreview::new();
        assert!(preview.get_blend_weight("nonexistent").is_none());
    }

    #[test]
    fn test_set_blend_weight_unknown_name_is_noop() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 5.0, vec!["smile".into()]);
        preview.set_blend_weight("unknown", 0.5);
        assert!(preview.get_blend_weight("unknown").is_none());
    }

    // --- Skeleton toggle ---

    #[test]
    fn test_toggle_skeleton() {
        let mut preview = AnimationPreview::new();
        assert!(!preview.show_skeleton);
        preview.toggle_skeleton();
        assert!(preview.show_skeleton);
        preview.toggle_skeleton();
        assert!(!preview.show_skeleton);
    }

    // --- Normalized time ---

    #[test]
    fn test_normalized_time() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 4.0, vec![]);
        preview.scrub_to(2.0);
        assert!(approx_eq(preview.normalized_time(), 0.5));
    }

    #[test]
    fn test_normalized_time_at_start() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 4.0, vec![]);
        assert!(approx_eq(preview.normalized_time(), 0.0));
    }

    #[test]
    fn test_normalized_time_at_end() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 4.0, vec![]);
        preview.scrub_to(4.0);
        assert!(approx_eq(preview.normalized_time(), 1.0));
    }

    #[test]
    fn test_normalized_time_zero_duration() {
        let preview = AnimationPreview::new();
        assert!(approx_eq(preview.normalized_time(), 0.0));
    }

    // --- Speed ---

    #[test]
    fn test_set_speed() {
        let mut preview = AnimationPreview::new();
        preview.set_speed(0.5);
        assert!(approx_eq(preview.speed, 0.5));
    }

    #[test]
    fn test_half_speed_advance() {
        let mut preview = AnimationPreview::new();
        preview.set_entity(1, 10.0, vec![]);
        preview.set_speed(0.5);
        preview.play();
        preview.advance(2.0);
        assert!(approx_eq(preview.current_time, 1.0));
    }
}
