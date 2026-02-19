//! Debug visualization data model for the RKIField editor.
//!
//! Provides debug overlay modes that map to the engine's MCP `debug_mode` tool,
//! frame statistics tracking, and frame time history for performance monitoring.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Debug visualization modes for the viewport.
///
/// Each variant maps to the engine's MCP `debug_mode` index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugVizMode {
    None,
    Normals,
    MaterialIds,
    LodLevels,
    BrickOccupancy,
    RadianceSlice,
    FrameTime,
}

impl DebugVizMode {
    /// All modes in order, used for cycling.
    const ALL: [DebugVizMode; 7] = [
        DebugVizMode::None,
        DebugVizMode::Normals,
        DebugVizMode::MaterialIds,
        DebugVizMode::LodLevels,
        DebugVizMode::BrickOccupancy,
        DebugVizMode::RadianceSlice,
        DebugVizMode::FrameTime,
    ];

    /// Return the index of this mode in the ALL array.
    fn ordinal(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap_or(0)
    }

    /// Advance to the next mode, wrapping around.
    pub fn next(self) -> DebugVizMode {
        let idx = (self.ordinal() + 1) % Self::ALL.len();
        Self::ALL[idx]
    }
}

/// Debug overlay state for the editor viewport.
#[derive(Debug, Clone)]
pub struct DebugOverlay {
    /// Currently active visualization mode.
    pub active_mode: DebugVizMode,
    /// Whether to show the FPS counter.
    pub show_fps: bool,
    /// Whether to show the frame time graph.
    pub show_frame_time: bool,
    /// Whether to show engine statistics.
    pub show_stats: bool,
    /// Radiance clipmap level for RadianceSlice mode (0-3).
    pub radiance_slice_level: u32,
    /// Slice axis for RadianceSlice mode (0=X, 1=Y, 2=Z).
    pub radiance_slice_axis: u8,
    /// Slice offset for RadianceSlice mode (0.0-1.0).
    pub radiance_slice_offset: f32,
}

impl Default for DebugOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugOverlay {
    /// Create a new debug overlay with all visualizations off.
    pub fn new() -> Self {
        Self {
            active_mode: DebugVizMode::None,
            show_fps: false,
            show_frame_time: false,
            show_stats: false,
            radiance_slice_level: 0,
            radiance_slice_axis: 0,
            radiance_slice_offset: 0.5,
        }
    }

    /// Set the active debug visualization mode.
    pub fn set_mode(&mut self, mode: DebugVizMode) {
        self.active_mode = mode;
    }

    /// Cycle to the next debug visualization mode (wraps around).
    pub fn cycle_mode(&mut self) {
        self.active_mode = self.active_mode.next();
    }

    /// Toggle the FPS counter visibility.
    pub fn toggle_fps(&mut self) {
        self.show_fps = !self.show_fps;
    }

    /// Toggle the stats panel visibility.
    pub fn toggle_stats(&mut self) {
        self.show_stats = !self.show_stats;
    }

    /// Map the active mode to the engine's MCP `debug_mode` index.
    ///
    /// Mapping: None=0, Normals=1, MaterialIds=3, LodLevels=0 (no engine equivalent),
    /// BrickOccupancy=0, RadianceSlice=0, FrameTime=0.
    /// Only modes with direct engine shader support get non-zero indices.
    pub fn to_debug_mode_index(&self) -> u32 {
        match self.active_mode {
            DebugVizMode::None => 0,
            DebugVizMode::Normals => 1,
            DebugVizMode::MaterialIds => 3,
            DebugVizMode::LodLevels => 2,         // positions mode shows LOD info
            DebugVizMode::BrickOccupancy => 4,     // diffuse-only
            DebugVizMode::RadianceSlice => 5,      // specular-only
            DebugVizMode::FrameTime => 0,          // overlay-only, no shader mode
        }
    }
}

/// Snapshot of per-frame engine statistics.
#[derive(Debug, Clone, Copy)]
pub struct FrameStats {
    pub frame_time_ms: f32,
    pub fps: f32,
    pub brick_count: u32,
    pub entity_count: u32,
    pub light_count: u32,
    pub draw_calls: u32,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            frame_time_ms: 0.0,
            fps: 0.0,
            brick_count: 0,
            entity_count: 0,
            light_count: 0,
            draw_calls: 0,
        }
    }
}

/// Rolling history of frame times for graphs and statistics.
///
/// Capped at 120 samples (2 seconds at 60fps).
#[derive(Debug, Clone)]
pub struct FrameTimeHistory {
    samples: VecDeque<f32>,
}

const MAX_SAMPLES: usize = 120;

impl Default for FrameTimeHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameTimeHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(MAX_SAMPLES),
        }
    }

    /// Push a new frame time sample (in milliseconds).
    ///
    /// If the history is at capacity, the oldest sample is dropped.
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() >= MAX_SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    /// Average frame time across all samples.
    ///
    /// Returns 0.0 if no samples.
    pub fn average(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.samples.iter().sum();
        sum / self.samples.len() as f32
    }

    /// Minimum frame time across all samples.
    ///
    /// Returns 0.0 if no samples.
    pub fn min(&self) -> f32 {
        self.samples
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Maximum frame time across all samples.
    ///
    /// Returns 0.0 if no samples.
    pub fn max(&self) -> f32 {
        self.samples
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Most recent frame time.
    ///
    /// Returns 0.0 if no samples.
    pub fn current(&self) -> f32 {
        self.samples.back().copied().unwrap_or(0.0)
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_overlay_new() {
        let overlay = DebugOverlay::new();
        assert_eq!(overlay.active_mode, DebugVizMode::None);
        assert!(!overlay.show_fps);
        assert!(!overlay.show_frame_time);
        assert!(!overlay.show_stats);
        assert_eq!(overlay.radiance_slice_level, 0);
        assert_eq!(overlay.radiance_slice_axis, 0);
        assert!((overlay.radiance_slice_offset - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cycle_mode_wrapping() {
        let mut overlay = DebugOverlay::new();
        assert_eq!(overlay.active_mode, DebugVizMode::None);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::Normals);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::MaterialIds);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::LodLevels);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::BrickOccupancy);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::RadianceSlice);

        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::FrameTime);

        // Wrap back to None
        overlay.cycle_mode();
        assert_eq!(overlay.active_mode, DebugVizMode::None);
    }

    #[test]
    fn test_set_mode() {
        let mut overlay = DebugOverlay::new();
        overlay.set_mode(DebugVizMode::Normals);
        assert_eq!(overlay.active_mode, DebugVizMode::Normals);
        overlay.set_mode(DebugVizMode::RadianceSlice);
        assert_eq!(overlay.active_mode, DebugVizMode::RadianceSlice);
    }

    #[test]
    fn test_to_debug_mode_index() {
        let mut overlay = DebugOverlay::new();

        overlay.set_mode(DebugVizMode::None);
        assert_eq!(overlay.to_debug_mode_index(), 0);

        overlay.set_mode(DebugVizMode::Normals);
        assert_eq!(overlay.to_debug_mode_index(), 1);

        overlay.set_mode(DebugVizMode::MaterialIds);
        assert_eq!(overlay.to_debug_mode_index(), 3);

        overlay.set_mode(DebugVizMode::LodLevels);
        assert_eq!(overlay.to_debug_mode_index(), 2);

        overlay.set_mode(DebugVizMode::BrickOccupancy);
        assert_eq!(overlay.to_debug_mode_index(), 4);

        overlay.set_mode(DebugVizMode::RadianceSlice);
        assert_eq!(overlay.to_debug_mode_index(), 5);

        overlay.set_mode(DebugVizMode::FrameTime);
        assert_eq!(overlay.to_debug_mode_index(), 0);
    }

    #[test]
    fn test_toggle_fps() {
        let mut overlay = DebugOverlay::new();
        assert!(!overlay.show_fps);
        overlay.toggle_fps();
        assert!(overlay.show_fps);
        overlay.toggle_fps();
        assert!(!overlay.show_fps);
    }

    #[test]
    fn test_toggle_stats() {
        let mut overlay = DebugOverlay::new();
        assert!(!overlay.show_stats);
        overlay.toggle_stats();
        assert!(overlay.show_stats);
        overlay.toggle_stats();
        assert!(!overlay.show_stats);
    }

    #[test]
    fn test_frame_time_history_push_and_current() {
        let mut history = FrameTimeHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.current(), 0.0);

        history.push(16.6);
        assert_eq!(history.len(), 1);
        assert!((history.current() - 16.6).abs() < f32::EPSILON);

        history.push(8.3);
        assert_eq!(history.len(), 2);
        assert!((history.current() - 8.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame_time_history_average() {
        let mut history = FrameTimeHistory::new();
        assert_eq!(history.average(), 0.0);

        history.push(10.0);
        history.push(20.0);
        history.push(30.0);
        assert!((history.average() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame_time_history_min_max() {
        let mut history = FrameTimeHistory::new();
        assert_eq!(history.min(), 0.0);
        assert_eq!(history.max(), 0.0);

        history.push(10.0);
        history.push(5.0);
        history.push(20.0);
        history.push(15.0);

        assert!((history.min() - 5.0).abs() < f32::EPSILON);
        assert!((history.max() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame_time_history_capped_at_120() {
        let mut history = FrameTimeHistory::new();
        for i in 0..200 {
            history.push(i as f32);
        }
        assert_eq!(history.len(), 120);
        // Oldest remaining should be 80 (200 - 120)
        assert!((history.min() - 80.0).abs() < f32::EPSILON);
        // Most recent should be 199
        assert!((history.current() - 199.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_frame_stats_default() {
        let stats = FrameStats::default();
        assert_eq!(stats.frame_time_ms, 0.0);
        assert_eq!(stats.fps, 0.0);
        assert_eq!(stats.brick_count, 0);
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.light_count, 0);
        assert_eq!(stats.draw_calls, 0);
    }

    #[test]
    fn test_debug_viz_mode_next_from_each() {
        assert_eq!(DebugVizMode::None.next(), DebugVizMode::Normals);
        assert_eq!(DebugVizMode::Normals.next(), DebugVizMode::MaterialIds);
        assert_eq!(DebugVizMode::FrameTime.next(), DebugVizMode::None);
    }

    #[test]
    fn test_debug_overlay_default() {
        let overlay = DebugOverlay::default();
        assert_eq!(overlay.active_mode, DebugVizMode::None);
    }

    #[test]
    fn test_frame_time_history_default() {
        let history = FrameTimeHistory::default();
        assert!(history.is_empty());
    }
}
