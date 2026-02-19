//! Edge case handling — graceful degradation, chunk boundaries, LOD transitions,
//! and window resize logic.
//!
//! These utilities help the engine adapt to resource pressure, smooth visual
//! discontinuities at chunk/LOD boundaries, and correctly reconfigure render
//! targets when the window is resized.

#![allow(dead_code)]

use glam::IVec3;

// ── Degradation ─────────────────────────────────────────────────────────────

/// How aggressively the engine should shed quality to stay within memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DegradationLevel {
    /// Full quality — everything enabled.
    Normal,
    /// Lower render scale, disable depth-of-field.
    ReduceQuality,
    /// Disable volumetrics, GI, bloom.
    DisableEffects,
    /// Minimal ray march, no shadows — survival mode.
    MinimalMode,
}

/// A recommended configuration change emitted by [`GracefulDegradation`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigChange {
    /// Dot-path of the config field to modify.
    pub field: String,
    /// Stringified previous value.
    pub old_value: String,
    /// Stringified recommended value.
    pub new_value: String,
    /// Why this change is recommended.
    pub reason: String,
}

/// Tracks memory pressure and recommends quality reductions to stay within
/// budget.
#[derive(Debug, Clone)]
pub struct GracefulDegradation {
    /// Hard memory ceiling in bytes.
    pub max_memory_bytes: u64,
    /// Most recently reported memory usage.
    pub current_memory_bytes: u64,
    /// Current degradation tier.
    pub degradation_level: DegradationLevel,
    /// Log of transitions for diagnostics.
    pub log_messages: Vec<String>,
}

impl GracefulDegradation {
    /// Create a new tracker with the given memory ceiling.
    pub fn new(max_memory_bytes: u64) -> Self {
        Self {
            max_memory_bytes,
            current_memory_bytes: 0,
            degradation_level: DegradationLevel::Normal,
            log_messages: Vec::new(),
        }
    }

    /// Update current memory usage and auto-adjust the degradation level.
    pub fn update_memory(&mut self, current_bytes: u64) {
        self.current_memory_bytes = current_bytes;

        let utilization = if self.max_memory_bytes > 0 {
            current_bytes as f64 / self.max_memory_bytes as f64
        } else {
            1.0
        };

        let new_level = if utilization > 0.95 {
            DegradationLevel::MinimalMode
        } else if utilization > 0.90 {
            DegradationLevel::DisableEffects
        } else if utilization > 0.80 {
            DegradationLevel::ReduceQuality
        } else {
            DegradationLevel::Normal
        };

        if new_level != self.degradation_level {
            self.log_messages.push(format!(
                "Degradation: {:?} -> {:?} (utilization {:.1}%)",
                self.degradation_level,
                new_level,
                utilization * 100.0,
            ));
            self.degradation_level = new_level;
        }
    }

    /// Return the list of config changes implied by the current level.
    pub fn recommended_changes(&self) -> Vec<ConfigChange> {
        match self.degradation_level {
            DegradationLevel::Normal => Vec::new(),
            DegradationLevel::ReduceQuality => vec![
                ConfigChange {
                    field: "render_scale".into(),
                    old_value: "0.75".into(),
                    new_value: "0.5".into(),
                    reason: "Memory pressure >80% — reduce internal resolution".into(),
                },
                ConfigChange {
                    field: "post_process.dof_enabled".into(),
                    old_value: "true".into(),
                    new_value: "false".into(),
                    reason: "Memory pressure >80% — disable depth of field".into(),
                },
            ],
            DegradationLevel::DisableEffects => vec![
                ConfigChange {
                    field: "render_scale".into(),
                    old_value: "0.75".into(),
                    new_value: "0.5".into(),
                    reason: "Memory pressure >90% — reduce internal resolution".into(),
                },
                ConfigChange {
                    field: "volumetrics.enabled".into(),
                    old_value: "true".into(),
                    new_value: "false".into(),
                    reason: "Memory pressure >90% — disable volumetrics".into(),
                },
                ConfigChange {
                    field: "gi.enabled".into(),
                    old_value: "true".into(),
                    new_value: "false".into(),
                    reason: "Memory pressure >90% — disable global illumination".into(),
                },
                ConfigChange {
                    field: "post_process.bloom_enabled".into(),
                    old_value: "true".into(),
                    new_value: "false".into(),
                    reason: "Memory pressure >90% — disable bloom".into(),
                },
            ],
            DegradationLevel::MinimalMode => vec![
                ConfigChange {
                    field: "render_scale".into(),
                    old_value: "0.75".into(),
                    new_value: "0.25".into(),
                    reason: "Memory critical >95% — minimal resolution".into(),
                },
                ConfigChange {
                    field: "ray_march.max_steps".into(),
                    old_value: "256".into(),
                    new_value: "64".into(),
                    reason: "Memory critical >95% — minimal ray march".into(),
                },
                ConfigChange {
                    field: "shading.shadows_enabled".into(),
                    old_value: "true".into(),
                    new_value: "false".into(),
                    reason: "Memory critical >95% — disable shadows".into(),
                },
            ],
        }
    }
}

// ── Chunk Boundary ──────────────────────────────────────────────────────────

/// Result of computing a blending region between two adjacent chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryFix {
    /// The two chunks involved.
    pub chunks_affected: [IVec3; 2],
    /// World-space minimum corner of the blend region.
    pub blend_region_min: glam::Vec3,
    /// World-space maximum corner of the blend region.
    pub blend_region_max: glam::Vec3,
    /// Number of voxels in the overlap zone per axis.
    pub overlap_voxels: u32,
}

/// Computes overlap / blend regions at chunk boundaries to eliminate seams.
#[derive(Debug, Clone)]
pub struct ChunkBoundaryFixer {
    /// Number of voxels to overlap per side at each boundary.
    pub overlap_voxels: u32,
    /// Width of the smoothstep blend in world units.
    pub blend_width: f32,
}

impl Default for ChunkBoundaryFixer {
    fn default() -> Self {
        Self {
            overlap_voxels: 2,
            blend_width: 0.5,
        }
    }
}

impl ChunkBoundaryFixer {
    /// Compute the boundary fix between two adjacent chunks.
    ///
    /// Chunks are 8m on a side. The blend region is placed at the interface
    /// between the two chunks, extending `blend_width` into each chunk.
    pub fn fix_boundary(&self, chunk_a_pos: IVec3, chunk_b_pos: IVec3) -> BoundaryFix {
        const CHUNK_SIZE: f32 = 8.0;

        let a_min = chunk_a_pos.as_vec3() * CHUNK_SIZE;
        let b_min = chunk_b_pos.as_vec3() * CHUNK_SIZE;

        // Find the interface: midpoint of the two chunk origins shifted by half-chunk.
        let a_max = a_min + glam::Vec3::splat(CHUNK_SIZE);
        let b_max = b_min + glam::Vec3::splat(CHUNK_SIZE);

        // Blend region is the overlap zone shrunk by blend_width on the shared face.
        let blend_min = a_min.max(b_min) - glam::Vec3::splat(self.blend_width);
        let blend_max = a_max.min(b_max) + glam::Vec3::splat(self.blend_width);

        BoundaryFix {
            chunks_affected: [chunk_a_pos, chunk_b_pos],
            blend_region_min: blend_min,
            blend_region_max: blend_max,
            overlap_voxels: self.overlap_voxels,
        }
    }
}

// ── LOD Transition ──────────────────────────────────────────────────────────

/// Blends between LOD tiers to prevent visible popping.
#[derive(Debug, Clone)]
pub struct LodTransitionBlender {
    /// Width of the transition zone in world units.
    pub transition_width: f32,
    /// Size of the dither pattern (NxN).
    pub dither_pattern_size: u32,
}

impl Default for LodTransitionBlender {
    fn default() -> Self {
        Self {
            transition_width: 1.0,
            dither_pattern_size: 4,
        }
    }
}

impl LodTransitionBlender {
    /// Compute the blend factor for a given distance to the LOD boundary.
    ///
    /// Returns 0.0 when fully on the near side, 1.0 on the far side, with a
    /// smooth Hermite (smoothstep) transition across `transition_width`.
    pub fn blend_factor(&self, distance_to_boundary: f32) -> f32 {
        if self.transition_width <= 0.0 {
            return if distance_to_boundary >= 0.0 { 1.0 } else { 0.0 };
        }
        let half = self.transition_width * 0.5;
        let t = (distance_to_boundary + half) / self.transition_width;
        let t = t.clamp(0.0, 1.0);
        // smoothstep: 3t² - 2t³
        t * t * (3.0 - 2.0 * t)
    }
}

// ── Window Resize ───────────────────────────────────────────────────────────

/// Describes what needs to be recreated after a window resize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResizeAction {
    /// New display resolution `[width, height]`.
    pub new_display_res: [u32; 2],
    /// New internal render resolution `[width, height]`.
    pub new_internal_res: [u32; 2],
    /// Whether the G-buffer textures must be recreated.
    pub recreate_gbuffer: bool,
    /// Whether upscale targets must be recreated.
    pub recreate_upscale: bool,
}

/// Handles window resize events, clamping to minimum dimensions and computing
/// the new internal render resolution.
#[derive(Debug, Clone)]
pub struct WindowResizeHandler {
    /// Minimum allowed display width.
    pub min_width: u32,
    /// Minimum allowed display height.
    pub min_height: u32,
    /// Current display width.
    pub current_width: u32,
    /// Current display height.
    pub current_height: u32,
    /// Whether a resize is pending processing.
    pub pending_resize: bool,
    /// Render scale factor (0.0–1.0) for computing internal resolution.
    render_scale: f32,
}

impl WindowResizeHandler {
    /// Create a handler with the given initial display size.
    ///
    /// Minimum dimensions default to 320×240. Render scale defaults to 0.75.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            min_width: 320,
            min_height: 240,
            current_width: width.max(320),
            current_height: height.max(240),
            pending_resize: false,
            render_scale: 0.75,
        }
    }

    /// Set the render scale used to compute internal resolution.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.1, 1.0);
    }

    /// Process a resize event and return the actions needed.
    pub fn handle_resize(&mut self, new_width: u32, new_height: u32) -> ResizeAction {
        let clamped_w = new_width.max(self.min_width);
        let clamped_h = new_height.max(self.min_height);

        let internal_w = ((clamped_w as f32 * self.render_scale).round() as u32).max(self.min_width);
        let internal_h = ((clamped_h as f32 * self.render_scale).round() as u32).max(self.min_height);

        let size_changed = clamped_w != self.current_width || clamped_h != self.current_height;

        self.current_width = clamped_w;
        self.current_height = clamped_h;
        self.pending_resize = false;

        ResizeAction {
            new_display_res: [clamped_w, clamped_h],
            new_internal_res: [internal_w, internal_h],
            recreate_gbuffer: size_changed,
            recreate_upscale: size_changed,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- GracefulDegradation --------------------------------------------------

    #[test]
    fn degradation_starts_normal() {
        let gd = GracefulDegradation::new(1_000_000);
        assert_eq!(gd.degradation_level, DegradationLevel::Normal);
        assert!(gd.log_messages.is_empty());
    }

    #[test]
    fn degradation_normal_under_80_percent() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(799_999);
        assert_eq!(gd.degradation_level, DegradationLevel::Normal);
    }

    #[test]
    fn degradation_reduce_quality_at_80_percent() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(850_000);
        assert_eq!(gd.degradation_level, DegradationLevel::ReduceQuality);
        assert_eq!(gd.log_messages.len(), 1);
    }

    #[test]
    fn degradation_disable_effects_at_90_percent() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(920_000);
        assert_eq!(gd.degradation_level, DegradationLevel::DisableEffects);
    }

    #[test]
    fn degradation_minimal_mode_above_95_percent() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(960_000);
        assert_eq!(gd.degradation_level, DegradationLevel::MinimalMode);
    }

    #[test]
    fn degradation_recovers_to_normal() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(960_000);
        assert_eq!(gd.degradation_level, DegradationLevel::MinimalMode);
        gd.update_memory(500_000);
        assert_eq!(gd.degradation_level, DegradationLevel::Normal);
        assert_eq!(gd.log_messages.len(), 2);
    }

    #[test]
    fn degradation_no_log_when_level_unchanged() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(100_000);
        gd.update_memory(200_000);
        assert!(gd.log_messages.is_empty());
    }

    #[test]
    fn recommended_changes_normal_empty() {
        let gd = GracefulDegradation::new(1_000_000);
        assert!(gd.recommended_changes().is_empty());
    }

    #[test]
    fn recommended_changes_reduce_quality() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(850_000);
        let changes = gd.recommended_changes();
        assert_eq!(changes.len(), 2);
        assert!(changes.iter().any(|c| c.field == "render_scale"));
        assert!(changes.iter().any(|c| c.field == "post_process.dof_enabled"));
    }

    #[test]
    fn recommended_changes_disable_effects() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(920_000);
        let changes = gd.recommended_changes();
        assert_eq!(changes.len(), 4);
        assert!(changes.iter().any(|c| c.field == "volumetrics.enabled"));
        assert!(changes.iter().any(|c| c.field == "gi.enabled"));
    }

    #[test]
    fn recommended_changes_minimal_mode() {
        let mut gd = GracefulDegradation::new(1_000_000);
        gd.update_memory(960_000);
        let changes = gd.recommended_changes();
        assert_eq!(changes.len(), 3);
        assert!(changes.iter().any(|c| c.field == "shading.shadows_enabled"));
        assert!(changes.iter().any(|c| c.field == "ray_march.max_steps"));
    }

    // -- ChunkBoundaryFixer ---------------------------------------------------

    #[test]
    fn chunk_boundary_fix_adjacent_x() {
        let fixer = ChunkBoundaryFixer::default();
        let fix = fixer.fix_boundary(IVec3::new(0, 0, 0), IVec3::new(1, 0, 0));
        assert_eq!(fix.chunks_affected, [IVec3::ZERO, IVec3::new(1, 0, 0)]);
        assert_eq!(fix.overlap_voxels, 2);
        // Blend region should span from the shared face minus blend_width.
        assert!(fix.blend_region_max.x > fix.blend_region_min.x);
    }

    #[test]
    fn chunk_boundary_fix_custom_overlap() {
        let fixer = ChunkBoundaryFixer {
            overlap_voxels: 4,
            blend_width: 1.0,
        };
        let fix = fixer.fix_boundary(IVec3::new(0, 0, 0), IVec3::new(0, 1, 0));
        assert_eq!(fix.overlap_voxels, 4);
        assert!(fix.blend_region_max.y > fix.blend_region_min.y);
    }

    #[test]
    fn chunk_boundary_default() {
        let fixer = ChunkBoundaryFixer::default();
        assert_eq!(fixer.overlap_voxels, 2);
        assert!((fixer.blend_width - 0.5).abs() < f32::EPSILON);
    }

    // -- LodTransitionBlender -------------------------------------------------

    #[test]
    fn lod_blend_factor_far_negative() {
        let blender = LodTransitionBlender::default();
        let f = blender.blend_factor(-10.0);
        assert!((f - 0.0).abs() < 1e-5);
    }

    #[test]
    fn lod_blend_factor_far_positive() {
        let blender = LodTransitionBlender::default();
        let f = blender.blend_factor(10.0);
        assert!((f - 1.0).abs() < 1e-5);
    }

    #[test]
    fn lod_blend_factor_at_boundary() {
        let blender = LodTransitionBlender::default();
        let f = blender.blend_factor(0.0);
        // Smoothstep at t=0.5 → 0.5
        assert!((f - 0.5).abs() < 1e-5);
    }

    #[test]
    fn lod_blend_factor_smoothstep_monotonic() {
        let blender = LodTransitionBlender::default();
        let mut prev = 0.0_f32;
        for i in 0..=20 {
            let d = -1.0 + (i as f32) * 0.1;
            let f = blender.blend_factor(d);
            assert!(f >= prev - 1e-7, "smoothstep should be monotonically non-decreasing");
            prev = f;
        }
    }

    #[test]
    fn lod_blend_zero_width() {
        let blender = LodTransitionBlender {
            transition_width: 0.0,
            dither_pattern_size: 4,
        };
        assert!((blender.blend_factor(-0.1) - 0.0).abs() < 1e-5);
        assert!((blender.blend_factor(0.0) - 1.0).abs() < 1e-5);
        assert!((blender.blend_factor(0.5) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn lod_default_values() {
        let blender = LodTransitionBlender::default();
        assert!((blender.transition_width - 1.0).abs() < f32::EPSILON);
        assert_eq!(blender.dither_pattern_size, 4);
    }

    // -- WindowResizeHandler --------------------------------------------------

    #[test]
    fn window_resize_handler_new() {
        let h = WindowResizeHandler::new(1280, 720);
        assert_eq!(h.current_width, 1280);
        assert_eq!(h.current_height, 720);
        assert!(!h.pending_resize);
    }

    #[test]
    fn window_resize_handler_clamps_minimum() {
        let h = WindowResizeHandler::new(100, 100);
        assert_eq!(h.current_width, 320);
        assert_eq!(h.current_height, 240);
    }

    #[test]
    fn window_resize_computes_internal_res() {
        let mut h = WindowResizeHandler::new(1920, 1080);
        h.set_render_scale(0.5);
        let action = h.handle_resize(1920, 1080);
        assert_eq!(action.new_display_res, [1920, 1080]);
        assert_eq!(action.new_internal_res, [960, 540]);
    }

    #[test]
    fn window_resize_clamps_to_min() {
        let mut h = WindowResizeHandler::new(1280, 720);
        let action = h.handle_resize(100, 50);
        assert_eq!(action.new_display_res, [320, 240]);
        // Internal also clamped to min
        assert!(action.new_internal_res[0] >= 320);
        assert!(action.new_internal_res[1] >= 240);
    }

    #[test]
    fn window_resize_recreate_on_change() {
        let mut h = WindowResizeHandler::new(1280, 720);
        let action = h.handle_resize(1920, 1080);
        assert!(action.recreate_gbuffer);
        assert!(action.recreate_upscale);
    }

    #[test]
    fn window_resize_no_recreate_same_size() {
        let mut h = WindowResizeHandler::new(1280, 720);
        let action = h.handle_resize(1280, 720);
        assert!(!action.recreate_gbuffer);
        assert!(!action.recreate_upscale);
    }
}
