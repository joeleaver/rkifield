//! Engine rendering state — stubbed pending v2 rewrite.
//!
//! In v2, this will use object-centric rendering with per-object brick maps
//! and BVH acceleration. Will be rewritten starting in Phase 5.

/// Display (output) resolution width.
pub const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height.
pub const DISPLAY_HEIGHT: u32 = 720;

/// Engine rendering state — placeholder for v2 rewrite.
///
/// In v2, this will hold the GpuScene, BVH, object metadata buffers,
/// and all render passes. Currently a minimal stub.
pub struct EngineState {
    _private: (),
}
