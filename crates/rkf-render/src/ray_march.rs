//! Ray march compute pass — stub pending v2 object-centric rewrite.

/// Default internal rendering resolution (width).
pub const INTERNAL_WIDTH: u32 = 960;
/// Default internal rendering resolution (height).
pub const INTERNAL_HEIGHT: u32 = 540;

/// Ray march compute pass — will be rewritten in Phase 5 for v2 object-centric
/// BVH + brick map evaluation.
pub struct RayMarchPass {
    _private: (),
}
