//! Segment and joint region data structures for skeletal animation.
//!
//! Characters are decomposed into rigid body **segments** (each attached to one
//! bone) and **joint regions** (small blending volumes at articulation points).
//!
//! - **Segments** store rest-pose SDF bricks. The ray marcher transforms rays by
//!   the inverse bone matrix to evaluate the rest-pose SDF.
//! - **Joint regions** are rebaked each frame via GPU compute using smooth-min
//!   blending from two adjacent segments. The ray marcher evaluates them directly
//!   in world space.

use rkf_core::aabb::Aabb;

/// A rigid body segment of an animated character.
///
/// Each segment is attached to one bone and stores a rest-pose SDF
/// in a contiguous range of brick pool slots.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Index of the bone this segment is attached to.
    pub bone_index: u32,
    /// First brick pool slot containing the rest-pose SDF.
    /// The full range is `brick_start..brick_start + brick_count`.
    pub brick_start: u32,
    /// Number of bricks in this segment.
    pub brick_count: u32,
    /// Rest-pose bounding box (local space, before bone transform).
    pub rest_aabb: Aabb,
}

/// A joint blending region between two adjacent segments.
///
/// Joint regions are rebaked each frame using smooth-min blending
/// from the two adjacent segments. Typical radius is 10-20 cm.
#[derive(Debug, Clone)]
pub struct JointRegion {
    /// Index of segment A (e.g., upper arm).
    pub segment_a: u32,
    /// Index of segment B (e.g., lower arm).
    pub segment_b: u32,
    /// Bone index this joint is associated with (typically the child bone).
    pub bone_index: u32,
    /// First brick pool slot for the joint's rebaked voxels.
    pub brick_start: u32,
    /// Number of bricks in this joint region.
    pub brick_count: u32,
    /// Rest-pose bounding box of the joint region (local space).
    pub rest_aabb: Aabb,
    /// Smooth-min blend radius parameter `k`.
    /// Small (~0.02) for mechanical joints, large (~0.08) for organic.
    pub blend_k: f32,
}

/// Flags for spatial index entries related to animation.
pub mod entry_flags {
    /// Spatial index entry is a rigid segment (transform ray by inverse bone matrix).
    pub const RIGID_SEGMENT: u8 = 1;
    /// Spatial index entry is a rebaked joint (evaluate directly in world space).
    pub const REBAKED_JOINT: u8 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn segment_fields_accessible() {
        let seg = Segment {
            bone_index: 3,
            brick_start: 100,
            brick_count: 12,
            rest_aabb: Aabb::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5)),
        };
        assert_eq!(seg.bone_index, 3);
        assert_eq!(seg.brick_start, 100);
        assert_eq!(seg.brick_count, 12);
        assert_eq!(seg.rest_aabb.min, Vec3::new(-0.5, -0.5, -0.5));
        assert_eq!(seg.rest_aabb.max, Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn joint_region_fields_accessible() {
        let joint = JointRegion {
            segment_a: 0,
            segment_b: 1,
            bone_index: 2,
            brick_start: 200,
            brick_count: 8,
            rest_aabb: Aabb::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1)),
            blend_k: 0.05,
        };
        assert_eq!(joint.segment_a, 0);
        assert_eq!(joint.segment_b, 1);
        assert_eq!(joint.bone_index, 2);
        assert_eq!(joint.brick_start, 200);
        assert_eq!(joint.brick_count, 8);
        assert!((joint.blend_k - 0.05).abs() < 1e-6);
    }

    #[test]
    fn entry_flags_are_distinct() {
        assert_ne!(entry_flags::RIGID_SEGMENT, entry_flags::REBAKED_JOINT);
        assert_eq!(entry_flags::RIGID_SEGMENT, 1);
        assert_eq!(entry_flags::REBAKED_JOINT, 2);
    }

    #[test]
    fn segment_clone() {
        let seg = Segment {
            bone_index: 1,
            brick_start: 50,
            brick_count: 6,
            rest_aabb: Aabb::new(Vec3::ZERO, Vec3::ONE),
        };
        let cloned = seg.clone();
        assert_eq!(cloned.bone_index, seg.bone_index);
        assert_eq!(cloned.brick_start, seg.brick_start);
        assert_eq!(cloned.brick_count, seg.brick_count);
    }

    #[test]
    fn joint_region_clone() {
        let joint = JointRegion {
            segment_a: 0,
            segment_b: 1,
            bone_index: 1,
            brick_start: 10,
            brick_count: 4,
            rest_aabb: Aabb::new(Vec3::splat(-0.2), Vec3::splat(0.2)),
            blend_k: 0.08,
        };
        let cloned = joint.clone();
        assert_eq!(cloned.segment_a, joint.segment_a);
        assert_eq!(cloned.blend_k, joint.blend_k);
    }
}
