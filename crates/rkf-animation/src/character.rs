//! Animated character assembly: skeleton + segments + joints.
//!
//! An [`AnimatedCharacter`] combines a skeleton hierarchy with segment and joint
//! region definitions. Each frame, [`update_pose`](AnimatedCharacter::update_pose)
//! evaluates an animation clip to produce bone matrices, which are then used to
//! compute world-space AABBs and inverse transforms for the ray marcher.

use glam::{Mat4, Vec3};
use rkf_core::aabb::Aabb;

use crate::clip::AnimationClip;
use crate::segment::{JointRegion, Segment};
use crate::skeleton::Skeleton;

/// An animated character: skeleton + segments + joints.
#[derive(Debug)]
pub struct AnimatedCharacter {
    /// The skeleton hierarchy.
    pub skeleton: Skeleton,
    /// Rigid body segments.
    pub segments: Vec<Segment>,
    /// Joint blending regions.
    pub joints: Vec<JointRegion>,
    /// Current bone matrices (updated each frame via evaluate).
    bone_matrices: Vec<Mat4>,
}

impl AnimatedCharacter {
    /// Create a new animated character.
    pub fn new(
        skeleton: Skeleton,
        segments: Vec<Segment>,
        joints: Vec<JointRegion>,
    ) -> Self {
        let bone_count = skeleton.bone_count();
        Self {
            skeleton,
            segments,
            joints,
            bone_matrices: vec![Mat4::IDENTITY; bone_count],
        }
    }

    /// Get the current bone matrices.
    pub fn bone_matrices(&self) -> &[Mat4] {
        &self.bone_matrices
    }

    /// Update bone matrices by evaluating the animation clip at the given time.
    pub fn update_pose(&mut self, clip: &AnimationClip, time: f32) {
        self.bone_matrices = self.skeleton.evaluate(clip, time);
    }

    /// Compute the world-space AABB for a segment given current bone matrices.
    pub fn segment_world_aabb(&self, segment_index: usize) -> Aabb {
        let seg = &self.segments[segment_index];
        let bone_mat = self.bone_matrices[seg.bone_index as usize];
        transform_aabb(&seg.rest_aabb, &bone_mat)
    }

    /// Compute the world-space AABB for a joint region.
    ///
    /// Uses the joint's rest AABB transformed by its associated bone matrix.
    pub fn joint_world_aabb(&self, joint_index: usize) -> Aabb {
        let joint = &self.joints[joint_index];
        let bone_mat = self.bone_matrices[joint.bone_index as usize];
        transform_aabb(&joint.rest_aabb, &bone_mat)
    }

    /// Get the inverse bone matrix for a segment (world -> segment local space).
    ///
    /// Used by the ray marcher to transform rays into segment rest-pose space.
    pub fn segment_inverse_bone(&self, segment_index: usize) -> Mat4 {
        let seg = &self.segments[segment_index];
        self.bone_matrices[seg.bone_index as usize].inverse()
    }

    /// Get all world-space AABBs for segments and joints (for spatial index updates).
    ///
    /// Returns `(segment_aabbs, joint_aabbs)`.
    pub fn all_world_aabbs(&self) -> (Vec<Aabb>, Vec<Aabb>) {
        let seg_aabbs = (0..self.segments.len())
            .map(|i| self.segment_world_aabb(i))
            .collect();
        let joint_aabbs = (0..self.joints.len())
            .map(|i| self.joint_world_aabb(i))
            .collect();
        (seg_aabbs, joint_aabbs)
    }
}

/// Transform an AABB by a matrix, producing a new axis-aligned bounding box.
///
/// This expands the AABB to contain all 8 transformed corners.
pub fn transform_aabb(aabb: &Aabb, mat: &Mat4) -> Aabb {
    let corners = [
        Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.min.z),
        Vec3::new(aabb.min.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.min.y, aabb.max.z),
        Vec3::new(aabb.min.x, aabb.max.y, aabb.max.z),
        Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
    ];

    let mut new_min = Vec3::splat(f32::MAX);
    let mut new_max = Vec3::splat(f32::MIN);
    for c in &corners {
        let t = mat.transform_point3(*c);
        new_min = new_min.min(t);
        new_max = new_max.max(t);
    }
    Aabb::new(new_min, new_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{AnimationClip, BoneChannel, Keyframe};
    use crate::skeleton::{Bone, Skeleton};
    use glam::Quat;
    use std::f32::consts::FRAC_PI_2;

    /// Helper: create a bone with translation-only bind pose.
    fn bone(name: &str, translation: Vec3) -> Bone {
        let bind = Mat4::from_translation(translation);
        Bone {
            name: name.to_string(),
            bind_transform: bind,
            inverse_bind: bind.inverse(),
        }
    }

    /// Helper: build a 3-bone skeleton (root, upper_arm, lower_arm) with 2 segments and 1 joint.
    fn make_test_character() -> AnimatedCharacter {
        let skeleton = Skeleton::new(
            vec![
                bone("root", Vec3::ZERO),
                bone("upper_arm", Vec3::new(0.0, 1.0, 0.0)),
                bone("lower_arm", Vec3::new(0.0, 0.5, 0.0)),
            ],
            vec![-1, 0, 1],
        )
        .unwrap();

        let segments = vec![
            Segment {
                bone_index: 1,
                brick_start: 0,
                brick_count: 10,
                rest_aabb: Aabb::new(
                    Vec3::new(-0.2, -0.5, -0.2),
                    Vec3::new(0.2, 0.5, 0.2),
                ),
            },
            Segment {
                bone_index: 2,
                brick_start: 10,
                brick_count: 8,
                rest_aabb: Aabb::new(
                    Vec3::new(-0.15, -0.25, -0.15),
                    Vec3::new(0.15, 0.25, 0.15),
                ),
            },
        ];

        let joints = vec![JointRegion {
            segment_a: 0,
            segment_b: 1,
            bone_index: 2,
            brick_start: 18,
            brick_count: 4,
            rest_aabb: Aabb::new(
                Vec3::new(-0.1, -0.1, -0.1),
                Vec3::new(0.1, 0.1, 0.1),
            ),
            blend_k: 0.05,
        }];

        AnimatedCharacter::new(skeleton, segments, joints)
    }

    /// Helper: create a bind-pose clip for the test character (identity transforms).
    fn bind_pose_clip() -> AnimationClip {
        AnimationClip::new(
            "bind".to_string(),
            1.0,
            vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::ZERO,
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
                BoneChannel {
                    bone_index: 1,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::new(0.0, 1.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
                BoneChannel {
                    bone_index: 2,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::new(0.0, 0.5, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
            ],
        )
    }

    #[test]
    fn create_character_with_skeleton_segments_joints() {
        let ch = make_test_character();
        assert_eq!(ch.skeleton.bone_count(), 3);
        assert_eq!(ch.segments.len(), 2);
        assert_eq!(ch.joints.len(), 1);
        assert_eq!(ch.bone_matrices().len(), 3);
    }

    #[test]
    fn update_pose_updates_bone_matrices() {
        let mut ch = make_test_character();

        // Before update: all identity.
        for m in ch.bone_matrices() {
            let diff = (*m - Mat4::IDENTITY).abs().to_cols_array();
            assert!(diff.iter().cloned().fold(0.0f32, f32::max) < 1e-5);
        }

        // Animate root to rotate 90 deg around Z.
        let clip = AnimationClip::new(
            "rotate".to_string(),
            1.0,
            vec![BoneChannel {
                bone_index: 0,
                keyframes: vec![Keyframe {
                    time: 0.0,
                    position: Vec3::ZERO,
                    rotation: Quat::from_rotation_z(FRAC_PI_2),
                    scale: Vec3::ONE,
                }],
            }],
        );
        ch.update_pose(&clip, 0.0);

        // Root bone matrix should no longer be identity.
        let root_mat = ch.bone_matrices()[0];
        let diff = (root_mat - Mat4::IDENTITY).abs().to_cols_array();
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_diff > 0.1, "root matrix should differ from identity after rotation");
    }

    #[test]
    fn segment_world_aabb_at_bind_pose_equals_rest_aabb() {
        let mut ch = make_test_character();
        let clip = bind_pose_clip();
        ch.update_pose(&clip, 0.0);

        // At bind pose, bone_matrix = world * inverse_bind = identity.
        // So segment_world_aabb should equal rest_aabb.
        let seg_aabb = ch.segment_world_aabb(0);
        let rest = &ch.segments[0].rest_aabb;
        assert!(
            (seg_aabb.min - rest.min).length() < 1e-4,
            "min mismatch: {:?} vs {:?}",
            seg_aabb.min,
            rest.min
        );
        assert!(
            (seg_aabb.max - rest.max).length() < 1e-4,
            "max mismatch: {:?} vs {:?}",
            seg_aabb.max,
            rest.max
        );
    }

    #[test]
    fn segment_world_aabb_with_rotation_produces_rotated_aabb() {
        let mut ch = make_test_character();

        // Rotate root 90 degrees around Z.
        let clip = AnimationClip::new(
            "rotate".to_string(),
            1.0,
            vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::ZERO,
                        rotation: Quat::from_rotation_z(FRAC_PI_2),
                        scale: Vec3::ONE,
                    }],
                },
                BoneChannel {
                    bone_index: 1,
                    keyframes: vec![Keyframe {
                        time: 0.0,
                        position: Vec3::new(0.0, 1.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    }],
                },
            ],
        );
        ch.update_pose(&clip, 0.0);

        let seg_aabb = ch.segment_world_aabb(0);
        let rest = &ch.segments[0].rest_aabb;

        // After 90-degree Z rotation, the AABB should differ from rest
        // because X and Y extents swap.
        let rest_size = rest.size();
        let rotated_size = seg_aabb.size();

        // rest is 0.4 x 1.0 x 0.4; after 90deg Z rotation: 1.0 x 0.4 x 0.4
        assert!(
            (rotated_size.x - rest_size.y).abs() < 1e-3,
            "expected rotated X ~ rest Y: {} vs {}",
            rotated_size.x,
            rest_size.y
        );
        assert!(
            (rotated_size.y - rest_size.x).abs() < 1e-3,
            "expected rotated Y ~ rest X: {} vs {}",
            rotated_size.y,
            rest_size.x
        );
    }

    #[test]
    fn segment_inverse_bone_inverts_correctly() {
        let mut ch = make_test_character();
        let clip = bind_pose_clip();
        ch.update_pose(&clip, 0.0);

        let bone_mat = ch.bone_matrices()[ch.segments[0].bone_index as usize];
        let inv = ch.segment_inverse_bone(0);
        let product = bone_mat * inv;

        let diff = (product - Mat4::IDENTITY).abs().to_cols_array();
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "bone * inverse should be identity, max diff = {max_diff}"
        );
    }

    #[test]
    fn joint_world_aabb_produces_valid_aabb() {
        let mut ch = make_test_character();
        let clip = bind_pose_clip();
        ch.update_pose(&clip, 0.0);

        let joint_aabb = ch.joint_world_aabb(0);
        // Should be a valid AABB (min <= max on all axes).
        assert!(joint_aabb.min.x <= joint_aabb.max.x);
        assert!(joint_aabb.min.y <= joint_aabb.max.y);
        assert!(joint_aabb.min.z <= joint_aabb.max.z);
        // Volume should be positive.
        assert!(joint_aabb.volume() > 0.0);
    }

    #[test]
    fn all_world_aabbs_returns_correct_counts() {
        let mut ch = make_test_character();
        let clip = bind_pose_clip();
        ch.update_pose(&clip, 0.0);

        let (seg_aabbs, joint_aabbs) = ch.all_world_aabbs();
        assert_eq!(seg_aabbs.len(), 2);
        assert_eq!(joint_aabbs.len(), 1);
    }

    #[test]
    fn transform_aabb_identity_returns_same() {
        let aabb = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = transform_aabb(&aabb, &Mat4::IDENTITY);
        assert!(
            (result.min - aabb.min).length() < 1e-5,
            "min: {:?} vs {:?}",
            result.min,
            aabb.min
        );
        assert!(
            (result.max - aabb.max).length() < 1e-5,
            "max: {:?} vs {:?}",
            result.max,
            aabb.max
        );
    }

    #[test]
    fn transform_aabb_translation_shifts() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let result = transform_aabb(&aabb, &mat);
        assert!(
            (result.min - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-5,
            "min should be (10,20,30), got {:?}",
            result.min
        );
        assert!(
            (result.max - Vec3::new(11.0, 21.0, 31.0)).length() < 1e-5,
            "max should be (11,21,31), got {:?}",
            result.max
        );
    }

    #[test]
    fn transform_aabb_90deg_rotation_swaps_axes() {
        // Unit cube [0,1]^3, rotated 90 degrees around Z.
        // X -> Y, Y -> -X.
        // Corner (0,0,0) -> (0,0,0)
        // Corner (1,0,0) -> (0,1,0)
        // Corner (0,1,0) -> (-1,0,0)
        // Corner (1,1,0) -> (-1,1,0)
        // So min.x = -1, max.x = 0; min.y = 0, max.y = 1; z unchanged.
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_rotation_z(FRAC_PI_2);
        let result = transform_aabb(&aabb, &mat);

        assert!(
            (result.min.x - (-1.0)).abs() < 1e-4,
            "min.x should be -1, got {}",
            result.min.x
        );
        assert!(
            result.max.x.abs() < 1e-4,
            "max.x should be ~0, got {}",
            result.max.x
        );
        assert!(
            result.min.y.abs() < 1e-4,
            "min.y should be ~0, got {}",
            result.min.y
        );
        assert!(
            (result.max.y - 1.0).abs() < 1e-4,
            "max.y should be 1, got {}",
            result.max.y
        );
        // Z unchanged: [0, 1].
        assert!(
            result.min.z.abs() < 1e-4,
            "min.z should be ~0, got {}",
            result.min.z
        );
        assert!(
            (result.max.z - 1.0).abs() < 1e-4,
            "max.z should be 1, got {}",
            result.max.z
        );
    }
}
