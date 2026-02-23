//! Animated character assembly — stub pending v2 rewrite.
//!
//! In v2, an AnimatedCharacter is a SceneObject with bones as child SceneNodes.
//! Animation = updating bone local transforms. No segments, no joint regions.
//! This will be rewritten in Phase 10.

use glam::{Mat4, Vec3};
use rkf_core::aabb::Aabb;

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
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn transform_aabb_identity_returns_same() {
        let aabb = Aabb::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0));
        let result = transform_aabb(&aabb, &Mat4::IDENTITY);
        assert!((result.min - aabb.min).length() < 1e-5);
        assert!((result.max - aabb.max).length() < 1e-5);
    }

    #[test]
    fn transform_aabb_translation_shifts() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let result = transform_aabb(&aabb, &mat);
        assert!((result.min - Vec3::new(10.0, 20.0, 30.0)).length() < 1e-5);
        assert!((result.max - Vec3::new(11.0, 21.0, 31.0)).length() < 1e-5);
    }

    #[test]
    fn transform_aabb_90deg_rotation_swaps_axes() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mat = Mat4::from_rotation_z(FRAC_PI_2);
        let result = transform_aabb(&aabb, &mat);
        assert!((result.min.x - (-1.0)).abs() < 1e-4);
        assert!(result.max.x.abs() < 1e-4);
        assert!(result.min.y.abs() < 1e-4);
        assert!((result.max.y - 1.0).abs() < 1e-4);
    }
}
