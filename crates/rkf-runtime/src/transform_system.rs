//! Transform hierarchy update system.
//!
//! Computes [`WorldTransform`] for all entities each frame in two passes:
//! 1. Root entities (no [`Parent`]) — camera-relative from [`Transform`]
//! 2. Child entities — parent's `WorldTransform` × local transform
//!
//! All `WorldTransform` matrices are camera-relative to avoid f32 precision loss.

use glam::{Mat4, Vec3};
use hecs::World;
use rkf_core::WorldPosition;

use crate::components::{Parent, Transform, WorldTransform};

/// Compute the camera-relative f32 offset from a WorldPosition.
///
/// Delegates to [`WorldPosition::relative_to`], which uses f64 arithmetic
/// internally to avoid catastrophic cancellation at large distances.
pub fn camera_relative_position(pos: &WorldPosition, camera: &WorldPosition) -> Vec3 {
    pos.relative_to(camera)
}

/// Build a local model matrix from Transform (without camera-relative position).
///
/// Returns rotation × scale as Mat4 (no translation — that is added separately
/// for camera-relative positioning).
pub fn local_matrix(transform: &Transform) -> Mat4 {
    Mat4::from_rotation_translation(transform.rotation, Vec3::ZERO)
        * Mat4::from_scale(Vec3::splat(transform.scale))
}

/// Build a camera-relative model matrix from Transform.
pub fn camera_relative_matrix(transform: &Transform, camera_pos: &WorldPosition) -> Mat4 {
    let pos = camera_relative_position(&transform.position, camera_pos);
    Mat4::from_rotation_translation(transform.rotation, pos)
        * Mat4::from_scale(Vec3::splat(transform.scale))
}

/// Update all WorldTransform components in the ECS world.
///
/// Two-pass approach:
/// 1. Root entities (no Parent component) — compute camera-relative matrix directly
/// 2. Child entities (have Parent) — multiply parent's WorldTransform by local transform
///
/// For depth > 2, call this function multiple times or extend with additional passes.
///
/// # Parameters
/// - `world`: the hecs World containing all entities
/// - `camera_pos`: the camera's WorldPosition for camera-relative transforms
pub fn update_transforms(world: &mut World, camera_pos: &WorldPosition) {
    // Pass 1: Root entities — no Parent component.
    // Collect into a vec to avoid borrow conflicts on `world`.
    let roots: Vec<_> = world
        .query::<&Transform>()
        .without::<&Parent>()
        .iter()
        .map(|(entity, transform)| {
            let matrix = camera_relative_matrix(transform, camera_pos);
            (entity, WorldTransform { matrix })
        })
        .collect();

    for (entity, wt) in roots {
        if let Ok(mut existing) = world.get::<&mut WorldTransform>(entity) {
            *existing = wt;
        }
    }

    // Pass 2: Child entities — have Parent component.
    // Collect (entity, transform clone, parent entity) to release the borrow.
    let children: Vec<_> = world
        .query::<(&Transform, &Parent)>()
        .iter()
        .map(|(entity, (transform, parent))| {
            (entity, transform.clone(), parent.entity, parent.bone_index)
        })
        .collect();

    for (entity, transform, parent_entity, _bone_index) in children {
        let parent_matrix = world
            .get::<&WorldTransform>(parent_entity)
            .map(|wt| wt.matrix)
            .unwrap_or(Mat4::IDENTITY);

        // The child's position is expressed in the parent's space (local offset).
        // We treat the child's WorldPosition as a local-space offset from the
        // parent origin, so we compute camera-relative displacement from world
        // origin (default) — this keeps the position in the same coordinate
        // space as the parent matrix.
        let local_pos = transform.position.relative_to(&WorldPosition::default());
        let local_mat = Mat4::from_rotation_translation(transform.rotation, local_pos)
            * Mat4::from_scale(Vec3::splat(transform.scale));

        // TODO: If bone_index is set, multiply by bone_matrices[bone_index]
        // (Phase 13+ — skeletal animation)

        let matrix = parent_matrix * local_mat;
        if let Ok(mut existing) = world.get::<&mut WorldTransform>(entity) {
            *existing = WorldTransform { matrix };
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};

    fn zero_pos() -> WorldPosition {
        WorldPosition::default()
    }

    fn make_pos(x: f32, y: f32, z: f32) -> WorldPosition {
        WorldPosition {
            chunk: IVec3::ZERO,
            local: Vec3::new(x, y, z),
        }
    }

    #[test]
    fn camera_relative_same_chunk() {
        let pos = make_pos(5.0, 3.0, 1.0);
        let cam = make_pos(2.0, 1.0, 0.0);
        let rel = camera_relative_position(&pos, &cam);
        assert!((rel.x - 3.0).abs() < 1e-5, "rel.x = {}", rel.x);
        assert!((rel.y - 2.0).abs() < 1e-5, "rel.y = {}", rel.y);
        assert!((rel.z - 1.0).abs() < 1e-5, "rel.z = {}", rel.z);
    }

    #[test]
    fn camera_relative_different_chunks() {
        let pos = WorldPosition {
            chunk: IVec3::new(1, 0, 0),
            local: Vec3::new(2.0, 0.0, 0.0),
        };
        let cam = WorldPosition {
            chunk: IVec3::new(0, 0, 0),
            local: Vec3::new(3.0, 0.0, 0.0),
        };
        let rel = camera_relative_position(&pos, &cam);
        // chunk_diff.x = 1 → 1*8.0 + (2.0 - 3.0) = 7.0
        assert!((rel.x - 7.0).abs() < 1e-5, "rel.x = {}", rel.x);
    }

    #[test]
    fn root_entity_transform() {
        let mut world = World::new();
        let t = Transform {
            position: make_pos(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
        };
        let e = world.spawn((t, WorldTransform::default()));

        update_transforms(&mut world, &zero_pos());

        let wt = world.get::<&WorldTransform>(e).unwrap();
        let pos = wt.matrix.col(3);
        assert!((pos.x - 10.0).abs() < 1e-5, "pos.x = {}", pos.x);
    }

    #[test]
    fn root_entity_camera_relative() {
        let mut world = World::new();
        let t = Transform {
            position: make_pos(10.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
        };
        let e = world.spawn((t, WorldTransform::default()));
        let cam = make_pos(3.0, 2.0, 0.0);

        update_transforms(&mut world, &cam);

        let wt = world.get::<&WorldTransform>(e).unwrap();
        let pos = wt.matrix.col(3);
        assert!((pos.x - 7.0).abs() < 1e-5, "pos.x = {}", pos.x);
        assert!((pos.y - 3.0).abs() < 1e-5, "pos.y = {}", pos.y);
    }

    #[test]
    fn child_inherits_parent() {
        let mut world = World::new();
        let parent_t = Transform {
            position: make_pos(10.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
        };
        let parent_e = world.spawn((parent_t, WorldTransform::default()));

        let child_t = Transform {
            position: make_pos(5.0, 0.0, 0.0), // local offset from parent
            rotation: Quat::IDENTITY,
            scale: 1.0,
        };
        let child_e = world.spawn((
            child_t,
            WorldTransform::default(),
            Parent {
                entity: parent_e,
                bone_index: None,
            },
        ));

        update_transforms(&mut world, &zero_pos());

        let child_wt = world.get::<&WorldTransform>(child_e).unwrap();
        let pos = child_wt.matrix.col(3);
        // Parent at 10 + child local 5 = 15
        assert!((pos.x - 15.0).abs() < 1e-5, "pos.x = {}", pos.x);
    }

    #[test]
    fn scale_propagates() {
        let mut world = World::new();
        let t = Transform {
            position: make_pos(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 2.0,
        };
        let e = world.spawn((t, WorldTransform::default()));

        update_transforms(&mut world, &zero_pos());

        let wt = world.get::<&WorldTransform>(e).unwrap();
        // Scale should appear in the matrix X-column length
        let sx = wt.matrix.col(0).length();
        assert!((sx - 2.0).abs() < 1e-5, "sx = {}", sx);
    }

    #[test]
    fn local_matrix_no_translation() {
        let t = Transform {
            position: make_pos(99.0, 99.0, 99.0), // should be ignored
            rotation: Quat::IDENTITY,
            scale: 3.0,
        };
        let m = local_matrix(&t);
        // Translation column should be zero
        assert!((m.col(3).x).abs() < 1e-5, "tx = {}", m.col(3).x);
        assert!((m.col(3).y).abs() < 1e-5, "ty = {}", m.col(3).y);
        assert!((m.col(3).z).abs() < 1e-5, "tz = {}", m.col(3).z);
        // Scale should be 3
        assert!((m.col(0).length() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn no_parent_does_not_write_without_world_transform() {
        // Entity with Transform but no WorldTransform — update should not panic.
        let mut world = World::new();
        let t = Transform::default();
        let _e = world.spawn((t,)); // no WorldTransform
        // Should complete without panic; entity is silently skipped in the write step.
        update_transforms(&mut world, &zero_pos());
    }

    #[test]
    fn rotation_preserved() {
        let mut world = World::new();
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let t = Transform {
            position: make_pos(0.0, 0.0, 0.0),
            rotation: rot,
            scale: 1.0,
        };
        let e = world.spawn((t, WorldTransform::default()));

        update_transforms(&mut world, &zero_pos());

        let wt = world.get::<&WorldTransform>(e).unwrap();
        // A 90° Y rotation maps +X → -Z. The X column of the matrix is (rot * X).
        let x_col = wt.matrix.col(0).truncate();
        // After 90° Y rotation: x_col ≈ (0, 0, -1)
        assert!(x_col.z.abs() > 0.9, "x_col = {:?}", x_col);
    }
}
