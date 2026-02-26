//! Transform baking — compute world transforms from parent-local chains.
//!
//! Objects in a [`Scene`] store position/rotation/scale relative to their
//! parent. The bake system walks the hierarchy in topological order
//! (parents before children) and accumulates world-space transforms.
//!
//! The resulting [`WorldTransform`] values are transient — they are computed
//! per-frame from the authoritative local transforms and never stored back
//! into the scene.

use std::collections::HashMap;

use glam::{Quat, Vec3};

use crate::scene::SceneObject;

/// A baked world-space transform for a single object.
#[derive(Debug, Clone, Copy)]
pub struct WorldTransform {
    /// World-space position.
    pub position: Vec3,
    /// World-space rotation.
    pub rotation: Quat,
    /// World-space per-axis scale (component-wise product of ancestor scales).
    pub scale: Vec3,
}

impl Default for WorldTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Batch-bake world transforms for all objects in the scene.
///
/// Processes objects in topological order (parents before children).
/// Returns a map from object ID → world transform.
///
/// Composition:
/// - `world_pos = parent_pos + parent_rot * (local_pos * parent_scale)`
/// - `world_rot = parent_rot * local_rot`
/// - `world_scale = parent_scale * local_scale` (component-wise)
pub fn bake_world_transforms(objects: &[SceneObject]) -> HashMap<u32, WorldTransform> {
    let mut result = HashMap::with_capacity(objects.len());

    // Build topological order via BFS from roots
    let mut order: Vec<u32> = Vec::with_capacity(objects.len());
    for obj in objects {
        if obj.parent_id.is_none() {
            order.push(obj.id);
        }
    }
    let mut i = 0;
    while i < order.len() {
        let current_id = order[i];
        for obj in objects {
            if obj.parent_id == Some(current_id) {
                order.push(obj.id);
            }
        }
        i += 1;
    }

    // Bake in topological order
    for &id in &order {
        let Some(obj) = objects.iter().find(|o| o.id == id) else {
            continue;
        };

        let parent_wt: WorldTransform = obj
            .parent_id
            .and_then(|pid| result.get(&pid))
            .copied()
            .unwrap_or_default();

        let world_pos =
            parent_wt.position + parent_wt.rotation * (obj.position * parent_wt.scale);
        let world_rot = parent_wt.rotation * obj.rotation;
        let world_scale = parent_wt.scale * obj.scale;

        result.insert(id, WorldTransform {
            position: world_pos,
            rotation: world_rot,
            scale: world_scale,
        });
    }

    result
}

/// Compute the world transform for a single object by walking up the parent chain.
///
/// Less efficient than [`bake_world_transforms`] for bulk queries, but useful
/// for single-object lookups (e.g. converting a gizmo delta to local space).
pub fn compute_world_transform(objects: &[SceneObject], obj_id: u32) -> WorldTransform {
    // Collect the chain from obj up to root
    let mut chain = Vec::new();
    let mut cursor = Some(obj_id);
    while let Some(id) = cursor {
        let Some(obj) = objects.iter().find(|o| o.id == id) else {
            break;
        };
        chain.push(obj);
        cursor = obj.parent_id;
    }

    // Compose from root (last) down to the object (first)
    let mut wt = WorldTransform::default();
    for obj in chain.iter().rev() {
        wt = WorldTransform {
            position: wt.position + wt.rotation * (obj.position * wt.scale),
            rotation: wt.rotation * obj.rotation,
            scale: wt.scale * obj.scale,
        };
    }

    wt
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aabb::Aabb;
    use crate::scene_node::SceneNode;
    use std::f32::consts::FRAC_PI_2;

    fn make_obj(id: u32, parent_id: Option<u32>, pos: Vec3, rot: Quat, scale: f32) -> SceneObject {
        SceneObject {
            id,
            name: format!("obj_{id}"),
            parent_id,
            position: pos,
            rotation: rot,
            scale: Vec3::splat(scale),
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        }
    }

    #[test]
    fn single_root_identity() {
        let objects = vec![make_obj(1, None, Vec3::ZERO, Quat::IDENTITY, 1.0)];
        let baked = bake_world_transforms(&objects);

        let wt = baked.get(&1).unwrap();
        assert!((wt.position - Vec3::ZERO).length() < 1e-6);
        assert!((wt.rotation - Quat::IDENTITY).length() < 1e-6);
        assert!((wt.scale - Vec3::ONE).length() < 1e-6);
    }

    #[test]
    fn single_root_with_transform() {
        let objects = vec![make_obj(
            1,
            None,
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(1.0),
            2.0,
        )];
        let baked = bake_world_transforms(&objects);

        let wt = baked.get(&1).unwrap();
        assert!((wt.position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert!((wt.rotation - Quat::from_rotation_y(1.0)).length() < 1e-5);
        assert!((wt.scale - Vec3::splat(2.0)).length() < 1e-6);
    }

    #[test]
    fn parent_child_translation() {
        let objects = vec![
            make_obj(1, None, Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, 1.0),
            make_obj(2, Some(1), Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY, 1.0),
        ];
        let baked = bake_world_transforms(&objects);

        let wt = baked.get(&2).unwrap();
        // child world pos = parent(10,0,0) + identity_rot * (5,0,0) * 1.0 = (15,0,0)
        assert!((wt.position - Vec3::new(15.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn parent_child_rotation() {
        // Parent rotated 90° around Y, child at local (1, 0, 0)
        let objects = vec![
            make_obj(1, None, Vec3::ZERO, Quat::from_rotation_y(FRAC_PI_2), 1.0),
            make_obj(2, Some(1), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 1.0),
        ];
        let baked = bake_world_transforms(&objects);

        let wt = baked.get(&2).unwrap();
        // After 90° Y rotation, local (1,0,0) → world (0,0,-1)
        assert!(wt.position.x.abs() < 1e-4, "x: {}", wt.position.x);
        assert!((wt.position.z - (-1.0)).abs() < 1e-4, "z: {}", wt.position.z);
    }

    #[test]
    fn parent_child_scale() {
        let objects = vec![
            make_obj(1, None, Vec3::ZERO, Quat::IDENTITY, 2.0),
            make_obj(2, Some(1), Vec3::new(3.0, 0.0, 0.0), Quat::IDENTITY, 0.5),
        ];
        let baked = bake_world_transforms(&objects);

        let wt_parent = baked.get(&1).unwrap();
        assert!((wt_parent.scale - Vec3::splat(2.0)).length() < 1e-6);

        let wt_child = baked.get(&2).unwrap();
        // child world pos = 0 + identity * (3,0,0) * 2.0 = (6,0,0)
        assert!((wt_child.position - Vec3::new(6.0, 0.0, 0.0)).length() < 1e-5);
        // child world scale = 2.0 * 0.5 = 1.0
        assert!((wt_child.scale - Vec3::ONE).length() < 1e-6);
    }

    #[test]
    fn three_level_hierarchy() {
        let objects = vec![
            make_obj(1, None, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 2.0),
            make_obj(2, Some(1), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 3.0),
            make_obj(3, Some(2), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.5),
        ];
        let baked = bake_world_transforms(&objects);

        // obj1: pos=(1,0,0), scale=2
        // obj2: pos = (1,0,0) + identity*(1,0,0)*2 = (3,0,0), scale = 6
        // obj3: pos = (3,0,0) + identity*(1,0,0)*6 = (9,0,0), scale = 3
        let wt1 = baked.get(&1).unwrap();
        let wt2 = baked.get(&2).unwrap();
        let wt3 = baked.get(&3).unwrap();

        assert!((wt1.position.x - 1.0).abs() < 1e-5);
        assert!((wt1.scale - Vec3::splat(2.0)).length() < 1e-5);
        assert!((wt2.position.x - 3.0).abs() < 1e-5);
        assert!((wt2.scale - Vec3::splat(6.0)).length() < 1e-5);
        assert!((wt3.position.x - 9.0).abs() < 1e-5);
        assert!((wt3.scale - Vec3::splat(3.0)).length() < 1e-5);
    }

    #[test]
    fn empty_objects() {
        let baked = bake_world_transforms(&[]);
        assert!(baked.is_empty());
    }

    #[test]
    fn compute_single_matches_batch() {
        let objects = vec![
            make_obj(1, None, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 2.0),
            make_obj(2, Some(1), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 3.0),
            make_obj(3, Some(2), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.5),
        ];
        let baked = bake_world_transforms(&objects);

        for &id in &[1, 2, 3] {
            let batch = baked.get(&id).unwrap();
            let single = compute_world_transform(&objects, id);
            assert!(
                (batch.position - single.position).length() < 1e-5,
                "id={id}: batch={:?} single={:?}",
                batch.position,
                single.position
            );
            assert!(
                (batch.scale - single.scale).length() < 1e-5,
                "id={id}: batch={:?} single={:?}",
                batch.scale,
                single.scale
            );
        }
    }

    #[test]
    fn rotation_and_scale_combined() {
        // Parent: 90° Y rotation, scale 2
        // Child at local (1, 0, 0) → world (0, 0, -2) (rotated + scaled)
        let objects = vec![
            make_obj(1, None, Vec3::ZERO, Quat::from_rotation_y(FRAC_PI_2), 2.0),
            make_obj(2, Some(1), Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 1.0),
        ];
        let baked = bake_world_transforms(&objects);

        let wt = baked.get(&2).unwrap();
        assert!(wt.position.x.abs() < 1e-4, "x: {}", wt.position.x);
        assert!((wt.position.z - (-2.0)).abs() < 1e-4, "z: {}", wt.position.z);
        assert!((wt.scale - Vec3::splat(2.0)).length() < 1e-5);
    }
}
