//! Transform flattening — depth-first traversal producing GPU-ready flat nodes.
//!
//! [`flatten_object`] walks a [`SceneObject`]'s node tree depth-first,
//! accumulating transforms and producing a flat [`Vec<FlatNode>`] suitable
//! for GPU upload. Each node carries a pre-computed inverse world matrix
//! and accumulated scale.
//!
//! # Camera-relative precision
//!
//! The root position is converted from [`WorldPosition`] to camera-relative
//! `f32` using `f64` arithmetic on the CPU. The GPU never sees absolute
//! world positions — only camera-relative offsets.

use glam::{Mat4, Vec3};

use crate::scene::SceneObject;
use crate::scene_node::{BlendMode, SceneNode, SdfSource};
use crate::world_position::WorldPosition;

/// A flattened node ready for GPU upload.
///
/// Produced by [`flatten_object`] via depth-first traversal.
#[derive(Debug, Clone)]
pub struct FlatNode {
    /// Pre-computed inverse world transform (camera-relative).
    /// Used in ray marching to transform ray into local space.
    pub inverse_world: Mat4,
    /// Product of all scales from root to this node.
    pub accumulated_scale: f32,
    /// Reference to the SDF source (cloned from the source node).
    pub sdf_source: SdfSource,
    /// How this node blends with siblings.
    pub blend_mode: BlendMode,
    /// Depth in the original tree (0 = root node of the object).
    pub depth: u32,
    /// Index of this node's parent in the flat array, or `u32::MAX` for root.
    pub parent_index: u32,
    /// Name of the source node (for debugging).
    pub name: String,
}

/// Flatten an object's node tree into a GPU-ready array.
///
/// Performs a depth-first traversal, accumulating transforms from root to leaf.
/// The root position is converted to camera-relative coordinates using `f64`
/// arithmetic for precision safety.
///
/// # Arguments
///
/// - `object` — the scene object whose tree to flatten
/// - `camera_pos` — current camera world position (for camera-relative transform)
///
/// # Returns
///
/// A flat array of nodes in depth-first order. Only nodes with non-None
/// `sdf_source` contribute to rendering, but all nodes are included for
/// correct transform propagation.
pub fn flatten_object(object: &SceneObject, camera_pos: &WorldPosition) -> Vec<FlatNode> {
    let mut result = Vec::new();

    // Compute camera-relative root position using f64 arithmetic.
    let camera_rel: Vec3 = object.world_position.relative_to(camera_pos);

    // Build root's camera-relative world transform.
    let root_world = Mat4::from_scale_rotation_translation(
        Vec3::splat(object.scale),
        object.rotation,
        camera_rel,
    );
    let root_accumulated_scale = object.scale;

    // Recurse depth-first.
    flatten_node(
        &object.root_node,
        root_world,
        root_accumulated_scale,
        0,     // depth
        u32::MAX, // parent_index (root has no parent)
        &mut result,
    );

    result
}

fn flatten_node(
    node: &SceneNode,
    parent_world: Mat4,
    parent_scale: f32,
    depth: u32,
    parent_index: u32,
    result: &mut Vec<FlatNode>,
) {
    // Compute this node's world transform.
    let local = node.local_transform.to_matrix();
    let world = parent_world * local;
    let accumulated_scale = parent_scale * node.local_transform.scale;

    // Compute inverse (for transforming rays into local space).
    let inverse_world = world.inverse();

    let my_index = result.len() as u32;
    result.push(FlatNode {
        inverse_world,
        accumulated_scale,
        sdf_source: node.sdf_source.clone(),
        blend_mode: node.blend_mode,
        depth,
        parent_index,
        name: node.name.clone(),
    });

    // Recurse into children.
    for child in &node.children {
        if child.metadata.visible {
            flatten_node(child, world, accumulated_scale, depth + 1, my_index, result);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_node::{SdfPrimitive, Transform};
    use glam::{IVec3, Quat};
    use std::f32::consts::FRAC_PI_2;

    fn make_object(root_node: SceneNode) -> SceneObject {
        SceneObject {
            id: 1,
            name: "test".into(),
            world_position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            root_node,
            aabb: crate::aabb::Aabb::new(Vec3::ZERO, Vec3::ZERO),
        }
    }

    fn camera_origin() -> WorldPosition {
        WorldPosition::default()
    }

    #[test]
    fn identity_transform_single_node() {
        let root = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.5 },
            1,
        );
        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].name, "sphere");
        assert!((flat[0].accumulated_scale - 1.0).abs() < 1e-6);
        assert_eq!(flat[0].depth, 0);
        assert_eq!(flat[0].parent_index, u32::MAX);

        // Inverse of identity should be identity.
        let inv = flat[0].inverse_world;
        let p = inv.transform_point3(Vec3::new(1.0, 2.0, 3.0));
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
    }

    #[test]
    fn nested_transforms_propagate() {
        let child = SceneNode::analytical(
            "child",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        )
        .with_transform(Transform::new(
            Vec3::new(2.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
        ));

        let mut root = SceneNode::new("root");
        root.add_child(child);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert_eq!(flat.len(), 2);
        assert_eq!(flat[0].name, "root");
        assert_eq!(flat[1].name, "child");
        assert_eq!(flat[1].depth, 1);
        assert_eq!(flat[1].parent_index, 0);

        // Transform a point at origin through the child's inverse —
        // origin in child's local space maps to (2,0,0) in world.
        let world_pos = flat[1].inverse_world.inverse().transform_point3(Vec3::ZERO);
        assert!((world_pos - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn scale_accumulation() {
        let child = SceneNode::new("child")
            .with_transform(Transform::new(Vec3::ZERO, Quat::IDENTITY, 0.5));

        let mut root = SceneNode::new("root");
        root.local_transform.scale = 1.0; // Root node's own scale
        root.add_child(child);

        let mut obj = make_object(root);
        obj.scale = 2.0; // Object root scale

        let flat = flatten_object(&obj, &camera_origin());

        // Root: object.scale * root_node.scale = 2.0 * 1.0 = 2.0
        assert!((flat[0].accumulated_scale - 2.0).abs() < 1e-6);
        // Child: 2.0 * 0.5 = 1.0
        assert!((flat[1].accumulated_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn deep_scale_accumulation() {
        let grandchild = SceneNode::new("gc")
            .with_transform(Transform::new(Vec3::ZERO, Quat::IDENTITY, 0.5));
        let mut child = SceneNode::new("child")
            .with_transform(Transform::new(Vec3::ZERO, Quat::IDENTITY, 3.0));
        child.add_child(grandchild);
        let mut root = SceneNode::new("root");
        root.add_child(child);

        let mut obj = make_object(root);
        obj.scale = 2.0;

        let flat = flatten_object(&obj, &camera_origin());
        assert_eq!(flat.len(), 3);
        // root: 2.0, child: 2.0*3.0=6.0, gc: 6.0*0.5=3.0
        assert!((flat[0].accumulated_scale - 2.0).abs() < 1e-6);
        assert!((flat[1].accumulated_scale - 6.0).abs() < 1e-6);
        assert!((flat[2].accumulated_scale - 3.0).abs() < 1e-6);
    }

    #[test]
    fn inverse_world_correctness() {
        // Object at (5, 0, 0) with child at local (3, 0, 0)
        let child = SceneNode::analytical(
            "child",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        )
        .with_transform(Transform::new(
            Vec3::new(3.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
        ));

        let mut root = SceneNode::new("root");
        root.add_child(child);

        let mut obj = make_object(root);
        obj.world_position = WorldPosition::new(IVec3::ZERO, Vec3::new(5.0, 0.0, 0.0));

        let flat = flatten_object(&obj, &camera_origin());

        // Child is at world (5+3, 0, 0) = (8, 0, 0)
        // Its inverse should map world (8,0,0) to local (0,0,0)
        let local = flat[1].inverse_world.transform_point3(Vec3::new(8.0, 0.0, 0.0));
        assert!(local.length() < 1e-3, "local = {local:?}");
    }

    #[test]
    fn camera_relative_precision() {
        // Object very far away, camera at same chunk
        let far_chunk = IVec3::new(1_000_000, 0, 0);
        let obj_pos = WorldPosition::new(far_chunk, Vec3::new(1.0, 0.0, 0.0));
        let cam_pos = WorldPosition::new(far_chunk, Vec3::new(0.0, 0.0, 0.0));

        let root = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        );
        let mut obj = make_object(root);
        obj.world_position = obj_pos;

        let flat = flatten_object(&obj, &cam_pos);

        // Object is 1 metre from camera — inverse should map (1,0,0) to (0,0,0)
        let local = flat[0].inverse_world.transform_point3(Vec3::new(1.0, 0.0, 0.0));
        assert!(local.length() < 1e-3, "camera-relative should be precise: {local:?}");
    }

    #[test]
    fn rotation_propagation() {
        // Root rotated 90° around Z, child at local (1, 0, 0)
        let child = SceneNode::analytical(
            "child",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        )
        .with_transform(Transform::new(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            1.0,
        ));

        let mut root = SceneNode::new("root");
        root.add_child(child);

        let mut obj = make_object(root);
        obj.rotation = Quat::from_rotation_z(FRAC_PI_2);

        let flat = flatten_object(&obj, &camera_origin());

        // After 90° Z rotation, child at local (1,0,0) should be at world (0,1,0)
        let world = flat[1].inverse_world.inverse().transform_point3(Vec3::ZERO);
        assert!((world.x).abs() < 1e-3, "x: {}", world.x);
        assert!((world.y - 1.0).abs() < 1e-3, "y: {}", world.y);
    }

    #[test]
    fn invisible_children_skipped() {
        let visible = SceneNode::analytical(
            "visible",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        );
        let mut invisible = SceneNode::analytical(
            "invisible",
            SdfPrimitive::Sphere { radius: 0.1 },
            1,
        );
        invisible.metadata.visible = false;

        let mut root = SceneNode::new("root");
        root.add_child(visible);
        root.add_child(invisible);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        // Root + visible child only (invisible skipped)
        assert_eq!(flat.len(), 2);
        assert_eq!(flat[1].name, "visible");
    }

    #[test]
    fn depth_first_order() {
        // Build tree: root -> [A -> [C], B]
        let c = SceneNode::new("C");
        let mut a = SceneNode::new("A");
        a.add_child(c);
        let b = SceneNode::new("B");
        let mut root = SceneNode::new("root");
        root.add_child(a);
        root.add_child(b);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].name, "root");
        assert_eq!(flat[1].name, "A");
        assert_eq!(flat[2].name, "C");
        assert_eq!(flat[3].name, "B");
    }

    #[test]
    fn parent_indices_correct() {
        let c = SceneNode::new("C");
        let mut a = SceneNode::new("A");
        a.add_child(c);
        let b = SceneNode::new("B");
        let mut root = SceneNode::new("root");
        root.add_child(a);
        root.add_child(b);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert_eq!(flat[0].parent_index, u32::MAX); // root
        assert_eq!(flat[1].parent_index, 0);         // A → root
        assert_eq!(flat[2].parent_index, 1);         // C → A
        assert_eq!(flat[3].parent_index, 0);         // B → root
    }

    #[test]
    fn blend_modes_preserved() {
        let child = SceneNode::new("child").with_blend_mode(BlendMode::Subtract);
        let mut root = SceneNode::new("root");
        root.add_child(child);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert!(matches!(flat[0].blend_mode, BlendMode::SmoothUnion(_)));
        assert!(matches!(flat[1].blend_mode, BlendMode::Subtract));
    }

    #[test]
    fn sdf_source_preserved() {
        let child = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.5 },
            42,
        );
        let mut root = SceneNode::new("group");
        root.add_child(child);

        let obj = make_object(root);
        let flat = flatten_object(&obj, &camera_origin());

        assert!(matches!(flat[0].sdf_source, SdfSource::None));
        match &flat[1].sdf_source {
            SdfSource::Analytical {
                primitive,
                material_id,
            } => {
                assert!(matches!(primitive, SdfPrimitive::Sphere { radius } if (*radius - 0.5).abs() < 1e-6));
                assert_eq!(*material_id, 42);
            }
            _ => panic!("expected Analytical"),
        }
    }
}
