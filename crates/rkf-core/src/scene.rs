//! Scene and SceneObject — v2 root-level containers.
//!
//! A [`Scene`] holds a flat collection of [`SceneObject`]s, each of which owns
//! a [`SceneNode`] tree. Objects carry local `Vec3` position relative to their
//! parent (or scene root if `parent_id` is `None`).
//!
//! # Design
//!
//! - Objects form a flat list with optional parent-child relationships via `parent_id`
//! - Position, rotation, and scale are **local** (relative to parent)
//! - World-space AABB is computed externally by transform baking
//! - Object IDs are monotonically increasing u32, never reused
//! - Child nodes within an object carry local `Transform` relative to parent node

use glam::{Quat, Vec3};

use crate::aabb::Aabb;
use crate::bvh::Bvh;
use crate::scene_node::SceneNode;

/// An object in the scene, owning an SDF node tree.
///
/// Objects carry a local `Vec3` position relative to their parent object
/// (or scene root if `parent_id` is `None`). Rotation and scale are also local.
/// The world-space `aabb` is computed externally by transform baking.
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// Unique monotonically-increasing ID within the scene.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Parent object ID, or `None` if this is a scene root.
    pub parent_id: Option<u32>,
    /// Local position relative to parent (or scene origin if root).
    pub position: Vec3,
    /// Local rotation relative to parent.
    pub rotation: Quat,
    /// Local scale (per-axis). Use `Vec3::ONE` for identity.
    pub scale: Vec3,
    /// Root of the SDF tree hierarchy.
    pub root_node: SceneNode,
    /// Cached world-space AABB (computed by transform baking).
    pub aabb: Aabb,
}

/// A collection of objects with optional parent-child relationships.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Scene name.
    pub name: String,
    /// All objects (both roots and children, stored flat).
    pub objects: Vec<SceneObject>,
    /// Next object ID to assign (monotonically increasing).
    pub next_id: u32,
}

impl Scene {
    /// Create a new empty scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            objects: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an object to the scene with auto-assigned ID and no parent.
    ///
    /// Returns the assigned ID.
    pub fn add_object(
        &mut self,
        name: impl Into<String>,
        position: Vec3,
        root_node: SceneNode,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let obj = SceneObject {
            id,
            name: name.into(),
            parent_id: None,
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            root_node,
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };
        self.objects.push(obj);
        id
    }

    /// Add a fully configured `SceneObject` to the scene.
    ///
    /// The object's `id` field is overwritten with an auto-assigned ID.
    /// Returns the assigned ID.
    pub fn add_object_full(&mut self, mut obj: SceneObject) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        obj.id = id;
        self.objects.push(obj);
        id
    }

    /// Remove an object by ID, returning it if found.
    ///
    /// Any children of the removed object are reparented to the removed
    /// object's parent (i.e. they become children of the grandparent,
    /// or scene roots if the removed object had no parent).
    pub fn remove_object(&mut self, id: u32) -> Option<SceneObject> {
        if let Some(idx) = self.objects.iter().position(|o| o.id == id) {
            let removed = self.objects.remove(idx);
            // Reparent children to the removed object's parent
            for obj in &mut self.objects {
                if obj.parent_id == Some(id) {
                    obj.parent_id = removed.parent_id;
                }
            }
            Some(removed)
        } else {
            None
        }
    }

    /// Find an object by name (first match).
    pub fn find_by_name(&self, name: &str) -> Option<&SceneObject> {
        self.objects.iter().find(|o| o.name == name)
    }

    /// Find an object by name (mutable, first match).
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SceneObject> {
        self.objects.iter_mut().find(|o| o.name == name)
    }

    /// Find an object by ID.
    pub fn find_by_id(&self, id: u32) -> Option<&SceneObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Find an object by ID (mutable).
    pub fn find_by_id_mut(&mut self, id: u32) -> Option<&mut SceneObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    /// Number of objects in the scene (both roots and children).
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Total number of nodes across all objects (each object's tree summed).
    pub fn total_node_count(&self) -> usize {
        self.objects
            .iter()
            .map(|o| o.root_node.node_count())
            .sum()
    }

    /// Collect `(object_id, aabb)` pairs for BVH construction.
    pub fn object_aabb_pairs(&self) -> Vec<(u32, Aabb)> {
        self.objects
            .iter()
            .map(|o| (o.id, o.aabb))
            .collect()
    }

    /// Build a BVH over all objects in the scene.
    pub fn build_bvh(&self) -> Bvh {
        Bvh::build(&self.object_aabb_pairs())
    }

    /// Refit an existing BVH using current object AABBs.
    pub fn refit_bvh(&self, bvh: &mut Bvh) {
        bvh.refit(&self.object_aabb_pairs());
    }

    /// Iterate over root objects (those with no parent).
    pub fn root_objects(&self) -> impl Iterator<Item = &SceneObject> {
        self.objects.iter().filter(|o| o.parent_id.is_none())
    }

    /// Iterate over children of a given object.
    pub fn children_of(&self, id: u32) -> impl Iterator<Item = &SceneObject> {
        self.objects.iter().filter(move |o| o.parent_id == Some(id))
    }

    /// Change an object's parent. Returns `false` if the child doesn't exist
    /// or if the reparent would create a cycle.
    pub fn reparent(&mut self, child_id: u32, new_parent_id: Option<u32>) -> bool {
        // Child must exist
        if self.find_by_id(child_id).is_none() {
            return false;
        }

        // If new_parent_id is Some, it must exist
        if let Some(pid) = new_parent_id {
            if self.find_by_id(pid).is_none() {
                return false;
            }
            // Check for cycles: walk up from new_parent_id — if we reach child_id, it's a cycle
            let mut cursor = Some(pid);
            while let Some(cur) = cursor {
                if cur == child_id {
                    return false; // cycle detected
                }
                cursor = self.find_by_id(cur).and_then(|o| o.parent_id);
            }
        }

        // Safe to reparent
        if let Some(obj) = self.find_by_id_mut(child_id) {
            obj.parent_id = new_parent_id;
            true
        } else {
            false
        }
    }

    /// Returns object IDs in topological (parent-before-child) order via BFS from roots.
    pub fn topological_order(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.objects.len());

        // Start with roots
        for obj in &self.objects {
            if obj.parent_id.is_none() {
                result.push(obj.id);
            }
        }

        // BFS: process each entry, appending its children
        let mut i = 0;
        while i < result.len() {
            let current_id = result[i];
            for obj in &self.objects {
                if obj.parent_id == Some(current_id) {
                    result.push(obj.id);
                }
            }
            i += 1;
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_node::SdfPrimitive;

    fn sphere_node(name: &str, radius: f32, mat: u16) -> SceneNode {
        SceneNode::analytical(name, SdfPrimitive::Sphere { radius }, mat)
    }

    #[test]
    fn scene_new_is_empty() {
        let scene = Scene::new("test");
        assert_eq!(scene.name, "test");
        assert_eq!(scene.object_count(), 0);
        assert_eq!(scene.total_node_count(), 0);
    }

    #[test]
    fn add_object_assigns_sequential_ids() {
        let mut scene = Scene::new("test");
        let id1 = scene.add_object("obj1", Vec3::ZERO, SceneNode::new("root1"));
        let id2 = scene.add_object("obj2", Vec3::ZERO, SceneNode::new("root2"));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(scene.object_count(), 2);
    }

    #[test]
    fn add_object_full_overrides_id() {
        let mut scene = Scene::new("test");
        let obj = SceneObject {
            id: 999, // will be overwritten
            name: "obj".into(),
            parent_id: None,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };
        let id = scene.add_object_full(obj);
        assert_eq!(id, 1);
        assert_eq!(scene.objects[0].id, 1);
    }

    #[test]
    fn remove_object_by_id() {
        let mut scene = Scene::new("test");
        let id1 = scene.add_object("obj1", Vec3::ZERO, SceneNode::new("r1"));
        let _id2 = scene.add_object("obj2", Vec3::ZERO, SceneNode::new("r2"));

        let removed = scene.remove_object(id1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "obj1");
        assert_eq!(scene.object_count(), 1);
        assert_eq!(scene.objects[0].name, "obj2");
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut scene = Scene::new("test");
        assert!(scene.remove_object(999).is_none());
    }

    #[test]
    fn remove_object_reparents_children() {
        let mut scene = Scene::new("test");
        let grandparent = scene.add_object("grandparent", Vec3::ZERO, SceneNode::new("r"));
        let parent = scene.add_object("parent", Vec3::new(1.0, 0.0, 0.0), SceneNode::new("r"));
        let child = scene.add_object("child", Vec3::new(0.0, 1.0, 0.0), SceneNode::new("r"));

        // Set up hierarchy: grandparent -> parent -> child
        scene.reparent(parent, Some(grandparent));
        scene.reparent(child, Some(parent));

        // Remove parent — child should reparent to grandparent
        scene.remove_object(parent);

        let child_obj = scene.find_by_id(child).unwrap();
        assert_eq!(child_obj.parent_id, Some(grandparent));
    }

    #[test]
    fn remove_root_reparents_children_to_root() {
        let mut scene = Scene::new("test");
        let root_obj = scene.add_object("root_obj", Vec3::ZERO, SceneNode::new("r"));
        let child = scene.add_object("child", Vec3::new(1.0, 0.0, 0.0), SceneNode::new("r"));

        scene.reparent(child, Some(root_obj));

        // Remove root_obj — child should become a root (parent_id = None)
        scene.remove_object(root_obj);

        let child_obj = scene.find_by_id(child).unwrap();
        assert_eq!(child_obj.parent_id, None);
    }

    #[test]
    fn find_by_name() {
        let mut scene = Scene::new("test");
        scene.add_object(
            "sphere",
            Vec3::new(1.0, 2.0, 3.0),
            sphere_node("root", 0.5, 1),
        );
        scene.add_object("box", Vec3::ZERO, SceneNode::new("root"));

        let found = scene.find_by_name("sphere");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "sphere");
        assert!(scene.find_by_name("missing").is_none());
    }

    #[test]
    fn find_by_name_mut() {
        let mut scene = Scene::new("test");
        scene.add_object("obj", Vec3::ZERO, SceneNode::new("root"));

        if let Some(obj) = scene.find_by_name_mut("obj") {
            obj.scale = Vec3::splat(2.0);
        }
        assert_eq!(scene.objects[0].scale, Vec3::splat(2.0));
    }

    #[test]
    fn find_by_id() {
        let mut scene = Scene::new("test");
        let id = scene.add_object("obj", Vec3::ZERO, SceneNode::new("root"));

        let found = scene.find_by_id(id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);
        assert!(scene.find_by_id(999).is_none());
    }

    #[test]
    fn find_by_id_mut() {
        let mut scene = Scene::new("test");
        let id = scene.add_object("obj", Vec3::ZERO, SceneNode::new("root"));

        if let Some(obj) = scene.find_by_id_mut(id) {
            obj.rotation = Quat::from_rotation_y(1.0);
        }
        let rot = scene.objects[0].rotation;
        assert!((rot - Quat::from_rotation_y(1.0)).length() < 1e-5);
    }

    #[test]
    fn total_node_count() {
        let mut scene = Scene::new("test");

        // Object with 3 nodes (root + 2 children)
        let mut root = SceneNode::new("root");
        root.add_child(sphere_node("a", 0.1, 1));
        root.add_child(sphere_node("b", 0.2, 2));
        scene.add_object("obj1", Vec3::ZERO, root);

        // Object with 1 node
        scene.add_object("obj2", Vec3::ZERO, SceneNode::new("root2"));

        assert_eq!(scene.total_node_count(), 4);
    }

    #[test]
    fn ids_never_reuse_after_removal() {
        let mut scene = Scene::new("test");
        let id1 = scene.add_object("a", Vec3::ZERO, SceneNode::new("r1"));
        let _id2 = scene.add_object("b", Vec3::ZERO, SceneNode::new("r2"));
        scene.remove_object(id1);

        // Next ID should be 3, not 1
        let id3 = scene.add_object("c", Vec3::ZERO, SceneNode::new("r3"));
        assert_eq!(id3, 3);
    }

    #[test]
    fn object_local_position() {
        let mut scene = Scene::new("test");
        scene.add_object("offset", Vec3::new(3.0, 1.5, 7.0), SceneNode::new("root"));

        let obj = scene.find_by_name("offset").unwrap();
        assert_eq!(obj.position, Vec3::new(3.0, 1.5, 7.0));
        assert_eq!(obj.parent_id, None);
    }

    #[test]
    fn build_bvh_empty_scene() {
        let scene = Scene::new("test");
        let bvh = scene.build_bvh();
        assert!(bvh.is_empty());
    }

    #[test]
    fn build_bvh_with_objects() {
        let mut scene = Scene::new("test");

        let n1 = SceneNode::analytical("s1", SdfPrimitive::Sphere { radius: 0.5 }, 1);
        scene.add_object("obj1", Vec3::new(-5.0, 0.0, 0.0), n1);

        let n2 = SceneNode::analytical("s2", SdfPrimitive::Sphere { radius: 0.5 }, 2);
        scene.add_object("obj2", Vec3::new(5.0, 0.0, 0.0), n2);

        // Set AABBs on objects
        scene.objects[0].aabb = Aabb::new(
            Vec3::new(-5.5, -0.5, -0.5),
            Vec3::new(-4.5, 0.5, 0.5),
        );
        scene.objects[1].aabb = Aabb::new(
            Vec3::new(4.5, -0.5, -0.5),
            Vec3::new(5.5, 0.5, 0.5),
        );

        let bvh = scene.build_bvh();
        assert_eq!(bvh.leaf_count(), 2);
        assert_eq!(bvh.node_count(), 3); // 1 internal + 2 leaves
    }

    #[test]
    fn refit_bvh_after_move() {
        let mut scene = Scene::new("test");

        scene.add_object("obj1", Vec3::ZERO, SceneNode::new("r1"));
        scene.add_object("obj2", Vec3::ZERO, SceneNode::new("r2"));

        scene.objects[0].aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        scene.objects[1].aabb = Aabb::new(Vec3::new(5.0, -1.0, -1.0), Vec3::new(7.0, 1.0, 1.0));

        let mut bvh = scene.build_bvh();

        // Move obj1 far away
        scene.objects[0].aabb = Aabb::new(Vec3::new(100.0, -1.0, -1.0), Vec3::new(102.0, 1.0, 1.0));
        scene.refit_bvh(&mut bvh);

        // Root AABB should extend to 102
        assert!(bvh.nodes[0].aabb.max.x >= 102.0);
    }

    #[test]
    fn object_aabb_pairs() {
        let mut scene = Scene::new("test");
        scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        scene.objects[0].aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);

        let pairs = scene.object_aabb_pairs();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 1); // object ID
    }

    #[test]
    fn object_with_voxelized_source() {
        use crate::scene_node::{BrickMapHandle, SdfSource};
        let mut scene = Scene::new("test");
        let mut root = SceneNode::new("mesh");
        root.sdf_source = SdfSource::Voxelized {
            brick_map_handle: BrickMapHandle {
                offset: 0,
                dims: glam::UVec3::new(8, 8, 8),
            },
            voxel_size: 0.02,
            aabb: Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        };
        scene.add_object("imported_mesh", Vec3::ZERO, root);

        let obj = scene.find_by_name("imported_mesh").unwrap();
        assert!(matches!(obj.root_node.sdf_source, SdfSource::Voxelized { .. }));
    }

    #[test]
    fn parent_id_defaults_to_none() {
        let mut scene = Scene::new("test");
        scene.add_object("obj", Vec3::ZERO, SceneNode::new("r"));
        assert_eq!(scene.objects[0].parent_id, None);
    }

    #[test]
    fn root_objects_iterator() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let _b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));
        let c = scene.add_object("c", Vec3::ZERO, SceneNode::new("r"));

        // Make c a child of a
        scene.reparent(c, Some(a));

        let roots: Vec<&str> = scene.root_objects().map(|o| o.name.as_str()).collect();
        assert_eq!(roots, vec!["a", "b"]);
    }

    #[test]
    fn children_of_iterator() {
        let mut scene = Scene::new("test");
        let parent = scene.add_object("parent", Vec3::ZERO, SceneNode::new("r"));
        let child1 = scene.add_object("child1", Vec3::new(1.0, 0.0, 0.0), SceneNode::new("r"));
        let child2 = scene.add_object("child2", Vec3::new(0.0, 1.0, 0.0), SceneNode::new("r"));
        let _other = scene.add_object("other", Vec3::ZERO, SceneNode::new("r"));

        scene.reparent(child1, Some(parent));
        scene.reparent(child2, Some(parent));

        let children: Vec<&str> = scene.children_of(parent).map(|o| o.name.as_str()).collect();
        assert_eq!(children, vec!["child1", "child2"]);

        // "other" has no children
        assert_eq!(scene.children_of(_other).count(), 0);
    }

    #[test]
    fn reparent_success() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));

        assert!(scene.reparent(b, Some(a)));
        assert_eq!(scene.find_by_id(b).unwrap().parent_id, Some(a));
    }

    #[test]
    fn reparent_to_none() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));

        scene.reparent(b, Some(a));
        assert!(scene.reparent(b, None));
        assert_eq!(scene.find_by_id(b).unwrap().parent_id, None);
    }

    #[test]
    fn reparent_prevents_self_cycle() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));

        assert!(!scene.reparent(a, Some(a)));
    }

    #[test]
    fn reparent_prevents_indirect_cycle() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));
        let c = scene.add_object("c", Vec3::ZERO, SceneNode::new("r"));

        // a -> b -> c
        scene.reparent(b, Some(a));
        scene.reparent(c, Some(b));

        // Trying to make a a child of c would create c -> a -> b -> c cycle
        assert!(!scene.reparent(a, Some(c)));
    }

    #[test]
    fn reparent_nonexistent_child() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        assert!(!scene.reparent(999, Some(a)));
    }

    #[test]
    fn reparent_nonexistent_parent() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        assert!(!scene.reparent(a, Some(999)));
    }

    #[test]
    fn topological_order_flat() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));
        let c = scene.add_object("c", Vec3::ZERO, SceneNode::new("r"));

        let order = scene.topological_order();
        assert_eq!(order, vec![a, b, c]);
    }

    #[test]
    fn topological_order_hierarchy() {
        let mut scene = Scene::new("test");
        let a = scene.add_object("a", Vec3::ZERO, SceneNode::new("r"));
        let b = scene.add_object("b", Vec3::ZERO, SceneNode::new("r"));
        let c = scene.add_object("c", Vec3::ZERO, SceneNode::new("r"));
        let d = scene.add_object("d", Vec3::ZERO, SceneNode::new("r"));

        // a -> b -> d, a -> c
        scene.reparent(b, Some(a));
        scene.reparent(c, Some(a));
        scene.reparent(d, Some(b));

        let order = scene.topological_order();
        // a must come before b and c; b must come before d
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        let pos_c = order.iter().position(|&id| id == c).unwrap();
        let pos_d = order.iter().position(|&id| id == d).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn topological_order_empty() {
        let scene = Scene::new("test");
        assert!(scene.topological_order().is_empty());
    }
}
