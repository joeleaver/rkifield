//! Scene and SceneObject — v2 root-level containers.
//!
//! A [`Scene`] holds a collection of [`SceneObject`]s, each of which owns
//! a [`SceneNode`] tree. Objects carry [`WorldPosition`] for precision-safe
//! placement in the world.
//!
//! # Design
//!
//! - Root objects sit directly in the Scene
//! - Objects use `WorldPosition` (chunk + local) for ±17 billion metre range
//! - Object IDs are monotonically increasing u32, never reused
//! - Child nodes within an object carry local `Transform` relative to parent

use glam::Quat;

use crate::aabb::Aabb;
use crate::scene_node::SceneNode;
use crate::world_position::WorldPosition;

/// A root-level object in the scene, owning an SDF node tree.
///
/// Objects carry a `WorldPosition` for precision-safe world placement.
/// Their `root_node` is the root of the SDF tree hierarchy.
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// Unique monotonically-increasing ID within the scene.
    pub id: u32,
    /// Human-readable name.
    pub name: String,
    /// Precision-safe world placement (chunk + local).
    pub world_position: WorldPosition,
    /// Root rotation (world space).
    pub rotation: Quat,
    /// Root scale (uniform only).
    pub scale: f32,
    /// Root of the SDF tree hierarchy.
    pub root_node: SceneNode,
    /// Cached world-space AABB (computed from the tree).
    pub aabb: Aabb,
}

/// A collection of root-level objects.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Scene name.
    pub name: String,
    /// All root-level objects.
    pub root_objects: Vec<SceneObject>,
    /// Next object ID to assign (monotonically increasing).
    next_id: u32,
}

impl Scene {
    /// Create a new empty scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            root_objects: Vec::new(),
            next_id: 1,
        }
    }

    /// Add an object to the scene with auto-assigned ID.
    ///
    /// Returns the assigned ID.
    pub fn add_object(
        &mut self,
        name: impl Into<String>,
        world_position: WorldPosition,
        root_node: SceneNode,
    ) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let obj = SceneObject {
            id,
            name: name.into(),
            world_position,
            rotation: Quat::IDENTITY,
            scale: 1.0,
            root_node,
            aabb: Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO),
        };
        self.root_objects.push(obj);
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
        self.root_objects.push(obj);
        id
    }

    /// Remove an object by ID, returning it if found.
    pub fn remove_object(&mut self, id: u32) -> Option<SceneObject> {
        if let Some(idx) = self.root_objects.iter().position(|o| o.id == id) {
            Some(self.root_objects.remove(idx))
        } else {
            None
        }
    }

    /// Find an object by name (first match).
    pub fn find_by_name(&self, name: &str) -> Option<&SceneObject> {
        self.root_objects.iter().find(|o| o.name == name)
    }

    /// Find an object by name (mutable, first match).
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SceneObject> {
        self.root_objects.iter_mut().find(|o| o.name == name)
    }

    /// Find an object by ID.
    pub fn find_by_id(&self, id: u32) -> Option<&SceneObject> {
        self.root_objects.iter().find(|o| o.id == id)
    }

    /// Find an object by ID (mutable).
    pub fn find_by_id_mut(&mut self, id: u32) -> Option<&mut SceneObject> {
        self.root_objects.iter_mut().find(|o| o.id == id)
    }

    /// Number of root objects in the scene.
    pub fn object_count(&self) -> usize {
        self.root_objects.len()
    }

    /// Total number of nodes across all objects (each object's tree summed).
    pub fn total_node_count(&self) -> usize {
        self.root_objects
            .iter()
            .map(|o| o.root_node.node_count())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_node::{SdfPrimitive, SdfSource};
    use glam::{IVec3, Vec3};

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
        let id1 = scene.add_object("obj1", WorldPosition::default(), SceneNode::new("root1"));
        let id2 = scene.add_object("obj2", WorldPosition::default(), SceneNode::new("root2"));
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
            world_position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };
        let id = scene.add_object_full(obj);
        assert_eq!(id, 1);
        assert_eq!(scene.root_objects[0].id, 1);
    }

    #[test]
    fn remove_object_by_id() {
        let mut scene = Scene::new("test");
        let id1 = scene.add_object("obj1", WorldPosition::default(), SceneNode::new("r1"));
        let _id2 = scene.add_object("obj2", WorldPosition::default(), SceneNode::new("r2"));

        let removed = scene.remove_object(id1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "obj1");
        assert_eq!(scene.object_count(), 1);
        assert_eq!(scene.root_objects[0].name, "obj2");
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut scene = Scene::new("test");
        assert!(scene.remove_object(999).is_none());
    }

    #[test]
    fn find_by_name() {
        let mut scene = Scene::new("test");
        scene.add_object(
            "sphere",
            WorldPosition::new(IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
            sphere_node("root", 0.5, 1),
        );
        scene.add_object("box", WorldPosition::default(), SceneNode::new("root"));

        let found = scene.find_by_name("sphere");
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "sphere");
        assert!(scene.find_by_name("missing").is_none());
    }

    #[test]
    fn find_by_name_mut() {
        let mut scene = Scene::new("test");
        scene.add_object("obj", WorldPosition::default(), SceneNode::new("root"));

        if let Some(obj) = scene.find_by_name_mut("obj") {
            obj.scale = 2.0;
        }
        assert_eq!(scene.root_objects[0].scale, 2.0);
    }

    #[test]
    fn find_by_id() {
        let mut scene = Scene::new("test");
        let id = scene.add_object("obj", WorldPosition::default(), SceneNode::new("root"));

        let found = scene.find_by_id(id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);
        assert!(scene.find_by_id(999).is_none());
    }

    #[test]
    fn find_by_id_mut() {
        let mut scene = Scene::new("test");
        let id = scene.add_object("obj", WorldPosition::default(), SceneNode::new("root"));

        if let Some(obj) = scene.find_by_id_mut(id) {
            obj.rotation = Quat::from_rotation_y(1.0);
        }
        let rot = scene.root_objects[0].rotation;
        assert!((rot - Quat::from_rotation_y(1.0)).length() < 1e-5);
    }

    #[test]
    fn total_node_count() {
        let mut scene = Scene::new("test");

        // Object with 3 nodes (root + 2 children)
        let mut root = SceneNode::new("root");
        root.add_child(sphere_node("a", 0.1, 1));
        root.add_child(sphere_node("b", 0.2, 2));
        scene.add_object("obj1", WorldPosition::default(), root);

        // Object with 1 node
        scene.add_object("obj2", WorldPosition::default(), SceneNode::new("root2"));

        assert_eq!(scene.total_node_count(), 4);
    }

    #[test]
    fn ids_never_reuse_after_removal() {
        let mut scene = Scene::new("test");
        let id1 = scene.add_object("a", WorldPosition::default(), SceneNode::new("r1"));
        let _id2 = scene.add_object("b", WorldPosition::default(), SceneNode::new("r2"));
        scene.remove_object(id1);

        // Next ID should be 3, not 1
        let id3 = scene.add_object("c", WorldPosition::default(), SceneNode::new("r3"));
        assert_eq!(id3, 3);
    }

    #[test]
    fn object_world_position() {
        let mut scene = Scene::new("test");
        let pos = WorldPosition::new(IVec3::new(100, 0, -50), Vec3::new(3.0, 1.5, 7.0));
        scene.add_object("far_away", pos, SceneNode::new("root"));

        let obj = scene.find_by_name("far_away").unwrap();
        assert_eq!(obj.world_position.chunk, IVec3::new(100, 0, -50));
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
        scene.add_object("imported_mesh", WorldPosition::default(), root);

        let obj = scene.find_by_name("imported_mesh").unwrap();
        assert!(matches!(obj.root_node.sdf_source, SdfSource::Voxelized { .. }));
    }
}
