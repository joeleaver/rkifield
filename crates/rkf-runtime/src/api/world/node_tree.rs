//! Node tree access and mutation operations.

use uuid::Uuid;

use rkf_core::scene_node::{BlendMode, SceneNode, SdfSource, Transform as NodeTransform};

use crate::components::SdfTree;

use super::{World, WorldError};

impl World {
    // ── Node tree access ────────────────────────────────────────────────

    /// Get a reference to the root scene node of an entity.
    pub fn root_node(&self, entity_id: Uuid) -> Result<&SceneNode, WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        record
            .sdf_object_id
            .ok_or(WorldError::MissingComponent(entity_id, "SdfObject"))?;
        // SAFETY: We have a shared reference to self which includes self.ecs.
        // hecs get returns a Ref guard that borrows the World, so we return
        // a reference tied to &self's lifetime by reading through the guard.
        self.ecs
            .get::<&SdfTree>(record.ecs_entity)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
            .map(|sdf| {
                // SAFETY: The SdfTree lives as long as the hecs World (which is &self).
                // hecs::Ref derefs to &T; we extend the lifetime to match &self.
                let root_ptr: *const SceneNode = &sdf.root;
                unsafe { &*root_ptr }
            })
    }

    /// Get a mutable reference to the root SDF scene node of an entity.
    ///
    /// Advanced: use this for per-frame animation updates that modify the
    /// scene node tree (e.g. skeletal animation bone transforms).
    pub fn root_node_mut(&mut self, entity_id: Uuid) -> Result<&mut SceneNode, WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        record
            .sdf_object_id
            .ok_or(WorldError::MissingComponent(entity_id, "SdfObject"))?;
        let ecs_entity = record.ecs_entity;
        self.ecs
            .get::<&mut SdfTree>(ecs_entity)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
            .map(|mut sdf| {
                let root_ptr: *mut SceneNode = &mut sdf.root;
                unsafe { &mut *root_ptr }
            })
    }

    /// Find a node by name within an entity's scene node tree.
    pub fn find_node(&self, entity_id: Uuid, name: &str) -> Result<&SceneNode, WorldError> {
        let root = self.root_node(entity_id)?;
        root.find_by_name(name)
            .ok_or_else(|| WorldError::NodeNotFound(name.to_string()))
    }

    /// Find a node by name (mutable) within an entity's scene node tree.
    pub fn find_node_mut(
        &mut self,
        entity_id: Uuid,
        name: &str,
    ) -> Result<&mut SceneNode, WorldError> {
        let root = self.root_node_mut(entity_id)?;
        root.find_by_name_mut(name)
            .ok_or_else(|| WorldError::NodeNotFound(name.to_string()))
    }

    /// Find a node by slash-separated path within an entity's scene node tree.
    pub fn find_node_by_path(
        &self,
        entity_id: Uuid,
        path: &str,
    ) -> Result<&SceneNode, WorldError> {
        let root = self.root_node(entity_id)?;
        root.find_by_path(path)
            .ok_or_else(|| WorldError::NodeNotFound(path.to_string()))
    }

    /// Find a node by slash-separated path (mutable) within an entity's scene node tree.
    pub fn find_node_by_path_mut(
        &mut self,
        entity_id: Uuid,
        path: &str,
    ) -> Result<&mut SceneNode, WorldError> {
        let root = self.root_node_mut(entity_id)?;
        root.find_by_path_mut(path)
            .ok_or_else(|| WorldError::NodeNotFound(path.to_string()))
    }

    /// Count the total number of nodes in an entity's scene node tree.
    pub fn node_count(&self, entity_id: Uuid) -> Result<usize, WorldError> {
        let root = self.root_node(entity_id)?;
        Ok(root.node_count())
    }

    // ── Node tree mutation ──────────────────────────────────────────────

    /// Set the local transform of a named node within an entity's tree.
    pub fn set_node_transform(
        &mut self,
        entity_id: Uuid,
        node_name: &str,
        transform: NodeTransform,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity_id, node_name)?;
        node.local_transform = transform;
        Ok(())
    }

    /// Add a child node to a named parent node within an entity's tree.
    pub fn add_child_node(
        &mut self,
        entity_id: Uuid,
        parent_name: &str,
        child: SceneNode,
    ) -> Result<(), WorldError> {
        let parent = self.find_node_mut(entity_id, parent_name)?;
        parent.add_child(child);
        Ok(())
    }

    /// Remove a named node from an entity's tree, returning it.
    ///
    /// Searches the tree for the node's parent and removes it from the
    /// parent's children. Cannot remove the root node itself.
    pub fn remove_child_node(
        &mut self,
        entity_id: Uuid,
        node_name: &str,
    ) -> Result<SceneNode, WorldError> {
        let root = self.root_node_mut(entity_id)?;
        // Cannot remove root itself
        if root.name == node_name {
            return Err(WorldError::NodeNotFound(format!(
                "cannot remove root node '{}'",
                node_name
            )));
        }
        let removed = remove_named_child(root, node_name)
            .ok_or_else(|| WorldError::NodeNotFound(node_name.to_string()))?;
        Ok(removed)
    }

    /// Set the blend mode of a named node within an entity's tree.
    pub fn set_node_blend_mode(
        &mut self,
        entity_id: Uuid,
        node_name: &str,
        mode: BlendMode,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity_id, node_name)?;
        node.blend_mode = mode;
        Ok(())
    }

    /// Set the SDF source of a named node within an entity's tree.
    pub fn set_node_sdf_source(
        &mut self,
        entity_id: Uuid,
        node_name: &str,
        source: SdfSource,
    ) -> Result<(), WorldError> {
        let node = self.find_node_mut(entity_id, node_name)?;
        node.sdf_source = source;
        Ok(())
    }
}

// ── Node tree helpers ──────────────────────────────────────────────────────

/// Recursively find and remove a child with the given name from the tree.
/// Returns the removed node, or None if not found.
fn remove_named_child(parent: &mut SceneNode, name: &str) -> Option<SceneNode> {
    // Check direct children first
    if let Some(pos) = parent.children.iter().position(|c| c.name == name) {
        return Some(parent.children.remove(pos));
    }
    // Recurse into children
    for child in &mut parent.children {
        if let Some(removed) = remove_named_child(child, name) {
            return Some(removed);
        }
    }
    None
}
