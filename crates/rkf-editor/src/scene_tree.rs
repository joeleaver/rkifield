//! Scene hierarchy data model for the RKIField editor.
//!
//! Provides a tree structure mirroring the ECS entity hierarchy. Each node tracks
//! entity ID, display name, visibility, selection, and expand/collapse state.
//! This is a pure data model that can be tested independently of the GUI framework.

#![allow(dead_code)]

use glam::{Quat, Vec3};

/// A single node in the scene hierarchy tree.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// ECS entity ID this node represents.
    pub entity_id: u64,
    /// Human-readable display name.
    pub name: String,
    /// Child nodes.
    pub children: Vec<SceneNode>,
    /// Whether this node's children are visible in the tree UI.
    pub expanded: bool,
    /// Whether this node is currently selected.
    pub selected: bool,
    /// Whether the entity is visible in the viewport.
    pub visible: bool,
    /// World-space position of this entity.
    pub position: Vec3,
    /// World-space rotation of this entity.
    pub rotation: Quat,
    /// Uniform scale of this entity.
    pub scale: f32,
    /// Asset path for SdfObject entities, used to roundtrip scene save/load.
    pub asset_path: Option<String>,
}

impl SceneNode {
    /// Create a new scene node with the given entity ID and name.
    pub fn new(entity_id: u64, name: impl Into<String>) -> Self {
        Self {
            entity_id,
            name: name.into(),
            children: Vec::new(),
            expanded: false,
            selected: false,
            visible: true,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
            asset_path: None,
        }
    }

    /// Recursively search for a node by entity ID.
    fn find(&self, entity_id: u64) -> Option<&SceneNode> {
        if self.entity_id == entity_id {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find(entity_id) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively search for a mutable node by entity ID.
    fn find_mut(&mut self, entity_id: u64) -> Option<&mut SceneNode> {
        if self.entity_id == entity_id {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_mut(entity_id) {
                return Some(found);
            }
        }
        None
    }

    /// Recursively clear selection on this node and all descendants.
    fn clear_selection(&mut self) {
        self.selected = false;
        for child in &mut self.children {
            child.clear_selection();
        }
    }

    /// Recursively collect selected entity IDs.
    fn collect_selected(&self, out: &mut Vec<u64>) {
        if self.selected {
            out.push(self.entity_id);
        }
        for child in &self.children {
            child.collect_selected(out);
        }
    }

    /// Remove a child (or deeper descendant) by entity ID. Returns the removed node if found.
    fn remove_child(&mut self, entity_id: u64) -> Option<SceneNode> {
        if let Some(idx) = self.children.iter().position(|c| c.entity_id == entity_id) {
            return Some(self.children.remove(idx));
        }
        for child in &mut self.children {
            if let Some(removed) = child.remove_child(entity_id) {
                return Some(removed);
            }
        }
        None
    }
}

/// The scene hierarchy tree, holding all root-level nodes and managing selection.
#[derive(Debug, Clone, Default)]
pub struct SceneTree {
    /// Root-level nodes (entities with no parent).
    pub roots: Vec<SceneNode>,
}

impl SceneTree {
    /// Create an empty scene tree.
    pub fn new() -> Self {
        Self { roots: Vec::new() }
    }

    /// Add a node as a root-level entry.
    pub fn add_node(&mut self, node: SceneNode) {
        self.roots.push(node);
    }

    /// Remove a node by entity ID from anywhere in the tree.
    /// Returns the removed node if found.
    pub fn remove_node(&mut self, entity_id: u64) -> Option<SceneNode> {
        // Check root level first.
        if let Some(idx) = self.roots.iter().position(|n| n.entity_id == entity_id) {
            return Some(self.roots.remove(idx));
        }
        // Search children recursively.
        for root in &mut self.roots {
            if let Some(removed) = root.remove_child(entity_id) {
                return Some(removed);
            }
        }
        None
    }

    /// Find a node by entity ID (immutable).
    pub fn find_node(&self, entity_id: u64) -> Option<&SceneNode> {
        for root in &self.roots {
            if let Some(found) = root.find(entity_id) {
                return Some(found);
            }
        }
        None
    }

    /// Find a node by entity ID (mutable).
    pub fn find_node_mut(&mut self, entity_id: u64) -> Option<&mut SceneNode> {
        for root in &mut self.roots {
            if let Some(found) = root.find_mut(entity_id) {
                return Some(found);
            }
        }
        None
    }

    /// Toggle the expanded state of a node.
    pub fn toggle_expanded(&mut self, entity_id: u64) {
        if let Some(node) = self.find_node_mut(entity_id) {
            node.expanded = !node.expanded;
        }
    }

    /// Select a single node, clearing all other selections.
    pub fn select_node(&mut self, entity_id: u64) {
        self.clear_selection();
        if let Some(node) = self.find_node_mut(entity_id) {
            node.selected = true;
        }
    }

    /// Add a node to the selection without clearing existing selections.
    pub fn multi_select_node(&mut self, entity_id: u64) {
        if let Some(node) = self.find_node_mut(entity_id) {
            node.selected = true;
        }
    }

    /// Clear all selections in the tree.
    pub fn clear_selection(&mut self) {
        for root in &mut self.roots {
            root.clear_selection();
        }
    }

    /// Return the entity IDs of all currently selected nodes.
    pub fn selected_entities(&self) -> Vec<u64> {
        let mut result = Vec::new();
        for root in &self.roots {
            root.collect_selected(&mut result);
        }
        result
    }

    /// Move a node to be a child of a new parent.
    ///
    /// If `new_parent_id` does not exist, the node becomes a root.
    /// Returns `true` if the reparent succeeded, `false` if the entity was not found
    /// or the entity would become its own ancestor.
    pub fn reparent(&mut self, entity_id: u64, new_parent_id: u64) -> bool {
        // Prevent reparenting to self.
        if entity_id == new_parent_id {
            return false;
        }

        // Check that the new parent is not a descendant of the node being moved
        // (would create a cycle).
        if let Some(node) = self.find_node(entity_id) {
            if node.find(new_parent_id).is_some() {
                return false;
            }
        }

        // Remove the node from its current location.
        let Some(node) = self.remove_node(entity_id) else {
            return false;
        };

        // Insert under new parent.
        if let Some(parent) = self.find_node_mut(new_parent_id) {
            parent.children.push(node);
            true
        } else {
            // Parent not found — re-add as root to avoid data loss.
            self.roots.push(node);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn sample_tree() -> SceneTree {
        let mut tree = SceneTree::new();
        let mut parent = SceneNode::new(1, "Parent");
        parent.children.push(SceneNode::new(2, "Child A"));
        parent.children.push(SceneNode::new(3, "Child B"));
        tree.add_node(parent);
        tree.add_node(SceneNode::new(4, "Root2"));
        tree
    }

    #[test]
    fn test_add_and_find_root() {
        let mut tree = SceneTree::new();
        tree.add_node(SceneNode::new(10, "Test"));
        assert!(tree.find_node(10).is_some());
        assert_eq!(tree.find_node(10).unwrap().name, "Test");
    }

    #[test]
    fn test_find_nested_node() {
        let tree = sample_tree();
        assert!(tree.find_node(2).is_some());
        assert_eq!(tree.find_node(2).unwrap().name, "Child A");
        assert!(tree.find_node(3).is_some());
    }

    #[test]
    fn test_find_nonexistent() {
        let tree = sample_tree();
        assert!(tree.find_node(999).is_none());
    }

    #[test]
    fn test_find_node_mut() {
        let mut tree = sample_tree();
        tree.find_node_mut(2).unwrap().name = "Renamed".to_string();
        assert_eq!(tree.find_node(2).unwrap().name, "Renamed");
    }

    #[test]
    fn test_remove_root() {
        let mut tree = sample_tree();
        let removed = tree.remove_node(4);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "Root2");
        assert!(tree.find_node(4).is_none());
    }

    #[test]
    fn test_remove_child() {
        let mut tree = sample_tree();
        let removed = tree.remove_node(2);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "Child A");
        assert!(tree.find_node(2).is_none());
        // Parent and sibling still exist
        assert!(tree.find_node(1).is_some());
        assert!(tree.find_node(3).is_some());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut tree = sample_tree();
        assert!(tree.remove_node(999).is_none());
    }

    #[test]
    fn test_select_node_clears_previous() {
        let mut tree = sample_tree();
        tree.select_node(2);
        assert!(tree.find_node(2).unwrap().selected);

        tree.select_node(3);
        assert!(!tree.find_node(2).unwrap().selected);
        assert!(tree.find_node(3).unwrap().selected);
    }

    #[test]
    fn test_multi_select() {
        let mut tree = sample_tree();
        tree.multi_select_node(2);
        tree.multi_select_node(4);
        assert!(tree.find_node(2).unwrap().selected);
        assert!(tree.find_node(4).unwrap().selected);
        assert!(!tree.find_node(1).unwrap().selected);
    }

    #[test]
    fn test_clear_selection() {
        let mut tree = sample_tree();
        tree.multi_select_node(1);
        tree.multi_select_node(2);
        tree.multi_select_node(3);
        tree.clear_selection();
        assert!(tree.selected_entities().is_empty());
    }

    #[test]
    fn test_selected_entities() {
        let mut tree = sample_tree();
        tree.multi_select_node(2);
        tree.multi_select_node(4);
        let sel = tree.selected_entities();
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(&2));
        assert!(sel.contains(&4));
    }

    #[test]
    fn test_toggle_expanded() {
        let mut tree = sample_tree();
        assert!(!tree.find_node(1).unwrap().expanded);
        tree.toggle_expanded(1);
        assert!(tree.find_node(1).unwrap().expanded);
        tree.toggle_expanded(1);
        assert!(!tree.find_node(1).unwrap().expanded);
    }

    #[test]
    fn test_reparent_to_existing_parent() {
        let mut tree = sample_tree();
        // Move Root2 (id=4) under Parent (id=1)
        assert!(tree.reparent(4, 1));
        assert!(tree.find_node(4).is_some());
        // 4 should now be a child of 1
        let parent = tree.find_node(1).unwrap();
        assert!(parent.children.iter().any(|c| c.entity_id == 4));
    }

    #[test]
    fn test_reparent_to_nonexistent_becomes_root() {
        let mut tree = sample_tree();
        // Move Child A (id=2) under nonexistent (id=999) — should become root
        assert!(!tree.reparent(2, 999));
        // Node should still exist as a root
        assert!(tree.find_node(2).is_some());
        assert!(tree.roots.iter().any(|r| r.entity_id == 2));
    }

    #[test]
    fn test_reparent_to_self_fails() {
        let mut tree = sample_tree();
        assert!(!tree.reparent(1, 1));
    }

    #[test]
    fn test_reparent_prevents_cycle() {
        let mut tree = sample_tree();
        // Parent (1) has Child A (2). Reparenting 1 under 2 would create a cycle.
        assert!(!tree.reparent(1, 2));
    }

    #[test]
    fn test_new_node_defaults() {
        let node = SceneNode::new(42, "Entity");
        assert_eq!(node.entity_id, 42);
        assert_eq!(node.name, "Entity");
        assert!(node.children.is_empty());
        assert!(!node.expanded);
        assert!(!node.selected);
        assert!(node.visible);
        assert_eq!(node.position, Vec3::ZERO);
        assert_eq!(node.rotation, Quat::IDENTITY);
        assert!((node.scale - 1.0).abs() < 1e-6);
        assert!(node.asset_path.is_none());
    }

    #[test]
    fn test_empty_tree() {
        let tree = SceneTree::new();
        assert!(tree.roots.is_empty());
        assert!(tree.selected_entities().is_empty());
        assert!(tree.find_node(1).is_none());
    }
}
