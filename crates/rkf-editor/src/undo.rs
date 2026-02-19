//! Unified undo/redo stack for the RKIField editor.
//!
//! Tracks transform changes, entity spawn/despawn, voxel edits, property changes,
//! and environment changes. Supports configurable max depth with oldest-first eviction.

#![allow(dead_code)]

use glam::{IVec3, Quat, Vec3};

/// The kind of action that can be undone/redone.
#[derive(Debug, Clone)]
pub enum UndoActionKind {
    /// Entity transform change.
    Transform {
        entity_id: u64,
        old_pos: Vec3,
        old_rot: Quat,
        old_scale: f32,
        new_pos: Vec3,
        new_rot: Quat,
        new_scale: f32,
    },
    /// An entity was spawned (undo = despawn it).
    SpawnEntity { entity_id: u64 },
    /// An entity was despawned (undo = respawn it).
    DespawnEntity { entity_id: u64 },
    /// A voxel edit operation on a chunk.
    VoxelEdit {
        chunk: IVec3,
        description: String,
    },
    /// A property was changed on an entity.
    PropertyChange {
        entity_id: u64,
        property_name: String,
        old_value: String,
        new_value: String,
    },
    /// An environment setting was changed.
    EnvironmentChange {
        field: String,
        old_value: String,
        new_value: String,
    },
}

/// A single undoable action with metadata.
#[derive(Debug, Clone)]
pub struct UndoAction {
    /// What kind of action this is.
    pub kind: UndoActionKind,
    /// When the action occurred (milliseconds since epoch or arbitrary counter).
    pub timestamp_ms: u64,
    /// Human-readable description of the action.
    pub description: String,
}

/// Dual-stack undo/redo system with configurable depth.
#[derive(Debug)]
pub struct UndoStack {
    undo_stack: Vec<UndoAction>,
    redo_stack: Vec<UndoAction>,
    max_depth: usize,
}

impl UndoStack {
    /// Create a new undo stack with the given maximum depth.
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    /// Push a new action onto the undo stack.
    ///
    /// Clears the redo stack (new action invalidates the redo history).
    /// If the stack exceeds `max_depth`, the oldest action is dropped.
    pub fn push(&mut self, action: UndoAction) {
        self.redo_stack.clear();
        self.undo_stack.push(action);

        // Enforce max depth by dropping the oldest (bottom of stack).
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the most recent action.
    ///
    /// Pops from the undo stack and pushes onto the redo stack.
    /// Returns `None` if there is nothing to undo.
    pub fn undo(&mut self) -> Option<UndoAction> {
        let action = self.undo_stack.pop()?;
        self.redo_stack.push(action.clone());
        Some(action)
    }

    /// Redo the most recently undone action.
    ///
    /// Pops from the redo stack and pushes onto the undo stack.
    /// Returns `None` if there is nothing to redo.
    pub fn redo(&mut self) -> Option<UndoAction> {
        let action = self.redo_stack.pop()?;
        self.undo_stack.push(action.clone());
        Some(action)
    }

    /// Whether there are actions that can be undone.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether there are actions that can be redone.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Description of the action that would be undone next.
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.last().map(|a| a.description.as_str())
    }

    /// Description of the action that would be redone next.
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|a| a.description.as_str())
    }

    /// Clear both undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Number of actions in the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of actions in the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_action(desc: &str) -> UndoAction {
        UndoAction {
            kind: UndoActionKind::Transform {
                entity_id: 1,
                old_pos: Vec3::ZERO,
                old_rot: Quat::IDENTITY,
                old_scale: 1.0,
                new_pos: Vec3::X,
                new_rot: Quat::IDENTITY,
                new_scale: 1.0,
            },
            timestamp_ms: 0,
            description: desc.to_string(),
        }
    }

    fn make_spawn_action(id: u64) -> UndoAction {
        UndoAction {
            kind: UndoActionKind::SpawnEntity { entity_id: id },
            timestamp_ms: 100,
            description: format!("Spawn entity {id}"),
        }
    }

    #[test]
    fn test_push_undo_redo_lifecycle() {
        let mut stack = UndoStack::new(100);

        stack.push(make_action("Move object"));
        stack.push(make_action("Rotate object"));
        assert_eq!(stack.undo_count(), 2);
        assert_eq!(stack.redo_count(), 0);

        // Undo one
        let undone = stack.undo();
        assert!(undone.is_some());
        assert_eq!(undone.unwrap().description, "Rotate object");
        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 1);

        // Redo it
        let redone = stack.redo();
        assert!(redone.is_some());
        assert_eq!(redone.unwrap().description, "Rotate object");
        assert_eq!(stack.undo_count(), 2);
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn test_redo_cleared_on_new_push() {
        let mut stack = UndoStack::new(100);

        stack.push(make_action("Action A"));
        stack.push(make_action("Action B"));

        // Undo B
        stack.undo();
        assert!(stack.can_redo());

        // Push a new action — redo should be cleared
        stack.push(make_action("Action C"));
        assert!(!stack.can_redo());
        assert_eq!(stack.redo_count(), 0);
    }

    #[test]
    fn test_max_depth_enforcement() {
        let mut stack = UndoStack::new(3);

        stack.push(make_action("A"));
        stack.push(make_action("B"));
        stack.push(make_action("C"));
        assert_eq!(stack.undo_count(), 3);

        // Pushing a 4th should evict the oldest
        stack.push(make_action("D"));
        assert_eq!(stack.undo_count(), 3);

        // The oldest remaining should be B (A was evicted)
        stack.undo(); // D
        stack.undo(); // C
        let oldest = stack.undo();
        assert!(oldest.is_some());
        assert_eq!(oldest.unwrap().description, "B");
    }

    #[test]
    fn test_empty_stack_returns_none() {
        let mut stack = UndoStack::new(100);

        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
        assert!(stack.undo().is_none());
        assert!(stack.redo().is_none());
        assert!(stack.undo_description().is_none());
        assert!(stack.redo_description().is_none());
    }

    #[test]
    fn test_descriptions() {
        let mut stack = UndoStack::new(100);

        stack.push(make_action("Move box"));
        assert_eq!(stack.undo_description(), Some("Move box"));
        assert_eq!(stack.redo_description(), None);

        stack.undo();
        assert_eq!(stack.undo_description(), None);
        assert_eq!(stack.redo_description(), Some("Move box"));
    }

    #[test]
    fn test_clear() {
        let mut stack = UndoStack::new(100);

        stack.push(make_action("A"));
        stack.push(make_action("B"));
        stack.undo();

        assert_eq!(stack.undo_count(), 1);
        assert_eq!(stack.redo_count(), 1);

        stack.clear();
        assert_eq!(stack.undo_count(), 0);
        assert_eq!(stack.redo_count(), 0);
        assert!(!stack.can_undo());
        assert!(!stack.can_redo());
    }

    #[test]
    fn test_spawn_despawn_action_kinds() {
        let mut stack = UndoStack::new(100);

        stack.push(make_spawn_action(42));
        stack.push(UndoAction {
            kind: UndoActionKind::DespawnEntity { entity_id: 42 },
            timestamp_ms: 200,
            description: "Despawn entity 42".to_string(),
        });

        assert_eq!(stack.undo_count(), 2);
        let undone = stack.undo().unwrap();
        assert_eq!(undone.description, "Despawn entity 42");
        if let UndoActionKind::DespawnEntity { entity_id } = undone.kind {
            assert_eq!(entity_id, 42);
        } else {
            panic!("Expected DespawnEntity kind");
        }
    }

    #[test]
    fn test_voxel_edit_kind() {
        let mut stack = UndoStack::new(100);
        stack.push(UndoAction {
            kind: UndoActionKind::VoxelEdit {
                chunk: IVec3::new(1, 2, 3),
                description: "Sculpt sphere".to_string(),
            },
            timestamp_ms: 300,
            description: "Voxel edit".to_string(),
        });
        assert!(stack.can_undo());
    }

    #[test]
    fn test_property_change_kind() {
        let mut stack = UndoStack::new(100);
        stack.push(UndoAction {
            kind: UndoActionKind::PropertyChange {
                entity_id: 5,
                property_name: "intensity".to_string(),
                old_value: "1.0".to_string(),
                new_value: "2.5".to_string(),
            },
            timestamp_ms: 400,
            description: "Change intensity".to_string(),
        });

        let undone = stack.undo().unwrap();
        if let UndoActionKind::PropertyChange {
            property_name,
            old_value,
            new_value,
            ..
        } = &undone.kind
        {
            assert_eq!(property_name, "intensity");
            assert_eq!(old_value, "1.0");
            assert_eq!(new_value, "2.5");
        } else {
            panic!("Expected PropertyChange kind");
        }
    }

    #[test]
    fn test_environment_change_kind() {
        let mut stack = UndoStack::new(100);
        stack.push(UndoAction {
            kind: UndoActionKind::EnvironmentChange {
                field: "fog_density".to_string(),
                old_value: "0.01".to_string(),
                new_value: "0.05".to_string(),
            },
            timestamp_ms: 500,
            description: "Change fog density".to_string(),
        });

        assert_eq!(stack.undo_description(), Some("Change fog density"));
    }

    #[test]
    fn test_multiple_undo_redo_sequence() {
        let mut stack = UndoStack::new(100);

        stack.push(make_action("A"));
        stack.push(make_action("B"));
        stack.push(make_action("C"));

        // Undo all three
        assert_eq!(stack.undo().unwrap().description, "C");
        assert_eq!(stack.undo().unwrap().description, "B");
        assert_eq!(stack.undo().unwrap().description, "A");
        assert!(stack.undo().is_none());

        // Redo all three
        assert_eq!(stack.redo().unwrap().description, "A");
        assert_eq!(stack.redo().unwrap().description, "B");
        assert_eq!(stack.redo().unwrap().description, "C");
        assert!(stack.redo().is_none());
    }
}
