//! Unified edit pipeline with undo/redo stack for ECS entity operations.
//!
//! The [`EditPipeline`] captures old state before applying edits, producing
//! [`UndoAction`] entries that allow full undo/redo of property changes,
//! component add/remove, entity spawn/despawn, and geometry edits.

use super::game_value::GameValue;
use super::registry::GameplayRegistry;
use super::stable_id::StableId;
use uuid::Uuid;

// ─── EditOp ──────────────────────────────────────────────────────────────

/// A requested edit operation. The pipeline captures old state and converts
/// this into an [`UndoAction`] before applying.
#[derive(Debug, Clone)]
pub enum EditOp {
    /// Set a single field on a component.
    SetProperty {
        /// Target entity.
        entity: hecs::Entity,
        /// Registered component name.
        component_name: String,
        /// Field within the component.
        field_name: String,
        /// New value to assign.
        value: GameValue,
    },
    /// Add a component to an entity (RON-encoded initial data).
    AddComponent {
        /// Target entity.
        entity: hecs::Entity,
        /// Registered component name.
        component_name: String,
        /// RON-serialized initial data.
        initial_data: String,
    },
    /// Remove a component from an entity.
    RemoveComponent {
        /// Target entity.
        entity: hecs::Entity,
        /// Registered component name.
        component_name: String,
    },
    /// Spawn a new entity.
    SpawnEntity {
        /// Display name for the entity.
        name: String,
        /// Optional parent entity.
        parent: Option<hecs::Entity>,
    },
    /// Despawn an existing entity.
    DespawnEntity {
        /// Entity to remove.
        entity: hecs::Entity,
    },
    /// Opaque geometry edit (sculpt, paint, etc.).
    GeometryEdit {
        /// Target entity.
        entity: hecs::Entity,
        /// Opaque data blob describing the edit.
        edit_data: String,
    },
}

// ─── UndoAction ──────────────────────────────────────────────────────────

/// A reversible action stored on the undo/redo stack.
#[derive(Debug, Clone)]
pub enum UndoAction {
    /// A property was changed from old to new.
    PropertyChange {
        /// Target entity.
        entity: hecs::Entity,
        /// Component name.
        component_name: String,
        /// Field name.
        field_name: String,
        /// Value before the edit.
        old_value: GameValue,
        /// Value after the edit.
        new_value: GameValue,
    },
    /// A component was added.
    ComponentAdd {
        /// Target entity.
        entity: hecs::Entity,
        /// Component name.
        component_name: String,
        /// RON-serialized data (for undo = remove).
        data: String,
    },
    /// A component was removed (data saved for undo = re-add).
    ComponentRemove {
        /// Target entity.
        entity: hecs::Entity,
        /// Component name.
        component_name: String,
        /// RON-serialized data (saved before removal).
        data: String,
    },
    /// An entity was spawned.
    EntitySpawn {
        /// The spawned entity handle.
        entity: hecs::Entity,
        /// All components as (name, RON) pairs.
        components: Vec<(String, String)>,
        /// Stable identifier.
        stable_id: Uuid,
        /// Parent stable id, if any.
        parent: Option<Uuid>,
    },
    /// An entity was despawned (data saved for undo = re-spawn).
    EntityDespawn {
        /// The despawned entity handle.
        entity: hecs::Entity,
        /// All components as (name, RON) pairs (saved before removal).
        components: Vec<(String, String)>,
        /// Stable identifier.
        stable_id: Uuid,
        /// Parent stable id, if any.
        parent: Option<Uuid>,
    },
    /// An opaque geometry edit.
    GeometryEdit {
        /// Target entity.
        entity: hecs::Entity,
        /// Human-readable description.
        description: String,
    },
    /// Multiple actions coalesced into one undo step.
    Coalesced {
        /// The individual actions.
        actions: Vec<UndoAction>,
    },
}

// ─── UndoStack ───────────────────────────────────────────────────────────

/// Linear undo/redo stack with coalescing support.
pub struct UndoStack {
    undo: Vec<UndoAction>,
    redo: Vec<UndoAction>,
    coalescing: bool,
    coalesce_buffer: Vec<UndoAction>,
}

impl UndoStack {
    /// Create an empty undo stack.
    pub fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            coalescing: false,
            coalesce_buffer: Vec::new(),
        }
    }

    /// Push an action onto the undo stack. Clears the redo stack.
    ///
    /// If coalescing is active, the action is buffered instead.
    pub fn push(&mut self, action: UndoAction) {
        if self.coalescing {
            self.coalesce_buffer.push(action);
        } else {
            self.undo.push(action);
            self.redo.clear();
        }
    }

    /// Pop the most recent undo action and move it to the redo stack.
    ///
    /// Returns the action so the caller can apply its reverse.
    pub fn undo(&mut self) -> Option<UndoAction> {
        let action = self.undo.pop()?;
        self.redo.push(action.clone());
        Some(action)
    }

    /// Pop the most recent redo action and move it back to the undo stack.
    ///
    /// Returns the action so the caller can re-apply it.
    pub fn redo(&mut self) -> Option<UndoAction> {
        let action = self.redo.pop()?;
        self.undo.push(action.clone());
        Some(action)
    }

    /// Begin coalescing: subsequent `push` calls accumulate in a buffer.
    pub fn begin_coalesce(&mut self) {
        self.coalescing = true;
        self.coalesce_buffer.clear();
    }

    /// End coalescing: flush the buffer as a single `Coalesced` action.
    ///
    /// If the buffer is empty, nothing is pushed. If it contains exactly
    /// one action, that action is pushed directly (no wrapping).
    ///
    /// Deduplication: when multiple `PropertyChange` actions target the same
    /// `(entity, component_name, field_name)`, only the last edit's
    /// `new_value` is kept, combined with the first edit's `old_value`.
    /// This collapses slider drags into a single undo step.
    pub fn end_coalesce(&mut self) {
        self.coalescing = false;
        let buffer = std::mem::take(&mut self.coalesce_buffer);
        let buffer = Self::dedup_property_edits(buffer);
        match buffer.len() {
            0 => {}
            1 => {
                let action = buffer.into_iter().next().unwrap();
                self.undo.push(action);
                self.redo.clear();
            }
            _ => {
                self.undo.push(UndoAction::Coalesced { actions: buffer });
                self.redo.clear();
            }
        }
    }

    /// Deduplicate `PropertyChange` actions that target the same
    /// (entity, component_name, field_name). Keeps the `old_value` from
    /// the first occurrence and the `new_value` from the last.
    /// Non-`PropertyChange` actions pass through in order.
    fn dedup_property_edits(buffer: Vec<UndoAction>) -> Vec<UndoAction> {
        use std::collections::HashMap;

        // Key: (entity, component_name, field_name) → index in result vec
        // We use entity.to_bits() as a hashable stand-in.
        type Key = (u64, String, String);

        let mut result: Vec<UndoAction> = Vec::with_capacity(buffer.len());
        let mut seen: HashMap<Key, usize> = HashMap::new();

        for action in buffer {
            if let UndoAction::PropertyChange {
                entity,
                ref component_name,
                ref field_name,
                ..
            } = action
            {
                let key: Key = (
                    entity.to_bits().get(),
                    component_name.clone(),
                    field_name.clone(),
                );
                if let Some(&idx) = seen.get(&key) {
                    // Update the existing entry: keep old_value, replace new_value.
                    if let UndoAction::PropertyChange {
                        ref mut new_value, ..
                    } = result[idx]
                    {
                        if let UndoAction::PropertyChange {
                            new_value: incoming_new,
                            ..
                        } = action
                        {
                            *new_value = incoming_new;
                        }
                    }
                } else {
                    seen.insert(key, result.len());
                    result.push(action);
                }
            } else {
                result.push(action);
            }
        }

        result
    }

    /// Clear both stacks entirely.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.coalescing = false;
        self.coalesce_buffer.clear();
    }

    /// Whether there are actions to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Whether there are actions to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Number of actions on the undo stack.
    pub fn undo_count(&self) -> usize {
        self.undo.len()
    }

    /// Number of actions on the redo stack.
    pub fn redo_count(&self) -> usize {
        self.redo.len()
    }
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new()
    }
}

// ─── EditPipeline ────────────────────────────────────────────────────────

/// Error type for edit pipeline operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EditError {
    /// The named component is not registered.
    #[error("unknown component: {0}")]
    UnknownComponent(String),
    /// A field read/write failed.
    #[error("field error: {0}")]
    FieldError(String),
    /// The target entity does not exist.
    #[error("entity not found")]
    EntityNotFound,
    /// The entity does not have the specified component.
    #[error("component not present: {0}")]
    ComponentNotPresent(String),
    /// Serialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),
}

/// Unified edit pipeline that captures old state before applying edits,
/// producing [`UndoAction`] entries on an internal [`UndoStack`].
pub struct EditPipeline {
    undo_stack: UndoStack,
}

impl EditPipeline {
    /// Create a new edit pipeline with an empty undo stack.
    pub fn new() -> Self {
        Self {
            undo_stack: UndoStack::new(),
        }
    }

    /// Access the undo stack (read-only).
    pub fn undo_stack(&self) -> &UndoStack {
        &self.undo_stack
    }

    /// Access the undo stack (mutable).
    pub fn undo_stack_mut(&mut self) -> &mut UndoStack {
        &mut self.undo_stack
    }

    /// Apply an edit operation, capturing undo state.
    ///
    /// For `SetProperty`, reads the old value via the registry before writing.
    /// For `SpawnEntity`/`DespawnEntity`, captures component state.
    pub fn apply(
        &mut self,
        op: EditOp,
        world: &mut hecs::World,
        registry: &GameplayRegistry,
    ) -> Result<(), EditError> {
        match op {
            EditOp::SetProperty {
                entity,
                component_name,
                field_name,
                value,
            } => {
                let entry = registry
                    .component_entry(&component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;

                if !(entry.has)(world, entity) {
                    return Err(EditError::ComponentNotPresent(component_name));
                }

                let old_value = (entry.get_field)(world, entity, &field_name)
                    .map_err(EditError::FieldError)?;

                (entry.set_field)(world, entity, &field_name, value.clone())
                    .map_err(EditError::FieldError)?;

                self.undo_stack.push(UndoAction::PropertyChange {
                    entity,
                    component_name,
                    field_name,
                    old_value,
                    new_value: value,
                });
            }

            EditOp::AddComponent {
                entity,
                component_name,
                initial_data,
            } => {
                let entry = registry
                    .component_entry(&component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;

                (entry.deserialize_insert)(world, entity, &initial_data)
                    .map_err(|e| EditError::SerializationError(e.to_string()))?;

                self.undo_stack.push(UndoAction::ComponentAdd {
                    entity,
                    component_name,
                    data: initial_data,
                });
            }

            EditOp::RemoveComponent {
                entity,
                component_name,
            } => {
                let entry = registry
                    .component_entry(&component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;

                if !(entry.has)(world, entity) {
                    return Err(EditError::ComponentNotPresent(component_name));
                }

                // Serialize component data before removing.
                let data = (entry.serialize)(world, entity)
                    .ok_or_else(|| {
                        EditError::SerializationError(format!(
                            "failed to serialize {component_name}"
                        ))
                    })?;

                (entry.remove)(world, entity);

                self.undo_stack.push(UndoAction::ComponentRemove {
                    entity,
                    component_name,
                    data,
                });
            }

            EditOp::SpawnEntity { name, parent } => {
                let stable_id = StableId::new();
                let entity = world.spawn((stable_id,));

                // Resolve parent stable id if parent entity exists.
                let parent_uuid = if let Some(parent_entity) = parent {
                    world
                        .get::<&StableId>(parent_entity)
                        .ok()
                        .map(|id| id.uuid())
                } else {
                    None
                };

                self.undo_stack.push(UndoAction::EntitySpawn {
                    entity,
                    components: vec![("StableId".to_string(), format!("{}", stable_id.uuid()))],
                    stable_id: stable_id.uuid(),
                    parent: parent_uuid,
                });

                // Store name if provided (as a simple marker — callers may add
                // a Name component via a follow-up AddComponent op).
                let _ = name;
            }

            EditOp::DespawnEntity { entity } => {
                // Capture stable id before despawn.
                let stable_id = world
                    .get::<&StableId>(entity)
                    .map(|id| id.uuid())
                    .unwrap_or_else(|_| Uuid::nil());

                // Serialize all known components.
                let mut components = Vec::new();
                for entry in registry.component_entries() {
                    if (entry.has)(world, entity) {
                        if let Some(data) = (entry.serialize)(world, entity) {
                            components.push((entry.name.to_string(), data));
                        }
                    }
                }

                world
                    .despawn(entity)
                    .map_err(|_| EditError::EntityNotFound)?;

                self.undo_stack.push(UndoAction::EntityDespawn {
                    entity,
                    components,
                    stable_id,
                    parent: None,
                });
            }

            EditOp::GeometryEdit {
                entity,
                edit_data,
            } => {
                self.undo_stack.push(UndoAction::GeometryEdit {
                    entity,
                    description: edit_data,
                });
            }
        }

        Ok(())
    }

    /// Undo the most recent action.
    ///
    /// Returns the action that was undone (caller may need to apply
    /// additional side effects like geometry restoration).
    pub fn undo(
        &mut self,
        world: &mut hecs::World,
        registry: &GameplayRegistry,
    ) -> Result<Option<UndoAction>, EditError> {
        let action = match self.undo_stack.undo() {
            Some(a) => a,
            None => return Ok(None),
        };
        self.apply_reverse(&action, world, registry)?;
        Ok(Some(action))
    }

    /// Redo the most recently undone action.
    ///
    /// Returns the action that was redone.
    pub fn redo(
        &mut self,
        world: &mut hecs::World,
        registry: &GameplayRegistry,
    ) -> Result<Option<UndoAction>, EditError> {
        let action = match self.undo_stack.redo() {
            Some(a) => a,
            None => return Ok(None),
        };
        self.apply_forward(&action, world, registry)?;
        Ok(Some(action))
    }

    /// Apply an action in reverse (for undo).
    fn apply_reverse(
        &self,
        action: &UndoAction,
        world: &mut hecs::World,
        registry: &GameplayRegistry,
    ) -> Result<(), EditError> {
        match action {
            UndoAction::PropertyChange {
                entity,
                component_name,
                field_name,
                old_value,
                ..
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.set_field)(world, *entity, field_name, old_value.clone())
                    .map_err(EditError::FieldError)?;
            }

            UndoAction::ComponentAdd {
                entity,
                component_name,
                ..
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.remove)(world, *entity);
            }

            UndoAction::ComponentRemove {
                entity,
                component_name,
                data,
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.deserialize_insert)(world, *entity, data)
                    .map_err(|e| EditError::SerializationError(e.to_string()))?;
            }

            UndoAction::EntitySpawn { entity, .. } => {
                let _ = world.despawn(*entity);
            }

            UndoAction::EntityDespawn {
                components,
                stable_id,
                ..
            } => {
                let entity = world.spawn((StableId::from_uuid(*stable_id),));
                // Re-add all serialized components.
                for (name, data) in components {
                    if name == "StableId" {
                        continue; // Already added.
                    }
                    if let Some(entry) = registry.component_entry(name) {
                        let _ = (entry.deserialize_insert)(world, entity, data);
                    }
                }
            }

            UndoAction::GeometryEdit { .. } => {
                // Geometry undo is opaque — caller handles restoration.
            }

            UndoAction::Coalesced { actions } => {
                // Reverse in opposite order.
                for action in actions.iter().rev() {
                    self.apply_reverse(action, world, registry)?;
                }
            }
        }
        Ok(())
    }

    /// Apply an action forward (for redo).
    fn apply_forward(
        &self,
        action: &UndoAction,
        world: &mut hecs::World,
        registry: &GameplayRegistry,
    ) -> Result<(), EditError> {
        match action {
            UndoAction::PropertyChange {
                entity,
                component_name,
                field_name,
                new_value,
                ..
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.set_field)(world, *entity, field_name, new_value.clone())
                    .map_err(EditError::FieldError)?;
            }

            UndoAction::ComponentAdd {
                entity,
                component_name,
                data,
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.deserialize_insert)(world, *entity, data)
                    .map_err(|e| EditError::SerializationError(e.to_string()))?;
            }

            UndoAction::ComponentRemove {
                entity,
                component_name,
                ..
            } => {
                let entry = registry
                    .component_entry(component_name)
                    .ok_or_else(|| EditError::UnknownComponent(component_name.clone()))?;
                (entry.remove)(world, *entity);
            }

            UndoAction::EntitySpawn { stable_id, .. } => {
                let _entity = world.spawn((StableId::from_uuid(*stable_id),));
            }

            UndoAction::EntityDespawn { entity, .. } => {
                let _ = world.despawn(*entity);
            }

            UndoAction::GeometryEdit { .. } => {
                // Geometry redo is opaque — caller handles.
            }

            UndoAction::Coalesced { actions } => {
                for action in actions {
                    self.apply_forward(action, world, registry)?;
                }
            }
        }
        Ok(())
    }
}

impl Default for EditPipeline {
    fn default() -> Self {
        Self::new()
    }
}

