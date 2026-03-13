//! Bridge between the editor's tool operations and the behavior system's
//! [`EditPipeline`] via [`ToolEditMapping`].
//!
//! The editor has its own undo system ([`crate::undo::UndoStack`]) that handles
//! transforms, sculpt strokes, paint strokes, etc. This module provides an
//! **optional** integration point that mirrors completed tool actions into the
//! behavior system's [`EditPipeline`], so that behavior scripts, the game state
//! store, and other pipeline consumers can observe and react to editor edits.
//!
//! This is additive — the editor's existing undo/redo and tool processing
//! continue to work exactly as before. The functions here are called *after*
//! the editor has already applied and recorded the action.

use glam::Vec3;
use rkf_runtime::behavior::{EditPipeline, GameplayRegistry, ToolEditMapping};
use rkf_runtime::behavior::game_value::GameValue;

/// Route a completed gizmo transform into the behavior system's edit pipeline.
///
/// Call this after the editor has already applied the transform and pushed its
/// own `UndoAction::Transform`. The `entity` must be the hecs entity
/// corresponding to the edited object (from `World::entity_for_object`).
///
/// Returns `Ok(())` if the pipeline accepted the edit, or `Err` if the
/// component/field is not registered in the gameplay registry (expected when
/// no gameplay components are registered — the error is safe to ignore).
pub fn route_gizmo_transform(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    entity: hecs::Entity,
    field: &str,
    old_value: Vec3,
    new_value: Vec3,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::gizmo_transform_edit(
        entity,
        field,
        GameValue::Vec3(old_value),
        GameValue::Vec3(new_value),
    );
    pipeline.apply(op, world, registry)
}

/// Route a completed sculpt stroke into the behavior system's edit pipeline.
///
/// Call this after the editor has already applied the sculpt edits and pushed
/// its own `UndoAction::SculptStroke`. The `geometry_data` is an opaque
/// descriptor (e.g. serialized snapshot metadata) — the actual geometry
/// restoration is still handled by the editor's own undo system.
pub fn route_sculpt_edit(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    entity: hecs::Entity,
    geometry_data: Vec<u8>,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::sculpt_edit(entity, geometry_data);
    pipeline.apply(op, world, registry)
}

/// Route a completed paint stroke into the behavior system's edit pipeline.
///
/// Call this after the editor has already applied the paint edits and pushed
/// its own `UndoAction::PaintStroke`.
pub fn route_paint_edit(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    entity: hecs::Entity,
    geometry_data: Vec<u8>,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::paint_edit(entity, geometry_data);
    pipeline.apply(op, world, registry)
}

/// Route a spawn action into the behavior system's edit pipeline.
///
/// Call this after the editor has spawned an entity and pushed its own
/// `UndoAction::SpawnEntity`.
pub fn route_spawn(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    name: String,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::spawn_edit(name);
    pipeline.apply(op, world, registry)
}

/// Route a despawn action into the behavior system's edit pipeline.
///
/// Call this after the editor has despawned an entity and pushed its own
/// `UndoAction::DespawnEntity`.
pub fn route_despawn(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    entity: hecs::Entity,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::despawn_edit(entity);
    pipeline.apply(op, world, registry)
}

/// Route an inspector property change into the behavior system's edit pipeline.
///
/// Call this after the editor has applied a property change via the inspector
/// panel (e.g. `SetObjectPosition`, `SetLightIntensity`).
pub fn route_inspector_edit(
    pipeline: &mut EditPipeline,
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    entity: hecs::Entity,
    component: &str,
    field: &str,
    old_value: GameValue,
    new_value: GameValue,
) -> Result<(), rkf_runtime::behavior::EditError> {
    let op = ToolEditMapping::inspector_field_edit(entity, component, field, old_value, new_value);
    pipeline.apply(op, world, registry)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_gizmo_returns_error_for_unregistered_component() {
        let mut pipeline = EditPipeline::new();
        let mut world = hecs::World::new();
        let registry = GameplayRegistry::new();
        let entity = world.spawn(());

        // Transform is not registered in an empty registry, so this should
        // return UnknownComponent — which is the expected behavior when no
        // gameplay components are set up.
        let result = route_gizmo_transform(
            &mut pipeline,
            &mut world,
            &registry,
            entity,
            "position",
            Vec3::ZERO,
            Vec3::new(1.0, 2.0, 3.0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn route_sculpt_pushes_geometry_edit() {
        let mut pipeline = EditPipeline::new();
        let mut world = hecs::World::new();
        let registry = GameplayRegistry::new();
        let entity = world.spawn(());

        // GeometryEdit does not require registry lookup, so it always succeeds.
        let result = route_sculpt_edit(
            &mut pipeline,
            &mut world,
            &registry,
            entity,
            vec![1, 2, 3],
        );
        assert!(result.is_ok());
        assert_eq!(pipeline.undo_stack().undo_count(), 1);
    }

    #[test]
    fn route_paint_pushes_geometry_edit() {
        let mut pipeline = EditPipeline::new();
        let mut world = hecs::World::new();
        let registry = GameplayRegistry::new();
        let entity = world.spawn(());

        let result = route_paint_edit(
            &mut pipeline,
            &mut world,
            &registry,
            entity,
            vec![10, 20],
        );
        assert!(result.is_ok());
        assert_eq!(pipeline.undo_stack().undo_count(), 1);
    }

    #[test]
    fn route_spawn_creates_entity_in_pipeline() {
        let mut pipeline = EditPipeline::new();
        let mut world = hecs::World::new();
        let registry = GameplayRegistry::new();

        let result = route_spawn(
            &mut pipeline,
            &mut world,
            &registry,
            "TestObject".to_string(),
        );
        assert!(result.is_ok());
        assert_eq!(pipeline.undo_stack().undo_count(), 1);
    }

    #[test]
    fn route_despawn_pushes_to_pipeline() {
        let mut pipeline = EditPipeline::new();
        let mut world = hecs::World::new();
        let registry = GameplayRegistry::new();
        let entity = world.spawn(());

        let result = route_despawn(
            &mut pipeline,
            &mut world,
            &registry,
            entity,
        );
        assert!(result.is_ok());
        assert_eq!(pipeline.undo_stack().undo_count(), 1);
    }
}
