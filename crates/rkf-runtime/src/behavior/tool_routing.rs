//! Tool-to-EditOp routing — maps editor tool operations to EditOp variants.
//!
//! Each editor tool (gizmo, inspector, sculpt, paint, MCP) produces domain-
//! specific events. This module converts those events into [`EditOp`] variants
//! so that all mutations flow through the unified [`EditPipeline`].

use super::edit_pipeline::EditOp;
use super::game_value::GameValue;

/// Maps editor tool operations to [`EditOp`] variants.
///
/// This is a stateless mapping layer — it does not apply edits, only creates
/// the correct `EditOp` for a given tool action. The caller feeds the returned
/// `EditOp` into [`EditPipeline::apply`].
pub struct ToolEditMapping;

impl ToolEditMapping {
    /// Create an `EditOp::SetProperty` for a gizmo transform drag.
    ///
    /// `field` is one of `"position"`, `"rotation"`, or `"scale"`.
    pub fn gizmo_transform_edit(
        entity: hecs::Entity,
        field: &str,
        _old_value: GameValue,
        new_value: GameValue,
    ) -> EditOp {
        EditOp::SetProperty {
            entity,
            component_name: "Transform".to_string(),
            field_name: field.to_string(),
            value: new_value,
        }
    }

    /// Create an `EditOp::SetProperty` for an inspector field change.
    pub fn inspector_field_edit(
        entity: hecs::Entity,
        component: &str,
        field: &str,
        _old_value: GameValue,
        new_value: GameValue,
    ) -> EditOp {
        EditOp::SetProperty {
            entity,
            component_name: component.to_string(),
            field_name: field.to_string(),
            value: new_value,
        }
    }

    /// Create an `EditOp::GeometryEdit` for a sculpt operation.
    ///
    /// `geometry_data` is an opaque blob describing the sculpt stroke.
    pub fn sculpt_edit(entity: hecs::Entity, geometry_data: Vec<u8>) -> EditOp {
        // Encode as a descriptive string — the geometry edit is opaque to the
        // pipeline; actual geometry restoration is handled by the caller.
        let description = format!("sculpt:{}", geometry_data.len());
        EditOp::GeometryEdit {
            entity,
            edit_data: description,
        }
    }

    /// Create an `EditOp::GeometryEdit` for a paint operation.
    ///
    /// `geometry_data` is an opaque blob describing the paint stroke.
    pub fn paint_edit(entity: hecs::Entity, geometry_data: Vec<u8>) -> EditOp {
        let description = format!("paint:{}", geometry_data.len());
        EditOp::GeometryEdit {
            entity,
            edit_data: description,
        }
    }

    /// Create an `EditOp::SetProperty` for an MCP mutation.
    pub fn mcp_mutation_edit(
        entity: hecs::Entity,
        component: &str,
        field: &str,
        _old_value: GameValue,
        new_value: GameValue,
    ) -> EditOp {
        EditOp::SetProperty {
            entity,
            component_name: component.to_string(),
            field_name: field.to_string(),
            value: new_value,
        }
    }

    /// Create an `EditOp::SpawnEntity`.
    pub fn spawn_edit(name: String) -> EditOp {
        EditOp::SpawnEntity {
            name,
            parent: None,
        }
    }

    /// Create an `EditOp::DespawnEntity`.
    pub fn despawn_edit(entity: hecs::Entity) -> EditOp {
        EditOp::DespawnEntity { entity }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn gizmo_creates_set_property() {
        let entity = hecs::Entity::DANGLING;
        let old = GameValue::Vec3(Vec3::ZERO);
        let new = GameValue::Vec3(Vec3::new(1.0, 2.0, 3.0));

        let op = ToolEditMapping::gizmo_transform_edit(entity, "position", old, new.clone());

        match op {
            EditOp::SetProperty {
                component_name,
                field_name,
                value,
                ..
            } => {
                assert_eq!(component_name, "Transform");
                assert_eq!(field_name, "position");
                assert_eq!(value, new);
            }
            _ => panic!("expected SetProperty, got {:?}", op),
        }
    }

    #[test]
    fn inspector_creates_set_property() {
        let entity = hecs::Entity::DANGLING;
        let old = GameValue::Float(0.5);
        let new = GameValue::Float(1.5);

        let op = ToolEditMapping::inspector_field_edit(
            entity,
            "FogVolumeComponent",
            "density",
            old,
            new.clone(),
        );

        match op {
            EditOp::SetProperty {
                component_name,
                field_name,
                value,
                ..
            } => {
                assert_eq!(component_name, "FogVolumeComponent");
                assert_eq!(field_name, "density");
                assert_eq!(value, new);
            }
            _ => panic!("expected SetProperty, got {:?}", op),
        }
    }

    #[test]
    fn sculpt_creates_geometry_edit() {
        let entity = hecs::Entity::DANGLING;
        let data = vec![1, 2, 3, 4, 5];

        let op = ToolEditMapping::sculpt_edit(entity, data);

        match op {
            EditOp::GeometryEdit { edit_data, .. } => {
                assert!(edit_data.starts_with("sculpt:"));
                assert!(edit_data.contains("5"));
            }
            _ => panic!("expected GeometryEdit, got {:?}", op),
        }
    }

    #[test]
    fn paint_creates_geometry_edit() {
        let entity = hecs::Entity::DANGLING;
        let data = vec![10, 20, 30];

        let op = ToolEditMapping::paint_edit(entity, data);

        match op {
            EditOp::GeometryEdit { edit_data, .. } => {
                assert!(edit_data.starts_with("paint:"));
                assert!(edit_data.contains("3"));
            }
            _ => panic!("expected GeometryEdit, got {:?}", op),
        }
    }

    #[test]
    fn spawn_creates_spawn_entity() {
        let op = ToolEditMapping::spawn_edit("MyEntity".to_string());

        match op {
            EditOp::SpawnEntity { name, parent } => {
                assert_eq!(name, "MyEntity");
                assert!(parent.is_none());
            }
            _ => panic!("expected SpawnEntity, got {:?}", op),
        }
    }

    #[test]
    fn despawn_creates_despawn_entity() {
        let entity = hecs::Entity::DANGLING;
        let op = ToolEditMapping::despawn_edit(entity);

        match op {
            EditOp::DespawnEntity { entity: e } => {
                assert_eq!(e, entity);
            }
            _ => panic!("expected DespawnEntity, got {:?}", op),
        }
    }

    #[test]
    fn mcp_creates_set_property() {
        let entity = hecs::Entity::DANGLING;
        let old = GameValue::String("old_name".to_string());
        let new = GameValue::String("new_name".to_string());

        let op = ToolEditMapping::mcp_mutation_edit(
            entity,
            "EditorMetadata",
            "name",
            old,
            new.clone(),
        );

        match op {
            EditOp::SetProperty {
                component_name,
                field_name,
                value,
                ..
            } => {
                assert_eq!(component_name, "EditorMetadata");
                assert_eq!(field_name, "name");
                assert_eq!(value, new);
            }
            _ => panic!("expected SetProperty, got {:?}", op),
        }
    }
}
