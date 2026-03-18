//! Inspector data model for the editor UI.
//!
//! Provides [`InspectorData`] — a snapshot of an entity's components and fields,
//! driven by [`ComponentMeta`] / [`ComponentEntry`] introspection. The editor UI
//! consumes these data types to render the property inspector without directly
//! touching hecs or the registry.

use super::game_value::GameValue;
use super::registry::{ComponentEntry, FieldType, GameplayRegistry};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Component names that cannot be removed from an entity via the inspector.
pub const MANDATORY_COMPONENTS: &[&str] = &["Transform", "EditorMetadata"];

/// Engine components that are rendered by dedicated UI panels (ObjectProperties).
/// The generic component inspector should hide these to avoid duplication.
pub const ENGINE_UI_COMPONENTS: &[&str] = &[
    "Transform", "EditorMetadata", "SdfTree",
    "EnvironmentSettings", "SceneEnvironment",
];

// ─── Inspector data types ───────────────────────────────────────────────────

/// Complete inspector snapshot for a single entity.
#[derive(Debug, Clone)]
pub struct InspectorData {
    /// The entity being inspected.
    pub entity: hecs::Entity,
    /// Per-component data, one entry per component present on the entity.
    pub components: Vec<ComponentInspectorData>,
}

/// Inspector data for a single component on an entity.
#[derive(Debug, Clone)]
pub struct ComponentInspectorData {
    /// Component type name (e.g., "Transform", "Health").
    pub name: String,
    /// Per-field data.
    pub fields: Vec<FieldInspectorData>,
    /// Whether this component can be removed from the entity.
    /// False for mandatory components (Transform, EditorMetadata).
    pub removable: bool,
}

/// Inspector data for a single field on a component.
#[derive(Debug, Clone)]
pub struct FieldInspectorData {
    /// Field name (e.g., "position", "health").
    pub name: String,
    /// Type classification for rendering.
    pub field_type: FieldType,
    /// Current value of the field.
    pub value: GameValue,
    /// Optional numeric range for slider display `(min, max)`.
    pub range: Option<(f64, f64)>,
    /// True if the field is transient (runtime state, not persisted).
    pub transient: bool,
    /// Sub-fields for `FieldType::Struct` fields. `None` for non-struct fields.
    pub sub_fields: Option<Vec<FieldInspectorData>>,
    /// File extension filter for `FieldType::AssetRef` fields.
    pub asset_filter: Option<String>,
    /// Required component type for `FieldType::ComponentRef` fields.
    pub component_filter: Option<String>,
}

// ─── Build inspector data ───────────────────────────────────────────────────

/// Build a complete [`InspectorData`] snapshot for the given entity.
///
/// Queries all registered components on the entity, reads field values via
/// `ComponentEntry.get_field`, and populates metadata from `ComponentEntry.meta`.
pub fn build_inspector_data(
    world: &hecs::World,
    entity: hecs::Entity,
    registry: &GameplayRegistry,
) -> InspectorData {
    let mut components = Vec::new();

    // Collect component names and sort for stable ordering.
    let mut names: Vec<&str> = registry.component_names().collect();
    names.sort();

    for name in names {
        let entry = match registry.component_entry(name) {
            Some(e) => e,
            None => continue,
        };

        // Skip components not present on this entity.
        if !(entry.has)(world, entity) {
            continue;
        }

        let comp_data = build_component_data(world, entity, entry);
        components.push(comp_data);
    }

    InspectorData { entity, components }
}

/// Build inspector data for a single component.
fn build_component_data(
    world: &hecs::World,
    entity: hecs::Entity,
    entry: &ComponentEntry,
) -> ComponentInspectorData {
    let removable = !MANDATORY_COMPONENTS.contains(&entry.name);

    let mut fields = Vec::new();
    for meta in entry.meta.iter() {
        let value = match (entry.get_field)(world, entity, meta.name) {
            Ok(v) => v,
            Err(_) => continue, // Skip fields we can't read.
        };

        fields.push(build_field_inspector_data(meta, value));
    }

    ComponentInspectorData {
        name: entry.name.to_string(),
        fields,
        removable,
    }
}

/// Build inspector data for a single field, recursing into struct sub-fields.
fn build_field_inspector_data(
    meta: &super::registry::FieldMeta,
    value: GameValue,
) -> FieldInspectorData {
    let sub_fields = if meta.field_type == FieldType::Struct {
        if let (Some(struct_meta), Some(struct_fields)) = (meta.struct_meta, value.as_struct()) {
            Some(
                struct_meta
                    .fields
                    .iter()
                    .filter_map(|sub_meta| {
                        let sub_val = struct_fields
                            .iter()
                            .find(|(n, _)| n == sub_meta.name)
                            .map(|(_, v)| v.clone())?;
                        Some(build_field_inspector_data(sub_meta, sub_val))
                    })
                    .collect(),
            )
        } else {
            None
        }
    } else {
        None
    };

    FieldInspectorData {
        name: meta.name.to_string(),
        field_type: meta.field_type,
        value,
        range: meta.range,
        transient: meta.transient,
        sub_fields,
        asset_filter: meta.asset_filter.map(|s| s.to_string()),
        component_filter: meta.component_filter.map(|s| s.to_string()),
    }
}

// ─── Add/Remove component logic ─────────────────────────────────────────────

/// Returns the names of components registered in the registry that are NOT
/// currently present on the given entity.
pub fn available_components_for_entity(
    world: &hecs::World,
    entity: hecs::Entity,
    registry: &GameplayRegistry,
) -> Vec<String> {
    let mut result: Vec<String> = registry
        .component_names()
        .filter(|name| {
            registry
                .component_entry(name)
                .map_or(false, |entry| !(entry.has)(world, entity))
        })
        .map(|s| s.to_string())
        .collect();
    result.sort();
    result
}

/// Add a component to an entity using its default RON representation.
///
/// Uses `ComponentEntry.deserialize_insert` with a default RON string. The
/// default RON is the simplest valid representation for each engine component
/// type (typically `Type()`). For gameplay components, this relies on `#[derive(Default)]`.
pub fn add_component_default(
    world: &mut hecs::World,
    entity: hecs::Entity,
    component_name: &str,
    registry: &GameplayRegistry,
) -> Result<(), String> {
    let entry = registry
        .component_entry(component_name)
        .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

    if (entry.has)(world, entity) {
        return Err(format!(
            "entity already has component '{component_name}'"
        ));
    }

    // Attempt to deserialize with default RON. For engine components that
    // derive Default + Serialize, `TypeName(field: default, ...)` works.
    // We try the simplest patterns: named-struct default, then unit-struct.
    let default_ron = format!("{component_name}()");
    (entry.deserialize_insert)(world, entity, &default_ron)
        .map_err(|e| format!("failed to add default '{component_name}': {e}"))
}

/// Remove a component from an entity.
///
/// Refuses to remove mandatory components (Transform, EditorMetadata).
pub fn remove_component(
    world: &mut hecs::World,
    entity: hecs::Entity,
    component_name: &str,
    registry: &GameplayRegistry,
) -> Result<(), String> {
    if MANDATORY_COMPONENTS.contains(&component_name) {
        return Err(format!(
            "cannot remove mandatory component '{component_name}'"
        ));
    }

    let entry = registry
        .component_entry(component_name)
        .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

    if !(entry.has)(world, entity) {
        return Err(format!(
            "entity does not have component '{component_name}'"
        ));
    }

    (entry.remove)(world, entity);
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::engine_components::engine_register;
    use crate::components::{EditorMetadata, Transform};
    use glam::Vec3;
    use rkf_core::WorldPosition;

    /// Helper: create a registry with engine components registered.
    fn test_registry() -> GameplayRegistry {
        let mut reg = GameplayRegistry::new();
        engine_register(&mut reg);
        reg
    }

    // ── build_inspector_data_with_engine_components ──────────────────────

    #[test]
    fn build_inspector_data_with_engine_components() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform {
                position: WorldPosition::new(glam::IVec3::ZERO, Vec3::new(1.0, 2.0, 3.0)),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            EditorMetadata {
                name: "TestObj".to_string(),
                tags: vec![],
                locked: false,
            },
        ));

        let data = build_inspector_data(&world, entity, &registry);
        assert_eq!(data.entity, entity);
        assert_eq!(data.components.len(), 2);

        // Find the Transform component data.
        let transform_data = data
            .components
            .iter()
            .find(|c| c.name == "Transform")
            .expect("should have Transform");
        assert_eq!(transform_data.fields.len(), 3);
        assert!(!transform_data.removable);

        // Verify position field.
        let pos_field = transform_data
            .fields
            .iter()
            .find(|f| f.name == "position")
            .expect("should have position field");
        assert_eq!(pos_field.field_type, FieldType::WorldPosition);
        assert!(!pos_field.transient);

        // Find the EditorMetadata component data.
        let meta_data = data
            .components
            .iter()
            .find(|c| c.name == "EditorMetadata")
            .expect("should have EditorMetadata");
        assert_eq!(meta_data.fields.len(), 2);
        assert!(!meta_data.removable);

        // Verify name field value.
        let name_field = meta_data
            .fields
            .iter()
            .find(|f| f.name == "name")
            .expect("should have name field");
        assert_eq!(name_field.value, GameValue::String("TestObj".to_string()));
    }

    // ── mandatory_components_not_removable ───────────────────────────────

    #[test]
    fn mandatory_components_not_removable() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            EditorMetadata::default(),
        ));

        let data = build_inspector_data(&world, entity, &registry);

        for comp in &data.components {
            if comp.name == "Transform" || comp.name == "EditorMetadata" {
                assert!(
                    !comp.removable,
                    "{} should not be removable",
                    comp.name
                );
            }
        }
    }

    // ── available_components_excludes_existing ───────────────────────────

    #[test]
    fn available_components_excludes_existing() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let available = available_components_for_entity(&world, entity, &registry);

        // Transform is on the entity — should NOT be in the list.
        assert!(
            !available.contains(&"Transform".to_string()),
            "Transform should be excluded"
        );

        // CameraComponent, EditorMetadata, FogVolumeComponent are NOT on the entity.
        assert!(
            available.contains(&"CameraComponent".to_string()),
            "CameraComponent should be available"
        );
        assert!(
            available.contains(&"EditorMetadata".to_string()),
            "EditorMetadata should be available"
        );
        assert!(
            available.contains(&"FogVolumeComponent".to_string()),
            "FogVolumeComponent should be available"
        );
    }

    // ── remove_component refuses mandatory ──────────────────────────────

    #[test]
    fn remove_component_refuses_mandatory() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let err = remove_component(&mut world, entity, "Transform", &registry).unwrap_err();
        assert!(err.contains("mandatory"));
    }

    // ── remove_component works for non-mandatory ────────────────────────

    #[test]
    fn remove_component_works_for_non_mandatory() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            crate::components::CameraComponent::default(),
        ));

        assert!(world.get::<&crate::components::CameraComponent>(entity).is_ok());
        remove_component(&mut world, entity, "CameraComponent", &registry).unwrap();
        assert!(world.get::<&crate::components::CameraComponent>(entity).is_err());
    }

    // ── remove_component errors for missing component ───────────────────

    #[test]
    fn remove_component_errors_for_missing() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let err = remove_component(&mut world, entity, "CameraComponent", &registry).unwrap_err();
        assert!(err.contains("does not have"));
    }

    // ── remove_component errors for unknown component ───────────────────

    #[test]
    fn remove_component_errors_for_unknown() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let err = remove_component(&mut world, entity, "Nonexistent", &registry).unwrap_err();
        assert!(err.contains("unknown component"));
    }

    // ── add_component_default errors for already present ────────────────

    #[test]
    fn add_component_default_errors_if_already_present() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let err = add_component_default(&mut world, entity, "Transform", &registry).unwrap_err();
        assert!(err.contains("already has"));
    }

    // ── add_component_default errors for unknown ────────────────────────

    #[test]
    fn add_component_default_errors_for_unknown() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn(());

        let err = add_component_default(&mut world, entity, "Nonexistent", &registry).unwrap_err();
        assert!(err.contains("unknown component"));
    }

    // ── inspector data for entity with no registered components ─────────

    #[test]
    fn inspector_data_empty_for_bare_entity() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn(());

        let data = build_inspector_data(&world, entity, &registry);
        assert_eq!(data.components.len(), 0);
    }

    // ── non-mandatory components are removable ──────────────────────────

    #[test]
    fn non_mandatory_components_are_removable() {
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((
            crate::components::CameraComponent::default(),
            crate::components::FogVolumeComponent::default(),
        ));

        let data = build_inspector_data(&world, entity, &registry);

        for comp in &data.components {
            assert!(
                comp.removable,
                "{} should be removable",
                comp.name
            );
        }
    }
}
