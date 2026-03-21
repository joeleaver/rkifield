//! Uniform treatment verification — ensures engine components flow through the
//! same code paths as gameplay components.
//!
//! These utilities verify that engine components (those with `engine: true`) are
//! properly registered in the [`GameplayRegistry`] and work correctly with the
//! inspector, edit pipeline, and field-level get/set.

use super::inspector::build_inspector_data;
use super::registry::GameplayRegistry;

/// Verifies that `get_field` and `set_field` work for each registered engine
/// component's fields.
///
/// For each engine component present on the given entity, reads every field via
/// `get_field`, then writes it back via `set_field`. Returns a list of
/// `(component_name, error_message)` for any failures.
pub fn verify_engine_component_fields(
    world: &mut hecs::World,
    entity: hecs::Entity,
    registry: &GameplayRegistry,
) -> Vec<(String, String)> {
    let mut failures = Vec::new();

    for entry in registry.component_entries() {
        if !entry.engine {
            continue;
        }

        if !(entry.has)(world, entity) {
            continue; // Component not on entity — skip, not a failure.
        }

        for field_meta in entry.meta.iter() {
            // Read the field value.
            let value = match (entry.get_field)(world, entity, field_meta.name) {
                Ok(v) => v,
                Err(e) => {
                    failures.push((
                        entry.name.to_string(),
                        format!("get_field('{}') failed: {}", field_meta.name, e),
                    ));
                    continue;
                }
            };

            // Write the same value back (roundtrip).
            if let Err(e) = (entry.set_field)(world, entity, field_meta.name, value) {
                failures.push((
                    entry.name.to_string(),
                    format!("set_field('{}') failed: {}", field_meta.name, e),
                ));
            }
        }
    }

    failures
}

/// Verifies that `build_inspector_data` includes engine components that are
/// present on the entity.
///
/// Returns `true` if every engine component present on the entity appears in
/// the inspector data.
pub fn verify_inspector_renders_engine_components(
    world: &hecs::World,
    entity: hecs::Entity,
    registry: &GameplayRegistry,
) -> bool {
    let data = build_inspector_data(world, entity, registry);

    for entry in registry.component_entries() {
        if !entry.engine {
            continue;
        }

        if !(entry.has)(world, entity) {
            continue; // Not on entity — skip.
        }

        // Verify this component appears in inspector data.
        let found = data.components.iter().any(|c| c.name == entry.name);
        if !found {
            return false;
        }
    }

    true
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::engine_components::engine_register;
    use crate::components::{CameraComponent, EditorMetadata, FogVolumeComponent, Transform};
    use glam::{Quat, Vec3};
    use rkf_core::WorldPosition;

    fn test_registry() -> GameplayRegistry {
        let mut reg = GameplayRegistry::new();
        engine_register(&mut reg);
        reg
    }

    #[test]
    fn engine_component_get_set_roundtrip() {
        let registry = test_registry();
        let mut world = hecs::World::new();

        let entity = world.spawn((
            Transform {
                position: WorldPosition::new(glam::IVec3::new(1, 2, 3), Vec3::new(0.5, 1.0, 1.5)),
                rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
                scale: Vec3::new(2.0, 3.0, 4.0),
            },
            CameraComponent {
                fov_degrees: 90.0,
                near: 0.1,
                far: 1000.0,
                active: true,
                label: "Main".to_string(),
                yaw: 45.0,
                pitch: -10.0,
                ..Default::default()
            },
            FogVolumeComponent {
                density: 0.5,
                color: [1.0, 1.0, 1.0],
                phase_g: 0.3,
                half_extents: Vec3::new(10.0, 5.0, 3.0),
            },
            EditorMetadata {
                name: "TestObj".to_string(),
                tags: vec!["test".to_string()],
                locked: false,
            },
        ));

        let failures = verify_engine_component_fields(&mut world, entity, &registry);
        assert!(
            failures.is_empty(),
            "field roundtrip failures: {:?}",
            failures
        );
    }

    #[test]
    fn inspector_includes_engine_components() {
        let registry = test_registry();
        let mut world = hecs::World::new();

        let entity = world.spawn((
            Transform::default(),
            CameraComponent::default(),
            FogVolumeComponent::default(),
            EditorMetadata::default(),
        ));

        assert!(verify_inspector_renders_engine_components(
            &world, entity, &registry
        ));
    }

    #[test]
    fn partial_entity_still_passes() {
        // Entity with only Transform — other engine components are absent but
        // that's not a failure (they're just not on this entity).
        let registry = test_registry();
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));

        let failures = verify_engine_component_fields(&mut world, entity, &registry);
        assert!(failures.is_empty());

        assert!(verify_inspector_renders_engine_components(
            &world, entity, &registry
        ));
    }
}
