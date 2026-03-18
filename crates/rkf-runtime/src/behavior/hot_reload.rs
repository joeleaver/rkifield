//! Hot-reload cycle for behavior system dylibs.
//!
//! Orchestrates swapping a game cdylib while preserving gameplay component data:
//! serialize gameplay components, unload old dylib, load new dylib, re-register,
//! deserialize components back onto entities.

use std::path::Path;

use super::dylib_loader::{DylibError, DylibLoader};
use super::engine_components::ENGINE_COMPONENT_NAMES;
use super::registry::GameplayRegistry;

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors that can occur during a hot-reload cycle.
#[derive(Debug, Clone)]
pub enum HotReloadError {
    /// Failed to load the new dylib.
    LoadFailed(DylibError),
    /// The new dylib's `rkf_register` call failed.
    RegisterFailed(DylibError),
    /// An I/O or filesystem error.
    Io(String),
}

impl std::fmt::Display for HotReloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HotReloadError::LoadFailed(e) => write!(f, "hot-reload load failed: {e}"),
            HotReloadError::RegisterFailed(e) => write!(f, "hot-reload register failed: {e}"),
            HotReloadError::Io(msg) => write!(f, "hot-reload I/O error: {msg}"),
        }
    }
}

impl std::error::Error for HotReloadError {}

// ─── Reload report ───────────────────────────────────────────────────────────

/// Summary of a completed hot-reload cycle.
#[derive(Debug, Clone)]
pub struct ReloadReport {
    /// Number of component instances successfully serialized before unload.
    pub components_saved: usize,
    /// Number of component instances successfully restored after load.
    pub components_restored: usize,
    /// Components that failed to restore: `(component_name, error_message)`.
    pub components_failed: Vec<(String, String)>,
}

// ─── Serialized component snapshot ───────────────────────────────────────────

/// A single serialized component instance, captured before dylib unload.
pub(crate) struct SavedComponent {
    entity: hecs::Entity,
    component_name: String,
    ron_data: String,
}

// ─── Serialize helpers ───────────────────────────────────────────────────────

/// Serialize all gameplay (non-engine) components from the world.
///
/// Returns a list of serialized component snapshots and the count of
/// components that were serialized.
pub(crate) fn serialize_gameplay_components(
    world: &hecs::World,
    registry: &GameplayRegistry,
) -> Vec<SavedComponent> {
    let mut saved = Vec::new();
    for entry in registry.component_entries() {
        // Skip engine components — they survive the reload.
        if ENGINE_COMPONENT_NAMES.contains(&entry.name) {
            continue;
        }
        // Iterate all entities and check if they have this component.
        for entity in world.iter().map(|e| e.entity()) {
            if !(entry.has)(world, entity) {
                continue;
            }
            if let Some(ron_data) = (entry.serialize)(world, entity) {
                saved.push(SavedComponent {
                    entity,
                    component_name: entry.name.to_owned(),
                    ron_data,
                });
            }
        }
    }
    saved
}

/// Remove all gameplay (non-engine) components from all entities.
///
/// This runs destructors in the old dylib's code before the dylib is unloaded.
pub fn remove_gameplay_components(
    world: &mut hecs::World,
    registry: &GameplayRegistry,
) {
    // Collect entities per component first to avoid borrow issues.
    let mut to_remove: Vec<(&'static str, Vec<hecs::Entity>)> = Vec::new();
    for entry in registry.component_entries() {
        if ENGINE_COMPONENT_NAMES.contains(&entry.name) {
            continue;
        }
        let entities: Vec<hecs::Entity> = world
            .iter()
            .map(|e| e.entity())
            .filter(|&e| (entry.has)(world, e))
            .collect();
        if !entities.is_empty() {
            to_remove.push((entry.name, entities));
        }
    }
    // Now remove them.
    for (comp_name, entities) in to_remove {
        if let Some(entry) = registry.component_entry(comp_name) {
            for entity in entities {
                (entry.remove)(world, entity);
            }
        }
    }
}

/// Restore serialized components after a new dylib has been loaded and registered.
///
/// Returns `(restored_count, failures)`.
pub(crate) fn restore_gameplay_components(
    world: &mut hecs::World,
    registry: &GameplayRegistry,
    saved: &[SavedComponent],
) -> (usize, Vec<(String, String)>) {
    let mut restored = 0;
    let mut failed = Vec::new();

    for item in saved {
        match registry.component_entry(&item.component_name) {
            None => {
                log::warn!(
                    "hot-reload: component '{}' no longer registered, skipping entity {:?}",
                    item.component_name,
                    item.entity
                );
                failed.push((
                    item.component_name.clone(),
                    "component no longer registered".to_string(),
                ));
            }
            Some(entry) => {
                // Check entity still exists in the world.
                if !world.contains(item.entity) {
                    log::warn!(
                        "hot-reload: entity {:?} no longer exists, skipping component '{}'",
                        item.entity,
                        item.component_name
                    );
                    failed.push((
                        item.component_name.clone(),
                        format!("entity {:?} no longer exists", item.entity),
                    ));
                    continue;
                }
                match (entry.deserialize_insert)(world, item.entity, &item.ron_data) {
                    Ok(()) => restored += 1,
                    Err(e) => {
                        log::warn!(
                            "hot-reload: failed to restore '{}' on entity {:?}: {}",
                            item.component_name,
                            item.entity,
                            e
                        );
                        failed.push((item.component_name.clone(), e));
                    }
                }
            }
        }
    }

    (restored, failed)
}

// ─── Main hot-reload orchestrator ────────────────────────────────────────────

/// Perform a full hot-reload cycle: serialize → unload → load → register → restore.
///
/// # Arguments
/// - `world` — the ECS world containing entities with gameplay components
/// - `registry` — the gameplay registry (will be cleared of gameplay entries and repopulated)
/// - `old_loader` — the currently loaded dylib (None on first load)
/// - `new_path` — path to the new cdylib to load
///
/// # Returns
/// The new `DylibLoader` and a `ReloadReport` summarizing the operation.
pub fn hot_reload(
    world: &mut hecs::World,
    registry: &mut GameplayRegistry,
    old_loader: Option<DylibLoader>,
    new_path: &Path,
) -> Result<(DylibLoader, ReloadReport), HotReloadError> {
    // 1. Serialize gameplay components.
    let saved = serialize_gameplay_components(world, registry);
    let components_saved = saved.len();

    // 2. Remove gameplay components (runs destructors in old dylib code).
    remove_gameplay_components(world, registry);

    // 3. Unload old dylib.
    if let Some(loader) = old_loader {
        loader.unload();
    }

    // 4. Load new dylib (ABI check happens inside DylibLoader::load).
    let new_loader =
        DylibLoader::load(new_path).map_err(HotReloadError::LoadFailed)?;

    // 5. Clear gameplay entries from registry (keep engine entries).
    registry.clear_gameplay(ENGINE_COMPONENT_NAMES);

    // 6. Register new dylib's components and systems.
    new_loader
        .call_register(registry)
        .map_err(HotReloadError::RegisterFailed)?;

    // 7. Deserialize components back onto entities.
    let (components_restored, components_failed) =
        restore_gameplay_components(world, registry, &saved);

    Ok((
        new_loader,
        ReloadReport {
            components_saved,
            components_restored,
            components_failed,
        },
    ))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::engine_components::{engine_register, ENGINE_COMPONENT_NAMES};
    use crate::behavior::registry::{ComponentEntry, FieldMeta, FieldType, GameplayRegistry};
    use crate::components::Transform;

    // ── Helper: create a mock gameplay ComponentEntry backed by a String component ──

    /// A simple gameplay component for testing — just a string value.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct MockHealth {
        pub hp: f32,
    }

    fn mock_health_entry() -> ComponentEntry {
        static META: [FieldMeta; 1] = [FieldMeta {
            name: "hp",
            field_type: FieldType::Float,
            transient: false,
            range: Some((0.0, 100.0)),
            default: None,
            persist: false,
            struct_meta: None,
            asset_filter: None,
            component_filter: None,
        }];
        ComponentEntry {
            name: "MockHealth",
            serialize: |world, entity| {
                world
                    .get::<&MockHealth>(entity)
                    .ok()
                    .map(|c| ron::to_string(&*c).unwrap())
            },
            deserialize_insert: |world, entity, ron_str| {
                let c: MockHealth = ron::from_str(ron_str).map_err(|e| e.to_string())?;
                world.insert_one(entity, c).map_err(|e| e.to_string())?;
                Ok(())
            },
            has: |world, entity| world.get::<&MockHealth>(entity).is_ok(),
            remove: |world, entity| {
                let _ = world.remove_one::<MockHealth>(entity);
            },
            get_field: |_, _, _| Err("stub".into()),
            set_field: |_, _, _, _| Err("stub".into()),
            meta: &META,
        }
    }

    /// A second gameplay component for multi-component tests.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct MockArmor {
        pub defense: u32,
    }

    fn mock_armor_entry() -> ComponentEntry {
        static META: [FieldMeta; 1] = [FieldMeta {
            name: "defense",
            field_type: FieldType::Int,
            transient: false,
            range: None,
            default: None,
            persist: false,
            struct_meta: None,
            asset_filter: None,
            component_filter: None,
        }];
        ComponentEntry {
            name: "MockArmor",
            serialize: |world, entity| {
                world
                    .get::<&MockArmor>(entity)
                    .ok()
                    .map(|c| ron::to_string(&*c).unwrap())
            },
            deserialize_insert: |world, entity, ron_str| {
                let c: MockArmor = ron::from_str(ron_str).map_err(|e| e.to_string())?;
                world.insert_one(entity, c).map_err(|e| e.to_string())?;
                Ok(())
            },
            has: |world, entity| world.get::<&MockArmor>(entity).is_ok(),
            remove: |world, entity| {
                let _ = world.remove_one::<MockArmor>(entity);
            },
            get_field: |_, _, _| Err("stub".into()),
            set_field: |_, _, _, _| Err("stub".into()),
            meta: &META,
        }
    }

    // ── 1. serialize and restore roundtrip ──────────────────────────────────

    #[test]
    fn serialize_and_restore_roundtrip() {
        let mut world = hecs::World::new();
        let mut registry = GameplayRegistry::new();

        // Register engine + gameplay components.
        engine_register(&mut registry);
        registry.register_component(mock_health_entry()).unwrap();

        // Spawn an entity with both engine (Transform) and gameplay (MockHealth).
        let entity = world.spawn((Transform::default(), MockHealth { hp: 42.5 }));

        // Serialize gameplay components.
        let saved = serialize_gameplay_components(&world, &registry);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].component_name, "MockHealth");
        assert_eq!(saved[0].entity, entity);
        assert!(saved[0].ron_data.contains("42.5"));

        // Remove gameplay components.
        remove_gameplay_components(&mut world, &registry);
        assert!(!world.get::<&MockHealth>(entity).is_ok());
        // Engine component should still be present.
        assert!(world.get::<&Transform>(entity).is_ok());

        // Re-register (simulating new dylib load).
        registry.clear_gameplay(ENGINE_COMPONENT_NAMES);
        registry.register_component(mock_health_entry()).unwrap();

        // Restore.
        let (restored, failed) = restore_gameplay_components(&mut world, &registry, &saved);
        assert_eq!(restored, 1);
        assert!(failed.is_empty());

        // Verify data survived.
        let hp = world.get::<&MockHealth>(entity).unwrap();
        assert!((hp.hp - 42.5).abs() < 1e-6);
    }

    // ── 2. engine components not serialized ─────────────────────────────────

    #[test]
    fn engine_components_not_serialized() {
        let mut world = hecs::World::new();
        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);

        // Spawn entity with only engine components.
        world.spawn((Transform::default(),));

        let saved = serialize_gameplay_components(&world, &registry);
        // No gameplay components → nothing serialized.
        assert!(saved.is_empty());
    }

    // ── 3. unknown component on restore is logged, not panicked ─────────────

    #[test]
    fn unknown_component_on_restore_logged() {
        let mut world = hecs::World::new();
        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);
        registry.register_component(mock_health_entry()).unwrap();

        let entity = world.spawn((MockHealth { hp: 10.0 },));

        // Serialize with MockHealth registered.
        let saved = serialize_gameplay_components(&world, &registry);
        assert_eq!(saved.len(), 1);

        // Now clear registry and DON'T re-register MockHealth.
        registry.clear_gameplay(ENGINE_COMPONENT_NAMES);

        // Restore — MockHealth is unknown, should skip gracefully.
        let (restored, failed) = restore_gameplay_components(&mut world, &registry, &saved);
        assert_eq!(restored, 0);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "MockHealth");
        assert!(failed[0].1.contains("no longer registered"));
    }

    // ── 4. reload report counts ─────────────────────────────────────────────

    #[test]
    fn reload_report_counts() {
        let mut world = hecs::World::new();
        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);
        registry.register_component(mock_health_entry()).unwrap();
        registry.register_component(mock_armor_entry()).unwrap();

        // Spawn entities with various component combinations.
        let e1 = world.spawn((Transform::default(), MockHealth { hp: 100.0 }));
        let e2 = world.spawn((
            Transform::default(),
            MockHealth { hp: 50.0 },
            MockArmor { defense: 10 },
        ));
        // Entity with only engine component.
        world.spawn((Transform::default(),));

        // Serialize.
        let saved = serialize_gameplay_components(&world, &registry);
        // e1: MockHealth, e2: MockHealth + MockArmor = 3 total.
        assert_eq!(saved.len(), 3);

        // Remove gameplay components.
        remove_gameplay_components(&mut world, &registry);
        assert!(!world.get::<&MockHealth>(e1).is_ok());
        assert!(!world.get::<&MockHealth>(e2).is_ok());
        assert!(!world.get::<&MockArmor>(e2).is_ok());

        // Re-register only MockHealth (not MockArmor) — simulates component removal.
        registry.clear_gameplay(ENGINE_COMPONENT_NAMES);
        registry.register_component(mock_health_entry()).unwrap();

        // Restore.
        let (restored, failed) = restore_gameplay_components(&mut world, &registry, &saved);
        assert_eq!(restored, 2); // Two MockHealth instances restored.
        assert_eq!(failed.len(), 1); // One MockArmor failed (not registered).
        assert_eq!(failed[0].0, "MockArmor");

        // Build a report manually.
        let report = ReloadReport {
            components_saved: saved.len(),
            components_restored: restored,
            components_failed: failed,
        };
        assert_eq!(report.components_saved, 3);
        assert_eq!(report.components_restored, 2);
        assert_eq!(report.components_failed.len(), 1);
    }

    // ── 5. clear_gameplay retains engine entries ─────────────────────────────

    #[test]
    fn clear_gameplay_retains_engine_entries() {
        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);
        registry.register_component(mock_health_entry()).unwrap();
        assert_eq!(registry.component_count(), 5); // 4 engine + 1 gameplay

        registry.clear_gameplay(ENGINE_COMPONENT_NAMES);
        assert_eq!(registry.component_count(), 4); // Only engine components remain
        assert!(registry.has_component("Transform"));
        assert!(registry.has_component("CameraComponent"));
        assert!(registry.has_component("FogVolumeComponent"));
        assert!(registry.has_component("EditorMetadata"));
        assert!(!registry.has_component("MockHealth"));
    }

    // ── 6. deserialize failure skips gracefully ─────────────────────────────

    #[test]
    fn deserialize_failure_skips_gracefully() {
        let mut world = hecs::World::new();
        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);
        registry.register_component(mock_health_entry()).unwrap();

        let entity = world.spawn((MockHealth { hp: 99.0 },));

        let saved = serialize_gameplay_components(&world, &registry);
        remove_gameplay_components(&mut world, &registry);

        // Re-register with a mock entry that always fails to deserialize.
        registry.clear_gameplay(ENGINE_COMPONENT_NAMES);
        registry
            .register_component(ComponentEntry {
                name: "MockHealth",
                serialize: |_, _| None,
                deserialize_insert: |_, _, _| Err("intentional failure".to_string()),
                has: |_, _| false,
                remove: |_, _| {},
                get_field: |_, _, _| Err("stub".into()),
                set_field: |_, _, _, _| Err("stub".into()),
                meta: &[],
            })
            .unwrap();

        let (restored, failed) = restore_gameplay_components(&mut world, &registry, &saved);
        assert_eq!(restored, 0);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, "MockHealth");
        assert!(failed[0].1.contains("intentional failure"));

        // Entity should still exist (not despawned).
        assert!(world.contains(entity));
    }
}
