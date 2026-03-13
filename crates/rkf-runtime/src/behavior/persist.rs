//! Persistence helpers for syncing component fields to/from the GameStore.
//!
//! Two approaches are available:
//!
//! 1. **Manual `Persistable` trait** — implement `sync_to_store`/`sync_from_store`
//!    directly for full control. Best for components with complex serialization
//!    (e.g., entity references, derived state).
//!
//! 2. **Automatic registry-driven persistence** via [`auto_sync_to_store`] and
//!    [`auto_sync_from_store`]. These use the component's `FieldMeta` (specifically
//!    `persist: true` fields) and the `get_field`/`set_field` function pointers on
//!    `ComponentEntry` to sync values without per-type code.
//!
//! # The `#[persist]` pattern
//!
//! With the `#[component]` proc macro, annotate fields with `#[persist]`:
//!
//! ```ignore
//! #[component]
//! pub struct Health {
//!     #[persist]
//!     pub current: f32,
//!     pub max: f32,  // design-time constant, not saved
//! }
//! ```
//!
//! The macro sets `FieldMeta { persist: true, .. }` for annotated fields.
//! Then call `auto_sync_to_store` / `auto_sync_from_store` with the component's
//! `ComponentEntry` to persist all marked fields automatically.
//!
//! Without the proc macro (manual `ComponentEntry`), set `persist: true` in the
//! `FieldMeta` declaration directly:
//!
//! ```ignore
//! static HEALTH_FIELDS: [FieldMeta; 2] = [
//!     FieldMeta {
//!         name: "current",
//!         field_type: FieldType::Float,
//!         transient: false,
//!         range: Some((0.0, 1000.0)),
//!         default: None,
//!         persist: true,  // <-- equivalent of #[persist]
//!     },
//!     FieldMeta {
//!         name: "max",
//!         field_type: FieldType::Float,
//!         transient: false,
//!         range: Some((0.0, 1000.0)),
//!         default: None,
//!         persist: false, // not persisted
//!     },
//! ];
//! ```
//!
//! # Store key format
//!
//! Keys follow the pattern `"entity/{stable_id}/{component_name}/{field_name}"`,
//! built by [`persist_key`] (simple) or [`persist_component_key`] (component-scoped).

use super::game_store::GameStore;
use super::registry::{ComponentEntry, FieldMeta};

/// Build a store key for a persisted field on an entity.
///
/// Format: `"entity/{stable_id}/{field_name}"`
///
/// This is the simple key format used by manual `Persistable` implementations.
/// For registry-driven auto-sync, use [`persist_component_key`] instead to
/// avoid name collisions between components with the same field name.
///
/// # Example
/// ```ignore
/// let key = persist_key("a1b2c3d4-...", "health");
/// // "entity/a1b2c3d4-.../health"
/// ```
pub fn persist_key(stable_id: &str, field_name: &str) -> String {
    format!("entity/{stable_id}/{field_name}")
}

/// Build a component-scoped store key for a persisted field.
///
/// Format: `"entity/{stable_id}/{component_name}/{field_name}"`
///
/// Used by [`auto_sync_to_store`] / [`auto_sync_from_store`] to avoid
/// collisions when two components have fields with the same name.
pub fn persist_component_key(
    stable_id: &str,
    component_name: &str,
    field_name: &str,
) -> String {
    format!("entity/{stable_id}/{component_name}/{field_name}")
}

/// Trait for components that can sync marked fields to/from the GameStore.
///
/// Each persisted field maps to a store key of the form
/// `"entity/{stable_id}/{field_name}"`. Implementations write fields via
/// `store.set()` and read them back via `store.get()`.
///
/// For most components, prefer the registry-driven approach
/// ([`auto_sync_to_store`] / [`auto_sync_from_store`]) which requires no
/// per-type implementation -- just mark fields with `persist: true` in
/// their `FieldMeta`.
///
/// # Manual Implementation
///
/// Use this trait directly when you need custom serialization logic
/// (e.g., entity references that need StableId remapping):
///
/// ```ignore
/// use rkf_runtime::behavior::{Persistable, persist_key, GameStore, GameValue};
///
/// struct Health { current: f32, max: f32 }
///
/// impl Persistable for Health {
///     fn sync_to_store(&self, stable_id: &str, store: &mut GameStore) {
///         store.set(&persist_key(stable_id, "current"), self.current);
///         // max is not persisted (design-time constant)
///     }
///     fn sync_from_store(&mut self, stable_id: &str, store: &GameStore) {
///         if let Some(v) = store.get::<f32>(&persist_key(stable_id, "current")) {
///             self.current = v;
///         }
///     }
/// }
/// ```
pub trait Persistable {
    /// Write persisted fields into the store, keyed by the entity's stable ID.
    fn sync_to_store(&self, stable_id: &str, store: &mut GameStore);

    /// Read persisted fields from the store, keyed by the entity's stable ID.
    /// Missing keys are silently skipped (field retains its current value).
    fn sync_from_store(&mut self, stable_id: &str, store: &GameStore);
}

// ─── Registry-driven auto-persistence ────────────────────────────────────

/// Return the list of field names marked `persist: true` in the metadata.
pub fn persisted_field_names(meta: &[FieldMeta]) -> Vec<&str> {
    meta.iter()
        .filter(|f| f.persist && !f.transient)
        .map(|f| f.name)
        .collect()
}

/// Automatically sync all `persist: true` fields of a component to the store.
///
/// Uses the component's `get_field` function pointer to read field values as
/// `GameValue`, then stores them under component-scoped keys.
///
/// Returns the number of fields successfully synced, or an error if the
/// entity does not have the component.
///
/// # Example
///
/// ```ignore
/// let entry = registry.component_entry("Health").unwrap();
/// auto_sync_to_store(entry, &world, entity, "stable-id-123", &mut store)?;
/// ```
pub fn auto_sync_to_store(
    entry: &ComponentEntry,
    world: &hecs::World,
    entity: hecs::Entity,
    stable_id: &str,
    store: &mut GameStore,
) -> Result<usize, String> {
    let fields = persisted_field_names(entry.meta);
    let mut count = 0;

    for field_name in &fields {
        let value = (entry.get_field)(world, entity, field_name)?;
        let key = persist_component_key(stable_id, entry.name, field_name);
        store.set(&key, value);
        count += 1;
    }

    Ok(count)
}

/// Automatically restore all `persist: true` fields of a component from the store.
///
/// Uses the component's `set_field` function pointer to write field values from
/// `GameValue` entries in the store. Missing keys are silently skipped.
///
/// Returns the number of fields successfully restored, or an error if a
/// `set_field` call fails (e.g., type mismatch).
///
/// # Example
///
/// ```ignore
/// let entry = registry.component_entry("Health").unwrap();
/// auto_sync_from_store(entry, &mut world, entity, "stable-id-123", &store)?;
/// ```
pub fn auto_sync_from_store(
    entry: &ComponentEntry,
    world: &mut hecs::World,
    entity: hecs::Entity,
    stable_id: &str,
    store: &GameStore,
) -> Result<usize, String> {
    let fields = persisted_field_names(entry.meta);
    let mut count = 0;

    for field_name in &fields {
        let key = persist_component_key(stable_id, entry.name, field_name);
        if let Some(value) = store.get_raw(&key) {
            (entry.set_field)(world, entity, field_name, value.clone())?;
            count += 1;
        }
    }

    Ok(count)
}

/// Sync all components on an entity that have `persist: true` fields.
///
/// Iterates over all registered component entries, checks if the entity has
/// each component, and syncs persisted fields to the store.
///
/// Returns the total number of fields synced across all components.
pub fn auto_sync_entity_to_store(
    entries: &[&ComponentEntry],
    world: &hecs::World,
    entity: hecs::Entity,
    stable_id: &str,
    store: &mut GameStore,
) -> usize {
    let mut total = 0;
    for entry in entries {
        if !(entry.has)(world, entity) {
            continue;
        }
        // Skip components with no persist fields.
        if !entry.meta.iter().any(|f| f.persist && !f.transient) {
            continue;
        }
        if let Ok(count) = auto_sync_to_store(entry, world, entity, stable_id, store) {
            total += count;
        }
    }
    total
}

/// Restore all components on an entity that have `persist: true` fields.
///
/// Iterates over all registered component entries, checks if the entity has
/// each component, and restores persisted fields from the store.
///
/// Returns the total number of fields restored across all components.
pub fn auto_sync_entity_from_store(
    entries: &[&ComponentEntry],
    world: &mut hecs::World,
    entity: hecs::Entity,
    stable_id: &str,
    store: &GameStore,
) -> usize {
    let mut total = 0;
    for entry in entries {
        if !(entry.has)(world, entity) {
            continue;
        }
        if !entry.meta.iter().any(|f| f.persist && !f.transient) {
            continue;
        }
        if let Ok(count) = auto_sync_from_store(entry, world, entity, stable_id, store) {
            total += count;
        }
    }
    total
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::game_value::GameValue;
    use super::super::registry::{ComponentEntry, FieldMeta, FieldType};

    #[test]
    fn persist_key_format() {
        let key = persist_key("abc-123", "health");
        assert_eq!(key, "entity/abc-123/health");

        let key2 = persist_key("some-uuid-v4", "score");
        assert_eq!(key2, "entity/some-uuid-v4/score");
    }

    #[test]
    fn persist_key_with_empty_parts() {
        let key = persist_key("", "field");
        assert_eq!(key, "entity//field");

        let key2 = persist_key("id", "");
        assert_eq!(key2, "entity/id/");
    }

    #[test]
    fn persist_component_key_format() {
        let key = persist_component_key("abc-123", "Health", "current");
        assert_eq!(key, "entity/abc-123/Health/current");
    }

    #[test]
    fn persisted_field_names_filters_correctly() {
        static META: [FieldMeta; 4] = [
            FieldMeta {
                name: "health",
                field_type: FieldType::Float,
                transient: false,
                range: None,
                default: None,
                persist: true,
            },
            FieldMeta {
                name: "max_health",
                field_type: FieldType::Float,
                transient: false,
                range: None,
                default: None,
                persist: false,
            },
            FieldMeta {
                name: "timer",
                field_type: FieldType::Float,
                transient: true,
                range: None,
                default: None,
                persist: false,
            },
            FieldMeta {
                name: "buggy",
                field_type: FieldType::Float,
                transient: true,
                range: None,
                default: None,
                persist: true, // should be excluded — transient overrides persist
            },
        ];

        let names = persisted_field_names(&META);
        assert_eq!(names, vec!["health"]);
    }

    #[test]
    fn persisted_field_names_empty_when_none_marked() {
        static META: [FieldMeta; 2] = [
            FieldMeta {
                name: "a",
                field_type: FieldType::Float,
                transient: false,
                range: None,
                default: None,
                persist: false,
            },
            FieldMeta {
                name: "b",
                field_type: FieldType::Int,
                transient: false,
                range: None,
                default: None,
                persist: false,
            },
        ];
        assert!(persisted_field_names(&META).is_empty());
    }

    // ── Manual Persistable roundtrip ─────────────────────────────────────

    #[derive(Debug, PartialEq)]
    struct TestComponent {
        health: f32,
        name: String,
        max_health: f32, // not persisted
    }

    impl Persistable for TestComponent {
        fn sync_to_store(&self, stable_id: &str, store: &mut GameStore) {
            store.set(&persist_key(stable_id, "health"), self.health);
            store.set(&persist_key(stable_id, "name"), self.name.as_str());
        }

        fn sync_from_store(&mut self, stable_id: &str, store: &GameStore) {
            if let Some(v) = store.get::<f32>(&persist_key(stable_id, "health")) {
                self.health = v;
            }
            if let Some(v) = store.get::<String>(&persist_key(stable_id, "name")) {
                self.name = v;
            }
        }
    }

    #[test]
    fn manual_persistable_roundtrip() {
        let original = TestComponent {
            health: 75.0,
            name: "Guard".to_owned(),
            max_health: 100.0,
        };

        let stable_id = "test-entity-uuid";
        let mut store = GameStore::new();

        // Sync to store
        original.sync_to_store(stable_id, &mut store);

        // Verify keys exist
        assert!(store.get_raw(&persist_key(stable_id, "health")).is_some());
        assert!(store.get_raw(&persist_key(stable_id, "name")).is_some());

        // Create a fresh component and sync from store
        let mut restored = TestComponent {
            health: 0.0,
            name: String::new(),
            max_health: 200.0, // different — should not be overwritten
        };
        restored.sync_from_store(stable_id, &store);

        assert!((restored.health - 75.0).abs() < 1e-6);
        assert_eq!(restored.name, "Guard");
        assert!((restored.max_health - 200.0).abs() < 1e-6); // unchanged
    }

    #[test]
    fn sync_from_store_missing_keys_keeps_defaults() {
        let store = GameStore::new();
        let mut component = TestComponent {
            health: 50.0,
            name: "Original".to_owned(),
            max_health: 100.0,
        };

        component.sync_from_store("nonexistent-id", &store);

        // All fields should remain unchanged
        assert!((component.health - 50.0).abs() < 1e-6);
        assert_eq!(component.name, "Original");
    }

    #[test]
    fn sync_from_store_partial_keys() {
        let mut store = GameStore::new();
        let stable_id = "partial-entity";

        // Only store health, not name
        store.set(&persist_key(stable_id, "health"), 30.0_f32);

        let mut component = TestComponent {
            health: 100.0,
            name: "Default".to_owned(),
            max_health: 100.0,
        };
        component.sync_from_store(stable_id, &store);

        assert!((component.health - 30.0).abs() < 1e-6);
        assert_eq!(component.name, "Default"); // unchanged — key was missing
    }

    // ── Registry-driven auto-sync tests ──────────────────────────────────

    // A test component for auto-sync testing via ComponentEntry.
    struct AutoHealth {
        current: f32,
        max: f32,
    }

    static AUTO_HEALTH_META: [FieldMeta; 2] = [
        FieldMeta {
            name: "current",
            field_type: FieldType::Float,
            transient: false,
            range: None,
            default: None,
            persist: true,
        },
        FieldMeta {
            name: "max",
            field_type: FieldType::Float,
            transient: false,
            range: None,
            default: None,
            persist: false,
        },
    ];

    fn auto_health_entry() -> ComponentEntry {
        ComponentEntry {
            name: "AutoHealth",
            meta: &AUTO_HEALTH_META,
            serialize: |_, _| None,
            deserialize_insert: |_, _, _| Ok(()),
            has: |world, entity| world.get::<&AutoHealth>(entity).is_ok(),
            remove: |world, entity| { let _ = world.remove_one::<AutoHealth>(entity); },
            get_field: |world, entity, field_name| {
                let c = world.get::<&AutoHealth>(entity)
                    .map_err(|_| "no AutoHealth".to_string())?;
                match field_name {
                    "current" => Ok(GameValue::Float(c.current as f64)),
                    "max" => Ok(GameValue::Float(c.max as f64)),
                    _ => Err(format!("unknown field '{}'", field_name)),
                }
            },
            set_field: |world, entity, field_name, value| {
                let mut c = world.get::<&mut AutoHealth>(entity)
                    .map_err(|_| "no AutoHealth".to_string())?;
                match field_name {
                    "current" => match value {
                        GameValue::Float(f) => c.current = f as f32,
                        _ => return Err("type mismatch".into()),
                    },
                    "max" => match value {
                        GameValue::Float(f) => c.max = f as f32,
                        _ => return Err("type mismatch".into()),
                    },
                    _ => return Err(format!("unknown field '{}'", field_name)),
                }
                Ok(())
            },
        }
    }

    #[test]
    fn auto_sync_to_store_persists_marked_fields() {
        let mut world = hecs::World::new();
        let entity = world.spawn((AutoHealth { current: 42.0, max: 100.0 },));
        let entry = auto_health_entry();
        let mut store = GameStore::new();

        let count = auto_sync_to_store(&entry, &world, entity, "ent-1", &mut store).unwrap();

        // Only "current" is persist: true
        assert_eq!(count, 1);

        let key = persist_component_key("ent-1", "AutoHealth", "current");
        let val = store.get::<f64>(&key).unwrap();
        assert!((val - 42.0).abs() < 1e-6);

        // "max" should NOT be in the store
        let max_key = persist_component_key("ent-1", "AutoHealth", "max");
        assert!(store.get_raw(&max_key).is_none());
    }

    #[test]
    fn auto_sync_from_store_restores_marked_fields() {
        let mut world = hecs::World::new();
        let entity = world.spawn((AutoHealth { current: 0.0, max: 50.0 },));
        let entry = auto_health_entry();

        // Pre-populate the store
        let mut store = GameStore::new();
        let key = persist_component_key("ent-1", "AutoHealth", "current");
        store.set(&key, 75.0_f64);

        let count = auto_sync_from_store(&entry, &mut world, entity, "ent-1", &store).unwrap();
        assert_eq!(count, 1);

        // Verify the field was restored
        let c = world.get::<&AutoHealth>(entity).unwrap();
        assert!((c.current - 75.0).abs() < 1e-4);
        assert!((c.max - 50.0).abs() < 1e-4); // unchanged
    }

    #[test]
    fn auto_sync_from_store_missing_keys_skipped() {
        let mut world = hecs::World::new();
        let entity = world.spawn((AutoHealth { current: 99.0, max: 100.0 },));
        let entry = auto_health_entry();
        let store = GameStore::new(); // empty

        let count = auto_sync_from_store(&entry, &mut world, entity, "ent-1", &store).unwrap();
        assert_eq!(count, 0);

        // Value unchanged
        let c = world.get::<&AutoHealth>(entity).unwrap();
        assert!((c.current - 99.0).abs() < 1e-4);
    }

    #[test]
    fn auto_sync_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn((AutoHealth { current: 33.0, max: 100.0 },));
        let entry = auto_health_entry();
        let mut store = GameStore::new();

        // Save
        auto_sync_to_store(&entry, &world, entity, "ent-rt", &mut store).unwrap();

        // Mutate the component
        {
            let mut c = world.get::<&mut AutoHealth>(entity).unwrap();
            c.current = 0.0;
            c.max = 999.0;
        }

        // Restore
        auto_sync_from_store(&entry, &mut world, entity, "ent-rt", &store).unwrap();

        let c = world.get::<&AutoHealth>(entity).unwrap();
        assert!((c.current - 33.0).abs() < 1e-4); // restored
        assert!((c.max - 999.0).abs() < 1e-4);    // NOT restored (persist: false)
    }

    #[test]
    fn auto_sync_entity_multiple_components() {
        let mut world = hecs::World::new();
        let entity = world.spawn((AutoHealth { current: 50.0, max: 100.0 },));

        // A second component with a persist field
        struct Score { value: i64 }

        static SCORE_META: [FieldMeta; 1] = [
            FieldMeta {
                name: "value",
                field_type: FieldType::Int,
                transient: false,
                range: None,
                default: None,
                persist: true,
            },
        ];

        let score_entry = ComponentEntry {
            name: "Score",
            meta: &SCORE_META,
            serialize: |_, _| None,
            deserialize_insert: |_, _, _| Ok(()),
            has: |world, entity| world.get::<&Score>(entity).is_ok(),
            remove: |world, entity| { let _ = world.remove_one::<Score>(entity); },
            get_field: |world, entity, field_name| {
                let c = world.get::<&Score>(entity)
                    .map_err(|_| "no Score".to_string())?;
                match field_name {
                    "value" => Ok(GameValue::Int(c.value)),
                    _ => Err(format!("unknown field '{}'", field_name)),
                }
            },
            set_field: |world, entity, field_name, value| {
                let mut c = world.get::<&mut Score>(entity)
                    .map_err(|_| "no Score".to_string())?;
                match field_name {
                    "value" => match value {
                        GameValue::Int(i) => c.value = i,
                        _ => return Err("type mismatch".into()),
                    },
                    _ => return Err(format!("unknown field '{}'", field_name)),
                }
                Ok(())
            },
        };

        // Add Score component to the entity
        world.insert_one(entity, Score { value: 42 }).unwrap();

        let health_entry = auto_health_entry();
        let entries: Vec<&ComponentEntry> = vec![&health_entry, &score_entry];
        let mut store = GameStore::new();

        // Sync all components to store
        let total = auto_sync_entity_to_store(&entries, &world, entity, "ent-multi", &mut store);
        assert_eq!(total, 2); // health.current + score.value

        // Mutate both
        {
            let mut c = world.get::<&mut AutoHealth>(entity).unwrap();
            c.current = 0.0;
        }
        {
            let mut c = world.get::<&mut Score>(entity).unwrap();
            c.value = 0;
        }

        // Restore
        let restored = auto_sync_entity_from_store(&entries, &mut world, entity, "ent-multi", &store);
        assert_eq!(restored, 2);

        let c = world.get::<&AutoHealth>(entity).unwrap();
        assert!((c.current - 50.0).abs() < 1e-4);

        let s = world.get::<&Score>(entity).unwrap();
        assert_eq!(s.value, 42);
    }

    #[test]
    fn auto_sync_entity_skips_absent_components() {
        let mut world = hecs::World::new();
        // Entity only has AutoHealth, not Score
        let entity = world.spawn((AutoHealth { current: 50.0, max: 100.0 },));

        let health_entry = auto_health_entry();
        // Create a dummy entry for a component the entity doesn't have
        let absent_entry = ComponentEntry {
            name: "Absent",
            meta: &[FieldMeta {
                name: "x",
                field_type: FieldType::Float,
                transient: false,
                range: None,
                default: None,
                persist: true,
            }],
            serialize: |_, _| None,
            deserialize_insert: |_, _, _| Ok(()),
            has: |_, _| false, // entity never has this
            remove: |_, _| {},
            get_field: |_, _, _| Err("stub".into()),
            set_field: |_, _, _, _| Err("stub".into()),
        };

        let entries: Vec<&ComponentEntry> = vec![&health_entry, &absent_entry];
        let mut store = GameStore::new();

        let total = auto_sync_entity_to_store(&entries, &world, entity, "ent-skip", &mut store);
        assert_eq!(total, 1); // only AutoHealth.current

        let restored = auto_sync_entity_from_store(&entries, &mut world, entity, "ent-skip", &store);
        assert_eq!(restored, 1);
    }
}
