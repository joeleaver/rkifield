//! Blueprint system — reusable entity templates.
//!
//! A [`Blueprint`] captures a named set of components as RON strings.
//! [`BlueprintCatalog`] stores blueprints by name, with filesystem persistence
//! via `.rkblueprint` files (RON format).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::registry::GameplayRegistry;

/// Components excluded from blueprints (runtime-only or identity-related).
const EXCLUDED_COMPONENTS: &[&str] = &[
    "Transform",
    "WorldTransform",
    "Parent",
    "StableId",
    "SceneOwnership",
];

/// A reusable entity template: a named bag of serialized components.
///
/// Components are stored as `component_name -> RON data` pairs. Excluded
/// components (Transform, WorldTransform, Parent, StableId, SceneOwnership)
/// are stripped on save and not instantiated from blueprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blueprint {
    /// Human-readable name (also used as the catalog key).
    pub name: String,
    /// Component data: `component_name -> RON string`.
    pub components: HashMap<String, String>,
}

impl Blueprint {
    /// Create a new empty blueprint with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            components: HashMap::new(),
        }
    }
}

/// Catalog of named blueprints, indexed by name.
pub struct BlueprintCatalog {
    blueprints: HashMap<String, Blueprint>,
}

impl Default for BlueprintCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl BlueprintCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            blueprints: HashMap::new(),
        }
    }

    /// Insert or replace a blueprint by name.
    pub fn insert(&mut self, blueprint: Blueprint) {
        self.blueprints.insert(blueprint.name.clone(), blueprint);
    }

    /// Look up a blueprint by name.
    pub fn get(&self, name: &str) -> Option<&Blueprint> {
        self.blueprints.get(name)
    }

    /// Remove a blueprint by name, returning it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<Blueprint> {
        self.blueprints.remove(name)
    }

    /// Iterate over all blueprint names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.blueprints.keys().map(|s| s.as_str())
    }

    /// Number of blueprints in the catalog.
    pub fn len(&self) -> usize {
        self.blueprints.len()
    }

    /// Returns true if the catalog contains no blueprints.
    pub fn is_empty(&self) -> bool {
        self.blueprints.is_empty()
    }

    /// Scan a directory for `.rkblueprint` files, parse each as RON, and insert
    /// into the catalog. Returns the number of blueprints successfully loaded.
    ///
    /// Files that fail to parse are logged and skipped.
    pub fn scan_directory(&mut self, dir: &Path) -> Result<usize, std::io::Error> {
        let entries = std::fs::read_dir(dir)?;
        let mut count = 0;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("rkblueprint") {
                continue;
            }

            let contents = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("Failed to read blueprint file {:?}: {}", path, e);
                    continue;
                }
            };

            match deserialize_blueprint(&contents) {
                Ok(bp) => {
                    self.insert(bp);
                    count += 1;
                }
                Err(e) => {
                    log::warn!("Failed to parse blueprint file {:?}: {}", path, e);
                }
            }
        }

        Ok(count)
    }
}

impl std::fmt::Debug for BlueprintCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlueprintCatalog")
            .field("count", &self.blueprints.len())
            .field("names", &self.blueprints.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Returns true if the component name is excluded from blueprints.
fn is_excluded(name: &str) -> bool {
    EXCLUDED_COMPONENTS.contains(&name)
}

/// Create a blueprint from an existing entity by serializing all registered
/// components (excluding Transform, WorldTransform, Parent, StableId,
/// SceneOwnership).
pub fn create_blueprint_from_entity(
    name: String,
    entity: hecs::Entity,
    world: &hecs::World,
    registry: &GameplayRegistry,
) -> Blueprint {
    let mut components = HashMap::new();

    for entry in registry.component_entries() {
        if is_excluded(entry.name) {
            continue;
        }

        if let Some(ron_data) = (entry.serialize)(world, entity) {
            components.insert(entry.name.to_owned(), ron_data);
        }
    }

    Blueprint { name, components }
}

/// Serialize a blueprint to a pretty-printed RON string.
pub fn serialize_blueprint(bp: &Blueprint) -> Result<String, ron::Error> {
    let pretty = ron::ser::PrettyConfig::default();
    ron::ser::to_string_pretty(bp, pretty)
}

/// Deserialize a blueprint from a RON string.
pub fn deserialize_blueprint(ron_str: &str) -> Result<Blueprint, ron::error::SpannedError> {
    ron::from_str(ron_str)
}

// ─── Editor blueprint actions ─────────────────────────────────────────────

/// Save an entity as a blueprint by reading all registered components
/// (excluding `EXCLUDED_COMPONENTS`) and serializing each to RON.
///
/// This is the editor-facing wrapper around [`create_blueprint_from_entity`].
pub fn save_entity_as_blueprint(
    world: &hecs::World,
    entity: hecs::Entity,
    name: &str,
    registry: &GameplayRegistry,
) -> Result<Blueprint, String> {
    // Verify the entity exists by checking at least one component or archetype.
    // hecs doesn't have a direct "entity_exists" — check via entity().
    if !world.contains(entity) {
        return Err("entity does not exist".to_string());
    }

    Ok(create_blueprint_from_entity(
        name.to_string(),
        entity,
        world,
        registry,
    ))
}

/// Spawn a new entity from a blueprint, deserializing each component and
/// setting the Transform position.
///
/// Creates a new entity, inserts all blueprint components via
/// `ComponentEntry::deserialize_insert`, then sets the Transform position
/// to the given `position`. If the blueprint lacks a Transform component,
/// a default Transform with the given position is inserted.
pub fn spawn_from_blueprint(
    world: &mut hecs::World,
    blueprint: &Blueprint,
    position: glam::Vec3,
    registry: &GameplayRegistry,
) -> Result<hecs::Entity, String> {
    let entity = world.spawn(());

    // Insert all blueprint components.
    for (comp_name, ron_data) in &blueprint.components {
        let entry = registry
            .component_entry(comp_name)
            .ok_or_else(|| format!("unknown component in blueprint: '{}'", comp_name))?;
        (entry.deserialize_insert)(world, entity, ron_data)
            .map_err(|e| format!("failed to deserialize '{}': {}", comp_name, e))?;
    }

    // Set position via Transform. If the entity already has a Transform from
    // the blueprint, update its position. Otherwise insert a default with the
    // given position.
    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, position);

    if let Some(transform_entry) = registry.component_entry("Transform") {
        if (transform_entry.has)(world, entity) {
            // Update position on existing Transform.
            (transform_entry.set_field)(
                world,
                entity,
                "position",
                super::game_value::GameValue::WorldPosition(wp),
            )
            .map_err(|e| format!("failed to set position: {}", e))?;
        } else {
            // Insert a default Transform with the desired position.
            let transform = crate::components::Transform {
                position: wp,
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            };
            world
                .insert_one(entity, transform)
                .map_err(|e| format!("failed to insert Transform: {}", e))?;
        }
    }

    Ok(entity)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_guard_blueprint() -> Blueprint {
        let mut components = HashMap::new();
        components.insert(
            "Health".to_owned(),
            "(current: 100.0, max: 100.0)".to_owned(),
        );
        components.insert(
            "EditorMetadata".to_owned(),
            "(name: \"Guard\", tags: [\"enemy\"], locked: false)".to_owned(),
        );
        Blueprint {
            name: "Guard".to_owned(),
            components,
        }
    }

    #[test]
    fn blueprint_creation() {
        let bp = Blueprint::new("Test");
        assert_eq!(bp.name, "Test");
        assert!(bp.components.is_empty());
    }

    #[test]
    fn blueprint_with_components() {
        let bp = make_guard_blueprint();
        assert_eq!(bp.name, "Guard");
        assert_eq!(bp.components.len(), 2);
        assert!(bp.components.contains_key("Health"));
        assert!(bp.components.contains_key("EditorMetadata"));
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let bp = make_guard_blueprint();
        let ron_str = serialize_blueprint(&bp).unwrap();
        let bp2 = deserialize_blueprint(&ron_str).unwrap();

        assert_eq!(bp.name, bp2.name);
        assert_eq!(bp.components.len(), bp2.components.len());
        for (k, v) in &bp.components {
            assert_eq!(bp2.components.get(k).unwrap(), v);
        }
    }

    #[test]
    fn empty_blueprint_roundtrip() {
        let bp = Blueprint::new("Empty");
        let ron_str = serialize_blueprint(&bp).unwrap();
        let bp2 = deserialize_blueprint(&ron_str).unwrap();

        assert_eq!(bp2.name, "Empty");
        assert!(bp2.components.is_empty());
    }

    #[test]
    fn invalid_ron_rejected() {
        let bad = "this is not valid RON {{{";
        let result = deserialize_blueprint(bad);
        assert!(result.is_err());
    }

    #[test]
    fn catalog_insert_and_get() {
        let mut catalog = BlueprintCatalog::new();
        assert!(catalog.is_empty());

        catalog.insert(make_guard_blueprint());
        assert_eq!(catalog.len(), 1);
        assert!(!catalog.is_empty());

        let bp = catalog.get("Guard").unwrap();
        assert_eq!(bp.name, "Guard");
        assert_eq!(bp.components.len(), 2);
    }

    #[test]
    fn catalog_get_missing() {
        let catalog = BlueprintCatalog::new();
        assert!(catalog.get("Nonexistent").is_none());
    }

    #[test]
    fn catalog_remove() {
        let mut catalog = BlueprintCatalog::new();
        catalog.insert(make_guard_blueprint());

        let removed = catalog.remove("Guard");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "Guard");
        assert!(catalog.is_empty());

        // Remove again returns None
        assert!(catalog.remove("Guard").is_none());
    }

    #[test]
    fn catalog_names() {
        let mut catalog = BlueprintCatalog::new();
        catalog.insert(Blueprint::new("Alpha"));
        catalog.insert(Blueprint::new("Beta"));
        catalog.insert(Blueprint::new("Gamma"));

        let mut names: Vec<&str> = catalog.names().collect();
        names.sort();
        assert_eq!(names, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn catalog_insert_replaces() {
        let mut catalog = BlueprintCatalog::new();
        catalog.insert(Blueprint::new("Guard"));
        assert!(catalog.get("Guard").unwrap().components.is_empty());

        catalog.insert(make_guard_blueprint());
        assert_eq!(catalog.get("Guard").unwrap().components.len(), 2);
        assert_eq!(catalog.len(), 1); // Still just one
    }

    #[test]
    fn scan_directory_loads_blueprints() {
        let dir = tempfile::tempdir().unwrap();

        // Write two valid .rkblueprint files
        let bp1 = make_guard_blueprint();
        let bp1_ron = serialize_blueprint(&bp1).unwrap();
        std::fs::write(dir.path().join("guard.rkblueprint"), &bp1_ron).unwrap();

        let bp2 = Blueprint::new("Villager");
        let bp2_ron = serialize_blueprint(&bp2).unwrap();
        std::fs::write(dir.path().join("villager.rkblueprint"), &bp2_ron).unwrap();

        // Write a non-blueprint file (should be ignored)
        std::fs::write(dir.path().join("readme.txt"), "ignore me").unwrap();

        // Write an invalid blueprint file (should be skipped with warning)
        std::fs::write(dir.path().join("bad.rkblueprint"), "not valid ron {{{").unwrap();

        let mut catalog = BlueprintCatalog::new();
        let count = catalog.scan_directory(dir.path()).unwrap();

        assert_eq!(count, 2);
        assert_eq!(catalog.len(), 2);
        assert!(catalog.get("Guard").is_some());
        assert!(catalog.get("Villager").is_some());
    }

    #[test]
    fn scan_directory_nonexistent() {
        let mut catalog = BlueprintCatalog::new();
        let result = catalog.scan_directory(Path::new("/tmp/nonexistent_rkf_dir_12345"));
        assert!(result.is_err());
    }

    #[test]
    fn blueprint_file_format_parsing() {
        // Test the exact .rkblueprint format from the spec
        let ron_str = r#"Blueprint(
    name: "Guard",
    components: {
        "Health": "(current: 100.0, max: 100.0)",
        "EditorMetadata": "(name: \"Guard\", tags: [\"enemy\"], locked: false)",
    },
)"#;
        let bp = deserialize_blueprint(ron_str).unwrap();
        assert_eq!(bp.name, "Guard");
        assert_eq!(bp.components.len(), 2);
        assert_eq!(
            bp.components.get("Health").unwrap(),
            "(current: 100.0, max: 100.0)"
        );
    }

    #[test]
    fn excluded_components_stripped_from_entity() {
        use crate::behavior::registry::{ComponentEntry, GameplayRegistry};

        // Build a registry with some excluded and non-excluded components
        let mut registry = GameplayRegistry::new();

        // Register a non-excluded component that serializes
        registry
            .register_component(ComponentEntry {
                name: "Health",
                serialize: |_, _| Some("(current: 50.0, max: 100.0)".to_owned()),
                deserialize_insert: |_, _, _| Ok(()),
                remove: |_, _| {},
                has: |_, _| true,
                get_field: |_, _, _| Err("stub".into()),
                set_field: |_, _, _, _| Err("stub".into()),
                meta: &[],
            })
            .unwrap();

        // Register an excluded component (Transform)
        registry
            .register_component(ComponentEntry {
                name: "Transform",
                serialize: |_, _| Some("(position: ..., rotation: ..., scale: ...)".to_owned()),
                deserialize_insert: |_, _, _| Ok(()),
                remove: |_, _| {},
                has: |_, _| true,
                get_field: |_, _, _| Err("stub".into()),
                set_field: |_, _, _, _| Err("stub".into()),
                meta: &[],
            })
            .unwrap();

        // Register another excluded component (StableId)
        registry
            .register_component(ComponentEntry {
                name: "StableId",
                serialize: |_, _| Some("(uuid: \"abc\")".to_owned()),
                deserialize_insert: |_, _, _| Ok(()),
                remove: |_, _| {},
                has: |_, _| true,
                get_field: |_, _, _| Err("stub".into()),
                set_field: |_, _, _, _| Err("stub".into()),
                meta: &[],
            })
            .unwrap();

        let world = hecs::World::new();
        // Entity doesn't need actual components — the dummy serialize fns always return Some
        let entity = world.reserve_entity();

        let bp = create_blueprint_from_entity("Test".to_owned(), entity, &world, &registry);

        assert_eq!(bp.name, "Test");
        // Only Health should survive — Transform and StableId are excluded
        assert_eq!(bp.components.len(), 1);
        assert!(bp.components.contains_key("Health"));
        assert!(!bp.components.contains_key("Transform"));
        assert!(!bp.components.contains_key("StableId"));
    }

    #[test]
    fn excluded_components_list() {
        // Verify all five excluded components are rejected
        assert!(is_excluded("Transform"));
        assert!(is_excluded("WorldTransform"));
        assert!(is_excluded("Parent"));
        assert!(is_excluded("StableId"));
        assert!(is_excluded("SceneOwnership"));
        // Non-excluded
        assert!(!is_excluded("Health"));
        assert!(!is_excluded("EditorMetadata"));
    }

    // ── Editor blueprint action tests ─────────────────────────────────────

    #[test]
    fn save_and_spawn_roundtrip() {
        use crate::behavior::engine_components::engine_register;
        use crate::components::EditorMetadata;

        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);

        let mut world = hecs::World::new();
        let entity = world.spawn((
            EditorMetadata {
                name: "Guard".to_string(),
                tags: vec!["enemy".to_string()],
                locked: false,
            },
        ));

        // Save as blueprint.
        let bp = save_entity_as_blueprint(&world, entity, "GuardBP", &registry).unwrap();
        assert_eq!(bp.name, "GuardBP");
        assert!(bp.components.contains_key("EditorMetadata"));

        // Spawn from blueprint.
        let pos = glam::Vec3::new(10.0, 20.0, 30.0);
        let spawned = spawn_from_blueprint(&mut world, &bp, pos, &registry).unwrap();

        // Verify the spawned entity has EditorMetadata with the same data.
        let meta = world.get::<&EditorMetadata>(spawned).unwrap();
        assert_eq!(meta.name, "Guard");
        assert_eq!(meta.tags, vec!["enemy".to_string()]);
    }

    #[test]
    fn excluded_components_not_saved() {
        use crate::behavior::engine_components::engine_register;
        use crate::components::{EditorMetadata, Transform};

        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);

        let mut world = hecs::World::new();
        let entity = world.spawn((
            Transform::default(),
            EditorMetadata {
                name: "Test".to_string(),
                tags: vec![],
                locked: false,
            },
        ));

        let bp = save_entity_as_blueprint(&world, entity, "TestBP", &registry).unwrap();

        // Transform is excluded from blueprints.
        assert!(!bp.components.contains_key("Transform"));
        // EditorMetadata is NOT excluded.
        assert!(bp.components.contains_key("EditorMetadata"));
    }

    #[test]
    fn spawn_sets_position() {
        use crate::behavior::engine_components::engine_register;
        use crate::components::Transform;

        let mut registry = GameplayRegistry::new();
        engine_register(&mut registry);

        // Blueprint with no components (Transform will be inserted by spawn).
        let bp = Blueprint::new("EmptyBP");
        let pos = glam::Vec3::new(5.0, 10.0, 15.0);

        let mut world = hecs::World::new();
        let spawned = spawn_from_blueprint(&mut world, &bp, pos, &registry).unwrap();

        // WorldPosition normalizes local to [0, CHUNK_SIZE=8), adjusting chunk.
        let expected = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
        let transform = world.get::<&Transform>(spawned).unwrap();
        assert_eq!(transform.position.chunk, expected.chunk);
        let local = transform.position.local;
        assert!((local.x - expected.local.x).abs() < 1e-6);
        assert!((local.y - expected.local.y).abs() < 1e-6);
        assert!((local.z - expected.local.z).abs() < 1e-6);
    }

    #[test]
    fn save_entity_as_blueprint_nonexistent_entity() {
        let registry = GameplayRegistry::new();
        let world = hecs::World::new();
        let entity = hecs::Entity::DANGLING;

        let result = save_entity_as_blueprint(&world, entity, "Bad", &registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }
}
