//! Save/load functions for v3 scene files and hecs World integration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use uuid::Uuid;

use super::types::{EntityRecord, SceneFileV3};

/// Preserve unknown components that don't match any registered type.
///
/// Attached to entities whose scene file contained component names that
/// couldn't be resolved in the current registry. Re-emitted as-is on save
/// so data isn't lost.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UnknownComponents {
    /// Component name -> raw RON string.
    pub data: HashMap<String, String>,
}

/// Save the current hecs World into a [`SceneFileV3`].
///
/// Iterates all entities, reads their StableId, serializes engine components
/// (Transform, EditorMetadata, SdfTree, CameraComponent, FogVolumeComponent),
/// and collects gameplay components via the ComponentEntry registry.
pub fn save_scene(
    ecs: &hecs::World,
    stable_index: &crate::behavior::StableIdIndex,
    registry: &crate::behavior::GameplayRegistry,
) -> SceneFileV3 {
    use crate::behavior::StableId;
    use crate::components::Parent;

    let mut scene = SceneFileV3::new();

    for entity in ecs.iter() {
        let entity_ref = entity;
        let hecs_entity = entity_ref.entity();

        // Get StableId (required for persistence)
        let stable_id = match ecs.get::<&StableId>(hecs_entity) {
            Ok(sid) => sid.0,
            Err(_) => continue, // Skip entities without StableId (transient)
        };

        let mut record = EntityRecord::new(stable_id);

        // Parent -> StableId UUID
        if let Ok(parent) = ecs.get::<&Parent>(hecs_entity) {
            if let Some(parent_sid) = stable_index.get_stable(parent.entity) {
                record.parent = Some(parent_sid);
            }
        }

        // Re-emit unknown components
        if let Ok(unknown) = ecs.get::<&UnknownComponents>(hecs_entity) {
            for (name, ron_str) in &unknown.data {
                record.components.insert(name.clone(), ron_str.clone());
            }
        }

        // All components via registry ComponentEntry (engine + gameplay, uniform path).
        for entry in registry.component_entries() {
            if record.components.contains_key(entry.name) {
                continue;
            }
            if (entry.has)(ecs, hecs_entity) {
                if let Some(ron_str) = (entry.serialize)(ecs, hecs_entity) {
                    record.components.insert(entry.name.to_string(), ron_str);
                }
            }
        }

        scene.entities.push(record);
    }

    scene
}

/// Load a [`SceneFileV3`] into a hecs World.
///
/// Creates entities, assigns StableIds, deserializes engine components,
/// and resolves parent references. Unknown components are preserved in
/// [`UnknownComponents`].
pub fn load_scene(
    scene: &SceneFileV3,
    ecs: &mut hecs::World,
    stable_index: &mut crate::behavior::StableIdIndex,
    registry: &crate::behavior::GameplayRegistry,
) {
    use crate::behavior::StableId;
    use crate::components::Parent;

    // Phase 1: Create entities and deserialize all components (except Parent).
    let mut uuid_to_hecs: HashMap<Uuid, hecs::Entity> = HashMap::new();

    for record in &scene.entities {
        let hecs_entity = ecs.spawn((StableId(record.stable_id),));
        stable_index.insert(record.stable_id, hecs_entity);
        uuid_to_hecs.insert(record.stable_id, hecs_entity);

        // All components via registry (engine + gameplay, uniform path).
        let mut unknown: HashMap<String, String> = HashMap::new();
        for (comp_name, ron_str) in &record.components {
            if let Some(entry) = registry.component_entry(comp_name) {
                if let Err(e) = (entry.deserialize_insert)(ecs, hecs_entity, ron_str) {
                    eprintln!(
                        "warning: failed to deserialize component '{}' on entity {}: {}",
                        comp_name, record.stable_id, e
                    );
                    unknown.insert(comp_name.clone(), ron_str.clone());
                }
            } else {
                // Not in registry — preserve as unknown so data isn't lost
                unknown.insert(comp_name.clone(), ron_str.clone());
            }
        }
        if !unknown.is_empty() {
            let _ = ecs.insert_one(hecs_entity, UnknownComponents { data: unknown });
        }
    }

    // Phase 2: Resolve parent references with cycle detection.
    // Build a UUID->parent UUID map for cycle checking.
    let parent_map: HashMap<Uuid, Uuid> = scene
        .entities
        .iter()
        .filter_map(|r| r.parent.map(|p| (r.stable_id, p)))
        .collect();

    for record in &scene.entities {
        if let Some(parent_uuid) = record.parent {
            // Walk the parent chain to detect cycles.
            let mut visited = HashSet::new();
            visited.insert(record.stable_id);
            let mut cursor = parent_uuid;
            let mut has_cycle = false;
            loop {
                if !visited.insert(cursor) {
                    // Already visited -- cycle detected.
                    has_cycle = true;
                    eprintln!(
                        "warning: parent cycle detected for entity {}, skipping parent assignment",
                        record.stable_id
                    );
                    break;
                }
                match parent_map.get(&cursor) {
                    Some(&next) => cursor = next,
                    None => break, // reached a root -- no cycle
                }
            }

            if has_cycle {
                continue;
            }

            if let (Some(&child), Some(&parent)) = (
                uuid_to_hecs.get(&record.stable_id),
                uuid_to_hecs.get(&parent_uuid),
            ) {
                let _ = ecs.insert_one(
                    child,
                    Parent {
                        entity: parent,
                        bone_index: None,
                    },
                );
            }
        }
    }
}
