//! Entity reference serialization helpers.
//!
//! These functions convert between live `hecs::Entity` handles and persistent
//! `uuid::Uuid` identifiers (via [`StableId`] components). They are used by
//! the `#[component]` macro-generated serialize/deserialize code for fields
//! of type `Entity`.

use hecs::{Entity, World};
use uuid::Uuid;

use super::stable_id::StableId;
use super::stable_id_index::StableIdIndex;

/// Serialize an entity handle to its persistent UUID.
///
/// Looks up the [`StableId`] component on `entity`. Returns `None` (with a
/// warning log) if the entity does not carry a `StableId`.
pub fn serialize_entity(entity: Entity, world: &World) -> Option<Uuid> {
    match world.get::<&StableId>(entity) {
        Ok(id) => Some(id.uuid()),
        Err(_) => {
            log::warn!(
                "serialize_entity: entity {entity:?} has no StableId component — \
                 cannot serialize reference"
            );
            None
        }
    }
}

/// Deserialize a persistent UUID back to a live entity handle.
///
/// Looks up `uuid` in the [`StableIdIndex`]. Returns `None` (with a warning
/// log) if the UUID is not present in the index.
pub fn deserialize_entity(uuid: Uuid, index: &StableIdIndex) -> Option<Entity> {
    match index.get_entity(uuid) {
        Some(entity) => Some(entity),
        None => {
            log::warn!(
                "deserialize_entity: UUID {uuid} not found in StableIdIndex — \
                 dangling entity reference"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spawn an entity with a StableId, register it in an index, return all parts.
    fn setup_entity(world: &mut World, index: &mut StableIdIndex) -> (Entity, Uuid) {
        let id = StableId::new();
        let uuid = id.uuid();
        let entity = world.spawn((id,));
        index.insert(uuid, entity);
        (entity, uuid)
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let mut world = World::new();
        let mut index = StableIdIndex::new();
        let (entity, _uuid) = setup_entity(&mut world, &mut index);

        // Serialize: entity → uuid
        let serialized = serialize_entity(entity, &world).expect("should serialize");

        // Deserialize: uuid → entity
        let resolved = deserialize_entity(serialized, &index).expect("should deserialize");

        assert_eq!(resolved, entity);
    }

    #[test]
    fn missing_stable_id_returns_none() {
        let mut world = World::new();
        // Spawn without StableId
        let entity = world.spawn(());

        let result = serialize_entity(entity, &world);
        assert!(result.is_none());
    }

    #[test]
    fn missing_uuid_in_index_returns_none() {
        let index = StableIdIndex::new();
        let unknown_uuid = Uuid::new_v4();

        let result = deserialize_entity(unknown_uuid, &index);
        assert!(result.is_none());
    }

    #[test]
    fn component_with_entity_field_roundtrip() {
        // Simulate a component that holds an Entity reference to another entity.
        // Spawn entity A (the "owner") and entity B (the "target").
        // Serialize A's reference to B as a UUID, then deserialize it back.
        let mut world = World::new();
        let mut index = StableIdIndex::new();

        let (entity_a, _uuid_a) = setup_entity(&mut world, &mut index);
        let (entity_b, _uuid_b) = setup_entity(&mut world, &mut index);

        // entity_a "references" entity_b (like a component field `target: Entity`)
        let target_ref: Entity = entity_b;

        // --- Serialize phase ---
        let serialized_target =
            serialize_entity(target_ref, &world).expect("target should have StableId");

        // Verify the serialized UUID matches entity_b's StableId
        let b_stable = world
            .get::<&StableId>(entity_b)
            .expect("entity_b has StableId");
        assert_eq!(serialized_target, b_stable.uuid());

        // --- Deserialize phase (potentially a different World, same index) ---
        let deserialized_target =
            deserialize_entity(serialized_target, &index).expect("UUID should resolve");

        assert_eq!(deserialized_target, entity_b);

        // Verify entity_a is unaffected / independently resolvable
        let a_uuid = serialize_entity(entity_a, &world).expect("a serializable");
        let a_resolved = deserialize_entity(a_uuid, &index).expect("a resolvable");
        assert_eq!(a_resolved, entity_a);
    }
}
