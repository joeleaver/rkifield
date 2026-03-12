use hecs::Entity;
use std::collections::HashMap;
use uuid::Uuid;

/// Bidirectional mapping between StableId UUIDs and live hecs::Entity handles.
///
/// Maintains a consistent two-way index so that entity references serialized
/// as UUIDs can be resolved back to live Entity handles after load, and vice
/// versa. Both maps are kept in sync by all mutating operations.
pub struct StableIdIndex {
    stable_to_entity: HashMap<Uuid, Entity>,
    entity_to_stable: HashMap<Entity, Uuid>,
}

impl StableIdIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            stable_to_entity: HashMap::new(),
            entity_to_stable: HashMap::new(),
        }
    }

    /// Insert a bidirectional mapping.
    ///
    /// # Panics
    /// Panics if either `uuid` or `entity` is already present in the index.
    /// This is an invariant violation — each UUID and each Entity must map
    /// to exactly one counterpart.
    pub fn insert(&mut self, uuid: Uuid, entity: Entity) {
        if self.stable_to_entity.contains_key(&uuid) {
            panic!("StableIdIndex: duplicate UUID {uuid}");
        }
        if self.entity_to_stable.contains_key(&entity) {
            panic!("StableIdIndex: duplicate Entity {entity:?}");
        }
        self.stable_to_entity.insert(uuid, entity);
        self.entity_to_stable.insert(entity, uuid);
    }

    /// Remove the mapping for a given entity. Returns the UUID if found.
    pub fn remove_by_entity(&mut self, entity: Entity) -> Option<Uuid> {
        if let Some(uuid) = self.entity_to_stable.remove(&entity) {
            self.stable_to_entity.remove(&uuid);
            Some(uuid)
        } else {
            None
        }
    }

    /// Remove the mapping for a given UUID. Returns the Entity if found.
    pub fn remove_by_stable(&mut self, uuid: Uuid) -> Option<Entity> {
        if let Some(entity) = self.stable_to_entity.remove(&uuid) {
            self.entity_to_stable.remove(&entity);
            Some(entity)
        } else {
            None
        }
    }

    /// Look up the Entity handle for a given UUID.
    pub fn get_entity(&self, uuid: Uuid) -> Option<Entity> {
        self.stable_to_entity.get(&uuid).copied()
    }

    /// Look up the UUID for a given Entity handle.
    pub fn get_stable(&self, entity: Entity) -> Option<Uuid> {
        self.entity_to_stable.get(&entity).copied()
    }

    /// Check whether an Entity is present in the index.
    pub fn contains_entity(&self, entity: Entity) -> bool {
        self.entity_to_stable.contains_key(&entity)
    }

    /// Check whether a UUID is present in the index.
    pub fn contains_stable(&self, uuid: Uuid) -> bool {
        self.stable_to_entity.contains_key(&uuid)
    }

    /// Number of mappings in the index.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.stable_to_entity.len(), self.entity_to_stable.len());
        self.stable_to_entity.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all mappings.
    pub fn clear(&mut self) {
        self.stable_to_entity.clear();
        self.entity_to_stable.clear();
    }

    /// Iterate all (UUID, Entity) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Uuid, Entity)> + '_ {
        self.stable_to_entity
            .iter()
            .map(|(&uuid, &entity)| (uuid, entity))
    }
}

impl Default for StableIdIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a fake Entity with a specific id for testing.
    /// hecs::Entity can be constructed via a World.
    fn make_entities(count: usize) -> (hecs::World, Vec<Entity>) {
        let mut world = hecs::World::new();
        let entities: Vec<Entity> = (0..count).map(|_| world.spawn(())).collect();
        (world, entities)
    }

    #[test]
    fn insert_and_lookup_both_directions() {
        let (_world, entities) = make_entities(1);
        let entity = entities[0];
        let uuid = Uuid::new_v4();

        let mut index = StableIdIndex::new();
        index.insert(uuid, entity);

        assert_eq!(index.get_entity(uuid), Some(entity));
        assert_eq!(index.get_stable(entity), Some(uuid));
    }

    #[test]
    fn remove_by_entity_cleans_both_maps() {
        let (_world, entities) = make_entities(1);
        let entity = entities[0];
        let uuid = Uuid::new_v4();

        let mut index = StableIdIndex::new();
        index.insert(uuid, entity);
        let removed = index.remove_by_entity(entity);

        assert_eq!(removed, Some(uuid));
        assert_eq!(index.get_entity(uuid), None);
        assert_eq!(index.get_stable(entity), None);
        assert!(index.is_empty());
    }

    #[test]
    fn remove_by_stable_cleans_both_maps() {
        let (_world, entities) = make_entities(1);
        let entity = entities[0];
        let uuid = Uuid::new_v4();

        let mut index = StableIdIndex::new();
        index.insert(uuid, entity);
        let removed = index.remove_by_stable(uuid);

        assert_eq!(removed, Some(entity));
        assert_eq!(index.get_entity(uuid), None);
        assert_eq!(index.get_stable(entity), None);
        assert!(index.is_empty());
    }

    #[test]
    #[should_panic(expected = "duplicate UUID")]
    fn insert_duplicate_uuid_panics() {
        let (_world, entities) = make_entities(2);
        let uuid = Uuid::new_v4();

        let mut index = StableIdIndex::new();
        index.insert(uuid, entities[0]);
        index.insert(uuid, entities[1]); // should panic
    }

    #[test]
    #[should_panic(expected = "duplicate Entity")]
    fn insert_duplicate_entity_panics() {
        let (_world, entities) = make_entities(1);
        let entity = entities[0];

        let mut index = StableIdIndex::new();
        index.insert(Uuid::new_v4(), entity);
        index.insert(Uuid::new_v4(), entity); // should panic
    }

    #[test]
    fn clear_empties_both_maps() {
        let (_world, entities) = make_entities(3);
        let mut index = StableIdIndex::new();
        for &e in &entities {
            index.insert(Uuid::new_v4(), e);
        }
        assert_eq!(index.len(), 3);

        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn len_tracks_correctly() {
        let (_world, entities) = make_entities(3);
        let mut index = StableIdIndex::new();
        assert_eq!(index.len(), 0);

        let uuids: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
        for i in 0..3 {
            index.insert(uuids[i], entities[i]);
        }
        assert_eq!(index.len(), 3);

        index.remove_by_entity(entities[1]);
        assert_eq!(index.len(), 2);

        index.remove_by_stable(uuids[0]);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn contains_checks() {
        let (_world, entities) = make_entities(1);
        let entity = entities[0];
        let uuid = Uuid::new_v4();

        let mut index = StableIdIndex::new();
        assert!(!index.contains_entity(entity));
        assert!(!index.contains_stable(uuid));

        index.insert(uuid, entity);
        assert!(index.contains_entity(entity));
        assert!(index.contains_stable(uuid));
    }

    #[test]
    fn iter_returns_all_mappings() {
        let (_world, entities) = make_entities(3);
        let mut index = StableIdIndex::new();
        let mut expected: HashMap<Uuid, Entity> = HashMap::new();

        for &e in &entities {
            let uuid = Uuid::new_v4();
            index.insert(uuid, e);
            expected.insert(uuid, e);
        }

        let collected: HashMap<Uuid, Entity> = index.iter().collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let (_world, entities) = make_entities(1);
        let mut index = StableIdIndex::new();

        assert_eq!(index.remove_by_entity(entities[0]), None);
        assert_eq!(index.remove_by_stable(Uuid::new_v4()), None);
    }

    #[test]
    fn roundtrip_insert_remove_verify_consistency() {
        let (_world, entities) = make_entities(5);
        let mut index = StableIdIndex::new();
        let uuids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();

        // Insert all
        for i in 0..5 {
            index.insert(uuids[i], entities[i]);
        }
        assert_eq!(index.len(), 5);

        // Remove some
        index.remove_by_entity(entities[1]);
        index.remove_by_stable(uuids[3]);
        assert_eq!(index.len(), 3);

        // Verify remaining are consistent
        for i in [0, 2, 4] {
            assert_eq!(index.get_entity(uuids[i]), Some(entities[i]));
            assert_eq!(index.get_stable(entities[i]), Some(uuids[i]));
        }

        // Verify removed are gone
        for i in [1, 3] {
            assert_eq!(index.get_entity(uuids[i]), None);
            assert_eq!(index.get_stable(entities[i]), None);
        }
    }
}
