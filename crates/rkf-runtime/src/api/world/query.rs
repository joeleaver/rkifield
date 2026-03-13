//! Entity query and lookup operations.

use uuid::Uuid;

use super::{SceneMeta, World, WorldError};

impl World {
    // ── Query ──────────────────────────────────────────────────────────────

    /// Find the first entity with the given name.
    pub fn find(&self, name: &str) -> Option<Uuid> {
        self.entities
            .iter()
            .find(|(_, r)| r.name == name)
            .map(|(id, _)| *id)
    }

    /// Find all entities with the given name.
    pub fn find_all(&self, name: &str) -> Vec<Uuid> {
        self.entities
            .iter()
            .filter(|(_, r)| r.name == name)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Iterate over all live entity UUIDs.
    pub fn entities(&self) -> impl Iterator<Item = Uuid> + '_ {
        self.entities.keys().copied()
    }

    /// Get the name of an entity.
    pub fn name(&self, entity_id: Uuid) -> Result<&str, WorldError> {
        self.entities
            .get(&entity_id)
            .map(|r| r.name.as_str())
            .ok_or(WorldError::NoSuchEntity(entity_id))
    }

    /// Check if an entity is still alive.
    pub fn is_alive(&self, entity_id: Uuid) -> bool {
        self.entities.contains_key(&entity_id)
    }

    /// Number of live entities.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Remove all entities from the world.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.sdf_to_entity.clear();
        self.entity_scene.clear();
        self.ecs = hecs::World::new();
        let name = self.scenes.first().map(|s| s.name.clone()).unwrap_or_else(|| "default".into());
        self.scenes = vec![SceneMeta {
            name,
            persistent: false,
        }];
        self.active_scene = 0;
        self.next_sdf_id = 0;
    }

    /// Look up an entity UUID by its SDF object ID.
    ///
    /// Used by the renderer bridge to convert GPU-picked object IDs back to UUIDs.
    pub fn find_by_sdf_id(&self, obj_id: u32) -> Option<Uuid> {
        self.sdf_to_entity.get(&obj_id).copied()
    }
}
