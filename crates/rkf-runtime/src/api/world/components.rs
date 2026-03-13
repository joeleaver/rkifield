//! ECS component operations (get, get_mut, insert, remove, has).

use uuid::Uuid;

use super::{World, WorldError};

impl World {
    // ── ECS Components ─────────────────────────────────────────────────────

    /// Get a shared reference to a component on an entity.
    pub fn get<C: hecs::Component>(&self, entity_id: Uuid) -> Result<hecs::Ref<'_, C>, WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .get::<&C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity_id, std::any::type_name::<C>()))
    }

    /// Get a mutable reference to a component on an entity.
    pub fn get_mut<C: hecs::Component>(
        &self,
        entity_id: Uuid,
    ) -> Result<hecs::RefMut<'_, C>, WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .get::<&mut C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity_id, std::any::type_name::<C>()))
    }

    /// Insert a component on an entity (overwrites if already present).
    pub fn insert<C: hecs::Component>(
        &mut self,
        entity_id: Uuid,
        component: C,
    ) -> Result<(), WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .insert_one(record.ecs_entity, component)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
    }

    /// Remove a component from an entity, returning it.
    pub fn remove<C: hecs::Component>(&mut self, entity_id: Uuid) -> Result<C, WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .remove_one::<C>(record.ecs_entity)
            .map_err(|_| WorldError::MissingComponent(entity_id, std::any::type_name::<C>()))
    }

    /// Check if an entity has a component.
    pub fn has<C: hecs::Component>(&self, entity_id: Uuid) -> bool {
        if let Some(record) = self.entities.get(&entity_id) {
            self.ecs.get::<&C>(record.ecs_entity).is_ok()
        } else {
            false
        }
    }
}
