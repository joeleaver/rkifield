//! Transform operations (position, rotation, scale).

use glam::{Quat, Vec3};
use uuid::Uuid;

use rkf_core::WorldPosition;

use super::{World, WorldError};

impl World {
    // ── Transforms ─────────────────────────────────────────────────────────

    /// Get the world position of an entity (reads from hecs Transform).
    pub fn position(&self, entity_id: Uuid) -> Result<WorldPosition, WorldError> {
        let record = self.entities.get(&entity_id).ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .get::<&crate::components::Transform>(record.ecs_entity)
            .map(|t| t.position)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
    }

    /// Set the world position of an entity (hecs is authoritative).
    pub fn set_position(
        &mut self,
        entity_id: Uuid,
        pos: WorldPosition,
    ) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        record.position = pos;
        let ecs_entity = record.ecs_entity;

        if let Ok(mut t) = self.ecs.get::<&mut crate::components::Transform>(ecs_entity) {
            t.position = pos;
        }
        Ok(())
    }

    /// Get the rotation of an entity (reads from hecs Transform).
    pub fn rotation(&self, entity_id: Uuid) -> Result<Quat, WorldError> {
        let record = self.entities.get(&entity_id).ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .get::<&crate::components::Transform>(record.ecs_entity)
            .map(|t| t.rotation)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
    }

    /// Set the rotation of an entity (hecs is authoritative).
    pub fn set_rotation(&mut self, entity_id: Uuid, rot: Quat) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        record.rotation = rot;
        let ecs_entity = record.ecs_entity;

        if let Ok(mut t) = self.ecs.get::<&mut crate::components::Transform>(ecs_entity) {
            t.rotation = rot;
        }
        Ok(())
    }

    /// Get the scale of an entity (reads from hecs Transform).
    pub fn scale(&self, entity_id: Uuid) -> Result<Vec3, WorldError> {
        let record = self.entities.get(&entity_id).ok_or(WorldError::NoSuchEntity(entity_id))?;
        self.ecs
            .get::<&crate::components::Transform>(record.ecs_entity)
            .map(|t| t.scale)
            .map_err(|_| WorldError::NoSuchEntity(entity_id))
    }

    /// Set the per-axis scale of an entity (hecs is authoritative).
    pub fn set_scale(&mut self, entity_id: Uuid, scale: Vec3) -> Result<(), WorldError> {
        let record = self
            .entities
            .get_mut(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        record.scale = scale;
        let ecs_entity = record.ecs_entity;

        if let Ok(mut t) = self.ecs.get::<&mut crate::components::Transform>(ecs_entity) {
            t.scale = scale;
        }
        Ok(())
    }
}
