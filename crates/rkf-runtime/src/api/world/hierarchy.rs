//! Entity hierarchy (parent/child relationships).

use uuid::Uuid;

use super::{World, WorldError};

impl World {
    // ── Hierarchy ──────────────────────────────────────────────────────────

    /// Set the parent of an entity.
    pub fn set_parent(
        &mut self,
        child: Uuid,
        parent: Uuid,
    ) -> Result<(), WorldError> {
        let _child_record = self
            .entities
            .get(&child)
            .ok_or(WorldError::NoSuchEntity(child))?;
        let _parent_record = self
            .entities
            .get(&parent)
            .ok_or(WorldError::NoSuchEntity(parent))?;

        // Cycle detection: walk up from parent and make sure we don't hit child
        let mut cursor = Some(parent);
        while let Some(cur_id) = cursor {
            if cur_id == child {
                return Err(WorldError::CycleDetected);
            }
            cursor = self
                .entities
                .get(&cur_id)
                .and_then(|r| r.parent_id);
        }

        // Update EntityRecord
        if let Some(record) = self.entities.get_mut(&child) {
            record.parent_id = Some(parent);
        }
        Ok(())
    }

    /// Remove an entity from its parent (make it a root).
    pub fn unparent(&mut self, entity_id: Uuid) -> Result<(), WorldError> {
        let _record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;

        if let Some(record) = self.entities.get_mut(&entity_id) {
            record.parent_id = None;
        }
        Ok(())
    }

    /// Iterate over the children of an entity.
    pub fn children(&self, entity_id: Uuid) -> impl Iterator<Item = Uuid> + '_ {
        self.entities
            .iter()
            .filter(move |(_, r)| r.parent_id == Some(entity_id))
            .map(|(id, _)| *id)
    }

    /// Get the parent of an entity, if it has one.
    pub fn parent(&self, entity_id: Uuid) -> Option<Uuid> {
        self.entities.get(&entity_id)?.parent_id
    }
}
