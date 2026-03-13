//! Entity spawning, despawning, and finalization.

use glam::{Quat, Vec3};
use uuid::Uuid;

use rkf_core::aabb::Aabb;
use rkf_core::scene_node::SceneNode;
use rkf_core::WorldPosition;

use crate::behavior::StableId;
use crate::components::{EditorMetadata, SdfTree};

use super::{EntityRecord, SceneLink, SpawnBuilder, World, WorldError};

impl World {
    // ── Spawning ───────────────────────────────────────────────────────────

    /// Begin building a new entity with the given name.
    ///
    /// Call methods on the returned [`SpawnBuilder`] to configure the entity,
    /// then call `.build()` to finalize.
    pub fn spawn(&mut self, name: impl Into<String>) -> SpawnBuilder<'_> {
        SpawnBuilder::new(self, name.into())
    }

    /// Despawn an entity, removing it from the world entirely.
    ///
    /// If the entity has children, they are also despawned recursively.
    pub fn despawn(&mut self, entity_id: Uuid) -> Result<(), WorldError> {
        if !self.entities.contains_key(&entity_id) {
            return Err(WorldError::NoSuchEntity(entity_id));
        }

        // Collect children to despawn recursively
        let children: Vec<Uuid> = self.children(entity_id).collect();
        for child in children {
            let _ = self.despawn(child);
        }

        let record = self.entities.remove(&entity_id).unwrap();

        // Remove from hecs
        let _ = self.ecs.despawn(record.ecs_entity);

        // Remove SDF tracking
        if let Some(obj_id) = record.sdf_object_id {
            self.sdf_to_entity.remove(&obj_id);
        }
        self.entity_scene.remove(&entity_id);

        Ok(())
    }

    // ── Internal spawning ──────────────────────────────────────────────────

    /// Finalize an SDF entity spawn (called by SpawnBuilder).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize_sdf_spawn(
        &mut self,
        name: String,
        position: WorldPosition,
        rotation: Quat,
        scale: Vec3,
        mut root_node: SceneNode,
        material_id: u16,
        blend_mode: Option<rkf_core::scene_node::BlendMode>,
        parent: Option<Uuid>,
        aabb: Aabb,
    ) -> Uuid {
        let _ = material_id; // Material is already set on the root_node

        // Resolve parent
        let resolved_parent_id = parent.filter(|p| self.entities.contains_key(p));

        if let Some(bm) = blend_mode {
            root_node.blend_mode = bm;
        }

        // Assign object ID from global counter
        let obj_id = self.next_sdf_id;
        self.next_sdf_id = obj_id + 1;

        let stable_id = StableId::new();
        let uuid = stable_id.uuid();

        // Create hecs entity with full component data (hecs is the sole authority)
        let transform = crate::components::Transform {
            position,
            rotation,
            scale,
        };
        let sdf_tree = SdfTree {
            root: root_node,
            asset_path: None,
            aabb,
        };
        let editor_meta = EditorMetadata {
            name: name.clone(),
            tags: Vec::new(),
            locked: false,
        };
        let ecs_entity = self.ecs.spawn((
            SceneLink { object_id: obj_id },
            transform,
            sdf_tree,
            editor_meta,
            stable_id,
        ));

        let record = EntityRecord {
            ecs_entity,
            sdf_object_id: Some(obj_id),
            parent_id: resolved_parent_id,
            position,
            rotation,
            scale,
            name,
        };

        self.entities.insert(uuid, record);
        self.sdf_to_entity.insert(obj_id, uuid);
        self.entity_scene.insert(uuid, self.active_scene);

        uuid
    }

    /// Finalize an ECS-only entity spawn (called by SpawnBuilder).
    pub(crate) fn finalize_ecs_spawn(&mut self, name: String) -> Uuid {
        let stable_id = StableId::new();
        let uuid = stable_id.uuid();

        let ecs_entity = self.ecs.spawn((
            crate::components::Transform::default(),
            crate::components::EditorMetadata {
                name: name.clone(),
                ..Default::default()
            },
            stable_id,
        ));

        let record = EntityRecord {
            ecs_entity,
            sdf_object_id: None,
            parent_id: None,
            position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            name,
        };

        self.entities.insert(uuid, record);
        self.entity_scene.insert(uuid, self.active_scene);
        uuid
    }
}
