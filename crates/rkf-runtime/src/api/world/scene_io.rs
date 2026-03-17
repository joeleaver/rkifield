//! Scene load/save operations (v3 format).

use std::path::Path;

use uuid::Uuid;

use super::{World, WorldError};

impl World {
    // ── Scene I/O ──────────────────────────────────────────────────────────

    /// Load entities from a v3 `.rkscene` file into a specific or new scene.
    ///
    /// - `target_scene: None` — creates a new scene and loads into it.
    /// - `target_scene: Some(idx)` — loads into the existing scene at `idx`.
    ///
    /// Returns `(scene_index, loaded_entities)`.
    pub fn load_scene_into(
        &mut self,
        path: impl AsRef<Path>,
        target_scene: Option<usize>,
    ) -> Result<(usize, Vec<Uuid>), WorldError> {
        let path_ref = path.as_ref();
        let scene_name = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("loaded")
            .to_string();

        let scene_idx = match target_scene {
            Some(idx) => {
                if idx >= self.scenes.len() {
                    return Err(WorldError::SceneOutOfRange(idx));
                }
                idx
            }
            None => self.create_scene(&scene_name),
        };

        let prev_active = self.active_scene;
        self.active_scene = scene_idx;
        let result = self.load_scene(path_ref);
        self.active_scene = prev_active;

        result.map(|entities| (scene_idx, entities))
    }

    /// Save the entities in a specific scene to a v3 `.rkscene` file.
    pub fn save_scene_at(
        &mut self,
        index: usize,
        path: impl AsRef<Path>,
    ) -> Result<(), WorldError> {
        if index >= self.scenes.len() {
            return Err(WorldError::SceneOutOfRange(index));
        }

        use crate::behavior::GameplayRegistry;
        use crate::scene_file_v3;

        // Ensure all entities have StableIds for v3 save
        self.ensure_stable_ids();

        let stable_index = self.build_stable_id_index();
        let registry = GameplayRegistry::new();

        let scene_v3 = scene_file_v3::save_scene(self.ecs_ref(), &stable_index, &registry);
        let ron_str = scene_file_v3::serialize_scene_v3(&scene_v3)
            .map_err(|e| WorldError::Io(std::io::Error::other(e.to_string())))?;

        std::fs::write(path.as_ref(), &ron_str)
            .map_err(WorldError::Io)
    }

    /// Load entities from a v3 `.rkscene` file, adding them to the active scene.
    ///
    /// Returns the UUIDs of loaded entities.
    pub fn load_scene(&mut self, path: impl AsRef<Path>) -> Result<Vec<Uuid>, WorldError> {
        use crate::behavior::{GameplayRegistry, StableIdIndex};
        use crate::scene_file_v3;

        let ron_str = std::fs::read_to_string(path.as_ref())
            .map_err(WorldError::Io)?;

        let scene_v3 = scene_file_v3::deserialize_scene_v3(&ron_str)
            .map_err(|e| WorldError::Parse(e.to_string()))?;

        let registry = GameplayRegistry::new();
        let mut stable_index = StableIdIndex::new();

        scene_file_v3::load_scene(&scene_v3, &mut self.ecs, &mut stable_index, &registry);

        // Rebuild entity tracking from the newly loaded hecs entities
        self.rebuild_entity_tracking_from_ecs();

        // Return loaded entity UUIDs from the stable index
        let loaded_entities: Vec<Uuid> = stable_index
            .iter()
            .filter_map(|(uuid, _hecs_entity)| {
                if self.entities.contains_key(&uuid) {
                    Some(uuid)
                } else {
                    None
                }
            })
            .collect();

        Ok(loaded_entities)
    }

    /// Save the world's entities to a v3 `.rkscene` file.
    pub fn save_scene(&mut self, path: impl AsRef<Path>) -> Result<(), WorldError> {
        self.save_scene_at(0, path)
    }

    /// Ensure all hecs entities with Transform have a StableId.
    fn ensure_stable_ids(&mut self) {
        use crate::behavior::StableId;

        let needs_id: Vec<hecs::Entity> = self
            .ecs
            .query::<&crate::components::Transform>()
            .without::<&StableId>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for ecs_entity in needs_id {
            let _ = self.ecs.insert_one(ecs_entity, StableId::new());
        }
    }

    /// Build a StableIdIndex from current hecs entities.
    fn build_stable_id_index(&self) -> crate::behavior::StableIdIndex {
        use crate::behavior::{StableId, StableIdIndex};
        let mut index = StableIdIndex::new();
        for (entity, id) in self.ecs.query::<&StableId>().iter() {
            index.insert(id.uuid(), entity);
        }
        index
    }
}
