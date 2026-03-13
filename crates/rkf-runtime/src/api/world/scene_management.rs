//! Multi-scene management, persistence, and swap operations.

use uuid::Uuid;

use rkf_core::scene::{Scene, SceneObject};
use rkf_core::WorldPosition;

use crate::behavior::StableId;
use crate::components::{EditorMetadata, SdfTree};

use super::{EntityRecord, SceneLink, SceneMeta, World, WorldError};

impl World {
    // ── Multi-scene management ──────────────────────────────────────────

    /// Create a new empty scene and return its index.
    pub fn create_scene(&mut self, name: impl Into<String>) -> usize {
        let idx = self.scenes.len();
        self.scenes.push(SceneMeta {
            name: name.into(),
            persistent: false,
        });
        idx
    }

    /// Number of loaded scenes.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Index of the currently active scene.
    pub fn active_scene_index(&self) -> usize {
        self.active_scene
    }

    /// Switch the active scene.
    ///
    /// All spawn operations target the active scene. Existing entities
    /// remain accessible regardless of which scene is active.
    pub fn set_active_scene(&mut self, index: usize) {
        assert!(index < self.scenes.len(), "scene index out of range");
        self.active_scene = index;
    }

    /// Get the name of a scene by index.
    pub fn scene_name(&self, index: usize) -> Option<&str> {
        self.scenes.get(index).map(|s| s.name.as_str())
    }

    /// Total number of SDF objects across all scenes (counts from entity tracking).
    pub fn total_object_count(&self) -> usize {
        self.entities.values().filter(|r| r.sdf_object_id.is_some()).count()
    }

    /// Iterate over all SDF entity IDs (for backward compat with tests expecting all_objects).
    pub fn all_sdf_entity_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.entities.values().filter_map(|r| r.sdf_object_id)
    }

    /// Build a merged Scene from hecs entity data for the render thread.
    ///
    /// Queries all entities with Transform + SdfTree + SceneLink components
    /// and reconstructs SceneObject data from them. This is the bridge that
    /// allows hecs to be the source of truth while the render engine still
    /// consumes `&Scene`.
    pub fn build_render_scene(&self) -> Scene {
        let mut scene = Scene::new("merged");

        for (_ecs_entity, (link, transform, sdf_tree)) in self
            .ecs
            .query::<(&SceneLink, &crate::components::Transform, &SdfTree)>()
            .iter()
        {
            let name = self
                .ecs
                .get::<&EditorMetadata>(_ecs_entity)
                .map(|m| m.name.clone())
                .unwrap_or_default();

            // Resolve parent_id from EntityRecord
            let parent_id = self
                .sdf_to_entity
                .get(&link.object_id)
                .and_then(|uuid| self.entities.get(uuid))
                .and_then(|record| {
                    // Look up parent's SDF object ID from parent UUID
                    record.parent_id.and_then(|parent_uuid| {
                        self.entities.get(&parent_uuid).and_then(|pr| pr.sdf_object_id)
                    })
                });

            let obj = SceneObject {
                id: link.object_id,
                name,
                parent_id,
                position: transform.position.to_vec3(),
                rotation: transform.rotation,
                scale: transform.scale,
                root_node: sdf_tree.root.clone(),
                aabb: sdf_tree.aabb,
            };
            scene.objects.push(obj);
        }

        scene
    }

    // ── Scene persistence ───────────────────────────────────────────────

    /// Mark a scene as persistent (survives scene swaps).
    pub fn set_scene_persistent(&mut self, index: usize, persistent: bool) {
        if let Some(sm) = self.scenes.get_mut(index) {
            sm.persistent = persistent;
        }
    }

    /// Check if a scene is persistent.
    pub fn is_scene_persistent(&self, index: usize) -> bool {
        self.scenes.get(index).map(|sm| sm.persistent).unwrap_or(false)
    }

    /// Unload all non-persistent scenes, despawning their entities.
    ///
    /// Returns the names of the removed scenes. Adjusts the active scene
    /// index to remain valid (falls back to 0).
    pub fn swap_scenes(&mut self) -> Vec<String> {
        let mut removed_names = Vec::new();

        // Collect indices of non-persistent scenes (reverse order for removal)
        let to_remove: Vec<usize> = self
            .scenes
            .iter()
            .enumerate()
            .filter(|(_, sm)| !sm.persistent)
            .map(|(i, _)| i)
            .collect();

        // Despawn entities belonging to non-persistent scenes
        let entities_to_despawn: Vec<Uuid> = self
            .entity_scene
            .iter()
            .filter(|(_, si)| to_remove.contains(si))
            .map(|(id, _)| *id)
            .collect();

        for entity_id in entities_to_despawn {
            let _ = self.despawn(entity_id);
        }

        // Remove scenes in reverse order to preserve indices
        for &idx in to_remove.iter().rev() {
            removed_names.push(self.scenes[idx].name.clone());
            self.scenes.remove(idx);
        }

        // If all scenes were removed, create a default empty one
        if self.scenes.is_empty() {
            self.scenes.push(SceneMeta {
                name: "default".into(),
                persistent: false,
            });
        }

        // Adjust active scene index
        if self.active_scene >= self.scenes.len() {
            self.active_scene = 0;
        }

        removed_names
    }

    /// Remove a scene by index.
    ///
    /// If the scene is persistent, `force` must be true. Cannot remove the
    /// last scene. Despawns all entities belonging to the removed scene.
    pub fn remove_scene(&mut self, index: usize, force: bool) -> Result<String, WorldError> {
        if index >= self.scenes.len() {
            return Err(WorldError::SceneOutOfRange(index));
        }
        if self.scenes.len() == 1 {
            return Err(WorldError::CannotRemoveLastScene);
        }
        if self.scenes[index].persistent && !force {
            return Err(WorldError::Parse(
                "scene is persistent; use force=true".to_string(),
            ));
        }

        // Despawn entities belonging to this scene
        let entities_to_despawn: Vec<Uuid> = self
            .entity_scene
            .iter()
            .filter(|(_, si)| **si == index)
            .map(|(id, _)| *id)
            .collect();
        for entity_id in entities_to_despawn {
            let _ = self.despawn(entity_id);
        }

        let name = self.scenes[index].name.clone();
        self.scenes.remove(index);

        // Update entity_scene indices for scenes after the removed one
        for si in self.entity_scene.values_mut() {
            if *si > index {
                *si -= 1;
            }
        }

        // Adjust active scene
        if self.active_scene >= self.scenes.len() {
            self.active_scene = self.scenes.len() - 1;
        }

        Ok(name)
    }

    // ── Transitional helpers ──────────────────────────────────────────

    /// Load entities from a transient Scene into the world.
    ///
    /// Clears existing entities and spawns all SceneObjects from the given
    /// Scene as hecs entities, preserving original object IDs.
    pub fn load_from_scene(&mut self, scene: &rkf_core::scene::Scene) {
        // Clear existing world state
        self.clear();

        // Spawn each object preserving its original ID
        for obj in &scene.objects {
            let position = WorldPosition::new(glam::IVec3::ZERO, obj.position);
            let obj_id = obj.id;

            // Ensure next_sdf_id stays ahead of assigned IDs
            if obj_id >= self.next_sdf_id {
                self.next_sdf_id = obj_id + 1;
            }

            let stable_id = StableId::new();
            let uuid = stable_id.uuid();

            let transform = crate::components::Transform {
                position,
                rotation: obj.rotation,
                scale: obj.scale,
            };
            let sdf_tree = SdfTree {
                root: obj.root_node.clone(),
                asset_path: None,
                aabb: obj.aabb,
            };
            let editor_meta = EditorMetadata {
                name: obj.name.clone(),
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
                parent_id: None, // Resolved below
                position,
                rotation: obj.rotation,
                scale: obj.scale,
                name: obj.name.clone(),
            };

            self.entities.insert(uuid, record);
            self.sdf_to_entity.insert(obj_id, uuid);
            self.entity_scene.insert(uuid, self.active_scene);
        }

        // Resolve parent relationships using original scene IDs
        for obj in &scene.objects {
            if let Some(parent_obj_id) = obj.parent_id {
                if let Some(&child_uuid) = self.sdf_to_entity.get(&obj.id) {
                    if let Some(&parent_uuid) = self.sdf_to_entity.get(&parent_obj_id) {
                        if let Some(record) = self.entities.get_mut(&child_uuid) {
                            record.parent_id = Some(parent_uuid);
                        }
                    }
                }
            }
        }
    }

    // ── Entity tracking rebuild ──────────────────────────────────────────

    /// Rebuild entity tracking from hecs entities.
    ///
    /// Scans all hecs entities with SceneLink + Transform + EditorMetadata
    /// and creates corresponding EntityRecord entries.
    ///
    /// Call this after loading a scene file into hecs to build the entity
    /// tracking maps.
    pub fn rebuild_entity_tracking_from_ecs(&mut self) {
        // Clear existing tracking
        self.entities.clear();
        self.sdf_to_entity.clear();
        self.entity_scene.clear();

        // Phase 0: Ensure entities with SdfTree get a SceneLink if they don't have one.
        // This handles entities loaded from v3 scene files, which have SdfTree but no SceneLink.
        let needs_link: Vec<hecs::Entity> = self
            .ecs
            .query::<(&SdfTree, &crate::components::Transform)>()
            .without::<&SceneLink>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for ecs_entity in needs_link {
            let obj_id = self.next_sdf_id;
            self.next_sdf_id += 1;
            let _ = self.ecs.insert_one(ecs_entity, SceneLink { object_id: obj_id });
        }

        // Phase 0b: Ensure all entities have StableIds
        let needs_stable_id: Vec<hecs::Entity> = self
            .ecs
            .query::<&crate::components::Transform>()
            .without::<&StableId>()
            .iter()
            .map(|(e, _)| e)
            .collect();
        for ecs_entity in needs_stable_id {
            let _ = self.ecs.insert_one(ecs_entity, StableId::new());
        }

        // Phase 1: Register SDF entities (those with SceneLink)
        for (ecs_entity, (link, transform, meta, stable_id)) in self
            .ecs
            .query::<(&SceneLink, &crate::components::Transform, &EditorMetadata, &StableId)>()
            .iter()
        {
            let obj_id = link.object_id;
            let uuid = stable_id.uuid();

            let record = EntityRecord {
                ecs_entity,
                sdf_object_id: Some(obj_id),
                parent_id: None, // Will be resolved in a second pass
                position: transform.position,
                rotation: transform.rotation,
                scale: transform.scale,
                name: meta.name.clone(),
            };

            self.entities.insert(uuid, record);
            self.sdf_to_entity.insert(obj_id, uuid);
            self.entity_scene.insert(uuid, self.active_scene);

            if obj_id >= self.next_sdf_id {
                self.next_sdf_id = obj_id + 1;
            }
        }

        // Phase 2: Pick up ECS-only entities (Transform + EditorMetadata + StableId but no SceneLink)
        let sdf_ecs_entities: std::collections::HashSet<hecs::Entity> = self
            .ecs
            .query::<&SceneLink>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for (ecs_entity, (transform, meta, stable_id)) in self
            .ecs
            .query::<(&crate::components::Transform, &EditorMetadata, &StableId)>()
            .iter()
        {
            if sdf_ecs_entities.contains(&ecs_entity) {
                continue; // Already handled above
            }

            let uuid = stable_id.uuid();

            let record = EntityRecord {
                ecs_entity,
                sdf_object_id: None,
                parent_id: None,
                position: transform.position,
                rotation: transform.rotation,
                scale: transform.scale,
                name: meta.name.clone(),
            };

            self.entities.insert(uuid, record);
            self.entity_scene.insert(uuid, self.active_scene);
        }
    }
}
