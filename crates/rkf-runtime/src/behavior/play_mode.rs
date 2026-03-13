//! Play/Stop mode infrastructure for the behavior system.
//!
//! Provides world cloning (edit → play), store snapshot/restore coordination,
//! the [`PlayModeManager`] state machine that orchestrates Play/Stop transitions,
//! frame execution integration, edit tool disabling, scene transitions during play,
//! and push-to-edit field transfer.

use std::collections::HashMap;
use std::path::PathBuf;

use super::executor::BehaviorExecutor;
use super::command_queue::CommandQueue;
use super::game_store::{GameStore, StoreSnapshot};
use super::game_value::GameValue;
use super::registry::GameplayRegistry;
use super::reload_queue::ReloadQueue;
use super::scene_ownership::SceneOwnership;
use super::stable_id::StableId;
use super::stable_id_index::StableIdIndex;
use crate::components::{
    CameraComponent, EditorMetadata, FogVolumeComponent, Parent, SdfTree, Transform,
};

// ─── PlayState ───────────────────────────────────────────────────────────────

/// Whether the engine is in Edit mode or Play mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayState {
    /// Normal editing — the edit world is live.
    Edit,
    /// Game is running — the play world is live, edit world is frozen.
    Playing,
}

// ─── clone_world_for_play ────────────────────────────────────────────────────

/// Clone the edit world into a fresh play world.
///
/// Only entities that carry a [`StableId`] component are cloned. Engine
/// components ([`Transform`], [`EditorMetadata`], [`SdfTree`],
/// [`CameraComponent`], [`FogVolumeComponent`]) are deep-cloned where present.
/// Gameplay components registered in the [`GameplayRegistry`] are cloned via
/// serialize/deserialize. [`Parent`] references are remapped from edit-entity
/// handles to play-entity handles.
///
/// Returns `(play_world, play_stable_index, edit_entity → play_entity mapping)`.
pub fn clone_world_for_play(
    edit_ecs: &hecs::World,
    edit_stable_index: &StableIdIndex,
    registry: &GameplayRegistry,
) -> (
    hecs::World,
    StableIdIndex,
    HashMap<hecs::Entity, hecs::Entity>,
) {
    let mut play_world = hecs::World::new();
    let mut play_stable_index = StableIdIndex::new();
    let mut entity_map: HashMap<hecs::Entity, hecs::Entity> = HashMap::new();

    // Phase 1: Spawn play entities with StableId + cloned components (no Parent yet).
    for (edit_entity, stable_id) in edit_ecs.query::<&StableId>().iter() {
        let play_entity = play_world.spawn((*stable_id,));
        play_stable_index.insert(stable_id.uuid(), play_entity);
        entity_map.insert(edit_entity, play_entity);

        // Clone engine components if present.
        // Dereference the hecs Ref to get the actual component, then clone.
        if let Ok(r) = edit_ecs.get::<&Transform>(edit_entity) {
            let cloned: Transform = (*r).clone();
            drop(r);
            let _ = play_world.insert_one(play_entity, cloned);
        }
        if let Ok(r) = edit_ecs.get::<&EditorMetadata>(edit_entity) {
            let cloned: EditorMetadata = (*r).clone();
            drop(r);
            let _ = play_world.insert_one(play_entity, cloned);
        }
        if let Ok(r) = edit_ecs.get::<&SdfTree>(edit_entity) {
            let cloned: SdfTree = (*r).clone();
            drop(r);
            let _ = play_world.insert_one(play_entity, cloned);
        }
        if let Ok(r) = edit_ecs.get::<&CameraComponent>(edit_entity) {
            let cloned: CameraComponent = (*r).clone();
            drop(r);
            let _ = play_world.insert_one(play_entity, cloned);
        }
        if let Ok(r) = edit_ecs.get::<&FogVolumeComponent>(edit_entity) {
            let cloned: FogVolumeComponent = *r;
            drop(r);
            let _ = play_world.insert_one(play_entity, cloned);
        }

        // Clone gameplay components via serialize/deserialize.
        // Skip engine components that were already cloned directly above.
        const DIRECT_CLONED: &[&str] = &[
            "Transform",
            "EditorMetadata",
            "SdfTree",
            "CameraComponent",
            "FogVolumeComponent",
        ];
        for entry in registry.component_entries() {
            if DIRECT_CLONED.contains(&entry.name) {
                continue;
            }
            if !(entry.has)(edit_ecs, edit_entity) {
                continue;
            }
            let ron_str = match (entry.serialize)(edit_ecs, edit_entity) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "play_mode: failed to serialize component '{}' on entity {:?} (serialize returned None)",
                        entry.name, edit_entity
                    );
                    continue;
                }
            };
            if let Err(e) = (entry.deserialize_insert)(&mut play_world, play_entity, &ron_str) {
                eprintln!(
                    "play_mode: failed to deserialize component '{}' into play entity {:?}: {}",
                    entry.name, play_entity, e
                );
            }
        }
    }

    // Phase 2: Resolve Parent references — remap edit entity to play entity.
    // Collect first to avoid borrow conflict on edit_ecs.
    let parents_to_remap: Vec<(hecs::Entity, Parent)> = edit_ecs
        .query::<(&StableId, &Parent)>()
        .iter()
        .filter_map(|(edit_entity, (_sid, parent))| {
            // Only remap if both child and parent were cloned.
            if entity_map.contains_key(&edit_entity) && entity_map.contains_key(&parent.entity) {
                Some((edit_entity, parent.clone()))
            } else {
                None
            }
        })
        .collect();

    for (edit_entity, edit_parent) in parents_to_remap {
        let play_child = entity_map[&edit_entity];
        let play_parent_entity = entity_map[&edit_parent.entity];
        let _ = play_world.insert_one(
            play_child,
            Parent {
                entity: play_parent_entity,
                bone_index: edit_parent.bone_index,
            },
        );
    }

    (play_world, play_stable_index, entity_map)
}

// ─── 12.4: Frame execution integration ──────────────────────────────────────

/// References needed during a play-mode frame tick.
///
/// Bundles the mutable world, store, and registry references along with the
/// frame delta time so that `run_play_frame` has everything in one struct.
pub struct PlayFrameContext<'a> {
    /// Seconds elapsed since the previous frame.
    pub dt: f32,
    /// The play world (mutable — systems may read/write via commands).
    pub world: &'a mut hecs::World,
    /// Gameplay state store.
    pub store: &'a mut GameStore,
    /// The gameplay registry (component + system catalog).
    pub registry: &'a GameplayRegistry,
}

/// Run one play-mode frame: wraps [`BehaviorExecutor::tick`].
///
/// Creates a [`CommandQueue`], calls `executor.tick()`, and returns.
/// The caller is responsible for checking [`should_run_systems`] before calling.
pub fn run_play_frame(
    executor: &mut BehaviorExecutor,
    world: &mut hecs::World,
    store: &mut GameStore,
    stable_ids: &mut StableIdIndex,
    registry: &GameplayRegistry,
    commands: &mut CommandQueue,
    dt: f32,
    total_time: f64,
    frame: u64,
) {
    executor.tick(registry, world, commands, store, stable_ids, dt, total_time, frame);
}

/// Returns `true` only when systems should be ticked (i.e. during [`PlayState::Playing`]).
pub fn should_run_systems(state: &PlayState) -> bool {
    *state == PlayState::Playing
}

// ─── 12.5: Edit tool disabling during play ──────────────────────────────────

/// Tool names that are disabled during play mode.
///
/// These tools modify the edit world (sculpt, paint, gizmo transforms,
/// inspector property edits) and must be blocked while the play world is live.
pub const DISABLED_PLAY_TOOLS: &[&str] = &[
    "sculpt",
    "paint",
    "gizmo",
    "inspector_edit",
    "transform_gizmo",
    "place",
    "delete",
];

/// Check whether a tool is allowed given the current play state.
///
/// During [`PlayState::Playing`], tools in [`DISABLED_PLAY_TOOLS`] are blocked.
/// During [`PlayState::Edit`], all tools are allowed. View/observation tools
/// (screenshot, camera orbit, etc.) are always allowed.
pub fn is_tool_allowed(state: &PlayState, tool: &str) -> bool {
    if *state == PlayState::Edit {
        return true;
    }
    // During Playing: block tools in the disabled list
    !DISABLED_PLAY_TOOLS.contains(&tool)
}

// ─── 12.7: Scene transitions during play ────────────────────────────────────

/// Load scene data into the play world.
///
/// Deserializes entities from `scene_data` (a `.rkscene` RON string in either
/// v2 or v3 format) and spawns them into `world`. Each spawned entity gets a
/// [`SceneOwnership`] component tagging it to the scene (using the tag `"loaded"`).
///
/// Returns the list of spawned entity handles.
pub fn load_scene_into_play_world(
    world: &mut hecs::World,
    scene_data: &[u8],
    _registry: &GameplayRegistry,
) -> Result<Vec<hecs::Entity>, String> {
    let ron_str = std::str::from_utf8(scene_data)
        .map_err(|e| format!("scene data is not valid UTF-8: {}", e))?;

    if ron_str.is_empty() {
        return Ok(Vec::new());
    }

    let scene = crate::scene_file_v3::deserialize_scene_v3(ron_str)
        .map_err(|e| format!("failed to parse scene: {}", e))?;

    let mut spawned = Vec::with_capacity(scene.entities.len());
    let mut uuid_to_hecs: HashMap<uuid::Uuid, hecs::Entity> = HashMap::new();

    // Phase 1: Spawn entities with StableId + SceneOwnership + components.
    for record in &scene.entities {
        let stable_id = StableId(record.stable_id);
        let entity = world.spawn((stable_id, SceneOwnership::for_scene("loaded")));
        uuid_to_hecs.insert(record.stable_id, entity);
        spawned.push(entity);

        // Engine components
        if let Some(Ok(t)) = record.get_component::<Transform>(
            crate::scene_file_v3::component_names::TRANSFORM,
        ) {
            let _ = world.insert_one(entity, t);
        }
        if let Some(Ok(m)) = record.get_component::<EditorMetadata>(
            crate::scene_file_v3::component_names::EDITOR_METADATA,
        ) {
            let _ = world.insert_one(entity, m);
        }
        if let Some(Ok(s)) = record.get_component::<SdfTree>(
            crate::scene_file_v3::component_names::SDF_TREE,
        ) {
            let _ = world.insert_one(entity, s);
        }
        if let Some(Ok(c)) = record.get_component::<CameraComponent>(
            crate::scene_file_v3::component_names::CAMERA,
        ) {
            let _ = world.insert_one(entity, c);
        }
        if let Some(Ok(f)) = record.get_component::<FogVolumeComponent>(
            crate::scene_file_v3::component_names::FOG_VOLUME,
        ) {
            let _ = world.insert_one(entity, f);
        }
    }

    // Phase 2: Resolve parent references.
    for record in &scene.entities {
        if let Some(parent_uuid) = record.parent {
            if let (Some(&child), Some(&parent)) = (
                uuid_to_hecs.get(&record.stable_id),
                uuid_to_hecs.get(&parent_uuid),
            ) {
                let _ = world.insert_one(
                    child,
                    Parent {
                        entity: parent,
                        bone_index: None,
                    },
                );
            }
        }
    }

    Ok(spawned)
}

/// Remove all entities owned by the named scene from the play world.
///
/// Iterates all entities with a [`SceneOwnership`] component and despawns those
/// whose `scene` matches `scene_name`. Persistent entities (`scene: None`) are
/// never removed.
pub fn unload_scene_from_play_world(world: &mut hecs::World, scene_name: &str) {
    let to_remove: Vec<hecs::Entity> = world
        .query::<&SceneOwnership>()
        .iter()
        .filter_map(|(entity, ownership)| {
            if ownership.belongs_to(scene_name) {
                Some(entity)
            } else {
                None
            }
        })
        .collect();

    for entity in to_remove {
        let _ = world.despawn(entity);
    }
}

// ─── 12.9: Push to edit ─────────────────────────────────────────────────────

/// Transfer a single field value from a component on an entity in the play world
/// to the same component on the corresponding entity in the edit world.
///
/// Uses [`ComponentEntry::get_field`] and [`ComponentEntry::set_field`] for
/// type-erased field access. The caller must identify the correct edit-world
/// entity (same `StableId`).
///
/// # Errors
///
/// Returns an error if:
/// - The component name is not found in the registry
/// - `get_field` fails on the play entity (missing component or field)
/// - `set_field` fails on the edit entity (missing component or field)
pub fn push_field_to_edit(
    play_world: &hecs::World,
    edit_world: &mut hecs::World,
    play_entity: hecs::Entity,
    edit_entity: hecs::Entity,
    component_name: &str,
    field_name: &str,
    registry: &GameplayRegistry,
) -> Result<(), String> {
    let entry = registry
        .component_entry(component_name)
        .ok_or_else(|| format!("component '{}' not found in registry", component_name))?;

    // Read the field value from the play world entity.
    let value: GameValue = (entry.get_field)(play_world, play_entity, field_name)?;

    // Write the field value to the edit world entity.
    (entry.set_field)(edit_world, edit_entity, field_name, value)?;

    Ok(())
}

// ─── PlayModeManager ─────────────────────────────────────────────────────────

/// Manages Play/Stop transitions: world cloning, store snapshotting, and state.
///
/// Includes an embedded [`ReloadQueue`] so that hot-reload builds that complete
/// during play are automatically deferred until Stop.
pub struct PlayModeManager {
    state: PlayState,
    store_snapshot: Option<StoreSnapshot>,
    reload_queue: ReloadQueue,
}

impl Default for PlayModeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayModeManager {
    /// Create a new manager in Edit state.
    pub fn new() -> Self {
        Self {
            state: PlayState::Edit,
            store_snapshot: None,
            reload_queue: ReloadQueue::new(),
        }
    }

    /// Access the embedded reload queue (e.g. to queue a build during play).
    pub fn reload_queue_mut(&mut self) -> &mut ReloadQueue {
        &mut self.reload_queue
    }

    /// Current play state.
    pub fn state(&self) -> PlayState {
        self.state
    }

    /// Whether the engine is currently in Play mode.
    pub fn is_playing(&self) -> bool {
        self.state == PlayState::Playing
    }

    /// Enter Play mode.
    ///
    /// 1. Snapshots the [`GameStore`].
    /// 2. Clones the edit world into a play world.
    /// 3. Sets state to [`PlayState::Playing`].
    ///
    /// Returns `(play_world, play_stable_index, entity_map)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if already playing.
    pub fn start_play(
        &mut self,
        edit_ecs: &hecs::World,
        edit_stable_index: &StableIdIndex,
        store: &GameStore,
        registry: &GameplayRegistry,
    ) -> Result<
        (
            hecs::World,
            StableIdIndex,
            HashMap<hecs::Entity, hecs::Entity>,
        ),
        String,
    > {
        if self.state == PlayState::Playing {
            return Err("already in Play mode".to_owned());
        }

        // Snapshot store before play.
        self.store_snapshot = Some(store.snapshot());

        // Clone world.
        let result = clone_world_for_play(edit_ecs, edit_stable_index, registry);

        self.state = PlayState::Playing;
        Ok(result)
    }

    /// Exit Play mode and restore the store.
    ///
    /// The caller is responsible for dropping the play world and switching back
    /// to the edit world.
    ///
    /// # Errors
    ///
    /// Returns `Err` if not currently playing.
    pub fn stop_play(&mut self, store: &mut GameStore) -> Result<(), String> {
        if self.state != PlayState::Playing {
            return Err("not in Play mode".to_owned());
        }

        // Restore store from snapshot.
        if let Some(snapshot) = self.store_snapshot.take() {
            store.restore(snapshot);
        }

        self.state = PlayState::Edit;
        Ok(())
    }

    /// Stop play mode and check an external reload queue for pending hot-reload paths.
    ///
    /// Combines `stop_play()` with a `ReloadQueue::take_pending()` check.
    /// Returns `Some(path)` if a hot-reload was queued during play.
    ///
    /// # Errors
    ///
    /// Returns `Err` if not currently playing.
    pub fn stop_and_check_reload(
        &mut self,
        store: &mut GameStore,
        reload_queue: &mut ReloadQueue,
    ) -> Result<Option<PathBuf>, String> {
        self.stop_play(store)?;
        Ok(reload_queue.take_pending())
    }

    /// Stop play mode and drain the embedded reload queue.
    ///
    /// If a hot-reload build completed during play (queued via
    /// [`reload_queue_mut()`]), this returns `Some(path)`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if not currently playing.
    pub fn stop_and_drain_reload(
        &mut self,
        store: &mut GameStore,
    ) -> Result<Option<PathBuf>, String> {
        self.stop_play(store)?;
        Ok(self.reload_queue.take_pending())
    }
}

// ─── Play inspector ─────────────────────────────────────────────────────────

/// Build an [`InspectorData`](super::inspector::InspectorData) snapshot from the play world.
///
/// Identical to [`build_inspector_data`](super::inspector::build_inspector_data)
/// but reads from the play world. All fields are populated normally — the caller
/// (editor UI) is responsible for rendering them as read-only during play mode.
pub fn build_play_inspector_data(
    play_world: &hecs::World,
    entity: hecs::Entity,
    registry: &GameplayRegistry,
) -> super::inspector::InspectorData {
    super::inspector::build_inspector_data(play_world, entity, registry)
}

