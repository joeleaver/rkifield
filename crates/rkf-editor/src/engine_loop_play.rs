//! Play mode transitions and behavior tick logic, extracted from engine_loop.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::automation::SharedState;
use crate::editor_state::UiSignals;
use crate::engine::EditorEngine;

/// Mutable play mode state that lives across frames in the engine loop.
pub(crate) struct PlayState {
    pub(crate) play_mode: rkf_runtime::behavior::PlayModeManager,
    pub(crate) play_world: Option<hecs::World>,
    pub(crate) play_stable_ids_store: Option<rkf_runtime::behavior::StableIdIndex>,
    pub(crate) play_commands: rkf_runtime::behavior::CommandQueue,
    pub(crate) play_frame_number: u64,
    pub(crate) play_total_time: f64,
    /// Monotonic frame counter (always increments, not just during play).
    pub(crate) global_frame: u64,
    /// Maps play hecs::Entity → SceneObject.id for syncing play transforms to the renderer.
    /// Built once at play start from the edit world's entity tracking.
    pub(crate) play_entity_to_obj_id: HashMap<hecs::Entity, u32>,
    /// Saved editor camera state before play mode (restored on stop).
    pub(crate) pre_play_camera: Option<crate::camera::SceneCamera>,
    /// The hecs entity of the active scene camera in the play world (if any).
    pub(crate) play_active_camera: Option<hecs::Entity>,
}

impl PlayState {
    /// Whether play mode is currently active.
    pub(crate) fn is_playing(&self) -> bool {
        self.play_mode.is_playing()
    }

    pub(crate) fn new() -> Self {
        Self {
            play_mode: rkf_runtime::behavior::PlayModeManager::new(),
            play_world: None,
            play_stable_ids_store: None,
            play_commands: rkf_runtime::behavior::CommandQueue::new(),
            play_frame_number: 0,
            play_total_time: 0.0,
            global_frame: 0,
            play_entity_to_obj_id: HashMap::new(),
            pre_play_camera: None,
            play_active_camera: None,
        }
    }
}

/// Handle play mode start/stop transitions and tick behavior systems.
///
/// Returns `true` if any play-world transforms were synced back to the scene
/// (caller should mark objects dirty).
pub(crate) fn tick_play_mode(
    ps: &mut PlayState,
    f_play_start: bool,
    f_play_stop: bool,
    scene_clone: &mut rkf_core::scene::Scene,
    engine: &mut EditorEngine,
    shared_state: &Arc<Mutex<SharedState>>,
    gameplay_registry: &Arc<Mutex<rkf_runtime::behavior::GameplayRegistry>>,
    behavior_executor: &mut rkf_runtime::behavior::BehaviorExecutor,
    game_store: &mut rkf_runtime::behavior::GameStore,
    editor_state: &Arc<Mutex<crate::editor_state::EditorState>>,
    stable_ids: &rkf_runtime::behavior::StableIdIndex,
    dt: f32,
) {
    // ── Start transition ───────────────────────────────────────────────
    if f_play_start && !ps.play_mode.is_playing() {
        log::info!("Play mode: starting");
        // Lock EditorState briefly to access the real hecs edit world for cloning.
        let start_result = {
            let es = editor_state.lock().expect("editor_state lock for play start");
            let edit_world = es.world.ecs_ref();
            let registry = gameplay_registry.lock().unwrap();
            ps.play_mode.start_play(edit_world, stable_ids, game_store, &registry)
        };
        match start_result {
            Ok((pw, psi, remap)) => {
                let entity_count = pw.len();
                // Build play_entity → SceneObject.id map using the entity remap
                // and the edit world's entity records.
                ps.play_entity_to_obj_id.clear();
                {
                    let es = editor_state.lock().expect("editor_state lock for obj_id map");
                    for (_, record) in es.world.entity_records() {
                        if let Some(obj_id) = record.sdf_object_id {
                            // remap maps edit_hecs_entity → play_hecs_entity
                            if let Some(&play_entity) = remap.get(&record.ecs_entity) {
                                ps.play_entity_to_obj_id.insert(play_entity, obj_id);
                            }
                        }
                    }
                }
                // Check for active camera in the play world.
                // Find the hecs entity with CameraComponent.active == true.
                let mut active_cam_entity = None;
                for (entity, cam) in pw.query::<&rkf_runtime::components::CameraComponent>().iter() {
                    if cam.active {
                        active_cam_entity = Some(entity);
                        break;
                    }
                }
                if active_cam_entity.is_some() {
                    // Save editor camera state for restoration on stop.
                    let es = editor_state.lock().expect("editor_state lock for camera save");
                    ps.pre_play_camera = Some(es.editor_camera);
                }
                ps.play_active_camera = active_cam_entity;

                ps.play_world = Some(pw);
                ps.play_stable_ids_store = Some(psi);
                ps.play_frame_number = 0;
                ps.play_total_time = 0.0;
                if let Ok(mut ss) = shared_state.lock() {
                    ss.play_mode_state = crate::automation::PlayModeState::Playing;
                }
                log::info!(
                    "Play mode: started with {} entities",
                    entity_count
                );
            }
            Err(e) => {
                log::error!("Play mode start failed: {e}");
            }
        }
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.play_state.set(true);
            }
        });
    }

    // ── Stop transition ────────────────────────────────────────────────
    if f_play_stop && ps.play_mode.is_playing() {
        log::info!("Play mode: stopping");
        let _ = ps.play_mode.stop_play(game_store);
        ps.play_world = None;
        ps.play_stable_ids_store = None;
        ps.play_commands = rkf_runtime::behavior::CommandQueue::new();
        ps.play_frame_number = 0;
        ps.play_total_time = 0.0;

        // Restore editor camera if it was saved before play.
        if let Some(saved_camera) = ps.pre_play_camera.take() {
            if let Ok(mut es) = editor_state.lock() {
                es.editor_camera = saved_camera;
            }
        }
        ps.play_active_camera = None;

        // Restore scene object transforms from the edit world.
        if let Ok(es) = editor_state.lock() {
            for (_, record) in es.world.entity_records() {
                if let Some(obj_id) = record.sdf_object_id {
                    if let Ok(t) = es.world.ecs_ref().get::<&rkf_runtime::components::Transform>(record.ecs_entity) {
                        let pos = t.position.to_vec3();
                        if let Some(obj) = scene_clone.objects.iter_mut().find(|o| o.id == obj_id) {
                            obj.position = pos;
                            obj.rotation = t.rotation;
                            obj.scale = t.scale;
                            engine.dirty_objects.insert(obj_id);
                        }
                    }
                }
            }
        }

        if let Ok(mut ss) = shared_state.lock() {
            ss.play_mode_state = crate::automation::PlayModeState::Stopped;
        }
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.play_state.set(false);
            }
        });
    }

    // ── Tick behavior systems during play ──────────────────────────────
    if ps.play_mode.is_playing() {
        if let (Some(world), Some(sids)) =
            (&mut ps.play_world, &mut ps.play_stable_ids_store)
        {
            ps.play_total_time += dt as f64;
            ps.play_frame_number += 1;
            let reg = gameplay_registry
                .lock()
                .expect("gameplay registry lock");
            rkf_runtime::behavior::run_play_frame(
                behavior_executor,
                world,
                game_store,
                sids,
                &reg,
                &mut ps.play_commands,
                &engine.console,
                dt,
                ps.play_total_time,
                ps.play_frame_number,
            );
            // Sync play world transforms back to the rendered scene.
            // Build obj_id → updated transform map from the play world.
            let mut obj_transforms: HashMap<u32, (glam::Vec3, glam::Quat, glam::Vec3)> = HashMap::new();
            for (play_entity, transform) in world
                .query::<&rkf_runtime::components::Transform>()
                .iter()
            {
                if let Some(&obj_id) = ps.play_entity_to_obj_id.get(&play_entity) {
                    obj_transforms.insert(obj_id, (
                        transform.position.to_vec3(),
                        transform.rotation,
                        transform.scale,
                    ));
                }
            }
            // Apply to scene objects by matching object ID.
            for obj in scene_clone.objects.iter_mut() {
                if let Some(&(pos, rot, scale)) = obj_transforms.get(&obj.id) {
                    obj.position = pos;
                    obj.rotation = rot;
                    obj.scale = scale;
                    engine.dirty_objects.insert(obj.id);
                }
            }

            // Sync active scene camera to editor camera during play.
            if let Some(cam_entity) = ps.play_active_camera {
                if let Ok(mut es) = editor_state.lock() {
                    es.editor_camera.sync_from_entity(world, cam_entity);

                    // Resolve environment from the active camera's profile → editor camera entity.
                    let profile_path = world
                        .get::<&rkf_runtime::components::CameraComponent>(cam_entity)
                        .ok()
                        .map(|c| c.environment_profile.clone())
                        .unwrap_or_default();
                    if !profile_path.is_empty() {
                        if let Ok(profile) = rkf_runtime::environment::load_environment(&profile_path) {
                            let settings = rkf_runtime::environment::EnvironmentSettings::from_profile(&profile);
                            if let Some(editor_cam_uuid) = es.editor_camera_entity {
                                if let Some(ee) = es.world.ecs_entity_for(editor_cam_uuid) {
                                    let _ = es.world.ecs_mut().insert_one(ee, settings);
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    // ── Push systems panel data (always, not just during play) ────────
    ps.global_frame += 1;
    if ps.global_frame % 10 == 1 {
        let reg = gameplay_registry
            .lock()
            .expect("gameplay registry lock for systems panel");
        let entries = rkf_runtime::behavior::build_systems_panel(&reg, behavior_executor);
        let summaries: Vec<crate::ui_snapshot::SystemSummary> = entries
            .into_iter()
            .map(|e| crate::ui_snapshot::SystemSummary {
                name: e.name,
                phase: format!("{:?}", e.phase),
                order: e.order,
                faulted: e.faulted,
                last_frame_us: e.last_frame_us,
            })
            .collect();
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.systems.set(summaries);
            }
        });
    }
}
