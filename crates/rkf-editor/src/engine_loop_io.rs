//! Scene and project I/O helpers for the engine loop.
//!
//! All scene persistence uses the v3 entity-centric format exclusively.

use std::sync::{Arc, Mutex};

use crate::editor_state::EditorState;
use crate::engine::EditorEngine;
use crate::layout::state::LayoutBacking;

use rkf_runtime::behavior::{GameplayRegistry, StableId, StableIdIndex};
use rkf_runtime::scene_file_v3::{self, SceneFileV3};

/// Shared context for scene/project loading — holds references to engine
/// and editor state resources used across multiple I/O helpers.
pub(crate) struct IoContext<'a> {
    pub(crate) engine: &'a mut EditorEngine,
    pub(crate) editor_state: &'a Arc<Mutex<EditorState>>,
    pub(crate) layout_backing: &'a LayoutBacking,
    pub(crate) gameplay_registry: &'a Arc<Mutex<GameplayRegistry>>,
}

// ─── Save helpers ────────────────────────────────────────────────────────

/// Camera snapshot for scene save/load.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct CameraSnapshot {
    pub position: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

/// Save the current scene to a v3 `.rkscene` file.
///
/// 1. Exports .rkf files for voxelized objects (updates SdfTree.asset_path).
/// 2. Serializes all hecs entities via `scene_file_v3::save_scene()`.
/// 3. Stores camera/environment/lights in the properties bag.
/// 4. Writes the `.rkscene` file.
fn save_scene_v3(
    es: &mut EditorState,
    engine: &EditorEngine,
    registry: &GameplayRegistry,
    path: &str,
) -> anyhow::Result<()> {
    let scene_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));

    // Export .rkf files for voxelized objects.
    export_rkf_assets(es, engine, scene_dir);

    // Ensure all entities have StableIds.
    let stable_index = ensure_stable_ids(es.world.ecs_mut());

    // Serialize hecs entities to v3 format.
    let mut scene_v3 = scene_file_v3::save_scene(es.world.ecs(), &stable_index, registry);

    // Store editor state in properties bag.
    let cam_snap = CameraSnapshot {
        position: es.editor_camera.position,
        yaw: es.editor_camera.fly_yaw,
        pitch: es.editor_camera.fly_pitch,
        fov_y: es.editor_camera.fov_y,
    };
    if let Ok(s) = ron::to_string(&cam_snap) {
        scene_v3.properties.insert("camera".into(), s);
    }
    if let Ok(s) = ron::to_string(&es.environment) {
        scene_v3.properties.insert("environment".into(), s);
    }
    if let Ok(s) = ron::to_string(es.light_editor.all_lights()) {
        scene_v3.properties.insert("lights".into(), s);
    }

    // Serialize and write.
    let ron_str = scene_file_v3::serialize_scene_v3(&scene_v3)
        .map_err(|e| anyhow::anyhow!("RON serialize: {e}"))?;
    std::fs::write(path, ron_str)?;

    Ok(())
}

/// Export .rkf files for all voxelized objects, updating SdfTree.asset_path
/// in hecs so the v3 save captures the asset references.
fn export_rkf_assets(
    es: &mut EditorState,
    engine: &EditorEngine,
    scene_dir: &std::path::Path,
) {
    use rkf_core::scene_node::SdfSource;

    // Collect entity info: (hecs_entity, obj_id, name, sdf_source data).
    let render_scene = es.world.build_render_scene();
    let mut exports: Vec<(u32, String, SdfSource)> = Vec::new();
    for obj in &render_scene.objects {
        if matches!(&obj.root_node.sdf_source, SdfSource::Voxelized { .. }) {
            exports.push((obj.id, obj.name.clone(), obj.root_node.sdf_source.clone()));
        }
    }

    for (obj_id, name, sdf_source) in exports {
        if let SdfSource::Voxelized { brick_map_handle: _, voxel_size, aabb } = &sdf_source {
            let filename = format!("{}_{}.rkf", sanitize_filename(&name), obj_id);
            let rkf_path = scene_dir.join(&filename);

            // Prefer v3 (geometry-first) when data is available.
            let save_result = if let Some(gfd) = engine.geometry_first_data.get(&obj_id) {
                crate::scene_io::export_voxelized_to_rkf_v3(
                    &rkf_path, gfd, *voxel_size, aabb,
                    &engine.cpu_geometry_pool, &engine.cpu_sdf_cache_pool,
                )
            } else {
                // No geometry-first data — skip (analytical objects don't need .rkf).
                continue;
            };

            match save_result {
                Ok(()) => {
                    // Update SdfTree.asset_path in hecs so it's captured by v3 save.
                    if let Some(entity) = es.world.find_by_sdf_id(obj_id) {
                        if let Some(ecs_e) = es.world.ecs_entity_for(entity) {
                            if let Ok(mut sdf) = es.world.ecs_mut()
                                .get::<&mut rkf_runtime::components::SdfTree>(ecs_e)
                            {
                                sdf.asset_path = Some(filename);
                            }
                        }
                    }
                }
                Err(e) => log::error!("Failed to save .rkf for '{}': {e}", name),
            }
        }
    }
}

/// Sanitize an object name for use as a filename.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ─── Load helpers ────────────────────────────────────────────────────────

/// Load a v3 `.rkscene` file into the engine and editor state.
///
/// 1. Reads and parses the v3 scene file.
/// 2. Clears the engine and world.
/// 3. Loads entities into hecs via `scene_file_v3::load_scene()`.
/// 4. Rebuilds World entity tracking from hecs.
/// 5. Loads .rkf assets from SdfTree.asset_path.
/// 6. Restores camera/environment/lights from properties.
pub(crate) fn load_scene_v3(
    path: &str,
    engine: &mut EditorEngine,
    es: &mut EditorState,
    registry: &GameplayRegistry,
) -> anyhow::Result<()> {
    let ron_str = std::fs::read_to_string(path)?;
    let scene_v3 = scene_file_v3::deserialize_scene_v3(&ron_str)
        .map_err(|e| anyhow::anyhow!("parse scene: {e}"))?;

    engine.clear_scene();
    es.world.clear();

    // Load entities into hecs.
    let mut stable_index = StableIdIndex::new();
    scene_file_v3::load_scene(&scene_v3, es.world.ecs_mut(), &mut stable_index, registry);

    // Rebuild World entity tracking from hecs components.
    es.world.rebuild_entity_tracking_from_ecs();

    // Load .rkf assets for voxelized objects.
    let scene_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    load_rkf_assets_from_hecs(engine, es, scene_dir);
    engine.reupload_brick_data();

    // Restore editor state from properties.
    restore_properties_v3(es, &scene_v3);

    Ok(())
}

/// Load .rkf assets for entities with SdfTree.asset_path set.
///
/// Iterates hecs entities with SdfTree, loads referenced .rkf files into
/// the brick pool, and updates SdfTree.sdf_source to Voxelized.
fn load_rkf_assets_from_hecs(
    engine: &mut EditorEngine,
    es: &mut EditorState,
    scene_dir: &std::path::Path,
) {
    use rkf_runtime::components::SdfTree;

    // Collect (ecs_entity, asset_path, sdf_object_id) tuples.
    let mut to_load: Vec<(hecs::Entity, String, Option<u32>)> = Vec::new();
    for (entity, record) in es.world.entity_records() {
        if let Ok(sdf) = es.world.ecs_ref().get::<&SdfTree>(record.ecs_entity) {
            if let Some(ref asset_path) = sdf.asset_path {
                to_load.push((record.ecs_entity, asset_path.clone(), record.sdf_object_id));
            }
        }
    }

    for (ecs_entity, asset_path, obj_id) in to_load {
        let rkf_path = scene_dir.join(&asset_path);
        let rkf_str = rkf_path.to_string_lossy().to_string();
        match crate::engine::load_rkf_auto(
            &rkf_str,
            &mut engine.cpu_brick_pool,
            &mut engine.cpu_brick_map_alloc,
            &mut engine.cpu_geometry_pool,
            &mut engine.cpu_sdf_cache_pool,
        ) {
            Ok((handle, voxel_size, grid_aabb, _count, gf_data)) => {
                // Update the SdfTree component in hecs.
                if let Ok(mut sdf) = es.world.ecs_mut().get::<&mut SdfTree>(ecs_entity) {
                    sdf.root.sdf_source = rkf_core::SdfSource::Voxelized {
                        brick_map_handle: handle,
                        voxel_size,
                        aabb: grid_aabb,
                    };
                    sdf.aabb = grid_aabb;
                }
                // Store geometry-first data keyed by object ID.
                if let (Some(gfd), Some(oid)) = (gf_data, obj_id) {
                    engine.geometry_first_data.insert(oid, gfd);
                }
            }
            Err(e) => log::error!("Failed to load .rkf '{}': {e}", asset_path),
        }
    }
}

/// Restore camera, environment, and lights from a v3 scene file's properties.
fn restore_properties_v3(es: &mut EditorState, scene_v3: &SceneFileV3) {
    if let Some(s) = scene_v3.properties.get("camera") {
        if let Ok(cam) = ron::from_str::<CameraSnapshot>(s) {
            es.editor_camera.position = cam.position;
            es.editor_camera.fly_yaw = cam.yaw;
            es.editor_camera.fly_pitch = cam.pitch;
            es.editor_camera.fov_y = cam.fov_y;
            let dir = glam::Vec3::new(
                -cam.yaw.sin() * cam.pitch.cos(),
                cam.pitch.sin(),
                -cam.yaw.cos() * cam.pitch.cos(),
            );
            es.editor_camera.target = es.editor_camera.position
                + dir * es.editor_camera.orbit_distance;
        }
    }
    if let Some(s) = scene_v3.properties.get("environment") {
        if let Ok(env) = ron::from_str::<crate::environment::EnvironmentState>(s) {
            es.environment = env;
            es.environment.mark_dirty();
        }
    }
    if let Some(s) = scene_v3.properties.get("lights") {
        if let Ok(lights) = ron::from_str::<Vec<crate::light_editor::SceneLight>>(s) {
            es.light_editor.replace_lights(lights);
        }
    }
}

/// Resolve the default scene path for a project file.
pub(crate) fn resolve_default_scene_path(
    pf: &rkf_runtime::project::ProjectFile,
    project_root: &std::path::Path,
) -> Option<std::path::PathBuf> {
    if let Some(ref ds) = pf.default_scene {
        pf.scenes.iter()
            .find(|s| &s.name == ds)
            .map(|s| project_root.join(&s.path))
    } else {
        pf.scenes.first()
            .map(|s| project_root.join(&s.path))
    }
}

// ─── Scene/Project action handlers ──────────────────────────────────────

/// Handle the "Scene Save" action.
pub(crate) fn handle_scene_save(
    es: &mut EditorState,
    engine: &EditorEngine,
    registry: &GameplayRegistry,
) {
    let save_path = es.pending_save_path.take().or_else(|| {
        es.current_scene_path.clone()
    });
    if let Some(path) = save_path {
        match save_scene_v3(es, engine, registry, &path) {
            Ok(()) => {
                es.current_scene_path = Some(path.clone());
                es.unsaved_changes.mark_saved();
                log::info!("Scene saved to {path}");
            }
            Err(e) => log::error!("Failed to save scene: {e}"),
        }
    } else {
        es.pending_save_as = true;
    }
}

/// Handle the "Scene Save As" action.
pub(crate) fn handle_scene_save_as(
    es: std::sync::MutexGuard<'_, EditorState>,
    editor_state: &Arc<Mutex<EditorState>>,
    engine: &EditorEngine,
    registry: &GameplayRegistry,
) {
    // Clone data needed for save before dropping the lock.
    drop(es);

    let dialog_result = rfd::FileDialog::new()
        .add_filter("Scene", &["rkscene"])
        .set_file_name("scene.rkscene")
        .save_file();

    if let Some(file_path) = dialog_result {
        let path = file_path.to_string_lossy().to_string();
        let mut es = editor_state.lock().unwrap();
        match save_scene_v3(&mut es, engine, registry, &path) {
            Ok(()) => {
                es.current_scene_path = Some(path.clone());
                es.unsaved_changes.mark_saved();
                log::info!("Scene saved to {path}");
            }
            Err(e) => log::error!("Failed to save scene: {e}"),
        }
    }
}

/// Handle the "Scene Open" action -- full implementation.
pub(crate) fn handle_scene_open_impl(
    mut es: std::sync::MutexGuard<'_, EditorState>,
    open_path: Option<String>,
    ctx: &mut IoContext<'_>,
) {
    let file_path = if let Some(p) = open_path {
        Some(std::path::PathBuf::from(p))
    } else {
        drop(es);
        let result = rfd::FileDialog::new()
            .add_filter("Scene", &["rkscene"])
            .pick_file();
        if result.is_none() {
            return;
        }
        result
    };

    if let Some(file_path) = file_path {
        let path_str = file_path.to_string_lossy().to_string();
        let registry = ctx.gameplay_registry.lock().unwrap();
        let mut es = ctx.editor_state.lock().unwrap();
        match load_scene_v3(&path_str, ctx.engine, &mut es, &registry) {
            Ok(()) => {
                es.current_scene_path = Some(path_str.clone());
                es.unsaved_changes.mark_saved();
                es.selected_entity = None;
                es.undo.clear();
                log::info!("Scene loaded from {path_str}");
            }
            Err(e) => {
                log::error!("Failed to load scene '{}': {e}", path_str);
            }
        }
    }
}

/// Handle the "New Project" action.
pub(crate) fn handle_new_project(
    ctx: &mut IoContext<'_>,
    frame_topology_changed: &mut bool,
) {
    let dialog_result = rfd::FileDialog::new()
        .set_title("Choose parent folder for new project")
        .pick_folder();

    if let Some(parent_dir) = dialog_result {
        let project_name = parent_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "NewProject".to_string());

        match rkf_runtime::project::create_project(&parent_dir, &project_name) {
            Ok(project_path) => {
                let project_path_str = project_path.to_string_lossy().to_string();
                match rkf_runtime::load_project(&project_path_str) {
                    Ok(pf) => {
                        let project_root = rkf_runtime::project::project_root(&project_path);
                        let default_scene_path = resolve_default_scene_path(&pf, &project_root);

                        let registry = ctx.gameplay_registry.lock().unwrap();
                        let mut es = ctx.editor_state.lock().unwrap();

                        if let Some(scene_path) = default_scene_path {
                            let sp = scene_path.to_string_lossy().to_string();
                            match load_scene_v3(&sp, ctx.engine, &mut es, &registry) {
                                Ok(()) => {
                                    es.current_scene_path = Some(sp);
                                }
                                Err(e) => log::error!("Failed to load default scene: {e}"),
                            }
                        } else {
                            ctx.engine.clear_scene();
                            es.world.clear();
                            es.current_scene_path = None;
                        }
                        es.current_project = Some(pf);
                        es.current_project_path = Some(project_path_str.clone());
                        es.selected_entity = None;
                        es.undo.clear();
                        es.unsaved_changes.mark_saved();
                        *frame_topology_changed = true;
                        crate::editor_config::set_last_project(Some(&project_path_str));
                        log::info!("New project created at {project_path_str}");
                    }
                    Err(e) => log::error!("Failed to load new project file: {e}"),
                }
            }
            Err(e) => log::error!("Failed to create project: {e}"),
        }
    }
}

/// Handle the "Open Project" action.
pub(crate) fn handle_open_project(
    open_path: Option<String>,
    ctx: &mut IoContext<'_>,
    frame_topology_changed: &mut bool,
) {
    let file_path = if let Some(p) = open_path {
        Some(std::path::PathBuf::from(p))
    } else {
        rfd::FileDialog::new()
            .add_filter("Project", &["rkproject"])
            .pick_file()
    };

    if let Some(file_path) = file_path {
        let path_str = file_path.to_string_lossy().to_string();
        match rkf_runtime::load_project(&path_str) {
            Ok(pf) => {
                let project_root = rkf_runtime::project::project_root(&file_path);
                let default_scene_path = resolve_default_scene_path(&pf, &project_root);

                let registry = ctx.gameplay_registry.lock().unwrap();
                let mut es = ctx.editor_state.lock().unwrap();

                if let Some(scene_path) = default_scene_path {
                    let sp = scene_path.to_string_lossy().to_string();
                    match load_scene_v3(&sp, ctx.engine, &mut es, &registry) {
                        Ok(()) => {
                            es.current_scene_path = Some(sp);
                        }
                        Err(e) => log::error!("Failed to load scene: {e}"),
                    }
                } else {
                    ctx.engine.clear_scene();
                    es.world.clear();
                    es.current_scene_path = None;
                }

                // Restore editor layout if saved in project.
                if let Some(ref layout_ron) = pf.editor_layout {
                    ctx.layout_backing.from_ron(layout_ron);
                    let lb = ctx.layout_backing.clone();
                    rinch::shell::rinch_runtime::run_on_main_thread(move || {
                        let layout = rinch::core::use_context::<crate::layout::state::LayoutState>();
                        layout.load_from_backing(&lb);
                    });
                }
                es.current_project = Some(pf);
                es.current_project_path = Some(path_str.clone());
                es.selected_entity = None;
                es.undo.clear();
                es.unsaved_changes.mark_saved();
                *frame_topology_changed = true;
                crate::editor_config::set_last_project(Some(&path_str));
                log::info!("Project opened from {path_str}");
            }
            Err(e) => log::error!("Failed to load project '{}': {e}", path_str),
        }
    }
}

/// Handle the "Save Project" action (save scene + .rkproject file).
pub(crate) fn handle_save_project(
    es: &mut EditorState,
    engine: &EditorEngine,
    layout_backing: &LayoutBacking,
    registry: &GameplayRegistry,
) {
    if es.current_project.is_some() {
        // Save scene.
        let save_path = es.current_scene_path.clone();
        if let Some(path) = save_path {
            match save_scene_v3(es, engine, registry, &path) {
                Ok(()) => {
                    es.unsaved_changes.mark_saved();
                    log::info!("Scene saved to {path}");
                }
                Err(e) => log::error!("Failed to save scene: {e}"),
            }
        } else {
            es.pending_save_as = true;
        }

        // Save .rkproject file (with layout config).
        if let Some(pp) = es.current_project_path.clone() {
            if let Some(ref mut pf) = es.current_project {
                pf.engine_version = env!("CARGO_PKG_VERSION").to_string();
                pf.editor_layout = layout_backing.to_ron();
                match rkf_runtime::save_project(&pp, pf) {
                    Ok(()) => log::info!("Project saved to {pp}"),
                    Err(e) => log::error!("Failed to save project: {e}"),
                }
            }
        }
    } else {
        es.pending_save = true;
    }
}

// ─── Game dylib loading ─────────────────────────────────────────────────

/// Build and load the project's game crate dylib, registering its
/// components and systems into the gameplay registry.
///
/// Returns the `DylibLoader` handle (must be kept alive for fn_ptr validity)
/// or `None` if the build/load fails.
pub(crate) fn build_and_load_game_dylib(
    project_root: &std::path::Path,
    game_crate_name: &str,
    registry: &Arc<Mutex<GameplayRegistry>>,
    old_loader: Option<rkf_runtime::behavior::DylibLoader>,
) -> Option<rkf_runtime::behavior::DylibLoader> {
    let game_crate_dir = project_root.join(game_crate_name);
    if !game_crate_dir.join("Cargo.toml").exists() {
        log::warn!(
            "Game crate directory '{}' has no Cargo.toml, skipping dylib load",
            game_crate_dir.display()
        );
        return old_loader;
    }

    log::info!("Building game crate at {}...", game_crate_dir.display());

    // Synchronous build — blocks briefly on first load.
    let mut watcher = rkf_runtime::behavior::BuildWatcher::new(game_crate_dir);
    watcher.trigger_build();

    // Poll until complete (timeout after 120 seconds).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        watcher.poll();
        match watcher.state() {
            rkf_runtime::behavior::BuildState::Success(dylib_path) => {
                log::info!("Game crate built: {}", dylib_path.display());

                let mut reg = registry.lock().expect("registry lock");

                // Clear old gameplay components (keep engine ones).
                let engine_names = rkf_runtime::behavior::engine_components::ENGINE_COMPONENT_NAMES;
                reg.clear_gameplay(engine_names);

                // Drop old loader before loading new one.
                drop(old_loader);

                match rkf_runtime::behavior::DylibLoader::load(&dylib_path) {
                    Ok(loader) => {
                        match loader.call_register(&mut reg) {
                            Ok(()) => {
                                log::info!(
                                    "Game crate loaded: {} components, {} systems",
                                    reg.component_count(),
                                    reg.system_list().len(),
                                );
                                return Some(loader);
                            }
                            Err(e) => {
                                log::error!("Game crate register failed: {e}");
                                return None;
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to load game dylib: {e}");
                        return None;
                    }
                }
            }
            rkf_runtime::behavior::BuildState::Error(err) => {
                log::error!("Game crate build failed:\n{err}");
                return old_loader;
            }
            rkf_runtime::behavior::BuildState::Compiling => {
                if std::time::Instant::now() > deadline {
                    log::error!("Game crate build timed out");
                    return old_loader;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            rkf_runtime::behavior::BuildState::Idle => {
                // Should not happen after trigger_build
                return old_loader;
            }
        }
    }
}

// ─── Stable ID management ───────────────────────────────────────────────

/// Ensure every entity in the hecs World has a [`StableId`] component,
/// and build a [`StableIdIndex`] from the result.
pub(crate) fn ensure_stable_ids(ecs: &mut hecs::World) -> StableIdIndex {
    let mut index = StableIdIndex::new();

    let mut existing: Vec<(hecs::Entity, StableId)> = Vec::new();
    let mut missing: Vec<hecs::Entity> = Vec::new();
    for entity_ref in ecs.iter() {
        let e = entity_ref.entity();
        match ecs.get::<&StableId>(e) {
            Ok(sid) => existing.push((e, *sid)),
            Err(_) => missing.push(e),
        }
    }

    for (e, sid) in existing {
        index.insert(sid.0, e);
    }

    for e in missing {
        let sid = StableId::new();
        let _ = ecs.insert_one(e, sid);
        index.insert(sid.0, e);
    }

    index
}
