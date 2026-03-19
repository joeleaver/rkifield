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
    pub(crate) material_library: &'a Arc<Mutex<rkf_core::material_library::MaterialLibrary>>,
}

// ─── Save helpers ────────────────────────────────────────────────────────

/// Camera snapshot for scene save/load.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SavedCameraSnapshot {
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

    // Filter out the editor camera entity — it's transient, not part of the scene.
    if let Some(editor_cam_id) = es.editor_camera_entity {
        scene_v3.entities.retain(|e| e.stable_id != editor_cam_id);
    }

    // Store editor state in properties bag.
    let snap = es.extract_camera_snapshot();
    let cam_snap = SavedCameraSnapshot {
        position: snap.position,
        yaw: snap.yaw,
        pitch: snap.pitch,
        fov_y: snap.fov_degrees.to_radians(),
    };
    if let Ok(s) = ron::to_string(&cam_snap) {
        scene_v3.properties.insert("camera".into(), s);
    }
    // Environment is stored as the EnvironmentSettings component on the
    // editor camera entity — no separate properties bag entry needed.
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

    let saved_snap = if es.editor_camera_entity.is_some() {
        Some(es.extract_camera_snapshot())
    } else {
        None
    };
    engine.clear_scene();
    es.world.clear();
    es.respawn_editor_camera(saved_snap);

    // Load entities into hecs.
    let mut stable_index = StableIdIndex::new();
    scene_file_v3::load_scene(&scene_v3, es.world.ecs_mut(), &mut stable_index, registry);

    // Rebuild World entity tracking from hecs components.
    es.world.rebuild_entity_tracking_from_ecs();

    // Migrate old scenes: restore environment to editor camera entity.
    if let Some(env_str) = scene_v3.properties.get("environment") {
        if let Ok(old_env) = ron::from_str::<crate::environment::EnvironmentState>(env_str) {
            if let Some(editor_cam) = es.editor_camera_entity {
                if let Some(ee) = es.world.ecs_entity_for(editor_cam) {
                    let settings = old_env.to_settings();
                    let _ = es.world.ecs_mut().insert_one(ee, settings);
                }
            }
        }
    }
    // Migrate SceneEnvironment entities: copy EnvironmentSettings to editor camera.
    if let Some(scene_env_e) = es.world.scene_environment_entity() {
        let settings = es.world.ecs_ref()
            .get::<&rkf_runtime::environment::EnvironmentSettings>(scene_env_e)
            .ok()
            .map(|s| (*s).clone());
        if let Some(settings) = settings {
            if let Some(editor_cam) = es.editor_camera_entity {
                if let Some(ee) = es.world.ecs_entity_for(editor_cam) {
                    let _ = es.world.ecs_mut().insert_one(ee, settings);
                }
            }
        }
    }

    // Load .rkf assets for voxelized objects.
    let scene_dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    load_rkf_assets_from_hecs(engine, es, scene_dir);
    engine.reupload_brick_data();

    // Upload color pool if any assets had per-voxel color, and rebuild the
    // shading pass bind group so it references the new GPU buffers.
    if !engine.cpu_color_bricks.is_empty() {
        let color_data: &[u8] = bytemuck::cast_slice(&engine.cpu_color_bricks);
        engine.gpu_color_pool = rkf_render::GpuColorPool::upload(
            &engine.ctx.device,
            color_data,
            &engine.color_companion_map,
        );
        engine.shading_pass.rebuild_group3(
            &engine.ctx.device,
            &engine.brush_overlay,
            &engine.gpu_color_pool,
        );
    }

    // Restore editor state from properties (camera, lights — NOT environment).
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
    for (_entity, record) in es.world.entity_records() {
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
            Ok((handle, voxel_size, grid_aabb, _count, gf_data, color_bricks)) => {
                // Update the SdfTree component in hecs.
                if let Ok(mut sdf) = es.world.ecs_mut().get::<&mut SdfTree>(ecs_entity) {
                    sdf.root.sdf_source = rkf_core::SdfSource::Voxelized {
                        brick_map_handle: handle,
                        voxel_size,
                        aabb: grid_aabb,
                    };
                    sdf.aabb = grid_aabb;
                } else {
                    log::error!("  FAILED to update SdfTree component!");
                }
                // Store geometry-first data keyed by object ID.
                if let (Some(gfd), Some(oid)) = (gf_data, obj_id) {
                    engine.geometry_first_data.insert(oid, gfd);
                }
                // Wire color bricks into the companion pool.
                if !color_bricks.is_empty() {
                    // Grow companion map to cover all brick pool slots.
                    let pool_cap = engine.cpu_brick_pool.capacity() as usize;
                    if engine.color_companion_map.len() < pool_cap {
                        engine.color_companion_map.resize(pool_cap, rkf_core::brick_map::EMPTY_SLOT);
                    }
                    for (brick_slot, color_brick) in color_bricks {
                        let color_idx = engine.cpu_color_bricks.len() as u32;
                        engine.cpu_color_bricks.push(color_brick);
                        engine.color_companion_map[brick_slot as usize] = color_idx;
                    }
                }
            }
            Err(e) => log::error!("Failed to load .rkf '{}': {e}", asset_path),
        }
    }
}

/// Restore camera, environment, and lights from a v3 scene file's properties.
fn restore_properties_v3(es: &mut EditorState, scene_v3: &SceneFileV3) {
    if let Some(s) = scene_v3.properties.get("camera") {
        if let Ok(cam) = ron::from_str::<SavedCameraSnapshot>(s) {
            // Write to entity (source of truth).
            if let Some(uuid) = es.editor_camera_entity {
                let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, cam.position);
                let _ = es.world.set_position(uuid, wp);
                if let Some(e) = es.world.ecs_entity_for(uuid) {
                    if let Ok(mut cc) = es.world.ecs_mut()
                        .get::<&mut rkf_runtime::components::CameraComponent>(e)
                    {
                        cc.yaw = cam.yaw.to_degrees();
                        cc.pitch = cam.pitch.to_degrees();
                        cc.fov_degrees = cam.fov_y.to_degrees();
                    }
                }
            }
        }
    }
    // Environment is restored from the EnvironmentSettings component
    // on the editor camera entity (handled in load_scene_v3).
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
    es: std::sync::MutexGuard<'_, EditorState>,
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
        .set_title("Create New Project")
        .set_file_name("MyProject.rkproject")
        .add_filter("RKIField Project", &["rkproject"])
        .save_file();

    if let Some(chosen_path) = dialog_result {
        // Extract project name from the chosen filename (strip .rkproject extension).
        let project_name = chosen_path
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "NewProject".to_string());
        let parent_dir = chosen_path
            .parent()
            .unwrap_or(std::path::Path::new("."));

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
                            let saved = if es.editor_camera_entity.is_some() {
                                Some(es.extract_camera_snapshot())
                            } else { None };
                            ctx.engine.clear_scene();
                            es.world.clear();
                            es.respawn_editor_camera(saved);
                            es.current_scene_path = None;
                        }

                        // Load material palette from the new project.
                        load_material_palette(&pf, &project_root, ctx.material_library);

                        // Set engine project root, scan for user shaders, and reinit file watcher.
                        ctx.engine.project_root = Some(project_root.clone());
                        ctx.engine.scan_user_shaders();
                        ctx.engine.reinit_file_watcher();

                        es.current_project = Some(pf);
                        es.current_project_path = Some(project_path_str.clone());
                        es.selected_entity = None;
                        es.undo.clear();
                        es.unsaved_changes.mark_saved();
                        *frame_topology_changed = true;
                        crate::editor_config::add_recent_project(&project_path_str, &project_name);
                        log::info!("New project created at {project_path_str}");
                    }
                    Err(e) => log::error!("Failed to load new project file: {e}"),
                }
            }
            Err(e) => log::error!("Failed to create project: {e}"),
        }
    }
    // NOTE: project_loaded UI push happens in the engine loop AFTER this
    // function returns and all mutex guards are dropped.
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
                    let saved = if es.editor_camera_entity.is_some() {
                        Some(es.extract_camera_snapshot())
                    } else { None };
                    ctx.engine.clear_scene();
                    es.world.clear();
                    es.respawn_editor_camera(saved);
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
                // Load material palette from project or engine library.
                load_material_palette(&pf, &project_root, ctx.material_library);

                // Set engine project root, scan for user shaders, and reinit file watcher.
                ctx.engine.project_root = Some(project_root.clone());
                ctx.engine.scan_user_shaders();
                ctx.engine.reinit_file_watcher();

                es.current_project = Some(pf);
                es.current_project_path = Some(path_str.clone());
                es.selected_entity = None;
                es.undo.clear();
                es.unsaved_changes.mark_saved();
                *frame_topology_changed = true;
                let project_name = project_root
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Project")
                    .to_string();
                crate::editor_config::add_recent_project(&path_str, &project_name);
                log::info!("Project opened from {path_str}");
            }
            Err(e) => log::error!("Failed to load project '{}': {e}", path_str),
        }
    }
    // NOTE: project_loaded UI push happens in the engine loop AFTER this
    // function returns and all mutex guards are dropped. This avoids
    // potential deadlocks from signal effects running while locks are held.
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

// ─── Material palette loading ────────────────────────────────────────────

/// Load a material palette for a project, falling back to the engine library.
///
/// Search order:
/// 1. `project.material_palette` relative to `project_root`
/// 2. Engine library `materials/default.rkmatlib`
fn load_material_palette(
    project: &rkf_runtime::project::ProjectFile,
    project_root: &std::path::Path,
    material_library: &std::sync::Arc<std::sync::Mutex<rkf_core::material_library::MaterialLibrary>>,
) {
    use rkf_core::material_library::MaterialLibrary;

    let palette_path = if let Some(ref rel) = project.material_palette {
        let project_path = project_root.join(rel);
        if project_path.exists() {
            project_path
        } else {
            rkf_runtime::project::engine_library_dir().join("materials/default.rkmatlib")
        }
    } else {
        rkf_runtime::project::engine_library_dir().join("materials/default.rkmatlib")
    };

    match MaterialLibrary::load_palette(&palette_path) {
        Ok(new_lib) => {
            let mut lib = material_library.lock().unwrap();
            *lib = new_lib;
            lib.clear_dirty();
            log::info!("Material palette loaded from {}", palette_path.display());
        }
        Err(e) => {
            log::warn!("Failed to load material palette from {}: {e}", palette_path.display());
        }
    }
}

// ─── Project-loaded UI push ──────────────────────────────────────────────

/// Push project-loaded state to the UI thread.
///
/// **Must be called after all mutex guards are dropped** to avoid deadlocks
/// from reactive signal effects that might try to lock `EditorState`.
pub(crate) fn push_project_loaded_to_ui() {
    let config = crate::editor_config::load_editor_config();
    let recents = config.recent_projects;
    rinch::shell::rinch_runtime::run_on_main_thread(move || {
        if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
            ui.project_loaded.set(true);
            ui.recent_projects.set(recents);
        }
    });
}

// ─── Game dylib loading ─────────────────────────────────────────────────

/// Non-blocking game plugin setup: scaffold the crate, try to load an
/// existing dylib, and prepare a [`BuildWatcher`] for the background build.
///
/// Returns `(dylib_loader, build_watcher)`:
/// - `dylib_loader` — `Some` if an existing (possibly stale) dylib was loaded,
///   `None` if no dylib exists yet.
/// - `build_watcher` — `Some` with a triggered background build, or `None` if
///   scaffolding failed or no scripts directory exists.
///
/// The caller should install the returned `BuildWatcher` on the engine so it
/// gets polled each frame via `process_file_events()`. When the build
/// completes, the hot-reload path loads the (re)built dylib.
pub(crate) fn build_and_load_game_dylib(
    project_root: &std::path::Path,
    registry: &Arc<Mutex<GameplayRegistry>>,
    old_loader: Option<rkf_runtime::behavior::DylibLoader>,
    console: Option<&rkf_runtime::behavior::ConsoleBuffer>,
) -> (
    Option<rkf_runtime::behavior::DylibLoader>,
    Option<rkf_runtime::behavior::BuildWatcher>,
) {
    use rkf_runtime::behavior::scaffold;

    let scripts_dir = scaffold::scripts_dir(project_root);
    if !scripts_dir.is_dir() {
        log::info!("No assets/scripts/ directory, skipping game plugin build");
        return (old_loader, None);
    }

    let game_crate_dir = scaffold::game_crate_dir(project_root);

    // Always generate/update the crate wrapper first (uses write-if-changed
    // internally, so it's cheap and preserves file mtimes when nothing changed).
    let engine_root = rkf_runtime::project::engine_library_dir()
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    match scaffold::generate_game_crate(project_root, &engine_root) {
        Ok(crate_dir) => {
            log::info!("Generated/updated game crate at {}", crate_dir.display());
        }
        Err(e) => {
            log::error!("Failed to generate game crate: {e}");
            return (old_loader, None);
        }
    }

    // Check if an existing dylib can be loaded immediately.
    let watcher_probe = rkf_runtime::behavior::BuildWatcher::new(
        game_crate_dir.clone(),
        scripts_dir.clone(),
    );
    let dylib_path = watcher_probe.expected_dylib_path();
    let dylib_exists = dylib_path.exists();

    log::info!(
        "Game dylib check: path={}, exists={}",
        dylib_path.display(), dylib_exists,
    );

    // Load existing dylib immediately for fast startup. Also trigger a
    // background rebuild — if the dylib is up-to-date, cargo finishes
    // instantly; if stale, the hot-reload path swaps in the new one.
    let loader = if dylib_exists {
        log::info!("Loading existing dylib (background build will update if stale)");
        load_game_dylib(&dylib_path, registry, old_loader)
    } else {
        log::info!("No existing dylib — will load after background build completes");
        drop(old_loader);
        None
    };

    let mut bw = rkf_runtime::behavior::BuildWatcher::new(
        game_crate_dir,
        scripts_dir,
    );
    bw.trigger_build(console.cloned());
    log::info!("Background game plugin build triggered");

    if let Some(c) = console {
        c.info("Compiling scripts...");
    }

    (loader, Some(bw))
}

/// Load a game dylib and register its components/systems (public for hot-reload).
pub(crate) fn load_game_dylib_public(
    dylib_path: &std::path::Path,
    registry: &Arc<Mutex<GameplayRegistry>>,
    old_loader: Option<rkf_runtime::behavior::DylibLoader>,
) -> Option<rkf_runtime::behavior::DylibLoader> {
    load_game_dylib(dylib_path, registry, old_loader)
}

/// Load a game dylib and register its components/systems.
fn load_game_dylib(
    dylib_path: &std::path::Path,
    registry: &Arc<Mutex<GameplayRegistry>>,
    old_loader: Option<rkf_runtime::behavior::DylibLoader>,
) -> Option<rkf_runtime::behavior::DylibLoader> {
    let mut reg = registry.lock().expect("registry lock");

    // Clear old gameplay components (keep engine ones).
    let engine_names = rkf_runtime::behavior::engine_components::ENGINE_COMPONENT_NAMES;
    reg.clear_gameplay(engine_names);

    // Drop old loader before loading new one.
    drop(old_loader);

    match rkf_runtime::behavior::DylibLoader::load(dylib_path) {
        Ok(loader) => {
            match loader.call_register(&mut reg) {
                Ok(()) => {
                    log::info!(
                        "Game crate loaded: {} components, {} systems",
                        reg.component_count(),
                        reg.system_list().len(),
                    );
                    Some(loader)
                }
                Err(e) => {
                    log::error!("Game crate register failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            log::error!("Failed to load game dylib: {e}");
            None
        }
    }
}

// ─── Diagnostics → UI ──────────────────────────────────────────────────

/// Parse build stderr into structured diagnostics and push them to the UI.
///
/// Called from the engine thread — uses `run_on_main_thread` to update the
/// reactive signal. An empty `stderr` clears existing diagnostics (build OK).
fn push_diagnostics_to_ui(stderr: &str, console: Option<&rkf_runtime::behavior::ConsoleBuffer>) {
    use crate::ui_snapshot::{DiagnosticEntry, DiagnosticSeverity};

    let entries: Vec<DiagnosticEntry> = if stderr.is_empty() {
        // Build succeeded — clear diagnostics.
        Vec::new()
    } else {
        let parsed = rkf_runtime::behavior::parse_cargo_errors(stderr);
        if parsed.is_empty() {
            // Unparseable stderr — show the raw output as a single error.
            vec![DiagnosticEntry {
                severity: DiagnosticSeverity::Error,
                message: stderr.lines().take(20).collect::<Vec<_>>().join("\n"),
                file: None,
                line: None,
                column: None,
            }]
        } else {
            parsed
                .into_iter()
                .map(|ce| DiagnosticEntry {
                    severity: DiagnosticSeverity::Error,
                    message: ce.message,
                    file: ce.file,
                    line: ce.line,
                    column: ce.column,
                })
                .collect()
        }
    };

    // Also push compile errors to the console buffer.
    if let Some(console) = console {
        for entry in &entries {
            let mut msg = entry.message.clone();
            if let Some(ref f) = entry.file {
                if let Some(line) = entry.line {
                    msg = format!("{f}:{line}: {msg}");
                }
            }
            match entry.severity {
                DiagnosticSeverity::Error => console.error(msg),
                DiagnosticSeverity::Warning => console.warn(msg),
                DiagnosticSeverity::Info => console.info(msg),
            }
        }
    }

    rinch::shell::rinch_runtime::run_on_main_thread(move || {
        if let Some(ui) =
            rinch::core::context::try_use_context::<crate::editor_state::UiSignals>()
        {
            ui.diagnostics.set(entries);
        }
    });
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
