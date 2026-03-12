//! Scene and project I/O helpers for the engine loop.
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use std::sync::{Arc, Mutex};

use crate::editor_state::EditorState;
use crate::engine::EditorEngine;
use crate::layout::state::LayoutBacking;

/// Shared context for scene/project loading — holds references to engine
/// and editor state resources used across multiple I/O helpers.
pub(crate) struct IoContext<'a> {
    pub(crate) engine: &'a mut EditorEngine,
    pub(crate) editor_state: &'a Arc<Mutex<EditorState>>,
    pub(crate) layout_backing: &'a LayoutBacking,
}

/// Restore camera, environment, and lights from a scene file's property bag.
pub(crate) fn restore_scene_properties(
    es: &mut EditorState,
    sf: &rkf_runtime::scene_file::SceneFile,
) {
    if let Some(s) = sf.properties.get("camera") {
        if let Ok(cam) = ron::from_str::<crate::scene_io::CameraSnapshot>(s) {
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
    if let Some(s) = sf.properties.get("environment") {
        if let Ok(env) = ron::from_str::<crate::environment::EnvironmentState>(s) {
            es.environment = env;
            es.environment.mark_dirty();
        }
    }
    if let Some(s) = sf.properties.get("lights") {
        if let Ok(lights) = ron::from_str::<Vec<crate::light_editor::SceneLight>>(s) {
            es.light_editor.replace_lights(lights);
        }
    }
}

/// Load voxelized objects from .rkf files referenced by a scene file,
/// populating the engine's brick pool and geometry data.
pub(crate) fn load_scene_rkf_assets(
    engine: &mut EditorEngine,
    sf: &rkf_runtime::scene_file::SceneFile,
    new_scene: &mut rkf_core::scene::Scene,
    scene_dir: &std::path::Path,
) {
    for (i, entry) in sf.objects.iter().enumerate() {
        if let Some(asset_path) = &entry.asset_path {
            let rkf_path = scene_dir.join(asset_path);
            let rkf_str = rkf_path.to_string_lossy().to_string();
            match crate::engine::load_rkf_auto(
                &rkf_str,
                &mut engine.cpu_brick_pool,
                &mut engine.cpu_brick_map_alloc,
                &mut engine.cpu_geometry_pool,
                &mut engine.cpu_sdf_cache_pool,
            ) {
                Ok((handle, voxel_size, grid_aabb, _count, gf_data)) => {
                    if let Some(obj) = new_scene.objects.get_mut(i) {
                        obj.root_node.sdf_source =
                            rkf_core::SdfSource::Voxelized {
                                brick_map_handle: handle,
                                voxel_size,
                                aabb: grid_aabb,
                            };
                        obj.aabb = grid_aabb;
                        if let Some(gfd) = gf_data {
                            engine.geometry_first_data.insert(obj.id, gfd);
                        }
                    }
                }
                Err(e) => {
                    log::error!("Failed to load .rkf '{}': {e}", asset_path);
                }
            }
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

/// Handle the "Scene Save" action. Returns `true` if the frame should continue
/// normally; returns `false` if we fell through to Save As.
pub(crate) fn handle_scene_save(
    es: &mut EditorState,
    engine: &EditorEngine,
) {
    let save_path = es.pending_save_path.take().or_else(|| {
        es.current_scene_path.clone()
    });
    if let Some(path) = save_path {
        let scene = es.world.scene().clone();
        let cam_snap = crate::scene_io::CameraSnapshot {
            position: es.editor_camera.position,
            yaw: es.editor_camera.fly_yaw,
            pitch: es.editor_camera.fly_pitch,
            fov_y: es.editor_camera.fov_y,
        };
        let props = crate::scene_io::SceneProperties {
            camera: Some(&cam_snap),
            environment: Some(&es.environment),
            lights: Some(es.light_editor.all_lights()),
        };
        match crate::scene_io::save_v2_scene_full(
            &scene, &path,
            &engine.cpu_brick_pool, &engine.cpu_brick_map_alloc,
            &props,
            Some(&engine.cpu_geometry_pool),
            Some(&engine.cpu_sdf_cache_pool),
            Some(&engine.geometry_first_data),
        ) {
            Ok(()) => {
                es.current_scene_path = Some(path.clone());
                es.unsaved_changes.mark_saved();
                log::info!("Scene saved to {path}");
            }
            Err(e) => log::error!("Failed to save scene: {e}"),
        }
    } else {
        // No path known — fall through to Save As.
        es.pending_save_as = true;
    }
}

/// Handle the "Scene Save As" action. Drops the EditorState lock before
/// showing the file dialog. Returns `true` to signal the caller should
/// `continue` (skip the rest of the frame).
pub(crate) fn handle_scene_save_as(
    es: std::sync::MutexGuard<'_, EditorState>,
    editor_state: &Arc<Mutex<EditorState>>,
    engine: &EditorEngine,
) {
    let scene = es.world.scene().clone();
    let cam_snap = crate::scene_io::CameraSnapshot {
        position: es.editor_camera.position,
        yaw: es.editor_camera.fly_yaw,
        pitch: es.editor_camera.fly_pitch,
        fov_y: es.editor_camera.fov_y,
    };
    let env_clone = es.environment.clone();
    let lights_clone: Vec<_> = es.light_editor.all_lights().to_vec();
    drop(es);

    let dialog_result = rfd::FileDialog::new()
        .add_filter("Scene", &["rkscene"])
        .set_file_name("scene.rkscene")
        .save_file();

    if let Some(file_path) = dialog_result {
        let path = file_path.to_string_lossy().to_string();
        let props = crate::scene_io::SceneProperties {
            camera: Some(&cam_snap),
            environment: Some(&env_clone),
            lights: Some(&lights_clone),
        };
        match crate::scene_io::save_v2_scene_full(
            &scene, &path,
            &engine.cpu_brick_pool, &engine.cpu_brick_map_alloc,
            &props,
            Some(&engine.cpu_geometry_pool),
            Some(&engine.cpu_sdf_cache_pool),
            Some(&engine.geometry_first_data),
        ) {
            Ok(()) => {
                let mut es = editor_state.lock().unwrap();
                es.current_scene_path = Some(path.clone());
                es.unsaved_changes.mark_saved();
                log::info!("Scene saved to {path}");
            }
            Err(e) => log::error!("Failed to save scene: {e}"),
        }
    }
}

/// Handle the "Scene Open" action -- full implementation.
///
/// `open_path`: pre-extracted path (if any) from `es.pending_open_path`.
/// The `es` lock is consumed (dropped if we need a file dialog).
pub(crate) fn handle_scene_open_impl(
    mut es: std::sync::MutexGuard<'_, EditorState>,
    open_path: Option<String>,
    ctx: &mut IoContext<'_>,
) {
    let file_path = if let Some(p) = open_path {
        Some(std::path::PathBuf::from(p))
    } else {
        // Drop the lock before blocking on file dialog.
        drop(es);
        let result = rfd::FileDialog::new()
            .add_filter("Scene", &["rkscene"])
            .pick_file();
        if result.is_none() {
            return; // User cancelled.
        }
        // Re-acquire lock handled below via re-lock.
        result
    };

    if let Some(file_path) = file_path {
        let path_str = file_path.to_string_lossy().to_string();
        match crate::scene_io::load_v2_scene(&path_str) {
            Ok(sf) => {
                let mut new_scene = crate::scene_io::reconstruct_v2_scene(&sf);
                ctx.engine.clear_scene();
                let scene_dir = file_path.parent()
                    .unwrap_or(std::path::Path::new("."));
                load_scene_rkf_assets(ctx.engine, &sf, &mut new_scene, scene_dir);
                ctx.engine.reupload_brick_data();

                // Replace the scene in EditorState.
                // We may need to re-acquire the lock if we dropped it for the dialog.
                let mut es = ctx.editor_state.lock().unwrap();
                *es.world.scene_mut() = new_scene;
                es.current_scene_path = Some(path_str.clone());
                es.unsaved_changes.mark_saved();
                es.selected_entity = None;
                es.undo.clear();
                restore_scene_properties(&mut es, &sf);
                log::info!("Scene loaded from {path_str}");
            }
            Err(e) => {
                log::error!("Failed to load scene '{}': {e}", path_str);
            }
        }
    }
}

/// Handle the "New Project" action. Drops the EditorState lock before
/// showing the folder dialog.
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

                        ctx.engine.clear_scene();
                        let mut es = ctx.editor_state.lock().unwrap();
                        if let Some(scene_path) = default_scene_path {
                            let sp = scene_path.to_string_lossy().to_string();
                            match crate::scene_io::load_v2_scene(&sp) {
                                Ok(sf) => {
                                    let new_scene = crate::scene_io::reconstruct_v2_scene(&sf);
                                    *es.world.scene_mut() = new_scene;
                                    es.current_scene_path = Some(sp);

                                    // Restore camera from scene properties.
                                    if let Some(s) = sf.properties.get("camera") {
                                        if let Ok(cam) = ron::from_str::<crate::scene_io::CameraSnapshot>(s) {
                                            es.editor_camera.position = cam.position;
                                            es.editor_camera.fly_yaw = cam.yaw;
                                            es.editor_camera.fly_pitch = cam.pitch;
                                            es.editor_camera.fov_y = cam.fov_y;
                                        }
                                    }
                                }
                                Err(e) => log::error!("Failed to load default scene: {e}"),
                            }
                        } else {
                            *es.world.scene_mut() = rkf_core::scene::Scene::new("default");
                            es.current_scene_path = None;
                        }
                        es.current_project = Some(pf);
                        es.current_project_path = Some(project_path_str.clone());
                        es.selected_entity = None;
                        es.undo.clear();
                        es.unsaved_changes.mark_saved();
                        es.world.resync_entity_tracking();
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

/// Handle the "Open Project" action. Drops the EditorState lock before
/// showing the file dialog if no path was provided.
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

                ctx.engine.clear_scene();
                let mut es = ctx.editor_state.lock().unwrap();

                if let Some(scene_path) = default_scene_path {
                    let sp = scene_path.to_string_lossy().to_string();
                    match crate::scene_io::load_v2_scene(&sp) {
                        Ok(sf) => {
                            let mut new_scene = crate::scene_io::reconstruct_v2_scene(&sf);
                            let scene_dir = scene_path.parent()
                                .unwrap_or(std::path::Path::new("."));
                            load_scene_rkf_assets(ctx.engine, &sf, &mut new_scene, scene_dir);
                            ctx.engine.reupload_brick_data();

                            *es.world.scene_mut() = new_scene;
                            es.current_scene_path = Some(sp);
                            restore_scene_properties(&mut es, &sf);
                        }
                        Err(e) => log::error!("Failed to load scene: {e}"),
                    }
                } else {
                    *es.world.scene_mut() = rkf_core::scene::Scene::new("default");
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
                es.world.resync_entity_tracking();
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
) {
    if es.current_project.is_some() {
        // Save scene.
        let save_path = es.current_scene_path.clone();
        if let Some(path) = save_path {
            let scene = es.world.scene().clone();
            let cam_snap = crate::scene_io::CameraSnapshot {
                position: es.editor_camera.position,
                yaw: es.editor_camera.fly_yaw,
                pitch: es.editor_camera.fly_pitch,
                fov_y: es.editor_camera.fov_y,
            };
            let props = crate::scene_io::SceneProperties {
                camera: Some(&cam_snap),
                environment: Some(&es.environment),
                lights: Some(es.light_editor.all_lights()),
            };
            match crate::scene_io::save_v2_scene_full(
                &scene, &path,
                &engine.cpu_brick_pool, &engine.cpu_brick_map_alloc,
                &props,
                Some(&engine.cpu_geometry_pool),
                Some(&engine.cpu_sdf_cache_pool),
                Some(&engine.geometry_first_data),
            ) {
                Ok(()) => {
                    es.unsaved_changes.mark_saved();
                    log::info!("Scene saved to {path}");
                }
                Err(e) => log::error!("Failed to save scene: {e}"),
            }
        } else {
            // No scene path — trigger Save As.
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
        // No project loaded — just save the scene.
        es.pending_save = true;
    }
}
