//! UI signal push and GPU pick/brush hit processing for the engine loop.
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use std::sync::{Arc, Mutex};

use crate::automation::SharedState;
use crate::editor_state::{EditorState, UiSignals};
use crate::engine::EditorEngine;

use crate::editor_state::SelectedEntity;
use crate::engine_loop::DirtyFlags;
use crate::engine_loop_commands::compute_material_usage;

/// l: Process GPU pick result -- sets selected_entity, pushes to UI signals.
///    Suppressed while actively sculpting to prevent accidental selection changes.
pub(crate) fn process_gpu_pick(
    shared_state: &Arc<Mutex<SharedState>>,
    editor_state: &Arc<Mutex<EditorState>>,
    engine: &EditorEngine,
    sculpting_active: bool,
) {
    let pick = shared_state.lock().ok()
        .and_then(|mut ss| ss.pick_result.take());
    if let Some(object_id) = pick {
        if !sculpting_active {
            let picked_entity = if object_id > 0 {
                Some(crate::editor_state::SelectedEntity::Object(object_id as u64))
            } else {
                None
            };
            let mat_usage = if let Ok(mut es) = editor_state.lock() {
                es.selected_entity = picked_entity;
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = picked_entity {
                    compute_material_usage(&es, engine, eid)
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };
            // Push selection + material usage to UI signals directly.
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<crate::editor_state::UiSignals>() {
                    // Set material usage BEFORE selection -- selection change
                    // triggers ObjectProperties rebuild which reads this signal.
                    ui.selected_object_materials.set(mat_usage);
                    let sliders = rinch::core::context::try_use_context::<crate::editor_state::SliderSignals>();
                    let tree_state = rinch::core::context::try_use_context::<rinch::prelude::UseTreeReturn>();
                    if let (Some(sliders), Some(tree_state)) = (sliders, tree_state) {
                        ui.set_selection(picked_entity, &sliders, &tree_state);
                        if picked_entity.is_some() {
                            ui.properties_tab.set(0);
                        }
                    }
                }
            });
        }
    }
}

/// m: Process GPU brush hit result -- drive sculpt/paint stroke lifecycle.
pub(crate) fn process_brush_hit(
    shared_state: &Arc<Mutex<SharedState>>,
    editor_state: &Arc<Mutex<EditorState>>,
) {
    let brush_hit = shared_state.lock().ok()
        .and_then(|mut ss| ss.brush_hit_result.take());
    if let Some(hit) = brush_hit {
        let (left_down, mode, selected_obj_id) = editor_state.lock().ok()
            .map(|es| {
                let sel_id = match es.selected_entity {
                    Some(crate::editor_state::SelectedEntity::Object(eid)) => Some(eid as u32),
                    _ => None,
                };
                (es.editor_input.viewport_left_down, es.mode, sel_id)
            })
            .unwrap_or((false, crate::editor_state::EditorMode::Default, None));

        // Only allow sculpt/paint on the currently selected object.
        let hit_on_selected = selected_obj_id == Some(hit.object_id);

        if left_down {
            if let Ok(mut es) = editor_state.lock() {
                match mode {
                    crate::editor_state::EditorMode::Sculpt if hit_on_selected => {
                        if es.sculpt.active_stroke.is_some() {
                            es.sculpt.continue_stroke(hit.position);
                        } else {
                            es.sculpt.begin_stroke(hit.position);
                        }
                        // Queue a real-time sculpt edit for this point.
                        let settings = es.sculpt.current_settings.clone();
                        es.pending_sculpt_edits.push(
                            crate::sculpt::SculptEditRequest {
                                object_id: hit.object_id,
                                world_position: hit.position,
                                settings,
                            },
                        );
                    }
                    crate::editor_state::EditorMode::Sculpt => {
                        // Hit a non-selected object -- ignore.
                    }
                    crate::editor_state::EditorMode::Paint if hit_on_selected => {
                        if es.paint.active_stroke.is_some() {
                            es.paint.continue_stroke(hit.position);
                        } else {
                            es.paint.begin_stroke(hit.position);
                        }
                        // Queue a real-time paint edit for this point.
                        let settings = es.paint.current_settings.clone();
                        es.pending_paint_edits.push(
                            crate::paint::PaintEditRequest {
                                object_id: hit.object_id,
                                world_position: hit.position,
                                settings,
                            },
                        );
                    }
                    crate::editor_state::EditorMode::Paint => {
                        // Hit a non-selected object -- ignore.
                    }
                    _ => {}
                }
            }
        }

        if let Ok(mut ss) = shared_state.lock() {
            ss.brush_preview_pos = Some(hit.position);
            ss.brush_preview_object_id = Some(hit.object_id);
        }
    } else {
        let (left_down, mode) = editor_state.lock().ok()
            .map(|es| (es.editor_input.viewport_left_down, es.mode))
            .unwrap_or((false, crate::editor_state::EditorMode::Default));

        if !left_down && matches!(mode, crate::editor_state::EditorMode::Sculpt | crate::editor_state::EditorMode::Paint) {
            if let Ok(mut es) = editor_state.lock() {
                match mode {
                    crate::editor_state::EditorMode::Sculpt => {
                        es.sculpt.end_stroke();
                        // Finalize sculpt undo: push accumulated geometry snapshots.
                        if let Some(acc) = es.sculpt_undo_accumulator.take() {
                            if !acc.snapshots.is_empty() {
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::SculptStroke {
                                        object_id: acc.object_id,
                                        geometry_snapshots: acc.snapshots,
                                    },
                                    timestamp_ms: 0,
                                    description: "Sculpt stroke".into(),
                                });
                            }
                        }
                    }
                    crate::editor_state::EditorMode::Paint => {
                        es.paint.end_stroke();
                        // Finalize paint undo: push accumulated geometry snapshots.
                        if let Some(acc) = es.paint_undo_accumulator.take() {
                            if !acc.snapshots.is_empty() {
                                es.undo.push(crate::undo::UndoAction {
                                    kind: crate::undo::UndoActionKind::PaintStroke {
                                        object_id: acc.object_id,
                                        geometry_snapshots: acc.snapshots,
                                    },
                                    timestamp_ms: 0,
                                    description: "Paint stroke".into(),
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// n0: Push dirty data into UI signals via run_on_main_thread.
///     Only push categories that actually changed this frame.
pub(crate) fn push_dirty_ui_signals(
    dirty: &DirtyFlags,
    editor_state: &Arc<Mutex<EditorState>>,
    engine: &EditorEngine,
) {
    if dirty.scene || dirty.lights || dirty.materials || dirty.shaders {
        if let Ok(es) = editor_state.lock() {
            // Scene objects.
            if dirty.scene {
                let objects = es.build_object_summaries();
                let scene_name = es.world.scene().name.clone();
                let scene_path = es.current_scene_path.clone();
                // Also compute material usage for selected object.
                let mat_usage = if let Some(crate::editor_state::SelectedEntity::Object(eid)) = es.selected_entity {
                    compute_material_usage(&es, engine, eid)
                } else {
                    Vec::new()
                };
                rinch::shell::rinch_runtime::run_on_main_thread(move || {
                    if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                        // Set material usage BEFORE objects -- objects.set() triggers
                        // ObjectProperties rebuild which reads selected_object_materials.
                        ui.selected_object_materials.set(mat_usage);
                        ui.objects.set(objects);
                        ui.scene_name.set(scene_name);
                        ui.scene_path.set(scene_path);
                    }
                });
            }
            // Lights.
            if dirty.lights {
                let lights = es.build_light_summaries();
                rinch::shell::rinch_runtime::run_on_main_thread(move || {
                    if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                        ui.lights.set(lights);
                    }
                });
            }
        }
    }

    // Materials (from engine's MaterialLibrary).
    if dirty.materials {
        if let Ok(lib) = engine.material_library.lock() {
            use crate::ui_snapshot::MaterialSummary;
            let mut materials: Vec<MaterialSummary> = lib.occupied_slots().map(|(slot, mat, info)| {
                MaterialSummary {
                    slot,
                    name: info.name.clone(),
                    category: info.category.clone(),
                    albedo: mat.albedo,
                    roughness: mat.roughness,
                    metallic: mat.metallic,
                    emission_strength: mat.emission_strength,
                    emission_color: mat.emission_color,
                    subsurface: mat.subsurface,
                    subsurface_color: mat.subsurface_color,
                    opacity: mat.opacity,
                    ior: mat.ior,
                    noise_scale: mat.noise_scale,
                    noise_strength: mat.noise_strength,
                    noise_channels: mat.noise_channels,
                    shader_name: info.shader_name.clone(),
                }
            }).collect();
            // Also include slot 0 (fallback) if not already present.
            for i in 0..lib.slot_count() {
                if lib.slot_info(i as u16).is_none() {
                    if i == 0 && materials.iter().all(|m| m.slot != 0) {
                        if let Some(mat) = lib.get_material(0) {
                            materials.push(MaterialSummary {
                                slot: 0,
                                name: "Material 0".to_string(),
                                category: "Other".to_string(),
                                albedo: mat.albedo,
                                roughness: mat.roughness,
                                metallic: mat.metallic,
                                emission_strength: mat.emission_strength,
                                emission_color: mat.emission_color,
                                subsurface: mat.subsurface,
                                subsurface_color: mat.subsurface_color,
                                opacity: mat.opacity,
                                ior: mat.ior,
                                noise_scale: mat.noise_scale,
                                noise_strength: mat.noise_strength,
                                noise_channels: mat.noise_channels,
                                shader_name: "pbr".to_string(),
                            });
                        }
                    }
                    break; // Only check slot 0
                }
            }
            materials.sort_by_key(|m| m.slot);
            rinch::shell::rinch_runtime::run_on_main_thread(move || {
                if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                    ui.materials.set(materials);
                }
            });
        }
    }

    // Shaders (from engine's ShaderComposer).
    if dirty.shaders {
        let shaders: Vec<crate::ui_snapshot::ShaderSummary> = engine.shader_composer.shader_summaries()
            .into_iter()
            .map(|s| crate::ui_snapshot::ShaderSummary {
                name: s.name,
                id: s.id,
                built_in: s.built_in,
                file_path: s.file_path,
            })
            .collect();
        rinch::shell::rinch_runtime::run_on_main_thread(move || {
            if let Some(ui) = rinch::core::context::try_use_context::<UiSignals>() {
                ui.shaders.set(shaders);
            }
        });
    }
}

/// g: Build wireframe overlays (selection, gizmos, grid, brush).
///
/// Returns the assembled wireframe vertex list. Also drives the brush
/// overlay on the engine (activate/deactivate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_wireframe_overlays(
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    shared_state: &Arc<Mutex<SharedState>>,
    scene_clone: &rkf_core::scene::Scene,
    camera_pos: glam::Vec3,
    selected: Option<SelectedEntity>,
    gizmo_mode: crate::gizmo::GizmoMode,
    gizmo_axis: crate::gizmo::GizmoAxis,
    show_grid: bool,
    editor_mode: crate::editor_state::EditorMode,
    brush_radius: f32,
    brush_falloff: f32,
    sculpting_active: bool,
) {
    let mut wf_verts: Vec<crate::wireframe::LineVertex> = Vec::new();

    // Light wireframe + translate gizmo for selected light.
    if let Some(SelectedEntity::Light(lid)) = selected {
        if let Ok(es) = editor_state.lock() {
            if let Some(light) = es.light_editor.get_light(lid) {
                let lc = [1.0, 0.9, 0.5, 1.0];
                match light.light_type {
                    crate::light_editor::SceneLightType::Point => {
                        wf_verts.extend(crate::wireframe::point_light_wireframe(
                            light.position, light.range, lc,
                        ));
                    }
                    crate::light_editor::SceneLightType::Spot => {
                        wf_verts.extend(crate::wireframe::spot_light_wireframe(
                            light.position, light.direction,
                            light.range, light.spot_outer_angle, lc,
                        ));
                    }
                }
                let gc = light.position;
                let cam_dist = (gc - camera_pos).length();
                let gizmo_size = cam_dist * 0.12;
                wf_verts.extend(crate::wireframe::translate_gizmo_wireframe(
                    gc, gizmo_size, gizmo_axis, camera_pos,
                ));
            }
        }
    }

    // Selection AABB + transform gizmo for selected objects.
    if let Some(SelectedEntity::Object(eid)) = selected.filter(|_| !sculpting_active) {
        let color = [0.3, 0.7, 1.0, 1.0];
        let mut gizmo_center: Option<glam::Vec3> = None;

        for obj in &scene_clone.objects {
            let obj_id = obj.id as u64;
            let root_world = glam::Mat4::from_scale_rotation_translation(
                obj.scale,
                obj.rotation,
                obj.position,
            );

            if eid == obj_id {
                if let Some((lmin, lmax)) =
                    crate::wireframe::compute_node_tree_aabb(&obj.root_node, glam::Mat4::IDENTITY)
                {
                    wf_verts.extend(crate::wireframe::obb_wireframe(
                        lmin, lmax, obj.position, obj.rotation, obj.scale, color,
                    ));
                    let center = obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale);
                    gizmo_center = Some(center);
                }
            } else if let Some((child_node, child_world)) =
                crate::wireframe::find_child_node_and_transform(
                    eid, obj_id, &obj.root_node, root_world,
                )
            {
                if let Some((lmin, lmax)) =
                    crate::wireframe::compute_node_tree_aabb(child_node, glam::Mat4::IDENTITY)
                {
                    let (child_scale, child_rot, child_pos) =
                        child_world.to_scale_rotation_translation();
                    wf_verts.extend(crate::wireframe::obb_wireframe(
                        lmin, lmax, child_pos, child_rot, child_scale, color,
                    ));
                    let center = child_pos + child_rot * ((lmin + lmax) * 0.5 * child_scale);
                    gizmo_center = Some(center);
                }
            }
        }

        if let Some(gc) = gizmo_center {
            let cam_dist = (gc - camera_pos).length();
            let gizmo_size = cam_dist * 0.12;
            match gizmo_mode {
                crate::gizmo::GizmoMode::Translate => {
                    wf_verts.extend(crate::wireframe::translate_gizmo_wireframe(
                        gc, gizmo_size, gizmo_axis, camera_pos,
                    ));
                }
                crate::gizmo::GizmoMode::Rotate => {
                    wf_verts.extend(crate::wireframe::rotate_gizmo_wireframe(
                        gc, gizmo_size, gizmo_axis, camera_pos,
                    ));
                }
                crate::gizmo::GizmoMode::Scale => {
                    wf_verts.extend(crate::wireframe::scale_gizmo_wireframe(
                        gc, gizmo_size, gizmo_axis, camera_pos,
                    ));
                }
            }
        }
    }

    // Ground grid overlay.
    if show_grid {
        wf_verts.extend(crate::wireframe::ground_grid_wireframe(
            camera_pos,
            40.0,
            1.0,
            [0.3, 0.3, 0.3, 0.4],
        ));
    }

    // Brush overlay.
    if matches!(editor_mode, crate::editor_state::EditorMode::Sculpt | crate::editor_state::EditorMode::Paint) {
        let (brush_pos, brush_obj_id) = shared_state.lock().ok()
            .map(|s| (s.brush_preview_pos, s.brush_preview_object_id))
            .unwrap_or((None, None));
        if let (Some(pos), Some(obj_id)) = (brush_pos, brush_obj_id) {
            let brush_color = match editor_mode {
                crate::editor_state::EditorMode::Sculpt => [0.0, 1.0, 1.0, 0.8],
                crate::editor_state::EditorMode::Paint  => [1.0, 0.8, 0.0, 0.8],
                _ => [1.0, 1.0, 1.0, 0.8],
            };
            engine.update_brush_overlay_from_flood(
                scene_clone, obj_id, pos, brush_radius, brush_falloff, brush_color,
            );
        } else {
            engine.deactivate_brush_overlay();
        }
    } else {
        engine.deactivate_brush_overlay();
    }

    engine.set_wireframe_vertices(wf_verts);
}
