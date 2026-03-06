//! Editor UI root component.
//!
//! Layout: titlebar → (left panel | viewport | right panel) → status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod asset_browser;
pub mod components;
pub mod materials_panel;
pub mod properties_panel;
pub mod scene_tree_panel;
pub mod shaders_panel;
mod slider_helpers;
pub mod titlebar;
pub mod right_panel;
pub mod status_bar;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiSignals};
use crate::gizmo;
use crate::input::{InputState, KeyCode, Modifiers};
use status_bar::StatusBar;
use titlebar::TitleBar;
use crate::wireframe;
use crate::layout::components::layout_root::LayoutRoot;
use crate::layout::components::floating_host::FloatingPanelHost;

// ── Style constants ─────────────────────────────────────────────────────────
// All colors use rinch theme CSS variables for the dark theme.
// Visual hierarchy: root=dark-9(#141414), panels=dark-8(#1f1f1f),
// titlebar=surface(#2e2e2e), text=#C9C9C9, dimmed=#828282, border=#424242.

pub(super) const PANEL_BG: &str = "background:var(--rinch-color-dark-8);";
pub(super) const PANEL_BORDER: &str = "border:1px solid var(--rinch-color-border);";
// Panel width constants removed — now driven by LayoutState signals.

pub(super) const LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

pub(super) const SECTION_STYLE: &str = "font-size:12px;color:var(--rinch-color-text);padding:6px 12px;";

pub(super) const VALUE_STYLE: &str = "font-size:12px;color:var(--rinch-color-dimmed);padding:2px 12px;\
    font-family:var(--rinch-font-family-monospace);";

pub(super) const DIVIDER_STYLE: &str = "height:1px;background:var(--rinch-color-border);margin:8px 0;";

// ── Root component ──────────────────────────────────────────────────────────

/// Root editor UI component.
///
/// Layout:
/// ```text
/// ┌──────────────────────────────────────────────────────────────────┐
/// │ RkiField   File Edit View  ·  Sculpt Paint        [─] [□] [×]  │  (titlebar 36px)
/// ├────────┬──────────────────────────────────────────┬──────────────┤
/// │  Left  │           RenderSurface                  │    Right     │
/// │ 250px  │            (flex: 1)                     │   300px      │
/// ├────────┴──────────────────────────────────────────┴──────────────┤
/// │  7 objects  61 fps  ┊                          Navigate mode     │  (status bar 25px)
/// └──────────────────────────────────────────────────────────────────┘
/// ```
#[component]
pub fn editor_ui() -> NodeHandle {
    // Dark theme overrides for rinch components that use light-mode colors.
    const DARK_OVERRIDES: &str = "\
        .rinch-tree__node-content:hover {\
            background-color: var(--rinch-color-dark-5) !important;\
            color: var(--rinch-color-text) !important;\
        }\
        .rinch-tree__node-content--selected {\
            background-color: var(--rinch-primary-color-9) !important;\
            color: var(--rinch-color-dark-0) !important;\
        }\
        .rinch-tree__node-content--selected:hover {\
            background-color: var(--rinch-primary-color-8) !important;\
        }\
        .rinch-tree__node-content--selected .rinch-tree__label,\
        .rinch-tree__node-content--selected .rinch-tree__icon,\
        .rinch-tree__node-content--selected .rinch-tree__chevron {\
            color: inherit !important;\
        }\
    ";

    // Retrieve contexts.
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();
    let surface_handle = use_context::<RenderSurfaceHandle>();

    // ── Global Effects ──────────────────────────────────────────────────────
    // These Effects live here (top-level, outside any reactive_component_dom)
    // so their Effect::new initial run doesn't cause re-entrant RefCell borrows.

    // Selection-change Effect: push object/light values into SliderSignals.
    {
        let ui = use_context::<UiSignals>();
        let sliders = use_context::<SliderSignals>();
        let snapshot = use_context::<crate::SnapshotReader>();
        Effect::new(move || {
            let sel = ui.selection.get();

            enum PushData {
                Object(glam::Vec3, glam::Vec3, glam::Vec3),
                Light(glam::Vec3, f32, f32),
                None,
            }
            let snap_guard = snapshot.0.load();
            let (push, oid, lid) = match sel {
                Some(SelectedEntity::Object(oid)) => {
                    let data = snap_guard
                        .objects
                        .iter()
                        .find(|o| o.id == oid)
                        .map(|o| PushData::Object(o.position, o.rotation_degrees, o.scale))
                        .unwrap_or(PushData::None);
                    (data, Some(oid), None)
                }
                Some(SelectedEntity::Light(lid)) => {
                    let data = snap_guard
                        .lights
                        .iter()
                        .find(|l| l.id == lid)
                        .map(|l| PushData::Light(l.position, l.intensity, l.range))
                        .unwrap_or(PushData::None);
                    (data, None, Some(lid))
                }
                _ => (PushData::None, None, None),
            };

            rinch::core::untracked(|| {
                sliders.bound_object_id.set(oid);
                sliders.bound_light_id.set(lid);
            });
            match push {
                PushData::Object(pos, rot_deg, scale) => {
                    rinch::core::untracked(|| {
                        sliders.push_object_values(pos, rot_deg, scale);
                    });
                }
                PushData::Light(pos, intensity, range) => {
                    rinch::core::untracked(|| {
                        sliders.push_light_values(pos, intensity, range);
                    });
                }
                PushData::None => {}
            }
        });
    }

    // Batch sync Effect: slider signals + toggle signals → engine commands.
    {
        let ui = use_context::<UiSignals>();
        let sliders = use_context::<SliderSignals>();
        let cmd = use_context::<crate::CommandSender>();
        Effect::new(move || {
            sliders.track_all();
            let _ = ui.atmo_enabled.get();
            let _ = ui.fog_enabled.get();
            let _ = ui.clouds_enabled.get();
            let _ = ui.bloom_enabled.get();
            let _ = ui.dof_enabled.get();
            let _ = ui.tone_map_mode.get();

            // Camera.
            let _ = cmd.0.send(EditorCommand::SetCameraFov {
                fov: sliders.fov.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetCameraSpeed {
                speed: sliders.fly_speed.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetCameraNearFar {
                near: sliders.near.get() as f32,
                far: sliders.far.get() as f32,
            });

            // Atmosphere.
            let az = (sliders.sun_azimuth.get() as f32).to_radians();
            let el = (sliders.sun_elevation.get() as f32).to_radians();
            let cos_el = el.cos();
            let sun_dir =
                glam::Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
            let _ = cmd.0.send(EditorCommand::SetAtmosphere {
                sun_direction: sun_dir,
                sun_intensity: sliders.sun_intensity.get() as f32,
                rayleigh_scale: sliders.rayleigh_scale.get() as f32,
                mie_scale: sliders.mie_scale.get() as f32,
            });

            // Fog.
            let _ = cmd.0.send(EditorCommand::SetFog {
                density: sliders.fog_density.get() as f32,
                height_falloff: sliders.fog_height_falloff.get() as f32,
                dust_density: sliders.dust_density.get() as f32,
                dust_asymmetry: sliders.dust_asymmetry.get() as f32,
            });

            // Clouds.
            let _ = cmd.0.send(EditorCommand::SetClouds {
                coverage: sliders.cloud_coverage.get() as f32,
                density: sliders.cloud_density.get() as f32,
                altitude: sliders.cloud_altitude.get() as f32,
                thickness: sliders.cloud_thickness.get() as f32,
                wind_speed: sliders.cloud_wind_speed.get() as f32,
            });

            // Post-process.
            let _ = cmd.0.send(EditorCommand::SetPostProcess {
                bloom_intensity: sliders.bloom_intensity.get() as f32,
                bloom_threshold: sliders.bloom_threshold.get() as f32,
                exposure: sliders.exposure.get() as f32,
                sharpen: sliders.sharpen.get() as f32,
                dof_focus_distance: sliders.dof_focus_dist.get() as f32,
                dof_focus_range: sliders.dof_focus_range.get() as f32,
                dof_max_coc: sliders.dof_max_coc.get() as f32,
                motion_blur: sliders.motion_blur.get() as f32,
                god_rays: sliders.god_rays.get() as f32,
                vignette: sliders.vignette.get() as f32,
                grain: sliders.grain.get() as f32,
                chromatic_aberration: sliders.chromatic_ab.get() as f32,
            });

            // Brush.
            let _ = cmd.0.send(EditorCommand::SetSculptSettings {
                radius: sliders.brush_radius.get() as f32,
                strength: sliders.brush_strength.get() as f32,
                falloff: sliders.brush_falloff.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetPaintSettings {
                radius: sliders.brush_radius.get() as f32,
                strength: sliders.brush_strength.get() as f32,
                falloff: sliders.brush_falloff.get() as f32,
            });

            // Object transform.
            let obj_id = rinch::core::untracked(|| sliders.bound_object_id.get());
            if let Some(oid) = obj_id {
                let _ = cmd.0.send(EditorCommand::SetObjectPosition {
                    entity_id: oid,
                    position: glam::Vec3::new(
                        sliders.obj_pos_x.get() as f32,
                        sliders.obj_pos_y.get() as f32,
                        sliders.obj_pos_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetObjectRotation {
                    entity_id: oid,
                    rotation: glam::Vec3::new(
                        sliders.obj_rot_x.get() as f32,
                        sliders.obj_rot_y.get() as f32,
                        sliders.obj_rot_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetObjectScale {
                    entity_id: oid,
                    scale: glam::Vec3::new(
                        sliders.obj_scale_x.get() as f32,
                        sliders.obj_scale_y.get() as f32,
                        sliders.obj_scale_z.get() as f32,
                    ),
                });
            }

            // Light properties.
            let light_id = rinch::core::untracked(|| sliders.bound_light_id.get());
            if let Some(lid) = light_id {
                let _ = cmd.0.send(EditorCommand::SetLightPosition {
                    light_id: lid,
                    position: glam::Vec3::new(
                        sliders.light_pos_x.get() as f32,
                        sliders.light_pos_y.get() as f32,
                        sliders.light_pos_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetLightIntensity {
                    light_id: lid,
                    intensity: sliders.light_intensity.get() as f32,
                });
                let _ = cmd.0.send(EditorCommand::SetLightRange {
                    light_id: lid,
                    range: sliders.light_range.get() as f32,
                });
            }

            // Toggles.
            let _ = cmd.0.send(EditorCommand::ToggleAtmosphere {
                enabled: ui.atmo_enabled.get(),
            });
            let _ = cmd.0.send(EditorCommand::ToggleFog {
                enabled: ui.fog_enabled.get(),
            });
            let _ = cmd.0.send(EditorCommand::ToggleClouds {
                enabled: ui.clouds_enabled.get(),
            });
            let _ = cmd.0.send(EditorCommand::ToggleBloom {
                enabled: ui.bloom_enabled.get(),
            });
            let _ = cmd.0.send(EditorCommand::ToggleDof {
                enabled: ui.dof_enabled.get(),
            });
            let _ = cmd.0.send(EditorCommand::SetToneMapMode {
                mode: ui.tone_map_mode.get(),
            });
        });
    }

    // Tree selection-sync Effect: update tree highlight when ui.selection changes.
    // Lives here (not in SceneTreePanel) to avoid .set() during render.
    {
        let ui = use_context::<UiSignals>();
        // Create tree state as context so SceneTreePanel can use it.
        let tree_state = UseTreeReturn::new(UseTreeOptions {
            initial_expanded: ["project".to_string(), "scene".to_string()]
                .into_iter()
                .collect(),
            ..Default::default()
        });
        create_context(tree_state);

        Effect::new(move || {
            let sel = ui.selection.get();
            rinch::core::untracked(|| {
                if let Some(sel) = sel {
                    let value = match sel {
                        SelectedEntity::Object(id) => format!("obj:{id}"),
                        SelectedEntity::Light(id) => format!("light:{id}"),
                        SelectedEntity::Camera => "camera".to_string(),
                        SelectedEntity::Scene => "scene".to_string(),
                        SelectedEntity::Project => "project".to_string(),
                    };
                    let current = tree_state.selected.get();
                    if !current.contains(&value) {
                        tree_state.controller.clear_selected();
                        tree_state.controller.select(&value);
                    }
                } else {
                    let current = tree_state.selected.get();
                    if !current.is_empty() {
                        tree_state.controller.clear_selected();
                    }
                }
            });
        });
    }

    // Wire SurfaceEvent → EditorState.editor_input + gizmo interaction.
    // The handler runs on the main thread every time the surface receives input.
    // All editor logic (gizmo hover/drag/end, mode switching) lives here.
    {
        let es = editor_state.clone();
        let ss = shared_state.clone();
        let sh = surface_handle.clone();
        let ui = use_context::<UiSignals>();
        let sliders = use_context::<SliderSignals>();
        let cmd_tx = use_context::<crate::CommandSender>().0.clone();
        let layout_for_events = use_context::<crate::layout::state::LayoutState>();
        let layout_backing = use_context::<crate::layout::state::LayoutBacking>();
        // Track last mouse position for delta computation (lock-free).
        let last_mx = std::cell::Cell::new(0.0f32);
        let last_my = std::cell::Cell::new(0.0f32);
        // Track drag-drop generation to suppress the spurious MouseUp that
        // rinch dispatches to the focused surface after ondrop+ondragend.
        let last_seen_dnd_gen = std::cell::Cell::new(0u64);
        surface_handle.set_event_handler(move |event| {
            use SurfaceEvent::*;
            use SurfaceMouseButton as Btn;

            // Helper: get viewport size for ray casting.
            let vp_size = || -> (f32, f32) {
                let (w, h) = sh.layout_size();
                (w.max(1) as f32, h.max(1) as f32)
            };

            match event {
                MouseMove { x, y } => {
                    // Send mouse delta through lock-free channel for camera.
                    let dx = x - last_mx.get();
                    let dy = y - last_my.get();
                    last_mx.set(x);
                    last_my.set(y);
                    let _ = cmd_tx.send(EditorCommand::MouseMove { x, y, dx, dy });

                    // Poll for layout config changes from the engine thread (project open).
                    if layout_backing.poll_dirty() {
                        layout_for_events.load_from_backing(&layout_backing);
                    }

                    // Brief lock for gizmo hover/drag (needs camera + scene state).
                    // Also update mouse_pos for gizmo ray casting.
                    let (mode, left_down) = if let Ok(mut state) = es.lock() {
                        state.editor_input.mouse_pos = glam::Vec2::new(x, y);

                        let mode = state.mode;
                        let left_down = state.editor_input.mouse_buttons[0];

                        // Gizmo hover detection + drag continue (Default mode only).
                        if mode == EditorMode::Default {
                            let (vp_w, vp_h) = vp_size();
                            if state.gizmo.dragging {
                                let (ray_o, ray_d) = crate::camera::screen_to_ray(
                                    &state.editor_camera, x, y, vp_w, vp_h,
                                );
                                if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                    let gizmo_mode = state.gizmo.mode;
                                    let pivot = state.gizmo.pivot;
                                    let initial_pos = state.gizmo.initial_position;
                                    let initial_rot = state.gizmo.initial_rotation;
                                    let initial_scale = state.gizmo.initial_scale;

                                    let (new_pos, new_rot, new_scale) = match gizmo_mode {
                                        gizmo::GizmoMode::Translate => {
                                            let delta = gizmo::compute_translate_delta(
                                                &state.gizmo, ray_o, ray_d,
                                            );
                                            (initial_pos + delta, initial_rot, initial_scale)
                                        }
                                        gizmo::GizmoMode::Rotate => {
                                            let rot_delta = gizmo::compute_rotate_delta(
                                                &state.gizmo, ray_o, ray_d, pivot,
                                            );
                                            let new_rot = rot_delta * initial_rot;
                                            let offset = initial_pos - pivot;
                                            let new_pos = pivot + rot_delta * offset;
                                            (new_pos, new_rot, initial_scale)
                                        }
                                        gizmo::GizmoMode::Scale => {
                                            let scale_delta = gizmo::compute_scale_delta(
                                                &state.gizmo, ray_o, ray_d,
                                            );
                                            (initial_pos, initial_rot, initial_scale * scale_delta)
                                        }
                                    };

                                    if let Some(entity) = state.world.find_entity_by_id(eid) {
                                        let wp = rkf_core::WorldPosition::new(
                                            glam::IVec3::ZERO, new_pos,
                                        );
                                        let _ = state.world.set_position(entity, wp);
                                        let _ = state.world.set_rotation(entity, new_rot);
                                        let _ = state.world.set_scale(entity, new_scale);

                                        // Push to slider signals so right panel tracks gizmo drag.
                                        let (rx, ry, rz) = new_rot.to_euler(glam::EulerRot::XYZ);
                                        sliders.push_object_values(
                                            new_pos,
                                            glam::Vec3::new(rx.to_degrees(), ry.to_degrees(), rz.to_degrees()),
                                            new_scale,
                                        );
                                    }
                                } else if let Some(SelectedEntity::Light(lid)) = state.selected_entity {
                                    // Light gizmo drag — translate only.
                                    let delta = gizmo::compute_translate_delta(
                                        &state.gizmo, ray_o, ray_d,
                                    );
                                    let new_pos = state.gizmo.initial_position + delta;
                                    state.light_editor.set_position(lid, new_pos);
                                    sliders.push_light_values(
                                        new_pos,
                                        state.light_editor.get_light(lid).map(|l| l.intensity).unwrap_or(1.0),
                                        state.light_editor.get_light(lid).map(|l| l.range).unwrap_or(10.0),
                                    );
                                }
                            } else {
                                // Hover detection: update hovered_axis.
                                let gc = match state.selected_entity {
                                    Some(SelectedEntity::Object(eid)) => {
                                        let scene = state.world.scene();
                                        scene.objects.iter().find(|o| o.id as u64 == eid)
                                            .and_then(|obj| {
                                                let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                                    &obj.root_node, glam::Mat4::IDENTITY,
                                                )?;
                                                Some(obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale))
                                            })
                                    }
                                    Some(SelectedEntity::Light(lid)) => {
                                        state.light_editor.get_light(lid).map(|l| l.position)
                                    }
                                    _ => None,
                                };
                                if let Some(gc) = gc {
                                    let cam_dist = (gc - state.editor_camera.position).length();
                                    let gizmo_size = cam_dist * 0.12;
                                    let (ray_o, ray_d) = crate::camera::screen_to_ray(
                                        &state.editor_camera, x, y, vp_w, vp_h,
                                    );
                                    state.gizmo.hovered_axis = gizmo::pick_gizmo_axis_for_mode(
                                        ray_o, ray_d, gc, gizmo_size, state.gizmo.mode,
                                    );
                                } else {
                                    state.gizmo.hovered_axis = gizmo::GizmoAxis::None;
                                }
                            }
                        }

                        (mode, left_down)
                    } else {
                        (EditorMode::Default, false)
                    };
                    // es lock released

                    // SINGLE ss lock: brush hit + pick check.
                    if let Ok(mut state) = ss.lock() {
                        // Brush hit request in Sculpt/Paint mode.
                        if left_down && matches!(mode, EditorMode::Sculpt | EditorMode::Paint) {
                            let scale = crate::engine_viewport::RENDER_SCALE;
                            let bx = (x * scale) as u32;
                            let by = (y * scale) as u32;
                            state.pending_brush_hit = Some((bx, by));
                        }

                        // Check if the engine thread completed a GPU pick.
                        if state.pick_completed {
                            state.pick_completed = false;
                            let picked = es.lock().ok().and_then(|s| s.selected_entity);
                            ui.selection.set(picked);
                            if picked.is_some() {
                                ui.properties_tab.set(0); // switch to Object tab
                            }
                        }
                    }
                }
                MouseDown { x, y, button } => {
                    // Send button state through lock-free channel for camera input.
                    let idx = match button { Btn::Left => 0, Btn::Right => 1, Btn::Middle => 2 };
                    let _ = cmd_tx.send(EditorCommand::MouseDown { button: idx, x, y });

                    // Brief lock for gizmo drag start (needs camera + scene state).
                    let mode = if let Ok(mut state) = es.lock() {
                        state.editor_input.mouse_buttons[idx] = true;
                        let mode = state.mode;

                        // Gizmo drag start: left-click in Default mode with object or light selected.
                        if button == Btn::Left && mode == EditorMode::Default {
                            let (vp_w, vp_h) = vp_size();
                            let right_down = state.editor_input.mouse_buttons[1];
                            if !right_down && !state.gizmo.dragging {
                                // Get gizmo center + initial transform for the selected entity.
                                let gizmo_info: Option<(glam::Vec3, glam::Vec3, glam::Quat, glam::Vec3)> =
                                    match state.selected_entity {
                                        Some(SelectedEntity::Object(eid)) => {
                                            let scene = state.world.scene();
                                            scene.objects.iter().find(|o| o.id as u64 == eid)
                                                .and_then(|obj| {
                                                    let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                                        &obj.root_node, glam::Mat4::IDENTITY,
                                                    )?;
                                                    let center = obj.position
                                                        + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale);
                                                    Some((center, obj.position, obj.rotation, obj.scale))
                                                })
                                        }
                                        Some(SelectedEntity::Light(lid)) => {
                                            state.light_editor.get_light(lid).map(|l| {
                                                (l.position, l.position, glam::Quat::IDENTITY, glam::Vec3::ONE)
                                            })
                                        }
                                        _ => None,
                                    };

                                if let Some((gc, entity_pos, entity_rot, entity_scale)) = gizmo_info {
                                    let cam_dist = (gc - state.editor_camera.position).length();
                                    let gizmo_size = cam_dist * 0.12;
                                    let (ray_o, ray_d) = crate::camera::screen_to_ray(
                                        &state.editor_camera, x, y, vp_w, vp_h,
                                    );
                                    let axis = gizmo::pick_gizmo_axis_for_mode(
                                        ray_o, ray_d, gc, gizmo_size, state.gizmo.mode,
                                    );
                                    if axis != gizmo::GizmoAxis::None {
                                        let start_point = match state.gizmo.mode {
                                            gizmo::GizmoMode::Translate | gizmo::GizmoMode::Scale => {
                                                if axis == gizmo::GizmoAxis::View {
                                                    let vn = (state.editor_camera.position - gc).normalize();
                                                    gizmo::project_to_plane(ray_o, ray_d, gc, vn)
                                                        .unwrap_or(gc)
                                                } else {
                                                    let t = gizmo::ray_axis_closest_point(
                                                        ray_o, ray_d, gc, axis.direction(),
                                                    );
                                                    gc + axis.direction() * t
                                                }
                                            }
                                            gizmo::GizmoMode::Rotate => {
                                                gizmo::project_to_plane(
                                                    ray_o, ray_d, gc, axis.plane_normal(),
                                                )
                                                .unwrap_or(gc)
                                            }
                                        };
                                        let view_normal =
                                            (state.editor_camera.position - gc).normalize();
                                        state.gizmo.begin_drag(
                                            axis, start_point, entity_pos, entity_rot, entity_scale,
                                            view_normal,
                                        );
                                        state.gizmo.pivot = gc;
                                    }
                                }
                            }
                        }

                        mode
                    } else {
                        EditorMode::Default
                    };
                    // es lock released

                    // SINGLE ss lock: brush hit request in Sculpt/Paint mode.
                    if button == Btn::Left
                        && matches!(mode, EditorMode::Sculpt | EditorMode::Paint)
                    {
                        let scale = crate::engine_viewport::RENDER_SCALE;
                        let bx = (x * scale) as u32;
                        let by = (y * scale) as u32;
                        if let Ok(mut state) = ss.lock() {
                            state.pending_brush_hit = Some((bx, by));
                        }
                    }
                }
                MouseUp { x, y, button } => {
                    // Rinch dispatches MouseUp to the focused surface AFTER
                    // ondrop+ondragend complete. Detect this by checking if
                    // the drag-drop generation changed since we last saw it.
                    let current_gen = ui.drag_drop_generation.get();
                    if current_gen != last_seen_dnd_gen.get() {
                        last_seen_dnd_gen.set(current_gen);
                        return; // Suppress this spurious post-drag MouseUp.
                    }
                    // Send button state through lock-free channel for camera input.
                    let idx = match button { Btn::Left => 0, Btn::Right => 1, Btn::Middle => 2 };
                    let _ = cmd_tx.send(EditorCommand::MouseUp { button: idx, x, y });

                    // Brief lock for gizmo drag end + undo.
                    let (mode, gizmo_was_dragging, picked_after_drag) = if let Ok(mut state) = es.lock() {
                        state.editor_input.mouse_buttons[idx] = false;
                        let mode = state.mode;
                        let was_dragging = state.gizmo.dragging;

                        if button == Btn::Left && was_dragging {
                            // Gizmo drag end: push undo action.
                            if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                let final_transform = {
                                    let scene = state.world.scene();
                                    scene.objects.iter().find(|o| o.id as u64 == eid)
                                        .map(|obj| (obj.position, obj.rotation, obj.scale))
                                };

                                if let Some((new_pos, new_rot, new_scale)) = final_transform {
                                    let desc = match state.gizmo.mode {
                                        gizmo::GizmoMode::Translate => "Move object",
                                        gizmo::GizmoMode::Rotate => "Rotate object",
                                        gizmo::GizmoMode::Scale => "Scale object",
                                    };
                                    let old_pos = state.gizmo.initial_position;
                                    let old_rot = state.gizmo.initial_rotation;
                                    let old_scale = state.gizmo.initial_scale;
                                    state.undo.push(crate::undo::UndoAction {
                                        kind: crate::undo::UndoActionKind::Transform {
                                            entity_id: eid,
                                            old_pos, old_rot, old_scale,
                                            new_pos, new_rot, new_scale,
                                        },
                                        timestamp_ms: 0,
                                        description: desc.to_string(),
                                    });
                                }
                            } else if let Some(SelectedEntity::Light(lid)) = state.selected_entity {
                                if let Some(light) = state.light_editor.get_light(lid) {
                                    let old_pos = state.gizmo.initial_position;
                                    let new_pos = light.position;
                                    state.undo.push(crate::undo::UndoAction {
                                        kind: crate::undo::UndoActionKind::Transform {
                                            entity_id: lid,
                                            old_pos,
                                            old_rot: glam::Quat::IDENTITY,
                                            old_scale: glam::Vec3::ONE,
                                            new_pos,
                                            new_rot: glam::Quat::IDENTITY,
                                            new_scale: glam::Vec3::ONE,
                                        },
                                        timestamp_ms: 0,
                                        description: "Move light".to_string(),
                                    });
                                }
                            }
                            state.gizmo.end_drag();
                            let sel = state.selected_entity;
                            (mode, true, sel)
                        } else {
                            (mode, false, None)
                        }
                    } else {
                        (EditorMode::Default, false, None)
                    };
                    // es lock released

                    if button == Btn::Left {
                        if gizmo_was_dragging {
                            // Signal update after lock released.
                            ui.selection.set(picked_after_drag);
                            if picked_after_drag.is_some() {
                                ui.properties_tab.set(0); // switch to Object tab
                            }
                        } else if matches!(mode, EditorMode::Sculpt | EditorMode::Paint) {
                            // Stroke ending is handled by the engine thread.
                        } else {
                            // No gizmo drag — send GPU pick request.
                            let (vp_w, vp_h) = sh.layout_size();
                            if vp_w > 0 && vp_h > 0 {
                                let scale = crate::engine_viewport::RENDER_SCALE;
                                let pick_x = (x * scale) as u32;
                                let pick_y = (y * scale) as u32;
                                if let Ok(mut state) = ss.lock() {
                                    state.pending_pick = Some((pick_x, pick_y));
                                }
                            }
                        }
                    }
                }
                MouseWheel { delta_y, .. } => {
                    let _ = cmd_tx.send(EditorCommand::Scroll { delta: delta_y });
                }
                KeyDown(key_data) => {
                    if let Some(kc) = translate_surface_key(&key_data.code) {
                        let mods = Modifiers {
                            shift: key_data.shift,
                            ctrl: key_data.ctrl,
                            alt: key_data.alt,
                        };
                        let _ = cmd_tx.send(EditorCommand::KeyDown { key: kc, modifiers: mods });
                    }
                }
                KeyUp(key_data) => {
                    if let Some(kc) = translate_surface_key(&key_data.code) {
                        let mods = Modifiers {
                            shift: key_data.shift,
                            ctrl: key_data.ctrl,
                            alt: key_data.alt,
                        };
                        let _ = cmd_tx.send(EditorCommand::KeyUp { key: kc, modifiers: mods });
                    }
                }
                FocusLost => {
                    if let Ok(mut state) = es.lock() {
                        state.editor_input = InputState::new();
                    }
                }
                _ => {}
            }

            let _ = ss;
        });
    }

    // Compute RenderSurface position from layout signals.
    // Must read layout state + backing to determine collapsed containers.
    let layout_for_surface = use_context::<crate::layout::state::LayoutState>();
    let backing_for_surface = use_context::<crate::layout::state::LayoutBacking>();

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;\
                    position:relative;\
                    background:var(--rinch-color-dark-9);color:var(--rinch-color-text);\
                    font-family:var(--rinch-font-family);",

            // Inject dark theme CSS overrides for rinch components.
            style { {DARK_OVERRIDES} }

            // ── RenderSurface overlay — absolute-positioned over center viewport ──
            // Lives outside any reactive_component_dom so it's never destroyed on
            // layout rebuilds. Position updates reactively via closure syntax.
            div {
                style: {
                    let backing = backing_for_surface.clone();
                    move || {
                        let lw = layout_for_surface.left_width.get();
                        let rw = layout_for_surface.right_width.get();
                        let bh = layout_for_surface.bottom_height.get();
                        let cfg = backing.load();
                        let left = if cfg.left.collapsed || cfg.left.zones.is_empty() { 0.0 } else { lw + 4.0 };
                        let right = if cfg.right.collapsed || cfg.right.zones.is_empty() { 0.0 } else { rw + 4.0 };
                        let bottom = 25.0 + if cfg.bottom.collapsed || cfg.bottom.zones.is_empty() { 0.0 } else { bh + 4.0 };
                        format!("position:absolute;top:36px;left:{left:.0}px;right:{right:.0}px;bottom:{bottom:.0}px;")
                    }
                },
                RenderSurface { surface: Some(surface_handle.clone()) }
            }

            // ── Titlebar spacer (36px) ──
            // Actual titlebar is position:absolute, rendered LAST for z-ordering.
            div { style: "height:36px;flex-shrink:0;background:var(--rinch-titlebar-bg);" }

            // ── Main content row (zone-based layout) ──
            LayoutRoot {}

            // ── Bottom status bar ──
            StatusBar {}

            // ── Floating panels (absolute overlay) ──
            FloatingPanelHost {}

            // ── Titlebar (absolute, last child for z-ordering / hit testing) ──
            TitleBar {}
        }
    }
}

/// Translate a `SurfaceKeyData.code` string (Physical key code) to `input::KeyCode`.
///
/// The `code` field uses the standard W3C physical key names (e.g. "KeyW", "ArrowLeft").
fn translate_surface_key(code: &str) -> Option<KeyCode> {
    match code {
        "KeyW" => Some(KeyCode::W),
        "KeyA" => Some(KeyCode::A),
        "KeyS" => Some(KeyCode::S),
        "KeyD" => Some(KeyCode::D),
        "KeyQ" => Some(KeyCode::Q),
        "KeyE" => Some(KeyCode::E),
        "KeyG" => Some(KeyCode::G),
        "KeyR" => Some(KeyCode::R),
        "KeyL" => Some(KeyCode::L),
        "KeyX" => Some(KeyCode::X),
        "KeyY" => Some(KeyCode::Y),
        "KeyZ" => Some(KeyCode::Z),
        "KeyF" => Some(KeyCode::F),
        "Delete" => Some(KeyCode::Delete),
        "Escape" => Some(KeyCode::Escape),
        "Space" => Some(KeyCode::Space),
        "Tab" => Some(KeyCode::Tab),
        "Enter" | "NumpadEnter" => Some(KeyCode::Return),
        "ShiftLeft" | "ShiftRight" => Some(KeyCode::ShiftLeft),
        "F5" => Some(KeyCode::F5),
        "F12" => Some(KeyCode::F12),
        "Digit1" => Some(KeyCode::Num1),
        "Digit2" => Some(KeyCode::Num2),
        "Digit3" => Some(KeyCode::Num3),
        _ => None,
    }
}
