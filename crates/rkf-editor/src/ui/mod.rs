//! Editor UI root component.
//!
//! Layout: titlebar → (left panel | viewport | right panel) → status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod camera_properties;
pub mod component_inspector;
pub mod components;
pub mod debug_panel;
pub mod environment_panel;
pub mod light_properties;
pub mod material_properties;
pub mod materials_panel;
pub mod object_properties;
pub mod scene_tree_panel;
pub mod shader_properties;
pub mod shaders_panel;
pub mod systems_panel;
pub mod library_panel;
pub mod models_panel;
pub mod welcome_screen;
mod slider_helpers;
pub mod brush_palette;
pub mod loading_modal;
pub mod titlebar;
pub mod right_panel;
pub mod status_bar;
pub mod viewport_toolbar;
pub mod widgets;
pub mod bound;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, SelectedEntity, UiSignals};
use crate::CommandSender;
use crate::gizmo;
use crate::input::{InputState, KeyCode, Modifiers};
use status_bar::StatusBar;
use titlebar::TitleBar;
use crate::wireframe;
use crate::layout::components::layout_root::LayoutRoot;
use crate::layout::components::floating_host::FloatingPanelHost;
use brush_palette::BrushPalette;
use loading_modal::LoadingModal;
use welcome_screen::WelcomeScreen;

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

    // ── Store setup ─────────────────────────────────────────────────────────
    // Selection-change sync and batch command sync are now store methods
    // (UiSignals::on_selection_changed) called from event handlers
    // instead of reactive Effects.

    // Create tree state as context so SceneTreePanel can use it.
    {
        let tree_state = UseTreeReturn::new(UseTreeOptions {
            initial_expanded: ["project".to_string(), "scene".to_string()]
                .into_iter()
                .collect(),
            ..Default::default()
        });
        create_context(tree_state);
    }

    // Wire SurfaceEvent → EditorState.editor_input + gizmo interaction.
    // The handler runs on the main thread every time the surface receives input.
    // All editor logic (gizmo hover/drag/end, mode switching) lives here.
    {
        let es = editor_state.clone();
        let ss = shared_state.clone();
        let sh = surface_handle.clone();
        let ui = use_context::<UiSignals>();
        let cmd = use_context::<CommandSender>();
        let tree_state = use_context::<UseTreeReturn>();
        let cmd_tx = use_context::<crate::CommandSender>().0.clone();
        // Track last mouse position for delta computation (lock-free).
        let last_mx = std::cell::Cell::new(0.0f32);
        let last_my = std::cell::Cell::new(0.0f32);
        // Track drag-drop generation to suppress the spurious MouseUp that
        // rinch dispatches to the focused surface after ondrop+ondragend.
        let last_seen_dnd_gen = std::cell::Cell::new(0u64);
        let model_drag_spawned = std::cell::Cell::new(false);
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
                    // Model drag: spawn on first move, update position on subsequent moves.
                    if ui.model_drag.is_active() {
                        if !model_drag_spawned.get() {
                            if let Some(path) = ui.model_drag.get() {
                                let _ = cmd_tx.send(EditorCommand::DragModelEnter { asset_path: path });
                                model_drag_spawned.set(true);
                            }
                        }
                        let _ = cmd_tx.send(EditorCommand::DragModelMove { x, y });
                        return; // Don't process gizmo/camera during model drag.
                    }

                    // Send mouse delta through lock-free channel for camera.
                    let dx = x - last_mx.get();
                    let dy = y - last_my.get();
                    last_mx.set(x);
                    last_my.set(y);
                    let _ = cmd_tx.send(EditorCommand::MouseMove { x, y, dx, dy });

                    // Brief lock for gizmo hover/drag (needs camera + scene state).
                    // Also update mouse_pos for gizmo ray casting.
                    let (mode, _left_down) = if let Ok(mut state) = es.lock() {
                        state.editor_input.mouse_pos = glam::Vec2::new(x, y);

                        let mode = state.mode;
                        let left_down = state.editor_input.viewport_left_down;

                        // Gizmo hover detection + drag continue (Default mode only).
                        if mode == EditorMode::Default {
                            let (vp_w, vp_h) = vp_size();
                            if state.gizmo.dragging {
                                let (ray_o, ray_d) = {
                                    let snap = state.extract_camera_snapshot();
                                    crate::camera::screen_to_ray_snapshot(&snap, x, y, vp_w, vp_h)
                                };
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

                                    if state.world.is_alive(eid) {
                                        // Send commands so the engine loop processes the
                                        // change, marks dirty, and pushes to UiSignals.
                                        let (rx, ry, rz) = new_rot.to_euler(glam::EulerRot::XYZ);
                                        let _ = cmd.0.send(EditorCommand::SetObjectPosition {
                                            entity_id: eid,
                                            position: new_pos,
                                        });
                                        let _ = cmd.0.send(EditorCommand::SetObjectRotation {
                                            entity_id: eid,
                                            rotation: glam::Vec3::new(
                                                rx.to_degrees(), ry.to_degrees(), rz.to_degrees(),
                                            ),
                                        });
                                        let _ = cmd.0.send(EditorCommand::SetObjectScale {
                                            entity_id: eid,
                                            scale: new_scale,
                                        });
                                    }
                                } else if let Some(SelectedEntity::Light(lid)) = state.selected_entity {
                                    // Light gizmo drag — translate only.
                                    let delta = gizmo::compute_translate_delta(
                                        &state.gizmo, ray_o, ray_d,
                                    );
                                    let new_pos = state.gizmo.initial_position + delta;
                                    let _ = cmd.0.send(EditorCommand::SetLightPosition {
                                        light_id: lid,
                                        position: new_pos,
                                    });
                                }
                            } else {
                                // Hover detection: update hovered_axis.
                                let gc = match state.selected_entity {
                                    Some(SelectedEntity::Object(eid)) => {
                                        // Read from hecs (authoritative).
                                        (|| -> Option<_> {
                                            let pos = state.world.position(eid).ok()?.to_vec3();
                                            let rot = state.world.rotation(eid).ok()?;
                                            let scale = state.world.scale(eid).ok()?;
                                            let hecs_e = state.world.ecs_entity_for(eid)?;
                                            let sdf = state.world.ecs_ref().get::<&rkf_runtime::components::SdfTree>(hecs_e).ok()?;
                                            let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                                &sdf.root, glam::Mat4::IDENTITY,
                                            )?;
                                            Some(pos + rot * ((lmin + lmax) * 0.5 * scale))
                                        })()
                                    }
                                    Some(SelectedEntity::Light(lid)) => {
                                        state.light_editor.get_light(lid).map(|l| l.position)
                                    }
                                    _ => None,
                                };
                                if let Some(gc) = gc {
                                    let cam_dist = (gc - state.extract_camera_snapshot().position).length();
                                    let gizmo_size = cam_dist * 0.12;
                                    let (ray_o, ray_d) = {
                                        let snap = state.extract_camera_snapshot();
                                        crate::camera::screen_to_ray_snapshot(&snap, x, y, vp_w, vp_h)
                                    };
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

                    // Brush hit request in Sculpt/Paint mode (hover + drag).
                    if matches!(mode, EditorMode::Sculpt | EditorMode::Paint) {
                        if let Ok(mut state) = ss.lock() {
                            let scale = crate::engine_viewport::RENDER_SCALE;
                            let bx = (x * scale) as u32;
                            let by = (y * scale) as u32;
                            state.pending_brush_hit = Some((bx, by));
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
                        if button == Btn::Left {
                            state.editor_input.viewport_left_down = true;
                        }
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
                                            // Read from hecs (authoritative).
                                            (|| -> Option<_> {
                                                let pos = state.world.position(eid).ok()?.to_vec3();
                                                let rot = state.world.rotation(eid).ok()?;
                                                let scale = state.world.scale(eid).ok()?;
                                                let hecs_e = state.world.ecs_entity_for(eid)?;
                                                let sdf = state.world.ecs_ref().get::<&rkf_runtime::components::SdfTree>(hecs_e).ok()?;
                                                let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                                    &sdf.root, glam::Mat4::IDENTITY,
                                                )?;
                                                let center = pos + rot * ((lmin + lmax) * 0.5 * scale);
                                                Some((center, pos, rot, scale))
                                            })()
                                        }
                                        Some(SelectedEntity::Light(lid)) => {
                                            state.light_editor.get_light(lid).map(|l| {
                                                (l.position, l.position, glam::Quat::IDENTITY, glam::Vec3::ONE)
                                            })
                                        }
                                        _ => None,
                                    };

                                if let Some((gc, entity_pos, entity_rot, entity_scale)) = gizmo_info {
                                    let cam_dist = (gc - state.extract_camera_snapshot().position).length();
                                    let gizmo_size = cam_dist * 0.12;
                                    let (ray_o, ray_d) = {
                                        let snap = state.extract_camera_snapshot();
                                        crate::camera::screen_to_ray_snapshot(&snap, x, y, vp_w, vp_h)
                                    };
                                    let axis = gizmo::pick_gizmo_axis_for_mode(
                                        ray_o, ray_d, gc, gizmo_size, state.gizmo.mode,
                                    );
                                    if axis != gizmo::GizmoAxis::None {
                                        let start_point = match state.gizmo.mode {
                                            gizmo::GizmoMode::Translate | gizmo::GizmoMode::Scale => {
                                                if axis == gizmo::GizmoAxis::View {
                                                    let vn = (state.extract_camera_snapshot().position - gc).normalize();
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
                                            (state.extract_camera_snapshot().position - gc).normalize();
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
                        // If a model was being drag-placed, finalize it.
                        if model_drag_spawned.get() {
                            let _ = cmd_tx.send(EditorCommand::DragModelDrop);
                            model_drag_spawned.set(false);
                        }
                        return; // Suppress this spurious post-drag MouseUp.
                    }
                    // Send button state through lock-free channel for camera input.
                    let idx = match button { Btn::Left => 0, Btn::Right => 1, Btn::Middle => 2 };
                    let _ = cmd_tx.send(EditorCommand::MouseUp { button: idx, x, y });

                    // Brief lock for gizmo drag end + undo.
                    let (mode, gizmo_was_dragging, picked_after_drag) = if let Ok(mut state) = es.lock() {
                        state.editor_input.mouse_buttons[idx] = false;
                        if button == Btn::Left {
                            state.editor_input.viewport_left_down = false;
                        }
                        let mode = state.mode;
                        let was_dragging = state.gizmo.dragging;

                        if button == Btn::Left && was_dragging {
                            // Gizmo drag end: push undo action.
                            if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                let final_transform = (|| -> Option<_> {
                                    let pos = state.world.position(eid).ok()?.to_vec3();
                                    let rot = state.world.rotation(eid).ok()?;
                                    let scale = state.world.scale(eid).ok()?;
                                    Some((pos, rot, scale))
                                })();

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
                                            entity_id: uuid::Uuid::from_u128(lid as u128),
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
                            ui.set_selection(picked_after_drag, &tree_state);
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
                        // Escape exits sculpt/paint mode — handle on UI thread
                        // so the toolbar signal updates immediately.
                        if kc == KeyCode::Escape && ui.editor_mode.get() != EditorMode::Default {
                            ui.editor_mode.set(EditorMode::Default);
                            let _ = cmd_tx.send(EditorCommand::SetEditorMode { mode: EditorMode::Default });
                        }
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
                    position:relative;overflow:hidden;\
                    background:var(--rinch-color-dark-9);color:var(--rinch-color-text);\
                    font-family:var(--rinch-font-family);",

            // Inject dark theme CSS overrides for rinch components.
            style { {DARK_OVERRIDES} }

            // ── Viewport overlay — absolute-positioned over center viewport ──
            // Contains the viewport toolbar + RenderSurface. Lives outside any
            // structural rebuild scope so it's never destroyed on layout rebuilds.
            div {
                style: {
                    let backing = backing_for_surface.clone();
                    move || {
                        let _ = layout_for_surface.structure_rev.get();
                        let lw = layout_for_surface.left_width.get();
                        let rw = layout_for_surface.right_width.get();
                        let bh = layout_for_surface.bottom_height.get();
                        let cfg = backing.load();
                        let left = if cfg.left.collapsed || cfg.left.zones.is_empty() { 0.0 } else { lw + 4.0 };
                        let right = if cfg.right.collapsed || cfg.right.zones.is_empty() { 0.0 } else { rw + 4.0 };
                        let bottom = 25.0 + if cfg.bottom.collapsed || cfg.bottom.zones.is_empty() { 0.0 } else { bh + 4.0 };
                        format!("position:absolute;z-index:1;top:36px;left:{left:.0}px;right:{right:.0}px;bottom:{bottom:.0}px;\
                                 display:flex;flex-direction:column;")
                    }
                },
                {viewport_toolbar::ViewportToolbar::default().render(__scope, &[])}
                div { style: "flex:1;min-height:0;",
                    RenderSurface { surface: Some(surface_handle.clone()) }
                }
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

            // ── Brush palette (compact overlay in Sculpt/Paint modes) ──
            BrushPalette {}

            // ── Welcome screen (absolute overlay, hides when project loaded) ──
            WelcomeScreen {}

            // ── Loading modal (absolute overlay, blocks interaction during builds) ──
            LoadingModal {}

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
