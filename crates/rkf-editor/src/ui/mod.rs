//! Editor UI root component.
//!
//! Layout: titlebar → (left panel | viewport | right panel) → status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod properties_panel;
pub mod scene_tree_panel;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiRevision, UiSignals};
use crate::gizmo;
use crate::input::{InputState, KeyCode, Modifiers};
use crate::wireframe;
use scene_tree_panel::SceneTreePanel;

// ── Style constants ─────────────────────────────────────────────────────────
// All colors use rinch theme CSS variables for the dark theme.
// Visual hierarchy: root=dark-9(#141414), panels=dark-8(#1f1f1f),
// titlebar=surface(#2e2e2e), text=#C9C9C9, dimmed=#828282, border=#424242.

const PANEL_BG: &str = "background:var(--rinch-color-dark-8);";
const PANEL_BORDER: &str = "border:1px solid var(--rinch-color-border);";
const LEFT_PANEL_WIDTH: &str = "width:250px;";
const RIGHT_PANEL_WIDTH: &str = "width:300px;";

const LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

const SECTION_STYLE: &str = "font-size:12px;color:var(--rinch-color-text);padding:6px 12px;";

const VALUE_STYLE: &str = "font-size:12px;color:var(--rinch-color-dimmed);padding:2px 12px;\
    font-family:var(--rinch-font-family-monospace);";

const DIVIDER_STYLE: &str = "height:1px;background:var(--rinch-color-border);margin:8px 0;";

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

    // Wire SurfaceEvent → EditorState.editor_input + gizmo interaction.
    // The handler runs on the main thread every time the surface receives input.
    // All editor logic (gizmo hover/drag/end, mode switching) lives here.
    {
        let es = editor_state.clone();
        let ss = shared_state.clone();
        let sh = surface_handle.clone();
        let rev = use_context::<UiRevision>();
        let ui = use_context::<UiSignals>();
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
                    // Read mode and left-button state before updating input.
                    let (mode, left_down) = es.lock().ok()
                        .map(|s| (s.mode, s.editor_input.mouse_buttons[0]))
                        .unwrap_or((EditorMode::Default, false));

                    if let Ok(mut state) = es.lock() {
                        let old = state.editor_input.mouse_pos;
                        state.editor_input.mouse_delta += glam::Vec2::new(x - old.x, y - old.y);
                        state.editor_input.mouse_pos = glam::Vec2::new(x, y);
                    }

                    // Gizmo hover detection + drag continue (main thread).
                    if mode == EditorMode::Default {
                        let (vp_w, vp_h) = vp_size();
                        if let Ok(mut state) = es.lock() {
                            if state.gizmo.dragging {
                                // Drag continue: update object transform.
                                if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                    let (ray_o, ray_d) = crate::camera::screen_to_ray(
                                        &state.editor_camera, x, y, vp_w, vp_h,
                                    );
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

                                    {
                                        let scene = state.world.scene_mut();
                                        if let Some(obj) = scene.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            obj.position = new_pos;
                                            obj.rotation = new_rot;
                                            obj.scale = new_scale;
                                        }
                                    }
                                }
                            } else {
                                // Hover detection: update hovered_axis.
                                if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                    let gc = {
                                        let scene = state.world.scene();
                                        scene.objects.iter().find(|o| o.id as u64 == eid)
                                            .and_then(|obj| {
                                                let (lmin, lmax) = wireframe::compute_node_tree_aabb(
                                                    &obj.root_node, glam::Mat4::IDENTITY,
                                                )?;
                                                Some(obj.position + obj.rotation * ((lmin + lmax) * 0.5 * obj.scale))
                                            })
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
                                } else {
                                    state.gizmo.hovered_axis = gizmo::GizmoAxis::None;
                                }
                            }
                        }
                    }

                    // In Sculpt/Paint mode with left button held, send
                    // continuous brush hit requests for stroke continuation.
                    if left_down && matches!(mode, EditorMode::Sculpt | EditorMode::Paint) {
                        let scale = crate::engine_viewport::RENDER_SCALE;
                        let bx = (x * scale) as u32;
                        let by = (y * scale) as u32;
                        if let Ok(mut state) = ss.lock() {
                            state.pending_brush_hit = Some((bx, by));
                        }
                    }

                    // Check if the engine thread completed a GPU pick.
                    let pick_done = ss.lock().ok()
                        .map(|mut s| {
                            let c = s.pick_completed;
                            if c { s.pick_completed = false; }
                            c
                        })
                        .unwrap_or(false);
                    if pick_done {
                        // Read the entity that was set by the render thread's pick handler.
                        let picked = es.lock().ok().and_then(|s| s.selected_entity);
                        ui.selection.set(picked);
                        rev.bump();
                    }
                }
                MouseDown { x, y, button } => {
                    let mode = es.lock().ok()
                        .map(|s| s.mode)
                        .unwrap_or(EditorMode::Default);

                    if let Ok(mut state) = es.lock() {
                        let idx = match button { Btn::Left => 0, Btn::Right => 1, Btn::Middle => 2 };
                        state.editor_input.mouse_buttons[idx] = true;
                    }

                    // Gizmo drag start: left-click in Default mode with object selected.
                    if button == Btn::Left && mode == EditorMode::Default {
                        let (vp_w, vp_h) = vp_size();
                        if let Ok(mut state) = es.lock() {
                            let right_down = state.editor_input.mouse_buttons[1];
                            if !right_down && !state.gizmo.dragging {
                                if let Some(SelectedEntity::Object(eid)) = state.selected_entity {
                                    let gc = {
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
                                    };

                                    if let Some((gc, obj_pos, obj_rot, obj_scale)) = gc {
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
                                                axis, start_point, obj_pos, obj_rot, obj_scale,
                                                view_normal,
                                            );
                                            state.gizmo.pivot = gc;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // In Sculpt/Paint mode, left-click starts a brush hit.
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
                    let mode = es.lock().ok()
                        .map(|s| s.mode)
                        .unwrap_or(EditorMode::Default);

                    if let Ok(mut state) = es.lock() {
                        let idx = match button { Btn::Left => 0, Btn::Right => 1, Btn::Middle => 2 };
                        state.editor_input.mouse_buttons[idx] = false;
                    }

                    if button == Btn::Left {
                        // Gizmo drag end: push undo action.
                        let gizmo_was_dragging = es.lock().ok()
                            .map(|s| s.gizmo.dragging)
                            .unwrap_or(false);

                        if gizmo_was_dragging {
                            if let Ok(mut state) = es.lock() {
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
                                }
                                state.gizmo.end_drag();
                            }
                            // Signal update after lock released.
                            ui.selection.set(
                                es.lock().ok().and_then(|s| s.selected_entity),
                            );
                            rev.bump();
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
                    if let Ok(mut state) = es.lock() {
                        state.editor_input.scroll_delta += delta_y;
                    }
                }
                KeyDown(key_data) => {
                    if let Some(kc) = translate_surface_key(&key_data.code) {
                        // Gizmo mode switching: G/R/L keys.
                        let mut gizmo_mode_changed = false;
                        if let Ok(mut state) = es.lock() {
                            state.editor_input.keys_pressed.insert(kc);
                            state.editor_input.keys_just_pressed.insert(kc);
                            state.editor_input.modifiers = Modifiers {
                                shift: key_data.shift,
                                ctrl: key_data.ctrl,
                                alt: key_data.alt,
                            };
                            if state.mode == EditorMode::Default {
                                match kc {
                                    KeyCode::G => {
                                        state.gizmo.mode = gizmo::GizmoMode::Translate;
                                        gizmo_mode_changed = true;
                                    }
                                    KeyCode::R => {
                                        state.gizmo.mode = gizmo::GizmoMode::Rotate;
                                        gizmo_mode_changed = true;
                                    }
                                    KeyCode::L => {
                                        state.gizmo.mode = gizmo::GizmoMode::Scale;
                                        gizmo_mode_changed = true;
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if gizmo_mode_changed {
                            let mode = es.lock().ok()
                                .map(|s| s.gizmo.mode)
                                .unwrap_or(gizmo::GizmoMode::Translate);
                            ui.gizmo_mode.set(mode);
                            rev.bump();
                        }
                    }
                }
                KeyUp(key_data) => {
                    if let Some(kc) = translate_surface_key(&key_data.code) {
                        if let Ok(mut state) = es.lock() {
                            state.editor_input.keys_pressed.remove(&kc);
                            state.editor_input.modifiers = Modifiers {
                                shift: key_data.shift,
                                ctrl: key_data.ctrl,
                                alt: key_data.alt,
                            };
                        }
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

    rsx! {
        div {
            style: "display:flex;flex-direction:column;width:100%;height:100%;\
                    position:relative;\
                    background:var(--rinch-color-dark-9);color:var(--rinch-color-text);\
                    font-family:var(--rinch-font-family);",

            // Inject dark theme CSS overrides for rinch components.
            style { {DARK_OVERRIDES} }

            // ── Titlebar spacer (36px) ──
            // Actual titlebar is position:absolute, rendered LAST for z-ordering.
            div { style: "height:36px;flex-shrink:0;background:var(--rinch-titlebar-bg);" }

            // ── Main content row (left + viewport + right) ──
            div {
                style: "display:flex;flex:1;min-height:0;",

                // Left panel — scene tree
                div {
                    style: {format!("{LEFT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-right:1px solid var(--rinch-color-border);\
                        display:flex;flex-direction:column;min-height:0;")},
                    SceneTreePanel {}
                }

                // Center viewport — zero-copy GPU compositing via RenderSurface
                div {
                    style: "flex:1;",
                    RenderSurface { surface: Some(surface_handle) }
                }

                // Right panel — mode-dependent
                div {
                    style: {format!("{RIGHT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-left:1px solid var(--rinch-color-border);\
                        display:flex;flex-direction:column;min-height:0;")},
                    RightPanel {}
                }
            }

            // ── Bottom status bar ──
            StatusBar {}

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

// ── Titlebar ────────────────────────────────────────────────────────────────

/// Combined titlebar with app title, menus, tool buttons, and window controls.
///
/// Replaces the separate MenuBar + Toolbar with a single 36px row in a
/// frameless window. Uses rinch's `rinch-app-menu-*` CSS classes for themed
/// dropdown rendering. The titlebar background has `data-drag-window` set so
/// rinch triggers `AppAction::DragWindow` on mousedown on empty areas.
#[component]
pub fn TitleBar() -> NodeHandle {
    use rinch::menu::{Menu, MenuItem, MenuEntryRef};

    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();
    let ui = use_context::<UiSignals>();

    // -1 = all closed, 0 = File, 1 = Edit, 2 = View.
    let active_menu: Signal<i32> = Signal::new(-1);

    // Outer wrapper — absolute positioning, last child of root for z-ordering.
    let wrapper = __scope.create_element("div");
    wrapper.set_attribute(
        "style",
        "position:absolute;top:0;left:0;width:100%;z-index:200;",
    );

    // ── Click-outside overlay ──────────────────────────────────────────
    let overlay = __scope.create_element("div");
    {
        let overlay_h = overlay.clone();
        Effect::new(move || {
            if active_menu.get() >= 0 {
                overlay_h.set_attribute("class", "rinch-app-menu-bar__overlay");
                overlay_h.set_attribute("style", "");
            } else {
                overlay_h.set_attribute("class", "rinch-app-menu-bar__overlay");
                overlay_h.set_attribute("style", "display:none;");
            }
        });
    }
    let overlay_click = __scope.register_handler(move || {
        active_menu.set(-1);
    });
    overlay.set_attribute("data-rid", &overlay_click.to_string());
    wrapper.append_child(&overlay);

    // ── Titlebar row ───────────────────────────────────────────────────
    // Uses `rinch-app-menu-bar` class which picks up `--rinch-titlebar-bg`
    // and `--rinch-titlebar-text` from the theme system.
    let bar = __scope.create_element("div");
    bar.set_attribute(
        "style",
        "position:relative;z-index:201;background:var(--rinch-titlebar-bg);",
    );
    {
        let bar_h = bar.clone();
        Effect::new(move || {
            let cls = if active_menu.get() >= 0 {
                "rinch-app-menu-bar rinch-app-menu-bar--engaged"
            } else {
                "rinch-app-menu-bar"
            };
            bar_h.set_attribute("class", cls);
        });
    }

    // Titlebar drag — rinch reads `data-drag-window` and emits AppAction::DragWindow
    // when a mousedown lands on this element (deepest-first: child handlers fire first).
    bar.set_attribute("data-drag-window", "1");

    // ── App title "RkiField" ───────────────────────────────────────────
    let title_container = __scope.create_element("div");
    title_container.set_attribute(
        "style",
        "position:relative;overflow:hidden;padding:0 12px 0 10px;\
         display:flex;align-items:center;",
    );
    let title_text = __scope.create_element("span");
    title_text.set_attribute(
        "style",
        "font-size:12px;font-weight:600;color:var(--rinch-color-text);\
         letter-spacing:0.5px;white-space:nowrap;",
    );
    title_text.append_child(&__scope.create_text("RkiField"));
    title_container.append_child(&title_text);
    // Gradient fade on right edge.
    let title_fade = __scope.create_element("span");
    title_fade.set_attribute(
        "style",
        "position:absolute;right:0;top:0;bottom:0;width:16px;\
         background:linear-gradient(90deg, transparent, var(--rinch-titlebar-bg));",
    );
    title_container.append_child(&title_fade);
    bar.append_child(&title_container);

    // ── Build menu data ────────────────────────────────────────────────
    let es = editor_state.clone();
    let rev = revision;
    let file_menu = Menu::new()
        .item(MenuItem::new("Open").shortcut("Ctrl+O").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.pending_open = true; }
                ui.bump_scene();
                ui.selection.set(None);
                rev.bump();
            }
        }))
        .item(MenuItem::new("Save").shortcut("Ctrl+S").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.pending_save = true; }
                rev.bump();
            }
        }))
        .item(MenuItem::new("Save As").shortcut("Ctrl+Shift+S").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.pending_save_as = true; }
                rev.bump();
            }
        }))
        .separator()
        .item(MenuItem::new("Quit").shortcut("Esc").on_click(move || {
            close_current_window();
        }));

    let spawn_primitives: &[(&str, &str)] = &[
        ("Box", "Box"),
        ("Sphere", "Sphere"),
        ("Capsule", "Capsule"),
        ("Torus", "Torus"),
        ("Cylinder", "Cylinder"),
    ];
    let mut spawn_menu = Menu::new();
    for &(label, prim_name) in spawn_primitives {
        spawn_menu = spawn_menu.item(MenuItem::new(label).on_click({
            let es = es.clone();
            let rev = rev;
            let name = prim_name.to_string();
            move || {
                if let Ok(mut s) = es.lock() {
                    s.pending_spawn = Some(name.clone());
                }
                ui.bump_scene();
                rev.bump();
            }
        }));
    }

    let edit_menu = Menu::new()
        .item(MenuItem::new("Undo").shortcut("Ctrl+Z").on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() { s.pending_undo = true; }
            }
        }))
        .item(MenuItem::new("Redo").shortcut("Ctrl+Y").on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() { s.pending_redo = true; }
            }
        }))
        .separator()
        .item(MenuItem::new("Delete").shortcut("Del").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.pending_delete = true; }
                ui.bump_scene();
                rev.bump();
            }
        }))
        .item(MenuItem::new("Duplicate").shortcut("Ctrl+D").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.pending_duplicate = true; }
                ui.bump_scene();
                rev.bump();
            }
        }))
        .separator()
        .submenu("Spawn", spawn_menu);

    let debug_modes: &[(&str, u32)] = &[
        ("Normal", 0),
        ("Normals", 1),
        ("Positions", 2),
        ("Material IDs", 3),
        ("Diffuse", 4),
        ("Specular", 5),
        ("GI Only", 6),
    ];
    let mut view_menu = Menu::new();
    for &(label, mode) in debug_modes {
        view_menu = view_menu.item(MenuItem::new(label).on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() {
                    s.debug_mode = mode;
                    s.pending_debug_mode = Some(mode);
                }
                ui.debug_mode.set(mode);
                rev.bump();
            }
        }));
    }

    // Camera presets (yaw, pitch in radians).
    view_menu = view_menu.separator();
    let cam_presets: &[(&str, f32, f32)] = &[
        ("Front",       0.0,                    0.0),
        ("Back",        std::f32::consts::PI,   0.0),
        ("Left",        std::f32::consts::FRAC_PI_2, 0.0),
        ("Right",       -std::f32::consts::FRAC_PI_2, 0.0),
        ("Top",         0.0,                    1.39),  // ~80°
        ("Perspective",  0.0,                    0.3),
    ];
    for &(label, yaw, pitch) in cam_presets {
        view_menu = view_menu.item(MenuItem::new(label).on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() {
                    s.editor_camera.set_orbit_angles(yaw, pitch);
                }
                rev.bump();
            }
        }));
    }

    // Grid overlay toggle.
    view_menu = view_menu.separator();
    view_menu = view_menu.item(MenuItem::new("Toggle Grid").shortcut("G").on_click({
        let es = es.clone();
        let rev = rev;
        move || {
            let new_val = es.lock().ok().map(|mut s| {
                s.show_grid = !s.show_grid;
                s.show_grid
            }).unwrap_or(false);
            ui.show_grid.set(new_val);
            rev.bump();
        }
    }));

    // ── Build DOM for each top-level menu ──────────────────────────────
    let menus: &[(&str, &Menu)] = &[
        ("File", &file_menu),
        ("Edit", &edit_menu),
        ("View", &view_menu),
    ];

    for (idx, &(label, menu)) in menus.iter().enumerate() {
        let index = idx as i32;

        let item = __scope.create_element("div");
        {
            let item_h = item.clone();
            Effect::new(move || {
                let cls = if active_menu.get() == index {
                    "rinch-app-menu-item rinch-app-menu-item--opened"
                } else {
                    "rinch-app-menu-item"
                };
                item_h.set_attribute("class", cls);
            });
        }

        let onenter_id = __scope.register_handler(move || {
            if active_menu.get() >= 0 {
                active_menu.set(index);
            }
        });
        item.set_attribute("data-onenter", &onenter_id.to_string());

        let label_node = __scope.create_element("div");
        label_node.set_attribute("class", "rinch-app-menu-item__label");
        label_node.append_child(&__scope.create_text(label));
        let label_click = __scope.register_handler(move || {
            let current = active_menu.get();
            if current == index {
                active_menu.set(-1);
            } else {
                active_menu.set(index);
            }
        });
        label_node.set_attribute("data-rid", &label_click.to_string());
        item.append_child(&label_node);

        let dropdown = __scope.create_element("div");
        {
            let dd_h = dropdown.clone();
            Effect::new(move || {
                if active_menu.get() == index {
                    dd_h.set_attribute(
                        "class",
                        "rinch-app-menu-item__dropdown rinch-app-menu-item__dropdown--visible",
                    );
                } else {
                    dd_h.set_attribute("class", "rinch-app-menu-item__dropdown");
                }
            });
        }

        for entry in menu.iter_entries() {
            match entry {
                MenuEntryRef::Item { label, shortcut, enabled, callback } => {
                    let entry_node = __scope.create_element("div");
                    let cls = if enabled {
                        "rinch-app-menu-entry"
                    } else {
                        "rinch-app-menu-entry rinch-app-menu-entry--disabled"
                    };
                    entry_node.set_attribute("class", cls);

                    let lbl = __scope.create_element("span");
                    lbl.set_attribute("class", "rinch-app-menu-entry__label");
                    lbl.append_child(&__scope.create_text(label));
                    entry_node.append_child(&lbl);

                    if let Some(sc) = shortcut {
                        let sc_span = __scope.create_element("span");
                        sc_span.set_attribute("class", "rinch-app-menu-entry__shortcut");
                        sc_span.append_child(&__scope.create_text(sc));
                        entry_node.append_child(&sc_span);
                    }

                    if enabled {
                        if let Some(cb) = callback {
                            let cb = std::rc::Rc::clone(cb);
                            let hid = __scope.register_handler(move || {
                                active_menu.set(-1);
                                cb();
                            });
                            entry_node.set_attribute("data-rid", &hid.to_string());
                        }
                    }

                    dropdown.append_child(&entry_node);
                }
                MenuEntryRef::Separator => {
                    let sep = __scope.create_element("div");
                    sep.set_attribute("class", "rinch-app-menu-separator");
                    dropdown.append_child(&sep);
                }
                MenuEntryRef::Submenu { .. } => {}
            }
        }

        item.append_child(&dropdown);
        bar.append_child(&item);
    }

    // ── Separator between menus and tool buttons ───────────────────────
    let separator = __scope.create_element("div");
    separator.set_attribute(
        "style",
        "width:1px;height:14px;background:var(--rinch-color-border);margin:0 8px;",
    );
    bar.append_child(&separator);

    // ── Tool buttons (Sculpt/Paint) ────────────────────────────────────
    {
        let es = editor_state.clone();
        let tool_container = __scope.create_element("div");
        tool_container.set_attribute(
            "style",
            "display:flex;align-items:center;gap:2px;",
        );

        rinch::core::reactive_component_dom(__scope, &tool_container, move |__scope| {
            let _ = ui.editor_mode.get();
            // Legacy fallback.
            revision.track();

            let current_mode = {
                let es = es.lock().unwrap();
                es.mode
            };

            let inner = __scope.create_element("div");
            inner.set_attribute("style", "display:flex;align-items:center;gap:2px;");

            for mode in EditorMode::TOOLS.iter() {
                let mode = *mode;
                let btn = __scope.create_element("div");
                let is_active = mode == current_mode;

                let style = if is_active {
                    "padding:2px 8px;cursor:pointer;border-radius:3px;\
                     background:var(--rinch-titlebar-active);\
                     font-size:11px;color:var(--rinch-color-text);user-select:none;\
                     border:1px solid var(--rinch-color-border);font-weight:600;"
                } else {
                    "padding:2px 8px;cursor:pointer;border-radius:3px;\
                     background:transparent;\
                     font-size:11px;color:var(--rinch-color-dimmed);user-select:none;\
                     border:1px solid transparent;font-weight:400;"
                };

                btn.set_attribute("style", style);

                btn.append_child(&__scope.create_text(mode.name()));

                let es = es.clone();
                let handler_id = __scope.register_handler(move || {
                    let new_mode = if let Ok(mut state) = es.lock() {
                        if state.mode == mode {
                            state.mode = EditorMode::Default;
                            EditorMode::Default
                        } else {
                            state.mode = mode;
                            mode
                        }
                    } else {
                        EditorMode::Default
                    };
                    ui.editor_mode.set(new_mode);
                    revision.bump();
                });
                btn.set_attribute("data-rid", &handler_id.to_string());

                inner.append_child(&btn);
            }

            inner
        });

        bar.append_child(&tool_container);
    }

    // ── Spacer ─────────────────────────────────────────────────────────
    let spacer = __scope.create_element("div");
    spacer.set_attribute("style", "flex:1;");
    bar.append_child(&spacer);

    // ── Window controls ────────────────────────────────────────────────
    let controls = __scope.create_element("div");
    controls.set_attribute(
        "style",
        "display:flex;align-items:center;height:100%;",
    );

    // Minimize [─]
    let min_btn = __scope.create_element("div");
    min_btn.set_attribute("class", "rinch-borderlesswindow__control");
    min_btn.set_attribute(
        "style",
        "width:46px;height:36px;display:flex;align-items:center;\
         justify-content:center;cursor:pointer;\
         color:var(--rinch-color-dimmed);font-size:14px;",
    );
    min_btn.append_child(&__scope.create_text("\u{2500}"));  // ─
    {
        let hid = __scope.register_handler(move || {
            minimize_current_window();
        });
        min_btn.set_attribute("data-rid", &hid.to_string());
    }
    controls.append_child(&min_btn);

    // Maximize [□]
    let max_btn = __scope.create_element("div");
    max_btn.set_attribute("class", "rinch-borderlesswindow__control");
    max_btn.set_attribute(
        "style",
        "width:46px;height:36px;display:flex;align-items:center;\
         justify-content:center;cursor:pointer;\
         color:var(--rinch-color-dimmed);font-size:14px;",
    );
    max_btn.append_child(&__scope.create_text("\u{25a1}"));  // □
    {
        let hid = __scope.register_handler(move || {
            toggle_maximize_current_window();
        });
        max_btn.set_attribute("data-rid", &hid.to_string());
    }
    controls.append_child(&max_btn);

    // Close [×]
    let close_btn = __scope.create_element("div");
    close_btn.set_attribute("class", "rinch-borderlesswindow__control rinch-borderlesswindow__control--close");
    close_btn.set_attribute(
        "style",
        "width:46px;height:36px;display:flex;align-items:center;\
         justify-content:center;cursor:pointer;\
         color:var(--rinch-color-dimmed);font-size:16px;",
    );
    close_btn.append_child(&__scope.create_text("\u{00d7}"));  // ×
    {
        let hid = __scope.register_handler(move || {
            close_current_window();
        });
        close_btn.set_attribute("data-rid", &hid.to_string());
    }
    controls.append_child(&close_btn);

    bar.append_child(&controls);

    wrapper.append_child(&bar);
    wrapper
}

// ── Slider helper ───────────────────────────────────────────────────────────

/// Build a labeled slider row and append it to the container.
///
/// Creates a row with label + reactive value display on top and a Slider beneath.
/// The slider updates `signal` on drag (fine-grained, no revision bump) and
/// calls `on_update` to write back to editor state.
fn build_slider_row(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    decimals: usize,
    on_update: impl Fn(f64) + 'static,
) {
    let row = scope.create_element("div");
    row.set_attribute("style", "padding:3px 12px;");

    // Label + reactive value display.
    let label_row = scope.create_element("div");
    label_row.set_attribute(
        "style",
        "display:flex;justify-content:space-between;\
         font-size:11px;color:var(--rinch-color-dimmed);margin-bottom:2px;",
    );
    label_row.append_child(&scope.create_text(label));

    let val_span = scope.create_element("span");
    val_span.set_attribute(
        "style",
        "font-family:var(--rinch-font-family-monospace);",
    );
    {
        let val_h = val_span.clone();
        let suffix = suffix.to_string();
        Effect::new(move || {
            let v = signal.get();
            let text = match decimals {
                0 => format!("{v:.0}{suffix}"),
                1 => format!("{v:.1}{suffix}"),
                _ => format!("{v:.2}{suffix}"),
            };
            val_h.set_text(&text);
        });
    }
    label_row.append_child(&val_span);
    row.append_child(&label_row);

    // Slider component.
    let slider = Slider {
        min: Some(min),
        max: Some(max),
        step: Some(step),
        value_signal: Some(signal),
        size: "xs".to_string(),
        onchange: Some(ValueCallback::new(move |v: f64| {
            signal.set(v);
            on_update(v);
        })),
        ..Default::default()
    };
    // Render the slider inside `untracked` so the Slider's internal
    // `value_signal.get()` (for initial value) doesn't subscribe the
    // parent reactive_component_dom scope to every slider signal.
    // Without this, dragging ANY slider rebuilds the entire RightPanel.
    let slider_node = rinch::core::untracked(|| slider.render(scope, &[]));
    row.append_child(&slider_node);
    container.append_child(&row);
}

/// Build a slider row bound to a `SliderSignals` signal (no lock closure).
///
/// The slider updates `signal` on drag. The batch sync `Effect` in
/// `RightPanel` writes all signal values to `EditorState` once per frame.
fn build_synced_slider(
    scope: &mut RenderScope,
    container: &NodeHandle,
    label: &str,
    suffix: &str,
    signal: Signal<f64>,
    min: f64,
    max: f64,
    step: f64,
    decimals: usize,
) {
    build_slider_row(
        scope, container, label, suffix, signal,
        min, max, step, decimals,
        move |_v| {
            // No-op — signal already set by build_slider_row's onchange.
            // Sync happens in the batch Effect.
        },
    );
}

// ── Mode-dependent right panel ──────────────────────────────────────────────

/// Right panel — always shows properties of the selected object + environment.
///
/// When a Sculpt/Paint tool is active, shows brush settings above the
/// properties section (placeholder for now). Camera selection shows
/// interactive FOV, fly speed, near/far sliders. Below properties, an
/// always-visible environment section shows atmosphere, fog, and quick
/// post-processing controls.
#[component]
pub fn RightPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:scroll;min-height:0;height:0;",
    );

    // Centralized slider signals — created once in main.rs, stored in rinch context.
    // A single batch Effect (below) syncs all values to EditorState per frame.
    let sliders = use_context::<SliderSignals>();

    // ── Batch sync Effect: slider signals + toggle signals → EditorState ──
    // This is the ONLY place that locks EditorState for slider/toggle updates.
    // One lock per frame, regardless of how many sliders changed.
    {
        let es = editor_state.clone();
        Effect::new(move || {
            // Subscribe to all slider signals.
            sliders.track_all();
            // Subscribe to toggle signals (from UiSignals).
            let _ = ui.atmo_enabled.get();
            let _ = ui.fog_enabled.get();
            let _ = ui.clouds_enabled.get();
            let _ = ui.bloom_enabled.get();
            let _ = ui.dof_enabled.get();
            let _ = ui.tone_map_mode.get();

            // Batch write to EditorState.
            if let Ok(mut es) = es.lock() {
                sliders.sync_to_state(&mut es);
                // Sync toggles.
                es.environment.atmosphere.enabled = ui.atmo_enabled.get();
                es.environment.fog.enabled = ui.fog_enabled.get();
                es.environment.clouds.enabled = ui.clouds_enabled.get();
                es.environment.post_process.bloom_enabled = ui.bloom_enabled.get();
                es.environment.post_process.dof_enabled = ui.dof_enabled.get();
                es.environment.post_process.tone_map_mode = ui.tone_map_mode.get();
                es.environment.mark_dirty();
            }
        });
    }

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        // Env toggle signals — needed because toggle labels are built in this closure.
        let _ = ui.atmo_enabled.get();
        let _ = ui.fog_enabled.get();
        let _ = ui.clouds_enabled.get();
        let _ = ui.bloom_enabled.get();
        let _ = ui.dof_enabled.get();
        let _ = ui.tone_map_mode.get();
        // Legacy fallback.
        revision.track();

        let (mode, selected_entity) = {
            let es = es.lock().unwrap();
            (es.mode, es.selected_entity)
        };

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // ── Tool-specific settings (when Sculpt/Paint active) ──
        match mode {
            EditorMode::Sculpt => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Sculpt Brush"));
                container.append_child(&header);

                // Brush type (read-only for now).
                {
                    let es_lock = es.lock().unwrap();
                    let type_name = match es_lock.sculpt.current_settings.brush_type {
                        crate::sculpt::BrushType::Add => "Add",
                        crate::sculpt::BrushType::Subtract => "Subtract",
                        crate::sculpt::BrushType::Smooth => "Smooth",
                        crate::sculpt::BrushType::Flatten => "Flatten",
                        crate::sculpt::BrushType::Sharpen => "Sharpen",
                    };
                    let row = __scope.create_element("div");
                    row.set_attribute("style", VALUE_STYLE);
                    row.append_child(&__scope.create_text(&format!("Type: {type_name}")));
                    container.append_child(&row);
                }

                build_synced_slider(__scope, &container, "Radius", "",
                    sliders.brush_radius, 0.1, 10.0, 0.1, 1);
                build_synced_slider(__scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2);

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);
            }
            EditorMode::Paint => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Paint Brush"));
                container.append_child(&header);

                build_synced_slider(__scope, &container, "Radius", "",
                    sliders.brush_radius, 0.1, 10.0, 0.1, 1);
                build_synced_slider(__scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2);

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);
            }
            EditorMode::Default => {}
        }

        // ── Properties (always shown) ──
        let header = __scope.create_element("div");
        header.set_attribute("style", LABEL_STYLE);
        header.append_child(&__scope.create_text("Properties"));
        container.append_child(&header);

        // ── Camera-specific property editing with sliders ──
        if let Some(SelectedEntity::Camera) = selected_entity {
            let pos = {
                let es = es.lock().unwrap();
                es.editor_camera.position
            };

            let name_row = __scope.create_element("div");
            name_row.set_attribute("style", SECTION_STYLE);
            name_row.append_child(&__scope.create_text("Camera"));
            container.append_child(&name_row);

            build_synced_slider(__scope, &container, "FOV", "\u{00b0}",
                sliders.fov, 30.0, 120.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Fly Speed", "",
                sliders.fly_speed, 0.5, 500.0, 0.5, 1);
            build_synced_slider(__scope, &container, "Near Plane", "",
                sliders.near, 0.01, 10.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Far Plane", "",
                sliders.far, 100.0, 10000.0, 100.0, 0);

            // Divider before position.
            let div = __scope.create_element("div");
            div.set_attribute(
                "style",
                "height:1px;background:var(--rinch-color-border);margin:6px 12px;",
            );
            container.append_child(&div);

            // Position (read-only).
            let pos_row = __scope.create_element("div");
            pos_row.set_attribute("style", VALUE_STYLE);
            pos_row.append_child(&__scope.create_text(
                &format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z),
            ));
            container.append_child(&pos_row);

            // ── Divider between camera props and environment ──
            let div = __scope.create_element("div");
            div.set_attribute(
                "style",
                DIVIDER_STYLE,
            );
            container.append_child(&div);

            // ── Atmosphere section ──
            let atmo_header = __scope.create_element("div");
            atmo_header.set_attribute("style", LABEL_STYLE);
            atmo_header.append_child(&__scope.create_text("Atmosphere"));
            container.append_child(&atmo_header);

            build_synced_slider(__scope, &container, "Sun Azimuth", "\u{00b0}",
                sliders.sun_azimuth, 0.0, 360.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Sun Elevation", "\u{00b0}",
                sliders.sun_elevation, -90.0, 90.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Sun Intensity", "",
                sliders.sun_intensity, 0.0, 10.0, 0.1, 1);

            // Sun color (read-only display).
            {
                let es_lock = es.lock().unwrap();
                let sc = es_lock.environment.atmosphere.sun_color;
                let color_row = __scope.create_element("div");
                color_row.set_attribute("style", VALUE_STYLE);
                color_row.append_child(&__scope.create_text(
                    &format!("Sun Color: ({:.2}, {:.2}, {:.2})", sc.x, sc.y, sc.z),
                ));
                container.append_child(&color_row);
            }

            // Atmosphere enable toggle.
            {
                let atmo_on = ui.atmo_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if atmo_on { "Atmosphere: ON" } else { "Atmosphere: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.atmo_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Rayleigh Scale", "",
                sliders.rayleigh_scale, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Mie Scale", "",
                sliders.mie_scale, 0.0, 5.0, 0.1, 1);

            // ── Fog section ──
            let fog_header = __scope.create_element("div");
            fog_header.set_attribute("style", LABEL_STYLE);
            fog_header.append_child(&__scope.create_text("Fog"));
            container.append_child(&fog_header);

            // Fog enable toggle.
            {
                let fog_on = ui.fog_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if fog_on { "Fog: ON" } else { "Fog: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.fog_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Fog Density", "",
                sliders.fog_density, 0.0, 0.5, 0.001, 3);
            build_synced_slider(__scope, &container, "Height Falloff", "",
                sliders.fog_height_falloff, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Dust Density", "",
                sliders.dust_density, 0.0, 0.1, 0.001, 3);
            build_synced_slider(__scope, &container, "Dust Asymmetry", "",
                sliders.dust_asymmetry, 0.0, 0.95, 0.05, 2);

            // ── Clouds section ──
            let cloud_header = __scope.create_element("div");
            cloud_header.set_attribute("style", LABEL_STYLE);
            cloud_header.append_child(&__scope.create_text("Clouds"));
            container.append_child(&cloud_header);

            // Cloud enable toggle.
            {
                let cloud_on = ui.clouds_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if cloud_on { "Clouds: ON" } else { "Clouds: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.clouds_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Coverage", "",
                sliders.cloud_coverage, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Cloud Density", "",
                sliders.cloud_density, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Altitude", "m",
                sliders.cloud_altitude, 0.0, 5000.0, 50.0, 0);
            build_synced_slider(__scope, &container, "Thickness", "m",
                sliders.cloud_thickness, 10.0, 10000.0, 50.0, 0);
            build_synced_slider(__scope, &container, "Wind Speed", "",
                sliders.cloud_wind_speed, 0.0, 50.0, 0.5, 1);

            // ── Post-Processing section ──
            let pp_header = __scope.create_element("div");
            pp_header.set_attribute("style", LABEL_STYLE);
            pp_header.append_child(&__scope.create_text("Post-Processing"));
            container.append_child(&pp_header);

            // Bloom enable toggle.
            {
                let bloom_on = ui.bloom_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if bloom_on { "Bloom: ON" } else { "Bloom: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.bloom_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Bloom Intensity", "",
                sliders.bloom_intensity, 0.0, 2.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Bloom Threshold", "",
                sliders.bloom_threshold, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Exposure", "",
                sliders.exposure, 0.1, 10.0, 0.1, 1);

            // Tone map mode toggle (ACES / AgX).
            {
                let tm_mode = ui.tone_map_mode.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if tm_mode == 0 { "Tone Map: ACES" } else { "Tone Map: AgX" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.tone_map_mode.update(|v| *v = if *v == 0 { 1 } else { 0 });
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Sharpen", "",
                sliders.sharpen, 0.0, 2.0, 0.05, 2);

            // DoF enable toggle.
            {
                let dof_on = ui.dof_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if dof_on { "DoF: ON" } else { "DoF: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.dof_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Focus Distance", "",
                sliders.dof_focus_dist, 0.1, 50.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Focus Range", "",
                sliders.dof_focus_range, 0.1, 20.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Max CoC", "px",
                sliders.dof_max_coc, 1.0, 32.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Motion Blur", "",
                sliders.motion_blur, 0.0, 3.0, 0.1, 1);
            build_synced_slider(__scope, &container, "God Rays", "",
                sliders.god_rays, 0.0, 2.0, 0.05, 2);
            build_synced_slider(__scope, &container, "Vignette", "",
                sliders.vignette, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Grain", "",
                sliders.grain, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Chromatic Ab.", "",
                sliders.chromatic_ab, 0.0, 1.0, 0.01, 2);

            // ── Animation controls ──
            {
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);

                let hdr = __scope.create_element("div");
                hdr.set_attribute("style", SECTION_STYLE);
                hdr.append_child(&__scope.create_text("Animation"));
                container.append_child(&hdr);

                // Play / Pause / Stop buttons.
                let btn_row = __scope.create_element("div");
                btn_row.set_attribute("style", "display:flex;gap:6px;padding:2px 12px;");

                for (label, state) in [
                    ("Play", crate::animation_preview::PlaybackState::Playing),
                    ("Pause", crate::animation_preview::PlaybackState::Paused),
                    ("Stop", crate::animation_preview::PlaybackState::Stopped),
                ] {
                    let btn = __scope.create_element("div");
                    let is_active = es.lock()
                        .map(|e| e.animation.playback_state == state)
                        .unwrap_or(false);
                    let bg = if is_active { "var(--rinch-primary-color)" } else { "var(--rinch-color-dark-7)" };
                    btn.set_attribute("style", &format!(
                        "padding:2px 8px;border-radius:3px;cursor:pointer;\
                         background:{bg};font-size:11px;color:var(--rinch-color-text);",
                    ));
                    btn.append_child(&__scope.create_text(label));

                    let hid = __scope.register_handler({
                        let es = es.clone();
                        let rev = revision;
                        move || {
                            if let Ok(mut es) = es.lock() {
                                es.animation.playback_state = state;
                                if matches!(state, crate::animation_preview::PlaybackState::Stopped) {
                                    es.animation.current_time = 0.0;
                                }
                            }
                            rev.bump();
                        }
                    });
                    btn.set_attribute("data-rid", &hid.to_string());
                    btn_row.append_child(&btn);
                }
                container.append_child(&btn_row);

                let anim_speed_signal: Signal<f64> = Signal::new(
                    es.lock().map(|e| e.animation.speed as f64).unwrap_or(1.0),
                );

                build_slider_row(
                    __scope, &container, "Speed", "x", anim_speed_signal,
                    0.0, 4.0, 0.1, 1,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.animation.speed = v as f32;
                        }
                    }},
                );
            }
        } else {
            // ── Non-camera entity properties ──
            match selected_entity {
                Some(SelectedEntity::Object(eid)) => {
                    // Read object info + transform from world scene.
                    let obj_info = es.lock().ok().and_then(|es_lock| {
                        let scene = es_lock.world.scene();
                        let obj = scene.objects.iter().find(|o| o.id as u64 == eid)?;
                        let name = obj.name.clone();
                        let child_count = scene.objects.iter()
                            .filter(|o| o.parent_id == Some(obj.id))
                            .count();
                        let (x, y, z) = obj.rotation.to_euler(glam::EulerRot::XYZ);
                        let is_vox = matches!(
                            obj.root_node.sdf_source,
                            rkf_core::scene_node::SdfSource::Voxelized { .. }
                        );
                        let non_uniform = (obj.scale - glam::Vec3::ONE).length() > 1e-4;
                        let xf = Some((
                            obj.position,
                            glam::Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees()),
                            obj.scale,
                        ));
                        let show_revoxelize = is_vox && non_uniform;
                        Some((name, child_count, xf, show_revoxelize))
                    });

                    if let Some((name, child_count, xf, show_revoxelize)) = obj_info {
                        // Name.
                        let name_row = __scope.create_element("div");
                        name_row.set_attribute("style", SECTION_STYLE);
                        name_row.append_child(&__scope.create_text(&name));
                        container.append_child(&name_row);

                        // Entity ID.
                        let id_row = __scope.create_element("div");
                        id_row.set_attribute("style", VALUE_STYLE);
                        id_row.append_child(
                            &__scope.create_text(&format!("Entity ID: {eid}")),
                        );
                        container.append_child(&id_row);

                        if child_count > 0 {
                            let cr = __scope.create_element("div");
                            cr.set_attribute("style", VALUE_STYLE);
                            cr.append_child(
                                &__scope.create_text(&format!("Children: {child_count}")),
                            );
                            container.append_child(&cr);
                        }

                        // Transform sliders (only when scene object found).
                        if let Some((pos, rot_deg, scale)) = xf {
                            let div = __scope.create_element("div");
                            div.set_attribute("style", DIVIDER_STYLE);
                            container.append_child(&div);

                            let xf_hdr = __scope.create_element("div");
                            xf_hdr.set_attribute("style", LABEL_STYLE);
                            xf_hdr.append_child(&__scope.create_text("Transform"));
                            container.append_child(&xf_hdr);

                            // Position.
                            let pos_x_sig = Signal::new(pos.x as f64);
                            let pos_y_sig = Signal::new(pos.y as f64);
                            let pos_z_sig = Signal::new(pos.z as f64);

                            build_slider_row(
                                __scope, &container, "Position X", "", pos_x_sig,
                                -500.0, 500.0, 0.01, 2,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        {
                                            let sc = es.world.scene_mut();
                                            if let Some(obj) = sc.objects.iter_mut()
                                                .find(|o| o.id as u64 == eid)
                                            {
                                                obj.position.x = v as f32;
                                            }
                                        }
                                    }
                                }},
                            );
                            build_slider_row(
                                __scope, &container, "Position Y", "", pos_y_sig,
                                -500.0, 500.0, 0.01, 2,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        let sc = es.world.scene_mut();
                                        if let Some(obj) = sc.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            obj.position.y = v as f32;
                                        }
                                    }
                                }},
                            );
                            build_slider_row(
                                __scope, &container, "Position Z", "", pos_z_sig,
                                -500.0, 500.0, 0.01, 2,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        let sc = es.world.scene_mut();
                                        if let Some(obj) = sc.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            obj.position.z = v as f32;
                                        }
                                    }
                                }},
                            );

                            // Rotation (Euler XYZ degrees).
                            let rot_x_sig = Signal::new(rot_deg.x as f64);
                            let rot_y_sig = Signal::new(rot_deg.y as f64);
                            let rot_z_sig = Signal::new(rot_deg.z as f64);

                            build_slider_row(
                                __scope, &container, "Rotation X", "\u{00b0}", rot_x_sig,
                                -180.0, 180.0, 0.5, 1,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        let sc = es.world.scene_mut();
                                        if let Some(obj) = sc.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            let (_, cy, cz) = obj.rotation
                                                .to_euler(glam::EulerRot::XYZ);
                                            obj.rotation = glam::Quat::from_euler(
                                                glam::EulerRot::XYZ,
                                                (v as f32).to_radians(), cy, cz,
                                            );
                                        }
                                    }
                                }},
                            );
                            build_slider_row(
                                __scope, &container, "Rotation Y", "\u{00b0}", rot_y_sig,
                                -180.0, 180.0, 0.5, 1,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        let sc = es.world.scene_mut();
                                        if let Some(obj) = sc.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            let (cx, _, cz) = obj.rotation
                                                .to_euler(glam::EulerRot::XYZ);
                                            obj.rotation = glam::Quat::from_euler(
                                                glam::EulerRot::XYZ,
                                                cx, (v as f32).to_radians(), cz,
                                            );
                                        }
                                    }
                                }},
                            );
                            build_slider_row(
                                __scope, &container, "Rotation Z", "\u{00b0}", rot_z_sig,
                                -180.0, 180.0, 0.5, 1,
                                { let es = es.clone(); move |v| {
                                    if let Ok(mut es) = es.lock() {
                                        let sc = es.world.scene_mut();
                                        if let Some(obj) = sc.objects.iter_mut()
                                            .find(|o| o.id as u64 == eid)
                                        {
                                            let (cx, cy, _) = obj.rotation
                                                .to_euler(glam::EulerRot::XYZ);
                                            obj.rotation = glam::Quat::from_euler(
                                                glam::EulerRot::XYZ,
                                                cx, cy, (v as f32).to_radians(),
                                            );
                                        }
                                    }
                                }},
                            );

                            // Scale X/Y/Z.
                            for (label, axis_idx, val) in [
                                ("Scale X", 0usize, scale.x),
                                ("Scale Y", 1usize, scale.y),
                                ("Scale Z", 2usize, scale.z),
                            ] {
                                let sig = Signal::new(val as f64);
                                build_slider_row(
                                    __scope, &container, label, "", sig,
                                    0.01, 50.0, 0.01, 2,
                                    { let es = es.clone(); move |v| {
                                        if let Ok(mut es) = es.lock() {
                                            let sc = es.world.scene_mut();
                                            if let Some(obj) = sc.objects.iter_mut()
                                                .find(|o| o.id as u64 == eid)
                                            {
                                                obj.scale[axis_idx] = v as f32;
                                            }
                                        }
                                    }},
                                );
                            }

                            // Re-voxelize button (only for voxelized objects with non-uniform scale).
                            if show_revoxelize {
                                let btn_row = __scope.create_element("div");
                                btn_row.set_attribute("style", "padding: 6px 8px;");
                                let btn = __scope.create_element("button");
                                btn.set_attribute("style",
                                    "width:100%; padding:4px 8px; background:#553322; \
                                     color:#ffcc99; border:1px solid #774433; \
                                     border-radius:3px; cursor:pointer; font-size:12px;");
                                btn.append_child(
                                    &__scope.create_text("Re-voxelize (bake scale)"),
                                );
                                let hid = __scope.register_handler({
                                    let es = es.clone();
                                    let rev = revision;
                                    move || {
                                        if let Ok(mut es) = es.lock() {
                                            es.pending_revoxelize = Some(eid as u32);
                                        }
                                        rev.bump();
                                    }
                                });
                                btn.set_attribute("data-rid", &hid.to_string());
                                btn_row.append_child(&btn);
                                container.append_child(&btn_row);
                            }
                        }
                    }
                }
                Some(SelectedEntity::Light(lid)) => {
                    // Read light data.
                    let light_data = es.lock().ok().and_then(|es_lock| {
                        es_lock.light_editor.get_light(lid).map(|light| {
                            (light.intensity, light.range, light.light_type, light.position)
                        })
                    });
                    if let Some((intensity, range, light_type, position)) = light_data {
                        let type_name = match light_type {
                            crate::light_editor::EditorLightType::Point => "Point Light",
                            crate::light_editor::EditorLightType::Spot => "Spot Light",
                        };
                        let hdr = __scope.create_element("div");
                        hdr.set_attribute("style", SECTION_STYLE);
                        hdr.append_child(&__scope.create_text(type_name));
                        container.append_child(&hdr);

                        // Position sliders.
                        let lpos_x = Signal::new(position.x as f64);
                        let lpos_y = Signal::new(position.y as f64);
                        let lpos_z = Signal::new(position.z as f64);

                        let lid_cap = lid;
                        build_slider_row(
                            __scope, &container, "Position X", "", lpos_x,
                            -500.0, 500.0, 0.01, 2,
                            { let es = es.clone(); move |v| {
                                if let Ok(mut es) = es.lock() {
                                    if let Some(l) = es.light_editor.get_light_mut(lid_cap) {
                                        l.position.x = v as f32;
                                    }
                                    es.light_editor.mark_dirty();
                                }
                            }},
                        );
                        build_slider_row(
                            __scope, &container, "Position Y", "", lpos_y,
                            -500.0, 500.0, 0.01, 2,
                            { let es = es.clone(); move |v| {
                                if let Ok(mut es) = es.lock() {
                                    if let Some(l) = es.light_editor.get_light_mut(lid_cap) {
                                        l.position.y = v as f32;
                                    }
                                    es.light_editor.mark_dirty();
                                }
                            }},
                        );
                        build_slider_row(
                            __scope, &container, "Position Z", "", lpos_z,
                            -500.0, 500.0, 0.01, 2,
                            { let es = es.clone(); move |v| {
                                if let Ok(mut es) = es.lock() {
                                    if let Some(l) = es.light_editor.get_light_mut(lid_cap) {
                                        l.position.z = v as f32;
                                    }
                                    es.light_editor.mark_dirty();
                                }
                            }},
                        );

                        let div = __scope.create_element("div");
                        div.set_attribute("style", DIVIDER_STYLE);
                        container.append_child(&div);

                        // Intensity and range sliders.
                        let light_intensity_signal = Signal::new(intensity as f64);
                        let light_range_signal = Signal::new(range as f64);

                        build_slider_row(
                            __scope, &container, "Intensity", "", light_intensity_signal,
                            0.0, 50.0, 0.1, 1,
                            { let es = es.clone(); move |v| {
                                if let Ok(mut es) = es.lock() {
                                    if let Some(l) = es.light_editor.get_light_mut(lid_cap) {
                                        l.intensity = v as f32;
                                    }
                                    es.light_editor.mark_dirty();
                                }
                            }},
                        );
                        build_slider_row(
                            __scope, &container, "Range", "m", light_range_signal,
                            0.1, 100.0, 0.5, 1,
                            { let es = es.clone(); move |v| {
                                if let Ok(mut es) = es.lock() {
                                    if let Some(l) = es.light_editor.get_light_mut(lid_cap) {
                                        l.range = v as f32;
                                    }
                                    es.light_editor.mark_dirty();
                                }
                            }},
                        );
                    }
                }
                Some(SelectedEntity::Scene) => {
                    let name = es.lock().ok()
                        .map(|e| e.world.scene().name.clone())
                        .unwrap_or_else(|| "Scene".to_string());
                    let count = es.lock().map(|e| e.world.scene().objects.len()).unwrap_or(0);

                    let name_row = __scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    name_row.append_child(&__scope.create_text(&name));
                    container.append_child(&name_row);

                    let detail = __scope.create_element("div");
                    detail.set_attribute("style", VALUE_STYLE);
                    detail.append_child(
                        &__scope.create_text(&format!("{count} objects")),
                    );
                    container.append_child(&detail);
                }
                Some(SelectedEntity::Project) => {
                    let hdr = __scope.create_element("div");
                    hdr.set_attribute("style", SECTION_STYLE);
                    hdr.append_child(&__scope.create_text("Project"));
                    container.append_child(&hdr);
                }
                _ => {
                    let msg = __scope.create_element("div");
                    msg.set_attribute(
                        "style",
                        &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                    );
                    msg.append_child(&__scope.create_text("No object selected"));
                    container.append_child(&msg);
                }
            }
        }

        container
    });

    root
}

// ── Status bar ──────────────────────────────────────────────────────────────

/// Reactive status bar showing object count, FPS, selected object, and mode.
#[component]
pub fn StatusBar() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let _shared_state = use_context::<Arc<Mutex<SharedState>>>();
    let revision = use_context::<UiRevision>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        &format!(
            "display:flex;align-items:center;height:25px;\
            background:var(--rinch-color-dark-9);border-top:1px solid var(--rinch-color-border);\
            padding:0 12px;gap:16px;\
            font-size:11px;color:var(--rinch-color-dimmed);"
        ),
    );

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking — only rebuild when these specific values change.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        let _ = ui.gizmo_mode.get();
        let _ = ui.debug_mode.get();
        let _ = ui.show_grid.get();
        let _ = ui.object_count.get();
        // Legacy fallback.
        revision.track();

        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display:flex;align-items:center;width:100%;gap:16px;",
        );

        let (obj_count, mode_name, selected_name, debug_name, gizmo_mode_name) = {
            let es = es.lock().unwrap();
            let sel_name = es.selected_entity.as_ref().map(|sel| match sel {
                SelectedEntity::Object(eid) => {
                    es.world.scene().objects.iter()
                        .find(|o| o.id as u64 == *eid)
                        .map(|o| o.name.clone())
                        .unwrap_or_else(|| format!("Object {eid}"))
                }
                SelectedEntity::Light(lid) => {
                    es.light_editor.get_light(*lid)
                        .map(|l| match l.light_type {
                            crate::light_editor::EditorLightType::Point => format!("Point Light {lid}"),
                            crate::light_editor::EditorLightType::Spot => format!("Spot Light {lid}"),
                        })
                        .unwrap_or_else(|| format!("Light {lid}"))
                }
                SelectedEntity::Camera => "Camera".to_string(),
                SelectedEntity::Scene => "Scene".to_string(),
                SelectedEntity::Project => "Project".to_string(),
            });
            let gizmo_name = match es.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => "Translate (W)",
                crate::gizmo::GizmoMode::Rotate => "Rotate (E)",
                crate::gizmo::GizmoMode::Scale => "Scale (R)",
            };
            (
                es.world.scene().objects.len(),
                es.mode.name().to_string(),
                sel_name,
                es.debug_mode_name().to_string(),
                gizmo_name.to_string(),
            )
        };
        let obj_div = __scope.create_element("div");
        obj_div.append_child(&__scope.create_text(&format!("{obj_count} objects")));
        container.append_child(&obj_div);

        // FPS: own reactive text node tracking UiSignals.fps (not revision).
        let fps_div = __scope.create_element("div");
        let fps_text = __scope.create_text("-- fps");
        fps_div.append_child(&fps_text);
        container.append_child(&fps_div);
        {
            let fps_text = fps_text.clone();
            Effect::new(move || {
                let ms = ui.fps.get();
                let label = if ms > 0.1 {
                    format!("{:.0} fps", 1000.0 / ms)
                } else {
                    "-- fps".to_string()
                };
                fps_text.set_text(&label);
            });
        }

        // Selected object name.
        if let Some(name) = &selected_name {
            let sel_div = __scope.create_element("div");
            sel_div.set_attribute("style", "color:var(--rinch-primary-color);");
            sel_div.append_child(&__scope.create_text(name));
            container.append_child(&sel_div);
        }

        // Debug mode indicator.
        if !debug_name.is_empty() {
            let dbg_div = __scope.create_element("div");
            dbg_div.set_attribute("style", "color:var(--rinch-color-yellow-5, #fcc419);");
            dbg_div.append_child(&__scope.create_text(&debug_name));
            container.append_child(&dbg_div);
        }

        let spacer = __scope.create_element("div");
        spacer.set_attribute("style", "flex:1;");
        container.append_child(&spacer);

        // Gizmo mode indicator.
        {
            let gizmo_div = __scope.create_element("div");
            gizmo_div.set_attribute("style", "color:var(--rinch-color-dimmed);");
            gizmo_div.append_child(&__scope.create_text(&gizmo_mode_name));
            container.append_child(&gizmo_div);
        }

        // Show active tool mode (only when Sculpt/Paint active).
        if !mode_name.is_empty() {
            let mode_div = __scope.create_element("div");
            mode_div.set_attribute("style", "color:var(--rinch-primary-color);");
            mode_div.append_child(&__scope.create_text(&format!("{mode_name} mode")));
            container.append_child(&mode_div);
        }

        // Grid indicator.
        {
            let show_grid = es.lock().map(|e| e.show_grid).unwrap_or(false);
            if show_grid {
                let grid_div = __scope.create_element("div");
                grid_div.set_attribute("style", "color:var(--rinch-color-dimmed);");
                grid_div.append_child(&__scope.create_text("Grid"));
                container.append_child(&grid_div);
            }
        }

        // F1 shortcut reference overlay.
        {
            let show_shortcuts = es.lock().map(|e| e.show_shortcuts).unwrap_or(false);
            if show_shortcuts {
                let overlay = __scope.create_element("div");
                overlay.set_attribute("style",
                    "position:fixed;bottom:30px;right:310px;z-index:100;\
                     background:var(--rinch-color-dark-9);border:1px solid var(--rinch-color-border);\
                     border-radius:6px;padding:12px 16px;font-size:11px;\
                     color:var(--rinch-color-text);line-height:1.8;white-space:pre;",
                );
                overlay.append_child(&__scope.create_text(
                    "Keyboard Shortcuts  (F1 to close)\n\
                     ─────────────────────────────────\n\
                     G/R/L      Grab / Rotate / scaLe\n\
                     B / N      Sculpt / Paint mode\n\
                     G          Toggle grid\n\
                     F3         Cycle debug mode\n\
                     Delete     Delete selected\n\
                     Ctrl+D     Duplicate selected\n\
                     Ctrl+S     Save scene\n\
                     Ctrl+O     Open scene\n\
                     RMB+WASD   Fly camera\n\
                     Scroll     Zoom (orbit)\n\
                     MMB        Pan\n\
                     F1         This reference",
                ));
                container.append_child(&overlay);
            }
        }

        container
    });

    root
}
