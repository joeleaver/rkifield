//! Combined titlebar component — menus, tool buttons, window controls.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, SliderSignals, UiSignals};
use crate::layout::state::{LayoutBacking, LayoutState};
use crate::layout::ContainerKind;
use crate::CommandSender;

// ── Window control button ───────────────────────────────────────────────────

#[component]
fn WindowControl(label: String, extra_class: String, onclick: Option<Callback>) -> NodeHandle {
    let cls = if extra_class.is_empty() {
        "rinch-borderlesswindow__control".to_string()
    } else {
        format!("rinch-borderlesswindow__control {extra_class}")
    };
    let label2 = label.clone();
    rsx! {
        div {
            class: {cls.clone()},
            style: "width:46px;height:36px;display:flex;align-items:center;\
                    justify-content:center;cursor:pointer;\
                    color:var(--rinch-color-dimmed);font-size:14px;",
            onclick: {
                let cb = onclick.clone();
                move || { if let Some(cb) = &cb { cb.invoke(); } }
            },
            {label2}
        }
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
    let ui = use_context::<UiSignals>();
    let _sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let tree_state = use_context::<UseTreeReturn>();
    let layout = use_context::<LayoutState>();
    let layout_backing = use_context::<LayoutBacking>();

    // -1 = all closed, 0 = File, 1 = Edit, 2 = View.
    let active_menu: Signal<i32> = Signal::new(-1);

    // Outer wrapper — absolute positioning, last child of root for z-ordering.
    let wrapper = __scope.create_element("div");
    wrapper.set_attribute(
        "style",
        "position:absolute;top:0;left:0;width:100%;z-index:200;",
    );

    // ── Click-outside overlay ──────────────────────────────────────────
    let overlay = rsx! {
        div {
            class: "rinch-app-menu-bar__overlay",
            style: {move || if active_menu.get() >= 0 { "" } else { "display:none;" }},
            onclick: move || active_menu.set(-1),
        }
    };
    wrapper.append_child(&overlay);

    // ── Titlebar row ───────────────────────────────────────────────────
    let bar = rsx! {
        div {
            class: {move || if active_menu.get() >= 0 {
                "rinch-app-menu-bar rinch-app-menu-bar--engaged"
            } else {
                "rinch-app-menu-bar"
            }},
            style: "position:relative;z-index:201;height:36px;background:var(--rinch-titlebar-bg);",
        }
    };

    // NOTE: data-drag-window must NOT be on the bar itself, because dropdown
    // menu entries are children of the bar.  The parent-walk in handle_click
    // checks data-rid first, then data-drag-window — but if a click lands on
    // the dropdown container (no data-rid), it would walk up to the bar and
    // start a window drag instead of closing the menu.  Apply data-drag-window
    // only to non-interactive regions (title, spacer).

    // ── App title "RkiField" ───────────────────────────────────────────
    let title_node = rsx! {
        div {
            style: "position:relative;overflow:hidden;padding:0 12px 0 10px;\
                    display:flex;align-items:center;",
            span {
                style: "font-size:12px;font-weight:600;color:var(--rinch-color-text);\
                        letter-spacing:0.5px;white-space:nowrap;",
                "RkiField"
            }
            span {
                style: "position:absolute;right:0;top:0;bottom:0;width:16px;\
                        background:linear-gradient(90deg, transparent, var(--rinch-titlebar-bg));",
            }
        }
    };
    title_node.set_attribute("data-drag-window", "1");
    bar.append_child(&title_node);

    // ── Build menu data ────────────────────────────────────────────────
    let es = editor_state.clone();
    let file_menu = Menu::new()
        .item(MenuItem::new("New Project").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::NewProject);
            }
        }))
        .item(MenuItem::new("Open Project").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::OpenProject { path: String::new() });
                ui.set_selection(None, &tree_state);
            }
        }))
        .separator()
        .item(MenuItem::new("Open Scene").shortcut("Ctrl+O").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::OpenScene { path: String::new() });
                ui.set_selection(None, &tree_state);
            }
        }))
        .item(MenuItem::new("Save").shortcut("Ctrl+S").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::SaveProject);
            }
        }))
        .item(MenuItem::new("Save Scene As").shortcut("Ctrl+Shift+S").on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() { s.pending_save_as = true; }
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
            let cmd = cmd.clone();
            let name = prim_name.to_string();
            move || {
                let _ = cmd.0.send(EditorCommand::SpawnPrimitive { name: name.clone() });
            }
        }));
    }

    let edit_menu = Menu::new()
        .item(MenuItem::new("Undo").shortcut("Ctrl+Z").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::Undo);
            }
        }))
        .item(MenuItem::new("Redo").shortcut("Ctrl+Y").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::Redo);
            }
        }))
        .separator()
        .item(MenuItem::new("Delete").shortcut("Del").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::DeleteSelected);
            }
        }))
        .item(MenuItem::new("Duplicate").shortcut("Ctrl+D").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::DuplicateSelected);
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
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::SetDebugMode { mode });
                ui.debug_mode.set(mode);
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
        ("Top",         0.0,                    1.39),  // ~80 deg
        ("Perspective",  0.0,                    0.3),
    ];
    for &(label, yaw, pitch) in cam_presets {
        view_menu = view_menu.item(MenuItem::new(label).on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() {
                    s.editor_camera.set_orbit_angles(yaw, pitch);
                }
            }
        }));
    }

    // Grid overlay toggle.
    view_menu = view_menu.separator();
    view_menu = view_menu.item(MenuItem::new("Toggle Grid").shortcut("G").on_click({
        let cmd = cmd.clone();
        move || {
            let _ = cmd.0.send(EditorCommand::ToggleGrid);
            ui.show_grid.update(|v| *v = !*v);
        }
    }));

    // Container visibility toggles.
    view_menu = view_menu.separator();
    let container_toggles: &[(&str, ContainerKind)] = &[
        ("Toggle Left Panel", ContainerKind::Left),
        ("Toggle Right Panel", ContainerKind::Right),
        ("Toggle Bottom Panel", ContainerKind::Bottom),
    ];
    for &(label, ck) in container_toggles {
        view_menu = view_menu.item(MenuItem::new(label).on_click({
            let backing = layout_backing.clone();
            move || {
                layout.toggle_container(&backing, ck);
            }
        }));
    }

    // Panels submenu — open/focus any panel by name.
    use crate::layout::PanelId;
    let mut panels_menu = Menu::new();
    for &panel in PanelId::ALL {
        panels_menu = panels_menu.item(MenuItem::new(panel.display_name()).on_click({
            let backing = layout_backing.clone();
            move || {
                layout.ensure_panel(&backing, panel);
            }
        }));
    }
    view_menu = view_menu.separator();
    view_menu = view_menu.submenu("Panels", panels_menu);

    // ── Build DOM for each top-level menu ──────────────────────────────
    let menus: &[(&str, &Menu)] = &[
        ("File", &file_menu),
        ("Edit", &edit_menu),
        ("View", &view_menu),
    ];

    for (idx, &(label, menu)) in menus.iter().enumerate() {
        let index = idx as i32;

        let item = rsx! {
            div {
                class: {move || if active_menu.get() == index {
                    "rinch-app-menu-item rinch-app-menu-item--opened"
                } else {
                    "rinch-app-menu-item"
                }},
            }
        };

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

        // IMPORTANT: Do NOT use the base `rinch-app-menu-item__dropdown` class
        // because it sets `visibility:hidden` which cascades to children and
        // blocks hit testing.  Instead, replicate its positioning + visual
        // styles inline and toggle display:none/block.
        let dropdown = __scope.create_element("div");
        {
            let dh = dropdown.clone();
            // Reactive style toggle — use scope-owned effect (not detached Effect::new)
            // so it's properly disposed with this component's lifecycle.
            __scope.create_effect(move || {
                if active_menu.get() == index {
                    dh.set_attribute("style",
                        "position:absolute;top:100%;left:0;min-width:220px;\
                         background:var(--rinch-color-body);\
                         border:1px solid var(--rinch-color-border,var(--rinch-color-gray-3));\
                         border-radius:var(--rinch-radius-md);\
                         box-shadow:0 4px 12px rgba(0,0,0,0.15);\
                         padding:4px;z-index:200;");
                } else {
                    dh.set_attribute("style", "display:none;");
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
                MenuEntryRef::Submenu { label, menu: submenu } => {
                    let sub_node = __scope.create_element("div");
                    sub_node.set_attribute("class", "rinch-app-menu-submenu");

                    let trigger = __scope.create_element("div");
                    trigger.set_attribute("class", "rinch-app-menu-submenu__trigger");

                    let lbl = __scope.create_element("span");
                    lbl.set_attribute("class", "rinch-app-menu-submenu__label");
                    lbl.append_child(&__scope.create_text(label));
                    trigger.append_child(&lbl);

                    let arrow = __scope.create_element("span");
                    arrow.set_attribute("class", "rinch-app-menu-submenu__arrow");
                    arrow.append_child(&__scope.create_text("\u{203A}"));
                    trigger.append_child(&arrow);
                    sub_node.append_child(&trigger);

                    // Avoid rinch-app-menu-submenu__dropdown class — its
                    // visibility:hidden cascades to children and blocks hit testing.
                    // Use inline styles matching the visual appearance instead.
                    let nested = __scope.create_element("div");
                    nested.set_attribute("style",
                        "position:absolute;left:100%;top:0;min-width:200px;\
                         background:var(--rinch-color-body);\
                         border:1px solid var(--rinch-color-border,var(--rinch-color-gray-3));\
                         border-radius:var(--rinch-radius-md);\
                         box-shadow:0 4px 12px rgba(0,0,0,0.15);\
                         padding:4px;z-index:200;");
                    for sub_entry in submenu.iter_entries() {
                        match sub_entry {
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

                                nested.append_child(&entry_node);
                            }
                            MenuEntryRef::Separator => {
                                let sep = __scope.create_element("div");
                                sep.set_attribute("class", "rinch-app-menu-separator");
                                nested.append_child(&sep);
                            }
                            _ => {}
                        }
                    }
                    sub_node.append_child(&nested);
                    dropdown.append_child(&sub_node);
                }
            }
        }

        item.append_child(&dropdown);
        bar.append_child(&item);
    }

    // ── Separator between menus and tool buttons ───────────────────────
    let separator = rsx! {
        div {
            style: "width:1px;height:14px;background:var(--rinch-color-border);margin:0 8px;",
        }
    };
    bar.append_child(&separator);

    // ── Tool buttons (Sculpt/Paint) ────────────────────────────────────
    {
        let tool_container = rsx! {
            div {
                style: "display:flex;align-items:center;gap:2px;",
            }
        };

        for &mode in EditorMode::TOOLS.iter() {
            let cmd = cmd.clone();
            let btn = rsx! {
                div {
                    style: {move || {
                        let is_active = ui.editor_mode.get() == mode;
                        if is_active {
                            "padding:2px 8px;cursor:pointer;border-radius:3px;\
                             background:var(--rinch-titlebar-active);\
                             font-size:11px;color:var(--rinch-color-text);user-select:none;\
                             border:1px solid var(--rinch-color-border);font-weight:600;"
                        } else {
                            "padding:2px 8px;cursor:pointer;border-radius:3px;\
                             background:transparent;\
                             font-size:11px;color:var(--rinch-color-dimmed);user-select:none;\
                             border:1px solid transparent;font-weight:400;"
                        }
                    }},
                    onclick: move || {
                        let current = ui.editor_mode.get();
                        let new_mode = if current == mode {
                            EditorMode::Default
                        } else {
                            mode
                        };
                        let _ = cmd.0.send(EditorCommand::SetEditorMode { mode: new_mode });
                        ui.editor_mode.set(new_mode);
                    },
                    {mode.name()}
                }
            };
            tool_container.append_child(&btn);
        }

        bar.append_child(&tool_container);
    }

    // ── Separator before play button ─────────────────────────────────────
    let play_sep = rsx! {
        div {
            style: "width:1px;height:14px;background:var(--rinch-color-border);margin:0 8px;",
        }
    };
    bar.append_child(&play_sep);

    // ── Play / Stop button ───────────────────────────────────────────────
    {
        let cmd = cmd.clone();
        let play_btn = rsx! {
            div {
                style: {move || {
                    let playing = ui.play_state.get();
                    let ready = ui.dylib_ready.get();
                    if playing {
                        "padding:2px 10px;cursor:pointer;border-radius:3px;\
                         background:#c0392b;font-size:11px;color:#fff;user-select:none;\
                         border:1px solid #a93226;font-weight:600;"
                    } else if !ready {
                        "padding:2px 10px;cursor:default;border-radius:3px;\
                         background:#555;font-size:11px;color:#999;user-select:none;\
                         border:1px solid #444;font-weight:600;opacity:0.6;\
                         pointer-events:none;"
                    } else {
                        "padding:2px 10px;cursor:pointer;border-radius:3px;\
                         background:#27ae60;font-size:11px;color:#fff;user-select:none;\
                         border:1px solid #1e8449;font-weight:600;"
                    }
                }},
                onclick: move || {
                    if !ui.dylib_ready.get() && !ui.play_state.get() {
                        return; // Ignore click when dylib not ready.
                    }
                    let playing = ui.play_state.get();
                    if playing {
                        let _ = cmd.0.send(EditorCommand::PlayStop);
                    } else {
                        let _ = cmd.0.send(EditorCommand::PlayStart);
                    }
                },
                {move || if ui.play_state.get() { "Stop" } else { "Play" }}
            }
        };
        bar.append_child(&play_btn);
    }

    // ── Spacer (draggable empty region) ─────────────────────────────────
    let spacer = rsx! {
        div { style: "flex:1;align-self:stretch;", }
    };
    spacer.set_attribute("data-drag-window", "1");
    bar.append_child(&spacer);

    // ── Window controls (rsx!) ──────────────────────────────────────────
    let controls_node = rsx! {
        div {
            style: "display:flex;align-items:center;height:100%;",
            WindowControl {
                label: "\u{2500}",
                onclick: minimize_current_window,
            }
            WindowControl {
                label: "\u{25a1}",
                onclick: toggle_maximize_current_window,
            }
            WindowControl {
                label: "\u{00d7}",
                extra_class: "rinch-borderlesswindow__control--close",
                onclick: close_current_window,
            }
        }
    };
    bar.append_child(&controls_node);

    wrapper.append_child(&bar);
    wrapper
}
