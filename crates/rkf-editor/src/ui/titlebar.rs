//! Combined titlebar component — menus, tool buttons, window controls.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, UiSignals};
use crate::layout::state::{LayoutBacking, LayoutState};
use crate::layout::ContainerKind;
use crate::{CommandSender, SnapshotReader};

// ── Window control button ───────────────────────────────────────────────────

#[component]
fn WindowControl(label: String, extra_class: String, onclick: Option<Callback>) -> NodeHandle {
    let cls = if extra_class.is_empty() {
        "rinch-borderlesswindow__control".to_string()
    } else {
        format!("rinch-borderlesswindow__control {extra_class}")
    };
    let el = __scope.create_element("div");
    el.set_attribute("class", &cls);
    el.set_attribute(
        "style",
        "width:46px;height:36px;display:flex;align-items:center;\
         justify-content:center;cursor:pointer;\
         color:var(--rinch-color-dimmed);font-size:14px;",
    );
    el.append_child(&__scope.create_text(&label));
    if let Some(cb) = onclick {
        let hid = __scope.register_handler(move || cb.invoke());
        el.set_attribute("data-rid", &hid.to_string());
    }
    el
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
    let cmd = use_context::<CommandSender>();
    let snapshot = use_context::<SnapshotReader>();
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

    // Titlebar drag — rinch reads `data-drag-window` and emits AppAction::DragWindow.
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
                ui.bump_scene();
                ui.selection.set(None);
            }
        }))
        .separator()
        .item(MenuItem::new("Open Scene").shortcut("Ctrl+O").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::OpenScene { path: String::new() });
                ui.bump_scene();
                ui.selection.set(None);
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
                ui.bump_scene();
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
                ui.bump_scene();
            }
        }))
        .item(MenuItem::new("Duplicate").shortcut("Ctrl+D").on_click({
            let cmd = cmd.clone();
            move || {
                let _ = cmd.0.send(EditorCommand::DuplicateSelected);
                ui.bump_scene();
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
        let snap = snapshot.clone();
        let cmd2 = cmd.clone();
        let tool_container = __scope.create_element("div");
        tool_container.set_attribute(
            "style",
            "display:flex;align-items:center;gap:2px;",
        );

        rinch::core::reactive_component_dom(__scope, &tool_container, move |__scope| {
            let _ = ui.editor_mode.get();

            let current_mode = snap.0.load().mode;

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

                let cmd = cmd2.clone();
                let snap = snap.clone();
                let handler_id = __scope.register_handler(move || {
                    let current = snap.0.load().mode;
                    let new_mode = if current == mode {
                        EditorMode::Default
                    } else {
                        mode
                    };
                    let _ = cmd.0.send(EditorCommand::SetEditorMode { mode: new_mode });
                    ui.editor_mode.set(new_mode);
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
