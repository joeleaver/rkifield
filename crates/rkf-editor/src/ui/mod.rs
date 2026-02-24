//! Editor UI root component.
//!
//! Layout: titlebar → (left panel | viewport | right panel) → status bar.
//! EditorState is shared via rinch context (create_context in main.rs).

pub mod properties_panel;
pub mod scene_tree_panel;

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_state::{EditorMode, EditorState, SelectedEntity, UiRevision};
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

// ── Root component ──────────────────────────────────────────────────────────

/// Root editor UI component.
///
/// Layout:
/// ```text
/// ┌──────────────────────────────────────────────────────────────────┐
/// │ RkiField   File Edit View  ·  Sculpt Paint        [─] [□] [×]  │  (titlebar 36px)
/// ├────────┬──────────────────────────────────────────┬──────────────┤
/// │  Left  │           GameViewport                   │    Right     │
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
                        display:flex;flex-direction:column;overflow:hidden;")},
                    SceneTreePanel {}
                }

                // Center viewport
                GameViewport { name: "main", style: "flex:1;" }

                // Right panel — mode-dependent
                div {
                    style: {format!("{RIGHT_PANEL_WIDTH}{PANEL_BG}{PANEL_BORDER}\
                        border-left:1px solid var(--rinch-color-border);\
                        display:flex;flex-direction:column;overflow:hidden;")},
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

// ── Titlebar ────────────────────────────────────────────────────────────────

/// Combined titlebar with app title, menus, tool buttons, and window controls.
///
/// Replaces the separate MenuBar + Toolbar with a single 36px row in a
/// frameless window. Uses rinch's `rinch-app-menu-*` CSS classes for themed
/// dropdown rendering. The titlebar background acts as a drag handle — clicks
/// on empty areas set `pending_drag` which the winit event loop consumes to
/// call `window.drag_window()`.
#[component]
pub fn TitleBar() -> NodeHandle {
    use rinch::menu::{Menu, MenuItem, MenuEntryRef};

    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();

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

    // Titlebar drag handler — clicks on empty background trigger window drag.
    // Child elements (menus, buttons, controls) have their own data-rid handlers
    // which fire first due to rinch's deepest-first hit testing.
    {
        let es = editor_state.clone();
        let drag_handler = __scope.register_handler(move || {
            if let Ok(mut s) = es.lock() {
                s.pending_drag = true;
            }
        });
        bar.set_attribute("data-rid", &drag_handler.to_string());
    }

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
        .item(MenuItem::new("Quit").shortcut("Esc").on_click({
            let es = es.clone();
            let rev = rev;
            move || {
                if let Ok(mut s) = es.lock() { s.wants_exit = true; }
                rev.bump();
            }
        }));

    let edit_menu = Menu::new()
        .item(MenuItem::new("Undo").shortcut("Ctrl+Z").on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() { s.undo.undo(); }
            }
        }))
        .item(MenuItem::new("Redo").shortcut("Ctrl+Y").on_click({
            let es = es.clone();
            move || {
                if let Ok(mut s) = es.lock() { s.undo.redo(); }
            }
        }));

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
                rev.bump();
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
        let es = editor_state.clone();
        let tool_container = __scope.create_element("div");
        tool_container.set_attribute(
            "style",
            "display:flex;align-items:center;gap:2px;",
        );

        rinch::core::reactive_component_dom(__scope, &tool_container, move |__scope| {
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
                    if let Ok(mut state) = es.lock() {
                        if state.mode == mode {
                            state.mode = EditorMode::Default;
                        } else {
                            state.mode = mode;
                        }
                    }
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
        let es = editor_state.clone();
        let hid = __scope.register_handler(move || {
            if let Ok(mut s) = es.lock() { s.pending_minimize = true; }
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
        let es = editor_state.clone();
        let hid = __scope.register_handler(move || {
            if let Ok(mut s) = es.lock() { s.pending_maximize = true; }
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
        let es = editor_state.clone();
        let rev = revision;
        let hid = __scope.register_handler(move || {
            if let Ok(mut s) = es.lock() { s.wants_exit = true; }
            rev.bump();
        });
        close_btn.set_attribute("data-rid", &hid.to_string());
    }
    controls.append_child(&close_btn);

    bar.append_child(&controls);

    wrapper.append_child(&bar);
    wrapper
}

// ── Mode-dependent right panel ──────────────────────────────────────────────

/// Right panel — always shows properties of the selected object.
///
/// When a Sculpt/Paint tool is active, shows brush settings above the
/// properties section (placeholder for now).
#[component]
pub fn RightPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
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

                let msg = __scope.create_element("div");
                msg.set_attribute(
                    "style",
                    &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                );
                msg.append_child(&__scope.create_text("Brush settings coming soon"));
                container.append_child(&msg);

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    "height:1px;background:var(--rinch-color-border);margin:8px 0;",
                );
                container.append_child(&div);
            }
            EditorMode::Paint => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Paint Brush"));
                container.append_child(&header);

                let msg = __scope.create_element("div");
                msg.set_attribute(
                    "style",
                    &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                );
                msg.append_child(&__scope.create_text("Paint settings coming soon"));
                container.append_child(&msg);

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    "height:1px;background:var(--rinch-color-border);margin:8px 0;",
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

        // Gather display info based on selected entity.
        let info = selected_entity.and_then(|sel| {
            let es = es.lock().ok()?;
            match sel {
                SelectedEntity::Object(eid) => {
                    let node = es.scene_tree.find_node(eid)?;
                    Some((node.name.clone(), format!("Entity ID: {eid}"), node.children.len()))
                }
                SelectedEntity::Light(lid) => {
                    let light = es.light_editor.get_light(lid)?;
                    let type_name = match light.light_type {
                        crate::light_editor::EditorLightType::Point => "Point Light",
                        crate::light_editor::EditorLightType::Spot => "Spot Light",
                        crate::light_editor::EditorLightType::Directional => "Directional Light",
                    };
                    Some((
                        type_name.to_string(),
                        format!("Intensity: {:.2}  Range: {:.1}", light.intensity, light.range),
                        0,
                    ))
                }
                SelectedEntity::Camera => {
                    let pos = es.editor_camera.position;
                    Some((
                        "Camera".to_string(),
                        format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z),
                        0,
                    ))
                }
                SelectedEntity::Scene => {
                    let name = es.v2_scene.as_ref()
                        .map(|s| s.name.clone())
                        .unwrap_or_else(|| "Scene".to_string());
                    Some((name, format!("{} objects", es.scene_tree.roots.len()), 0))
                }
                SelectedEntity::Project => {
                    Some(("Project".to_string(), String::new(), 0))
                }
            }
        });

        if let Some((name, detail, child_count)) = info {
            let name_row = __scope.create_element("div");
            name_row.set_attribute("style", SECTION_STYLE);
            name_row.append_child(&__scope.create_text(&name));
            container.append_child(&name_row);

            if !detail.is_empty() {
                let detail_row = __scope.create_element("div");
                detail_row.set_attribute("style", VALUE_STYLE);
                detail_row.append_child(&__scope.create_text(&detail));
                container.append_child(&detail_row);
            }

            if child_count > 0 {
                let cr = __scope.create_element("div");
                cr.set_attribute("style", VALUE_STYLE);
                cr.append_child(
                    &__scope.create_text(&format!("Children: {child_count}")),
                );
                container.append_child(&cr);
            }
        } else {
            let msg = __scope.create_element("div");
            msg.set_attribute(
                "style",
                &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
            );
            msg.append_child(&__scope.create_text("No object selected"));
            container.append_child(&msg);
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
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();
    let revision = use_context::<UiRevision>();

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
    let ss = shared_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        revision.track();

        let container = __scope.create_element("div");
        container.set_attribute(
            "style",
            "display:flex;align-items:center;width:100%;gap:16px;",
        );

        let (obj_count, mode_name, selected_name, debug_name) = {
            let es = es.lock().unwrap();
            let sel_name = es.selected_entity.as_ref().map(|sel| match sel {
                SelectedEntity::Object(eid) => {
                    es.scene_tree.find_node(*eid)
                        .map(|n| n.name.clone())
                        .unwrap_or_else(|| format!("Object {eid}"))
                }
                SelectedEntity::Light(lid) => {
                    es.light_editor.get_light(*lid)
                        .map(|l| match l.light_type {
                            crate::light_editor::EditorLightType::Point => format!("Point Light {lid}"),
                            crate::light_editor::EditorLightType::Spot => format!("Spot Light {lid}"),
                            crate::light_editor::EditorLightType::Directional => format!("Dir Light {lid}"),
                        })
                        .unwrap_or_else(|| format!("Light {lid}"))
                }
                SelectedEntity::Camera => "Camera".to_string(),
                SelectedEntity::Scene => "Scene".to_string(),
                SelectedEntity::Project => "Project".to_string(),
            });
            (
                es.scene_tree.roots.len(),
                es.mode.name().to_string(),
                sel_name,
                es.debug_mode_name().to_string(),
            )
        };
        let frame_time_ms = ss.lock().map(|s| s.frame_time_ms).unwrap_or(0.0);
        let fps = if frame_time_ms > 0.1 {
            format!("{:.0} fps", 1000.0 / frame_time_ms)
        } else {
            "-- fps".to_string()
        };

        let obj_div = __scope.create_element("div");
        obj_div.append_child(&__scope.create_text(&format!("{obj_count} objects")));
        container.append_child(&obj_div);

        let fps_div = __scope.create_element("div");
        fps_div.append_child(&__scope.create_text(&fps));
        container.append_child(&fps_div);

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

        // Show active tool mode (only when Sculpt/Paint active).
        if !mode_name.is_empty() {
            let mode_div = __scope.create_element("div");
            mode_div.set_attribute("style", "color:var(--rinch-primary-color);");
            mode_div.append_child(&__scope.create_text(&format!("{mode_name} mode")));
            container.append_child(&mode_div);
        }

        container
    });

    root
}
