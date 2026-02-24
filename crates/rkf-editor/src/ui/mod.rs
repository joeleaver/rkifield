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
    row.append_child(&slider.render(scope, &[]));
    container.append_child(&row);
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

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    // Persistent signals for camera sliders (survive reactive rebuilds).
    let fov_signal: Signal<f64> = Signal::new(70.0);
    let speed_signal: Signal<f64> = Signal::new(5.0);
    let near_signal: Signal<f64> = Signal::new(0.1);
    let far_signal: Signal<f64> = Signal::new(1000.0);

    // Persistent signals for environment sliders.
    let sun_azimuth_signal: Signal<f64> = Signal::new(0.0);
    let sun_elevation_signal: Signal<f64> = Signal::new(45.0);
    let sun_intensity_signal: Signal<f64> = Signal::new(3.0);
    let fog_density_signal: Signal<f64> = Signal::new(0.02);
    let fog_height_falloff_signal: Signal<f64> = Signal::new(0.1);
    let bloom_intensity_signal: Signal<f64> = Signal::new(0.3);
    let bloom_threshold_signal: Signal<f64> = Signal::new(1.0);
    let exposure_signal: Signal<f64> = Signal::new(1.0);
    // Brush settings signals.
    let brush_radius_signal: Signal<f64> = Signal::new(1.0);
    let brush_strength_signal: Signal<f64> = Signal::new(0.5);
    let brush_falloff_signal: Signal<f64> = Signal::new(0.5);
    // Extended post-processing signals.
    let sharpen_signal: Signal<f64> = Signal::new(0.5);
    let dof_focus_dist_signal: Signal<f64> = Signal::new(2.0);
    let dof_focus_range_signal: Signal<f64> = Signal::new(3.0);
    let dof_max_coc_signal: Signal<f64> = Signal::new(8.0);
    let motion_blur_signal: Signal<f64> = Signal::new(1.0);
    let vignette_signal: Signal<f64> = Signal::new(0.0);
    let grain_signal: Signal<f64> = Signal::new(0.0);
    let chromatic_signal: Signal<f64> = Signal::new(0.0);

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

                // Sync brush signals.
                {
                    let es_lock = es.lock().unwrap();
                    brush_radius_signal.set(es_lock.sculpt.current_settings.radius as f64);
                    brush_strength_signal.set(es_lock.sculpt.current_settings.strength as f64);
                    brush_falloff_signal.set(es_lock.sculpt.current_settings.falloff as f64);
                }

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

                build_slider_row(
                    __scope, &container, "Radius", "", brush_radius_signal,
                    0.1, 10.0, 0.1, 1,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.sculpt.set_radius(v as f32);
                        }
                    }},
                );
                build_slider_row(
                    __scope, &container, "Strength", "", brush_strength_signal,
                    0.0, 1.0, 0.01, 2,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.sculpt.set_strength(v as f32);
                        }
                    }},
                );
                build_slider_row(
                    __scope, &container, "Falloff", "", brush_falloff_signal,
                    0.0, 1.0, 0.01, 2,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.sculpt.current_settings.falloff = v as f32;
                        }
                    }},
                );

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

                // Sync brush signals (paint uses the same radius/strength/falloff).
                {
                    let es_lock = es.lock().unwrap();
                    brush_radius_signal.set(es_lock.paint.current_settings.radius as f64);
                    brush_strength_signal.set(es_lock.paint.current_settings.strength as f64);
                    brush_falloff_signal.set(es_lock.paint.current_settings.falloff as f64);
                }

                build_slider_row(
                    __scope, &container, "Radius", "", brush_radius_signal,
                    0.1, 10.0, 0.1, 1,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.paint.current_settings.radius = v as f32;
                        }
                    }},
                );
                build_slider_row(
                    __scope, &container, "Strength", "", brush_strength_signal,
                    0.0, 1.0, 0.01, 2,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.paint.current_settings.strength = v as f32;
                        }
                    }},
                );
                build_slider_row(
                    __scope, &container, "Falloff", "", brush_falloff_signal,
                    0.0, 1.0, 0.01, 2,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.paint.current_settings.falloff = v as f32;
                        }
                    }},
                );

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

        // ── Camera-specific property editing with sliders ──
        if let Some(SelectedEntity::Camera) = selected_entity {
            // Sync persistent signals from current editor state.
            let pos = {
                let es = es.lock().unwrap();
                fov_signal.set(es.editor_camera.fov_y.to_degrees() as f64);
                speed_signal.set(es.editor_camera.fly_speed as f64);
                near_signal.set(es.editor_camera.near as f64);
                far_signal.set(es.editor_camera.far as f64);
                es.editor_camera.position
            };

            let name_row = __scope.create_element("div");
            name_row.set_attribute("style", SECTION_STYLE);
            name_row.append_child(&__scope.create_text("Camera"));
            container.append_child(&name_row);

            build_slider_row(
                __scope, &container, "FOV", "\u{00b0}", fov_signal,
                30.0, 120.0, 1.0, 0,
                { let es = es.clone(); move |v| {
                    if let Ok(mut es) = es.lock() { es.editor_camera.fov_y = (v as f32).to_radians(); }
                }},
            );
            build_slider_row(
                __scope, &container, "Fly Speed", "", speed_signal,
                0.5, 50.0, 0.5, 1,
                { let es = es.clone(); move |v| {
                    if let Ok(mut es) = es.lock() { es.editor_camera.fly_speed = v as f32; }
                }},
            );
            build_slider_row(
                __scope, &container, "Near Plane", "", near_signal,
                0.01, 10.0, 0.01, 2,
                { let es = es.clone(); move |v| {
                    if let Ok(mut es) = es.lock() { es.editor_camera.near = v as f32; }
                }},
            );
            build_slider_row(
                __scope, &container, "Far Plane", "", far_signal,
                100.0, 10000.0, 100.0, 0,
                { let es = es.clone(); move |v| {
                    if let Ok(mut es) = es.lock() { es.editor_camera.far = v as f32; }
                }},
            );

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
        } else {
            // ── Generic info display for non-camera entities ──
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
                    SelectedEntity::Camera => unreachable!(),
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
        }

        // ── Divider between properties and environment ──
        let div = __scope.create_element("div");
        div.set_attribute(
            "style",
            "height:1px;background:var(--rinch-color-border);margin:8px 0;",
        );
        container.append_child(&div);

        // ── Environment (always visible) ──

        // Sync environment signals from editor state.
        {
            let es = es.lock().unwrap();
            let atmo = &es.environment.atmosphere;
            // Convert sun_direction → azimuth/elevation.
            let d = atmo.sun_direction.normalize_or_zero();
            let elevation = d.y.asin().to_degrees();
            let azimuth = d.x.atan2(d.z).to_degrees().rem_euclid(360.0);
            sun_azimuth_signal.set(azimuth as f64);
            sun_elevation_signal.set(elevation as f64);
            sun_intensity_signal.set(atmo.sun_intensity as f64);

            fog_density_signal.set(es.environment.fog.density as f64);
            fog_height_falloff_signal.set(es.environment.fog.height_falloff as f64);

            let pp = &es.environment.post_process;
            bloom_intensity_signal.set(pp.bloom_intensity as f64);
            bloom_threshold_signal.set(pp.bloom_threshold as f64);
            exposure_signal.set(pp.exposure as f64);
            sharpen_signal.set(pp.sharpen_strength as f64);
            dof_focus_dist_signal.set(pp.dof_focus_distance as f64);
            dof_focus_range_signal.set(pp.dof_focus_range as f64);
            dof_max_coc_signal.set(pp.dof_max_coc as f64);
            motion_blur_signal.set(pp.motion_blur_intensity as f64);
            vignette_signal.set(pp.vignette_intensity as f64);
            grain_signal.set(pp.grain_intensity as f64);
            chromatic_signal.set(pp.chromatic_aberration as f64);
        }

        // ── Atmosphere section ──
        let atmo_header = __scope.create_element("div");
        atmo_header.set_attribute("style", LABEL_STYLE);
        atmo_header.append_child(&__scope.create_text("Atmosphere"));
        container.append_child(&atmo_header);

        build_slider_row(
            __scope, &container, "Sun Azimuth", "\u{00b0}", sun_azimuth_signal,
            0.0, 360.0, 1.0, 0,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    let az = (v as f32).to_radians();
                    let el = es.environment.atmosphere.sun_direction.y.asin();
                    let cos_el = el.cos();
                    es.environment.atmosphere.sun_direction =
                        glam::Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Sun Elevation", "\u{00b0}", sun_elevation_signal,
            -90.0, 90.0, 1.0, 0,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    let d = es.environment.atmosphere.sun_direction.normalize_or_zero();
                    let az = d.x.atan2(d.z);
                    let el = (v as f32).to_radians();
                    let cos_el = el.cos();
                    es.environment.atmosphere.sun_direction =
                        glam::Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Sun Intensity", "", sun_intensity_signal,
            0.0, 10.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.atmosphere.sun_intensity = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );

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

        // ── Fog section ──
        let fog_header = __scope.create_element("div");
        fog_header.set_attribute("style", LABEL_STYLE);
        fog_header.append_child(&__scope.create_text("Fog"));
        container.append_child(&fog_header);

        // Fog enable toggle (simple clickable label).
        {
            let fog_enabled = es.lock().map(|e| e.environment.fog.enabled).unwrap_or(false);
            let toggle_row = __scope.create_element("div");
            toggle_row.set_attribute(
                "style",
                "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                 cursor:pointer;user-select:none;",
            );
            let label = if fog_enabled { "Fog: ON" } else { "Fog: OFF" };
            toggle_row.append_child(&__scope.create_text(label));
            let es_toggle = es.clone();
            let rev = revision;
            let hid = __scope.register_handler(move || {
                if let Ok(mut es) = es_toggle.lock() {
                    es.environment.fog.enabled = !es.environment.fog.enabled;
                    es.environment.mark_dirty();
                }
                rev.bump();
            });
            toggle_row.set_attribute("data-rid", &hid.to_string());
            container.append_child(&toggle_row);
        }

        build_slider_row(
            __scope, &container, "Fog Density", "", fog_density_signal,
            0.0, 0.5, 0.001, 3,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.fog.density = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Height Falloff", "", fog_height_falloff_signal,
            0.0, 1.0, 0.01, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.fog.height_falloff = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );

        // ── Post-Processing section ──
        let pp_header = __scope.create_element("div");
        pp_header.set_attribute("style", LABEL_STYLE);
        pp_header.append_child(&__scope.create_text("Post-Processing"));
        container.append_child(&pp_header);

        // Bloom enable toggle.
        {
            let bloom_on = es.lock().map(|e| e.environment.post_process.bloom_enabled).unwrap_or(true);
            let toggle_row = __scope.create_element("div");
            toggle_row.set_attribute(
                "style",
                "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                 cursor:pointer;user-select:none;",
            );
            let label = if bloom_on { "Bloom: ON" } else { "Bloom: OFF" };
            toggle_row.append_child(&__scope.create_text(label));
            let es_toggle = es.clone();
            let rev = revision;
            let hid = __scope.register_handler(move || {
                if let Ok(mut es) = es_toggle.lock() {
                    es.environment.post_process.bloom_enabled = !es.environment.post_process.bloom_enabled;
                    es.environment.mark_dirty();
                }
                rev.bump();
            });
            toggle_row.set_attribute("data-rid", &hid.to_string());
            container.append_child(&toggle_row);
        }

        build_slider_row(
            __scope, &container, "Bloom Intensity", "", bloom_intensity_signal,
            0.0, 2.0, 0.01, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.bloom_intensity = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Bloom Threshold", "", bloom_threshold_signal,
            0.0, 5.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.bloom_threshold = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Exposure", "", exposure_signal,
            0.1, 10.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.exposure = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );

        // Tone map mode toggle (ACES / AgX).
        {
            let tm_mode = es.lock().map(|e| e.environment.post_process.tone_map_mode).unwrap_or(0);
            let toggle_row = __scope.create_element("div");
            toggle_row.set_attribute(
                "style",
                "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                 cursor:pointer;user-select:none;",
            );
            let label = if tm_mode == 0 { "Tone Map: ACES" } else { "Tone Map: AgX" };
            toggle_row.append_child(&__scope.create_text(label));
            let es_toggle = es.clone();
            let rev = revision;
            let hid = __scope.register_handler(move || {
                if let Ok(mut es) = es_toggle.lock() {
                    es.environment.post_process.tone_map_mode =
                        if es.environment.post_process.tone_map_mode == 0 { 1 } else { 0 };
                    es.environment.mark_dirty();
                }
                rev.bump();
            });
            toggle_row.set_attribute("data-rid", &hid.to_string());
            container.append_child(&toggle_row);
        }

        build_slider_row(
            __scope, &container, "Sharpen", "", sharpen_signal,
            0.0, 2.0, 0.05, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.sharpen_strength = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );

        // DoF enable toggle.
        {
            let dof_on = es.lock().map(|e| e.environment.post_process.dof_enabled).unwrap_or(false);
            let toggle_row = __scope.create_element("div");
            toggle_row.set_attribute(
                "style",
                "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                 cursor:pointer;user-select:none;",
            );
            let label = if dof_on { "DoF: ON" } else { "DoF: OFF" };
            toggle_row.append_child(&__scope.create_text(label));
            let es_toggle = es.clone();
            let rev = revision;
            let hid = __scope.register_handler(move || {
                if let Ok(mut es) = es_toggle.lock() {
                    es.environment.post_process.dof_enabled = !es.environment.post_process.dof_enabled;
                    es.environment.mark_dirty();
                }
                rev.bump();
            });
            toggle_row.set_attribute("data-rid", &hid.to_string());
            container.append_child(&toggle_row);
        }

        build_slider_row(
            __scope, &container, "Focus Distance", "", dof_focus_dist_signal,
            0.1, 50.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.dof_focus_distance = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Focus Range", "", dof_focus_range_signal,
            0.1, 20.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.dof_focus_range = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Max CoC", "px", dof_max_coc_signal,
            1.0, 32.0, 1.0, 0,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.dof_max_coc = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Motion Blur", "", motion_blur_signal,
            0.0, 3.0, 0.1, 1,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.motion_blur_intensity = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Vignette", "", vignette_signal,
            0.0, 1.0, 0.01, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.vignette_intensity = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Grain", "", grain_signal,
            0.0, 1.0, 0.01, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.grain_intensity = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );
        build_slider_row(
            __scope, &container, "Chromatic Ab.", "", chromatic_signal,
            0.0, 1.0, 0.01, 2,
            { let es = es.clone(); move |v| {
                if let Ok(mut es) = es.lock() {
                    es.environment.post_process.chromatic_aberration = v as f32;
                    es.environment.mark_dirty();
                }
            }},
        );

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

        let (obj_count, mode_name, selected_name, debug_name, gizmo_mode_name) = {
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
            let gizmo_name = match es.gizmo.mode {
                crate::gizmo::GizmoMode::Translate => "Translate (W)",
                crate::gizmo::GizmoMode::Rotate => "Rotate (E)",
                crate::gizmo::GizmoMode::Scale => "Scale (R)",
            };
            (
                es.scene_tree.roots.len(),
                es.mode.name().to_string(),
                sel_name,
                es.debug_mode_name().to_string(),
                gizmo_name.to_string(),
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

        container
    });

    root
}
