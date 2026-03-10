//! LayoutRoot component — zone-based panel layout with reactive signals.
//!
//! Architecture:
//! - `structure_rev` → rebuilds the container/zone tree (rare: zone add/remove, tab move)
//! - `tab_rev` → swaps zone content panels (tab switch, frequent)
//! - `zone_frac_rev` → updates zone flex CSS during splitter drags (high-freq, CSS-only)
//! - Pixel signals (`left_width`, `right_width`, `bottom_height`) → container CSS (high-freq)
//!
//! The material preview RenderSurface lives inside AssetPropertiesPanel.

use std::cell::RefCell;
use std::rc::Rc;

use rinch::prelude::*;
use crate::layout::panel_registry;
use crate::layout::state::{LayoutBacking, LayoutState};
use crate::layout::ContainerKind;

/// Shared collection of all drop overlay region handles.
/// Tab drag handlers iterate this to show/hide all overlays imperatively.
type OverlayRegions = Rc<RefCell<Vec<NodeHandle>>>;

/// Root layout component that renders the zone-based panel layout.
#[component]
pub fn LayoutRoot() -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();

    // Provide overlay_regions as context so ContainerZones can add to it.
    let overlay_regions: OverlayRegions = Rc::new(RefCell::new(Vec::new()));
    create_context(overlay_regions.clone());

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex:1;min-height:0;");

    rinch::core::for_each_dom_typed(
        __scope, &root,
        move || vec![layout.structure_rev.get()],
        |rev| format!("{rev}"),
        move |_rev, __scope| {
        let config = layout.read_config(&backing);

        // Clear overlay regions from previous render.
        overlay_regions.borrow_mut().clear();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex:1;min-height:0;width:100%;");

        // ── Left container ──
        if !config.left.collapsed && !config.left.zones.is_empty() {
            let left_div = rsx! {
                div {
                    style: {move || format!(
                        "flex-shrink:0;\
                         background:var(--rinch-color-dark-8);\
                         border-right:1px solid var(--rinch-color-border);\
                         display:flex;flex-direction:column;min-height:0;\
                         overflow:hidden;\
                         width:{:.0}px",
                        layout.left_width.get()
                    )},
                    ContainerZones { kind: ContainerKind::Left }
                }
            };
            container.append_child(&left_div);

            let splitter = ContainerSplitter { side: ContainerKind::Left, ..Default::default() };
            container.append_child(&rinch::core::untracked(|| splitter.render(__scope, &[])));
        }

        // ── Center column (viewport + bottom) ──
        {
            let center_col = __scope.create_element("div");
            center_col.set_attribute("style", "flex:1;display:flex;flex-direction:column;min-height:0;min-width:0;");

            if !config.center.collapsed && !config.center.zones.is_empty() {
                let center_div = rsx! {
                    div {
                        style: "flex:1;min-height:0;display:flex;flex-direction:column;",
                        ContainerZones { kind: ContainerKind::Center }
                    }
                };
                center_col.append_child(&center_div);
            } else {
                center_col.append_child(&rsx! {
                    div { style: "flex:1;min-height:0;" }
                });
            }

            if !config.bottom.collapsed && !config.bottom.zones.is_empty() {
                let splitter = ContainerSplitter { side: ContainerKind::Bottom, ..Default::default() };
                center_col.append_child(&rinch::core::untracked(|| splitter.render(__scope, &[])));

                let bottom_div = rsx! {
                    div {
                        style: {move || format!(
                            "flex-shrink:0;\
                             background:var(--rinch-color-dark-8);\
                             border-top:1px solid var(--rinch-color-border);\
                             display:flex;flex-direction:row;min-height:0;\
                             height:{:.0}px",
                            layout.bottom_height.get()
                        )},
                        ContainerZones { kind: ContainerKind::Bottom }
                    }
                };
                center_col.append_child(&bottom_div);
            }

            container.append_child(&center_col);
        }

        // ── Right container ──
        if !config.right.collapsed && !config.right.zones.is_empty() {
            let splitter = ContainerSplitter { side: ContainerKind::Right, ..Default::default() };
            container.append_child(&rinch::core::untracked(|| splitter.render(__scope, &[])));

            let right_div = rsx! {
                div {
                    style: {move || format!(
                        "flex-shrink:0;\
                         background:var(--rinch-color-dark-8);\
                         border-left:1px solid var(--rinch-color-border);\
                         display:flex;flex-direction:column;min-height:0;\
                         overflow:hidden;\
                         width:{:.0}px",
                        layout.right_width.get()
                    )},
                    ContainerZones { kind: ContainerKind::Right }
                }
            };
            container.append_child(&right_div);
        }

        container
    },
    );

    root
}

// ── Container zones component ─────────────────────────────────────────────

/// Renders all zones within a container, including tab bars, splitters,
/// content areas, and drop overlays.
///
/// Uses `use_context` for all dependencies — no function parameters needed
/// beyond `kind`.
#[component]
fn ContainerZones(
    kind: ContainerKind,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let overlay_regions = use_context::<OverlayRegions>();

    // Read config non-reactively — parent (LayoutRoot) already subscribes
    // to structure_rev and rebuilds us on structural changes.
    let config = rinch::core::untracked(|| layout.read_config(&backing));
    let container = match config.container(kind) {
        Some(c) => c.clone(),
        None => return __scope.create_text("").into(),
    };

    let zone_count = container.zones.len();
    let wrapper = __scope.create_element("div");
    wrapper.set_attribute("style", "display:contents;");

    for (zi, zone) in container.zones.iter().enumerate() {
        if zone.tabs.is_empty() {
            continue;
        }

        // Zone wrapper with reactive flex from zone_frac_rev.
        let zone_node = ZoneWrapper {
            kind,
            zone_idx: zi as u32,
            initial_fraction: zone.size_fraction,
            ..Default::default()
        };
        let zone_div = rinch::core::untracked(|| zone_node.render(__scope, &[]));

        // Tab bar.
        let show_tab_bar = zone.tabs.len() > 1 || kind != ContainerKind::Center;
        if show_tab_bar {
            let tab_tabs = zone.tabs.clone();
            let tab_backing = backing.clone();
            let tab_regions = overlay_regions.clone();
            let tab_bar_wrapper = __scope.create_element("div");
            tab_bar_wrapper.set_attribute("style", "flex-shrink:0;");
            rinch::core::for_each_dom_typed(
                __scope, &tab_bar_wrapper,
                move || vec![layout.tab_rev.get()],
                |rev| format!("{rev}"),
                move |_rev, __scope| {
                    let cfg = layout.read_config(&tab_backing);
                    let active = cfg
                        .container(kind)
                        .and_then(|c| c.zones.get(zi))
                        .map(|z| z.active_tab.min(z.tabs.len() - 1))
                        .unwrap_or(0);
                    build_tab_bar(
                        __scope,
                        &tab_tabs,
                        active,
                        kind,
                        zi,
                        layout,
                        tab_backing.clone(),
                        &tab_regions,
                    )
                },
            );
            zone_div.append_child(&tab_bar_wrapper);
        }

        // Zone content — subscribes to `tab_rev` for active tab changes.
        let content_div = __scope.create_element("div");
        let pe = if kind == ContainerKind::Center { "pointer-events:none;" } else { "" };
        content_div.set_attribute(
            "style",
            &format!("flex:1 1 0px;overflow-y:auto;position:relative;z-index:0;\
                      display:flex;flex-direction:column;{pe}"),
        );

        let content_backing = backing.clone();
        rinch::core::for_each_dom_typed(
            __scope, &content_div,
            move || vec![layout.tab_rev.get()],
            |rev| format!("{rev}"),
            move |_rev, __scope| {
                let cfg = layout.read_config(&content_backing);
                let active_panel = cfg
                    .container(kind)
                    .and_then(|c| c.zones.get(zi))
                    .map(|z| z.tabs[z.active_tab.min(z.tabs.len() - 1)])
                    .unwrap_or(crate::layout::PanelId::SceneView);

                panel_registry::render_panel(__scope, active_panel)
            },
        );

        // Drop edge overlay.
        if kind != ContainerKind::Center {
            build_drop_edge_overlay(__scope, &content_div, kind, zi, layout, backing.clone(), &overlay_regions);
        }

        zone_div.append_child(&content_div);
        wrapper.append_child(&zone_div);

        if zi < zone_count - 1 {
            let zone_splitter = ZoneSplitter { kind, zone_idx: zi as u32, ..Default::default() };
            wrapper.append_child(&rinch::core::untracked(|| zone_splitter.render(__scope, &[])));
        }
    }

    wrapper
}

// ── Zone wrapper component ─────────────────────────────────────────────────

/// Zone wrapper with reactive flex from `zone_frac_rev` signal.
///
/// The flex value updates on splitter drags without rebuilding the layout tree.
#[component]
fn ZoneWrapper(
    kind: ContainerKind,
    zone_idx: u32,
    initial_fraction: f32,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let horizontal = matches!(kind, ContainerKind::Bottom | ContainerKind::Center);
    let extra: &'static str = if horizontal { "min-width:0;height:100%;" } else { "min-height:0;" };
    let zi = zone_idx as usize;

    rsx! {
        div {
            style: {
                let backing = backing.clone();
                move || {
                    let _rev = layout.zone_frac_rev.get();
                    let fraction = rinch::core::untracked(|| {
                        let cfg = layout.read_config(&backing);
                        cfg.container(kind)
                            .and_then(|c| c.zones.get(zi))
                            .map(|z| z.size_fraction)
                            .unwrap_or(initial_fraction)
                    });
                    format!(
                        "flex:{fraction} 1 0px;\
                         display:flex;flex-direction:column;overflow:hidden;{extra}"
                    )
                }
            },
        }
    }
}

// ── Tab bar ────────────────────────────────────────────────────────────────

/// Build a tab bar for a zone.
///
/// Uses register_handler + data attributes for drag-and-drop, which can't
/// be expressed in rsx! (ondragstart/ondragenter/ondragleave/ondrop need
/// data-* attribute wiring, not rsx! event syntax).
fn build_tab_bar(
    scope: &mut RenderScope,
    tabs: &[crate::layout::PanelId],
    active_tab: usize,
    container_kind: ContainerKind,
    zone_idx: usize,
    layout: LayoutState,
    backing: LayoutBacking,
    overlay_regions: &OverlayRegions,
) -> NodeHandle {
    let tab_bar = scope.create_element("div");
    tab_bar.set_attribute(
        "style",
        "display:flex;flex-shrink:0;height:26px;\
         background:var(--rinch-color-dark-9);\
         border-bottom:1px solid var(--rinch-color-border);\
         overflow-x:auto;",
    );

    for (ti, &panel_id) in tabs.iter().enumerate() {
        let is_active = ti == active_tab;
        let tab = scope.create_element("div");

        let bg = if is_active {
            "background:var(--rinch-color-dark-8);"
        } else {
            ""
        };
        let border_bottom = if is_active {
            "border-bottom:2px solid var(--rinch-primary-color-9);"
        } else {
            "border-bottom:2px solid transparent;"
        };

        tab.set_attribute(
            "style",
            &format!(
                "padding:4px 10px;font-size:11px;cursor:pointer;\
                 color:var(--rinch-color-text);white-space:nowrap;\
                 user-select:none;{bg}{border_bottom}"
            ),
        );

        let label = scope.create_text(panel_id.display_name());
        tab.append_child(&label);

        tab.set_attribute("draggable", "true");

        // Click → switch active tab.
        let click_hid = scope.register_handler({
            let backing = backing.clone();
            move || {
                layout.set_active_tab(&backing, container_kind, zone_idx, ti);
            }
        });
        tab.set_attribute("data-rid", &click_hid.to_string());

        // Drag start → show all overlay regions, record drag data.
        let drag_start_hid = scope.register_handler({
            let regions = overlay_regions.clone();
            move || {
                layout.tab_drag.set(Some(crate::layout::state::TabDragData {
                    panel: panel_id,
                    source_container: container_kind,
                    source_zone: zone_idx,
                }));
                for region in regions.borrow().iter() {
                    region.set_style("display", "block");
                }
            }
        });
        tab.set_attribute("data-ondragstart", &drag_start_hid.to_string());

        // Drag move → track cursor position for ghost overlay.
        let drag_move_hid = scope.register_handler(move || {
            let ctx = get_click_context();
            layout.drag_cursor.set(Some((ctx.mouse_x, ctx.mouse_y)));
        });
        tab.set_attribute("data-ondragmove", &drag_move_hid.to_string());

        // Drag end → float panel if dropped outside all targets, then clean up.
        let drag_end_hid = scope.register_handler({
            let regions = overlay_regions.clone();
            let backing = backing.clone();
            move || {
                let drag_data = layout.tab_drag.get();
                let drop = layout.drop_target.get();
                let cursor = layout.drag_cursor.get();

                if drop.is_none() {
                    if let (Some(data), Some((cx, cy))) = (drag_data, cursor) {
                        layout.float_tab(&backing, data.panel, cx, cy, 300.0, 200.0);
                    }
                }

                layout.tab_drag.set(None);
                layout.drop_target.set(None);
                layout.drag_cursor.set(None);
                for region in regions.borrow().iter() {
                    region.set_style("display", "none");
                }
            }
        });
        tab.set_attribute("data-ondragend", &drag_end_hid.to_string());

        tab_bar.append_child(&tab);
    }

    // Tab bar drop indicator — visible "+" zone during drags.
    // Created via rsx! for reactive style; data-* attributes attached imperatively.
    let drop_indicator = {
        let my_target = crate::layout::state::DropTarget::Zone {
            container: container_kind,
            zone_idx,
        };
        let __scope = &mut *scope;
        rsx! {
            div {
                style: {move || {
                    let dragging = layout.tab_drag.get().is_some();
                    let display = if dragging { "block" } else { "none" };
                    let drop = layout.drop_target.get();
                    let (bg, border_color, color) = if drop == Some(my_target) {
                        ("rgba(59,130,246,0.2)", "rgba(59,130,246,0.6)", "rgba(59,130,246,0.9)")
                    } else {
                        ("rgba(59,130,246,0.05)", "rgba(59,130,246,0.4)", "rgba(59,130,246,0.6)")
                    };
                    format!(
                        "display:{display};padding:4px 10px;font-size:11px;\
                         color:{color};white-space:nowrap;\
                         border:1px dashed {border_color};\
                         background:{bg};\
                         border-radius:3px;margin:2px 4px;\
                         user-select:none;"
                    )
                }},
                "+"
            }
        }
    };

    let enter_hid = scope.register_handler({
        let ck = container_kind;
        let zi = zone_idx;
        move || {
            layout.drop_target.set(Some(crate::layout::state::DropTarget::Zone {
                container: ck,
                zone_idx: zi,
            }));
        }
    });
    drop_indicator.set_attribute("data-ondragenter", &enter_hid.to_string());

    let leave_hid = scope.register_handler(move || {
        layout.drop_target.set(None);
    });
    drop_indicator.set_attribute("data-ondragleave", &leave_hid.to_string());

    let drop_hid = scope.register_handler({
        let backing = backing.clone();
        let ck = container_kind;
        let zi = zone_idx;
        let regions = overlay_regions.clone();
        move || {
            if let Some(data) = layout.tab_drag.get() {
                layout.move_tab(&backing, data.panel, ck, zi);
            }
            layout.tab_drag.set(None);
            layout.drop_target.set(None);
            for region in regions.borrow().iter() {
                region.set_style("display", "none");
            }
        }
    });
    drop_indicator.set_attribute("data-ondrop", &drop_hid.to_string());

    overlay_regions.borrow_mut().push(drop_indicator.clone());
    tab_bar.append_child(&drop_indicator);

    tab_bar.into()
}

// ── Splitters ──────────────────────────────────────────────────────────────

/// Zone splitter — drag to resize adjacent zones.
#[component]
fn ZoneSplitter(
    kind: ContainerKind,
    zone_idx: u32,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let zi = zone_idx as usize;
    let horizontal_zones = matches!(kind, ContainerKind::Bottom | ContainerKind::Center);

    let style = if horizontal_zones {
        "width:3px;min-width:3px;flex-shrink:0;\
         cursor:col-resize;background:transparent;\
         position:relative;"
    } else {
        "height:3px;min-height:3px;flex-shrink:0;\
         cursor:row-resize;background:transparent;\
         position:relative;"
    };

    let zone_a = zi;
    let zone_b = zi + 1;

    rsx! {
        div {
            style: {style},
            onclick: {move || {
                let ctx = get_click_context();
                let start_pos = if horizontal_zones { ctx.mouse_x } else { ctx.mouse_y };
                let cfg = layout.read_config(&backing);
                let container_cfg = match kind {
                    ContainerKind::Left => &cfg.left,
                    ContainerKind::Right => &cfg.right,
                    ContainerKind::Bottom => &cfg.bottom,
                    ContainerKind::Center => &cfg.center,
                    _ => return,
                };
                let container_px = match kind {
                    ContainerKind::Left | ContainerKind::Right => {
                        let (_, wh) = layout.window_size.get();
                        (wh - 62.0).max(100.0)
                    }
                    ContainerKind::Bottom | ContainerKind::Center => {
                        let (ww, _) = layout.window_size.get();
                        (ww - layout.left_width.get() - layout.right_width.get()).max(100.0)
                    }
                    _ => 400.0,
                };
                let start_frac_a = container_cfg.zones.get(zone_a).map(|z| z.size_fraction).unwrap_or(0.5);
                let start_frac_b = container_cfg.zones.get(zone_b).map(|z| z.size_fraction).unwrap_or(0.5);
                let total_frac = start_frac_a + start_frac_b;
                let min_frac = 0.1;
                let backing = backing.clone();

                Drag::absolute()
                    .on_move(move |mx, my| {
                        let current = if horizontal_zones { mx } else { my };
                        let delta_px = current - start_pos;
                        let delta_frac = delta_px / container_px * total_frac;
                        let new_a = (start_frac_a + delta_frac).clamp(min_frac, total_frac - min_frac);
                        let new_b = total_frac - new_a;
                        layout.set_zone_fractions(&backing, kind, zone_a, new_a, zone_b, new_b);
                    })
                    .start();
            }},
        }
    }
}

/// Container splitter — drag to resize left/right/bottom containers.
#[component]
fn ContainerSplitter(
    side: ContainerKind,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let is_horizontal = matches!(side, ContainerKind::Bottom);

    let style = if is_horizontal {
        "height:4px;min-height:4px;flex-shrink:0;\
         cursor:row-resize;background:transparent;\
         position:relative;z-index:10;"
    } else {
        "width:4px;min-width:4px;flex-shrink:0;\
         cursor:col-resize;background:transparent;\
         position:relative;z-index:10;"
    };

    let s = side;
    rsx! {
        div {
            style: {style},
            onclick: {move || {
                let ctx = get_click_context();
                let start_mouse = if matches!(s, ContainerKind::Bottom) { ctx.mouse_y } else { ctx.mouse_x };
                let start_size = match s {
                    ContainerKind::Left => layout.left_width.get(),
                    ContainerKind::Right => layout.right_width.get(),
                    ContainerKind::Bottom => layout.bottom_height.get(),
                    _ => 0.0,
                };
                let backing = backing.clone();
                Drag::absolute()
                    .on_move(move |mx, my| {
                        let current = if matches!(s, ContainerKind::Bottom) { my } else { mx };
                        let delta = current - start_mouse;
                        let min_size = 80.0;
                        let new_size = match s {
                            ContainerKind::Left => (start_size + delta).max(min_size),
                            ContainerKind::Right => (start_size - delta).max(min_size),
                            ContainerKind::Bottom => (start_size - delta).max(min_size),
                            _ => start_size,
                        };
                        match s {
                            ContainerKind::Left => layout.left_width.set(new_size),
                            ContainerKind::Right => layout.right_width.set(new_size),
                            ContainerKind::Bottom => layout.bottom_height.set(new_size),
                            _ => {}
                        }
                        layout.publish_to_backing(&backing);
                    })
                    .start();
            }},
        }
    }
}

// ── Drop edge overlay ──────────────────────────────────────────────────────

/// Build drop edge overlay regions for a zone content area.
///
/// Creates 2 edge regions (matching the container's stacking direction),
/// all starting `display:none`. Uses register_handler for drag events.
fn build_drop_edge_overlay(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    kind: ContainerKind,
    zi: usize,
    layout: LayoutState,
    backing: LayoutBacking,
    overlay_regions: &OverlayRegions,
) {
    use crate::layout::state::SplitEdge;

    let overlay = scope.create_element("div");
    overlay.set_attribute(
        "style",
        "position:absolute;inset:0;z-index:10;pointer-events:none;",
    );

    let highlight_color = "rgba(59,130,246,0.2)";
    let highlight_border = "2px dashed rgba(59,130,246,0.6)";

    let vertical = matches!(kind, ContainerKind::Left | ContainerKind::Right);
    let edges: Vec<(SplitEdge, &str)> = if vertical {
        vec![
            (SplitEdge::Top,    "display:none;pointer-events:auto;position:absolute;top:0;left:0;right:0;height:50%;"),
            (SplitEdge::Bottom, "display:none;pointer-events:auto;position:absolute;bottom:0;left:0;right:0;height:50%;"),
        ]
    } else {
        vec![
            (SplitEdge::Left,   "display:none;pointer-events:auto;position:absolute;top:0;left:0;bottom:0;width:50%;"),
            (SplitEdge::Right,  "display:none;pointer-events:auto;position:absolute;top:0;right:0;bottom:0;width:50%;"),
        ]
    };

    for (edge, base_style) in edges {
        // Created via rsx! for reactive style; data-* attributes attached imperatively.
        let my_target = crate::layout::state::DropTarget::Split {
            container: kind,
            zone_idx: zi,
            edge,
        };
        let region = {
            let __scope = &mut *scope;
            rsx! {
                div {
                    style: {move || {
                        let dragging = layout.tab_drag.get().is_some();
                        let display = if dragging { "block" } else { "none" };
                        let drop = layout.drop_target.get();
                        let border_side = match edge {
                            SplitEdge::Top => "border-bottom",
                            SplitEdge::Bottom => "border-top",
                            SplitEdge::Left => "border-right",
                            SplitEdge::Right => "border-left",
                        };
                        let (bg, border_css) = if drop == Some(my_target) {
                            (highlight_color, format!("{border_side}:{highlight_border};"))
                        } else {
                            ("transparent", format!("{border_side}:none;"))
                        };
                        // Strip "display:none;" from base_style and use reactive display.
                        let pos_style = base_style.replace("display:none;", "");
                        format!(
                            "display:{display};{pos_style}\
                             background:{bg};{border_css}"
                        )
                    }},
                }
            }
        };

        let enter_hid = scope.register_handler({
            let edge = edge;
            move || {
                layout.drop_target.set(Some(crate::layout::state::DropTarget::Split {
                    container: kind,
                    zone_idx: zi,
                    edge,
                }));
            }
        });
        region.set_attribute("data-ondragenter", &enter_hid.to_string());

        let leave_hid = scope.register_handler({
            move || {
                layout.drop_target.set(None);
            }
        });
        region.set_attribute("data-ondragleave", &leave_hid.to_string());

        let drop_hid = scope.register_handler({
            let backing = backing.clone();
            let regions = overlay_regions.clone();
            move || {
                if let Some(data) = layout.tab_drag.get() {
                    layout.split_tab(&backing, data.panel, kind, zi, edge);
                }
                layout.tab_drag.set(None);
                layout.drop_target.set(None);
                for r in regions.borrow().iter() {
                    r.set_style("display", "none");
                }
            }
        });
        region.set_attribute("data-ondrop", &drop_hid.to_string());

        overlay_regions.borrow_mut().push(region.clone());
        overlay.append_child(&region);
    }

    parent.append_child(&overlay);
}
