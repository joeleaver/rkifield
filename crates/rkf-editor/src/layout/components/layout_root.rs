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
    let _backing = use_context::<LayoutBacking>();

    // Provide overlay_regions as context so ContainerZones can add to it.
    let overlay_regions: OverlayRegions = Rc::new(RefCell::new(Vec::new()));
    create_context(overlay_regions.clone());

    rsx! {
        div {
            style: "display:flex;flex:1;min-height:0;",
            for _rev in vec![layout.structure_rev.get()] {
                div {
                    key: format!("{_rev}"),
                    style: "display:contents;",
                    LayoutStructure { overlay_regions: overlay_regions.clone() }
                }
            }
        }
    }
}

/// Inner structure component — rebuilds when `structure_rev` changes.
///
/// Extracted from LayoutRoot to replace `for_each_dom_typed` with `for` in rsx.
#[component]
fn LayoutStructure(overlay_regions: OverlayRegions) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let config = layout.read_config(&backing);

    // Clear overlay regions from previous render.
    overlay_regions.borrow_mut().clear();

    let show_left = !config.left.collapsed && !config.left.zones.is_empty();
    let show_center = !config.center.collapsed && !config.center.zones.is_empty();
    let show_bottom = !config.bottom.collapsed && !config.bottom.zones.is_empty();
    let show_right = !config.right.collapsed && !config.right.zones.is_empty();

    rsx! {
        div {
            style: "display:flex;flex:1;min-height:0;width:100%;",

            // ── Left container ──
            if show_left {
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
                ContainerSplitter { side: ContainerKind::Left }
            }

            // ── Center column (viewport + bottom) ──
            div {
                style: "flex:1;display:flex;flex-direction:column;min-height:0;min-width:0;",

                if show_center {
                    div {
                        style: "flex:1;min-height:0;display:flex;flex-direction:column;",
                        ContainerZones { kind: ContainerKind::Center }
                    }
                }
                if !show_center {
                    div { style: "flex:1;min-height:0;" }
                }

                if show_bottom {
                    ContainerSplitter { side: ContainerKind::Bottom }
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
                }
            }

            // ── Right container ──
            if show_right {
                ContainerSplitter { side: ContainerKind::Right }
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
            }
        }
    }
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
        None => return rsx! { span {} },
    };

    let zone_count = container.zones.len();
    let horizontal = matches!(kind, ContainerKind::Bottom | ContainerKind::Center);
    let extra: &'static str = if horizontal { "min-width:0;height:100%;" } else { "min-height:0;" };

    // Build zone children via Rust loop. This component is non-reactive (rebuilt
    // fully on structure_rev), so a Rust loop collecting NodeHandles is correct.
    let wrapper = rsx! { div { style: "display:contents;" } };

    for (zi, zone) in container.zones.iter().enumerate() {
        if zone.tabs.is_empty() {
            continue;
        }
        let initial_fraction = zone.size_fraction;
        let show_tab_bar = zone.tabs.len() > 1 || kind != ContainerKind::Center;
        let tabs = zone.tabs.clone();

        // Clone for each iteration — rsx moves captures into closures.
        let iter_overlay = overlay_regions.clone();
        let iter_backing = backing.clone();

        // Build tab bar conditionally — wrapped in ReactiveTabBar for
        // reactive rebuild on tab_rev changes.
        let tab_bar_node = if show_tab_bar {
            let tb_overlay = iter_overlay.clone();
            Some(rinch::core::untracked(|| {
                (ReactiveTabBar {
                    kind,
                    zone_idx: zi,
                    tabs: tabs.clone(),
                    overlay_regions: tb_overlay,
                    ..Default::default()
                }).render(__scope, &[])
            }))
        } else {
            None
        };

        // Zone wrapper div with reactive flex.
        let zone_div = rsx! {
            div {
                style: {
                    let backing = iter_backing.clone();
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
        };

        // Append tab bar and content as children.
        if let Some(tab_bar) = tab_bar_node {
            zone_div.append_child(&tab_bar);
        }
        let content = rinch::core::untracked(|| {
            (ZoneContentWrapper {
                kind,
                zone_idx: zi,
                overlay_regions: iter_overlay,
                ..Default::default()
            }).render(__scope, &[])
        });
        zone_div.append_child(&content);
        wrapper.append_child(&zone_div);

        // Zone splitter between adjacent zones.
        if zi < zone_count - 1 {
            let splitter = rinch::core::untracked(|| {
                (ZoneSplitter { kind, zone_idx: zi as u32, ..Default::default() })
                    .render(__scope, &[])
            });
            wrapper.append_child(&splitter);
        }
    }

    wrapper
}

/// Wrapper for zone content area + drop edge overlay.
///
/// Extracted to keep the overlay region tracking (push into OverlayRegions)
/// within a dedicated component scope.
#[component]
fn ZoneContentWrapper(
    kind: ContainerKind,
    zone_idx: usize,
    overlay_regions: OverlayRegions,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();

    let content_style: &'static str = if kind == ContainerKind::Center {
        "flex:1 1 0px;overflow-y:auto;position:relative;z-index:0;\
         display:flex;flex-direction:column;pointer-events:none;"
    } else {
        "flex:1 1 0px;overflow-y:auto;position:relative;z-index:0;\
         display:flex;flex-direction:column;"
    };

    let backing = use_context::<LayoutBacking>();

    rsx! {
        div {
            style: {content_style},
            // Keyed by the active panel for THIS zone — only rebuilds when
            // this specific zone's active tab changes, not on every tab_rev bump.
            for active_panel_key in vec![{
                let _ = layout.tab_rev.get(); // subscribe to tab changes
                let cfg = layout.read_config(&backing);
                cfg.container(kind)
                    .and_then(|c| c.zones.get(zone_idx))
                    .map(|z| format!("{:?}", z.tabs[z.active_tab.min(z.tabs.len() - 1)]))
                    .unwrap_or_else(|| "none".to_string())
            }] {
                div {
                    key: active_panel_key.clone(),
                    style: "display:contents;",
                    ZoneContent {
                        kind: kind,
                        zone_idx: zone_idx,
                    }
                }
            }

            // Drop edge overlay (non-center zones only).
            if kind != ContainerKind::Center {
                DropEdgeOverlay {
                    kind: kind,
                    zone_idx: zone_idx,
                    overlay_regions: overlay_regions.clone(),
                }
            }
        }
    }
}

// ── Tab bar reactive wrapper ──────────────────────────────────────────────

/// Wraps TabBarContent in a reactive `for` that rebuilds on `tab_rev` changes.
/// Keyed by this zone's active tab index so only the affected zone rebuilds.
#[component]
fn ReactiveTabBar(
    kind: ContainerKind,
    zone_idx: usize,
    tabs: Vec<crate::layout::PanelId>,
    overlay_regions: OverlayRegions,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();

    rsx! {
        div {
            style: "flex-shrink:0;",
            for active_key in vec![{
                let _ = layout.tab_rev.get(); // subscribe to tab changes
                let cfg = layout.read_config(&backing);
                cfg.container(kind)
                    .and_then(|c| c.zones.get(zone_idx))
                    .map(|z| z.active_tab)
                    .unwrap_or(0)
            }] {
                div {
                    key: format!("{active_key}"),
                    style: "display:contents;",
                    TabBarContent {
                        kind: kind,
                        zone_idx: zone_idx,
                        tabs: tabs.clone(),
                        overlay_regions: overlay_regions.clone(),
                    }
                }
            }
        }
    }
}

// ── Tab bar content component ──────────────────────────────────────────────

/// Renders the tab bar for a zone, rebuilt when `tab_rev` changes.
#[component]
fn TabBarContent(
    kind: ContainerKind,
    zone_idx: usize,
    tabs: Vec<crate::layout::PanelId>,
    overlay_regions: OverlayRegions,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let cfg = layout.read_config(&backing);
    let active = cfg
        .container(kind)
        .and_then(|c| c.zones.get(zone_idx))
        .map(|z| z.active_tab.min(z.tabs.len() - 1))
        .unwrap_or(0);

    let my_target = crate::layout::state::DropTarget::Zone {
        container: kind,
        zone_idx,
    };

    // Build tab items with precomputed styles.
    let tab_items: Vec<(usize, crate::layout::PanelId, String)> = tabs
        .iter()
        .enumerate()
        .map(|(ti, &panel_id)| {
            let is_active = ti == active;
            let bg = if is_active { "background:var(--rinch-color-dark-8);" } else { "" };
            let border_bottom = if is_active {
                "border-bottom:2px solid var(--rinch-primary-color-9);"
            } else {
                "border-bottom:2px solid transparent;"
            };
            let style = format!(
                "padding:4px 10px;font-size:11px;cursor:pointer;\
                 color:var(--rinch-color-text);white-space:nowrap;\
                 user-select:none;{bg}{border_bottom}"
            );
            (ti, panel_id, style)
        })
        .collect();

    // Build the drop indicator and track it in overlay_regions.
    let drop_indicator = rsx! {
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
            ondragenter: {
                let ck = kind;
                let zi = zone_idx;
                move || {
                    layout.drop_target.set(Some(crate::layout::state::DropTarget::Zone {
                        container: ck,
                        zone_idx: zi,
                    }));
                }
            },
            ondragleave: move || {
                layout.drop_target.set(None);
            },
            ondrop: {
                let backing = backing.clone();
                let ck = kind;
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
            },
            "+"
        }
    };

    overlay_regions.borrow_mut().push(drop_indicator.clone());

    // Build tab bar container and append tab items via Rust loop.
    // Tab items have per-tab styles (String) and closures capturing non-Copy
    // types, so a Rust loop with individual rsx! calls is correct here.
    let tab_bar = rsx! {
        div {
            style: "display:flex;flex-shrink:0;height:26px;\
                    background:var(--rinch-color-dark-9);\
                    border-bottom:1px solid var(--rinch-color-border);\
                    overflow-x:auto;",
        }
    };

    for (ti, panel_id, tab_style) in tab_items {
        let tab_node = rsx! {
            div {
                style: {&tab_style},
                draggable: "true",
                onclick: {
                    let backing = backing.clone();
                    move || {
                        layout.set_active_tab(&backing, kind, zone_idx, ti);
                    }
                },
                ondragstart: {
                    let regions = overlay_regions.clone();
                    move || {
                        layout.tab_drag.set(Some(crate::layout::state::TabDragData {
                            panel: panel_id,
                            source_container: kind,
                            source_zone: zone_idx,
                        }));
                        for region in regions.borrow().iter() {
                            region.set_style("display", "block");
                        }
                    }
                },
                ondragmove: move || {
                    let ctx = get_click_context();
                    layout.drag_cursor.set(Some((ctx.mouse_x, ctx.mouse_y)));
                },
                ondragend: {
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
                },
                {panel_id.display_name()}
            }
        };
        tab_bar.append_child(&tab_node);
    }

    tab_bar.append_child(&drop_indicator);
    tab_bar
}

// ── Zone content component ────────────────────────────────────────────────

/// Renders the active panel content for a zone, rebuilt when `tab_rev` changes.
#[component]
fn ZoneContent(
    kind: ContainerKind,
    zone_idx: usize,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let cfg = layout.read_config(&backing);
    let active_panel = cfg
        .container(kind)
        .and_then(|c| c.zones.get(zone_idx))
        .map(|z| z.tabs[z.active_tab.min(z.tabs.len() - 1)])
        .unwrap_or(crate::layout::PanelId::SceneView);

    panel_registry::render_panel(__scope, active_panel)
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

/// Drop edge overlay regions for a zone content area.
///
/// Creates 2 edge regions (matching the container's stacking direction)
/// with reactive styles and drag event handlers. Regions are tracked in
/// `overlay_regions` so tab drag handlers can show/hide them.
#[component]
fn DropEdgeOverlay(
    kind: ContainerKind,
    zone_idx: usize,
    overlay_regions: OverlayRegions,
) -> NodeHandle {
    use crate::layout::state::SplitEdge;

    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();

    let highlight_color = "rgba(59,130,246,0.2)";
    let highlight_border = "2px dashed rgba(59,130,246,0.6)";

    let vertical = matches!(kind, ContainerKind::Left | ContainerKind::Right);
    let edges: Vec<(SplitEdge, &'static str)> = if vertical {
        vec![
            (SplitEdge::Top,    "pointer-events:auto;position:absolute;top:0;left:0;right:0;height:50%;"),
            (SplitEdge::Bottom, "pointer-events:auto;position:absolute;bottom:0;left:0;right:0;height:50%;"),
        ]
    } else {
        vec![
            (SplitEdge::Left,   "pointer-events:auto;position:absolute;top:0;left:0;bottom:0;width:50%;"),
            (SplitEdge::Right,  "pointer-events:auto;position:absolute;top:0;right:0;bottom:0;width:50%;"),
        ]
    };

    // Build region nodes, push into overlay_regions, then embed in the overlay div.
    let mut region_nodes: Vec<NodeHandle> = Vec::new();
    for (edge, pos_style) in edges {
        let my_target = crate::layout::state::DropTarget::Split {
            container: kind,
            zone_idx,
            edge,
        };

        let region = rsx! {
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
                    format!(
                        "display:{display};{pos_style}\
                         background:{bg};{border_css}"
                    )
                }},
                ondragenter: {
                    let edge = edge;
                    move || {
                        layout.drop_target.set(Some(crate::layout::state::DropTarget::Split {
                            container: kind,
                            zone_idx,
                            edge,
                        }));
                    }
                },
                ondragleave: move || {
                    layout.drop_target.set(None);
                },
                ondrop: {
                    let backing = backing.clone();
                    let regions = overlay_regions.clone();
                    move || {
                        if let Some(data) = layout.tab_drag.get() {
                            layout.split_tab(&backing, data.panel, kind, zone_idx, edge);
                        }
                        layout.tab_drag.set(None);
                        layout.drop_target.set(None);
                        for r in regions.borrow().iter() {
                            r.set_style("display", "none");
                        }
                    }
                },
            }
        };

        overlay_regions.borrow_mut().push(region.clone());
        region_nodes.push(region);
    }

    // Build the overlay container and append region nodes.
    let overlay = rsx! {
        div {
            style: "position:absolute;inset:0;z-index:10;pointer-events:none;",
        }
    };
    for node in region_nodes {
        overlay.append_child(&node);
    }
    overlay
}
