//! LayoutRoot component — replaces the hardcoded div layout in editor_ui.
//!
//! Reads layout config from `LayoutBacking` (non-reactive) and subscribes to
//! revision signals for reactive rendering:
//! - `structure_rev` → rebuilds the container/zone tree (rare)
//! - `tab_rev` → swaps zone content panels (tab switch, frequent)
//!
//! Splitter drags update pixel signals only — no DOM rebuild.

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

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex:1;min-height:0;");

    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Subscribe to structural changes only — rebuilds the layout tree.
        // Splitter sizes are applied via Effects (no rebuild on drag).
        let _rev = layout.structure_rev.get();
        // Read config from backing (non-reactive — no signal borrow).
        let config = layout.read_config(&backing);

        // Shared collection — build_drop_edge_overlay pushes into this,
        // build_tab_bar's drag handlers read from it to show/hide.
        let overlay_regions: OverlayRegions = Rc::new(RefCell::new(Vec::new()));

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex:1;min-height:0;width:100%;");

        // ── Left container ──
        if !config.left.collapsed && !config.left.zones.is_empty() {
            let left_div = __scope.create_element("div");
            left_div.set_attribute(
                "style",
                "flex-shrink:0;\
                 background:var(--rinch-color-dark-8);\
                 border-right:1px solid var(--rinch-color-border);\
                 display:flex;flex-direction:column;min-height:0;",
            );
            {
                let el = left_div.clone();
                Effect::new(move || {
                    let w = layout.left_width.get();
                    el.set_style("width", &format!("{w:.0}px"));
                });
            }
            render_container_zones(__scope, &left_div, &config.left, ContainerKind::Left, layout, &backing, &overlay_regions);
            container.append_child(&left_div);

            let splitter = build_container_splitter(__scope, ContainerKind::Left, layout, backing.clone());
            container.append_child(&splitter);
        }

        // ── Center column (viewport + bottom) ──
        {
            let center_col = __scope.create_element("div");
            center_col.set_attribute(
                "style",
                "flex:1;display:flex;flex-direction:column;min-height:0;min-width:0;",
            );

            if !config.center.collapsed && !config.center.zones.is_empty() {
                let center_div = __scope.create_element("div");
                center_div.set_attribute(
                    "style",
                    "flex:1;min-height:0;display:flex;flex-direction:column;",
                );
                render_container_zones(
                    __scope, &center_div, &config.center, ContainerKind::Center, layout, &backing, &overlay_regions,
                );
                center_col.append_child(&center_div);
            } else {
                let empty = __scope.create_element("div");
                empty.set_attribute("style", "flex:1;min-height:0;");
                center_col.append_child(&empty);
            }

            if !config.bottom.collapsed && !config.bottom.zones.is_empty() {
                let splitter = build_container_splitter(__scope, ContainerKind::Bottom, layout, backing.clone());
                center_col.append_child(&splitter);

                let bottom_div = __scope.create_element("div");
                bottom_div.set_attribute(
                    "style",
                    "flex-shrink:0;\
                     background:var(--rinch-color-dark-8);\
                     border-top:1px solid var(--rinch-color-border);\
                     display:flex;flex-direction:row;min-height:0;",
                );
                {
                    let el = bottom_div.clone();
                    Effect::new(move || {
                        let h = layout.bottom_height.get();
                        el.set_style("height", &format!("{h:.0}px"));
                    });
                }
                render_container_zones(
                    __scope, &bottom_div, &config.bottom, ContainerKind::Bottom, layout, &backing, &overlay_regions,
                );
                center_col.append_child(&bottom_div);
            }

            container.append_child(&center_col);
        }

        // ── Right container ──
        if !config.right.collapsed && !config.right.zones.is_empty() {
            let splitter = build_container_splitter(__scope, ContainerKind::Right, layout, backing.clone());
            container.append_child(&splitter);

            let right_div = __scope.create_element("div");
            right_div.set_attribute(
                "style",
                "flex-shrink:0;\
                 background:var(--rinch-color-dark-8);\
                 border-left:1px solid var(--rinch-color-border);\
                 display:flex;flex-direction:column;min-height:0;",
            );
            {
                let el = right_div.clone();
                Effect::new(move || {
                    let w = layout.right_width.get();
                    el.set_style("width", &format!("{w:.0}px"));
                });
            }
            render_container_zones(__scope, &right_div, &config.right, ContainerKind::Right, layout, &backing, &overlay_regions);
            container.append_child(&right_div);
        }

        // Reactive effect: show/hide overlay regions when tab_drag changes.
        // Works for both tab bar DnD drags and floating panel component drags.
        {
            let regions = overlay_regions.clone();
            Effect::new(move || {
                let drag = layout.tab_drag.get();
                let regions = regions.borrow();
                if drag.is_some() {
                    for region in regions.iter() {
                        region.set_style("display", "block");
                    }
                } else {
                    for region in regions.iter() {
                        region.set_style("display", "none");
                    }
                }
            });
        }

        container
    });

    root
}

/// Render all zones within a container.
fn render_container_zones(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    container: &crate::layout::ContainerConfig,
    kind: ContainerKind,
    layout: LayoutState,
    backing: &LayoutBacking,
    overlay_regions: &OverlayRegions,
) {
    let zone_count = container.zones.len();

    for (zi, zone) in container.zones.iter().enumerate() {
        if zone.tabs.is_empty() {
            continue;
        }

        let fraction = zone.size_fraction;
        let zone_div = scope.create_element("div");
        let horizontal_container = matches!(kind, ContainerKind::Bottom | ContainerKind::Center);
        let extra = if horizontal_container { "min-width:0;height:100%;" } else { "min-height:0;" };
        zone_div.set_attribute(
            "style",
            &format!(
                "flex:{fraction} 1 0px;\
                 display:flex;flex-direction:column;overflow:hidden;{extra}"
            ),
        );

        // Tab bar — show if multiple tabs or if not center.
        let show_tab_bar = zone.tabs.len() > 1 || kind != ContainerKind::Center;

        if show_tab_bar {
            let tab_bar_node = {
                let tab_tabs = zone.tabs.clone();
                let tab_backing = backing.clone();
                let tab_kind = kind;
                let tab_zi = zi;
                let tab_regions = overlay_regions.clone();
                let tab_bar_wrapper = scope.create_element("div");
                tab_bar_wrapper.set_attribute("style", "flex-shrink:0;");
                rinch::core::reactive_component_dom(scope, &tab_bar_wrapper, move |__scope| {
                    let _tab_rev = layout.tab_rev.get();
                    let cfg = layout.read_config(&tab_backing);
                    let active = cfg
                        .container(tab_kind)
                        .and_then(|c| c.zones.get(tab_zi))
                        .map(|z| z.active_tab.min(z.tabs.len() - 1))
                        .unwrap_or(0);
                    build_tab_bar(
                        __scope,
                        &tab_tabs,
                        active,
                        tab_kind,
                        tab_zi,
                        layout,
                        tab_backing.clone(),
                        &tab_regions,
                    )
                });
                tab_bar_wrapper
            };
            zone_div.append_child(&tab_bar_node);
        }

        // Zone content — subscribes to `tab_rev` for active tab changes.
        let content_div = scope.create_element("div");
        let pe = if kind == ContainerKind::Center { "pointer-events:none;" } else { "" };
        content_div.set_attribute(
            "style",
            &format!("flex:1 1 0px;overflow-y:auto;position:relative;z-index:0;\
                      display:flex;flex-direction:column;{pe}"),
        );

        let content_backing = backing.clone();
        let content_ck = kind;
        let content_zi = zi;
        rinch::core::reactive_component_dom(scope, &content_div, move |__scope| {
            let _tab_rev = layout.tab_rev.get();
            let cfg = layout.read_config(&content_backing);
            let active_panel = cfg
                .container(content_ck)
                .and_then(|c| c.zones.get(content_zi))
                .map(|z| z.tabs[z.active_tab.min(z.tabs.len() - 1)])
                .unwrap_or(crate::layout::PanelId::SceneView);

            panel_registry::render_panel(__scope, active_panel)
        });

        // Drop edge overlay (hidden until a tab drag starts).
        // Center (canvas) panels don't participate in tab drag — skip overlay.
        if kind != ContainerKind::Center {
            build_drop_edge_overlay(scope, &content_div, kind, zi, layout, backing.clone(), overlay_regions);
        }

        zone_div.append_child(&content_div);
        parent.append_child(&zone_div);

        if zi < zone_count - 1 {
            let zone_splitter = build_zone_splitter(scope, kind, zi, layout, backing.clone());
            parent.append_child(&zone_splitter);
        }
    }
}

/// Build a tab bar for a zone.
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
                // Directly show all overlay regions.
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

                // Float panel if dropped outside all drop targets.
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

    // Tab bar drop indicator — visible "+" zone at the end of the tab bar
    // that appears during drags, making it clear the tab will join this zone.
    let drop_indicator = scope.create_element("div");
    drop_indicator.set_attribute(
        "style",
        "display:none;padding:4px 10px;font-size:11px;\
         color:rgba(59,130,246,0.6);white-space:nowrap;\
         border:1px dashed rgba(59,130,246,0.4);\
         background:rgba(59,130,246,0.05);\
         border-radius:3px;margin:2px 4px;\
         user-select:none;",
    );
    let drop_label = scope.create_text("+");
    drop_indicator.append_child(&drop_label);

    // Highlight on dragenter.
    let enter_hid = scope.register_handler({
        let indicator = drop_indicator.clone();
        let ck = container_kind;
        let zi = zone_idx;
        move || {
            indicator.set_style("background", "rgba(59,130,246,0.2)");
            indicator.set_style("border-color", "rgba(59,130,246,0.6)");
            indicator.set_style("color", "rgba(59,130,246,0.9)");
            layout.drop_target.set(Some(crate::layout::state::DropTarget::Zone {
                container: ck,
                zone_idx: zi,
            }));
        }
    });
    drop_indicator.set_attribute("data-ondragenter", &enter_hid.to_string());

    // Clear on dragleave.
    let leave_hid = scope.register_handler({
        let indicator = drop_indicator.clone();
        move || {
            indicator.set_style("background", "rgba(59,130,246,0.05)");
            indicator.set_style("border-color", "rgba(59,130,246,0.4)");
            indicator.set_style("color", "rgba(59,130,246,0.6)");
            layout.drop_target.set(None);
        }
    });
    drop_indicator.set_attribute("data-ondragleave", &leave_hid.to_string());

    // Reactive highlight for component drags (floating panels).
    {
        let indicator = drop_indicator.clone();
        let ck = container_kind;
        let zi = zone_idx;
        let my_target = crate::layout::state::DropTarget::Zone {
            container: ck,
            zone_idx: zi,
        };
        Effect::new(move || {
            let drop = layout.drop_target.get();
            if drop == Some(my_target) {
                indicator.set_style("background", "rgba(59,130,246,0.2)");
                indicator.set_style("border-color", "rgba(59,130,246,0.6)");
                indicator.set_style("color", "rgba(59,130,246,0.9)");
            } else {
                indicator.set_style("background", "rgba(59,130,246,0.05)");
                indicator.set_style("border-color", "rgba(59,130,246,0.4)");
                indicator.set_style("color", "rgba(59,130,246,0.6)");
            }
        });
    }

    // Drop on indicator → add tab to this zone.
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

    // Register the indicator in overlay_regions so it shows/hides with drags.
    overlay_regions.borrow_mut().push(drop_indicator.clone());
    tab_bar.append_child(&drop_indicator);

    tab_bar.into()
}

/// Build a zone splitter (drag to resize adjacent zones).
fn build_zone_splitter(
    scope: &mut RenderScope,
    kind: ContainerKind,
    zi: usize,
    layout: LayoutState,
    backing: LayoutBacking,
) -> NodeHandle {
    let horizontal_zones = matches!(kind, ContainerKind::Bottom | ContainerKind::Center);
    let zone_splitter = scope.create_element("div");
    if horizontal_zones {
        zone_splitter.set_attribute(
            "style",
            "width:3px;min-width:3px;flex-shrink:0;\
             cursor:col-resize;background:transparent;\
             position:relative;",
        );
    } else {
        zone_splitter.set_attribute(
            "style",
            "height:3px;min-height:3px;flex-shrink:0;\
             cursor:row-resize;background:transparent;\
             position:relative;",
        );
    }

    let zone_a = zi;
    let zone_b = zi + 1;
    let hid = scope.register_handler({
        move || {
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
                ContainerKind::Bottom => {
                    let (ww, _) = layout.window_size.get();
                    (ww - layout.left_width.get() - layout.right_width.get()).max(100.0)
                }
                ContainerKind::Center => {
                    let (ww, _) = layout.window_size.get();
                    (ww - layout.left_width.get() - layout.right_width.get()).max(100.0)
                }
                _ => 400.0,
            };
            let start_frac_a = container_cfg
                .zones
                .get(zone_a)
                .map(|z| z.size_fraction)
                .unwrap_or(0.5);
            let start_frac_b = container_cfg
                .zones
                .get(zone_b)
                .map(|z| z.size_fraction)
                .unwrap_or(0.5);
            let total_frac = start_frac_a + start_frac_b;
            let min_frac = 0.1;
            let backing = backing.clone();

            start_drag_absolute(move |mx, my| {
                let current = if horizontal_zones { mx } else { my };
                let delta_px = current - start_pos;
                let delta_frac = delta_px / container_px * total_frac;
                let new_a = (start_frac_a + delta_frac).clamp(min_frac, total_frac - min_frac);
                let new_b = total_frac - new_a;
                layout.set_zone_fractions(&backing, kind, zone_a, new_a, zone_b, new_b);
            });
        }
    });
    zone_splitter.set_attribute("data-rid", &hid.to_string());

    zone_splitter.into()
}

/// Build a container splitter (drag to resize left/right/bottom containers).
fn build_container_splitter(
    scope: &mut RenderScope,
    side: ContainerKind,
    layout: LayoutState,
    backing: LayoutBacking,
) -> NodeHandle {
    let is_horizontal = matches!(side, ContainerKind::Bottom);

    let el = scope.create_element("div");
    if is_horizontal {
        el.set_attribute(
            "style",
            "height:4px;min-height:4px;flex-shrink:0;\
             cursor:row-resize;background:transparent;\
             position:relative;z-index:10;",
        );
    } else {
        el.set_attribute(
            "style",
            "width:4px;min-width:4px;flex-shrink:0;\
             cursor:col-resize;background:transparent;\
             position:relative;z-index:10;",
        );
    }

    let hid = scope.register_handler({
        let s = side;
        move || {
            let ctx = get_click_context();
            let start_mouse = if matches!(s, ContainerKind::Bottom) {
                ctx.mouse_y
            } else {
                ctx.mouse_x
            };
            let start_size = match s {
                ContainerKind::Left => layout.left_width.get(),
                ContainerKind::Right => layout.right_width.get(),
                ContainerKind::Bottom => layout.bottom_height.get(),
                _ => 0.0,
            };
            let backing = backing.clone();
            start_drag_absolute(move |mx, my| {
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
            });
        }
    });
    el.set_attribute("data-rid", &hid.to_string());

    el.into()
}

/// Build drop edge overlay regions for a zone content area.
///
/// Creates 2 edge regions (matching the container's stacking direction),
/// all starting `display:none`. The ondragstart/ondragend handlers in
/// build_tab_bar show/hide them imperatively via the shared
/// `overlay_regions` collection. The tab bar "+" indicator handles
/// "add to zone" — these regions are only for "create new zone" splits.
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

    // Overlay container — always present, just holds the positioned regions.
    // pointer-events:none so it doesn't block clicks to the content underneath.
    let overlay = scope.create_element("div");
    overlay.set_attribute(
        "style",
        "position:absolute;inset:0;z-index:10;pointer-events:none;",
    );

    let highlight_color = "rgba(59,130,246,0.2)";
    let highlight_border = "2px dashed rgba(59,130,246,0.6)";

    // Only show edges that match the container's stacking direction:
    // - Left/Right containers stack vertically → Top/Bottom splits
    // - Bottom/Center containers stack horizontally → Left/Right splits
    // Each edge takes 50% — full coverage, no center region needed
    // (the tab bar "+" indicator handles "add to zone").
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
        let region = scope.create_element("div");
        region.set_attribute("style", base_style);

        let enter_hid = scope.register_handler({
            let region = region.clone();
            let edge = edge;
            move || {
                let border_side = match edge {
                    SplitEdge::Top => "border-bottom",
                    SplitEdge::Bottom => "border-top",
                    SplitEdge::Left => "border-right",
                    SplitEdge::Right => "border-left",
                };
                region.set_style("background", highlight_color);
                region.set_style(border_side, highlight_border);
                layout.drop_target.set(Some(crate::layout::state::DropTarget::Split {
                    container: kind,
                    zone_idx: zi,
                    edge,
                }));
            }
        });
        region.set_attribute("data-ondragenter", &enter_hid.to_string());

        let leave_hid = scope.register_handler({
            let region = region.clone();
            let edge = edge;
            move || {
                let border_side = match edge {
                    SplitEdge::Top => "border-bottom",
                    SplitEdge::Bottom => "border-top",
                    SplitEdge::Left => "border-right",
                    SplitEdge::Right => "border-left",
                };
                region.set_style("background", "transparent");
                region.set_style(border_side, "none");
                layout.drop_target.set(None);
            }
        });
        region.set_attribute("data-ondragleave", &leave_hid.to_string());

        // Reactive highlight: also highlight/unhighlight based on drop_target signal.
        // This handles component drags (floating panels) where DnD events don't fire.
        {
            let region = region.clone();
            let my_target = crate::layout::state::DropTarget::Split {
                container: kind,
                zone_idx: zi,
                edge,
            };
            Effect::new(move || {
                let drop = layout.drop_target.get();
                let border_side = match edge {
                    SplitEdge::Top => "border-bottom",
                    SplitEdge::Bottom => "border-top",
                    SplitEdge::Left => "border-right",
                    SplitEdge::Right => "border-left",
                };
                if drop == Some(my_target) {
                    region.set_style("background", highlight_color);
                    region.set_style(border_side, highlight_border);
                } else {
                    region.set_style("background", "transparent");
                    region.set_style(border_side, "none");
                }
            });
        }

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
