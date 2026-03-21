//! Floating panel host — renders floating panels as overlays.

use rinch::prelude::*;

use crate::layout::panel_registry;
use crate::layout::state::{LayoutBacking, LayoutState, TabDragData};
use crate::layout::ContainerKind;
use crate::layout::PanelId;

/// Renders all floating panels as draggable, resizable overlays.
#[component]
pub fn FloatingPanelHost() -> NodeHandle {
    let layout = use_context::<LayoutState>();

    rsx! {
        div {
            style: "position:absolute;inset:0;z-index:200;pointer-events:none;",
            // Rebuild all floating panels whenever structure_rev changes.
            div {
                for _rev in vec![layout.structure_rev.get()] {
                    div {
                        key: format!("{_rev}"),
                        style: "display:contents;",
                        FloatingPanelsContent {}
                    }
                }
            }
            // Drag ghost overlay
            div {
                style: {move || {
                    let drag = layout.tab_drag.get();
                    let drop = layout.drop_target.get();
                    let cursor = layout.drag_cursor.get();

                    let base = "position:absolute;width:200px;height:80px;\
                        border:2px dashed rgba(59,130,246,0.6);\
                        background:rgba(59,130,246,0.08);\
                        border-radius:8px;\
                        pointer-events:none;\
                        z-index:9999;\
                        align-items:center;justify-content:center;\
                        color:rgba(59,130,246,0.8);font-size:12px;";

                    if let (Some(data), None, Some((cx, cy))) = (drag, drop, cursor) {
                        if data.source_container != ContainerKind::Floating {
                            let left = cx + 10.0;
                            let top = cy + 10.0;
                            format!(
                                "{base}display:flex;left:{left:.0}px;top:{top:.0}px;"
                            )
                        } else {
                            format!("{base}display:none;")
                        }
                    } else {
                        format!("{base}display:none;")
                    }
                }},
                {move || {
                    let drag = layout.tab_drag.get();
                    let drop = layout.drop_target.get();
                    let cursor = layout.drag_cursor.get();

                    if let (Some(data), None, Some(_)) = (drag, drop, cursor) {
                        if data.source_container != ContainerKind::Floating {
                            return data.panel.display_name().to_string();
                        }
                    }
                    String::new()
                }}
            }
        }
    }
}

/// Renders all floating panel instances.
#[component]
fn FloatingPanelsContent() -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();
    let cfg = layout.read_config(&backing);

    let panels = Signal::new(
        cfg.floating
            .iter()
            .enumerate()
            .map(|(fi, fp)| (fi, fp.panel, fp.x, fp.y, fp.width, fp.height))
            .collect::<Vec<_>>(),
    );

    rsx! {
        div {
            for (fi, panel_id, px, py, pw, ph) in panels.get() {
                LayoutFloatingPanel {
                    panel_id: panel_id,
                    fi: fi,
                    init_x: px,
                    init_y: py,
                    init_w: pw,
                    init_h: ph,
                }
            }
        }
    }
}

/// A floating panel that uses rinch's `FloatingPanel` component and adds
/// layout integration (tab drag-to-dock, drop targets, position persistence).
#[component]
fn LayoutFloatingPanel(
    panel_id: PanelId,
    fi: usize,
    init_x: f32,
    init_y: f32,
    init_w: f32,
    init_h: f32,
) -> NodeHandle {
    let layout = use_context::<LayoutState>();
    let backing = use_context::<LayoutBacking>();

    let x = Signal::new(init_x);
    let y = Signal::new(init_y);
    let w = Signal::new(init_w);
    let h = Signal::new(init_h);

    // Use rinch's FloatingPanel for rendering (gets inset fast path for free).
    // Override the drag behavior to integrate with the layout drop-target system.
    let panel_content = panel_registry::render_panel(__scope, panel_id);

    // Build the FloatingPanel with rinch's component.
    let panel = FloatingPanel {
        title: panel_id.display_name().to_string(),
        x: Some(x),
        y: Some(y),
        width: Some(w),
        height: Some(h),
        min_width: Some(200.0),
        min_height: Some(100.0),
        on_close: Some(Callback::new({
            let backing = backing.clone();
            move || {
                layout.dock_floating_panel(&backing, fi);
            }
        })),
        ..Default::default()
    };
    let root = panel.render(__scope, &[panel_content]);

    // Enable pointer events on the panel itself — the host container has
    // pointer-events:none so events pass through to the viewport behind,
    // but floating panels need to receive clicks and drags.
    root.set_style("pointer-events", "auto");

    // Override the header's click handler to add layout-aware drag behavior
    // (drop targets, position persistence). The FloatingPanel's default drag
    // only moves the panel — we need tab-drag-to-dock support.
    //
    // Find the header element (first child with rinch-floating-panel__header class)
    // and replace its handler.
    if let Some(header) = root.children().first().cloned() {
        let backing_for_handler = backing.clone();
        let handler_id = __scope.register_handler(move || {
            let ctx = get_click_context();
            let start_x = x.get();
            let start_y = y.get();
            let start_mx = ctx.mouse_x;
            let start_my = ctx.mouse_y;

            layout.tab_drag.set(Some(TabDragData {
                panel: panel_id,
                source_container: ContainerKind::Floating,
                source_zone: fi,
            }));

            let backing_drag = backing_for_handler.clone();
            let backing_end = backing_for_handler.clone();
            Drag::absolute()
                .on_move(move |mx, my| {
                    x.set(start_x + mx - start_mx);
                    y.set(start_y + my - start_my);
                    layout.drag_cursor.set(Some((mx, my)));
                    let target = layout.hit_test_drop(&backing_drag, mx, my);
                    layout.drop_target.set(target);
                })
                .on_end(move |_mx, _my| {
                    let drop = layout.drop_target.get();
                    if let Some(data) = layout.tab_drag.get() {
                        match drop {
                            Some(crate::layout::state::DropTarget::Zone { container, zone_idx }) => {
                                layout.move_tab(&backing_end, data.panel, container, zone_idx);
                            }
                            Some(crate::layout::state::DropTarget::Split { container, zone_idx, edge }) => {
                                layout.split_tab(&backing_end, data.panel, container, zone_idx, edge);
                            }
                            None => {
                                let new_x = x.get();
                                let new_y = y.get();
                                let mut cfg = layout.read_config(&backing_end);
                                if fi < cfg.floating.len() {
                                    cfg.floating[fi].x = new_x;
                                    cfg.floating[fi].y = new_y;
                                    backing_end.store(cfg);
                                }
                            }
                        }
                    }
                    layout.tab_drag.set(None);
                    layout.drop_target.set(None);
                    layout.drag_cursor.set(None);
                })
                .start();
        });
        header.set_attribute("data-rid", &handler_id.to_string());
    }

    root
}
