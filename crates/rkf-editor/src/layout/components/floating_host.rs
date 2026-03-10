//! Floating panel host — renders floating panels as overlays.

use rinch::prelude::*;

use crate::layout::panel_registry;
use crate::layout::state::{LayoutBacking, LayoutState, TabDragData};
use crate::layout::ContainerKind;

/// Renders all floating panels as draggable, resizable overlays.
#[component]
pub fn FloatingPanelHost() -> NodeHandle {
    let layout = use_context::<LayoutState>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "position:absolute;inset:0;z-index:200;pointer-events:none;",
    );

    // Rebuild all floating panels whenever structure_rev changes.
    rinch::core::for_each_dom_typed(
        __scope,
        &root,
        move || vec![layout.structure_rev.get()],
        |rev| format!("{rev}"),
        |_rev, __scope| {
            let layout = use_context::<LayoutState>();
            let backing = use_context::<LayoutBacking>();
            let cfg = layout.read_config(&backing);
            let container = __scope.create_element("div");

            for (fi, fp) in cfg.floating.iter().enumerate() {
                let panel_id = fp.panel;

                let x = Signal::new(fp.x);
                let y = Signal::new(fp.y);
                let w = Signal::new(fp.width);
                let h = Signal::new(fp.height);

                // Build fp_root via rsx! with a reactive style closure for position/size.
                let fp_root = rsx! {
                    div {
                        class: "rinch-floating-panel",
                        style: {move || format!(
                            "left:{:.0}px;top:{:.0}px;width:{:.0}px;height:{:.0}px;\
                             pointer-events:auto;z-index:auto;",
                            x.get(), y.get(), w.get(), h.get()
                        )},
                    }
                };

                // Header
                let header = __scope.create_element("div");
                header.set_attribute("class", "rinch-floating-panel__header");

                let title_span = __scope.create_element("span");
                title_span.set_attribute("class", "rinch-floating-panel__title");
                let title_text = __scope.create_text(panel_id.display_name());
                title_span.append_child(&title_text);
                header.append_child(&title_span);

                // Close button
                {
                    let close_btn = __scope.create_element("button");
                    close_btn.set_attribute("class", "rinch-floating-panel__close");
                    let close_text = __scope.create_text("\u{00D7}");
                    close_btn.append_child(&close_text);
                    let close_hid = __scope.register_handler({
                        let backing = backing.clone();
                        move || {
                            layout.dock_floating_panel(&backing, fi);
                        }
                    });
                    close_btn.set_attribute("data-rid", &close_hid.to_string());
                    header.append_child(&close_btn);
                }

                // Header drag: real-time panel movement via Drag builder.
                // On drag start, we also set tab_drag so drop zone overlays appear.
                // On drag end, if cursor is over a drop zone, dock the panel there;
                // otherwise persist the new position.
                {
                    let hid = __scope.register_handler({
                        let backing = backing.clone();
                        move || {
                            let ctx = get_click_context();
                            let start_x = x.get();
                            let start_y = y.get();
                            let start_mx = ctx.mouse_x;
                            let start_my = ctx.mouse_y;

                            // Signal that a tab drag is active so overlay regions appear.
                            layout.tab_drag.set(Some(TabDragData {
                                panel: panel_id,
                                source_container: ContainerKind::Floating,
                                source_zone: fi,
                            }));

                            let backing_drag = backing.clone();
                            let backing_end = backing.clone();
                            Drag::absolute()
                                .on_move(move |mx, my| {
                                    x.set(start_x + mx - start_mx);
                                    y.set(start_y + my - start_my);
                                    // Update cursor for ghost overlay.
                                    layout.drag_cursor.set(Some((mx, my)));
                                    // Geometric drop target detection.
                                    let target = layout.hit_test_drop(&backing_drag, mx, my);
                                    layout.drop_target.set(target);
                                })
                                .on_end(move |_mx, _my| {
                                    // Drag ended — check if we're over a drop zone.
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
                                                // Just moved — persist the new position.
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
                        }
                    });
                    header.set_attribute("data-rid", &hid.to_string());
                }

                fp_root.append_child(&header);

                // Body
                let body = __scope.create_element("div");
                body.set_attribute("class", "rinch-floating-panel__body");
                let content = panel_registry::render_panel(__scope, panel_id);
                body.append_child(&content);
                fp_root.append_child(&body);

                // Resize handle
                {
                    let resize = __scope.create_element("div");
                    resize.set_attribute("class", "rinch-floating-panel__resize");
                    let rhid = __scope.register_handler(move || {
                        let ctx = get_click_context();
                        let start_w = w.get();
                        let start_h = h.get();
                        let start_mx = ctx.mouse_x;
                        let start_my = ctx.mouse_y;
                        Drag::absolute()
                            .on_move(move |mx, my| {
                                w.set((start_w + mx - start_mx).max(200.0));
                                h.set((start_h + my - start_my).max(100.0));
                            })
                            .start();
                    });
                    resize.set_attribute("data-rid", &rhid.to_string());
                    fp_root.append_child(&resize);
                }

                container.append_child(&fp_root);
            }

            container
        },
    );

    // Drag ghost overlay — uses rsx! with reactive style and text closures.
    let ghost = rsx! {
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
    };

    root.append_child(&ghost);

    root
}
