//! Models panel — browse and place `.rkf` voxelized model files.
//!
//! Scans the project for `.rkf` files and displays them as a list.
//! Click "Place" to place at camera target. Click and drag into the
//! viewport to place with GPU-raycasted positioning.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

const ITEM_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:4px 12px;font-size:11px;cursor:grab;\
    border-bottom:1px solid var(--rinch-color-border);";

const ITEM_HOVER_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:4px 12px;font-size:11px;cursor:grab;\
    border-bottom:1px solid var(--rinch-color-border);\
    background:var(--rinch-color-dark-5);";

const ITEM_NAME_STYLE: &str = "\
    flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;\
    white-space:nowrap;color:var(--rinch-color-text);";

const ITEM_PATH_STYLE: &str = "\
    color:var(--rinch-color-placeholder);font-size:9px;\
    font-family:var(--rinch-font-family-monospace);flex-shrink:0;";

const PLACE_BTN_STYLE: &str = "\
    padding:1px 6px;font-size:9px;cursor:pointer;border-radius:3px;\
    background:var(--rinch-primary-color-9);color:var(--rinch-color-text);\
    border:1px solid var(--rinch-primary-color-7);flex-shrink:0;";

const EMPTY_STYLE: &str = "\
    font-size:11px;color:var(--rinch-color-placeholder);padding:12px;";

/// Models panel component — lists `.rkf` files from the project.
#[component]
pub fn ModelsPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;",
            // Count badge.
            div {
                style: "font-size:10px;color:var(--rinch-color-placeholder);\
                        font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                        border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
                {move || {
                    let count = ui.models.get().len();
                    if count == 0 {
                        "No .rkf models found".to_string()
                    } else {
                        format!("{count} models")
                    }
                }}
            }

            // List of models.
            div {
                style: "overflow-y:auto;flex:1;min-height:0;",
                for (name, path) in ui.models.get() {
                    ModelItem {
                        key: path.clone(),
                        name: name,
                        path: path,
                    }
                }
            }
        }
    }
}

/// A single model item row.
///
/// Click the row to start a drag-to-place operation (uses rinch Drag API,
/// not HTML drag-and-drop, so mouse events reach the render surface).
/// Click "Place" button to place at camera target without dragging.
#[component]
fn ModelItem(name: String, path: String) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let hovered = Signal::new(false);

    let row = __scope.create_element("div");
    // Reactive hover style.
    {
        let row = row.clone();
        __scope.create_effect(move || {
            let style = if hovered.get() { ITEM_HOVER_STYLE } else { ITEM_STYLE };
            row.set_attribute("style", style);
        });
    }
    {
        let handler_id = __scope.register_handler({ let h = hovered; move || h.set(true) });
        row.set_attribute("data-onmouseenter", &handler_id.to_string());
    }
    {
        let handler_id = __scope.register_handler({ let h = hovered; move || h.set(false) });
        row.set_attribute("data-onmouseleave", &handler_id.to_string());
    }

    // Click on row starts drag-to-place via Drag::absolute().
    {
        let path = path.clone();
        let cmd = cmd.clone();
        let handler_id = __scope.register_handler(move || {
            let spawned = std::rc::Rc::new(std::cell::Cell::new(false));
            let spawned_move = spawned.clone();
            let spawned_end = spawned.clone();
            let path_enter = path.clone();
            let cmd_enter = cmd.clone();
            let cmd_move = cmd.clone();
            let cmd_end = cmd.clone();

            Drag::absolute()
                .on_move(move |mx, my| {
                    if !spawned_move.get() {
                        // First move — spawn the entity.
                        spawned_move.set(true);
                        let _ = cmd_enter.0.send(EditorCommand::DragModelEnter {
                            asset_path: path_enter.clone(),
                        });
                    }
                    let _ = cmd_move.0.send(EditorCommand::DragModelMove { x: mx, y: my });
                })
                .on_end(move |_mx, _my| {
                    if spawned_end.get() {
                        let _ = cmd_end.0.send(EditorCommand::DragModelDrop);
                    } else {
                        // Clicked without moving — no-op (use Place button instead).
                    }
                })
                .start();
        });
        row.set_attribute("data-rid", &handler_id.to_string());
    }

    // Model name.
    let name_span = __scope.create_element("span");
    name_span.set_attribute("style", ITEM_NAME_STYLE);
    name_span.set_text(&name);
    row.append_child(&name_span);

    // File extension badge.
    let ext_span = __scope.create_element("span");
    ext_span.set_attribute("style", ITEM_PATH_STYLE);
    ext_span.set_text(".rkf");
    row.append_child(&ext_span);

    // Place button (click to place at camera target, no drag).
    let place_btn = __scope.create_element("button");
    place_btn.set_attribute("style", PLACE_BTN_STYLE);
    place_btn.set_text("Place");
    {
        let path = path.clone();
        let cmd = cmd.clone();
        let handler_id = __scope.register_handler(move || {
            let _ = cmd.0.send(EditorCommand::PlaceModel { asset_path: path.clone() });
        });
        place_btn.set_attribute("data-rid", &handler_id.to_string());
    }
    row.append_child(&place_btn);

    row
}
