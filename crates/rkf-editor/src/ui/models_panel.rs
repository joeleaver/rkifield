//! Models panel — browse and place `.rkf` voxelized model files.
//!
//! Scans the project for `.rkf` files and displays them as a list.
//! Double-click to place a model in the scene at the camera target position.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

const ITEM_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:4px 12px;font-size:11px;cursor:pointer;\
    border-bottom:1px solid var(--rinch-color-border);";

const ITEM_HOVER_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:4px 12px;font-size:11px;cursor:pointer;\
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
#[component]
fn ModelItem(name: String, path: String) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let hovered = Signal::new(false);
    let place_path = path.clone();

    rsx! {
        div {
            style: {move || if hovered.get() { ITEM_HOVER_STYLE } else { ITEM_STYLE }},
            onmouseenter: move || hovered.set(true),
            onmouseleave: move || hovered.set(false),

            // Model name.
            span { style: {ITEM_NAME_STYLE}, {name} }

            // File extension badge.
            span { style: {ITEM_PATH_STYLE}, ".rkf" }

            // Place button.
            button {
                style: {PLACE_BTN_STYLE},
                onclick: {
                    let cmd = cmd.clone();
                    let path = place_path.clone();
                    move || {
                        let _ = cmd.0.send(EditorCommand::PlaceModel { asset_path: path.clone() });
                    }
                },
                "Place"
            }
        }
    }
}
