//! Properties panel — shows details of the currently selected entity.
//!
//! Tracks the shared `UiRevision` signal so it re-renders whenever the
//! selection changes (from scene tree clicks or viewport picks).

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::{EditorState, UiRevision};

// ── Style constants ──────────────────────────────────────────────────────────

const LABEL_STYLE: &str = "font-size:11px;color:rgba(255,255,255,0.45);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

const SECTION_STYLE: &str = "font-size:12px;color:rgba(255,255,255,0.7);padding:6px 12px;";

const VALUE_STYLE: &str = "font-size:12px;color:rgba(255,255,255,0.55);padding:2px 12px;\
    font-family:monospace;";

// ── Component ────────────────────────────────────────────────────────────────

/// Properties panel component.
///
/// Displays the name and basic info about the currently selected entity.
/// Re-renders reactively when the shared `UiRevision` signal changes.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    // Reactive content — rebuilds when revision changes.
    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        revision.track();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // Header.
        let header = __scope.create_element("div");
        header.set_attribute("style", LABEL_STYLE);
        let header_text = __scope.create_text("Properties");
        header.append_child(&header_text);
        container.append_child(&header);

        // Read selected entity info.
        let es = es.lock().unwrap();
        let selected = es.scene_tree.selected_entities();

        if selected.is_empty() {
            let msg = __scope.create_element("div");
            msg.set_attribute(
                "style",
                &format!("{SECTION_STYLE}color:rgba(255,255,255,0.35);"),
            );
            let msg_text = __scope.create_text("No object selected");
            msg.append_child(&msg_text);
            container.append_child(&msg);
        } else {
            for &entity_id in &selected {
                if let Some(node) = es.scene_tree.find_node(entity_id) {
                    // Entity name.
                    let name_row = __scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    let name_text = __scope.create_text(&node.name);
                    name_row.append_child(&name_text);
                    container.append_child(&name_row);

                    // Entity ID.
                    let id_row = __scope.create_element("div");
                    id_row.set_attribute("style", VALUE_STYLE);
                    let id_text =
                        __scope.create_text(&format!("Entity ID: {entity_id}"));
                    id_row.append_child(&id_text);
                    container.append_child(&id_row);

                    // Children count.
                    if !node.children.is_empty() {
                        let children_row = __scope.create_element("div");
                        children_row.set_attribute("style", VALUE_STYLE);
                        let children_text = __scope.create_text(&format!(
                            "Children: {}",
                            node.children.len()
                        ));
                        children_row.append_child(&children_text);
                        container.append_child(&children_row);
                    }
                }
            }
        }

        container
    });

    root
}
