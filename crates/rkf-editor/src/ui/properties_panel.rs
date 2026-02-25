//! Properties panel — shows details of the currently selected entity.
//!
//! Tracks the shared `UiRevision` signal so it re-renders whenever the
//! selection changes (from scene tree clicks or viewport picks).

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::{EditorState, SelectedEntity, UiRevision};

// ── Style constants ──────────────────────────────────────────────────────────

const LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

const SECTION_STYLE: &str = "font-size:12px;color:var(--rinch-color-text);padding:6px 12px;";

const VALUE_STYLE: &str = "font-size:12px;color:var(--rinch-color-dimmed);padding:2px 12px;\
    font-family:var(--rinch-font-family-monospace);";

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

        if let Some(ref sel) = es.selected_entity {
            match sel {
                SelectedEntity::Object(entity_id) => {
                    if let Some(node) = es.scene_tree.find_node(*entity_id) {
                        let name_row = __scope.create_element("div");
                        name_row.set_attribute("style", SECTION_STYLE);
                        name_row.append_child(&__scope.create_text(&node.name));
                        container.append_child(&name_row);

                        let id_row = __scope.create_element("div");
                        id_row.set_attribute("style", VALUE_STYLE);
                        id_row.append_child(
                            &__scope.create_text(&format!("Entity ID: {entity_id}")),
                        );
                        container.append_child(&id_row);

                        if !node.children.is_empty() {
                            let cr = __scope.create_element("div");
                            cr.set_attribute("style", VALUE_STYLE);
                            cr.append_child(
                                &__scope.create_text(&format!("Children: {}", node.children.len())),
                            );
                            container.append_child(&cr);
                        }
                    }
                }
                SelectedEntity::Light(lid) => {
                    if let Some(light) = es.light_editor.get_light(*lid) {
                        let name_row = __scope.create_element("div");
                        name_row.set_attribute("style", SECTION_STYLE);
                        let type_name = match light.light_type {
                            crate::light_editor::EditorLightType::Point => "Point Light",
                            crate::light_editor::EditorLightType::Spot => "Spot Light",
                        };
                        name_row.append_child(&__scope.create_text(type_name));
                        container.append_child(&name_row);

                        let detail_row = __scope.create_element("div");
                        detail_row.set_attribute("style", VALUE_STYLE);
                        detail_row.append_child(&__scope.create_text(
                            &format!("Intensity: {:.2}  Range: {:.1}", light.intensity, light.range),
                        ));
                        container.append_child(&detail_row);
                    }
                }
                SelectedEntity::Camera => {
                    let name_row = __scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    name_row.append_child(&__scope.create_text("Camera"));
                    container.append_child(&name_row);

                    let pos = es.editor_camera.position;
                    let detail_row = __scope.create_element("div");
                    detail_row.set_attribute("style", VALUE_STYLE);
                    detail_row.append_child(&__scope.create_text(
                        &format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z),
                    ));
                    container.append_child(&detail_row);
                }
                _ => {
                    let msg = __scope.create_element("div");
                    msg.set_attribute(
                        "style",
                        &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                    );
                    msg.append_child(&__scope.create_text("No properties"));
                    container.append_child(&msg);
                }
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
