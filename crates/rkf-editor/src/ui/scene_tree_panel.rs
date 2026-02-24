//! Scene tree panel — shows the hierarchical scene graph in the left panel.
//!
//! Reads the scene tree from `EditorState` (shared via rinch context) and
//! renders it as a flat list of indented rows. Supports expand/collapse
//! for nodes with children, and click-to-select.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::EditorState;
use crate::scene_tree::SceneNode;

// ── Style constants ──────────────────────────────────────────────────────────

const LABEL_STYLE: &str = "font-size:11px;color:rgba(255,255,255,0.45);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

// ── Flattened tree item ──────────────────────────────────────────────────────

/// A single visible row in the flattened scene tree.
#[derive(Clone, Debug)]
struct TreeItem {
    entity_id: u64,
    name: String,
    depth: u32,
    selected: bool,
    expanded: bool,
    has_children: bool,
}

/// Flatten the recursive scene tree into a list of visible items.
///
/// Respects expand/collapse state — children of collapsed nodes are omitted.
fn flatten_tree(roots: &[SceneNode]) -> Vec<TreeItem> {
    let mut items = Vec::new();
    fn visit(node: &SceneNode, depth: u32, out: &mut Vec<TreeItem>) {
        out.push(TreeItem {
            entity_id: node.entity_id,
            name: node.name.clone(),
            depth,
            selected: node.selected,
            expanded: node.expanded,
            has_children: !node.children.is_empty(),
        });
        if node.expanded {
            for child in &node.children {
                visit(child, depth + 1, out);
            }
        }
    }
    for root in roots {
        visit(root, 0, &mut items);
    }
    items
}

// ── Component ────────────────────────────────────────────────────────────────

/// Scene tree panel component.
///
/// Renders the editor's scene hierarchy as an indented list with
/// expand/collapse arrows and click-to-select. Updates are driven by
/// a local revision signal that click handlers increment.
#[component]
pub fn SceneTreePanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_signal(|| 0u64);

    // Touch the revision signal to create a render dependency.
    let _ = revision.get();

    // Read scene tree (brief lock).
    let items = {
        let es = editor_state.lock().unwrap();
        flatten_tree(&es.scene_tree.roots)
    };
    let obj_count = items.len();

    // Build the panel programmatically for full control over per-row handlers.
    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    // Header with object count.
    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    let header_text = __scope.create_text(&format!("Scene ({obj_count})"));
    header.append_child(&header_text);
    root.append_child(&header);

    // Tree rows.
    for item in &items {
        let row = __scope.create_element("div");

        let pad_left = 12 + item.depth * 16;
        let bg = if item.selected {
            "rgba(76,154,255,0.2)"
        } else {
            "transparent"
        };
        let text_color = if item.selected {
            "rgba(255,255,255,0.95)"
        } else {
            "rgba(255,255,255,0.7)"
        };

        row.set_attribute(
            "style",
            &format!(
                "padding:4px 8px 4px {pad_left}px;cursor:pointer;background:{bg};\
                 font-size:12px;color:{text_color};user-select:none;\
                 display:flex;align-items:center;gap:4px;"
            ),
        );

        // Expand/collapse arrow for nodes with children.
        if item.has_children {
            let arrow = __scope.create_element("span");
            arrow.set_attribute(
                "style",
                "font-size:9px;width:14px;text-align:center;opacity:0.5;cursor:pointer;",
            );
            let arrow_char = if item.expanded { "\u{25BC}" } else { "\u{25B6}" };
            let arrow_text = __scope.create_text(arrow_char);
            arrow.append_child(&arrow_text);

            // Arrow click → toggle expand (stop propagation by using separate handler).
            let entity_id = item.entity_id;
            let es = editor_state.clone();
            let rev = revision;
            let handler_id = __scope.register_handler(move || {
                if let Ok(mut state) = es.lock() {
                    state.scene_tree.toggle_expanded(entity_id);
                }
                rev.update(|r| *r += 1);
            });
            arrow.set_attribute("data-rid", &handler_id.to_string());
            row.append_child(&arrow);
        } else {
            // Spacer for alignment.
            let spacer = __scope.create_element("span");
            spacer.set_attribute("style", "width:14px;");
            row.append_child(&spacer);
        }

        // Node name.
        let name_span = __scope.create_element("span");
        let name_text = __scope.create_text(&item.name);
        name_span.append_child(&name_text);
        row.append_child(&name_span);

        // Row click → select node.
        let entity_id = item.entity_id;
        let es = editor_state.clone();
        let rev = revision;
        let handler_id = __scope.register_handler(move || {
            if let Ok(mut state) = es.lock() {
                state.scene_tree.select_node(entity_id);
            }
            rev.update(|r| *r += 1);
        });
        row.set_attribute("data-rid", &handler_id.to_string());

        root.append_child(&row);
    }

    root
}
