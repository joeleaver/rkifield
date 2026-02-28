//! Scene tree panel — shows the hierarchical scene graph in the left panel.
//!
//! Uses rinch's `Tree` component with `TreeNodeData` to render the project
//! hierarchy: Project → Camera + Scene → objects + lights.
//! Reads directly from `world.scene().objects` — no intermediate SceneTree mirror.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;
use rinch_tabler_icons::TablerIcon;

use crate::editor_state::{EditorState, SelectedEntity, UiRevision, UiSignals};
use crate::light_editor::EditorLightType;

// ── Style constants ──────────────────────────────────────────────────────────

const LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    text-transform:uppercase;letter-spacing:1px;padding:8px 12px;";

// ── Tree data builder ────────────────────────────────────────────────────────

/// Build the full tree data from editor state.
///
/// Reads objects from `world.scene().objects` and lights from `light_editor`.
/// Objects with `parent_id` are nested under their parent; root objects
/// (parent_id == None) appear directly under the Scene node.
fn build_tree_data(es: &EditorState) -> Vec<TreeNodeData> {
    // Camera node.
    let camera_node = TreeNodeData::new("camera", "Camera")
        .with_icon(TablerIcon::Camera);

    // Scene children: SDF objects (roots only) + lights.
    let mut scene_children: Vec<TreeNodeData> = Vec::new();

    let scene = es.world.scene();

    // Build a map of parent_id → children for hierarchy.
    // First pass: collect root objects (no parent).
    for obj in &scene.objects {
        if obj.parent_id.is_none() {
            scene_children.push(build_object_node(obj, &scene.objects));
        }
    }

    for light in es.light_editor.all_lights() {
        let value = format!("light:{}", light.id);
        let label = match light.light_type {
            EditorLightType::Point => format!("Point Light {}", light.id),
            EditorLightType::Spot => format!("Spot Light {}", light.id),
        };
        let node = TreeNodeData::new(value, label)
            .with_icon(TablerIcon::Bulb);
        scene_children.push(node);
    }

    let scene_name = scene.name.clone();

    let scene_node = TreeNodeData::new("scene", scene_name)
        .with_icon(TablerIcon::World)
        .with_children(scene_children);

    let project_name = es.current_scene_path.as_ref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled Project".to_string());

    let project_node = TreeNodeData::new("project", project_name)
        .with_icon(TablerIcon::FolderOpen)
        .with_children(vec![camera_node, scene_node]);

    vec![project_node]
}

/// Build a `TreeNodeData` for a `SceneObject` and recursively attach children.
fn build_object_node(
    obj: &rkf_core::scene::SceneObject,
    all_objects: &[rkf_core::scene::SceneObject],
) -> TreeNodeData {
    let value = format!("obj:{}", obj.id);
    let mut tree_node = TreeNodeData::new(value, &obj.name)
        .with_icon(TablerIcon::Cube);

    // Find children (objects whose parent_id == this object's id).
    for child_obj in all_objects {
        if child_obj.parent_id == Some(obj.id) {
            tree_node = tree_node.with_child(build_object_node(child_obj, all_objects));
        }
    }

    tree_node
}

/// Convert a `SelectedEntity` to the tree node value string.
fn selected_to_value(sel: &SelectedEntity) -> String {
    match sel {
        SelectedEntity::Object(id) => format!("obj:{id}"),
        SelectedEntity::Light(id) => format!("light:{id}"),
        SelectedEntity::Camera => "camera".to_string(),
        SelectedEntity::Scene => "scene".to_string(),
        SelectedEntity::Project => "project".to_string(),
    }
}

/// Parse a tree node value string into a `SelectedEntity`.
fn parse_value(value: &str) -> Option<SelectedEntity> {
    if let Some(id_str) = value.strip_prefix("obj:") {
        id_str.parse::<u64>().ok().map(SelectedEntity::Object)
    } else if let Some(id_str) = value.strip_prefix("light:") {
        id_str.parse::<u64>().ok().map(SelectedEntity::Light)
    } else {
        match value {
            "camera" => Some(SelectedEntity::Camera),
            "scene" => Some(SelectedEntity::Scene),
            "project" => Some(SelectedEntity::Project),
            _ => None,
        }
    }
}

// ── Component ────────────────────────────────────────────────────────────────

/// Scene tree panel component.
///
/// Renders the editor's scene hierarchy using rinch's `Tree` component.
/// Hierarchy: Project → Camera + Scene → objects + lights.
#[component]
pub fn SceneTreePanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let revision = use_context::<UiRevision>();
    let ui = use_context::<UiSignals>();

    // Persistent tree expand/select state — created once, lives in rinch context.
    let tree_state = UseTreeReturn::new(UseTreeOptions {
        initial_expanded: ["project".to_string(), "scene".to_string()]
            .into_iter()
            .collect(),
        ..Default::default()
    });

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
    );

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Track selection + scene structure — only rebuild when these change.
        let _ = ui.selection.get();
        let _ = ui.scene_revision.get();
        // Legacy fallback: also track revision for non-migrated code paths.
        revision.track();

        // Read tree data and selection from editor state, then RELEASE the
        // lock before syncing tree signals. clear_selected()/select() trigger
        // reactive cascades (→ RightPanel::render) that also need the lock;
        // holding it here would deadlock (std::Mutex is not reentrant).
        let (tree_data, obj_count, sel_entity) = {
            let es = es.lock().unwrap();
            let data = build_tree_data(&es);
            let count = es.world.scene().objects.len() + es.light_editor.all_lights().len();
            let sel = es.selected_entity;
            (data, count, sel)
        };

        // Sync selection state from EditorState → tree signals (lock released).
        rinch::core::untracked(|| {
            if let Some(ref sel) = sel_entity {
                let value = selected_to_value(sel);
                let current = tree_state.selected.get();
                if !current.contains(&value) {
                    tree_state.controller.clear_selected();
                    tree_state.controller.select(&value);
                }
            } else {
                let current = tree_state.selected.get();
                if !current.is_empty() {
                    tree_state.controller.clear_selected();
                }
            }
        });

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // Header.
        let header = __scope.create_element("div");
        header.set_attribute("style", LABEL_STYLE);
        let header_text = __scope.create_text(&format!("Scene ({obj_count})"));
        header.append_child(&header_text);
        container.append_child(&header);

        // Selection callback → update EditorState.selected_entity.
        let es_cb = es.clone();
        let onselect = ValueCallback::new(move |value: String| {
            if let Some(entity) = parse_value(&value) {
                if let Ok(mut state) = es_cb.lock() {
                    state.selected_entity = Some(entity);
                }
                // Signal update after lock released.
                ui.selection.set(Some(entity));
            }
            revision.bump();
        });

        // Tree component.
        let tree_component = Tree {
            data: tree_data,
            tree: Some(tree_state),
            select_on_click: true,
            expand_on_click: true,
            level_offset: "sm".to_string(),
            onselect: Some(onselect),
            ..Default::default()
        };
        let tree = tree_component.render(__scope, &[]);
        container.append_child(&tree);

        container
    });

    root
}
