//! Scene tree panel — shows the hierarchical scene graph in the left panel.
//!
//! Uses rinch's `Tree` component with `TreeNodeData` to render the project
//! hierarchy: Project → Camera + Scene → objects + lights.
//! Reads from the lock-free `SnapshotReader` — no mutex locks needed.

use rinch::prelude::*;
use rinch_tabler_icons::TablerIcon;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SelectedEntity, UiSignals};
use crate::light_editor::SceneLightType;
use crate::ui_snapshot::UiSnapshot;
use crate::{CommandSender, SnapshotReader};

// ── Tree data builder ────────────────────────────────────────────────────────

/// Build the full tree data from a lock-free snapshot.
fn build_tree_data_from_snapshot(snap: &UiSnapshot) -> Vec<TreeNodeData> {
    let camera_node = TreeNodeData::new("camera", "Camera")
        .with_icon(TablerIcon::Camera);

    let mut scene_children: Vec<TreeNodeData> = Vec::new();

    // Root objects (no parent).
    for obj in &snap.objects {
        if obj.parent_id.is_none() {
            scene_children.push(build_snapshot_object_node(obj, &snap.objects));
        }
    }

    for light in &snap.lights {
        let value = format!("light:{}", light.id);
        let label = match light.light_type {
            SceneLightType::Point => format!("Point Light {}", light.id),
            SceneLightType::Spot => format!("Spot Light {}", light.id),
        };
        let node = TreeNodeData::new(value, label).with_icon(TablerIcon::Bulb);
        scene_children.push(node);
    }

    let scene_node = TreeNodeData::new("scene", snap.scene_name.clone())
        .with_icon(TablerIcon::World)
        .with_children(scene_children);

    let project_name = snap.current_scene_path.as_ref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled Project".to_string());

    let project_node = TreeNodeData::new("project", project_name)
        .with_icon(TablerIcon::FolderOpen)
        .with_children(vec![camera_node, scene_node]);

    vec![project_node]
}

/// Build a `TreeNodeData` for a snapshot object, recursively attaching children.
fn build_snapshot_object_node(
    obj: &crate::ui_snapshot::ObjectSummary,
    all_objects: &[crate::ui_snapshot::ObjectSummary],
) -> TreeNodeData {
    let value = format!("obj:{}", obj.id);
    let mut tree_node = TreeNodeData::new(value, &obj.name)
        .with_icon(TablerIcon::Cube);

    for child in all_objects {
        if child.parent_id.map(|p| p as u64) == Some(obj.id) {
            tree_node = tree_node.with_child(build_snapshot_object_node(child, all_objects));
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
/// Tree data rebuilds only on scene structure changes (spawn/delete/open).
/// Selection sync is a separate lightweight Effect — no tree rebuild needed.
#[component]
pub fn SceneTreePanel() -> NodeHandle {
    let snap = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    // Tree state lives as a context (created in editor_ui) so the
    // selection-sync Effect can live outside the render path.
    let tree_state = use_context::<UseTreeReturn>();

    // Header count — derived value, no .set() needed.
    let snap_for_header = snap.clone();
    let header_label = Memo::new(move || {
        let _ = ui.scene_revision.get();
        let guard = snap_for_header.0.load();
        let count = guard.objects.len() + guard.lights.len();
        format!("Scene ({count})")
    });

    // NOTE: Selection sync Effect lives in editor_ui() (ui/mod.rs),
    // NOT here. Effects must not .set() signals during render.

    // Selection callback → send command (no lock).
    let onselect = ValueCallback::new(move |value: String| {
        if let Some(entity) = parse_value(&value) {
            let _ = cmd.0.send(EditorCommand::SelectEntity { entity: Some(entity) });
            ui.selection.set(Some(entity));
            ui.properties_tab.set(0); // switch to Object tab
        }
    });

    // Reactive data source for the Tree — rebuilds only on scene_revision.
    let snap2 = snap.clone();
    let data_source: std::rc::Rc<dyn Fn() -> Vec<TreeNodeData>> =
        std::rc::Rc::new(move || {
            let _ = ui.scene_revision.get();
            let guard = snap2.0.load();
            build_tree_data_from_snapshot(&guard)
        });

    // Initial data for the Tree (read once, data_source handles updates).
    let initial_data = {
        let guard = snap.0.load();
        build_tree_data_from_snapshot(&guard)
    };

    rsx! {
        div {
            style: "flex:1;overflow-y:auto;display:flex;flex-direction:column;",
            div {
                style: "font-size:11px;color:var(--rinch-color-dimmed);\
                        text-transform:uppercase;letter-spacing:1px;padding:8px 12px;",
                {move || header_label.get()}
            }
            Tree {
                data: initial_data,
                tree: Some(tree_state),
                select_on_click: true,
                expand_on_click: true,
                level_offset: "sm",
                onselect: onselect,
                data_source: Some(data_source),
            }
        }
    }
}
