//! Scene tree panel — shows the hierarchical scene graph in the left panel.
//!
//! Uses rinch's `Tree` component with `TreeNodeData` to render the project
//! hierarchy: Project → Camera + Scene → objects + lights.
//! Reads from reactive UiSignals — no mutex locks needed.
//!
//! ## Store migration status
//!
//! The scene tree reads `ui.objects` (Vec<ObjectSummary>) and `ui.lights`
//! (Vec<LightSummary>) which are complex typed collections. These remain on
//! UiSignals because `UiValue` doesn't support typed lists. Full migration
//! would require either a `UiValue::List` variant or a parallel typed-signal
//! mechanism in the store. Selection is read from `ui.selection` (source of
//! truth) and mirrored to `editor/selected` in the store for read-only use
//! by other widgets. See `engine_loop_store::push_collection_counts_to_store`.

use rinch::prelude::*;
use rinch_tabler_icons::TablerIcon;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SelectedEntity, SliderSignals, UiSignals};
use crate::light_editor::SceneLightType;
use crate::ui_snapshot::{LightSummary, ObjectSummary};
use crate::CommandSender;

// ── Tree data builder ────────────────────────────────────────────────────────

/// Build the full tree data from object/light summaries.
fn build_tree_data(
    objects: &[ObjectSummary],
    lights: &[LightSummary],
    scene_name: &str,
    scene_path: &Option<String>,
) -> Vec<TreeNodeData> {
    let mut scene_children: Vec<TreeNodeData> = Vec::new();

    // Root objects (no parent).
    for obj in objects {
        if obj.parent_id.is_none() {
            scene_children.push(build_snapshot_object_node(obj, objects));
        }
    }

    for light in lights {
        let value = format!("light:{}", light.id);
        let label = match light.light_type {
            SceneLightType::Point => format!("Point Light {}", light.id),
            SceneLightType::Spot => format!("Spot Light {}", light.id),
        };
        let node = TreeNodeData::new(value, label).with_icon(TablerIcon::Bulb);
        scene_children.push(node);
    }

    let scene_node = TreeNodeData::new("scene", scene_name)
        .with_icon(TablerIcon::World)
        .with_children(scene_children);

    let project_name = scene_path.as_ref()
        .and_then(|p| std::path::Path::new(p).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled Project".to_string());

    let project_node = TreeNodeData::new("project", project_name)
        .with_icon(TablerIcon::FolderOpen)
        .with_children(vec![scene_node]);

    vec![project_node]
}

/// Build a `TreeNodeData` for a snapshot object, recursively attaching children.
fn build_snapshot_object_node(
    obj: &crate::ui_snapshot::ObjectSummary,
    all_objects: &[crate::ui_snapshot::ObjectSummary],
) -> TreeNodeData {
    let value = format!("obj:{}", obj.id);
    let icon = if obj.is_camera {
        TablerIcon::Camera
    } else if obj.object_type == crate::ui_snapshot::ObjectType::None {
        TablerIcon::Point
    } else {
        TablerIcon::Cube
    };
    let mut tree_node = TreeNodeData::new(value, &obj.name)
        .with_icon(icon);

    for child in all_objects {
        if child.parent_id == Some(obj.id) {
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
        SelectedEntity::Scene => "scene".to_string(),
        SelectedEntity::Project => "project".to_string(),
    }
}

/// Parse a tree node value string into a `SelectedEntity`.
fn parse_value(value: &str) -> Option<SelectedEntity> {
    if let Some(id_str) = value.strip_prefix("obj:") {
        uuid::Uuid::parse_str(id_str).ok().map(SelectedEntity::Object)
    } else if let Some(id_str) = value.strip_prefix("light:") {
        id_str.parse::<u64>().ok().map(SelectedEntity::Light)
    } else {
        match value {
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
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    // Tree state lives as a context (created in editor_ui) so the
    // selection-sync Effect can live outside the render path.
    let tree_state = use_context::<UseTreeReturn>();

    // Header count — derived value, no .set() needed.
    let header_label = Memo::new(move || {
        let count = ui.objects.get().len() + ui.lights.get().len();
        format!("Scene ({count})")
    });

    // NOTE: Selection sync Effect lives in editor_ui() (ui/mod.rs),
    // NOT here. Effects must not .set() signals during render.

    // Selection callback → send command + sync slider values (no lock).
    let _sliders = use_context::<SliderSignals>();
    let onselect = ValueCallback::new(move |value: String| {
        if let Some(entity) = parse_value(&value) {
            let _ = cmd.0.send(EditorCommand::SelectEntity { entity: Some(entity) });
            ui.selection.set(Some(entity));
            ui.properties_tab.set(0); // switch to Object tab
        }
    });

    // Reactive data source for the Tree — rebuilds when objects/lights/scene change.
    let data_source: std::rc::Rc<dyn Fn() -> Vec<TreeNodeData>> =
        std::rc::Rc::new(move || {
            let objects = ui.objects.get();
            let lights = ui.lights.get();
            let scene_name = ui.scene_name.get();
            let scene_path = ui.scene_path.get();
            build_tree_data(&objects, &lights, &scene_name, &scene_path)
        });

    // Initial data for the Tree (read once, data_source handles updates).
    let initial_data = {
        let objects = ui.objects.get();
        let lights = ui.lights.get();
        let scene_name = ui.scene_name.get();
        let scene_path = ui.scene_path.get();
        build_tree_data(&objects, &lights, &scene_name, &scene_path)
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
