//! Panel registry — maps PanelId to rinch component.
//!
//! Each panel type renders its own component. This module provides the
//! dispatch function that the zone component calls.

use rinch::prelude::*;

use super::PanelId;

/// Render the content of a panel by its ID.
///
/// Returns a `NodeHandle` for the panel's component tree.
pub fn render_panel(scope: &mut RenderScope, panel: PanelId) -> NodeHandle {
    match panel {
        PanelId::SceneTree => {
            let c = crate::ui::scene_tree_panel::SceneTreePanel::default();
            c.render(scope, &[])
        }
        PanelId::EditorCamera => {
            let c = crate::ui::camera_properties::EditorCameraPanel::default();
            c.render(scope, &[])
        }
        PanelId::ObjectProperties => {
            let c = crate::ui::right_panel::PropertiesPanel::default();
            c.render(scope, &[])
        }
        PanelId::AssetProperties => {
            let c = crate::ui::right_panel::AssetPropertiesPanel::default();
            c.render(scope, &[])
        }
        PanelId::Materials => {
            let c = crate::ui::materials_panel::MaterialsPanel::default();
            c.render(scope, &[])
        }
        PanelId::Shaders => {
            let c = crate::ui::shaders_panel::ShadersPanel::default();
            c.render(scope, &[])
        }
        PanelId::SceneView => {
            // Scene view is an empty transparent placeholder — the actual
            // RenderSurface lives at the editor_ui level (absolute-positioned)
            // so it's never destroyed by reactive scope rebuilds.
            let __scope = scope;
            rsx! { div { style: "flex:1;min-height:0;" } }
        }
        PanelId::Systems => {
            let c = crate::ui::systems_panel::SystemsPanel::default();
            c.render(scope, &[])
        }
        PanelId::Library => {
            let c = crate::ui::library_panel::LibraryPanel::default();
            c.render(scope, &[])
        }
        PanelId::Models => {
            let c = crate::ui::models_panel::ModelsPanel::default();
            c.render(scope, &[])
        }
        PanelId::Console => {
            let c = crate::ui::debug_panel::DebugPanel::default();
            c.render(scope, &[])
        }
        PanelId::GameView | PanelId::AnimationEditor => {
            placeholder(scope, panel)
        }
    }
}

fn placeholder(scope: &mut RenderScope, panel: PanelId) -> NodeHandle {
    let __scope = scope;
    rsx! {
        div {
            style: "flex:1;min-height:0;padding:12px;color:var(--rinch-color-dimmed);",
            {format!("{} (coming soon)", panel.display_name())}
        }
    }
}
