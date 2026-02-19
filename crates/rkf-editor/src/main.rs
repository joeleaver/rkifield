#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod camera;
mod gizmo;
mod input;
mod paint;
mod placement;
mod properties;
mod scene_tree;
mod sculpt;

use rinch::prelude::*;

#[component]
fn editor_app() -> NodeHandle {
    rsx! {
        div {
            style: "display: flex; flex-direction: column; width: 100%; height: 100%;",
            // Menu bar
            div {
                style: "display: flex; flex-direction: row; height: 32px; background: #2b2b2b; align-items: center; padding: 0 8px; gap: 12px;",
                span { style: "color: #aaa; font-weight: bold;", "RKIField Editor" }
                span { style: "color: #888; cursor: pointer;", "File" }
                span { style: "color: #888; cursor: pointer;", "Edit" }
                span { style: "color: #888; cursor: pointer;", "View" }
                span { style: "color: #888; cursor: pointer;", "Tools" }
            }
            // Main content area
            div {
                style: "display: flex; flex-direction: row; flex: 1; overflow: hidden;",
                // Left panel (scene hierarchy)
                div {
                    style: "width: 250px; background: #1e1e1e; border-right: 1px solid #333; overflow-y: auto; padding: 8px;",
                    span { style: "color: #ccc; font-weight: bold;", "Scene Hierarchy" }
                }
                // Viewport
                div {
                    style: "flex: 1; background: #111; display: flex; align-items: center; justify-content: center;",
                    span { style: "color: #555;", "Viewport (engine renders here)" }
                }
                // Right panel (properties)
                div {
                    style: "width: 300px; background: #1e1e1e; border-left: 1px solid #333; overflow-y: auto; padding: 8px;",
                    span { style: "color: #ccc; font-weight: bold;", "Properties" }
                }
            }
            // Bottom panel (status bar)
            div {
                style: "height: 24px; background: #2b2b2b; display: flex; align-items: center; padding: 0 8px;",
                span { style: "color: #666; font-size: 12px;", "Ready" }
            }
        }
    }
}

fn main() {
    env_logger::init();
    rinch::run("RKIField Editor", 1280, 720, editor_app);
}
