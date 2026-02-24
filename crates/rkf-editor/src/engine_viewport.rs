//! Engine rendering state — stubbed pending v2 rewrite.
//!
//! In v2, this will use object-centric rendering with per-object brick maps
//! and BVH acceleration. Will be rewritten starting in Phase 5.

/// Display (output) resolution width.
pub const DISPLAY_WIDTH: u32 = 1280;
/// Display (output) resolution height.
pub const DISPLAY_HEIGHT: u32 = 720;

/// Engine rendering state — placeholder for v2 rewrite.
///
/// In v2, this will hold the GpuScene, BVH, object metadata buffers,
/// and all render passes. Currently a minimal stub.
pub struct EngineState {
    _private: (),
}

/// Build a flat list of `FlatNode`s from all objects in a v2 scene.
///
/// Iterates over every `SceneObject` in `scene.root_objects`, calls
/// `rkf_core::transform_flatten::flatten_object` on each, and concatenates the
/// results into a single `Vec<FlatNode>`. The list is camera-relative: all
/// transforms are expressed relative to `camera_pos`.
///
/// The returned list is suitable for uploading to a `GpuScene` node buffer.
pub fn flatten_v2_scene(
    scene: &rkf_core::scene::Scene,
    camera_pos: rkf_core::WorldPosition,
) -> Vec<rkf_core::transform_flatten::FlatNode> {
    let mut nodes = Vec::new();
    for obj in &scene.root_objects {
        let mut flat = rkf_core::transform_flatten::flatten_object(obj, &camera_pos);
        nodes.append(&mut flat);
    }
    nodes
}

/// Compute the editor viewport dimensions given window size and panel widths.
///
/// Returns `(width, height)` clamped to a minimum of 64 pixels in each
/// dimension. The viewport width is the window width minus `ui_panel_width`;
/// the height is the full window height minus the top/bottom chrome assumed
/// by the editor (62 px top bar + 25 px status bar = 87 px total).
pub fn compute_viewport(width: u32, height: u32, ui_panel_width: u32) -> (u32, u32) {
    const TOP_BOTTOM_CHROME: u32 = 87; // matches EditorState defaults
    let vp_width = width.saturating_sub(ui_panel_width).max(64);
    let vp_height = height.saturating_sub(TOP_BOTTOM_CHROME).max(64);
    (vp_width, vp_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_viewport_basic() {
        let (w, h) = compute_viewport(1280, 720, 552); // 251 left + 301 right
        assert_eq!(w, 1280 - 552);
        assert_eq!(h, 720 - 87);
    }

    #[test]
    fn test_compute_viewport_min_clamp() {
        // Very small window — should clamp to 64
        let (w, h) = compute_viewport(100, 100, 600);
        assert_eq!(w, 64);
        // height: 100 - 87 = 13 < 64, clamped to 64
        assert_eq!(h, 64);
    }

    #[test]
    fn test_compute_viewport_no_panel() {
        let (w, h) = compute_viewport(1920, 1080, 0);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080 - 87);
    }

    #[test]
    fn test_flatten_v2_scene_empty() {
        use rkf_core::{scene::Scene, WorldPosition};

        let scene = Scene::new("empty");
        let nodes = flatten_v2_scene(&scene, WorldPosition::default());
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_flatten_v2_scene_single_object() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("test");
        let node = SceneNode::analytical("sphere", SdfPrimitive::Sphere { radius: 0.5 }, 1);
        scene.add_object("obj1", WorldPosition::default(), node);

        let nodes = flatten_v2_scene(&scene, WorldPosition::default());
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "sphere");
    }

    #[test]
    fn test_flatten_v2_scene_multiple_objects() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("test");
        // Object 1: 1 node
        scene.add_object(
            "obj1",
            WorldPosition::default(),
            SceneNode::analytical("a", SdfPrimitive::Sphere { radius: 0.5 }, 1),
        );
        // Object 2: root + child = 2 nodes
        let mut root = SceneNode::new("root_b");
        root.add_child(SceneNode::analytical("b_child", SdfPrimitive::Sphere { radius: 0.3 }, 2));
        scene.add_object("obj2", WorldPosition::default(), root);

        let nodes = flatten_v2_scene(&scene, WorldPosition::default());
        assert_eq!(nodes.len(), 3); // 1 + 2
    }
}
