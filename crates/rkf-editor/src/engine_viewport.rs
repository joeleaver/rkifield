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

    #[test]
    fn test_flatten_v2_scene_multiple_objects_different_positions() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("test");
        // Object 1 at origin
        scene.add_object(
            "obj1",
            WorldPosition::default(),
            SceneNode::analytical("sphere1", SdfPrimitive::Sphere { radius: 0.5 }, 1),
        );
        // Object 2 at chunk (1, 0, 0), local (5.0, 5.0, 5.0)
        let pos2 = WorldPosition {
            chunk: glam::IVec3::new(1, 0, 0),
            local: glam::Vec3::new(5.0, 5.0, 5.0),
        };
        scene.add_object(
            "obj2",
            pos2,
            SceneNode::analytical("sphere2", SdfPrimitive::Sphere { radius: 0.3 }, 2),
        );
        // Object 3 at chunk (-1, 2, 1), local (-3.0, 1.0, 7.0)
        let pos3 = WorldPosition {
            chunk: glam::IVec3::new(-1, 2, 1),
            local: glam::Vec3::new(-3.0, 1.0, 7.0),
        };
        scene.add_object(
            "obj3",
            pos3,
            SceneNode::analytical("sphere3", SdfPrimitive::Sphere { radius: 0.1 }, 3),
        );

        let camera_pos = WorldPosition::default();
        let nodes = flatten_v2_scene(&scene, camera_pos);
        assert_eq!(nodes.len(), 3);
        // Verify distinct names are present
        let names: Vec<_> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"sphere1"));
        assert!(names.contains(&"sphere2"));
        assert!(names.contains(&"sphere3"));
    }

    #[test]
    fn test_flatten_v2_scene_nested_hierarchy() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("test");
        // Build a multi-level hierarchy: root -> child -> grandchild
        let mut root = SceneNode::new("root");
        let mut child = SceneNode::new("child");
        child.add_child(SceneNode::analytical(
            "grandchild",
            SdfPrimitive::Sphere { radius: 0.1 },
            3,
        ));
        root.add_child(child);

        scene.add_object("obj_hierarchy", WorldPosition::default(), root);

        let nodes = flatten_v2_scene(&scene, WorldPosition::default());
        // Should have 3 nodes: root, child, grandchild
        assert_eq!(nodes.len(), 3);
        let names: Vec<_> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"root"));
        assert!(names.contains(&"child"));
        assert!(names.contains(&"grandchild"));
    }

    #[test]
    fn test_compute_viewport_zero_width() {
        let (w, h) = compute_viewport(0, 720, 0);
        // Width should clamp to minimum 64
        assert_eq!(w, 64);
        // Height: 720 - 87 = 633
        assert_eq!(h, 633);
    }

    #[test]
    fn test_compute_viewport_zero_height() {
        let (w, h) = compute_viewport(1280, 0, 0);
        // Width: 1280 - 0 = 1280
        assert_eq!(w, 1280);
        // Height should clamp to minimum 64 (0 - 87 saturates to 0, then clamped)
        assert_eq!(h, 64);
    }

    #[test]
    fn test_compute_viewport_zero_dimensions() {
        let (w, h) = compute_viewport(0, 0, 0);
        // Both should clamp to minimum 64
        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_compute_viewport_very_large_dimensions() {
        let (w, h) = compute_viewport(8000, 6000, 500);
        // Width: 8000 - 500 = 7500
        assert_eq!(w, 7500);
        // Height: 6000 - 87 = 5913
        assert_eq!(h, 5913);
    }

    #[test]
    fn test_compute_viewport_large_ui_panel() {
        let (w, h) = compute_viewport(2000, 1500, 1900);
        // Width: 2000 - 1900 = 100
        assert_eq!(w, 100);
        // Height: 1500 - 87 = 1413
        assert_eq!(h, 1413);
    }

    #[test]
    fn test_viewport_aspect_ratio_standard() {
        let (w, h) = compute_viewport(1920, 1080, 400);
        // Standard 16:9 aspect
        let aspect = w as f32 / h as f32;
        // 1520 / 993 ≈ 1.53, roughly 16:9 (1.777...)
        assert!(aspect > 1.0);
        assert!(aspect < 2.0);
    }

    #[test]
    fn test_viewport_aspect_ratio_extreme_narrow() {
        let (w, h) = compute_viewport(200, 1200, 100);
        // Very narrow viewport
        let aspect = w as f32 / h as f32;
        // 100 / 1113 ≈ 0.09
        assert!(aspect < 0.2);
    }

    #[test]
    fn test_viewport_aspect_ratio_extreme_wide() {
        let (w, h) = compute_viewport(5000, 200, 100);
        // Very wide viewport
        let aspect = w as f32 / h as f32;
        // 4900 / 113 ≈ 43.36
        assert!(aspect > 20.0);
    }

    #[test]
    fn test_flatten_v2_scene_deeply_nested() {
        use rkf_core::{
            scene::Scene,
            scene_node::{SceneNode, SdfPrimitive},
            WorldPosition,
        };

        let mut scene = Scene::new("test");
        // Build a deeply nested hierarchy (5 levels)
        let mut level0 = SceneNode::new("level0");
        let mut level1 = SceneNode::new("level1");
        let mut level2 = SceneNode::new("level2");
        let mut level3 = SceneNode::new("level3");
        let level4 = SceneNode::analytical("level4_leaf", SdfPrimitive::Sphere { radius: 0.05 }, 5);

        level3.add_child(level4);
        level2.add_child(level3);
        level1.add_child(level2);
        level0.add_child(level1);

        scene.add_object("deep_obj", WorldPosition::default(), level0);

        let nodes = flatten_v2_scene(&scene, WorldPosition::default());
        // Should have all 5 nodes
        assert_eq!(nodes.len(), 5);
        let names: Vec<_> = nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"level0"));
        assert!(names.contains(&"level1"));
        assert!(names.contains(&"level2"));
        assert!(names.contains(&"level3"));
        assert!(names.contains(&"level4_leaf"));
    }
}
