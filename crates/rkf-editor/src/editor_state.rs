//! Central editor state, aggregating all data model modules.
//!
//! `EditorState` holds instances of every editor subsystem. It is shared via
//! `Arc<Mutex<EditorState>>` between the winit event loop and the rinch component tree.

use crate::animation_preview::AnimationPreview;
use crate::camera::{CameraMode, EditorCamera};
use crate::debug_viz::{DebugOverlay, FrameTimeHistory};
use crate::environment::EnvironmentState;
use crate::gizmo::GizmoState;
use crate::input::InputState;
use crate::light_editor::LightEditor;
use crate::overlay::OverlayConfig;
use crate::paint::PaintState;
use crate::placement::{AssetBrowser, GridSnap, PlacementQueue};
use crate::properties::PropertySheet;
use crate::scene_io::{RecentFiles, UnsavedChangesState};
use crate::scene_tree::SceneTree;
use crate::sculpt::SculptState;
use crate::undo::UndoStack;

use glam::Vec3;
use rinch::prelude::Signal;

/// What is currently selected in the scene tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedEntity {
    /// An SDF object by entity ID.
    Object(u64),
    /// A light by light ID.
    Light(u64),
    /// The main editor camera.
    Camera,
    /// The scene node itself.
    Scene,
    /// The project root node.
    Project,
}

/// Shared reactive revision counter for triggering UI re-renders.
///
/// Stored in rinch context — any component can `use_context::<UiRevision>()`
/// and track `signal.get()` inside a `reactive_component_dom` Effect.
/// Bumped by scene tree clicks, viewport picks, and other state mutations.
#[derive(Clone, Copy)]
pub struct UiRevision(pub Signal<u64>);

impl UiRevision {
    /// Create a new revision counter starting at 0.
    pub fn new() -> Self {
        Self(Signal::new(0u64))
    }

    /// Bump the revision to trigger UI re-renders.
    pub fn bump(&self) {
        self.0.update(|r| *r += 1);
    }

    /// Read the revision value (creates a reactive tracking dependency).
    pub fn track(&self) {
        let _ = self.0.get();
    }
}

/// Region of the window occupied by the engine viewport (excludes UI panels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportRect {
    /// Left edge in window pixels.
    pub x: u32,
    /// Top edge in window pixels.
    pub y: u32,
    /// Viewport width in pixels.
    pub width: u32,
    /// Viewport height in pixels.
    pub height: u32,
}

impl Default for ViewportRect {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 1280,
            height: 720,
        }
    }
}

/// Editor tool mode — determines viewport interaction behaviour.
///
/// Navigation (orbit/fly) and selection (click-to-pick) are always active
/// regardless of mode. Only Sculpt and Paint change how mouse drags in the
/// viewport are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditorMode {
    /// Default mode — navigate + select always active, no brush tool.
    #[default]
    Default,
    /// CSG brush strokes on voxelized objects.
    Sculpt,
    /// Material/color painting on object surfaces.
    Paint,
}

impl EditorMode {
    /// Tool modes that appear as toolbar toggle buttons.
    pub const TOOLS: [EditorMode; 2] = [Self::Sculpt, Self::Paint];

    /// Display name for the status bar / toolbar.
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Sculpt => "Sculpt",
            Self::Paint => "Paint",
        }
    }
}

/// Aggregated editor state, shared between the event loop and the UI.
pub struct EditorState {
    // ── Mode ─────────────────────────────────────────────────
    pub mode: EditorMode,

    // ── Camera & Input ───────────────────────────────────────
    pub editor_camera: EditorCamera,
    pub editor_input: InputState,

    // ── Scene ────────────────────────────────────────────────
    pub scene_tree: SceneTree,
    pub selected_entity: Option<SelectedEntity>,
    pub selected_properties: Option<PropertySheet>,

    // ── Gizmo ────────────────────────────────────────────────
    pub gizmo: GizmoState,

    // ── Tool states ──────────────────────────────────────────
    pub sculpt: SculptState,
    pub paint: PaintState,
    pub placement_queue: PlacementQueue,
    pub asset_browser: AssetBrowser,
    pub grid_snap: GridSnap,

    // ── Lights & Environment ─────────────────────────────────
    pub light_editor: LightEditor,
    pub environment: EnvironmentState,

    // ── Animation ────────────────────────────────────────────
    pub animation: AnimationPreview,

    // ── Overlays & Debug ─────────────────────────────────────
    pub overlay_config: OverlayConfig,
    pub debug_viz: DebugOverlay,
    pub frame_time_history: FrameTimeHistory,

    // ── Undo/Redo ────────────────────────────────────────────
    pub undo: UndoStack,

    // ── Scene I/O ────────────────────────────────────────────
    pub unsaved_changes: UnsavedChangesState,
    pub recent_files: RecentFiles,
    pub current_scene_path: Option<String>,

    // ── Viewport layout ──────────────────────────────────────
    /// Current viewport area (engine output region).
    pub viewport: ViewportRect,
    /// Left panel width in pixels.
    pub left_panel_width: u32,
    /// Right panel width in pixels.
    pub right_panel_width: u32,
    /// Top bar height in pixels (menu + toolbar).
    pub top_bar_height: u32,
    /// Bottom bar height in pixels (status bar).
    pub bottom_bar_height: u32,

    // ── Pending commands (UI → render loop) ──────────────────
    /// Set by UI menus, consumed by the render loop.
    pub pending_debug_mode: Option<u32>,
    /// Current debug visualization mode (0=normal, 1-6=debug).
    pub debug_mode: u32,
    /// Set by File > Quit, consumed by the event loop.
    pub wants_exit: bool,
    /// Set by File > Open, consumed by the event loop.
    pub pending_open: bool,
    /// Set by File > Save, consumed by the event loop.
    pub pending_save: bool,
    /// Set by File > Save As, consumed by the event loop.
    pub pending_save_as: bool,
    /// Set by Edit > Spawn, consumed by the event loop. Value is the primitive name.
    pub pending_spawn: Option<String>,
    /// Set by Delete key, consumed by the event loop.
    pub pending_delete: bool,
    /// Set by Ctrl+D, consumed by the event loop.
    pub pending_duplicate: bool,
    /// Set by Edit > Undo, consumed by the event loop.
    pub pending_undo: bool,
    /// Set by Edit > Redo, consumed by the event loop.
    pub pending_redo: bool,
    /// Whether the ground grid overlay is visible (toggled via View menu).
    pub show_grid: bool,
    /// Whether the shortcut reference overlay is visible (F1 toggle).
    pub show_shortcuts: bool,
    /// Set by titlebar drag handler, consumed by the event loop.
    pub pending_drag: bool,
    /// Set by minimize window control, consumed by the event loop.
    pub pending_minimize: bool,
    /// Set by maximize window control, consumed by the event loop.
    pub pending_maximize: bool,
    /// Set by "Re-voxelize" button in properties panel. Contains the object ID
    /// of the voxelized object whose brick map should be resampled at the
    /// current non-uniform scale, resetting scale to (1,1,1) afterwards.
    pub pending_revoxelize: Option<u32>,

    // ── v2 Scene model ───────────────────────────────────────
    /// Optional v2 scene reference. When set, `sync_v2_scene` mirrors it
    /// into the editor scene tree.
    pub v2_scene: Option<rkf_core::scene::Scene>,
}

impl EditorState {
    /// Create a new editor state with default values and fly-mode camera
    /// positioned to match the engine's initial viewpoint.
    pub fn new() -> Self {
        let mut cam = EditorCamera::new();
        cam.mode = CameraMode::Fly;
        cam.position = Vec3::new(0.0, 2.5, 5.0);
        cam.fly_yaw = 0.0;
        cam.fly_pitch = -0.15;
        cam.fov_y = 70.0_f32.to_radians();
        cam.fly_speed = 5.0;

        Self {
            mode: EditorMode::Default,
            editor_camera: cam,
            editor_input: InputState::new(),
            scene_tree: SceneTree::new(),
            selected_entity: None,
            selected_properties: None,
            gizmo: GizmoState::new(),
            sculpt: SculptState::new(),
            paint: PaintState::new(),
            placement_queue: PlacementQueue::new(),
            asset_browser: AssetBrowser::new(),
            grid_snap: GridSnap::default(),
            light_editor: LightEditor::new(),
            environment: {
                let mut env = EnvironmentState::new();
                env.mark_dirty(); // Ensure first frame applies defaults to engine
                env
            },
            animation: AnimationPreview::new(),
            overlay_config: OverlayConfig::default(),
            debug_viz: DebugOverlay::new(),
            frame_time_history: FrameTimeHistory::new(),
            undo: UndoStack::new(100),
            unsaved_changes: UnsavedChangesState::new(),
            recent_files: RecentFiles::new(),
            current_scene_path: None,
            viewport: ViewportRect::default(),
            left_panel_width: 251,  // 250px content + 1px border-right
            right_panel_width: 301, // 300px content + 1px border-left
            top_bar_height: 37,    // titlebar 36px + 1px border
            bottom_bar_height: 25, // status bar 24px + 1px border-top
            pending_debug_mode: None,
            debug_mode: 0,
            wants_exit: false,
            pending_open: false,
            pending_save: false,
            pending_save_as: false,
            pending_spawn: None,
            pending_delete: false,
            pending_duplicate: false,
            pending_undo: false,
            pending_redo: false,
            show_grid: false,
            show_shortcuts: false,
            pending_drag: false,
            pending_minimize: false,
            pending_maximize: false,
            pending_revoxelize: None,
            v2_scene: None,
        }
    }

    /// Recompute the viewport rect from current panel sizes and window dimensions.
    ///
    /// Returns `true` if the viewport changed (callers should rebuild GPU passes).
    pub fn compute_viewport(&mut self, window_width: u32, window_height: u32) -> bool {
        let new = ViewportRect {
            x: self.left_panel_width,
            y: self.top_bar_height,
            width: window_width
                .saturating_sub(self.left_panel_width + self.right_panel_width)
                .max(64),
            height: window_height
                .saturating_sub(self.top_bar_height + self.bottom_bar_height)
                .max(64),
        };
        let changed = new != self.viewport;
        self.viewport = new;
        changed
    }

    /// Update camera from current input state.
    ///
    /// This method exists to work around Rust's borrow checker: calling
    /// `self.editor_camera.update(&self.editor_input, dt)` through a
    /// `MutexGuard` doesn't allow simultaneous mutable + immutable borrows,
    /// but a method on `Self` can borrow separate fields.
    pub fn update_camera(&mut self, dt: f32) {
        self.editor_camera.update(&self.editor_input, dt);
    }

    /// Sync editor camera state to an engine `Camera`.
    ///
    /// Copies position, orientation, and FOV from the `EditorCamera` to
    /// the render engine's `Camera` struct.
    pub fn sync_to_engine_camera(&self, engine_cam: &mut rkf_render::camera::Camera) {
        engine_cam.position = self.editor_camera.position;
        engine_cam.yaw = self.editor_camera.fly_yaw;
        engine_cam.pitch = self.editor_camera.fly_pitch;
        engine_cam.fov_degrees = self.editor_camera.fov_y.to_degrees();
    }

    /// Name of the current debug visualization mode (empty for normal shading).
    pub fn debug_mode_name(&self) -> &'static str {
        match self.debug_mode {
            0 => "",
            1 => "Normals",
            2 => "Positions",
            3 => "Material IDs",
            4 => "Diffuse",
            5 => "Specular",
            6 => "GI Only",
            _ => "Debug",
        }
    }

    /// Reset per-frame input deltas (mouse delta, scroll) after processing.
    pub fn reset_frame_deltas(&mut self) {
        self.editor_input.mouse_delta = glam::Vec2::ZERO;
        self.editor_input.scroll_delta = 0.0;
        self.editor_input.keys_just_pressed.clear();
    }

    /// Mirror the v2 scene into the editor scene tree.
    ///
    /// If `v2_scene` is `Some`, calls [`crate::scene_tree::sync_from_v2_scene`]
    /// to rebuild `scene_tree.roots` from the v2 object list. Does nothing when
    /// `v2_scene` is `None`.
    pub fn sync_v2_scene(&mut self) {
        if let Some(ref scene) = self.v2_scene {
            crate::scene_tree::sync_from_v2_scene(&mut self.scene_tree, scene);
        }
    }

    /// Pick a scene object via ray-AABB intersection (CPU fallback).
    ///
    /// Returns the nearest hit object's id, or `None`. This is a rough
    /// approximation — the primary pick path uses GPU readback from the
    /// material G-buffer (see `pending_pick`/`pick_result` in SharedState).
    pub fn pick_object_aabb(
        &self,
        pixel_x: f32,
        pixel_y: f32,
        vp_width: f32,
        vp_height: f32,
    ) -> Option<u64> {
        let (ray_o, ray_d) = crate::camera::screen_to_ray(
            &self.editor_camera,
            pixel_x,
            pixel_y,
            vp_width,
            vp_height,
        );

        let scene = self.v2_scene.as_ref()?;
        let world_transforms = rkf_core::transform_bake::bake_world_transforms(&scene.objects);
        let default_wt = rkf_core::transform_bake::WorldTransform::default();

        let mut best_t = f32::MAX;
        let mut best_id = None;

        for obj in &scene.objects {
            let wt = world_transforms.get(&obj.id).unwrap_or(&default_wt);
            let smin = obj.aabb.min * wt.scale;
            let smax = obj.aabb.max * wt.scale;
            let corners = [
                glam::Vec3::new(smin.x, smin.y, smin.z), glam::Vec3::new(smax.x, smin.y, smin.z),
                glam::Vec3::new(smin.x, smax.y, smin.z), glam::Vec3::new(smax.x, smax.y, smin.z),
                glam::Vec3::new(smin.x, smin.y, smax.z), glam::Vec3::new(smax.x, smin.y, smax.z),
                glam::Vec3::new(smin.x, smax.y, smax.z), glam::Vec3::new(smax.x, smax.y, smax.z),
            ];
            let mut wmin = glam::Vec3::splat(f32::MAX);
            let mut wmax = glam::Vec3::splat(f32::MIN);
            for c in &corners {
                let r = wt.rotation * *c + wt.position;
                wmin = wmin.min(r);
                wmax = wmax.max(r);
            }
            if let Some(t) = ray_aabb_distance(ray_o, ray_d, wmin, wmax) {
                if t < best_t {
                    best_t = t;
                    best_id = Some(obj.id as u64);
                }
            }
        }

        best_id
    }

    /// Load a scene file and populate the editor state from it.
    ///
    /// Populates the scene tree with entities from the file and applies
    /// environment settings if present. Returns the loaded `SceneFile`
    /// for further processing (e.g. setting up engine geometry).
    pub fn load_scene(&mut self, path: &str) -> Result<crate::scene_io::SceneFile, String> {
        use crate::scene_io::{load_scene_from_path, ComponentData};
        use crate::scene_tree::SceneNode;

        let scene = load_scene_from_path(path)?;

        // Clear existing scene tree
        self.scene_tree = crate::scene_tree::SceneTree::new();

        // Populate scene tree from entities.
        // Light-only entities are handled exclusively by the light editor below;
        // skip them here to avoid double-counting on save.
        for entity in &scene.entities {
            let is_light_only = !entity.components.is_empty()
                && entity
                    .components
                    .iter()
                    .all(|c| matches!(c, ComponentData::Light { .. }));
            if is_light_only {
                continue;
            }

            let mut node = SceneNode::new(entity.entity_id, &entity.name);
            node.visible = true;
            node.position = entity.position;
            node.rotation = entity.rotation;
            node.scale = entity.scale;
            for comp in &entity.components {
                if let ComponentData::SdfObject { asset_path } = comp {
                    node.asset_path = Some(asset_path.clone());
                }
            }
            self.scene_tree.add_node(node);
        }

        // Rebuild parent-child relationships (only for non-light entities)
        for entity in &scene.entities {
            let is_light_only = !entity.components.is_empty()
                && entity
                    .components
                    .iter()
                    .all(|c| matches!(c, ComponentData::Light { .. }));
            if is_light_only {
                continue;
            }
            if let Some(parent_id) = entity.parent_id {
                // Remove from roots and re-add as child
                if let Some(child) = self.scene_tree.remove_node(entity.entity_id) {
                    if let Some(parent) = self.scene_tree.find_node_mut(parent_id) {
                        parent.children.push(child);
                    } else {
                        // Parent not found — keep as root
                        self.scene_tree.add_node(child);
                    }
                }
            }
        }

        // Populate light editor from Light components
        self.light_editor = crate::light_editor::LightEditor::new();
        for entity in &scene.entities {
            for comp in &entity.components {
                if let ComponentData::Light {
                    light_type,
                    color,
                    intensity,
                    range,
                } = comp
                {
                    use crate::light_editor::EditorLightType;
                    let lt = match light_type.as_str() {
                        "spot" => EditorLightType::Spot,
                        _ => EditorLightType::Point,
                    };
                    let id = self.light_editor.add_light(lt);
                    self.light_editor.set_position(id, entity.position);
                    self.light_editor.set_color(
                        id,
                        Vec3::new(color[0], color[1], color[2]),
                    );
                    self.light_editor.set_intensity(id, *intensity);
                    if *range > 0.0 {
                        self.light_editor.set_range(id, *range);
                    }
                }
            }
        }

        // Apply environment settings if present
        if !scene.environment_ron.is_empty() {
            if let Ok(env) =
                crate::environment::EnvironmentState::deserialize_from_ron(&scene.environment_ron)
            {
                self.environment = env;
                self.environment.mark_dirty();
            }
        }

        // Track the current scene path
        self.current_scene_path = Some(path.to_string());
        self.unsaved_changes.mark_saved();

        Ok(scene)
    }

    /// Construct a [`SceneFile`] from the current editor state.
    ///
    /// Walks the scene tree to collect SDF/animated/physics entities (with their
    /// stored transforms and asset paths), then appends light entities from the
    /// light editor. The environment is serialized via RON. The resulting
    /// `SceneFile` can be passed to [`crate::scene_io::save_scene_to_path`].
    pub fn save_current_scene(&self) -> crate::scene_io::SceneFile {
        use crate::light_editor::EditorLightType;
        use crate::scene_io::{ComponentData, SceneEntity, SceneFile};
        use crate::scene_tree::SceneNode;
        use glam::Quat;

        let mut entities: Vec<SceneEntity> = Vec::new();

        // Recursively collect scene-tree nodes into flat entity list.
        fn collect_nodes(
            node: &SceneNode,
            parent_id: Option<u64>,
            out: &mut Vec<SceneEntity>,
        ) {
            let mut components = Vec::new();
            if let Some(ref path) = node.asset_path {
                components.push(ComponentData::SdfObject {
                    asset_path: path.clone(),
                });
            }
            out.push(SceneEntity {
                entity_id: node.entity_id,
                name: node.name.clone(),
                parent_id,
                position: node.position,
                rotation: node.rotation,
                scale: node.scale,
                components,
            });
            for child in &node.children {
                collect_nodes(child, Some(node.entity_id), out);
            }
        }

        for root in &self.scene_tree.roots {
            collect_nodes(root, None, &mut entities);
        }

        // Append light entities from the light editor.
        for (idx, light) in self.light_editor.all_lights().iter().enumerate() {
            let light_type_str = match light.light_type {
                EditorLightType::Point => "point",
                EditorLightType::Spot => "spot",
            };
            let name = format!(
                "{} Light {}",
                match light.light_type {
                    EditorLightType::Point => "Point",
                    EditorLightType::Spot => "Spot",
                },
                idx + 1
            );
            let range = if light.range.is_infinite() {
                0.0
            } else {
                light.range
            };
            entities.push(SceneEntity {
                entity_id: light.id,
                name,
                parent_id: None,
                position: light.position,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
                components: vec![ComponentData::Light {
                    light_type: light_type_str.to_string(),
                    color: [light.color.x, light.color.y, light.color.z],
                    intensity: light.intensity,
                    range,
                }],
            });
        }

        let environment_ron = self.environment.serialize_to_ron().unwrap_or_default();

        let name = self
            .current_scene_path
            .as_ref()
            .and_then(|p| std::path::Path::new(p).file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".to_string());

        SceneFile {
            version: 1,
            name,
            entities,
            environment_ron,
        }
    }
}

/// Ray-AABB intersection returning the entry distance along the ray.
///
/// Returns `Some(t)` where `t >= 0` if the ray hits the box, `None` otherwise.
fn ray_aabb_distance(ray_o: Vec3, ray_d: Vec3, min: Vec3, max: Vec3) -> Option<f32> {
    let inv_d = Vec3::new(
        if ray_d.x.abs() > 1e-8 { 1.0 / ray_d.x } else { f32::MAX.copysign(ray_d.x) },
        if ray_d.y.abs() > 1e-8 { 1.0 / ray_d.y } else { f32::MAX.copysign(ray_d.y) },
        if ray_d.z.abs() > 1e-8 { 1.0 / ray_d.z } else { f32::MAX.copysign(ray_d.z) },
    );
    let t1 = (min - ray_o) * inv_d;
    let t2 = (max - ray_o) * inv_d;
    let t_min = t1.min(t2);
    let t_max = t1.max(t2);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    if t_exit >= t_enter.max(0.0) {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_io::{save_scene, ComponentData, SceneEntity, SceneFile};
    use glam::{Quat, Vec3};

    fn write_test_scene(path: &str) {
        let scene = SceneFile {
            version: 1,
            name: "Test".to_string(),
            entities: vec![
                SceneEntity {
                    entity_id: 1,
                    name: "Ground".to_string(),
                    parent_id: None,
                    position: Vec3::new(0.0, -0.5, 0.0),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                    components: vec![ComponentData::SdfObject {
                        asset_path: "procedural://ground".to_string(),
                    }],
                },
                SceneEntity {
                    entity_id: 2,
                    name: "Child".to_string(),
                    parent_id: Some(1),
                    position: Vec3::new(1.0, 2.0, 3.0),
                    rotation: Quat::from_rotation_y(1.0),
                    scale: Vec3::splat(0.5),
                    components: vec![ComponentData::SdfObject {
                        asset_path: "procedural://pillar".to_string(),
                    }],
                },
                SceneEntity {
                    entity_id: 3,
                    name: "Sun".to_string(),
                    parent_id: None,
                    position: Vec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                    components: vec![ComponentData::Light {
                        light_type: "directional".to_string(),
                        color: [1.0, 0.95, 0.8],
                        intensity: 3.0,
                        range: 0.0,
                    }],
                },
            ],
            environment_ron: String::new(),
        };
        let ron_str = save_scene(&scene).unwrap();
        std::fs::write(path, ron_str).unwrap();
    }

    #[test]
    fn test_load_scene_populates_scene_tree() {
        let path = "/tmp/rkf_test_load_tree.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        let scene = state.load_scene(path).unwrap();
        assert_eq!(scene.entities.len(), 3);
        // Only SDF entities go in scene tree (lights go to light_editor)
        assert_eq!(state.scene_tree.roots.len(), 1); // "Ground" root
        assert_eq!(state.scene_tree.roots[0].name, "Ground");
        // "Child" is reparented under "Ground"
        assert_eq!(state.scene_tree.roots[0].children.len(), 1);
        assert_eq!(state.scene_tree.roots[0].children[0].name, "Child");
    }

    #[test]
    fn test_load_scene_stores_transform_and_asset_path() {
        let path = "/tmp/rkf_test_load_xform.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.load_scene(path).unwrap();
        let ground = &state.scene_tree.roots[0];
        assert_eq!(ground.position, Vec3::new(0.0, -0.5, 0.0));
        assert_eq!(
            ground.asset_path.as_deref(),
            Some("procedural://ground")
        );
        let child = &ground.children[0];
        assert_eq!(child.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(child.scale.abs_diff_eq(Vec3::splat(0.5), 1e-6));
        assert_eq!(
            child.asset_path.as_deref(),
            Some("procedural://pillar")
        );
    }

    #[test]
    fn test_load_scene_populates_light_editor() {
        let path = "/tmp/rkf_test_load_lights.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.load_scene(path).unwrap();
        assert_eq!(state.light_editor.all_lights().len(), 1);
        let light = &state.light_editor.all_lights()[0];
        assert_eq!(light.intensity, 3.0);
    }

    #[test]
    fn test_load_scene_sets_path_and_clears_dirty() {
        let path = "/tmp/rkf_test_load_path.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.unsaved_changes.mark_changed();
        state.load_scene(path).unwrap();
        assert_eq!(state.current_scene_path.as_deref(), Some(path));
        assert!(!state.unsaved_changes.needs_save());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let mut state = EditorState::new();
        let result = state.load_scene("/nonexistent/path.rkscene");
        assert!(result.is_err());
    }

    #[test]
    fn test_save_empty_scene() {
        let state = EditorState::new();
        let saved = state.save_current_scene();
        assert_eq!(saved.version, 1);
        assert_eq!(saved.name, "Untitled");
        assert!(saved.entities.is_empty());
    }

    #[test]
    fn test_save_roundtrip_preserves_entities() {
        let path = "/tmp/rkf_test_save_rt.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.load_scene(path).unwrap();

        let saved = state.save_current_scene();
        assert_eq!(saved.version, 1);
        // 2 SDF entities (Ground + Child) + 1 light = 3
        assert_eq!(saved.entities.len(), 3);

        // Check SDF entities
        let ground = saved.entities.iter().find(|e| e.name == "Ground").unwrap();
        assert_eq!(ground.position, Vec3::new(0.0, -0.5, 0.0));
        assert!(ground.components.iter().any(|c| matches!(
            c,
            ComponentData::SdfObject { asset_path } if asset_path == "procedural://ground"
        )));

        let child = saved.entities.iter().find(|e| e.name == "Child").unwrap();
        assert_eq!(child.parent_id, Some(1));
        assert!(child.scale.abs_diff_eq(Vec3::splat(0.5), 1e-6));

        // Check light entity
        let light_ent = saved
            .entities
            .iter()
            .find(|e| e.components.iter().any(|c| matches!(c, ComponentData::Light { .. })))
            .unwrap();
        assert!(light_ent
            .components
            .iter()
            .any(|c| matches!(c, ComponentData::Light { intensity, .. } if (*intensity - 3.0).abs() < 1e-6)));
    }

    #[test]
    fn test_v2_scene_field_defaults_to_none() {
        let state = EditorState::new();
        assert!(state.v2_scene.is_none());
    }

    #[test]
    fn test_sync_v2_scene_populates_tree() {
        use rkf_core::scene::Scene;
        use rkf_core::scene_node::SceneNode as CoreNode;
        let mut state = EditorState::new();
        let mut v2 = Scene::new("test");
        v2.add_object("SphereObj", Vec3::ZERO, CoreNode::new("root"));
        v2.add_object("BoxObj", Vec3::ZERO, CoreNode::new("root2"));

        state.v2_scene = Some(v2);
        state.sync_v2_scene();

        assert_eq!(state.scene_tree.roots.len(), 2);
        assert_eq!(state.scene_tree.roots[0].name, "SphereObj");
        assert_eq!(state.scene_tree.roots[1].name, "BoxObj");
    }

    #[test]
    fn test_sync_v2_scene_no_op_when_none() {
        let mut state = EditorState::new();
        // Add a node manually
        use crate::scene_tree::SceneNode;
        state.scene_tree.add_node(SceneNode::new(1, "ManualNode"));

        // sync with no v2_scene — tree should be unchanged
        state.sync_v2_scene();
        assert_eq!(state.scene_tree.roots.len(), 1);
        assert_eq!(state.scene_tree.roots[0].name, "ManualNode");
    }

    #[test]
    fn test_save_scene_name_from_path() {
        let path = "/tmp/rkf_test_save_name.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.load_scene(path).unwrap();
        let saved = state.save_current_scene();
        assert_eq!(saved.name, "rkf_test_save_name");
    }
}
