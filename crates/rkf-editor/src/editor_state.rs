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

/// Editor tool mode — determines viewport interaction, right-panel content,
/// and active keyboard shortcuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    /// Orbit/fly camera only.
    Navigate,
    /// Pick entities, transform gizmo.
    Select,
    /// Instantiate from asset browser.
    Place,
    /// CSG brush strokes.
    Sculpt,
    /// Material/color painting.
    Paint,
    /// Place and tune lights.
    Light,
    /// Preview animations, pose.
    Animate,
    /// Tune fog/clouds/atmosphere/post-process.
    Environment,
}

impl EditorMode {
    /// All modes in toolbar order.
    pub const ALL: [EditorMode; 8] = [
        Self::Navigate,
        Self::Select,
        Self::Place,
        Self::Sculpt,
        Self::Paint,
        Self::Light,
        Self::Animate,
        Self::Environment,
    ];

    /// Display name for the status bar.
    pub fn name(self) -> &'static str {
        match self {
            Self::Navigate => "Navigate",
            Self::Select => "Select",
            Self::Place => "Place",
            Self::Sculpt => "Sculpt",
            Self::Paint => "Paint",
            Self::Light => "Light",
            Self::Animate => "Animate",
            Self::Environment => "Environment",
        }
    }

    /// Short label for the toolbar button.
    pub fn short_name(self) -> &'static str {
        match self {
            Self::Navigate => "Nav",
            Self::Select => "Sel",
            Self::Place => "Place",
            Self::Sculpt => "Sculpt",
            Self::Paint => "Paint",
            Self::Light => "Light",
            Self::Animate => "Anim",
            Self::Environment => "Env",
        }
    }

    /// Numeric index (0-7) for signal-based UI reactivity.
    pub fn index(self) -> u8 {
        match self {
            Self::Navigate => 0,
            Self::Select => 1,
            Self::Place => 2,
            Self::Sculpt => 3,
            Self::Paint => 4,
            Self::Light => 5,
            Self::Animate => 6,
            Self::Environment => 7,
        }
    }

    /// Convert a numeric index (0-7) back to an `EditorMode`.
    /// Returns `Navigate` for out-of-range values.
    pub fn from_index(i: u8) -> Self {
        match i {
            0 => Self::Navigate,
            1 => Self::Select,
            2 => Self::Place,
            3 => Self::Sculpt,
            4 => Self::Paint,
            5 => Self::Light,
            6 => Self::Animate,
            7 => Self::Environment,
            _ => Self::Navigate,
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

    // ── Pending commands (UI → render loop) ──────────────────
    /// Set by UI menus, consumed by the render loop.
    pub pending_debug_mode: Option<u32>,
    /// Set by File > Quit, consumed by the event loop.
    pub wants_exit: bool,
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
            mode: EditorMode::Navigate,
            editor_camera: cam,
            editor_input: InputState::new(),
            scene_tree: SceneTree::new(),
            selected_properties: None,
            gizmo: GizmoState::new(),
            sculpt: SculptState::new(),
            paint: PaintState::new(),
            placement_queue: PlacementQueue::new(),
            asset_browser: AssetBrowser::new(),
            grid_snap: GridSnap::default(),
            light_editor: LightEditor::new(),
            environment: EnvironmentState::new(),
            animation: AnimationPreview::new(),
            overlay_config: OverlayConfig::default(),
            debug_viz: DebugOverlay::new(),
            frame_time_history: FrameTimeHistory::new(),
            undo: UndoStack::new(100),
            unsaved_changes: UnsavedChangesState::new(),
            recent_files: RecentFiles::new(),
            current_scene_path: None,
            pending_debug_mode: None,
            wants_exit: false,
        }
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

    /// Reset per-frame input deltas (mouse delta, scroll) after processing.
    pub fn reset_frame_deltas(&mut self) {
        self.editor_input.mouse_delta = glam::Vec2::ZERO;
        self.editor_input.scroll_delta = 0.0;
    }
}
