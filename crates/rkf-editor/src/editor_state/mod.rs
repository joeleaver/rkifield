//! Central editor state, aggregating all data model modules.
//!
//! `EditorState` holds instances of every editor subsystem. It is shared via
//! `Arc<Mutex<EditorState>>` between the winit event loop and the rinch component tree.

mod impl_state;
mod slider_signals;
#[cfg(test)]
mod tests;

pub use slider_signals::SliderSignals;

use crate::animation_preview::AnimationPreview;
use crate::camera::{CameraMode, SceneCamera};
use crate::debug_viz::{DebugOverlay, FrameTimeHistory};
use crate::gizmo::{GizmoMode, GizmoState};
use crate::input::InputState;
use crate::light_editor::LightManager;
use crate::overlay::OverlayConfig;
use crate::paint::PaintState;
use crate::placement::{AssetBrowser, GridSnap, PlacementQueue};
use crate::properties::PropertySheet;
use crate::scene_io::{RecentFiles, UnsavedChangesState};
use crate::sculpt::SculptState;
use crate::undo::UndoStack;

use std::collections::HashSet;

use glam::Vec3;
use rinch::prelude::{Signal, DragContext};
use uuid::Uuid;

use rkf_runtime::api::World;

use crate::ui_snapshot::{
    DiagnosticEntry, LightSummary, MaterialSummary, ObjectMaterialUsage, ObjectSummary,
    ShaderSummary,
};

/// What is currently selected in the scene tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SelectedEntity {
    /// An SDF object by entity UUID.
    Object(Uuid),
    /// A light by light ID.
    Light(u64),
    /// The scene node itself.
    Scene,
    /// The project root node.
    Project,
}

/// Dedicated reactive signal for the FPS counter.
///
/// Per-property reactive signals for fine-grained UI updates.
///
/// Each signal holds the actual value (not a counter). UI components subscribe
/// to specific signals via `.get()` inside reactive closures, so changes
/// to one property only update the UI elements that care about it.
///
/// **Engine→UI data flow**: The engine thread pushes structural data (objects,
/// lights, materials) into these signals via `run_on_main_thread`. No ArcSwap,
/// no polling — changes flow through the reactive system automatically.
///
/// **Important**: Never call `.set()` while holding `editor_state.lock()` — signal
/// updates trigger synchronous reactive effects that may also lock `EditorState`.
/// Always release the lock first, then update signals.
#[derive(Clone, Copy)]
pub struct UiSignals {
    // ── Selection & modes ────────────────────────────────────────
    pub selection: Signal<Option<SelectedEntity>>,
    pub editor_mode: Signal<EditorMode>,
    pub gizmo_mode: Signal<GizmoMode>,
    pub debug_mode: Signal<u32>,
    pub show_grid: Signal<bool>,
    pub fps: Signal<f64>,

    // ── Environment toggles ──────────────────────────────────────
    pub atmo_enabled: Signal<bool>,
    pub fog_enabled: Signal<bool>,
    pub clouds_enabled: Signal<bool>,
    pub bloom_enabled: Signal<bool>,
    pub dof_enabled: Signal<bool>,
    pub tone_map_mode: Signal<u32>,

    // ── Animation ────────────────────────────────────────────────
    pub animation_state: Signal<u32>,

    // ── Material browser ─────────────────────────────────────────
    pub selected_material: Signal<Option<u16>>,

    // ── Properties panel tab (0 = Object, 1 = Asset) ─────────────
    pub properties_tab: Signal<u32>,

    // ── Asset browser tab (0 = Materials, 1 = Shaders) ───────────
    pub asset_browser_tab: Signal<u32>,

    // ── Selected shader name (for properties display) ────────────
    pub selected_shader: Signal<Option<String>>,

    // ── Material preview ──────────────────────────────────────────
    pub preview_primitive_type: Signal<u32>,

    // ── Drag-and-drop ────────────────────────────────────────────
    pub shader_drag: DragContext<String>,
    pub shader_drop_highlight: Signal<bool>,
    pub material_drag: DragContext<u16>,
    pub material_drop_highlight: Signal<Option<u16>>,
    /// Counter incremented on drag-drop completion.
    pub drag_drop_generation: Signal<u64>,

    // ── Engine→UI structural data (pushed via run_on_main_thread) ─
    /// Scene objects — pushed by engine when scene structure changes.
    pub objects: Signal<Vec<ObjectSummary>>,
    /// Scene lights — pushed by engine when light list changes.
    pub lights: Signal<Vec<LightSummary>>,
    /// Material table — pushed by engine when materials change.
    pub materials: Signal<Vec<MaterialSummary>>,
    /// Shader registry — pushed by engine on startup and hot-reload.
    pub shaders: Signal<Vec<ShaderSummary>>,
    /// Per-material voxel counts for the selected object.
    pub selected_object_materials: Signal<Vec<ObjectMaterialUsage>>,
    /// Camera position for display readout (throttled push from engine).
    pub camera_display_pos: Signal<Vec3>,
    /// Current scene name.
    pub scene_name: Signal<String>,
    /// Current scene file path.
    pub scene_path: Signal<Option<String>>,
    /// Sculpt brush type name — pushed by engine when brush type changes.
    pub brush_type: Signal<String>,
    /// Paint mode (Material / Color).
    pub paint_mode: Signal<crate::paint::PaintMode>,
    /// Paint color (RGB, 0.0–1.0).
    pub paint_color: Signal<Vec3>,
    /// Play/stop mode state (true = playing, false = editing).
    pub play_state: Signal<bool>,

    // ── Component inspector (pushed via run_on_main_thread) ──
    /// Inspector data for the selected entity's components.
    pub inspector_data: Signal<Option<InspectorSnapshot>>,
    /// Components available to add to the selected entity.
    pub available_components: Signal<Vec<String>>,
    /// Behavior systems list — pushed by engine when play mode is active.
    pub systems: Signal<Vec<crate::ui_snapshot::SystemSummary>>,

    // ── Loading indicator ────────────────────────────────────────
    /// Loading status message — `None` means idle, `Some("...")` shows a message
    /// in the status bar. Engine thread uses `.set()` via `run_on_main_thread()`.
    pub loading_status: Signal<Option<String>>,

    // ── Diagnostics (Debug panel) ────────────────────────────────
    /// Compile errors, warnings, and info messages from the game plugin build.
    /// Pushed from the engine thread via `run_on_main_thread`.
    pub diagnostics: Signal<Vec<DiagnosticEntry>>,

    // ── Console ─────────────────────────────────────────────────
    /// Console entries snapshot, pushed from engine thread when revision changes.
    pub console_entries: Signal<Vec<rkf_runtime::behavior::ConsoleEntry>>,
    /// Console filter state (show/hide info/warn/error).
    pub console_filter: Signal<rkf_runtime::behavior::ConsoleFilter>,

    // ── Project state ─────────────────────────────────────────────
    /// Whether a project is currently loaded. Drives welcome screen visibility.
    pub project_loaded: Signal<bool>,
    /// Recent projects list for the welcome screen.
    pub recent_projects: Signal<Vec<crate::editor_config::RecentProject>>,

    // ── Dylib readiness ─────────────────────────────────────────────
    /// Whether the game plugin dylib has been loaded successfully.
    /// `false` while the initial build is in progress; `true` once loaded.
    /// UI elements that depend on the dylib (component inspector, systems
    /// panel, play button) should check this before enabling interaction.
    pub dylib_ready: Signal<bool>,

    // ── Camera linking ─────────────────────────────────────────────
    /// UUID of the scene camera the editor is linked to, if any.
    pub linked_camera: Signal<Option<Uuid>>,

    /// UUID of the scene camera currently driving the viewport, if any.
    pub viewport_camera: Signal<Option<Uuid>>,

    // ── Scene environment ─────────────────────────────────────────
    /// UUID of the scene environment singleton entity.
    /// Used by the environment panel to send SetComponentField commands.
    pub scene_env_uuid: Signal<Option<Uuid>>,
}

/// Snapshot of inspector data, safe to send to the UI thread.
///
/// Mirrors `rkf_runtime::behavior::inspector::InspectorData` but uses
/// owned types that are `Clone + PartialEq` for signal comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectorSnapshot {
    /// The entity UUID.
    pub entity_id: uuid::Uuid,
    /// Per-component data.
    pub components: Vec<ComponentSnapshot>,
}

/// Snapshot of a single component for the UI inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentSnapshot {
    /// Component type name.
    pub name: String,
    /// Per-field data.
    pub fields: Vec<FieldSnapshot>,
    /// Whether this component can be removed.
    pub removable: bool,
}

/// Snapshot of a single field for the UI inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSnapshot {
    /// Field name.
    pub name: String,
    /// Type classification.
    pub field_type: rkf_runtime::behavior::registry::FieldType,
    /// Current value as a displayable string.
    pub display_value: String,
    /// Current float value (for Float fields with sliders).
    pub float_value: Option<f64>,
    /// Current int value (for Int fields).
    pub int_value: Option<i64>,
    /// Current bool value (for Bool fields).
    pub bool_value: Option<bool>,
    /// Current Vec3 value (for Vec3 fields).
    pub vec3_value: Option<glam::Vec3>,
    /// Current string value (for String fields).
    pub string_value: Option<String>,
    /// Optional numeric range for slider display `(min, max)`.
    pub range: Option<(f64, f64)>,
    /// True if the field is transient.
    pub transient: bool,
    /// Sub-fields for `Struct` type fields.
    pub sub_fields: Option<Vec<FieldSnapshot>>,
    /// File extension filter for `AssetRef` fields (e.g., `"rkenv"`).
    pub asset_filter: Option<String>,
    /// Required component type for `ComponentRef` fields (e.g., `"Transform"`).
    pub component_filter: Option<String>,
}

impl UiSignals {
    /// Create a new set of UI signals with default values.
    /// Must be called on the main thread (signals use thread-local reactive state).
    pub fn new() -> Self {
        Self {
            selection: Signal::new(None),
            editor_mode: Signal::new(EditorMode::Default),
            gizmo_mode: Signal::new(GizmoMode::Translate),
            debug_mode: Signal::new(0),
            show_grid: Signal::new(false),
            fps: Signal::new(0.0),
            atmo_enabled: Signal::new(true),
            fog_enabled: Signal::new(false),
            clouds_enabled: Signal::new(false),
            bloom_enabled: Signal::new(true),
            dof_enabled: Signal::new(false),
            tone_map_mode: Signal::new(0),
            animation_state: Signal::new(0),
            selected_material: Signal::new(None),
            properties_tab: Signal::new(0),
            asset_browser_tab: Signal::new(0),
            selected_shader: Signal::new(None),
            preview_primitive_type: Signal::new(0),
            shader_drag: DragContext::new(),
            shader_drop_highlight: Signal::new(false),
            material_drag: DragContext::new(),
            material_drop_highlight: Signal::new(None),
            drag_drop_generation: Signal::new(0),
            // Engine→UI structural data — populated by engine thread.
            objects: Signal::new(Vec::new()),
            lights: Signal::new(Vec::new()),
            materials: Signal::new(Vec::new()),
            shaders: Signal::new(Vec::new()),
            selected_object_materials: Signal::new(Vec::new()),
            camera_display_pos: Signal::new(Vec3::new(0.0, 2.5, 5.0)),
            scene_name: Signal::new("Untitled".into()),
            scene_path: Signal::new(None),
            brush_type: Signal::new("Add".to_string()),
            paint_mode: Signal::new(crate::paint::PaintMode::Material),
            paint_color: Signal::new(Vec3::ONE),
            play_state: Signal::new(false),
            inspector_data: Signal::new(None),
            available_components: Signal::new(Vec::new()),
            systems: Signal::new(Vec::new()),
            loading_status: Signal::new(None),
            diagnostics: Signal::new(Vec::new()),
            console_entries: Signal::new(Vec::new()),
            console_filter: Signal::new(rkf_runtime::behavior::ConsoleFilter::default()),
            project_loaded: Signal::new(false),
            recent_projects: Signal::new(Vec::new()),
            dylib_ready: Signal::new(false),
            linked_camera: Signal::new(None),
            viewport_camera: Signal::new(None),
            scene_env_uuid: Signal::new(None),
        }
    }

    /// Set selection and sync dependent state (sliders + tree highlight).
    ///
    /// Replaces the former selection-change and tree-sync Effects.
    /// Call this instead of `selection.set()` directly.
    pub fn set_selection(
        &self,
        sel: Option<SelectedEntity>,
        tree_state: &rinch::prelude::UseTreeReturn,
    ) {
        self.selection.set(sel);
        self.sync_tree_selection(tree_state);
    }

    /// Sync tree selection highlight when ui.selection changes.
    ///
    /// Store method — called from event handlers that change selection,
    /// replacing the tree selection-sync Effect.
    pub fn sync_tree_selection(&self, tree_state: &rinch::prelude::UseTreeReturn) {
        let sel = self.selection.get();
        if let Some(sel) = sel {
            let value = match sel {
                SelectedEntity::Object(id) => format!("obj:{}", &id.to_string()[..8]),
                SelectedEntity::Light(id) => format!("light:{id}"),
                SelectedEntity::Scene => "scene".to_string(),
                SelectedEntity::Project => "project".to_string(),
            };
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

/// Accumulates geometry-first snapshots during a sculpt stroke for undo support.
///
/// Created when a sculpt stroke begins. Before each edit, any not-yet-captured
/// geometry slots are snapshot from the geometry and SDF cache pools. On stroke
/// end, the accumulated snapshots are pushed as an undo action.
pub struct SculptUndoAccumulator {
    /// The object being sculpted (entity UUID).
    pub object_id: Uuid,
    /// Geometry pool slots that have already been captured.
    pub captured_slots: HashSet<u32>,
    /// Pre-edit snapshots: (geo_slot, geometry, sdf_cache, brick_pool_slot).
    pub snapshots: Vec<GeometryUndoEntry>,
}

/// A single geometry-first undo entry for one brick slot.
#[derive(Clone)]
pub struct GeometryUndoEntry {
    /// Geometry pool slot index.
    pub geo_slot: u32,
    /// Snapshot of the BrickGeometry before modification.
    pub geometry: rkf_core::brick_geometry::BrickGeometry,
    /// Snapshot of the SdfCache before modification.
    pub sdf_cache: rkf_core::sdf_cache::SdfCache,
    /// Corresponding brick pool slot (for GPU re-upload after undo).
    pub brick_slot: u32,
}

impl std::fmt::Debug for GeometryUndoEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeometryUndoEntry")
            .field("geo_slot", &self.geo_slot)
            .field("brick_slot", &self.brick_slot)
            .field("solid_count", &self.geometry.solid_count())
            .finish()
    }
}

/// State for the material browser panel.
#[derive(Debug, Clone)]
pub struct MaterialBrowserState {
    /// Currently selected material slot in the browser.
    pub selected_slot: Option<u16>,
    /// Filter text for material search.
    pub filter_text: String,
    /// Filter by category (empty = show all).
    pub filter_category: String,
}

impl Default for MaterialBrowserState {
    fn default() -> Self {
        Self {
            selected_slot: None,
            filter_text: String::new(),
            filter_category: String::new(),
        }
    }
}

/// Aggregated editor state, shared between the event loop and the UI.
pub struct EditorState {
    // ── Mode ─────────────────────────────────────────────────
    pub mode: EditorMode,

    // ── Camera & Input ───────────────────────────────────────
    pub editor_camera: SceneCamera,
    pub editor_input: InputState,

    // ── Scene ────────────────────────────────────────────────
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

    // ── Lights ────────────────────────────────────────────────
    pub light_editor: LightManager,

    // ── Animation ────────────────────────────────────────────
    pub animation: AnimationPreview,

    // ── Overlays & Debug ─────────────────────────────────────
    pub overlay_config: OverlayConfig,
    pub debug_viz: DebugOverlay,
    pub frame_time_history: FrameTimeHistory,

    // ── Undo/Redo ────────────────────────────────────────────
    pub undo: UndoStack,

    // ── Material Browser ──────────────────────────────────────
    pub material_browser: MaterialBrowserState,

    // ── Scene I/O ────────────────────────────────────────────
    pub unsaved_changes: UnsavedChangesState,
    pub recent_files: RecentFiles,
    pub current_scene_path: Option<String>,

    // ── Project I/O ──────────────────────────────────────────
    /// Currently loaded project descriptor.
    pub current_project: Option<rkf_runtime::ProjectFile>,
    /// Path to the current `.rkproject` file.
    pub current_project_path: Option<String>,
    /// Set by File > New Project, consumed by the engine loop.
    pub pending_new_project: bool,
    /// Set by File > Open Project, consumed by the engine loop.
    pub pending_open_project: bool,
    /// Pre-supplied path for Open Project (skips file dialog).
    pub pending_open_project_path: Option<String>,
    /// Set by Save — saves both scene and project when a project is loaded.
    pub pending_save_project: bool,

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
    /// Set by File > Open, consumed by the engine loop.
    pub pending_open: bool,
    /// Pre-supplied path for File > Open (skips file dialog). Set by MCP/commands.
    pub pending_open_path: Option<String>,
    /// Set by File > Save, consumed by the engine loop.
    pub pending_save: bool,
    /// Set by File > Save As, consumed by the engine loop.
    pub pending_save_as: bool,
    /// Pre-supplied path for Save (skips file dialog). Set by MCP/commands.
    pub pending_save_path: Option<String>,
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
    /// Set by Play button, consumed by engine loop to enter play mode.
    pub pending_play_start: bool,
    /// Set by Stop button, consumed by engine loop to exit play mode.
    pub pending_play_stop: bool,
    /// Set by titlebar drag handler, consumed by the event loop.
    pub pending_drag: bool,
    /// Set by minimize window control, consumed by the event loop.
    pub pending_minimize: bool,
    /// Set by maximize window control, consumed by the event loop.
    pub pending_maximize: bool,
    /// Set by "Convert to Voxel Object" button. Contains the object ID
    /// of an analytical primitive to convert to geometry-first voxelized form.
    pub pending_convert_to_voxel: Option<(Uuid, f32)>,
    /// Set by material drag-and-drop in object properties panel.
    /// Contains (object_id, from_material, to_material).
    pub pending_remap_material: Option<(Uuid, u16, u16)>,
    /// Set by material assignment on analytical primitives.
    /// Contains (object_id, material_id).
    pub pending_set_primitive_material: Option<(Uuid, u16)>,

    // ── Sculpt pipeline (UI → render loop) ──────────────────
    /// Queued sculpt edit requests — one per brush-hit point during a stroke.
    /// Drained by the render loop each frame and applied to the CPU brick pool.
    pub pending_sculpt_edits: Vec<crate::sculpt::SculptEditRequest>,

    /// Accumulates brick snapshots during a sculpt stroke for undo.
    /// Created on stroke begin, finalized on stroke end.
    pub sculpt_undo_accumulator: Option<SculptUndoAccumulator>,

    /// Pending sculpt undo: (object_id, geometry_snapshots) to restore.
    /// Set by apply_undo_action, consumed by the render loop.
    pub pending_sculpt_undo: Option<(Uuid, Vec<GeometryUndoEntry>)>,

    // ── Paint pipeline (UI → render loop) ────────────────────
    /// Queued paint edit requests — one per brush-hit point during a stroke.
    /// Drained by the render loop each frame and applied to surface voxels.
    pub pending_paint_edits: Vec<crate::paint::PaintEditRequest>,

    /// Accumulates geometry-first snapshots during a paint stroke for undo.
    /// Created on stroke begin, finalized on stroke end.
    pub paint_undo_accumulator: Option<SculptUndoAccumulator>,

    /// Pending paint undo: (object_id, geometry_snapshots) to restore.
    /// Set by apply_undo_action, consumed by the render loop.
    pub pending_paint_undo: Option<(Uuid, Vec<GeometryUndoEntry>)>,

    // ── World (unified game state) ──────────────────────────
    /// The unified world container. Wraps `Scene` (SDF objects) + ECS +
    /// brick pool. Replaces the former `v2_scene: Option<Scene>`.
    pub world: World,
}
