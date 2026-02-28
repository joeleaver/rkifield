//! Central editor state, aggregating all data model modules.
//!
//! `EditorState` holds instances of every editor subsystem. It is shared via
//! `Arc<Mutex<EditorState>>` between the winit event loop and the rinch component tree.

use crate::animation_preview::AnimationPreview;
use crate::camera::{CameraMode, EditorCamera};
use crate::debug_viz::{DebugOverlay, FrameTimeHistory};
use crate::environment::EnvironmentState;
use crate::gizmo::{GizmoMode, GizmoState};
use crate::input::InputState;
use crate::light_editor::LightEditor;
use crate::overlay::OverlayConfig;
use crate::paint::PaintState;
use crate::placement::{AssetBrowser, GridSnap, PlacementQueue};
use crate::properties::PropertySheet;
use crate::scene_io::{RecentFiles, UnsavedChangesState};
use crate::sculpt::SculptState;
use crate::undo::UndoStack;

use glam::Vec3;
use rinch::prelude::Signal;
use rkf_runtime::api::World;

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

/// Dedicated reactive signal for the FPS counter.
///
/// Updated from the render thread via `run_on_main_thread`. Only the status
/// bar's FPS text node tracks this — bumping it does NOT rebuild other panels.
#[derive(Clone, Copy)]
pub struct FpsSignal(pub Signal<f64>);

impl FpsSignal {
    pub fn new() -> Self {
        Self(Signal::new(0.0))
    }
}

/// Per-property reactive signals for fine-grained UI updates.
///
/// Each signal holds the actual value (not a counter). UI components subscribe
/// to specific signals via `.get()` inside `reactive_component_dom`, so changes
/// to one property only rebuild the panels that care about it.
///
/// **Important**: Never call `.set()` while holding `editor_state.lock()` — signal
/// updates trigger synchronous reactive effects that may also lock `EditorState`.
/// Always release the lock first, then update signals.
#[derive(Clone, Copy)]
pub struct UiSignals {
    // Selection & modes
    pub selection: Signal<Option<SelectedEntity>>,
    pub editor_mode: Signal<EditorMode>,
    pub gizmo_mode: Signal<GizmoMode>,
    pub debug_mode: Signal<u32>,
    pub show_grid: Signal<bool>,
    pub object_count: Signal<usize>,
    pub fps: Signal<f64>,

    // Environment toggles
    pub atmo_enabled: Signal<bool>,
    pub fog_enabled: Signal<bool>,
    pub clouds_enabled: Signal<bool>,
    pub bloom_enabled: Signal<bool>,
    pub dof_enabled: Signal<bool>,
    pub tone_map_mode: Signal<u32>,

    // Scene structure — counter because tree data is too complex for a Signal
    pub scene_revision: Signal<u64>,
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
            object_count: Signal::new(0),
            fps: Signal::new(0.0),
            atmo_enabled: Signal::new(true),
            fog_enabled: Signal::new(false),
            clouds_enabled: Signal::new(false),
            bloom_enabled: Signal::new(true),
            dof_enabled: Signal::new(false),
            tone_map_mode: Signal::new(0),
            scene_revision: Signal::new(0),
        }
    }

    /// Increment scene_revision to trigger tree/count rebuilds.
    pub fn bump_scene(&self) {
        self.scene_revision.update(|r| *r += 1);
    }
}

/// Centralized reactive signals for all persistent slider properties.
///
/// Each signal holds the actual slider value (f64). The UI creates sliders
/// bound to these signals via `build_synced_slider`, and a single batch
/// `Effect` in `RightPanel` syncs all values to `EditorState` in one lock
/// per frame — eliminating per-slider lock closures.
///
/// **Adding a new slider property:**
/// 1. Add `pub field_name: Signal<f64>` here
/// 2. Init from `EditorState` in `new()`
/// 3. Add sync line in `sync_to_state()`
/// 4. Call `build_synced_slider()` in the UI
#[derive(Clone, Copy)]
pub struct SliderSignals {
    // Camera
    pub fov: Signal<f64>,
    pub fly_speed: Signal<f64>,
    pub near: Signal<f64>,
    pub far: Signal<f64>,
    // Atmosphere
    pub sun_azimuth: Signal<f64>,
    pub sun_elevation: Signal<f64>,
    pub sun_intensity: Signal<f64>,
    pub rayleigh_scale: Signal<f64>,
    pub mie_scale: Signal<f64>,
    // Fog
    pub fog_density: Signal<f64>,
    pub fog_height_falloff: Signal<f64>,
    pub dust_density: Signal<f64>,
    pub dust_asymmetry: Signal<f64>,
    // Clouds
    pub cloud_coverage: Signal<f64>,
    pub cloud_density: Signal<f64>,
    pub cloud_altitude: Signal<f64>,
    pub cloud_thickness: Signal<f64>,
    pub cloud_wind_speed: Signal<f64>,
    // Post-processing
    pub bloom_intensity: Signal<f64>,
    pub bloom_threshold: Signal<f64>,
    pub exposure: Signal<f64>,
    pub sharpen: Signal<f64>,
    pub dof_focus_dist: Signal<f64>,
    pub dof_focus_range: Signal<f64>,
    pub dof_max_coc: Signal<f64>,
    pub motion_blur: Signal<f64>,
    pub god_rays: Signal<f64>,
    pub vignette: Signal<f64>,
    pub grain: Signal<f64>,
    pub chromatic_ab: Signal<f64>,
    // Brush
    pub brush_radius: Signal<f64>,
    pub brush_strength: Signal<f64>,
    pub brush_falloff: Signal<f64>,
}

impl SliderSignals {
    /// Create slider signals initialized from the current `EditorState`.
    /// Must be called on the main thread (signals use thread-local reactive state).
    pub fn new(es: &EditorState) -> Self {
        let d = &es.environment.atmosphere.sun_direction;
        let pp = &es.environment.post_process;
        Self {
            fov: Signal::new(es.editor_camera.fov_y.to_degrees() as f64),
            fly_speed: Signal::new(es.editor_camera.fly_speed as f64),
            near: Signal::new(es.editor_camera.near as f64),
            far: Signal::new(es.editor_camera.far as f64),
            sun_azimuth: Signal::new(d.x.atan2(d.z).to_degrees().rem_euclid(360.0) as f64),
            sun_elevation: Signal::new(d.y.asin().to_degrees() as f64),
            sun_intensity: Signal::new(es.environment.atmosphere.sun_intensity as f64),
            rayleigh_scale: Signal::new(es.environment.atmosphere.rayleigh_scale as f64),
            mie_scale: Signal::new(es.environment.atmosphere.mie_scale as f64),
            fog_density: Signal::new(es.environment.fog.density as f64),
            fog_height_falloff: Signal::new(es.environment.fog.height_falloff as f64),
            dust_density: Signal::new(es.environment.fog.ambient_dust_density as f64),
            dust_asymmetry: Signal::new(es.environment.fog.dust_asymmetry as f64),
            cloud_coverage: Signal::new(es.environment.clouds.coverage as f64),
            cloud_density: Signal::new(es.environment.clouds.density as f64),
            cloud_altitude: Signal::new(es.environment.clouds.altitude as f64),
            cloud_thickness: Signal::new(es.environment.clouds.thickness as f64),
            cloud_wind_speed: Signal::new(es.environment.clouds.wind_speed as f64),
            bloom_intensity: Signal::new(pp.bloom_intensity as f64),
            bloom_threshold: Signal::new(pp.bloom_threshold as f64),
            exposure: Signal::new(pp.exposure as f64),
            sharpen: Signal::new(pp.sharpen_strength as f64),
            dof_focus_dist: Signal::new(pp.dof_focus_distance as f64),
            dof_focus_range: Signal::new(pp.dof_focus_range as f64),
            dof_max_coc: Signal::new(pp.dof_max_coc as f64),
            motion_blur: Signal::new(pp.motion_blur_intensity as f64),
            god_rays: Signal::new(pp.god_rays_intensity as f64),
            vignette: Signal::new(pp.vignette_intensity as f64),
            grain: Signal::new(pp.grain_intensity as f64),
            chromatic_ab: Signal::new(pp.chromatic_aberration as f64),
            brush_radius: Signal::new(es.sculpt.current_settings.radius as f64),
            brush_strength: Signal::new(es.sculpt.current_settings.strength as f64),
            brush_falloff: Signal::new(es.sculpt.current_settings.falloff as f64),
        }
    }

    /// Subscribe the current reactive scope to every signal.
    /// Call inside an `Effect` to re-run whenever any slider changes.
    pub fn track_all(&self) {
        let _ = self.fov.get();
        let _ = self.fly_speed.get();
        let _ = self.near.get();
        let _ = self.far.get();
        let _ = self.sun_azimuth.get();
        let _ = self.sun_elevation.get();
        let _ = self.sun_intensity.get();
        let _ = self.rayleigh_scale.get();
        let _ = self.mie_scale.get();
        let _ = self.fog_density.get();
        let _ = self.fog_height_falloff.get();
        let _ = self.dust_density.get();
        let _ = self.dust_asymmetry.get();
        let _ = self.cloud_coverage.get();
        let _ = self.cloud_density.get();
        let _ = self.cloud_altitude.get();
        let _ = self.cloud_thickness.get();
        let _ = self.cloud_wind_speed.get();
        let _ = self.bloom_intensity.get();
        let _ = self.bloom_threshold.get();
        let _ = self.exposure.get();
        let _ = self.sharpen.get();
        let _ = self.dof_focus_dist.get();
        let _ = self.dof_focus_range.get();
        let _ = self.dof_max_coc.get();
        let _ = self.motion_blur.get();
        let _ = self.god_rays.get();
        let _ = self.vignette.get();
        let _ = self.grain.get();
        let _ = self.chromatic_ab.get();
        let _ = self.brush_radius.get();
        let _ = self.brush_strength.get();
        let _ = self.brush_falloff.get();
    }

    /// Write all signal values back to `EditorState` in one shot.
    ///
    /// Called from the batch sync `Effect` after `track_all()`. Toggle
    /// signals (atmosphere enabled, fog enabled, etc.) are synced by the
    /// caller from `UiSignals`.
    pub fn sync_to_state(&self, es: &mut EditorState) {
        // Camera
        es.editor_camera.fov_y = (self.fov.get() as f32).to_radians();
        es.editor_camera.fly_speed = self.fly_speed.get() as f32;
        es.editor_camera.near = self.near.get() as f32;
        es.editor_camera.far = self.far.get() as f32;

        // Atmosphere — compute sun_direction from azimuth + elevation
        let az = (self.sun_azimuth.get() as f32).to_radians();
        let el = (self.sun_elevation.get() as f32).to_radians();
        let cos_el = el.cos();
        es.environment.atmosphere.sun_direction =
            Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
        es.environment.atmosphere.sun_intensity = self.sun_intensity.get() as f32;
        es.environment.atmosphere.rayleigh_scale = self.rayleigh_scale.get() as f32;
        es.environment.atmosphere.mie_scale = self.mie_scale.get() as f32;

        // Fog
        es.environment.fog.density = self.fog_density.get() as f32;
        es.environment.fog.height_falloff = self.fog_height_falloff.get() as f32;
        es.environment.fog.ambient_dust_density = self.dust_density.get() as f32;
        es.environment.fog.dust_asymmetry = self.dust_asymmetry.get() as f32;

        // Clouds
        es.environment.clouds.coverage = self.cloud_coverage.get() as f32;
        es.environment.clouds.density = self.cloud_density.get() as f32;
        es.environment.clouds.altitude = self.cloud_altitude.get() as f32;
        es.environment.clouds.thickness = self.cloud_thickness.get() as f32;
        es.environment.clouds.wind_speed = self.cloud_wind_speed.get() as f32;

        // Post-processing
        es.environment.post_process.bloom_intensity = self.bloom_intensity.get() as f32;
        es.environment.post_process.bloom_threshold = self.bloom_threshold.get() as f32;
        es.environment.post_process.exposure = self.exposure.get() as f32;
        es.environment.post_process.sharpen_strength = self.sharpen.get() as f32;
        es.environment.post_process.dof_focus_distance = self.dof_focus_dist.get() as f32;
        es.environment.post_process.dof_focus_range = self.dof_focus_range.get() as f32;
        es.environment.post_process.dof_max_coc = self.dof_max_coc.get() as f32;
        es.environment.post_process.motion_blur_intensity = self.motion_blur.get() as f32;
        es.environment.post_process.god_rays_intensity = self.god_rays.get() as f32;
        es.environment.post_process.vignette_intensity = self.vignette.get() as f32;
        es.environment.post_process.grain_intensity = self.grain.get() as f32;
        es.environment.post_process.chromatic_aberration = self.chromatic_ab.get() as f32;

        // Brush — sync to both sculpt and paint settings
        let radius = self.brush_radius.get() as f32;
        let strength = self.brush_strength.get() as f32;
        let falloff = self.brush_falloff.get() as f32;
        es.sculpt.set_radius(radius);
        es.sculpt.set_strength(strength);
        es.sculpt.current_settings.falloff = falloff;
        es.paint.current_settings.radius = radius;
        es.paint.current_settings.strength = strength;
        es.paint.current_settings.falloff = falloff;

        // Mark environment dirty so the engine picks up changes.
        es.environment.mark_dirty();
    }
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

    // ── World (unified game state) ──────────────────────────
    /// The unified world container. Wraps `Scene` (SDF objects) + ECS +
    /// brick pool. Replaces the former `v2_scene: Option<Scene>`.
    pub world: World,
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
            world: World::new("editor"),
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

    /// Apply a single undo/redo action to the world.
    ///
    /// When `reverse` is `true` (undo), applies old values.
    /// When `reverse` is `false` (redo), applies new values.
    pub fn apply_undo_action(&mut self, action: &crate::undo::UndoAction, reverse: bool) {
        use crate::undo::UndoActionKind;

        match &action.kind {
            UndoActionKind::Transform {
                entity_id,
                old_pos, old_rot, old_scale,
                new_pos, new_rot, new_scale,
            } => {
                let (pos, rot, scale) = if reverse {
                    (*old_pos, *old_rot, *old_scale)
                } else {
                    (*new_pos, *new_rot, *new_scale)
                };
                if let Some(entity) = self.world.find_entity_by_id(*entity_id) {
                    let wp = rkf_core::WorldPosition::new(glam::IVec3::ZERO, pos);
                    let _ = self.world.set_position(entity, wp);
                    let _ = self.world.set_rotation(entity, rot);
                    let _ = self.world.set_scale(entity, scale);
                }
            }
            UndoActionKind::SpawnEntity { entity_id } => {
                if reverse {
                    // Undo spawn = despawn
                    if let Some(entity) = self.world.find_entity_by_id(*entity_id) {
                        let _ = self.world.despawn(entity);
                    }
                }
                // Redo spawn would need stored object data — not supported yet.
            }
            UndoActionKind::DespawnEntity { entity_id: _ } => {
                // Undo despawn would need stored object data — not supported yet.
            }
            _ => {
                // VoxelEdit, PropertyChange, EnvironmentChange — future work.
            }
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

        let scene = self.world.scene();
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
    /// Populates `world.scene()` with SDF objects from the file, lights into
    /// the light editor, and applies environment settings. Returns the loaded
    /// `SceneFile` for further processing (e.g. setting up engine geometry).
    pub fn load_scene(&mut self, path: &str) -> Result<crate::scene_io::SceneFile, String> {
        use crate::scene_io::{load_scene_from_path, ComponentData};

        let scene_file = load_scene_from_path(path)?;

        // Clear existing world scene.
        let world_scene = self.world.scene_mut();
        world_scene.objects.clear();

        // Populate world scene from non-light entities.
        for entity in &scene_file.entities {
            let is_light_only = !entity.components.is_empty()
                && entity
                    .components
                    .iter()
                    .all(|c| matches!(c, ComponentData::Light { .. }));
            if is_light_only {
                continue;
            }

            let root_node = rkf_core::scene_node::SceneNode::new(&entity.name);
            let mut obj = rkf_core::scene::SceneObject {
                id: entity.entity_id as u32,
                name: entity.name.clone(),
                parent_id: entity.parent_id.map(|p| p as u32),
                position: entity.position,
                rotation: entity.rotation,
                scale: entity.scale,
                root_node,
                aabb: rkf_core::aabb::Aabb::new(Vec3::ZERO, Vec3::ZERO),
            };
            // Store asset path in the root node's name for roundtrip.
            for comp in &entity.components {
                if let ComponentData::SdfObject { asset_path } = comp {
                    obj.root_node.name = asset_path.clone();
                }
            }
            world_scene.objects.push(obj);
        }

        // Update next_id to avoid collisions with loaded objects.
        if let Some(max_id) = world_scene.objects.iter().map(|o| o.id).max() {
            world_scene.next_id = max_id + 1;
        }

        // Populate light editor from Light components.
        self.light_editor = crate::light_editor::LightEditor::new();
        for entity in &scene_file.entities {
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

        // Apply environment settings if present.
        if !scene_file.environment_ron.is_empty() {
            if let Ok(env) =
                crate::environment::EnvironmentState::deserialize_from_ron(&scene_file.environment_ron)
            {
                self.environment = env;
                self.environment.mark_dirty();
            }
        }

        // Resync entity tracking to register all loaded objects.
        self.world.resync_entity_tracking();

        // Track the current scene path.
        self.current_scene_path = Some(path.to_string());
        self.unsaved_changes.mark_saved();

        Ok(scene_file)
    }

    /// Construct a [`SceneFile`] from the current editor state.
    ///
    /// Reads SDF objects directly from `world.scene()`, then appends light
    /// entities from the light editor. The environment is serialized via RON.
    /// The resulting `SceneFile` can be passed to
    /// [`crate::scene_io::save_scene_to_path`].
    pub fn save_current_scene(&self) -> crate::scene_io::SceneFile {
        use crate::light_editor::EditorLightType;
        use crate::scene_io::{ComponentData, SceneEntity, SceneFile};
        use glam::Quat;

        let mut entities: Vec<SceneEntity> = Vec::new();

        // Collect SDF objects from world scene.
        let scene = self.world.scene();
        for obj in &scene.objects {
            let mut components = Vec::new();
            // If the root node name looks like an asset path, store it.
            if !obj.root_node.name.is_empty() && obj.root_node.name.contains("://") {
                components.push(ComponentData::SdfObject {
                    asset_path: obj.root_node.name.clone(),
                });
            }
            entities.push(SceneEntity {
                entity_id: obj.id as u64,
                name: obj.name.clone(),
                parent_id: obj.parent_id.map(|p| p as u64),
                position: obj.position,
                rotation: obj.rotation,
                scale: obj.scale,
                components,
            });
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
    fn test_load_scene_populates_world() {
        let path = "/tmp/rkf_test_load_tree.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        let scene = state.load_scene(path).unwrap();
        assert_eq!(scene.entities.len(), 3);
        // Only SDF entities go in world scene (lights go to light_editor)
        let objects = &state.world.scene().objects;
        assert_eq!(objects.len(), 2); // "Ground" + "Child"
        assert_eq!(objects[0].name, "Ground");
        assert_eq!(objects[1].name, "Child");
        // "Child" has parent_id pointing to Ground
        assert_eq!(objects[1].parent_id, Some(objects[0].id));
    }

    #[test]
    fn test_load_scene_stores_transform_and_asset_path() {
        let path = "/tmp/rkf_test_load_xform.rkscene";
        write_test_scene(path);
        let mut state = EditorState::new();
        state.load_scene(path).unwrap();
        let objects = &state.world.scene().objects;
        let ground = &objects[0];
        assert_eq!(ground.position, Vec3::new(0.0, -0.5, 0.0));
        // Asset path stored in root_node.name
        assert_eq!(ground.root_node.name, "procedural://ground");
        let child = &objects[1];
        assert_eq!(child.position, Vec3::new(1.0, 2.0, 3.0));
        assert!(child.scale.abs_diff_eq(Vec3::splat(0.5), 1e-6));
        assert_eq!(child.root_node.name, "procedural://pillar");
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
    fn test_world_defaults_to_empty() {
        let state = EditorState::new();
        assert_eq!(state.world.scene().objects.len(), 0);
    }

    #[test]
    fn test_world_scene_is_single_source_of_truth() {
        use rkf_core::scene_node::SceneNode as CoreNode;
        let mut state = EditorState::new();
        let scene = state.world.scene_mut();
        scene.add_object("SphereObj", Vec3::ZERO, CoreNode::new("root"));
        scene.add_object("BoxObj", Vec3::ZERO, CoreNode::new("root2"));

        // World scene is the single source of truth — no sync needed.
        assert_eq!(state.world.scene().objects.len(), 2);
        assert_eq!(state.world.scene().objects[0].name, "SphereObj");
        assert_eq!(state.world.scene().objects[1].name, "BoxObj");
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

    // ── Undo/redo application tests ───────────────────────────────────────

    #[test]
    fn undo_transform_restores_position() {
        use rkf_core::scene_node::SdfPrimitive;
        let mut state = EditorState::new();
        let entity = state.world.spawn("obj")
            .position_vec3(Vec3::new(1.0, 2.0, 3.0))
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(0)
            .build();

        let action = crate::undo::UndoAction {
            kind: crate::undo::UndoActionKind::Transform {
                entity_id: entity.to_u64(),
                old_pos: Vec3::new(1.0, 2.0, 3.0),
                old_rot: Quat::IDENTITY,
                old_scale: Vec3::ONE,
                new_pos: Vec3::new(10.0, 20.0, 30.0),
                new_rot: Quat::from_rotation_y(1.0),
                new_scale: Vec3::splat(2.0),
            },
            timestamp_ms: 0,
            description: "Move".into(),
        };

        // Apply redo (new values).
        state.apply_undo_action(&action, false);
        let pos = state.world.position(entity).unwrap().to_vec3();
        assert!(pos.abs_diff_eq(Vec3::new(10.0, 20.0, 30.0), 1e-6));

        // Apply undo (old values).
        state.apply_undo_action(&action, true);
        let pos = state.world.position(entity).unwrap().to_vec3();
        assert!(pos.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-6));
    }

    #[test]
    fn redo_transform_reapplies() {
        use rkf_core::scene_node::SdfPrimitive;
        let mut state = EditorState::new();
        let entity = state.world.spawn("obj")
            .position_vec3(Vec3::ZERO)
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(0)
            .build();

        let action = crate::undo::UndoAction {
            kind: crate::undo::UndoActionKind::Transform {
                entity_id: entity.to_u64(),
                old_pos: Vec3::ZERO,
                old_rot: Quat::IDENTITY,
                old_scale: Vec3::ONE,
                new_pos: Vec3::new(5.0, 0.0, 0.0),
                new_rot: Quat::IDENTITY,
                new_scale: Vec3::splat(3.0),
            },
            timestamp_ms: 0,
            description: "Move".into(),
        };

        // Undo then redo should restore new values.
        state.apply_undo_action(&action, true);
        state.apply_undo_action(&action, false);
        let scale = state.world.scale(entity).unwrap();
        assert!(scale.abs_diff_eq(Vec3::splat(3.0), 1e-6));
    }

    #[test]
    fn undo_spawn_despawns_entity() {
        use rkf_core::scene_node::SdfPrimitive;
        let mut state = EditorState::new();
        let entity = state.world.spawn("spawned")
            .sdf(SdfPrimitive::Sphere { radius: 0.5 })
            .material(0)
            .build();
        assert!(state.world.is_alive(entity));

        let action = crate::undo::UndoAction {
            kind: crate::undo::UndoActionKind::SpawnEntity {
                entity_id: entity.to_u64(),
            },
            timestamp_ms: 0,
            description: "Spawn".into(),
        };

        // Undo spawn = despawn.
        state.apply_undo_action(&action, true);
        assert!(!state.world.is_alive(entity));
    }

    #[test]
    fn undo_with_empty_stack_noop() {
        let mut state = EditorState::new();
        // No actions pushed — undo/redo should return None silently.
        assert!(state.undo.undo().is_none());
        assert!(state.undo.redo().is_none());
    }
}
