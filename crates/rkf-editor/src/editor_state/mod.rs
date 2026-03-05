//! Central editor state, aggregating all data model modules.
//!
//! `EditorState` holds instances of every editor subsystem. It is shared via
//! `Arc<Mutex<EditorState>>` between the winit event loop and the rinch component tree.

mod impl_state;
#[cfg(test)]
mod tests;

use crate::animation_preview::AnimationPreview;
use crate::camera::{CameraMode, SceneCamera};
use crate::debug_viz::{DebugOverlay, FrameTimeHistory};
use crate::environment::EnvironmentState;
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
use rkf_core::brick::Brick;
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

    // Animation playback state (0=stopped, 1=playing, 2=paused)
    pub animation_state: Signal<u32>,

    // Material browser
    pub selected_material: Signal<Option<u16>>,
    pub material_revision: Signal<u64>,

    // Properties panel tab (0 = Object, 1 = Asset)
    pub properties_tab: Signal<u32>,

    // Asset browser tab (0 = Materials, 1 = Shaders)
    pub asset_browser_tab: Signal<u32>,

    // Selected shader name (for properties display)
    pub selected_shader: Signal<Option<String>>,

    // Shader drag-and-drop context (from shader card to material shader slot)
    pub shader_drag: DragContext<String>,

    // Whether the shader drop target is being hovered during a drag
    pub shader_drop_highlight: Signal<bool>,

    // Material drag-and-drop context (from asset browser material card to object material row)
    pub material_drag: DragContext<u16>,

    // Which material row is highlighted during drag (stores from_material_id)
    pub material_drop_highlight: Signal<Option<u16>>,

    /// Counter incremented on drag-drop completion. The viewport's MouseUp
    /// handler compares against its own snapshot to suppress the spurious
    /// MouseUp that rinch dispatches after ondrop+ondragend.
    pub drag_drop_generation: Signal<u64>,
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
            animation_state: Signal::new(0),
            selected_material: Signal::new(None),
            material_revision: Signal::new(0),
            properties_tab: Signal::new(0),
            asset_browser_tab: Signal::new(0),
            selected_shader: Signal::new(None),
            shader_drag: DragContext::new(),
            shader_drop_highlight: Signal::new(false),
            material_drag: DragContext::new(),
            material_drop_highlight: Signal::new(None),
            drag_drop_generation: Signal::new(0),
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

    // Object transform (bound to currently selected object)
    pub obj_pos_x: Signal<f64>,
    pub obj_pos_y: Signal<f64>,
    pub obj_pos_z: Signal<f64>,
    pub obj_rot_x: Signal<f64>,
    pub obj_rot_y: Signal<f64>,
    pub obj_rot_z: Signal<f64>,
    pub obj_scale_x: Signal<f64>,
    pub obj_scale_y: Signal<f64>,
    pub obj_scale_z: Signal<f64>,
    pub bound_object_id: Signal<Option<u64>>,

    // Light properties (bound to currently selected light)
    pub light_pos_x: Signal<f64>,
    pub light_pos_y: Signal<f64>,
    pub light_pos_z: Signal<f64>,
    pub light_intensity: Signal<f64>,
    pub light_range: Signal<f64>,
    pub bound_light_id: Signal<Option<u64>>,
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
            // Object transform — initialized empty, populated on selection change.
            obj_pos_x: Signal::new(0.0),
            obj_pos_y: Signal::new(0.0),
            obj_pos_z: Signal::new(0.0),
            obj_rot_x: Signal::new(0.0),
            obj_rot_y: Signal::new(0.0),
            obj_rot_z: Signal::new(0.0),
            obj_scale_x: Signal::new(1.0),
            obj_scale_y: Signal::new(1.0),
            obj_scale_z: Signal::new(1.0),
            bound_object_id: Signal::new(None),
            // Light properties — initialized empty, populated on selection change.
            light_pos_x: Signal::new(0.0),
            light_pos_y: Signal::new(0.0),
            light_pos_z: Signal::new(0.0),
            light_intensity: Signal::new(1.0),
            light_range: Signal::new(10.0),
            bound_light_id: Signal::new(None),
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
        // Object transform
        let _ = self.obj_pos_x.get();
        let _ = self.obj_pos_y.get();
        let _ = self.obj_pos_z.get();
        let _ = self.obj_rot_x.get();
        let _ = self.obj_rot_y.get();
        let _ = self.obj_rot_z.get();
        let _ = self.obj_scale_x.get();
        let _ = self.obj_scale_y.get();
        let _ = self.obj_scale_z.get();
        // Light properties
        let _ = self.light_pos_x.get();
        let _ = self.light_pos_y.get();
        let _ = self.light_pos_z.get();
        let _ = self.light_intensity.get();
        let _ = self.light_range.get();
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

        // Object transform — write back to scene if bound.
        // Read bound IDs with untracked() to avoid subscribing to selection changes.
        let obj_id = rinch::core::untracked(|| self.bound_object_id.get());
        if let Some(oid) = obj_id {
            let sc = es.world.scene_mut();
            if let Some(obj) = sc.objects.iter_mut().find(|o| o.id as u64 == oid) {
                obj.position.x = self.obj_pos_x.get() as f32;
                obj.position.y = self.obj_pos_y.get() as f32;
                obj.position.z = self.obj_pos_z.get() as f32;
                obj.rotation = glam::Quat::from_euler(
                    glam::EulerRot::XYZ,
                    (self.obj_rot_x.get() as f32).to_radians(),
                    (self.obj_rot_y.get() as f32).to_radians(),
                    (self.obj_rot_z.get() as f32).to_radians(),
                );
                obj.scale.x = self.obj_scale_x.get() as f32;
                obj.scale.y = self.obj_scale_y.get() as f32;
                obj.scale.z = self.obj_scale_z.get() as f32;
            }
        }

        // Light properties — write back if bound.
        let light_id = rinch::core::untracked(|| self.bound_light_id.get());
        if let Some(lid) = light_id {
            if let Some(light) = es.light_editor.get_light_mut(lid) {
                light.position.x = self.light_pos_x.get() as f32;
                light.position.y = self.light_pos_y.get() as f32;
                light.position.z = self.light_pos_z.get() as f32;
                light.intensity = self.light_intensity.get() as f32;
                light.range = self.light_range.get() as f32;
            }
            es.light_editor.mark_dirty();
        }
    }

    /// Push object transform values into the signals (selection changed).
    /// Called from an untracked context to avoid triggering the batch sync Effect.
    pub fn push_object_values(&self, pos: Vec3, rot_deg: Vec3, scale: Vec3) {
        self.obj_pos_x.set(pos.x as f64);
        self.obj_pos_y.set(pos.y as f64);
        self.obj_pos_z.set(pos.z as f64);
        self.obj_rot_x.set(rot_deg.x as f64);
        self.obj_rot_y.set(rot_deg.y as f64);
        self.obj_rot_z.set(rot_deg.z as f64);
        self.obj_scale_x.set(scale.x as f64);
        self.obj_scale_y.set(scale.y as f64);
        self.obj_scale_z.set(scale.z as f64);
    }

    /// Push light property values into the signals (selection changed).
    pub fn push_light_values(&self, pos: Vec3, intensity: f32, range: f32) {
        self.light_pos_x.set(pos.x as f64);
        self.light_pos_y.set(pos.y as f64);
        self.light_pos_z.set(pos.z as f64);
        self.light_intensity.set(intensity as f64);
        self.light_range.set(range as f64);
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

/// Accumulates brick snapshots during a sculpt stroke for undo support.
///
/// Created when a sculpt stroke begins. Before each edit, any not-yet-captured
/// brick slots are snapshot from the CPU brick pool. On stroke end, the
/// accumulated snapshots are pushed as an undo action.
pub struct SculptUndoAccumulator {
    /// The object being sculpted.
    pub object_id: u64,
    /// Brick pool slots that have already been captured.
    pub captured_slots: HashSet<u32>,
    /// Pre-edit snapshots: (slot_index, brick_data_before_edit).
    pub snapshots: Vec<(u32, Brick)>,
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

    // ── Lights & Environment ─────────────────────────────────
    pub light_editor: LightManager,
    pub environment: EnvironmentState,

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
    /// Set by "Fix SDFs" button in sculpt panel. Contains the object ID whose
    /// SDF magnitudes should be recomputed from zero-crossings via BFS.
    pub pending_fix_sdfs: Option<u32>,
    /// Set by material drag-and-drop in object properties panel.
    /// Contains (object_id, from_material, to_material).
    pub pending_remap_material: Option<(u64, u16, u16)>,

    // ── Sculpt pipeline (UI → render loop) ──────────────────
    /// Queued sculpt edit requests — one per brush-hit point during a stroke.
    /// Drained by the render loop each frame and applied to the CPU brick pool.
    pub pending_sculpt_edits: Vec<crate::sculpt::SculptEditRequest>,

    /// Accumulates brick snapshots during a sculpt stroke for undo.
    /// Created on stroke begin, finalized on stroke end.
    pub sculpt_undo_accumulator: Option<SculptUndoAccumulator>,

    /// Pending sculpt undo: brick snapshots to restore.
    /// Set by apply_undo_action, consumed by the render loop.
    pub pending_sculpt_undo: Option<Vec<(u32, Brick)>>,

    // ── World (unified game state) ──────────────────────────
    /// The unified world container. Wraps `Scene` (SDF objects) + ECS +
    /// brick pool. Replaces the former `v2_scene: Option<Scene>`.
    pub world: World,
}
