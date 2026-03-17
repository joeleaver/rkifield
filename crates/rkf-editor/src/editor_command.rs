//! Command types sent from the UI thread to the render/engine thread.
//!
//! The UI thread never locks `EditorState` directly. Instead, it sends
//! `EditorCommand` variants through a `crossbeam::channel`. The engine
//! thread drains these at frame start and applies them to its owned
//! `EditorState`.

use glam::Vec3;
use uuid::Uuid;

use crate::editor_state::{EditorMode, SelectedEntity};
use crate::gizmo::GizmoMode;
use crate::input::{KeyCode, Modifiers};

/// A command from the UI thread to the engine thread.
///
/// Sent via `crossbeam::channel::Sender<EditorCommand>`. The engine thread
/// drains all pending commands at the start of each frame.
pub enum EditorCommand {
    // ── Input ────────────────────────────────────────────────────────────
    MouseMove { x: f32, y: f32, dx: f32, dy: f32 },
    MouseDown { button: usize, x: f32, y: f32 },
    MouseUp { button: usize, x: f32, y: f32 },
    Scroll { delta: f32 },
    KeyDown { key: KeyCode, modifiers: Modifiers },
    KeyUp { key: KeyCode, modifiers: Modifiers },

    // ── Scene mutations ──────────────────────────────────────────────────
    SpawnPrimitive { name: String },
    DeleteSelected,
    DuplicateSelected,
    Undo,
    Redo,
    SelectEntity { entity: Option<SelectedEntity> },

    // ── Gizmo ────────────────────────────────────────────────────────────
    SetGizmoMode { mode: GizmoMode },

    // ── Tool settings ────────────────────────────────────────────────────
    SetEditorMode { mode: EditorMode },
    SetSculptSettings { radius: f32, strength: f32, falloff: f32 },
    SetPaintSettings { radius: f32, strength: f32, falloff: f32 },
    SetPaintMode { mode: crate::paint::PaintMode },
    SetPaintColor { r: f32, g: f32, b: f32 },

    // ── Camera settings ──────────────────────────────────────────────────
    SetCameraFov { fov: f32 },
    SetCameraSpeed { speed: f32 },
    SetCameraNearFar { near: f32, far: f32 },

    // ── Environment ──────────────────────────────────────────────────────
    SetAtmosphere {
        sun_direction: Vec3,
        sun_intensity: f32,
        rayleigh_scale: f32,
        mie_scale: f32,
    },
    SetFog {
        density: f32,
        height_falloff: f32,
        dust_density: f32,
        dust_asymmetry: f32,
    },
    SetClouds {
        coverage: f32,
        density: f32,
        altitude: f32,
        thickness: f32,
        wind_speed: f32,
    },
    SetPostProcess {
        bloom_intensity: f32,
        bloom_threshold: f32,
        exposure: f32,
        sharpen: f32,
        dof_focus_distance: f32,
        dof_focus_range: f32,
        dof_max_coc: f32,
        motion_blur: f32,
        god_rays: f32,
        vignette: f32,
        grain: f32,
        chromatic_aberration: f32,
    },
    ToggleAtmosphere { enabled: bool },
    ToggleFog { enabled: bool },
    ToggleClouds { enabled: bool },
    ToggleBloom { enabled: bool },
    ToggleDof { enabled: bool },
    SetToneMapMode { mode: u32 },

    // ── Lights ────────────────────────────────────────────────────────────
    SetLightPosition { light_id: u64, position: Vec3 },
    SetLightIntensity { light_id: u64, intensity: f32 },
    SetLightRange { light_id: u64, range: f32 },

    // ── Debug / view ─────────────────────────────────────────────────────
    SetDebugMode { mode: u32 },
    ToggleGrid,
    ToggleShortcuts,

    // ── Object properties ────────────────────────────────────────────────
    SetObjectPosition { entity_id: Uuid, position: Vec3 },
    SetObjectRotation { entity_id: Uuid, rotation: Vec3 },
    SetObjectScale { entity_id: Uuid, scale: Vec3 },

    // ── Scene I/O ────────────────────────────────────────────────────────
    OpenScene { path: String },
    SaveScene { path: Option<String> },

    // ── Project I/O ──────────────────────────────────────────────────────
    NewProject,
    OpenProject { path: String },
    SaveProject,
    RemoveRecentProject { path: String },

    // ── Voxel ops ────────────────────────────────────────────────────────
    ConvertToVoxel { object_id: Uuid, voxel_size: f32 },

    // ── Materials ────────────────────────────────────────────────────────
    SelectMaterial { slot: u16 },
    SetMaterial { slot: u16, material: rkf_core::material::Material },
    SetMaterialShader { slot: u16, shader_name: String },
    /// Remap all voxels using `from_material` to `to_material` on a specific object.
    RemapMaterial { object_id: Uuid, from_material: u16, to_material: u16 },
    /// Set the material_id on an analytical primitive.
    SetPrimitiveMaterial { object_id: Uuid, material_id: u16 },

    // ── Animation ────────────────────────────────────────────────────────
    SetAnimationState { state: u32 },
    SetAnimationSpeed { speed: f32 },

    // ── Play mode ────────────────────────────────────────────────────────
    PlayStart,
    PlayStop,

    // ── Component inspector ──────────────────────────────────────────────
    /// Set a field value on a component of an entity.
    SetComponentField {
        entity_id: Uuid,
        component_name: String,
        field_name: String,
        value: rkf_runtime::behavior::game_value::GameValue,
    },
    /// Add a component (with defaults) to an entity.
    AddComponent {
        entity_id: Uuid,
        component_name: String,
    },
    /// Remove a component from an entity.
    RemoveComponent {
        entity_id: Uuid,
        component_name: String,
    },

    // ── Window management ────────────────────────────────────────────────
    WindowDrag,
    WindowMinimize,
    WindowMaximize,
    RequestExit,
}

// KeyCode and Modifiers are re-exported from crate::input.
