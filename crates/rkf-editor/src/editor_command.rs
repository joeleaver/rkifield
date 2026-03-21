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
    SpawnCamera,
    SpawnPointLight,
    SpawnSpotLight,
    /// Place a `.rkf` model at the camera target position.
    PlaceModel { asset_path: String },
    /// Begin dragging a model into the viewport.
    /// Spawns the entity, loads the .rkf, and starts drag-placement mode.
    DragModelEnter { asset_path: String },
    /// Update the dragged model's position based on GPU raycast at mouse position.
    DragModelMove { x: f32, y: f32 },
    /// Finalize the dragged model placement (push undo).
    DragModelDrop,
    /// Cancel the model drag (despawn the entity).
    DragModelCancel,
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
    /// Set orbit yaw/pitch (radians) — for camera presets (Top, Front, etc.).
    SetCameraOrbitAngles { yaw: f32, pitch: f32 },

    // ── Environment ──────────────────────────────────────────────────────
    // Environment settings flow through SetComponentField targeting the
    // active camera entity's EnvironmentSettings component.

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

    // ── Camera linking ─────────────────────────────────────────────────
    /// Set which scene camera drives the viewport (or None for editor camera).
    SetViewportCamera { camera_id: Option<Uuid> },
    /// Link the editor camera's environment to a scene camera's `.rkenv` profile.
    /// None = use editor defaults (no file backing). Slider edits auto-save to
    /// the linked camera's profile when set.
    SetLinkedEnvCamera { camera_id: Option<Uuid> },
    /// Copy a scene camera's transform + FOV to the editor camera.
    SnapToCamera { camera_id: Uuid },
    /// Spawn a new camera entity at the editor camera's current position/rotation/FOV.
    CreateCameraFromView,
    /// Enter/exit pilot mode (editor viewport drives a scene camera entity).
    PilotCamera { camera_id: Option<Uuid> },

    // ── Window management ────────────────────────────────────────────────
    WindowDrag,
    WindowMinimize,
    WindowMaximize,
    RequestExit,
}

// KeyCode and Modifiers are re-exported from crate::input.
