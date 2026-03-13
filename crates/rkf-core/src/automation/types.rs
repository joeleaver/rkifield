//! Error type, observation result types, and mutation input types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`AutomationApi`](super::AutomationApi) methods.
#[derive(Debug, Error)]
pub enum AutomationError {
    /// The method has not yet been implemented by the engine.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// No entity with the given ID exists in the scene.
    #[error("entity not found: {0}")]
    EntityNotFound(String),

    /// A parameter value is out of range or malformed.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// An internal engine error occurred.
    #[error("engine error: {0}")]
    EngineError(String),
}

/// Convenience alias for automation results.
pub type AutomationResult<T> = Result<T, AutomationError>;

// ---------------------------------------------------------------------------
// Observation result types
// ---------------------------------------------------------------------------

/// One node in the scene entity tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityNode {
    /// Unique entity identifier (UUID string).
    pub id: String,
    /// Human-readable name (may be empty).
    pub name: String,
    /// Parent entity id (UUID string), or `None` for root entities.
    pub parent: Option<String>,
    /// Coarse classification of the entity (e.g. `"sdf_object"`, `"light"`, `"camera"`).
    pub entity_type: String,
    /// World-space transform: `[tx, ty, tz, rx, ry, rz, rw, sx, sy, sz]`
    /// (translation, quaternion rotation, uniform scale repeated three times).
    pub transform: [f32; 10],
}

/// Snapshot of the full scene entity hierarchy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraphSnapshot {
    /// All entities in the scene, in no guaranteed order.
    pub entities: Vec<EntityNode>,
}

/// All components attached to a single entity, serialised as JSON values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySnapshot {
    /// Entity identifier (UUID string).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Map from component type name to component data.
    pub components: HashMap<String, serde_json::Value>,
}

/// Per-pass GPU timing and overall frame statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderStats {
    /// Total frame time in milliseconds.
    pub frame_time_ms: f64,
    /// Per-pass timings keyed by pass name (e.g. `"ray_march"`, `"shade"`).
    pub pass_timings: HashMap<String, f64>,
    /// Fraction of the brick pool that is currently allocated (`0.0` – `1.0`).
    pub brick_pool_usage: f32,
    /// Approximate GPU memory used in megabytes.
    pub memory_mb: f32,
}

/// Current camera position, orientation, and projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSnapshot {
    /// Integer chunk coordinates of the camera's world position `[cx, cy, cz]`.
    pub chunk: [i32; 3],
    /// Sub-chunk local position in metres `[lx, ly, lz]`.
    pub local: [f32; 3],
    /// Rotation as a unit quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
}

/// Brick pool streaming and upload queue status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetStatusReport {
    /// Number of chunks currently loaded.
    pub loaded_chunks: u32,
    /// Number of brick uploads pending.
    pub pending_uploads: u32,
    /// Total number of bricks across all loaded chunks.
    pub total_bricks: u64,
    /// Total brick pool capacity.
    pub pool_capacity: u64,
}

/// Severity level for log entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Trace-level detail (extremely verbose).
    Trace,
    /// Debug-level information.
    Debug,
    /// Informational message.
    Info,
    /// Warning — something unexpected but recoverable.
    Warn,
    /// Error — something failed.
    Error,
}

/// A single log entry from the engine's ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Severity level.
    pub level: LogLevel,
    /// Log message text.
    pub message: String,
    /// Timestamp in milliseconds since engine startup.
    pub timestamp_ms: u64,
}

/// Raw brick pool occupancy counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickPoolStats {
    /// Total number of brick slots.
    pub capacity: u64,
    /// Number of currently allocated slots.
    pub allocated: u64,
    /// Number of slots on the free list.
    pub free_list_size: u64,
}

/// Result of a spatial SDF query at a world-space position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQueryResult {
    /// Signed distance to the nearest surface (negative = inside).
    pub distance: f32,
    /// Material ID at the query position (0 if outside all objects).
    pub material_id: u16,
    /// Whether the query point is inside geometry.
    pub inside: bool,
}

/// Compact brick-level 3D shape overview of an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectShapeResult {
    /// Object ID.
    pub object_id: u32,
    /// Brick-map dimensions `[width, height, depth]`.
    pub dims: [u32; 3],
    /// Voxel size in metres.
    pub voxel_size: f32,
    /// Local-space AABB minimum.
    pub aabb_min: [f32; 3],
    /// Local-space AABB maximum.
    pub aabb_max: [f32; 3],
    /// Number of EMPTY_SLOT bricks.
    pub empty_count: u32,
    /// Number of INTERIOR_SLOT bricks.
    pub interior_count: u32,
    /// Number of allocated (surface) bricks.
    pub surface_count: u32,
    /// Per-Y-level ASCII slices showing brick status.
    pub y_slices: Vec<String>,
}

/// Raw SDF slice data from an object's voxel volume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelSliceResult {
    /// XZ origin of the slice in object-local space.
    pub origin: [f32; 2],
    /// Spacing between samples in metres.
    pub spacing: f32,
    /// Number of columns (X).
    pub width: u32,
    /// Number of rows (Z).
    pub height: u32,
    /// Y coordinate of the slice.
    pub y_coord: f32,
    /// Row-major distance values.
    pub distances: Vec<f32>,
    /// Per-sample slot status: 0=EMPTY, 1=INTERIOR, 2=allocated.
    pub slot_status: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Mutation input types
// ---------------------------------------------------------------------------

/// Describes a component to attach to or update on an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComponentDef {
    /// World-space transform: chunk `[cx,cy,cz]`, local `[lx,ly,lz]`, quat `[rx,ry,rz,rw]`.
    Transform {
        /// Integer chunk coordinates.
        chunk: [i32; 3],
        /// Sub-chunk local coordinates in metres.
        local: [f32; 3],
        /// Unit quaternion rotation `[x, y, z, w]`.
        rotation: [f32; 4],
    },
    /// SDF object source asset path.
    SdfObject {
        /// Path to the `.rkf` asset.
        asset_path: String,
    },
    /// Directional or point light parameters.
    Light {
        /// Colour as linear RGB.
        color: [f32; 3],
        /// Luminous intensity in candela.
        intensity: f32,
        /// `"directional"` or `"point"`.
        kind: String,
    },
    /// Arbitrary component specified by a fully-qualified Rust type name.
    Custom {
        /// Fully-qualified Rust type name (e.g. `"rkf_runtime::MyComponent"`).
        type_name: String,
        /// Component data as a JSON object.
        data: serde_json::Value,
    },
}

/// Describes a new entity to spawn into the scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDef {
    /// Display name for the entity.
    pub name: String,
    /// Initial components to attach.
    pub components: Vec<ComponentDef>,
}

/// Partial update to a material entry in the global material table.
///
/// All fields are optional; omitted fields leave the existing value unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterialDef {
    /// Base colour as linear RGB.
    pub base_color: Option<[f32; 3]>,
    /// Metallic factor `[0, 1]`.
    pub metallic: Option<f32>,
    /// Roughness factor `[0, 1]`.
    pub roughness: Option<f32>,
    /// Emissive colour as linear RGB.
    pub emissive: Option<[f32; 3]>,
    /// Subsurface scattering radius in metres.
    pub sss_radius: Option<f32>,
    /// Index of refraction.
    pub ior: Option<f32>,
    /// Transmission factor `[0, 1]` (glass, translucency).
    pub transmission: Option<f32>,
    /// Shader model name (e.g. "pbr", "toon", "unlit", "hologram").
    pub shader: Option<String>,
}

/// Summary info about a material slot — returned by `material_list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialInfo {
    /// GPU material table slot index.
    pub slot: u16,
    /// Human-readable name (empty if no metadata).
    pub name: String,
    /// Category (e.g. "Metal", "Stone", "Organic").
    pub category: String,
    /// Base color (linear RGB).
    pub albedo: [f32; 3],
    /// Roughness factor.
    pub roughness: f32,
    /// Metallic factor.
    pub metallic: f32,
    /// Whether this material has non-zero emission.
    pub is_emissive: bool,
}

/// Info about a registered shader model — returned by `shader_list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderInfo {
    /// Shader name (e.g. "pbr", "toon", "hologram").
    pub name: String,
    /// Numeric ID used in the GPU dispatch switch.
    pub id: u32,
    /// Whether this is a built-in shader (vs user-provided).
    pub built_in: bool,
}

/// Full snapshot of a material's properties — returned by `material_get`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialSnapshot {
    /// GPU material table slot index.
    pub slot: u16,
    /// Human-readable name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Category.
    pub category: String,
    /// Base color (linear RGB).
    pub albedo: [f32; 3],
    /// Roughness factor.
    pub roughness: f32,
    /// Metallic factor.
    pub metallic: f32,
    /// Emissive color (linear RGB).
    pub emission_color: [f32; 3],
    /// Emissive intensity.
    pub emission_strength: f32,
    /// Subsurface scattering strength.
    pub subsurface: f32,
    /// Subsurface color.
    pub subsurface_color: [f32; 3],
    /// Opacity (1.0 = solid).
    pub opacity: f32,
    /// Index of refraction.
    pub ior: f32,
    /// Procedural noise scale.
    pub noise_scale: f32,
    /// Procedural noise strength.
    pub noise_strength: f32,
    /// Noise channel bitfield.
    pub noise_channels: u32,
}

// ---------------------------------------------------------------------------
// Behavior system types (Phase 14.1-14.4)
// ---------------------------------------------------------------------------

/// Describes a registered component type and its fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component type name (e.g. `"Health"`, `"Transform"`).
    pub name: String,
    /// Fields exposed on this component.
    pub fields: Vec<FieldInfo>,
}

/// Describes a single field on a component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    /// Field name.
    pub name: String,
    /// Type description (e.g. `"f32"`, `"Vec3"`, `"String"`).
    pub field_type: String,
    /// Optional valid range `(min, max)` for numeric fields.
    pub range: Option<(f64, f64)>,
}

/// Info about a registered behavior system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// System name (e.g. `"gravity"`, `"health_regen"`).
    pub name: String,
    /// Execution phase (e.g. `"update"`, `"fixed_update"`, `"post_update"`).
    pub phase: String,
    /// Whether this system has faulted and been disabled.
    pub faulted: bool,
}

/// Info about a spawnable blueprint (prefab).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintInfo {
    /// Blueprint name.
    pub name: String,
    /// Names of components included in this blueprint.
    pub component_names: Vec<String>,
}

/// Render quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    /// Minimal quality for weak hardware or fast iteration.
    Low,
    /// Balanced quality for mid-range hardware.
    Medium,
    /// High quality suitable for most consumer GPUs.
    High,
    /// Maximum quality for high-end GPUs.
    Ultra,
    /// User-defined quality settings.
    Custom,
}
