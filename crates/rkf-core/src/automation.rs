//! Automation API — the engine's control surface for MCP tools and agents.
//!
//! This module defines the [`AutomationApi`] trait implemented by `rkf-runtime`
//! and called by `rkf-mcp`. All observation methods are safe in any mode;
//! mutation methods require editor mode.
//!
//! A [`StubAutomationApi`] is provided for testing and as a placeholder until
//! runtime implements the trait.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors returned by [`AutomationApi`] methods.
#[derive(Debug, Error)]
pub enum AutomationError {
    /// The method has not yet been implemented by the engine.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// No entity with the given ID exists in the scene.
    #[error("entity not found: {0}")]
    EntityNotFound(u64),

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
    /// Unique entity identifier.
    pub id: u64,
    /// Human-readable name (may be empty).
    pub name: String,
    /// Parent entity id, or `None` for root entities.
    pub parent: Option<u64>,
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
    /// Entity identifier.
    pub id: u64,
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

/// Streaming and upload queue status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetStatusReport {
    /// Number of chunks currently resident in the brick pool.
    pub loaded_chunks: u32,
    /// Number of chunks queued for GPU upload.
    pub pending_uploads: u32,
    /// Total number of allocated bricks across all resident chunks.
    pub total_bricks: u64,
    /// Maximum number of bricks the pool can hold.
    pub pool_capacity: u64,
}

/// Severity level of a log entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Detailed diagnostic information.
    Trace,
    /// General informational messages.
    Info,
    /// Potential issues that do not prevent execution.
    Warn,
    /// Errors that may degrade functionality.
    Error,
}

/// A single captured log message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Severity level.
    pub level: LogLevel,
    /// Message text.
    pub message: String,
    /// Milliseconds since engine start.
    pub timestamp_ms: u64,
}

/// Low-level brick pool occupancy counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickPoolStats {
    /// Total brick slots in the pool.
    pub capacity: u64,
    /// Currently allocated brick slots.
    pub allocated: u64,
    /// Number of entries in the free-list (should equal `capacity - allocated`).
    pub free_list_size: u64,
}

/// Result of evaluating the distance field at a world-space point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialQueryResult {
    /// Signed distance in metres — negative means the point is inside geometry.
    pub distance: f32,
    /// Material id of the nearest surface voxel.
    pub material_id: u16,
    /// `true` if the point is inside (distance < 0).
    pub inside: bool,
}

/// Compact 3D shape overview of an object at brick granularity.
///
/// Each brick (8x8x8 voxels) is categorized as empty (`.`), interior (`#`),
/// or surface/allocated (`+`). Per-Y-level ASCII slices provide a spatial
/// overview in a single MCP call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectShapeResult {
    /// Scene object ID.
    pub object_id: u32,
    /// Brick grid dimensions `[x, y, z]`.
    pub dims: [u32; 3],
    /// Voxel size in metres.
    pub voxel_size: f32,
    /// Object-local AABB minimum `[x, y, z]`.
    pub aabb_min: [f32; 3],
    /// Object-local AABB maximum `[x, y, z]`.
    pub aabb_max: [f32; 3],
    /// Number of empty brick slots.
    pub empty_count: u32,
    /// Number of interior brick slots.
    pub interior_count: u32,
    /// Number of surface/allocated brick slots.
    pub surface_count: u32,
    /// Per-Y-level ASCII slices (index = brick Y coordinate).
    /// Each string is Z rows of X characters, newline-separated.
    pub y_slices: Vec<String>,
}

/// Result of sampling a 2D slice of voxel SDF distances for an object.
///
/// The slice is an XZ grid at a fixed Y coordinate in object-local space.
/// Used for diagnosing sculpt artifacts by inspecting raw voxel data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoxelSliceResult {
    /// Origin of the grid in object-local space `[x_min, z_min]`.
    pub origin: [f32; 2],
    /// Sample spacing (= voxel_size).
    pub spacing: f32,
    /// Number of samples along X.
    pub width: u32,
    /// Number of samples along Z.
    pub height: u32,
    /// The Y coordinate of the slice in object-local space.
    pub y_coord: f32,
    /// Row-major distance values (width * height floats).
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

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// The engine's automation surface — called by `rkf-mcp`.
///
/// Implemented by `rkf-runtime` to provide read and write access to engine
/// state from MCP tools and AI agents.
///
/// # Mode restrictions
/// Observation methods are available in **Editor** and **Debug** modes.
/// Mutation methods (`entity_spawn`, `material_set`, etc.) are only available
/// in **Editor** mode; they return [`AutomationError::EngineError`] otherwise.
pub trait AutomationApi: Send + Sync {
    // --- Observation -------------------------------------------------------

    /// Capture the current viewport as a PNG-encoded byte vector.
    ///
    /// The engine will render an off-screen frame at the requested resolution.
    fn screenshot(&self, width: u32, height: u32) -> AutomationResult<Vec<u8>>;

    /// Capture the full editor window (UI + viewport composited) as a PNG-encoded byte vector.
    ///
    /// Uses the rinch debug protocol to read the composited window texture.
    fn screenshot_window(&self) -> AutomationResult<Vec<u8>> {
        Err(AutomationError::NotImplemented("screenshot_window"))
    }

    /// Return the full scene entity hierarchy.
    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot>;

    /// Return all components attached to the entity with the given id.
    fn entity_inspect(&self, entity_id: u64) -> AutomationResult<EntitySnapshot>;

    /// Return the most recent frame timing and resource statistics.
    fn render_stats(&self) -> AutomationResult<RenderStats>;

    /// Return brick pool streaming and upload queue status.
    fn asset_status(&self) -> AutomationResult<AssetStatusReport>;

    /// Return the last `lines` log entries (most recent last).
    fn read_log(&self, lines: usize) -> AutomationResult<Vec<LogEntry>>;

    /// Return the current camera position, orientation, and projection.
    fn camera_state(&self) -> AutomationResult<CameraSnapshot>;

    /// Return raw brick pool occupancy counters.
    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats>;

    /// Sample the signed distance field at a world-space position.
    ///
    /// `chunk` is the integer chunk coordinate; `local` is the sub-chunk
    /// position in metres.
    fn spatial_query(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult>;

    // --- Mutation (editor mode only) ---------------------------------------

    /// Spawn a new entity described by `def` and return its id.
    fn entity_spawn(&self, def: EntityDef) -> AutomationResult<u64>;

    /// Remove the entity with the given id from the scene.
    fn entity_despawn(&self, entity_id: u64) -> AutomationResult<()>;

    /// Add or replace a component on the given entity.
    fn entity_set_component(
        &self,
        entity_id: u64,
        component: ComponentDef,
    ) -> AutomationResult<()>;

    /// Update fields of the material at slot `id` in the global material table.
    fn material_set(&self, id: u16, material: MaterialDef) -> AutomationResult<()>;

    /// Apply a CSG / sculpt brush operation described by `op`.
    ///
    /// The `op` JSON object is forwarded to `rkf-edit`; its schema is defined
    /// in that crate's brush operation registry.
    fn brush_apply(&self, op: serde_json::Value) -> AutomationResult<()>;

    /// Load and activate the scene at the given `.rkscene` path.
    fn scene_load(&self, path: &str) -> AutomationResult<()>;

    /// Serialise the current scene to the given `.rkscene` path.
    fn scene_save(&self, path: &str) -> AutomationResult<()>;

    /// Teleport the camera to the given world-space position and orientation.
    fn camera_set(
        &self,
        chunk: [i32; 3],
        local: [f32; 3],
        rotation: [f32; 4],
    ) -> AutomationResult<()>;

    /// Switch the renderer to the given quality preset.
    fn quality_preset(&self, preset: QualityPreset) -> AutomationResult<()>;

    /// Execute an engine console command and return its output string.
    fn execute_command(&self, command: &str) -> AutomationResult<String>;

    // --- Tool discovery (default: empty/unsupported) -----------------------

    /// List available MCP tools from the engine as a JSON array of tool definition objects.
    ///
    /// Each element has `name`, `description`, and `inputSchema` fields matching
    /// the MCP `tools/list` response schema.
    fn list_tools_json(&self) -> AutomationResult<serde_json::Value> {
        Ok(serde_json::json!([]))
    }

    /// Forward a raw MCP tool call to the engine.
    ///
    /// Returns the full `ToolsCallResult` JSON (content array + isError flag).
    #[allow(unused_variables)]
    fn call_tool_json(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> AutomationResult<serde_json::Value> {
        Err(AutomationError::NotImplemented("call_tool_json"))
    }

    // --- v2 object-centric methods (default: unsupported) ------------------

    /// Spawn a new SDF object in the v2 scene and return its object ID.
    ///
    /// `primitive_type` is a string such as `"sphere"`, `"box"`, `"capsule"`, etc.
    /// `params` are primitive-specific shape parameters (radius, half-extents, …).
    /// `position` is the initial world-space position in metres.
    /// `material_id` is the index into the global material table.
    #[allow(unused_variables)]
    fn object_spawn(
        &self,
        name: &str,
        primitive_type: &str,
        params: &[f32],
        position: [f32; 3],
        material_id: u16,
    ) -> Result<u32, String> {
        Err("object_spawn not supported".into())
    }

    /// Despawn an SDF object by ID, removing it from the v2 scene.
    #[allow(unused_variables)]
    fn object_despawn(&self, object_id: u32) -> Result<(), String> {
        Err("object_despawn not supported".into())
    }

    /// Set a node's local transform in the v2 scene.
    ///
    /// `position` is the local translation in metres, `rotation` is a unit
    /// quaternion `[x, y, z, w]`, and `scale` is per-axis `[sx, sy, sz]`.
    #[allow(unused_variables)]
    fn node_set_transform(
        &self,
        object_id: u32,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
    ) -> Result<(), String> {
        Err("node_set_transform not supported".into())
    }

    /// Return a human-readable description of the current environment profile.
    fn environment_get(&self) -> Result<String, String> {
        Err("environment_get not supported".into())
    }

    /// Begin blending the environment toward the profile at `target_index` over
    /// `duration` seconds.
    #[allow(unused_variables)]
    fn environment_blend(&self, target_index: usize, duration: f32) -> Result<(), String> {
        Err("environment_blend not supported".into())
    }

    /// Override a single environment property by name.
    ///
    /// `property` is a dot-separated path such as `"sun.intensity"` or
    /// `"fog.density"`.  `value` is the new scalar value.
    #[allow(unused_variables)]
    fn env_override(&self, property: &str, value: f32) -> Result<(), String> {
        Err("env_override not supported".into())
    }

    // --- Node tree operations (default: unsupported) -------------------------

    /// Find a node by name within an object's scene node tree.
    /// Returns a JSON string describing the node.
    #[allow(unused_variables)]
    fn node_find(&self, object_id: u32, node_name: &str) -> Result<String, String> {
        Err("node_find not supported".into())
    }

    /// Add a child node to a named parent node within an object's tree.
    #[allow(unused_variables)]
    fn node_add_child(
        &self,
        object_id: u32,
        parent_node: &str,
        child_primitive: &str,
        params: &[f32],
        name: &str,
        material_id: u16,
    ) -> Result<(), String> {
        Err("node_add_child not supported".into())
    }

    /// Remove a named node from an object's tree.
    #[allow(unused_variables)]
    fn node_remove(&self, object_id: u32, node_name: &str) -> Result<(), String> {
        Err("node_remove not supported".into())
    }

    // --- Multi-scene operations (default: unsupported) -----------------------

    /// Create a new empty scene, returning its index.
    #[allow(unused_variables)]
    fn scene_create(&self, name: &str) -> Result<usize, String> {
        Err("scene_create not supported".into())
    }

    /// List all scenes as a JSON string.
    fn scene_list(&self) -> Result<String, String> {
        Err("scene_list not supported".into())
    }

    /// Set the active scene by index.
    #[allow(unused_variables)]
    fn scene_set_active(&self, index: usize) -> Result<(), String> {
        Err("scene_set_active not supported".into())
    }

    /// Mark a scene as persistent or not.
    #[allow(unused_variables)]
    fn scene_set_persistent(&self, index: usize, persistent: bool) -> Result<(), String> {
        Err("scene_set_persistent not supported".into())
    }

    /// Swap scenes: unload non-persistent scenes. Returns names of removed scenes.
    fn scene_swap(&self) -> Result<String, String> {
        Err("scene_swap not supported".into())
    }

    // --- Camera entity operations (default: unsupported) ---------------------

    /// Spawn a camera entity. Returns the entity ID as u64.
    #[allow(unused_variables)]
    fn camera_spawn(
        &self,
        label: &str,
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        fov: f32,
    ) -> Result<u64, String> {
        Err("camera_spawn not supported".into())
    }

    /// List all camera entities as a JSON string.
    fn camera_list(&self) -> Result<String, String> {
        Err("camera_list not supported".into())
    }

    /// Snap the rendering camera to a camera entity.
    #[allow(unused_variables)]
    fn camera_snap_to(&self, entity_id: u64) -> Result<(), String> {
        Err("camera_snap_to not supported".into())
    }

    // --- Diagnostic tools (default: unsupported) -----------------------------

    /// Sample a 2D XZ slice of raw SDF distances from an object's voxel data.
    ///
    /// `object_id` identifies the scene object. `y_coord` is the object-local
    /// Y coordinate for the slice. Returns a grid of distance values and
    /// per-sample brick slot status.
    #[allow(unused_variables)]
    fn voxel_slice(
        &self,
        object_id: u32,
        y_coord: f32,
    ) -> AutomationResult<VoxelSliceResult> {
        Err(AutomationError::NotImplemented("voxel_slice"))
    }

    /// Apply a single sculpt brush hit to an object via MCP.
    ///
    /// Creates a one-shot undo entry. `position` is world-space `[x, y, z]`.
    /// `mode` is `"add"`, `"subtract"`, or `"smooth"`.
    #[allow(unused_variables)]
    fn sculpt_apply(
        &self,
        object_id: u32,
        position: [f32; 3],
        mode: &str,
        radius: f32,
        strength: f32,
        material_id: u16,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("sculpt_apply"))
    }

    /// Return a compact brick-level 3D shape overview of an object.
    ///
    /// Each brick is categorized as empty, interior, or surface/allocated.
    /// Returns per-Y-level ASCII slices for spatial reasoning.
    #[allow(unused_variables)]
    fn object_shape(&self, object_id: u32) -> AutomationResult<ObjectShapeResult> {
        Err(AutomationError::NotImplemented("object_shape"))
    }

    // --- Material library methods (default: unsupported) ---------------------

    /// List all materials in the library with summary info.
    fn material_list(&self) -> AutomationResult<Vec<MaterialInfo>> {
        Err(AutomationError::NotImplemented("material_list"))
    }

    /// Get full properties of a material at the given slot.
    #[allow(unused_variables)]
    fn material_get(&self, slot: u16) -> AutomationResult<MaterialSnapshot> {
        Err(AutomationError::NotImplemented("material_get"))
    }

    /// List available shader models (name, id, built_in).
    fn shader_list(&self) -> AutomationResult<Vec<ShaderInfo>> {
        Err(AutomationError::NotImplemented("shader_list"))
    }
}

// ---------------------------------------------------------------------------
// Stub implementation
// ---------------------------------------------------------------------------

/// Stub implementation that returns [`AutomationError::NotImplemented`] for
/// every method.
///
/// Used during testing and as a placeholder before `rkf-runtime` provides a
/// real implementation.
pub struct StubAutomationApi;

impl AutomationApi for StubAutomationApi {
    fn screenshot(&self, _width: u32, _height: u32) -> AutomationResult<Vec<u8>> {
        Err(AutomationError::NotImplemented("screenshot"))
    }

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        Err(AutomationError::NotImplemented("scene_graph"))
    }

    fn entity_inspect(&self, _entity_id: u64) -> AutomationResult<EntitySnapshot> {
        Err(AutomationError::NotImplemented("entity_inspect"))
    }

    fn render_stats(&self) -> AutomationResult<RenderStats> {
        Err(AutomationError::NotImplemented("render_stats"))
    }

    fn asset_status(&self) -> AutomationResult<AssetStatusReport> {
        Err(AutomationError::NotImplemented("asset_status"))
    }

    fn read_log(&self, _lines: usize) -> AutomationResult<Vec<LogEntry>> {
        Err(AutomationError::NotImplemented("read_log"))
    }

    fn camera_state(&self) -> AutomationResult<CameraSnapshot> {
        Err(AutomationError::NotImplemented("camera_state"))
    }

    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats> {
        Err(AutomationError::NotImplemented("brick_pool_stats"))
    }

    fn spatial_query(
        &self,
        _chunk: [i32; 3],
        _local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
        Err(AutomationError::NotImplemented("spatial_query"))
    }

    fn entity_spawn(&self, _def: EntityDef) -> AutomationResult<u64> {
        Err(AutomationError::NotImplemented("entity_spawn"))
    }

    fn entity_despawn(&self, _entity_id: u64) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_despawn"))
    }

    fn entity_set_component(
        &self,
        _entity_id: u64,
        _component: ComponentDef,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_set_component"))
    }

    fn material_set(&self, _id: u16, _material: MaterialDef) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("material_set"))
    }

    fn brush_apply(&self, _op: serde_json::Value) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("brush_apply"))
    }

    fn scene_load(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_load"))
    }

    fn scene_save(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_save"))
    }

    fn camera_set(
        &self,
        _chunk: [i32; 3],
        _local: [f32; 3],
        _rotation: [f32; 4],
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("camera_set"))
    }

    fn quality_preset(&self, _preset: QualityPreset) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("quality_preset"))
    }

    fn execute_command(&self, _command: &str) -> AutomationResult<String> {
        Err(AutomationError::NotImplemented("execute_command"))
    }

    // The v2 methods have default implementations in the trait that return
    // Err("... not supported").  StubAutomationApi inherits those defaults and
    // does not need to override them; the trait defaults are tested directly.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time object-safety check — if the trait is not object-safe this
    // function will fail to compile.
    fn _assert_object_safe(_: &dyn AutomationApi) {}

    // Verify StubAutomationApi can be constructed.
    fn make_stub() -> StubAutomationApi {
        StubAutomationApi
    }

    // Helper: assert that an AutomationResult is the NotImplemented variant.
    fn assert_not_implemented<T: std::fmt::Debug>(result: AutomationResult<T>) {
        match result {
            Err(AutomationError::NotImplemented(_)) => {}
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn stub_screenshot_not_implemented() {
        assert_not_implemented(make_stub().screenshot(1920, 1080));
    }

    #[test]
    fn stub_scene_graph_not_implemented() {
        assert_not_implemented(make_stub().scene_graph());
    }

    #[test]
    fn stub_entity_inspect_not_implemented() {
        assert_not_implemented(make_stub().entity_inspect(42));
    }

    #[test]
    fn stub_render_stats_not_implemented() {
        assert_not_implemented(make_stub().render_stats());
    }

    #[test]
    fn stub_asset_status_not_implemented() {
        assert_not_implemented(make_stub().asset_status());
    }

    #[test]
    fn stub_read_log_not_implemented() {
        assert_not_implemented(make_stub().read_log(10));
    }

    #[test]
    fn stub_camera_state_not_implemented() {
        assert_not_implemented(make_stub().camera_state());
    }

    #[test]
    fn stub_brick_pool_stats_not_implemented() {
        assert_not_implemented(make_stub().brick_pool_stats());
    }

    #[test]
    fn stub_spatial_query_not_implemented() {
        assert_not_implemented(make_stub().spatial_query([0, 0, 0], [0.0, 0.0, 0.0]));
    }

    #[test]
    fn stub_entity_spawn_not_implemented() {
        let def = EntityDef {
            name: "test".into(),
            components: vec![],
        };
        assert_not_implemented(make_stub().entity_spawn(def));
    }

    #[test]
    fn stub_entity_despawn_not_implemented() {
        assert_not_implemented(make_stub().entity_despawn(1));
    }

    #[test]
    fn stub_entity_set_component_not_implemented() {
        let comp = ComponentDef::Custom {
            type_name: "Foo".into(),
            data: serde_json::Value::Null,
        };
        assert_not_implemented(make_stub().entity_set_component(1, comp));
    }

    #[test]
    fn stub_material_set_not_implemented() {
        assert_not_implemented(make_stub().material_set(0, MaterialDef::default()));
    }

    #[test]
    fn stub_brush_apply_not_implemented() {
        assert_not_implemented(make_stub().brush_apply(serde_json::json!({})));
    }

    #[test]
    fn stub_scene_load_not_implemented() {
        assert_not_implemented(make_stub().scene_load("foo.rkscene"));
    }

    #[test]
    fn stub_scene_save_not_implemented() {
        assert_not_implemented(make_stub().scene_save("foo.rkscene"));
    }

    #[test]
    fn stub_camera_set_not_implemented() {
        assert_not_implemented(make_stub().camera_set([0, 0, 0], [0.0; 3], [0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn stub_quality_preset_not_implemented() {
        assert_not_implemented(make_stub().quality_preset(QualityPreset::High));
    }

    #[test]
    fn stub_execute_command_not_implemented() {
        assert_not_implemented(make_stub().execute_command("help"));
    }

    // --- Serde round-trip tests for all result types -----------------------

    #[test]
    fn scene_graph_snapshot_serde_roundtrip() {
        let snap = SceneGraphSnapshot {
            entities: vec![EntityNode {
                id: 1,
                name: "root".into(),
                parent: None,
                entity_type: "sdf_object".into(),
                transform: [0.0; 10],
            }],
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: SceneGraphSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.entities.len(), 1);
        assert_eq!(back.entities[0].id, 1);
    }

    #[test]
    fn entity_snapshot_serde_roundtrip() {
        let snap = EntitySnapshot {
            id: 7,
            name: "lamp".into(),
            components: HashMap::from([("light".into(), serde_json::json!({"intensity": 100.0}))]),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: EntitySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, 7);
        assert!(back.components.contains_key("light"));
    }

    #[test]
    fn render_stats_serde_roundtrip() {
        let stats = RenderStats {
            frame_time_ms: 16.6,
            pass_timings: HashMap::from([("ray_march".into(), 8.3)]),
            brick_pool_usage: 0.42,
            memory_mb: 512.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: RenderStats = serde_json::from_str(&json).unwrap();
        assert!((back.frame_time_ms - 16.6).abs() < 1e-9);
    }

    #[test]
    fn camera_snapshot_serde_roundtrip() {
        let cam = CameraSnapshot {
            chunk: [1, 2, 3],
            local: [0.5, 1.0, -0.5],
            rotation: [0.0, 0.0, 0.0, 1.0],
            fov_degrees: 75.0,
        };
        let json = serde_json::to_string(&cam).unwrap();
        let back: CameraSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.chunk, [1, 2, 3]);
        assert!((back.fov_degrees - 75.0).abs() < 1e-6);
    }

    #[test]
    fn asset_status_serde_roundtrip() {
        let report = AssetStatusReport {
            loaded_chunks: 16,
            pending_uploads: 4,
            total_bricks: 8192,
            pool_capacity: 65536,
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: AssetStatusReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.loaded_chunks, 16);
    }

    #[test]
    fn log_entry_serde_roundtrip() {
        let entry = LogEntry {
            level: LogLevel::Warn,
            message: "brick pool near capacity".into(),
            timestamp_ms: 12345,
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: LogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level, LogLevel::Warn);
        assert_eq!(back.timestamp_ms, 12345);
    }

    #[test]
    fn brick_pool_stats_serde_roundtrip() {
        let stats = BrickPoolStats {
            capacity: 65536,
            allocated: 1024,
            free_list_size: 64512,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: BrickPoolStats = serde_json::from_str(&json).unwrap();
        assert_eq!(back.allocated, 1024);
    }

    #[test]
    fn object_shape_result_serde_roundtrip() {
        let res = ObjectShapeResult {
            object_id: 2,
            dims: [32, 16, 32],
            voxel_size: 0.03,
            aabb_min: [-3.84, -1.92, -3.84],
            aabb_max: [3.84, 1.92, 3.84],
            empty_count: 14000,
            interior_count: 200,
            surface_count: 2184,
            y_slices: vec!["....++++....".to_string()],
        };
        let json = serde_json::to_string(&res).unwrap();
        let back: ObjectShapeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.object_id, 2);
        assert_eq!(back.dims, [32, 16, 32]);
        assert_eq!(back.y_slices.len(), 1);
    }

    #[test]
    fn voxel_slice_result_serde_roundtrip() {
        let res = VoxelSliceResult {
            origin: [-0.5, -0.5],
            spacing: 0.04,
            width: 25,
            height: 25,
            y_coord: 0.0,
            distances: vec![0.1; 625],
            slot_status: vec![2; 625],
        };
        let json = serde_json::to_string(&res).unwrap();
        let back: VoxelSliceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.width, 25);
        assert_eq!(back.height, 25);
        assert!((back.spacing - 0.04).abs() < 1e-6);
        assert_eq!(back.distances.len(), 625);
    }

    #[test]
    fn spatial_query_result_serde_roundtrip() {
        let res = SpatialQueryResult {
            distance: -0.05,
            material_id: 7,
            inside: true,
        };
        let json = serde_json::to_string(&res).unwrap();
        let back: SpatialQueryResult = serde_json::from_str(&json).unwrap();
        assert!(back.inside);
        assert_eq!(back.material_id, 7);
    }

    #[test]
    fn stub_material_list_not_implemented() {
        assert_not_implemented(make_stub().material_list());
    }

    #[test]
    fn stub_material_get_not_implemented() {
        assert_not_implemented(make_stub().material_get(0));
    }

    #[test]
    fn material_info_serde_roundtrip() {
        let info = MaterialInfo {
            slot: 1,
            name: "Stone".into(),
            category: "Stone".into(),
            albedo: [0.45, 0.43, 0.40],
            roughness: 0.85,
            metallic: 0.0,
            is_emissive: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: MaterialInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.slot, 1);
        assert_eq!(back.name, "Stone");
    }

    #[test]
    fn material_snapshot_serde_roundtrip() {
        let snap = MaterialSnapshot {
            slot: 2,
            name: "Gold".into(),
            description: "Shiny metallic gold".into(),
            category: "Metal".into(),
            albedo: [1.0, 0.84, 0.0],
            roughness: 0.2,
            metallic: 1.0,
            emission_color: [0.0, 0.0, 0.0],
            emission_strength: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0, 1.0, 1.0],
            opacity: 1.0,
            ior: 1.5,
            noise_scale: 0.0,
            noise_strength: 0.0,
            noise_channels: 0,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: MaterialSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(back.slot, 2);
        assert_eq!(back.name, "Gold");
        assert!((back.metallic - 1.0).abs() < 1e-6);
    }

    #[test]
    fn material_def_serde_roundtrip() {
        let mat = MaterialDef {
            base_color: Some([1.0, 0.0, 0.0]),
            metallic: Some(0.0),
            roughness: Some(0.5),
            ..Default::default()
        };
        let json = serde_json::to_string(&mat).unwrap();
        let back: MaterialDef = serde_json::from_str(&json).unwrap();
        assert_eq!(back.base_color, Some([1.0, 0.0, 0.0]));
        assert_eq!(back.sss_radius, None);
    }

    #[test]
    fn component_def_custom_serde_roundtrip() {
        let comp = ComponentDef::Custom {
            type_name: "rkf_runtime::Foo".into(),
            data: serde_json::json!({"bar": 42}),
        };
        let json = serde_json::to_string(&comp).unwrap();
        let back: ComponentDef = serde_json::from_str(&json).unwrap();
        match back {
            ComponentDef::Custom { type_name, .. } => {
                assert_eq!(type_name, "rkf_runtime::Foo");
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn quality_preset_serde_roundtrip() {
        for preset in [
            QualityPreset::Low,
            QualityPreset::Medium,
            QualityPreset::High,
            QualityPreset::Ultra,
            QualityPreset::Custom,
        ] {
            let json = serde_json::to_string(&preset).unwrap();
            let back: QualityPreset = serde_json::from_str(&json).unwrap();
            assert_eq!(back, preset);
        }
    }

    // --- Tool discovery default method tests --------------------------------

    #[test]
    fn default_list_tools_json_returns_empty_array() {
        let result = make_stub().list_tools_json().unwrap();
        assert_eq!(result, serde_json::json!([]));
    }

    #[test]
    fn default_call_tool_json_returns_not_implemented() {
        assert_not_implemented(make_stub().call_tool_json("foo", serde_json::json!({})));
    }

    // --- v2 default method tests -------------------------------------------

    // Helper: assert a Result<_, String> is Err containing the expected text.
    fn assert_err_contains<T: std::fmt::Debug>(result: Result<T, String>, needle: &str) {
        match result {
            Err(msg) => assert!(
                msg.contains(needle),
                "expected error containing {:?}, got {:?}",
                needle,
                msg
            ),
            Ok(v) => panic!("expected Err, got Ok({v:?})"),
        }
    }

    #[test]
    fn default_object_spawn_returns_error() {
        assert_err_contains(
            make_stub().object_spawn("cube", "box", &[0.5, 0.5, 0.5], [0.0; 3], 0),
            "not supported",
        );
    }

    #[test]
    fn default_object_despawn_returns_error() {
        assert_err_contains(make_stub().object_despawn(1), "not supported");
    }

    #[test]
    fn default_node_set_transform_returns_error() {
        assert_err_contains(
            make_stub().node_set_transform(1, [0.0; 3], [0.0, 0.0, 0.0, 1.0], [1.0; 3]),
            "not supported",
        );
    }

    #[test]
    fn default_environment_get_returns_error() {
        assert_err_contains(make_stub().environment_get(), "not supported");
    }

    #[test]
    fn default_environment_blend_returns_error() {
        assert_err_contains(make_stub().environment_blend(0, 2.0), "not supported");
    }

    #[test]
    fn default_env_override_returns_error() {
        assert_err_contains(make_stub().env_override("sun.intensity", 1.0), "not supported");
    }

    #[test]
    fn automation_error_display() {
        let e = AutomationError::EntityNotFound(99);
        assert!(e.to_string().contains("99"));

        let e2 = AutomationError::InvalidParameter("negative size".into());
        assert!(e2.to_string().contains("negative size"));

        let e3 = AutomationError::NotImplemented("screenshot");
        assert!(e3.to_string().contains("screenshot"));
    }
}
