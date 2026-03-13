//! UI data types for engine→UI communication.
//!
//! These lightweight summary structs are pushed from the engine thread
//! into reactive Signals via `run_on_main_thread`. No snapshot struct,
//! no ArcSwap — data flows directly into the reactive system.

use glam::Vec3;
use uuid::Uuid;

use crate::light_editor::SceneLightType;

/// Per-material voxel usage within the selected object.
#[derive(Debug, Clone)]
pub struct ObjectMaterialUsage {
    pub material_id: u16,
    pub voxel_count: u32,
}

/// What kind of SDF source an object has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    /// No SDF source.
    None,
    /// Analytical SDF primitive.
    Analytical,
    /// Voxelized SDF.
    Voxelized,
}

/// Lightweight summary of a scene object for UI display.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub id: Uuid,
    pub name: String,
    pub position: Vec3,
    pub rotation_degrees: Vec3,
    pub scale: Vec3,
    pub parent_id: Option<Uuid>,
    pub object_type: ObjectType,
}

/// Lightweight summary of a material slot for UI display.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialSummary {
    pub slot: u16,
    pub name: String,
    pub category: String,
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emission_strength: f32,
    pub emission_color: [f32; 3],
    pub subsurface: f32,
    pub subsurface_color: [f32; 3],
    pub opacity: f32,
    pub ior: f32,
    pub noise_scale: f32,
    pub noise_strength: f32,
    pub noise_channels: u32,
    pub shader_name: String,
}

/// Lightweight summary of a light for UI display.
#[derive(Debug, Clone)]
pub struct LightSummary {
    pub id: u64,
    pub light_type: SceneLightType,
    pub position: Vec3,
    pub intensity: f32,
    pub range: f32,
}

/// Lightweight summary of a shader for UI display.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderSummary {
    pub name: String,
    pub id: u32,
    pub built_in: bool,
    pub file_path: String,
}

/// Lightweight summary of a registered behavior system for UI display.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemSummary {
    /// System function name (e.g., "patrol_system").
    pub name: String,
    /// Phase label ("Update" or "LateUpdate").
    pub phase: String,
    /// Execution order within its phase (0-based).
    pub order: usize,
    /// Whether the system faulted (panicked) during the last frame.
    pub faulted: bool,
    /// Last frame execution time in microseconds, if available.
    pub last_frame_us: Option<u64>,
}

/// Helper for debug mode display.
pub fn debug_mode_name(mode: u32) -> &'static str {
    match mode {
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
