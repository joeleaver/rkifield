//! Read-only snapshot of editor state for the UI thread.
//!
//! Published by the render thread each frame via `ArcSwap<UiSnapshot>`.
//! The UI thread reads the latest snapshot lock-free — never blocks.

use glam::Vec3;

use crate::editor_state::{EditorMode, SelectedEntity};
use crate::gizmo::GizmoMode;
use crate::light_editor::SceneLightType;

/// Per-material voxel usage within the selected object.
#[derive(Debug, Clone)]
pub struct ObjectMaterialUsage {
    pub material_id: u16,
    pub voxel_count: u32,
}

/// Lightweight summary of a scene object for UI display.
#[derive(Debug, Clone)]
pub struct ObjectSummary {
    pub id: u64,
    pub name: String,
    pub position: Vec3,
    pub rotation_degrees: Vec3,
    pub scale: Vec3,
    pub parent_id: Option<u32>,
}

/// Lightweight summary of a material slot for UI display.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct ShaderSummary {
    pub name: String,
    pub id: u32,
    pub built_in: bool,
    pub file_path: String,
}

/// Complete read-only snapshot of editor state for the UI thread.
///
/// Published by the render thread each frame. The UI reads via
/// `arc_swap::ArcSwap<UiSnapshot>::load()` — zero-cost atomic pointer read.
#[derive(Debug, Clone)]
pub struct UiSnapshot {
    // ── Camera ───────────────────────────────────────────────────────
    pub camera_position: Vec3,
    pub camera_yaw: f32,
    pub camera_pitch: f32,
    pub camera_fov: f32,
    pub camera_speed: f32,
    pub camera_near: f32,
    pub camera_far: f32,

    // ── Selection & modes ────────────────────────────────────────────
    pub selected_entity: Option<SelectedEntity>,
    pub mode: EditorMode,
    pub gizmo_mode: GizmoMode,
    pub debug_mode: u32,
    pub show_grid: bool,
    pub show_shortcuts: bool,

    // ── Scene (lightweight for UI display) ───────────────────────────
    pub scene_name: String,
    pub objects: Vec<ObjectSummary>,
    pub lights: Vec<LightSummary>,
    pub scene_revision: u64,

    // ── Brush settings ───────────────────────────────────────────────
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub brush_falloff: f32,

    // ── Animation ────────────────────────────────────────────────────
    pub animation_state: u32,
    pub animation_speed: f32,

    // ── Environment ──────────────────────────────────────────────────
    pub atmo_enabled: bool,
    pub sun_azimuth: f32,
    pub sun_elevation: f32,
    pub sun_intensity: f32,
    pub rayleigh_scale: f32,
    pub mie_scale: f32,

    pub fog_enabled: bool,
    pub fog_density: f32,
    pub fog_height_falloff: f32,
    pub dust_density: f32,
    pub dust_asymmetry: f32,

    pub clouds_enabled: bool,
    pub cloud_coverage: f32,
    pub cloud_density: f32,
    pub cloud_altitude: f32,
    pub cloud_thickness: f32,
    pub cloud_wind_speed: f32,

    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub dof_enabled: bool,
    pub dof_focus_distance: f32,
    pub dof_focus_range: f32,
    pub dof_max_coc: f32,
    pub exposure: f32,
    pub sharpen: f32,
    pub motion_blur: f32,
    pub god_rays: f32,
    pub vignette: f32,
    pub grain: f32,
    pub chromatic_aberration: f32,
    pub tone_map_mode: u32,

    // ── Scene I/O ────────────────────────────────────────────────────
    pub current_scene_path: Option<String>,

    // ── Materials ─────────────────────────────────────────────────────
    pub materials: Vec<MaterialSummary>,
    pub material_revision: u64,

    // ── Shader registry ────────────────────────────────────────────────
    /// Available shaders from the ShaderComposer (for UI display + dropdowns).
    pub shaders: Vec<ShaderSummary>,

    // ── Per-object material usage ───────────────────────────────────
    /// Materials used by the selected voxelized object, sorted by voxel count descending.
    pub selected_object_materials: Vec<ObjectMaterialUsage>,

    // ── Stats ────────────────────────────────────────────────────────
    pub fps_ms: f64,
    pub object_count: usize,
}

impl UiSnapshot {
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

    pub fn gizmo_mode_name(&self) -> &'static str {
        match self.gizmo_mode {
            GizmoMode::Translate => "Translate (W)",
            GizmoMode::Rotate => "Rotate (E)",
            GizmoMode::Scale => "Scale (R)",
        }
    }
}

impl Default for UiSnapshot {
    fn default() -> Self {
        Self {
            camera_position: Vec3::new(0.0, 2.5, 5.0),
            camera_yaw: 0.0,
            camera_pitch: -0.15,
            camera_fov: 70.0,
            camera_speed: 5.0,
            camera_near: 0.1,
            camera_far: 1000.0,
            selected_entity: None,
            mode: EditorMode::Default,
            gizmo_mode: GizmoMode::Translate,
            debug_mode: 0,
            show_grid: false,
            show_shortcuts: false,
            scene_name: "Untitled".into(),
            objects: Vec::new(),
            lights: Vec::new(),
            scene_revision: 0,
            brush_radius: 0.5,
            brush_strength: 0.5,
            brush_falloff: 0.5,
            animation_state: 0,
            animation_speed: 1.0,
            atmo_enabled: true,
            sun_azimuth: 0.0,
            sun_elevation: 45.0,
            sun_intensity: 8.0,
            rayleigh_scale: 1.0,
            mie_scale: 1.0,
            fog_enabled: false,
            fog_density: 0.001,
            fog_height_falloff: 0.5,
            dust_density: 0.0,
            dust_asymmetry: 0.7,
            clouds_enabled: false,
            cloud_coverage: 0.5,
            cloud_density: 0.3,
            cloud_altitude: 800.0,
            cloud_thickness: 400.0,
            cloud_wind_speed: 5.0,
            bloom_enabled: true,
            bloom_intensity: 0.8,
            bloom_threshold: 1.0,
            dof_enabled: false,
            dof_focus_distance: 5.0,
            dof_focus_range: 3.0,
            dof_max_coc: 8.0,
            exposure: 1.0,
            sharpen: 0.3,
            motion_blur: 0.5,
            god_rays: 0.5,
            vignette: 0.2,
            grain: 0.05,
            chromatic_aberration: 0.0,
            tone_map_mode: 0,
            current_scene_path: None,
            materials: Vec::new(),
            material_revision: 0,
            shaders: vec![
                ShaderSummary { name: "pbr".into(), id: 0, built_in: true, file_path: "crates/rkf-render/shaders/shade_pbr.wgsl".into() },
                ShaderSummary { name: "unlit".into(), id: 1, built_in: true, file_path: "crates/rkf-render/shaders/shade_unlit.wgsl".into() },
                ShaderSummary { name: "toon".into(), id: 2, built_in: true, file_path: "crates/rkf-render/shaders/shade_toon.wgsl".into() },
                ShaderSummary { name: "emissive".into(), id: 3, built_in: true, file_path: "crates/rkf-render/shaders/shade_emissive.wgsl".into() },
            ],
            selected_object_materials: Vec::new(),
            fps_ms: 0.0,
            object_count: 0,
        }
    }
}
