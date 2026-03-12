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

use rkf_runtime::api::World;

use crate::ui_snapshot::{
    LightSummary, MaterialSummary, ObjectMaterialUsage, ObjectSummary, ShaderSummary,
};

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
            shaders: Signal::new(vec![
                ShaderSummary { name: "pbr".into(), id: 0, built_in: true, file_path: "crates/rkf-render/shaders/shade_pbr.wgsl".into() },
                ShaderSummary { name: "unlit".into(), id: 1, built_in: true, file_path: "crates/rkf-render/shaders/shade_unlit.wgsl".into() },
                ShaderSummary { name: "toon".into(), id: 2, built_in: true, file_path: "crates/rkf-render/shaders/shade_toon.wgsl".into() },
                ShaderSummary { name: "emissive".into(), id: 3, built_in: true, file_path: "crates/rkf-render/shaders/shade_emissive.wgsl".into() },
            ]),
            selected_object_materials: Signal::new(Vec::new()),
            camera_display_pos: Signal::new(Vec3::new(0.0, 2.5, 5.0)),
            scene_name: Signal::new("Untitled".into()),
            scene_path: Signal::new(None),
            brush_type: Signal::new("Add".to_string()),
            paint_mode: Signal::new(crate::paint::PaintMode::Material),
            paint_color: Signal::new(Vec3::ONE),
        }
    }

    /// Set selection and sync dependent state (sliders + tree highlight).
    ///
    /// Replaces the former selection-change and tree-sync Effects.
    /// Call this instead of `selection.set()` directly.
    pub fn set_selection(
        &self,
        sel: Option<SelectedEntity>,
        sliders: &SliderSignals,
        tree_state: &rinch::prelude::UseTreeReturn,
    ) {
        self.selection.set(sel);
        self.on_selection_changed(sliders);
        self.sync_tree_selection(tree_state);
    }

    /// Push object/light data into SliderSignals when selection changes.
    ///
    /// Reads from reactive signals (objects/lights) — always up-to-date
    /// since the engine pushes changes via `run_on_main_thread`.
    pub fn on_selection_changed(
        &self,
        sliders: &SliderSignals,
    ) {
        let sel = self.selection.get();

        enum PushData {
            Object(glam::Vec3, glam::Vec3, glam::Vec3),
            Light(glam::Vec3, f32, f32),
            None,
        }
        let (push, oid, lid) = match sel {
            Some(SelectedEntity::Object(oid)) => {
                let objects = self.objects.get();
                let data = objects
                    .iter()
                    .find(|o| o.id == oid)
                    .map(|o| PushData::Object(o.position, o.rotation_degrees, o.scale))
                    .unwrap_or(PushData::None);
                (data, Some(oid), None)
            }
            Some(SelectedEntity::Light(lid)) => {
                let lights = self.lights.get();
                let data = lights
                    .iter()
                    .find(|l| l.id == lid)
                    .map(|l| PushData::Light(l.position, l.intensity, l.range))
                    .unwrap_or(PushData::None);
                (data, None, Some(lid))
            }
            _ => (PushData::None, None, None),
        };

        sliders.bound_object_id.set(oid);
        sliders.bound_light_id.set(lid);
        match push {
            PushData::Object(pos, rot_deg, scale) => {
                sliders.push_object_values(pos, rot_deg, scale);
            }
            PushData::Light(pos, intensity, range) => {
                sliders.push_light_values(pos, intensity, range);
            }
            PushData::None => {}
        }
    }

    /// Sync tree selection highlight when ui.selection changes.
    ///
    /// Store method — called from event handlers that change selection,
    /// replacing the tree selection-sync Effect.
    pub fn sync_tree_selection(&self, tree_state: &rinch::prelude::UseTreeReturn) {
        let sel = self.selection.get();
        if let Some(sel) = sel {
            let value = match sel {
                SelectedEntity::Object(id) => format!("obj:{id}"),
                SelectedEntity::Light(id) => format!("light:{id}"),
                SelectedEntity::Camera => "camera".to_string(),
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

    /// Send all slider and toggle values as EditorCommands.
    ///
    /// Called from slider onchange callbacks (replacing the batch sync Effect).
    /// Each slider calls this when its value changes, which sends all commands.
    /// This is a store method per the rinch store pattern.
    ///
    /// Prefer the targeted `send_*_commands()` methods when only one UI section
    /// changed — they avoid sending unrelated commands.
    pub fn send_all_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        self.send_camera_commands(cmd);
        self.send_atmosphere_commands(cmd, ui);
        self.send_fog_commands(cmd, ui);
        self.send_cloud_commands(cmd, ui);
        self.send_post_process_commands(cmd, ui);
        self.send_brush_commands(cmd);
        self.send_object_transform_commands(cmd);
        self.send_light_commands(cmd);
    }

    /// Send camera-related commands (FOV, speed, near/far).
    pub fn send_camera_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetCameraFov {
            fov: self.fov.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetCameraSpeed {
            speed: self.fly_speed.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetCameraNearFar {
            near: self.near.get() as f32,
            far: self.far.get() as f32,
        });
    }

    /// Send atmosphere commands (sun direction, intensity, scattering).
    /// Also sends the atmosphere toggle from `ui`.
    pub fn send_atmosphere_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        use crate::editor_command::EditorCommand;
        let az = (self.sun_azimuth.get() as f32).to_radians();
        let el = (self.sun_elevation.get() as f32).to_radians();
        let cos_el = el.cos();
        let sun_dir =
            Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
        let _ = cmd.0.send(EditorCommand::SetAtmosphere {
            sun_direction: sun_dir,
            sun_intensity: self.sun_intensity.get() as f32,
            rayleigh_scale: self.rayleigh_scale.get() as f32,
            mie_scale: self.mie_scale.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::ToggleAtmosphere {
            enabled: ui.atmo_enabled.get(),
        });
    }

    /// Send fog commands (density, height falloff, dust).
    /// Also sends the fog toggle from `ui`.
    pub fn send_fog_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetFog {
            density: self.fog_density.get() as f32,
            height_falloff: self.fog_height_falloff.get() as f32,
            dust_density: self.dust_density.get() as f32,
            dust_asymmetry: self.dust_asymmetry.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::ToggleFog {
            enabled: ui.fog_enabled.get(),
        });
    }

    /// Send cloud commands (coverage, density, altitude, thickness, wind).
    /// Also sends the clouds toggle from `ui`.
    pub fn send_cloud_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetClouds {
            coverage: self.cloud_coverage.get() as f32,
            density: self.cloud_density.get() as f32,
            altitude: self.cloud_altitude.get() as f32,
            thickness: self.cloud_thickness.get() as f32,
            wind_speed: self.cloud_wind_speed.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::ToggleClouds {
            enabled: ui.clouds_enabled.get(),
        });
    }

    /// Send post-processing commands (bloom, exposure, DoF, etc.).
    /// Also sends bloom/DoF toggles and tone map mode from `ui`.
    pub fn send_post_process_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetPostProcess {
            bloom_intensity: self.bloom_intensity.get() as f32,
            bloom_threshold: self.bloom_threshold.get() as f32,
            exposure: self.exposure.get() as f32,
            sharpen: self.sharpen.get() as f32,
            dof_focus_distance: self.dof_focus_dist.get() as f32,
            dof_focus_range: self.dof_focus_range.get() as f32,
            dof_max_coc: self.dof_max_coc.get() as f32,
            motion_blur: self.motion_blur.get() as f32,
            god_rays: self.god_rays.get() as f32,
            vignette: self.vignette.get() as f32,
            grain: self.grain.get() as f32,
            chromatic_aberration: self.chromatic_ab.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::ToggleBloom {
            enabled: ui.bloom_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::ToggleDof {
            enabled: ui.dof_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::SetToneMapMode {
            mode: ui.tone_map_mode.get(),
        });
    }

    /// Send brush/sculpt/paint settings commands.
    pub fn send_brush_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::SetSculptSettings {
            radius: self.brush_radius.get() as f32,
            strength: self.brush_strength.get() as f32,
            falloff: self.brush_falloff.get() as f32,
        });
        let _ = cmd.0.send(EditorCommand::SetPaintSettings {
            radius: self.brush_radius.get() as f32,
            strength: self.brush_strength.get() as f32,
            falloff: self.brush_falloff.get() as f32,
        });
    }

    /// Send object transform commands (position, rotation, scale) for the bound object.
    pub fn send_object_transform_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        if let Some(oid) = self.bound_object_id.get() {
            let _ = cmd.0.send(EditorCommand::SetObjectPosition {
                entity_id: oid,
                position: Vec3::new(
                    self.obj_pos_x.get() as f32,
                    self.obj_pos_y.get() as f32,
                    self.obj_pos_z.get() as f32,
                ),
            });
            let _ = cmd.0.send(EditorCommand::SetObjectRotation {
                entity_id: oid,
                rotation: Vec3::new(
                    self.obj_rot_x.get() as f32,
                    self.obj_rot_y.get() as f32,
                    self.obj_rot_z.get() as f32,
                ),
            });
            let _ = cmd.0.send(EditorCommand::SetObjectScale {
                entity_id: oid,
                scale: Vec3::new(
                    self.obj_scale_x.get() as f32,
                    self.obj_scale_y.get() as f32,
                    self.obj_scale_z.get() as f32,
                ),
            });
        }
    }

    /// Send light property commands (position, intensity, range) for the bound light.
    pub fn send_light_commands(&self, cmd: &crate::CommandSender) {
        use crate::editor_command::EditorCommand;
        if let Some(lid) = self.bound_light_id.get() {
            let _ = cmd.0.send(EditorCommand::SetLightPosition {
                light_id: lid,
                position: Vec3::new(
                    self.light_pos_x.get() as f32,
                    self.light_pos_y.get() as f32,
                    self.light_pos_z.get() as f32,
                ),
            });
            let _ = cmd.0.send(EditorCommand::SetLightIntensity {
                light_id: lid,
                intensity: self.light_intensity.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetLightRange {
                light_id: lid,
                range: self.light_range.get() as f32,
            });
        }
    }

    /// Send toggle/state commands (atmosphere, fog, clouds, bloom, DoF, tone map mode).
    pub fn send_toggle_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        use crate::editor_command::EditorCommand;
        let _ = cmd.0.send(EditorCommand::ToggleAtmosphere {
            enabled: ui.atmo_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::ToggleFog {
            enabled: ui.fog_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::ToggleClouds {
            enabled: ui.clouds_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::ToggleBloom {
            enabled: ui.bloom_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::ToggleDof {
            enabled: ui.dof_enabled.get(),
        });
        let _ = cmd.0.send(EditorCommand::SetToneMapMode {
            mode: ui.tone_map_mode.get(),
        });
    }

    /// Subscribe the current reactive scope to every signal.
    /// Call inside an `Effect` to re-run whenever any slider changes.
    #[allow(dead_code)]
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

/// Accumulates geometry-first snapshots during a sculpt stroke for undo support.
///
/// Created when a sculpt stroke begins. Before each edit, any not-yet-captured
/// geometry slots are snapshot from the geometry and SDF cache pools. On stroke
/// end, the accumulated snapshots are pushed as an undo action.
pub struct SculptUndoAccumulator {
    /// The object being sculpted.
    pub object_id: u64,
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
    /// Set by titlebar drag handler, consumed by the event loop.
    pub pending_drag: bool,
    /// Set by minimize window control, consumed by the event loop.
    pub pending_minimize: bool,
    /// Set by maximize window control, consumed by the event loop.
    pub pending_maximize: bool,
    /// Set by "Convert to Voxel Object" button. Contains the object ID
    /// of an analytical primitive to convert to geometry-first voxelized form.
    pub pending_convert_to_voxel: Option<u32>,
    /// Set by material drag-and-drop in object properties panel.
    /// Contains (object_id, from_material, to_material).
    pub pending_remap_material: Option<(u64, u16, u16)>,
    /// Set by material assignment on analytical primitives.
    /// Contains (object_id, material_id).
    pub pending_set_primitive_material: Option<(u64, u16)>,

    // ── Sculpt pipeline (UI → render loop) ──────────────────
    /// Queued sculpt edit requests — one per brush-hit point during a stroke.
    /// Drained by the render loop each frame and applied to the CPU brick pool.
    pub pending_sculpt_edits: Vec<crate::sculpt::SculptEditRequest>,

    /// Accumulates brick snapshots during a sculpt stroke for undo.
    /// Created on stroke begin, finalized on stroke end.
    pub sculpt_undo_accumulator: Option<SculptUndoAccumulator>,

    /// Pending sculpt undo: (object_id, geometry_snapshots) to restore.
    /// Set by apply_undo_action, consumed by the render loop.
    pub pending_sculpt_undo: Option<(u64, Vec<GeometryUndoEntry>)>,

    // ── Paint pipeline (UI → render loop) ────────────────────
    /// Queued paint edit requests — one per brush-hit point during a stroke.
    /// Drained by the render loop each frame and applied to surface voxels.
    pub pending_paint_edits: Vec<crate::paint::PaintEditRequest>,

    /// Accumulates geometry-first snapshots during a paint stroke for undo.
    /// Created on stroke begin, finalized on stroke end.
    pub paint_undo_accumulator: Option<SculptUndoAccumulator>,

    /// Pending paint undo: (object_id, geometry_snapshots) to restore.
    /// Set by apply_undo_action, consumed by the render loop.
    pub pending_paint_undo: Option<(u64, Vec<GeometryUndoEntry>)>,

    // ── World (unified game state) ──────────────────────────
    /// The unified world container. Wraps `Scene` (SDF objects) + ECS +
    /// brick pool. Replaces the former `v2_scene: Option<Scene>`.
    pub world: World,
}
