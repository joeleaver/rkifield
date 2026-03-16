//! Centralized reactive signals for all persistent slider properties.
//!
//! Each signal holds the actual slider value (f64). The UI creates sliders
//! bound to these signals via `build_synced_slider`, and a single batch
//! `Effect` in `RightPanel` syncs all values to `EditorState` in one lock
//! per frame — eliminating per-slider lock closures.

use glam::Vec3;
use rinch::prelude::Signal;

use super::EditorState;
use super::UiSignals;

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

    // Object transform and light properties are NO LONGER here.
    // They read from UiSignals via Memo (single source of truth).
    // See TransformEditor and LightProperties components.
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
        // Object transform and light properties are handled by their
        // respective UI components via EditorCommands (single source of truth).
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

        // Object transform and light properties are handled by their
        // respective UI components via EditorCommands (single source of truth).
    }

}
