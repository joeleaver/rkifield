//! Centralized reactive signals for all persistent slider properties.
//!
//! Each signal holds the actual slider value (f64). The UI creates sliders
//! bound to these signals via `SliderRow` / `SyncedSliderRow` components,
//! and targeted `send_*_commands()` methods push changes to the engine.

use glam::Vec3;
use rinch::prelude::Signal;

use super::EditorState;
use super::UiSignals;

/// Centralized reactive signals for all persistent slider properties.
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
    pub vol_ambient_intensity: Signal<f64>,
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
    pub gi_intensity: Signal<f64>,
    // Brush
    pub brush_radius: Signal<f64>,
    pub brush_strength: Signal<f64>,
    pub brush_falloff: Signal<f64>,
}

/// Send a single `SetComponentField` command for the scene environment entity.
fn send_env_field(
    cmd: &crate::CommandSender,
    ui: &UiSignals,
    field: &str,
    value: rkf_runtime::behavior::game_value::GameValue,
) {
    use crate::editor_command::EditorCommand;
    if let Some(uuid) = ui.active_camera_uuid.get() {
        let _ = cmd.0.send(EditorCommand::SetComponentField {
            entity_id: uuid,
            component_name: "EnvironmentSettings".to_string(),
            field_name: field.to_string(),
            value,
        });
    }
}

/// Shorthand for sending a float field to the scene environment.
fn send_env_float(cmd: &crate::CommandSender, ui: &UiSignals, field: &str, v: f64) {
    send_env_field(cmd, ui, field, rkf_runtime::behavior::game_value::GameValue::Float(v));
}

/// Shorthand for sending a bool field to the scene environment.
fn send_env_bool(cmd: &crate::CommandSender, ui: &UiSignals, field: &str, v: bool) {
    send_env_field(cmd, ui, field, rkf_runtime::behavior::game_value::GameValue::Bool(v));
}

/// Shorthand for sending an int field to the scene environment.
fn send_env_int(cmd: &crate::CommandSender, ui: &UiSignals, field: &str, v: i64) {
    send_env_field(cmd, ui, field, rkf_runtime::behavior::game_value::GameValue::Int(v));
}

/// Shorthand for sending a Vec3 field to the scene environment.
fn send_env_vec3(cmd: &crate::CommandSender, ui: &UiSignals, field: &str, v: Vec3) {
    send_env_field(cmd, ui, field, rkf_runtime::behavior::game_value::GameValue::Vec3(v));
}

/// Send a color (Vec3) environment field. Public for use by color picker callbacks.
pub fn send_env_color(cmd: &crate::CommandSender, ui: &UiSignals, field: &str, v: Vec3) {
    send_env_vec3(cmd, ui, field, v);
}

impl SliderSignals {
    /// Create slider signals initialized from the current `EditorState`.
    /// Must be called on the main thread (signals use thread-local reactive state).
    ///
    /// Environment sliders init from `EnvironmentSettings::default()` — the
    /// engine thread will push actual values from the ECS singleton on first frame.
    pub fn new(es: &EditorState) -> Self {
        let env = rkf_runtime::environment::EnvironmentSettings::default();
        let d = &env.atmosphere.sun_direction;
        let pp = &env.post_process;
        let az = d[0].atan2(d[2]).to_degrees().rem_euclid(360.0);
        let el = d[1].asin().to_degrees();
        Self {
            fov: Signal::new(es.editor_camera_fov_degrees() as f64),
            fly_speed: Signal::new(es.camera_control.fly_speed as f64),
            near: Signal::new(es.editor_camera_near() as f64),
            far: Signal::new(es.editor_camera_far() as f64),
            sun_azimuth: Signal::new(az as f64),
            sun_elevation: Signal::new(el as f64),
            sun_intensity: Signal::new(env.atmosphere.sun_intensity as f64),
            rayleigh_scale: Signal::new(env.atmosphere.rayleigh_scale as f64),
            mie_scale: Signal::new(env.atmosphere.mie_scale as f64),
            fog_density: Signal::new(env.fog.density as f64),
            fog_height_falloff: Signal::new(env.fog.height_falloff as f64),
            dust_density: Signal::new(env.fog.ambient_dust_density as f64),
            dust_asymmetry: Signal::new(env.fog.dust_asymmetry as f64),
            vol_ambient_intensity: Signal::new(env.fog.vol_ambient_intensity as f64),
            cloud_coverage: Signal::new(env.clouds.coverage as f64),
            cloud_density: Signal::new(env.clouds.density as f64),
            cloud_altitude: Signal::new(env.clouds.altitude as f64),
            cloud_thickness: Signal::new(env.clouds.thickness as f64),
            cloud_wind_speed: Signal::new(env.clouds.wind_speed as f64),
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
            gi_intensity: Signal::new(pp.gi_intensity as f64),
            brush_radius: Signal::new(es.sculpt.current_settings.radius as f64),
            brush_strength: Signal::new(es.sculpt.current_settings.strength as f64),
            brush_falloff: Signal::new(es.sculpt.current_settings.falloff as f64),
        }
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

    /// Send atmosphere fields to the scene environment ECS entity.
    /// Send sun direction computed from azimuth/elevation sliders.
    ///
    /// Only sends the derived sun_direction Vec3 — all other atmosphere
    /// fields are now handled by store-bound widgets.
    pub fn send_atmosphere_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        let az = (self.sun_azimuth.get() as f32).to_radians();
        let el = (self.sun_elevation.get() as f32).to_radians();
        let cos_el = el.cos();
        let sun_dir = Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
        send_env_vec3(cmd, ui, "atmosphere.sun_direction", sun_dir);
    }

    // send_fog_commands — removed (migrated to store-bound widgets)
    // send_cloud_commands — removed (migrated to store-bound widgets)
    // send_post_process_commands — removed (migrated to store-bound widgets)

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

    /// Send all non-store slider values as EditorCommands.
    ///
    /// Environment fields (fog, clouds, post-process) are now handled by
    /// store-bound widgets. This only sends camera, sun direction, and brush.
    pub fn send_all_commands(&self, cmd: &crate::CommandSender, ui: &UiSignals) {
        self.send_camera_commands(cmd);
        self.send_atmosphere_commands(cmd, ui);
        self.send_brush_commands(cmd);
    }

    /// Write camera + brush slider values back to `EditorState`.
    ///
    /// Environment values are no longer synced here — they flow through the
    /// ECS `EnvironmentSettings` component via `SetComponentField` commands.
    pub fn sync_to_state(&self, es: &mut EditorState) {
        // Camera
        es.set_editor_camera_component_field(|c| c.fov_degrees = self.fov.get() as f32);
        es.camera_control.fly_speed = self.fly_speed.get() as f32;
        es.set_editor_camera_component_field(|c| { c.near = self.near.get() as f32; c.far = self.far.get() as f32; });

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
    }
}
