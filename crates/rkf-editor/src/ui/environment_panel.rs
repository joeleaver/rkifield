//! Environment panel — atmosphere, fog, clouds, post-processing.
//!
//! Each section is a separate `#[component]` for fine-grained reactivity:
//! only the section whose toggle changes will rebuild.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SliderSignals, UiSignals};
use crate::CommandSender;

use super::components::ToggleRow;
use super::slider_helpers::SliderRow;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE};

fn vec3_to_hex(c: glam::Vec3) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        (c.x.clamp(0.0, 1.0) * 255.0) as u8,
        (c.y.clamp(0.0, 1.0) * 255.0) as u8,
        (c.z.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.strip_prefix('#')?;
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some((r, g, b))
    } else {
        None
    }
}

const COLOR_ROW_STYLE: &str = "display:flex;align-items:center;gap:6px;padding:1px 6px;font-size:10px;color:var(--rinch-color-text);";

// ── Color picker sub-components ─────────────────────────────────────────────

/// Build a color picker row using manual DOM construction (avoids rsx! type recursion).
fn env_color_row(
    scope: &mut RenderScope,
    label_text: &str,
    color_signal: Signal<glam::Vec3>,
    field_path: &'static str,
    cmd: &CommandSender,
    ui: &UiSignals,
) -> NodeHandle {
    let row = scope.create_element("div");
    row.set_attribute("style", COLOR_ROW_STYLE);

    let label = scope.create_element("span");
    label.set_text(label_text);
    row.append_child(&label);

    let picker = ColorPicker {
        format: "hex".into(),
        value: vec3_to_hex(color_signal.get()),
        alpha: false,
        with_input: false,
        size: "xs".into(),
        onchange: Some(InputCallback::new({
            let cmd = cmd.clone();
            let ui = *ui;
            move |hex: String| {
                if let Some((r, g, b)) = parse_hex_color(&hex) {
                    let v = glam::Vec3::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
                    color_signal.set(v);
                    crate::editor_state::send_env_color(&cmd, &ui, field_path, v);
                }
            }
        })),
        ..Default::default()
    };
    let picker_node = rinch::core::untracked(|| picker.render(scope, &[]));
    row.append_child(&picker_node);

    row
}

#[component]
pub fn SunColorPicker() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    env_color_row(__scope, "Sun Color", ui.sun_color, "atmosphere.sun_color", &cmd, &ui)
}

#[component]
pub fn FogColorPicker() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    env_color_row(__scope, "Fog Color", ui.fog_color, "fog.color", &cmd, &ui)
}

#[component]
pub fn VolAmbientColorPicker() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    env_color_row(__scope, "Vol Ambient", ui.vol_ambient_color, "fog.vol_ambient_color", &cmd, &ui)
}

// ── Atmosphere section ──────────────────────────────────────────────────────

/// Sun direction sliders + atmosphere toggle + Rayleigh/Mie scale.
#[component]
pub fn AtmosphereSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Atmosphere" }
            SliderRow {
                label: "Sun Azimuth",
                suffix: "\u{00b0}",
                signal: Some(sliders.sun_azimuth),
                min: 0.0, max: 360.0, step: 1.0, decimals: 0,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Sun Elevation",
                suffix: "\u{00b0}",
                signal: Some(sliders.sun_elevation),
                min: Some(-90.0), max: 90.0, step: 1.0, decimals: 0,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Sun Intensity",
                suffix: "",
                signal: Some(sliders.sun_intensity),
                min: 0.0, max: 10.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
            SunColorPicker {}
            ToggleRow {
                label: "Atmosphere",
                enabled: Some(ui.atmo_enabled),
                on_change: { let cmd = cmd.clone(); move || { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Rayleigh Scale",
                suffix: "",
                signal: Some(sliders.rayleigh_scale),
                min: 0.0, max: 5.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Mie Scale",
                suffix: "",
                signal: Some(sliders.mie_scale),
                min: 0.0, max: 5.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_atmosphere_commands(&cmd, &ui); } },
            }
        }
    }
}

// ── Fog section ─────────────────────────────────────────────────────────────

/// Fog toggle + density, height falloff, dust density, dust asymmetry sliders.
#[component]
pub fn FogSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Fog" }
            ToggleRow {
                label: "Fog",
                enabled: Some(ui.fog_enabled),
                on_change: { let cmd = cmd.clone(); move || { sliders.send_fog_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Fog Density",
                suffix: "",
                signal: Some(sliders.fog_density),
                min: 0.0, max: 0.5, step: 0.001, decimals: 3,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_fog_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Height Falloff",
                suffix: "",
                signal: Some(sliders.fog_height_falloff),
                min: 0.0, max: 1.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_fog_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Dust Density",
                suffix: "",
                signal: Some(sliders.dust_density),
                min: 0.0, max: 0.1, step: 0.001, decimals: 3,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_fog_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Dust Asymmetry",
                suffix: "",
                signal: Some(sliders.dust_asymmetry),
                min: 0.0, max: 0.95, step: 0.05, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_fog_commands(&cmd, &ui); } },
            }
            FogColorPicker {}
            VolAmbientColorPicker {}
        }
    }
}

// ── Clouds section ──────────────────────────────────────────────────────────

/// Cloud toggle + coverage, density, altitude, thickness, wind speed sliders.
#[component]
pub fn CloudsSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Clouds" }
            ToggleRow {
                label: "Clouds",
                enabled: Some(ui.clouds_enabled),
                on_change: { let cmd = cmd.clone(); move || { sliders.send_cloud_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Coverage",
                suffix: "",
                signal: Some(sliders.cloud_coverage),
                min: 0.0, max: 1.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_cloud_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Cloud Density",
                suffix: "",
                signal: Some(sliders.cloud_density),
                min: 0.0, max: 5.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_cloud_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Altitude",
                suffix: "m",
                signal: Some(sliders.cloud_altitude),
                min: 0.0, max: 5000.0, step: 50.0, decimals: 0,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_cloud_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Thickness",
                suffix: "m",
                signal: Some(sliders.cloud_thickness),
                min: 10.0, max: 10000.0, step: 50.0, decimals: 0,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_cloud_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Wind Speed",
                suffix: "",
                signal: Some(sliders.cloud_wind_speed),
                min: 0.0, max: 50.0, step: 0.5, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_cloud_commands(&cmd, &ui); } },
            }
        }
    }
}

// ── Post-processing section ─────────────────────────────────────────────────

/// Bloom, exposure, tone mapping, sharpen, DoF, motion blur, god rays,
/// vignette, grain, chromatic aberration.
#[component]
pub fn PostProcessSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let tm_signal = ui.tone_map_mode;

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Post-Processing" }
            ToggleRow {
                label: "Bloom",
                enabled: Some(ui.bloom_enabled),
                on_change: { let cmd = cmd.clone(); move || { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Bloom Intensity",
                suffix: "",
                signal: Some(sliders.bloom_intensity),
                min: 0.0, max: 2.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Bloom Threshold",
                suffix: "",
                signal: Some(sliders.bloom_threshold),
                min: 0.0, max: 5.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Exposure",
                suffix: "",
                signal: Some(sliders.exposure),
                min: 0.1, max: 10.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            div {
                style: "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                        cursor:pointer;user-select:none;",
                onclick: {
                    let cmd = cmd.clone();
                    move || {
                        tm_signal.update(|v| *v = if *v == 0 { 1 } else { 0 });
                        sliders.send_post_process_commands(&cmd, &ui);
                    }
                },
                {|| if tm_signal.get() == 0 { "Tone Map: ACES".to_string() } else { "Tone Map: AgX".to_string() }}
            }
            SliderRow {
                label: "Sharpen",
                suffix: "",
                signal: Some(sliders.sharpen),
                min: 0.0, max: 2.0, step: 0.05, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            ToggleRow {
                label: "DoF",
                enabled: Some(ui.dof_enabled),
                on_change: { let cmd = cmd.clone(); move || { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Focus Distance",
                suffix: "",
                signal: Some(sliders.dof_focus_dist),
                min: 0.1, max: 50.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Focus Range",
                suffix: "",
                signal: Some(sliders.dof_focus_range),
                min: 0.1, max: 20.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Max CoC",
                suffix: "px",
                signal: Some(sliders.dof_max_coc),
                min: 1.0, max: 32.0, step: 1.0, decimals: 0,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Motion Blur",
                suffix: "",
                signal: Some(sliders.motion_blur),
                min: 0.0, max: 3.0, step: 0.1, decimals: 1,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "God Rays",
                suffix: "",
                signal: Some(sliders.god_rays),
                min: 0.0, max: 2.0, step: 0.05, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Vignette",
                suffix: "",
                signal: Some(sliders.vignette),
                min: 0.0, max: 1.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Grain",
                suffix: "",
                signal: Some(sliders.grain),
                min: 0.0, max: 1.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
            SliderRow {
                label: "Chromatic Ab.",
                suffix: "",
                signal: Some(sliders.chromatic_ab),
                min: 0.0, max: 1.0, step: 0.01, decimals: 2,
                on_change: { let cmd = cmd.clone(); move |_v: f64| { sliders.send_post_process_commands(&cmd, &ui); } },
            }
        }
    }
}


