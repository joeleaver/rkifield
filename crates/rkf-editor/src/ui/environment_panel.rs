//! Environment panel — atmosphere, fog, clouds, post-processing.
//!
//! Each section is a separate `#[component]` for fine-grained reactivity:
//! only the section whose toggle changes will rebuild.
//!
//! All environment fields use store-bound widgets (BoundSlider, BoundToggle,
//! BoundColor) that read from and write to the UI Store. The only exception
//! is Sun Azimuth/Elevation which are derived fields that compute a Vec3
//! sun_direction — they use manual SliderRows until a DirectionInput widget
//! is built.

use rinch::prelude::*;

use crate::editor_state::SliderSignals;
use crate::CommandSender;

use super::slider_helpers::SliderRow;
use super::LABEL_STYLE;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::bound::bound_color::BoundColor;
use crate::ui::bound::bound_slider::BoundSlider;
use crate::ui::bound::bound_toggle::BoundToggle;

// ── Atmosphere section ──────────────────────────────────────────────────────

/// Sun direction sliders + atmosphere toggle + Rayleigh/Mie scale.
#[component]
pub fn AtmosphereSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();

    // Bound widgets rendered outside rsx! to avoid type-inference conflicts
    // between SliderRow (Option<f64> min) and BoundSlider (f64 min).
    let sun_intensity = BoundSlider {
        path: String::from("env/atmosphere.sun_intensity"),
        label: String::from("Sun Intensity"),
        min: 0.0, max: 10.0, step: 0.1, decimals: 1,
        suffix: String::new(),
    }.render(__scope, &[]);

    let sun_color = BoundColor {
        path: String::from("env/atmosphere.sun_color"),
        label: String::from("Sun Color"),
    }.render(__scope, &[]);

    let atmo_toggle = BoundToggle {
        path: String::from("env/atmosphere.enabled"),
        label: String::from("Atmosphere"),
    }.render(__scope, &[]);

    let rayleigh = BoundSlider {
        path: String::from("env/atmosphere.rayleigh_scale"),
        label: String::from("Rayleigh Scale"),
        min: 0.0, max: 5.0, step: 0.1, decimals: 1,
        suffix: String::new(),
    }.render(__scope, &[]);

    let mie = BoundSlider {
        path: String::from("env/atmosphere.mie_scale"),
        label: String::from("Mie Scale"),
        min: 0.0, max: 5.0, step: 0.1, decimals: 1,
        suffix: String::new(),
    }.render(__scope, &[]);

    // Sun Azimuth/Elevation are derived (compute Vec3 sun_direction).
    // They use manual SliderRows until a DirectionInput widget is built.
    // ui param required by send_atmosphere_commands signature but only
    // sun_direction is actually sent.
    let ui = use_context::<crate::editor_state::UiSignals>();

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
            {sun_intensity}
            {sun_color}
            {atmo_toggle}
            {rayleigh}
            {mie}
        }
    }
}

// ── Fog section ─────────────────────────────────────────────────────────────

/// Fog toggle + density, height falloff, dust density, dust asymmetry sliders.
#[component]
pub fn FogSection() -> NodeHandle {
    let fog_toggle = BoundToggle { path: "env/fog.enabled".into(), label: "Fog".into() }.render(__scope, &[]);
    let fog_density = BoundSlider { path: "env/fog.density".into(), label: "Fog Density".into(), min: 0.0, max: 0.5, step: 0.001, decimals: 3, suffix: String::new() }.render(__scope, &[]);
    let height_falloff = BoundSlider { path: "env/fog.height_falloff".into(), label: "Height Falloff".into(), min: 0.0, max: 1.0, step: 0.01, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let dust_density = BoundSlider { path: "env/fog.ambient_dust_density".into(), label: "Dust Density".into(), min: 0.0, max: 0.1, step: 0.001, decimals: 3, suffix: String::new() }.render(__scope, &[]);
    let dust_asymmetry = BoundSlider { path: "env/fog.dust_asymmetry".into(), label: "Dust Asymmetry".into(), min: 0.0, max: 0.95, step: 0.05, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let fog_color = BoundColor { path: "env/fog.color".into(), label: "Fog Color".into() }.render(__scope, &[]);
    let vol_ambient_color = BoundColor { path: "env/fog.vol_ambient_color".into(), label: "Vol Ambient".into() }.render(__scope, &[]);
    let vol_ambient_intensity = BoundSlider { path: "env/fog.vol_ambient_intensity".into(), label: "Ambient Intensity".into(), min: 0.0, max: 5.0, step: 0.05, decimals: 2, suffix: String::new() }.render(__scope, &[]);

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Fog" }
            {fog_toggle}
            {fog_density}
            {height_falloff}
            {dust_density}
            {dust_asymmetry}
            {fog_color}
            {vol_ambient_color}
            {vol_ambient_intensity}
        }
    }
}

// ── Clouds section ──────────────────────────────────────────────────────────

/// Cloud toggle + coverage, density, altitude, thickness, wind speed sliders.
#[component]
pub fn CloudsSection() -> NodeHandle {
    let clouds_toggle = BoundToggle { path: "env/clouds.enabled".into(), label: "Clouds".into() }.render(__scope, &[]);
    let coverage = BoundSlider { path: "env/clouds.coverage".into(), label: "Coverage".into(), min: 0.0, max: 1.0, step: 0.01, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let density = BoundSlider { path: "env/clouds.density".into(), label: "Cloud Density".into(), min: 0.0, max: 5.0, step: 0.1, decimals: 1, suffix: String::new() }.render(__scope, &[]);
    let altitude = BoundSlider { path: "env/clouds.altitude".into(), label: "Altitude".into(), min: 500.0, max: 5000.0, step: 100.0, decimals: 0, suffix: "m".into() }.render(__scope, &[]);
    let thickness = BoundSlider { path: "env/clouds.thickness".into(), label: "Thickness".into(), min: 100.0, max: 3000.0, step: 50.0, decimals: 0, suffix: "m".into() }.render(__scope, &[]);
    let wind_speed = BoundSlider { path: "env/clouds.wind_speed".into(), label: "Wind Speed".into(), min: 0.0, max: 50.0, step: 0.5, decimals: 1, suffix: String::new() }.render(__scope, &[]);

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Clouds" }
            {clouds_toggle}
            {coverage}
            {density}
            {altitude}
            {thickness}
            {wind_speed}
        }
    }
}

// ── Post-processing section ─────────────────────────────────────────────────

/// Post-processing: GI, bloom, exposure, tone mapping, sharpen.
#[component]
pub fn PostProcessSection() -> NodeHandle {
    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Post-Processing" }
            PostProcessLightingGroup {}
            PostProcessEffectsGroup {}
        }
    }
}

/// GI, bloom, exposure, tone mapping, sharpen.
#[component]
fn PostProcessLightingGroup() -> NodeHandle {
    let store = use_context::<UiStore>();

    // Tone map mode: read as Int from store, toggle on click, write back as Int.
    let tm_signal = store.read("env/post_process.tone_map_mode");

    let store_for_tm = store.clone();
    let tone_map_row = rsx! {
        div {
            style: "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                    cursor:pointer;user-select:none;",
            onclick: {
                move || {
                    let cur = tm_signal.get().as_int().unwrap_or(0);
                    let next = if cur == 0 { 1 } else { 0 };
                    store_for_tm.set("env/post_process.tone_map_mode", UiValue::Int(next));
                }
            },
            {move || if tm_signal.get().as_int().unwrap_or(0) == 0 { "Tone Map: ACES".to_string() } else { "Tone Map: AgX".to_string() }}
        }
    };

    let gi = BoundSlider {
        path: "env/post_process.gi_intensity".into(), label: "GI Intensity".into(),
        min: 0.0, max: 3.0, step: 0.05, decimals: 2, suffix: String::new(),
    }.render(__scope, &[]);

    let bloom_toggle = BoundToggle {
        path: "env/post_process.bloom_enabled".into(), label: "Bloom".into(),
    }.render(__scope, &[]);

    let bloom_intensity = BoundSlider {
        path: "env/post_process.bloom_intensity".into(), label: "Bloom Intensity".into(),
        min: 0.0, max: 2.0, step: 0.01, decimals: 2, suffix: String::new(),
    }.render(__scope, &[]);

    let bloom_threshold = BoundSlider {
        path: "env/post_process.bloom_threshold".into(), label: "Bloom Threshold".into(),
        min: 0.0, max: 5.0, step: 0.1, decimals: 1, suffix: String::new(),
    }.render(__scope, &[]);

    let exposure = BoundSlider {
        path: "env/post_process.exposure".into(), label: "Exposure".into(),
        min: 0.1, max: 10.0, step: 0.1, decimals: 1, suffix: String::new(),
    }.render(__scope, &[]);

    let sharpen = BoundSlider {
        path: "env/post_process.sharpen_strength".into(), label: "Sharpen".into(),
        min: 0.0, max: 2.0, step: 0.05, decimals: 2, suffix: String::new(),
    }.render(__scope, &[]);

    rsx! {
        div {
            {gi}
            {bloom_toggle}
            {bloom_intensity}
            {bloom_threshold}
            {exposure}
            {tone_map_row}
            {sharpen}
        }
    }
}

/// DoF, motion blur, god rays, vignette, grain, chromatic aberration.
#[component]
fn PostProcessEffectsGroup() -> NodeHandle {
    let dof_toggle = BoundToggle { path: "env/post_process.dof_enabled".into(), label: "DoF".into() }.render(__scope, &[]);
    let focus_dist = BoundSlider { path: "env/post_process.dof_focus_distance".into(), label: "Focus Distance".into(), min: 0.1, max: 50.0, step: 0.1, decimals: 1, suffix: String::new() }.render(__scope, &[]);
    let focus_range = BoundSlider { path: "env/post_process.dof_focus_range".into(), label: "Focus Range".into(), min: 0.1, max: 20.0, step: 0.1, decimals: 1, suffix: String::new() }.render(__scope, &[]);
    let max_coc = BoundSlider { path: "env/post_process.dof_max_coc".into(), label: "Max CoC".into(), min: 1.0, max: 32.0, step: 1.0, decimals: 0, suffix: "px".into() }.render(__scope, &[]);
    let motion_blur = BoundSlider { path: "env/post_process.motion_blur_intensity".into(), label: "Motion Blur".into(), min: 0.0, max: 3.0, step: 0.1, decimals: 1, suffix: String::new() }.render(__scope, &[]);
    let god_rays = BoundSlider { path: "env/post_process.god_rays_intensity".into(), label: "God Rays".into(), min: 0.0, max: 2.0, step: 0.05, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let vignette = BoundSlider { path: "env/post_process.vignette_intensity".into(), label: "Vignette".into(), min: 0.0, max: 1.0, step: 0.01, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let grain = BoundSlider { path: "env/post_process.grain_intensity".into(), label: "Grain".into(), min: 0.0, max: 1.0, step: 0.01, decimals: 2, suffix: String::new() }.render(__scope, &[]);
    let chromatic_ab = BoundSlider { path: "env/post_process.chromatic_aberration".into(), label: "Chromatic Ab.".into(), min: 0.0, max: 1.0, step: 0.01, decimals: 2, suffix: String::new() }.render(__scope, &[]);

    rsx! {
        div {
            {dof_toggle}
            {focus_dist}
            {focus_range}
            {max_coc}
            {motion_blur}
            {god_rays}
            {vignette}
            {grain}
            {chromatic_ab}
        }
    }
}
