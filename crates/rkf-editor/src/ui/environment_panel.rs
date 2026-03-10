//! Environment panel — atmosphere, fog, clouds, post-processing, animation.
//!
//! Each section is a separate `#[component]` for fine-grained reactivity:
//! only the section whose toggle changes will rebuild.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SliderSignals, UiSignals};
use crate::CommandSender;

use super::components::ToggleRow;
use super::slider_helpers::build_slider_row;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE};

// ── Atmosphere section ──────────────────────────────────────────────────────

/// Sun direction sliders + atmosphere toggle + Rayleigh/Mie scale.
#[component]
pub fn AtmosphereSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let pre_toggle = __scope.create_element("div");
    build_slider_row(
        __scope, &pre_toggle, "Sun Azimuth", "\u{00b0}",
        sliders.sun_azimuth, 0.0, 360.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &pre_toggle, "Sun Elevation", "\u{00b0}",
        sliders.sun_elevation, -90.0, 90.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &pre_toggle, "Sun Intensity", "",
        sliders.sun_intensity, 0.0, 10.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );

    let atmo_toggle = ToggleRow {
        label: "Atmosphere".to_string(),
        enabled: Some(ui.atmo_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_atmosphere_commands(&cmd, &ui); }
        })),
    }
    .render(__scope, &[]);

    let post_toggle = __scope.create_element("div");
    build_slider_row(
        __scope, &post_toggle, "Rayleigh Scale", "",
        sliders.rayleigh_scale, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &post_toggle, "Mie Scale", "",
        sliders.mie_scale, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Atmosphere" }
            {pre_toggle}
            {atmo_toggle}
            {post_toggle}
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

    let fog_toggle = ToggleRow {
        label: "Fog".to_string(),
        enabled: Some(ui.fog_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_fog_commands(&cmd, &ui); }
        })),
    }
    .render(__scope, &[]);

    let sliders_div = __scope.create_element("div");
    build_slider_row(
        __scope, &sliders_div, "Fog Density", "",
        sliders.fog_density, 0.0, 0.5, 0.001, 3,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Height Falloff", "",
        sliders.fog_height_falloff, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Dust Density", "",
        sliders.dust_density, 0.0, 0.1, 0.001, 3,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Dust Asymmetry", "",
        sliders.dust_asymmetry, 0.0, 0.95, 0.05, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Fog" }
            {fog_toggle}
            {sliders_div}
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

    let cloud_toggle = ToggleRow {
        label: "Clouds".to_string(),
        enabled: Some(ui.clouds_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_cloud_commands(&cmd, &ui); }
        })),
    }
    .render(__scope, &[]);

    let sliders_div = __scope.create_element("div");
    build_slider_row(
        __scope, &sliders_div, "Coverage", "",
        sliders.cloud_coverage, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Cloud Density", "",
        sliders.cloud_density, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Altitude", "m",
        sliders.cloud_altitude, 0.0, 5000.0, 50.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Thickness", "m",
        sliders.cloud_thickness, 10.0, 10000.0, 50.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &sliders_div, "Wind Speed", "",
        sliders.cloud_wind_speed, 0.0, 50.0, 0.5, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Clouds" }
            {cloud_toggle}
            {sliders_div}
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

    // Bloom toggle (rendered imperatively — ToggleRow props don't work inline in rsx!).
    let bloom_toggle = ToggleRow {
        label: "Bloom".to_string(),
        enabled: Some(ui.bloom_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_post_process_commands(&cmd, &ui); }
        })),
    }
    .render(__scope, &[]);

    // Bloom sliders.
    let bloom_sliders = __scope.create_element("div");
    build_slider_row(
        __scope, &bloom_sliders, "Bloom Intensity", "",
        sliders.bloom_intensity, 0.0, 2.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &bloom_sliders, "Bloom Threshold", "",
        sliders.bloom_threshold, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &bloom_sliders, "Exposure", "",
        sliders.exposure, 0.1, 10.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );

    // Sharpen slider.
    let sharpen_slider = __scope.create_element("div");
    build_slider_row(
        __scope, &sharpen_slider, "Sharpen", "",
        sliders.sharpen, 0.0, 2.0, 0.05, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );

    // DoF toggle.
    let dof_toggle = ToggleRow {
        label: "DoF".to_string(),
        enabled: Some(ui.dof_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_post_process_commands(&cmd, &ui); }
        })),
    }
    .render(__scope, &[]);

    // DoF + remaining sliders.
    let dof_sliders = __scope.create_element("div");
    build_slider_row(
        __scope, &dof_sliders, "Focus Distance", "",
        sliders.dof_focus_dist, 0.1, 50.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Focus Range", "",
        sliders.dof_focus_range, 0.1, 20.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Max CoC", "px",
        sliders.dof_max_coc, 1.0, 32.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Motion Blur", "",
        sliders.motion_blur, 0.0, 3.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "God Rays", "",
        sliders.god_rays, 0.0, 2.0, 0.05, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Vignette", "",
        sliders.vignette, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Grain", "",
        sliders.grain, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &dof_sliders, "Chromatic Ab.", "",
        sliders.chromatic_ab, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );

    // Tone map mode toggle (ACES / AgX).
    let tm_mode = ui.tone_map_mode.get();
    let tm_signal = ui.tone_map_mode;
    let tm_label = if tm_mode == 0 { "Tone Map: ACES" } else { "Tone Map: AgX" };

    rsx! {
        div {
            div { style: {LABEL_STYLE}, "Post-Processing" }
            {bloom_toggle}
            {bloom_sliders}
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
                {tm_label}
            }
            {sharpen_slider}
            {dof_toggle}
            {dof_sliders}
        }
    }
}

// ── Animation section ───────────────────────────────────────────────────────

/// Play/pause/stop buttons + animation speed slider.
#[component]
pub fn AnimationSection() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let anim_state_signal = ui.animation_state;
    let cur_state = ui.animation_state.get();
    let anim_speed_signal: Signal<f64> = Signal::new(1.0);

    let speed_slider = __scope.create_element("div");
    build_slider_row(
        __scope,
        &speed_slider,
        "Speed",
        "x",
        anim_speed_signal,
        0.0,
        4.0,
        0.1,
        1,
        {
            let cmd = cmd.clone();
            move |v| {
                let _ = cmd.0.send(EditorCommand::SetAnimationSpeed { speed: v as f32 });
            }
        },
    );

    let btn_style = |active: bool| {
        let bg = if active {
            "var(--rinch-primary-color)"
        } else {
            "var(--rinch-color-dark-7)"
        };
        format!(
            "padding:2px 8px;border-radius:3px;cursor:pointer;\
             background:{bg};font-size:11px;color:var(--rinch-color-text);",
        )
    };

    rsx! {
        div {
            div { style: {DIVIDER_STYLE} }
            div { style: {SECTION_STYLE}, "Animation" }
            div {
                style: "display:flex;gap:6px;padding:2px 12px;",
                div {
                    style: {btn_style(cur_state == 1)},
                    onclick: {
                        let cmd = cmd.clone();
                        move || {
                            let _ = cmd.0.send(EditorCommand::SetAnimationState { state: 1 });
                            anim_state_signal.set(1);
                        }
                    },
                    "Play"
                }
                div {
                    style: {btn_style(cur_state == 2)},
                    onclick: {
                        let cmd = cmd.clone();
                        move || {
                            let _ = cmd.0.send(EditorCommand::SetAnimationState { state: 2 });
                            anim_state_signal.set(2);
                        }
                    },
                    "Pause"
                }
                div {
                    style: {btn_style(cur_state == 0)},
                    onclick: {
                        let cmd = cmd.clone();
                        move || {
                            let _ = cmd.0.send(EditorCommand::SetAnimationState { state: 0 });
                            anim_state_signal.set(0);
                        }
                    },
                    "Stop"
                }
            }
            {speed_slider}
        }
    }
}

// ── Combined environment panel ──────────────────────────────────────────────

/// Combines all environment + post-process sections into a single panel.
#[component]
pub fn EnvironmentPanel() -> NodeHandle {
    rsx! {
        div {
            AtmosphereSection {}
            FogSection {}
            CloudsSection {}
            PostProcessSection {}
            AnimationSection {}
        }
    }
}
