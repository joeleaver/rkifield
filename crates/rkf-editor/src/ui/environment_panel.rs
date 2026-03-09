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

    let root = __scope.create_element("div");

    // Header.
    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Atmosphere"));
    root.append_child(&header);

    build_slider_row(
        __scope, &root, "Sun Azimuth", "\u{00b0}",
        sliders.sun_azimuth, 0.0, 360.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Sun Elevation", "\u{00b0}",
        sliders.sun_elevation, -90.0, 90.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Sun Intensity", "",
        sliders.sun_intensity, 0.0, 10.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );

    // Atmosphere toggle.
    let toggle = ToggleRow {
        label: "Atmosphere".to_string(),
        enabled: Some(ui.atmo_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_atmosphere_commands(&cmd, &ui); }
        })),
    };
    root.append_child(&toggle.render(__scope, &[]));

    build_slider_row(
        __scope, &root, "Rayleigh Scale", "",
        sliders.rayleigh_scale, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Mie Scale", "",
        sliders.mie_scale, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_atmosphere_commands(&cmd, &ui); } },
    );

    root.into()
}

// ── Fog section ─────────────────────────────────────────────────────────────

/// Fog toggle + density, height falloff, dust density, dust asymmetry sliders.
#[component]
pub fn FogSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");

    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Fog"));
    root.append_child(&header);

    let fog_toggle = ToggleRow {
        label: "Fog".to_string(),
        enabled: Some(ui.fog_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_fog_commands(&cmd, &ui); }
        })),
    };
    root.append_child(&fog_toggle.render(__scope, &[]));

    build_slider_row(
        __scope, &root, "Fog Density", "",
        sliders.fog_density, 0.0, 0.5, 0.001, 3,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Height Falloff", "",
        sliders.fog_height_falloff, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Dust Density", "",
        sliders.dust_density, 0.0, 0.1, 0.001, 3,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Dust Asymmetry", "",
        sliders.dust_asymmetry, 0.0, 0.95, 0.05, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_fog_commands(&cmd, &ui); } },
    );

    root.into()
}

// ── Clouds section ──────────────────────────────────────────────────────────

/// Cloud toggle + coverage, density, altitude, thickness, wind speed sliders.
#[component]
pub fn CloudsSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");

    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Clouds"));
    root.append_child(&header);

    let cloud_toggle = ToggleRow {
        label: "Clouds".to_string(),
        enabled: Some(ui.clouds_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_cloud_commands(&cmd, &ui); }
        })),
    };
    root.append_child(&cloud_toggle.render(__scope, &[]));

    build_slider_row(
        __scope, &root, "Coverage", "",
        sliders.cloud_coverage, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Cloud Density", "",
        sliders.cloud_density, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Altitude", "m",
        sliders.cloud_altitude, 0.0, 5000.0, 50.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Thickness", "m",
        sliders.cloud_thickness, 10.0, 10000.0, 50.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Wind Speed", "",
        sliders.cloud_wind_speed, 0.0, 50.0, 0.5, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_cloud_commands(&cmd, &ui); } },
    );

    root.into()
}

// ── Post-processing section ─────────────────────────────────────────────────

/// Bloom, exposure, tone mapping, sharpen, DoF, motion blur, god rays,
/// vignette, grain, chromatic aberration.
#[component]
pub fn PostProcessSection() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");

    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Post-Processing"));
    root.append_child(&header);

    // Bloom toggle.
    let bloom_toggle = ToggleRow {
        label: "Bloom".to_string(),
        enabled: Some(ui.bloom_enabled),
        on_change: Some(Callback::new({
            let cmd = cmd.clone();
            move || { sliders.send_post_process_commands(&cmd, &ui); }
        })),
    };
    root.append_child(&bloom_toggle.render(__scope, &[]));

    build_slider_row(
        __scope, &root, "Bloom Intensity", "",
        sliders.bloom_intensity, 0.0, 2.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Bloom Threshold", "",
        sliders.bloom_threshold, 0.0, 5.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Exposure", "",
        sliders.exposure, 0.1, 10.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );

    // Tone map mode toggle (ACES / AgX).
    {
        let tm_mode = ui.tone_map_mode.get();
        let tm_signal = ui.tone_map_mode;
        let toggle_row = __scope.create_element("div");
        toggle_row.set_attribute(
            "style",
            "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
             cursor:pointer;user-select:none;",
        );
        let label = if tm_mode == 0 {
            "Tone Map: ACES"
        } else {
            "Tone Map: AgX"
        };
        toggle_row.append_child(&__scope.create_text(label));
        let hid = __scope.register_handler({
            let cmd = cmd.clone();
            move || {
                tm_signal.update(|v| *v = if *v == 0 { 1 } else { 0 });
                sliders.send_post_process_commands(&cmd, &ui);
            }
        });
        toggle_row.set_attribute("data-rid", &hid.to_string());
        root.append_child(&toggle_row);
    }

    build_slider_row(
        __scope, &root, "Sharpen", "",
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
    };
    root.append_child(&dof_toggle.render(__scope, &[]));

    build_slider_row(
        __scope, &root, "Focus Distance", "",
        sliders.dof_focus_dist, 0.1, 50.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Focus Range", "",
        sliders.dof_focus_range, 0.1, 20.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Max CoC", "px",
        sliders.dof_max_coc, 1.0, 32.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Motion Blur", "",
        sliders.motion_blur, 0.0, 3.0, 0.1, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "God Rays", "",
        sliders.god_rays, 0.0, 2.0, 0.05, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Vignette", "",
        sliders.vignette, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Grain", "",
        sliders.grain, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );
    build_slider_row(
        __scope, &root, "Chromatic Ab.", "",
        sliders.chromatic_ab, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_post_process_commands(&cmd, &ui); } },
    );

    root.into()
}

// ── Animation section ───────────────────────────────────────────────────────

/// Play/pause/stop buttons + animation speed slider.
#[component]
pub fn AnimationSection() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");

    // Divider before animation section.
    let div = __scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    root.append_child(&div);

    let anim_hdr = __scope.create_element("div");
    anim_hdr.set_attribute("style", SECTION_STYLE);
    anim_hdr.append_child(&__scope.create_text("Animation"));
    root.append_child(&anim_hdr);

    // Play / Pause / Stop buttons.
    let btn_row = __scope.create_element("div");
    btn_row.set_attribute("style", "display:flex;gap:6px;padding:2px 12px;");
    let anim_state_signal = ui.animation_state;

    for (label, state) in [
        (
            "Play",
            crate::animation_preview::PlaybackState::Playing,
        ),
        (
            "Pause",
            crate::animation_preview::PlaybackState::Paused,
        ),
        (
            "Stop",
            crate::animation_preview::PlaybackState::Stopped,
        ),
    ] {
        let btn = __scope.create_element("div");
        let state_val = match state {
            crate::animation_preview::PlaybackState::Stopped => 0u32,
            crate::animation_preview::PlaybackState::Playing => 1,
            crate::animation_preview::PlaybackState::Paused => 2,
        };
        let is_active = ui.animation_state.get() == state_val;
        let bg = if is_active {
            "var(--rinch-primary-color)"
        } else {
            "var(--rinch-color-dark-7)"
        };
        btn.set_attribute(
            "style",
            &format!(
                "padding:2px 8px;border-radius:3px;cursor:pointer;\
                 background:{bg};font-size:11px;color:var(--rinch-color-text);",
            ),
        );
        btn.append_child(&__scope.create_text(label));

        let hid = __scope.register_handler({
            let cmd = cmd.clone();
            move || {
                let anim_val = match state {
                    crate::animation_preview::PlaybackState::Stopped => 0u32,
                    crate::animation_preview::PlaybackState::Playing => 1,
                    crate::animation_preview::PlaybackState::Paused => 2,
                };
                let _ = cmd.0.send(EditorCommand::SetAnimationState { state: anim_val });
                anim_state_signal.set(anim_val);
            }
        });
        btn.set_attribute("data-rid", &hid.to_string());
        btn_row.append_child(&btn);
    }
    root.append_child(&btn_row);

    let anim_speed_signal: Signal<f64> = Signal::new(1.0);

    build_slider_row(
        __scope,
        &root,
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

    root.into()
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
