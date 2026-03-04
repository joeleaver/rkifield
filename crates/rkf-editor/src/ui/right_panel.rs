//! Right panel — properties, environment, sculpt/paint settings.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiSignals};

use super::slider_helpers::{build_slider_row, build_synced_slider};
use super::{LABEL_STYLE, SECTION_STYLE, VALUE_STYLE, DIVIDER_STYLE};

// ── Mode-dependent right panel ──────────────────────────────────────────────

/// Right panel — always shows properties of the selected object + environment.
///
/// When a Sculpt/Paint tool is active, shows brush settings above the
/// properties section (placeholder for now). Camera selection shows
/// interactive FOV, fly speed, near/far sliders. Below properties, an
/// always-visible environment section shows atmosphere, fog, and quick
/// post-processing controls.
#[component]
pub fn RightPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let ui = use_context::<UiSignals>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:scroll;min-height:0;height:0;",
    );

    // Centralized slider signals — created once in main.rs, stored in rinch context.
    // A single batch Effect (below) syncs all values to EditorState per frame.
    let sliders = use_context::<SliderSignals>();

    // ── Selection-change Effect: push object/light values into SliderSignals ──
    // When the selected entity changes, read its transform from EditorState
    // and push into the persistent slider signals. This runs once per selection
    // change, not per slider drag.
    //
    // CRITICAL: Release the es lock BEFORE calling push_*_values(), because
    // signal .set() triggers effect flushes → batch sync Effect → es.lock()
    // → deadlock if we still hold the lock here.
    {
        let es = editor_state.clone();
        Effect::new(move || {
            let sel = ui.selection.get();

            // Phase 1: Read values from EditorState (lock held briefly).
            enum PushData {
                Object(glam::Vec3, glam::Vec3, glam::Vec3), // pos, rot_deg, scale
                Light(glam::Vec3, f32, f32),                 // pos, intensity, range
                None,
            }
            let (push, oid, lid) = match sel {
                Some(SelectedEntity::Object(oid)) => {
                    let data = es.lock().ok().and_then(|es| {
                        let scene = es.world.scene();
                        let obj = scene.objects.iter().find(|o| o.id as u64 == oid)?;
                        let (rx, ry, rz) = obj.rotation.to_euler(glam::EulerRot::XYZ);
                        Some(PushData::Object(
                            obj.position,
                            glam::Vec3::new(rx.to_degrees(), ry.to_degrees(), rz.to_degrees()),
                            obj.scale,
                        ))
                    }).unwrap_or(PushData::None);
                    (data, Some(oid), None)
                }
                Some(SelectedEntity::Light(lid)) => {
                    let data = es.lock().ok().and_then(|es| {
                        let light = es.light_editor.get_light(lid)?;
                        Some(PushData::Light(light.position, light.intensity, light.range))
                    }).unwrap_or(PushData::None);
                    (data, None, Some(lid))
                }
                _ => (PushData::None, None, None),
            };
            // es lock released here.

            // Phase 2: Push values into signals (no lock held).
            rinch::core::untracked(|| {
                sliders.bound_object_id.set(oid);
                sliders.bound_light_id.set(lid);
            });
            match push {
                PushData::Object(pos, rot_deg, scale) => {
                    rinch::core::untracked(|| {
                        sliders.push_object_values(pos, rot_deg, scale);
                    });
                }
                PushData::Light(pos, intensity, range) => {
                    rinch::core::untracked(|| {
                        sliders.push_light_values(pos, intensity, range);
                    });
                }
                PushData::None => {}
            }
        });
    }

    // ── Batch sync Effect: slider signals + toggle signals → EditorState ──
    // This is the ONLY place that locks EditorState for slider/toggle updates.
    // One lock per frame, regardless of how many sliders changed.
    {
        let es = editor_state.clone();
        Effect::new(move || {
            // Subscribe to all slider signals.
            sliders.track_all();
            // Subscribe to toggle signals (from UiSignals).
            let _ = ui.atmo_enabled.get();
            let _ = ui.fog_enabled.get();
            let _ = ui.clouds_enabled.get();
            let _ = ui.bloom_enabled.get();
            let _ = ui.dof_enabled.get();
            let _ = ui.tone_map_mode.get();

            // Batch write to EditorState.
            if let Ok(mut es) = es.lock() {
                sliders.sync_to_state(&mut es);
                // Sync toggles.
                es.environment.atmosphere.enabled = ui.atmo_enabled.get();
                es.environment.fog.enabled = ui.fog_enabled.get();
                es.environment.clouds.enabled = ui.clouds_enabled.get();
                es.environment.post_process.bloom_enabled = ui.bloom_enabled.get();
                es.environment.post_process.dof_enabled = ui.dof_enabled.get();
                es.environment.post_process.tone_map_mode = ui.tone_map_mode.get();
                es.environment.mark_dirty();
            }
        });
    }

    let es = editor_state.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        // Env toggle signals — needed because toggle labels are built in this closure.
        let _ = ui.atmo_enabled.get();
        let _ = ui.fog_enabled.get();
        let _ = ui.clouds_enabled.get();
        let _ = ui.bloom_enabled.get();
        let _ = ui.dof_enabled.get();
        let _ = ui.tone_map_mode.get();

        let (mode, selected_entity) = match es.lock().ok() {
            Some(es) => (es.mode, es.selected_entity),
            None => return __scope.create_element("div"),
        };

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // ── Tool-specific settings (when Sculpt/Paint active) ──
        match mode {
            EditorMode::Sculpt => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Sculpt Brush"));
                container.append_child(&header);

                // Brush type (read-only for now).
                if let Some(es_lock) = es.lock().ok() {
                    let type_name = match es_lock.sculpt.current_settings.brush_type {
                        crate::sculpt::BrushType::Add => "Add",
                        crate::sculpt::BrushType::Subtract => "Subtract",
                        crate::sculpt::BrushType::Smooth => "Smooth",
                        crate::sculpt::BrushType::Flatten => "Flatten",
                        crate::sculpt::BrushType::Sharpen => "Sharpen",
                    };
                    let row = __scope.create_element("div");
                    row.set_attribute("style", VALUE_STYLE);
                    row.append_child(&__scope.create_text(&format!("Type: {type_name}")));
                    container.append_child(&row);
                }

                build_synced_slider(__scope, &container, "Radius", "",
                    sliders.brush_radius, 0.01, 10.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2);

                // Fix SDFs button — recomputes correct SDF magnitudes from zero-crossings.
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = selected_entity {
                    let btn_row = __scope.create_element("div");
                    btn_row.set_attribute("style", "padding: 6px 8px;");
                    let btn = __scope.create_element("button");
                    btn.set_attribute("style",
                        "width:100%; padding:4px 8px; background:#223355; \
                         color:#99ccff; border:1px solid #3355aa; \
                         border-radius:3px; cursor:pointer; font-size:12px;");
                    btn.append_child(&__scope.create_text("Fix SDFs"));
                    let hid = __scope.register_handler({
                        let es = es.clone();
                        move || {
                            if let Ok(mut es) = es.lock() {
                                es.pending_fix_sdfs = Some(eid as u32);
                            }
                        }
                    });
                    btn.set_attribute("data-rid", &hid.to_string());
                    btn_row.append_child(&btn);
                    container.append_child(&btn_row);
                }

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);
            }
            EditorMode::Paint => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Paint Brush"));
                container.append_child(&header);

                build_synced_slider(__scope, &container, "Radius", "",
                    sliders.brush_radius, 0.01, 10.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2);
                build_synced_slider(__scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2);

                // Divider.
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);
            }
            EditorMode::Default => {}
        }

        // ── Properties (always shown) ──
        let header = __scope.create_element("div");
        header.set_attribute("style", LABEL_STYLE);
        header.append_child(&__scope.create_text("Properties"));
        container.append_child(&header);

        // ── Camera-specific property editing with sliders ──
        if let Some(SelectedEntity::Camera) = selected_entity {
            let pos = es.lock().ok()
                .map(|es| es.editor_camera.position)
                .unwrap_or(glam::Vec3::ZERO);

            let name_row = __scope.create_element("div");
            name_row.set_attribute("style", SECTION_STYLE);
            name_row.append_child(&__scope.create_text("Camera"));
            container.append_child(&name_row);

            build_synced_slider(__scope, &container, "FOV", "\u{00b0}",
                sliders.fov, 30.0, 120.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Fly Speed", "",
                sliders.fly_speed, 0.5, 500.0, 0.5, 1);
            build_synced_slider(__scope, &container, "Near Plane", "",
                sliders.near, 0.01, 10.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Far Plane", "",
                sliders.far, 100.0, 10000.0, 100.0, 0);

            // Divider before position.
            let div = __scope.create_element("div");
            div.set_attribute(
                "style",
                "height:1px;background:var(--rinch-color-border);margin:6px 12px;",
            );
            container.append_child(&div);

            // Position (read-only).
            let pos_row = __scope.create_element("div");
            pos_row.set_attribute("style", VALUE_STYLE);
            pos_row.append_child(&__scope.create_text(
                &format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z),
            ));
            container.append_child(&pos_row);

            // ── Divider between camera props and environment ──
            let div = __scope.create_element("div");
            div.set_attribute(
                "style",
                DIVIDER_STYLE,
            );
            container.append_child(&div);

            // ── Atmosphere section ──
            let atmo_header = __scope.create_element("div");
            atmo_header.set_attribute("style", LABEL_STYLE);
            atmo_header.append_child(&__scope.create_text("Atmosphere"));
            container.append_child(&atmo_header);

            build_synced_slider(__scope, &container, "Sun Azimuth", "\u{00b0}",
                sliders.sun_azimuth, 0.0, 360.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Sun Elevation", "\u{00b0}",
                sliders.sun_elevation, -90.0, 90.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Sun Intensity", "",
                sliders.sun_intensity, 0.0, 10.0, 0.1, 1);

            // Sun color (read-only display).
            if let Some(es_lock) = es.lock().ok() {
                let sc = es_lock.environment.atmosphere.sun_color;
                let color_row = __scope.create_element("div");
                color_row.set_attribute("style", VALUE_STYLE);
                color_row.append_child(&__scope.create_text(
                    &format!("Sun Color: ({:.2}, {:.2}, {:.2})", sc.x, sc.y, sc.z),
                ));
                container.append_child(&color_row);
            }

            // Atmosphere enable toggle.
            {
                let atmo_on = ui.atmo_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if atmo_on { "Atmosphere: ON" } else { "Atmosphere: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.atmo_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Rayleigh Scale", "",
                sliders.rayleigh_scale, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Mie Scale", "",
                sliders.mie_scale, 0.0, 5.0, 0.1, 1);

            // ── Fog section ──
            let fog_header = __scope.create_element("div");
            fog_header.set_attribute("style", LABEL_STYLE);
            fog_header.append_child(&__scope.create_text("Fog"));
            container.append_child(&fog_header);

            // Fog enable toggle.
            {
                let fog_on = ui.fog_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if fog_on { "Fog: ON" } else { "Fog: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.fog_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Fog Density", "",
                sliders.fog_density, 0.0, 0.5, 0.001, 3);
            build_synced_slider(__scope, &container, "Height Falloff", "",
                sliders.fog_height_falloff, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Dust Density", "",
                sliders.dust_density, 0.0, 0.1, 0.001, 3);
            build_synced_slider(__scope, &container, "Dust Asymmetry", "",
                sliders.dust_asymmetry, 0.0, 0.95, 0.05, 2);

            // ── Clouds section ──
            let cloud_header = __scope.create_element("div");
            cloud_header.set_attribute("style", LABEL_STYLE);
            cloud_header.append_child(&__scope.create_text("Clouds"));
            container.append_child(&cloud_header);

            // Cloud enable toggle.
            {
                let cloud_on = ui.clouds_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if cloud_on { "Clouds: ON" } else { "Clouds: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.clouds_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Coverage", "",
                sliders.cloud_coverage, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Cloud Density", "",
                sliders.cloud_density, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Altitude", "m",
                sliders.cloud_altitude, 0.0, 5000.0, 50.0, 0);
            build_synced_slider(__scope, &container, "Thickness", "m",
                sliders.cloud_thickness, 10.0, 10000.0, 50.0, 0);
            build_synced_slider(__scope, &container, "Wind Speed", "",
                sliders.cloud_wind_speed, 0.0, 50.0, 0.5, 1);

            // ── Post-Processing section ──
            let pp_header = __scope.create_element("div");
            pp_header.set_attribute("style", LABEL_STYLE);
            pp_header.append_child(&__scope.create_text("Post-Processing"));
            container.append_child(&pp_header);

            // Bloom enable toggle.
            {
                let bloom_on = ui.bloom_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if bloom_on { "Bloom: ON" } else { "Bloom: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.bloom_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Bloom Intensity", "",
                sliders.bloom_intensity, 0.0, 2.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Bloom Threshold", "",
                sliders.bloom_threshold, 0.0, 5.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Exposure", "",
                sliders.exposure, 0.1, 10.0, 0.1, 1);

            // Tone map mode toggle (ACES / AgX).
            {
                let tm_mode = ui.tone_map_mode.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if tm_mode == 0 { "Tone Map: ACES" } else { "Tone Map: AgX" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.tone_map_mode.update(|v| *v = if *v == 0 { 1 } else { 0 });
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Sharpen", "",
                sliders.sharpen, 0.0, 2.0, 0.05, 2);

            // DoF enable toggle.
            {
                let dof_on = ui.dof_enabled.get();
                let toggle_row = __scope.create_element("div");
                toggle_row.set_attribute(
                    "style",
                    "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);\
                     cursor:pointer;user-select:none;",
                );
                let label = if dof_on { "DoF: ON" } else { "DoF: OFF" };
                toggle_row.append_child(&__scope.create_text(label));
                let hid = __scope.register_handler(move || {
                    ui.dof_enabled.update(|v| *v = !*v);
                });
                toggle_row.set_attribute("data-rid", &hid.to_string());
                container.append_child(&toggle_row);
            }

            build_synced_slider(__scope, &container, "Focus Distance", "",
                sliders.dof_focus_dist, 0.1, 50.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Focus Range", "",
                sliders.dof_focus_range, 0.1, 20.0, 0.1, 1);
            build_synced_slider(__scope, &container, "Max CoC", "px",
                sliders.dof_max_coc, 1.0, 32.0, 1.0, 0);
            build_synced_slider(__scope, &container, "Motion Blur", "",
                sliders.motion_blur, 0.0, 3.0, 0.1, 1);
            build_synced_slider(__scope, &container, "God Rays", "",
                sliders.god_rays, 0.0, 2.0, 0.05, 2);
            build_synced_slider(__scope, &container, "Vignette", "",
                sliders.vignette, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Grain", "",
                sliders.grain, 0.0, 1.0, 0.01, 2);
            build_synced_slider(__scope, &container, "Chromatic Ab.", "",
                sliders.chromatic_ab, 0.0, 1.0, 0.01, 2);

            // ── Animation controls ──
            {
                let div = __scope.create_element("div");
                div.set_attribute(
                    "style",
                    DIVIDER_STYLE,
                );
                container.append_child(&div);

                let hdr = __scope.create_element("div");
                hdr.set_attribute("style", SECTION_STYLE);
                hdr.append_child(&__scope.create_text("Animation"));
                container.append_child(&hdr);

                // Play / Pause / Stop buttons.
                let btn_row = __scope.create_element("div");
                btn_row.set_attribute("style", "display:flex;gap:6px;padding:2px 12px;");

                for (label, state) in [
                    ("Play", crate::animation_preview::PlaybackState::Playing),
                    ("Pause", crate::animation_preview::PlaybackState::Paused),
                    ("Stop", crate::animation_preview::PlaybackState::Stopped),
                ] {
                    let btn = __scope.create_element("div");
                    let is_active = es.lock()
                        .map(|e| e.animation.playback_state == state)
                        .unwrap_or(false);
                    let bg = if is_active { "var(--rinch-primary-color)" } else { "var(--rinch-color-dark-7)" };
                    btn.set_attribute("style", &format!(
                        "padding:2px 8px;border-radius:3px;cursor:pointer;\
                         background:{bg};font-size:11px;color:var(--rinch-color-text);",
                    ));
                    btn.append_child(&__scope.create_text(label));

                    let hid = __scope.register_handler({
                        let es = es.clone();
                        move || {
                            let anim_val = match state {
                                crate::animation_preview::PlaybackState::Stopped => 0u32,
                                crate::animation_preview::PlaybackState::Playing => 1,
                                crate::animation_preview::PlaybackState::Paused => 2,
                            };
                            if let Ok(mut es) = es.lock() {
                                es.animation.playback_state = state;
                                if matches!(state, crate::animation_preview::PlaybackState::Stopped) {
                                    es.animation.current_time = 0.0;
                                }
                            }
                            ui.animation_state.set(anim_val);
                        }
                    });
                    btn.set_attribute("data-rid", &hid.to_string());
                    btn_row.append_child(&btn);
                }
                container.append_child(&btn_row);

                let anim_speed_signal: Signal<f64> = Signal::new(
                    es.lock().ok().map(|e| e.animation.speed as f64).unwrap_or(1.0),
                );

                build_slider_row(
                    __scope, &container, "Speed", "x", anim_speed_signal,
                    0.0, 4.0, 0.1, 1,
                    { let es = es.clone(); move |v| {
                        if let Ok(mut es) = es.lock() {
                            es.animation.speed = v as f32;
                        }
                    }},
                );
            }
        } else {
            // ── Non-camera entity properties ──
            match selected_entity {
                Some(SelectedEntity::Object(eid)) => {
                    // Read object info + transform from world scene.
                    let obj_info = es.lock().ok().and_then(|es_lock| {
                        let scene = es_lock.world.scene();
                        let obj = scene.objects.iter().find(|o| o.id as u64 == eid)?;
                        let name = obj.name.clone();
                        let child_count = scene.objects.iter()
                            .filter(|o| o.parent_id == Some(obj.id))
                            .count();
                        let (x, y, z) = obj.rotation.to_euler(glam::EulerRot::XYZ);
                        let is_vox = matches!(
                            obj.root_node.sdf_source,
                            rkf_core::scene_node::SdfSource::Voxelized { .. }
                        );
                        let non_uniform = (obj.scale - glam::Vec3::ONE).length() > 1e-4;
                        let xf = Some((
                            obj.position,
                            glam::Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees()),
                            obj.scale,
                        ));
                        let show_revoxelize = is_vox && non_uniform;
                        Some((name, child_count, xf, show_revoxelize))
                    });

                    if let Some((name, child_count, xf, show_revoxelize)) = obj_info {
                        // Name.
                        let name_row = __scope.create_element("div");
                        name_row.set_attribute("style", SECTION_STYLE);
                        name_row.append_child(&__scope.create_text(&name));
                        container.append_child(&name_row);

                        // Entity ID.
                        let id_row = __scope.create_element("div");
                        id_row.set_attribute("style", VALUE_STYLE);
                        id_row.append_child(
                            &__scope.create_text(&format!("Entity ID: {eid}")),
                        );
                        container.append_child(&id_row);

                        if child_count > 0 {
                            let cr = __scope.create_element("div");
                            cr.set_attribute("style", VALUE_STYLE);
                            cr.append_child(
                                &__scope.create_text(&format!("Children: {child_count}")),
                            );
                            container.append_child(&cr);
                        }

                        // Transform sliders (only when scene object found).
                        if xf.is_some() {
                            let div = __scope.create_element("div");
                            div.set_attribute("style", DIVIDER_STYLE);
                            container.append_child(&div);

                            let xf_hdr = __scope.create_element("div");
                            xf_hdr.set_attribute("style", LABEL_STYLE);
                            xf_hdr.append_child(&__scope.create_text("Transform"));
                            container.append_child(&xf_hdr);

                            // Position — uses centralized SliderSignals (batch sync).
                            build_synced_slider(__scope, &container, "Position X", "",
                                sliders.obj_pos_x, -500.0, 500.0, 0.01, 2);
                            build_synced_slider(__scope, &container, "Position Y", "",
                                sliders.obj_pos_y, -500.0, 500.0, 0.01, 2);
                            build_synced_slider(__scope, &container, "Position Z", "",
                                sliders.obj_pos_z, -500.0, 500.0, 0.01, 2);

                            // Rotation (Euler XYZ degrees).
                            build_synced_slider(__scope, &container, "Rotation X", "\u{00b0}",
                                sliders.obj_rot_x, -180.0, 180.0, 0.5, 1);
                            build_synced_slider(__scope, &container, "Rotation Y", "\u{00b0}",
                                sliders.obj_rot_y, -180.0, 180.0, 0.5, 1);
                            build_synced_slider(__scope, &container, "Rotation Z", "\u{00b0}",
                                sliders.obj_rot_z, -180.0, 180.0, 0.5, 1);

                            // Scale X/Y/Z.
                            build_synced_slider(__scope, &container, "Scale X", "",
                                sliders.obj_scale_x, 0.01, 50.0, 0.01, 2);
                            build_synced_slider(__scope, &container, "Scale Y", "",
                                sliders.obj_scale_y, 0.01, 50.0, 0.01, 2);
                            build_synced_slider(__scope, &container, "Scale Z", "",
                                sliders.obj_scale_z, 0.01, 50.0, 0.01, 2);

                            // Re-voxelize button (only for voxelized objects with non-uniform scale).
                            if show_revoxelize {
                                let btn_row = __scope.create_element("div");
                                btn_row.set_attribute("style", "padding: 6px 8px;");
                                let btn = __scope.create_element("button");
                                btn.set_attribute("style",
                                    "width:100%; padding:4px 8px; background:#553322; \
                                     color:#ffcc99; border:1px solid #774433; \
                                     border-radius:3px; cursor:pointer; font-size:12px;");
                                btn.append_child(
                                    &__scope.create_text("Re-voxelize (bake scale)"),
                                );
                                let hid = __scope.register_handler({
                                    let es = es.clone();
                                    move || {
                                        if let Ok(mut es) = es.lock() {
                                            es.pending_revoxelize = Some(eid as u32);
                                        }
                                    }
                                });
                                btn.set_attribute("data-rid", &hid.to_string());
                                btn_row.append_child(&btn);
                                container.append_child(&btn_row);
                            }
                        }
                    }
                }
                Some(SelectedEntity::Light(lid)) => {
                    // Read light data.
                    let light_data = es.lock().ok().and_then(|es_lock| {
                        es_lock.light_editor.get_light(lid).map(|light| {
                            (light.intensity, light.range, light.light_type, light.position)
                        })
                    });
                    if let Some((_intensity, _range, light_type, _position)) = light_data {
                        let type_name = match light_type {
                            crate::light_editor::EditorLightType::Point => "Point Light",
                            crate::light_editor::EditorLightType::Spot => "Spot Light",
                        };
                        let hdr = __scope.create_element("div");
                        hdr.set_attribute("style", SECTION_STYLE);
                        hdr.append_child(&__scope.create_text(type_name));
                        container.append_child(&hdr);

                        // Position — uses centralized SliderSignals (batch sync).
                        build_synced_slider(__scope, &container, "Position X", "",
                            sliders.light_pos_x, -500.0, 500.0, 0.01, 2);
                        build_synced_slider(__scope, &container, "Position Y", "",
                            sliders.light_pos_y, -500.0, 500.0, 0.01, 2);
                        build_synced_slider(__scope, &container, "Position Z", "",
                            sliders.light_pos_z, -500.0, 500.0, 0.01, 2);

                        let div = __scope.create_element("div");
                        div.set_attribute("style", DIVIDER_STYLE);
                        container.append_child(&div);

                        // Intensity and range — uses centralized SliderSignals.
                        build_synced_slider(__scope, &container, "Intensity", "",
                            sliders.light_intensity, 0.0, 50.0, 0.1, 1);
                        build_synced_slider(__scope, &container, "Range", "m",
                            sliders.light_range, 0.1, 100.0, 0.5, 1);
                    }
                }
                Some(SelectedEntity::Scene) => {
                    let name = es.lock().ok()
                        .map(|e| e.world.scene().name.clone())
                        .unwrap_or_else(|| "Scene".to_string());
                    let count = es.lock().ok().map(|e| e.world.scene().objects.len()).unwrap_or(0);

                    let name_row = __scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    name_row.append_child(&__scope.create_text(&name));
                    container.append_child(&name_row);

                    let detail = __scope.create_element("div");
                    detail.set_attribute("style", VALUE_STYLE);
                    detail.append_child(
                        &__scope.create_text(&format!("{count} objects")),
                    );
                    container.append_child(&detail);
                }
                Some(SelectedEntity::Project) => {
                    let hdr = __scope.create_element("div");
                    hdr.set_attribute("style", SECTION_STYLE);
                    hdr.append_child(&__scope.create_text("Project"));
                    container.append_child(&hdr);
                }
                _ => {
                    let msg = __scope.create_element("div");
                    msg.set_attribute(
                        "style",
                        &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                    );
                    msg.append_child(&__scope.create_text("No object selected"));
                    container.append_child(&msg);
                }
            }
        }

        container
    });

    root
}
