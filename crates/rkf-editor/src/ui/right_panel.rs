//! Right panel — properties, environment, sculpt/paint settings.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, EditorState, SelectedEntity, SliderSignals, UiSignals};
use crate::{CommandSender, SnapshotReader};

use super::components::{
    build_synced_slider_row, DragValue, ToggleRow, TransformEditor, Vec3Editor,
};
use super::slider_helpers::build_slider_row;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE, VALUE_STYLE};

// ── Mode-dependent right panel ──────────────────────────────────────────────

/// Right panel — always shows properties of the selected object + environment.
///
/// When a Sculpt/Paint tool is active, shows brush settings above the
/// properties section. Camera selection shows interactive FOV, fly speed,
/// near/far sliders and environment/post-processing sections.
#[component]
pub fn RightPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();

    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "flex:1;overflow-y:scroll;min-height:0;height:0;",
    );

    let sliders = use_context::<SliderSignals>();

    // ── Selection-change Effect: push object/light values into SliderSignals ──
    {
        let snap = snapshot.clone();
        Effect::new(move || {
            let sel = ui.selection.get();

            enum PushData {
                Object(glam::Vec3, glam::Vec3, glam::Vec3),
                Light(glam::Vec3, f32, f32),
                None,
            }
            let snap_guard = snap.0.load();
            let (push, oid, lid) = match sel {
                Some(SelectedEntity::Object(oid)) => {
                    let data = snap_guard
                        .objects
                        .iter()
                        .find(|o| o.id == oid)
                        .map(|o| PushData::Object(o.position, o.rotation_degrees, o.scale))
                        .unwrap_or(PushData::None);
                    (data, Some(oid), None)
                }
                Some(SelectedEntity::Light(lid)) => {
                    let data = snap_guard
                        .lights
                        .iter()
                        .find(|l| l.id == lid)
                        .map(|l| PushData::Light(l.position, l.intensity, l.range))
                        .unwrap_or(PushData::None);
                    (data, None, Some(lid))
                }
                _ => (PushData::None, None, None),
            };

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

    // ── Batch sync Effect: slider signals + toggle signals → engine commands ──
    {
        let cmd = cmd.clone();
        Effect::new(move || {
            sliders.track_all();
            let _ = ui.atmo_enabled.get();
            let _ = ui.fog_enabled.get();
            let _ = ui.clouds_enabled.get();
            let _ = ui.bloom_enabled.get();
            let _ = ui.dof_enabled.get();
            let _ = ui.tone_map_mode.get();

            // Camera.
            let _ = cmd
                .0
                .send(EditorCommand::SetCameraFov {
                    fov: sliders.fov.get() as f32,
                });
            let _ = cmd.0.send(EditorCommand::SetCameraSpeed {
                speed: sliders.fly_speed.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetCameraNearFar {
                near: sliders.near.get() as f32,
                far: sliders.far.get() as f32,
            });

            // Atmosphere.
            let az = (sliders.sun_azimuth.get() as f32).to_radians();
            let el = (sliders.sun_elevation.get() as f32).to_radians();
            let cos_el = el.cos();
            let sun_dir =
                glam::Vec3::new(az.sin() * cos_el, el.sin(), az.cos() * cos_el).normalize();
            let _ = cmd.0.send(EditorCommand::SetAtmosphere {
                sun_direction: sun_dir,
                sun_intensity: sliders.sun_intensity.get() as f32,
                rayleigh_scale: sliders.rayleigh_scale.get() as f32,
                mie_scale: sliders.mie_scale.get() as f32,
            });

            // Fog.
            let _ = cmd.0.send(EditorCommand::SetFog {
                density: sliders.fog_density.get() as f32,
                height_falloff: sliders.fog_height_falloff.get() as f32,
                dust_density: sliders.dust_density.get() as f32,
                dust_asymmetry: sliders.dust_asymmetry.get() as f32,
            });

            // Clouds.
            let _ = cmd.0.send(EditorCommand::SetClouds {
                coverage: sliders.cloud_coverage.get() as f32,
                density: sliders.cloud_density.get() as f32,
                altitude: sliders.cloud_altitude.get() as f32,
                thickness: sliders.cloud_thickness.get() as f32,
                wind_speed: sliders.cloud_wind_speed.get() as f32,
            });

            // Post-process.
            let _ = cmd.0.send(EditorCommand::SetPostProcess {
                bloom_intensity: sliders.bloom_intensity.get() as f32,
                bloom_threshold: sliders.bloom_threshold.get() as f32,
                exposure: sliders.exposure.get() as f32,
                sharpen: sliders.sharpen.get() as f32,
                dof_focus_distance: sliders.dof_focus_dist.get() as f32,
                dof_focus_range: sliders.dof_focus_range.get() as f32,
                dof_max_coc: sliders.dof_max_coc.get() as f32,
                motion_blur: sliders.motion_blur.get() as f32,
                god_rays: sliders.god_rays.get() as f32,
                vignette: sliders.vignette.get() as f32,
                grain: sliders.grain.get() as f32,
                chromatic_aberration: sliders.chromatic_ab.get() as f32,
            });

            // Brush.
            let _ = cmd.0.send(EditorCommand::SetSculptSettings {
                radius: sliders.brush_radius.get() as f32,
                strength: sliders.brush_strength.get() as f32,
                falloff: sliders.brush_falloff.get() as f32,
            });
            let _ = cmd.0.send(EditorCommand::SetPaintSettings {
                radius: sliders.brush_radius.get() as f32,
                strength: sliders.brush_strength.get() as f32,
                falloff: sliders.brush_falloff.get() as f32,
            });

            // Object transform.
            let obj_id = rinch::core::untracked(|| sliders.bound_object_id.get());
            if let Some(oid) = obj_id {
                let _ = cmd.0.send(EditorCommand::SetObjectPosition {
                    entity_id: oid,
                    position: glam::Vec3::new(
                        sliders.obj_pos_x.get() as f32,
                        sliders.obj_pos_y.get() as f32,
                        sliders.obj_pos_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetObjectRotation {
                    entity_id: oid,
                    rotation: glam::Vec3::new(
                        sliders.obj_rot_x.get() as f32,
                        sliders.obj_rot_y.get() as f32,
                        sliders.obj_rot_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetObjectScale {
                    entity_id: oid,
                    scale: glam::Vec3::new(
                        sliders.obj_scale_x.get() as f32,
                        sliders.obj_scale_y.get() as f32,
                        sliders.obj_scale_z.get() as f32,
                    ),
                });
            }

            // Light properties.
            let light_id = rinch::core::untracked(|| sliders.bound_light_id.get());
            if let Some(lid) = light_id {
                let _ = cmd.0.send(EditorCommand::SetLightPosition {
                    light_id: lid,
                    position: glam::Vec3::new(
                        sliders.light_pos_x.get() as f32,
                        sliders.light_pos_y.get() as f32,
                        sliders.light_pos_z.get() as f32,
                    ),
                });
                let _ = cmd.0.send(EditorCommand::SetLightIntensity {
                    light_id: lid,
                    intensity: sliders.light_intensity.get() as f32,
                });
                let _ = cmd.0.send(EditorCommand::SetLightRange {
                    light_id: lid,
                    range: sliders.light_range.get() as f32,
                });
            }

            // Toggles.
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
        });
    }

    let es = editor_state.clone();
    let snap = snapshot.clone();
    let cmd2 = cmd.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        // Fine-grained signal tracking.
        let _ = ui.selection.get();
        let _ = ui.editor_mode.get();
        let _ = ui.atmo_enabled.get();
        let _ = ui.fog_enabled.get();
        let _ = ui.clouds_enabled.get();
        let _ = ui.bloom_enabled.get();
        let _ = ui.dof_enabled.get();
        let _ = ui.tone_map_mode.get();

        let snap_guard = snap.0.load();
        let mode = snap_guard.mode;
        // Use signal for selection — snapshot may be stale on the frame of change.
        let selected_entity = ui.selection.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        // ── Tool-specific settings (when Sculpt/Paint active) ──
        match mode {
            EditorMode::Sculpt => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Sculpt Brush"));
                container.append_child(&header);

                if let Ok(es_lock) = es.lock() {
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

                build_synced_slider_row(
                    __scope, &container, "Radius", "",
                    sliders.brush_radius, 0.01, 10.0, 0.01, 2,
                );
                build_synced_slider_row(
                    __scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2,
                );
                build_synced_slider_row(
                    __scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
                );

                // Fix SDFs button.
                if let Some(crate::editor_state::SelectedEntity::Object(eid)) = selected_entity {
                    let btn_row = __scope.create_element("div");
                    btn_row.set_attribute("style", "padding: 6px 8px;");
                    let btn = __scope.create_element("button");
                    btn.set_attribute(
                        "style",
                        "width:100%; padding:4px 8px; background:#223355; \
                         color:#99ccff; border:1px solid #3355aa; \
                         border-radius:3px; cursor:pointer; font-size:12px;",
                    );
                    btn.append_child(&__scope.create_text("Fix SDFs"));
                    let hid = __scope.register_handler({
                        let cmd = cmd2.clone();
                        move || {
                            let _ =
                                cmd.0.send(EditorCommand::FixSdfs {
                                    object_id: eid as u32,
                                });
                        }
                    });
                    btn.set_attribute("data-rid", &hid.to_string());
                    btn_row.append_child(&btn);
                    container.append_child(&btn_row);
                }

                append_divider(__scope, &container);
            }
            EditorMode::Paint => {
                let header = __scope.create_element("div");
                header.set_attribute("style", LABEL_STYLE);
                header.append_child(&__scope.create_text("Paint Brush"));
                container.append_child(&header);

                build_synced_slider_row(
                    __scope, &container, "Radius", "",
                    sliders.brush_radius, 0.01, 10.0, 0.01, 2,
                );
                build_synced_slider_row(
                    __scope, &container, "Strength", "",
                    sliders.brush_strength, 0.0, 1.0, 0.01, 2,
                );
                build_synced_slider_row(
                    __scope, &container, "Falloff", "",
                    sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
                );

                append_divider(__scope, &container);
            }
            EditorMode::Default => {}
        }

        // ── Properties header ──
        let header = __scope.create_element("div");
        header.set_attribute("style", LABEL_STYLE);
        header.append_child(&__scope.create_text("Properties"));
        container.append_child(&header);

        // ── Camera-specific properties ──
        if let Some(SelectedEntity::Camera) = selected_entity {
            build_camera_properties(
                __scope, &container, &snap_guard, &sliders, &ui, &cmd2, &es,
            );
        } else {
            // ── Non-camera entity properties ──
            match selected_entity {
                Some(SelectedEntity::Object(eid)) => {
                    build_object_properties(
                        __scope, &container, eid, &snap_guard, &sliders, &cmd2, &es,
                    );
                }
                Some(SelectedEntity::Light(lid)) => {
                    build_light_properties(__scope, &container, lid, &snap_guard, &sliders);
                }
                Some(SelectedEntity::Scene) => {
                    let name_row = __scope.create_element("div");
                    name_row.set_attribute("style", SECTION_STYLE);
                    name_row.append_child(&__scope.create_text(&snap_guard.scene_name));
                    container.append_child(&name_row);

                    let detail = __scope.create_element("div");
                    detail.set_attribute("style", VALUE_STYLE);
                    detail.append_child(
                        &__scope.create_text(&format!("{} objects", snap_guard.object_count)),
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

// ── Helper: divider ──────────────────────────────────────────────────────────

fn append_divider(scope: &mut RenderScope, container: &NodeHandle) {
    let div = scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    container.append_child(&div);
}

// ── Camera properties ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_camera_properties(
    scope: &mut RenderScope,
    container: &NodeHandle,
    snap_guard: &crate::ui_snapshot::UiSnapshot,
    sliders: &SliderSignals,
    ui: &UiSignals,
    cmd: &CommandSender,
    _es: &Arc<Mutex<EditorState>>,
) {
    let pos = snap_guard.camera_position;

    let name_row = scope.create_element("div");
    name_row.set_attribute("style", SECTION_STYLE);
    name_row.append_child(&scope.create_text("Camera"));
    container.append_child(&name_row);

    build_synced_slider_row(scope, container, "FOV", "\u{00b0}", sliders.fov, 30.0, 120.0, 1.0, 0);
    build_synced_slider_row(scope, container, "Fly Speed", "", sliders.fly_speed, 0.5, 500.0, 0.5, 1);
    build_synced_slider_row(scope, container, "Near Plane", "", sliders.near, 0.01, 10.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Far Plane", "", sliders.far, 100.0, 10000.0, 100.0, 0);

    append_divider(scope, container);

    // Position (read-only).
    let pos_row = scope.create_element("div");
    pos_row.set_attribute("style", VALUE_STYLE);
    pos_row.append_child(
        &scope.create_text(&format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z)),
    );
    container.append_child(&pos_row);

    append_divider(scope, container);

    // ── Atmosphere ──
    let atmo_header = scope.create_element("div");
    atmo_header.set_attribute("style", LABEL_STYLE);
    atmo_header.append_child(&scope.create_text("Atmosphere"));
    container.append_child(&atmo_header);

    build_synced_slider_row(scope, container, "Sun Azimuth", "\u{00b0}", sliders.sun_azimuth, 0.0, 360.0, 1.0, 0);
    build_synced_slider_row(scope, container, "Sun Elevation", "\u{00b0}", sliders.sun_elevation, -90.0, 90.0, 1.0, 0);
    build_synced_slider_row(scope, container, "Sun Intensity", "", sliders.sun_intensity, 0.0, 10.0, 0.1, 1);

    // Atmosphere toggle.
    let toggle = ToggleRow {
        label: "Atmosphere".to_string(),
        enabled: Some(ui.atmo_enabled),
    };
    container.append_child(&toggle.render(scope, &[]));

    build_synced_slider_row(scope, container, "Rayleigh Scale", "", sliders.rayleigh_scale, 0.0, 5.0, 0.1, 1);
    build_synced_slider_row(scope, container, "Mie Scale", "", sliders.mie_scale, 0.0, 5.0, 0.1, 1);

    // ── Fog ──
    let fog_header = scope.create_element("div");
    fog_header.set_attribute("style", LABEL_STYLE);
    fog_header.append_child(&scope.create_text("Fog"));
    container.append_child(&fog_header);

    let fog_toggle = ToggleRow {
        label: "Fog".to_string(),
        enabled: Some(ui.fog_enabled),
    };
    container.append_child(&fog_toggle.render(scope, &[]));

    build_synced_slider_row(scope, container, "Fog Density", "", sliders.fog_density, 0.0, 0.5, 0.001, 3);
    build_synced_slider_row(scope, container, "Height Falloff", "", sliders.fog_height_falloff, 0.0, 1.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Dust Density", "", sliders.dust_density, 0.0, 0.1, 0.001, 3);
    build_synced_slider_row(scope, container, "Dust Asymmetry", "", sliders.dust_asymmetry, 0.0, 0.95, 0.05, 2);

    // ── Clouds ──
    let cloud_header = scope.create_element("div");
    cloud_header.set_attribute("style", LABEL_STYLE);
    cloud_header.append_child(&scope.create_text("Clouds"));
    container.append_child(&cloud_header);

    let cloud_toggle = ToggleRow {
        label: "Clouds".to_string(),
        enabled: Some(ui.clouds_enabled),
    };
    container.append_child(&cloud_toggle.render(scope, &[]));

    build_synced_slider_row(scope, container, "Coverage", "", sliders.cloud_coverage, 0.0, 1.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Cloud Density", "", sliders.cloud_density, 0.0, 5.0, 0.1, 1);
    build_synced_slider_row(scope, container, "Altitude", "m", sliders.cloud_altitude, 0.0, 5000.0, 50.0, 0);
    build_synced_slider_row(scope, container, "Thickness", "m", sliders.cloud_thickness, 10.0, 10000.0, 50.0, 0);
    build_synced_slider_row(scope, container, "Wind Speed", "", sliders.cloud_wind_speed, 0.0, 50.0, 0.5, 1);

    // ── Post-Processing ──
    let pp_header = scope.create_element("div");
    pp_header.set_attribute("style", LABEL_STYLE);
    pp_header.append_child(&scope.create_text("Post-Processing"));
    container.append_child(&pp_header);

    let bloom_toggle = ToggleRow {
        label: "Bloom".to_string(),
        enabled: Some(ui.bloom_enabled),
    };
    container.append_child(&bloom_toggle.render(scope, &[]));

    build_synced_slider_row(scope, container, "Bloom Intensity", "", sliders.bloom_intensity, 0.0, 2.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Bloom Threshold", "", sliders.bloom_threshold, 0.0, 5.0, 0.1, 1);
    build_synced_slider_row(scope, container, "Exposure", "", sliders.exposure, 0.1, 10.0, 0.1, 1);

    // Tone map mode toggle (ACES / AgX).
    {
        let tm_mode = ui.tone_map_mode.get();
        let tm_signal = ui.tone_map_mode;
        let toggle_row = scope.create_element("div");
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
        toggle_row.append_child(&scope.create_text(label));
        let hid = scope.register_handler(move || {
            tm_signal.update(|v| *v = if *v == 0 { 1 } else { 0 });
        });
        toggle_row.set_attribute("data-rid", &hid.to_string());
        container.append_child(&toggle_row);
    }

    build_synced_slider_row(scope, container, "Sharpen", "", sliders.sharpen, 0.0, 2.0, 0.05, 2);

    let dof_toggle = ToggleRow {
        label: "DoF".to_string(),
        enabled: Some(ui.dof_enabled),
    };
    container.append_child(&dof_toggle.render(scope, &[]));

    build_synced_slider_row(scope, container, "Focus Distance", "", sliders.dof_focus_dist, 0.1, 50.0, 0.1, 1);
    build_synced_slider_row(scope, container, "Focus Range", "", sliders.dof_focus_range, 0.1, 20.0, 0.1, 1);
    build_synced_slider_row(scope, container, "Max CoC", "px", sliders.dof_max_coc, 1.0, 32.0, 1.0, 0);
    build_synced_slider_row(scope, container, "Motion Blur", "", sliders.motion_blur, 0.0, 3.0, 0.1, 1);
    build_synced_slider_row(scope, container, "God Rays", "", sliders.god_rays, 0.0, 2.0, 0.05, 2);
    build_synced_slider_row(scope, container, "Vignette", "", sliders.vignette, 0.0, 1.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Grain", "", sliders.grain, 0.0, 1.0, 0.01, 2);
    build_synced_slider_row(scope, container, "Chromatic Ab.", "", sliders.chromatic_ab, 0.0, 1.0, 0.01, 2);

    // ── Animation controls ──
    append_divider(scope, container);

    let anim_hdr = scope.create_element("div");
    anim_hdr.set_attribute("style", SECTION_STYLE);
    anim_hdr.append_child(&scope.create_text("Animation"));
    container.append_child(&anim_hdr);

    // Play / Pause / Stop buttons.
    let btn_row = scope.create_element("div");
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
        let btn = scope.create_element("div");
        let state_val = match state {
            crate::animation_preview::PlaybackState::Stopped => 0u32,
            crate::animation_preview::PlaybackState::Playing => 1,
            crate::animation_preview::PlaybackState::Paused => 2,
        };
        let is_active = snap_guard.animation_state == state_val;
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
        btn.append_child(&scope.create_text(label));

        let hid = scope.register_handler({
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
    container.append_child(&btn_row);

    let anim_speed_signal: Signal<f64> = Signal::new(snap_guard.animation_speed as f64);

    build_slider_row(
        scope,
        container,
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
}

// ── Object properties ────────────────────────────────────────────────────────

fn build_object_properties(
    scope: &mut RenderScope,
    container: &NodeHandle,
    eid: u64,
    snap_guard: &crate::ui_snapshot::UiSnapshot,
    sliders: &SliderSignals,
    cmd: &CommandSender,
    es: &Arc<Mutex<EditorState>>,
) {
    let obj_info = snap_guard.objects.iter().find(|o| o.id == eid).map(|o| {
        let name = o.name.clone();
        let child_count = snap_guard
            .objects
            .iter()
            .filter(|c| c.parent_id.map(|p| p as u64) == Some(eid))
            .count();
        let has_xf = true;
        let show_revoxelize = es
            .lock()
            .ok()
            .map(|es_lock| {
                es_lock
                    .world
                    .scene()
                    .objects
                    .iter()
                    .find(|obj| obj.id as u64 == eid)
                    .map(|obj| {
                        let is_vox = matches!(
                            obj.root_node.sdf_source,
                            rkf_core::scene_node::SdfSource::Voxelized { .. }
                        );
                        let non_uniform = (obj.scale - glam::Vec3::ONE).length() > 1e-4;
                        is_vox && non_uniform
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        (name, child_count, has_xf, show_revoxelize)
    });

    if let Some((name, child_count, _has_xf, show_revoxelize)) = obj_info {
        // Name.
        let name_row = scope.create_element("div");
        name_row.set_attribute("style", SECTION_STYLE);
        name_row.append_child(&scope.create_text(&name));
        container.append_child(&name_row);

        // Entity ID.
        let id_row = scope.create_element("div");
        id_row.set_attribute("style", VALUE_STYLE);
        id_row.append_child(&scope.create_text(&format!("Entity ID: {eid}")));
        container.append_child(&id_row);

        if child_count > 0 {
            let cr = scope.create_element("div");
            cr.set_attribute("style", VALUE_STYLE);
            cr.append_child(&scope.create_text(&format!("Children: {child_count}")));
            container.append_child(&cr);
        }

        // Transform editor.
        append_divider(scope, container);

        let xf_hdr = scope.create_element("div");
        xf_hdr.set_attribute("style", LABEL_STYLE);
        xf_hdr.append_child(&scope.create_text("Transform"));
        container.append_child(&xf_hdr);

        let transform = TransformEditor {
            pos_x: sliders.obj_pos_x,
            pos_y: sliders.obj_pos_y,
            pos_z: sliders.obj_pos_z,
            rot_x: sliders.obj_rot_x,
            rot_y: sliders.obj_rot_y,
            rot_z: sliders.obj_rot_z,
            scale_x: sliders.obj_scale_x,
            scale_y: sliders.obj_scale_y,
            scale_z: sliders.obj_scale_z,
        };
        let xf_node = rinch::core::untracked(|| transform.render(scope, &[]));
        container.append_child(&xf_node);

        // Re-voxelize button.
        if show_revoxelize {
            let btn_row = scope.create_element("div");
            btn_row.set_attribute("style", "padding: 6px 8px;");
            let btn = scope.create_element("button");
            btn.set_attribute(
                "style",
                "width:100%; padding:4px 8px; background:#553322; \
                 color:#ffcc99; border:1px solid #774433; \
                 border-radius:3px; cursor:pointer; font-size:12px;",
            );
            btn.append_child(&scope.create_text("Re-voxelize (bake scale)"));
            let hid = scope.register_handler({
                let cmd = cmd.clone();
                move || {
                    let _ = cmd.0.send(EditorCommand::Revoxelize {
                        object_id: eid as u32,
                    });
                }
            });
            btn.set_attribute("data-rid", &hid.to_string());
            btn_row.append_child(&btn);
            container.append_child(&btn_row);
        }
    }
}

// ── Light properties ─────────────────────────────────────────────────────────

fn build_light_properties(
    scope: &mut RenderScope,
    container: &NodeHandle,
    lid: u64,
    snap_guard: &crate::ui_snapshot::UiSnapshot,
    sliders: &SliderSignals,
) {
    let light_data = snap_guard.lights.iter().find(|l| l.id == lid);
    if let Some(light) = light_data {
        let type_name = match light.light_type {
            crate::light_editor::SceneLightType::Point => "Point Light",
            crate::light_editor::SceneLightType::Spot => "Spot Light",
        };
        let hdr = scope.create_element("div");
        hdr.set_attribute("style", SECTION_STYLE);
        hdr.append_child(&scope.create_text(type_name));
        container.append_child(&hdr);

        // Position — Vec3Editor with colored axis labels.
        let pos_label = scope.create_element("div");
        pos_label.set_attribute("style", LABEL_STYLE);
        pos_label.append_child(&scope.create_text("Position"));
        container.append_child(&pos_label);

        let pos_editor = Vec3Editor {
            x: sliders.light_pos_x,
            y: sliders.light_pos_y,
            z: sliders.light_pos_z,
            step: 0.01,
            min: -500.0,
            max: 500.0,
            decimals: 2,
            suffix: String::new(),
        };
        let pos_row = scope.create_element("div");
        pos_row.set_attribute("style", "padding: 2px 12px;");
        let pos_node = rinch::core::untracked(|| pos_editor.render(scope, &[]));
        pos_row.append_child(&pos_node);
        container.append_child(&pos_row);

        append_divider(scope, container);

        // Intensity — DragValue.
        let int_label_row = scope.create_element("div");
        int_label_row.set_attribute(
            "style",
            "display:flex;align-items:center;padding:2px 12px;gap:8px;",
        );
        let int_label = scope.create_element("span");
        int_label.set_attribute(
            "style",
            "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;",
        );
        int_label.append_child(&scope.create_text("Intensity"));
        int_label_row.append_child(&int_label);

        let int_dv = DragValue {
            value: sliders.light_intensity,
            step: 0.1,
            min: 0.0,
            max: 50.0,
            decimals: 1,
            ..Default::default()
        };
        let int_node = rinch::core::untracked(|| int_dv.render(scope, &[]));
        int_label_row.append_child(&int_node);
        container.append_child(&int_label_row);

        // Range — DragValue.
        let range_label_row = scope.create_element("div");
        range_label_row.set_attribute(
            "style",
            "display:flex;align-items:center;padding:2px 12px;gap:8px;",
        );
        let range_label = scope.create_element("span");
        range_label.set_attribute(
            "style",
            "font-size:11px;color:var(--rinch-color-dimmed);min-width:56px;",
        );
        range_label.append_child(&scope.create_text("Range"));
        range_label_row.append_child(&range_label);

        let range_dv = DragValue {
            value: sliders.light_range,
            step: 0.5,
            min: 0.1,
            max: 100.0,
            decimals: 1,
            suffix: "m".to_string(),
            ..Default::default()
        };
        let range_node = rinch::core::untracked(|| range_dv.render(scope, &[]));
        range_label_row.append_child(&range_node);
        container.append_child(&range_label_row);
    }
}
