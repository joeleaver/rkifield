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

    // NOTE: Selection-sync and batch-sync Effects live in editor_ui() (ui/mod.rs),
    // NOT here. Effects must not be created inside render paths (reactive_component_dom)
    // because Effect::new runs immediately, and .set() during render causes re-entrant
    // RefCell borrows → panic.

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

                // Show active material name from library.
                {
                    let paint_mat_id = if let Ok(es_lock) = es.lock() {
                        es_lock.paint.current_settings.material_id
                    } else {
                        0
                    };
                    let mat_name = snap_guard
                        .materials
                        .iter()
                        .find(|m| m.slot == paint_mat_id)
                        .map(|m| m.name.as_str())
                        .unwrap_or("Unknown");
                    let mat_row = __scope.create_element("div");
                    mat_row.set_attribute("style", VALUE_STYLE);
                    mat_row.append_child(
                        &__scope.create_text(&format!("Material: {mat_name} (#{paint_mat_id})")),
                    );
                    container.append_child(&mat_row);
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

                append_divider(__scope, &container);
            }
            EditorMode::Default => {}
        }

        // ── Properties tab bar ──
        let active_tab = ui.properties_tab.get();
        let selected_mat_slot = ui.selected_material.get();
        let tab_bar = __scope.create_element("div");
        tab_bar.set_attribute(
            "style",
            "display:flex;gap:0;border-bottom:1px solid var(--rinch-color-border);\
             margin-bottom:4px;flex-shrink:0;",
        );
        for (idx, label) in [(0u32, "Object"), (1u32, "Asset")] {
            let tab = __scope.create_element("div");
            let is_active = active_tab == idx;
            let bg = if is_active { "var(--rinch-color-dark-7)" } else { "transparent" };
            let color = if is_active { "var(--rinch-color-text)" } else { "var(--rinch-color-dimmed)" };
            let border_bottom = if is_active { "2px solid var(--rinch-primary-color)" } else { "2px solid transparent" };
            tab.set_attribute(
                "style",
                &format!(
                    "flex:1;text-align:center;padding:4px 0;font-size:11px;\
                     cursor:pointer;background:{bg};color:{color};\
                     border-bottom:{border_bottom};text-transform:uppercase;\
                     letter-spacing:0.5px;"
                ),
            );
            tab.append_child(&__scope.create_text(label));
            let hid = __scope.register_handler({
                let ui = ui;
                move || { ui.properties_tab.set(idx); }
            });
            tab.set_attribute("data-rid", &hid.to_string());
            tab_bar.append_child(&tab);
        }
        container.append_child(&tab_bar);

        if active_tab == 0 {
            // ── Object tab: scene entity properties ──
            if let Some(SelectedEntity::Camera) = selected_entity {
                build_camera_properties(
                    __scope, &container, &snap_guard, &sliders, &ui, &cmd2, &es,
                );
            } else {
                match selected_entity {
                    Some(SelectedEntity::Object(eid)) => {
                        build_object_properties(
                            __scope, &container, eid, &snap_guard, &sliders, &cmd2, &es, &ui,
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
        } else {
            // ── Asset tab: material or shader properties ──
            let selected_shader_name = ui.selected_shader.get();
            if let Some(slot) = selected_mat_slot {
                build_material_properties(
                    __scope, &container, slot, &snap_guard, &cmd2, &ui,
                );
            } else if let Some(ref shader_name) = selected_shader_name {
                // Show shader properties.
                if let Some(shader) = snap_guard.shaders.iter().find(|s| &s.name == shader_name) {
                    build_shader_properties(__scope, &container, shader);
                }
            } else {
                let msg = __scope.create_element("div");
                msg.set_attribute(
                    "style",
                    &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
                );
                msg.append_child(&__scope.create_text("No asset selected"));
                container.append_child(&msg);
            }
        }

        container
    });

    root
}

// ── Standalone panel components ──────────────────────────────────────────────
//
// These thin wrappers expose individual sections of the right panel as
// independent `#[component]`s for the zone-based layout system.
// They reuse the same helper functions that RightPanel calls internally.

/// Object Properties panel — shows selected object/light/camera properties.
///
/// When the camera is selected, also shows environment/post-processing controls.
/// When in Sculpt/Paint mode, shows brush settings at the top.
#[component]
pub fn PropertiesPanel() -> NodeHandle {
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();
    let sliders = use_context::<SliderSignals>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex-direction:column;");

    let es = editor_state.clone();
    let snap = snapshot.clone();
    let cmd2 = cmd.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
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
                            let _ = cmd.0.send(EditorCommand::FixSdfs {
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

                {
                    let paint_mat_id = if let Ok(es_lock) = es.lock() {
                        es_lock.paint.current_settings.material_id
                    } else {
                        0
                    };
                    let mat_name = snap_guard
                        .materials
                        .iter()
                        .find(|m| m.slot == paint_mat_id)
                        .map(|m| m.name.as_str())
                        .unwrap_or("Unknown");
                    let mat_row = __scope.create_element("div");
                    mat_row.set_attribute("style", VALUE_STYLE);
                    mat_row.append_child(
                        &__scope.create_text(&format!("Material: {mat_name} (#{paint_mat_id})")),
                    );
                    container.append_child(&mat_row);
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

                append_divider(__scope, &container);
            }
            EditorMode::Default => {}
        }

        // ── Entity properties ──
        if let Some(SelectedEntity::Camera) = selected_entity {
            build_camera_properties(
                __scope, &container, &snap_guard, &sliders, &ui, &cmd2, &es,
            );
        } else {
            match selected_entity {
                Some(SelectedEntity::Object(eid)) => {
                    build_object_properties(
                        __scope, &container, eid, &snap_guard, &sliders, &cmd2, &es, &ui,
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

/// Asset Properties panel — shows material or shader properties for the selected asset.
#[component]
pub fn AssetPropertiesPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let snapshot = use_context::<SnapshotReader>();
    let cmd = use_context::<CommandSender>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex-direction:column;");

    let snap = snapshot.clone();
    let cmd2 = cmd.clone();
    rinch::core::reactive_component_dom(__scope, &root, move |__scope| {
        let _ = ui.selected_material.get();
        let _ = ui.selected_shader.get();
        let _ = ui.material_revision.get();

        let snap_guard = snap.0.load();
        let selected_mat_slot = ui.selected_material.get();
        let selected_shader_name = ui.selected_shader.get();

        let container = __scope.create_element("div");
        container.set_attribute("style", "display:flex;flex-direction:column;");

        if let Some(slot) = selected_mat_slot {
            build_material_properties(
                __scope, &container, slot, &snap_guard, &cmd2, &ui,
            );
        } else if let Some(ref shader_name) = selected_shader_name {
            if let Some(shader) = snap_guard.shaders.iter().find(|s| &s.name == shader_name) {
                build_shader_properties(__scope, &container, shader);
            }
        } else {
            let msg = __scope.create_element("div");
            msg.set_attribute(
                "style",
                &format!("{SECTION_STYLE}color:var(--rinch-color-placeholder);"),
            );
            msg.append_child(&__scope.create_text("Select a material or shader"));
            container.append_child(&msg);
        }

        container
    });

    root
}

/// Sculpt settings panel — brush type, radius, strength, falloff.
#[component]
pub fn SculptPanel() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "padding:4px 0;");

    let header = __scope.create_element("div");
    header.set_attribute("style", super::LABEL_STYLE);
    header.append_child(&__scope.create_text("Sculpt Brush"));
    root.append_child(&header);

    build_synced_slider_row(
        __scope, &root, "Radius", "",
        sliders.brush_radius, 0.01, 10.0, 0.01, 2,
    );
    build_synced_slider_row(
        __scope, &root, "Strength", "",
        sliders.brush_strength, 0.0, 1.0, 0.01, 2,
    );
    build_synced_slider_row(
        __scope, &root, "Falloff", "",
        sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
    );

    root.into()
}

/// Paint settings panel — material, radius, strength, falloff.
#[component]
pub fn PaintPanel() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();

    let root = __scope.create_element("div");
    root.set_attribute("style", "padding:4px 0;");

    let header = __scope.create_element("div");
    header.set_attribute("style", super::LABEL_STYLE);
    header.append_child(&__scope.create_text("Paint Brush"));
    root.append_child(&header);

    build_synced_slider_row(
        __scope, &root, "Radius", "",
        sliders.brush_radius, 0.01, 10.0, 0.01, 2,
    );
    build_synced_slider_row(
        __scope, &root, "Strength", "",
        sliders.brush_strength, 0.0, 1.0, 0.01, 2,
    );
    build_synced_slider_row(
        __scope, &root, "Falloff", "",
        sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
    );

    root.into()
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
    ui: &UiSignals,
) {
    let obj_info = snap_guard.objects.iter().find(|o| o.id == eid).map(|o| {
        let name = o.name.clone();
        let child_count = snap_guard
            .objects
            .iter()
            .filter(|c| c.parent_id.map(|p| p as u64) == Some(eid))
            .count();
        let has_xf = true;
        let (show_revoxelize, is_voxelized, is_analytical) = es
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
                        let is_anal = matches!(
                            obj.root_node.sdf_source,
                            rkf_core::scene_node::SdfSource::Analytical { .. }
                        );
                        let non_uniform = (obj.scale - glam::Vec3::ONE).length() > 1e-4;
                        (is_vox && non_uniform, is_vox, is_anal)
                    })
                    .unwrap_or((false, false, false))
            })
            .unwrap_or((false, false, false));
        (name, child_count, has_xf, show_revoxelize, is_voxelized, is_analytical)
    });

    if let Some((name, child_count, _has_xf, show_revoxelize, is_voxelized, is_analytical)) = obj_info {
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

        // Convert to Voxel Object button (analytical primitives only).
        if is_analytical {
            let btn_row = scope.create_element("div");
            btn_row.set_attribute("style", "padding: 6px 8px;");
            let btn = scope.create_element("button");
            btn.set_attribute(
                "style",
                "width:100%; padding:4px 8px; background:#223355; \
                 color:#99ccff; border:1px solid #334477; \
                 border-radius:3px; cursor:pointer; font-size:12px;",
            );
            btn.append_child(&scope.create_text("Convert to Voxel Object"));
            let hid = scope.register_handler({
                let cmd = cmd.clone();
                move || {
                    let _ = cmd.0.send(EditorCommand::ConvertToVoxel {
                        object_id: eid as u32,
                    });
                }
            });
            btn.set_attribute("data-rid", &hid.to_string());
            btn_row.append_child(&btn);
            container.append_child(&btn_row);
        }

        // ── Materials section (voxelized objects only) ──
        if is_voxelized {
            append_divider(scope, container);

            let mat_hdr = scope.create_element("div");
            mat_hdr.set_attribute("style", LABEL_STYLE);
            mat_hdr.append_child(&scope.create_text("Materials"));
            container.append_child(&mat_hdr);

            for usage in &snap_guard.selected_object_materials {
                let mat_info = snap_guard.materials.iter().find(|m| m.slot == usage.material_id);
                let mat_name = mat_info.map(|m| m.name.as_str()).unwrap_or("Unknown");
                let (r, g, b) = mat_info
                    .map(|m| {
                        (
                            (m.albedo[0] * 255.0).round() as u8,
                            (m.albedo[1] * 255.0).round() as u8,
                            (m.albedo[2] * 255.0).round() as u8,
                        )
                    })
                    .unwrap_or((128, 128, 128));

                let from_mat = std::rc::Rc::new(std::cell::Cell::new(usage.material_id));
                let count_str = if usage.voxel_count >= 1_000_000 {
                    format!("{:.1}M", usage.voxel_count as f64 / 1_000_000.0)
                } else if usage.voxel_count >= 1_000 {
                    format!("{:.1}K", usage.voxel_count as f64 / 1_000.0)
                } else {
                    format!("{}", usage.voxel_count)
                };

                let row = scope.create_element("div");
                row.set_attribute(
                    "style",
                    "display:flex;align-items:center;gap:6px;padding:2px 12px;\
                     min-height:22px;border:2px solid transparent;",
                );

                // Surgical highlight Effect — only updates this row's border,
                // never rebuilds the panel.
                {
                    let row_ref = row.clone();
                    let ui = *ui;
                    let from_mat = from_mat.clone();
                    Effect::new(move || {
                        let highlighted = ui.material_drop_highlight.get() == Some(from_mat.get());
                        let border = if highlighted {
                            "border:2px dashed var(--rinch-primary-color);border-radius:4px;"
                        } else {
                            "border:2px solid transparent;"
                        };
                        row_ref.set_attribute(
                            "style",
                            &format!(
                                "display:flex;align-items:center;gap:6px;padding:2px 12px;\
                                 min-height:22px;{border}"
                            ),
                        );
                    });
                }

                // Albedo swatch.
                let swatch = scope.create_element("div");
                swatch.set_attribute(
                    "style",
                    &format!(
                        "width:12px;height:12px;border-radius:50%;\
                         background:rgb({r},{g},{b});flex-shrink:0;\
                         border:1px solid rgba(255,255,255,0.15);"
                    ),
                );
                row.append_child(&swatch);

                // Material name.
                let name_el = scope.create_element("span");
                name_el.set_attribute(
                    "style",
                    "font-size:11px;color:var(--rinch-color-text);\
                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;",
                );
                name_el.append_child(&scope.create_text(mat_name));
                row.append_child(&name_el);

                // Voxel count.
                let count_el = scope.create_element("span");
                count_el.set_attribute(
                    "style",
                    "font-size:10px;color:var(--rinch-color-placeholder);\
                     font-family:var(--rinch-font-family-monospace);flex-shrink:0;",
                );
                count_el.append_child(&scope.create_text(&count_str));
                row.append_child(&count_el);

                // Drop target handlers — same pattern as shader drag-and-drop.
                let drop_hid = scope.register_handler({
                    let ui = *ui;
                    let cmd = cmd.clone();
                    let name_el = name_el.clone();
                    let swatch = swatch.clone();
                    let from_mat = from_mat.clone();
                    let snap = snap_guard.materials.iter()
                        .map(|m| (m.slot, m.name.clone(), m.albedo))
                        .collect::<Vec<_>>();
                    move || {
                        ui.material_drop_highlight.set(None);
                        let current_from = from_mat.get();
                        if let Some(to_mat) = ui.material_drag.take() {
                            if to_mat != current_from {
                                // Optimistically update name and swatch immediately.
                                if let Some((_, name, albedo)) = snap.iter().find(|(s, _, _)| *s == to_mat) {
                                    name_el.set_text(name);
                                    let nr = (albedo[0] * 255.0).round() as u8;
                                    let ng = (albedo[1] * 255.0).round() as u8;
                                    let nb = (albedo[2] * 255.0).round() as u8;
                                    swatch.set_attribute(
                                        "style",
                                        &format!(
                                            "width:12px;height:12px;border-radius:50%;\
                                             background:rgb({nr},{ng},{nb});flex-shrink:0;\
                                             border:1px solid rgba(255,255,255,0.15);"
                                        ),
                                    );
                                }
                                let _ = cmd.0.send(EditorCommand::RemapMaterial {
                                    object_id: eid,
                                    from_material: current_from,
                                    to_material: to_mat,
                                });
                                // Update the cell so subsequent drops use the new material.
                                from_mat.set(to_mat);
                            }
                        }
                    }
                });
                row.set_attribute("data-ondrop", &drop_hid.to_string());

                let enter_hid = scope.register_handler({
                    let ui = *ui;
                    let from_mat = from_mat.clone();
                    move || {
                        if ui.material_drag.is_active() {
                            ui.material_drop_highlight.set(Some(from_mat.get()));
                        }
                    }
                });
                row.set_attribute("data-ondragenter", &enter_hid.to_string());

                let leave_hid = scope.register_handler({
                    let ui = *ui;
                    move || {
                        ui.material_drop_highlight.set(None);
                    }
                });
                row.set_attribute("data-ondragleave", &leave_hid.to_string());

                container.append_child(&row);
            }
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

// ── Material properties ─────────────────────────────────────────────────────

fn build_material_properties(
    scope: &mut RenderScope,
    container: &NodeHandle,
    slot: u16,
    snap_guard: &crate::ui_snapshot::UiSnapshot,
    cmd: &CommandSender,
    ui: &UiSignals,
) {
    use super::slider_helpers::build_slider_row;

    let mat = snap_guard.materials.iter().find(|m| m.slot == slot);
    let mat = match mat {
        Some(m) => m,
        None => return,
    };

    // Header.
    let hdr = scope.create_element("div");
    hdr.set_attribute("style", LABEL_STYLE);
    hdr.append_child(&scope.create_text("Material Properties"));
    container.append_child(&hdr);

    // Name + slot.
    let name_row = scope.create_element("div");
    name_row.set_attribute("style", SECTION_STYLE);
    name_row.append_child(&scope.create_text(&format!("{} (#{slot})", mat.name)));
    container.append_child(&name_row);

    if !mat.category.is_empty() {
        let cat_row = scope.create_element("div");
        cat_row.set_attribute("style", VALUE_STYLE);
        cat_row.append_child(&scope.create_text(&format!("Category: {}", mat.category)));
        container.append_child(&cat_row);
    }

    // ── Shader display (drop target for shader drag-and-drop) ──
    {
        let shader_row = scope.create_element("div");
        shader_row.set_attribute(
            "style",
            "display:flex;align-items:center;justify-content:space-between;\
             padding:4px 12px;border:2px solid transparent;",
        );

        // Surgical highlight Effect — updates only the row's border style,
        // never rebuilds the panel.
        {
            let row = shader_row.clone();
            let ui = *ui;
            Effect::new(move || {
                let highlighted = ui.shader_drop_highlight.get();
                let border = if highlighted {
                    "border:2px dashed var(--rinch-primary-color);border-radius:4px;"
                } else {
                    "border:2px solid transparent;"
                };
                row.set_attribute(
                    "style",
                    &format!(
                        "display:flex;align-items:center;justify-content:space-between;\
                         padding:4px 12px;{border}"
                    ),
                );
            });
        }

        let shader_label = scope.create_element("span");
        shader_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
        shader_label.append_child(&scope.create_text("Shader"));
        shader_row.append_child(&shader_label);

        let shader_value = scope.create_element("span");
        shader_value.set_attribute("style", "font-size:11px;color:var(--rinch-color-text);font-weight:600;");
        shader_value.append_child(&scope.create_text(&mat.shader_name));
        shader_row.append_child(&shader_value);

        // Drop target handlers for shader drag-and-drop.
        let drop_hid = scope.register_handler({
            let ui = *ui;
            let cmd = cmd.clone();
            let shader_value = shader_value.clone();
            move || {
                ui.shader_drop_highlight.set(false);
                if let Some(shader_name) = ui.shader_drag.take() {
                    // Optimistically update the displayed text immediately.
                    shader_value.set_text(&shader_name);
                    let _ = cmd.0.send(EditorCommand::SetMaterialShader {
                        slot,
                        shader_name,
                    });
                }
            }
        });
        shader_row.set_attribute("data-ondrop", &drop_hid.to_string());

        let enter_hid = scope.register_handler({
            let ui = *ui;
            move || {
                if ui.shader_drag.is_active() {
                    ui.shader_drop_highlight.set(true);
                }
            }
        });
        shader_row.set_attribute("data-ondragenter", &enter_hid.to_string());

        let leave_hid = scope.register_handler({
            let ui = *ui;
            move || {
                ui.shader_drop_highlight.set(false);
            }
        });
        shader_row.set_attribute("data-ondragleave", &leave_hid.to_string());

        container.append_child(&shader_row);
    }

    append_divider(scope, container);

    // ── PBR Sliders ──
    // Each slider creates a local signal initialized from snapshot data,
    // with an on_update callback that sends SetMaterial command.

    // Helper: build a material slider that sends the full material on change.
    let build_mat_slider = |scope: &mut RenderScope,
                            container: &NodeHandle,
                            label: &str,
                            suffix: &str,
                            initial: f64,
                            min: f64,
                            max: f64,
                            step: f64,
                            decimals: usize,
                            field_setter: fn(&mut rkf_core::material::Material, f32),
                            cmd: CommandSender,
                            mat_snapshot: &crate::ui_snapshot::MaterialSummary| {
        let sig: Signal<f64> = Signal::new(initial);
        let mat_copy = mat_snapshot.clone();
        build_slider_row(
            scope, container, label, suffix, sig, min, max, step, decimals,
            move |v| {
                let mut m = snapshot_to_material(&mat_copy);
                field_setter(&mut m, v as f32);
                let _ = cmd.0.send(EditorCommand::SetMaterial { slot, material: m });
            },
        );
    };

    // Albedo R/G/B.
    let albedo_label = scope.create_element("div");
    albedo_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    albedo_label.append_child(&scope.create_text("Albedo"));
    container.append_child(&albedo_label);

    // Color preview swatch.
    let swatch = scope.create_element("div");
    let r = (mat.albedo[0] * 255.0).round() as u8;
    let g = (mat.albedo[1] * 255.0).round() as u8;
    let b = (mat.albedo[2] * 255.0).round() as u8;
    swatch.set_attribute(
        "style",
        &format!(
            "width:calc(100% - 24px);height:16px;margin:0 12px 4px;\
             border-radius:3px;background:rgb({r},{g},{b});\
             border:1px solid var(--rinch-color-border);"
        ),
    );
    container.append_child(&swatch);

    build_mat_slider(
        scope, container, "R", "", mat.albedo[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[0] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "G", "", mat.albedo[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[1] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "B", "", mat.albedo[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[2] = v, cmd.clone(), mat,
    );

    build_mat_slider(
        scope, container, "Roughness", "", mat.roughness as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.roughness = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "Metallic", "", mat.metallic as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.metallic = v, cmd.clone(), mat,
    );

    append_divider(scope, container);

    // Emission.
    let em_label = scope.create_element("div");
    em_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    em_label.append_child(&scope.create_text("Emission"));
    container.append_child(&em_label);

    build_mat_slider(
        scope, container, "Strength", "", mat.emission_strength as f64, 0.0, 20.0, 0.1, 1,
        |m, v| m.emission_strength = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "Em. R", "", mat.emission_color[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[0] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "Em. G", "", mat.emission_color[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[1] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "Em. B", "", mat.emission_color[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[2] = v, cmd.clone(), mat,
    );

    append_divider(scope, container);

    // Subsurface.
    build_mat_slider(
        scope, container, "Subsurface", "", mat.subsurface as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.subsurface = v, cmd.clone(), mat,
    );

    // Opacity + IOR.
    build_mat_slider(
        scope, container, "Opacity", "", mat.opacity as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.opacity = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "IOR", "", mat.ior as f64, 1.0, 3.0, 0.01, 2,
        |m, v| m.ior = v, cmd.clone(), mat,
    );

    append_divider(scope, container);

    // Noise.
    let noise_label = scope.create_element("div");
    noise_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    noise_label.append_child(&scope.create_text("Noise"));
    container.append_child(&noise_label);

    build_mat_slider(
        scope, container, "Scale", "", mat.noise_scale as f64, 0.0, 50.0, 0.1, 1,
        |m, v| m.noise_scale = v, cmd.clone(), mat,
    );
    build_mat_slider(
        scope, container, "Noise Str", "", mat.noise_strength as f64, 0.0, 2.0, 0.01, 2,
        |m, v| m.noise_strength = v, cmd.clone(), mat,
    );
}

/// Build shader properties panel (read-only info display).
fn build_shader_properties(
    scope: &mut RenderScope,
    container: &NodeHandle,
    shader: &crate::ui_snapshot::ShaderSummary,
) {
    // Header.
    let hdr = scope.create_element("div");
    hdr.set_attribute("style", SECTION_STYLE);
    hdr.append_child(&scope.create_text(&shader.name));
    container.append_child(&hdr);

    // Type badge.
    let type_row = scope.create_element("div");
    type_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
    let type_label = scope.create_element("span");
    type_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
    type_label.append_child(&scope.create_text("Type"));
    type_row.append_child(&type_label);
    let type_value = scope.create_element("span");
    let (badge_text, badge_bg) = if shader.built_in {
        ("built-in", "rgba(60,120,200,0.3)")
    } else {
        ("custom", "rgba(60,180,100,0.3)")
    };
    type_value.set_attribute(
        "style",
        &format!(
            "font-size:10px;color:var(--rinch-color-text);padding:1px 6px;\
             border-radius:3px;background:{badge_bg};"
        ),
    );
    type_value.append_child(&scope.create_text(badge_text));
    type_row.append_child(&type_value);
    container.append_child(&type_row);

    // Shader ID.
    let id_row = scope.create_element("div");
    id_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
    let id_label = scope.create_element("span");
    id_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
    id_label.append_child(&scope.create_text("ID"));
    id_row.append_child(&id_label);
    let id_value = scope.create_element("span");
    id_value.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-text);\
         font-family:var(--rinch-font-family-monospace);",
    );
    id_value.append_child(&scope.create_text(&format!("{}", shader.id)));
    id_row.append_child(&id_value);
    container.append_child(&id_row);

    // File path.
    if !shader.file_path.is_empty() {
        let path_row = scope.create_element("div");
        path_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
        let path_label = scope.create_element("span");
        path_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
        path_label.append_child(&scope.create_text("File"));
        path_row.append_child(&path_label);
        let path_value = scope.create_element("span");
        path_value.set_attribute(
            "style",
            "font-size:9px;color:var(--rinch-color-placeholder);\
             font-family:var(--rinch-font-family-monospace);\
             word-break:break-all;text-align:right;max-width:160px;",
        );
        path_value.append_child(&scope.create_text(&shader.file_path));
        path_row.append_child(&path_value);
        container.append_child(&path_row);
    }

    append_divider(scope, container);
}

/// Reconstruct a GPU `Material` from a `MaterialSummary` snapshot.
fn snapshot_to_material(s: &crate::ui_snapshot::MaterialSummary) -> rkf_core::material::Material {
    rkf_core::material::Material {
        albedo: s.albedo,
        roughness: s.roughness,
        metallic: s.metallic,
        emission_color: s.emission_color,
        emission_strength: s.emission_strength,
        subsurface: s.subsurface,
        subsurface_color: s.subsurface_color,
        opacity: s.opacity,
        ior: s.ior,
        noise_scale: s.noise_scale,
        noise_strength: s.noise_strength,
        noise_channels: s.noise_channels,
        shader_id: 0,
        _padding: [0.0; 5],
    }
}
