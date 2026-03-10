//! Material properties panel — shows name, category, shader (with drag-drop),
//! material preview thumbnail, albedo RGB, roughness, metallic, emission,
//! subsurface, opacity, IOR, noise.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::automation::SharedState;
use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

use super::slider_helpers::build_slider_row;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE, VALUE_STYLE};

/// Material properties panel — editable PBR properties with shader drag-drop.
#[component]
pub fn MaterialProperties(
    slot: u16,
) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    let shared_state = use_context::<Arc<Mutex<SharedState>>>();

    let materials = ui.materials.get();

    let mat = materials.iter().find(|m| m.slot == slot);
    let mat = match mat {
        Some(m) => m,
        None => {
            return rsx! { div { style: "display:flex;flex-direction:column;" } };
        }
    };

    // ── Primitive selector buttons (imperative — for loop over tuples) ──
    let current_prim = ui.preview_primitive_type.get();
    let prim_bar = __scope.create_element("div");
    prim_bar.set_attribute(
        "style",
        "display:flex;gap:2px;justify-content:center;",
    );
    for (idx, label) in [
        (0u32, "Sphere"), (1, "Box"), (2, "Capsule"),
        (3, "Torus"), (4, "Cylinder"), (5, "Plane"),
    ] {
        let btn = __scope.create_element("div");
        let is_active = current_prim == idx;
        let bg = if is_active {
            "var(--rinch-primary-color)"
        } else {
            "var(--rinch-color-dark-5)"
        };
        let color = if is_active {
            "#fff"
        } else {
            "var(--rinch-color-dimmed)"
        };
        btn.set_attribute(
            "style",
            &format!(
                "padding:2px 6px;font-size:9px;border-radius:3px;\
                 cursor:pointer;background:{bg};color:{color};\
                 user-select:none;"
            ),
        );
        btn.append_child(&__scope.create_text(label));
        let hid = __scope.register_handler({
            let ui = ui;
            let ss = shared_state.clone();
            move || {
                ui.preview_primitive_type.set(idx);
                if let Ok(mut ss) = ss.lock() {
                    ss.preview_primitive_type = idx;
                    ss.preview_dirty = true;
                }
            }
        });
        btn.set_attribute("data-rid", &hid.to_string());
        prim_bar.append_child(&btn);
    }

    // ── Shader display row (imperative — optimistic DOM update on drop) ──
    let shader_row = rsx! {
        div {
            style: {
                let ui = ui;
                move || {
                    let highlighted = ui.shader_drop_highlight.get();
                    let border = if highlighted {
                        "border:2px dashed var(--rinch-primary-color);border-radius:4px;"
                    } else {
                        "border:2px solid transparent;"
                    };
                    format!(
                        "display:flex;align-items:center;justify-content:space-between;\
                         padding:4px 12px;{border}"
                    )
                }
            },
        }
    };

    let shader_label = __scope.create_element("span");
    shader_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
    shader_label.append_child(&__scope.create_text("Shader"));
    shader_row.append_child(&shader_label);

    let shader_value = __scope.create_element("span");
    shader_value.set_attribute("style", "font-size:11px;color:var(--rinch-color-text);font-weight:600;");
    shader_value.append_child(&__scope.create_text(&mat.shader_name));
    shader_row.append_child(&shader_value);

    // Drop target handlers for shader drag-and-drop.
    let drop_hid = __scope.register_handler({
        let ui = ui;
        let cmd = cmd.clone();
        let shader_value = shader_value.clone();
        move || {
            ui.shader_drop_highlight.set(false);
            if let Some(shader_name) = ui.shader_drag.take() {
                shader_value.set_text(&shader_name);
                let _ = cmd.0.send(EditorCommand::SetMaterialShader {
                    slot,
                    shader_name,
                });
            }
        }
    });
    shader_row.set_attribute("data-ondrop", &drop_hid.to_string());

    let enter_hid = __scope.register_handler({
        let ui = ui;
        move || {
            if ui.shader_drag.is_active() {
                ui.shader_drop_highlight.set(true);
            }
        }
    });
    shader_row.set_attribute("data-ondragenter", &enter_hid.to_string());

    let leave_hid = __scope.register_handler({
        let ui = ui;
        move || {
            ui.shader_drop_highlight.set(false);
        }
    });
    shader_row.set_attribute("data-ondragleave", &leave_hid.to_string());

    // ── PBR Sliders ──
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
                            mat_snapshot: &crate::ui_snapshot::MaterialSummary,
                            ss: Arc<Mutex<SharedState>>| {
        let sig: Signal<f64> = Signal::new(initial);
        let mat_copy = mat_snapshot.clone();
        build_slider_row(
            scope, container, label, suffix, sig, min, max, step, decimals,
            move |v| {
                let mut m = snapshot_to_material(&mat_copy);
                field_setter(&mut m, v as f32);
                let _ = cmd.0.send(EditorCommand::SetMaterial { slot, material: m });
                if let Ok(mut ss) = ss.lock() {
                    ss.preview_dirty = true;
                }
            },
        );
    };

    // Build slider containers imperatively (build_slider_row appends to a container).
    let albedo_sliders = __scope.create_element("div");
    build_mat_slider(
        __scope, &albedo_sliders, "R", "", mat.albedo[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[0] = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &albedo_sliders, "G", "", mat.albedo[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[1] = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &albedo_sliders, "B", "", mat.albedo[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[2] = v, cmd.clone(), mat, shared_state.clone(),
    );

    let pbr_sliders = __scope.create_element("div");
    build_mat_slider(
        __scope, &pbr_sliders, "Roughness", "", mat.roughness as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.roughness = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &pbr_sliders, "Metallic", "", mat.metallic as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.metallic = v, cmd.clone(), mat, shared_state.clone(),
    );

    let emission_sliders = __scope.create_element("div");
    build_mat_slider(
        __scope, &emission_sliders, "Strength", "", mat.emission_strength as f64, 0.0, 20.0, 0.1, 1,
        |m, v| m.emission_strength = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &emission_sliders, "Em. R", "", mat.emission_color[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[0] = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &emission_sliders, "Em. G", "", mat.emission_color[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[1] = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &emission_sliders, "Em. B", "", mat.emission_color[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[2] = v, cmd.clone(), mat, shared_state.clone(),
    );

    let misc_sliders = __scope.create_element("div");
    build_mat_slider(
        __scope, &misc_sliders, "Subsurface", "", mat.subsurface as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.subsurface = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &misc_sliders, "Opacity", "", mat.opacity as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.opacity = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &misc_sliders, "IOR", "", mat.ior as f64, 1.0, 3.0, 0.01, 2,
        |m, v| m.ior = v, cmd.clone(), mat, shared_state.clone(),
    );

    let noise_sliders = __scope.create_element("div");
    build_mat_slider(
        __scope, &noise_sliders, "Scale", "", mat.noise_scale as f64, 0.0, 50.0, 0.1, 1,
        |m, v| m.noise_scale = v, cmd.clone(), mat, shared_state.clone(),
    );
    build_mat_slider(
        __scope, &noise_sliders, "Noise Str", "", mat.noise_strength as f64, 0.0, 2.0, 0.01, 2,
        |m, v| m.noise_strength = v, cmd.clone(), mat, shared_state.clone(),
    );

    let mat_name_display = format!("{} (#{slot})", mat.name);
    let category = mat.category.clone();
    let has_category = !category.is_empty();
    let r = (mat.albedo[0] * 255.0).round() as u8;
    let g = (mat.albedo[1] * 255.0).round() as u8;
    let b = (mat.albedo[2] * 255.0).round() as u8;
    let swatch_style = format!(
        "width:calc(100% - 24px);height:16px;margin:0 12px 4px;\
         border-radius:3px;background:rgb({r},{g},{b});\
         border:1px solid var(--rinch-color-border);"
    );

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            // Header.
            div { style: {LABEL_STYLE}, "Material Properties" }

            // Primitive selector.
            div { style: "padding:4px 12px;display:flex;flex-direction:column;gap:4px;",
                {prim_bar}
            }

            // Name + slot.
            div { style: {SECTION_STYLE}, {mat_name_display} }

            // Category (conditional).
            if has_category {
                div { style: {VALUE_STYLE}, {format!("Category: {category}")} }
            }

            // Shader display (drop target).
            {shader_row}

            div { style: {DIVIDER_STYLE} }

            // Albedo.
            div { style: "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
                "Albedo"
            }
            div { style: {swatch_style.as_str()} }
            {albedo_sliders}
            {pbr_sliders}

            div { style: {DIVIDER_STYLE} }

            // Emission.
            div { style: "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
                "Emission"
            }
            {emission_sliders}

            div { style: {DIVIDER_STYLE} }

            // Subsurface, Opacity, IOR.
            {misc_sliders}

            div { style: {DIVIDER_STYLE} }

            // Noise.
            div { style: "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
                "Noise"
            }
            {noise_sliders}
        }
    }
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
