//! Material properties panel — shows name, category, shader (with drag-drop),
//! albedo RGB, roughness, metallic, emission, subsurface, opacity, IOR, noise.

use rinch::prelude::*;

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

    let container = __scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    let materials = ui.materials.get();

    let mat = materials.iter().find(|m| m.slot == slot);
    let mat = match mat {
        Some(m) => m,
        None => return container,
    };

    // Header.
    let hdr = __scope.create_element("div");
    hdr.set_attribute("style", LABEL_STYLE);
    hdr.append_child(&__scope.create_text("Material Properties"));
    container.append_child(&hdr);

    // Name + slot.
    let name_row = __scope.create_element("div");
    name_row.set_attribute("style", SECTION_STYLE);
    name_row.append_child(&__scope.create_text(&format!("{} (#{slot})", mat.name)));
    container.append_child(&name_row);

    if !mat.category.is_empty() {
        let cat_row = __scope.create_element("div");
        cat_row.set_attribute("style", VALUE_STYLE);
        cat_row.append_child(&__scope.create_text(&format!("Category: {}", mat.category)));
        container.append_child(&cat_row);
    }

    // ── Shader display (drop target for shader drag-and-drop) ──
    {
        let shader_row = __scope.create_element("div");
        shader_row.set_attribute(
            "style",
            "display:flex;align-items:center;justify-content:space-between;\
             padding:4px 12px;border:2px solid transparent;",
        );

        // Surgical highlight — updates only the row's border style,
        // never rebuilds the panel.
        {
            let row = shader_row.clone();
            let ui = ui;
            let trigger = __scope.create_element("div");
            trigger.set_attribute("style", "display:none;");
            rinch::core::reactive_component_dom(__scope, &trigger, move |__scope| {
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
                __scope.create_text("")
            });
            shader_row.append_child(&trigger);
        }

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

        container.append_child(&shader_row);
    }

    append_divider(__scope, &container);

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
    let albedo_label = __scope.create_element("div");
    albedo_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    albedo_label.append_child(&__scope.create_text("Albedo"));
    container.append_child(&albedo_label);

    // Color preview swatch.
    let swatch = __scope.create_element("div");
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
        __scope, &container, "R", "", mat.albedo[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[0] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "G", "", mat.albedo[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[1] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "B", "", mat.albedo[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.albedo[2] = v, cmd.clone(), mat,
    );

    build_mat_slider(
        __scope, &container, "Roughness", "", mat.roughness as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.roughness = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "Metallic", "", mat.metallic as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.metallic = v, cmd.clone(), mat,
    );

    append_divider(__scope, &container);

    // Emission.
    let em_label = __scope.create_element("div");
    em_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    em_label.append_child(&__scope.create_text("Emission"));
    container.append_child(&em_label);

    build_mat_slider(
        __scope, &container, "Strength", "", mat.emission_strength as f64, 0.0, 20.0, 0.1, 1,
        |m, v| m.emission_strength = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "Em. R", "", mat.emission_color[0] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[0] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "Em. G", "", mat.emission_color[1] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[1] = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "Em. B", "", mat.emission_color[2] as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.emission_color[2] = v, cmd.clone(), mat,
    );

    append_divider(__scope, &container);

    // Subsurface.
    build_mat_slider(
        __scope, &container, "Subsurface", "", mat.subsurface as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.subsurface = v, cmd.clone(), mat,
    );

    // Opacity + IOR.
    build_mat_slider(
        __scope, &container, "Opacity", "", mat.opacity as f64, 0.0, 1.0, 0.01, 2,
        |m, v| m.opacity = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "IOR", "", mat.ior as f64, 1.0, 3.0, 0.01, 2,
        |m, v| m.ior = v, cmd.clone(), mat,
    );

    append_divider(__scope, &container);

    // Noise.
    let noise_label = __scope.create_element("div");
    noise_label.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-dimmed);padding:3px 12px;",
    );
    noise_label.append_child(&__scope.create_text("Noise"));
    container.append_child(&noise_label);

    build_mat_slider(
        __scope, &container, "Scale", "", mat.noise_scale as f64, 0.0, 50.0, 0.1, 1,
        |m, v| m.noise_scale = v, cmd.clone(), mat,
    );
    build_mat_slider(
        __scope, &container, "Noise Str", "", mat.noise_strength as f64, 0.0, 2.0, 0.01, 2,
        |m, v| m.noise_strength = v, cmd.clone(), mat,
    );

    container
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

fn append_divider(scope: &mut RenderScope, container: &NodeHandle) {
    let div = scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    container.append_child(&div);
}
