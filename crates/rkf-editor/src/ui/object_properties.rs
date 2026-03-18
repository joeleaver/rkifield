//! Object properties panel — shows name, entity ID, transform editor,
//! convert-to-voxel button, and material usage for the selected object.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

use super::components::TransformEditor;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE, VALUE_STYLE};

/// Object properties panel — shows name, entity ID, transform editor,
/// convert-to-voxel button, and material usage for the selected object.
#[component]
pub fn ObjectProperties() -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let eid = match ui.selection.get() {
        Some(crate::editor_state::SelectedEntity::Object(id)) => id,
        _ => return rsx! { div {} },
    };

    let objects = ui.objects.get();

    let obj_info = objects.iter().find(|o| o.id == eid).map(|o| {
        let name = o.name.clone();
        let child_count = objects
            .iter()
            .filter(|c| c.parent_id == Some(eid))
            .count();
        let is_voxelized = o.object_type == crate::ui_snapshot::ObjectType::Voxelized;
        let is_analytical = o.object_type == crate::ui_snapshot::ObjectType::Analytical;
        (name, child_count, is_voxelized, is_analytical)
    });

    let Some((name, child_count, is_voxelized, is_analytical)) = obj_info else {
        return rsx! { div { style: "display:flex;flex-direction:column;" } };
    };

    // Build transform editor — reads from UiSignals, writes via EditorCommands.
    let transform = TransformEditor { entity_id: eid };
    let xf_node = rinch::core::untracked(|| transform.render(__scope, &[]));

    // Build material rows imperatively — they use drag-drop handlers and
    // optimistic DOM updates that require element references.
    let materials_section = __scope.create_element("div");
    if is_voxelized || is_analytical {
        // Add divider + label inside the materials section container.
        let divider = __scope.create_element("div");
        divider.set_attribute("style", DIVIDER_STYLE);
        materials_section.append_child(&divider);
        let mat_hdr = __scope.create_element("div");
        mat_hdr.set_attribute("style", LABEL_STYLE);
        mat_hdr.append_child(&__scope.create_text("Materials"));
        materials_section.append_child(&mat_hdr);

        let section = &materials_section;

        let sel_obj_mats = ui.selected_object_materials.get();
        let all_materials = ui.materials.get();
        for usage in sel_obj_mats.iter() {
            let mat_info = all_materials.iter().find(|m| m.slot == usage.material_id);
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
            let from_mat_for_handler = from_mat.clone();
            let show_voxel_count = is_voxelized;

            let row = rsx! {
                div {
                    style: {
                        let ui = ui;
                        let from_mat = from_mat.clone();
                        move || {
                            let highlighted = ui.material_drop_highlight.get() == Some(from_mat.get());
                            let border = if highlighted {
                                "border:2px dashed var(--rinch-primary-color);border-radius:4px;"
                            } else {
                                "border:2px solid transparent;"
                            };
                            format!(
                                "display:flex;align-items:center;gap:6px;padding:2px 12px;\
                                 min-height:22px;{border}"
                            )
                        }
                    },
                }
            };

            // Albedo swatch.
            let swatch = __scope.create_element("div");
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
            let name_el = __scope.create_element("span");
            name_el.set_attribute(
                "style",
                "font-size:11px;color:var(--rinch-color-text);\
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;",
            );
            name_el.append_child(&__scope.create_text(mat_name));
            row.append_child(&name_el);

            // Voxel count (voxelized objects only).
            if show_voxel_count {
                let count_str = if usage.voxel_count >= 1_000_000 {
                    format!("{:.1}M", usage.voxel_count as f64 / 1_000_000.0)
                } else if usage.voxel_count >= 1_000 {
                    format!("{:.1}K", usage.voxel_count as f64 / 1_000.0)
                } else {
                    format!("{}", usage.voxel_count)
                };
                let count_el = __scope.create_element("span");
                count_el.set_attribute(
                    "style",
                    "font-size:10px;color:var(--rinch-color-placeholder);\
                     font-family:var(--rinch-font-family-monospace);flex-shrink:0;",
                );
                count_el.append_child(&__scope.create_text(&count_str));
                row.append_child(&count_el);
            }

            // Drop target handlers — optimistic updates require element references.
            let drop_hid = __scope.register_handler({
                let ui = ui;
                let cmd = cmd.clone();
                let name_el = name_el.clone();
                let swatch = swatch.clone();
                let from_mat = from_mat_for_handler.clone();
                let snap = all_materials.iter()
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
                            if is_analytical {
                                let _ = cmd.0.send(EditorCommand::SetPrimitiveMaterial {
                                    object_id: eid,
                                    material_id: to_mat,
                                });
                            } else {
                                let _ = cmd.0.send(EditorCommand::RemapMaterial {
                                    object_id: eid,
                                    from_material: current_from,
                                    to_material: to_mat,
                                });
                            }
                            from_mat.set(to_mat);
                        }
                    }
                }
            });
            row.set_attribute("data-ondrop", &drop_hid.to_string());

            let enter_hid = __scope.register_handler({
                let ui = ui;
                let from_mat = from_mat_for_handler;
                move || {
                    if ui.material_drag.is_active() {
                        ui.material_drop_highlight.set(Some(from_mat.get()));
                    }
                }
            });
            row.set_attribute("data-ondragenter", &enter_hid.to_string());

            let leave_hid = __scope.register_handler({
                let ui = ui;
                move || {
                    ui.material_drop_highlight.set(None);
                }
            });
            row.set_attribute("data-ondragleave", &leave_hid.to_string());

            section.append_child(&row);
        }
    }

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            // Name.
            div { style: {SECTION_STYLE}, {name} }

            // Entity ID.
            div { style: {VALUE_STYLE}, {format!("Entity ID: {}", &eid.to_string()[..8])} }

            // Children count (conditional).
            if child_count > 0 {
                div { style: {VALUE_STYLE}, {format!("Children: {child_count}")} }
            }

            // Transform editor.
            div { style: {DIVIDER_STYLE} }
            div { style: {LABEL_STYLE}, "Transform" }
            {xf_node}

            // Convert to Voxel Object panel (analytical primitives only).
            // Read from signal reactively so the panel disappears immediately
            // when the object is converted to voxelized.
            if ui.objects.get().iter().find(|o| o.id == eid)
                .map(|o| o.object_type == crate::ui_snapshot::ObjectType::Analytical)
                .unwrap_or(false)
            {
                VoxelizePanel { entity_id: eid }
            }

            // Materials section (voxelized objects only — empty div if not voxelized).
            {materials_section}
        }
    }
}

/// Standard resolution tiers with labels for the voxelize UI.
const VOXEL_TIERS: [(f32, &str); 4] = [
    (0.005, "0.5cm — Fine detail"),
    (0.02,  "2cm — Standard"),
    (0.08,  "8cm — Large"),
    (0.32,  "32cm — Terrain"),
];

/// Pick the best default tier index for an object based on its world-space size.
fn default_tier_for_object(primitive: &rkf_core::SdfPrimitive, scale: glam::Vec3) -> usize {
    let scaled_half = crate::engine::primitive_half_extents(primitive) * scale;
    let max_dim = scaled_half.x.max(scaled_half.y).max(scaled_half.z) * 2.0;
    // Heuristic: pick the coarsest tier that gives at least 8 bricks on the longest axis.
    for (i, &(vs, _)) in VOXEL_TIERS.iter().enumerate().rev() {
        let bricks = (max_dim / (vs * 8.0)).ceil();
        if bricks >= 8.0 {
            return i;
        }
    }
    0 // Finest if the object is tiny.
}

fn format_memory(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

/// Expandable panel for converting analytical primitives to voxelized form.
/// Shows tier selection with live estimates before committing.
#[component]
fn VoxelizePanel(entity_id: uuid::Uuid) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    let eid = entity_id;

    let expanded = Signal::new(false);

    // Get the primitive info and scale for estimates.
    let obj_data = ui.objects.get().iter().find(|o| o.id == eid).map(|o| {
        (o.primitive, o.scale)
    });
    let Some((Some(primitive), scale)) = obj_data else {
        return rsx! { div {} };
    };

    let default_tier = default_tier_for_object(&primitive, scale);
    let selected_tier = Signal::new(default_tier);

    rsx! {
        div { style: "padding: 6px 8px;",
            if !expanded.get() {
                button {
                    style: "width:100%; padding:4px 8px; background:#223355; \
                           color:#99ccff; border:1px solid #334477; \
                           border-radius:3px; cursor:pointer; font-size:12px;",
                    onclick: move || expanded.set(true),
                    "Convert to Voxel Object"
                }
            }

            if expanded.get() {
                div { style: "background:#1a1a2e; border:1px solid #334477; \
                             border-radius:4px; padding:8px; display:flex; \
                             flex-direction:column; gap:4px;",
                    div { style: "font-size:11px; color:#99ccff; font-weight:600; \
                                 margin-bottom:2px;",
                        "Voxelize Settings"
                    }

                    // Tier selection buttons.
                    for (i, (_vs, label)) in VOXEL_TIERS.iter().copied().enumerate() {
                        div { key: i,
                            style: {
                                move || {
                                    let sel = selected_tier.get() == i;
                                    format!(
                                        "padding:3px 8px; font-size:10px; cursor:pointer; \
                                         border-radius:3px; border:1px solid {}; \
                                         background:{}; color:{};",
                                        if sel { "#4488cc" } else { "#333344" },
                                        if sel { "#223355" } else { "transparent" },
                                        if sel { "#99ccff" } else { "#8899aa" },
                                    )
                                }
                            },
                            onclick: move || selected_tier.set(i),
                            {label}
                        }
                    }

                    // Estimates.
                    div { style: "font-size:10px; color:#8899aa; font-family:monospace; \
                                 line-height:1.6; margin-top:4px; \
                                 white-space:pre;",
                        {|| {
                            let tier = selected_tier.get();
                            let vs = VOXEL_TIERS[tier].0;
                            let (dims, bricks, mem) =
                                crate::engine_loop_edits::estimate_voxelization(&primitive, scale, vs);
                            format!(
                                "Grid:   {}×{}×{} bricks\n\
                                 Bricks: ~{}\n\
                                 Memory: ~{}",
                                dims.x, dims.y, dims.z,
                                bricks, format_memory(mem),
                            )
                        }}
                    }

                    // Buttons.
                    div { style: "display:flex; gap:6px; margin-top:4px;",
                        button {
                            style: "flex:1; padding:4px 8px; background:#224433; \
                                   color:#88ddaa; border:1px solid #336644; \
                                   border-radius:3px; cursor:pointer; font-size:11px;",
                            onclick: {
                                let cmd = cmd.clone();
                                move || {
                                    let tier = selected_tier.get();
                                    let vs = VOXEL_TIERS[tier].0;
                                    let _ = cmd.0.send(EditorCommand::ConvertToVoxel {
                                        object_id: eid,
                                        voxel_size: vs,
                                    });
                                }
                            },
                            "Convert"
                        }
                        button {
                            style: "flex:1; padding:4px 8px; background:#332222; \
                                   color:#aa8888; border:1px solid #443333; \
                                   border-radius:3px; cursor:pointer; font-size:11px;",
                            onclick: move || expanded.set(false),
                            "Cancel"
                        }
                    }
                }
            }
        }
    }
}
