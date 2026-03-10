//! Object properties panel — shows name, entity ID, transform editor,
//! convert-to-voxel button, and material usage for the selected object.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorState, SliderSignals, UiSignals};
use crate::CommandSender;

use super::components::TransformEditor;
use super::{DIVIDER_STYLE, LABEL_STYLE, SECTION_STYLE, VALUE_STYLE};

/// Object properties panel — shows name, entity ID, transform editor,
/// convert-to-voxel button, and material usage for the selected object.
#[component]
pub fn ObjectProperties(
    entity_id: u64,
) -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    let objects = ui.objects.get();
    let eid = entity_id;

    let obj_info = objects.iter().find(|o| o.id == eid).map(|o| {
        let name = o.name.clone();
        let child_count = objects
            .iter()
            .filter(|c| c.parent_id.map(|p| p as u64) == Some(eid))
            .count();
        let (is_voxelized, is_analytical) = editor_state
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
                        (is_vox, is_anal)
                    })
                    .unwrap_or((false, false))
            })
            .unwrap_or((false, false));
        (name, child_count, is_voxelized, is_analytical)
    });

    let Some((name, child_count, is_voxelized, is_analytical)) = obj_info else {
        return rsx! { div { style: "display:flex;flex-direction:column;" } };
    };

    // Build transform editor imperatively (component struct).
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
    let xf_node = rinch::core::untracked(|| transform.render(__scope, &[]));

    // Build material rows imperatively — they use drag-drop handlers and
    // optimistic DOM updates that require element references.
    let materials_section = __scope.create_element("div");
    if is_voxelized {
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
            let count_str = if usage.voxel_count >= 1_000_000 {
                format!("{:.1}M", usage.voxel_count as f64 / 1_000_000.0)
            } else if usage.voxel_count >= 1_000 {
                format!("{:.1}K", usage.voxel_count as f64 / 1_000.0)
            } else {
                format!("{}", usage.voxel_count)
            };

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

            // Voxel count.
            let count_el = __scope.create_element("span");
            count_el.set_attribute(
                "style",
                "font-size:10px;color:var(--rinch-color-placeholder);\
                 font-family:var(--rinch-font-family-monospace);flex-shrink:0;",
            );
            count_el.append_child(&__scope.create_text(&count_str));
            row.append_child(&count_el);

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
                            let _ = cmd.0.send(EditorCommand::RemapMaterial {
                                object_id: eid,
                                from_material: current_from,
                                to_material: to_mat,
                            });
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
            div { style: {VALUE_STYLE}, {format!("Entity ID: {eid}")} }

            // Children count (conditional).
            if child_count > 0 {
                div { style: {VALUE_STYLE}, {format!("Children: {child_count}")} }
            }

            // Transform editor.
            div { style: {DIVIDER_STYLE} }
            div { style: {LABEL_STYLE}, "Transform" }
            {xf_node}

            // Convert to Voxel Object button (analytical primitives only).
            if is_analytical {
                div { style: "padding: 6px 8px;",
                    button {
                        style: "width:100%; padding:4px 8px; background:#223355; \
                               color:#99ccff; border:1px solid #334477; \
                               border-radius:3px; cursor:pointer; font-size:12px;",
                        onclick: {
                            let cmd = cmd.clone();
                            move || {
                                let _ = cmd.0.send(EditorCommand::ConvertToVoxel {
                                    object_id: eid as u32,
                                });
                            }
                        },
                        "Convert to Voxel Object"
                    }
                }
            }

            // Materials section (voxelized objects only — empty div if not voxelized).
            {materials_section}
        }
    }
}
