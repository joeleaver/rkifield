//! Component inspector panel — shows editable component fields for the selected entity.
//!
//! Reads from `UiSignals::inspector_data` (pushed by the engine thread) and
//! sends `EditorCommand` variants for field edits, component add/remove.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{ComponentSnapshot, FieldSnapshot, UiSignals};
use crate::CommandSender;

use super::{DIVIDER_STYLE, LABEL_STYLE};

// ── Style constants ──────────────────────────────────────────────────────

const FIELD_ROW_STYLE: &str = "display:flex;align-items:center;\
    justify-content:space-between;padding:2px 12px;min-height:22px;gap:8px;";

const FIELD_LABEL_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);\
    white-space:nowrap;flex-shrink:0;min-width:70px;";

const FIELD_VALUE_STYLE: &str = "font-size:11px;color:var(--rinch-color-text);\
    font-family:var(--rinch-font-family-monospace);\
    flex:1;text-align:right;overflow:hidden;text-overflow:ellipsis;\
    white-space:nowrap;";

const REMOVE_BTN_STYLE: &str = "font-size:10px;color:var(--rinch-color-dimmed);\
    cursor:pointer;padding:0 4px;border:none;background:none;\
    line-height:1;";

const ADD_BTN_STYLE: &str = "width:100%;padding:4px 8px;background:rgba(255,255,255,0.06);\
    color:var(--rinch-color-dimmed);border:1px solid var(--rinch-color-border);\
    border-radius:3px;cursor:pointer;font-size:11px;text-align:left;";

const DROPDOWN_STYLE: &str = "position:absolute;left:12px;right:12px;\
    background:var(--rinch-color-dark-7);border:1px solid var(--rinch-color-border);\
    border-radius:4px;z-index:100;max-height:200px;overflow-y:auto;\
    box-shadow:0 4px 12px rgba(0,0,0,0.4);";

const DROPDOWN_ITEM_STYLE: &str = "padding:4px 8px;font-size:11px;\
    color:var(--rinch-color-text);cursor:pointer;";

const DROPDOWN_ITEM_HOVER: &str = "padding:4px 8px;font-size:11px;\
    color:var(--rinch-color-text);cursor:pointer;\
    background:var(--rinch-color-dark-5);";

// ── Component Inspector ──────────────────────────────────────────────────

/// Component inspector panel. Renders all components and fields for the
/// selected entity, with editable inputs and add/remove controls.
///
/// Designed to be appended below ObjectProperties in the right panel.
/// When the game plugin dylib is not yet loaded (`dylib_ready == false`),
/// shows a "Building scripts..." placeholder instead of the component list.
#[component]
pub fn ComponentInspector(entity_id: uuid::Uuid) -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();
    let eid = entity_id;

    // Signal to control add-component dropdown visibility.
    let dropdown_open = Signal::new(false);

    // Build container imperatively — the content reacts to inspector_data signal.
    let container = __scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    // Divider before the component section.
    let divider = __scope.create_element("div");
    divider.set_attribute("style", DIVIDER_STYLE);
    container.append_child(&divider);

    // Section header.
    let header = __scope.create_element("div");
    header.set_attribute("style", LABEL_STYLE);
    header.append_child(&__scope.create_text("Components"));
    container.append_child(&header);

    // "Building scripts..." placeholder — shown when dylib not ready.
    let building_msg = __scope.create_element("div");
    building_msg.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-placeholder);padding:8px 12px;",
    );
    building_msg.append_child(&__scope.create_text("Building scripts..."));
    {
        let msg = building_msg.clone();
        __scope.create_effect(move || {
            if ui.dylib_ready.get() {
                msg.set_attribute("style", "display:none;");
            } else {
                msg.set_attribute(
                    "style",
                    "font-size:11px;color:var(--rinch-color-placeholder);padding:8px 12px;",
                );
            }
        });
    }
    container.append_child(&building_msg);

    // Content wrapper — hidden when dylib not ready.
    let content = __scope.create_element("div");
    {
        let c = content.clone();
        __scope.create_effect(move || {
            if ui.dylib_ready.get() {
                c.set_attribute("style", "display:flex;flex-direction:column;");
            } else {
                c.set_attribute("style", "display:none;");
            }
        });
    }

    // Reactive component list — rebuilds when inspector_data changes.
    let components_container = __scope.create_element("div");
    components_container.set_attribute("style", "display:flex;flex-direction:column;");
    content.append_child(&components_container.clone());

    let cmd_for_list = cmd.clone();
    rinch::core::for_each_dom_typed(
        __scope,
        &components_container,
        move || {
            let data = ui.inspector_data.get();
            match data {
                Some(ref snap) if snap.entity_id == eid => snap.components.clone(),
                _ => Vec::new(),
            }
        },
        |comp| comp.name.clone(),
        move |comp, scope| build_component_section(scope, &comp, eid, cmd_for_list.clone()),
    );

    // Add Component section.
    let add_section = __scope.create_element("div");
    add_section.set_attribute("style", "padding:6px 12px;position:relative;");

    // Add Component button.
    let add_btn = __scope.create_element("button");
    add_btn.set_attribute("style", ADD_BTN_STYLE);
    add_btn.append_child(&__scope.create_text("+ Add Component"));
    let toggle_hid = __scope.register_handler(move || {
        dropdown_open.update(|v| *v = !*v);
    });
    add_btn.set_attribute("data-rid", &toggle_hid.to_string());
    add_section.append_child(&add_btn);

    // Dropdown list (reactive visibility).
    let dropdown = __scope.create_element("div");
    {
        let dropdown_clone = dropdown.clone();
        __scope.create_effect(move || {
            if dropdown_open.get() {
                dropdown_clone.set_attribute("style", DROPDOWN_STYLE);
            } else {
                dropdown_clone.set_attribute("style", &format!("{DROPDOWN_STYLE}display:none;"));
            }
        });
    }
    add_section.append_child(&dropdown.clone());

    // Populate dropdown with available components (reactive).
    let cmd_for_dropdown = cmd.clone();
    rinch::core::for_each_dom_typed(
        __scope,
        &dropdown,
        move || ui.available_components.get(),
        |name| name.clone(),
        move |comp_name, scope| {
            let item = scope.create_element("div");
            item.set_attribute("style", DROPDOWN_ITEM_STYLE);
            item.append_child(&scope.create_text(&comp_name));

            let name_for_click = comp_name.clone();
            let cmd = cmd_for_dropdown.clone();
            let hid = scope.register_handler(move || {
                let _ = cmd.0.send(EditorCommand::AddComponent {
                    entity_id: eid,
                    component_name: name_for_click.clone(),
                });
                dropdown_open.set(false);
            });
            item.set_attribute("data-rid", &hid.to_string());

            // Hover highlight.
            let item_for_enter = item.clone();
            let enter_hid = scope.register_handler(move || {
                item_for_enter.set_attribute("style", DROPDOWN_ITEM_HOVER);
            });
            item.set_attribute("data-onmouseenter", &enter_hid.to_string());

            let item_for_leave = item.clone();
            let leave_hid = scope.register_handler(move || {
                item_for_leave.set_attribute("style", DROPDOWN_ITEM_STYLE);
            });
            item.set_attribute("data-onmouseleave", &leave_hid.to_string());

            item
        },
    );

    content.append_child(&add_section);
    container.append_child(&content);

    container
}

// ── Component section builder ────────────────────────────────────────────

/// Build a collapsible section for a single component.
fn build_component_section(
    scope: &mut RenderScope,
    comp: &ComponentSnapshot,
    entity_id: uuid::Uuid,
    cmd: CommandSender,
) -> NodeHandle {
    let section = scope.create_element("div");
    section.set_attribute(
        "style",
        "display:flex;flex-direction:column;margin-bottom:2px;",
    );

    // Component header row.
    let collapsed = Signal::new(false);
    let header = scope.create_element("div");
    header.set_attribute(
        "style",
        "display:flex;align-items:center;padding:4px 12px;cursor:pointer;\
         user-select:none;gap:4px;",
    );

    // Collapse triangle.
    let triangle = scope.create_element("span");
    {
        let tri = triangle.clone();
        scope.create_effect(move || {
            if collapsed.get() {
                tri.set_attribute(
                    "style",
                    "font-size:8px;color:var(--rinch-color-dimmed);",
                );
                tri.set_text("\u{25B6}"); // right-pointing triangle
            } else {
                tri.set_attribute(
                    "style",
                    "font-size:8px;color:var(--rinch-color-dimmed);\
                     transform:rotate(90deg);display:inline-block;",
                );
                tri.set_text("\u{25B6}");
            }
        });
    }
    header.append_child(&triangle);

    // Component name.
    let name_span = scope.create_element("span");
    name_span.set_attribute(
        "style",
        "font-size:12px;font-weight:600;color:var(--rinch-color-text);flex:1;",
    );
    name_span.append_child(&scope.create_text(&comp.name));
    header.append_child(&name_span);

    // Toggle collapse on header click.
    let toggle_hid = scope.register_handler(move || {
        collapsed.update(|v| *v = !*v);
    });
    header.set_attribute("data-rid", &toggle_hid.to_string());

    // Remove button (only for removable components).
    if comp.removable {
        let remove_btn = scope.create_element("span");
        remove_btn.set_attribute("style", REMOVE_BTN_STYLE);
        remove_btn.append_child(&scope.create_text("\u{2715}")); // ×

        let comp_name = comp.name.clone();
        let cmd_clone = cmd.clone();
        let remove_hid = scope.register_handler(move || {
            let _ = cmd_clone.0.send(EditorCommand::RemoveComponent {
                entity_id,
                component_name: comp_name.clone(),
            });
        });
        remove_btn.set_attribute("data-rid", &remove_hid.to_string());
        header.append_child(&remove_btn);
    }

    section.append_child(&header);

    // Fields container (hidden when collapsed).
    let fields_container = scope.create_element("div");
    {
        let fc = fields_container.clone();
        scope.create_effect(move || {
            if collapsed.get() {
                fc.set_attribute("style", "display:none;");
            } else {
                fc.set_attribute("style", "display:flex;flex-direction:column;");
            }
        });
    }

    for field in &comp.fields {
        let row = build_field_row(scope, field, entity_id, &comp.name, cmd.clone());
        fields_container.append_child(&row);
    }

    section.append_child(&fields_container);
    section
}

// ── Field row builder ────────────────────────────────────────────────────

/// Build a single field row with an appropriate editor for the field type.
fn build_field_row(
    scope: &mut RenderScope,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) -> NodeHandle {
    use rkf_runtime::behavior::registry::FieldType;

    let row = scope.create_element("div");
    row.set_attribute("style", FIELD_ROW_STYLE);

    // Field label.
    let label = scope.create_element("span");
    label.set_attribute("style", FIELD_LABEL_STYLE);
    label.append_child(&scope.create_text(&field.name));
    row.append_child(&label);

    match field.field_type {
        FieldType::Float => {
            build_float_editor(scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Int => {
            build_int_editor(scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Bool => {
            build_bool_editor(scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Vec3 => {
            build_vec3_editor(scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::String => {
            build_string_editor(scope, &row, field, entity_id, component_name, cmd);
        }
        _ => {
            // Read-only display for unsupported types.
            let val_span = scope.create_element("span");
            val_span.set_attribute("style", FIELD_VALUE_STYLE);
            val_span.append_child(&scope.create_text(&field.display_value));
            row.append_child(&val_span);
        }
    }

    // Transient badge.
    if field.transient {
        let badge = scope.create_element("span");
        badge.set_attribute(
            "style",
            "font-size:9px;color:var(--rinch-color-placeholder);\
             border:1px solid var(--rinch-color-border);border-radius:2px;\
             padding:0 3px;flex-shrink:0;",
        );
        badge.append_child(&scope.create_text("rt"));
        row.append_child(&badge);
    }

    row
}

// ── Float editor ──────────────────────────────────────────────────────────

fn build_float_editor(
    scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field.float_value.unwrap_or(0.0);

    if let Some((min, max)) = field.range {
        // Slider for ranged values.
        let signal = Signal::new(current_val);
        let comp_name = component_name.to_string();
        let field_name = field.name.clone();

        let slider = Slider {
            min: Some(min),
            max: Some(max),
            step: Some((max - min) / 200.0),
            value_signal: Some(signal),
            size: "xs".to_string(),
            onchange: Some(ValueCallback::new(move |v: f64| {
                signal.set(v);
                let _ = cmd.0.send(EditorCommand::SetComponentField {
                    entity_id,
                    component_name: comp_name.clone(),
                    field_name: field_name.clone(),
                    value: rkf_runtime::behavior::game_value::GameValue::Float(v),
                });
            })),
            ..Default::default()
        };
        let slider_container = scope.create_element("div");
        slider_container.set_attribute("style", "flex:1;min-width:60px;");
        let slider_node = rinch::core::untracked(|| slider.render(scope, &[]));
        slider_container.append_child(&slider_node);
        row.append_child(&slider_container);

        // Value readout.
        let val_text = scope.create_text("");
        {
            let val_text = val_text.clone();
            scope.create_effect(move || {
                let v = signal.get();
                val_text.set_text(&format!("{v:.2}"));
            });
        }
        let val_span = scope.create_element("span");
        val_span.set_attribute(
            "style",
            "font-size:10px;color:var(--rinch-color-dimmed);\
             font-family:var(--rinch-font-family-monospace);\
             min-width:40px;text-align:right;flex-shrink:0;",
        );
        val_span.append_child(&val_text);
        row.append_child(&val_span);
    } else {
        // DragValue for unranged floats.
        let signal = Signal::new(current_val);
        let comp_name = component_name.to_string();
        let field_name = field.name.clone();

        let mut dv = super::components::DragValue::from_signal(signal, Some(ValueCallback::new(move |v: f64| {
            let _ = cmd.0.send(EditorCommand::SetComponentField {
                entity_id,
                component_name: comp_name.clone(),
                field_name: field_name.clone(),
                value: rkf_runtime::behavior::game_value::GameValue::Float(v),
            });
        })));
        dv.step = 0.01;
        dv.decimals = 3;
        let dv_container = scope.create_element("div");
        dv_container.set_attribute("style", "flex:1;min-width:60px;");
        dv_container.append_child(&dv.render(scope, &[]));
        row.append_child(&dv_container);
    }
}

// ── Int editor ────────────────────────────────────────────────────────────

fn build_int_editor(
    scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field.int_value.unwrap_or(0) as f64;
    let signal = Signal::new(current_val);
    let comp_name = component_name.to_string();
    let field_name = field.name.clone();

    if let Some((min, max)) = field.range {
        let slider = Slider {
            min: Some(min),
            max: Some(max),
            step: Some(1.0),
            value_signal: Some(signal),
            size: "xs".to_string(),
            onchange: Some(ValueCallback::new(move |v: f64| {
                let iv = v.round() as i64;
                signal.set(iv as f64);
                let _ = cmd.0.send(EditorCommand::SetComponentField {
                    entity_id,
                    component_name: comp_name.clone(),
                    field_name: field_name.clone(),
                    value: rkf_runtime::behavior::game_value::GameValue::Int(iv),
                });
            })),
            ..Default::default()
        };
        let slider_container = scope.create_element("div");
        slider_container.set_attribute("style", "flex:1;min-width:60px;");
        let slider_node = rinch::core::untracked(|| slider.render(scope, &[]));
        slider_container.append_child(&slider_node);
        row.append_child(&slider_container);
    } else {
        let mut dv = super::components::DragValue::from_signal(signal, Some(ValueCallback::new(move |v: f64| {
            let iv = v.round() as i64;
            let _ = cmd.0.send(EditorCommand::SetComponentField {
                entity_id,
                component_name: comp_name.clone(),
                field_name: field_name.clone(),
                value: rkf_runtime::behavior::game_value::GameValue::Int(iv),
            });
        })));
        dv.step = 1.0;
        dv.decimals = 0;
        let dv_container = scope.create_element("div");
        dv_container.set_attribute("style", "flex:1;min-width:60px;");
        dv_container.append_child(&dv.render(scope, &[]));
        row.append_child(&dv_container);
    }
}

// ── Bool editor ───────────────────────────────────────────────────────────

fn build_bool_editor(
    scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field.bool_value.unwrap_or(false);
    let comp_name = component_name.to_string();
    let field_name = field.name.clone();

    let checkbox = scope.create_element("div");
    checkbox.set_attribute(
        "style",
        &format!(
            "width:16px;height:16px;border-radius:3px;cursor:pointer;\
             border:1px solid var(--rinch-color-border);\
             background:{};flex-shrink:0;",
            if current_val {
                "var(--rinch-primary-color)"
            } else {
                "rgba(255,255,255,0.06)"
            }
        ),
    );

    // Checkmark text.
    if current_val {
        let check = scope.create_element("span");
        check.set_attribute(
            "style",
            "color:#fff;font-size:11px;display:flex;\
             align-items:center;justify-content:center;height:100%;",
        );
        check.append_child(&scope.create_text("\u{2713}"));
        checkbox.append_child(&check);
    }

    let cb_clone = checkbox.clone();
    let toggle_hid = scope.register_handler(move || {
        let new_val = !current_val;
        let _ = cmd.0.send(EditorCommand::SetComponentField {
            entity_id,
            component_name: comp_name.clone(),
            field_name: field_name.clone(),
            value: rkf_runtime::behavior::game_value::GameValue::Bool(new_val),
        });
        // Optimistic update.
        let bg = if new_val {
            "var(--rinch-primary-color)"
        } else {
            "rgba(255,255,255,0.06)"
        };
        cb_clone.set_attribute(
            "style",
            &format!(
                "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                 border:1px solid var(--rinch-color-border);\
                 background:{bg};flex-shrink:0;"
            ),
        );
    });
    checkbox.set_attribute("data-rid", &toggle_hid.to_string());

    // Spacer to push checkbox to the right.
    let spacer = scope.create_element("div");
    spacer.set_attribute("style", "flex:1;");
    row.append_child(&spacer);
    row.append_child(&checkbox);
}

// ── Vec3 editor ──────────────────────────────────────────────────────────

fn build_vec3_editor(
    scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let v = field.vec3_value.unwrap_or(glam::Vec3::ZERO);
    let sx = Signal::new(v.x as f64);
    let sy = Signal::new(v.y as f64);
    let sz = Signal::new(v.z as f64);

    let comp_name = component_name.to_string();
    let field_name = field.name.clone();

    let on_change = ValueCallback::new(move |v: [f64; 3]| {
        sx.set(v[0]);
        sy.set(v[1]);
        sz.set(v[2]);
        let new_vec = glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
        let _ = cmd.0.send(EditorCommand::SetComponentField {
            entity_id,
            component_name: comp_name.clone(),
            field_name: field_name.clone(),
            value: rkf_runtime::behavior::game_value::GameValue::Vec3(new_vec),
        });
    });

    let editor = super::components::Vec3Editor {
        x: Memo::new(move || sx.get()),
        y: Memo::new(move || sy.get()),
        z: Memo::new(move || sz.get()),
        on_change: Some(on_change),
        on_commit: None,
        step: 0.01,
        min: -1e9,
        max: 1e9,
        decimals: 2,
        suffix: String::new(),
    };

    let editor_container = scope.create_element("div");
    editor_container.set_attribute("style", "flex:1;");
    editor_container.append_child(&editor.render(scope, &[]));
    row.append_child(&editor_container);
}

// ── String editor ────────────────────────────────────────────────────────

fn build_string_editor(
    scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field
        .string_value
        .as_deref()
        .unwrap_or("")
        .to_string();
    let comp_name = component_name.to_string();
    let field_name = field.name.clone();

    let input = scope.create_element("input");
    input.set_attribute("type", "text");
    input.set_attribute("value", &current_val);
    input.set_attribute(
        "style",
        "flex:1;background:rgba(255,255,255,0.06);border:1px solid var(--rinch-color-border);\
         border-radius:3px;padding:1px 4px;font-size:11px;color:var(--rinch-color-text);\
         outline:none;min-width:40px;height:18px;font-family:var(--rinch-font-family-monospace);",
    );

    let input_hid = scope.register_input_handler(move |text: String| {
        let _ = cmd.0.send(EditorCommand::SetComponentField {
            entity_id,
            component_name: comp_name.clone(),
            field_name: field_name.clone(),
            value: rkf_runtime::behavior::game_value::GameValue::String(text),
        });
    });
    input.set_attribute("data-oninput", &input_hid.to_string());

    row.append_child(&input);
}
