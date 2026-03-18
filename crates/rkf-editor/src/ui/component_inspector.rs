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
pub fn ComponentInspector() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();
    let eid = match ui.selection.get() {
        Some(crate::editor_state::SelectedEntity::Object(id)) => id,
        _ => return rsx! { div {} },
    };

    // Signal to control add-component dropdown visibility.
    let dropdown_open = Signal::new(false);

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            // Divider before the component section.
            div { style: {DIVIDER_STYLE} }

            // Section header.
            div { style: {LABEL_STYLE}, "Components" }

            // "Building scripts..." placeholder — shown when dylib not ready.
            div {
                style: {|| if ui.dylib_ready.get() {
                    "display:none;"
                } else {
                    "font-size:11px;color:var(--rinch-color-placeholder);padding:8px 12px;"
                }},
                "Building scripts..."
            }

            // Content wrapper — hidden when dylib not ready.
            div {
                style: {|| if ui.dylib_ready.get() {
                    "display:flex;flex-direction:column;"
                } else {
                    "display:none;"
                }},

                // Reactive component list.
                div { style: "display:flex;flex-direction:column;",
                    for comp in get_components(ui, eid) {
                        div {
                            key: comp.name.clone(),
                            {build_component_section(__scope, &comp, eid, cmd.clone())}
                        }
                    }
                }

                // Add Component section.
                div { style: "padding:6px 12px;position:relative;",
                    button {
                        style: {ADD_BTN_STYLE},
                        onclick: move || dropdown_open.update(|v| *v = !*v),
                        "+ Add Component"
                    }

                    // Dropdown list (reactive visibility).
                    div {
                        style: {move || if dropdown_open.get() {
                            DROPDOWN_STYLE.to_string()
                        } else {
                            format!("{DROPDOWN_STYLE}display:none;")
                        }},

                        for comp_name in ui.available_components.get() {
                            DropdownItem {
                                key: comp_name.clone(),
                                comp_name: comp_name,
                                entity_id: eid,
                                dropdown_open: dropdown_open,
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Get the component list for the current entity from inspector data.
fn get_components(ui: UiSignals, eid: uuid::Uuid) -> Vec<ComponentSnapshot> {
    let data = ui.inspector_data.get();
    match data {
        Some(ref snap) if snap.entity_id == eid => snap.components.clone(),
        _ => Vec::new(),
    }
}

/// A single dropdown item for the add-component menu.
#[component]
fn DropdownItem(
    comp_name: String,
    entity_id: uuid::Uuid,
    dropdown_open: Option<Signal<bool>>,
) -> NodeHandle {
    let cmd = use_context::<CommandSender>();
    let hovered = Signal::new(false);
    let dropdown_open = dropdown_open.unwrap_or_else(|| Signal::new(false));
    let name_for_click = comp_name.clone();
    let name_display = comp_name.clone();

    rsx! {
        div {
            style: {move || if hovered.get() { DROPDOWN_ITEM_HOVER } else { DROPDOWN_ITEM_STYLE }},
            onclick: move || {
                eprintln!("\n\n!!! ADD_COMPONENT entity_id={entity_id} component={} !!!\n", name_for_click);
                let _ = cmd.0.send(EditorCommand::AddComponent {
                    entity_id,
                    component_name: name_for_click.clone(),
                });
                dropdown_open.set(false);
            },
            onmouseenter: move || hovered.set(true),
            onmouseleave: move || hovered.set(false),
            {name_display}
        }
    }
}

// ── Component section builder ────────────────────────────────────────────

/// Build a collapsible section for a single component.
fn build_component_section(
    __scope: &mut RenderScope,
    comp: &ComponentSnapshot,
    entity_id: uuid::Uuid,
    cmd: CommandSender,
) -> NodeHandle {
    let collapsed = Signal::new(false);
    let comp_name_str = comp.name.clone();
    let removable = comp.removable;
    let comp_name_display = comp.name.clone();

    // Build header.
    let header = rsx! {
        div {
            style: "display:flex;align-items:center;padding:4px 12px;cursor:pointer;\
                    user-select:none;gap:4px;",
            onclick: move || collapsed.update(|v| *v = !*v),

            // Collapse triangle.
            span {
                style: {|| if collapsed.get() {
                    "font-size:8px;color:var(--rinch-color-dimmed);"
                } else {
                    "font-size:8px;color:var(--rinch-color-dimmed);\
                     transform:rotate(90deg);display:inline-block;"
                }},
                "\u{25B6}"
            }

            // Component name.
            span {
                style: "font-size:12px;font-weight:600;color:var(--rinch-color-text);flex:1;",
                {comp_name_display}
            }
        }
    };

    // Add remove button if removable.
    if removable {
        let cmd_clone = cmd.clone();
        let name_for_remove = comp_name_str.clone();
        let remove_btn = rsx! {
            span {
                style: {REMOVE_BTN_STYLE},
                onclick: move || {
                    let _ = cmd_clone.0.send(EditorCommand::RemoveComponent {
                        entity_id,
                        component_name: name_for_remove.clone(),
                    });
                },
                "\u{2715}"
            }
        };
        header.append_child(&remove_btn);
    }

    // Build fields container.
    let fields_container = rsx! {
        div {
            style: {|| if collapsed.get() {
                "display:none;"
            } else {
                "display:flex;flex-direction:column;"
            }},
        }
    };

    for field in &comp.fields {
        let row = build_field_row(__scope, field, entity_id, &comp_name_str, cmd.clone());
        fields_container.append_child(&row);
    }

    // Assemble section.
    let section = rsx! {
        div { style: "display:flex;flex-direction:column;margin-bottom:2px;" }
    };
    section.append_child(&header);
    section.append_child(&fields_container);
    section
}

// ── Field row builder ────────────────────────────────────────────────────

/// Build a single field row with an appropriate editor for the field type.
fn build_field_row(
    __scope: &mut RenderScope,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) -> NodeHandle {
    use rkf_runtime::behavior::registry::FieldType;

    let row = rsx! {
        div { style: {FIELD_ROW_STYLE},
            span { style: {FIELD_LABEL_STYLE}, {field.name.clone()} }
        }
    };

    match field.field_type {
        FieldType::Float => {
            build_float_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Int => {
            build_int_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Bool => {
            build_bool_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Vec3 => {
            build_vec3_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::String => {
            build_string_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::Struct => {
            return build_struct_editor(
                __scope, field, entity_id, component_name, &field.name, cmd,
            );
        }
        FieldType::AssetRef => {
            build_asset_ref_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        FieldType::ComponentRef => {
            build_component_ref_editor(__scope, &row, field, entity_id, component_name, cmd);
        }
        _ => {
            let val_span = rsx! {
                span { style: {FIELD_VALUE_STYLE}, {field.display_value.clone()} }
            };
            row.append_child(&val_span);
        }
    }

    if field.transient {
        let badge = rsx! {
            span {
                style: "font-size:9px;color:var(--rinch-color-placeholder);\
                        border:1px solid var(--rinch-color-border);border-radius:2px;\
                        padding:0 3px;flex-shrink:0;",
                "rt"
            }
        };
        row.append_child(&badge);
    }

    row
}

// ── Float editor ──────────────────────────────────────────────────────────

fn build_float_editor(
    __scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field.float_value.unwrap_or(0.0);

    if let Some((min, max)) = field.range {
        let signal = Signal::new(current_val);
        let comp_name = component_name.to_string();
        let field_name = field.name.clone();

        let slider_container = rsx! {
            div { style: "flex:1;min-width:60px;",
                Slider {
                    min: Some(min),
                    max: Some(max),
                    step: Some((max - min) / 200.0),
                    value_signal: Some(signal),
                    size: "xs",
                    onchange: move |v: f64| {
                        signal.set(v);
                        let _ = cmd.0.send(EditorCommand::SetComponentField {
                            entity_id,
                            component_name: comp_name.clone(),
                            field_name: field_name.clone(),
                            value: rkf_runtime::behavior::game_value::GameValue::Float(v),
                        });
                    },
                }
            }
        };
        row.append_child(&slider_container);

        let val_span = rsx! {
            span {
                style: "font-size:10px;color:var(--rinch-color-dimmed);\
                        font-family:var(--rinch-font-family-monospace);\
                        min-width:40px;text-align:right;flex-shrink:0;",
                {|| format!("{:.2}", signal.get())}
            }
        };
        row.append_child(&val_span);
    } else {
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

        let dv_container = rsx! {
            div { style: "flex:1;min-width:60px;" }
        };
        dv_container.append_child(&dv.render(__scope, &[]));
        row.append_child(&dv_container);
    }
}

// ── Int editor ────────────────────────────────────────────────────────────

fn build_int_editor(
    __scope: &mut RenderScope,
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
        let slider_container = rsx! {
            div { style: "flex:1;min-width:60px;",
                Slider {
                    min: Some(min),
                    max: Some(max),
                    step: Some(1.0),
                    value_signal: Some(signal),
                    size: "xs",
                    onchange: move |v: f64| {
                        let iv = v.round() as i64;
                        signal.set(iv as f64);
                        let _ = cmd.0.send(EditorCommand::SetComponentField {
                            entity_id,
                            component_name: comp_name.clone(),
                            field_name: field_name.clone(),
                            value: rkf_runtime::behavior::game_value::GameValue::Int(iv),
                        });
                    },
                }
            }
        };
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

        let dv_container = rsx! {
            div { style: "flex:1;min-width:60px;" }
        };
        dv_container.append_child(&dv.render(__scope, &[]));
        row.append_child(&dv_container);
    }
}

// ── Bool editor ───────────────────────────────────────────────────────────

fn build_bool_editor(
    __scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    let current_val = field.bool_value.unwrap_or(false);
    let checked = Signal::new(current_val);
    let comp_name = component_name.to_string();
    let field_name = field.name.clone();

    let spacer = rsx! { div { style: "flex:1;" } };
    row.append_child(&spacer);

    let checkbox = rsx! {
        div {
            style: {|| {
                let bg = if checked.get() {
                    "var(--rinch-primary-color)"
                } else {
                    "rgba(255,255,255,0.06)"
                };
                format!(
                    "width:16px;height:16px;border-radius:3px;cursor:pointer;\
                     border:1px solid var(--rinch-color-border);\
                     background:{bg};flex-shrink:0;"
                )
            }},
            onclick: move || {
                let new_val = !checked.get();
                checked.set(new_val);
                let _ = cmd.0.send(EditorCommand::SetComponentField {
                    entity_id,
                    component_name: comp_name.clone(),
                    field_name: field_name.clone(),
                    value: rkf_runtime::behavior::game_value::GameValue::Bool(new_val),
                });
            },

            if checked.get() {
                span {
                    style: "color:#fff;font-size:11px;display:flex;\
                            align-items:center;justify-content:center;height:100%;",
                    "\u{2713}"
                }
            }
        }
    };
    row.append_child(&checkbox);
}

// ── Vec3 editor ──────────────────────────────────────────────────────────

fn build_vec3_editor(
    __scope: &mut RenderScope,
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

    let editor_container = rsx! {
        div { style: "flex:1;" }
    };
    editor_container.append_child(&editor.render(__scope, &[]));
    row.append_child(&editor_container);
}

// ── String editor ────────────────────────────────────────────────────────

fn build_string_editor(
    __scope: &mut RenderScope,
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

    let input = rsx! {
        input {
            r#type: "text",
            value: {current_val.clone()},
            style: "flex:1;background:rgba(255,255,255,0.06);border:1px solid var(--rinch-color-border);\
                    border-radius:3px;padding:1px 4px;font-size:11px;color:var(--rinch-color-text);\
                    outline:none;min-width:40px;height:18px;font-family:var(--rinch-font-family-monospace);",
            oninput: move |text: String| {
                let _ = cmd.0.send(EditorCommand::SetComponentField {
                    entity_id,
                    component_name: comp_name.clone(),
                    field_name: field_name.clone(),
                    value: rkf_runtime::behavior::game_value::GameValue::String(text),
                });
            },
        }
    };
    row.append_child(&input);
}

// ── Struct editor ────────────────────────────────────────────────────────

const STRUCT_HEADER_STYLE: &str = "display:flex;align-items:center;padding:2px 12px;\
    cursor:pointer;user-select:none;gap:4px;";

/// Build a collapsible struct editor that recursively renders sub-fields.
/// Each sub-field sends dot-notated field names in `SetComponentField`.
fn build_struct_editor(
    __scope: &mut RenderScope,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    field_path: &str,
    cmd: CommandSender,
) -> NodeHandle {
    let collapsed = Signal::new(false);
    let field_display = field.name.clone();

    let section = rsx! {
        div { style: "display:flex;flex-direction:column;margin-left:8px;",
            // Collapsible header.
            div {
                style: {STRUCT_HEADER_STYLE},
                onclick: move || collapsed.update(|v| *v = !*v),

                span {
                    style: {|| if collapsed.get() {
                        "font-size:8px;color:var(--rinch-color-dimmed);"
                    } else {
                        "font-size:8px;color:var(--rinch-color-dimmed);\
                         transform:rotate(90deg);display:inline-block;"
                    }},
                    "\u{25B6}"
                }

                span {
                    style: "font-size:11px;color:var(--rinch-color-dimmed);flex:1;",
                    {field_display}
                }
            }
        }
    };

    // Sub-fields container (hidden when collapsed).
    let fields_container = rsx! {
        div {
            style: {|| if collapsed.get() {
                "display:none;"
            } else {
                "display:flex;flex-direction:column;margin-left:12px;\
                 border-left:1px solid var(--rinch-color-border);padding-left:4px;"
            }},
        }
    };

    if let Some(sub_fields) = &field.sub_fields {
        for sub in sub_fields {
            let dotted_path = format!("{}.{}", field_path, sub.name);

            if sub.field_type == rkf_runtime::behavior::registry::FieldType::Struct {
                // Recurse for nested structs.
                let nested = build_struct_editor(
                    __scope,
                    sub,
                    entity_id,
                    component_name,
                    &dotted_path,
                    cmd.clone(),
                );
                fields_container.append_child(&nested);
            } else {
                // Build a sub-field row with the dot-notated path.
                let mut sub_with_dotted_name = sub.clone();
                sub_with_dotted_name.name = sub.name.clone();
                let row = build_field_row_with_path(
                    __scope,
                    &sub_with_dotted_name,
                    entity_id,
                    component_name,
                    &dotted_path,
                    cmd.clone(),
                );
                fields_container.append_child(&row);
            }
        }
    }

    section.append_child(&fields_container);
    section
}

/// Build a field row that sends a specific dot-notated path as the field name.
fn build_field_row_with_path(
    __scope: &mut RenderScope,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    dotted_path: &str,
    cmd: CommandSender,
) -> NodeHandle {
    use rkf_runtime::behavior::registry::FieldType;

    let row = rsx! {
        div { style: {FIELD_ROW_STYLE},
            span { style: {FIELD_LABEL_STYLE}, {field.name.clone()} }
        }
    };

    // Create a temporary FieldSnapshot with the dotted path as the name,
    // so the editor callbacks send the correct path.
    let mut path_field = field.clone();
    path_field.name = dotted_path.to_string();

    match field.field_type {
        FieldType::Float => {
            build_float_editor(__scope, &row, &path_field, entity_id, component_name, cmd);
        }
        FieldType::Int => {
            build_int_editor(__scope, &row, &path_field, entity_id, component_name, cmd);
        }
        FieldType::Bool => {
            build_bool_editor(__scope, &row, &path_field, entity_id, component_name, cmd);
        }
        FieldType::Vec3 => {
            build_vec3_editor(__scope, &row, &path_field, entity_id, component_name, cmd);
        }
        FieldType::String | FieldType::AssetRef | FieldType::ComponentRef => {
            build_string_editor(__scope, &row, &path_field, entity_id, component_name, cmd);
        }
        _ => {
            let val_span = rsx! {
                span { style: {FIELD_VALUE_STYLE}, {field.display_value.clone()} }
            };
            row.append_child(&val_span);
        }
    }

    row
}

// ── Asset ref editor ─────────────────────────────────────────────────────

const BADGE_STYLE: &str = "font-size:9px;color:var(--rinch-color-placeholder);\
    border:1px solid var(--rinch-color-border);border-radius:2px;\
    padding:0 3px;flex-shrink:0;";

fn build_asset_ref_editor(
    __scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    // Reuse string editor for the path input.
    build_string_editor(__scope, row, field, entity_id, component_name, cmd);

    // Add filter badge showing allowed extension.
    if let Some(ext) = &field.asset_filter {
        let badge_text = format!(".{ext}");
        let badge = rsx! {
            span { style: {BADGE_STYLE}, {badge_text} }
        };
        row.append_child(&badge);
    }
}

// ── Component ref editor ─────────────────────────────────────────────────

fn build_component_ref_editor(
    __scope: &mut RenderScope,
    row: &NodeHandle,
    field: &FieldSnapshot,
    entity_id: uuid::Uuid,
    component_name: &str,
    cmd: CommandSender,
) {
    // Reuse string editor for the UUID input.
    build_string_editor(__scope, row, field, entity_id, component_name, cmd);

    // Add component type badge.
    if let Some(comp) = &field.component_filter {
        let badge_text = comp.clone();
        let badge = rsx! {
            span { style: {BADGE_STYLE}, {badge_text} }
        };
        row.append_child(&badge);
    }
}
