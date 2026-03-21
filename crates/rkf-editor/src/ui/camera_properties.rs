//! Editor camera panel — FOV, fly speed, near/far, position, linked camera,
//! action buttons, and environment settings.
//!
//! Registered as its own panel (`PanelId::EditorCamera`) in the layout system.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SelectedEntity, UiSignals};
use crate::CommandSender;

use super::bound::bound_slider::BoundSlider;
use super::component_inspector::ComponentInspector;
use super::{DIVIDER_STYLE, VALUE_STYLE};

/// Editor camera panel — shows camera settings, linked camera, and environment.
#[component]
pub fn EditorCameraPanel() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    // Get the editor camera entity UUID for the component inspector.
    let editor_cam_uuid = ui.editor_camera_inspector.get()
        .map(|snap| snap.entity_id);

    rsx! {
        div { style: "display:flex;flex-direction:column;overflow-y:auto;",
            CameraSettingsSection {}
            div { style: {DIVIDER_STYLE} }
            LinkedCameraDropdown {}
            CameraActionButtons {}
            div { style: {DIVIDER_STYLE} }
            EnvironmentProfileHeader {}
            // Environment settings via the component inspector (reusable code).
            if let Some(eid) = editor_cam_uuid {
                ComponentInspector {
                    entity_id: eid,
                    filter: vec!["EnvironmentSettings".to_string()],
                    show_header: Some(false),
                }
            }
        }
    }
}

/// Camera FOV, fly speed, near/far, position readout.
#[component]
fn CameraSettingsSection() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let fov = BoundSlider {
        path: "camera/fov".into(), label: "FOV".into(),
        min: 30.0, max: 120.0, step: 1.0, decimals: 0, suffix: "\u{00b0}".into(),
    }.render(__scope, &[]);
    let fly_speed = BoundSlider {
        path: "camera/fly_speed".into(), label: "Fly Speed".into(),
        min: 0.5, max: 500.0, step: 0.5, decimals: 1, suffix: String::new(),
    }.render(__scope, &[]);
    let near = BoundSlider {
        path: "camera/near".into(), label: "Near Plane".into(),
        min: 0.01, max: 10.0, step: 0.01, decimals: 2, suffix: String::new(),
    }.render(__scope, &[]);
    let far = BoundSlider {
        path: "camera/far".into(), label: "Far Plane".into(),
        min: 100.0, max: 10000.0, step: 100.0, decimals: 0, suffix: String::new(),
    }.render(__scope, &[]);

    rsx! {
        div {
            {fov}
            {fly_speed}
            {near}
            {far}
            div { style: {DIVIDER_STYLE} }
            div { style: {VALUE_STYLE},
                {|| {
                    let pos = ui.camera_display_pos.get();
                    format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z)
                }}
            }
        }
    }
}


const BTN_STYLE: &str = "padding:3px 8px;font-size:10px;cursor:pointer;\
    background:var(--rinch-color-dark-7);color:var(--rinch-color-text);\
    border:1px solid var(--rinch-color-border);border-radius:3px;";

/// Shows the active environment profile name or "Default".
#[component]
fn EnvironmentProfileHeader() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let profile_name = ui.environment_profile_name.get();
    let label = if profile_name.is_empty() {
        "Default".to_string()
    } else {
        profile_name
    };

    rsx! {
        div { style: "padding:2px 6px;font-size:10px;color:var(--rinch-color-text-dim);",
            span { style: "opacity:0.7;", "Profile: " }
            span { "{label}" }
        }
    }
}

/// Compute a fingerprint of the camera list for keyed rebuild.
fn camera_panel_list_key(ui: &UiSignals) -> String {
    let objects = ui.objects.get();
    let cameras: Vec<_> = objects.iter().filter(|o| o.is_camera).collect();
    let mut key = format!("cam:{}", cameras.len());
    for c in &cameras {
        key.push(':');
        key.push_str(&c.id.to_string()[..8]);
    }
    key
}

/// Linked environment camera selector — dropdown to link the editor camera's
/// environment to a scene camera's `.rkenv` profile.
///
/// This does NOT change the viewport — it only controls which environment
/// profile the editor camera inherits. Slider edits auto-save to the linked
/// camera's `.rkenv` file when set. "None" means use editor defaults.
#[component]
fn LinkedCameraDropdown() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "padding:2px 6px;",
            div { style: "font-size:10px;color:var(--rinch-color-text-dim);margin-bottom:2px;", "Environment Source" }
            for _key in vec![camera_panel_list_key(&ui)] {
                div {
                    key: _key,
                    style: "display:contents;",
                    LinkedCameraSelect {}
                }
            }
        }
    }
}

/// Inner Select for the linked camera dropdown — rebuilt when the camera list changes.
#[component]
fn LinkedCameraSelect() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();

    let objects = ui.objects.get();
    let mut options = vec![SelectOption::new("", "None (editor defaults)")];
    for obj in objects.iter().filter(|o| o.is_camera) {
        options.push(SelectOption::new(obj.id.to_string(), obj.name.clone()));
    }

    rsx! {
        Select {
            size: "xs",
            placeholder: "None",
            value_fn: {
                let ui = ui;
                move || {
                    ui.linked_env_camera.get()
                        .map(|id| id.to_string())
                        .unwrap_or_default()
                }
            },
            onchange: {
                let cmd = cmd.clone();
                let ui = ui;
                move |value: String| {
                    if value.is_empty() {
                        let _ = cmd.0.send(EditorCommand::SetLinkedEnvCamera { camera_id: None });
                        ui.linked_env_camera.set(None);
                    } else if let Ok(uuid) = uuid::Uuid::parse_str(&value) {
                        let _ = cmd.0.send(EditorCommand::SetLinkedEnvCamera { camera_id: Some(uuid) });
                        ui.linked_env_camera.set(Some(uuid));
                    }
                }
            },
            data: options,
        }
    }
}

/// Action buttons: Create Camera from View, Snap to Selected.
#[component]
fn CameraActionButtons() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "display:flex;gap:4px;padding:4px 12px;flex-wrap:wrap;",

            div {
                style: {BTN_STYLE},
                onclick: {
                    let cmd = cmd.clone();
                    move || {
                        let _ = cmd.0.send(EditorCommand::CreateCameraFromView);
                    }
                },
                "Create Camera"
            }

            div {
                style: {
                    let ui = ui;
                    move || {
                        let sel = ui.selection.get();
                        // Only enable for entities that are cameras.
                        let is_camera = if let Some(SelectedEntity::Object(eid)) = sel {
                            ui.objects.get().iter().any(|o| o.id == eid && o.is_camera)
                        } else {
                            false
                        };
                        if is_camera {
                            BTN_STYLE.to_string()
                        } else {
                            format!("{BTN_STYLE}opacity:0.4;pointer-events:none;")
                        }
                    }
                },
                onclick: {
                    let cmd = cmd.clone();
                    let ui = ui;
                    move || {
                        if let Some(SelectedEntity::Object(eid)) = ui.selection.get() {
                            let _ = cmd.0.send(EditorCommand::SnapToCamera { camera_id: eid });
                        }
                    }
                },
                "Snap to Selected"
            }
        }
    }
}
