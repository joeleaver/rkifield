//! Editor camera panel — FOV, fly speed, near/far, position, linked camera,
//! action buttons, and environment settings.
//!
//! Registered as its own panel (`PanelId::EditorCamera`) in the layout system.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{SelectedEntity, SliderSignals, UiSignals};
use crate::CommandSender;

use super::slider_helpers::SliderRow;
use super::{DIVIDER_STYLE, VALUE_STYLE};

/// Editor camera panel — shows camera settings, linked camera, and environment.
#[component]
pub fn EditorCameraPanel() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "display:flex;flex-direction:column;overflow-y:auto;",
            // ── Camera settings ──────────────────────────────────
            SliderRow {
                label: "FOV",
                suffix: "\u{00b0}",
                signal: Some(sliders.fov),
                min: 30.0,
                max: 120.0,
                step: 1.0,
                decimals: 0,
                on_change: {
                    let cmd = cmd.clone();
                    move |_v: f64| { sliders.send_camera_commands(&cmd); }
                },
            }
            SliderRow {
                label: "Fly Speed",
                suffix: "",
                signal: Some(sliders.fly_speed),
                min: 0.5,
                max: 500.0,
                step: 0.5,
                decimals: 1,
                on_change: {
                    let cmd = cmd.clone();
                    move |_v: f64| { sliders.send_camera_commands(&cmd); }
                },
            }
            SliderRow {
                label: "Near Plane",
                suffix: "",
                signal: Some(sliders.near),
                min: 0.01,
                max: 10.0,
                step: 0.01,
                decimals: 2,
                on_change: {
                    let cmd = cmd.clone();
                    move |_v: f64| { sliders.send_camera_commands(&cmd); }
                },
            }
            SliderRow {
                label: "Far Plane",
                suffix: "",
                signal: Some(sliders.far),
                min: 100.0,
                max: 10000.0,
                step: 100.0,
                decimals: 0,
                on_change: {
                    let cmd = cmd.clone();
                    move |_v: f64| { sliders.send_camera_commands(&cmd); }
                },
            }
            div { style: {DIVIDER_STYLE} }
            div { style: {VALUE_STYLE},
                {|| {
                    let pos = ui.camera_display_pos.get();
                    format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z)
                }}
            }
            div { style: {DIVIDER_STYLE} }

            // ── Linked Camera ────────────────────────────────────
            LinkedCameraDropdown {}

            // ── Action buttons ───────────────────────────────────
            CameraActionButtons {}

        }
    }
}

const BTN_STYLE: &str = "padding:3px 8px;font-size:10px;cursor:pointer;\
    background:var(--rinch-color-dark-7);color:var(--rinch-color-text);\
    border:1px solid var(--rinch-color-border);border-radius:3px;";

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

/// Linked camera selector — dropdown to switch viewport between scene cameras.
///
/// Shows "Editor Camera" or a scene camera name. Changing the selection sends
/// `SetViewportCamera` and updates the `viewport_camera` signal, which in turn
/// causes the engine loop to load the selected camera's environment profile.
#[component]
fn LinkedCameraDropdown() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "padding:2px 6px;",
            div { style: "font-size:10px;color:var(--rinch-color-text-dim);margin-bottom:2px;", "Viewport Camera" }
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
    let mut options = vec![SelectOption::new("", "Editor Camera")];
    for obj in objects.iter().filter(|o| o.is_camera) {
        options.push(SelectOption::new(obj.id.to_string(), obj.name.clone()));
    }

    rsx! {
        Select {
            size: "xs",
            placeholder: "Editor Camera",
            value_fn: {
                let ui = ui;
                move || {
                    ui.viewport_camera.get()
                        .map(|id| id.to_string())
                        .unwrap_or_default()
                }
            },
            onchange: {
                let cmd = cmd.clone();
                let ui = ui;
                move |value: String| {
                    if value.is_empty() {
                        let _ = cmd.0.send(EditorCommand::SetViewportCamera { camera_id: None });
                        ui.viewport_camera.set(None);
                    } else if let Ok(uuid) = uuid::Uuid::parse_str(&value) {
                        let _ = cmd.0.send(EditorCommand::SetViewportCamera { camera_id: Some(uuid) });
                        ui.viewport_camera.set(Some(uuid));
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
