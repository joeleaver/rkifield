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

/// Linked camera selector — click to cycle through available cameras.
///
/// Shows "None" or the linked camera's name. Click cycles through:
/// None → Camera1 → Camera2 → ... → None.
#[component]
fn LinkedCameraDropdown() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let cmd = use_context::<CommandSender>();

    rsx! {
        div {
            style: "display:flex;align-items:center;gap:8px;padding:4px 12px;",
            div {
                style: "font-size:10px;color:var(--rinch-color-dimmed);width:80px;flex-shrink:0;",
                "Linked Camera"
            }
            div {
                style: "flex:1;font-size:10px;padding:3px 8px;\
                        background:var(--rinch-color-dark-6);color:var(--rinch-color-text);\
                        border:1px solid var(--rinch-color-border);border-radius:3px;\
                        cursor:pointer;user-select:none;",
                onclick: {
                    let cmd = cmd.clone();
                    let ui = ui;
                    move || {
                        let objects = ui.objects.get();
                        let cameras: Vec<_> = objects.iter()
                            .filter(|o| o.is_camera)
                            .collect();

                        let current = ui.linked_camera.get();
                        // Find current index, then advance to next (or None).
                        let current_idx = current.and_then(|id| cameras.iter().position(|c| c.id == id));
                        let next = match current_idx {
                            Some(i) if i + 1 < cameras.len() => Some(cameras[i + 1].id),
                            Some(_) => None, // wrap around to None
                            None if !cameras.is_empty() => Some(cameras[0].id),
                            None => None, // no cameras available
                        };

                        let _ = cmd.0.send(EditorCommand::LinkCamera { camera_id: next });
                        ui.linked_camera.set(next);
                    }
                },
                {move || {
                    let linked = ui.linked_camera.get();
                    match linked {
                        None => "None".to_string(),
                        Some(id) => {
                            let objects = ui.objects.get();
                            objects.iter()
                                .find(|o| o.id == id)
                                .map(|o| o.name.clone())
                                .unwrap_or_else(|| format!("{}...", &id.to_string()[..8]))
                        }
                    }
                }}
            }
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
