//! Viewport toolbar — sits above the render surface in the center viewport.
//!
//! Contains: camera selector, gizmo mode buttons, editor mode buttons.
//! Mode and gizmo buttons are driven by the action registry via `ActionButton`.

use rinch::prelude::*;
use rinch_tabler_icons::{TablerIcon, TablerIconStyle, render_tabler_icon};

use crate::editor_state::UiSignals;
use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::action_button::ActionButton;

const TOOLBAR_STYLE: &str = "\
    display:flex;align-items:center;gap:2px;padding:2px 8px;\
    height:28px;flex-shrink:0;\
    background:var(--rinch-color-dark-8);border-bottom:1px solid var(--rinch-color-border);\
    pointer-events:auto;z-index:2;";

/// Viewport toolbar component.
#[component]
pub fn ViewportToolbar() -> NodeHandle {
    rsx! {
        div { style: {TOOLBAR_STYLE},
            // ── Gizmo mode buttons ───────────────────────────
            div { style: "display:flex;gap:1px;",
                ActionButton { action_id: "gizmo.translate".to_string() }
                ActionButton { action_id: "gizmo.rotate".to_string() }
                ActionButton { action_id: "gizmo.scale".to_string() }
            }

            Separator {}

            // ── Editor mode buttons (Sculpt, Paint) ──────────
            div { style: "display:flex;gap:1px;",
                ActionButton { action_id: "mode.sculpt".to_string() }
                ActionButton { action_id: "mode.paint".to_string() }
            }

            Separator {}

            // ── Camera selector ──────────────────────────────
            CameraSelector {}

            // ── Spacer ───────────────────────────────────────
            div { style: "flex:1;" }
        }
    }
}

/// Vertical separator line.
#[component]
fn Separator() -> NodeHandle {
    rsx! {
        div { style: "width:1px;height:16px;background:var(--rinch-color-border);margin:0 4px;" }
    }
}

/// Camera selector — switch viewport between scene cameras.
///
/// Rebuilds the Select when the camera list changes (keyed by camera count + ids).
#[component]
fn CameraSelector() -> NodeHandle {
    let ui = use_context::<UiSignals>();

    rsx! {
        div { style: "display:flex;align-items:center;gap:4px;color:var(--rinch-color-dimmed);",
            {render_tabler_icon(__scope, TablerIcon::Video, TablerIconStyle::Outline)}
            span { style: "font-size:11px;", "Camera" }
            // Key by camera fingerprint so the Select rebuilds when cameras change.
            for camera_key in vec![camera_list_key(&ui)] {
                div {
                    key: camera_key,
                    style: "display:contents;",
                    CameraSelectorInner {}
                }
            }
        }
    }
}

/// Compute a fingerprint of the current camera list for keyed rebuild.
fn camera_list_key(ui: &UiSignals) -> String {
    let objects = ui.objects.get();
    let cameras: Vec<_> = objects.iter().filter(|o| o.is_camera).collect();
    let mut key = format!("cam:{}", cameras.len());
    for c in &cameras {
        key.push(':');
        key.push_str(&c.id.to_string()[..8]);
    }
    key
}

/// Inner Select component — rendered fresh each time the camera list changes.
#[component]
fn CameraSelectorInner() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let store = use_context::<UiStore>();

    let objects = ui.objects.get();
    let mut options = vec![SelectOption::new("", "Editor Camera")];
    for obj in objects.iter().filter(|o| o.is_camera) {
        options.push(SelectOption::new(obj.id.to_string(), obj.name.clone()));
    }

    let cam_signal = store.read("viewport/camera");

    rsx! {
        Select {
            size: "xs",
            placeholder: "Editor",
            value_fn: {
                move || {
                    cam_signal.get().as_string().unwrap_or_default().to_string()
                }
            },
            onchange: {
                let store = store.clone();
                move |value: String| {
                    store.set("viewport/camera", UiValue::String(value));
                }
            },
            data: options,
        }
    }
}
