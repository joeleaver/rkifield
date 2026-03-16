//! TransformEditor — Position, Rotation, Scale using Vec3Editor.
//!
//! Single source of truth: reads from UiSignals.objects via Memo.
//! Writes via EditorCommands. No intermediate signal layer.

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::UiSignals;
use crate::CommandSender;

use super::Vec3Editor;

/// Edits a full transform: Position, Rotation, Scale.
///
/// Derives Memos from UiSignals.objects for the given entity.
/// Sends EditorCommands on change — no SliderSignals dependency.
#[derive(Debug, Default)]
pub struct TransformEditor {
    pub entity_id: uuid::Uuid,
}

impl Component for TransformEditor {
    fn render(&self, __scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let cmd = use_context::<CommandSender>();
        let ui = use_context::<UiSignals>();
        let eid = self.entity_id;

        // Memos derived from UiSignals — reactive, Copy, single source of truth.
        let pos_x = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.position.x as f64).unwrap_or(0.0));
        let pos_y = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.position.y as f64).unwrap_or(0.0));
        let pos_z = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.position.z as f64).unwrap_or(0.0));
        let rot_x = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.rotation_degrees.x as f64).unwrap_or(0.0));
        let rot_y = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.rotation_degrees.y as f64).unwrap_or(0.0));
        let rot_z = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.rotation_degrees.z as f64).unwrap_or(0.0));
        let scale_x = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.scale.x as f64).unwrap_or(1.0));
        let scale_y = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.scale.y as f64).unwrap_or(1.0));
        let scale_z = Memo::new(move || ui.objects.get().iter().find(|o| o.id == eid).map(|o| o.scale.z as f64).unwrap_or(1.0));

        // Commit callbacks — send EditorCommands.
        let pos_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: [f64; 3]| {
                let _ = cmd.0.send(EditorCommand::SetObjectPosition {
                    entity_id: eid,
                    position: glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32),
                });
            }
        });
        let rot_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: [f64; 3]| {
                let _ = cmd.0.send(EditorCommand::SetObjectRotation {
                    entity_id: eid,
                    rotation: glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32),
                });
            }
        });
        let scale_cb = ValueCallback::new({
            let cmd = cmd.clone();
            move |v: [f64; 3]| {
                let _ = cmd.0.send(EditorCommand::SetObjectScale {
                    entity_id: eid,
                    scale: glam::Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32),
                });
            }
        });

        let sections: [(&str, Memo<f64>, Memo<f64>, Memo<f64>,
                         ValueCallback<[f64; 3]>, f64, u32, &str, f64, f64); 3] = [
            ("Position", pos_x, pos_y, pos_z, pos_cb.clone(), 0.01, 2, "", -1e9, 1e9),
            ("Rotation", rot_x, rot_y, rot_z, rot_cb.clone(), 0.5, 1, "\u{00b0}", -180.0, 180.0),
            ("Scale", scale_x, scale_y, scale_z, scale_cb.clone(), 0.01, 2, "", 0.01, 1e9),
        ];

        let container = rsx! {
            div { style: "display: flex; flex-direction: column; gap: 4px;" }
        };

        for (label, x, y, z, cb, step, decimals, suffix, min_val, max_val) in sections {
            let editor = Vec3Editor {
                x, y, z,
                on_change: Some(cb.clone()),
                on_commit: Some(cb),
                step,
                min: min_val,
                max: max_val,
                decimals,
                suffix: suffix.into(),
            };

            let row = rsx! {
                div {
                    style: "display: flex; align-items: center; padding: 2px 12px; gap: 8px;",
                    span {
                        style: "font-size: 11px; color: #999; min-width: 56px; user-select: none;",
                        {label}
                    }
                }
            };

            row.append_child(&editor.render(__scope, &[]));
            container.append_child(&row);
        }

        container
    }
}
