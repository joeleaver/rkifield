//! Write routing — converts a `(PathRoute, UiValue)` pair into an `EditorCommand`.
//!
//! This is the outbound half of the store: when a widget changes a value,
//! the store calls `route_write` to produce the command that the engine
//! thread will apply.

use glam::Vec3;
use rkf_runtime::behavior::game_value::GameValue;

use super::path::PathRoute;
use super::types::UiValue;
use crate::editor_command::EditorCommand;
use crate::editor_state::EditorMode;
use crate::gizmo::GizmoMode;
use uuid::Uuid;

/// Convert a `UiValue` to the appropriate `EditorCommand` for the given path route.
///
/// Returns `None` if the route/value combination doesn't map to a command
/// (e.g. unknown field names, type mismatches, or routes handled specially
/// by the store layer).
pub fn route_write(route: &PathRoute, value: UiValue) -> Vec<EditorCommand> {
    match route {
        PathRoute::EcsField {
            entity_id,
            component,
            field,
        } => {
            let game_value: GameValue = value.into();
            vec![EditorCommand::SetComponentField {
                entity_id: *entity_id,
                component_name: component.clone(),
                field_name: field.clone(),
                value: game_value,
            }]
        }

        PathRoute::EnvField { .. } => {
            // EnvField needs the active camera UUID — the store resolves this
            // to an EcsField before calling route_write.
            vec![]
        }

        PathRoute::LightField { light_id, field } => route_light(*light_id, field, value).into_iter().collect(),

        PathRoute::CameraField { field } => route_camera(field, value).into_iter().collect(),

        PathRoute::EditorState { field } => route_editor(field, value).into_iter().collect(),

        PathRoute::ToolState { tool, field } => route_tool(tool, field, value),

        PathRoute::MaterialField { slot, field: _ } => {
            // Material fields require building a full Material struct.
            // Not yet wired — individual field mutation needs a read-modify-write
            // pattern that the store will handle at a higher level.
            let _ = slot;
            vec![]
        }

        PathRoute::SystemPath { path } => route_system(path, value),
    }
}

// ─── Light routing ───────────────────────────────────────────────────────

fn route_light(light_id: u64, field: &str, value: UiValue) -> Option<EditorCommand> {
    match field {
        "position" => {
            let v = value.as_vec3()?;
            Some(EditorCommand::SetLightPosition {
                light_id,
                position: Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32),
            })
        }
        // Per-axis position: uses NaN sentinel for unchanged axes.
        "position.x" => {
            let f = value.as_float()? as f32;
            Some(EditorCommand::SetLightPosition {
                light_id,
                position: Vec3::new(f, f32::NAN, f32::NAN),
            })
        }
        "position.y" => {
            let f = value.as_float()? as f32;
            Some(EditorCommand::SetLightPosition {
                light_id,
                position: Vec3::new(f32::NAN, f, f32::NAN),
            })
        }
        "position.z" => {
            let f = value.as_float()? as f32;
            Some(EditorCommand::SetLightPosition {
                light_id,
                position: Vec3::new(f32::NAN, f32::NAN, f),
            })
        }
        "intensity" => {
            let f = value.as_float()?;
            Some(EditorCommand::SetLightIntensity {
                light_id,
                intensity: f as f32,
            })
        }
        "range" => {
            let f = value.as_float()?;
            Some(EditorCommand::SetLightRange {
                light_id,
                range: f as f32,
            })
        }
        _ => None,
    }
}

// ─── Camera routing ──────────────────────────────────────────────────────

fn route_camera(field: &str, value: UiValue) -> Option<EditorCommand> {
    match field {
        "fov" => {
            let f = value.as_float()?;
            Some(EditorCommand::SetCameraFov { fov: f as f32 })
        }
        "fly_speed" => {
            let f = value.as_float()?;
            Some(EditorCommand::SetCameraSpeed { speed: f as f32 })
        }
        "near" => {
            // near requires far — use a sentinel that the engine interprets as
            // "keep existing". For now, set far to a large default.
            let f = value.as_float()?;
            Some(EditorCommand::SetCameraNearFar {
                near: f as f32,
                far: f32::NAN, // NaN = keep existing
            })
        }
        "far" => {
            let f = value.as_float()?;
            Some(EditorCommand::SetCameraNearFar {
                near: f32::NAN, // NaN = keep existing
                far: f as f32,
            })
        }
        "orbit_angles" => {
            let v = value.as_vec3()?;
            Some(EditorCommand::SetCameraOrbitAngles {
                yaw: v[0] as f32,
                pitch: v[1] as f32,
            })
        }
        _ => None,
    }
}

// ─── Editor state routing ────────────────────────────────────────────────

fn route_editor(field: &str, value: UiValue) -> Option<EditorCommand> {
    match field {
        "debug_mode" => {
            let i = value.as_int()?;
            Some(EditorCommand::SetDebugMode { mode: i as u32 })
        }
        "show_grid" => {
            // ToggleGrid is a toggle, not a set — fire it regardless of value.
            let _ = value.as_bool()?;
            Some(EditorCommand::ToggleGrid)
        }
        "mode" => {
            let s = value.as_string()?;
            let mode = match s {
                "default" | "Default" => EditorMode::Default,
                "sculpt" | "Sculpt" => EditorMode::Sculpt,
                "paint" | "Paint" => EditorMode::Paint,
                _ => return None,
            };
            Some(EditorCommand::SetEditorMode { mode })
        }
        _ => None,
    }
}

// ─── Tool state routing ─────────────────────────────────────────────────

fn route_tool(tool: &str, field: &str, value: UiValue) -> Vec<EditorCommand> {
    match tool {
        "sculpt" => route_sculpt_tool(field, value),
        "paint" => route_paint_tool(field, value).into_iter().collect(),
        "gizmo" => route_gizmo_tool(field, value).into_iter().collect(),
        _ => vec![],
    }
}

fn route_sculpt_tool(field: &str, value: UiValue) -> Vec<EditorCommand> {
    // Brush settings (radius, strength, falloff) are shared between sculpt and
    // paint modes. When changed, send both SetSculptSettings and SetPaintSettings
    // so both stay in sync. Individual field edits use NaN for unchanged fields.
    match field {
        "radius" => {
            let Some(f) = value.as_float() else { return vec![] };
            vec![
                EditorCommand::SetSculptSettings {
                    radius: f as f32,
                    strength: f32::NAN,
                    falloff: f32::NAN,
                },
                EditorCommand::SetPaintSettings {
                    radius: f as f32,
                    strength: f32::NAN,
                    falloff: f32::NAN,
                },
            ]
        }
        "strength" => {
            let Some(f) = value.as_float() else { return vec![] };
            vec![
                EditorCommand::SetSculptSettings {
                    radius: f32::NAN,
                    strength: f as f32,
                    falloff: f32::NAN,
                },
                EditorCommand::SetPaintSettings {
                    radius: f32::NAN,
                    strength: f as f32,
                    falloff: f32::NAN,
                },
            ]
        }
        "falloff" => {
            let Some(f) = value.as_float() else { return vec![] };
            vec![
                EditorCommand::SetSculptSettings {
                    radius: f32::NAN,
                    strength: f32::NAN,
                    falloff: f as f32,
                },
                EditorCommand::SetPaintSettings {
                    radius: f32::NAN,
                    strength: f32::NAN,
                    falloff: f as f32,
                },
            ]
        }
        _ => vec![],
    }
}

fn route_paint_tool(field: &str, value: UiValue) -> Option<EditorCommand> {
    match field {
        "radius" | "strength" | "falloff" => {
            let f = value.as_float()?;
            let (mut r, mut s, mut fo) = (f32::NAN, f32::NAN, f32::NAN);
            match field {
                "radius" => r = f as f32,
                "strength" => s = f as f32,
                "falloff" => fo = f as f32,
                _ => unreachable!(),
            }
            Some(EditorCommand::SetPaintSettings {
                radius: r,
                strength: s,
                falloff: fo,
            })
        }
        "color" => {
            let v = value.as_vec3()?;
            Some(EditorCommand::SetPaintColor {
                r: v[0] as f32,
                g: v[1] as f32,
                b: v[2] as f32,
            })
        }
        _ => None,
    }
}

fn route_gizmo_tool(field: &str, value: UiValue) -> Option<EditorCommand> {
    match field {
        "mode" => {
            let s = value.as_string()?;
            let mode = match s {
                "translate" | "Translate" => GizmoMode::Translate,
                "rotate" | "Rotate" => GizmoMode::Rotate,
                "scale" | "Scale" => GizmoMode::Scale,
                _ => return None,
            };
            Some(EditorCommand::SetGizmoMode { mode })
        }
        _ => None,
    }
}

// ─── System path routing ─────────────────────────────────────────────────

fn route_system(path: &str, value: UiValue) -> Vec<EditorCommand> {
    match path {
        "viewport/camera" => {
            let s = value.as_string().unwrap_or_default();
            let camera_id = if s.is_empty() {
                None
            } else {
                Uuid::parse_str(&s).ok()
            };
            vec![EditorCommand::SetViewportCamera { camera_id }]
        }
        _ => vec![],
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // ── EcsField ─────────────────────────────────────────────────────────

    #[test]
    fn ecs_field_routes_to_set_component_field() {
        let uuid = Uuid::new_v4();
        let route = PathRoute::EcsField {
            entity_id: uuid,
            component: "Transform".into(),
            field: "position".into(),
        };
        let cmd = route_write(&route, UiValue::Vec3([1.0, 2.0, 3.0])).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetComponentField {
                entity_id,
                component_name,
                field_name,
                value,
            } => {
                assert_eq!(entity_id, uuid);
                assert_eq!(component_name, "Transform");
                assert_eq!(field_name, "position");
                let v = value.as_vec3().into_iter().next().unwrap();
                assert!((v.x - 1.0).abs() < 1e-6);
                assert!((v.y - 2.0).abs() < 1e-6);
                assert!((v.z - 3.0).abs() < 1e-6);
            }
            _ => panic!("expected SetComponentField"),
        }
    }

    #[test]
    fn ecs_field_with_float_value() {
        let uuid = Uuid::new_v4();
        let route = PathRoute::EcsField {
            entity_id: uuid,
            component: "EnvironmentSettings".into(),
            field: "fog.density".into(),
        };
        let cmd = route_write(&route, UiValue::Float(0.5)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetComponentField {
                field_name, value, ..
            } => {
                assert_eq!(field_name, "fog.density");
                assert_eq!(value.as_float(), Some(0.5));
            }
            _ => panic!("expected SetComponentField"),
        }
    }

    // ── EnvField ─────────────────────────────────────────────────────────

    #[test]
    fn env_field_returns_none() {
        let route = PathRoute::EnvField {
            field: "fog.density".into(),
        };
        assert!(route_write(&route, UiValue::Float(0.5)).is_empty());
    }

    // ── LightField ───────────────────────────────────────────────────────

    #[test]
    fn light_intensity_routes_correctly() {
        let route = PathRoute::LightField {
            light_id: 3,
            field: "intensity".into(),
        };
        let cmd = route_write(&route, UiValue::Float(2.5)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightIntensity {
                light_id,
                intensity,
            } => {
                assert_eq!(light_id, 3);
                assert!((intensity - 2.5).abs() < 1e-6);
            }
            _ => panic!("expected SetLightIntensity"),
        }
    }

    #[test]
    fn light_position_routes_correctly() {
        let route = PathRoute::LightField {
            light_id: 1,
            field: "position".into(),
        };
        let cmd = route_write(&route, UiValue::Vec3([10.0, 20.0, 30.0])).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightPosition {
                light_id,
                position,
            } => {
                assert_eq!(light_id, 1);
                assert!((position.x - 10.0).abs() < 1e-5);
                assert!((position.y - 20.0).abs() < 1e-5);
                assert!((position.z - 30.0).abs() < 1e-5);
            }
            _ => panic!("expected SetLightPosition"),
        }
    }

    #[test]
    fn light_position_x_routes_with_nan_sentinel() {
        let route = PathRoute::LightField {
            light_id: 2,
            field: "position.x".into(),
        };
        let cmd = route_write(&route, UiValue::Float(5.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightPosition {
                light_id,
                position,
            } => {
                assert_eq!(light_id, 2);
                assert!((position.x - 5.0).abs() < 1e-5);
                assert!(position.y.is_nan());
                assert!(position.z.is_nan());
            }
            _ => panic!("expected SetLightPosition"),
        }
    }

    #[test]
    fn light_position_y_routes_with_nan_sentinel() {
        let route = PathRoute::LightField {
            light_id: 2,
            field: "position.y".into(),
        };
        let cmd = route_write(&route, UiValue::Float(7.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightPosition {
                light_id,
                position,
            } => {
                assert_eq!(light_id, 2);
                assert!(position.x.is_nan());
                assert!((position.y - 7.0).abs() < 1e-5);
                assert!(position.z.is_nan());
            }
            _ => panic!("expected SetLightPosition"),
        }
    }

    #[test]
    fn light_position_z_routes_with_nan_sentinel() {
        let route = PathRoute::LightField {
            light_id: 2,
            field: "position.z".into(),
        };
        let cmd = route_write(&route, UiValue::Float(-3.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightPosition {
                light_id,
                position,
            } => {
                assert_eq!(light_id, 2);
                assert!(position.x.is_nan());
                assert!(position.y.is_nan());
                assert!((position.z - -3.0).abs() < 1e-5);
            }
            _ => panic!("expected SetLightPosition"),
        }
    }

    #[test]
    fn light_range_routes_correctly() {
        let route = PathRoute::LightField {
            light_id: 0,
            field: "range".into(),
        };
        let cmd = route_write(&route, UiValue::Float(50.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetLightRange { light_id, range } => {
                assert_eq!(light_id, 0);
                assert!((range - 50.0).abs() < 1e-6);
            }
            _ => panic!("expected SetLightRange"),
        }
    }

    #[test]
    fn light_unknown_field_returns_none() {
        let route = PathRoute::LightField {
            light_id: 0,
            field: "unknown_field".into(),
        };
        assert!(route_write(&route, UiValue::Float(1.0)).is_empty());
    }

    #[test]
    fn light_intensity_wrong_type_returns_none() {
        let route = PathRoute::LightField {
            light_id: 0,
            field: "intensity".into(),
        };
        // Bool instead of Float — as_float returns None → route returns None.
        assert!(route_write(&route, UiValue::Bool(true)).is_empty());
    }

    // ── CameraField ──────────────────────────────────────────────────────

    #[test]
    fn camera_fov_routes_correctly() {
        let route = PathRoute::CameraField {
            field: "fov".into(),
        };
        let cmd = route_write(&route, UiValue::Float(90.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetCameraFov { fov } => {
                assert!((fov - 90.0).abs() < 1e-6);
            }
            _ => panic!("expected SetCameraFov"),
        }
    }

    #[test]
    fn camera_fly_speed_routes_correctly() {
        let route = PathRoute::CameraField {
            field: "fly_speed".into(),
        };
        let cmd = route_write(&route, UiValue::Float(5.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetCameraSpeed { speed } => {
                assert!((speed - 5.0).abs() < 1e-6);
            }
            _ => panic!("expected SetCameraSpeed"),
        }
    }

    #[test]
    fn camera_orbit_angles_routes_correctly() {
        let route = PathRoute::CameraField {
            field: "orbit_angles".into(),
        };
        let cmd = route_write(&route, UiValue::Vec3([1.57, -0.5, 0.0])).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetCameraOrbitAngles { yaw, pitch } => {
                assert!((yaw - 1.57).abs() < 1e-5);
                assert!((pitch - -0.5).abs() < 1e-5);
            }
            _ => panic!("expected SetCameraOrbitAngles"),
        }
    }

    #[test]
    fn camera_near_routes_correctly() {
        let route = PathRoute::CameraField {
            field: "near".into(),
        };
        let cmd = route_write(&route, UiValue::Float(0.1)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetCameraNearFar { near, far } => {
                assert!((near - 0.1).abs() < 1e-6);
                assert!(far.is_nan()); // far should be NaN (keep existing)
            }
            _ => panic!("expected SetCameraNearFar"),
        }
    }

    #[test]
    fn camera_unknown_field_returns_none() {
        let route = PathRoute::CameraField {
            field: "nonexistent".into(),
        };
        assert!(route_write(&route, UiValue::Float(1.0)).is_empty());
    }

    // ── EditorState ──────────────────────────────────────────────────────

    #[test]
    fn editor_debug_mode_routes_correctly() {
        let route = PathRoute::EditorState {
            field: "debug_mode".into(),
        };
        let cmd = route_write(&route, UiValue::Int(3)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetDebugMode { mode } => {
                assert_eq!(mode, 3);
            }
            _ => panic!("expected SetDebugMode"),
        }
    }

    #[test]
    fn editor_show_grid_routes_to_toggle() {
        let route = PathRoute::EditorState {
            field: "show_grid".into(),
        };
        let cmd = route_write(&route, UiValue::Bool(true)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::ToggleGrid => {}
            _ => panic!("expected ToggleGrid"),
        }
    }

    #[test]
    fn editor_mode_routes_correctly() {
        let route = PathRoute::EditorState {
            field: "mode".into(),
        };

        // Lowercase
        let cmd = route_write(&route, UiValue::String("sculpt".into())).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetEditorMode { mode } => {
                assert_eq!(mode, EditorMode::Sculpt);
            }
            _ => panic!("expected SetEditorMode"),
        }

        // Capitalized
        let cmd = route_write(&route, UiValue::String("Paint".into())).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetEditorMode { mode } => {
                assert_eq!(mode, EditorMode::Paint);
            }
            _ => panic!("expected SetEditorMode"),
        }
    }

    #[test]
    fn editor_mode_unknown_returns_none() {
        let route = PathRoute::EditorState {
            field: "mode".into(),
        };
        assert!(route_write(&route, UiValue::String("invalid".into())).is_empty());
    }

    #[test]
    fn editor_unknown_field_returns_none() {
        let route = PathRoute::EditorState {
            field: "nonexistent".into(),
        };
        assert!(route_write(&route, UiValue::Float(1.0)).is_empty());
    }

    // ── ToolState ────────────────────────────────────────────────────────

    #[test]
    fn sculpt_radius_routes_correctly() {
        let route = PathRoute::ToolState {
            tool: "sculpt".into(),
            field: "radius".into(),
        };
        let cmds = route_write(&route, UiValue::Float(2.0));
        // Brush fields send both sculpt and paint commands.
        assert_eq!(cmds.len(), 2);
        match &cmds[0] {
            EditorCommand::SetSculptSettings {
                radius,
                strength,
                falloff,
            } => {
                assert!((radius - 2.0).abs() < 1e-6);
                assert!(strength.is_nan());
                assert!(falloff.is_nan());
            }
            _ => panic!("expected SetSculptSettings"),
        }
        match &cmds[1] {
            EditorCommand::SetPaintSettings {
                radius,
                strength,
                falloff,
            } => {
                assert!((radius - 2.0).abs() < 1e-6);
                assert!(strength.is_nan());
                assert!(falloff.is_nan());
            }
            _ => panic!("expected SetPaintSettings"),
        }
    }

    #[test]
    fn sculpt_strength_routes_correctly() {
        let route = PathRoute::ToolState {
            tool: "sculpt".into(),
            field: "strength".into(),
        };
        let cmds = route_write(&route, UiValue::Float(0.8));
        assert_eq!(cmds.len(), 2);
        match &cmds[0] {
            EditorCommand::SetSculptSettings {
                radius,
                strength,
                falloff,
            } => {
                assert!(radius.is_nan());
                assert!((strength - 0.8).abs() < 1e-6);
                assert!(falloff.is_nan());
            }
            _ => panic!("expected SetSculptSettings"),
        }
    }

    #[test]
    fn paint_color_routes_correctly() {
        let route = PathRoute::ToolState {
            tool: "paint".into(),
            field: "color".into(),
        };
        let cmd = route_write(&route, UiValue::Vec3([1.0, 0.5, 0.0])).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetPaintColor { r, g, b } => {
                assert!((r - 1.0).abs() < 1e-6);
                assert!((g - 0.5).abs() < 1e-5);
                assert!((b - 0.0).abs() < 1e-6);
            }
            _ => panic!("expected SetPaintColor"),
        }
    }

    #[test]
    fn paint_radius_routes_correctly() {
        let route = PathRoute::ToolState {
            tool: "paint".into(),
            field: "radius".into(),
        };
        let cmd = route_write(&route, UiValue::Float(3.0)).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetPaintSettings { radius, .. } => {
                assert!((radius - 3.0).abs() < 1e-6);
            }
            _ => panic!("expected SetPaintSettings"),
        }
    }

    #[test]
    fn gizmo_mode_routes_correctly() {
        let route = PathRoute::ToolState {
            tool: "gizmo".into(),
            field: "mode".into(),
        };
        let cmd = route_write(&route, UiValue::String("rotate".into())).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetGizmoMode { mode } => {
                assert_eq!(mode, GizmoMode::Rotate);
            }
            _ => panic!("expected SetGizmoMode"),
        }
    }

    #[test]
    fn gizmo_mode_unknown_returns_none() {
        let route = PathRoute::ToolState {
            tool: "gizmo".into(),
            field: "mode".into(),
        };
        assert!(route_write(&route, UiValue::String("stretch".into())).is_empty());
    }

    #[test]
    fn unknown_tool_returns_none() {
        let route = PathRoute::ToolState {
            tool: "eraser".into(),
            field: "size".into(),
        };
        assert!(route_write(&route, UiValue::Float(1.0)).is_empty());
    }

    #[test]
    fn sculpt_unknown_field_returns_none() {
        let route = PathRoute::ToolState {
            tool: "sculpt".into(),
            field: "opacity".into(),
        };
        assert!(route_write(&route, UiValue::Float(1.0)).is_empty());
    }

    // ── MaterialField ────────────────────────────────────────────────────

    #[test]
    fn material_field_returns_none() {
        let route = PathRoute::MaterialField {
            slot: 5,
            field: "roughness".into(),
        };
        assert!(route_write(&route, UiValue::Float(0.5)).is_empty());
    }

    // ── SystemPath ───────────────────────────────────────────────────────

    #[test]
    fn system_path_unknown_returns_none() {
        let route = PathRoute::SystemPath {
            path: "console/output".into(),
        };
        assert!(route_write(&route, UiValue::String("test".into())).is_empty());
    }

    #[test]
    fn viewport_camera_routes_correctly() {
        let uuid = Uuid::new_v4();
        let route = PathRoute::SystemPath {
            path: "viewport/camera".into(),
        };
        let cmd = route_write(&route, UiValue::String(uuid.to_string())).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetViewportCamera { camera_id } => {
                assert_eq!(camera_id, Some(uuid));
            }
            _ => panic!("expected SetViewportCamera"),
        }
    }

    #[test]
    fn viewport_camera_empty_clears() {
        let route = PathRoute::SystemPath {
            path: "viewport/camera".into(),
        };
        let cmd = route_write(&route, UiValue::String(String::new())).into_iter().next().unwrap();
        match cmd {
            EditorCommand::SetViewportCamera { camera_id } => {
                assert_eq!(camera_id, None);
            }
            _ => panic!("expected SetViewportCamera"),
        }
    }
}
