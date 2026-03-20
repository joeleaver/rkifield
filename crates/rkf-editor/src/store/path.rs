use std::collections::HashMap;

/// Parsed route from a UI store path string.
///
/// Each variant corresponds to a different data domain that the store routes to.
#[derive(Debug, Clone, PartialEq)]
pub enum PathRoute {
    /// `entity:{uuid}/{ComponentName}/{field.path}`
    EcsField {
        entity_id: uuid::Uuid,
        component: String,
        field: String,
    },
    /// `env/{field.path}` — shortcut for active camera EnvironmentSettings
    EnvField { field: String },
    /// `light:{id}/{field}`
    LightField { light_id: u64, field: String },
    /// `material:{slot}/{field}`
    MaterialField { slot: u16, field: String },
    /// `camera/{field}`
    CameraField { field: String },
    /// `editor/{field}`
    EditorState { field: String },
    /// `sculpt/{field}`, `paint/{field}`, `gizmo/{field}`
    ToolState { tool: String, field: String },
    /// `console/{anything}`, `drag/{anything}`, `viewport/{field}`
    SystemPath { path: String },
}

/// Parse a UI store path string into a `PathRoute`.
///
/// Returns `Err` with a description if the path is malformed.
pub fn parse_path(s: &str) -> Result<PathRoute, String> {
    if s.is_empty() {
        return Err("empty path".into());
    }

    // Prefixes with colon-delimited ID: entity:{uuid}/..., light:{id}/..., material:{slot}/...
    if let Some(rest) = s.strip_prefix("entity:") {
        return parse_entity_path(rest);
    }
    if let Some(rest) = s.strip_prefix("light:") {
        return parse_light_path(rest);
    }
    if let Some(rest) = s.strip_prefix("material:") {
        return parse_material_path(rest);
    }

    // Slash-prefixed domains
    if let Some(field) = s.strip_prefix("camera/") {
        return if field.is_empty() {
            Err("camera path missing field".into())
        } else {
            Ok(PathRoute::CameraField {
                field: field.to_string(),
            })
        };
    }
    if let Some(field) = s.strip_prefix("editor/") {
        return if field.is_empty() {
            Err("editor path missing field".into())
        } else {
            Ok(PathRoute::EditorState {
                field: field.to_string(),
            })
        };
    }
    if let Some(field) = s.strip_prefix("env/") {
        return if field.is_empty() {
            Err("env path missing field".into())
        } else {
            Ok(PathRoute::EnvField {
                field: field.to_string(),
            })
        };
    }

    // Tool paths
    for tool in &["sculpt", "paint", "gizmo"] {
        if let Some(field) = s.strip_prefix(tool).and_then(|r| r.strip_prefix('/')) {
            return if field.is_empty() {
                Err(format!("{tool} path missing field"))
            } else {
                Ok(PathRoute::ToolState {
                    tool: (*tool).to_string(),
                    field: field.to_string(),
                })
            };
        }
    }

    // System paths
    for prefix in &["console", "drag", "viewport"] {
        if let Some(rest) = s.strip_prefix(prefix).and_then(|r| r.strip_prefix('/')) {
            return if rest.is_empty() {
                Err(format!("{prefix} path missing field"))
            } else {
                Ok(PathRoute::SystemPath {
                    path: s.to_string(),
                })
            };
        }
    }

    Err(format!("unrecognized path prefix: {s}"))
}

/// Parse `{uuid}/{Component}/{field}` after the `entity:` prefix.
fn parse_entity_path(rest: &str) -> Result<PathRoute, String> {
    let slash = rest.find('/').ok_or_else(|| {
        "entity path missing component (expected entity:{uuid}/{Component}/{field})".to_string()
    })?;
    let uuid_str = &rest[..slash];
    let after_uuid = &rest[slash + 1..];

    let entity_id = uuid::Uuid::parse_str(uuid_str)
        .map_err(|e| format!("invalid entity UUID '{uuid_str}': {e}"))?;

    let slash2 = after_uuid.find('/').ok_or_else(|| {
        "entity path missing field (expected entity:{uuid}/{Component}/{field})".to_string()
    })?;
    let component = &after_uuid[..slash2];
    let field = &after_uuid[slash2 + 1..];

    if component.is_empty() {
        return Err("entity path has empty component name".into());
    }
    if field.is_empty() {
        return Err("entity path has empty field".into());
    }

    Ok(PathRoute::EcsField {
        entity_id,
        component: component.to_string(),
        field: field.to_string(),
    })
}

/// Parse `{id}/{field}` after the `light:` prefix.
fn parse_light_path(rest: &str) -> Result<PathRoute, String> {
    let slash = rest
        .find('/')
        .ok_or_else(|| "light path missing field (expected light:{id}/{field})".to_string())?;
    let id_str = &rest[..slash];
    let field = &rest[slash + 1..];

    let light_id: u64 = id_str
        .parse()
        .map_err(|_| format!("invalid light ID '{id_str}': expected non-negative integer"))?;

    if field.is_empty() {
        return Err("light path has empty field".into());
    }

    Ok(PathRoute::LightField {
        light_id,
        field: field.to_string(),
    })
}

/// Parse `{slot}/{field}` after the `material:` prefix.
fn parse_material_path(rest: &str) -> Result<PathRoute, String> {
    let slash = rest.find('/').ok_or_else(|| {
        "material path missing field (expected material:{slot}/{field})".to_string()
    })?;
    let slot_str = &rest[..slash];
    let field = &rest[slash + 1..];

    let slot: u16 = slot_str
        .parse()
        .map_err(|_| format!("invalid material slot '{slot_str}': expected u16 integer"))?;

    if field.is_empty() {
        return Err("material path has empty field".into());
    }

    Ok(PathRoute::MaterialField {
        slot,
        field: field.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Path interning registry
// ---------------------------------------------------------------------------

struct PathEntry {
    path: String,
    route: PathRoute,
}

/// Interns path strings to compact `u32` IDs and caches their parsed routes.
pub struct PathRegistry {
    path_to_id: HashMap<String, u32>,
    entries: Vec<PathEntry>,
}

impl PathRegistry {
    pub fn new() -> Self {
        Self {
            path_to_id: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Intern a path, returning its stable ID. Creates a new entry if the path
    /// has not been seen before. Returns `Err` if the path fails to parse.
    pub fn intern(&mut self, path: &str) -> Result<u32, String> {
        if let Some(&id) = self.path_to_id.get(path) {
            return Ok(id);
        }
        let route = parse_path(path)?;
        let id = self.entries.len() as u32;
        self.path_to_id.insert(path.to_string(), id);
        self.entries.push(PathEntry {
            path: path.to_string(),
            route,
        });
        Ok(id)
    }

    /// Look up the parsed route for a previously interned ID.
    pub fn get_route(&self, id: u32) -> Option<&PathRoute> {
        self.entries.get(id as usize).map(|e| &e.route)
    }

    /// Look up the ID for a previously interned path (without creating it).
    pub fn get_id(&self, path: &str) -> Option<u32> {
        self.path_to_id.get(path).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- EcsField parsing --

    #[test]
    fn parse_ecs_field() {
        let uuid = uuid::Uuid::new_v4();
        let path = format!("entity:{uuid}/Transform/position");
        let route = parse_path(&path).unwrap();
        assert_eq!(
            route,
            PathRoute::EcsField {
                entity_id: uuid,
                component: "Transform".into(),
                field: "position".into(),
            }
        );
    }

    #[test]
    fn parse_ecs_field_nested() {
        let uuid = uuid::Uuid::new_v4();
        let path = format!("entity:{uuid}/EnvironmentSettings/fog.density");
        let route = parse_path(&path).unwrap();
        assert_eq!(
            route,
            PathRoute::EcsField {
                entity_id: uuid,
                component: "EnvironmentSettings".into(),
                field: "fog.density".into(),
            }
        );
    }

    #[test]
    fn parse_ecs_field_field_with_slashes() {
        // field part can contain slashes (everything after second slash)
        let uuid = uuid::Uuid::new_v4();
        let path = format!("entity:{uuid}/Comp/a/b/c");
        let route = parse_path(&path).unwrap();
        assert_eq!(
            route,
            PathRoute::EcsField {
                entity_id: uuid,
                component: "Comp".into(),
                field: "a/b/c".into(),
            }
        );
    }

    #[test]
    fn parse_ecs_field_invalid_uuid() {
        let result = parse_path("entity:not-a-uuid/Transform/position");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid entity UUID"));
    }

    #[test]
    fn parse_ecs_field_missing_component() {
        let uuid = uuid::Uuid::new_v4();
        let result = parse_path(&format!("entity:{uuid}"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_ecs_field_missing_field() {
        let uuid = uuid::Uuid::new_v4();
        let result = parse_path(&format!("entity:{uuid}/Transform"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_ecs_field_empty_component() {
        let uuid = uuid::Uuid::new_v4();
        let result = parse_path(&format!("entity:{uuid}//field"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_ecs_field_empty_field() {
        let uuid = uuid::Uuid::new_v4();
        let result = parse_path(&format!("entity:{uuid}/Comp/"));
        assert!(result.is_err());
    }

    // -- EnvField parsing --

    #[test]
    fn parse_env_field() {
        let route = parse_path("env/fog.density").unwrap();
        assert_eq!(
            route,
            PathRoute::EnvField {
                field: "fog.density".into(),
            }
        );
    }

    #[test]
    fn parse_env_field_nested() {
        let route = parse_path("env/atmosphere.sun_intensity").unwrap();
        assert_eq!(
            route,
            PathRoute::EnvField {
                field: "atmosphere.sun_intensity".into(),
            }
        );
    }

    #[test]
    fn parse_env_field_empty() {
        assert!(parse_path("env/").is_err());
    }

    // -- LightField parsing --

    #[test]
    fn parse_light_field() {
        let route = parse_path("light:3/intensity").unwrap();
        assert_eq!(
            route,
            PathRoute::LightField {
                light_id: 3,
                field: "intensity".into(),
            }
        );
    }

    #[test]
    fn parse_light_field_zero() {
        let route = parse_path("light:0/color").unwrap();
        assert_eq!(
            route,
            PathRoute::LightField {
                light_id: 0,
                field: "color".into(),
            }
        );
    }

    #[test]
    fn parse_light_field_invalid_id() {
        assert!(parse_path("light:abc/intensity").is_err());
    }

    #[test]
    fn parse_light_field_negative_id() {
        assert!(parse_path("light:-1/intensity").is_err());
    }

    #[test]
    fn parse_light_field_missing_field() {
        assert!(parse_path("light:3").is_err());
    }

    #[test]
    fn parse_light_field_empty_field() {
        assert!(parse_path("light:3/").is_err());
    }

    // -- MaterialField parsing --

    #[test]
    fn parse_material_field() {
        let route = parse_path("material:5/roughness").unwrap();
        assert_eq!(
            route,
            PathRoute::MaterialField {
                slot: 5,
                field: "roughness".into(),
            }
        );
    }

    #[test]
    fn parse_material_field_invalid_slot() {
        assert!(parse_path("material:xyz/roughness").is_err());
    }

    #[test]
    fn parse_material_field_slot_overflow() {
        // u16 max is 65535
        assert!(parse_path("material:99999/roughness").is_err());
    }

    #[test]
    fn parse_material_field_missing_field() {
        assert!(parse_path("material:5").is_err());
    }

    #[test]
    fn parse_material_field_empty_field() {
        assert!(parse_path("material:5/").is_err());
    }

    // -- CameraField parsing --

    #[test]
    fn parse_camera_field() {
        let route = parse_path("camera/fov").unwrap();
        assert_eq!(
            route,
            PathRoute::CameraField {
                field: "fov".into(),
            }
        );
    }

    #[test]
    fn parse_camera_field_fly_speed() {
        let route = parse_path("camera/fly_speed").unwrap();
        assert_eq!(
            route,
            PathRoute::CameraField {
                field: "fly_speed".into(),
            }
        );
    }

    #[test]
    fn parse_camera_field_empty() {
        assert!(parse_path("camera/").is_err());
    }

    // -- EditorState parsing --

    #[test]
    fn parse_editor_state() {
        let route = parse_path("editor/mode").unwrap();
        assert_eq!(
            route,
            PathRoute::EditorState {
                field: "mode".into(),
            }
        );
    }

    #[test]
    fn parse_editor_state_show_grid() {
        let route = parse_path("editor/show_grid").unwrap();
        assert_eq!(
            route,
            PathRoute::EditorState {
                field: "show_grid".into(),
            }
        );
    }

    #[test]
    fn parse_editor_state_empty() {
        assert!(parse_path("editor/").is_err());
    }

    // -- ToolState parsing --

    #[test]
    fn parse_sculpt_tool() {
        let route = parse_path("sculpt/radius").unwrap();
        assert_eq!(
            route,
            PathRoute::ToolState {
                tool: "sculpt".into(),
                field: "radius".into(),
            }
        );
    }

    #[test]
    fn parse_paint_tool() {
        let route = parse_path("paint/color").unwrap();
        assert_eq!(
            route,
            PathRoute::ToolState {
                tool: "paint".into(),
                field: "color".into(),
            }
        );
    }

    #[test]
    fn parse_gizmo_tool() {
        let route = parse_path("gizmo/mode").unwrap();
        assert_eq!(
            route,
            PathRoute::ToolState {
                tool: "gizmo".into(),
                field: "mode".into(),
            }
        );
    }

    #[test]
    fn parse_tool_empty_field() {
        assert!(parse_path("sculpt/").is_err());
        assert!(parse_path("paint/").is_err());
        assert!(parse_path("gizmo/").is_err());
    }

    // -- SystemPath parsing --

    #[test]
    fn parse_console_system() {
        let route = parse_path("console/output").unwrap();
        assert_eq!(
            route,
            PathRoute::SystemPath {
                path: "console/output".into(),
            }
        );
    }

    #[test]
    fn parse_drag_system() {
        let route = parse_path("drag/active").unwrap();
        assert_eq!(
            route,
            PathRoute::SystemPath {
                path: "drag/active".into(),
            }
        );
    }

    #[test]
    fn parse_viewport_system() {
        let route = parse_path("viewport/width").unwrap();
        assert_eq!(
            route,
            PathRoute::SystemPath {
                path: "viewport/width".into(),
            }
        );
    }

    #[test]
    fn parse_system_empty_field() {
        assert!(parse_path("console/").is_err());
        assert!(parse_path("drag/").is_err());
        assert!(parse_path("viewport/").is_err());
    }

    // -- Error cases --

    #[test]
    fn parse_empty_path() {
        assert!(parse_path("").is_err());
    }

    #[test]
    fn parse_unrecognized_prefix() {
        assert!(parse_path("unknown/something").is_err());
    }

    #[test]
    fn parse_bare_prefix_no_slash() {
        assert!(parse_path("camera").is_err());
        assert!(parse_path("editor").is_err());
    }

    // -- PathRegistry tests --

    #[test]
    fn registry_intern_and_lookup() {
        let mut reg = PathRegistry::new();
        let id = reg.intern("camera/fov").unwrap();
        assert_eq!(id, 0);

        let route = reg.get_route(id).unwrap();
        assert_eq!(
            *route,
            PathRoute::CameraField {
                field: "fov".into(),
            }
        );
    }

    #[test]
    fn registry_intern_same_path_returns_same_id() {
        let mut reg = PathRegistry::new();
        let id1 = reg.intern("camera/fov").unwrap();
        let id2 = reg.intern("camera/fov").unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn registry_intern_different_paths_different_ids() {
        let mut reg = PathRegistry::new();
        let id1 = reg.intern("camera/fov").unwrap();
        let id2 = reg.intern("camera/fly_speed").unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn registry_get_id() {
        let mut reg = PathRegistry::new();
        assert_eq!(reg.get_id("camera/fov"), None);
        let id = reg.intern("camera/fov").unwrap();
        assert_eq!(reg.get_id("camera/fov"), Some(id));
    }

    #[test]
    fn registry_get_route_invalid_id() {
        let reg = PathRegistry::new();
        assert!(reg.get_route(999).is_none());
    }

    #[test]
    fn registry_intern_invalid_path() {
        let mut reg = PathRegistry::new();
        let result = reg.intern("bogus");
        assert!(result.is_err());
        // Registry should not have added anything
        assert_eq!(reg.get_id("bogus"), None);
    }

    #[test]
    fn registry_sequential_ids() {
        let mut reg = PathRegistry::new();
        let id0 = reg.intern("editor/mode").unwrap();
        let id1 = reg.intern("editor/selected").unwrap();
        let id2 = reg.intern("light:0/color").unwrap();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn registry_mixed_route_types() {
        let mut reg = PathRegistry::new();
        let uuid = uuid::Uuid::new_v4();

        reg.intern(&format!("entity:{uuid}/Transform/position"))
            .unwrap();
        reg.intern("env/fog.density").unwrap();
        reg.intern("light:1/intensity").unwrap();
        reg.intern("material:0/albedo").unwrap();
        reg.intern("camera/fov").unwrap();
        reg.intern("editor/mode").unwrap();
        reg.intern("sculpt/radius").unwrap();
        reg.intern("console/output").unwrap();

        // All 8 entries, IDs 0..7
        for i in 0..8u32 {
            assert!(reg.get_route(i).is_some());
        }
        assert!(reg.get_route(8).is_none());
    }
}
