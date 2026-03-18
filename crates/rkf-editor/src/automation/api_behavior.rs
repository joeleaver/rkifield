//! Behavior system AutomationApi method implementations.
//!
//! Implements component_list/get/set/add/remove, state_get/set/list, and
//! blueprint_spawn on [`EditorAutomationApi`].

use std::collections::HashMap;

use rkf_core::automation::*;
use rkf_runtime::behavior::game_value::GameValue;
use rkf_runtime::behavior::registry::FieldType;

use super::EditorAutomationApi;

impl EditorAutomationApi {
    // ── Component tools ──────────────────────────────────────────────────

    pub(crate) fn component_list_impl(&self) -> Vec<ComponentInfo> {
        let Ok(reg) = self.gameplay_registry.lock() else {
            return vec![];
        };

        reg.component_names()
            .map(|name| {
                let entry = reg.component_entry(name).unwrap();
                let fields = entry
                    .meta
                    .iter()
                    .filter(|f| !f.transient)
                    .map(|f| field_meta_to_info(f))
                    .collect();
                ComponentInfo {
                    name: name.to_string(),
                    fields,
                }
            })
            .collect()
    }

    pub(crate) fn component_get_impl(
        &self,
        entity_id: &str,
        component_name: &str,
    ) -> Result<HashMap<String, String>, String> {
        let es = self.editor_state.lock().map_err(|e| e.to_string())?;
        let reg = self.gameplay_registry.lock().map_err(|e| e.to_string())?;

        let uuid = uuid::Uuid::parse_str(entity_id)
            .map_err(|_| format!("invalid entity UUID: {entity_id}"))?;
        let ecs_entity = es
            .world
            .ecs_entity_for(uuid)
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        let entry = reg
            .component_entry(component_name)
            .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

        if !(entry.has)(es.world.ecs_ref(), ecs_entity) {
            return Err(format!(
                "entity {entity_id} does not have component '{component_name}'"
            ));
        }

        let mut result = HashMap::new();
        for field in entry.meta.iter().filter(|f| !f.transient) {
            match (entry.get_field)(es.world.ecs_ref(), ecs_entity, field.name) {
                Ok(value) => {
                    result.insert(field.name.to_string(), game_value_to_string(&value));
                }
                Err(e) => {
                    result.insert(field.name.to_string(), format!("<error: {e}>"));
                }
            }
        }
        Ok(result)
    }

    pub(crate) fn component_set_impl(
        &self,
        entity_id: &str,
        component_name: &str,
        fields: HashMap<String, String>,
    ) -> Result<(), String> {
        let mut es = self.editor_state.lock().map_err(|e| e.to_string())?;
        let reg = self.gameplay_registry.lock().map_err(|e| e.to_string())?;

        let uuid = uuid::Uuid::parse_str(entity_id)
            .map_err(|_| format!("invalid entity UUID: {entity_id}"))?;
        let ecs_entity = es
            .world
            .ecs_entity_for(uuid)
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        let entry = reg
            .component_entry(component_name)
            .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

        if !(entry.has)(es.world.ecs_ref(), ecs_entity) {
            return Err(format!(
                "entity {entity_id} does not have component '{component_name}'"
            ));
        }

        for (field_name, value_str) in &fields {
            if field_name.contains('.') {
                // Dot-notation: find top-level field, read→modify→write.
                let top_field = field_name.split('.').next().unwrap();
                let rest = &field_name[top_field.len() + 1..];

                // Find the leaf field type by traversing struct metadata.
                let leaf_type = find_nested_field_type(entry.meta, field_name)
                    .ok_or_else(|| {
                        format!("unknown dotted field '{field_name}' on '{component_name}'")
                    })?;
                let value = parse_game_value(value_str, leaf_type)?;

                let mut parent_val =
                    (entry.get_field)(es.world.ecs_ref(), ecs_entity, top_field)?;
                rkf_runtime::behavior::set_nested_field(&mut parent_val, rest, value)?;
                (entry.set_field)(es.world.ecs_mut(), ecs_entity, top_field, parent_val)?;
            } else {
                // Simple flat field.
                let field_meta = entry
                    .meta
                    .iter()
                    .find(|f| f.name == field_name.as_str())
                    .ok_or_else(|| {
                        format!("unknown field '{field_name}' on component '{component_name}'")
                    })?;

                let value = parse_game_value(value_str, field_meta.field_type)?;
                (entry.set_field)(es.world.ecs_mut(), ecs_entity, field_meta.name, value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn component_add_impl(
        &self,
        entity_id: &str,
        component_name: &str,
        fields: HashMap<String, String>,
    ) -> Result<(), String> {
        let mut es = self.editor_state.lock().map_err(|e| e.to_string())?;
        let reg = self.gameplay_registry.lock().map_err(|e| e.to_string())?;

        let uuid = uuid::Uuid::parse_str(entity_id)
            .map_err(|_| format!("invalid entity UUID: {entity_id}"))?;
        let ecs_entity = es
            .world
            .ecs_entity_for(uuid)
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        let entry = reg
            .component_entry(component_name)
            .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

        // Build RON string from provided fields, or use defaults.
        let ron_data = if fields.is_empty() {
            // Use default RON (empty struct).
            "()".to_string()
        } else {
            // Build a RON struct from the provided fields.
            let mut parts = Vec::new();
            for (k, v) in &fields {
                parts.push(format!("{k}: {v}"));
            }
            format!("({})", parts.join(", "))
        };

        (entry.deserialize_insert)(es.world.ecs_mut(), ecs_entity, &ron_data)?;

        // If caller provided fields that need setting (since RON parsing may not
        // capture typed values correctly), apply them via set_field.
        if !fields.is_empty() {
            for (field_name, value_str) in &fields {
                if let Some(field_meta) =
                    entry.meta.iter().find(|f| f.name == field_name.as_str())
                {
                    if let Ok(value) = parse_game_value(value_str, field_meta.field_type) {
                        let _ = (entry.set_field)(
                            es.world.ecs_mut(),
                            ecs_entity,
                            field_meta.name,
                            value,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn component_remove_impl(
        &self,
        entity_id: &str,
        component_name: &str,
    ) -> Result<(), String> {
        let mut es = self.editor_state.lock().map_err(|e| e.to_string())?;
        let reg = self.gameplay_registry.lock().map_err(|e| e.to_string())?;

        let uuid = uuid::Uuid::parse_str(entity_id)
            .map_err(|_| format!("invalid entity UUID: {entity_id}"))?;
        let ecs_entity = es
            .world
            .ecs_entity_for(uuid)
            .ok_or_else(|| format!("entity {entity_id} not found"))?;

        let entry = reg
            .component_entry(component_name)
            .ok_or_else(|| format!("unknown component: '{component_name}'"))?;

        (entry.remove)(es.world.ecs_mut(), ecs_entity);
        Ok(())
    }

    // ── State tools ──────────────────────────────────────────────────────

    pub(crate) fn state_get_impl(&self, key: &str) -> Result<Option<String>, String> {
        let store = self.game_store.lock().map_err(|e| e.to_string())?;
        Ok(store.get_raw(key).map(game_value_to_string))
    }

    pub(crate) fn state_set_impl(
        &self,
        key: &str,
        value: &str,
        value_type: &str,
    ) -> Result<(), String> {
        let mut store = self.game_store.lock().map_err(|e| e.to_string())?;

        let gv = match value_type {
            "bool" => {
                let b = match value {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => return Err(format!("invalid bool: '{value}'")),
                };
                GameValue::Bool(b)
            }
            "i32" | "i64" | "int" => {
                let i: i64 = value
                    .parse()
                    .map_err(|_| format!("invalid integer: '{value}'"))?;
                GameValue::Int(i)
            }
            "f32" | "f64" | "float" => {
                let f: f64 = value
                    .parse()
                    .map_err(|_| format!("invalid float: '{value}'"))?;
                GameValue::Float(f)
            }
            "string" | "str" => GameValue::String(value.to_string()),
            _ => {
                // Default: try to parse as number, then bool, then string.
                if let Ok(i) = value.parse::<i64>() {
                    GameValue::Int(i)
                } else if let Ok(f) = value.parse::<f64>() {
                    GameValue::Float(f)
                } else if value == "true" || value == "false" {
                    GameValue::Bool(value == "true")
                } else {
                    GameValue::String(value.to_string())
                }
            }
        };

        store.set(key, gv);
        Ok(())
    }

    pub(crate) fn state_list_impl(&self, prefix: &str) -> Vec<String> {
        let Ok(store) = self.game_store.lock() else {
            return vec![];
        };
        store.list(prefix).map(|(k, _)| k.to_string()).collect()
    }

    // ── Blueprint spawn ──────────────────────────────────────────────────

    pub(crate) fn blueprint_spawn_impl(
        &self,
        name: &str,
        position: [f32; 3],
    ) -> Result<String, String> {
        let mut es = self.editor_state.lock().map_err(|e| e.to_string())?;
        let reg = self.gameplay_registry.lock().map_err(|e| e.to_string())?;

        let blueprint = reg
            .blueprint_catalog
            .get(name)
            .ok_or_else(|| format!("blueprint '{name}' not found"))?
            .clone();

        let pos = glam::Vec3::new(position[0], position[1], position[2]);

        let hecs_entity = rkf_runtime::behavior::blueprint::spawn_from_blueprint(
            es.world.ecs_mut(),
            &blueprint,
            pos,
            &reg,
        )?;

        // Ensure the entity has a StableId so we can return a UUID.
        let stable_id = if let Ok(sid) = es.world.ecs_ref().get::<&rkf_runtime::behavior::StableId>(hecs_entity) {
            *sid
        } else {
            let sid = rkf_runtime::behavior::StableId::new();
            es.world.ecs_mut().insert_one(hecs_entity, sid)
                .map_err(|e| format!("failed to insert StableId: {e}"))?;
            sid
        };

        Ok(stable_id.0.to_string())
    }
}

// ── Helper functions ──────────────────────────────────────────────────────

/// Resolve a dot-notation field path to the leaf `FieldType`.
///
/// For example, `find_nested_field_type(meta, "fog.density")` returns `FieldType::Float`.
fn find_nested_field_type(
    meta: &[rkf_runtime::behavior::registry::FieldMeta],
    path: &str,
) -> Option<FieldType> {
    let mut parts = path.split('.');
    let first = parts.next()?;
    let field = meta.iter().find(|f| f.name == first)?;

    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        return Some(field.field_type);
    }

    // Recurse into struct metadata.
    let sm = field.struct_meta?;
    find_nested_field_type(sm.fields, &rest.join("."))
}

/// Convert a `FieldMeta` to a `FieldInfo` for MCP, including struct/asset/component metadata.
fn field_meta_to_info(f: &rkf_runtime::behavior::registry::FieldMeta) -> FieldInfo {
    let struct_meta = f.struct_meta.map(|sm| {
        StructFieldInfo {
            name: sm.name.to_string(),
            fields: sm
                .fields
                .iter()
                .map(|sf| field_meta_to_info(sf))
                .collect(),
        }
    });

    FieldInfo {
        name: f.name.to_string(),
        field_type: field_type_string(f.field_type),
        range: f.range,
        struct_meta,
        asset_filter: f.asset_filter.map(|s| s.to_string()),
        component_filter: f.component_filter.map(|s| s.to_string()),
    }
}

/// Convert a FieldType enum to a human-readable string.
fn field_type_string(ft: FieldType) -> String {
    match ft {
        FieldType::Float => "f32".to_string(),
        FieldType::Int => "i32".to_string(),
        FieldType::Bool => "bool".to_string(),
        FieldType::Vec3 => "Vec3".to_string(),
        FieldType::WorldPosition => "WorldPosition".to_string(),
        FieldType::Quat => "Quat".to_string(),
        FieldType::String => "String".to_string(),
        FieldType::Entity => "Entity".to_string(),
        FieldType::Enum => "Enum".to_string(),
        FieldType::List => "List".to_string(),
        FieldType::Color => "Color".to_string(),
        FieldType::Struct => "Struct".to_string(),
        FieldType::AssetRef => "AssetRef".to_string(),
        FieldType::ComponentRef => "ComponentRef".to_string(),
    }
}

/// Convert a GameValue to a display string.
fn game_value_to_string(v: &GameValue) -> String {
    match v {
        GameValue::Bool(b) => b.to_string(),
        GameValue::Int(i) => i.to_string(),
        GameValue::Float(f) => f.to_string(),
        GameValue::String(s) => s.clone(),
        GameValue::Vec3(v) => format!("[{}, {}, {}]", v.x, v.y, v.z),
        GameValue::WorldPosition(wp) => {
            format!(
                "{{chunk: [{}, {}, {}], local: [{}, {}, {}]}}",
                wp.chunk.x, wp.chunk.y, wp.chunk.z, wp.local.x, wp.local.y, wp.local.z,
            )
        }
        GameValue::Quat(q) => format!("[{}, {}, {}, {}]", q.x, q.y, q.z, q.w),
        GameValue::Color(c) => format!("[{}, {}, {}, {}]", c[0], c[1], c[2], c[3]),
        GameValue::List(items) => {
            let parts: Vec<String> = items.iter().map(game_value_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        GameValue::Struct(fields) => {
            let parts: Vec<String> = fields
                .iter()
                .map(|(name, val)| format!("{}: {}", name, game_value_to_string(val)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        GameValue::Ron(s) => s.clone(),
    }
}

/// Parse a string into a GameValue based on the expected FieldType.
fn parse_game_value(s: &str, field_type: FieldType) -> Result<GameValue, String> {
    match field_type {
        FieldType::Bool => match s {
            "true" | "1" => Ok(GameValue::Bool(true)),
            "false" | "0" => Ok(GameValue::Bool(false)),
            _ => Err(format!("invalid bool: '{s}'")),
        },
        FieldType::Int => {
            let i: i64 = s.parse().map_err(|_| format!("invalid integer: '{s}'"))?;
            Ok(GameValue::Int(i))
        }
        FieldType::Float => {
            let f: f64 = s.parse().map_err(|_| format!("invalid float: '{s}'"))?;
            Ok(GameValue::Float(f))
        }
        FieldType::String | FieldType::Enum => Ok(GameValue::String(s.to_string())),
        FieldType::Vec3 => {
            // Parse "[x, y, z]" or "x, y, z".
            let clean = s.trim_matches(|c| c == '[' || c == ']');
            let parts: Vec<f32> = clean
                .split(',')
                .map(|p| p.trim().parse::<f32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid Vec3: '{s}'"))?;
            if parts.len() != 3 {
                return Err(format!("Vec3 requires 3 components, got {}", parts.len()));
            }
            Ok(GameValue::Vec3(glam::Vec3::new(parts[0], parts[1], parts[2])))
        }
        FieldType::Quat => {
            let clean = s.trim_matches(|c| c == '[' || c == ']');
            let parts: Vec<f32> = clean
                .split(',')
                .map(|p| p.trim().parse::<f32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid Quat: '{s}'"))?;
            if parts.len() != 4 {
                return Err(format!("Quat requires 4 components, got {}", parts.len()));
            }
            Ok(GameValue::Quat(glam::Quat::from_xyzw(
                parts[0], parts[1], parts[2], parts[3],
            )))
        }
        FieldType::Color => {
            let clean = s.trim_matches(|c| c == '[' || c == ']');
            let parts: Vec<f32> = clean
                .split(',')
                .map(|p| p.trim().parse::<f32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid Color: '{s}'"))?;
            if parts.len() != 4 {
                return Err(format!("Color requires 4 components, got {}", parts.len()));
            }
            Ok(GameValue::Color([parts[0], parts[1], parts[2], parts[3]]))
        }
        FieldType::WorldPosition => {
            // Try to parse as Vec3 for local position (chunk = 0,0,0).
            let clean = s.trim_matches(|c| c == '[' || c == ']');
            let parts: Vec<f32> = clean
                .split(',')
                .map(|p| p.trim().parse::<f32>())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid WorldPosition: '{s}'"))?;
            if parts.len() != 3 {
                return Err(format!(
                    "WorldPosition requires 3 local components, got {}",
                    parts.len()
                ));
            }
            Ok(GameValue::WorldPosition(rkf_core::WorldPosition::new(
                glam::IVec3::ZERO,
                glam::Vec3::new(parts[0], parts[1], parts[2]),
            )))
        }
        FieldType::Entity | FieldType::List => {
            // For entity references and lists, just store as string.
            Ok(GameValue::String(s.to_string()))
        }
        FieldType::AssetRef | FieldType::ComponentRef => {
            // Asset refs and component refs are stored as strings.
            Ok(GameValue::String(s.to_string()))
        }
        FieldType::Struct => {
            // Struct values can't be parsed from a flat string.
            Err("cannot parse Struct from string — use dot-notation for sub-fields".to_string())
        }
    }
}
