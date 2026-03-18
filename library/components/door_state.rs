//! DoorState component — a door that can be opened, optionally requiring a key.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// A door that can be opened, optionally requiring a key.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DoorState {
    pub open: bool,
    pub key_required: Option<String>,
}

impl Default for DoorState {
    fn default() -> Self {
        Self { open: false, key_required: None }
    }
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "open",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "key_required",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "DoorState",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&DoorState>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: DoorState = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world
                .insert_one(entity, comp)
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&DoorState>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<DoorState>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&DoorState>(entity)
                .map_err(|_| "entity does not have component 'DoorState'".to_string())?;
            match field_name {
                "open" => Ok(GameValue::Bool(c.open)),
                "key_required" => Ok(GameValue::String(
                    c.key_required.clone().unwrap_or_default(),
                )),
                _ => Err(format!(
                    "unknown field '{}' on component 'DoorState'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut DoorState>(entity)
                .map_err(|_| "entity does not have component 'DoorState'".to_string())?;
            match field_name {
                "open" => match value {
                    GameValue::Bool(b) => c.open = b,
                    _ => return Err("type mismatch for field 'open'".into()),
                },
                "key_required" => match value {
                    GameValue::String(s) => {
                        c.key_required = if s.is_empty() { None } else { Some(s) };
                    }
                    _ => return Err("type mismatch for field 'key_required'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'DoorState'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}
