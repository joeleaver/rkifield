//! DoorState component — a door that can be opened, optionally requiring a key.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// A door that can be opened, optionally requiring a key.
pub struct DoorState {
    pub open: bool,
    pub key_required: Option<String>,
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "open",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: true,
    },
    FieldMeta {
        name: "key_required",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
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
                .map(|c| {
                    let key = match &c.key_required {
                        Some(k) => format!("Some(\"{}\")", k),
                        None => "None".to_string(),
                    };
                    format!("(open: {}, key_required: {})", c.open, key)
                })
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(
                    entity,
                    DoorState {
                        open: false,
                        key_required: None,
                    },
                )
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
