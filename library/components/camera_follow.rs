//! CameraFollow component — smoothly tracks a target entity's position.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Camera follow component: smoothly tracks a target entity's position.
pub struct CameraFollow {
    pub target: hecs::Entity,
    pub offset: glam::Vec3,
    pub smoothing: f32,
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "offset",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "smoothing",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: false,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "CameraFollow",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&CameraFollow>(entity)
                .ok()
                .map(|c| {
                    format!(
                        "(offset: ({}, {}, {}), smoothing: {})",
                        c.offset.x, c.offset.y, c.offset.z, c.smoothing
                    )
                })
        },
        deserialize_insert: |world, entity, _ron_str| {
            let placeholder = world.reserve_entity();
            world
                .insert_one(
                    entity,
                    CameraFollow {
                        target: placeholder,
                        offset: glam::Vec3::new(0.0, 5.0, -10.0),
                        smoothing: 0.1,
                    },
                )
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&CameraFollow>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<CameraFollow>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&CameraFollow>(entity)
                .map_err(|_| "entity does not have component 'CameraFollow'".to_string())?;
            match field_name {
                "offset" => Ok(GameValue::Vec3(c.offset)),
                "smoothing" => Ok(GameValue::Float(c.smoothing as f64)),
                _ => Err(format!(
                    "unknown field '{}' on component 'CameraFollow'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut CameraFollow>(entity)
                .map_err(|_| "entity does not have component 'CameraFollow'".to_string())?;
            match field_name {
                "offset" => match value {
                    GameValue::Vec3(v) => c.offset = v,
                    _ => return Err("type mismatch for field 'offset'".into()),
                },
                "smoothing" => match value {
                    GameValue::Float(f) => c.smoothing = f as f32,
                    _ => return Err("type mismatch for field 'smoothing'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'CameraFollow'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}
