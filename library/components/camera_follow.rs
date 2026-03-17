//! CameraFollow component — smoothly tracks a target entity's position.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Camera follow component: smoothly tracks a target entity's position.
pub struct CameraFollow {
    pub target: hecs::Entity,
    pub offset: glam::Vec3,
    pub smoothing: f32,
}

impl Default for CameraFollow {
    fn default() -> Self {
        Self {
            target: hecs::Entity::DANGLING,
            offset: glam::Vec3::new(0.0, 5.0, -10.0),
            smoothing: 0.1,
        }
    }
}

/// Proxy struct for serializing CameraFollow (hecs::Entity is not serde-compatible).
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct CameraFollowProxy {
    target_bits: u64,
    offset: glam::Vec3,
    smoothing: f32,
}

impl Default for CameraFollowProxy {
    fn default() -> Self {
        let def = CameraFollow::default();
        Self {
            target_bits: def.target.to_bits().get(),
            offset: def.offset,
            smoothing: def.smoothing,
        }
    }
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
                    let proxy = CameraFollowProxy {
                        target_bits: c.target.to_bits().get(),
                        offset: c.offset,
                        smoothing: c.smoothing,
                    };
                    ron::to_string(&proxy).unwrap_or_default()
                })
        },
        deserialize_insert: |world, entity, ron_str| {
            let proxy: CameraFollowProxy = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            let target = hecs::Entity::from_bits(proxy.target_bits).unwrap_or_else(|| world.reserve_entity());
            world
                .insert_one(
                    entity,
                    CameraFollow {
                        target,
                        offset: proxy.offset,
                        smoothing: proxy.smoothing,
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
