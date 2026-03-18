//! CameraFollow component — smoothly tracks a target entity's position.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Camera follow component: smoothly tracks a target entity's position.
pub struct CameraFollow {
    pub target: hecs::Entity,
    /// StableId UUID string of the target entity (for inspector/persistence).
    pub target_id: String,
    pub offset: glam::Vec3,
    pub smoothing: f32,
}

impl Default for CameraFollow {
    fn default() -> Self {
        Self {
            target: hecs::Entity::DANGLING,
            target_id: String::new(),
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
    target_id: String,
    offset: glam::Vec3,
    smoothing: f32,
}

impl Default for CameraFollowProxy {
    fn default() -> Self {
        let def = CameraFollow::default();
        Self {
            target_bits: def.target.to_bits().get(),
            target_id: String::new(),
            offset: def.offset,
            smoothing: def.smoothing,
        }
    }
}

static FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "target_id",
        field_type: FieldType::ComponentRef,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: Some("Transform"),
    },
    FieldMeta {
        name: "offset",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "smoothing",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
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
                        target_id: c.target_id.clone(),
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
                        target_id: proxy.target_id,
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
                "target_id" => Ok(GameValue::String(c.target_id.clone())),
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
                "target_id" => match value {
                    GameValue::String(s) => c.target_id = s,
                    _ => return Err("type mismatch for field 'target_id'".into()),
                },
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
