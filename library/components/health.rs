//! Health component — hit points for a damageable entity.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Hit points for a damageable entity.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Default for Health {
    fn default() -> Self {
        Self { current: 100.0, max: 100.0 }
    }
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "current",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1000.0)),
        default: None,
        persist: true,
    },
    FieldMeta {
        name: "max",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1000.0)),
        default: None,
        persist: false,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "Health",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&Health>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: Health = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world
                .insert_one(entity, comp)
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&Health>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<Health>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&Health>(entity)
                .map_err(|_| "entity does not have component 'Health'".to_string())?;
            match field_name {
                "current" => Ok(GameValue::Float(c.current as f64)),
                "max" => Ok(GameValue::Float(c.max as f64)),
                _ => Err(format!("unknown field '{}' on component 'Health'", field_name)),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut Health>(entity)
                .map_err(|_| "entity does not have component 'Health'".to_string())?;
            match field_name {
                "current" => match value {
                    GameValue::Float(f) => c.current = f as f32,
                    _ => return Err("type mismatch for field 'current'".into()),
                },
                "max" => match value {
                    GameValue::Float(f) => c.max = f as f32,
                    _ => return Err("type mismatch for field 'max'".into()),
                },
                _ => return Err(format!("unknown field '{}' on component 'Health'", field_name)),
            }
            Ok(())
        },
    }
}
