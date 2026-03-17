//! Spin component — rotates an entity around the Y axis.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Spin component: rotates an entity around the Y axis.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Spin {
    /// Rotation speed in radians per second.
    pub speed: f32,
}

impl Default for Spin {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

static FIELDS: [FieldMeta; 1] = [
    FieldMeta { name: "speed", field_type: FieldType::Float, transient: false, range: Some((-10.0, 10.0)), default: None, persist: true },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "Spin",
        meta: &FIELDS,
        serialize: |world, entity| {
            world.get::<&Spin>(entity).ok().map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: Spin = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, comp).map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&Spin>(entity).is_ok(),
        remove: |world, entity| { let _ = world.remove_one::<Spin>(entity); },
        get_field: |world, entity, field| {
            let c = world.get::<&Spin>(entity).map_err(|_| "no Spin".to_string())?;
            match field {
                "speed" => Ok(GameValue::Float(c.speed as f64)),
                _ => Err(format!("unknown field '{field}' on Spin")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut Spin>(entity).map_err(|_| "no Spin".to_string())?;
            match (field, value) {
                ("speed", GameValue::Float(f)) => c.speed = f as f32,
                _ => return Err(format!("unknown or mismatched field '{field}' on Spin")),
            }
            Ok(())
        },
    }
}
