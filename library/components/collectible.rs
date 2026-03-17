//! Collectible component — a collectible item with a value and spinning animation.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// A collectible item with a value and spinning animation.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Collectible {
    pub value: i32,
    pub spin_speed: f32,
}

impl Default for Collectible {
    fn default() -> Self {
        Self { value: 10, spin_speed: 1.0 }
    }
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "value",
        field_type: FieldType::Int,
        transient: false,
        range: Some((0.0, 9999.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "spin_speed",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 10.0)),
        default: None,
        persist: false,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "Collectible",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&Collectible>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: Collectible = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world
                .insert_one(entity, comp)
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&Collectible>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<Collectible>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&Collectible>(entity)
                .map_err(|_| "entity does not have component 'Collectible'".to_string())?;
            match field_name {
                "value" => Ok(GameValue::Int(c.value as i64)),
                "spin_speed" => Ok(GameValue::Float(c.spin_speed as f64)),
                _ => Err(format!(
                    "unknown field '{}' on component 'Collectible'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut Collectible>(entity)
                .map_err(|_| "entity does not have component 'Collectible'".to_string())?;
            match field_name {
                "value" => match value {
                    GameValue::Int(i) => c.value = i as i32,
                    _ => return Err("type mismatch for field 'value'".into()),
                },
                "spin_speed" => match value {
                    GameValue::Float(f) => c.spin_speed = f as f32,
                    _ => return Err("type mismatch for field 'spin_speed'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'Collectible'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}
