//! Patrol component — move between waypoints at a given speed.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Patrol behavior: move between waypoints at a given speed.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Patrol {
    pub waypoints: Vec<glam::Vec3>,
    pub speed: f32,
    pub current_index: usize,
}

impl Default for Patrol {
    fn default() -> Self {
        Self { waypoints: Vec::new(), speed: 5.0, current_index: 0 }
    }
}

static FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "waypoints",
        field_type: FieldType::List,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "speed",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 100.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "current_index",
        field_type: FieldType::Int,
        transient: true,
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
        name: "Patrol",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&Patrol>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: Patrol = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world
                .insert_one(entity, comp)
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&Patrol>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<Patrol>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&Patrol>(entity)
                .map_err(|_| "entity does not have component 'Patrol'".to_string())?;
            match field_name {
                "waypoints" => {
                    let list = c
                        .waypoints
                        .iter()
                        .map(|v| GameValue::Vec3(*v))
                        .collect();
                    Ok(GameValue::List(list))
                }
                "speed" => Ok(GameValue::Float(c.speed as f64)),
                "current_index" => Ok(GameValue::Int(c.current_index as i64)),
                _ => Err(format!("unknown field '{}' on component 'Patrol'", field_name)),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut Patrol>(entity)
                .map_err(|_| "entity does not have component 'Patrol'".to_string())?;
            match field_name {
                "speed" => match value {
                    GameValue::Float(f) => c.speed = f as f32,
                    _ => return Err("type mismatch for field 'speed'".into()),
                },
                "current_index" => match value {
                    GameValue::Int(i) => c.current_index = i as usize,
                    _ => return Err("type mismatch for field 'current_index'".into()),
                },
                "waypoints" => match value {
                    GameValue::List(items) => {
                        c.waypoints = items
                            .iter()
                            .filter_map(|v| v.as_vec3())
                            .collect();
                    }
                    _ => return Err("type mismatch for field 'waypoints'".into()),
                },
                _ => return Err(format!("unknown field '{}' on component 'Patrol'", field_name)),
            }
            Ok(())
        },
    }
}
