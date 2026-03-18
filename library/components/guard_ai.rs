//! GuardAi component — guard AI state machine with patrol/chase/return behavior.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Guard AI state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GuardState {
    /// Moving along a path near the patrol origin.
    Patrol,
    /// Chasing toward a detected target.
    Chase,
    /// Returning to patrol origin after losing the target.
    Return,
}

/// Guard AI component: patrol/chase/return behavior around a patrol origin.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GuardAi {
    pub state: GuardState,
    pub patrol_origin: glam::Vec3,
    pub patrol_radius: f32,
    pub chase_speed: f32,
    pub detection_range: f32,
}

impl Default for GuardAi {
    fn default() -> Self {
        Self {
            state: GuardState::Patrol,
            patrol_origin: glam::Vec3::ZERO,
            patrol_radius: 10.0,
            chase_speed: 8.0,
            detection_range: 15.0,
        }
    }
}

static FIELDS: [FieldMeta; 5] = [
    FieldMeta {
        name: "state",
        field_type: FieldType::String,
        transient: true,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "patrol_origin",
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
        name: "patrol_radius",
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
        name: "chase_speed",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 50.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "detection_range",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 200.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "GuardAi",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&GuardAi>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: GuardAi = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world
                .insert_one(entity, comp)
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&GuardAi>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<GuardAi>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&GuardAi>(entity)
                .map_err(|_| "entity does not have component 'GuardAi'".to_string())?;
            match field_name {
                "state" => Ok(GameValue::String(format!("{:?}", c.state))),
                "patrol_origin" => Ok(GameValue::Vec3(c.patrol_origin)),
                "patrol_radius" => Ok(GameValue::Float(c.patrol_radius as f64)),
                "chase_speed" => Ok(GameValue::Float(c.chase_speed as f64)),
                "detection_range" => Ok(GameValue::Float(c.detection_range as f64)),
                _ => Err(format!("unknown field '{}' on component 'GuardAi'", field_name)),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut GuardAi>(entity)
                .map_err(|_| "entity does not have component 'GuardAi'".to_string())?;
            match field_name {
                "state" => match value {
                    GameValue::String(s) => {
                        c.state = match s.as_str() {
                            "Patrol" => GuardState::Patrol,
                            "Chase" => GuardState::Chase,
                            "Return" => GuardState::Return,
                            _ => return Err(format!("unknown GuardState '{}'", s)),
                        };
                    }
                    _ => return Err("type mismatch for field 'state'".into()),
                },
                "patrol_origin" => match value {
                    GameValue::Vec3(v) => c.patrol_origin = v,
                    _ => return Err("type mismatch for field 'patrol_origin'".into()),
                },
                "patrol_radius" => match value {
                    GameValue::Float(f) => c.patrol_radius = f as f32,
                    _ => return Err("type mismatch for field 'patrol_radius'".into()),
                },
                "chase_speed" => match value {
                    GameValue::Float(f) => c.chase_speed = f as f32,
                    _ => return Err("type mismatch for field 'chase_speed'".into()),
                },
                "detection_range" => match value {
                    GameValue::Float(f) => c.detection_range = f as f32,
                    _ => return Err("type mismatch for field 'detection_range'".into()),
                },
                _ => return Err(format!("unknown field '{}' on component 'GuardAi'", field_name)),
            }
            Ok(())
        },
    }
}
