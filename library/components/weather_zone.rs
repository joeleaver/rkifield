//! WeatherZone component — defines a zone with fog, wind, and an environment profile.
//!
//! Demonstrates all three new field types:
//! - `FieldType::Struct` (nested `FogConfig`)
//! - `FieldType::AssetRef` (environment profile path)
//! - `FieldType::ComponentRef` (inherited from zone chain)

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue, StructMeta};

/// Fog configuration within a weather zone.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FogConfig {
    pub density: f32,
    pub height_falloff: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
}

impl Default for FogConfig {
    fn default() -> Self {
        Self {
            density: 0.02,
            height_falloff: 0.5,
            color_r: 0.7,
            color_g: 0.8,
            color_b: 0.9,
        }
    }
}

/// Weather zone component.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WeatherZone {
    pub fog: FogConfig,
    pub wind_strength: f32,
    pub profile_path: String,
    pub parent_zone_id: String,
}

impl Default for WeatherZone {
    fn default() -> Self {
        Self {
            fog: FogConfig::default(),
            wind_strength: 1.0,
            profile_path: String::new(),
            parent_zone_id: String::new(),
        }
    }
}

// ── Struct metadata (static) ──────────────────────────────────────────

static FOG_CONFIG_FIELDS: [FieldMeta; 5] = [
    FieldMeta {
        name: "density",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "height_falloff",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 5.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "color_r",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "color_g",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "color_b",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

static FOG_CONFIG_META: StructMeta = StructMeta {
    name: "FogConfig",
    fields: &FOG_CONFIG_FIELDS,
};

static FIELDS: [FieldMeta; 4] = [
    FieldMeta {
        name: "fog",
        field_type: FieldType::Struct,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: Some(&FOG_CONFIG_META),
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "wind_strength",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 50.0)),
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "profile_path",
        field_type: FieldType::AssetRef,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: Some("rkenv"),
        component_filter: None,
    },
    FieldMeta {
        name: "parent_zone_id",
        field_type: FieldType::ComponentRef,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: None,
        component_filter: Some("WeatherZone"),
    },
];

// ── Conversion helpers ───────────────────────────────────────────────────

fn fog_config_to_game_value(fog: &FogConfig) -> GameValue {
    GameValue::Struct(vec![
        ("density".into(), GameValue::Float(fog.density as f64)),
        ("height_falloff".into(), GameValue::Float(fog.height_falloff as f64)),
        ("color_r".into(), GameValue::Float(fog.color_r as f64)),
        ("color_g".into(), GameValue::Float(fog.color_g as f64)),
        ("color_b".into(), GameValue::Float(fog.color_b as f64)),
    ])
}

fn game_value_to_fog_config(val: &GameValue) -> Result<FogConfig, String> {
    let fields = val.as_struct().ok_or("expected Struct for FogConfig")?;
    let mut fog = FogConfig::default();
    for (name, v) in fields {
        match name.as_str() {
            "density" => fog.density = v.as_float().ok_or("expected float")? as f32,
            "height_falloff" => fog.height_falloff = v.as_float().ok_or("expected float")? as f32,
            "color_r" => fog.color_r = v.as_float().ok_or("expected float")? as f32,
            "color_g" => fog.color_g = v.as_float().ok_or("expected float")? as f32,
            "color_b" => fog.color_b = v.as_float().ok_or("expected float")? as f32,
            _ => {}
        }
    }
    Ok(fog)
}

// ── ComponentEntry ───────────────────────────────────────────────────────

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "WeatherZone",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&WeatherZone>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap_or_default())
        },
        deserialize_insert: |world, entity, ron_str| {
            let wz: WeatherZone = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, wz).map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&WeatherZone>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<WeatherZone>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&WeatherZone>(entity)
                .map_err(|_| "entity does not have component 'WeatherZone'".to_string())?;
            match field_name {
                "fog" => Ok(fog_config_to_game_value(&c.fog)),
                "wind_strength" => Ok(GameValue::Float(c.wind_strength as f64)),
                "profile_path" => Ok(GameValue::String(c.profile_path.clone())),
                "parent_zone_id" => Ok(GameValue::String(c.parent_zone_id.clone())),
                _ => Err(format!("unknown field '{}' on WeatherZone", field_name)),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut WeatherZone>(entity)
                .map_err(|_| "entity does not have component 'WeatherZone'".to_string())?;
            match field_name {
                "fog" => c.fog = game_value_to_fog_config(&value)?,
                "wind_strength" => match value {
                    GameValue::Float(f) => c.wind_strength = f as f32,
                    _ => return Err("type mismatch for 'wind_strength'".into()),
                },
                "profile_path" => match value {
                    GameValue::String(s) => c.profile_path = s,
                    _ => return Err("type mismatch for 'profile_path'".into()),
                },
                "parent_zone_id" => match value {
                    GameValue::String(s) => c.parent_zone_id = s,
                    _ => return Err("type mismatch for 'parent_zone_id'".into()),
                },
                _ => return Err(format!("unknown field '{}' on WeatherZone", field_name)),
            }
            Ok(())
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fog_config_roundtrip() {
        let fog = FogConfig {
            density: 0.1,
            height_falloff: 2.0,
            color_r: 0.5,
            color_g: 0.6,
            color_b: 0.7,
        };
        let gv = fog_config_to_game_value(&fog);
        let back = game_value_to_fog_config(&gv).unwrap();
        assert!((back.density - 0.1).abs() < 1e-5);
        assert!((back.height_falloff - 2.0).abs() < 1e-5);
        assert!((back.color_r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn weather_zone_entry_get_set_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn((WeatherZone::default(),));
        let e = entry();

        // Get fog struct.
        let fog_val = (e.get_field)(&world, entity, "fog").unwrap();
        assert!(fog_val.as_struct().is_some());

        // Set wind_strength.
        (e.set_field)(&mut world, entity, "wind_strength", GameValue::Float(5.0)).unwrap();
        let ws = (e.get_field)(&world, entity, "wind_strength").unwrap();
        assert_eq!(ws.as_float().unwrap(), 5.0);

        // Set profile_path (AssetRef field).
        (e.set_field)(
            &mut world,
            entity,
            "profile_path",
            GameValue::String("environments/storm.rkenv".into()),
        ).unwrap();
        let pp = (e.get_field)(&world, entity, "profile_path").unwrap();
        assert_eq!(pp.as_string().unwrap(), "environments/storm.rkenv");

        // Set parent_zone_id (ComponentRef field).
        let uuid_str = "a1b2c3d4-0000-0000-0000-000000000000";
        (e.set_field)(
            &mut world,
            entity,
            "parent_zone_id",
            GameValue::String(uuid_str.into()),
        ).unwrap();
        let pz = (e.get_field)(&world, entity, "parent_zone_id").unwrap();
        assert_eq!(pz.as_string().unwrap(), uuid_str);
    }

    #[test]
    fn weather_zone_serialize_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn((WeatherZone {
            fog: FogConfig { density: 0.05, ..Default::default() },
            wind_strength: 3.0,
            profile_path: "test.rkenv".into(),
            parent_zone_id: String::new(),
        },));
        let e = entry();

        let ron_str = (e.serialize)(&world, entity).unwrap();

        let entity2 = world.reserve_entity();
        (e.deserialize_insert)(&mut world, entity2, &ron_str).unwrap();

        let c = world.get::<&WeatherZone>(entity2).unwrap();
        assert!((c.fog.density - 0.05).abs() < 1e-5);
        assert!((c.wind_strength - 3.0).abs() < 1e-5);
        assert_eq!(c.profile_path, "test.rkenv");
    }

    #[test]
    fn field_metadata_correct() {
        assert_eq!(FIELDS[0].field_type, FieldType::Struct);
        assert!(FIELDS[0].struct_meta.is_some());
        assert_eq!(FIELDS[0].struct_meta.unwrap().name, "FogConfig");
        assert_eq!(FIELDS[0].struct_meta.unwrap().fields.len(), 5);

        assert_eq!(FIELDS[2].field_type, FieldType::AssetRef);
        assert_eq!(FIELDS[2].asset_filter, Some("rkenv"));

        assert_eq!(FIELDS[3].field_type, FieldType::ComponentRef);
        assert_eq!(FIELDS[3].component_filter, Some("WeatherZone"));
    }
}
