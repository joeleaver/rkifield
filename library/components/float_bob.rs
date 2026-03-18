//! FloatBob component — makes an entity bob up and down sinusoidally.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Float/bob component: makes an entity bob up and down sinusoidally.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FloatBob {
    /// Amplitude of the bobbing motion in metres.
    pub amplitude: f32,
    /// Frequency of bobbing in Hz.
    pub frequency: f32,
    /// Phase offset in radians (allows staggering multiple bobbers).
    pub phase: f32,
    /// Base Y position — initialized from entity position on first tick.
    /// Not shown in inspector (runtime state, not a user-facing parameter).
    pub base_y: Option<f32>,
}

impl Default for FloatBob {
    fn default() -> Self {
        Self { amplitude: 0.3, frequency: 1.0, phase: 0.0, base_y: None }
    }
}

// Only expose the user-facing fields in the inspector.
static FIELDS: [FieldMeta; 3] = [
    FieldMeta { name: "amplitude", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "frequency", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "phase", field_type: FieldType::Float, transient: false, range: Some((0.0, 6.283)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "FloatBob",
        meta: &FIELDS,
        serialize: |world, entity| {
            world.get::<&FloatBob>(entity).ok().map(|c| {
                ron::to_string(&*c).unwrap_or_default()
            })
        },
        deserialize_insert: |world, entity, ron_str| {
            let comp: FloatBob = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, comp).map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&FloatBob>(entity).is_ok(),
        remove: |world, entity| { let _ = world.remove_one::<FloatBob>(entity); },
        get_field: |world, entity, field| {
            let c = world.get::<&FloatBob>(entity).map_err(|_| "no FloatBob".to_string())?;
            match field {
                "amplitude" => Ok(GameValue::Float(c.amplitude as f64)),
                "frequency" => Ok(GameValue::Float(c.frequency as f64)),
                "phase" => Ok(GameValue::Float(c.phase as f64)),
                _ => Err(format!("unknown field '{field}' on FloatBob")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut FloatBob>(entity).map_err(|_| "no FloatBob".to_string())?;
            match (field, value) {
                ("amplitude", GameValue::Float(f)) => c.amplitude = f as f32,
                ("frequency", GameValue::Float(f)) => c.frequency = f as f32,
                ("phase", GameValue::Float(f)) => c.phase = f as f32,
                _ => return Err(format!("unknown or mismatched field '{field}' on FloatBob")),
            }
            Ok(())
        },
    }
}
