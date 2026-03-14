//! FloatBob component — makes an entity bob up and down sinusoidally.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Float/bob component: makes an entity bob up and down sinusoidally.
pub struct FloatBob {
    /// Amplitude of the bobbing motion in metres.
    pub amplitude: f32,
    /// Frequency of bobbing in Hz.
    pub frequency: f32,
    /// Phase offset in radians (allows staggering multiple bobbers).
    pub phase: f32,
    /// Base Y position to oscillate around (initialized from entity position on first tick).
    pub base_y: Option<f32>,
}

static FIELDS: [FieldMeta; 4] = [
    FieldMeta { name: "amplitude", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true },
    FieldMeta { name: "frequency", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true },
    FieldMeta { name: "phase", field_type: FieldType::Float, transient: false, range: Some((0.0, 6.283)), default: None, persist: true },
    FieldMeta { name: "base_y", field_type: FieldType::Float, transient: false, range: None, default: None, persist: true },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "FloatBob",
        meta: &FIELDS,
        serialize: |world, entity| {
            world.get::<&FloatBob>(entity).ok().map(|c| {
                let base = c.base_y.unwrap_or(0.0);
                format!("(amplitude: {}, frequency: {}, phase: {}, base_y: {})", c.amplitude, c.frequency, c.phase, base)
            })
        },
        deserialize_insert: |world, entity, _ron_str| {
            world.insert_one(entity, FloatBob { amplitude: 0.3, frequency: 1.0, phase: 0.0, base_y: None })
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&FloatBob>(entity).is_ok(),
        remove: |world, entity| { let _ = world.remove_one::<FloatBob>(entity); },
        get_field: |world, entity, field| {
            let c = world.get::<&FloatBob>(entity).map_err(|_| "no FloatBob".to_string())?;
            match field {
                "amplitude" => Ok(GameValue::Float(c.amplitude as f64)),
                "frequency" => Ok(GameValue::Float(c.frequency as f64)),
                "phase" => Ok(GameValue::Float(c.phase as f64)),
                "base_y" => Ok(GameValue::Float(c.base_y.unwrap_or(0.0) as f64)),
                _ => Err(format!("unknown field '{field}' on FloatBob")),
            }
        },
        set_field: |world, entity, field, value| {
            let mut c = world.get::<&mut FloatBob>(entity).map_err(|_| "no FloatBob".to_string())?;
            match (field, value) {
                ("amplitude", GameValue::Float(f)) => c.amplitude = f as f32,
                ("frequency", GameValue::Float(f)) => c.frequency = f as f32,
                ("phase", GameValue::Float(f)) => c.phase = f as f32,
                ("base_y", GameValue::Float(f)) => c.base_y = Some(f as f32),
                _ => return Err(format!("unknown or mismatched field '{field}' on FloatBob")),
            }
            Ok(())
        },
    }
}
