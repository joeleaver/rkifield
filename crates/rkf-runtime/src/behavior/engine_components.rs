//! Manual `ComponentEntry` registrations for engine components.
//!
//! Engine components (`Transform`, `CameraComponent`, `FogVolumeComponent`,
//! `EditorMetadata`) are defined in `rkf-runtime::components` and predate the
//! `#[component]` proc macro. This module creates hand-written `ComponentEntry`
//! values with correct field metadata, get/set, serialize/deserialize, has, and
//! remove implementations.

use super::game_value::GameValue;
use super::registry::{ComponentEntry, FieldMeta, FieldType, GameplayRegistry, StructMeta};
use crate::components::{CameraComponent, EditorCameraMarker, EditorMetadata, FogVolumeComponent, Transform};
use crate::environment::EnvironmentSettings;

// ─── Field metadata (static) ─────────────────────────────────────────────

static TRANSFORM_FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "position",
        field_type: FieldType::WorldPosition,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "rotation",
        field_type: FieldType::Quat,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "scale",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

static CAMERA_FIELDS: [FieldMeta; 8] = [
    FieldMeta {
        name: "fov_degrees",
        field_type: FieldType::Float,
        transient: false,
        range: Some((1.0, 179.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "near",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.001, 100.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "far",
        field_type: FieldType::Float,
        transient: false,
        range: Some((1.0, 100000.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "active",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "label",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "yaw",
        field_type: FieldType::Float,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "pitch",
        field_type: FieldType::Float,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "environment_profile",
        field_type: FieldType::AssetRef,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: None,
        asset_filter: Some("rkenv"),
        component_filter: None,
    },
];

static FOG_VOLUME_FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "density",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 10.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "phase_g",
        field_type: FieldType::Float,
        transient: false,
        range: Some((-1.0, 1.0)),
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "half_extents",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

static EDITOR_METADATA_FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "name",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "locked",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: false,
        struct_meta: None,
        asset_filter: None,
        component_filter: None,
    },
];

// ─── ComponentEntry constructors ─────────────────────────────────────────

/// Create the `ComponentEntry` for `Transform`.
fn transform_entry() -> ComponentEntry {
    ComponentEntry {
        name: "Transform",
        meta: &TRANSFORM_FIELDS,
        serialize: |world, entity| {
            world
                .get::<&Transform>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap())
        },
        deserialize_insert: |world, entity, ron_str| {
            let c: Transform = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, c).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&Transform>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<Transform>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&Transform>(entity)
                .map_err(|_| "entity does not have component 'Transform'".to_string())?;
            match field_name {
                "position" => Ok(GameValue::WorldPosition(c.position.clone())),
                "rotation" => Ok(GameValue::Quat(c.rotation)),
                "scale" => Ok(GameValue::Vec3(c.scale)),
                _ => Err(format!(
                    "unknown field '{}' on component 'Transform'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut Transform>(entity)
                .map_err(|_| "entity does not have component 'Transform'".to_string())?;
            match field_name {
                "position" => match value {
                    GameValue::WorldPosition(wp) => c.position = wp,
                    _ => return Err("type mismatch for field 'position'".into()),
                },
                "rotation" => match value {
                    GameValue::Quat(q) => c.rotation = q,
                    _ => return Err("type mismatch for field 'rotation'".into()),
                },
                "scale" => match value {
                    GameValue::Vec3(v) => c.scale = v,
                    _ => return Err("type mismatch for field 'scale'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'Transform'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}

/// Create the `ComponentEntry` for `CameraComponent`.
fn camera_entry() -> ComponentEntry {
    ComponentEntry {
        name: "CameraComponent",
        meta: &CAMERA_FIELDS,
        serialize: |world, entity| {
            world
                .get::<&CameraComponent>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap())
        },
        deserialize_insert: |world, entity, ron_str| {
            let c: CameraComponent = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, c).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&CameraComponent>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<CameraComponent>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&CameraComponent>(entity)
                .map_err(|_| "entity does not have component 'CameraComponent'".to_string())?;
            match field_name {
                "fov_degrees" => Ok(GameValue::Float(c.fov_degrees as f64)),
                "near" => Ok(GameValue::Float(c.near as f64)),
                "far" => Ok(GameValue::Float(c.far as f64)),
                "active" => Ok(GameValue::Bool(c.active)),
                "label" => Ok(GameValue::String(c.label.clone())),
                "yaw" => Ok(GameValue::Float(c.yaw as f64)),
                "pitch" => Ok(GameValue::Float(c.pitch as f64)),
                "environment_profile" => Ok(GameValue::String(c.environment_profile.clone())),
                _ => Err(format!(
                    "unknown field '{}' on component 'CameraComponent'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut CameraComponent>(entity)
                .map_err(|_| "entity does not have component 'CameraComponent'".to_string())?;
            match field_name {
                "fov_degrees" => match value {
                    GameValue::Float(f) => c.fov_degrees = f as f32,
                    _ => return Err("type mismatch for field 'fov_degrees'".into()),
                },
                "near" => match value {
                    GameValue::Float(f) => c.near = f as f32,
                    _ => return Err("type mismatch for field 'near'".into()),
                },
                "far" => match value {
                    GameValue::Float(f) => c.far = f as f32,
                    _ => return Err("type mismatch for field 'far'".into()),
                },
                "active" => match value {
                    GameValue::Bool(b) => c.active = b,
                    _ => return Err("type mismatch for field 'active'".into()),
                },
                "label" => match value {
                    GameValue::String(s) => c.label = s,
                    _ => return Err("type mismatch for field 'label'".into()),
                },
                "yaw" => match value {
                    GameValue::Float(f) => c.yaw = f as f32,
                    _ => return Err("type mismatch for field 'yaw'".into()),
                },
                "pitch" => match value {
                    GameValue::Float(f) => c.pitch = f as f32,
                    _ => return Err("type mismatch for field 'pitch'".into()),
                },
                "environment_profile" => match value {
                    GameValue::String(s) => c.environment_profile = s,
                    _ => return Err("type mismatch for field 'environment_profile'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'CameraComponent'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}

/// Create the `ComponentEntry` for `FogVolumeComponent`.
fn fog_volume_entry() -> ComponentEntry {
    ComponentEntry {
        name: "FogVolumeComponent",
        meta: &FOG_VOLUME_FIELDS,
        serialize: |world, entity| {
            world
                .get::<&FogVolumeComponent>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap())
        },
        deserialize_insert: |world, entity, ron_str| {
            let c: FogVolumeComponent = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, c).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&FogVolumeComponent>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<FogVolumeComponent>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world.get::<&FogVolumeComponent>(entity).map_err(|_| {
                "entity does not have component 'FogVolumeComponent'".to_string()
            })?;
            match field_name {
                "density" => Ok(GameValue::Float(c.density as f64)),
                "phase_g" => Ok(GameValue::Float(c.phase_g as f64)),
                "half_extents" => Ok(GameValue::Vec3(c.half_extents)),
                _ => Err(format!(
                    "unknown field '{}' on component 'FogVolumeComponent'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world.get::<&mut FogVolumeComponent>(entity).map_err(|_| {
                "entity does not have component 'FogVolumeComponent'".to_string()
            })?;
            match field_name {
                "density" => match value {
                    GameValue::Float(f) => c.density = f as f32,
                    _ => return Err("type mismatch for field 'density'".into()),
                },
                "phase_g" => match value {
                    GameValue::Float(f) => c.phase_g = f as f32,
                    _ => return Err("type mismatch for field 'phase_g'".into()),
                },
                "half_extents" => match value {
                    GameValue::Vec3(v) => c.half_extents = v,
                    _ => return Err("type mismatch for field 'half_extents'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'FogVolumeComponent'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}

/// Create the `ComponentEntry` for `EditorMetadata`.
fn editor_metadata_entry() -> ComponentEntry {
    ComponentEntry {
        name: "EditorMetadata",
        meta: &EDITOR_METADATA_FIELDS,
        serialize: |world, entity| {
            world
                .get::<&EditorMetadata>(entity)
                .ok()
                .map(|c| ron::to_string(&*c).unwrap())
        },
        deserialize_insert: |world, entity, ron_str| {
            let c: EditorMetadata = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, c).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&EditorMetadata>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<EditorMetadata>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&EditorMetadata>(entity)
                .map_err(|_| "entity does not have component 'EditorMetadata'".to_string())?;
            match field_name {
                "name" => Ok(GameValue::String(c.name.clone())),
                "locked" => Ok(GameValue::Bool(c.locked)),
                _ => Err(format!(
                    "unknown field '{}' on component 'EditorMetadata'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut EditorMetadata>(entity)
                .map_err(|_| "entity does not have component 'EditorMetadata'".to_string())?;
            match field_name {
                "name" => match value {
                    GameValue::String(s) => c.name = s,
                    _ => return Err("type mismatch for field 'name'".into()),
                },
                "locked" => match value {
                    GameValue::Bool(b) => c.locked = b,
                    _ => return Err("type mismatch for field 'locked'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'EditorMetadata'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}

// ─── EnvironmentSettings (full scene environment) ─────────────────────────

// Sub-struct metadata for EnvironmentSettings inspector rendering.

static FOG_FIELDS: [FieldMeta; 10] = [
    FieldMeta { name: "enabled", field_type: FieldType::Bool, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "density", field_type: FieldType::Float, transient: false, range: Some((0.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "color", field_type: FieldType::Color, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "start_distance", field_type: FieldType::Float, transient: false, range: Some((0.0, 1000.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "end_distance", field_type: FieldType::Float, transient: false, range: Some((0.0, 2000.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "height_falloff", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "ambient_dust_density", field_type: FieldType::Float, transient: false, range: Some((0.0, 0.1)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "dust_asymmetry", field_type: FieldType::Float, transient: false, range: Some((-1.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "vol_ambient_color", field_type: FieldType::Color, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "vol_ambient_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
];

static FOG_META: StructMeta = StructMeta { name: "FogSettings", fields: &FOG_FIELDS };

static ATMOSPHERE_FIELDS: [FieldMeta; 6] = [
    FieldMeta { name: "enabled", field_type: FieldType::Bool, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "rayleigh_scale", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "mie_scale", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "sun_direction", field_type: FieldType::Vec3, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "sun_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 20.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "sun_color", field_type: FieldType::Color, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
];

static ATMOSPHERE_META: StructMeta = StructMeta { name: "AtmosphereSettings", fields: &ATMOSPHERE_FIELDS };

static CLOUD_FIELDS: [FieldMeta; 7] = [
    FieldMeta { name: "enabled", field_type: FieldType::Bool, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "coverage", field_type: FieldType::Float, transient: false, range: Some((0.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "density", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "altitude", field_type: FieldType::Float, transient: false, range: Some((0.0, 10000.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "thickness", field_type: FieldType::Float, transient: false, range: Some((0.0, 5000.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "wind_direction", field_type: FieldType::Vec3, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "wind_speed", field_type: FieldType::Float, transient: false, range: Some((0.0, 100.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
];

static CLOUD_META: StructMeta = StructMeta { name: "CloudSettings", fields: &CLOUD_FIELDS };

static POST_PROCESS_FIELDS: [FieldMeta; 18] = [
    FieldMeta { name: "bloom_enabled", field_type: FieldType::Bool, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "bloom_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 5.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "bloom_threshold", field_type: FieldType::Float, transient: false, range: Some((0.0, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "exposure", field_type: FieldType::Float, transient: false, range: Some((0.01, 10.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "contrast", field_type: FieldType::Float, transient: false, range: Some((0.0, 3.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "saturation", field_type: FieldType::Float, transient: false, range: Some((0.0, 3.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "vignette_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "tone_map_mode", field_type: FieldType::Int, transient: false, range: Some((0.0, 3.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "sharpen_strength", field_type: FieldType::Float, transient: false, range: Some((0.0, 2.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "dof_enabled", field_type: FieldType::Bool, transient: false, range: None, default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "dof_focus_distance", field_type: FieldType::Float, transient: false, range: Some((0.1, 100.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "dof_focus_range", field_type: FieldType::Float, transient: false, range: Some((0.1, 100.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "dof_max_coc", field_type: FieldType::Float, transient: false, range: Some((1.0, 32.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "motion_blur_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 5.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "god_rays_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 5.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "grain_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "chromatic_aberration", field_type: FieldType::Float, transient: false, range: Some((0.0, 1.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
    FieldMeta { name: "gi_intensity", field_type: FieldType::Float, transient: false, range: Some((0.0, 5.0)), default: None, persist: true, struct_meta: None, asset_filter: None, component_filter: None },
];

static POST_PROCESS_META: StructMeta = StructMeta { name: "PostProcessSettings", fields: &POST_PROCESS_FIELDS };

// Top-level metadata: 4 struct fields (fog, atmosphere, clouds, post_process).
// Each has StructMeta so the inspector can expand and render sub-fields.
static ENV_SETTINGS_FIELDS: [FieldMeta; 4] = [
    FieldMeta {
        name: "fog",
        field_type: FieldType::Struct,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: Some(&FOG_META),
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "atmosphere",
        field_type: FieldType::Struct,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: Some(&ATMOSPHERE_META),
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "clouds",
        field_type: FieldType::Struct,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: Some(&CLOUD_META),
        asset_filter: None,
        component_filter: None,
    },
    FieldMeta {
        name: "post_process",
        field_type: FieldType::Struct,
        transient: false,
        range: None,
        default: None,
        persist: true,
        struct_meta: Some(&POST_PROCESS_META),
        asset_filter: None,
        component_filter: None,
    },
];

/// Get a field from `EnvironmentSettings` using dot-notation.
fn env_settings_get_field(
    c: &EnvironmentSettings,
    field_name: &str,
) -> Result<GameValue, String> {
    match field_name {
        // Top-level struct accessors (for inspector sub-field expansion)
        "fog" => Ok(GameValue::Struct(vec![
            ("enabled".into(), GameValue::Bool(c.fog.enabled)),
            ("density".into(), GameValue::Float(c.fog.density as f64)),
            ("color".into(), GameValue::Color([c.fog.color[0], c.fog.color[1], c.fog.color[2], 1.0])),
            ("start_distance".into(), GameValue::Float(c.fog.start_distance as f64)),
            ("end_distance".into(), GameValue::Float(c.fog.end_distance as f64)),
            ("height_falloff".into(), GameValue::Float(c.fog.height_falloff as f64)),
            ("ambient_dust_density".into(), GameValue::Float(c.fog.ambient_dust_density as f64)),
            ("dust_asymmetry".into(), GameValue::Float(c.fog.dust_asymmetry as f64)),
            ("vol_ambient_color".into(), GameValue::Color([c.fog.vol_ambient_color[0], c.fog.vol_ambient_color[1], c.fog.vol_ambient_color[2], 1.0])),
            ("vol_ambient_intensity".into(), GameValue::Float(c.fog.vol_ambient_intensity as f64)),
        ])),
        "atmosphere" => Ok(GameValue::Struct(vec![
            ("enabled".into(), GameValue::Bool(c.atmosphere.enabled)),
            ("rayleigh_scale".into(), GameValue::Float(c.atmosphere.rayleigh_scale as f64)),
            ("mie_scale".into(), GameValue::Float(c.atmosphere.mie_scale as f64)),
            ("sun_direction".into(), GameValue::Vec3(glam::Vec3::new(c.atmosphere.sun_direction[0], c.atmosphere.sun_direction[1], c.atmosphere.sun_direction[2]))),
            ("sun_intensity".into(), GameValue::Float(c.atmosphere.sun_intensity as f64)),
            ("sun_color".into(), GameValue::Color([c.atmosphere.sun_color[0], c.atmosphere.sun_color[1], c.atmosphere.sun_color[2], 1.0])),
        ])),
        "clouds" => Ok(GameValue::Struct(vec![
            ("enabled".into(), GameValue::Bool(c.clouds.enabled)),
            ("coverage".into(), GameValue::Float(c.clouds.coverage as f64)),
            ("density".into(), GameValue::Float(c.clouds.density as f64)),
            ("altitude".into(), GameValue::Float(c.clouds.altitude as f64)),
            ("thickness".into(), GameValue::Float(c.clouds.thickness as f64)),
            ("wind_direction".into(), GameValue::Vec3(glam::Vec3::new(c.clouds.wind_direction[0], c.clouds.wind_direction[1], c.clouds.wind_direction[2]))),
            ("wind_speed".into(), GameValue::Float(c.clouds.wind_speed as f64)),
        ])),
        "post_process" => Ok(GameValue::Struct(vec![
            ("bloom_enabled".into(), GameValue::Bool(c.post_process.bloom_enabled)),
            ("bloom_intensity".into(), GameValue::Float(c.post_process.bloom_intensity as f64)),
            ("bloom_threshold".into(), GameValue::Float(c.post_process.bloom_threshold as f64)),
            ("exposure".into(), GameValue::Float(c.post_process.exposure as f64)),
            ("contrast".into(), GameValue::Float(c.post_process.contrast as f64)),
            ("saturation".into(), GameValue::Float(c.post_process.saturation as f64)),
            ("vignette_intensity".into(), GameValue::Float(c.post_process.vignette_intensity as f64)),
            ("tone_map_mode".into(), GameValue::Int(c.post_process.tone_map_mode as i64)),
            ("sharpen_strength".into(), GameValue::Float(c.post_process.sharpen_strength as f64)),
            ("dof_enabled".into(), GameValue::Bool(c.post_process.dof_enabled)),
            ("dof_focus_distance".into(), GameValue::Float(c.post_process.dof_focus_distance as f64)),
            ("dof_focus_range".into(), GameValue::Float(c.post_process.dof_focus_range as f64)),
            ("dof_max_coc".into(), GameValue::Float(c.post_process.dof_max_coc as f64)),
            ("motion_blur_intensity".into(), GameValue::Float(c.post_process.motion_blur_intensity as f64)),
            ("god_rays_intensity".into(), GameValue::Float(c.post_process.god_rays_intensity as f64)),
            ("grain_intensity".into(), GameValue::Float(c.post_process.grain_intensity as f64)),
            ("chromatic_aberration".into(), GameValue::Float(c.post_process.chromatic_aberration as f64)),
            ("gi_intensity".into(), GameValue::Float(c.post_process.gi_intensity as f64)),
        ])),
        // Fog (dot-notation)
        "fog.enabled" => Ok(GameValue::Bool(c.fog.enabled)),
        "fog.density" => Ok(GameValue::Float(c.fog.density as f64)),
        "fog.color" => Ok(GameValue::Color([c.fog.color[0], c.fog.color[1], c.fog.color[2], 1.0])),
        "fog.start_distance" => Ok(GameValue::Float(c.fog.start_distance as f64)),
        "fog.end_distance" => Ok(GameValue::Float(c.fog.end_distance as f64)),
        "fog.height_falloff" => Ok(GameValue::Float(c.fog.height_falloff as f64)),
        "fog.ambient_dust_density" => Ok(GameValue::Float(c.fog.ambient_dust_density as f64)),
        "fog.dust_asymmetry" => Ok(GameValue::Float(c.fog.dust_asymmetry as f64)),
        "fog.vol_ambient_color" => Ok(GameValue::Color([c.fog.vol_ambient_color[0], c.fog.vol_ambient_color[1], c.fog.vol_ambient_color[2], 1.0])),
        "fog.vol_ambient_intensity" => Ok(GameValue::Float(c.fog.vol_ambient_intensity as f64)),
        // Atmosphere
        "atmosphere.enabled" => Ok(GameValue::Bool(c.atmosphere.enabled)),
        "atmosphere.rayleigh_scale" => Ok(GameValue::Float(c.atmosphere.rayleigh_scale as f64)),
        "atmosphere.mie_scale" => Ok(GameValue::Float(c.atmosphere.mie_scale as f64)),
        "atmosphere.sun_direction" => Ok(GameValue::Vec3(glam::Vec3::new(
            c.atmosphere.sun_direction[0], c.atmosphere.sun_direction[1], c.atmosphere.sun_direction[2],
        ))),
        "atmosphere.sun_intensity" => Ok(GameValue::Float(c.atmosphere.sun_intensity as f64)),
        "atmosphere.sun_color" => Ok(GameValue::Color([c.atmosphere.sun_color[0], c.atmosphere.sun_color[1], c.atmosphere.sun_color[2], 1.0])),
        // Clouds
        "clouds.enabled" => Ok(GameValue::Bool(c.clouds.enabled)),
        "clouds.coverage" => Ok(GameValue::Float(c.clouds.coverage as f64)),
        "clouds.density" => Ok(GameValue::Float(c.clouds.density as f64)),
        "clouds.altitude" => Ok(GameValue::Float(c.clouds.altitude as f64)),
        "clouds.thickness" => Ok(GameValue::Float(c.clouds.thickness as f64)),
        "clouds.wind_direction" => Ok(GameValue::Vec3(glam::Vec3::new(
            c.clouds.wind_direction[0], c.clouds.wind_direction[1], c.clouds.wind_direction[2],
        ))),
        "clouds.wind_speed" => Ok(GameValue::Float(c.clouds.wind_speed as f64)),
        // Post-process
        "post_process.bloom_enabled" => Ok(GameValue::Bool(c.post_process.bloom_enabled)),
        "post_process.bloom_intensity" => Ok(GameValue::Float(c.post_process.bloom_intensity as f64)),
        "post_process.bloom_threshold" => Ok(GameValue::Float(c.post_process.bloom_threshold as f64)),
        "post_process.exposure" => Ok(GameValue::Float(c.post_process.exposure as f64)),
        "post_process.contrast" => Ok(GameValue::Float(c.post_process.contrast as f64)),
        "post_process.saturation" => Ok(GameValue::Float(c.post_process.saturation as f64)),
        "post_process.vignette_intensity" => Ok(GameValue::Float(c.post_process.vignette_intensity as f64)),
        "post_process.tone_map_mode" => Ok(GameValue::Int(c.post_process.tone_map_mode as i64)),
        "post_process.sharpen_strength" => Ok(GameValue::Float(c.post_process.sharpen_strength as f64)),
        "post_process.dof_enabled" => Ok(GameValue::Bool(c.post_process.dof_enabled)),
        "post_process.dof_focus_distance" => Ok(GameValue::Float(c.post_process.dof_focus_distance as f64)),
        "post_process.dof_focus_range" => Ok(GameValue::Float(c.post_process.dof_focus_range as f64)),
        "post_process.dof_max_coc" => Ok(GameValue::Float(c.post_process.dof_max_coc as f64)),
        "post_process.motion_blur_intensity" => Ok(GameValue::Float(c.post_process.motion_blur_intensity as f64)),
        "post_process.god_rays_intensity" => Ok(GameValue::Float(c.post_process.god_rays_intensity as f64)),
        "post_process.grain_intensity" => Ok(GameValue::Float(c.post_process.grain_intensity as f64)),
        "post_process.chromatic_aberration" => Ok(GameValue::Float(c.post_process.chromatic_aberration as f64)),
        "post_process.gi_intensity" => Ok(GameValue::Float(c.post_process.gi_intensity as f64)),
        _ => Err(format!("unknown field '{}' on component 'EnvironmentSettings'", field_name)),
    }
}

/// Set a field on `EnvironmentSettings` using dot-notation.
fn env_settings_set_field(
    c: &mut EnvironmentSettings,
    field_name: &str,
    value: GameValue,
) -> Result<(), String> {
    match field_name {
        // Fog
        "fog.enabled" => { c.fog.enabled = value.as_bool().ok_or("type mismatch")?; }
        "fog.density" => { c.fog.density = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.color" => {
            if let Some(rgba) = value.as_color() {
                c.fog.color = [rgba[0], rgba[1], rgba[2]];
            } else if let Some(v) = value.as_vec3() {
                c.fog.color = [v.x, v.y, v.z];
            } else { return Err("type mismatch".into()); }
        }
        "fog.start_distance" => { c.fog.start_distance = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.end_distance" => { c.fog.end_distance = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.height_falloff" => { c.fog.height_falloff = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.ambient_dust_density" => { c.fog.ambient_dust_density = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.dust_asymmetry" => { c.fog.dust_asymmetry = value.as_float().ok_or("type mismatch")? as f32; }
        "fog.vol_ambient_color" => {
            if let Some(rgba) = value.as_color() {
                c.fog.vol_ambient_color = [rgba[0], rgba[1], rgba[2]];
            } else if let Some(v) = value.as_vec3() {
                c.fog.vol_ambient_color = [v.x, v.y, v.z];
            } else { return Err("type mismatch".into()); }
        }
        "fog.vol_ambient_intensity" => { c.fog.vol_ambient_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        // Atmosphere
        "atmosphere.enabled" => { c.atmosphere.enabled = value.as_bool().ok_or("type mismatch")?; }
        "atmosphere.rayleigh_scale" => { c.atmosphere.rayleigh_scale = value.as_float().ok_or("type mismatch")? as f32; }
        "atmosphere.mie_scale" => { c.atmosphere.mie_scale = value.as_float().ok_or("type mismatch")? as f32; }
        "atmosphere.sun_direction" => {
            let v = value.as_vec3().ok_or("type mismatch")?;
            c.atmosphere.sun_direction = [v.x, v.y, v.z];
        }
        "atmosphere.sun_intensity" => { c.atmosphere.sun_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "atmosphere.sun_color" => {
            if let Some(rgba) = value.as_color() {
                c.atmosphere.sun_color = [rgba[0], rgba[1], rgba[2]];
            } else if let Some(v) = value.as_vec3() {
                c.atmosphere.sun_color = [v.x, v.y, v.z];
            } else { return Err("type mismatch".into()); }
        }
        // Clouds
        "clouds.enabled" => { c.clouds.enabled = value.as_bool().ok_or("type mismatch")?; }
        "clouds.coverage" => { c.clouds.coverage = value.as_float().ok_or("type mismatch")? as f32; }
        "clouds.density" => { c.clouds.density = value.as_float().ok_or("type mismatch")? as f32; }
        "clouds.altitude" => { c.clouds.altitude = value.as_float().ok_or("type mismatch")? as f32; }
        "clouds.thickness" => { c.clouds.thickness = value.as_float().ok_or("type mismatch")? as f32; }
        "clouds.wind_direction" => {
            let v = value.as_vec3().ok_or("type mismatch")?;
            c.clouds.wind_direction = [v.x, v.y, v.z];
        }
        "clouds.wind_speed" => { c.clouds.wind_speed = value.as_float().ok_or("type mismatch")? as f32; }
        // Post-process
        "post_process.bloom_enabled" => { c.post_process.bloom_enabled = value.as_bool().ok_or("type mismatch")?; }
        "post_process.bloom_intensity" => { c.post_process.bloom_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.bloom_threshold" => { c.post_process.bloom_threshold = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.exposure" => { c.post_process.exposure = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.contrast" => { c.post_process.contrast = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.saturation" => { c.post_process.saturation = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.vignette_intensity" => { c.post_process.vignette_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.tone_map_mode" => { c.post_process.tone_map_mode = value.as_int().ok_or("type mismatch")? as u32; }
        "post_process.sharpen_strength" => { c.post_process.sharpen_strength = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.dof_enabled" => { c.post_process.dof_enabled = value.as_bool().ok_or("type mismatch")?; }
        "post_process.dof_focus_distance" => { c.post_process.dof_focus_distance = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.dof_focus_range" => { c.post_process.dof_focus_range = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.dof_max_coc" => { c.post_process.dof_max_coc = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.motion_blur_intensity" => { c.post_process.motion_blur_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.god_rays_intensity" => { c.post_process.god_rays_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.grain_intensity" => { c.post_process.grain_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.chromatic_aberration" => { c.post_process.chromatic_aberration = value.as_float().ok_or("type mismatch")? as f32; }
        "post_process.gi_intensity" => { c.post_process.gi_intensity = value.as_float().ok_or("type mismatch")? as f32; }
        _ => return Err(format!("unknown field '{}' on component 'EnvironmentSettings'", field_name)),
    }
    Ok(())
}

/// Create the `ComponentEntry` for `EnvironmentSettings`.
fn env_settings_entry() -> ComponentEntry {
    ComponentEntry {
        name: "EnvironmentSettings",
        meta: &ENV_SETTINGS_FIELDS,
        serialize: |world, entity| {
            world.get::<&EnvironmentSettings>(entity).ok()
                .map(|c| ron::to_string(&*c).unwrap())
        },
        deserialize_insert: |world, entity, ron_str| {
            let c: EnvironmentSettings = ron::from_str(ron_str).map_err(|e| e.to_string())?;
            world.insert_one(entity, c).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&EnvironmentSettings>(entity).is_ok(),
        remove: |world, entity| { let _ = world.remove_one::<EnvironmentSettings>(entity); },
        get_field: |world, entity, field_name| {
            let c = world.get::<&EnvironmentSettings>(entity)
                .map_err(|_| "entity does not have 'EnvironmentSettings'".to_string())?;
            env_settings_get_field(&*c, field_name)
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world.get::<&mut EnvironmentSettings>(entity)
                .map_err(|_| "entity does not have 'EnvironmentSettings'".to_string())?;
            env_settings_set_field(&mut *c, field_name, value)
        },
    }
}

static EDITOR_CAMERA_MARKER_FIELDS: [FieldMeta; 0] = [];

fn editor_camera_marker_entry() -> ComponentEntry {
    ComponentEntry {
        name: "EditorCameraMarker",
        meta: &EDITOR_CAMERA_MARKER_FIELDS,
        serialize: |world, entity| {
            world.get::<&EditorCameraMarker>(entity).ok()
                .map(|_| "()".to_string())
        },
        deserialize_insert: |world, entity, _ron_str| {
            world.insert_one(entity, EditorCameraMarker).map_err(|e| e.to_string())?;
            Ok(())
        },
        has: |world, entity| world.get::<&EditorCameraMarker>(entity).is_ok(),
        remove: |world, entity| { let _ = world.remove_one::<EditorCameraMarker>(entity); },
        get_field: |_world, _entity, field_name| {
            Err(format!("EditorCameraMarker has no field '{}'", field_name))
        },
        set_field: |_world, _entity, field_name, _value| {
            Err(format!("EditorCameraMarker has no field '{}'", field_name))
        },
    }
}

// ─── Registration ────────────────────────────────────────────────────────

/// All engine component names, for distinguishing engine vs gameplay entries.
pub const ENGINE_COMPONENT_NAMES: &[&str] = &[
    "Transform",
    "CameraComponent",
    "FogVolumeComponent",
    "EditorMetadata",
    "EnvironmentSettings",
    "EditorCameraMarker",
];

/// Register all engine components into the given registry.
///
/// Called once at startup, before any gameplay dylib loads. Engine entries
/// survive `clear_gameplay()` because their names are in `ENGINE_COMPONENT_NAMES`.
pub fn engine_register(registry: &mut GameplayRegistry) {
    let entries = [
        transform_entry(),
        camera_entry(),
        fog_volume_entry(),
        editor_metadata_entry(),
        env_settings_entry(),
        editor_camera_marker_entry(),
    ];
    for entry in entries {
        registry
            .register_component(entry)
            .expect("engine component registration should not conflict");
    }
}

// Note: inventory::submit! requires const construction, which isn't possible
// for ComponentEntry with function pointers. Engine components are registered
// via `engine_register()` called at startup instead.

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};
    use rkf_core::WorldPosition;

    // ── 1. engine_register populates registry ───────────────────────────

    #[test]
    fn engine_register_populates_registry() {
        let mut reg = GameplayRegistry::new();
        engine_register(&mut reg);
        assert_eq!(reg.component_count(), 6);
        assert!(reg.has_component("Transform"));
        assert!(reg.has_component("CameraComponent"));
        assert!(reg.has_component("EditorCameraMarker"));
        assert!(reg.has_component("FogVolumeComponent"));
        assert!(reg.has_component("EditorMetadata"));
        assert!(reg.has_component("EnvironmentSettings"));
    }

    // ── 2. Transform get_field ──────────────────────────────────────────

    #[test]
    fn transform_get_field() {
        let mut world = hecs::World::new();
        let pos = WorldPosition::new(IVec3::new(1, 2, 3), Vec3::new(0.5, 1.0, 1.5));
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let scale = Vec3::new(2.0, 3.0, 4.0);
        let entity = world.spawn((Transform {
            position: pos.clone(),
            rotation: rot,
            scale,
        },));

        let entry = transform_entry();
        let val = (entry.get_field)(&world, entity, "position").unwrap();
        assert_eq!(val.as_world_position(), Some(&pos));

        let val = (entry.get_field)(&world, entity, "rotation").unwrap();
        let q = val.as_quat().unwrap();
        assert!((q.x - rot.x).abs() < 1e-6);

        let val = (entry.get_field)(&world, entity, "scale").unwrap();
        assert_eq!(val.as_vec3(), Some(scale));
    }

    // ── 3. Transform set_field ──────────────────────────────────────────

    #[test]
    fn transform_set_field() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let entry = transform_entry();

        let new_pos = WorldPosition::new(IVec3::new(5, 6, 7), Vec3::ZERO);
        (entry.set_field)(
            &mut world,
            entity,
            "position",
            GameValue::WorldPosition(new_pos.clone()),
        )
        .unwrap();

        let val = (entry.get_field)(&world, entity, "position").unwrap();
        assert_eq!(val.as_world_position(), Some(&new_pos));

        (entry.set_field)(
            &mut world,
            entity,
            "scale",
            GameValue::Vec3(Vec3::new(10.0, 20.0, 30.0)),
        )
        .unwrap();
        let val = (entry.get_field)(&world, entity, "scale").unwrap();
        assert_eq!(val.as_vec3(), Some(Vec3::new(10.0, 20.0, 30.0)));
    }

    // ── 4. CameraComponent get_field ────────────────────────────────────

    #[test]
    fn camera_get_field() {
        let mut world = hecs::World::new();
        let cam = CameraComponent {
            fov_degrees: 90.0,
            near: 0.5,
            far: 500.0,
            active: true,
            label: "Main".to_string(),
            yaw: 45.0,
            pitch: -15.0,
            ..Default::default()
        };
        let entity = world.spawn((cam,));
        let entry = camera_entry();

        let val = (entry.get_field)(&world, entity, "fov_degrees").unwrap();
        assert!((val.as_float().unwrap() - 90.0).abs() < 1e-6);

        let val = (entry.get_field)(&world, entity, "active").unwrap();
        assert_eq!(val.as_bool(), Some(true));

        let val = (entry.get_field)(&world, entity, "label").unwrap();
        assert_eq!(val.as_string(), Some("Main"));

        let val = (entry.get_field)(&world, entity, "yaw").unwrap();
        assert!((val.as_float().unwrap() - 45.0).abs() < 1e-6);
    }

    // ── 5. CameraComponent set_field ────────────────────────────────────

    #[test]
    fn camera_set_field() {
        let mut world = hecs::World::new();
        let entity = world.spawn((CameraComponent::default(),));
        let entry = camera_entry();

        (entry.set_field)(
            &mut world,
            entity,
            "fov_degrees",
            GameValue::Float(120.0),
        )
        .unwrap();
        let val = (entry.get_field)(&world, entity, "fov_degrees").unwrap();
        assert!((val.as_float().unwrap() - 120.0).abs() < 1e-6);

        (entry.set_field)(
            &mut world,
            entity,
            "label",
            GameValue::String("Cinematic".into()),
        )
        .unwrap();
        let val = (entry.get_field)(&world, entity, "label").unwrap();
        assert_eq!(val.as_string(), Some("Cinematic"));
    }

    // ── 6. FogVolumeComponent get_field ─────────────────────────────────

    #[test]
    fn fog_volume_get_field() {
        let mut world = hecs::World::new();
        let fog = FogVolumeComponent {
            density: 0.7,
            color: [1.0, 0.5, 0.0],
            phase_g: 0.4,
            half_extents: Vec3::new(10.0, 5.0, 3.0),
        };
        let entity = world.spawn((fog,));
        let entry = fog_volume_entry();

        let val = (entry.get_field)(&world, entity, "density").unwrap();
        assert!((val.as_float().unwrap() - 0.7).abs() < 1e-6);

        let val = (entry.get_field)(&world, entity, "half_extents").unwrap();
        assert_eq!(val.as_vec3(), Some(Vec3::new(10.0, 5.0, 3.0)));
    }

    // ── 7. FogVolumeComponent set_field ─────────────────────────────────

    #[test]
    fn fog_volume_set_field() {
        let mut world = hecs::World::new();
        let entity = world.spawn((FogVolumeComponent::default(),));
        let entry = fog_volume_entry();

        (entry.set_field)(&mut world, entity, "density", GameValue::Float(1.5)).unwrap();
        let val = (entry.get_field)(&world, entity, "density").unwrap();
        assert!((val.as_float().unwrap() - 1.5).abs() < 1e-6);

        (entry.set_field)(
            &mut world,
            entity,
            "half_extents",
            GameValue::Vec3(Vec3::new(20.0, 10.0, 5.0)),
        )
        .unwrap();
        let val = (entry.get_field)(&world, entity, "half_extents").unwrap();
        assert_eq!(val.as_vec3(), Some(Vec3::new(20.0, 10.0, 5.0)));
    }

    // ── 8. EditorMetadata get_field ─────────────────────────────────────

    #[test]
    fn editor_metadata_get_field() {
        let mut world = hecs::World::new();
        let meta = EditorMetadata {
            name: "Guard".into(),
            tags: vec!["npc".into()],
            locked: true,
        };
        let entity = world.spawn((meta,));
        let entry = editor_metadata_entry();

        let val = (entry.get_field)(&world, entity, "name").unwrap();
        assert_eq!(val.as_string(), Some("Guard"));

        let val = (entry.get_field)(&world, entity, "locked").unwrap();
        assert_eq!(val.as_bool(), Some(true));
    }

    // ── 9. EditorMetadata set_field ─────────────────────────────────────

    #[test]
    fn editor_metadata_set_field() {
        let mut world = hecs::World::new();
        let entity = world.spawn((EditorMetadata::default(),));
        let entry = editor_metadata_entry();

        (entry.set_field)(
            &mut world,
            entity,
            "name",
            GameValue::String("Tower".into()),
        )
        .unwrap();
        let val = (entry.get_field)(&world, entity, "name").unwrap();
        assert_eq!(val.as_string(), Some("Tower"));

        (entry.set_field)(&mut world, entity, "locked", GameValue::Bool(true)).unwrap();
        let val = (entry.get_field)(&world, entity, "locked").unwrap();
        assert_eq!(val.as_bool(), Some(true));
    }

    // ── 10. has and remove ──────────────────────────────────────────────

    #[test]
    fn has_and_remove() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let entry = transform_entry();

        assert!((entry.has)(&world, entity));
        (entry.remove)(&mut world, entity);
        assert!(!(entry.has)(&world, entity));
    }

    // ── 11. serialize / deserialize_insert round-trip ────────────────────

    #[test]
    fn transform_serialize_roundtrip() {
        let mut world = hecs::World::new();
        let pos = WorldPosition::new(IVec3::new(1, 2, 3), Vec3::new(0.5, 1.0, 1.5));
        let rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        let entity = world.spawn((Transform {
            position: pos.clone(),
            rotation: rot,
            scale: Vec3::new(2.0, 3.0, 4.0),
        },));
        let entry = transform_entry();

        let ron_str = (entry.serialize)(&world, entity).unwrap();

        // Deserialize onto a new entity.
        let entity2 = world.spawn(());
        (entry.deserialize_insert)(&mut world, entity2, &ron_str).unwrap();

        let val = (entry.get_field)(&world, entity2, "position").unwrap();
        assert_eq!(val.as_world_position(), Some(&pos));

        let val = (entry.get_field)(&world, entity2, "scale").unwrap();
        assert_eq!(val.as_vec3(), Some(Vec3::new(2.0, 3.0, 4.0)));
    }

    #[test]
    fn camera_serialize_roundtrip() {
        let mut world = hecs::World::new();
        let cam = CameraComponent {
            fov_degrees: 90.0,
            near: 0.5,
            far: 500.0,
            active: true,
            label: "Cinematic".into(),
            yaw: 30.0,
            pitch: -10.0,
            ..Default::default()
        };
        let entity = world.spawn((cam,));
        let entry = camera_entry();

        let ron_str = (entry.serialize)(&world, entity).unwrap();
        let entity2 = world.spawn(());
        (entry.deserialize_insert)(&mut world, entity2, &ron_str).unwrap();

        let val = (entry.get_field)(&world, entity2, "fov_degrees").unwrap();
        assert!((val.as_float().unwrap() - 90.0).abs() < 1e-6);

        let val = (entry.get_field)(&world, entity2, "label").unwrap();
        assert_eq!(val.as_string(), Some("Cinematic"));
    }

    // ── 12. unknown field errors ────────────────────────────────────────

    #[test]
    fn unknown_field_errors() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let entry = transform_entry();

        let err = (entry.get_field)(&world, entity, "nonexistent").unwrap_err();
        assert!(err.contains("unknown field"));

        let err = (entry.set_field)(
            &mut world,
            entity,
            "nonexistent",
            GameValue::Float(1.0),
        )
        .unwrap_err();
        assert!(err.contains("unknown field"));
    }

    // ── 13. type mismatch errors ────────────────────────────────────────

    #[test]
    fn type_mismatch_errors() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Transform::default(),));
        let entry = transform_entry();

        // Try setting position with a Float instead of WorldPosition.
        let err = (entry.set_field)(
            &mut world,
            entity,
            "position",
            GameValue::Float(1.0),
        )
        .unwrap_err();
        assert!(err.contains("type mismatch"));
    }

    // ── 14. field meta correctness ──────────────────────────────────────

    #[test]
    fn field_meta_correctness() {
        let entry = transform_entry();
        assert_eq!(entry.meta.len(), 3);
        assert_eq!(entry.meta[0].name, "position");
        assert_eq!(entry.meta[0].field_type, FieldType::WorldPosition);
        assert_eq!(entry.meta[1].name, "rotation");
        assert_eq!(entry.meta[1].field_type, FieldType::Quat);
        assert_eq!(entry.meta[2].name, "scale");
        assert_eq!(entry.meta[2].field_type, FieldType::Vec3);

        let entry = camera_entry();
        assert_eq!(entry.meta.len(), 8);
        assert_eq!(entry.meta[0].name, "fov_degrees");
        assert_eq!(entry.meta[0].field_type, FieldType::Float);
        assert!(entry.meta[0].range.is_some());

        let entry = fog_volume_entry();
        assert_eq!(entry.meta.len(), 3);

        let entry = editor_metadata_entry();
        assert_eq!(entry.meta.len(), 2);
    }

    // ── 15. serialize returns None for missing component ────────────────

    #[test]
    fn serialize_returns_none_for_missing() {
        let world = hecs::World::new();
        let entity = world.reserve_entity();
        let entry = transform_entry();
        assert!((entry.serialize)(&world, entity).is_none());
    }
}
