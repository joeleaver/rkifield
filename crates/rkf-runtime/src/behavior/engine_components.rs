//! Manual `ComponentEntry` registrations for engine components.
//!
//! Engine components (`Transform`, `CameraComponent`, `FogVolumeComponent`,
//! `EditorMetadata`) are defined in `rkf-runtime::components` and predate the
//! `#[component]` proc macro. This module creates hand-written `ComponentEntry`
//! values with correct field metadata, get/set, serialize/deserialize, has, and
//! remove implementations.

use super::game_value::GameValue;
use super::registry::{ComponentEntry, FieldMeta, FieldType, GameplayRegistry};
use crate::components::{CameraComponent, EditorMetadata, FogVolumeComponent, Transform};

// ─── Field metadata (static) ─────────────────────────────────────────────

static TRANSFORM_FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "position",
        field_type: FieldType::WorldPosition,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "rotation",
        field_type: FieldType::Quat,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "scale",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
];

static CAMERA_FIELDS: [FieldMeta; 7] = [
    FieldMeta {
        name: "fov_degrees",
        field_type: FieldType::Float,
        transient: false,
        range: Some((1.0, 179.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "near",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.001, 100.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "far",
        field_type: FieldType::Float,
        transient: false,
        range: Some((1.0, 100000.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "active",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "label",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "yaw",
        field_type: FieldType::Float,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "pitch",
        field_type: FieldType::Float,
        transient: false,
        range: None,
        default: None,
        persist: false,
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
    },
    FieldMeta {
        name: "phase_g",
        field_type: FieldType::Float,
        transient: false,
        range: Some((-1.0, 1.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "half_extents",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
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
    },
    FieldMeta {
        name: "locked",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: false,
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

// ─── Registration ────────────────────────────────────────────────────────

/// All engine component names, for distinguishing engine vs gameplay entries.
pub const ENGINE_COMPONENT_NAMES: &[&str] = &[
    "Transform",
    "CameraComponent",
    "FogVolumeComponent",
    "EditorMetadata",
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
        assert_eq!(reg.component_count(), 4);
        assert!(reg.has_component("Transform"));
        assert!(reg.has_component("CameraComponent"));
        assert!(reg.has_component("FogVolumeComponent"));
        assert!(reg.has_component("EditorMetadata"));
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
        assert_eq!(entry.meta.len(), 7);
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
