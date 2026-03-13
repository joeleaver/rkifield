//! Game crate scaffolding for the behavior system.
//!
//! [`scaffold_game_crate`] creates a complete starter game crate with example
//! components, systems, and blueprints — ready to compile as a hot-reloadable
//! cdylib.

use std::fs;
use std::path::{Path, PathBuf};

// ─── ScaffoldError ──────────────────────────────────────────────────────────

/// Errors that can occur during game crate scaffolding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaffoldError {
    /// The target directory already exists.
    AlreadyExists,
    /// An I/O error occurred.
    IoError(String),
}

impl std::fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyExists => write!(f, "game crate directory already exists"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for ScaffoldError {}

impl From<std::io::Error> for ScaffoldError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

// ─── scaffold_game_crate ────────────────────────────────────────────────────

/// Create a complete starter game crate with example components, systems, and
/// blueprints.
///
/// Creates the following structure under `project_dir/<crate_name>`:
/// ```text
/// <crate_name>/
///   Cargo.toml                    — cdylib, depends on rkf-runtime + glam + hecs
///   src/
///     lib.rs                      — #[no_mangle] rkf_register + rkf_abi_version
///     blueprints.rs               — guard_blueprint, collectible_blueprint
///     components/
///       mod.rs                    — pub mod per-component, re-exports
///       health.rs                 — Health struct + field metadata + ComponentEntry
///       patrol.rs                 — Patrol struct + entry
///       collectible.rs            — Collectible struct + entry
///       door_state.rs             — DoorState + entry
///       guard_ai.rs               — GuardState enum + GuardAi struct + entry
///       camera_follow.rs          — CameraFollow struct + entry
///       float_bob.rs              — FloatBob struct + entry
///       spin.rs                   — Spin struct + entry
///     systems/
///       mod.rs                    — pub mod per-system, re-exports
///       patrol.rs                 — patrol_system fn
///       door.rs                   — door_system fn
///       death.rs                  — death_system fn
///       collectible.rs            — collectible_system fn
///       guard_ai.rs               — guard_ai_system fn
///       camera_follow.rs          — camera_follow_system fn
///       float_bob.rs              — float_bob_system fn
///       spin.rs                   — spin_system fn
/// ```
///
/// `engine_root` is the workspace root (parent of `crates/`). The generated
/// `Cargo.toml` uses the absolute path to `rkf-runtime` within that workspace.
///
/// Returns the path to the created crate directory.
pub fn scaffold_game_crate(
    project_dir: &Path,
    crate_name: &str,
    engine_root: &Path,
) -> Result<PathBuf, ScaffoldError> {
    let crate_dir = project_dir.join(crate_name);

    if crate_dir.exists() {
        return Err(ScaffoldError::AlreadyExists);
    }

    // Create directory structure.
    let src_dir = crate_dir.join("src");
    let components_dir = src_dir.join("components");
    let systems_dir = src_dir.join("systems");

    fs::create_dir_all(&components_dir)?;
    fs::create_dir_all(&systems_dir)?;

    let runtime_path = engine_root.join("crates/rkf-runtime");
    let runtime_path_str = runtime_path.display().to_string();

    // Write all files.
    fs::write(crate_dir.join("Cargo.toml"), gen_cargo_toml(crate_name, &runtime_path_str))?;
    fs::write(src_dir.join("lib.rs"), gen_lib_rs())?;
    fs::write(src_dir.join("blueprints.rs"), gen_blueprints_rs())?;

    // Components
    fs::write(components_dir.join("mod.rs"), gen_components_mod_rs())?;
    fs::write(components_dir.join("health.rs"), gen_component_health())?;
    fs::write(components_dir.join("patrol.rs"), gen_component_patrol())?;
    fs::write(components_dir.join("collectible.rs"), gen_component_collectible())?;
    fs::write(components_dir.join("door_state.rs"), gen_component_door_state())?;
    fs::write(components_dir.join("guard_ai.rs"), gen_component_guard_ai())?;
    fs::write(components_dir.join("camera_follow.rs"), gen_component_camera_follow())?;
    fs::write(components_dir.join("float_bob.rs"), gen_component_float_bob())?;
    fs::write(components_dir.join("spin.rs"), gen_component_spin())?;

    // Systems
    fs::write(systems_dir.join("mod.rs"), gen_systems_mod_rs())?;
    fs::write(systems_dir.join("patrol.rs"), gen_system_patrol())?;
    fs::write(systems_dir.join("door.rs"), gen_system_door())?;
    fs::write(systems_dir.join("death.rs"), gen_system_death())?;
    fs::write(systems_dir.join("collectible.rs"), gen_system_collectible())?;
    fs::write(systems_dir.join("guard_ai.rs"), gen_system_guard_ai())?;
    fs::write(systems_dir.join("camera_follow.rs"), gen_system_camera_follow())?;
    fs::write(systems_dir.join("float_bob.rs"), gen_system_float_bob())?;
    fs::write(systems_dir.join("spin.rs"), gen_system_spin())?;

    Ok(crate_dir)
}

// ─── Template generators ────────────────────────────────────────────────────

fn gen_cargo_toml(crate_name: &str, runtime_path: &str) -> String {
    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
rkf-runtime = {{ path = "{runtime_path}" }}
glam = {{ version = "0.29", features = ["bytemuck", "serde"] }}
hecs = "0.10"
serde = {{ version = "1", features = ["derive"] }}
"#
    )
}

fn gen_lib_rs() -> String {
    r#"//! Game behavior crate — hot-reloadable gameplay logic.
//!
//! This crate is compiled as a cdylib and loaded by the engine at runtime.
//! It registers components, systems, and blueprints via `rkf_register`.

mod blueprints;
mod components;
mod systems;

use rkf_runtime::behavior::{GameplayRegistry, Phase, SystemMeta};

/// Called by the engine on load/reload to register components and systems.
#[no_mangle]
pub extern "C" fn rkf_register(registry: &mut GameplayRegistry) {
    // Register all components.
    let entries = [
        components::health::entry(),
        components::patrol::entry(),
        components::collectible::entry(),
        components::door_state::entry(),
        components::guard_ai::entry(),
        components::camera_follow::entry(),
        components::float_bob::entry(),
        components::spin::entry(),
    ];
    for e in entries {
        registry
            .register_component(e)
            .expect("component registration should not conflict");
    }

    // Register systems.
    registry.register_system(SystemMeta {
        name: "patrol_system",
        module_path: "systems::patrol::patrol_system",
        phase: Phase::Update,
        after: &[],
        before: &["death_system"],
        fn_ptr: systems::patrol::patrol_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "door_system",
        module_path: "systems::door::door_system",
        phase: Phase::Update,
        after: &[],
        before: &[],
        fn_ptr: systems::door::door_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "death_system",
        module_path: "systems::death::death_system",
        phase: Phase::Update,
        after: &["patrol_system"],
        before: &[],
        fn_ptr: systems::death::death_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "collectible_system",
        module_path: "systems::collectible::collectible_system",
        phase: Phase::LateUpdate,
        after: &[],
        before: &[],
        fn_ptr: systems::collectible::collectible_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "guard_ai_system",
        module_path: "systems::guard_ai::guard_ai_system",
        phase: Phase::Update,
        after: &[],
        before: &["death_system"],
        fn_ptr: systems::guard_ai::guard_ai_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "camera_follow_system",
        module_path: "systems::camera_follow::camera_follow_system",
        phase: Phase::LateUpdate,
        after: &[],
        before: &[],
        fn_ptr: systems::camera_follow::camera_follow_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "float_bob_system",
        module_path: "systems::float_bob::float_bob_system",
        phase: Phase::Update,
        after: &[],
        before: &[],
        fn_ptr: systems::float_bob::float_bob_system as *const (),
    });

    registry.register_system(SystemMeta {
        name: "spin_system",
        module_path: "systems::spin::spin_system",
        phase: Phase::LateUpdate,
        after: &[],
        before: &[],
        fn_ptr: systems::spin::spin_system as *const (),
    });

    // Register blueprints.
    registry.blueprint_catalog.insert(blueprints::guard_blueprint());
    registry.blueprint_catalog.insert(blueprints::collectible_blueprint());
}

/// ABI version check — engine compares this against its own version.
#[no_mangle]
pub extern "C" fn rkf_abi_version() -> *const u8 {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr()
}
"#
    .to_string()
}

fn gen_blueprints_rs() -> String {
    r#"//! Blueprint definitions for pre-configured entity archetypes.

use rkf_runtime::behavior::Blueprint;
use std::collections::HashMap;

/// Create a Guard blueprint: Health + Patrol components.
pub fn guard_blueprint() -> Blueprint {
    let mut components = HashMap::new();
    components.insert(
        "Health".to_owned(),
        "(current: 100.0, max: 100.0)".to_owned(),
    );
    components.insert(
        "Patrol".to_owned(),
        "(speed: 5.0, current_index: 0)".to_owned(),
    );
    Blueprint {
        name: "Guard".to_owned(),
        components,
    }
}

/// Create a Collectible blueprint: Collectible component.
pub fn collectible_blueprint() -> Blueprint {
    let mut components = HashMap::new();
    components.insert(
        "Collectible".to_owned(),
        "(value: 10, spin_speed: 1.0)".to_owned(),
    );
    Blueprint {
        name: "Collectible".to_owned(),
        components,
    }
}
"#
    .to_string()
}

fn gen_components_mod_rs() -> String {
    r#"//! Game-specific components.

pub mod health;
pub mod patrol;
pub mod collectible;
pub mod door_state;
pub mod guard_ai;
pub mod camera_follow;
pub mod float_bob;
pub mod spin;

pub use health::Health;
pub use patrol::Patrol;
pub use collectible::Collectible;
pub use door_state::DoorState;
pub use guard_ai::{GuardAi, GuardState};
pub use camera_follow::CameraFollow;
pub use float_bob::FloatBob;
pub use spin::Spin;
"#
    .to_string()
}

fn gen_systems_mod_rs() -> String {
    r#"//! Game-specific systems.

pub mod patrol;
pub mod door;
pub mod death;
pub mod collectible;
pub mod guard_ai;
pub mod camera_follow;
pub mod float_bob;
pub mod spin;
"#
    .to_string()
}

// ─── Component templates ────────────────────────────────────────────────────

fn gen_component_health() -> String {
    r#"//! Health component — hit points for a damageable entity.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Hit points for a damageable entity.
pub struct Health {
    pub current: f32,
    pub max: f32,
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
                .map(|c| format!("(current: {}, max: {})", c.current, c.max))
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(entity, Health { current: 100.0, max: 100.0 })
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
"#
    .to_string()
}

fn gen_component_patrol() -> String {
    r#"//! Patrol component — move between waypoints at a given speed.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Patrol behavior: move between waypoints at a given speed.
pub struct Patrol {
    pub waypoints: Vec<glam::Vec3>,
    pub speed: f32,
    pub current_index: usize,
}

static FIELDS: [FieldMeta; 3] = [
    FieldMeta {
        name: "waypoints",
        field_type: FieldType::List,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "speed",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 100.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "current_index",
        field_type: FieldType::Int,
        transient: true,
        range: None,
        default: None,
        persist: false,
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
                .map(|c| format!("(speed: {}, current_index: {})", c.speed, c.current_index))
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(
                    entity,
                    Patrol {
                        waypoints: Vec::new(),
                        speed: 5.0,
                        current_index: 0,
                    },
                )
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
"#
    .to_string()
}

fn gen_component_collectible() -> String {
    r#"//! Collectible component — a collectible item with a value and spinning animation.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// A collectible item with a value and spinning animation.
pub struct Collectible {
    pub value: i32,
    pub spin_speed: f32,
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
                .map(|c| format!("(value: {}, spin_speed: {})", c.value, c.spin_speed))
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(
                    entity,
                    Collectible {
                        value: 10,
                        spin_speed: 1.0,
                    },
                )
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
"#
    .to_string()
}

fn gen_component_door_state() -> String {
    r#"//! DoorState component — a door that can be opened, optionally requiring a key.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// A door that can be opened, optionally requiring a key.
pub struct DoorState {
    pub open: bool,
    pub key_required: Option<String>,
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "open",
        field_type: FieldType::Bool,
        transient: false,
        range: None,
        default: None,
        persist: true,
    },
    FieldMeta {
        name: "key_required",
        field_type: FieldType::String,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "DoorState",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&DoorState>(entity)
                .ok()
                .map(|c| {
                    let key = match &c.key_required {
                        Some(k) => format!("Some(\"{}\")", k),
                        None => "None".to_string(),
                    };
                    format!("(open: {}, key_required: {})", c.open, key)
                })
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(
                    entity,
                    DoorState {
                        open: false,
                        key_required: None,
                    },
                )
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&DoorState>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<DoorState>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&DoorState>(entity)
                .map_err(|_| "entity does not have component 'DoorState'".to_string())?;
            match field_name {
                "open" => Ok(GameValue::Bool(c.open)),
                "key_required" => Ok(GameValue::String(
                    c.key_required.clone().unwrap_or_default(),
                )),
                _ => Err(format!(
                    "unknown field '{}' on component 'DoorState'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut DoorState>(entity)
                .map_err(|_| "entity does not have component 'DoorState'".to_string())?;
            match field_name {
                "open" => match value {
                    GameValue::Bool(b) => c.open = b,
                    _ => return Err("type mismatch for field 'open'".into()),
                },
                "key_required" => match value {
                    GameValue::String(s) => {
                        c.key_required = if s.is_empty() { None } else { Some(s) };
                    }
                    _ => return Err("type mismatch for field 'key_required'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'DoorState'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}
"#
    .to_string()
}

fn gen_component_guard_ai() -> String {
    r#"//! GuardAi component — guard AI state machine with patrol/chase/return behavior.

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
pub struct GuardAi {
    pub state: GuardState,
    pub patrol_origin: glam::Vec3,
    pub patrol_radius: f32,
    pub chase_speed: f32,
    pub detection_range: f32,
}

static FIELDS: [FieldMeta; 5] = [
    FieldMeta {
        name: "state",
        field_type: FieldType::String,
        transient: true,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "patrol_origin",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "patrol_radius",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 100.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "chase_speed",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 50.0)),
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "detection_range",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 200.0)),
        default: None,
        persist: false,
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
                .map(|c| {
                    format!(
                        "(state: {:?}, patrol_origin: ({}, {}, {}), patrol_radius: {}, chase_speed: {}, detection_range: {})",
                        c.state, c.patrol_origin.x, c.patrol_origin.y, c.patrol_origin.z,
                        c.patrol_radius, c.chase_speed, c.detection_range
                    )
                })
        },
        deserialize_insert: |world, entity, _ron_str| {
            world
                .insert_one(
                    entity,
                    GuardAi {
                        state: GuardState::Patrol,
                        patrol_origin: glam::Vec3::ZERO,
                        patrol_radius: 10.0,
                        chase_speed: 8.0,
                        detection_range: 15.0,
                    },
                )
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
"#
    .to_string()
}

fn gen_component_camera_follow() -> String {
    r#"//! CameraFollow component — smoothly tracks a target entity's position.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Camera follow component: smoothly tracks a target entity's position.
pub struct CameraFollow {
    pub target: hecs::Entity,
    pub offset: glam::Vec3,
    pub smoothing: f32,
}

static FIELDS: [FieldMeta; 2] = [
    FieldMeta {
        name: "offset",
        field_type: FieldType::Vec3,
        transient: false,
        range: None,
        default: None,
        persist: false,
    },
    FieldMeta {
        name: "smoothing",
        field_type: FieldType::Float,
        transient: false,
        range: Some((0.0, 1.0)),
        default: None,
        persist: false,
    },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "CameraFollow",
        meta: &FIELDS,
        serialize: |world, entity| {
            world
                .get::<&CameraFollow>(entity)
                .ok()
                .map(|c| {
                    format!(
                        "(offset: ({}, {}, {}), smoothing: {})",
                        c.offset.x, c.offset.y, c.offset.z, c.smoothing
                    )
                })
        },
        deserialize_insert: |world, entity, _ron_str| {
            let placeholder = world.reserve_entity();
            world
                .insert_one(
                    entity,
                    CameraFollow {
                        target: placeholder,
                        offset: glam::Vec3::new(0.0, 5.0, -10.0),
                        smoothing: 0.1,
                    },
                )
                .map_err(|e| e.to_string())
        },
        has: |world, entity| world.get::<&CameraFollow>(entity).is_ok(),
        remove: |world, entity| {
            let _ = world.remove_one::<CameraFollow>(entity);
        },
        get_field: |world, entity, field_name| {
            let c = world
                .get::<&CameraFollow>(entity)
                .map_err(|_| "entity does not have component 'CameraFollow'".to_string())?;
            match field_name {
                "offset" => Ok(GameValue::Vec3(c.offset)),
                "smoothing" => Ok(GameValue::Float(c.smoothing as f64)),
                _ => Err(format!(
                    "unknown field '{}' on component 'CameraFollow'",
                    field_name
                )),
            }
        },
        set_field: |world, entity, field_name, value| {
            let mut c = world
                .get::<&mut CameraFollow>(entity)
                .map_err(|_| "entity does not have component 'CameraFollow'".to_string())?;
            match field_name {
                "offset" => match value {
                    GameValue::Vec3(v) => c.offset = v,
                    _ => return Err("type mismatch for field 'offset'".into()),
                },
                "smoothing" => match value {
                    GameValue::Float(f) => c.smoothing = f as f32,
                    _ => return Err("type mismatch for field 'smoothing'".into()),
                },
                _ => {
                    return Err(format!(
                        "unknown field '{}' on component 'CameraFollow'",
                        field_name
                    ))
                }
            }
            Ok(())
        },
    }
}
"#
    .to_string()
}

fn gen_component_float_bob() -> String {
    r#"//! FloatBob component — makes an entity bob up and down sinusoidally.

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
"#
    .to_string()
}

fn gen_component_spin() -> String {
    r#"//! Spin component — rotates an entity around the Y axis.

use rkf_runtime::behavior::{ComponentEntry, FieldMeta, FieldType, GameValue};

/// Spin component: rotates an entity around the Y axis.
pub struct Spin {
    /// Rotation speed in radians per second.
    pub speed: f32,
}

static FIELDS: [FieldMeta; 1] = [
    FieldMeta { name: "speed", field_type: FieldType::Float, transient: false, range: Some((-10.0, 10.0)), default: None, persist: true },
];

pub fn entry() -> ComponentEntry {
    ComponentEntry {
        name: "Spin",
        meta: &FIELDS,
        serialize: |world, entity| {
            world.get::<&Spin>(entity).ok().map(|c| format!("(speed: {})", c.speed))
        },
        deserialize_insert: |world, entity, _ron_str| {
            world.insert_one(entity, Spin { speed: 1.0 }).map_err(|e| e.to_string())
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
"#
    .to_string()
}

// ─── System templates ───────────────────────────────────────────────────────

fn gen_system_patrol() -> String {
    r#"//! Patrol system — moves entities with Patrol along their waypoints.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Patrol;

/// Moves entities with `Patrol` along their waypoints.
pub fn patrol_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let updates: Vec<_> = ctx
        .query::<(&Patrol,)>()
        .iter()
        .filter_map(|(entity, (patrol,))| {
            if patrol.waypoints.is_empty() {
                return None;
            }
            let next_index = (patrol.current_index + 1) % patrol.waypoints.len();
            Some((entity, next_index, patrol.speed * dt))
        })
        .collect();

    for (entity, next_index, _step) in updates {
        ctx.insert(
            entity,
            Patrol {
                waypoints: Vec::new(),
                speed: 0.0,
                current_index: next_index,
            },
        );
    }
}
"#
    .to_string()
}

fn gen_system_door() -> String {
    r#"//! Door system — checks for interact events and toggles door state.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::DoorState;

/// Checks for interact events and toggles door state.
pub fn door_system(ctx: &mut SystemContext) {
    let doors: Vec<_> = ctx
        .query::<(&DoorState,)>()
        .iter()
        .map(|(entity, (door,))| (entity, door.open))
        .collect();

    let has_interact = ctx.store_ref().events("interact").next().is_some();
    if !has_interact {
        return;
    }

    for (entity, was_open) in doors {
        ctx.insert(
            entity,
            DoorState {
                open: !was_open,
                key_required: None,
            },
        );
    }
}
"#
    .to_string()
}

fn gen_system_death() -> String {
    r#"//! Death system — despawns entities with Health <= 0.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Health;

/// Checks entities with Health <= 0 and queues them for despawn.
pub fn death_system(ctx: &mut SystemContext) {
    let dead: Vec<_> = ctx
        .query::<(&Health,)>()
        .iter()
        .filter(|(_, (h,))| h.current <= 0.0)
        .map(|(entity, _)| entity)
        .collect();

    for entity in dead {
        ctx.despawn(entity);
    }
}
"#
    .to_string()
}

fn gen_system_collectible() -> String {
    r#"//! Collectible system — counts active collectibles and stores the count.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Collectible;

/// Spin animation for collectibles (increments a counter in the store).
pub fn collectible_system(ctx: &mut SystemContext) {
    let _dt = ctx.delta_time();
    let count = ctx.query::<(&Collectible,)>().iter().count();
    if count > 0 {
        ctx.store().set("collectible_count", count as i64);
    }
}
"#
    .to_string()
}

fn gen_system_guard_ai() -> String {
    r#"//! Guard AI system — patrol/chase/return state machine.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::{GuardAi, GuardState};

/// Guard AI state machine: patrol near origin, chase detected targets, return when lost.
pub fn guard_ai_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let total_time = ctx.total_time() as f32;

    // Dummy player target position — in a real game this would come from a Player query.
    let player_pos = glam::Vec3::new(0.0, 0.0, 0.0);

    let guards: Vec<_> = ctx
        .query::<(&GuardAi, &Transform)>()
        .iter()
        .map(|(entity, (ai, transform))| {
            let pos = transform.position.local;
            (entity, ai.state, ai.patrol_origin, ai.patrol_radius, ai.chase_speed, ai.detection_range, pos)
        })
        .collect();

    for (entity, state, patrol_origin, patrol_radius, chase_speed, detection_range, pos) in guards {
        let dist_to_player = (player_pos - pos).length();
        let dist_to_origin = (patrol_origin - pos).length();

        let (new_state, new_pos) = match state {
            GuardState::Patrol => {
                let angle = total_time * 0.5;
                let target = patrol_origin + glam::Vec3::new(
                    angle.cos() * patrol_radius,
                    0.0,
                    angle.sin() * patrol_radius,
                );
                let dir = target - pos;
                let step = if dir.length() > 0.01 {
                    dir.normalize() * chase_speed * 0.5 * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_player < detection_range {
                    (GuardState::Chase, moved)
                } else {
                    (GuardState::Patrol, moved)
                }
            }
            GuardState::Chase => {
                let dir = player_pos - pos;
                let step = if dir.length() > 0.01 {
                    dir.normalize() * chase_speed * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_player > detection_range * 1.5 {
                    (GuardState::Return, moved)
                } else {
                    (GuardState::Chase, moved)
                }
            }
            GuardState::Return => {
                let dir = patrol_origin - pos;
                let step = if dir.length() > 0.5 {
                    dir.normalize() * chase_speed * 0.75 * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_origin < 1.0 {
                    (GuardState::Patrol, moved)
                } else {
                    (GuardState::Return, moved)
                }
            }
        };

        ctx.insert(
            entity,
            GuardAi {
                state: new_state,
                patrol_origin,
                patrol_radius,
                chase_speed,
                detection_range,
            },
        );
        ctx.insert(
            entity,
            Transform {
                position: new_pos.into(),
                ..Transform::default()
            },
        );
    }
}
"#
    .to_string()
}

fn gen_system_camera_follow() -> String {
    r#"//! Camera follow system — smoothly lerps camera toward target entity.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::CameraFollow;

/// Smoothly lerps camera position toward a target entity's position plus offset.
pub fn camera_follow_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();

    let followers: Vec<_> = ctx
        .query::<(&CameraFollow, &Transform)>()
        .iter()
        .map(|(entity, (follow, transform))| {
            (entity, follow.target, follow.offset, follow.smoothing, transform.position.local)
        })
        .collect();

    for (entity, target, offset, smoothing, cam_pos) in followers {
        let target_pos = match ctx.get::<Transform>(target) {
            Ok(t) => t.position.local,
            Err(_) => continue,
        };

        let desired = target_pos + offset;
        let alpha = 1.0 - (-dt / smoothing.max(0.001)).exp();
        let new_pos = cam_pos + (desired - cam_pos) * alpha;

        ctx.insert(
            entity,
            Transform {
                position: new_pos.into(),
                ..Transform::default()
            },
        );
    }
}
"#
    .to_string()
}

fn gen_system_float_bob() -> String {
    r#"//! Float bob system — makes entities with FloatBob oscillate up and down.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::FloatBob;

/// Float bob system: makes entities with FloatBob oscillate up and down.
pub fn float_bob_system(ctx: &mut SystemContext) {
    let total = ctx.total_time() as f32;
    let updates: Vec<_> = ctx
        .query::<(&FloatBob, &Transform)>()
        .iter()
        .map(|(entity, (bob, t))| {
            let current_y = t.position.to_vec3().y;
            let base = bob.base_y.unwrap_or(current_y);
            let y = base + bob.amplitude * (total * bob.frequency * std::f32::consts::TAU + bob.phase).sin();
            let mut pos = t.position.to_vec3();
            pos.y = y;
            (entity, base, Transform { position: pos.into(), rotation: t.rotation, scale: t.scale })
        })
        .collect();
    for (entity, base, new_t) in updates {
        ctx.insert(entity, new_t);
        let needs_init = ctx.get::<FloatBob>(entity).ok().filter(|b| b.base_y.is_none())
            .map(|b| FloatBob { amplitude: b.amplitude, frequency: b.frequency, phase: b.phase, base_y: Some(base) });
        if let Some(updated) = needs_init {
            ctx.insert(entity, updated);
        }
    }
}
"#
    .to_string()
}

fn gen_system_spin() -> String {
    r#"//! Spin system — rotates entities with Spin around the Y axis.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::Spin;

/// Spin system: rotates entities with Spin around the Y axis.
pub fn spin_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let updates: Vec<_> = ctx
        .query::<(&Spin, &Transform)>()
        .iter()
        .map(|(entity, (spin, t))| {
            let new_rot = t.rotation * glam::Quat::from_rotation_y(spin.speed * dt);
            (entity, Transform { position: t.position, rotation: new_rot, scale: t.scale })
        })
        .collect();
    for (entity, new_t) in updates {
        ctx.insert(entity, new_t);
    }
}
"#
    .to_string()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_root() -> &'static Path {
        // crates/rkf-runtime => engine root is two levels up from CARGO_MANIFEST_DIR
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        // CARGO_MANIFEST_DIR = .../rkifield/crates/rkf-runtime
        // engine root = .../rkifield
        manifest.parent().unwrap().parent().unwrap()
    }

    #[test]
    fn scaffold_creates_files() {
        let tmp = tempfile::tempdir().unwrap();
        let result = scaffold_game_crate(tmp.path(), "my-game", engine_root());
        assert!(result.is_ok());
        let crate_dir = result.unwrap();

        assert!(crate_dir.join("Cargo.toml").is_file());
        assert!(crate_dir.join("src/lib.rs").is_file());
        assert!(crate_dir.join("src/blueprints.rs").is_file());
        assert!(crate_dir.join("src/components/mod.rs").is_file());
        assert!(crate_dir.join("src/components/health.rs").is_file());
        assert!(crate_dir.join("src/components/patrol.rs").is_file());
        assert!(crate_dir.join("src/components/collectible.rs").is_file());
        assert!(crate_dir.join("src/components/door_state.rs").is_file());
        assert!(crate_dir.join("src/components/guard_ai.rs").is_file());
        assert!(crate_dir.join("src/components/camera_follow.rs").is_file());
        assert!(crate_dir.join("src/components/float_bob.rs").is_file());
        assert!(crate_dir.join("src/components/spin.rs").is_file());
        assert!(crate_dir.join("src/systems/mod.rs").is_file());
        assert!(crate_dir.join("src/systems/patrol.rs").is_file());
        assert!(crate_dir.join("src/systems/door.rs").is_file());
        assert!(crate_dir.join("src/systems/death.rs").is_file());
        assert!(crate_dir.join("src/systems/collectible.rs").is_file());
        assert!(crate_dir.join("src/systems/guard_ai.rs").is_file());
        assert!(crate_dir.join("src/systems/camera_follow.rs").is_file());
        assert!(crate_dir.join("src/systems/float_bob.rs").is_file());
        assert!(crate_dir.join("src/systems/spin.rs").is_file());
    }

    #[test]
    fn scaffold_already_exists_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let first = scaffold_game_crate(tmp.path(), "my-game", engine_root());
        assert!(first.is_ok());

        let second = scaffold_game_crate(tmp.path(), "my-game", engine_root());
        assert_eq!(second, Err(ScaffoldError::AlreadyExists));
    }

    #[test]
    fn scaffolded_cargo_toml_is_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let crate_dir = scaffold_game_crate(tmp.path(), "my-game", engine_root()).unwrap();
        let content = fs::read_to_string(crate_dir.join("Cargo.toml")).unwrap();

        assert!(content.contains("name = \"my-game\""));
        assert!(content.contains("crate-type = [\"cdylib\"]"));
        assert!(content.contains("rkf-runtime"));
        assert!(content.contains("[package]"));
        assert!(content.contains("[lib]"));
        assert!(content.contains("[dependencies]"));
        assert!(content.contains("glam"));
        assert!(content.contains("hecs"));
        assert!(content.contains("serde"));
    }

    #[test]
    fn scaffolded_lib_rs_has_register_functions() {
        let tmp = tempfile::tempdir().unwrap();
        let crate_dir = scaffold_game_crate(tmp.path(), "my-game", engine_root()).unwrap();
        let content = fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();

        assert!(content.contains("pub extern \"C\" fn rkf_register"));
        assert!(content.contains("pub extern \"C\" fn rkf_abi_version"));
        assert!(content.contains("GameplayRegistry"));
        assert!(content.contains("#[no_mangle]"));
        assert!(content.contains("mod components"));
        assert!(content.contains("mod systems"));
        assert!(content.contains("mod blueprints"));
    }

    #[test]
    fn scaffolded_components_are_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let crate_dir = scaffold_game_crate(tmp.path(), "my-game", engine_root()).unwrap();

        // Check components/mod.rs re-exports all components.
        let mod_rs = fs::read_to_string(crate_dir.join("src/components/mod.rs")).unwrap();
        for name in ["health", "patrol", "collectible", "door_state", "guard_ai", "camera_follow", "float_bob", "spin"] {
            assert!(mod_rs.contains(&format!("pub mod {name};")), "missing pub mod {name}");
        }

        // Check a component file has entry function and struct.
        let health = fs::read_to_string(crate_dir.join("src/components/health.rs")).unwrap();
        assert!(health.contains("pub struct Health"));
        assert!(health.contains("pub fn entry()"));
        assert!(health.contains("ComponentEntry"));
        assert!(health.contains("get_field"));
        assert!(health.contains("set_field"));
    }

    #[test]
    fn scaffolded_systems_are_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let crate_dir = scaffold_game_crate(tmp.path(), "my-game", engine_root()).unwrap();

        // Check systems/mod.rs declares all system modules.
        let mod_rs = fs::read_to_string(crate_dir.join("src/systems/mod.rs")).unwrap();
        for name in ["patrol", "door", "death", "collectible", "guard_ai", "camera_follow", "float_bob", "spin"] {
            assert!(mod_rs.contains(&format!("pub mod {name};")), "missing pub mod {name}");
        }

        // Check a system file has the system function.
        let patrol = fs::read_to_string(crate_dir.join("src/systems/patrol.rs")).unwrap();
        assert!(patrol.contains("pub fn patrol_system"));
        assert!(patrol.contains("SystemContext"));
    }

    #[test]
    fn scaffolded_blueprints_are_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let crate_dir = scaffold_game_crate(tmp.path(), "my-game", engine_root()).unwrap();

        let content = fs::read_to_string(crate_dir.join("src/blueprints.rs")).unwrap();
        assert!(content.contains("guard_blueprint"));
        assert!(content.contains("collectible_blueprint"));
        assert!(content.contains("Blueprint"));
    }
}
