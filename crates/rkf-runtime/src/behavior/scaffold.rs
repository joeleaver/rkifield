//! Game crate scaffolding for the behavior system.
//!
//! [`scaffold_game_crate`] creates a complete starter game crate with example
//! components, systems, and blueprints — ready to compile as a hot-reloadable
//! cdylib.

use std::fs;
use std::path::{Path, PathBuf};

use crate::project::engine_library_dir;

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

// ─── Library file reader ─────────────────────────────────────────────────────

/// Read a template file from the engine's `library/` directory.
///
/// `rel_path` is relative to `library/`, e.g. `"components/health.rs"`.
fn read_library_file(rel_path: &str) -> Result<String, ScaffoldError> {
    let path = engine_library_dir().join(rel_path);
    fs::read_to_string(&path).map_err(|e| {
        ScaffoldError::IoError(format!(
            "failed to read library template '{}': {}",
            path.display(),
            e
        ))
    })
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
    fs::write(src_dir.join("blueprints.rs"), read_library_file("blueprints.rs")?)?;

    // Components
    fs::write(components_dir.join("mod.rs"), gen_components_mod_rs())?;
    fs::write(components_dir.join("health.rs"), read_library_file("components/health.rs")?)?;
    fs::write(components_dir.join("patrol.rs"), read_library_file("components/patrol.rs")?)?;
    fs::write(components_dir.join("collectible.rs"), read_library_file("components/collectible.rs")?)?;
    fs::write(components_dir.join("door_state.rs"), read_library_file("components/door_state.rs")?)?;
    fs::write(components_dir.join("guard_ai.rs"), read_library_file("components/guard_ai.rs")?)?;
    fs::write(components_dir.join("camera_follow.rs"), read_library_file("components/camera_follow.rs")?)?;
    fs::write(components_dir.join("float_bob.rs"), read_library_file("components/float_bob.rs")?)?;
    fs::write(components_dir.join("spin.rs"), read_library_file("components/spin.rs")?)?;

    // Systems
    fs::write(systems_dir.join("mod.rs"), gen_systems_mod_rs())?;
    fs::write(systems_dir.join("patrol.rs"), read_library_file("systems/patrol.rs")?)?;
    fs::write(systems_dir.join("door.rs"), read_library_file("systems/door.rs")?)?;
    fs::write(systems_dir.join("death.rs"), read_library_file("systems/death.rs")?)?;
    fs::write(systems_dir.join("collectible.rs"), read_library_file("systems/collectible.rs")?)?;
    fs::write(systems_dir.join("guard_ai.rs"), read_library_file("systems/guard_ai.rs")?)?;
    fs::write(systems_dir.join("camera_follow.rs"), read_library_file("systems/camera_follow.rs")?)?;
    fs::write(systems_dir.join("float_bob.rs"), read_library_file("systems/float_bob.rs")?)?;
    fs::write(systems_dir.join("spin.rs"), read_library_file("systems/spin.rs")?)?;

    Ok(crate_dir)
}

// ─── Template generators ────────────────────────────────────────────────────
//
// Only gen_cargo_toml, gen_lib_rs, gen_components_mod_rs, and gen_systems_mod_rs
// remain inline — they need project-specific content or are module declarations.
//
// Component, system, and blueprint templates now live in library/ and are read
// at scaffold time via read_library_file().

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
