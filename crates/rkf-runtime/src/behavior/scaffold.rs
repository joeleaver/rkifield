//! Game crate generation for the behavior system.
//!
//! [`generate_game_crate`] scans `assets/scripts/{components,systems}/` in the
//! project directory and generates a compilable Rust crate in the editor cache
//! directory (`.rkeditorcache/game/`). This crate is compiled as a hot-reloadable
//! cdylib — the user never sees or manages the crate structure.

use std::fs;
use std::path::{Path, PathBuf};

// ─── ScaffoldError ──────────────────────────────────────────────────────────

/// Errors that can occur during game crate generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaffoldError {
    /// An I/O error occurred.
    IoError(String),
}

impl std::fmt::Display for ScaffoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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

// ─── Constants ──────────────────────────────────────────────────────────────

/// Editor cache directory name (placed in the project root).
pub const EDITOR_CACHE_DIR: &str = ".rkeditorcache";

/// Name of the generated game crate within the editor cache.
pub const GAME_CRATE_NAME: &str = "game";

// ─── generate_game_crate ────────────────────────────────────────────────────

/// Generate a compilable game crate from the project's `assets/scripts/` directory.
///
/// Scans `project_dir/assets/scripts/{components,systems}/` for `.rs` files and
/// generates a complete Rust crate at `project_dir/.rkeditorcache/game/` with:
/// - `Cargo.toml` (cdylib, depends on rkf-runtime + glam + hecs)
/// - `src/lib.rs` (rkf_register + rkf_abi_version, auto-generated from discovered files)
/// - `src/blueprints.rs` (symlinked/copied from assets/scripts/blueprints.rs)
/// - `src/components/mod.rs` + symlinks to each component `.rs` file
/// - `src/systems/mod.rs` + symlinks to each system `.rs` file
///
/// `engine_root` is the workspace root (parent of `crates/`). The generated
/// `Cargo.toml` uses the absolute path to `rkf-runtime` within that workspace.
///
/// Returns the path to the generated crate directory.
pub fn generate_game_crate(
    project_dir: &Path,
    engine_root: &Path,
) -> Result<PathBuf, ScaffoldError> {
    let scripts_dir = project_dir.join("assets/scripts");
    let cache_dir = project_dir.join(EDITOR_CACHE_DIR);
    let crate_dir = cache_dir.join(GAME_CRATE_NAME);

    // Discover component and system source files.
    let components = discover_rs_files(&scripts_dir.join("components"));
    let systems = discover_rs_files(&scripts_dir.join("systems"));
    let has_blueprints = scripts_dir.join("blueprints.rs").is_file();

    // Create crate directory structure.
    let src_dir = crate_dir.join("src");
    let components_dir = src_dir.join("components");
    let systems_dir = src_dir.join("systems");

    fs::create_dir_all(&components_dir)?;
    fs::create_dir_all(&systems_dir)?;

    let runtime_path = engine_root.join("crates/rkf-runtime");
    let runtime_path_str = runtime_path.display().to_string();

    // Write Cargo.toml (only if changed — preserves mtime for cargo fingerprinting).
    write_if_changed(
        &crate_dir.join("Cargo.toml"),
        &gen_cargo_toml(GAME_CRATE_NAME, &runtime_path_str),
    )?;

    // Remove stale source files from the crate that no longer have a matching script.
    // This prevents cargo from compiling deleted scripts.
    remove_stale_sources(&components_dir, &components)?;
    remove_stale_sources(&systems_dir, &systems)?;

    // Copy source files from assets/scripts/ into the crate's src/ (only if changed).
    for name in &components {
        let src = scripts_dir.join("components").join(format!("{name}.rs"));
        let dst = components_dir.join(format!("{name}.rs"));
        copy_if_changed(&src, &dst)?;
    }
    for name in &systems {
        let src = scripts_dir.join("systems").join(format!("{name}.rs"));
        let dst = systems_dir.join(format!("{name}.rs"));
        copy_if_changed(&src, &dst)?;
    }
    if has_blueprints {
        let src = scripts_dir.join("blueprints.rs");
        let dst = src_dir.join("blueprints.rs");
        copy_if_changed(&src, &dst)?;
    }

    // Generate mod.rs files (only if changed).
    write_if_changed(&components_dir.join("mod.rs"), &gen_components_mod_rs(&components))?;
    write_if_changed(&systems_dir.join("mod.rs"), &gen_systems_mod_rs(&systems))?;

    // Generate lib.rs (only if changed).
    write_if_changed(
        &src_dir.join("lib.rs"),
        &gen_lib_rs(&components, &systems, has_blueprints),
    )?;

    Ok(crate_dir)
}

/// Return the path to the generated game crate directory for a project.
///
/// Does not check whether the crate has been generated yet.
pub fn game_crate_dir(project_dir: &Path) -> PathBuf {
    project_dir.join(EDITOR_CACHE_DIR).join(GAME_CRATE_NAME)
}

/// Return the path to the project's scripts directory.
pub fn scripts_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("assets/scripts")
}

// ─── Stale file cleanup ─────────────────────────────────────────────────────

/// Remove `.rs` source files from `dir` that are not in the `expected` names list.
///
/// Skips `mod.rs`. This ensures that scripts deleted from `assets/scripts/` are
/// also removed from the generated crate, so cargo doesn't try to compile them.
fn remove_stale_sources(dir: &Path, expected: &[String]) -> Result<(), ScaffoldError> {
    if !dir.is_dir() {
        return Ok(());
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            if stem == "mod" {
                continue;
            }
            if !expected.iter().any(|name| name == stem) {
                let _ = fs::remove_file(&path);
            }
        }
    }
    Ok(())
}

// ─── File discovery ─────────────────────────────────────────────────────────

/// Discover `.rs` files in a directory, returning their stem names sorted.
///
/// For example, `["camera_follow", "health", "patrol", "spin"]`.
fn discover_rs_files(dir: &Path) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // Skip mod.rs if present.
                    if stem != "mod" {
                        names.push(stem.to_string());
                    }
                }
            }
        }
    }
    names.sort();
    names
}

/// Write content to a file only if it differs from the current content.
///
/// Returns `true` if the file was actually written (content changed or didn't exist).
fn write_if_changed(path: &Path, content: &str) -> Result<bool, ScaffoldError> {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == content {
            return Ok(false);
        }
    }
    fs::write(path, content)?;
    Ok(true)
}

/// Copy a file only if the content differs from the destination.
///
/// Returns `true` if the file was actually copied (content changed or didn't exist).
fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool, ScaffoldError> {
    if dst.exists() {
        // Compare content to avoid updating mtime when nothing changed.
        let src_content = fs::read(src)?;
        if let Ok(dst_content) = fs::read(dst) {
            if src_content == dst_content {
                return Ok(false);
            }
        }
    }
    fs::copy(src, dst)?;
    Ok(true)
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

# The game dylib shares hecs::World with the editor across the FFI boundary.
# The editor always runs in release mode. hecs has #[cfg(debug_assertions)]
# fields that change struct layout between debug/release, so the dylib MUST
# be built in release mode to match. opt-level=1 keeps compile times fast.
[profile.release]
opt-level = 1
"#
    )
}

fn gen_lib_rs(components: &[String], systems: &[String], has_blueprints: bool) -> String {
    let mut s = String::new();
    s.push_str("//! Auto-generated game behavior crate — hot-reloadable gameplay logic.\n");
    s.push_str("//!\n");
    s.push_str("//! Generated by the editor from assets/scripts/. Do not edit manually.\n\n");

    if has_blueprints {
        s.push_str("mod blueprints;\n");
    }
    if !components.is_empty() {
        s.push_str("mod components;\n");
    }
    if !systems.is_empty() {
        s.push_str("mod systems;\n");
    }
    s.push('\n');

    s.push_str("use rkf_runtime::behavior::{GameplayRegistry, Phase, SystemMeta};\n\n");

    s.push_str("/// Called by the engine on load/reload to register components and systems.\n");
    s.push_str("#[no_mangle]\n");
    s.push_str("pub extern \"C\" fn rkf_register(registry: &mut GameplayRegistry) {\n");

    // Register components.
    if !components.is_empty() {
        s.push_str("    // Register all components.\n");
        s.push_str("    let entries = [\n");
        for name in components {
            s.push_str(&format!("        components::{name}::entry(),\n"));
        }
        s.push_str("    ];\n");
        s.push_str("    for e in entries {\n");
        s.push_str("        registry\n");
        s.push_str("            .register_component(e)\n");
        s.push_str("            .expect(\"component registration should not conflict\");\n");
        s.push_str("    }\n\n");
    }

    // Register systems.
    // Each system module is expected to export a function with the same name as the module
    // suffixed with `_system`.  We register with Phase::Update and no ordering constraints
    // by default.  Users can customise ordering in their system files (future: metadata attr).
    for name in systems {
        let fn_name = format!("{name}_system");
        s.push_str(&format!(
            "    registry.register_system(SystemMeta {{\n\
             \x20       name: \"{fn_name}\",\n\
             \x20       module_path: \"systems::{name}::{fn_name}\",\n\
             \x20       phase: Phase::Update,\n\
             \x20       after: &[],\n\
             \x20       before: &[],\n\
             \x20       fn_ptr: systems::{name}::{fn_name} as *const (),\n\
             \x20   }});\n\n"
        ));
    }

    // Register blueprints.
    if has_blueprints {
        s.push_str("    // Register blueprints.\n");
        s.push_str("    for bp in blueprints::all_blueprints() {\n");
        s.push_str("        registry.blueprint_catalog.insert(bp);\n");
        s.push_str("    }\n");
    }

    s.push_str("}\n\n");

    s.push_str("/// ABI version check — engine compares this against its own version.\n");
    s.push_str("#[no_mangle]\n");
    s.push_str("pub extern \"C\" fn rkf_abi_version() -> *const u8 {\n");
    s.push_str("    concat!(env!(\"CARGO_PKG_VERSION\"), \"\\0\").as_ptr()\n");
    s.push_str("}\n");

    s
}

fn gen_components_mod_rs(components: &[String]) -> String {
    let mut s = String::from("//! Game-specific components.\n\n");
    for name in components {
        s.push_str(&format!("pub mod {name};\n"));
    }
    // Glob re-exports so systems can use any public type from component modules
    // (e.g., `use crate::components::GuardState` alongside `GuardAi`).
    s.push('\n');
    for name in components {
        s.push_str(&format!("pub use {name}::*;\n"));
    }
    s
}

fn gen_systems_mod_rs(systems: &[String]) -> String {
    let mut s = String::from("//! Game-specific systems.\n\n");
    for name in systems {
        s.push_str(&format!("pub mod {name};\n"));
    }
    s
}

/// Convert a snake_case name to PascalCase.
#[allow(dead_code)]
fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().to_string() + c.as_str(),
            }
        })
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_root() -> &'static Path {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest.parent().unwrap().parent().unwrap()
    }

    #[test]
    fn generate_game_crate_from_scripts() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("test_project");

        // Set up assets/scripts/ with a couple of files.
        let comp_dir = project.join("assets/scripts/components");
        let sys_dir = project.join("assets/scripts/systems");
        fs::create_dir_all(&comp_dir).unwrap();
        fs::create_dir_all(&sys_dir).unwrap();

        // Copy library files to simulate a real project.
        let lib_dir = crate::project::engine_library_dir();
        fs::copy(
            lib_dir.join("components/health.rs"),
            comp_dir.join("health.rs"),
        ).unwrap();
        fs::copy(
            lib_dir.join("components/spin.rs"),
            comp_dir.join("spin.rs"),
        ).unwrap();
        fs::copy(
            lib_dir.join("systems/spin.rs"),
            sys_dir.join("spin.rs"),
        ).unwrap();

        let result = generate_game_crate(&project, engine_root());
        assert!(result.is_ok());
        let crate_dir = result.unwrap();

        assert!(crate_dir.join("Cargo.toml").is_file());
        assert!(crate_dir.join("src/lib.rs").is_file());
        assert!(crate_dir.join("src/components/mod.rs").is_file());
        assert!(crate_dir.join("src/systems/mod.rs").is_file());

        // The source files should be linked/copied.
        assert!(crate_dir.join("src/components/health.rs").exists());
        assert!(crate_dir.join("src/components/spin.rs").exists());
        assert!(crate_dir.join("src/systems/spin.rs").exists());

        // lib.rs should reference the discovered files.
        let lib_rs = fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
        assert!(lib_rs.contains("components::health::entry()"));
        assert!(lib_rs.contains("components::spin::entry()"));
        assert!(lib_rs.contains("systems::spin::spin_system"));

        // mod.rs should declare the modules.
        let comp_mod = fs::read_to_string(crate_dir.join("src/components/mod.rs")).unwrap();
        assert!(comp_mod.contains("pub mod health;"));
        assert!(comp_mod.contains("pub mod spin;"));
    }

    #[test]
    fn generate_game_crate_empty_scripts() {
        let tmp = tempfile::tempdir().unwrap();
        let project = tmp.path().join("empty_project");
        let comp_dir = project.join("assets/scripts/components");
        let sys_dir = project.join("assets/scripts/systems");
        fs::create_dir_all(&comp_dir).unwrap();
        fs::create_dir_all(&sys_dir).unwrap();

        let result = generate_game_crate(&project, engine_root());
        assert!(result.is_ok());
        let crate_dir = result.unwrap();

        // Should still generate a valid crate with empty registrations.
        let lib_rs = fs::read_to_string(crate_dir.join("src/lib.rs")).unwrap();
        assert!(lib_rs.contains("rkf_register"));
        assert!(lib_rs.contains("rkf_abi_version"));
        // No mod declarations for empty dirs.
        assert!(!lib_rs.contains("mod components"));
        assert!(!lib_rs.contains("mod systems"));
    }

    #[test]
    fn game_crate_dir_path() {
        let project = Path::new("/home/user/projects/MyGame");
        assert_eq!(
            game_crate_dir(project),
            PathBuf::from("/home/user/projects/MyGame/.rkeditorcache/game"),
        );
    }

    #[test]
    fn to_pascal_case_works() {
        assert_eq!(to_pascal_case("health"), "Health");
        assert_eq!(to_pascal_case("camera_follow"), "CameraFollow");
        assert_eq!(to_pascal_case("guard_ai"), "GuardAi");
        assert_eq!(to_pascal_case("door_state"), "DoorState");
    }
}
