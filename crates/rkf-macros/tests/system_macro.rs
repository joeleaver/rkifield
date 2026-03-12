//! Integration tests for the `#[system]` proc macro.
//!
//! These tests verify that the macro:
//! - Preserves the original function unchanged
//! - Generates a `__register_system_<name>` registration function
//! - Correctly handles `phase`, `after`, and `before` attributes
//! - Supports both single-string and bracketed-list dependencies

use rkf_macros::system;
use rkf_runtime::behavior::{GameplayRegistry, Phase};

// ─── Basic: phase only ──────────────────────────────────────────────────

#[system(phase = Update)]
fn simple_system() {}

#[test]
fn basic_phase_update() {
    let mut reg = GameplayRegistry::new();
    __register_system_simple_system(&mut reg);

    let systems = reg.system_list();
    assert_eq!(systems.len(), 1);
    assert_eq!(systems[0].name, "simple_system");
    assert_eq!(systems[0].phase, Phase::Update);
    assert!(systems[0].after.is_empty());
    assert!(systems[0].before.is_empty());
    assert!(!systems[0].fn_ptr.is_null());
}

#[system(phase = LateUpdate)]
fn late_system() {}

#[test]
fn basic_phase_late_update() {
    let mut reg = GameplayRegistry::new();
    __register_system_late_system(&mut reg);

    let systems = reg.system_list();
    assert_eq!(systems.len(), 1);
    assert_eq!(systems[0].phase, Phase::LateUpdate);
}

// ─── Single after dependency ────────────────────────────────────────────

#[system(phase = Update, after = "movement")]
fn after_single() {}

#[test]
fn single_after_dependency() {
    let mut reg = GameplayRegistry::new();
    __register_system_after_single(&mut reg);

    let systems = reg.system_list();
    assert_eq!(systems[0].after, &["movement"]);
    assert!(systems[0].before.is_empty());
}

// ─── Single before dependency ───────────────────────────────────────────

#[system(phase = Update, before = "rendering")]
fn before_single() {}

#[test]
fn single_before_dependency() {
    let mut reg = GameplayRegistry::new();
    __register_system_before_single(&mut reg);

    let systems = reg.system_list();
    assert!(systems[0].after.is_empty());
    assert_eq!(systems[0].before, &["rendering"]);
}

// ─── Both after and before ──────────────────────────────────────────────

#[system(phase = Update, after = "movement", before = "rendering")]
fn after_and_before() {}

#[test]
fn both_after_and_before() {
    let mut reg = GameplayRegistry::new();
    __register_system_after_and_before(&mut reg);

    let systems = reg.system_list();
    assert_eq!(systems[0].after, &["movement"]);
    assert_eq!(systems[0].before, &["rendering"]);
}

// ─── Bracketed list dependencies ────────────────────────────────────────

#[system(phase = LateUpdate, after = ["physics", "ai"], before = ["cleanup", "debug"])]
fn multi_deps() {}

#[test]
fn bracketed_list_dependencies() {
    let mut reg = GameplayRegistry::new();
    __register_system_multi_deps(&mut reg);

    let systems = reg.system_list();
    assert_eq!(systems[0].phase, Phase::LateUpdate);
    assert_eq!(systems[0].after, &["physics", "ai"]);
    assert_eq!(systems[0].before, &["cleanup", "debug"]);
}

// ─── Function body preserved ────────────────────────────────────────────

#[system(phase = Update)]
fn returns_value() -> i32 {
    42
}

#[test]
fn function_body_preserved() {
    // The original function should still be callable.
    assert_eq!(returns_value(), 42);
}

// ─── Function with arguments preserved ──────────────────────────────────

#[system(phase = Update)]
fn with_args(x: i32, y: i32) -> i32 {
    x + y
}

#[test]
fn function_with_args_preserved() {
    assert_eq!(with_args(3, 4), 7);
}

// ─── fn_ptr points to correct function ──────────────────────────────────

#[system(phase = Update)]
fn ptr_test_fn() {}

#[test]
fn fn_ptr_points_to_function() {
    let mut reg = GameplayRegistry::new();
    __register_system_ptr_test_fn(&mut reg);

    let expected = ptr_test_fn as *const ();
    assert_eq!(reg.system_list()[0].fn_ptr, expected);
}

// ─── module_path is set ─────────────────────────────────────────────────

#[system(phase = Update)]
fn module_path_test() {}

#[test]
fn module_path_populated() {
    let mut reg = GameplayRegistry::new();
    __register_system_module_path_test(&mut reg);

    let mp = reg.system_list()[0].module_path;
    // Integration test module path will contain "system_macro".
    assert!(
        !mp.is_empty(),
        "module_path should not be empty, got: {mp}"
    );
}

// ─── Multiple registrations on same registry ────────────────────────────

#[system(phase = Update)]
fn sys_a() {}

#[system(phase = LateUpdate)]
fn sys_b() {}

#[test]
fn multiple_registrations() {
    let mut reg = GameplayRegistry::new();
    __register_system_sys_a(&mut reg);
    __register_system_sys_b(&mut reg);

    assert_eq!(reg.system_list().len(), 2);
    assert_eq!(reg.system_list()[0].name, "sys_a");
    assert_eq!(reg.system_list()[0].phase, Phase::Update);
    assert_eq!(reg.system_list()[1].name, "sys_b");
    assert_eq!(reg.system_list()[1].phase, Phase::LateUpdate);
}

// ─── Attribute order doesn't matter ─────────────────────────────────────

#[system(before = "z", phase = Update, after = "a")]
fn reordered_attrs() {}

#[test]
fn attribute_order_irrelevant() {
    let mut reg = GameplayRegistry::new();
    __register_system_reordered_attrs(&mut reg);

    let s = &reg.system_list()[0];
    assert_eq!(s.phase, Phase::Update);
    assert_eq!(s.after, &["a"]);
    assert_eq!(s.before, &["z"]);
}
