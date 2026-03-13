//! Systems panel data model for the editor UI.
//!
//! Provides [`SystemPanelEntry`] — a display-ready snapshot of a registered
//! system, including its name, phase, order, and optional timing data.

use super::executor::BehaviorExecutor;
use super::registry::{GameplayRegistry, Phase};

// ─── SystemPanelEntry ───────────────────────────────────────────────────────

/// Display data for a single system in the Systems panel.
#[derive(Debug, Clone)]
pub struct SystemPanelEntry {
    /// System function name (e.g., "patrol_system").
    pub name: String,
    /// Which execution phase this system runs in.
    pub phase: Phase,
    /// Execution order within its phase (0-based).
    pub order: usize,
    /// Whether the system faulted (panicked) during the last frame.
    pub faulted: bool,
    /// Last frame execution time in microseconds, if available.
    pub last_frame_us: Option<u64>,
}

// ─── Build systems panel ────────────────────────────────────────────────────

/// Build a list of [`SystemPanelEntry`] from the registry and executor.
///
/// Systems are ordered by phase (Update before LateUpdate), then by their
/// schedule order within each phase. Timing data is read from the executor's
/// per-system instrumentation (populated during `tick()`).
pub fn build_systems_panel(
    registry: &GameplayRegistry,
    executor: &BehaviorExecutor,
) -> Vec<SystemPanelEntry> {
    let systems = registry.system_list();
    if systems.is_empty() {
        return Vec::new();
    }

    let mut entries = Vec::new();

    // Group by phase, preserving registration order as a proxy for schedule order.
    let mut update_order: usize = 0;
    let mut late_update_order: usize = 0;

    for (sys_idx, meta) in systems.iter().enumerate() {
        let (order, phase) = match meta.phase {
            Phase::Update => {
                let o = update_order;
                update_order += 1;
                (o, Phase::Update)
            }
            Phase::LateUpdate => {
                let o = late_update_order;
                late_update_order += 1;
                (o, Phase::LateUpdate)
            }
        };

        entries.push(SystemPanelEntry {
            name: meta.name.to_string(),
            phase,
            order,
            faulted: executor.is_faulted(sys_idx),
            last_frame_us: executor.system_timing(sys_idx),
        });
    }

    // Sort: Update phase first, then LateUpdate, within each phase by order.
    entries.sort_by(|a, b| {
        let phase_cmp = phase_sort_key(a.phase).cmp(&phase_sort_key(b.phase));
        phase_cmp.then(a.order.cmp(&b.order))
    });

    entries
}

/// Sort key for phases: Update=0, LateUpdate=1.
fn phase_sort_key(phase: Phase) -> u8 {
    match phase {
        Phase::Update => 0,
        Phase::LateUpdate => 1,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::registry::SystemMeta;

    fn dummy_system_meta(name: &'static str, phase: Phase) -> SystemMeta {
        SystemMeta {
            name,
            module_path: name,
            phase,
            after: &[],
            before: &[],
            fn_ptr: std::ptr::null(),
        }
    }

    // ── build_systems_panel_empty ────────────────────────────────────────

    #[test]
    fn build_systems_panel_empty() {
        let registry = GameplayRegistry::new();
        let executor = BehaviorExecutor::new(&registry).unwrap();

        let entries = build_systems_panel(&registry, &executor);
        assert!(entries.is_empty());
    }

    // ── systems_panel_entry_fields ──────────────────────────────────────

    #[test]
    fn systems_panel_entry_fields() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(dummy_system_meta("movement_system", Phase::Update));
        registry.register_system(dummy_system_meta("camera_follow", Phase::LateUpdate));
        registry.register_system(dummy_system_meta("ai_system", Phase::Update));

        let executor = BehaviorExecutor::new(&registry).unwrap();
        let entries = build_systems_panel(&registry, &executor);

        assert_eq!(entries.len(), 3);

        // Update systems should come first.
        assert_eq!(entries[0].phase, Phase::Update);
        assert_eq!(entries[1].phase, Phase::Update);
        assert_eq!(entries[2].phase, Phase::LateUpdate);

        // Verify fields.
        assert_eq!(entries[2].name, "camera_follow");
        assert_eq!(entries[2].order, 0); // first in LateUpdate
        assert!(!entries[2].faulted);
        assert!(entries[2].last_frame_us.is_none());
    }

    // ── ordering within phase ───────────────────────────────────────────

    #[test]
    fn update_systems_ordered_before_late_update() {
        let mut registry = GameplayRegistry::new();
        // Register LateUpdate first, then Update — output should still be Update first.
        registry.register_system(dummy_system_meta("late_sys", Phase::LateUpdate));
        registry.register_system(dummy_system_meta("update_sys", Phase::Update));

        let executor = BehaviorExecutor::new(&registry).unwrap();
        let entries = build_systems_panel(&registry, &executor);

        assert_eq!(entries[0].name, "update_sys");
        assert_eq!(entries[0].phase, Phase::Update);
        assert_eq!(entries[1].name, "late_sys");
        assert_eq!(entries[1].phase, Phase::LateUpdate);
    }

    // ── faulted defaults to false ───────────────────────────────────────

    #[test]
    fn faulted_defaults_to_false() {
        let mut registry = GameplayRegistry::new();
        registry.register_system(dummy_system_meta("sys", Phase::Update));

        let executor = BehaviorExecutor::new(&registry).unwrap();
        let entries = build_systems_panel(&registry, &executor);

        assert!(!entries[0].faulted);
    }
}
