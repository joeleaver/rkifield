//! System scheduler — topological sort of systems within phases.
//!
//! Builds a dependency graph from `SystemMeta::after` / `before` constraints
//! and produces a linear execution order for each phase. Detects cycles and
//! missing dependency targets at build time.

use super::registry::{Phase, SystemMeta};
use std::collections::{HashMap, HashSet, VecDeque};

/// A resolved execution schedule: systems in topological order per phase.
#[derive(Debug)]
pub struct Schedule {
    /// Systems to run in Phase::Update, in order.
    pub update: Vec<usize>,
    /// Systems to run in Phase::LateUpdate, in order.
    pub late_update: Vec<usize>,
}

/// Error from building a schedule.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ScheduleError {
    /// A dependency cycle was detected.
    #[error("dependency cycle detected among systems: {0:?}")]
    Cycle(Vec<String>),
    /// A system references a dependency that doesn't exist.
    #[error("system '{system}' has dependency on unknown system '{dependency}'")]
    MissingDependency {
        /// The system with the bad dependency.
        system: String,
        /// The dependency that doesn't exist.
        dependency: String,
    },
}

/// Build a [`Schedule`] from a slice of registered systems.
///
/// Systems are grouped by phase, then topologically sorted within each phase
/// using Kahn's algorithm. Returns indices into the input slice.
pub fn build_schedule(systems: &[SystemMeta]) -> Result<Schedule, ScheduleError> {
    let update = sort_phase(systems, Phase::Update)?;
    let late_update = sort_phase(systems, Phase::LateUpdate)?;
    Ok(Schedule {
        update,
        late_update,
    })
}

/// Topological sort of systems within a single phase.
fn sort_phase(all_systems: &[SystemMeta], phase: Phase) -> Result<Vec<usize>, ScheduleError> {
    // Collect indices of systems in this phase
    let phase_indices: Vec<usize> = all_systems
        .iter()
        .enumerate()
        .filter(|(_, s)| s.phase == phase)
        .map(|(i, _)| i)
        .collect();

    if phase_indices.is_empty() {
        return Ok(Vec::new());
    }

    // Map system name → index in phase_indices for fast lookup
    let name_to_pos: HashMap<&str, usize> = phase_indices
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (all_systems[idx].name, pos))
        .collect();

    let n = phase_indices.len();
    // Adjacency: edges[a] contains b means a must run before b
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    // Build graph from after/before constraints
    for (pos, &idx) in phase_indices.iter().enumerate() {
        let sys = &all_systems[idx];

        // "after" = these systems must run before me
        for &dep in sys.after {
            if let Some(&dep_pos) = name_to_pos.get(dep) {
                edges[dep_pos].push(pos);
                in_degree[pos] += 1;
            } else {
                // Check if the dependency exists in another phase (not an error — cross-phase deps are implicit)
                let exists_elsewhere = all_systems
                    .iter()
                    .any(|s| s.name == dep && s.phase != phase);
                if !exists_elsewhere {
                    return Err(ScheduleError::MissingDependency {
                        system: sys.name.to_owned(),
                        dependency: dep.to_owned(),
                    });
                }
                // Cross-phase dependency — silently ignore (Update always runs before LateUpdate)
            }
        }

        // "before" = I must run before these systems
        for &dep in sys.before {
            if let Some(&dep_pos) = name_to_pos.get(dep) {
                edges[pos].push(dep_pos);
                in_degree[dep_pos] += 1;
            } else {
                let exists_elsewhere = all_systems
                    .iter()
                    .any(|s| s.name == dep && s.phase != phase);
                if !exists_elsewhere {
                    return Err(ScheduleError::MissingDependency {
                        system: sys.name.to_owned(),
                        dependency: dep.to_owned(),
                    });
                }
            }
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (pos, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(pos);
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(pos) = queue.pop_front() {
        order.push(phase_indices[pos]);
        for &next in &edges[pos] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }

    if order.len() != n {
        // Cycle detected — collect names of systems still in the graph
        let cycle_names: Vec<String> = phase_indices
            .iter()
            .enumerate()
            .filter(|(pos, _)| in_degree[*pos] > 0)
            .map(|(_, &idx)| all_systems[idx].name.to_owned())
            .collect();
        return Err(ScheduleError::Cycle(cycle_names));
    }

    Ok(order)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(
        name: &'static str,
        phase: Phase,
        after: &'static [&'static str],
        before: &'static [&'static str],
    ) -> SystemMeta {
        SystemMeta {
            name,
            module_path: name,
            phase,
            after,
            before,
            fn_ptr: std::ptr::null(),
        }
    }

    #[test]
    fn empty_schedule() {
        let schedule = build_schedule(&[]).unwrap();
        assert!(schedule.update.is_empty());
        assert!(schedule.late_update.is_empty());
    }

    #[test]
    fn single_system() {
        let systems = [meta("movement", Phase::Update, &[], &[])];
        let schedule = build_schedule(&systems).unwrap();
        assert_eq!(schedule.update, vec![0]);
        assert!(schedule.late_update.is_empty());
    }

    #[test]
    fn independent_systems_stable_order() {
        let systems = [
            meta("a", Phase::Update, &[], &[]),
            meta("b", Phase::Update, &[], &[]),
            meta("c", Phase::Update, &[], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        // Independent systems should all appear (order is stable from Kahn's)
        assert_eq!(schedule.update.len(), 3);
        let set: HashSet<usize> = schedule.update.iter().copied().collect();
        assert!(set.contains(&0));
        assert!(set.contains(&1));
        assert!(set.contains(&2));
    }

    #[test]
    fn after_constraint() {
        let systems = [
            meta("render", Phase::Update, &["physics"], &[]),
            meta("physics", Phase::Update, &[], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        let physics_pos = schedule.update.iter().position(|&i| i == 1).unwrap();
        let render_pos = schedule.update.iter().position(|&i| i == 0).unwrap();
        assert!(
            physics_pos < render_pos,
            "physics must run before render"
        );
    }

    #[test]
    fn before_constraint() {
        let systems = [
            meta("input", Phase::Update, &[], &["movement"]),
            meta("movement", Phase::Update, &[], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        let input_pos = schedule.update.iter().position(|&i| i == 0).unwrap();
        let movement_pos = schedule.update.iter().position(|&i| i == 1).unwrap();
        assert!(
            input_pos < movement_pos,
            "input must run before movement"
        );
    }

    #[test]
    fn chain_of_dependencies() {
        // input → movement → physics → render
        let systems = [
            meta("input", Phase::Update, &[], &["movement"]),
            meta("movement", Phase::Update, &["input"], &["physics"]),
            meta("physics", Phase::Update, &["movement"], &["render"]),
            meta("render", Phase::Update, &["physics"], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        assert_eq!(schedule.update, vec![0, 1, 2, 3]);
    }

    #[test]
    fn separate_phases() {
        let systems = [
            meta("movement", Phase::Update, &[], &[]),
            meta("camera_follow", Phase::LateUpdate, &[], &[]),
            meta("physics", Phase::Update, &["movement"], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();

        // Update: movement(0) then physics(2)
        assert_eq!(schedule.update.len(), 2);
        let movement_pos = schedule.update.iter().position(|&i| i == 0).unwrap();
        let physics_pos = schedule.update.iter().position(|&i| i == 2).unwrap();
        assert!(movement_pos < physics_pos);

        // LateUpdate: camera_follow(1) only
        assert_eq!(schedule.late_update, vec![1]);
    }

    #[test]
    fn cycle_detected() {
        let systems = [
            meta("a", Phase::Update, &["b"], &[]),
            meta("b", Phase::Update, &["a"], &[]),
        ];
        let err = build_schedule(&systems).unwrap_err();
        assert!(matches!(err, ScheduleError::Cycle(_)));
    }

    #[test]
    fn missing_dependency() {
        let systems = [meta("movement", Phase::Update, &["nonexistent"], &[])];
        let err = build_schedule(&systems).unwrap_err();
        match err {
            ScheduleError::MissingDependency {
                system,
                dependency,
            } => {
                assert_eq!(system, "movement");
                assert_eq!(dependency, "nonexistent");
            }
            _ => panic!("expected MissingDependency"),
        }
    }

    #[test]
    fn cross_phase_dep_not_error() {
        // movement in Update, camera_follow in LateUpdate references movement
        // This should NOT error — cross-phase ordering is implicit (Update < LateUpdate)
        let systems = [
            meta("movement", Phase::Update, &[], &[]),
            meta("camera_follow", Phase::LateUpdate, &["movement"], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        assert_eq!(schedule.update, vec![0]);
        assert_eq!(schedule.late_update, vec![1]);
    }

    #[test]
    fn three_node_cycle() {
        let systems = [
            meta("a", Phase::Update, &["c"], &[]),
            meta("b", Phase::Update, &["a"], &[]),
            meta("c", Phase::Update, &["b"], &[]),
        ];
        let err = build_schedule(&systems).unwrap_err();
        match err {
            ScheduleError::Cycle(names) => {
                assert_eq!(names.len(), 3);
            }
            _ => panic!("expected Cycle"),
        }
    }

    #[test]
    fn diamond_dependency() {
        //    A
        //   / \
        //  B   C
        //   \ /
        //    D
        let systems = [
            meta("a", Phase::Update, &[], &["b", "c"]),
            meta("b", Phase::Update, &["a"], &["d"]),
            meta("c", Phase::Update, &["a"], &["d"]),
            meta("d", Phase::Update, &["b", "c"], &[]),
        ];
        let schedule = build_schedule(&systems).unwrap();
        assert_eq!(schedule.update.len(), 4);

        let pos = |name_idx: usize| {
            schedule
                .update
                .iter()
                .position(|&i| i == name_idx)
                .unwrap()
        };
        assert!(pos(0) < pos(1)); // a before b
        assert!(pos(0) < pos(2)); // a before c
        assert!(pos(1) < pos(3)); // b before d
        assert!(pos(2) < pos(3)); // c before d
    }
}
