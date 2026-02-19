//! Stress test framework — scenario definitions, result tracking, and evaluation.
//!
//! This module provides the data types and logic for defining GPU/engine stress
//! scenarios, recording their results, and evaluating whether they meet
//! performance thresholds. The actual frame-loop execution is external; this
//! module is the bookkeeping layer.

#![allow(dead_code)]

use std::fmt;

// ── Scenario ─────────────────────────────────────────────────────────────────

/// Pre-defined stress scenarios that exercise different engine bottlenecks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StressScenario {
    /// Many chunks loaded simultaneously — tests spatial index and streaming.
    LargeWorld,
    /// Many shadow-casting lights — tests tile culling and shading budget.
    ManyLights,
    /// Many animated characters — tests joint rebaking throughput.
    ManyCharacters,
    /// High edit operations per frame — tests CSG pipeline and journal.
    HeavyEditing,
    /// Fast camera movement through many chunks — tests streaming I/O.
    RapidStreaming,
    /// All stressors simultaneously.
    Combined,
}

impl fmt::Display for StressScenario {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LargeWorld => write!(f, "LargeWorld"),
            Self::ManyLights => write!(f, "ManyLights"),
            Self::ManyCharacters => write!(f, "ManyCharacters"),
            Self::HeavyEditing => write!(f, "HeavyEditing"),
            Self::RapidStreaming => write!(f, "RapidStreaming"),
            Self::Combined => write!(f, "Combined"),
        }
    }
}

// ── Config ───────────────────────────────────────────────────────────────────

/// Parameters for a single stress test run.
#[derive(Debug, Clone)]
pub struct StressConfig {
    /// Which scenario to run.
    pub scenario: StressScenario,
    /// Number of chunks to load.
    pub chunk_count: u32,
    /// Number of shadow-casting lights.
    pub light_count: u32,
    /// Number of animated characters.
    pub character_count: u32,
    /// CSG / edit operations dispatched per frame.
    pub edit_operations_per_frame: u32,
    /// Camera movement speed in world-units/second.
    pub camera_speed: f32,
    /// How many frames to run the scenario.
    pub duration_frames: u32,
}

impl StressConfig {
    /// Build the default config for a given scenario.
    pub fn for_scenario(scenario: StressScenario) -> Self {
        match scenario {
            StressScenario::LargeWorld => Self {
                scenario,
                chunk_count: 100,
                light_count: 10,
                character_count: 0,
                edit_operations_per_frame: 0,
                camera_speed: 10.0,
                duration_frames: 300,
            },
            StressScenario::ManyLights => Self {
                scenario,
                chunk_count: 4,
                light_count: 100,
                character_count: 0,
                edit_operations_per_frame: 0,
                camera_speed: 10.0,
                duration_frames: 300,
            },
            StressScenario::ManyCharacters => Self {
                scenario,
                chunk_count: 4,
                light_count: 10,
                character_count: 20,
                edit_operations_per_frame: 0,
                camera_speed: 10.0,
                duration_frames: 300,
            },
            StressScenario::HeavyEditing => Self {
                scenario,
                chunk_count: 4,
                light_count: 10,
                character_count: 0,
                edit_operations_per_frame: 50,
                camera_speed: 10.0,
                duration_frames: 300,
            },
            StressScenario::RapidStreaming => Self {
                scenario,
                chunk_count: 50,
                light_count: 10,
                character_count: 0,
                edit_operations_per_frame: 0,
                camera_speed: 100.0,
                duration_frames: 300,
            },
            StressScenario::Combined => Self {
                scenario,
                chunk_count: 50,
                light_count: 50,
                character_count: 10,
                edit_operations_per_frame: 20,
                camera_speed: 50.0,
                duration_frames: 300,
            },
        }
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

/// Recorded results from a completed stress test run.
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Which scenario was tested.
    pub scenario: StressScenario,
    /// How many frames were actually completed.
    pub frames_completed: u32,
    /// Average frame time in milliseconds.
    pub avg_frame_time_ms: f32,
    /// Worst single-frame time in milliseconds.
    pub worst_frame_time_ms: f32,
    /// 99th percentile frame time in milliseconds.
    pub p99_frame_time_ms: f32,
    /// Peak GPU + CPU memory usage in bytes.
    pub memory_peak_bytes: u64,
    /// Memory usage at the end of the run in bytes.
    pub memory_final_bytes: u64,
    /// Whether the run passed its thresholds.
    pub passed: bool,
    /// Reason for failure, if any.
    pub failure_reason: Option<String>,
}

// ── Thresholds ───────────────────────────────────────────────────────────────

/// Performance thresholds that a stress result is evaluated against.
#[derive(Debug, Clone)]
pub struct StressThresholds {
    /// Maximum acceptable average frame time in milliseconds.
    pub max_avg_frame_ms: f32,
    /// Maximum acceptable 99th percentile frame time in milliseconds.
    pub max_p99_frame_ms: f32,
    /// Maximum acceptable peak memory in bytes.
    pub max_memory_bytes: u64,
}

impl Default for StressThresholds {
    fn default() -> Self {
        Self {
            max_avg_frame_ms: 33.0,             // 30 fps
            max_p99_frame_ms: 50.0,             // brief spikes OK
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
        }
    }
}

/// Evaluate whether a [`StressResult`] meets the given [`StressThresholds`].
///
/// Returns `true` if all thresholds are satisfied.
pub fn evaluate_result(result: &StressResult, thresholds: &StressThresholds) -> bool {
    result.avg_frame_time_ms <= thresholds.max_avg_frame_ms
        && result.p99_frame_time_ms <= thresholds.max_p99_frame_ms
        && result.memory_peak_bytes <= thresholds.max_memory_bytes
}

// ── Suite ────────────────────────────────────────────────────────────────────

/// A collection of stress test configs and their results.
#[derive(Debug, Clone)]
pub struct StressSuite {
    /// Scenario configurations to run.
    pub configs: Vec<StressConfig>,
    /// Recorded results (populated as tests complete).
    pub results: Vec<StressResult>,
}

impl StressSuite {
    /// Create an empty suite.
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a scenario configuration to the suite.
    pub fn add_scenario(&mut self, config: StressConfig) {
        self.configs.push(config);
    }

    /// Record a completed result.
    pub fn record_result(&mut self, result: StressResult) {
        self.results.push(result);
    }

    /// Returns `true` if all recorded results passed.
    pub fn all_passed(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(|r| r.passed)
    }

    /// Returns references to all results that failed.
    pub fn failed_scenarios(&self) -> Vec<&StressResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Human-readable summary of all results.
    pub fn summary_string(&self) -> String {
        if self.results.is_empty() {
            return "No results recorded.".to_string();
        }

        let mut lines = Vec::new();
        lines.push("Stress Test Summary".to_string());
        lines.push("===================".to_string());

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            lines.push(format!(
                "[{}] {} — avg {:.1}ms, p99 {:.1}ms, worst {:.1}ms, peak {:.1}MB",
                status,
                result.scenario,
                result.avg_frame_time_ms,
                result.p99_frame_time_ms,
                result.worst_frame_time_ms,
                result.memory_peak_bytes as f64 / (1024.0 * 1024.0),
            ));
            if let Some(reason) = &result.failure_reason {
                lines.push(format!("       Reason: {}", reason));
            }
        }

        let passed = self.results.iter().filter(|r| r.passed).count();
        let total = self.results.len();
        lines.push(format!("\n{}/{} scenarios passed.", passed, total));

        lines.join("\n")
    }
}

impl Default for StressSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a suite with all 6 default scenarios.
pub fn build_default_suite() -> StressSuite {
    let mut suite = StressSuite::new();
    for scenario in [
        StressScenario::LargeWorld,
        StressScenario::ManyLights,
        StressScenario::ManyCharacters,
        StressScenario::HeavyEditing,
        StressScenario::RapidStreaming,
        StressScenario::Combined,
    ] {
        suite.add_scenario(StressConfig::for_scenario(scenario));
    }
    suite
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default configs ──────────────────────────────────────────────────

    #[test]
    fn default_large_world() {
        let cfg = StressConfig::for_scenario(StressScenario::LargeWorld);
        assert_eq!(cfg.chunk_count, 100);
        assert_eq!(cfg.light_count, 10);
        assert_eq!(cfg.character_count, 0);
        assert_eq!(cfg.edit_operations_per_frame, 0);
    }

    #[test]
    fn default_many_lights() {
        let cfg = StressConfig::for_scenario(StressScenario::ManyLights);
        assert_eq!(cfg.chunk_count, 4);
        assert_eq!(cfg.light_count, 100);
        assert_eq!(cfg.character_count, 0);
    }

    #[test]
    fn default_many_characters() {
        let cfg = StressConfig::for_scenario(StressScenario::ManyCharacters);
        assert_eq!(cfg.chunk_count, 4);
        assert_eq!(cfg.light_count, 10);
        assert_eq!(cfg.character_count, 20);
    }

    #[test]
    fn default_heavy_editing() {
        let cfg = StressConfig::for_scenario(StressScenario::HeavyEditing);
        assert_eq!(cfg.chunk_count, 4);
        assert_eq!(cfg.edit_operations_per_frame, 50);
    }

    #[test]
    fn default_rapid_streaming() {
        let cfg = StressConfig::for_scenario(StressScenario::RapidStreaming);
        assert_eq!(cfg.chunk_count, 50);
        assert_eq!(cfg.camera_speed, 100.0);
    }

    #[test]
    fn default_combined() {
        let cfg = StressConfig::for_scenario(StressScenario::Combined);
        assert_eq!(cfg.chunk_count, 50);
        assert_eq!(cfg.light_count, 50);
        assert_eq!(cfg.character_count, 10);
        assert_eq!(cfg.edit_operations_per_frame, 20);
        assert_eq!(cfg.camera_speed, 50.0);
    }

    // ── Evaluate ─────────────────────────────────────────────────────────

    fn make_passing_result(scenario: StressScenario) -> StressResult {
        StressResult {
            scenario,
            frames_completed: 300,
            avg_frame_time_ms: 16.0,
            worst_frame_time_ms: 35.0,
            p99_frame_time_ms: 30.0,
            memory_peak_bytes: 500 * 1024 * 1024, // 500 MB
            memory_final_bytes: 400 * 1024 * 1024,
            passed: true,
            failure_reason: None,
        }
    }

    fn make_failing_result(scenario: StressScenario) -> StressResult {
        StressResult {
            scenario,
            frames_completed: 300,
            avg_frame_time_ms: 50.0,
            worst_frame_time_ms: 120.0,
            p99_frame_time_ms: 80.0,
            memory_peak_bytes: 3 * 1024 * 1024 * 1024, // 3 GB
            memory_final_bytes: 2 * 1024 * 1024 * 1024,
            passed: false,
            failure_reason: Some("Exceeded frame time budget".into()),
        }
    }

    #[test]
    fn evaluate_passing() {
        let result = make_passing_result(StressScenario::LargeWorld);
        let thresholds = StressThresholds::default();
        assert!(evaluate_result(&result, &thresholds));
    }

    #[test]
    fn evaluate_failing_avg_frame() {
        let mut result = make_passing_result(StressScenario::LargeWorld);
        result.avg_frame_time_ms = 40.0; // > 33ms
        let thresholds = StressThresholds::default();
        assert!(!evaluate_result(&result, &thresholds));
    }

    #[test]
    fn evaluate_failing_p99() {
        let mut result = make_passing_result(StressScenario::LargeWorld);
        result.p99_frame_time_ms = 60.0; // > 50ms
        let thresholds = StressThresholds::default();
        assert!(!evaluate_result(&result, &thresholds));
    }

    #[test]
    fn evaluate_failing_memory() {
        let mut result = make_passing_result(StressScenario::LargeWorld);
        result.memory_peak_bytes = 3 * 1024 * 1024 * 1024; // 3 GB > 2 GB
        let thresholds = StressThresholds::default();
        assert!(!evaluate_result(&result, &thresholds));
    }

    #[test]
    fn evaluate_custom_thresholds() {
        let result = make_passing_result(StressScenario::LargeWorld);
        let strict = StressThresholds {
            max_avg_frame_ms: 10.0,
            max_p99_frame_ms: 20.0,
            max_memory_bytes: 256 * 1024 * 1024,
        };
        // 16ms avg > 10ms max → fail.
        assert!(!evaluate_result(&result, &strict));
    }

    // ── Suite ────────────────────────────────────────────────────────────

    #[test]
    fn suite_all_passed_empty() {
        let suite = StressSuite::new();
        assert!(!suite.all_passed()); // No results → not "all passed".
    }

    #[test]
    fn suite_all_passed_true() {
        let mut suite = StressSuite::new();
        suite.record_result(make_passing_result(StressScenario::LargeWorld));
        suite.record_result(make_passing_result(StressScenario::ManyLights));
        assert!(suite.all_passed());
    }

    #[test]
    fn suite_all_passed_false() {
        let mut suite = StressSuite::new();
        suite.record_result(make_passing_result(StressScenario::LargeWorld));
        suite.record_result(make_failing_result(StressScenario::ManyLights));
        assert!(!suite.all_passed());
    }

    #[test]
    fn suite_failed_scenarios() {
        let mut suite = StressSuite::new();
        suite.record_result(make_passing_result(StressScenario::LargeWorld));
        suite.record_result(make_failing_result(StressScenario::ManyLights));
        suite.record_result(make_failing_result(StressScenario::Combined));
        let failed = suite.failed_scenarios();
        assert_eq!(failed.len(), 2);
        assert_eq!(failed[0].scenario, StressScenario::ManyLights);
        assert_eq!(failed[1].scenario, StressScenario::Combined);
    }

    #[test]
    fn suite_add_and_record() {
        let mut suite = StressSuite::new();
        suite.add_scenario(StressConfig::for_scenario(StressScenario::LargeWorld));
        assert_eq!(suite.configs.len(), 1);
        suite.record_result(make_passing_result(StressScenario::LargeWorld));
        assert_eq!(suite.results.len(), 1);
    }

    // ── Build default suite ──────────────────────────────────────────────

    #[test]
    fn build_default_has_six_scenarios() {
        let suite = build_default_suite();
        assert_eq!(suite.configs.len(), 6);
    }

    #[test]
    fn build_default_scenarios_unique() {
        let suite = build_default_suite();
        let scenarios: Vec<StressScenario> = suite.configs.iter().map(|c| c.scenario).collect();
        for (i, s) in scenarios.iter().enumerate() {
            assert!(
                !scenarios[i + 1..].contains(s),
                "duplicate scenario: {:?}",
                s
            );
        }
    }

    // ── Summary string ───────────────────────────────────────────────────

    #[test]
    fn summary_empty() {
        let suite = StressSuite::new();
        let s = suite.summary_string();
        assert!(s.contains("No results"));
    }

    #[test]
    fn summary_has_pass_fail_markers() {
        let mut suite = StressSuite::new();
        suite.record_result(make_passing_result(StressScenario::LargeWorld));
        suite.record_result(make_failing_result(StressScenario::ManyLights));
        let s = suite.summary_string();
        assert!(s.contains("[PASS]"));
        assert!(s.contains("[FAIL]"));
        assert!(s.contains("1/2 scenarios passed"));
    }

    #[test]
    fn summary_includes_failure_reason() {
        let mut suite = StressSuite::new();
        suite.record_result(make_failing_result(StressScenario::Combined));
        let s = suite.summary_string();
        assert!(s.contains("Exceeded frame time budget"));
    }

    // ── Display ──────────────────────────────────────────────────────────

    #[test]
    fn scenario_display() {
        assert_eq!(format!("{}", StressScenario::LargeWorld), "LargeWorld");
        assert_eq!(format!("{}", StressScenario::Combined), "Combined");
    }

    // ── Default thresholds ───────────────────────────────────────────────

    #[test]
    fn default_thresholds() {
        let t = StressThresholds::default();
        assert_eq!(t.max_avg_frame_ms, 33.0);
        assert_eq!(t.max_p99_frame_ms, 50.0);
        assert_eq!(t.max_memory_bytes, 2 * 1024 * 1024 * 1024);
    }
}
