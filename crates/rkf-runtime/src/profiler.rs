#![allow(dead_code)]
//! Performance profiling for the RKIField engine.
//!
//! Provides CPU timing, per-pass profiling, frame history, and summary statistics.

use std::collections::VecDeque;
use std::time::Instant;

/// Timing data for a single render pass.
#[derive(Debug, Clone)]
pub struct PassTiming {
    /// Name of the render pass.
    pub name: String,
    /// CPU-side time in milliseconds.
    pub cpu_time_ms: f32,
    /// GPU-side time in milliseconds (requires GPU timestamp queries).
    pub gpu_time_ms: Option<f32>,
}

/// Complete profile data for a single frame.
#[derive(Debug, Clone)]
pub struct FrameProfile {
    /// Frame number (monotonically increasing).
    pub frame_number: u64,
    /// Total CPU frame time in milliseconds.
    pub total_cpu_ms: f32,
    /// Total GPU frame time in milliseconds (if available).
    pub total_gpu_ms: Option<f32>,
    /// Per-pass timing breakdown.
    pub pass_timings: Vec<PassTiming>,
    /// Number of brick reads this frame.
    pub brick_reads: u64,
    /// Average ray march steps per pixel.
    pub ray_steps_avg: f32,
    /// Total shadow rays cast.
    pub shadow_rays: u32,
    /// Total cone traces performed.
    pub cone_traces: u32,
}

/// Rolling history of frame profiles.
#[derive(Debug)]
pub struct ProfileHistory {
    frames: VecDeque<FrameProfile>,
    max_length: usize,
}

impl ProfileHistory {
    /// Create a new history with the given maximum capacity.
    pub fn new(max_length: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(max_length),
            max_length,
        }
    }

    /// Push a frame profile, evicting the oldest if at capacity.
    pub fn push(&mut self, profile: FrameProfile) {
        if self.frames.len() >= self.max_length {
            self.frames.pop_front();
        }
        self.frames.push_back(profile);
    }

    /// Average total CPU frame time across all stored frames.
    pub fn average_frame_time(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.frames.iter().map(|f| f.total_cpu_ms).sum();
        sum / self.frames.len() as f32
    }

    /// Worst (maximum) CPU frame time.
    pub fn worst_frame_time(&self) -> f32 {
        self.frames
            .iter()
            .map(|f| f.total_cpu_ms)
            .fold(0.0_f32, f32::max)
    }

    /// Best (minimum) CPU frame time.
    pub fn best_frame_time(&self) -> f32 {
        self.frames
            .iter()
            .map(|f| f.total_cpu_ms)
            .fold(f32::INFINITY, f32::min)
    }

    /// Average CPU time for a specific named pass.
    pub fn average_pass_time(&self, name: &str) -> Option<f32> {
        let mut sum = 0.0_f32;
        let mut count = 0u32;
        for frame in &self.frames {
            for pass in &frame.pass_timings {
                if pass.name == name {
                    sum += pass.cpu_time_ms;
                    count += 1;
                }
            }
        }
        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    /// Percentile CPU frame time (e.g., p=99.0 for 99th percentile).
    pub fn percentile_frame_time(&self, p: f32) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let mut times: Vec<f32> = self.frames.iter().map(|f| f.total_cpu_ms).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let index = ((p / 100.0) * (times.len() - 1) as f32).round() as usize;
        let index = index.min(times.len() - 1);
        times[index]
    }

    /// Number of stored frames.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Clear all stored frame profiles.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

/// Simple CPU timer based on `std::time::Instant`.
#[derive(Debug)]
pub struct CpuTimer {
    start: Instant,
}

impl CpuTimer {
    /// Start a new CPU timer.
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Elapsed time since start in milliseconds.
    pub fn elapsed_ms(&self) -> f32 {
        self.start.elapsed().as_secs_f32() * 1000.0
    }
}

/// Configuration for the profiling system.
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Whether profiling is enabled.
    pub enabled: bool,
    /// Whether GPU timestamp queries are enabled.
    pub gpu_timing_enabled: bool,
    /// Maximum number of frames to keep in history.
    pub history_length: usize,
    /// How often (in frames) to log a summary.
    pub log_interval_frames: u32,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            gpu_timing_enabled: false,
            history_length: 300,
            log_interval_frames: 300,
        }
    }
}

/// Main profiler that manages frame lifecycle and history.
#[derive(Debug)]
pub struct Profiler {
    config: ProfilingConfig,
    history: ProfileHistory,
    current_frame: Option<FrameProfile>,
    frame_counter: u64,
    frame_start: Option<Instant>,
}

impl Profiler {
    /// Create a new profiler with the given configuration.
    pub fn new(config: ProfilingConfig) -> Self {
        let history = ProfileHistory::new(config.history_length);
        Self {
            config,
            history,
            current_frame: None,
            frame_counter: 0,
            frame_start: None,
        }
    }

    /// Begin profiling a new frame.
    pub fn begin_frame(&mut self) {
        if !self.config.enabled {
            return;
        }
        self.frame_start = Some(Instant::now());
        self.current_frame = Some(FrameProfile {
            frame_number: self.frame_counter,
            total_cpu_ms: 0.0,
            total_gpu_ms: None,
            pass_timings: Vec::new(),
            brick_reads: 0,
            ray_steps_avg: 0.0,
            shadow_rays: 0,
            cone_traces: 0,
        });
    }

    /// Begin timing a named render pass. Returns a `CpuTimer`.
    pub fn begin_pass(&self, _name: &str) -> CpuTimer {
        CpuTimer::start()
    }

    /// End timing a named render pass.
    pub fn end_pass(&mut self, name: &str, timer: CpuTimer, gpu_ms: Option<f32>) {
        if let Some(ref mut frame) = self.current_frame {
            frame.pass_timings.push(PassTiming {
                name: name.to_string(),
                cpu_time_ms: timer.elapsed_ms(),
                gpu_time_ms: gpu_ms,
            });
        }
    }

    /// End the current frame and push it to history.
    pub fn end_frame(&mut self) {
        if let Some(mut frame) = self.current_frame.take() {
            if let Some(start) = self.frame_start.take() {
                frame.total_cpu_ms = start.elapsed().as_secs_f32() * 1000.0;
            }
            self.history.push(frame);
            self.frame_counter += 1;
        }
    }

    /// Access the profile history.
    pub fn history(&self) -> &ProfileHistory {
        &self.history
    }

    /// Whether profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Generate a formatted multi-line performance summary.
    pub fn summary_string(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Performance Summary ===\n");
        s.push_str(&format!(
            "Frames: {}\n",
            self.history.frame_count()
        ));
        s.push_str(&format!(
            "Avg frame: {:.2} ms ({:.0} fps)\n",
            self.history.average_frame_time(),
            if self.history.average_frame_time() > 0.0 {
                1000.0 / self.history.average_frame_time()
            } else {
                0.0
            }
        ));
        s.push_str(&format!(
            "Best: {:.2} ms | Worst: {:.2} ms | P99: {:.2} ms\n",
            self.history.best_frame_time(),
            self.history.worst_frame_time(),
            self.history.percentile_frame_time(99.0)
        ));

        // Collect unique pass names from latest frame
        if let Some(latest) = self.history.frames.back() {
            s.push_str("--- Pass Timings ---\n");
            for pass in &latest.pass_timings {
                let avg = self
                    .history
                    .average_pass_time(&pass.name)
                    .unwrap_or(0.0);
                s.push_str(&format!(
                    "  {}: {:.2} ms (avg {:.2} ms)\n",
                    pass.name, pass.cpu_time_ms, avg
                ));
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(frame_number: u64, total_cpu_ms: f32, passes: Vec<(&str, f32)>) -> FrameProfile {
        FrameProfile {
            frame_number,
            total_cpu_ms,
            total_gpu_ms: None,
            pass_timings: passes
                .into_iter()
                .map(|(name, cpu)| PassTiming {
                    name: name.to_string(),
                    cpu_time_ms: cpu,
                    gpu_time_ms: None,
                })
                .collect(),
            brick_reads: 0,
            ray_steps_avg: 0.0,
            shadow_rays: 0,
            cone_traces: 0,
        }
    }

    #[test]
    fn history_push_and_count() {
        let mut history = ProfileHistory::new(5);
        assert_eq!(history.frame_count(), 0);
        history.push(make_profile(0, 16.0, vec![]));
        history.push(make_profile(1, 17.0, vec![]));
        assert_eq!(history.frame_count(), 2);
    }

    #[test]
    fn history_caps_at_max_length() {
        let mut history = ProfileHistory::new(3);
        for i in 0..10 {
            history.push(make_profile(i, i as f32, vec![]));
        }
        assert_eq!(history.frame_count(), 3);
        // Oldest should be frame 7
        assert_eq!(history.frames.front().unwrap().frame_number, 7);
    }

    #[test]
    fn average_frame_time() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(0, 10.0, vec![]));
        history.push(make_profile(1, 20.0, vec![]));
        history.push(make_profile(2, 30.0, vec![]));
        assert!((history.average_frame_time() - 20.0).abs() < 0.01);
    }

    #[test]
    fn worst_frame_time() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(0, 10.0, vec![]));
        history.push(make_profile(1, 50.0, vec![]));
        history.push(make_profile(2, 25.0, vec![]));
        assert!((history.worst_frame_time() - 50.0).abs() < 0.01);
    }

    #[test]
    fn best_frame_time() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(0, 10.0, vec![]));
        history.push(make_profile(1, 50.0, vec![]));
        history.push(make_profile(2, 25.0, vec![]));
        assert!((history.best_frame_time() - 10.0).abs() < 0.01);
    }

    #[test]
    fn average_pass_time() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(0, 16.0, vec![("ray_march", 8.0), ("shade", 4.0)]));
        history.push(make_profile(1, 18.0, vec![("ray_march", 10.0), ("shade", 6.0)]));
        assert!((history.average_pass_time("ray_march").unwrap() - 9.0).abs() < 0.01);
        assert!((history.average_pass_time("shade").unwrap() - 5.0).abs() < 0.01);
        assert!(history.average_pass_time("nonexistent").is_none());
    }

    #[test]
    fn percentile_frame_time() {
        let mut history = ProfileHistory::new(100);
        // Push 100 frames: 1.0, 2.0, ..., 100.0
        for i in 1..=100 {
            history.push(make_profile(i, i as f32, vec![]));
        }
        let p50 = history.percentile_frame_time(50.0);
        assert!((p50 - 50.0).abs() < 1.5, "p50 = {}", p50);
        let p99 = history.percentile_frame_time(99.0);
        assert!((p99 - 99.0).abs() < 1.5, "p99 = {}", p99);
    }

    #[test]
    fn empty_history_returns_zeros() {
        let history = ProfileHistory::new(10);
        assert_eq!(history.average_frame_time(), 0.0);
        assert_eq!(history.worst_frame_time(), 0.0);
        assert_eq!(history.percentile_frame_time(99.0), 0.0);
    }

    #[test]
    fn clear_history() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(0, 16.0, vec![]));
        history.push(make_profile(1, 17.0, vec![]));
        history.clear();
        assert_eq!(history.frame_count(), 0);
    }

    #[test]
    fn cpu_timer_elapsed() {
        let timer = CpuTimer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ms = timer.elapsed_ms();
        assert!(ms >= 5.0, "expected >= 5ms, got {}", ms);
    }

    #[test]
    fn profiler_begin_end_frame() {
        let config = ProfilingConfig {
            enabled: true,
            ..Default::default()
        };
        let mut profiler = Profiler::new(config);

        profiler.begin_frame();
        let timer = profiler.begin_pass("ray_march");
        profiler.end_pass("ray_march", timer, None);
        profiler.end_frame();

        assert_eq!(profiler.history().frame_count(), 1);
    }

    #[test]
    fn profiler_disabled_skips_frame() {
        let config = ProfilingConfig {
            enabled: false,
            ..Default::default()
        };
        let mut profiler = Profiler::new(config);

        profiler.begin_frame();
        profiler.end_frame();

        // When disabled, begin_frame doesn't create a frame, so end_frame has nothing to push
        assert_eq!(profiler.history().frame_count(), 0);
    }

    #[test]
    fn profiler_is_enabled() {
        let enabled = Profiler::new(ProfilingConfig {
            enabled: true,
            ..Default::default()
        });
        assert!(enabled.is_enabled());

        let disabled = Profiler::new(ProfilingConfig {
            enabled: false,
            ..Default::default()
        });
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn summary_string_contains_pass_names() {
        let mut history = ProfileHistory::new(10);
        history.push(make_profile(
            0,
            16.5,
            vec![("ray_march", 8.0), ("shade", 4.0), ("volumetrics", 3.0)],
        ));
        let config = ProfilingConfig::default();
        let profiler = Profiler {
            config,
            history,
            current_frame: None,
            frame_counter: 1,
            frame_start: None,
        };
        let summary = profiler.summary_string();
        assert!(summary.contains("ray_march"), "summary missing ray_march");
        assert!(summary.contains("shade"), "summary missing shade");
        assert!(summary.contains("volumetrics"), "summary missing volumetrics");
        assert!(summary.contains("Performance Summary"), "summary missing header");
    }

    #[test]
    fn profiling_config_default() {
        let config = ProfilingConfig::default();
        assert!(config.enabled);
        assert!(!config.gpu_timing_enabled);
        assert_eq!(config.history_length, 300);
        assert_eq!(config.log_interval_frames, 300);
    }
}
