//! Streaming budget enforcement for the brick pool streaming system.
//!
//! This module tracks memory utilization (brick pool + staging buffer) and
//! I/O bandwidth, throttling new chunk loads when resource limits are reached.

const BYTES_PER_MB: u64 = 1024 * 1024;

/// Memory and I/O budget for the streaming system.
///
/// Tracks brick pool utilization and staging buffer usage,
/// throttling chunk loads when budgets are exceeded.
#[derive(Debug, Clone)]
pub struct StreamingBudget {
    /// Maximum brick pool size in megabytes (e.g., 512).
    pub max_pool_mb: u32,
    /// Maximum staging buffer size in megabytes (e.g., 64).
    pub max_staging_mb: u32,
    /// Maximum I/O bandwidth in MB/s (e.g., 200.0).
    pub max_io_bandwidth_mb_s: f32,
    /// Start throttling at this pool utilization fraction (0.0–1.0).
    pub throttle_threshold: f32,
}

impl Default for StreamingBudget {
    fn default() -> Self {
        Self {
            max_pool_mb: 512,
            max_staging_mb: 64,
            max_io_bandwidth_mb_s: 200.0,
            throttle_threshold: 0.85,
        }
    }
}

/// Current resource utilization tracked by the budget system.
#[derive(Debug, Clone, Default)]
pub struct BudgetState {
    /// Current pool usage in bytes.
    pub pool_used_bytes: u64,
    /// Current staging buffer usage in bytes.
    pub staging_used_bytes: u64,
    /// Bytes loaded this second (rolling window for bandwidth).
    pub io_bytes_this_second: u64,
    /// Timestamp of the current second window start (frame number).
    pub io_window_start_frame: u64,
    /// Frames per second estimate (for bandwidth calculation).
    pub fps_estimate: f32,
}

/// Monitors streaming resource usage and enforces budgets.
///
/// Call `update_pool_usage()` / `update_staging_usage()` / `record_io()` each
/// frame with current utilization.  Query `can_load()` to check whether a new
/// chunk load is allowed.
pub struct BudgetMonitor {
    /// Budget configuration.
    budget: StreamingBudget,
    /// Current utilization state.
    state: BudgetState,
}

impl BudgetMonitor {
    /// Create a new monitor with the given budget and zeroed state.
    pub fn new(budget: StreamingBudget) -> Self {
        Self {
            budget,
            state: BudgetState::default(),
        }
    }

    /// Update pool usage from the number of allocated bricks and their size.
    ///
    /// `pool_used_bytes` is set to `allocated_bricks * brick_size_bytes`.
    pub fn update_pool_usage(&mut self, allocated_bricks: u32, brick_size_bytes: u32) {
        self.state.pool_used_bytes =
            allocated_bricks as u64 * brick_size_bytes as u64;
    }

    /// Update the staging buffer utilization.
    pub fn update_staging_usage(&mut self, staging_bytes: u64) {
        self.state.staging_used_bytes = staging_bytes;
    }

    /// Record bytes loaded in the current frame for bandwidth accounting.
    ///
    /// The rolling window is `fps_estimate` frames wide (≈ 1 second).
    /// When `current_frame` has advanced past the window start by more than
    /// `fps_estimate` frames, the counter resets to the bytes just recorded.
    pub fn record_io(&mut self, bytes_loaded: u64, current_frame: u64) {
        let window_frames = self.state.fps_estimate.max(1.0) as u64;
        if current_frame >= self.state.io_window_start_frame + window_frames {
            // New window: reset counter.
            self.state.io_bytes_this_second = bytes_loaded;
            self.state.io_window_start_frame = current_frame;
        } else {
            self.state.io_bytes_this_second += bytes_loaded;
        }
    }

    /// Update the frames-per-second estimate used for bandwidth calculations.
    pub fn set_fps(&mut self, fps: f32) {
        self.state.fps_estimate = fps.max(1.0);
    }

    /// Returns `true` if all budget conditions allow a new chunk load.
    ///
    /// Conditions:
    /// 1. Pool usage is below the pool limit.
    /// 2. Staging usage is below the staging limit.
    /// 3. Either pool utilization is below the throttle threshold *or*
    ///    I/O bandwidth is under the configured limit.
    pub fn can_load(&self) -> bool {
        let max_pool_bytes = self.budget.max_pool_mb as u64 * BYTES_PER_MB;
        let max_staging_bytes = self.budget.max_staging_mb as u64 * BYTES_PER_MB;

        if self.state.pool_used_bytes >= max_pool_bytes {
            return false;
        }
        if self.state.staging_used_bytes >= max_staging_bytes {
            return false;
        }

        // Not throttled OR I/O bandwidth is available — either is fine.
        !self.is_throttled() || self.io_bandwidth_utilization() < 1.0
    }

    /// Current pool usage as a fraction of the maximum (0.0–1.0+).
    pub fn pool_utilization(&self) -> f32 {
        let max_bytes = self.budget.max_pool_mb as u64 * BYTES_PER_MB;
        if max_bytes == 0 {
            return 0.0;
        }
        self.state.pool_used_bytes as f32 / max_bytes as f32
    }

    /// Current staging buffer usage as a fraction of the maximum (0.0–1.0+).
    pub fn staging_utilization(&self) -> f32 {
        let max_bytes = self.budget.max_staging_mb as u64 * BYTES_PER_MB;
        if max_bytes == 0 {
            return 0.0;
        }
        self.state.staging_used_bytes as f32 / max_bytes as f32
    }

    /// I/O bandwidth utilization as a fraction of the configured maximum.
    ///
    /// Computed as `(io_bytes_this_second / fps_estimate) / (max_io_mb_s * 1MB)`.
    /// A value ≥ 1.0 means the bandwidth limit is exceeded.
    pub fn io_bandwidth_utilization(&self) -> f32 {
        // bytes accumulated over one window (≈ 1 second of I/O).
        let io_bytes_per_second = self.state.io_bytes_this_second as f32;
        let max_bytes_per_second =
            self.budget.max_io_bandwidth_mb_s * BYTES_PER_MB as f32;
        if max_bytes_per_second == 0.0 {
            return 0.0;
        }
        io_bytes_per_second / max_bytes_per_second
    }

    /// Returns `true` when pool utilization meets or exceeds the throttle threshold.
    pub fn is_throttled(&self) -> bool {
        self.pool_utilization() >= self.budget.throttle_threshold
    }

    /// Read-only access to the budget configuration.
    pub fn budget(&self) -> &StreamingBudget {
        &self.budget
    }

    /// Mutable access to the budget configuration.
    pub fn budget_mut(&mut self) -> &mut StreamingBudget {
        &mut self.budget
    }

    /// Read-only access to the current utilization state.
    pub fn state(&self) -> &BudgetState {
        &self.state
    }

    /// Remaining pool capacity in bytes (saturating at 0).
    pub fn pool_headroom_bytes(&self) -> u64 {
        let max_pool_bytes = self.budget.max_pool_mb as u64 * BYTES_PER_MB;
        max_pool_bytes.saturating_sub(self.state.pool_used_bytes)
    }

    /// Remaining pool capacity expressed as a number of bricks.
    ///
    /// Returns 0 if `brick_size_bytes` is 0 or the pool is full.
    pub fn pool_headroom_bricks(&self, brick_size_bytes: u32) -> u32 {
        if brick_size_bytes == 0 {
            return 0;
        }
        (self.pool_headroom_bytes() / brick_size_bytes as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRICK_SIZE: u32 = 4096; // 4 KB per brick

    fn default_monitor() -> BudgetMonitor {
        BudgetMonitor::new(StreamingBudget::default())
    }

    // 1. New monitor has zeroed state.
    #[test]
    fn new_monitor_default() {
        let m = default_monitor();
        assert_eq!(m.state().pool_used_bytes, 0);
        assert_eq!(m.state().staging_used_bytes, 0);
        assert_eq!(m.state().io_bytes_this_second, 0);
        assert_eq!(m.state().fps_estimate, 0.0);
    }

    // 2. update_pool_usage sets pool_used_bytes correctly.
    #[test]
    fn update_pool_usage() {
        let mut m = default_monitor();
        m.update_pool_usage(10, BRICK_SIZE);
        assert_eq!(m.state().pool_used_bytes, 10 * BRICK_SIZE as u64);
    }

    // 3. pool_utilization: 256 MB of 512 MB = 0.5.
    #[test]
    fn pool_utilization_calculation() {
        let mut m = default_monitor();
        let half_pool = 256u64 * BYTES_PER_MB;
        let bricks = half_pool / BRICK_SIZE as u64;
        m.update_pool_usage(bricks as u32, BRICK_SIZE);
        let util = m.pool_utilization();
        assert!((util - 0.5).abs() < 1e-4, "expected ~0.5 got {util}");
    }

    // 4. staging_utilization: 32 MB of 64 MB = 0.5.
    #[test]
    fn staging_utilization_calculation() {
        let mut m = default_monitor();
        let half_staging = 32u64 * BYTES_PER_MB;
        m.update_staging_usage(half_staging);
        let util = m.staging_utilization();
        assert!((util - 0.5).abs() < 1e-4, "expected ~0.5 got {util}");
    }

    // 5. can_load returns true when everything is under budget.
    #[test]
    fn can_load_when_under_budget() {
        let mut m = default_monitor();
        m.update_pool_usage(1, BRICK_SIZE); // tiny pool usage
        m.update_staging_usage(1024);       // tiny staging usage
        assert!(m.can_load());
    }

    // 6. cannot load when pool is at max.
    #[test]
    fn cannot_load_pool_full() {
        let mut m = default_monitor();
        let max_bytes = m.budget().max_pool_mb as u64 * BYTES_PER_MB;
        let bricks = max_bytes / BRICK_SIZE as u64;
        m.update_pool_usage(bricks as u32, BRICK_SIZE);
        assert!(!m.can_load());
    }

    // 7. cannot load when staging is at max.
    #[test]
    fn cannot_load_staging_full() {
        let mut m = default_monitor();
        let max_staging = m.budget().max_staging_mb as u64 * BYTES_PER_MB;
        m.update_staging_usage(max_staging);
        assert!(!m.can_load());
    }

    // 8. is_throttled when pool utilization >= throttle_threshold.
    #[test]
    fn throttle_threshold() {
        let mut m = default_monitor(); // threshold = 0.85
        // 90% of 512 MB
        let target_bytes = (512u64 * BYTES_PER_MB * 90) / 100;
        let bricks = target_bytes / BRICK_SIZE as u64;
        m.update_pool_usage(bricks as u32, BRICK_SIZE);
        assert!(m.is_throttled());
        assert!(m.pool_utilization() >= 0.85);
    }

    // 9. record_io tracks bandwidth utilization.
    #[test]
    fn record_io_tracks_bandwidth() {
        let mut m = default_monitor();
        m.set_fps(60.0);
        // Record 100 MB in one window — max is 200 MB/s, so utilization ≈ 0.5.
        let one_hundred_mb = 100u64 * BYTES_PER_MB;
        m.record_io(one_hundred_mb, 0);
        let util = m.io_bandwidth_utilization();
        assert!((util - 0.5).abs() < 1e-4, "expected ~0.5 got {util}");
    }

    // 10. io counter resets when the frame window advances.
    #[test]
    fn io_window_resets() {
        let mut m = default_monitor();
        m.set_fps(60.0);
        // Record 100 MB at frame 0.
        m.record_io(100 * BYTES_PER_MB, 0);
        assert_eq!(m.state().io_bytes_this_second, 100 * BYTES_PER_MB);

        // Advance past one full window (60+ frames).
        m.record_io(10 * BYTES_PER_MB, 60);
        // Counter should reset to just the new bytes.
        assert_eq!(m.state().io_bytes_this_second, 10 * BYTES_PER_MB);
    }

    // 11. pool_headroom_bytes is correct.
    #[test]
    fn pool_headroom_bytes() {
        let mut m = default_monitor(); // max 512 MB
        let used = 256u64 * BYTES_PER_MB;
        let bricks = used / BRICK_SIZE as u64;
        m.update_pool_usage(bricks as u32, BRICK_SIZE);
        let headroom = m.pool_headroom_bytes();
        let expected = 256u64 * BYTES_PER_MB;
        // Allow for rounding (brick granularity).
        assert!(
            headroom.abs_diff(expected) < BRICK_SIZE as u64,
            "headroom {headroom} far from expected {expected}"
        );
    }

    // 12. pool_headroom_bricks is headroom / brick_size.
    #[test]
    fn pool_headroom_bricks() {
        let mut m = default_monitor(); // max 512 MB
        m.update_pool_usage(0, BRICK_SIZE);
        let bricks = m.pool_headroom_bricks(BRICK_SIZE);
        let expected = (512u64 * BYTES_PER_MB / BRICK_SIZE as u64) as u32;
        assert_eq!(bricks, expected);
    }

    // 13. budget_mut allows modifying the budget after construction.
    #[test]
    fn budget_mut_updates() {
        let mut m = default_monitor();
        m.budget_mut().max_pool_mb = 1024;
        assert_eq!(m.budget().max_pool_mb, 1024);
        // Pool should now be effectively empty relative to new limit.
        assert!(m.pool_utilization() < 0.01);
    }
}
