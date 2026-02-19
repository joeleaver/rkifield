#![allow(dead_code)]
//! GPU memory audit and leak detection for the RKIField engine.
//!
//! Tracks per-pool memory usage, detects monotonic allocation trends
//! (potential leaks), and provides formatted summary reports.

use std::collections::VecDeque;

/// Standard pool name for SDF brick data.
pub const POOL_SDF_BRICKS: &str = "sdf_bricks";
/// Standard pool name for per-voxel color brick data.
pub const POOL_COLOR_BRICKS: &str = "color_bricks";
/// Standard pool name for bone/animation brick data.
pub const POOL_BONE_BRICKS: &str = "bone_bricks";
/// Standard pool name for volumetric brick data.
pub const POOL_VOLUMETRIC_BRICKS: &str = "volumetric_bricks";
/// Standard pool name for CPU-to-GPU staging buffers.
pub const POOL_STAGING: &str = "staging";

/// Memory statistics for a single GPU pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Human-readable pool name.
    pub name: String,
    /// Total pool capacity in bytes.
    pub capacity_bytes: u64,
    /// Currently used bytes.
    pub used_bytes: u64,
    /// Peak used bytes since last reset.
    pub peak_bytes: u64,
    /// Total number of allocations.
    pub allocation_count: u32,
    /// Total number of deallocations.
    pub deallocation_count: u32,
}

impl PoolStats {
    /// Current utilization as a fraction (0.0 - 1.0).
    pub fn utilization(&self) -> f32 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.capacity_bytes as f32
    }

    /// Peak utilization as a fraction (0.0 - 1.0).
    pub fn peak_utilization(&self) -> f32 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.peak_bytes as f32 / self.capacity_bytes as f32
    }

    /// Net allocations (allocations minus deallocations). Can be negative.
    pub fn net_allocations(&self) -> i64 {
        self.allocation_count as i64 - self.deallocation_count as i64
    }

    /// Whether usage exceeds the given threshold (0.0 - 1.0).
    pub fn is_over_budget(&self, threshold: f32) -> bool {
        self.utilization() > threshold
    }
}

/// A snapshot of all pool statistics at a point in time.
#[derive(Debug, Clone)]
pub struct MemoryAudit {
    /// Per-pool statistics.
    pub pools: Vec<PoolStats>,
    /// Timestamp in milliseconds since some epoch.
    pub timestamp_ms: u64,
    /// Frame number when this audit was taken.
    pub frame_number: u64,
}

impl MemoryAudit {
    /// Create a new empty audit.
    pub fn new(frame_number: u64, timestamp_ms: u64) -> Self {
        Self {
            pools: Vec::new(),
            timestamp_ms,
            frame_number,
        }
    }

    /// Add pool statistics to this audit.
    pub fn add_pool(&mut self, stats: PoolStats) {
        self.pools.push(stats);
    }

    /// Total used bytes across all pools.
    pub fn total_used_bytes(&self) -> u64 {
        self.pools.iter().map(|p| p.used_bytes).sum()
    }

    /// Total capacity bytes across all pools.
    pub fn total_capacity_bytes(&self) -> u64 {
        self.pools.iter().map(|p| p.capacity_bytes).sum()
    }

    /// Overall utilization across all pools.
    pub fn total_utilization(&self) -> f32 {
        let cap = self.total_capacity_bytes();
        if cap == 0 {
            return 0.0;
        }
        self.total_used_bytes() as f32 / cap as f32
    }

    /// Find pool stats by name.
    pub fn find_pool(&self, name: &str) -> Option<&PoolStats> {
        self.pools.iter().find(|p| p.name == name)
    }

    /// Return all pools exceeding the given utilization threshold.
    pub fn pools_over_budget(&self, threshold: f32) -> Vec<&PoolStats> {
        self.pools
            .iter()
            .filter(|p| p.is_over_budget(threshold))
            .collect()
    }

    /// Generate a formatted multi-line summary.
    pub fn summary_string(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "=== Memory Audit (frame {}) ===\n",
            self.frame_number
        ));
        s.push_str(&format!(
            "Total: {:.1} MB / {:.1} MB ({:.1}%)\n",
            self.total_used_bytes() as f64 / (1024.0 * 1024.0),
            self.total_capacity_bytes() as f64 / (1024.0 * 1024.0),
            self.total_utilization() * 100.0,
        ));
        for pool in &self.pools {
            s.push_str(&format!(
                "  {}: {:.1} MB / {:.1} MB ({:.1}%) [peak {:.1}%] alloc={} dealloc={}\n",
                pool.name,
                pool.used_bytes as f64 / (1024.0 * 1024.0),
                pool.capacity_bytes as f64 / (1024.0 * 1024.0),
                pool.utilization() * 100.0,
                pool.peak_utilization() * 100.0,
                pool.allocation_count,
                pool.deallocation_count,
            ));
        }
        s
    }
}

/// Rolling history of memory audits for trend analysis.
#[derive(Debug)]
pub struct MemoryHistory {
    snapshots: VecDeque<MemoryAudit>,
    max_length: usize,
}

impl MemoryHistory {
    /// Create a new history with the given maximum capacity.
    pub fn new(max_length: usize) -> Self {
        Self {
            snapshots: VecDeque::with_capacity(max_length.min(64)),
            max_length,
        }
    }

    /// Push an audit snapshot, evicting the oldest if at capacity.
    pub fn push(&mut self, audit: MemoryAudit) {
        if self.snapshots.len() >= self.max_length {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(audit);
    }

    /// Most recent audit snapshot.
    pub fn latest(&self) -> Option<&MemoryAudit> {
        self.snapshots.back()
    }

    /// Average utilization of a named pool across all snapshots.
    pub fn average_utilization(&self, pool_name: &str) -> Option<f32> {
        let mut sum = 0.0_f32;
        let mut count = 0u32;
        for audit in &self.snapshots {
            if let Some(pool) = audit.find_pool(pool_name) {
                sum += pool.utilization();
                count += 1;
            }
        }
        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    /// Peak utilization of a named pool across all snapshots.
    pub fn peak_utilization(&self, pool_name: &str) -> Option<f32> {
        let mut peak = None;
        for audit in &self.snapshots {
            if let Some(pool) = audit.find_pool(pool_name) {
                let u = pool.utilization();
                peak = Some(peak.map_or(u, |p: f32| p.max(u)));
            }
        }
        peak
    }

    /// Heuristic: does the named pool show a monotonically increasing
    /// `used_bytes` trend over the last 30+ snapshots with non-zero net allocations?
    pub fn has_leak_indication(&self, pool_name: &str) -> bool {
        if self.snapshots.len() < 30 {
            return false;
        }

        let usage: Vec<(u64, i64)> = self
            .snapshots
            .iter()
            .filter_map(|a| {
                a.find_pool(pool_name)
                    .map(|p| (p.used_bytes, p.net_allocations()))
            })
            .collect();

        if usage.len() < 30 {
            return false;
        }

        // Check last 30 entries for monotonic increase with non-zero net allocs
        let tail = &usage[usage.len() - 30..];
        let monotonic = tail.windows(2).all(|w| w[1].0 >= w[0].0);
        let has_net_allocs = tail.iter().any(|(_, net)| *net > 0);
        // Require actual growth (first != last)
        let grew = tail.last().unwrap().0 > tail.first().unwrap().0;

        monotonic && has_net_allocs && grew
    }

    /// Number of stored snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }
}

/// Report of a suspected memory leak in a specific pool.
#[derive(Debug, Clone)]
pub struct LeakReport {
    /// Name of the pool with suspected leak.
    pub pool_name: String,
    /// Estimated bytes-per-frame growth rate.
    pub trend_bytes_per_frame: f64,
    /// Number of frames observed in the trend window.
    pub frames_observed: u32,
}

/// Analyze memory history for each pool and return leak reports.
pub fn detect_leaks(history: &MemoryHistory) -> Vec<LeakReport> {
    if history.snapshots.len() < 30 {
        return Vec::new();
    }

    // Collect all unique pool names
    let mut pool_names: Vec<String> = Vec::new();
    for audit in &history.snapshots {
        for pool in &audit.pools {
            if !pool_names.contains(&pool.name) {
                pool_names.push(pool.name.clone());
            }
        }
    }

    let mut reports = Vec::new();
    for name in &pool_names {
        if history.has_leak_indication(name) {
            // Calculate trend
            let usage: Vec<u64> = history
                .snapshots
                .iter()
                .filter_map(|a| a.find_pool(name).map(|p| p.used_bytes))
                .collect();
            let window = &usage[usage.len().saturating_sub(30)..];
            if window.len() >= 2 {
                let first = window.first().copied().unwrap_or(0) as f64;
                let last = window.last().copied().unwrap_or(0) as f64;
                let frames = (window.len() - 1) as f64;
                let trend = if frames > 0.0 {
                    (last - first) / frames
                } else {
                    0.0
                };
                reports.push(LeakReport {
                    pool_name: name.clone(),
                    trend_bytes_per_frame: trend,
                    frames_observed: window.len() as u32,
                });
            }
        }
    }

    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(name: &str, capacity: u64, used: u64, peak: u64, allocs: u32, deallocs: u32) -> PoolStats {
        PoolStats {
            name: name.to_string(),
            capacity_bytes: capacity,
            used_bytes: used,
            peak_bytes: peak,
            allocation_count: allocs,
            deallocation_count: deallocs,
        }
    }

    fn make_audit(frame: u64, pools: Vec<PoolStats>) -> MemoryAudit {
        let mut audit = MemoryAudit::new(frame, frame * 16);
        for p in pools {
            audit.add_pool(p);
        }
        audit
    }

    #[test]
    fn pool_utilization() {
        let pool = make_pool("test", 1000, 750, 900, 10, 5);
        assert!((pool.utilization() - 0.75).abs() < 0.001);
        assert!((pool.peak_utilization() - 0.9).abs() < 0.001);
    }

    #[test]
    fn pool_utilization_zero_capacity() {
        let pool = make_pool("empty", 0, 0, 0, 0, 0);
        assert_eq!(pool.utilization(), 0.0);
        assert_eq!(pool.peak_utilization(), 0.0);
    }

    #[test]
    fn pool_net_allocations() {
        let pool = make_pool("test", 1000, 500, 500, 100, 30);
        assert_eq!(pool.net_allocations(), 70);
    }

    #[test]
    fn pool_net_allocations_negative() {
        let pool = make_pool("test", 1000, 100, 500, 10, 40);
        assert_eq!(pool.net_allocations(), -30);
    }

    #[test]
    fn pool_over_budget() {
        let pool = make_pool("test", 1000, 950, 950, 10, 0);
        assert!(pool.is_over_budget(0.9));
        assert!(!pool.is_over_budget(0.96));
    }

    #[test]
    fn audit_totals() {
        let audit = make_audit(
            0,
            vec![
                make_pool(POOL_SDF_BRICKS, 1000, 500, 700, 10, 5),
                make_pool(POOL_COLOR_BRICKS, 2000, 800, 1200, 20, 10),
            ],
        );
        assert_eq!(audit.total_used_bytes(), 1300);
        assert_eq!(audit.total_capacity_bytes(), 3000);
        assert!((audit.total_utilization() - (1300.0 / 3000.0)).abs() < 0.001);
    }

    #[test]
    fn audit_find_pool() {
        let audit = make_audit(
            0,
            vec![
                make_pool(POOL_SDF_BRICKS, 1000, 500, 700, 10, 5),
                make_pool(POOL_STAGING, 500, 100, 200, 5, 3),
            ],
        );
        assert!(audit.find_pool(POOL_SDF_BRICKS).is_some());
        assert!(audit.find_pool(POOL_STAGING).is_some());
        assert!(audit.find_pool("nonexistent").is_none());
    }

    #[test]
    fn audit_pools_over_budget() {
        let audit = make_audit(
            0,
            vec![
                make_pool("high", 1000, 950, 950, 10, 0),
                make_pool("low", 1000, 100, 200, 5, 3),
                make_pool("medium", 1000, 850, 900, 8, 4),
            ],
        );
        let over = audit.pools_over_budget(0.9);
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].name, "high");
    }

    #[test]
    fn audit_summary_string_format() {
        let audit = make_audit(
            42,
            vec![
                make_pool(POOL_SDF_BRICKS, 1024 * 1024, 512 * 1024, 768 * 1024, 100, 50),
            ],
        );
        let summary = audit.summary_string();
        assert!(summary.contains("Memory Audit"));
        assert!(summary.contains("frame 42"));
        assert!(summary.contains(POOL_SDF_BRICKS));
    }

    #[test]
    fn history_push_and_latest() {
        let mut history = MemoryHistory::new(5);
        history.push(make_audit(0, vec![]));
        history.push(make_audit(1, vec![]));
        assert_eq!(history.snapshot_count(), 2);
        assert_eq!(history.latest().unwrap().frame_number, 1);
    }

    #[test]
    fn history_caps_at_max() {
        let mut history = MemoryHistory::new(3);
        for i in 0..10 {
            history.push(make_audit(i, vec![]));
        }
        assert_eq!(history.snapshot_count(), 3);
        assert_eq!(history.snapshots.front().unwrap().frame_number, 7);
    }

    #[test]
    fn history_average_utilization() {
        let mut history = MemoryHistory::new(10);
        history.push(make_audit(0, vec![make_pool("p", 1000, 200, 200, 5, 0)]));
        history.push(make_audit(1, vec![make_pool("p", 1000, 400, 400, 10, 0)]));
        history.push(make_audit(2, vec![make_pool("p", 1000, 600, 600, 15, 0)]));
        let avg = history.average_utilization("p").unwrap();
        assert!((avg - 0.4).abs() < 0.001); // (0.2 + 0.4 + 0.6) / 3 = 0.4
    }

    #[test]
    fn history_peak_utilization() {
        let mut history = MemoryHistory::new(10);
        history.push(make_audit(0, vec![make_pool("p", 1000, 200, 200, 5, 0)]));
        history.push(make_audit(1, vec![make_pool("p", 1000, 800, 800, 10, 0)]));
        history.push(make_audit(2, vec![make_pool("p", 1000, 300, 300, 15, 0)]));
        let peak = history.peak_utilization("p").unwrap();
        assert!((peak - 0.8).abs() < 0.001);
    }

    #[test]
    fn leak_detection_monotonic_increase() {
        let mut history = MemoryHistory::new(60);
        // 40 frames with monotonically increasing usage and positive net allocs
        for i in 0..40 {
            let used = 1000 + i * 100;
            history.push(make_audit(
                i,
                vec![make_pool("leaky", 100000, used, used, (i + 1) as u32, 0)],
            ));
        }
        assert!(history.has_leak_indication("leaky"));
        let reports = detect_leaks(&history);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].pool_name, "leaky");
        assert!(reports[0].trend_bytes_per_frame > 0.0);
    }

    #[test]
    fn no_false_positive_leaks_fluctuating() {
        let mut history = MemoryHistory::new(60);
        // 40 frames with fluctuating usage (not monotonic)
        for i in 0..40u64 {
            let used = if i % 2 == 0 { 5000 } else { 3000 };
            history.push(make_audit(
                i,
                vec![make_pool("stable", 100000, used, 5000, 10, 8)],
            ));
        }
        assert!(!history.has_leak_indication("stable"));
        let reports = detect_leaks(&history);
        assert!(reports.is_empty());
    }

    #[test]
    fn no_leak_with_insufficient_history() {
        let mut history = MemoryHistory::new(60);
        for i in 0..10 {
            history.push(make_audit(
                i,
                vec![make_pool("short", 100000, 1000 + i * 100, 5000, (i + 1) as u32, 0)],
            ));
        }
        assert!(!history.has_leak_indication("short"));
        let reports = detect_leaks(&history);
        assert!(reports.is_empty());
    }

    #[test]
    fn pool_name_constants() {
        // Verify constants are distinct and non-empty
        let names = [
            POOL_SDF_BRICKS,
            POOL_COLOR_BRICKS,
            POOL_BONE_BRICKS,
            POOL_VOLUMETRIC_BRICKS,
            POOL_STAGING,
        ];
        for name in &names {
            assert!(!name.is_empty());
        }
        // All unique
        let mut sorted = names.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }

    #[test]
    fn empty_audit_totals() {
        let audit = MemoryAudit::new(0, 0);
        assert_eq!(audit.total_used_bytes(), 0);
        assert_eq!(audit.total_capacity_bytes(), 0);
        assert_eq!(audit.total_utilization(), 0.0);
    }
}
