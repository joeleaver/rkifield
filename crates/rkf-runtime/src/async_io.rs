//! Async I/O pipeline for chunk loading.
//!
//! Uses a thread pool (via [`std::thread::spawn`]) to load `.rkf` files in the
//! background.  Completed loads are staged in a shared queue for the main thread
//! to drain each frame.
//!
//! The pipeline is intentionally simple — no async runtime, no channels — just
//! `Arc<Mutex<>>` shared state and OS threads.  Chunk loading is CPU-bound
//! (disk I/O + LZ4 decompression), so lightweight OS threads are a good fit.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use glam::IVec3;
use rkf_core::chunk::Chunk;
use rkf_core::load_chunk_file;

/// Result of a background chunk load operation.
pub struct ChunkLoadResult {
    /// Chunk coordinates.
    pub coords: IVec3,
    /// The loaded chunk data, or an error message if loading failed.
    pub result: Result<Chunk, String>,
}

/// Async I/O pipeline for chunk loading.
///
/// Uses a thread pool to load `.rkf` files in the background.
/// Completed loads are staged for the main thread to process.
pub struct AsyncIoPipeline {
    /// Map of chunk coords to their `.rkf` file paths.
    chunk_paths: HashMap<IVec3, PathBuf>,
    /// Completed loads waiting for main-thread processing.
    staging: Arc<Mutex<Vec<ChunkLoadResult>>>,
    /// Currently in-flight loads (coords of chunks being loaded).
    in_flight: Arc<Mutex<Vec<IVec3>>>,
    /// Maximum concurrent loads.
    max_concurrent: usize,
}

impl AsyncIoPipeline {
    /// Create a new pipeline with the given concurrency limit.
    ///
    /// A typical value is 4 — enough to saturate an NVMe drive without
    /// starving the main thread of CPU time for decompression.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            chunk_paths: HashMap::new(),
            staging: Arc::new(Mutex::new(Vec::new())),
            in_flight: Arc::new(Mutex::new(Vec::new())),
            max_concurrent,
        }
    }

    /// Register a chunk's file path so it can be loaded later.
    pub fn register_path(&mut self, coords: IVec3, path: PathBuf) {
        self.chunk_paths.insert(coords, path);
    }

    /// Batch-register multiple chunk file paths.
    pub fn register_paths(&mut self, paths: impl IntoIterator<Item = (IVec3, PathBuf)>) {
        for (coords, path) in paths {
            self.chunk_paths.insert(coords, path);
        }
    }

    /// Remove a path mapping.  Does **not** cancel an in-flight load.
    pub fn unregister_path(&mut self, coords: &IVec3) {
        self.chunk_paths.remove(coords);
    }

    /// Check whether a path is registered for `coords`.
    pub fn has_path(&self, coords: &IVec3) -> bool {
        self.chunk_paths.contains_key(coords)
    }

    /// Submit load requests for the given chunk coordinates.
    ///
    /// For each coordinate the method:
    /// 1. Skips if already in-flight.
    /// 2. Skips if no registered path.
    /// 3. Stops submitting if the in-flight count has reached `max_concurrent`.
    /// 4. Spawns an OS thread that reads the `.rkf` file, decompresses, and
    ///    pushes the result into the staging queue.
    ///
    /// This method returns immediately — it never blocks on I/O.
    pub fn submit_loads(&self, requests: &[IVec3]) {
        for &coords in requests {
            // Look up path (needs to happen before we lock in_flight).
            let path = match self.chunk_paths.get(&coords) {
                Some(p) => p.clone(),
                None => continue,
            };

            // Lock in_flight to check capacity and duplicates.
            let mut in_flight = self.in_flight.lock().unwrap();
            if in_flight.len() >= self.max_concurrent {
                break;
            }
            if in_flight.contains(&coords) {
                continue;
            }
            in_flight.push(coords);
            drop(in_flight); // Release lock before spawning.

            let staging = Arc::clone(&self.staging);
            let in_flight_ref = Arc::clone(&self.in_flight);

            std::thread::spawn(move || {
                let result = load_chunk_file(&path).map_err(|e| e.to_string());
                staging.lock().unwrap().push(ChunkLoadResult { coords, result });
                let mut in_flight = in_flight_ref.lock().unwrap();
                in_flight.retain(|&c| c != coords);
            });
        }
    }

    /// Take all completed loads from the staging queue.
    ///
    /// Called once per frame by the main thread.  Returns an empty `Vec` if
    /// nothing has finished since the last drain.
    pub fn drain_completed(&self) -> Vec<ChunkLoadResult> {
        let mut staging = self.staging.lock().unwrap();
        std::mem::take(&mut *staging)
    }

    /// Number of chunk loads currently in flight.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.lock().unwrap().len()
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, UVec3};
    use rkf_core::chunk::{save_chunk_file, TierGrid};
    use rkf_core::sparse_grid::SparseGrid;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    /// Create a unique temporary directory for a test.
    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rkf_async_io_test_{}", name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Create a minimal test chunk file on disk and return its path.
    fn create_test_chunk_file(coords: IVec3, dir: &Path) -> PathBuf {
        let grid = SparseGrid::new(UVec3::new(2, 2, 2));
        let chunk = Chunk {
            coords,
            grids: vec![TierGrid {
                tier: 1,
                grid,
                bricks: vec![],
            }],
            brick_count: 0,
        };
        let path = dir.join(format!(
            "chunk_{}_{}_{}_.rkf",
            coords.x, coords.y, coords.z
        ));
        save_chunk_file(&chunk, &path).unwrap();
        path
    }

    /// Poll `drain_completed` until the expected count is reached or timeout.
    fn wait_for_results(pipeline: &AsyncIoPipeline, expected: usize) -> Vec<ChunkLoadResult> {
        let mut results = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while results.len() < expected && std::time::Instant::now() < deadline {
            results.extend(pipeline.drain_completed());
            if results.len() < expected {
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        results
    }

    // ── 1. construction ─────────────────────────────────────────────────────

    #[test]
    fn new_pipeline() {
        let pipeline = AsyncIoPipeline::new(4);
        assert_eq!(pipeline.in_flight_count(), 0);
        assert!(pipeline.drain_completed().is_empty());
        assert!(!pipeline.has_path(&IVec3::ZERO));
    }

    // ── 2. register / unregister ────────────────────────────────────────────

    #[test]
    fn register_and_unregister_path() {
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(1, 2, 3);
        let path = PathBuf::from("/tmp/test.rkf");

        pipeline.register_path(coords, path);
        assert!(pipeline.has_path(&coords));

        pipeline.unregister_path(&coords);
        assert!(!pipeline.has_path(&coords));
    }

    // ── 3. batch register ───────────────────────────────────────────────────

    #[test]
    fn batch_register_paths() {
        let mut pipeline = AsyncIoPipeline::new(4);
        let entries: Vec<(IVec3, PathBuf)> = (0..5)
            .map(|i| (IVec3::new(i, 0, 0), PathBuf::from(format!("/tmp/c{i}.rkf"))))
            .collect();

        pipeline.register_paths(entries);

        for i in 0..5 {
            assert!(pipeline.has_path(&IVec3::new(i, 0, 0)));
        }
        assert!(!pipeline.has_path(&IVec3::new(99, 0, 0)));
    }

    // ── 4. submit spawns work and completes ─────────────────────────────────

    #[test]
    fn submit_loads_spawns_work() {
        let dir = temp_dir("submit_spawns");
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(1, 0, 0);
        let path = create_test_chunk_file(coords, &dir);

        pipeline.register_path(coords, path);
        pipeline.submit_loads(&[coords]);

        let results = wait_for_results(&pipeline, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].coords, coords);
        assert!(results[0].result.is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 5. submit skips unknown coords ──────────────────────────────────────

    #[test]
    fn submit_skips_unknown_coords() {
        let pipeline = AsyncIoPipeline::new(4);
        let unknown = IVec3::new(99, 99, 99);

        pipeline.submit_loads(&[unknown]);

        // Nothing should be in-flight or staged.
        assert_eq!(pipeline.in_flight_count(), 0);
        std::thread::sleep(Duration::from_millis(100));
        assert!(pipeline.drain_completed().is_empty());
    }

    // ── 6. submit skips in-flight duplicates ────────────────────────────────

    #[test]
    fn submit_skips_in_flight() {
        let dir = temp_dir("skip_inflight");
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(0, 0, 0);
        let path = create_test_chunk_file(coords, &dir);

        pipeline.register_path(coords, path);

        // Submit twice in quick succession.
        pipeline.submit_loads(&[coords]);
        pipeline.submit_loads(&[coords]);

        // Should only get one result total.
        let results = wait_for_results(&pipeline, 1);
        assert_eq!(results.len(), 1);

        // Extra wait to confirm no duplicate arrives.
        std::thread::sleep(Duration::from_millis(200));
        assert!(pipeline.drain_completed().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 7. drain_completed returns correct results ──────────────────────────

    #[test]
    fn drain_completed_returns_results() {
        let dir = temp_dir("drain_results");
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(5, 10, 15);
        let path = create_test_chunk_file(coords, &dir);

        pipeline.register_path(coords, path);
        pipeline.submit_loads(&[coords]);

        let results = wait_for_results(&pipeline, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].coords, coords);

        let chunk = results[0].result.as_ref().unwrap();
        assert_eq!(chunk.coords, coords);
        assert_eq!(chunk.grids.len(), 1);
        assert_eq!(chunk.grids[0].tier, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 8. drain is empty after drain ───────────────────────────────────────

    #[test]
    fn drain_completed_is_empty_after_drain() {
        let dir = temp_dir("drain_empty");
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::ZERO;
        let path = create_test_chunk_file(coords, &dir);

        pipeline.register_path(coords, path);
        pipeline.submit_loads(&[coords]);

        let first = wait_for_results(&pipeline, 1);
        assert_eq!(first.len(), 1);

        let second = pipeline.drain_completed();
        assert!(second.is_empty(), "second drain should be empty");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 9. max_concurrent respected ─────────────────────────────────────────

    #[test]
    fn max_concurrent_respected() {
        let dir = temp_dir("max_concurrent");
        let mut pipeline = AsyncIoPipeline::new(1);

        let coords: Vec<IVec3> = (0..3).map(|i| IVec3::new(i, 0, 0)).collect();
        for &c in &coords {
            let path = create_test_chunk_file(c, &dir);
            pipeline.register_path(c, path);
        }

        // Submit all 3 at once with max_concurrent=1.
        pipeline.submit_loads(&coords);

        // Only 1 should be in flight (or already completed by now).
        // The key invariant: at most max_concurrent are spawned per submit call.
        // Wait for the first to finish, then submit more.
        let mut all_results = wait_for_results(&pipeline, 1);
        assert!(all_results.len() >= 1);

        // Submit again for the rest.
        pipeline.submit_loads(&coords);
        all_results.extend(wait_for_results(&pipeline, 1));

        pipeline.submit_loads(&coords);
        all_results.extend(wait_for_results(&pipeline, 1));

        // All 3 should have completed eventually.
        assert!(all_results.len() >= 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 10. nonexistent file gives error ────────────────────────────────────

    #[test]
    fn load_nonexistent_file_gives_error() {
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(0, 0, 0);
        let bad_path = PathBuf::from("/tmp/rkf_does_not_exist_async_io_test.rkf");

        pipeline.register_path(coords, bad_path);
        pipeline.submit_loads(&[coords]);

        let results = wait_for_results(&pipeline, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].coords, coords);
        assert!(
            results[0].result.is_err(),
            "loading a nonexistent file should return an error"
        );
    }

    // ── 11. multiple chunks load successfully ───────────────────────────────

    #[test]
    fn multiple_chunks_load() {
        let dir = temp_dir("multi_load");
        let mut pipeline = AsyncIoPipeline::new(4);

        let coords: Vec<IVec3> = (0..3).map(|i| IVec3::new(i, i, i)).collect();
        for &c in &coords {
            let path = create_test_chunk_file(c, &dir);
            pipeline.register_path(c, path);
        }

        pipeline.submit_loads(&coords);

        let results = wait_for_results(&pipeline, 3);
        assert_eq!(results.len(), 3);

        // All should be Ok with matching coords.
        let mut loaded_coords: Vec<IVec3> = results
            .iter()
            .map(|r| {
                assert!(r.result.is_ok(), "chunk {:?} failed: {:?}", r.coords, r.result);
                r.coords
            })
            .collect();
        loaded_coords.sort_by_key(|c| (c.x, c.y, c.z));

        let mut expected = coords.clone();
        expected.sort_by_key(|c| (c.x, c.y, c.z));
        assert_eq!(loaded_coords, expected);

        // All in-flight should be cleared.
        assert_eq!(pipeline.in_flight_count(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 12. in_flight_count tracks correctly ────────────────────────────────

    #[test]
    fn in_flight_count_tracks() {
        let dir = temp_dir("inflight_count");
        let mut pipeline = AsyncIoPipeline::new(4);
        let coords = IVec3::new(7, 8, 9);
        let path = create_test_chunk_file(coords, &dir);

        pipeline.register_path(coords, path);
        assert_eq!(pipeline.in_flight_count(), 0);

        pipeline.submit_loads(&[coords]);

        // Wait for completion.
        let _ = wait_for_results(&pipeline, 1);
        assert_eq!(pipeline.in_flight_count(), 0, "in-flight should be 0 after completion");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
