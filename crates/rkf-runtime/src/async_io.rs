//! Async I/O pipeline for loading .rkf v2 object files on background threads.
//!
//! The pipeline uses `std::thread` workers (not async/tokio). Callers submit
//! load requests via [`AsyncIoPipeline::submit`] and drain completed results
//! via [`AsyncIoPipeline::poll_results`]. Workers call the `rkf-core` asset
//! file functions directly.
//!
//! # Example
//!
//! ```no_run
//! use rkf_runtime::async_io::{AsyncIoConfig, AsyncIoPipeline};
//!
//! let pipeline = AsyncIoPipeline::new(AsyncIoConfig::default());
//!
//! let queued = pipeline.submit(1, "/assets/rock.rkf".into(), 0);
//! assert!(queued);
//!
//! // Later, in the frame loop:
//! for result in pipeline.poll_results() {
//!     match result.result {
//!         Ok(lod_data) => { /* upload to GPU */ }
//!         Err(e) => eprintln!("load failed: {e}"),
//!     }
//! }
//!
//! pipeline.shutdown();
//! ```

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use rkf_core::asset_file::{LodData, load_object_header, load_object_lod};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the async I/O pipeline.
#[derive(Debug, Clone)]
pub struct AsyncIoConfig {
    /// Number of worker threads that execute file I/O concurrently.
    ///
    /// Default: 2.
    pub max_workers: usize,

    /// Maximum number of in-flight + queued requests.
    ///
    /// [`AsyncIoPipeline::submit`] returns `false` when this limit is reached.
    /// Default: 16.
    pub max_pending: usize,
}

impl Default for AsyncIoConfig {
    fn default() -> Self {
        Self {
            max_workers: 2,
            max_pending: 16,
        }
    }
}

/// The outcome of one load request.
#[derive(Debug)]
pub struct LoadResult {
    /// The object ID passed to [`AsyncIoPipeline::submit`].
    pub object_id: u32,
    /// The LOD level that was requested.
    pub lod_level: usize,
    /// The loaded data, or an error message if the load failed.
    ///
    /// Errors use `String` rather than [`rkf_core::asset_file::AssetError`] so
    /// the result is easily `Send` across thread boundaries.
    pub result: Result<LodData, String>,
}

// ---------------------------------------------------------------------------
// Internal request type
// ---------------------------------------------------------------------------

/// A load request sent from the main thread to a worker.
struct Request {
    object_id: u32,
    asset_path: String,
    lod_level: usize,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Background I/O pipeline for loading `.rkf` v2 object files.
///
/// Spawn worker threads with [`AsyncIoPipeline::new`], then submit requests
/// with [`submit`][AsyncIoPipeline::submit] and drain results with
/// [`poll_results`][AsyncIoPipeline::poll_results].
pub struct AsyncIoPipeline {
    /// Channel used to send requests to workers.
    ///
    /// Wrapped in `Option` so it can be dropped (closing the channel) during
    /// shutdown before joining worker threads.
    request_tx: Option<Sender<Request>>,

    /// Shared request receiver kept alive so the channel stays open even if
    /// no worker threads have been spawned (e.g. `max_workers: 0` in tests).
    _request_rx: Arc<Mutex<Receiver<Request>>>,

    /// Channel used to receive completed results from workers.
    result_rx: Receiver<LoadResult>,

    /// Worker thread handles, held until [`shutdown`][AsyncIoPipeline::shutdown].
    workers: Vec<JoinHandle<()>>,

    /// Number of requests that have been submitted but whose results have not
    /// yet been drained via [`poll_results`].
    pending_count: Arc<Mutex<usize>>,

    /// Maximum number of pending requests (from config).
    max_pending: usize,
}

impl AsyncIoPipeline {
    /// Create a new pipeline and spawn `config.max_workers` background threads.
    pub fn new(config: AsyncIoConfig) -> Self {
        // Unbounded request channel — backpressure is enforced by `pending_count`
        // alone, so workers are never blocked waiting for the main thread to recv.
        let (request_tx, request_rx) = mpsc::channel::<Request>();
        // Unbounded result channel — workers are never blocked on the consumer.
        let (result_tx, result_rx) = mpsc::channel::<LoadResult>();

        let pending_count = Arc::new(Mutex::new(0usize));

        // Shared receiver for the request channel.
        let shared_rx = Arc::new(Mutex::new(request_rx));

        let mut workers = Vec::with_capacity(config.max_workers);
        for _ in 0..config.max_workers {
            let rx_clone = Arc::clone(&shared_rx);
            let tx_clone = result_tx.clone();
            let pending_clone = Arc::clone(&pending_count);

            let handle = thread::spawn(move || {
                worker_loop(rx_clone, tx_clone, pending_clone);
            });
            workers.push(handle);
        }

        Self {
            request_tx: Some(request_tx),
            _request_rx: shared_rx,
            result_rx,
            workers,
            pending_count,
            max_pending: config.max_pending,
        }
    }

    /// Submit a load request for `lod_level` of the object at `asset_path`.
    ///
    /// Returns `true` if the request was accepted, or `false` if the pending
    /// queue is full (see [`AsyncIoConfig::max_pending`]).
    pub fn submit(&self, object_id: u32, asset_path: String, lod_level: usize) -> bool {
        // Check pending count before sending.
        {
            let mut count = self.pending_count.lock().unwrap();
            if *count >= self.max_pending {
                return false;
            }
            *count += 1;
        }

        let request = Request {
            object_id,
            asset_path,
            lod_level,
        };

        match self.request_tx.as_ref() {
            Some(tx) => {
                if tx.send(request).is_err() {
                    // Channel closed (pipeline is shutting down). Undo the
                    // pending count increment.
                    let mut count = self.pending_count.lock().unwrap();
                    *count = count.saturating_sub(1);
                    false
                } else {
                    true
                }
            }
            None => {
                // Already shut down.
                let mut count = self.pending_count.lock().unwrap();
                *count = count.saturating_sub(1);
                false
            }
        }
    }

    /// Drain all completed results without blocking.
    ///
    /// Returns an empty `Vec` if no results are ready yet. The caller is
    /// responsible for handling each [`LoadResult`].
    pub fn poll_results(&self) -> Vec<LoadResult> {
        let mut results = Vec::new();
        loop {
            match self.result_rx.try_recv() {
                Ok(r) => results.push(r),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    /// Number of requests that have been submitted but not yet drained via
    /// [`poll_results`].
    ///
    /// This includes both requests currently being processed by a worker and
    /// requests waiting in the queue.
    pub fn pending_count(&self) -> usize {
        *self.pending_count.lock().unwrap()
    }

    /// Stop all worker threads and wait for them to finish.
    ///
    /// Drops the request sender (closing the channel), signals workers to exit,
    /// then joins every thread. Any in-flight work completes normally; queued
    /// requests that haven't started will be discarded when workers read the
    /// closed channel.
    pub fn shutdown(mut self) {
        // Drop the sender to close the channel; workers will exit their recv loop.
        drop(self.request_tx.take());

        for handle in self.workers.drain(..) {
            // Ignore thread panics during shutdown.
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Worker function
// ---------------------------------------------------------------------------

/// Main loop executed by each worker thread.
///
/// Reads requests from `rx`, loads the asset, then sends the result to `tx`.
/// Decrements `pending_count` after each request (success or error) so the
/// main thread's count stays accurate.
fn worker_loop(
    rx: Arc<Mutex<mpsc::Receiver<Request>>>,
    tx: mpsc::Sender<LoadResult>,
    pending_count: Arc<Mutex<usize>>,
) {
    loop {
        // Acquire the receiver lock to claim one request.
        let request = {
            let locked = rx.lock().unwrap();
            match locked.recv() {
                Ok(r) => r,
                // Channel closed — pipeline is shutting down.
                Err(_) => break,
            }
        };

        let result = load_lod(&request.asset_path, request.lod_level);

        // Decrement pending count before sending the result so that callers
        // who check pending_count immediately after draining results see 0.
        {
            let mut count = pending_count.lock().unwrap();
            *count = count.saturating_sub(1);
        }

        let load_result = LoadResult {
            object_id: request.object_id,
            lod_level: request.lod_level,
            result,
        };

        // If the pipeline is already shut down the send may fail — that's OK.
        let _ = tx.send(load_result);
    }
}

/// Open a .rkf file at `path`, read the header, then load the specified LOD level.
fn load_lod(path: &str, lod_level: usize) -> Result<LodData, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open '{}': {}", path, e))?;
    let mut reader = std::io::BufReader::new(file);
    let header =
        load_object_header(&mut reader).map_err(|e| format!("load_object_header: {}", e))?;
    load_object_lod(&mut reader, &header, lod_level)
        .map_err(|e| format!("load_object_lod(lod={}): {}", lod_level, e))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // ── Helper: write a minimal valid .rkf v2 file ───────────────────────────

    fn write_minimal_rkf(path: &std::path::Path) {
        use rkf_core::aabb::Aabb;
        use rkf_core::asset_file::{SaveLodLevel, save_object};
        use rkf_core::brick_map::BrickMap;
        use glam::{UVec3, Vec3};

        let dims = UVec3::new(2, 2, 2);
        let brick_map = BrickMap::new(dims);
        let lod = SaveLodLevel {
            voxel_size: 0.08,
            brick_map,
            brick_data: vec![],
        };

        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let mut file = std::fs::File::create(path).expect("create test rkf");
        save_object(&mut file, &aabb, None, &[], &[lod]).expect("save_object");
    }

    // ── 1. new_pipeline_empty ────────────────────────────────────────────────

    #[test]
    fn new_pipeline_empty() {
        let pipeline = AsyncIoPipeline::new(AsyncIoConfig::default());
        assert_eq!(pipeline.pending_count(), 0);
        assert!(pipeline.poll_results().is_empty());
        pipeline.shutdown();
    }

    // ── 2. submit_and_poll_not_found ─────────────────────────────────────────

    #[test]
    fn submit_and_poll_not_found() {
        let pipeline = AsyncIoPipeline::new(AsyncIoConfig::default());

        let queued = pipeline.submit(42, "/nonexistent/path/to/object.rkf".into(), 0);
        assert!(queued, "submit should succeed when queue is not full");

        // Wait until the result arrives (with a timeout to avoid hanging CI).
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let results = pipeline.poll_results();
            if !results.is_empty() {
                let r = &results[0];
                assert_eq!(r.object_id, 42);
                assert_eq!(r.lod_level, 0);
                assert!(r.result.is_err(), "nonexistent path must produce an error");
                break;
            }
            assert!(Instant::now() < deadline, "timed out waiting for load result");
            std::thread::sleep(Duration::from_millis(10));
        }

        pipeline.shutdown();
    }

    // ── 3. submit_respects_max_pending ───────────────────────────────────────

    #[test]
    fn submit_respects_max_pending() {
        let config = AsyncIoConfig {
            max_workers: 0, // zero workers — nothing will consume from the queue
            max_pending: 1,
        };
        let pipeline = AsyncIoPipeline::new(config);

        // First submit fills the single slot.
        let first = pipeline.submit(1, "/no/file.rkf".into(), 0);
        assert!(first, "first submit should succeed");

        // Second submit should be rejected.
        let second = pipeline.submit(2, "/no/file.rkf".into(), 0);
        assert!(!second, "second submit should be rejected when max_pending=1");

        pipeline.shutdown();
    }

    // ── 4. pending_count_tracks ──────────────────────────────────────────────

    #[test]
    fn pending_count_tracks() {
        let config = AsyncIoConfig {
            max_workers: 0, // no workers — pending count stays elevated
            max_pending: 4,
        };
        let pipeline = AsyncIoPipeline::new(config);

        assert_eq!(pipeline.pending_count(), 0);

        pipeline.submit(1, "/a.rkf".into(), 0);
        assert_eq!(pipeline.pending_count(), 1);

        pipeline.submit(2, "/b.rkf".into(), 0);
        assert_eq!(pipeline.pending_count(), 2);

        pipeline.shutdown();
    }

    // ── 5. shutdown_cleans_up ────────────────────────────────────────────────

    #[test]
    fn shutdown_cleans_up() {
        let pipeline = AsyncIoPipeline::new(AsyncIoConfig::default());
        // Submit a request that will fail quickly (nonexistent path).
        pipeline.submit(99, "/does/not/exist.rkf".into(), 0);
        // Shutdown must not hang.
        pipeline.shutdown();
    }

    // ── 6. poll_results_nonblocking ──────────────────────────────────────────

    #[test]
    fn poll_results_nonblocking() {
        let pipeline = AsyncIoPipeline::new(AsyncIoConfig::default());

        // Don't submit anything; poll should return immediately with empty vec.
        let start = Instant::now();
        let results = pipeline.poll_results();
        let elapsed = start.elapsed();

        assert!(results.is_empty());
        // Should complete in well under a millisecond.
        assert!(
            elapsed < Duration::from_millis(100),
            "poll_results blocked for {:?}",
            elapsed
        );

        pipeline.shutdown();
    }

    // ── 7. multiple_submits ──────────────────────────────────────────────────

    #[test]
    fn multiple_submits() {
        let tmp = std::env::temp_dir().join("rkf_async_io_test");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let path1 = tmp.join("obj1.rkf");
        let path2 = tmp.join("obj2.rkf");
        let path3 = tmp.join("obj3.rkf");

        write_minimal_rkf(&path1);
        write_minimal_rkf(&path2);
        write_minimal_rkf(&path3);

        let config = AsyncIoConfig {
            max_workers: 2,
            max_pending: 8,
        };
        let pipeline = AsyncIoPipeline::new(config);

        pipeline.submit(1, path1.to_string_lossy().into_owned(), 0);
        pipeline.submit(2, path2.to_string_lossy().into_owned(), 0);
        pipeline.submit(3, path3.to_string_lossy().into_owned(), 0);

        // Collect results with a timeout.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut collected: Vec<LoadResult> = Vec::new();

        while collected.len() < 3 {
            assert!(Instant::now() < deadline, "timed out waiting for all results");
            let mut batch = pipeline.poll_results();
            collected.append(&mut batch);
            if collected.len() < 3 {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        // All results should succeed.
        for r in &collected {
            assert!(
                r.result.is_ok(),
                "object {} failed: {:?}",
                r.object_id,
                r.result
            );
        }

        // All three object IDs are represented.
        let mut ids: Vec<u32> = collected.iter().map(|r| r.object_id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);

        pipeline.shutdown();
    }

    // ── 8. shutdown_with_inflight_requests ───────────────────────────────────
    // Verify that shutdown() does not hang when requests are in-flight.

    #[test]
    fn shutdown_with_inflight_requests() {
        let tmp = std::env::temp_dir().join("rkf_async_io_test_shutdown");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let path = tmp.join("shutdown_test.rkf");
        write_minimal_rkf(&path);

        let config = AsyncIoConfig {
            max_workers: 2,
            max_pending: 8,
        };
        let pipeline = AsyncIoPipeline::new(config);

        // Submit multiple requests that will be in-flight.
        pipeline.submit(100, path.to_string_lossy().into_owned(), 0);
        pipeline.submit(101, path.to_string_lossy().into_owned(), 0);
        pipeline.submit(102, path.to_string_lossy().into_owned(), 0);

        // Shutdown without draining results — workers should complete in-flight
        // work and then exit gracefully when they read the closed channel.
        let start = Instant::now();
        pipeline.shutdown();
        let elapsed = start.elapsed();

        // Shutdown should complete quickly (within 5 seconds), not hang indefinitely.
        assert!(
            elapsed < Duration::from_secs(5),
            "shutdown took {:?}, possible hang detected",
            elapsed
        );
    }

    // ── 9. submit_returns_false_after_shutdown ───────────────────────────────
    // Verify that submit() returns false when the request channel is closed
    // (simulated by zero workers, no channel send).

    #[test]
    fn submit_returns_false_after_shutdown() {
        let config = AsyncIoConfig {
            max_workers: 0,
            max_pending: 1,
        };
        let pipeline = AsyncIoPipeline::new(config);

        // Submit one request to fill the pending slot.
        let first = pipeline.submit(200, "/some/path.rkf".into(), 0);
        assert!(first, "first submit should succeed");

        // Second submit should be rejected because max_pending=1 and slot is full.
        let second = pipeline.submit(201, "/some/path.rkf".into(), 0);
        assert!(
            !second,
            "second submit should be rejected when queue is full, but returned true"
        );

        // Now test the channel-closed path by dropping the request sender
        // (simulating what shutdown does). Create a new pipeline with a sender
        // we can manually drop.
        let config2 = AsyncIoConfig {
            max_workers: 1,
            max_pending: 8,
        };
        let mut pipeline2 = AsyncIoPipeline::new(config2);

        // Drop the request sender to close the channel.
        drop(pipeline2.request_tx.take());

        // Now submit should return false because send will fail.
        let result = pipeline2.submit(202, "/some/path.rkf".into(), 0);
        assert!(
            !result,
            "submit should return false when channel is closed, but returned true"
        );

        pipeline.shutdown();
        pipeline2.shutdown();
    }

    // ── 10. result_ordering_doesnt_matter ────────────────────────────────────
    // Verify that all submitted IDs eventually appear in results, regardless of
    // the order in which they are returned by workers.

    #[test]
    fn result_ordering_doesnt_matter() {
        let tmp = std::env::temp_dir().join("rkf_async_io_test_ordering");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");

        // Create 5 test files with distinct paths.
        let paths: Vec<_> = (0..5)
            .map(|i| {
                let p = tmp.join(format!("obj_{}.rkf", i));
                write_minimal_rkf(&p);
                p
            })
            .collect();

        let config = AsyncIoConfig {
            max_workers: 3,
            max_pending: 16,
        };
        let pipeline = AsyncIoPipeline::new(config);

        // Submit 5 requests in order (object_id = 300-304).
        for (i, path) in paths.iter().enumerate() {
            pipeline.submit(
                300 + i as u32,
                path.to_string_lossy().into_owned(),
                0,
            );
        }

        // Collect all results with a timeout.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut collected: Vec<LoadResult> = Vec::new();

        while collected.len() < 5 {
            assert!(Instant::now() < deadline, "timed out waiting for all results");
            let mut batch = pipeline.poll_results();
            collected.append(&mut batch);
            if collected.len() < 5 {
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        // Verify that all 5 object IDs are present (order irrelevant).
        let mut ids: Vec<u32> = collected.iter().map(|r| r.object_id).collect();
        ids.sort();
        assert_eq!(
            ids, vec![300, 301, 302, 303, 304],
            "not all submitted IDs appeared in results"
        );

        // Verify all results are successful.
        for r in &collected {
            assert!(r.result.is_ok(), "result for object {} failed", r.object_id);
        }

        pipeline.shutdown();
    }

    // ── 11. rapid_submit_poll_cycling ────────────────────────────────────────
    // Verify that rapid submit/poll cycling works correctly without data loss
    // or missed results.

    #[test]
    fn rapid_submit_poll_cycling() {
        let tmp = std::env::temp_dir().join("rkf_async_io_test_rapid");
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        let path = tmp.join("rapid_test.rkf");
        write_minimal_rkf(&path);

        let config = AsyncIoConfig {
            max_workers: 2,
            max_pending: 16,
        };
        let pipeline = AsyncIoPipeline::new(config);

        let path_str = path.to_string_lossy().into_owned();

        // Rapidly submit 10 requests in a tight loop.
        for i in 0..10 {
            let queued = pipeline.submit(400 + i, path_str.clone(), 0);
            assert!(queued, "submit {} should succeed", i);
        }

        // Rapidly poll results in a tight loop with brief sleeps.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut collected: Vec<LoadResult> = Vec::new();

        while collected.len() < 10 {
            assert!(Instant::now() < deadline, "timed out waiting for all results");

            // Poll multiple times without sleeping to stress the channel.
            for _ in 0..5 {
                let mut batch = pipeline.poll_results();
                collected.append(&mut batch);
            }

            if collected.len() < 10 {
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        // Verify all 10 results arrived and succeeded.
        assert_eq!(collected.len(), 10, "did not collect all 10 results");
        for r in &collected {
            assert!(r.result.is_ok(), "result for object {} failed", r.object_id);
        }

        // Verify all object IDs are present.
        let mut ids: Vec<u32> = collected.iter().map(|r| r.object_id).collect();
        ids.sort();
        assert_eq!(ids, (400..410).collect::<Vec<_>>());

        pipeline.shutdown();
    }
}
