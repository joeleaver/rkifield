//! Integration test: streaming world.
//!
//! Ties together procedural chunk generation, streaming system, async I/O
//! pipeline, brick pool, LRU eviction, and budget monitoring into end-to-end
//! scenarios that exercise the full chunk lifecycle.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use glam::{IVec3, Vec3};
use rkf_core::brick_pool::{BrickPool, Pool};
use rkf_core::world_position::CHUNK_SIZE;
use rkf_core::WorldPosition;
use rkf_runtime::async_io::AsyncIoPipeline;
use rkf_runtime::lru_eviction::{EvictionPolicy, LruTracker};
use rkf_runtime::procgen::{generate_world, ProcgenConfig};
use rkf_runtime::streaming::{ChunkState, StreamingConfig, StreamingSystem};
use rkf_runtime::streaming_budget::{BudgetMonitor, StreamingBudget};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a small test world and return (chunk_coords_with_paths, temp_dir_path).
fn setup_test_world(extent_x: u32, extent_z: u32) -> (Vec<(IVec3, PathBuf)>, PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "rkf_streaming_test_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config = ProcgenConfig {
        tier: 2,          // coarse for speed
        amplitude: 1.0,
        frequency: 0.5,
        base_height: 4.0,
    };
    let chunks = generate_world(&dir, extent_x, extent_z, &config).unwrap();
    (chunks, dir)
}

/// Clean up a temporary directory.
fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Poll `drain_completed` until the expected count is reached or timeout.
fn wait_for_loads(
    pipeline: &AsyncIoPipeline,
    expected: usize,
) -> Vec<rkf_runtime::async_io::ChunkLoadResult> {
    let mut results = Vec::new();
    for _ in 0..50 {
        results.extend(pipeline.drain_completed());
        if results.len() >= expected {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    results
}

// ---------------------------------------------------------------------------
// 1. Generate and register chunks
// ---------------------------------------------------------------------------

#[test]
fn generate_and_register_chunks() {
    let (chunks, dir) = setup_test_world(4, 4);
    assert_eq!(chunks.len(), 16, "4x4 world should produce 16 chunks");

    let mut streaming = StreamingSystem::new(StreamingConfig::default());
    let mut pipeline = AsyncIoPipeline::new(4);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    // All chunks should be Unloaded and registered in the pipeline.
    for (coords, _) in &chunks {
        assert_eq!(
            streaming.chunk_state(coords),
            Some(ChunkState::Unloaded),
            "chunk {:?} should be Unloaded after registration",
            coords
        );
        assert!(
            pipeline.has_path(coords),
            "chunk {:?} should have a registered path",
            coords
        );
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 2. Camera triggers nearby loads
// ---------------------------------------------------------------------------

#[test]
fn camera_triggers_nearby_loads() {
    let (chunks, dir) = setup_test_world(4, 4);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 40.0, // ~5 chunks away
        max_radius: 60.0,
        max_loads_per_frame: 16,
        max_evicts_per_frame: 4,
        eviction_grace_frames: 5,
    });

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
    }

    // Camera at the origin.
    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);

    let load_requests = streaming.drain_load_requests();
    assert!(
        !load_requests.is_empty(),
        "should have load requests for nearby chunks"
    );

    // All requested chunks should be PendingLoad.
    for coords in &load_requests {
        assert_eq!(streaming.chunk_state(coords), Some(ChunkState::PendingLoad));
    }

    // Some distant chunks should remain Unloaded.
    let mut any_unloaded = false;
    for (coords, _) in &chunks {
        if streaming.chunk_state(coords) == Some(ChunkState::Unloaded) {
            any_unloaded = true;
            break;
        }
    }
    // The 4x4 grid spans -2..2 in X and Z. The farthest chunk centre is at
    // roughly chunk (1,0,1) * 8m + 4m offset ~= 12m * sqrt(2) ~= 17m from origin.
    // With load_radius=40 all 16 may qualify. So only assert load_requests is sorted
    // nearest-first.
    if load_requests.len() >= 2 {
        // Verify nearest-first ordering.
        let mut prev_dist = 0.0_f64;
        for coords in &load_requests {
            let centre = WorldPosition::new(
                *coords,
                Vec3::splat(CHUNK_SIZE / 2.0),
            );
            let dist = camera.distance_f64(&centre);
            assert!(
                dist >= prev_dist - 1e-6,
                "load requests should be sorted nearest-first, got {dist} after {prev_dist}"
            );
            prev_dist = dist;
        }
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 3. Async pipeline loads chunks
// ---------------------------------------------------------------------------

#[test]
fn async_pipeline_loads_chunks() {
    let (chunks, dir) = setup_test_world(3, 3);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_loads_per_frame: 16,
        ..StreamingConfig::default()
    });
    let mut pipeline = AsyncIoPipeline::new(16);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);

    let load_requests = streaming.drain_load_requests();
    pipeline.submit_loads(&load_requests);

    let results = wait_for_loads(&pipeline, load_requests.len());
    assert_eq!(
        results.len(),
        load_requests.len(),
        "should have received all load results"
    );

    for r in &results {
        let chunk = r.result.as_ref().expect("chunk load should succeed");
        assert!(
            chunk.brick_count > 0,
            "loaded chunk {:?} should have bricks",
            r.coords
        );
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 4. Loaded chunks integrate into pool
// ---------------------------------------------------------------------------

#[test]
fn loaded_chunks_integrate_into_pool() {
    let (chunks, dir) = setup_test_world(2, 2);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_loads_per_frame: 16,
        ..StreamingConfig::default()
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);

    let load_requests = streaming.drain_load_requests();
    pipeline.submit_loads(&load_requests);

    let results = wait_for_loads(&pipeline, load_requests.len());

    let mut pool: BrickPool = Pool::new(65536);
    assert_eq!(pool.allocated_count(), 0);

    let mut total_slots = 0u32;
    for r in results {
        let mut chunk = r.result.expect("load should succeed");
        let slots = chunk
            .load_into_pool(&mut pool)
            .expect("pool should have capacity");
        assert!(!slots.is_empty(), "chunk should produce pool slots");
        total_slots += slots.len() as u32;
        streaming.mark_loaded(&r.coords, slots);
    }

    assert!(
        pool.allocated_count() > 0,
        "pool should have bricks after loading"
    );
    assert_eq!(pool.allocated_count(), total_slots);

    for (coords, _) in &chunks {
        assert_eq!(
            streaming.chunk_state(coords),
            Some(ChunkState::Loaded),
            "chunk {:?} should be Loaded",
            coords
        );
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 5. LRU tracks loaded chunks
// ---------------------------------------------------------------------------

#[test]
fn lru_tracks_loaded_chunks() {
    let (chunks, dir) = setup_test_world(2, 2);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_loads_per_frame: 16,
        ..StreamingConfig::default()
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);
    let load_requests = streaming.drain_load_requests();
    pipeline.submit_loads(&load_requests);
    let results = wait_for_loads(&pipeline, load_requests.len());

    let mut pool: BrickPool = Pool::new(65536);
    let mut lru = LruTracker::new(EvictionPolicy::default());
    let mut all_slot_count = 0usize;

    for r in results {
        let mut chunk = r.result.expect("load should succeed");
        let slots = chunk.load_into_pool(&mut pool).expect("pool capacity");
        for &slot in &slots {
            lru.track(slot, r.coords, 1);
        }
        all_slot_count += slots.len();
        streaming.mark_loaded(&r.coords, slots);
    }

    assert_eq!(
        lru.tracked_count(),
        all_slot_count,
        "LRU should track all allocated slots"
    );

    // Verify all entries have last_frame == 1.
    for (coords, _) in &chunks {
        let slots = lru.slots_for_chunk(coords);
        for slot in slots {
            let entry = lru.entry(slot).expect("slot should be tracked");
            assert_eq!(entry.last_frame, 1);
            assert_eq!(entry.chunk_coords, *coords);
        }
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 6. Budget monitor tracks usage
// ---------------------------------------------------------------------------

#[test]
fn budget_monitor_tracks_usage() {
    let (chunks, dir) = setup_test_world(2, 2);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_loads_per_frame: 16,
        ..StreamingConfig::default()
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);
    let load_requests = streaming.drain_load_requests();
    pipeline.submit_loads(&load_requests);
    let results = wait_for_loads(&pipeline, load_requests.len());

    let mut pool: BrickPool = Pool::new(65536);
    for r in results {
        let mut chunk = r.result.expect("load ok");
        let slots = chunk.load_into_pool(&mut pool).expect("pool capacity");
        streaming.mark_loaded(&r.coords, slots);
    }

    // Budget with generous limits should allow loading.
    let mut budget = BudgetMonitor::new(StreamingBudget::default());
    budget.update_pool_usage(pool.allocated_count(), 4096);

    assert!(
        budget.pool_utilization() > 0.0,
        "pool utilization should be positive after loading"
    );
    assert!(budget.can_load(), "should be able to load with default budget");

    // With a very small max_pool_mb, can_load should return false.
    let mut tiny_budget = BudgetMonitor::new(StreamingBudget {
        max_pool_mb: 1, // 1 MB — less than even a few bricks
        ..StreamingBudget::default()
    });
    tiny_budget.update_pool_usage(pool.allocated_count(), 4096);
    // If the bricks exceed 1 MB the pool is "full" from the budget's perspective.
    let pool_bytes = pool.allocated_count() as u64 * 4096;
    if pool_bytes >= 1024 * 1024 {
        assert!(
            !tiny_budget.can_load(),
            "should not be able to load when pool exceeds budget"
        );
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 7. Eviction frees pool slots
// ---------------------------------------------------------------------------

#[test]
fn eviction_frees_pool_slots() {
    let (chunks, dir) = setup_test_world(3, 3);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_radius: 30.0,         // tight max radius to trigger eviction
        eviction_grace_frames: 5, // short grace
        max_loads_per_frame: 16,
        max_evicts_per_frame: 16,
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    // Load all nearby chunks.
    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);
    let load_requests = streaming.drain_load_requests();
    pipeline.submit_loads(&load_requests);
    let results = wait_for_loads(&pipeline, load_requests.len());

    let mut pool: BrickPool = Pool::new(65536);
    let mut lru = LruTracker::new(EvictionPolicy::default());
    let mut chunk_slots: HashMap<IVec3, Vec<u32>> = HashMap::new();

    for r in results {
        let mut chunk = r.result.expect("load ok");
        let slots = chunk.load_into_pool(&mut pool).expect("pool capacity");
        for &slot in &slots {
            lru.track(slot, r.coords, 1);
        }
        chunk_slots.insert(r.coords, slots.clone());
        streaming.mark_loaded(&r.coords, slots);
    }

    let loaded_before = streaming.loaded_count();
    let allocated_before = pool.allocated_count();
    assert!(loaded_before > 0, "should have loaded some chunks");

    // Move camera very far away so all loaded chunks exceed max_radius.
    let far_camera = WorldPosition::new(IVec3::new(1000, 0, 0), Vec3::ZERO);

    // Run enough frames to exceed the grace period.
    for _ in 0..20 {
        streaming.update(&far_camera);
    }

    let evict_requests = streaming.drain_evict_requests();
    assert!(
        !evict_requests.is_empty(),
        "should have eviction requests after moving camera far away"
    );

    // Process evictions: deallocate pool slots, untrack from LRU, mark evicted.
    for coords in &evict_requests {
        if let Some(slots) = chunk_slots.remove(coords) {
            for slot in &slots {
                pool.deallocate(*slot);
                lru.untrack(*slot);
            }
        }
        streaming.mark_evicted(coords);
    }

    assert!(
        pool.allocated_count() < allocated_before,
        "pool should have fewer bricks after eviction"
    );
    assert!(
        pool.free_count() > 65536 - allocated_before,
        "free_count should have increased"
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 8. Full streaming loop
// ---------------------------------------------------------------------------

#[test]
fn full_streaming_loop() {
    let (chunks, dir) = setup_test_world(6, 6);
    assert_eq!(chunks.len(), 36, "6x6 world should produce 36 chunks");

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 50.0,
        max_radius: 60.0,
        eviction_grace_frames: 5,
        max_loads_per_frame: 8,
        max_evicts_per_frame: 4,
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let mut pool: BrickPool = Pool::new(65536);
    let mut lru = LruTracker::new(EvictionPolicy {
        high_watermark: 0.9,
        low_watermark: 0.8,
        min_age_frames: 10,
        max_evictions_per_frame: 64,
    });
    let mut budget = BudgetMonitor::new(StreamingBudget::default());
    let mut chunk_slots: HashMap<IVec3, Vec<u32>> = HashMap::new();

    // --- Phase 1: Camera at origin, load nearby chunks ---
    let camera_origin = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);

    for frame in 0..20u64 {
        streaming.update(&camera_origin);

        let load_requests = streaming.drain_load_requests();
        if !load_requests.is_empty() {
            pipeline.submit_loads(&load_requests);
        }

        // Drain completed loads — skip duplicates.
        let completed = pipeline.drain_completed();
        for r in completed {
            if chunk_slots.contains_key(&r.coords) {
                continue;
            }
            if let Ok(mut chunk) = r.result {
                if let Ok(slots) = chunk.load_into_pool(&mut pool) {
                    for &slot in &slots {
                        lru.track(slot, r.coords, frame);
                    }
                    chunk_slots.insert(r.coords, slots.clone());
                    streaming.mark_loaded(&r.coords, slots);
                }
            }
        }

        budget.update_pool_usage(pool.allocated_count(), 4096);

        // Small sleep to let background threads work.
        std::thread::sleep(Duration::from_millis(20));
    }

    // Drain any remaining in-flight loads from Phase 1.
    for _ in 0..50 {
        let completed = pipeline.drain_completed();
        if completed.is_empty() && pipeline.in_flight_count() == 0 {
            break;
        }
        for r in completed {
            if chunk_slots.contains_key(&r.coords) {
                continue;
            }
            if let Ok(mut chunk) = r.result {
                if let Ok(slots) = chunk.load_into_pool(&mut pool) {
                    for &slot in &slots {
                        lru.track(slot, r.coords, 19);
                    }
                    chunk_slots.insert(r.coords, slots.clone());
                    streaming.mark_loaded(&r.coords, slots);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let loaded_after_phase1 = streaming.loaded_count();
    assert!(
        loaded_after_phase1 > 0,
        "should have loaded some chunks after phase 1"
    );

    // --- Phase 2: Move camera to edge, run many frames to trigger eviction ---
    let camera_far = WorldPosition::new(IVec3::new(100, 0, 100), Vec3::ZERO);

    let loaded_before_eviction = streaming.loaded_count();
    for frame in 20..220u64 {
        streaming.update(&camera_far);

        // Handle new load requests (for chunks near new position, if any).
        let load_requests = streaming.drain_load_requests();
        if !load_requests.is_empty() {
            pipeline.submit_loads(&load_requests);
        }

        // Drain completed loads — skip duplicates (same coords already loaded).
        let completed = pipeline.drain_completed();
        for r in completed {
            if chunk_slots.contains_key(&r.coords) {
                // Already loaded — discard duplicate result.
                continue;
            }
            if let Ok(mut chunk) = r.result {
                if let Ok(slots) = chunk.load_into_pool(&mut pool) {
                    for &slot in &slots {
                        lru.track(slot, r.coords, frame);
                    }
                    chunk_slots.insert(r.coords, slots.clone());
                    streaming.mark_loaded(&r.coords, slots);
                }
            }
        }

        // Process eviction requests.
        let evict_requests = streaming.drain_evict_requests();
        for coords in &evict_requests {
            if let Some(slots) = chunk_slots.remove(coords) {
                for slot in &slots {
                    pool.deallocate(*slot);
                    lru.untrack(*slot);
                }
            }
            streaming.mark_evicted(coords);
        }

        budget.update_pool_usage(pool.allocated_count(), 4096);
    }

    // Verify eviction occurred.
    let loaded_after_phase2 = streaming.loaded_count();
    assert!(
        loaded_after_phase2 < loaded_before_eviction,
        "some chunks should have been evicted (loaded: {} -> {})",
        loaded_before_eviction,
        loaded_after_phase2,
    );

    // Verify pool consistency: allocated_count should match tracked slot count.
    // chunk_slots contains only slots that haven't been evicted.
    let expected_allocated: u32 = chunk_slots.values().map(|v| v.len() as u32).sum();
    assert_eq!(
        pool.allocated_count(),
        expected_allocated,
        "pool allocated count should match tracked chunk_slots"
    );

    // Verify no orphaned LRU entries: tracked_count should match allocated pool slots.
    assert_eq!(
        lru.tracked_count() as u32,
        expected_allocated,
        "LRU tracked_count should match pool allocated_count"
    );

    // Pool should not have leaked: free + allocated = capacity.
    assert_eq!(
        pool.free_count() + pool.allocated_count(),
        pool.capacity(),
        "pool free + allocated should equal capacity (no leaks)"
    );

    cleanup(&dir);
}

// ---------------------------------------------------------------------------
// 9. Multiple load-evict cycles
// ---------------------------------------------------------------------------

#[test]
fn multiple_load_evict_cycles() {
    let (chunks, dir) = setup_test_world(4, 4);

    let mut streaming = StreamingSystem::new(StreamingConfig {
        load_radius: 200.0,
        max_radius: 20.0,        // tight: most chunks will be "out of range" immediately
        eviction_grace_frames: 3,
        max_loads_per_frame: 16,
        max_evicts_per_frame: 16,
    });
    let mut pipeline = AsyncIoPipeline::new(8);

    for (coords, path) in &chunks {
        streaming.register_chunk(*coords, path.clone(), false);
        pipeline.register_path(*coords, path.clone());
    }

    let mut pool: BrickPool = Pool::new(65536);
    let mut chunk_slots: HashMap<IVec3, Vec<u32>> = HashMap::new();

    // Cycle 1: load at origin.
    let camera = WorldPosition::new(IVec3::ZERO, Vec3::ZERO);
    streaming.update(&camera);
    let reqs = streaming.drain_load_requests();
    pipeline.submit_loads(&reqs);
    let results = wait_for_loads(&pipeline, reqs.len());

    for r in results {
        let mut chunk = r.result.expect("load ok");
        let slots = chunk.load_into_pool(&mut pool).expect("pool capacity");
        chunk_slots.insert(r.coords, slots.clone());
        streaming.mark_loaded(&r.coords, slots);
    }

    let loaded_cycle1 = streaming.loaded_count();
    assert!(loaded_cycle1 > 0);

    // Move far away and evict.
    let far = WorldPosition::new(IVec3::new(500, 0, 0), Vec3::ZERO);
    for _ in 0..10 {
        streaming.update(&far);
    }
    let evicts = streaming.drain_evict_requests();
    for coords in &evicts {
        if let Some(slots) = chunk_slots.remove(coords) {
            for s in &slots {
                pool.deallocate(*s);
            }
        }
        streaming.mark_evicted(coords);
    }

    let loaded_after_evict = streaming.loaded_count();
    assert!(
        loaded_after_evict < loaded_cycle1,
        "should have evicted some chunks"
    );

    // Cycle 2: move back to origin, re-load.
    streaming.update(&camera);
    let reqs2 = streaming.drain_load_requests();
    assert!(
        !reqs2.is_empty(),
        "evicted chunks should be re-loadable when camera returns"
    );

    pipeline.submit_loads(&reqs2);
    let results2 = wait_for_loads(&pipeline, reqs2.len());
    for r in results2 {
        let mut chunk = r.result.expect("re-load ok");
        let slots = chunk.load_into_pool(&mut pool).expect("pool capacity");
        chunk_slots.insert(r.coords, slots.clone());
        streaming.mark_loaded(&r.coords, slots);
    }

    assert!(
        streaming.loaded_count() >= loaded_cycle1,
        "should have re-loaded chunks after camera returns"
    );

    // Pool consistency check.
    assert_eq!(
        pool.free_count() + pool.allocated_count(),
        pool.capacity(),
        "pool should not leak across load/evict cycles"
    );

    cleanup(&dir);
}
