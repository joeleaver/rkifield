//! Tests for object_streaming.

use super::*;
use std::collections::HashMap;

fn default_system() -> ObjectStreamingSystem {
    ObjectStreamingSystem::new(StreamingConfig::default())
}

fn make_aabbs(entries: &[(u32, Vec3, f32)]) -> HashMap<u32, (Vec3, f32)> {
    entries.iter().map(|&(id, c, r)| (id, (c, r))).collect()
}

// -----------------------------------------------------------------------
// 1. new_system_empty
// -----------------------------------------------------------------------

#[test]
fn new_system_empty() {
    let sys = default_system();
    assert_eq!(sys.object_count(), 0);
    assert_eq!(sys.loading_count(), 0);
    assert_eq!(sys.loaded_count(), 0);
}

// -----------------------------------------------------------------------
// 2. register_and_query
// -----------------------------------------------------------------------

#[test]
fn register_and_query() {
    let mut sys = default_system();
    sys.register_object(42, "rock.rkf".into(), 3, 1.0);

    assert_eq!(sys.object_count(), 1);
    assert_eq!(sys.get_state(42), Some(&ObjectStreamState::Unloaded));
    assert_eq!(sys.get_state(99), None);
}

/// Registering the same ID twice keeps the original entry.
#[test]
fn register_idempotent() {
    let mut sys = default_system();
    sys.register_object(1, "a.rkf".into(), 2, 1.0);
    sys.notify_load_complete(1, 0); // put into Loaded
    // Re-register should be a no-op.
    sys.register_object(1, "b.rkf".into(), 5, 2.0);
    // State must not be reset.
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));
}

// -----------------------------------------------------------------------
// 3. unregister_removes
// -----------------------------------------------------------------------

#[test]
fn unregister_removes() {
    let mut sys = default_system();
    sys.register_object(7, "tree.rkf".into(), 2, 1.0);
    assert_eq!(sys.object_count(), 1);
    sys.unregister_object(7);
    assert_eq!(sys.object_count(), 0);
    assert_eq!(sys.get_state(7), None);
}

/// Unregistering an ID that was never registered is a no-op.
#[test]
fn unregister_unknown_is_noop() {
    let mut sys = default_system();
    sys.unregister_object(999);
    assert_eq!(sys.object_count(), 0);
}

// -----------------------------------------------------------------------
// 4. update_priorities_computes_coverage
// -----------------------------------------------------------------------

#[test]
fn update_priorities_computes_coverage() {
    let mut sys = default_system();
    sys.register_object(1, "a.rkf".into(), 2, 1.0);

    // Object at distance 10, radius 1, viewport_height 1080, fov_y 60deg.
    // Expected: (1*2*1080) / (2*10*tan(30deg)) ~ 2160 / 11.547 ~ 187 pixels.
    let center = Vec3::new(10.0, 0.0, 0.0);
    let camera = Vec3::ZERO;
    let aabbs = make_aabbs(&[(1, center, 1.0)]);
    sys.update_priorities(camera, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let obj = sys.objects.get(&1).unwrap();
    assert!(
        obj.screen_coverage > 100.0 && obj.screen_coverage < 300.0,
        "coverage {} out of expected range 100-300 pixels",
        obj.screen_coverage
    );
    assert!((obj.priority - obj.screen_coverage).abs() < 1e-3);
}

#[test]
fn update_priorities_zero_coverage_for_absent_object() {
    let mut sys = default_system();
    sys.register_object(5, "b.rkf".into(), 1, 1.0);
    // No AABB provided for object 5.
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &HashMap::new());

    let obj = sys.objects.get(&5).unwrap();
    assert_eq!(obj.screen_coverage, 0.0);
    assert_eq!(obj.priority, 0.0);
}

// -----------------------------------------------------------------------
// 5. collect_load_requests_prioritized
// -----------------------------------------------------------------------

#[test]
fn collect_load_requests_prioritized() {
    let config = StreamingConfig {
        max_concurrent_loads: 10,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);

    // Register three objects at different distances.
    for id in 1_u32..=3 {
        sys.register_object(id, format!("obj{id}.rkf"), 2, 1.0);
    }

    // Object 1 is closest (highest coverage), object 3 is farthest.
    let aabbs = make_aabbs(&[
        (1, Vec3::new(2.0, 0.0, 0.0), 1.0),
        (2, Vec3::new(10.0, 0.0, 0.0), 1.0),
        (3, Vec3::new(50.0, 0.0, 0.0), 1.0),
    ]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let requests = sys.collect_load_requests();

    // All three should be requested (all above min coverage).
    assert_eq!(requests.len(), 3, "expected 3 requests, got {}", requests.len());

    // Must be in descending priority order (closest first).
    for window in requests.windows(2) {
        assert!(
            window[0].priority >= window[1].priority,
            "requests not sorted by priority: {} < {}",
            window[0].priority,
            window[1].priority
        );
    }
}

// -----------------------------------------------------------------------
// 6. collect_load_requests_respects_max_concurrent
// -----------------------------------------------------------------------

#[test]
fn collect_load_requests_respects_max_concurrent() {
    let config = StreamingConfig {
        max_concurrent_loads: 2,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);

    for id in 1_u32..=5 {
        sys.register_object(id, format!("obj{id}.rkf"), 2, 1.0);
    }

    let aabbs: HashMap<u32, (Vec3, f32)> = (1_u32..=5)
        .map(|id| (id, (Vec3::new(5.0, 0.0, 0.0), 1.0)))
        .collect();
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let first = sys.collect_load_requests();
    assert_eq!(first.len(), 2, "first batch: expected 2, got {}", first.len());

    // Second call -- 2 are already Loading, 0 slots available.
    let second = sys.collect_load_requests();
    assert_eq!(second.len(), 0, "second batch: expected 0, got {}", second.len());
}

// -----------------------------------------------------------------------
// 7. load_complete_transitions_state
// -----------------------------------------------------------------------

#[test]
fn load_complete_transitions_state() {
    let mut sys = default_system();
    sys.register_object(1, "a.rkf".into(), 3, 1.0);

    // Force above min coverage.
    let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let requests = sys.collect_load_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].lod_level, 0, "first load should be coarsest LOD");
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loading { lod_level: 0 }));

    sys.notify_load_complete(1, 0);
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));
    assert_eq!(sys.loaded_count(), 1);
}

// -----------------------------------------------------------------------
// 8. upgrade_triggers_on_high_coverage
// -----------------------------------------------------------------------

#[test]
fn upgrade_triggers_on_high_coverage() {
    let config = StreamingConfig {
        lod_upgrade_threshold: 64.0,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);
    sys.register_object(1, "a.rkf".into(), 3, 1.0);

    // Bring object to Loaded(0).
    sys.notify_load_complete(1, 0);
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 0 }));

    // High coverage to trigger upgrade.
    let aabbs = make_aabbs(&[(1, Vec3::new(0.5, 0.0, 0.0), 1.0)]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let requests = sys.collect_load_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].lod_level, 1, "should request LOD 1 upgrade");

    assert!(
        matches!(sys.get_state(1), Some(ObjectStreamState::Upgrading { current_lod: 0, target_lod: 1 })),
        "expected Upgrading(0->1), got {:?}",
        sys.get_state(1)
    );

    // Complete upgrade.
    sys.notify_load_complete(1, 1);
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Loaded { lod_level: 1 }));
}

// -----------------------------------------------------------------------
// 9. eviction_candidates_respect_grace
// -----------------------------------------------------------------------

#[test]
fn eviction_candidates_respect_grace() {
    let config = StreamingConfig {
        eviction_grace_frames: 10,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);
    sys.register_object(1, "a.rkf".into(), 1, 1.0);

    // Simulate: object was visible at frame 0.
    {
        let obj = sys.objects.get_mut(&1).unwrap();
        obj.last_visible_frame = 0;
        obj.state = ObjectStreamState::Loaded { lod_level: 0 };
    }

    // At frame 9 (9 frames invisible) -- still within grace.
    let candidates = sys.collect_eviction_candidates(9);
    assert!(candidates.is_empty(), "should not evict within grace period");

    // At frame 10 -- exactly at grace boundary.
    let candidates = sys.collect_eviction_candidates(10);
    assert_eq!(candidates, vec![1], "should evict at grace boundary");

    // At frame 100 -- well past grace.
    let candidates = sys.collect_eviction_candidates(100);
    assert_eq!(candidates, vec![1]);
}

// -----------------------------------------------------------------------
// 10. no_eviction_when_visible
// -----------------------------------------------------------------------

#[test]
fn no_eviction_when_visible() {
    let config = StreamingConfig {
        eviction_grace_frames: 5,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);
    sys.register_object(1, "a.rkf".into(), 1, 1.0);

    // Bring to Loaded.
    {
        let obj = sys.objects.get_mut(&1).unwrap();
        obj.state = ObjectStreamState::Loaded { lod_level: 0 };
    }

    // Update visibility to frame 50.
    let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 50, &aabbs);

    // Check eviction at frame 52 -- only 2 frames since last visible.
    let candidates = sys.collect_eviction_candidates(52);
    assert!(candidates.is_empty(), "visible object should not be evicted");
}

// -----------------------------------------------------------------------
// 11. load_error_handling
// -----------------------------------------------------------------------

#[test]
fn load_error_handling() {
    let mut sys = default_system();
    sys.register_object(1, "missing.rkf".into(), 2, 1.0);

    // Trigger load.
    let aabbs = make_aabbs(&[(1, Vec3::new(1.0, 0.0, 0.0), 1.0)]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);
    sys.collect_load_requests();

    sys.notify_load_error(1, "file not found".into());
    assert!(
        matches!(sys.get_state(1), Some(ObjectStreamState::Error(msg)) if msg == "file not found"),
        "expected Error state, got {:?}",
        sys.get_state(1)
    );

    // Error objects must not be re-queued by collect_load_requests.
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 1, &aabbs);
    let requests = sys.collect_load_requests();
    assert!(requests.is_empty(), "errored objects must not be re-requested");
}

/// notify_load_error for an unknown ID is silently ignored.
#[test]
fn load_error_unknown_id_ignored() {
    let mut sys = default_system();
    sys.notify_load_error(999, "boom".into()); // must not panic
    assert_eq!(sys.object_count(), 0);
}

// -----------------------------------------------------------------------
// 12. loading_count_tracking
// -----------------------------------------------------------------------

#[test]
fn loading_count_tracking() {
    let config = StreamingConfig {
        max_concurrent_loads: 10,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);

    for id in 1_u32..=3 {
        sys.register_object(id, format!("o{id}.rkf"), 2, 1.0);
    }

    let aabbs: HashMap<u32, (Vec3, f32)> = (1_u32..=3)
        .map(|id| (id, (Vec3::new(2.0, 0.0, 0.0), 1.0)))
        .collect();
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    assert_eq!(sys.loading_count(), 0);
    sys.collect_load_requests();
    assert_eq!(sys.loading_count(), 3);

    sys.notify_load_complete(1, 0);
    assert_eq!(sys.loading_count(), 2);
    assert_eq!(sys.loaded_count(), 1);

    sys.notify_load_complete(2, 0);
    sys.notify_load_complete(3, 0);
    assert_eq!(sys.loading_count(), 0);
    assert_eq!(sys.loaded_count(), 3);
}

// -----------------------------------------------------------------------
// Additional edge-case tests
// -----------------------------------------------------------------------

/// Evicting an object that was Loaded transitions it to Unloaded.
#[test]
fn eviction_lifecycle() {
    let config = StreamingConfig {
        eviction_grace_frames: 0,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);
    sys.register_object(1, "a.rkf".into(), 1, 1.0);
    sys.notify_load_complete(1, 0);
    assert_eq!(sys.loaded_count(), 1);

    // last_visible_frame is 0, frame_number is 0 -> grace = 0 -> evict.
    let candidates = sys.collect_eviction_candidates(0);
    assert_eq!(candidates, vec![1]);

    // Simulate caller marking for eviction (state set externally here since
    // the eviction-initiation step is the caller's responsibility).
    sys.objects.get_mut(&1).unwrap().state = ObjectStreamState::Evicting;
    sys.notify_evicted(1);
    assert_eq!(sys.get_state(1), Some(&ObjectStreamState::Unloaded));
}

/// Objects with zero LODs are clamped to lod_count=1.
#[test]
fn lod_count_clamped_to_one() {
    let mut sys = default_system();
    sys.register_object(1, "a.rkf".into(), 0, 1.0);
    assert_eq!(sys.objects[&1].lod_count, 1);
}

/// Importance bias scales priority correctly.
#[test]
fn importance_bias_scales_priority() {
    let config = StreamingConfig {
        max_concurrent_loads: 10,
        ..Default::default()
    };
    let mut sys = ObjectStreamingSystem::new(config);
    // Two objects at the same distance but different importance bias.
    sys.register_object(1, "a.rkf".into(), 2, 2.0); // 2x bias
    sys.register_object(2, "b.rkf".into(), 2, 1.0); // 1x bias

    let aabbs = make_aabbs(&[
        (1, Vec3::new(10.0, 0.0, 0.0), 1.0),
        (2, Vec3::new(10.0, 0.0, 0.0), 1.0),
    ]);
    sys.update_priorities(Vec3::ZERO, 1080.0, 60_f32.to_radians(), 0, &aabbs);

    let p1 = sys.objects[&1].priority;
    let p2 = sys.objects[&2].priority;
    assert!(
        (p1 - 2.0 * p2).abs() < 1e-3,
        "priority with 2x bias should be double: {p1} vs 2x{p2}"
    );
}
