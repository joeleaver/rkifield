    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // --- ray_axis_closest_point tests ---

    #[test]
    fn test_ray_axis_perpendicular() {
        // Ray along +Z, axis along +X at origin
        let t = ray_axis_closest_point(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, Vec3::ZERO, Vec3::X);
        // Closest point on X axis to a ray along Z through origin is at t=0
        assert!(approx_eq(t, 0.0), "t = {t}");
    }

    #[test]
    fn test_ray_axis_offset() {
        // Ray along +Z passing through (3, 0, z), axis along +X at origin
        let t = ray_axis_closest_point(
            Vec3::new(3.0, 0.0, -5.0),
            Vec3::Z,
            Vec3::ZERO,
            Vec3::X,
        );
        // Closest point on X-axis is x=3
        assert!(approx_eq(t, 3.0), "t = {t}");
    }

    #[test]
    fn test_ray_axis_parallel() {
        // Ray parallel to axis — degenerate, should return 0
        let t = ray_axis_closest_point(Vec3::new(0.0, 1.0, 0.0), Vec3::X, Vec3::ZERO, Vec3::X);
        assert!(approx_eq(t, 0.0), "parallel case should return 0: t = {t}");
    }

    #[test]
    fn test_ray_axis_negative_t() {
        // Ray along +Z passing through (-2, 0, z), axis along +X at origin
        let t = ray_axis_closest_point(
            Vec3::new(-2.0, 0.0, -5.0),
            Vec3::Z,
            Vec3::ZERO,
            Vec3::X,
        );
        assert!(approx_eq(t, -2.0), "t = {t}");
    }

    // --- project_to_plane tests ---

    #[test]
    fn test_plane_intersection_basic() {
        // Ray from (0,5,0) downward onto XZ plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        assert!(vec3_approx_eq(hit.unwrap(), Vec3::ZERO), "hit = {:?}", hit);
    }

    #[test]
    fn test_plane_intersection_offset() {
        // Ray from (3, 10, 7) downward onto XZ plane at y=0
        let hit = project_to_plane(Vec3::new(3.0, 10.0, 7.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        let p = hit.unwrap();
        assert!(approx_eq(p.x, 3.0) && approx_eq(p.y, 0.0) && approx_eq(p.z, 7.0));
    }

    #[test]
    fn test_plane_parallel_returns_none() {
        // Ray parallel to the plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), Vec3::X, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_none());
    }

    #[test]
    fn test_plane_behind_ray_returns_none() {
        // Ray pointing away from the plane
        let hit = project_to_plane(Vec3::new(0.0, 5.0, 0.0), Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_none());
    }

    // --- pick_gizmo_axis tests ---

    #[test]
    fn test_pick_x_axis() {
        // Ray from front, aiming right along X axis handle
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray grazing the X axis at y=0, z offset close
        let axis = pick_gizmo_axis(
            Vec3::new(0.5, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::X);
    }

    #[test]
    fn test_pick_y_axis() {
        let center = Vec3::ZERO;
        let size = 1.0;
        let axis = pick_gizmo_axis(
            Vec3::new(0.0, 0.5, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::Y);
    }

    #[test]
    fn test_pick_z_axis() {
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray from the side, aimed to pass near the Z axis handle
        let axis = pick_gizmo_axis(
            Vec3::new(5.0, 0.0, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::Z);
    }

    #[test]
    fn test_pick_miss() {
        let center = Vec3::ZERO;
        let size = 1.0;
        // Ray far from any axis
        let axis = pick_gizmo_axis(
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(0.0, 0.0, -1.0),
            center,
            size,
        );
        assert_eq!(axis, GizmoAxis::None);
    }

    // --- compute_translate_delta tests ---

    #[test]
    fn test_translate_delta_x_axis() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::X;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Ray from (3, 0, 5) looking along -Z, should project to x=3 on X axis
        let delta = compute_translate_delta(&state, Vec3::new(3.0, 0.0, 5.0), -Vec3::Z);
        // Delta should be purely along X
        assert!(approx_eq(delta.y, 0.0), "y should be 0: {:?}", delta);
        assert!(approx_eq(delta.z, 0.0), "z should be 0: {:?}", delta);
        assert!(approx_eq(delta.x, 3.0), "x should be ~3: {:?}", delta);
    }

    #[test]
    fn test_translate_delta_none_axis() {
        let state = GizmoState::new();
        let delta = compute_translate_delta(&state, Vec3::ZERO, Vec3::Z);
        assert!(vec3_approx_eq(delta, Vec3::ZERO));
    }

    #[test]
    fn test_translate_delta_plane_xz() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::XZ;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        // Drag started at (1, 0, 1) on the XZ plane
        state.drag_start = Vec3::new(1.0, 0.0, 1.0);

        // Ray from above hitting (3, 0, 2) on XZ plane
        let delta = compute_translate_delta(&state, Vec3::new(3.0, 5.0, 2.0), -Vec3::Y);
        // Delta should be (3-1, 0, 2-1) = (2, 0, 1)
        assert!(approx_eq(delta.x, 2.0), "x: {:?}", delta);
        assert!(approx_eq(delta.y, 0.0), "y: {:?}", delta);
        assert!(approx_eq(delta.z, 1.0), "z: {:?}", delta);
    }

    // --- compute_rotate_delta tests ---

    #[test]
    fn test_rotate_delta_y_axis_90_degrees() {
        let mut state = GizmoState::new();
        state.mode = GizmoMode::Rotate;
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        let center = Vec3::ZERO;
        // Start at +X on the XZ plane
        state.drag_start = Vec3::new(1.0, 0.0, 0.0);

        // Current ray hits +Z on the XZ plane (90 degrees CCW around Y)
        let rot = compute_rotate_delta(
            &state,
            Vec3::new(0.0, 5.0, 1.0),
            -Vec3::Y,
            center,
        );

        // Should be ~90 degrees around Y
        let (axis, angle) = rot.to_axis_angle();
        assert!(
            approx_eq(angle.abs(), PI / 2.0),
            "angle should be ~90 deg: {} (axis: {:?})",
            angle.to_degrees(),
            axis
        );
    }

    #[test]
    fn test_rotate_delta_no_axis() {
        let state = GizmoState::new();
        let rot = compute_rotate_delta(&state, Vec3::ZERO, Vec3::Z, Vec3::ZERO);
        // Should be identity when no axis selected
        assert!(approx_eq(rot.w, 1.0), "should be identity: {:?}", rot);
    }

    #[test]
    fn test_rotate_delta_zero_movement() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        state.drag_start = Vec3::new(1.0, 0.0, 0.0);

        // Ray hits the same point as drag_start
        let rot = compute_rotate_delta(
            &state,
            Vec3::new(1.0, 5.0, 0.0),
            -Vec3::Y,
            Vec3::ZERO,
        );

        let (_, angle) = rot.to_axis_angle();
        assert!(
            approx_eq(angle, 0.0),
            "no movement should give ~0 angle: {}",
            angle
        );
    }

    // --- compute_scale_delta tests ---

    #[test]
    fn test_scale_delta_no_movement() {
        let mut state = GizmoState::new();
        state.mode = GizmoMode::Scale;
        state.active_axis = GizmoAxis::X;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Ray at origin along X — t=0, start_offset=0, so delta=0 → scale=(1,1,1)
        let scale = compute_scale_delta(&state, Vec3::new(0.0, 0.0, 5.0), -Vec3::Z);
        // X-axis drag with no movement: X should be ~1.0, Y/Z stay 1.0
        assert!(
            approx_eq(scale.x, 1.0),
            "no movement should give x scale ~1.0: {:?}",
            scale
        );
        assert!(approx_eq(scale.y, 1.0) && approx_eq(scale.z, 1.0));
    }

    #[test]
    fn test_scale_delta_none_axis() {
        let state = GizmoState::new();
        let scale = compute_scale_delta(&state, Vec3::ZERO, Vec3::Z);
        assert!(vec3_approx_eq(scale, Vec3::ONE));
    }

    #[test]
    fn test_scale_delta_clamped() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::Y;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Extreme negative scale should be clamped to 0.01 on Y axis
        let scale = compute_scale_delta(&state, Vec3::new(0.0, -100000.0, 5.0), -Vec3::Z);
        assert!(scale.y >= 0.01, "scale.y should be clamped: {:?}", scale);
        assert!(approx_eq(scale.x, 1.0), "x should be 1.0: {:?}", scale);

        // Extreme positive scale should be clamped to 100.0 on Y axis
        let scale = compute_scale_delta(&state, Vec3::new(0.0, 100000.0, 5.0), -Vec3::Z);
        assert!(scale.y <= 100.0, "scale.y should be clamped: {:?}", scale);
    }

    #[test]
    fn test_scale_delta_per_axis() {
        let mut state = GizmoState::new();
        state.active_axis = GizmoAxis::X;
        state.dragging = true;
        state.initial_position = Vec3::ZERO;
        state.drag_start = Vec3::ZERO;

        // Drag along +X: only X component should change
        let scale = compute_scale_delta(&state, Vec3::new(5.0, 0.0, 5.0), -Vec3::Z);
        assert!(scale.x != 1.0, "x should have changed: {:?}", scale);
        assert!(approx_eq(scale.y, 1.0), "y should be 1.0: {:?}", scale);
        assert!(approx_eq(scale.z, 1.0), "z should be 1.0: {:?}", scale);
    }

    // --- GizmoState tests ---

    #[test]
    fn test_gizmo_state_default() {
        let state = GizmoState::new();
        assert_eq!(state.mode, GizmoMode::Translate);
        assert_eq!(state.active_axis, GizmoAxis::None);
        assert!(!state.dragging);
    }

    #[test]
    fn test_gizmo_begin_end_drag() {
        let mut state = GizmoState::new();
        state.begin_drag(
            GizmoAxis::X,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            Vec3::Z,
        );

        assert!(state.dragging);
        assert_eq!(state.active_axis, GizmoAxis::X);
        assert_eq!(state.drag_start, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(state.initial_scale, Vec3::ONE);

        state.end_drag();
        assert!(!state.dragging);
        assert_eq!(state.active_axis, GizmoAxis::None);
    }

    // --- GizmoAxis tests ---

    #[test]
    fn test_axis_direction() {
        assert_eq!(GizmoAxis::X.direction(), Vec3::X);
        assert_eq!(GizmoAxis::Y.direction(), Vec3::Y);
        assert_eq!(GizmoAxis::Z.direction(), Vec3::Z);
        assert_eq!(GizmoAxis::XY.direction(), Vec3::ZERO);
        assert_eq!(GizmoAxis::None.direction(), Vec3::ZERO);
    }

    #[test]
    fn test_axis_plane_normal() {
        assert_eq!(GizmoAxis::XY.plane_normal(), Vec3::Z);
        assert_eq!(GizmoAxis::XZ.plane_normal(), Vec3::Y);
        assert_eq!(GizmoAxis::YZ.plane_normal(), Vec3::X);
    }

    // --- Degenerate cases ---

    #[test]
    fn test_ray_axis_zero_direction() {
        // Zero-length ray direction
        let t = ray_axis_closest_point(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::X);
        // Should handle gracefully (denom = 0)
        assert!(t.is_finite(), "should return finite value");
    }

    #[test]
    fn test_plane_intersection_at_origin() {
        let hit = project_to_plane(Vec3::new(0.0, 1.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y);
        assert!(hit.is_some());
        assert!(vec3_approx_eq(hit.unwrap(), Vec3::ZERO));
    }

    #[test]
    fn test_gizmo_mode_eq() {
        assert_eq!(GizmoMode::Translate, GizmoMode::Translate);
        assert_ne!(GizmoMode::Translate, GizmoMode::Rotate);
        assert_ne!(GizmoMode::Rotate, GizmoMode::Scale);
    }

    // --- GizmoResult / apply_to_v2_object tests ---

    #[test]
    fn test_apply_to_v2_object_sets_fields() {
        use rkf_core::{aabb::Aabb, scene::SceneObject, scene_node::SceneNode};

        let mut obj = SceneObject {
            id: 1,
            name: "test".into(),
            parent_id: None,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };

        let result = GizmoResult {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: Vec3::splat(2.5),
        };

        apply_to_v2_object(&result, &mut obj);

        assert!(vec3_approx_eq(obj.position, Vec3::new(1.0, 2.0, 3.0)));
        assert!((obj.scale - Vec3::splat(2.5)).length() < EPS);
        // Rotation should match
        let expected = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        assert!((obj.rotation - expected).length() < EPS);
    }

    #[test]
    fn test_apply_to_v2_object_sets_position_directly() {
        use rkf_core::{aabb::Aabb, scene::SceneObject, scene_node::SceneNode};

        let mut obj = SceneObject {
            id: 2,
            name: "far".into(),
            parent_id: None,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            root_node: SceneNode::new("root"),
            aabb: Aabb::new(Vec3::ZERO, Vec3::ZERO),
        };

        let result = GizmoResult {
            position: Vec3::new(80.5, 0.5, -39.5),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };

        apply_to_v2_object(&result, &mut obj);

        assert!(vec3_approx_eq(obj.position, Vec3::new(80.5, 0.5, -39.5)));
    }
