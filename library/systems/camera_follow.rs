//! Camera follow system — smoothly lerps camera toward target entity.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::CameraFollow;

/// Smoothly lerps camera position toward a target entity's position plus offset.
pub fn camera_follow_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();

    let followers: Vec<_> = ctx
        .query::<(&CameraFollow, &Transform)>()
        .iter()
        .map(|(entity, (follow, transform))| {
            (entity, follow.target, follow.offset, follow.smoothing, transform.position.local)
        })
        .collect();

    for (entity, target, offset, smoothing, cam_pos) in followers {
        let target_pos = match ctx.get::<Transform>(target) {
            Ok(t) => t.position.local,
            Err(_) => continue,
        };

        let desired = target_pos + offset;
        let alpha = 1.0 - (-dt / smoothing.max(0.001)).exp();
        let new_pos = cam_pos + (desired - cam_pos) * alpha;

        ctx.insert(
            entity,
            Transform {
                position: new_pos.into(),
                ..Transform::default()
            },
        );
    }
}
