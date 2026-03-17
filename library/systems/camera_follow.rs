//! Camera follow system — smoothly lerps camera toward target entity.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::CameraFollow;

/// Smoothly lerps camera position toward a target entity's position plus offset.
pub fn camera_follow_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();

    let followers: Vec<_> = ctx
        .query::<&CameraFollow>()
        .iter()
        .filter_map(|(entity, follow)| {
            let cam_pos = ctx.engine().position(entity)?.to_vec3();
            Some((entity, follow.target, follow.offset, follow.smoothing, cam_pos))
        })
        .collect();

    for (entity, target, offset, smoothing, cam_pos) in followers {
        let Some(target_pos) = ctx.engine().position(target) else { continue };
        let target_local = target_pos.to_vec3();

        let desired = target_local + offset;
        let alpha = 1.0 - (-dt / smoothing.max(0.001)).exp();
        let new_pos = cam_pos + (desired - cam_pos) * alpha;

        ctx.set_position(entity, new_pos.into());
    }
}
