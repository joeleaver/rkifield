//! Spin system — rotates entities with Spin around the Y axis.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Spin;

/// Spin system: rotates entities with Spin around the Y axis.
pub fn spin_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let spins: Vec<_> = ctx
        .query::<&Spin>()
        .iter()
        .map(|(entity, spin)| (entity, spin.speed))
        .collect();

    for (entity, speed) in spins {
        let Some((pos, rot, scale)) = ctx.get_transform(entity) else { continue };
        let new_rot = rot * glam::Quat::from_rotation_y(speed * dt);
        ctx.set_transform(entity, pos, new_rot, scale);
    }
}
