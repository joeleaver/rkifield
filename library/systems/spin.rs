//! Spin system — rotates entities with Spin around the Y axis.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::Spin;

/// Spin system: rotates entities with Spin around the Y axis.
pub fn spin_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let updates: Vec<_> = ctx
        .query::<(&Spin, &Transform)>()
        .iter()
        .map(|(entity, (spin, t))| {
            let new_rot = t.rotation * glam::Quat::from_rotation_y(spin.speed * dt);
            (entity, Transform { position: t.position, rotation: new_rot, scale: t.scale })
        })
        .collect();
    for (entity, new_t) in updates {
        ctx.insert(entity, new_t);
    }
}
