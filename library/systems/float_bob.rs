//! Float bob system — makes entities with FloatBob oscillate up and down.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::FloatBob;

/// Float bob system: makes entities with FloatBob oscillate up and down.
pub fn float_bob_system(ctx: &mut SystemContext) {
    let total = ctx.total_time() as f32;
    let updates: Vec<_> = ctx
        .query::<(&FloatBob, &Transform)>()
        .iter()
        .map(|(entity, (bob, t))| {
            let current_y = t.position.to_vec3().y;
            let base = bob.base_y.unwrap_or(current_y);
            let y = base + bob.amplitude * (total * bob.frequency * std::f32::consts::TAU + bob.phase).sin();
            let mut pos = t.position.to_vec3();
            pos.y = y;
            (entity, base, Transform { position: pos.into(), rotation: t.rotation, scale: t.scale })
        })
        .collect();
    for (entity, base, new_t) in updates {
        ctx.insert(entity, new_t);
        let needs_init = ctx.get::<FloatBob>(entity).ok().filter(|b| b.base_y.is_none())
            .map(|b| FloatBob { amplitude: b.amplitude, frequency: b.frequency, phase: b.phase, base_y: Some(base) });
        if let Some(updated) = needs_init {
            ctx.insert(entity, updated);
        }
    }
}
