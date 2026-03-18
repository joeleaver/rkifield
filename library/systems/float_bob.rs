//! Float bob system — makes entities with FloatBob oscillate up and down.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::FloatBob;

/// Float bob system: makes entities with FloatBob oscillate up and down.
///
/// On the first tick, captures the entity's Y position as `base_y`.
/// Subsequent ticks oscillate around that base.
pub fn float_bob_system(ctx: &mut SystemContext) {
    let total = ctx.total_time() as f32;

    let bobs: Vec<_> = ctx
        .query::<&FloatBob>()
        .iter()
        .map(|(entity, bob)| (entity, bob.amplitude, bob.frequency, bob.phase, bob.base_y))
        .collect();

    for (entity, amplitude, frequency, phase, base_y) in bobs {
        let Some((pos, rot, scale)) = ctx.get_transform(entity) else { continue };
        let base = base_y.unwrap_or(pos.to_vec3().y);
        let y = base + amplitude * (total * frequency * std::f32::consts::TAU + phase).sin();
        let mut new_pos = pos.to_vec3();
        new_pos.y = y;
        ctx.set_transform(entity, new_pos.into(), rot, scale);

        if base_y.is_none() {
            ctx.insert(entity, FloatBob { amplitude, frequency, phase, base_y: Some(base) });
        }
    }
}
