//! Float bob system — makes entities with FloatBob oscillate up and down.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::FloatBob;

/// Float bob system: makes entities with FloatBob oscillate up and down.
pub fn float_bob_system(ctx: &mut SystemContext) {
    let total = ctx.total_time() as f32;

    let bobs: Vec<_> = ctx
        .query::<&FloatBob>()
        .iter()
        .map(|(entity, bob)| (entity, bob.amplitude, bob.frequency, bob.phase, bob.base_y))
        .collect();

    for (entity, amplitude, frequency, phase, base_y) in bobs {
        let Some((pos, rot, scale)) = ctx.get_transform(entity) else { continue };
        let current_y = pos.to_vec3().y;
        let base = base_y.unwrap_or(current_y);
        let y = base + amplitude * (total * frequency * std::f32::consts::TAU + phase).sin();
        let mut new_pos = pos.to_vec3();
        new_pos.y = y;
        ctx.set_transform(entity, new_pos.into(), rot, scale);

        // Initialize base_y on first tick.
        if base_y.is_none() {
            ctx.insert(entity, FloatBob { amplitude, frequency, phase, base_y: Some(base) });
        }
    }
}
