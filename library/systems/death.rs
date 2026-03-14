//! Death system — despawns entities with Health <= 0.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Health;

/// Checks entities with Health <= 0 and queues them for despawn.
pub fn death_system(ctx: &mut SystemContext) {
    let dead: Vec<_> = ctx
        .query::<(&Health,)>()
        .iter()
        .filter(|(_, (h,))| h.current <= 0.0)
        .map(|(entity, _)| entity)
        .collect();

    for entity in dead {
        ctx.despawn(entity);
    }
}
