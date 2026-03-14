//! Patrol system — moves entities with Patrol along their waypoints.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Patrol;

/// Moves entities with `Patrol` along their waypoints.
pub fn patrol_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let updates: Vec<_> = ctx
        .query::<(&Patrol,)>()
        .iter()
        .filter_map(|(entity, (patrol,))| {
            if patrol.waypoints.is_empty() {
                return None;
            }
            let next_index = (patrol.current_index + 1) % patrol.waypoints.len();
            Some((entity, next_index, patrol.speed * dt))
        })
        .collect();

    for (entity, next_index, _step) in updates {
        ctx.insert(
            entity,
            Patrol {
                waypoints: Vec::new(),
                speed: 0.0,
                current_index: next_index,
            },
        );
    }
}
