//! Collectible system — counts active collectibles and stores the count.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::Collectible;

/// Spin animation for collectibles (increments a counter in the store).
pub fn collectible_system(ctx: &mut SystemContext) {
    let _dt = ctx.delta_time();
    let count = ctx.query::<(&Collectible,)>().iter().count();
    if count > 0 {
        ctx.store().set("collectible_count", count as i64);
    }
}
