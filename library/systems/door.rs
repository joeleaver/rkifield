//! Door system — checks for interact events and toggles door state.

use rkf_runtime::behavior::system_context::SystemContext;
use crate::components::DoorState;

/// Checks for interact events and toggles door state.
pub fn door_system(ctx: &mut SystemContext) {
    let doors: Vec<_> = ctx
        .query::<(&DoorState,)>()
        .iter()
        .map(|(entity, (door,))| (entity, door.open))
        .collect();

    let has_interact = ctx.store_ref().events("interact").next().is_some();
    if !has_interact {
        return;
    }

    for (entity, was_open) in doors {
        ctx.insert(
            entity,
            DoorState {
                open: !was_open,
                key_required: None,
            },
        );
    }
}
