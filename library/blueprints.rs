//! Blueprint definitions for pre-configured entity archetypes.

use rkf_runtime::behavior::Blueprint;
use std::collections::HashMap;

/// Create a Guard blueprint: Health + Patrol components.
pub fn guard_blueprint() -> Blueprint {
    let mut components = HashMap::new();
    components.insert(
        "Health".to_owned(),
        "(current: 100.0, max: 100.0)".to_owned(),
    );
    components.insert(
        "Patrol".to_owned(),
        "(speed: 5.0, current_index: 0)".to_owned(),
    );
    Blueprint {
        name: "Guard".to_owned(),
        components,
    }
}

/// Create a Collectible blueprint: Collectible component.
pub fn collectible_blueprint() -> Blueprint {
    let mut components = HashMap::new();
    components.insert(
        "Collectible".to_owned(),
        "(value: 10, spin_speed: 1.0)".to_owned(),
    );
    Blueprint {
        name: "Collectible".to_owned(),
        components,
    }
}
