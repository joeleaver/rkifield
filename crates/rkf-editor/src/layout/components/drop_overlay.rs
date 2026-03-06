//! Drop edge overlay — visual indicators for auto-split during tab drag.

use rinch::prelude::*;

/// Overlay that shows drop zones at the edges of a zone during tab drag.
/// The zone is divided into 5 regions: top, bottom, left, right, and center.
/// Edge drops trigger `split_zone()`, center drops add the tab to the zone.
#[component]
pub fn DropEdgeOverlay() -> NodeHandle {
    // Placeholder — will be implemented in Phase E (Tab Drag-and-Drop).
    rsx! { div {} }
}
