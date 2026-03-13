//! Scene file format v3 (.rkscene) — entity-centric with component bags.
//!
//! Each entity is stored as a [`StableId`] UUID with a bag of named components
//! serialized as RON strings. Engine components are handled directly; gameplay
//! components go through the `ComponentEntry` registry (type-erased dylib).
//!
//! # Format
//!
//! ```ron
//! SceneFileV3(
//!     version: 3,
//!     entities: [
//!         EntityRecord(
//!             stable_id: "550e8400-e29b-41d4-a716-446655440000",
//!             parent: None,
//!             components: {
//!                 "Transform": "(position:(chunk:(1,0,0),local:(3.0,2.0,1.0)),rotation:(0,0,0,1),scale:(1,1,1))",
//!                 "EditorMetadata": "(name:\"Guard\",tags:[],locked:false)",
//!                 "SdfTree": "(asset_path:Some(\"assets/guard.rkf\"),aabb:(min:(-1,-1,-1),max:(1,1,1)))",
//!             },
//!         ),
//!     ],
//! )
//! ```

mod types;
mod save_load;

#[cfg(test)]
mod tests;

// Re-export all public items.
pub use types::*;
pub use save_load::*;
