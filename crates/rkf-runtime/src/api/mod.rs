//! # RKIField Engine API
//!
//! The public API for building games and tools with the RKIField SDF engine.
//!
//! This module provides two main types:
//! - [`World`] — the game state container (scenes + ECS)
//! - [`Renderer`] — the GPU rendering pipeline
//!
//! Entities are identified by [`uuid::Uuid`] (via the `StableId` component).
//!
//! # Quick Start
//!
//! ```ignore
//! let mut world = World::new("my_game");
//! let cube_id = world.spawn("cube")
//!     .position_vec3(Vec3::new(0.0, 1.0, -3.0))
//!     .sdf(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) })
//!     .material(1)
//!     .build();
//! ```

pub mod error;
pub mod renderer;
pub mod spawn;
pub mod world;

pub use error::WorldError;
pub use renderer::{BrushHitResult, RenderEnvironment, Renderer, RendererConfig, RenderTarget};
pub use spawn::SpawnBuilder;
pub use world::World;

// Re-export wireframe types from rkf-render for convenience.
pub use rkf_render::wireframe::{
    aabb_wireframe, circle_wireframe, crosshair, directional_light_wireframe,
    ground_grid_wireframe, obb_wireframe, point_light_wireframe, sphere_wireframe,
    spot_light_wireframe, LineVertex, WireframePass,
};
