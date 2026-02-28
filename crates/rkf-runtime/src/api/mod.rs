//! # RKIField Engine API
//!
//! The public API for building games and tools with the RKIField SDF engine.
//!
//! This module provides three main types:
//! - [`Entity`] — opaque handle to an object in the world
//! - [`World`] — the game state container (scenes + ECS)
//! - [`Renderer`] — the GPU rendering pipeline
//!
//! # Quick Start
//!
//! ```ignore
//! let mut world = World::new("my_game");
//! let cube = world.spawn("cube")
//!     .position_vec3(Vec3::new(0.0, 1.0, -3.0))
//!     .sdf(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) })
//!     .material(1)
//!     .build();
//! ```

pub mod entity;
pub mod error;
pub mod renderer;
pub mod spawn;
pub mod world;

pub use entity::Entity;
pub use error::WorldError;
pub use renderer::{Renderer, RendererConfig, RenderTarget};
pub use spawn::SpawnBuilder;
pub use world::World;
