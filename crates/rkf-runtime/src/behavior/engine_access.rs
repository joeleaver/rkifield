//! Cross-dylib bridge for engine component access.
//!
//! Game dylibs have different `TypeId`s for shared types like `Transform`.
//! `EngineAccess` provides a trait-object vtable that dispatches reads to
//! host code, which uses the correct (host) TypeIds.

use glam::{Quat, Vec3};
use rkf_core::WorldPosition;

use crate::components::Transform;

/// Read-only access to engine components via host TypeIds.
///
/// The host creates a `WorldEngineAccess` that implements this trait.
/// Game systems call through the vtable, dispatching to host code that
/// can actually find the components in the hecs world.
pub trait EngineAccess {
    /// Read the position of an entity's Transform.
    fn position(&self, entity: hecs::Entity) -> Option<WorldPosition>;
    /// Read the rotation of an entity's Transform.
    fn rotation(&self, entity: hecs::Entity) -> Option<Quat>;
    /// Read the scale of an entity's Transform.
    fn scale(&self, entity: hecs::Entity) -> Option<Vec3>;
    /// Read the full transform (position, rotation, scale).
    fn transform(&self, entity: hecs::Entity) -> Option<(WorldPosition, Quat, Vec3)>;
    /// Collect all entities that have a Transform component.
    /// Returns (entity, position, rotation, scale) tuples.
    fn all_transforms(&self) -> Vec<(hecs::Entity, WorldPosition, Quat, Vec3)>;
}

/// Pending transform write, applied by the executor after each system.
#[allow(missing_docs)]
pub struct TransformUpdate {
    pub entity: hecs::Entity,
    pub position: Option<WorldPosition>,
    pub rotation: Option<Quat>,
    pub scale: Option<Vec3>,
}

/// Host-side `EngineAccess` backed by a raw pointer to the hecs World.
///
/// # Safety
///
/// The world pointer must remain valid and unaliased for the lifetime of
/// this struct. Only shared (read) access occurs through the pointer.
pub struct WorldEngineAccess {
    world: *const hecs::World,
}

impl WorldEngineAccess {
    /// Create a new `WorldEngineAccess` from a raw world pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The pointer is valid for the duration of all `EngineAccess` calls.
    /// - No mutable access to the world occurs while an `EngineAccess`
    ///   method is executing.
    pub unsafe fn new(world: *const hecs::World) -> Self {
        Self { world }
    }
}

impl EngineAccess for WorldEngineAccess {
    fn position(&self, entity: hecs::Entity) -> Option<WorldPosition> {
        let world = unsafe { &*self.world };
        world.get::<&Transform>(entity).ok().map(|t| t.position.clone())
    }

    fn rotation(&self, entity: hecs::Entity) -> Option<Quat> {
        let world = unsafe { &*self.world };
        world.get::<&Transform>(entity).ok().map(|t| t.rotation)
    }

    fn scale(&self, entity: hecs::Entity) -> Option<Vec3> {
        let world = unsafe { &*self.world };
        world.get::<&Transform>(entity).ok().map(|t| t.scale)
    }

    fn transform(&self, entity: hecs::Entity) -> Option<(WorldPosition, Quat, Vec3)> {
        let world = unsafe { &*self.world };
        world
            .get::<&Transform>(entity)
            .ok()
            .map(|t| (t.position.clone(), t.rotation, t.scale))
    }

    fn all_transforms(&self) -> Vec<(hecs::Entity, WorldPosition, Quat, Vec3)> {
        let world = unsafe { &*self.world };
        world
            .query::<&Transform>()
            .iter()
            .map(|(e, t)| (e, t.position.clone(), t.rotation, t.scale))
            .collect()
    }
}
