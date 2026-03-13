//! Internal and advanced accessors.

use glam::Vec3;
use uuid::Uuid;

use rkf_core::aabb::Aabb;
use rkf_core::brick_map::BrickMapAllocator;
use rkf_core::brick_pool::BrickPool;

use super::{World, WorldError};

impl World {
    // ── Internal accessors ─────────────────────────────────────────────────

    /// Get a reference to the brick pool.
    pub(crate) fn brick_pool(&self) -> &BrickPool {
        &self.brick_pool
    }

    /// Get a reference to the brick map allocator.
    pub(crate) fn brick_map_alloc(&self) -> &BrickMapAllocator {
        &self.brick_map_alloc
    }

    /// Get the hecs entity for an API entity (for SpawnBuilder component insertion).
    pub fn ecs_entity_for(&self, entity_id: Uuid) -> Option<hecs::Entity> {
        self.entities.get(&entity_id).map(|r| r.ecs_entity)
    }

    /// Get a shared reference to the underlying hecs world.
    ///
    /// Advanced: used by the automation API and inspector to call type-erased
    /// `ComponentEntry` function pointers that operate on raw `hecs::World`.
    pub fn ecs_ref(&self) -> &hecs::World {
        &self.ecs
    }

    /// Alias for [`ecs_ref`](Self::ecs_ref).
    pub fn ecs(&self) -> &hecs::World {
        &self.ecs
    }

    /// Get a mutable reference to the hecs world.
    pub fn ecs_mut(&mut self) -> &mut hecs::World {
        &mut self.ecs
    }

    /// Iterate over all entity records (for I/O operations that need object metadata).
    pub fn entity_records(&self) -> impl Iterator<Item = (&Uuid, &super::EntityRecord)> {
        self.entities.iter()
    }

    // ── Advanced access ─────────────────────────────────────────────────────

    /// Voxelize an SDF function into this world's brick pool.
    ///
    /// Returns the brick map handle and brick count on success.
    pub fn voxelize<F>(
        &mut self,
        sdf_fn: F,
        aabb: &Aabb,
        voxel_size: f32,
    ) -> Result<(rkf_core::BrickMapHandle, u32), WorldError>
    where
        F: Fn(Vec3) -> (f32, u16),
    {
        rkf_core::voxelize_sdf(
            sdf_fn,
            aabb,
            voxel_size,
            &mut self.brick_pool,
            &mut self.brick_map_alloc,
        )
        .ok_or(WorldError::Voxelize("voxelization produced no bricks".into()))
    }
}
