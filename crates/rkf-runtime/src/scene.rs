//! Scene management via hecs ECS.
//!
//! Thin wrapper around [`hecs::World`] providing convenience methods
//! for common entity operations in the RKIField engine.

use hecs::World;

use crate::components::{
    CameraComponent, FogVolumeComponent, Parent, SdfObject, Transform, WorldTransform,
};

/// The scene world — owns all entities and their components.
pub struct Scene {
    /// The hecs ECS world.
    pub world: World,
}

impl Scene {
    /// Create an empty scene.
    pub fn new() -> Self {
        Self {
            world: World::new(),
        }
    }

    /// Spawn a static SDF object with transform.
    pub fn spawn_sdf_object(
        &mut self,
        transform: Transform,
        sdf: SdfObject,
    ) -> hecs::Entity {
        self.world.spawn((transform, WorldTransform::default(), sdf))
    }

    /// Spawn a light entity.
    pub fn spawn_light(&mut self, transform: Transform) -> hecs::Entity {
        self.world.spawn((transform, WorldTransform::default()))
    }

    /// Spawn a camera entity.
    pub fn spawn_camera(
        &mut self,
        transform: Transform,
        camera: CameraComponent,
    ) -> hecs::Entity {
        self.world.spawn((transform, WorldTransform::default(), camera))
    }

    /// Spawn a fog volume entity.
    pub fn spawn_fog_volume(
        &mut self,
        transform: Transform,
        fog: FogVolumeComponent,
    ) -> hecs::Entity {
        self.world.spawn((transform, WorldTransform::default(), fog))
    }

    /// Despawn an entity.
    pub fn despawn(&mut self, entity: hecs::Entity) -> Result<(), hecs::NoSuchEntity> {
        self.world.despawn(entity)
    }

    /// Set an entity's parent (for transform hierarchy).
    pub fn set_parent(
        &mut self,
        child: hecs::Entity,
        parent_entity: hecs::Entity,
        bone_index: Option<u32>,
    ) -> Result<(), hecs::NoSuchEntity> {
        // Verify parent exists.
        if !self.world.contains(parent_entity) {
            return Err(hecs::NoSuchEntity);
        }
        self.world
            .insert_one(
                child,
                Parent {
                    entity: parent_entity,
                    bone_index,
                },
            )
            .map_err(|_| hecs::NoSuchEntity)?;
        Ok(())
    }

    /// Remove an entity's parent (make it a root entity).
    pub fn remove_parent(&mut self, entity: hecs::Entity) -> Result<(), hecs::ComponentError> {
        self.world.remove_one::<Parent>(entity).map(|_| ())
    }

    /// Count all entities in the scene.
    pub fn entity_count(&self) -> u32 {
        self.world.len()
    }

    /// Clear all entities.
    pub fn clear(&mut self) {
        self.world.clear();
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scene_empty() {
        let scene = Scene::new();
        assert_eq!(scene.entity_count(), 0);
    }

    #[test]
    fn spawn_and_count() {
        let mut scene = Scene::new();
        let _e = scene.spawn_camera(Transform::default(), CameraComponent::default());
        assert_eq!(scene.entity_count(), 1);
    }

    #[test]
    fn spawn_sdf_object() {
        let mut scene = Scene::new();
        let sdf = SdfObject {
            brick_start: 0,
            brick_count: 10,
            tier: 0,
        };
        let e = scene.spawn_sdf_object(Transform::default(), sdf);
        assert!(scene.world.get::<&SdfObject>(e).is_ok());
        assert!(scene.world.get::<&WorldTransform>(e).is_ok());
    }

    #[test]
    fn spawn_fog_volume() {
        let mut scene = Scene::new();
        let e = scene.spawn_fog_volume(Transform::default(), FogVolumeComponent::default());
        assert!(scene.world.get::<&FogVolumeComponent>(e).is_ok());
    }

    #[test]
    fn despawn() {
        let mut scene = Scene::new();
        let e = scene.spawn_camera(Transform::default(), CameraComponent::default());
        assert_eq!(scene.entity_count(), 1);
        scene.despawn(e).unwrap();
        assert_eq!(scene.entity_count(), 0);
    }

    #[test]
    fn parent_child() {
        let mut scene = Scene::new();
        let parent = scene.spawn_sdf_object(
            Transform::default(),
            SdfObject {
                brick_start: 0,
                brick_count: 5,
                tier: 0,
            },
        );
        let child = scene.spawn_sdf_object(
            Transform::default(),
            SdfObject {
                brick_start: 5,
                brick_count: 3,
                tier: 0,
            },
        );
        scene.set_parent(child, parent, None).unwrap();
        let p = scene.world.get::<&Parent>(child).unwrap();
        assert_eq!(p.entity, parent);
        assert!(p.bone_index.is_none());
    }

    #[test]
    fn remove_parent() {
        let mut scene = Scene::new();
        let parent = scene.spawn_light(Transform::default());
        let child = scene.spawn_light(Transform::default());
        scene.set_parent(child, parent, Some(3)).unwrap();
        assert!(scene.world.get::<&Parent>(child).is_ok());
        scene.remove_parent(child).unwrap();
        assert!(scene.world.get::<&Parent>(child).is_err());
    }

    #[test]
    fn clear() {
        let mut scene = Scene::new();
        for _ in 0..5 {
            scene.spawn_light(Transform::default());
        }
        assert_eq!(scene.entity_count(), 5);
        scene.clear();
        assert_eq!(scene.entity_count(), 0);
    }
}
