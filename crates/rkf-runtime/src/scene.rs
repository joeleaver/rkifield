//! Scene management — v2 object-centric scenes + hecs ECS for non-SDF entities.
//!
//! [`RuntimeScene`] holds:
//! - A v2 [`rkf_core::Scene`] containing SDF objects with scene node trees
//! - A hecs [`World`] for non-SDF entities (lights, cameras, fog volumes)
//!
//! In v2, SDF objects are managed through the scene hierarchy (SceneObject → SceneNode tree).
//! Non-SDF entities (lights, cameras, etc.) remain in the ECS world.

use hecs::World;

use crate::components::{
    CameraComponent, FogVolumeComponent, Parent, Transform, WorldTransform,
};

/// The runtime scene — owns both v2 SDF scene and hecs world for non-SDF entities.
pub struct RuntimeScene {
    /// v2 scene containing SDF objects (SceneObject with SceneNode trees).
    pub sdf_scene: rkf_core::scene::Scene,
    /// hecs ECS world for non-SDF entities (lights, cameras, fog volumes, etc.).
    pub world: World,
}

impl RuntimeScene {
    /// Create an empty runtime scene.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            sdf_scene: rkf_core::scene::Scene::new(name),
            world: World::new(),
        }
    }

    /// Spawn a camera entity in the ECS world.
    pub fn spawn_camera(
        &mut self,
        transform: Transform,
        camera: CameraComponent,
    ) -> hecs::Entity {
        self.world
            .spawn((transform, WorldTransform::default(), camera))
    }

    /// Spawn a light entity in the ECS world.
    pub fn spawn_light(&mut self, transform: Transform) -> hecs::Entity {
        self.world.spawn((transform, WorldTransform::default()))
    }

    /// Spawn a fog volume entity in the ECS world.
    pub fn spawn_fog_volume(
        &mut self,
        transform: Transform,
        fog: FogVolumeComponent,
    ) -> hecs::Entity {
        self.world
            .spawn((transform, WorldTransform::default(), fog))
    }

    /// Set an entity's parent (for transform hierarchy in ECS).
    pub fn set_parent(
        &mut self,
        child: hecs::Entity,
        parent_entity: hecs::Entity,
        bone_index: Option<u32>,
    ) -> Result<(), hecs::NoSuchEntity> {
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

    /// Despawn an ECS entity.
    pub fn despawn(&mut self, entity: hecs::Entity) -> Result<(), hecs::NoSuchEntity> {
        self.world.despawn(entity)
    }

    /// Count ECS entities (lights, cameras, etc.).
    pub fn ecs_entity_count(&self) -> u32 {
        self.world.len()
    }

    /// Count SDF objects in the v2 scene.
    pub fn sdf_object_count(&self) -> usize {
        self.sdf_scene.object_count()
    }

    /// Clear everything — both SDF scene and ECS world.
    pub fn clear(&mut self) {
        self.sdf_scene.objects.clear();
        self.world.clear();
    }
}

impl Default for RuntimeScene {
    fn default() -> Self {
        Self::new("default")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::scene_node::{SceneNode, SdfPrimitive};
    use glam::Vec3;

    #[test]
    fn new_scene_empty() {
        let scene = RuntimeScene::new("test");
        assert_eq!(scene.ecs_entity_count(), 0);
        assert_eq!(scene.sdf_object_count(), 0);
    }

    #[test]
    fn spawn_camera() {
        let mut scene = RuntimeScene::new("test");
        let _e = scene.spawn_camera(Transform::default(), CameraComponent::default());
        assert_eq!(scene.ecs_entity_count(), 1);
    }

    #[test]
    fn spawn_fog_volume() {
        let mut scene = RuntimeScene::new("test");
        let _e = scene.spawn_fog_volume(Transform::default(), FogVolumeComponent::default());
        assert_eq!(scene.ecs_entity_count(), 1);
    }

    #[test]
    fn add_sdf_object() {
        let mut scene = RuntimeScene::new("test");
        let node = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.5 },
            1,
        );
        scene.sdf_scene.add_object("sphere_obj", Vec3::ZERO, node);
        assert_eq!(scene.sdf_object_count(), 1);
    }

    #[test]
    fn mixed_sdf_and_ecs() {
        let mut scene = RuntimeScene::new("test");

        // SDF objects
        let node = SceneNode::new("root");
        scene.sdf_scene.add_object("obj1", Vec3::ZERO, node);

        // ECS entities
        scene.spawn_light(Transform::default());
        scene.spawn_camera(Transform::default(), CameraComponent::default());

        assert_eq!(scene.sdf_object_count(), 1);
        assert_eq!(scene.ecs_entity_count(), 2);
    }

    #[test]
    fn despawn_ecs_entity() {
        let mut scene = RuntimeScene::new("test");
        let e = scene.spawn_light(Transform::default());
        assert_eq!(scene.ecs_entity_count(), 1);
        scene.despawn(e).unwrap();
        assert_eq!(scene.ecs_entity_count(), 0);
    }

    #[test]
    fn parent_child() {
        let mut scene = RuntimeScene::new("test");
        let parent = scene.spawn_light(Transform::default());
        let child = scene.spawn_light(Transform::default());
        scene.set_parent(child, parent, None).unwrap();
        let p = scene.world.get::<&Parent>(child).unwrap();
        assert_eq!(p.entity, parent);
    }

    #[test]
    fn clear() {
        let mut scene = RuntimeScene::new("test");
        scene.spawn_light(Transform::default());
        let node = SceneNode::new("root");
        scene.sdf_scene.add_object("obj", Vec3::ZERO, node);

        scene.clear();
        assert_eq!(scene.ecs_entity_count(), 0);
        assert_eq!(scene.sdf_object_count(), 0);
    }
}
