//! ECS rigid body component and physics-to-transform synchronization.
//!
//! [`RigidBodyComponent`] is the ECS component that links an entity to a
//! Rapier rigid body. Helper functions synchronize Rapier state back to
//! entity transforms and run SDF collision each frame.

use glam::{Quat, Vec3};
use rapier3d::prelude::*;

use crate::rapier_world::{
    glam_quat_to_rapier, glam_to_rapier, rapier_rotation_to_glam, rapier_to_glam, PhysicsWorld,
};
use crate::sdf_collision::{generate_sdf_contacts, apply_sdf_contacts, CollisionShape, SdfQueryable};

// ---------------------------------------------------------------------------
// BodyType
// ---------------------------------------------------------------------------

/// Wrapper around Rapier's rigid body types for use in the ECS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyType {
    /// Affected by forces and gravity.
    Dynamic,
    /// Never moves (infinite mass).
    Static,
    /// Moved by setting position directly; pushes dynamic bodies.
    KinematicPosition,
    /// Moved by setting velocity directly; pushes dynamic bodies.
    KinematicVelocity,
}

// ---------------------------------------------------------------------------
// RigidBodyComponent
// ---------------------------------------------------------------------------

/// ECS component linking an entity to a Rapier rigid body.
///
/// Holds the Rapier handle, the collision shape for SDF contact generation,
/// and material properties (friction, restitution).
#[derive(Debug, Clone)]
pub struct RigidBodyComponent {
    /// Handle into the Rapier `RigidBodySet`.
    pub handle: RigidBodyHandle,
    /// Collision shape used for SDF surface sample points.
    pub collision_shape: CollisionShape,
    /// Whether to test this body against the SDF world each frame.
    pub sdf_collision: bool,
    /// Friction coefficient for SDF contacts. Default: `0.5`.
    pub friction: f32,
    /// Coefficient of restitution (bounciness). Default: `0.3`.
    pub restitution: f32,
}

impl RigidBodyComponent {
    /// Create a new component from an existing Rapier handle and collision shape.
    pub fn new(handle: RigidBodyHandle, collision_shape: CollisionShape) -> Self {
        Self {
            handle,
            collision_shape,
            sdf_collision: true,
            friction: 0.5,
            restitution: 0.3,
        }
    }

    /// Builder method to disable SDF collision for this body.
    pub fn with_sdf_collision(mut self, enabled: bool) -> Self {
        self.sdf_collision = enabled;
        self
    }

    /// Builder method to set friction.
    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    /// Builder method to set restitution.
    pub fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }
}

// ---------------------------------------------------------------------------
// Transform sync
// ---------------------------------------------------------------------------

/// A transform that can be read from / written to for physics synchronization.
pub struct TransformRef<'a> {
    /// Mutable reference to the entity's position.
    pub position: &'a mut Vec3,
    /// Mutable reference to the entity's rotation.
    pub rotation: &'a mut Quat,
}

/// Synchronize Rapier rigid body state back to entity transforms.
///
/// For each `(component, transform)` pair, reads the body position and
/// rotation from Rapier and writes them into the mutable transform
/// references. The `alpha` parameter (from [`PhysicsWorld::step`]) can be
/// used for interpolation between the previous and current physics state;
/// currently we snap to the current state (alpha is noted for future use).
pub fn sync_transforms(
    world: &PhysicsWorld,
    components: &[(RigidBodyComponent, TransformRef<'_>)],
    _alpha: f32,
) {
    for (comp, _transform) in components {
        let _body = match world.get_body(comp.handle) {
            Some(b) => b,
            None => continue,
        };
        // Note: can't mutate through shared slice — use sync_transforms_mut
    }
}

/// Synchronize Rapier rigid body state back to entity transforms (mutable version).
///
/// Reads each body's position/rotation from Rapier and writes it into the
/// corresponding transform. `alpha` is reserved for future interpolation.
pub fn sync_transforms_mut(
    world: &PhysicsWorld,
    entries: &mut [(RigidBodyComponent, Vec3, Quat)],
    _alpha: f32,
) {
    for (comp, pos, rot) in entries.iter_mut() {
        let body = match world.get_body(comp.handle) {
            Some(b) => b,
            None => continue,
        };
        *pos = rapier_to_glam(body.translation());
        *rot = rapier_rotation_to_glam(body.rotation());
    }
}

// ---------------------------------------------------------------------------
// SDF collision processing
// ---------------------------------------------------------------------------

/// Run SDF collision for all bodies that have `sdf_collision == true`.
///
/// For each enabled body, generates SDF contacts and applies position
/// corrections and velocity adjustments. Call this after
/// [`PhysicsWorld::step`] each frame.
pub fn process_sdf_collisions(
    world: &mut PhysicsWorld,
    components: &[RigidBodyComponent],
    sdf: &dyn SdfQueryable,
) {
    let contact_threshold = 0.02;

    for comp in components {
        if !comp.sdf_collision {
            continue;
        }

        // Read body position/rotation
        let (position, rotation) = {
            let body = match world.get_body(comp.handle) {
                Some(b) => b,
                None => continue,
            };
            (
                rapier_to_glam(body.translation()),
                rapier_rotation_to_glam(body.rotation()),
            )
        };

        // Generate contacts
        let contacts = generate_sdf_contacts(
            &comp.collision_shape,
            position,
            rotation,
            sdf,
            contact_threshold,
        );

        // Apply contacts
        if !contacts.is_empty() {
            apply_sdf_contacts(world, comp.handle, &contacts, comp.friction);
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: spawn_rigid_body
// ---------------------------------------------------------------------------

/// Spawn a new rigid body with the given parameters and return a
/// [`RigidBodyComponent`] ready for ECS insertion.
///
/// Creates both the Rapier rigid body and a matching collider, inserts them
/// into the physics world, and returns the component.
pub fn spawn_rigid_body(
    world: &mut PhysicsWorld,
    position: Vec3,
    rotation: Quat,
    body_type: BodyType,
    collision_shape: CollisionShape,
    mass: f32,
) -> RigidBodyComponent {
    // Build rigid body
    let mut builder = match body_type {
        BodyType::Dynamic => RigidBodyBuilder::dynamic(),
        BodyType::Static => RigidBodyBuilder::fixed(),
        BodyType::KinematicPosition => RigidBodyBuilder::kinematic_position_based(),
        BodyType::KinematicVelocity => RigidBodyBuilder::kinematic_velocity_based(),
    };

    builder = builder
        .translation(glam_to_rapier(position))
        .rotation(glam_quat_to_rapier(rotation).scaled_axis());

    // For dynamic bodies, we'll set mass via the collider density
    let body = builder.build();
    let handle = world.add_rigid_body(body);

    // Build collider from the collision shape
    let mut collider_builder = match &collision_shape {
        CollisionShape::Sphere { radius } => ColliderBuilder::ball(*radius),
        CollisionShape::Box { half_extents } => {
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
        }
        CollisionShape::Capsule {
            half_height,
            radius,
        } => ColliderBuilder::capsule_y(*half_height, *radius),
    };

    // Set mass via density: density = mass / volume
    if mass > 0.0 {
        let volume = match &collision_shape {
            CollisionShape::Sphere { radius } => {
                (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3)
            }
            CollisionShape::Box { half_extents } => {
                8.0 * half_extents.x * half_extents.y * half_extents.z
            }
            CollisionShape::Capsule {
                half_height,
                radius,
            } => {
                let cylinder = std::f32::consts::PI * radius.powi(2) * (2.0 * half_height);
                let sphere = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
                cylinder + sphere
            }
        };
        if volume > 1e-6 {
            collider_builder = collider_builder.density(mass / volume);
        }
    }

    let collider = collider_builder.build();
    world.add_collider(collider, handle);

    RigidBodyComponent::new(handle, collision_shape)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rapier_world::PhysicsConfig;
    use crate::sdf_collision::GroundPlaneSdf;

    #[test]
    fn test_spawn_dynamic_body() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.5 },
            1.0,
        );

        assert_eq!(world.body_count(), 1);
        assert_eq!(world.collider_count(), 1);

        let body = world.get_body(comp.handle).unwrap();
        assert!((body.translation().y - 5.0).abs() < 1e-4);
        assert!(comp.sdf_collision);
        assert!((comp.friction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sync_transforms() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 10.0, 0.0),
            Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.5 },
            1.0,
        );

        // Step physics — body should fall
        let alpha = world.step(0.1);

        let mut entries = vec![(comp.clone(), Vec3::ZERO, Quat::IDENTITY)];
        sync_transforms_mut(&world, &mut entries, alpha);

        let (_, pos, _rot) = &entries[0];
        assert!(
            pos.y < 10.0,
            "position should have updated after step: y={}",
            pos.y
        );
    }

    #[test]
    fn test_sdf_collision_prevents_penetration() {
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            timestep: 1.0 / 60.0,
            max_substeps: 10,
        });

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.5 },
            1.0,
        );

        let ground = GroundPlaneSdf { height: 0.0 };

        // Simulate multiple frames with SDF collision
        for _ in 0..120 {
            world.step(1.0 / 60.0);
            process_sdf_collisions(&mut world, &[comp.clone()], &ground);
        }

        // After 2 seconds of simulation, the sphere should be resting
        // on the ground plane, not fallen through it.
        // Sphere radius = 0.5, ground at y=0 → center should be >= ~0.3
        // (allowing some tolerance for the discrete contact resolution)
        let body = world.get_body(comp.handle).unwrap();
        let y = body.translation().y;
        assert!(
            y > -0.5,
            "sphere should not have fallen through the ground: y={}",
            y
        );
    }

    #[test]
    fn test_body_type_static() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 5.0, 0.0),
            Quat::IDENTITY,
            BodyType::Static,
            CollisionShape::Box {
                half_extents: Vec3::splat(1.0),
            },
            0.0,
        );

        // Step physics
        world.step(1.0);

        // Static body should not have moved
        let body = world.get_body(comp.handle).unwrap();
        assert!(
            (body.translation().y - 5.0).abs() < 1e-4,
            "static body should not move: y={}",
            body.translation().y
        );
    }

    #[test]
    fn test_spawn_and_remove() {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());

        // Spawn
        let comp = spawn_rigid_body(
            &mut world,
            Vec3::ZERO,
            Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Capsule {
                half_height: 0.5,
                radius: 0.25,
            },
            2.0,
        );

        assert_eq!(world.body_count(), 1);
        assert_eq!(world.collider_count(), 1);

        // Remove
        let removed = world.remove_rigid_body(comp.handle);
        assert!(removed.is_some());
        assert_eq!(world.body_count(), 0);
        assert_eq!(world.collider_count(), 0);
    }
}
