//! Physics playground — composite SDF primitives and integration tests.
//!
//! Provides [`CompositeSdf`] (ground + sphere obstacles) and [`HillySdf`]
//! (sinusoidal terrain) for testing the full physics stack: rigid bodies
//! falling onto SDF ground, character controller walking on hills, and
//! destruction generating debris.

use glam::Vec3;

use crate::sdf_collision::SdfQueryable;

// ---------------------------------------------------------------------------
// CompositeSdf
// ---------------------------------------------------------------------------

/// A composite SDF built from a ground plane and sphere obstacles.
///
/// `evaluate()` returns the minimum distance across all primitives (union).
/// Useful for integration tests that need a simple multi-object world.
pub struct CompositeSdf {
    /// Y coordinate of the infinite ground plane.
    pub ground_height: f32,
    /// Sphere obstacles: `(center, radius)`.
    pub spheres: Vec<(Vec3, f32)>,
}

impl CompositeSdf {
    /// Create a composite SDF with a ground plane and no obstacles.
    pub fn ground_only(height: f32) -> Self {
        Self {
            ground_height: height,
            spheres: Vec::new(),
        }
    }

    /// Builder: add a sphere obstacle.
    pub fn with_sphere(mut self, center: Vec3, radius: f32) -> Self {
        self.spheres.push((center, radius));
        self
    }

    /// Find the index of the closest primitive at `pos`.
    /// Returns `None` for ground, `Some(i)` for sphere `i`.
    fn closest_primitive(&self, pos: Vec3) -> Option<usize> {
        let ground_dist = pos.y - self.ground_height;
        let mut min_dist = ground_dist;
        let mut closest = None;

        for (i, (center, radius)) in self.spheres.iter().enumerate() {
            let d = (pos - *center).length() - radius;
            if d < min_dist {
                min_dist = d;
                closest = Some(i);
            }
        }

        closest
    }
}

impl SdfQueryable for CompositeSdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        let mut min_dist = pos.y - self.ground_height;

        for (center, radius) in &self.spheres {
            let d = (pos - *center).length() - radius;
            min_dist = min_dist.min(d);
        }

        min_dist
    }

    fn gradient(&self, pos: Vec3) -> Vec3 {
        match self.closest_primitive(pos) {
            None => {
                // Ground plane — normal is always up
                Vec3::Y
            }
            Some(i) => {
                // Sphere — normal points outward from center
                let (center, _) = &self.spheres[i];
                (pos - *center).normalize_or_zero()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HillySdf
// ---------------------------------------------------------------------------

/// Ground surface with sinusoidal hills.
///
/// Height at `(x, z)` is `base + amplitude * sin(x * freq) * sin(z * freq)`.
/// The SDF is `pos.y - height_at(pos.x, pos.z)` — positive above, negative below.
pub struct HillySdf {
    /// Base ground height. Default: `0.0`.
    pub base: f32,
    /// Hill amplitude. Default: `1.0`.
    pub amplitude: f32,
    /// Spatial frequency of the sine waves. Default: `1.0`.
    pub frequency: f32,
}

impl HillySdf {
    /// Create hilly terrain with default parameters.
    pub fn new(base: f32, amplitude: f32, frequency: f32) -> Self {
        Self {
            base,
            amplitude,
            frequency,
        }
    }

    /// Compute terrain height at `(x, z)`.
    #[inline]
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        self.base + self.amplitude * (x * self.frequency).sin() * (z * self.frequency).sin()
    }
}

impl SdfQueryable for HillySdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        pos.y - self.height_at(pos.x, pos.z)
    }

    fn gradient(&self, pos: Vec3) -> Vec3 {
        // Analytic normal from height derivatives:
        // h(x,z) = base + A * sin(x*f) * sin(z*f)
        // dh/dx  = A * f * cos(x*f) * sin(z*f)
        // dh/dz  = A * f * sin(x*f) * cos(z*f)
        // Normal = normalize(-dh/dx, 1, -dh/dz)
        let a = self.amplitude;
        let f = self.frequency;
        let sx = (pos.x * f).sin();
        let cx = (pos.x * f).cos();
        let sz = (pos.z * f).sin();
        let cz = (pos.z * f).cos();

        let dhdx = a * f * cx * sz;
        let dhdz = a * f * sx * cz;

        Vec3::new(-dhdx, 1.0, -dhdz).normalize_or_zero()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::character_controller::SdfCharacterController;
    use crate::destruction::{DebrisConfig, DestructionEvent, generate_debris};
    use crate::rapier_world::{PhysicsConfig, PhysicsWorld};
    use crate::rigid_body::{
        process_sdf_collisions, spawn_rigid_body, BodyType,
    };
    use crate::sdf_collision::CollisionShape;

    #[test]
    fn test_rigid_body_falls_to_ground() {
        let sdf = CompositeSdf::ground_only(0.0);
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
            timestep: 1.0 / 60.0,
            max_substeps: 10,
        });

        // Drop a sphere from height 5
        let comp = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 5.0, 0.0),
            glam::Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.5 },
            1.0,
        );

        // Simulate 3 seconds (180 frames at 60Hz)
        for _ in 0..180 {
            world.step(1.0 / 60.0);
            process_sdf_collisions(&mut world, &[comp.clone()], &sdf);
        }

        // Sphere should be resting near the ground, not fallen through
        let body = world.get_body(comp.handle).unwrap();
        let y = body.translation().y;
        assert!(
            y > -0.5 && y < 2.0,
            "sphere should rest on ground: y={y}"
        );
    }

    #[test]
    fn test_character_walks_on_hills() {
        let hills = HillySdf::new(0.0, 0.5, 1.0);
        let mut cc = SdfCharacterController::new();

        // Start on top of the terrain
        let start_height = hills.height_at(0.0, 0.0);
        cc.position = Vec3::new(0.0, start_height + 0.8, 0.0);

        // Walk forward in +X for 60 steps
        let dt = 1.0 / 60.0;
        let walk_speed = Vec3::new(3.0, 0.0, 0.0);

        for _ in 0..60 {
            cc.update_ground_state(&hills);
            cc.apply_gravity(9.81, dt);

            let move_vel = Vec3::new(walk_speed.x, cc.velocity.y, walk_speed.z);
            cc.move_and_slide(move_vel, &hills, dt);
        }

        // Should have moved forward in X
        assert!(
            cc.position.x > 1.0,
            "character should have walked forward: x={}",
            cc.position.x
        );

        // Should still be roughly on the terrain surface (not fallen through
        // or launched into the sky)
        let expected_height = hills.height_at(cc.position.x, cc.position.z);
        let height_diff = (cc.position.y - (expected_height + 0.8)).abs();
        assert!(
            height_diff < 2.0,
            "character should be near terrain: pos.y={}, expected ~{}, diff={}",
            cc.position.y,
            expected_height + 0.8,
            height_diff
        );
    }

    #[test]
    fn test_character_stops_at_wall() {
        // Ground at y=0 with a sphere wall at x=3
        let sdf = CompositeSdf::ground_only(0.0)
            .with_sphere(Vec3::new(3.0, 0.5, 0.0), 1.0);

        let mut cc = SdfCharacterController::new();
        cc.position = Vec3::new(0.0, 0.8, 0.0);

        // Walk toward the wall
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            cc.update_ground_state(&sdf);
            cc.apply_gravity(9.81, dt);
            let move_vel = Vec3::new(5.0, cc.velocity.y, 0.0);
            cc.move_and_slide(move_vel, &sdf, dt);
        }

        // Should NOT have passed through the sphere center at x=3
        // The capsule (radius 0.3) should stop before the sphere surface (at x=2.0)
        assert!(
            cc.position.x < 3.0,
            "character should not pass through wall: x={}",
            cc.position.x
        );
    }

    #[test]
    fn test_destruction_generates_debris() {
        let event = DestructionEvent {
            center: Vec3::new(10.0, 5.0, 3.0),
            radius: 2.0,
            material_id: 5,
            force: 15.0,
        };
        let config = DebrisConfig {
            max_debris_count: 25,
            ..Default::default()
        };

        let debris = generate_debris(&event, &config, 42);

        assert_eq!(debris.len(), 25);

        // All debris should be near the destruction center
        for d in &debris {
            let dist = (d.position - event.center).length();
            assert!(
                dist <= event.radius + 0.01,
                "debris should spawn within radius: dist={dist}"
            );
        }

        // All should have outward velocity
        for d in &debris {
            assert!(
                d.velocity.length() > 0.1,
                "debris should have velocity"
            );
        }

        // All should have correct material
        for d in &debris {
            assert_eq!(d.material_id, 5);
        }
    }

    #[test]
    fn test_multiple_rigid_bodies() {
        let sdf = CompositeSdf::ground_only(0.0);
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: glam::Vec3::new(0.0, -9.81, 0.0),
            timestep: 1.0 / 60.0,
            max_substeps: 10,
        });

        // Spawn 3 spheres at different heights
        let comp1 = spawn_rigid_body(
            &mut world,
            Vec3::new(-2.0, 3.0, 0.0),
            glam::Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.5 },
            1.0,
        );
        let comp2 = spawn_rigid_body(
            &mut world,
            Vec3::new(0.0, 6.0, 0.0),
            glam::Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.3 },
            0.5,
        );
        let comp3 = spawn_rigid_body(
            &mut world,
            Vec3::new(2.0, 10.0, 0.0),
            glam::Quat::IDENTITY,
            BodyType::Dynamic,
            CollisionShape::Sphere { radius: 0.4 },
            0.8,
        );

        let comps = [comp1.clone(), comp2.clone(), comp3.clone()];

        // Simulate 3 seconds
        for _ in 0..180 {
            world.step(1.0 / 60.0);
            process_sdf_collisions(&mut world, &comps, &sdf);
        }

        // All three should be resting near the ground
        for (i, comp) in comps.iter().enumerate() {
            let body = world.get_body(comp.handle).unwrap();
            let y = body.translation().y;
            assert!(
                y > -1.0 && y < 3.0,
                "body[{i}] should rest near ground: y={y}"
            );
        }

        // They should be at different X positions (didn't merge)
        let x1 = world.get_body(comp1.handle).unwrap().translation().x;
        let x2 = world.get_body(comp2.handle).unwrap().translation().x;
        let x3 = world.get_body(comp3.handle).unwrap().translation().x;
        assert!(
            (x1 - x2).abs() > 0.5 && (x2 - x3).abs() > 0.5,
            "bodies should remain separated: x1={x1}, x2={x2}, x3={x3}"
        );
    }
}
