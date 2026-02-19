//! SDF collision adapter for generating contacts between rigid bodies
//! and the SDF world.
//!
//! Instead of mesh colliders, this module samples the SDF at points on a
//! body's surface and generates contact normals/depths from the distance
//! field gradient. The resulting contacts can be applied as position
//! corrections and velocity adjustments.

use glam::{Quat, Vec3};

use crate::rapier_world::PhysicsWorld;
use rapier3d::prelude::*;

// ---------------------------------------------------------------------------
// ContactPoint
// ---------------------------------------------------------------------------

/// A contact between a rigid body surface point and the SDF world.
#[derive(Debug, Clone)]
pub struct ContactPoint {
    /// World-space position of the contact.
    pub position: Vec3,
    /// Outward-pointing surface normal from the SDF (points away from solid).
    pub normal: Vec3,
    /// Penetration depth. Positive means the sample point is inside the SDF.
    pub penetration: f32,
}

// ---------------------------------------------------------------------------
// SdfQueryable trait
// ---------------------------------------------------------------------------

/// Abstraction for evaluating a signed distance field at arbitrary positions.
///
/// Negative distance means inside the surface. Implementors must provide both
/// [`evaluate`](SdfQueryable::evaluate) and [`gradient`](SdfQueryable::gradient).
/// Use [`gradient_central_diff`] for a finite-difference gradient if no analytic form exists.
pub trait SdfQueryable {
    /// Evaluate SDF distance at a world position. Negative = inside.
    fn evaluate(&self, pos: Vec3) -> f32;

    /// Compute the outward surface normal at `pos`.
    fn gradient(&self, pos: Vec3) -> Vec3;
}

/// Compute the SDF gradient at `pos` using central finite differences.
///
/// The epsilon is 0.01 world units — fine enough for physics contacts but
/// coarse enough to smooth over single-voxel noise.
pub fn gradient_central_diff(sdf: &dyn SdfQueryable, pos: Vec3) -> Vec3 {
    let eps = 0.01;
    let dx = sdf.evaluate(pos + Vec3::X * eps) - sdf.evaluate(pos - Vec3::X * eps);
    let dy = sdf.evaluate(pos + Vec3::Y * eps) - sdf.evaluate(pos - Vec3::Y * eps);
    let dz = sdf.evaluate(pos + Vec3::Z * eps) - sdf.evaluate(pos - Vec3::Z * eps);
    Vec3::new(dx, dy, dz).normalize_or_zero()
}

// ---------------------------------------------------------------------------
// CollisionShape
// ---------------------------------------------------------------------------

/// Collision shape used for sample-point generation.
///
/// These are simpler than the full SDF shapes — they define surface sample
/// points for the SDF contact test, not the SDF itself.
#[derive(Debug, Clone)]
pub enum CollisionShape {
    /// Sphere with the given radius.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Axis-aligned box with the given half-extents.
    Box {
        /// Half-extents along each axis.
        half_extents: Vec3,
    },
    /// Capsule aligned along the Y axis.
    Capsule {
        /// Half the height of the cylindrical shaft (not including hemisphere caps).
        half_height: f32,
        /// Radius of the capsule.
        radius: f32,
    },
}

impl CollisionShape {
    /// Generate surface sample points in local space.
    ///
    /// These points are transformed to world space by the caller before
    /// SDF evaluation. The number of samples is fixed per shape type to
    /// keep the contact generation cost predictable.
    pub fn sample_points(&self) -> Vec<Vec3> {
        match self {
            CollisionShape::Sphere { radius } => {
                let r = *radius;
                let d3 = r / 3.0_f32.sqrt();
                vec![
                    // 6 axis-aligned points
                    Vec3::new(r, 0.0, 0.0),
                    Vec3::new(-r, 0.0, 0.0),
                    Vec3::new(0.0, r, 0.0),
                    Vec3::new(0.0, -r, 0.0),
                    Vec3::new(0.0, 0.0, r),
                    Vec3::new(0.0, 0.0, -r),
                    // 8 diagonal points (corners of inscribed cube)
                    Vec3::new(d3, d3, d3),
                    Vec3::new(d3, d3, -d3),
                    Vec3::new(d3, -d3, d3),
                    Vec3::new(d3, -d3, -d3),
                    Vec3::new(-d3, d3, d3),
                    Vec3::new(-d3, d3, -d3),
                    Vec3::new(-d3, -d3, d3),
                    Vec3::new(-d3, -d3, -d3),
                ]
            }
            CollisionShape::Box { half_extents } => {
                let h = *half_extents;
                let mut points = Vec::with_capacity(14);
                // 8 corners
                for sx in [-1.0_f32, 1.0] {
                    for sy in [-1.0_f32, 1.0] {
                        for sz in [-1.0_f32, 1.0] {
                            points.push(Vec3::new(h.x * sx, h.y * sy, h.z * sz));
                        }
                    }
                }
                // 6 face centers
                points.push(Vec3::new(h.x, 0.0, 0.0));
                points.push(Vec3::new(-h.x, 0.0, 0.0));
                points.push(Vec3::new(0.0, h.y, 0.0));
                points.push(Vec3::new(0.0, -h.y, 0.0));
                points.push(Vec3::new(0.0, 0.0, h.z));
                points.push(Vec3::new(0.0, 0.0, -h.z));
                points
            }
            CollisionShape::Capsule {
                half_height,
                radius,
            } => {
                let hh = *half_height;
                let r = *radius;
                vec![
                    // Top hemisphere: pole + 4 equatorial
                    Vec3::new(0.0, hh + r, 0.0),
                    Vec3::new(r, hh, 0.0),
                    Vec3::new(-r, hh, 0.0),
                    Vec3::new(0.0, hh, r),
                    Vec3::new(0.0, hh, -r),
                    // Bottom hemisphere: pole + 4 equatorial
                    Vec3::new(0.0, -(hh + r), 0.0),
                    Vec3::new(r, -hh, 0.0),
                    Vec3::new(-r, -hh, 0.0),
                    Vec3::new(0.0, -hh, r),
                    Vec3::new(0.0, -hh, -r),
                    // 2 shaft midpoints
                    Vec3::new(r, 0.0, 0.0),
                    Vec3::new(-r, 0.0, 0.0),
                ]
            }
        }
    }

    /// Build a Rapier [`Collider`] matching this shape.
    pub fn to_rapier_collider(&self) -> Collider {
        match self {
            CollisionShape::Sphere { radius } => ColliderBuilder::ball(*radius).build(),
            CollisionShape::Box { half_extents } => {
                ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build()
            }
            CollisionShape::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(*half_height, *radius).build(),
        }
    }
}

// ---------------------------------------------------------------------------
// Contact generation
// ---------------------------------------------------------------------------

/// Generate SDF contacts for a shape at a given position and rotation.
///
/// For each sample point on the shape's surface, evaluates the SDF.
/// Points with SDF distance below `contact_threshold` generate a
/// [`ContactPoint`]. Results are sorted by penetration depth (deepest first).
pub fn generate_sdf_contacts(
    shape: &CollisionShape,
    position: Vec3,
    rotation: Quat,
    sdf: &dyn SdfQueryable,
    contact_threshold: f32,
) -> Vec<ContactPoint> {
    let local_points = shape.sample_points();
    let mut contacts = Vec::new();

    for lp in &local_points {
        // Transform local sample point to world space
        let world_point = position + rotation * *lp;

        let distance = sdf.evaluate(world_point);

        // Penetration is how far inside the surface the point is.
        // If distance < threshold, we have a contact.
        // Penetration = threshold - distance (positive when penetrating).
        if distance < contact_threshold {
            let normal = sdf.gradient(world_point);
            contacts.push(ContactPoint {
                position: world_point,
                normal,
                penetration: contact_threshold - distance,
            });
        }
    }

    // Sort by penetration depth, deepest first
    contacts.sort_by(|a, b| {
        b.penetration
            .partial_cmp(&a.penetration)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    contacts
}

// ---------------------------------------------------------------------------
// Contact application
// ---------------------------------------------------------------------------

/// Apply SDF contacts to a rigid body as position corrections and velocity damping.
///
/// For each contact:
/// 1. Push the body out along the contact normal by the penetration depth.
/// 2. Cancel velocity into the surface (project out the normal component).
/// 3. Apply friction to the tangential velocity.
///
/// This is a simplified penalty/projection method — it does not solve a full
/// LCP like Rapier's internal solver, but is sufficient for SDF world collision.
pub fn apply_sdf_contacts(
    world: &mut PhysicsWorld,
    handle: RigidBodyHandle,
    contacts: &[ContactPoint],
    friction: f32,
) {
    if contacts.is_empty() {
        return;
    }

    let body = match world.rigid_body_set.get_mut(handle) {
        Some(b) => b,
        None => return,
    };

    // Only apply to dynamic bodies
    if !body.is_dynamic() {
        return;
    }

    for contact in contacts {
        let n = contact.normal;
        if n.length_squared() < 0.001 {
            continue;
        }

        // 1. Position correction: push body out of the SDF surface
        let correction = n * contact.penetration;
        let current_translation = crate::rapier_world::rapier_to_glam(body.translation());
        let new_translation = current_translation + correction;
        body.set_translation(
            crate::rapier_world::glam_to_rapier(new_translation),
            true,
        );

        // 2. Velocity correction: cancel velocity into the surface
        let vel = crate::rapier_world::rapier_to_glam(body.linvel());
        let vel_normal = vel.dot(n);
        if vel_normal < 0.0 {
            // Moving into the surface — cancel normal component
            let vel_corrected = vel - n * vel_normal;
            // 3. Apply friction to tangential component
            let tangent = vel_corrected;
            let friction_vel = tangent * (1.0 - friction.clamp(0.0, 1.0));
            body.set_linvel(
                crate::rapier_world::glam_to_rapier(friction_vel),
                true,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test SDF implementations
// ---------------------------------------------------------------------------

/// An infinite ground plane at a given Y height.
///
/// Useful for testing and simple scenes. The SDF is simply `pos.y - height`.
pub struct GroundPlaneSdf {
    /// Y coordinate of the ground plane.
    pub height: f32,
}

impl SdfQueryable for GroundPlaneSdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        pos.y - self.height
    }

    fn gradient(&self, _pos: Vec3) -> Vec3 {
        Vec3::Y
    }
}

/// A sphere SDF for testing.
pub struct SphereSdf {
    /// Center of the sphere.
    pub center: Vec3,
    /// Radius of the sphere.
    pub radius: f32,
}

impl SdfQueryable for SphereSdf {
    fn evaluate(&self, pos: Vec3) -> f32 {
        (pos - self.center).length() - self.radius
    }

    fn gradient(&self, pos: Vec3) -> Vec3 {
        (pos - self.center).normalize_or_zero()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_plane_contact() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let shape = CollisionShape::Sphere { radius: 0.5 };

        // Sphere centered at y=2.0, well above ground → no contacts
        let contacts = generate_sdf_contacts(
            &shape,
            Vec3::new(0.0, 2.0, 0.0),
            Quat::IDENTITY,
            &ground,
            0.02,
        );
        assert!(
            contacts.is_empty(),
            "sphere above ground should have no contacts, got {}",
            contacts.len()
        );

        // Sphere centered at y=0.3, bottom sample at y=-0.2 → penetrating
        let contacts = generate_sdf_contacts(
            &shape,
            Vec3::new(0.0, 0.3, 0.0),
            Quat::IDENTITY,
            &ground,
            0.02,
        );
        assert!(
            !contacts.is_empty(),
            "sphere intersecting ground should have contacts"
        );

        // All contacts should have upward normal (Y)
        for c in &contacts {
            assert!(
                c.normal.y > 0.9,
                "ground contact normal should point up, got {:?}",
                c.normal
            );
        }
    }

    #[test]
    fn test_contact_penetration_depth() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let shape = CollisionShape::Sphere { radius: 0.5 };

        // Sphere at y=0.0 → bottom point at y=-0.5, penetration = 0.5 + threshold
        let threshold = 0.02;
        let contacts = generate_sdf_contacts(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            &ground,
            threshold,
        );
        assert!(!contacts.is_empty());

        // The deepest contact should be the bottom pole at y=-0.5
        let deepest = &contacts[0];
        // SDF at y=-0.5 is -0.5, penetration = threshold - (-0.5) = 0.52
        assert!(
            (deepest.penetration - 0.52).abs() < 0.05,
            "expected penetration ~0.52, got {}",
            deepest.penetration
        );
    }

    #[test]
    fn test_gradient_central_diff() {
        let sphere = SphereSdf {
            center: Vec3::ZERO,
            radius: 1.0,
        };

        // Gradient at (2, 0, 0) should point in +X direction
        let grad = gradient_central_diff(&sphere, Vec3::new(2.0, 0.0, 0.0));
        assert!(
            grad.x > 0.95,
            "gradient at +X should point in +X, got {:?}",
            grad
        );
        assert!(grad.y.abs() < 0.1);
        assert!(grad.z.abs() < 0.1);

        // Gradient at (0, -2, 0) should point in -Y direction
        let grad = gradient_central_diff(&sphere, Vec3::new(0.0, -2.0, 0.0));
        assert!(
            grad.y < -0.95,
            "gradient at -Y should point in -Y, got {:?}",
            grad
        );
    }

    #[test]
    fn test_sample_points_sphere() {
        let shape = CollisionShape::Sphere { radius: 1.0 };
        let points = shape.sample_points();
        assert_eq!(points.len(), 14, "sphere should have 14 sample points");

        // All points should be at distance 1.0 from origin
        for (i, p) in points.iter().enumerate() {
            let dist = p.length();
            assert!(
                (dist - 1.0).abs() < 0.01,
                "point {i} at {:?} has distance {dist}, expected 1.0",
                p
            );
        }
    }

    #[test]
    fn test_sample_points_box() {
        let shape = CollisionShape::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let points = shape.sample_points();
        assert_eq!(points.len(), 14, "box should have 14 sample points");

        // First 8 should be corners
        for p in &points[0..8] {
            assert!(
                (p.x.abs() - 1.0).abs() < 1e-6
                    && (p.y.abs() - 2.0).abs() < 1e-6
                    && (p.z.abs() - 3.0).abs() < 1e-6,
                "corner {:?} doesn't match half_extents",
                p
            );
        }

        // Last 6 should be face centers
        let face_centers = &points[8..14];
        // +X face center
        assert!((face_centers[0] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
        // -X face center
        assert!((face_centers[1] - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn test_sample_points_capsule() {
        let shape = CollisionShape::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        let points = shape.sample_points();
        assert_eq!(points.len(), 12, "capsule should have 12 sample points");

        // Top pole should be at (0, half_height + radius, 0)
        assert!(
            (points[0] - Vec3::new(0.0, 1.5, 0.0)).length() < 1e-6,
            "top pole: {:?}",
            points[0]
        );

        // Bottom pole should be at (0, -(half_height + radius), 0)
        assert!(
            (points[5] - Vec3::new(0.0, -1.5, 0.0)).length() < 1e-6,
            "bottom pole: {:?}",
            points[5]
        );
    }

    #[test]
    fn test_contacts_sorted_by_depth() {
        let ground = GroundPlaneSdf { height: 0.0 };
        let shape = CollisionShape::Sphere { radius: 1.0 };

        // Sphere centered at y=0 → multiple contact points at different depths
        let contacts = generate_sdf_contacts(
            &shape,
            Vec3::ZERO,
            Quat::IDENTITY,
            &ground,
            0.02,
        );
        assert!(contacts.len() >= 2, "should have multiple contacts");

        // Verify sorted by penetration (deepest first)
        for i in 1..contacts.len() {
            assert!(
                contacts[i - 1].penetration >= contacts[i].penetration,
                "contacts not sorted: [{}].pen={} > [{}].pen={}",
                i - 1,
                contacts[i - 1].penetration,
                i,
                contacts[i].penetration
            );
        }
    }

    #[test]
    fn test_apply_contacts_resolves_penetration() {
        let mut world = PhysicsWorld::new(crate::rapier_world::PhysicsConfig::default());

        // Create a dynamic body at the origin (sphere radius 0.5)
        let body = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 0.0, 0.0])
            .build();
        let handle = world.add_rigid_body(body);
        let collider = ColliderBuilder::ball(0.5).build();
        world.add_collider(collider, handle);

        // Simulate a contact pushing the body upward
        let contacts = vec![ContactPoint {
            position: Vec3::new(0.0, -0.5, 0.0),
            normal: Vec3::Y,
            penetration: 0.5,
        }];

        apply_sdf_contacts(&mut world, handle, &contacts, 0.5);

        // Body should have moved up
        let body = world.get_body(handle).unwrap();
        assert!(
            body.translation().y > 0.0,
            "body should have been pushed up: y={}",
            body.translation().y
        );
    }
}
