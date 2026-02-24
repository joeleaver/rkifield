//! V2 object-centric SDF collision interface for particle simulation.
//!
//! In the v2 architecture, scene geometry is represented as a set of
//! per-object SDFs organised under a BVH. Instead of querying a global
//! chunk-backed distance field, collision queries walk the BVH and
//! evaluate each candidate object's SDF, returning the closest hit.
//!
//! This module defines the query result type, the query function signature,
//! and a CPU-side helper that applies a collision response (bounce + slide)
//! to a particle's position and velocity.

use glam::Vec3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a v2 particle–scene collision query.
#[derive(Debug, Clone, Copy)]
pub struct ParticleCollisionV2 {
    /// Signed distance to the nearest surface (negative = inside geometry).
    pub distance: f32,
    /// Outward surface normal at the nearest point (unit length).
    pub normal: Vec3,
    /// ID of the object that was hit.
    ///
    /// - `0` is reserved for terrain / world geometry.
    /// - `u32::MAX` indicates no hit (particle is far from all surfaces).
    pub object_id: u32,
}

impl ParticleCollisionV2 {
    /// Construct a "no hit" result (particle far from all surfaces).
    pub fn no_hit(distance: f32) -> Self {
        Self {
            distance,
            normal: Vec3::Y,
            object_id: u32::MAX,
        }
    }

    /// Returns `true` when the particle is at or below the surface.
    #[inline]
    pub fn is_penetrating(&self) -> bool {
        self.distance <= 0.0
    }
}

/// Query function signature for v2 object-centric SDF collision.
///
/// Receives a world-space position and returns the [`ParticleCollisionV2`]
/// describing the nearest surface.  The closure must be `Send + Sync` so
/// that it can be shared across threads during parallel particle simulation.
pub type CollisionQueryV2 = Box<dyn Fn(Vec3) -> ParticleCollisionV2 + Send + Sync>;

// ---------------------------------------------------------------------------
// Collision response
// ---------------------------------------------------------------------------

/// Apply a v2 collision response to a particle.
///
/// When the particle has penetrated a surface (`collision.distance <= 0`):
/// 1. **Position correction** — push the particle out along the surface
///    normal by the penetration depth so it sits exactly on the surface.
/// 2. **Velocity response** — decompose velocity into normal and tangential
///    components, then:
///    - Negate and scale the normal component by `restitution` (bounce).
///    - Scale the tangential component by `(1 − friction)` (slide damping).
///
/// When the particle is outside the surface (`collision.distance > 0`) no
/// correction is applied and the inputs are returned unchanged.
///
/// # Returns
/// `(corrected_position, corrected_velocity)`
pub fn apply_collision_v2(
    position: Vec3,
    velocity: Vec3,
    collision: &ParticleCollisionV2,
    restitution: f32,
    friction: f32,
) -> (Vec3, Vec3) {
    // Nothing to do if particle is outside the surface.
    if collision.distance > 0.0 {
        return (position, velocity);
    }

    let n = collision.normal;

    // Push out of the surface.
    let corrected_pos = position - n * collision.distance; // distance is ≤ 0, so this adds |distance|*n

    // Decompose velocity.
    let v_normal = n * velocity.dot(n);
    let v_tangent = velocity - v_normal;

    // Bounce normal component, damp tangential component.
    let corrected_vel = -v_normal * restitution + v_tangent * (1.0 - friction);

    (corrected_pos, corrected_vel)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a floor collision at y=0 (normal pointing up).
    fn floor_collision(y_pos: f32) -> ParticleCollisionV2 {
        ParticleCollisionV2 {
            distance: y_pos, // positive above floor, negative below
            normal: Vec3::Y,
            object_id: 0,
        }
    }

    // -----------------------------------------------------------------------
    // collision_v2_no_hit
    // -----------------------------------------------------------------------
    /// A particle far above the floor should experience no correction.
    #[test]
    fn collision_v2_no_hit() {
        let pos = Vec3::new(0.0, 5.0, 0.0);
        let vel = Vec3::new(1.0, -2.0, 0.0);

        let collision = floor_collision(5.0); // 5 m above floor → no hit

        let (new_pos, new_vel) = apply_collision_v2(pos, vel, &collision, 0.5, 0.3);

        assert_eq!(new_pos, pos, "position must not change when far from surface");
        assert_eq!(new_vel, vel, "velocity must not change when far from surface");

        // object_id should survive untouched through the query result.
        assert_eq!(collision.object_id, 0);
    }

    // -----------------------------------------------------------------------
    // collision_v2_floor_bounce
    // -----------------------------------------------------------------------
    /// A particle that has penetrated the floor should bounce back.
    #[test]
    fn collision_v2_floor_bounce() {
        // Particle is 0.1 m below the floor (distance = -0.1).
        let pos = Vec3::new(0.0, -0.1, 0.0);
        let vel = Vec3::new(0.0, -3.0, 0.0); // moving downward

        let collision = ParticleCollisionV2 {
            distance: -0.1,
            normal: Vec3::Y,
            object_id: 0,
        };

        let restitution = 0.6;
        let friction = 0.0; // no friction so tangential is unaffected (zero here)

        let (new_pos, new_vel) = apply_collision_v2(pos, vel, &collision, restitution, friction);

        // Position should be pushed back above the floor (y ≥ 0).
        assert!(
            new_pos.y >= 0.0,
            "particle should be pushed above floor, got y={}",
            new_pos.y
        );

        // The y velocity should be positive (bouncing upward).
        assert!(
            new_vel.y > 0.0,
            "particle should bounce upward, got vy={}",
            new_vel.y
        );

        // Magnitude should be scaled by restitution.
        let expected_vy = 3.0 * restitution;
        assert!(
            (new_vel.y - expected_vy).abs() < 1e-5,
            "bounce velocity should be {} but got {}",
            expected_vy,
            new_vel.y
        );
    }

    // -----------------------------------------------------------------------
    // collision_v2_surface_slide
    // -----------------------------------------------------------------------
    /// A particle sliding along the floor should have its normal velocity
    /// cancelled and its tangential velocity damped by friction.
    #[test]
    fn collision_v2_surface_slide() {
        // Particle exactly on the surface (distance = 0) — treated as penetrating.
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let vel = Vec3::new(5.0, -0.5, 0.0); // mostly horizontal, slight sink

        let collision = ParticleCollisionV2 {
            distance: 0.0,
            normal: Vec3::Y,
            object_id: 0,
        };

        let restitution = 0.0; // fully inelastic (no bounce)
        let friction = 0.2;    // 20 % friction damping

        let (new_pos, new_vel) = apply_collision_v2(pos, vel, &collision, restitution, friction);

        // Position should not change (distance == 0, no penetration to correct).
        assert_eq!(new_pos, pos);

        // Normal (y) velocity: -0.5 * restitution = 0.0 → cancelled.
        assert!(
            new_vel.y.abs() < 1e-5,
            "normal velocity should be cancelled, got vy={}",
            new_vel.y
        );

        // Tangential (x) velocity: 5.0 * (1 - friction) = 4.0.
        let expected_vx = 5.0 * (1.0 - friction);
        assert!(
            (new_vel.x - expected_vx).abs() < 1e-5,
            "tangential velocity should be {} but got {}",
            expected_vx,
            new_vel.x
        );
    }

    // -----------------------------------------------------------------------
    // collision_v2_object_identity
    // -----------------------------------------------------------------------
    /// The object_id stored in the collision result should be preserved
    /// exactly — the response function does not overwrite it.
    #[test]
    fn collision_v2_object_identity() {
        let arbitrary_id: u32 = 42;

        let collision = ParticleCollisionV2 {
            distance: -0.05,
            normal: Vec3::Y,
            object_id: arbitrary_id,
        };

        let pos = Vec3::new(0.0, -0.05, 0.0);
        let vel = Vec3::new(0.0, -1.0, 0.0);

        // apply_collision_v2 does not touch object_id; verify it is preserved.
        assert_eq!(
            collision.object_id, arbitrary_id,
            "object_id must be preserved"
        );

        // Also verify no_hit carries u32::MAX.
        let no_hit = ParticleCollisionV2::no_hit(10.0);
        assert_eq!(no_hit.object_id, u32::MAX, "no_hit object_id must be u32::MAX");

        // Sanity: collision response still works for this object.
        let (new_pos, _) = apply_collision_v2(pos, vel, &collision, 0.5, 0.1);
        assert!(new_pos.y >= 0.0, "particle should be above floor after correction");
    }
}
