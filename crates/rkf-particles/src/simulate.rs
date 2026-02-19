//! CPU-side particle simulation (mirrors GPU compute logic).

use glam::Vec3;

use crate::particle::{flags, ParticleBuffer};

/// Gravity constant (m/s^2).
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Simulation parameters.
#[derive(Debug, Clone)]
pub struct SimulationParams {
    /// Time step in seconds.
    pub dt: f32,
    /// Wind velocity.
    pub wind: Vec3,
    /// Global gravity scale.
    pub gravity_scale: f32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            wind: Vec3::ZERO,
            gravity_scale: 1.0,
        }
    }
}

/// Step the particle simulation forward by one time step (CPU fallback).
///
/// For each alive particle:
/// 1. Apply gravity (if flagged)
/// 2. Apply wind and drag
/// 3. Integrate position
/// 4. Age particle, kill if expired
pub fn simulate_step(buffer: &mut ParticleBuffer, params: &SimulationParams) {
    let dt = params.dt;

    for i in 0..buffer.alive_count as usize {
        let p = &mut buffer.particles[i];
        if !p.is_alive() {
            continue;
        }

        // Gravity
        if p.flags & flags::GRAVITY != 0 {
            let vel = Vec3::from(p.velocity);
            let new_vel = vel + GRAVITY * params.gravity_scale * dt;
            p.velocity = new_vel.into();
        }

        // Wind + drag
        let vel = Vec3::from(p.velocity);
        let wind_force = (params.wind - vel) * 0.1; // simple drag towards wind
        let new_vel = vel + wind_force * dt;
        p.velocity = new_vel.into();

        // Integrate position
        let pos = Vec3::from(p.position);
        let new_pos = pos + Vec3::from(p.velocity) * dt;
        p.position = new_pos.into();

        // Age
        p.lifetime -= dt;
        if p.lifetime <= 0.0 {
            p.flags &= !flags::ALIVE;
        }
    }

    // Compact dead particles
    buffer.compact();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{flags, Particle, ParticleBuffer};

    #[test]
    fn simulate_applies_gravity() {
        let mut buf = ParticleBuffer::new(10);
        let p = Particle::new([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], 5.0);
        buf.spawn(p);

        let params = SimulationParams {
            dt: 1.0 / 60.0,
            ..Default::default()
        };
        simulate_step(&mut buf, &params);

        // Particle should have moved downward
        assert!(buf.particles[0].velocity[1] < 0.0, "gravity should pull y velocity negative");
        assert!(buf.particles[0].position[1] < 10.0, "particle should fall");
    }

    #[test]
    fn simulate_kills_expired() {
        let mut buf = ParticleBuffer::new(10);
        let p = Particle::new([0.0; 3], [0.0; 3], 0.01);
        buf.spawn(p);

        let params = SimulationParams {
            dt: 0.02,
            ..Default::default()
        };
        simulate_step(&mut buf, &params);

        // Particle should be dead and compacted away
        assert_eq!(buf.alive_count, 0);
    }

    #[test]
    fn simulate_compact_removes_dead() {
        let mut buf = ParticleBuffer::new(10);
        // Long-lived particle
        buf.spawn(Particle::new([1.0, 0.0, 0.0], [0.0; 3], 10.0));
        // Short-lived particle
        buf.spawn(Particle::new([2.0, 0.0, 0.0], [0.0; 3], 0.001));
        // Long-lived particle
        buf.spawn(Particle::new([3.0, 0.0, 0.0], [0.0; 3], 10.0));

        assert_eq!(buf.alive_count, 3);

        let params = SimulationParams {
            dt: 0.01,
            ..Default::default()
        };
        simulate_step(&mut buf, &params);

        assert_eq!(buf.alive_count, 2);
        // The remaining two should be the long-lived ones
        assert!((buf.particles[0].position[0] - 1.0).abs() < 0.5);
        assert!((buf.particles[1].position[0] - 3.0).abs() < 0.5);
    }

    #[test]
    fn simulate_position_integrates() {
        let mut buf = ParticleBuffer::new(10);
        // No gravity flag, just pure velocity integration
        let mut p = Particle::new([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], 5.0);
        p.flags = flags::ALIVE; // no gravity, no fade
        buf.spawn(p);

        let params = SimulationParams {
            dt: 0.1,
            wind: Vec3::ZERO,
            gravity_scale: 1.0,
        };
        simulate_step(&mut buf, &params);

        // With drag towards wind=0, velocity decreases slightly, but position should advance
        // vel after drag: v + (0 - v)*0.1*dt = 10 + (-10)*0.1*0.1 = 10 - 0.1 = 9.9
        // pos: 0 + 9.9 * 0.1 = 0.99
        let pos_x = buf.particles[0].position[0];
        assert!(pos_x > 0.9, "position should advance, got {}", pos_x);
        assert!(pos_x < 1.1, "position should be ~0.99, got {}", pos_x);
    }

    #[test]
    fn simulate_wind_applies_force() {
        let mut buf = ParticleBuffer::new(10);
        let mut p = Particle::new([0.0; 3], [0.0; 3], 5.0);
        p.flags = flags::ALIVE; // no gravity
        buf.spawn(p);

        let params = SimulationParams {
            dt: 0.1,
            wind: Vec3::new(10.0, 0.0, 0.0),
            gravity_scale: 1.0,
        };
        simulate_step(&mut buf, &params);

        // Wind should push particle in +x
        assert!(buf.particles[0].velocity[0] > 0.0, "wind should push velocity +x");
        assert!(buf.particles[0].position[0] > 0.0, "wind should push position +x");
    }

    #[test]
    fn simulate_no_gravity_when_flag_unset() {
        let mut buf = ParticleBuffer::new(10);
        let mut p = Particle::new([0.0, 10.0, 0.0], [0.0; 3], 5.0);
        p.flags = flags::ALIVE; // no GRAVITY flag
        buf.spawn(p);

        let params = SimulationParams {
            dt: 1.0 / 60.0,
            wind: Vec3::ZERO,
            gravity_scale: 1.0,
        };
        simulate_step(&mut buf, &params);

        // Without gravity, y velocity should stay near 0 (only tiny drag effect)
        let vy = buf.particles[0].velocity[1];
        assert!(vy.abs() < 0.01, "without gravity flag, vy should be near 0, got {}", vy);
    }
}
