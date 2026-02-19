//! Particle emitter definitions.

use glam::Vec3;

/// Range for randomized per-particle properties.
#[derive(Debug, Clone, Copy)]
pub struct RangeF32 {
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
}

impl RangeF32 {
    /// Constant range (min == max).
    pub fn constant(v: f32) -> Self {
        Self { min: v, max: v }
    }

    /// Sample a value from the range using t in `[0, 1]`.
    pub fn sample(&self, t: f32) -> f32 {
        self.min + (self.max - self.min) * t
    }
}

/// Color range for per-particle color randomization.
#[derive(Debug, Clone, Copy)]
pub struct ColorRange {
    /// Minimum color (linear RGB).
    pub min: [f32; 3],
    /// Maximum color (linear RGB).
    pub max: [f32; 3],
}

impl ColorRange {
    /// Constant color.
    pub fn constant(r: f32, g: f32, b: f32) -> Self {
        Self {
            min: [r, g, b],
            max: [r, g, b],
        }
    }

    /// Sample color using t in `[0, 1]`.
    pub fn sample(&self, t: f32) -> [f32; 3] {
        [
            self.min[0] + (self.max[0] - self.min[0]) * t,
            self.min[1] + (self.max[1] - self.min[1]) * t,
            self.min[2] + (self.max[2] - self.min[2]) * t,
        ]
    }
}

/// Emitter shape for spawn position generation.
#[derive(Debug, Clone, Copy)]
pub enum EmitterShape {
    /// Spawn at a single point.
    Point,
    /// Spawn within a sphere of given radius.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Spawn within an axis-aligned box.
    Box {
        /// Half-extents of the box.
        half_extents: Vec3,
    },
    /// Spawn in a cone direction with half-angle (radians).
    Cone {
        /// Cone direction (normalized).
        direction: Vec3,
        /// Half-angle in radians.
        half_angle: f32,
    },
}

/// Particle render type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleRenderType {
    /// Volumetric density splat.
    Volumetric,
    /// SDF micro-object.
    SdfMicro,
    /// Screen-space overlay.
    ScreenSpace,
}

/// A particle emitter that spawns particles with randomized properties.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    /// World-space position of the emitter.
    pub position: Vec3,
    /// Emitter shape.
    pub shape: EmitterShape,
    /// Render type for spawned particles.
    pub render_type: ParticleRenderType,
    /// Spawn rate (particles per second, 0 for burst-only).
    pub rate: f32,
    /// Burst count (one-shot spawn, 0 for continuous-only).
    pub burst_count: u32,
    /// Whether burst has been fired.
    pub burst_fired: bool,
    /// Accumulator for continuous emission.
    pub spawn_accumulator: f32,

    /// Per-particle lifetime range.
    pub lifetime: RangeF32,
    /// Initial speed range.
    pub speed: RangeF32,
    /// Particle size range.
    pub size: RangeF32,
    /// Color range.
    pub color: ColorRange,
    /// Emission intensity range.
    pub emission: RangeF32,

    /// Gravity scale (1.0 = normal gravity).
    pub gravity_scale: f32,
    /// Drag coefficient.
    pub drag: f32,
    /// Whether particles collide with SDF world.
    pub collision: bool,
    /// Material ID for SDF micro-objects.
    pub material_id: u16,
    /// Whether emitter is active.
    pub active: bool,
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            shape: EmitterShape::Point,
            render_type: ParticleRenderType::Volumetric,
            rate: 100.0,
            burst_count: 0,
            burst_fired: false,
            spawn_accumulator: 0.0,
            lifetime: RangeF32 {
                min: 0.5,
                max: 2.0,
            },
            speed: RangeF32 {
                min: 1.0,
                max: 3.0,
            },
            size: RangeF32 {
                min: 0.02,
                max: 0.05,
            },
            color: ColorRange::constant(1.0, 1.0, 1.0),
            emission: RangeF32 {
                min: 0.0,
                max: 1.0,
            },
            gravity_scale: 1.0,
            drag: 0.1,
            collision: false,
            material_id: 0,
            active: true,
        }
    }
}

impl ParticleEmitter {
    /// Calculate how many particles to spawn this frame.
    pub fn particles_to_spawn(&mut self, dt: f32) -> u32 {
        if !self.active {
            return 0;
        }

        let mut count = 0u32;

        // Burst
        if self.burst_count > 0 && !self.burst_fired {
            count += self.burst_count;
            self.burst_fired = true;
        }

        // Continuous
        if self.rate > 0.0 {
            self.spawn_accumulator += self.rate * dt;
            let spawns = self.spawn_accumulator as u32;
            self.spawn_accumulator -= spawns as f32;
            count += spawns;
        }

        count
    }

    /// Generate a spawn position based on emitter shape using t values `[0,1]`.
    pub fn spawn_position(&self, t1: f32, t2: f32, t3: f32) -> Vec3 {
        match self.shape {
            EmitterShape::Point => self.position,
            EmitterShape::Sphere { radius } => {
                // Uniform random in sphere volume, scaled by t3
                let theta = t1 * std::f32::consts::TAU;
                let phi = (t2 * 2.0 - 1.0).acos();
                let r = radius * t3.cbrt();
                let sp = phi.sin();
                self.position
                    + Vec3::new(r * sp * theta.cos(), r * sp * theta.sin(), r * phi.cos())
            }
            EmitterShape::Box { half_extents } => {
                self.position
                    + Vec3::new(
                        (t1 * 2.0 - 1.0) * half_extents.x,
                        (t2 * 2.0 - 1.0) * half_extents.y,
                        (t3 * 2.0 - 1.0) * half_extents.z,
                    )
            }
            EmitterShape::Cone {
                direction,
                half_angle,
            } => {
                let theta = t1 * std::f32::consts::TAU;
                let cone_angle = t2 * half_angle;
                let ca = cone_angle.cos();
                let sa = cone_angle.sin();
                let dir = direction.normalize_or_zero();
                // Build orthonormal basis
                let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let right = dir.cross(up).normalize();
                let fwd = right.cross(dir).normalize();
                let spawn_dir = dir * ca + right * sa * theta.cos() + fwd * sa * theta.sin();
                self.position + spawn_dir * 0.01 // tiny offset in spawn direction
            }
        }
    }

    /// Generate a spawn velocity based on emitter shape and speed range.
    pub fn spawn_velocity(&self, speed_t: f32, dir_t1: f32, dir_t2: f32) -> Vec3 {
        let speed = self.speed.sample(speed_t);
        match self.shape {
            EmitterShape::Point | EmitterShape::Sphere { .. } | EmitterShape::Box { .. } => {
                // Random direction on unit sphere
                let theta = dir_t1 * std::f32::consts::TAU;
                let phi = (dir_t2 * 2.0 - 1.0).acos();
                let sp = phi.sin();
                Vec3::new(sp * theta.cos(), sp * theta.sin(), phi.cos()) * speed
            }
            EmitterShape::Cone {
                direction,
                half_angle,
            } => {
                let theta = dir_t1 * std::f32::consts::TAU;
                let cone_angle = dir_t2 * half_angle;
                let ca = cone_angle.cos();
                let sa = cone_angle.sin();
                let dir = direction.normalize_or_zero();
                let up = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let right = dir.cross(up).normalize();
                let fwd = right.cross(dir).normalize();
                (dir * ca + right * sa * theta.cos() + fwd * sa * theta.sin()) * speed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_default_values() {
        let e = ParticleEmitter::default();
        assert_eq!(e.position, Vec3::ZERO);
        assert_eq!(e.rate, 100.0);
        assert!(e.active);
        assert!(!e.burst_fired);
        assert_eq!(e.burst_count, 0);
        assert_eq!(e.gravity_scale, 1.0);
    }

    #[test]
    fn range_constant() {
        let r = RangeF32::constant(5.0);
        assert_eq!(r.sample(0.0), 5.0);
        assert_eq!(r.sample(0.5), 5.0);
        assert_eq!(r.sample(1.0), 5.0);
    }

    #[test]
    fn range_sample_linear() {
        let r = RangeF32 { min: 1.0, max: 3.0 };
        assert!((r.sample(0.0) - 1.0).abs() < 1e-6);
        assert!((r.sample(0.5) - 2.0).abs() < 1e-6);
        assert!((r.sample(1.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn color_range_constant() {
        let c = ColorRange::constant(0.5, 0.3, 0.8);
        let s = c.sample(0.5);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.3).abs() < 1e-6);
        assert!((s[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn color_range_interpolation() {
        let c = ColorRange {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let s = c.sample(0.5);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.5).abs() < 1e-6);
        assert!((s[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn emitter_burst_fires_once() {
        let mut e = ParticleEmitter {
            burst_count: 10,
            rate: 0.0,
            ..Default::default()
        };
        assert_eq!(e.particles_to_spawn(1.0 / 60.0), 10);
        assert_eq!(e.particles_to_spawn(1.0 / 60.0), 0);
        assert!(e.burst_fired);
    }

    #[test]
    fn emitter_continuous_rate() {
        let mut e = ParticleEmitter {
            rate: 60.0,
            burst_count: 0,
            ..Default::default()
        };
        let count = e.particles_to_spawn(1.0 / 60.0);
        assert_eq!(count, 1);
    }

    #[test]
    fn emitter_inactive_spawns_zero() {
        let mut e = ParticleEmitter {
            rate: 1000.0,
            active: false,
            ..Default::default()
        };
        assert_eq!(e.particles_to_spawn(1.0), 0);
    }

    #[test]
    fn spawn_position_point() {
        let e = ParticleEmitter {
            position: Vec3::new(5.0, 10.0, 15.0),
            shape: EmitterShape::Point,
            ..Default::default()
        };
        let pos = e.spawn_position(0.5, 0.5, 0.5);
        assert_eq!(pos, Vec3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn spawn_position_box() {
        let e = ParticleEmitter {
            position: Vec3::ZERO,
            shape: EmitterShape::Box {
                half_extents: Vec3::new(1.0, 1.0, 1.0),
            },
            ..Default::default()
        };
        // t=0.5 maps to center (0.5*2-1 = 0)
        let pos = e.spawn_position(0.5, 0.5, 0.5);
        assert!((pos.x).abs() < 1e-6);
        assert!((pos.y).abs() < 1e-6);
        assert!((pos.z).abs() < 1e-6);
    }

    #[test]
    fn spawn_position_sphere_within_radius() {
        let e = ParticleEmitter {
            position: Vec3::ZERO,
            shape: EmitterShape::Sphere { radius: 2.0 },
            ..Default::default()
        };
        // Any t values should produce position within radius
        for i in 0..10 {
            let t = i as f32 / 10.0;
            let pos = e.spawn_position(t, t, t);
            assert!(pos.length() <= 2.0 + 1e-5);
        }
    }

    #[test]
    fn spawn_velocity_nonzero() {
        let e = ParticleEmitter::default();
        let vel = e.spawn_velocity(0.5, 0.3, 0.7);
        assert!(vel.length() > 0.0);
    }
}
