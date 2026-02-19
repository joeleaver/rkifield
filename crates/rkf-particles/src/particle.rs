//! GPU particle data types and buffer management.

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Render type tags for particles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RenderType {
    /// Volumetric density splat (glowing effects).
    Volumetric = 0,
    /// SDF micro-object (solid, fully shaded).
    SdfMicro = 1,
    /// Screen-space overlay (rain, snow).
    ScreenSpace = 2,
}

/// Particle flags (bitfield).
pub mod flags {
    /// Apply gravity to this particle.
    pub const GRAVITY: u8 = 1;
    /// Check SDF collision for this particle.
    pub const COLLISION: u8 = 2;
    /// Fade opacity with age.
    pub const FADE_OUT: u8 = 4;
    /// Particle is alive.
    pub const ALIVE: u8 = 8;
}

/// GPU particle -- 48 bytes, matches WGSL layout.
///
/// Layout:
/// - position `[f32; 3]` + lifetime `f32` = 16 bytes
/// - velocity `[f32; 3]` + max_lifetime `f32` = 16 bytes
/// - color_emission `[f16; 4]` = 8 bytes
/// - size `f16` + render_type `u8` + flags `u8` + material_id `u16` + _pad `u16` = 8 bytes
///
/// Total: 48 bytes
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Particle {
    /// World-space position.
    pub position: [f32; 3],
    /// Remaining lifetime in seconds.
    pub lifetime: f32,
    /// World-space velocity.
    pub velocity: [f32; 3],
    /// Initial lifetime (for age ratio).
    pub max_lifetime: f32,
    /// RGBA color + emission intensity (f16x4).
    pub color_emission: [u16; 4],
    /// Particle radius in world units (f16).
    pub size: u16,
    /// Render backend type.
    pub render_type: u8,
    /// Bitfield flags.
    pub flags: u8,
    /// Material ID (for SDF micro-objects).
    pub material_id: u16,
    /// Padding to 48 bytes.
    pub _pad: u16,
}

unsafe impl Zeroable for Particle {}
unsafe impl Pod for Particle {}

impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            lifetime: 0.0,
            velocity: [0.0; 3],
            max_lifetime: 1.0,
            color_emission: [
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(0.0).to_bits(),
            ],
            size: f16::from_f32(0.05).to_bits(),
            render_type: RenderType::Volumetric as u8,
            flags: flags::ALIVE | flags::GRAVITY | flags::FADE_OUT,
            material_id: 0,
            _pad: 0,
        }
    }
}

impl Particle {
    /// Create a new particle at position with velocity.
    pub fn new(position: [f32; 3], velocity: [f32; 3], lifetime: f32) -> Self {
        Self {
            position,
            velocity,
            lifetime,
            max_lifetime: lifetime,
            ..Default::default()
        }
    }

    /// Whether this particle is alive.
    pub fn is_alive(&self) -> bool {
        self.flags & flags::ALIVE != 0
    }

    /// Age ratio `[0, 1]` where 0 = just born, 1 = about to die.
    pub fn age_ratio(&self) -> f32 {
        if self.max_lifetime > 0.0 {
            1.0 - (self.lifetime / self.max_lifetime).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Set color (linear RGB) and emission intensity.
    pub fn set_color_emission(&mut self, r: f32, g: f32, b: f32, emission: f32) {
        self.color_emission = [
            f16::from_f32(r).to_bits(),
            f16::from_f32(g).to_bits(),
            f16::from_f32(b).to_bits(),
            f16::from_f32(emission).to_bits(),
        ];
    }

    /// Set particle size (radius in world units).
    pub fn set_size(&mut self, radius: f32) {
        self.size = f16::from_f32(radius).to_bits();
    }
}

/// CPU-side particle buffer for staging before GPU upload.
#[derive(Debug)]
pub struct ParticleBuffer {
    /// All particle slots.
    pub particles: Vec<Particle>,
    /// Number of alive particles (always <= particles.len()).
    pub alive_count: u32,
    /// Maximum capacity.
    pub capacity: u32,
}

impl ParticleBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(capacity: u32) -> Self {
        Self {
            particles: vec![Particle::default(); capacity as usize],
            alive_count: 0,
            capacity,
        }
    }

    /// Spawn a particle, returning its index or None if full.
    pub fn spawn(&mut self, particle: Particle) -> Option<u32> {
        if self.alive_count >= self.capacity {
            return None;
        }
        let idx = self.alive_count;
        self.particles[idx as usize] = particle;
        self.alive_count += 1;
        Some(idx)
    }

    /// Compact: remove dead particles by swapping with last alive.
    pub fn compact(&mut self) {
        let mut write = 0usize;
        for read in 0..self.alive_count as usize {
            if self.particles[read].is_alive() {
                if write != read {
                    self.particles[write] = self.particles[read];
                }
                write += 1;
            }
        }
        self.alive_count = write as u32;
    }

    /// Get alive particles slice.
    pub fn alive_slice(&self) -> &[Particle] {
        &self.particles[..self.alive_count as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particle_size_48_bytes() {
        assert_eq!(std::mem::size_of::<Particle>(), 48);
    }

    #[test]
    fn particle_default_is_alive() {
        let p = Particle::default();
        assert!(p.is_alive());
        assert!(p.flags & flags::GRAVITY != 0);
        assert!(p.flags & flags::FADE_OUT != 0);
    }

    #[test]
    fn particle_new_sets_lifetime() {
        let p = Particle::new([1.0, 2.0, 3.0], [0.0; 3], 5.0);
        assert_eq!(p.lifetime, 5.0);
        assert_eq!(p.max_lifetime, 5.0);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn particle_age_ratio_fresh() {
        let p = Particle::new([0.0; 3], [0.0; 3], 2.0);
        assert!((p.age_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn particle_age_ratio_half() {
        let mut p = Particle::new([0.0; 3], [0.0; 3], 2.0);
        p.lifetime = 1.0;
        assert!((p.age_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn particle_age_ratio_dead() {
        let mut p = Particle::new([0.0; 3], [0.0; 3], 2.0);
        p.lifetime = 0.0;
        assert!((p.age_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn particle_age_ratio_zero_max_lifetime() {
        let mut p = Particle::default();
        p.max_lifetime = 0.0;
        assert!((p.age_ratio() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn particle_set_color_emission() {
        let mut p = Particle::default();
        p.set_color_emission(0.5, 0.25, 0.75, 2.0);
        let r = f16::from_bits(p.color_emission[0]).to_f32();
        let g = f16::from_bits(p.color_emission[1]).to_f32();
        let b = f16::from_bits(p.color_emission[2]).to_f32();
        let e = f16::from_bits(p.color_emission[3]).to_f32();
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.25).abs() < 0.01);
        assert!((b - 0.75).abs() < 0.01);
        assert!((e - 2.0).abs() < 0.01);
    }

    #[test]
    fn particle_set_size() {
        let mut p = Particle::default();
        p.set_size(0.1);
        let sz = f16::from_bits(p.size).to_f32();
        assert!((sz - 0.1).abs() < 0.001);
    }

    #[test]
    fn buffer_spawn_and_compact() {
        let mut buf = ParticleBuffer::new(10);
        let p1 = Particle::new([1.0, 0.0, 0.0], [0.0; 3], 1.0);
        let mut p2 = Particle::new([2.0, 0.0, 0.0], [0.0; 3], 1.0);
        p2.flags &= !flags::ALIVE; // dead on arrival
        let p3 = Particle::new([3.0, 0.0, 0.0], [0.0; 3], 1.0);

        assert_eq!(buf.spawn(p1), Some(0));
        assert_eq!(buf.spawn(p2), Some(1));
        assert_eq!(buf.spawn(p3), Some(2));
        assert_eq!(buf.alive_count, 3);

        buf.compact();
        assert_eq!(buf.alive_count, 2);
        assert_eq!(buf.particles[0].position[0], 1.0);
        assert_eq!(buf.particles[1].position[0], 3.0);
    }

    #[test]
    fn buffer_spawn_full() {
        let mut buf = ParticleBuffer::new(2);
        let p = Particle::new([0.0; 3], [0.0; 3], 1.0);
        assert_eq!(buf.spawn(p), Some(0));
        assert_eq!(buf.spawn(p), Some(1));
        assert_eq!(buf.spawn(p), None);
    }

    #[test]
    fn buffer_alive_slice() {
        let mut buf = ParticleBuffer::new(10);
        let p = Particle::new([0.0; 3], [0.0; 3], 1.0);
        buf.spawn(p);
        buf.spawn(p);
        assert_eq!(buf.alive_slice().len(), 2);
    }
}
