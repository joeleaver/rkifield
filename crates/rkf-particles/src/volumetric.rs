//! Volumetric particle density accumulation.
//!
//! Provides CPU-reference implementations of the density field that
//! volumetric particles contribute to the scene. The GPU shader helpers
//! in `particle_volumetric.wgsl` mirror these functions.

use glam::Vec3;
use half::f16;

use crate::binning::ParticleGrid3D;
use crate::particle::{flags, Particle};

/// Smoothstep interpolation (Hermite).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Result of accumulating volumetric particle contributions at a point.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricSample {
    /// Total accumulated density.
    pub density: f32,
    /// Total emission intensity.
    pub emission: f32,
    /// Weighted average scattering color (linear RGB).
    pub color: Vec3,
}

impl Default for VolumetricSample {
    fn default() -> Self {
        Self {
            density: 0.0,
            emission: 0.0,
            color: Vec3::ZERO,
        }
    }
}

/// Compute density and emission contribution of a single particle at a
/// world-space position.
///
/// Returns `(density, emission)`. Both are zero if the position is outside
/// the particle radius.
///
/// - Falloff: `1.0 - smoothstep(0.0, size, distance)`
/// - Age fade (if `FADE_OUT` flag set): `lifetime / max_lifetime`
/// - Emission: `falloff * color_emission.w`
pub fn particle_density_at(pos: Vec3, particle: &Particle) -> (f32, f32) {
    let center = Vec3::from(particle.position);
    let size = f16::from_bits(particle.size).to_f32();
    let dist = pos.distance(center);

    if dist > size {
        return (0.0, 0.0);
    }

    let falloff = 1.0 - smoothstep(0.0, size, dist);

    let age_fade = if particle.flags & flags::FADE_OUT != 0 {
        if particle.max_lifetime > 0.0 {
            (particle.lifetime / particle.max_lifetime).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        1.0
    };

    let density = falloff * age_fade;
    let emission_intensity = f16::from_bits(particle.color_emission[3]).to_f32();
    let emission = falloff * emission_intensity;

    (density, emission)
}

/// Accumulate volumetric particle contributions at a world position.
///
/// Looks up the cell containing `pos` and all 26 neighbors (3x3x3),
/// then sums density and emission from every particle in those cells.
/// Color is density-weighted.
pub fn accumulate_particles(
    pos: Vec3,
    grid: &ParticleGrid3D,
    particles: &[Particle],
) -> VolumetricSample {
    let cell = grid.world_to_cell(pos);
    let cx = cell[0] as i32;
    let cy = cell[1] as i32;
    let cz = cell[2] as i32;

    let mut total_density = 0.0f32;
    let mut total_emission = 0.0f32;
    let mut weighted_color = Vec3::ZERO;

    // Iterate 3x3x3 neighborhood.
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = cx + dx;
                let ny = cy + dy;
                let nz = cz + dz;

                if nx < 0
                    || ny < 0
                    || nz < 0
                    || nx >= grid.grid_dims[0] as i32
                    || ny >= grid.grid_dims[1] as i32
                    || nz >= grid.grid_dims[2] as i32
                {
                    continue;
                }

                let indices = grid.particles_in_cell(nx as u32, ny as u32, nz as u32);
                for &pidx in indices {
                    let p = &particles[pidx as usize];
                    let (density, emission) = particle_density_at(pos, p);
                    if density > 0.0 {
                        total_density += density;
                        total_emission += emission;
                        let r = f16::from_bits(p.color_emission[0]).to_f32();
                        let g = f16::from_bits(p.color_emission[1]).to_f32();
                        let b = f16::from_bits(p.color_emission[2]).to_f32();
                        weighted_color += Vec3::new(r, g, b) * density;
                    }
                }
            }
        }
    }

    let color = if total_density > 0.0 {
        weighted_color / total_density
    } else {
        Vec3::ZERO
    };

    VolumetricSample {
        density: total_density,
        emission: total_emission,
        color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::bin_particles;
    use crate::particle::{flags, Particle, RenderType};
    use half::f16;

    fn make_particle(pos: [f32; 3], size: f32, lifetime: f32, max_lifetime: f32, fade: bool) -> Particle {
        let mut p_flags = flags::ALIVE;
        if fade {
            p_flags |= flags::FADE_OUT;
        }
        Particle {
            position: pos,
            lifetime,
            velocity: [0.0; 3],
            max_lifetime,
            color_emission: [
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(0.5).to_bits(),
                f16::from_f32(0.2).to_bits(),
                f16::from_f32(3.0).to_bits(), // emission intensity
            ],
            size: f16::from_f32(size).to_bits(),
            render_type: RenderType::Volumetric as u8,
            flags: p_flags,
            material_id: 0,
            _pad: 0,
        }
    }

    #[test]
    fn test_density_at_center() {
        let p = make_particle([5.0, 5.0, 5.0], 1.0, 2.0, 2.0, false);
        let (density, _emission) = particle_density_at(Vec3::new(5.0, 5.0, 5.0), &p);
        // At center, distance=0, smoothstep(0,1,0)=0, falloff=1-0=1, no fade → density=1.0
        assert!((density - 1.0).abs() < 1e-5, "density at center should be 1.0, got {}", density);
    }

    #[test]
    fn test_density_beyond_radius() {
        let p = make_particle([0.0, 0.0, 0.0], 0.5, 1.0, 1.0, false);
        let (density, emission) = particle_density_at(Vec3::new(2.0, 0.0, 0.0), &p);
        assert_eq!(density, 0.0);
        assert_eq!(emission, 0.0);
    }

    #[test]
    fn test_density_falloff() {
        let p = make_particle([0.0, 0.0, 0.0], 1.0, 1.0, 1.0, false);
        let (d_near, _) = particle_density_at(Vec3::new(0.1, 0.0, 0.0), &p);
        let (d_mid, _) = particle_density_at(Vec3::new(0.5, 0.0, 0.0), &p);
        let (d_far, _) = particle_density_at(Vec3::new(0.9, 0.0, 0.0), &p);
        assert!(d_near > d_mid, "closer should be denser: {} > {}", d_near, d_mid);
        assert!(d_mid > d_far, "mid should be denser than far: {} > {}", d_mid, d_far);
        assert!(d_far > 0.0, "inside radius should be > 0");
    }

    #[test]
    fn test_density_age_fade() {
        // With FADE_OUT, density scales by lifetime/max_lifetime.
        let p_full = make_particle([0.0, 0.0, 0.0], 1.0, 2.0, 2.0, true);
        let p_half = make_particle([0.0, 0.0, 0.0], 1.0, 1.0, 2.0, true);

        let (d_full, _) = particle_density_at(Vec3::ZERO, &p_full);
        let (d_half, _) = particle_density_at(Vec3::ZERO, &p_half);

        // At center with fade: d_full = 1.0 * (2/2) = 1.0, d_half = 1.0 * (1/2) = 0.5
        assert!((d_full - 1.0).abs() < 1e-5, "full lifetime density should be 1.0, got {}", d_full);
        assert!((d_half - 0.5).abs() < 1e-5, "half lifetime density should be 0.5, got {}", d_half);
    }

    #[test]
    fn test_density_no_fade_flag() {
        // Without FADE_OUT, age doesn't affect density.
        let p_full = make_particle([0.0, 0.0, 0.0], 1.0, 2.0, 2.0, false);
        let p_half = make_particle([0.0, 0.0, 0.0], 1.0, 1.0, 2.0, false);

        let (d_full, _) = particle_density_at(Vec3::ZERO, &p_full);
        let (d_half, _) = particle_density_at(Vec3::ZERO, &p_half);

        assert!((d_full - d_half).abs() < 1e-5, "without FADE_OUT, age should not matter: {} vs {}", d_full, d_half);
    }

    #[test]
    fn test_accumulate_single_particle() {
        let p = make_particle([1.5, 1.5, 1.5], 0.5, 1.0, 1.0, false);
        let particles = [p];
        let grid = bin_particles(&particles, Vec3::ZERO, 1.0, [4, 4, 4]);

        let sample = accumulate_particles(Vec3::new(1.5, 1.5, 1.5), &grid, &particles);
        // At particle center, density should be 1.0
        assert!((sample.density - 1.0).abs() < 1e-4, "density at center: {}", sample.density);
        // Emission = falloff(1.0) * 3.0 = 3.0
        assert!((sample.emission - 3.0).abs() < 0.1, "emission at center: {}", sample.emission);
        // Color should be (1.0, 0.5, 0.2)
        assert!((sample.color.x - 1.0).abs() < 0.01);
        assert!((sample.color.y - 0.5).abs() < 0.01);
        assert!((sample.color.z - 0.2).abs() < 0.02);
    }

    #[test]
    fn test_accumulate_across_cells() {
        // Place particle at cell boundary — it is in cell (1,1,1), but its
        // radius should reach into cell (0,1,1) when queried from there.
        let p = make_particle([1.0, 1.5, 1.5], 0.5, 1.0, 1.0, false);
        let particles = [p];
        let grid = bin_particles(&particles, Vec3::ZERO, 1.0, [4, 4, 4]);

        // Query from cell (0,1,1), just inside particle radius.
        let query = Vec3::new(0.8, 1.5, 1.5); // distance to particle = 0.2 < 0.5
        let sample = accumulate_particles(query, &grid, &particles);
        assert!(sample.density > 0.0, "should find particle across cell boundary, got density {}", sample.density);
    }
}
