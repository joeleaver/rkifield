//! Spatial binning of particles into a 3D grid for efficient neighbor lookup.
//!
//! Used by volumetric particle rendering to quickly find nearby particles
//! at any world position. The grid is rebuilt each frame from the alive
//! particle slice.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::particle::{Particle, RenderType};

/// A 3D grid that bins particles into cells for spatial lookup.
///
/// Built via [`bin_particles()`] each frame. Cell contents are stored as
/// a prefix-sum offset array + a sorted index array, matching the GPU
/// layout for zero-copy upload.
#[derive(Debug, Clone)]
pub struct ParticleGrid3D {
    /// World-space origin of the grid (minimum corner).
    pub origin: Vec3,
    /// Side length of each cubic cell.
    pub cell_size: f32,
    /// Grid dimensions (x, y, z).
    pub grid_dims: [u32; 3],
    /// Prefix-sum offsets per cell. Length = total_cells + 1.
    /// `cell_offsets[i]..cell_offsets[i+1]` is the range in `particle_indices`.
    pub cell_offsets: Vec<u32>,
    /// Particle indices sorted by cell. Length = number of binned particles.
    pub particle_indices: Vec<u32>,
}

/// GPU-friendly grid parameters for uniform upload.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuParticleGrid {
    /// Total number of cells.
    pub cell_count: u32,
    /// Cell side length.
    pub cell_size: f32,
    /// Grid origin (world-space minimum corner).
    pub origin: [f32; 3],
    /// Padding after origin for 16-byte alignment.
    pub _pad0: f32,
    /// Grid dimensions.
    pub dims: [u32; 3],
    /// Padding for 16-byte alignment.
    pub _pad1: u32,
}

unsafe impl Zeroable for GpuParticleGrid {}
unsafe impl Pod for GpuParticleGrid {}

impl ParticleGrid3D {
    /// Total number of cells in the grid.
    pub fn total_cells(&self) -> u32 {
        self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
    }

    /// Convert a world-space position to cell coordinates, clamped to grid bounds.
    pub fn world_to_cell(&self, pos: Vec3) -> [u32; 3] {
        let rel = pos - self.origin;
        let inv = 1.0 / self.cell_size;
        let cx = (rel.x * inv).floor().max(0.0).min((self.grid_dims[0] - 1) as f32) as u32;
        let cy = (rel.y * inv).floor().max(0.0).min((self.grid_dims[1] - 1) as f32) as u32;
        let cz = (rel.z * inv).floor().max(0.0).min((self.grid_dims[2] - 1) as f32) as u32;
        [cx, cy, cz]
    }

    /// Flat index from 3D cell coordinates.
    fn cell_index(&self, cx: u32, cy: u32, cz: u32) -> u32 {
        cx + cy * self.grid_dims[0] + cz * self.grid_dims[0] * self.grid_dims[1]
    }

    /// Return the particle indices stored in the given cell.
    pub fn particles_in_cell(&self, cx: u32, cy: u32, cz: u32) -> &[u32] {
        if cx >= self.grid_dims[0] || cy >= self.grid_dims[1] || cz >= self.grid_dims[2] {
            return &[];
        }
        let idx = self.cell_index(cx, cy, cz) as usize;
        let start = self.cell_offsets[idx] as usize;
        let end = self.cell_offsets[idx + 1] as usize;
        &self.particle_indices[start..end]
    }

    /// Produce GPU-friendly flat buffers for upload.
    ///
    /// Returns `(cell_offsets, particle_indices, params)`.
    pub fn to_gpu_buffers(&self) -> (Vec<u32>, Vec<u32>, GpuParticleGrid) {
        let params = GpuParticleGrid {
            cell_count: self.total_cells(),
            cell_size: self.cell_size,
            origin: self.origin.into(),
            _pad0: 0.0,
            dims: self.grid_dims,
            _pad1: 0,
        };
        (self.cell_offsets.clone(), self.particle_indices.clone(), params)
    }
}

/// Bin alive particles into a 3D grid.
///
/// Only particles with `RenderType::Volumetric` (render_type == 0) are binned.
/// The input slice should be the alive portion of a [`ParticleBuffer`].
///
/// Algorithm:
/// 1. Count particles per cell.
/// 2. Prefix-sum the counts to get cell offsets.
/// 3. Scatter particle indices into the sorted array.
pub fn bin_particles(
    particles: &[Particle],
    origin: Vec3,
    cell_size: f32,
    grid_dims: [u32; 3],
) -> ParticleGrid3D {
    let total_cells = (grid_dims[0] * grid_dims[1] * grid_dims[2]) as usize;
    let inv = 1.0 / cell_size;
    let stride_y = grid_dims[0];
    let stride_z = grid_dims[0] * grid_dims[1];

    // Pass 1: count particles per cell.
    let mut counts = vec![0u32; total_cells];
    for p in particles {
        if p.render_type != RenderType::Volumetric as u8 {
            continue;
        }
        let rel = Vec3::from(p.position) - origin;
        let cx = (rel.x * inv).floor().max(0.0).min((grid_dims[0] - 1) as f32) as u32;
        let cy = (rel.y * inv).floor().max(0.0).min((grid_dims[1] - 1) as f32) as u32;
        let cz = (rel.z * inv).floor().max(0.0).min((grid_dims[2] - 1) as f32) as u32;
        let idx = (cx + cy * stride_y + cz * stride_z) as usize;
        counts[idx] += 1;
    }

    // Pass 2: prefix sum → cell_offsets (length = total_cells + 1).
    let mut cell_offsets = Vec::with_capacity(total_cells + 1);
    cell_offsets.push(0u32);
    let mut running = 0u32;
    for &c in &counts {
        running += c;
        cell_offsets.push(running);
    }
    let total_binned = running as usize;

    // Pass 3: scatter indices (use a write-cursor copy of offsets).
    let mut write_cursors = cell_offsets[..total_cells].to_vec();
    let mut particle_indices = vec![0u32; total_binned];
    for (i, p) in particles.iter().enumerate() {
        if p.render_type != RenderType::Volumetric as u8 {
            continue;
        }
        let rel = Vec3::from(p.position) - origin;
        let cx = (rel.x * inv).floor().max(0.0).min((grid_dims[0] - 1) as f32) as u32;
        let cy = (rel.y * inv).floor().max(0.0).min((grid_dims[1] - 1) as f32) as u32;
        let cz = (rel.z * inv).floor().max(0.0).min((grid_dims[2] - 1) as f32) as u32;
        let idx = (cx + cy * stride_y + cz * stride_z) as usize;
        let slot = write_cursors[idx] as usize;
        particle_indices[slot] = i as u32;
        write_cursors[idx] += 1;
    }

    ParticleGrid3D {
        origin,
        cell_size,
        grid_dims,
        cell_offsets,
        particle_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{flags, Particle, RenderType};
    use half::f16;

    fn make_volumetric_particle(pos: [f32; 3]) -> Particle {
        Particle {
            position: pos,
            lifetime: 1.0,
            velocity: [0.0; 3],
            max_lifetime: 1.0,
            color_emission: [
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(0.0).to_bits(),
            ],
            size: f16::from_f32(0.1).to_bits(),
            render_type: RenderType::Volumetric as u8,
            flags: flags::ALIVE,
            material_id: 0,
            _pad: 0,
        }
    }

    #[test]
    fn test_empty_buffer() {
        let grid = bin_particles(&[], Vec3::ZERO, 1.0, [4, 4, 4]);
        assert_eq!(grid.particle_indices.len(), 0);
        assert_eq!(grid.cell_offsets.len(), 65); // 64 cells + 1
        for i in 0..64 {
            assert!(grid.particles_in_cell(i % 4, (i / 4) % 4, i / 16).is_empty());
        }
    }

    #[test]
    fn test_single_particle() {
        let p = make_volumetric_particle([0.5, 0.5, 0.5]);
        let grid = bin_particles(&[p], Vec3::ZERO, 1.0, [4, 4, 4]);
        assert_eq!(grid.particle_indices.len(), 1);
        // Cell (0,0,0) should contain index 0
        let cell = grid.particles_in_cell(0, 0, 0);
        assert_eq!(cell, &[0]);
        // Other cells empty
        assert!(grid.particles_in_cell(1, 0, 0).is_empty());
    }

    #[test]
    fn test_multiple_cells() {
        let p0 = make_volumetric_particle([0.5, 0.5, 0.5]); // cell (0,0,0)
        let p1 = make_volumetric_particle([2.5, 0.5, 0.5]); // cell (2,0,0)
        let p2 = make_volumetric_particle([0.5, 3.5, 0.5]); // cell (0,3,0)

        let grid = bin_particles(&[p0, p1, p2], Vec3::ZERO, 1.0, [4, 4, 4]);
        assert_eq!(grid.particle_indices.len(), 3);

        let c0 = grid.particles_in_cell(0, 0, 0);
        assert_eq!(c0.len(), 1);
        assert_eq!(c0[0], 0);

        let c2 = grid.particles_in_cell(2, 0, 0);
        assert_eq!(c2.len(), 1);
        assert_eq!(c2[0], 1);

        let c3 = grid.particles_in_cell(0, 3, 0);
        assert_eq!(c3.len(), 1);
        assert_eq!(c3[0], 2);
    }

    #[test]
    fn test_world_to_cell() {
        let grid = ParticleGrid3D {
            origin: Vec3::new(-2.0, -2.0, -2.0),
            cell_size: 1.0,
            grid_dims: [4, 4, 4],
            cell_offsets: vec![0; 65],
            particle_indices: vec![],
        };

        // Position at origin → cell (0,0,0)
        assert_eq!(grid.world_to_cell(Vec3::new(-2.0, -2.0, -2.0)), [0, 0, 0]);
        // Position at +1.5 offset → cell (1,1,1)
        assert_eq!(grid.world_to_cell(Vec3::new(-0.5, -0.5, -0.5)), [1, 1, 1]);
        // Position beyond grid → clamped to max
        assert_eq!(grid.world_to_cell(Vec3::new(100.0, 100.0, 100.0)), [3, 3, 3]);
        // Position below origin → clamped to 0
        assert_eq!(grid.world_to_cell(Vec3::new(-10.0, -10.0, -10.0)), [0, 0, 0]);
    }

    #[test]
    fn test_only_volumetric_binned() {
        let p0 = make_volumetric_particle([0.5, 0.5, 0.5]);
        let mut p1 = make_volumetric_particle([1.5, 0.5, 0.5]);
        p1.render_type = RenderType::SdfMicro as u8; // NOT volumetric
        let mut p2 = make_volumetric_particle([2.5, 0.5, 0.5]);
        p2.render_type = RenderType::ScreenSpace as u8; // NOT volumetric

        let grid = bin_particles(&[p0, p1, p2], Vec3::ZERO, 1.0, [4, 4, 4]);
        // Only particle 0 should be binned
        assert_eq!(grid.particle_indices.len(), 1);
        assert_eq!(grid.particles_in_cell(0, 0, 0), &[0]);
    }

    #[test]
    fn test_gpu_buffers_roundtrip() {
        let p0 = make_volumetric_particle([0.5, 0.5, 0.5]);
        let p1 = make_volumetric_particle([1.5, 0.5, 0.5]);
        let grid = bin_particles(&[p0, p1], Vec3::ZERO, 1.0, [4, 4, 4]);

        let (offsets, indices, params) = grid.to_gpu_buffers();
        assert_eq!(offsets.len(), 65); // 64 cells + 1
        assert_eq!(indices.len(), 2); // 2 volumetric particles
        assert_eq!(params.cell_count, 64);
        assert_eq!(params.cell_size, 1.0);
        assert_eq!(params.dims, [4, 4, 4]);
        assert_eq!(params.origin, [0.0, 0.0, 0.0]);
    }
}
