//! Coarse acceleration field — low-resolution world-space distance field.
//!
//! A 3D R16Float texture storing conservative minimum distances to any surface.
//! The ray marcher uses this for empty-space skipping: large steps through empty
//! regions before switching to per-object evaluation near surfaces.
//!
//! Cell values are the unsigned distance from each cell center to the nearest
//! object AABB surface (conservative — always ≤ true SDF distance).
//!
//! # Bind Group (group 3 in the ray march pipeline)
//!
//! | Binding | Resource | Content |
//! |---------|----------|---------|
//! | 0 | Texture 3D (f32, sampled) | Distance field |
//! | 1 | Sampler (linear) | Trilinear interpolation |
//! | 2 | Uniform | Field metadata (origin, dims, voxel size) |

use glam::{UVec3, Vec3};

/// Default voxel size for the coarse field (32cm).
pub const COARSE_VOXEL_SIZE: f32 = 0.32;

/// Maximum cells per dimension (limits GPU memory).
pub const MAX_COARSE_DIM: u32 = 128;

/// Uniform data describing the coarse field layout.
///
/// Sent to the ray march shader so it can compute texture coordinates.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoarseFieldUniforms {
    /// Camera-relative origin of the field (minimum corner). Updated each frame.
    pub origin_cam_rel: [f32; 4],
    /// Field dimensions in cells (x, y, z, 0).
    pub dims: [u32; 4],
    /// Voxel size in world units.
    pub voxel_size: f32,
    /// Inverse voxel size (1.0 / voxel_size).
    pub inv_voxel_size: f32,
    /// Padding.
    pub _pad: [f32; 2],
}

/// Coarse acceleration field for empty-space skipping.
pub struct CoarseField {
    /// 3D texture on GPU (R16Float).
    texture: wgpu::Texture,
    /// Texture view for sampling.
    #[allow(dead_code)]
    texture_view: wgpu::TextureView,
    /// Sampler (linear interpolation).
    #[allow(dead_code)]
    sampler: wgpu::Sampler,
    /// CPU-side distance data (f16, row-major: x + y*dx + z*dx*dy).
    data: Vec<half::f16>,
    /// World-space origin (minimum corner) of the field.
    pub origin: Vec3,
    /// Dimensions in cells.
    pub dims: UVec3,
    /// Voxel size in world units.
    pub voxel_size: f32,
    /// Bind group layout for the ray marcher (group 3).
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group: texture + sampler + uniforms.
    pub bind_group: wgpu::BindGroup,
    /// Uniform buffer for field metadata.
    uniform_buffer: wgpu::Buffer,
    /// Dirty region (cell coordinates, inclusive). If set, only these cells need re-upload.
    dirty_min: Option<UVec3>,
    dirty_max: Option<UVec3>,
}

/// Unsigned distance from a point to an AABB surface.
///
/// Returns 0 if the point is inside the AABB.
fn aabb_unsigned_distance(p: Vec3, aabb_min: Vec3, aabb_max: Vec3) -> f32 {
    let d = (aabb_min - p).max(p - aabb_max).max(Vec3::ZERO);
    d.length()
}

impl CoarseField {
    /// Create a coarse acceleration field covering the given world-space region.
    ///
    /// `origin` is the minimum corner of the field volume. `extent` is the
    /// world-space size (width, height, depth). The field is subdivided at
    /// `voxel_size` resolution, capped to [`MAX_COARSE_DIM`] per axis.
    pub fn new(
        device: &wgpu::Device,
        origin: Vec3,
        extent: Vec3,
        voxel_size: f32,
    ) -> Self {
        let dims = UVec3::new(
            ((extent.x / voxel_size).ceil() as u32).min(MAX_COARSE_DIM).max(1),
            ((extent.y / voxel_size).ceil() as u32).min(MAX_COARSE_DIM).max(1),
            ((extent.z / voxel_size).ceil() as u32).min(MAX_COARSE_DIM).max(1),
        );

        let cell_count = (dims.x * dims.y * dims.z) as usize;
        let data = vec![half::f16::from_f32(voxel_size * dims.x.max(dims.y).max(dims.z) as f32); cell_count];

        // GPU 3D texture (R16Float).
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("coarse_field"),
            size: wgpu::Extent3d {
                width: dims.x,
                height: dims.y,
                depth_or_array_layers: dims.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("coarse_field_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coarse_field_uniforms"),
            size: std::mem::size_of::<CoarseFieldUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coarse_field layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("coarse_field"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            texture,
            texture_view,
            sampler,
            data,
            origin,
            dims,
            voxel_size,
            bind_group_layout,
            bind_group,
            uniform_buffer,
            dirty_min: None,
            dirty_max: None,
        }
    }

    /// Create a coarse field that covers the union of the given AABBs with margin.
    ///
    /// `margin` is extra padding around the scene AABB (in world units).
    pub fn from_scene_aabbs(
        device: &wgpu::Device,
        aabbs: &[(Vec3, Vec3)],
        voxel_size: f32,
        margin: f32,
    ) -> Self {
        let (scene_min, scene_max) = if aabbs.is_empty() {
            (Vec3::ZERO, Vec3::splat(voxel_size))
        } else {
            let mut lo = Vec3::splat(f32::MAX);
            let mut hi = Vec3::splat(f32::MIN);
            for (amin, amax) in aabbs {
                lo = lo.min(*amin);
                hi = hi.max(*amax);
            }
            (lo, hi)
        };

        let origin = scene_min - Vec3::splat(margin);
        let extent = (scene_max + Vec3::splat(margin)) - origin;

        Self::new(device, origin, extent, voxel_size)
    }

    /// Populate the CPU-side distance data from scene object AABBs.
    ///
    /// For each cell center, computes the unsigned distance to the nearest
    /// object AABB surface. This is conservative (≤ true SDF distance) and
    /// safe for empty-space skipping.
    pub fn populate(&mut self, aabbs: &[(Vec3, Vec3)]) {
        let dx = self.dims.x;
        let dy = self.dims.y;
        let dz = self.dims.z;
        let vs = self.voxel_size;

        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let world_pos = self.origin + Vec3::new(
                        (x as f32 + 0.5) * vs,
                        (y as f32 + 0.5) * vs,
                        (z as f32 + 0.5) * vs,
                    );

                    let mut min_dist = vs * dx.max(dy).max(dz) as f32;
                    for (amin, amax) in aabbs {
                        let d = aabb_unsigned_distance(world_pos, *amin, *amax);
                        min_dist = min_dist.min(d);
                    }

                    let idx = (x + y * dx + z * dx * dy) as usize;
                    self.data[idx] = half::f16::from_f32(min_dist);
                }
            }
        }
    }

    /// Upload the CPU distance data to the GPU 3D texture and update uniforms.
    ///
    /// `camera_pos` is the current camera world-space position, used to compute
    /// the camera-relative field origin for the shader.
    pub fn upload(&self, queue: &wgpu::Queue, camera_pos: Vec3) {
        // Upload texture data.
        let bytes_per_row = self.dims.x * 2; // R16Float = 2 bytes per texel
        let rows_per_image = self.dims.y;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width: self.dims.x,
                height: self.dims.y,
                depth_or_array_layers: self.dims.z,
            },
        );

        // Update uniforms with camera-relative origin.
        let cam_rel_origin = self.origin - camera_pos;
        let uniforms = CoarseFieldUniforms {
            origin_cam_rel: [cam_rel_origin.x, cam_rel_origin.y, cam_rel_origin.z, 0.0],
            dims: [self.dims.x, self.dims.y, self.dims.z, 0],
            voxel_size: self.voxel_size,
            inv_voxel_size: 1.0 / self.voxel_size,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    /// Update only the uniforms (camera-relative origin). Call each frame when
    /// the camera moves but the field data hasn't changed.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, camera_pos: Vec3) {
        let cam_rel_origin = self.origin - camera_pos;
        let uniforms = CoarseFieldUniforms {
            origin_cam_rel: [cam_rel_origin.x, cam_rel_origin.y, cam_rel_origin.z, 0.0],
            dims: [self.dims.x, self.dims.y, self.dims.z, 0],
            voxel_size: self.voxel_size,
            inv_voxel_size: 1.0 / self.voxel_size,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    // ------------------------------------------------------------------
    // Dirty region tracking and incremental update
    // ------------------------------------------------------------------

    /// Convert a world-space position to cell coordinates.
    /// Returns `None` if the position is outside the field bounds.
    fn world_to_cell(&self, world_pos: Vec3) -> Option<UVec3> {
        let local = world_pos - self.origin;
        let cell = local / self.voxel_size;
        if cell.x < 0.0 || cell.y < 0.0 || cell.z < 0.0 {
            return None;
        }
        let c = UVec3::new(cell.x as u32, cell.y as u32, cell.z as u32);
        if c.x >= self.dims.x || c.y >= self.dims.y || c.z >= self.dims.z {
            return None;
        }
        Some(c)
    }

    /// Mark a world-space AABB region as dirty. The dirty region expands to
    /// cover the union of all marked regions. Call this when an object moves,
    /// is added, or is removed — pass both the old and new AABBs.
    pub fn mark_dirty(&mut self, world_min: Vec3, world_max: Vec3) {
        // Convert to cell coordinates, clamped to field bounds.
        let inv_vs = 1.0 / self.voxel_size;
        let local_min = (world_min - self.origin) * inv_vs;
        let local_max = (world_max - self.origin) * inv_vs;

        let clamped_min = local_min.floor().max(Vec3::ZERO);
        let clamped_max = local_max.ceil().max(Vec3::ZERO);
        let last = self.dims - UVec3::ONE;

        let cell_min = UVec3::new(
            (clamped_min.x as u32).min(last.x),
            (clamped_min.y as u32).min(last.y),
            (clamped_min.z as u32).min(last.z),
        );
        let cell_max = UVec3::new(
            (clamped_max.x as u32).min(last.x),
            (clamped_max.y as u32).min(last.y),
            (clamped_max.z as u32).min(last.z),
        );

        self.dirty_min = Some(match self.dirty_min {
            Some(prev) => prev.min(cell_min),
            None => cell_min,
        });
        self.dirty_max = Some(match self.dirty_max {
            Some(prev) => prev.max(cell_max),
            None => cell_max,
        });
    }

    /// Mark the entire field as dirty. Call after full scene reload.
    pub fn mark_all_dirty(&mut self) {
        self.dirty_min = Some(UVec3::ZERO);
        self.dirty_max = Some(self.dims - UVec3::ONE);
    }

    /// Returns true if any cells are marked dirty.
    pub fn is_dirty(&self) -> bool {
        self.dirty_min.is_some()
    }

    /// Re-evaluate only the dirty cells and upload the changed region to the GPU.
    ///
    /// `aabbs` is the complete set of scene object AABBs (old and new positions).
    /// Only cells within the dirty region are re-evaluated. After upload, the
    /// dirty region is cleared.
    ///
    /// Returns the number of cells updated, or 0 if nothing was dirty.
    pub fn update_dirty(&mut self, queue: &wgpu::Queue, aabbs: &[(Vec3, Vec3)]) -> u32 {
        let (cell_min, cell_max) = match (self.dirty_min, self.dirty_max) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => return 0,
        };

        let dx = self.dims.x;
        let dy = self.dims.y;
        let vs = self.voxel_size;
        let max_dist = vs * dx.max(dy).max(self.dims.z) as f32;
        let mut cells_updated = 0u32;

        for z in cell_min.z..=cell_max.z {
            for y in cell_min.y..=cell_max.y {
                for x in cell_min.x..=cell_max.x {
                    let world_pos = self.origin + Vec3::new(
                        (x as f32 + 0.5) * vs,
                        (y as f32 + 0.5) * vs,
                        (z as f32 + 0.5) * vs,
                    );

                    let mut min_dist = max_dist;
                    for (amin, amax) in aabbs {
                        let d = aabb_unsigned_distance(world_pos, *amin, *amax);
                        min_dist = min_dist.min(d);
                    }

                    let idx = (x + y * dx + z * dx * dy) as usize;
                    self.data[idx] = half::f16::from_f32(min_dist);
                    cells_updated += 1;
                }
            }
        }

        // Upload only the dirty z-slices to the GPU.
        // wgpu write_texture works on contiguous z-slice ranges, so we upload
        // the minimal range of z-slices containing the dirty region.
        let slice_size = (dx * dy) as usize;
        let z_start = cell_min.z;
        let z_count = cell_max.z - cell_min.z + 1;
        let data_offset = (z_start * dx * dy) as usize;
        let data_len = (z_count * dx * dy) as usize;

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: z_start },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&self.data[data_offset..data_offset + data_len]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dx * 2),
                rows_per_image: Some(dy),
            },
            wgpu::Extent3d {
                width: dx,
                height: dy,
                depth_or_array_layers: z_count,
            },
        );

        // Clear dirty region.
        self.dirty_min = None;
        self.dirty_max = None;

        let _ = slice_size; // suppress unused
        cells_updated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_distance_outside() {
        let d = aabb_unsigned_distance(
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!((d - 2.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_distance_inside() {
        let d = aabb_unsigned_distance(
            Vec3::ZERO,
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert_eq!(d, 0.0);
    }

    #[test]
    fn aabb_distance_corner() {
        let d = aabb_unsigned_distance(
            Vec3::new(2.0, 2.0, 2.0),
            Vec3::ZERO,
            Vec3::ONE,
        );
        // Distance from (2,2,2) to corner (1,1,1) = sqrt(3)
        assert!((d - 3.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn dims_calculation() {
        let origin = Vec3::new(-2.0, -1.0, -3.0);
        let extent = Vec3::new(4.0, 2.0, 6.0);
        let voxel_size = 0.32;

        let dx = (extent.x / voxel_size).ceil() as u32;
        let dy = (extent.y / voxel_size).ceil() as u32;
        let dz = (extent.z / voxel_size).ceil() as u32;

        assert_eq!(dx, 13); // 4.0 / 0.32 = 12.5 → 13
        assert_eq!(dy, 7);  // 2.0 / 0.32 = 6.25 → 7
        assert_eq!(dz, 19); // 6.0 / 0.32 = 18.75 → 19

        // Verify the region we cover.
        let actual_extent = Vec3::new(dx as f32, dy as f32, dz as f32) * voxel_size;
        assert!(actual_extent.x >= extent.x);
        assert!(actual_extent.y >= extent.y);
        assert!(actual_extent.z >= extent.z);

        // Total cells.
        assert_eq!(dx * dy * dz, 1729);
        // Memory: 1729 * 2 bytes (R16Float) = 3,458 bytes ≈ 3.4 KB.
        let _ = origin; // suppress unused warning
    }

    #[test]
    fn populate_empty_scene() {
        // With no objects, all cells should have max distance.
        // (Can't create GPU texture without device, so just test the math.)
        let origin = Vec3::ZERO;
        let _extent = Vec3::splat(1.0);
        let voxel_size = 0.5;
        let dims = UVec3::new(2, 2, 2);
        let cell_count = 8;

        let max_d = voxel_size * 2.0; // max dim = 2 cells
        let mut data = vec![half::f16::from_f32(max_d); cell_count];

        // Populate with no AABBs.
        let aabbs: &[(Vec3, Vec3)] = &[];
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let world_pos = origin + Vec3::new(
                        (x as f32 + 0.5) * voxel_size,
                        (y as f32 + 0.5) * voxel_size,
                        (z as f32 + 0.5) * voxel_size,
                    );
                    let mut min_dist = max_d;
                    for (amin, amax) in aabbs {
                        min_dist = min_dist.min(aabb_unsigned_distance(world_pos, *amin, *amax));
                    }
                    let idx = (x + y * dims.x + z * dims.x * dims.y) as usize;
                    data[idx] = half::f16::from_f32(min_dist);
                }
            }
        }

        // All cells should have max distance.
        for d in &data {
            assert_eq!(d.to_f32(), max_d);
        }
    }

    #[test]
    fn populate_one_object() {
        let origin = Vec3::new(-2.0, -2.0, -2.0);
        let voxel_size = 1.0;
        let dims = UVec3::new(4, 4, 4);

        // Object AABB at origin, size 1×1×1.
        let aabbs = vec![(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5))];

        let mut data = vec![half::f16::from_f32(100.0); 64];
        for z in 0..dims.z {
            for y in 0..dims.y {
                for x in 0..dims.x {
                    let world_pos = origin + Vec3::new(
                        (x as f32 + 0.5) * voxel_size,
                        (y as f32 + 0.5) * voxel_size,
                        (z as f32 + 0.5) * voxel_size,
                    );
                    let mut min_dist = 100.0_f32;
                    for (amin, amax) in &aabbs {
                        min_dist = min_dist.min(aabb_unsigned_distance(world_pos, *amin, *amax));
                    }
                    let idx = (x + y * dims.x + z * dims.x * dims.y) as usize;
                    data[idx] = half::f16::from_f32(min_dist);
                }
            }
        }

        // Cell at (2,2,2) has center at origin+(2.5, 2.5, 2.5) = (0.5, 0.5, 0.5).
        // That's exactly at the AABB corner, so distance = 0.
        let idx_center = 2 + 2 * 4 + 2 * 16;
        assert_eq!(data[idx_center].to_f32(), 0.0);

        // Cell at (0,0,0) has center at (-1.5, -1.5, -1.5).
        // Distance to AABB [-0.5..0.5] = length(max((-0.5 - -1.5), (-1.5 - 0.5), 0))
        //   = length(max((1.0), (-2.0), 0)) per component = length(1.0, 1.0, 1.0) = sqrt(3)
        let idx_corner = 0;
        let expected = (1.0_f32 + 1.0 + 1.0).sqrt();
        assert!((data[idx_corner].to_f32() - expected).abs() < 0.01);
    }

    // --- Dirty tracking tests (no GPU needed) ---

    #[test]
    fn world_to_cell_inside() {
        // Manual CoarseField-like test without GPU.
        let origin = Vec3::new(-2.0, -1.0, -3.0);
        let dims = UVec3::new(13, 7, 19);
        let voxel_size = 0.32_f32;

        // Test point at origin + (0.5*vs, 0.5*vs, 0.5*vs) = cell (0,0,0).
        let local = (origin + Vec3::splat(0.16)) - origin;
        let cell = local / voxel_size;
        assert_eq!(cell.x as u32, 0);
        assert_eq!(cell.y as u32, 0);
        assert_eq!(cell.z as u32, 0);

        // Test point at origin + (4.0, 1.5, 5.0).
        let local2 = Vec3::new(4.0, 1.5, 5.0);
        let c2 = (local2 / voxel_size).floor();
        assert_eq!(c2.x as u32, 12); // 4.0/0.32 = 12.5 → floor 12
        assert_eq!(c2.y as u32, 4);  // 1.5/0.32 = 4.6875 → floor 4
        let _ = dims;
    }

    #[test]
    fn mark_dirty_single() {
        // Test dirty tracking logic without GPU.
        let origin = Vec3::ZERO;
        let voxel_size = 1.0_f32;
        let dims = UVec3::new(10, 10, 10);
        let inv_vs = 1.0 / voxel_size;
        let last = dims - UVec3::ONE;

        // Simulate mark_dirty for AABB (2.0..4.0, 3.0..5.0, 1.0..6.0).
        let world_min = Vec3::new(2.0, 3.0, 1.0);
        let world_max = Vec3::new(4.0, 5.0, 6.0);
        let local_min = (world_min - origin) * inv_vs;
        let local_max = (world_max - origin) * inv_vs;

        let clamped_min = local_min.floor().max(Vec3::ZERO);
        let clamped_max = local_max.ceil().max(Vec3::ZERO);

        let cell_min = UVec3::new(
            (clamped_min.x as u32).min(last.x),
            (clamped_min.y as u32).min(last.y),
            (clamped_min.z as u32).min(last.z),
        );
        let cell_max = UVec3::new(
            (clamped_max.x as u32).min(last.x),
            (clamped_max.y as u32).min(last.y),
            (clamped_max.z as u32).min(last.z),
        );

        assert_eq!(cell_min, UVec3::new(2, 3, 1));
        assert_eq!(cell_max, UVec3::new(4, 5, 6));
    }

    #[test]
    fn mark_dirty_union() {
        // Two dirty regions should merge.
        let mut dirty_min: Option<UVec3> = None;
        let mut dirty_max: Option<UVec3> = None;

        // First region: (2,3,1) to (4,5,6).
        let r1_min = UVec3::new(2, 3, 1);
        let r1_max = UVec3::new(4, 5, 6);
        dirty_min = Some(r1_min);
        dirty_max = Some(r1_max);

        // Second region: (0,0,5) to (3,2,8).
        let r2_min = UVec3::new(0, 0, 5);
        let r2_max = UVec3::new(3, 2, 8);
        dirty_min = Some(dirty_min.unwrap().min(r2_min));
        dirty_max = Some(dirty_max.unwrap().max(r2_max));

        assert_eq!(dirty_min.unwrap(), UVec3::new(0, 0, 1));
        assert_eq!(dirty_max.unwrap(), UVec3::new(4, 5, 8));
    }

    #[test]
    fn dirty_cell_count() {
        // Test that the expected number of cells in a dirty region is correct.
        let cell_min = UVec3::new(2, 3, 1);
        let cell_max = UVec3::new(4, 5, 6);
        let count = (cell_max.x - cell_min.x + 1)
            * (cell_max.y - cell_min.y + 1)
            * (cell_max.z - cell_min.z + 1);
        assert_eq!(count, 3 * 3 * 6); // 54 cells
    }
}
