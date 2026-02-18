use bytemuck::{Pod, Zeroable};

/// Maximum number of active fog volumes in a scene.
pub const MAX_FOG_VOLUMES: usize = 64;

/// GPU-uploadable fog volume descriptor (64 bytes).
///
/// Describes a single local fog volume with AABB bounds and properties.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GpuFogVolume {
    /// AABB minimum corner (world space). w = density_scale.
    pub aabb_min: [f32; 4],
    /// AABB maximum corner (world space). w = emission_scale.
    pub aabb_max: [f32; 4],
    /// Scattering color (linear RGB). w = phase_g (HG asymmetry).
    pub color: [f32; 4],
    /// x = falloff (edge softness), y = noise_scale, z = noise_speed, w = active (0/1).
    pub properties: [f32; 4],
}

impl GpuFogVolume {
    /// Create a fog volume with given bounds and density.
    pub fn new(
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        density_scale: f32,
        color: [f32; 3],
        phase_g: f32,
    ) -> Self {
        Self {
            aabb_min: [aabb_min[0], aabb_min[1], aabb_min[2], density_scale],
            aabb_max: [aabb_max[0], aabb_max[1], aabb_max[2], 0.0],
            color: [color[0], color[1], color[2], phase_g],
            properties: [1.0, 0.0, 0.0, 1.0], // falloff=1, no noise, active
        }
    }
}

/// GPU-uploadable fog volume header (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[allow(missing_docs)]
pub struct FogVolumeHeader {
    /// Number of active fog volumes.
    pub count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// GPU buffer holding fog volume metadata.
///
/// Layout: FogVolumeHeader (16 bytes) + MAX_FOG_VOLUMES × GpuFogVolume (64 bytes each)
/// Total: 16 + 64 × 64 = 4112 bytes
pub struct FogVolumeBuffer {
    buffer: wgpu::Buffer,
    count: u32,
}

impl FogVolumeBuffer {
    /// Create the fog volume buffer with no active volumes.
    pub fn new(device: &wgpu::Device) -> Self {
        let total_size = std::mem::size_of::<FogVolumeHeader>()
            + MAX_FOG_VOLUMES * std::mem::size_of::<GpuFogVolume>();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog volume buffer"),
            size: total_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, count: 0 }
    }

    /// Upload fog volumes to the GPU buffer.
    ///
    /// Writes the header (count) and all volume descriptors.
    pub fn upload(&mut self, queue: &wgpu::Queue, volumes: &[GpuFogVolume]) {
        let count = volumes.len().min(MAX_FOG_VOLUMES) as u32;
        self.count = count;

        let header = FogVolumeHeader {
            count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&header));

        if count > 0 {
            let header_size = std::mem::size_of::<FogVolumeHeader>() as u64;
            let data = bytemuck::cast_slice(&volumes[..count as usize]);
            queue.write_buffer(&self.buffer, header_size, data);
        }
    }

    /// Get a reference to the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Number of active fog volumes.
    pub fn count(&self) -> u32 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_fog_volume_size_is_64() {
        assert_eq!(std::mem::size_of::<GpuFogVolume>(), 64);
    }

    #[test]
    fn fog_volume_header_size_is_16() {
        assert_eq!(std::mem::size_of::<FogVolumeHeader>(), 16);
    }

    #[test]
    fn gpu_fog_volume_pod_roundtrip() {
        let v = GpuFogVolume::new(
            [-5.0, 0.0, -5.0],
            [5.0, 3.0, 5.0],
            0.5,
            [0.8, 0.85, 0.9],
            0.3,
        );
        let bytes = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 64);
        let v2: &GpuFogVolume = bytemuck::from_bytes(bytes);
        assert_eq!(v.aabb_min[3], 0.5); // density_scale
        assert_eq!(v2.color[3], 0.3); // phase_g
        assert_eq!(v2.properties[3], 1.0); // active
    }

    #[test]
    fn fog_volume_new_packs_correctly() {
        let v = GpuFogVolume::new(
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            0.8,
            [1.0, 0.5, 0.2],
            0.6,
        );
        // density_scale in aabb_min.w
        assert_eq!(v.aabb_min, [1.0, 2.0, 3.0, 0.8]);
        // phase_g in color.w
        assert_eq!(v.color, [1.0, 0.5, 0.2, 0.6]);
        // active flag in properties.w
        assert_eq!(v.properties[3], 1.0);
    }

    #[test]
    fn buffer_total_size() {
        let expected = 16 + MAX_FOG_VOLUMES * 64;
        assert_eq!(expected, 4112);
    }

    #[test]
    fn max_fog_volumes() {
        assert_eq!(MAX_FOG_VOLUMES, 64);
    }

    #[test]
    fn fog_volume_field_offsets() {
        assert_eq!(std::mem::offset_of!(GpuFogVolume, aabb_min), 0);
        assert_eq!(std::mem::offset_of!(GpuFogVolume, aabb_max), 16);
        assert_eq!(std::mem::offset_of!(GpuFogVolume, color), 32);
        assert_eq!(std::mem::offset_of!(GpuFogVolume, properties), 48);
    }
}
