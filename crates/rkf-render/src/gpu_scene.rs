//! GPU scene v2 — object-centric bind group layout and resource management.
//!
//! [`GpuSceneV2`] replaces the deleted v1 chunk-based GpuScene. It manages the
//! GPU resources and bind group for the v2 object-centric ray marcher:
//!
//! | Binding | Resource | Content |
//! |---------|----------|---------|
//! | 0 | Storage (read) | Brick pool — `array<VoxelSample>` |
//! | 1 | Storage (read) | Brick maps — `array<u32>` |
//! | 2 | Storage (read) | Object metadata — `array<GpuObject>` |
//! | 3 | Uniform | Camera uniforms |
//! | 4 | Uniform | Scene uniforms (num_objects, max_steps, max_distance) |
//! | 5 | Storage (read) | BVH nodes — `array<BvhNode>` |

use crate::camera::CameraUniforms;
use crate::gpu_bvh::GpuBvh;
use crate::gpu_brick_maps::GpuBrickMaps;
use crate::gpu_object::GpuObject;

/// Scene-level uniform data for the ray marcher.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    /// Number of active objects in the metadata buffer.
    pub num_objects: u32,
    /// Maximum sphere-tracing steps per ray.
    pub max_steps: u32,
    /// Maximum ray march distance.
    pub max_distance: f32,
    /// Minimum surface hit distance.
    pub hit_threshold: f32,
}

impl Default for SceneUniforms {
    fn default() -> Self {
        Self {
            num_objects: 0,
            max_steps: 128,
            max_distance: 100.0,
            hit_threshold: 0.001,
        }
    }
}

/// Default initial capacity for the object metadata buffer (in objects).
const DEFAULT_OBJECT_CAPACITY: u32 = 256;

/// Default initial capacity for the brick maps buffer (in u32 entries).
const DEFAULT_BRICK_MAP_CAPACITY: u32 = 65536;

/// GPU scene v2 — manages all object-centric rendering resources.
pub struct GpuSceneV2 {
    /// Packed brick maps buffer (all objects' brick maps).
    pub brick_maps: GpuBrickMaps,

    /// Object metadata storage buffer.
    pub object_buffer: wgpu::Buffer,
    /// Current capacity of object_buffer (in GpuObjects).
    object_capacity: u32,
    /// Number of active objects.
    num_objects: u32,

    /// BVH node storage buffer.
    pub bvh: GpuBvh,

    /// Camera uniform buffer.
    pub camera_buffer: wgpu::Buffer,
    /// Scene uniform buffer.
    pub scene_buffer: wgpu::Buffer,

    /// Bind group layout for the scene.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group — recreated when buffers are resized.
    pub bind_group: wgpu::BindGroup,

    /// Brick pool storage buffer (reference — owned elsewhere, e.g. by BrickPoolManager).
    /// Set via [`set_brick_pool`](GpuSceneV2::set_brick_pool).
    brick_pool_buffer: wgpu::Buffer,
}

impl GpuSceneV2 {
    /// Create a new GPU scene v2 with default capacities.
    ///
    /// `brick_pool_buffer` is the GPU storage buffer containing the brick pool
    /// (array of VoxelSample). It is borrowed — the scene does not own it.
    pub fn new(device: &wgpu::Device, brick_pool_buffer: wgpu::Buffer) -> Self {
        let brick_maps = GpuBrickMaps::new(device, DEFAULT_BRICK_MAP_CAPACITY);
        let bvh = GpuBvh::new(device);

        let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object_metadata"),
            size: (DEFAULT_OBJECT_CAPACITY as u64) * std::mem::size_of::<GpuObject>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_uniforms"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scene_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene_uniforms"),
            size: std::mem::size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = Self::create_layout(device);
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &brick_pool_buffer,
            &brick_maps.buffer,
            &object_buffer,
            &camera_buffer,
            &scene_buffer,
            &bvh.buffer,
        );

        Self {
            brick_maps,
            object_buffer,
            object_capacity: DEFAULT_OBJECT_CAPACITY,
            num_objects: 0,
            bvh,
            camera_buffer,
            scene_buffer,
            bind_group_layout,
            bind_group,
            brick_pool_buffer,
        }
    }

    /// Upload object metadata to the GPU.
    ///
    /// If the data exceeds current capacity, a new buffer is created and the
    /// bind group is rebuilt.
    pub fn upload_objects(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        objects: &[GpuObject],
    ) {
        let count = objects.len() as u32;

        if count > self.object_capacity {
            let new_cap = count.max(self.object_capacity * 2).max(DEFAULT_OBJECT_CAPACITY);
            self.object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("object_metadata"),
                size: (new_cap as u64) * std::mem::size_of::<GpuObject>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.object_capacity = new_cap;
            self.rebuild_bind_group(device);
        }

        if !objects.is_empty() {
            queue.write_buffer(&self.object_buffer, 0, bytemuck::cast_slice(objects));
        }
        self.num_objects = count;
    }

    /// Upload brick maps data to the GPU.
    ///
    /// Rebuilds the bind group if the buffer is resized.
    pub fn upload_brick_maps(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u32],
    ) {
        let old_cap = self.brick_maps.capacity();
        self.brick_maps.upload(device, queue, data);
        if self.brick_maps.capacity() != old_cap {
            self.rebuild_bind_group(device);
        }
    }

    /// Upload BVH to the GPU.
    ///
    /// Rebuilds the bind group if the buffer is resized.
    pub fn upload_bvh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bvh: &rkf_core::Bvh,
    ) {
        if self.bvh.upload(device, queue, bvh) {
            self.rebuild_bind_group(device);
        }
    }

    /// Update camera uniforms.
    pub fn update_camera(&self, queue: &wgpu::Queue, camera: &CameraUniforms) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));
    }

    /// Update scene uniforms.
    pub fn update_scene_uniforms(&self, queue: &wgpu::Queue, uniforms: &SceneUniforms) {
        queue.write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Replace the brick pool buffer reference and rebuild the bind group.
    pub fn set_brick_pool(&mut self, device: &wgpu::Device, buffer: wgpu::Buffer) {
        self.brick_pool_buffer = buffer;
        self.rebuild_bind_group(device);
    }

    /// Number of active objects.
    #[inline]
    pub fn num_objects(&self) -> u32 {
        self.num_objects
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_scene_v2"),
            entries: &[
                // 0: Brick pool (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: Brick maps (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Object metadata (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Camera uniforms (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Scene uniforms (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: BVH nodes (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        brick_pool: &wgpu::Buffer,
        brick_maps: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        scene: &wgpu::Buffer,
        bvh: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_scene_v2"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brick_pool.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_maps.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: objects.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bvh.as_entire_binding(),
                },
            ],
        })
    }

    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = Self::create_bind_group(
            device,
            &self.bind_group_layout,
            &self.brick_pool_buffer,
            &self.brick_maps.buffer,
            &self.object_buffer,
            &self.camera_buffer,
            &self.scene_buffer,
            &self.bvh.buffer,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_uniforms_size() {
        assert_eq!(std::mem::size_of::<SceneUniforms>(), 16);
    }

    #[test]
    fn scene_uniforms_default() {
        let u = SceneUniforms::default();
        assert_eq!(u.num_objects, 0);
        assert_eq!(u.max_steps, 128);
        assert!((u.max_distance - 100.0).abs() < 1e-6);
        assert!((u.hit_threshold - 0.001).abs() < 1e-6);
    }

    #[test]
    fn scene_uniforms_pod_roundtrip() {
        let u = SceneUniforms {
            num_objects: 42,
            max_steps: 256,
            max_distance: 200.0,
            hit_threshold: 0.0005,
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 16);
        let restored: &SceneUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(restored.num_objects, 42);
        assert_eq!(restored.max_steps, 256);
    }
}
