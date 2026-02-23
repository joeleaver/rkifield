//! GPU BVH buffer — uploads BVH nodes for compute shader traversal.
//!
//! [`GpuBvhNode`] is a 32-byte Pod struct matching the WGSL layout:
//!
//! ```text
//! struct BvhNode {
//!     aabb_min: vec3<f32>,   // 12 bytes
//!     left: u32,             //  4 bytes
//!     aabb_max: vec3<f32>,   // 12 bytes
//!     right_or_object: u32,  //  4 bytes
//! }
//! ```
//!
//! For internal nodes, `left` and `right_or_object` are child indices.
//! For leaf nodes (left == 0xFFFFFFFF), `right_or_object` is the object index.

use rkf_core::bvh::Bvh;

/// GPU-ready BVH node (32 bytes, bytemuck Pod).
///
/// Layout matches WGSL struct for direct upload.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    /// AABB minimum (xyz), left child index (w).
    pub aabb_min_x: f32,
    /// AABB minimum Y.
    pub aabb_min_y: f32,
    /// AABB minimum Z.
    pub aabb_min_z: f32,
    /// Left child index (INVALID for leaves).
    pub left: u32,
    /// AABB maximum X.
    pub aabb_max_x: f32,
    /// AABB maximum Y.
    pub aabb_max_y: f32,
    /// AABB maximum Z.
    pub aabb_max_z: f32,
    /// Right child index (internal) or object index (leaf).
    pub right_or_object: u32,
}

impl GpuBvhNode {
    /// Convert a CPU BvhNode to GPU format.
    pub fn from_cpu(node: &rkf_core::bvh::BvhNode) -> Self {
        Self {
            aabb_min_x: node.aabb.min.x,
            aabb_min_y: node.aabb.min.y,
            aabb_min_z: node.aabb.min.z,
            left: node.left,
            aabb_max_x: node.aabb.max.x,
            aabb_max_y: node.aabb.max.y,
            aabb_max_z: node.aabb.max.z,
            right_or_object: if node.is_leaf() {
                node.object_index
            } else {
                node.right
            },
        }
    }
}

/// Default initial capacity for the BVH buffer (in nodes).
const DEFAULT_BVH_CAPACITY: u32 = 512;

/// GPU BVH storage buffer.
pub struct GpuBvh {
    /// The wgpu storage buffer.
    pub buffer: wgpu::Buffer,
    /// Current buffer capacity in nodes.
    capacity: u32,
    /// Number of valid nodes.
    node_count: u32,
}

impl GpuBvh {
    /// Create a new GPU BVH buffer with default capacity.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bvh_nodes"),
            size: (DEFAULT_BVH_CAPACITY as u64) * std::mem::size_of::<GpuBvhNode>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity: DEFAULT_BVH_CAPACITY,
            node_count: 0,
        }
    }

    /// Upload a full BVH to the GPU.
    ///
    /// Converts CPU nodes to GPU format and writes to the storage buffer.
    /// Grows the buffer if needed.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bvh: &Bvh) -> bool {
        let gpu_nodes: Vec<GpuBvhNode> = bvh.nodes.iter().map(GpuBvhNode::from_cpu).collect();
        let count = gpu_nodes.len() as u32;
        let mut resized = false;

        if count > self.capacity {
            let new_cap = count.max(self.capacity * 2).max(DEFAULT_BVH_CAPACITY);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bvh_nodes"),
                size: (new_cap as u64) * std::mem::size_of::<GpuBvhNode>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.capacity = new_cap;
            resized = true;
        }

        if !gpu_nodes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&gpu_nodes));
        }
        self.node_count = count;
        resized
    }

    /// Number of valid nodes.
    #[inline]
    pub fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Whether the BVH is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Current buffer capacity in nodes.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::bvh::{BvhNode, INVALID};
    use rkf_core::Aabb;
    use glam::Vec3;

    #[test]
    fn gpu_bvh_node_size() {
        assert_eq!(std::mem::size_of::<GpuBvhNode>(), 32);
    }

    #[test]
    fn gpu_bvh_node_pod_roundtrip() {
        let node = GpuBvhNode {
            aabb_min_x: -1.0,
            aabb_min_y: -2.0,
            aabb_min_z: -3.0,
            left: 5,
            aabb_max_x: 1.0,
            aabb_max_y: 2.0,
            aabb_max_z: 3.0,
            right_or_object: 10,
        };
        let bytes = bytemuck::bytes_of(&node);
        assert_eq!(bytes.len(), 32);
        let restored: &GpuBvhNode = bytemuck::from_bytes(bytes);
        assert_eq!(restored.left, 5);
        assert_eq!(restored.right_or_object, 10);
    }

    #[test]
    fn from_cpu_leaf() {
        let cpu = BvhNode {
            aabb: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            left: INVALID,
            right: INVALID,
            object_index: 42,
        };
        let gpu = GpuBvhNode::from_cpu(&cpu);
        assert_eq!(gpu.left, INVALID);
        assert_eq!(gpu.right_or_object, 42); // object index for leaf
    }

    #[test]
    fn from_cpu_internal() {
        let cpu = BvhNode {
            aabb: Aabb::new(Vec3::ZERO, Vec3::ONE),
            left: 1,
            right: 2,
            object_index: INVALID,
        };
        let gpu = GpuBvhNode::from_cpu(&cpu);
        assert_eq!(gpu.left, 1);
        assert_eq!(gpu.right_or_object, 2); // right child for internal
    }
}
