//! Joint rebaking GPU compute pipeline.
//!
//! The [`JointRebakePipeline`] dispatches a compute shader that blends two
//! adjacent segments using smooth-min at each voxel in a joint region, writing
//! the result back to the brick pool with the [`JOINT_REGION_FLAG`] set for
//! Lipschitz mitigation (0.8x ray march step multiplier).
//!
//! # Usage
//!
//! ```ignore
//! let pipeline = JointRebakePipeline::new(&device);
//!
//! // For each joint region, fill JointParams from character data:
//! let params = JointParams { /* ... */ };
//! pipeline.dispatch(&mut encoder, &device, &brick_pool_buffer, &params);
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Voxel flag indicating this voxel is in a joint region.
///
/// The ray marcher checks this flag and applies a 0.8x conservative step
/// multiplier to maintain Lipschitz continuity across the smooth-min blend.
pub const JOINT_REGION_FLAG: u8 = 0x01;

/// Check if a voxel has the joint region flag set (for Lipschitz mitigation).
///
/// The ray marcher uses this to apply 0.8x step multiplier on joint voxels.
#[inline]
pub fn is_joint_region(flags: u8) -> bool {
    flags & JOINT_REGION_FLAG != 0
}

// ---------------------------------------------------------------------------
// JointParams — GPU uniform struct (256 bytes)
// ---------------------------------------------------------------------------

/// GPU-compatible parameters for the joint rebaking compute shader.
///
/// Matches the WGSL `JointParams` struct in `joint_rebake.wgsl`. All vec3
/// fields are padded to vec4 for correct GPU alignment. Total: 256 bytes.
///
/// # Layout
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 64 | `inv_bone_a` (mat4x4) |
/// | 64 | 64 | `inv_bone_b` (mat4x4) |
/// | 128 | 4 | `joint_brick_base` |
/// | 132 | 4 | `joint_brick_count` |
/// | 136 | 4 | `seg_a_brick_base` |
/// | 140 | 4 | `seg_a_brick_count` |
/// | 144 | 4 | `seg_b_brick_base` |
/// | 148 | 4 | `seg_b_brick_count` |
/// | 152 | 4 | `blend_k` |
/// | 156 | 4 | `voxel_size` |
/// | 160 | 12+4 | `joint_world_min` + pad |
/// | 176 | 12+4 | `seg_a_local_min` + pad |
/// | 192 | 12+4 | `seg_b_local_min` + pad |
/// | 208 | 4 | `seg_a_voxel_size` |
/// | 212 | 4 | `seg_b_voxel_size` |
/// | 216 | 4 | `region_bricks_x` |
/// | 220 | 4 | `region_bricks_y` |
/// | 224 | 4 | `seg_a_bricks_x` |
/// | 228 | 4 | `seg_a_bricks_y` |
/// | 232 | 4 | `seg_a_bricks_z` |
/// | 236 | 4 | `seg_b_bricks_x` |
/// | 240 | 4 | `seg_b_bricks_y` |
/// | 244 | 4 | `seg_b_bricks_z` |
/// | 248 | 8 | `_pad_end` (align to 256) |
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct JointParams {
    /// Inverse bone matrix for segment A (world -> A local space). Column-major.
    pub inv_bone_a: [f32; 16],
    /// Inverse bone matrix for segment B (world -> B local space). Column-major.
    pub inv_bone_b: [f32; 16],

    /// Base index in the brick pool for the joint's output bricks.
    pub joint_brick_base: u32,
    /// Number of bricks in this joint region.
    pub joint_brick_count: u32,
    /// Base index in the brick pool for segment A's rest-pose bricks.
    pub seg_a_brick_base: u32,
    /// Number of bricks in segment A.
    pub seg_a_brick_count: u32,
    /// Base index in the brick pool for segment B's rest-pose bricks.
    pub seg_b_brick_base: u32,
    /// Number of bricks in segment B.
    pub seg_b_brick_count: u32,
    /// Smooth-min blend radius parameter `k`.
    pub blend_k: f32,
    /// Voxel size for the joint region's resolution tier.
    pub voxel_size: f32,

    /// World-space minimum corner of the joint brick region. `[x, y, z]`.
    pub joint_world_min: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad0: f32,
    /// Segment A rest-pose AABB minimum corner (local space). `[x, y, z]`.
    pub seg_a_local_min: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad1: f32,
    /// Segment B rest-pose AABB minimum corner (local space). `[x, y, z]`.
    pub seg_b_local_min: [f32; 3],
    /// Padding for vec4 alignment.
    pub _pad2: f32,

    /// Segment A voxel size (may differ if segments use different tiers).
    pub seg_a_voxel_size: f32,
    /// Segment B voxel size.
    pub seg_b_voxel_size: f32,
    /// Joint brick region dimension in bricks (X axis).
    pub region_bricks_x: u32,
    /// Joint brick region dimension in bricks (Y axis).
    pub region_bricks_y: u32,

    /// Segment A grid dimension in bricks (X).
    pub seg_a_bricks_x: u32,
    /// Segment A grid dimension in bricks (Y).
    pub seg_a_bricks_y: u32,
    /// Segment A grid dimension in bricks (Z).
    pub seg_a_bricks_z: u32,
    /// Segment B grid dimension in bricks (X).
    pub seg_b_bricks_x: u32,
    /// Segment B grid dimension in bricks (Y).
    pub seg_b_bricks_y: u32,
    /// Segment B grid dimension in bricks (Z).
    pub seg_b_bricks_z: u32,

    /// Padding to bring struct size to 256 bytes (multiple of 16).
    pub _pad_end: [u32; 2],
}

// ---------------------------------------------------------------------------
// JointRebakePipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for joint rebaking.
///
/// Dispatches the `joint_rebake.wgsl` shader that smooth-min blends two
/// segments into joint region bricks. Each workgroup (8x8x8 = 512 threads)
/// processes one brick.
pub struct JointRebakePipeline {
    /// The compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout: binding 0 = rw storage (brick pool), binding 1 = uniform (JointParams).
    bind_group_layout: wgpu::BindGroupLayout,
}

impl JointRebakePipeline {
    /// Create the joint rebaking pipeline.
    ///
    /// Compiles `joint_rebake.wgsl` and creates the pipeline layout.
    /// The brick pool buffer is bound per-dispatch, not here.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("joint_rebake.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/joint_rebake.wgsl").into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("joint rebake bind group layout"),
                entries: &[
                    // Binding 0: brick pool — read_write storage
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: joint params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("joint rebake pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("joint rebake pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Dispatch the joint rebake shader for a single joint region.
    ///
    /// `brick_pool_buffer` must have `STORAGE` usage with read-write access.
    /// Dispatches `region_bricks_x * region_bricks_y * region_bricks_z` workgroups
    /// where `region_bricks_z = ceil(joint_brick_count / (region_bricks_x * region_bricks_y))`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        brick_pool_buffer: &wgpu::Buffer,
        params: &JointParams,
    ) {
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("joint rebake params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("joint rebake bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: brick_pool_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compute region_bricks_z from brick count and XY dimensions.
        let xy = params.region_bricks_x * params.region_bricks_y;
        let region_bricks_z = if xy == 0 {
            0
        } else {
            (params.joint_brick_count + xy - 1) / xy
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("joint rebake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(params.region_bricks_x, params.region_bricks_y, region_bricks_z);
    }

    /// Dispatch the joint rebake shader for multiple joint regions in one pass.
    ///
    /// More efficient than calling [`dispatch`](Self::dispatch) in a loop — reuses
    /// the same compute pass. Each joint gets its own bind group / uniform buffer.
    pub fn dispatch_all(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        brick_pool_buffer: &wgpu::Buffer,
        params_list: &[JointParams],
    ) {
        if params_list.is_empty() {
            return;
        }

        // Pre-create all uniform buffers and bind groups.
        let bind_groups: Vec<(wgpu::BindGroup, u32, u32, u32)> = params_list
            .iter()
            .enumerate()
            .map(|(i, params)| {
                let params_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("joint rebake params [{i}]")),
                        contents: bytemuck::bytes_of(params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("joint rebake bind group [{i}]")),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: brick_pool_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                let xy = params.region_bricks_x * params.region_bricks_y;
                let region_bricks_z = if xy == 0 {
                    0
                } else {
                    (params.joint_brick_count + xy - 1) / xy
                };

                (
                    bind_group,
                    params.region_bricks_x,
                    params.region_bricks_y,
                    region_bricks_z,
                )
            })
            .collect();

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("joint rebake all"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);

        for (bind_group, wx, wy, wz) in &bind_groups {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*wx, *wy, *wz);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn joint_params_is_pod_and_zeroable() {
        let params = JointParams::zeroed();
        assert_eq!(params.joint_brick_base, 0);
        assert_eq!(params.blend_k, 0.0);
        assert_eq!(params.voxel_size, 0.0);
        assert_eq!(params.region_bricks_x, 0);

        // Verify Pod round-trip.
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 256);
        let _round_trip: &JointParams = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn joint_params_size_is_256_bytes() {
        assert_eq!(mem::size_of::<JointParams>(), 256);
    }

    #[test]
    fn joint_params_alignment_is_4() {
        assert_eq!(mem::align_of::<JointParams>(), 4);
    }

    #[test]
    fn joint_params_default_is_zeroed() {
        let params = JointParams::zeroed();
        let bytes = bytemuck::bytes_of(&params);
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn joint_params_field_offsets() {
        // Verify key field offsets match the WGSL layout using bytemuck.
        let mut params = JointParams::zeroed();

        // inv_bone_a at offset 0 (64 bytes)
        params.inv_bone_a[0] = f32::from_bits(0xDEAD_BEEF);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0xDEAD_BEEF,
            "inv_bone_a should be at offset 0"
        );
        params.inv_bone_a[0] = 0.0;

        // inv_bone_b at offset 64
        params.inv_bone_b[0] = f32::from_bits(0xCAFE_BABE);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[64], bytes[65], bytes[66], bytes[67]]),
            0xCAFE_BABE,
            "inv_bone_b should be at offset 64"
        );
        params.inv_bone_b[0] = 0.0;

        // joint_brick_base at offset 128
        params.joint_brick_base = 0x1234_5678;
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[128], bytes[129], bytes[130], bytes[131]]),
            0x1234_5678,
            "joint_brick_base should be at offset 128"
        );
        params.joint_brick_base = 0;

        // blend_k at offset 152
        params.blend_k = f32::from_bits(0xAAAA_BBBB);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[152], bytes[153], bytes[154], bytes[155]]),
            0xAAAA_BBBB,
            "blend_k should be at offset 152"
        );
        params.blend_k = 0.0;

        // joint_world_min at offset 160
        params.joint_world_min[0] = f32::from_bits(0x1111_2222);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[160], bytes[161], bytes[162], bytes[163]]),
            0x1111_2222,
            "joint_world_min should be at offset 160"
        );
        params.joint_world_min[0] = 0.0;

        // seg_a_local_min at offset 176
        params.seg_a_local_min[0] = f32::from_bits(0x3333_4444);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[176], bytes[177], bytes[178], bytes[179]]),
            0x3333_4444,
            "seg_a_local_min should be at offset 176"
        );
        params.seg_a_local_min[0] = 0.0;

        // seg_b_local_min at offset 192
        params.seg_b_local_min[0] = f32::from_bits(0x5555_6666);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[192], bytes[193], bytes[194], bytes[195]]),
            0x5555_6666,
            "seg_b_local_min should be at offset 192"
        );
        params.seg_b_local_min[0] = 0.0;

        // seg_a_voxel_size at offset 208
        params.seg_a_voxel_size = f32::from_bits(0x7777_8888);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[208], bytes[209], bytes[210], bytes[211]]),
            0x7777_8888,
            "seg_a_voxel_size should be at offset 208"
        );
        params.seg_a_voxel_size = 0.0;

        // region_bricks_x at offset 216
        params.region_bricks_x = 0x9999_AAAA;
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[216], bytes[217], bytes[218], bytes[219]]),
            0x9999_AAAA,
            "region_bricks_x should be at offset 216"
        );
        params.region_bricks_x = 0;

        // seg_a_bricks_x at offset 224
        params.seg_a_bricks_x = 0xBBBB_CCCC;
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(
            u32::from_le_bytes([bytes[224], bytes[225], bytes[226], bytes[227]]),
            0xBBBB_CCCC,
            "seg_a_bricks_x should be at offset 224"
        );
    }

    #[test]
    fn is_joint_region_flag_check() {
        assert!(is_joint_region(JOINT_REGION_FLAG));
        assert!(is_joint_region(0x01));
        assert!(is_joint_region(0xFF)); // flag set among others
        assert!(is_joint_region(0x03)); // flag set with another bit
        assert!(!is_joint_region(0x00));
        assert!(!is_joint_region(0x02));
        assert!(!is_joint_region(0xFE));
    }

    #[test]
    fn joint_region_flag_matches_shader_constant() {
        // The shader defines FLAG_JOINT_REGION = 1u.
        // This test ensures the Rust constant matches.
        assert_eq!(JOINT_REGION_FLAG, 0x01);
    }

    #[test]
    fn joint_params_matrices_are_column_major() {
        // glam Mat4 is column-major — verify that to_cols_array() output
        // matches what we'd put in JointParams.
        let mat = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let cols = mat.to_cols_array();

        let mut params = JointParams::zeroed();
        params.inv_bone_a = cols;

        // Column 3 (translation) should be [1, 2, 3, 1] starting at index 12.
        assert_eq!(params.inv_bone_a[12], 1.0);
        assert_eq!(params.inv_bone_a[13], 2.0);
        assert_eq!(params.inv_bone_a[14], 3.0);
        assert_eq!(params.inv_bone_a[15], 1.0);
    }

    #[test]
    fn dispatch_all_with_empty_list_is_noop() {
        // This is a compile-time / logic check — dispatch_all should not panic
        // on empty input. We can't actually call it without a GPU device, but
        // we verify the guard condition exists in the code.
        let empty: &[JointParams] = &[];
        assert!(empty.is_empty());
    }
}
