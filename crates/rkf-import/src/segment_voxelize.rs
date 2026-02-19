//! Per-segment voxelization and joint region extraction.
//!
//! After automatic segmentation, each rigid segment's triangles are isolated
//! and voxelized independently. Joint regions between segments are also
//! voxelized with overlapping segments for smooth-min blending.

use glam::Vec3;
use rkf_animation::segment::{JointRegion, Segment};

use crate::mesh::MeshData;
use crate::skeleton_extract::SegmentationResult;
use crate::voxelize::{voxelize_mesh, VoxelizeConfig, VoxelizeResult};

/// Result of per-segment voxelization.
pub struct SegmentVoxelization {
    /// Voxelized segments with brick data.
    pub segments: Vec<SegmentData>,
    /// Voxelized joint regions.
    pub joints: Vec<JointData>,
}

/// A voxelized segment.
pub struct SegmentData {
    /// Engine segment descriptor.
    pub segment: Segment,
    /// The voxelization result for this segment.
    pub voxelize_result: VoxelizeResult,
}

/// A voxelized joint region.
pub struct JointData {
    /// Engine joint region descriptor.
    pub joint: JointRegion,
    /// The voxelization result for this joint region.
    pub voxelize_result: VoxelizeResult,
}

/// Extract a sub-mesh containing only the specified triangles.
fn extract_submesh(mesh: &MeshData, triangle_mask: &[bool]) -> MeshData {
    // Collect all referenced vertex indices
    let mut vertex_used = vec![false; mesh.positions.len()];
    let mut new_indices = Vec::new();
    let mut new_material_indices = Vec::new();

    for (tri_idx, &included) in triangle_mask.iter().enumerate() {
        if !included {
            continue;
        }
        let base = tri_idx * 3;
        for k in 0..3 {
            let vi = mesh.indices[base + k] as usize;
            vertex_used[vi] = true;
        }
    }

    // Build old-to-new vertex index map
    let mut old_to_new = vec![0u32; mesh.positions.len()];
    let mut new_positions = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_uvs = Vec::new();
    let mut new_idx = 0u32;

    for (old_idx, &used) in vertex_used.iter().enumerate() {
        if used {
            old_to_new[old_idx] = new_idx;
            new_positions.push(mesh.positions[old_idx]);
            if old_idx < mesh.normals.len() {
                new_normals.push(mesh.normals[old_idx]);
            }
            if old_idx < mesh.uvs.len() {
                new_uvs.push(mesh.uvs[old_idx]);
            }
            new_idx += 1;
        }
    }

    // Remap indices
    for (tri_idx, &included) in triangle_mask.iter().enumerate() {
        if !included {
            continue;
        }
        let base = tri_idx * 3;
        for k in 0..3 {
            new_indices.push(old_to_new[mesh.indices[base + k] as usize]);
        }
        if tri_idx < mesh.material_indices.len() {
            new_material_indices.push(mesh.material_indices[tri_idx]);
        }
    }

    // Compute bounds
    let (bounds_min, bounds_max) = if new_positions.is_empty() {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        let mut bmin = Vec3::splat(f32::MAX);
        let mut bmax = Vec3::splat(f32::MIN);
        for &p in &new_positions {
            bmin = bmin.min(p);
            bmax = bmax.max(p);
        }
        (bmin, bmax)
    };

    MeshData {
        positions: new_positions,
        normals: new_normals,
        uvs: new_uvs,
        indices: new_indices,
        material_indices: new_material_indices,
        materials: mesh.materials.clone(),
        bounds_min,
        bounds_max,
    }
}

/// Voxelize each segment and joint region independently.
///
/// For each rigid segment: extract its triangles, voxelize, create [`Segment`] descriptor.
/// For each joint pair: extract joint triangles plus adjacent segment triangles from both
/// sides, voxelize, compute `blend_k` from bone distance heuristic, create [`JointRegion`]
/// descriptor.
///
/// Brick offsets are assigned sequentially: all segment bricks first, then joint bricks.
pub fn voxelize_segments(
    mesh: &MeshData,
    segmentation: &SegmentationResult,
    config: &VoxelizeConfig,
) -> SegmentVoxelization {
    let tri_count = mesh.triangle_count();
    let mut segments = Vec::new();
    let mut joints = Vec::new();
    let mut brick_offset = 0u32;

    // Voxelize each rigid segment
    for &bone_idx in &segmentation.segment_bones {
        let mask: Vec<bool> = (0..tri_count)
            .map(|i| segmentation.triangle_segments[i] == bone_idx)
            .collect();

        let submesh = extract_submesh(mesh, &mask);
        if submesh.triangle_count() == 0 {
            continue;
        }

        let result = voxelize_mesh(&submesh, config);
        let segment = Segment {
            bone_index: bone_idx,
            brick_start: brick_offset,
            brick_count: result.brick_count,
            rest_aabb: result.aabb,
        };
        brick_offset += result.brick_count;

        segments.push(SegmentData {
            segment,
            voxelize_result: result,
        });
    }

    // Voxelize each joint region
    for (jp_idx, &(bone_a, bone_b)) in segmentation.joint_pairs.iter().enumerate() {
        // Include joint triangles + nearby segment triangles from both sides
        let mask: Vec<bool> = (0..tri_count)
            .map(|i| {
                segmentation.triangle_joint_index[i] == jp_idx as u32
                    || segmentation.triangle_segments[i] == bone_a
                    || segmentation.triangle_segments[i] == bone_b
            })
            .collect();

        let submesh = extract_submesh(mesh, &mask);
        if submesh.triangle_count() == 0 {
            continue;
        }

        let result = voxelize_mesh(&submesh, config);

        // Find segment indices for this joint pair
        let seg_a_idx = segments
            .iter()
            .position(|s| s.segment.bone_index == bone_a)
            .unwrap_or(0) as u32;
        let seg_b_idx = segments
            .iter()
            .position(|s| s.segment.bone_index == bone_b)
            .unwrap_or(0) as u32;

        // Compute blend_k from AABB size heuristic (smaller region -> sharper blend)
        let extent = result.aabb.max - result.aabb.min;
        let min_extent = extent.x.min(extent.y).min(extent.z);
        let blend_k = (min_extent * 0.15).clamp(0.02, 0.08);

        let joint = JointRegion {
            segment_a: seg_a_idx,
            segment_b: seg_b_idx,
            bone_index: bone_b, // child bone
            brick_start: brick_offset,
            brick_count: result.brick_count,
            rest_aabb: result.aabb,
            blend_k,
        };
        brick_offset += result.brick_count;

        joints.push(JointData {
            joint,
            voxelize_result: result,
        });
    }

    SegmentVoxelization { segments, joints }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ImportMaterial;
    use crate::skeleton_extract::SegmentationResult;
    use glam::Vec3;

    /// Helper: build a closed tetrahedron mesh for voxelization tests.
    /// Large enough to produce bricks at tier 1 (2cm voxels).
    fn make_tetrahedron(scale: f32) -> MeshData {
        let v0 = Vec3::new(0.0, 0.0, 0.0) * scale;
        let v1 = Vec3::new(1.0, 0.0, 0.0) * scale;
        let v2 = Vec3::new(0.5, 1.0, 0.0) * scale;
        let v3 = Vec3::new(0.5, 0.3, 0.8) * scale;

        let bounds_min = v0.min(v1).min(v2).min(v3);
        let bounds_max = v0.max(v1).max(v2).max(v3);

        MeshData {
            positions: vec![v0, v1, v2, v3],
            normals: vec![Vec3::Y; 4],
            uvs: Vec::new(),
            indices: vec![
                0, 2, 1, // bottom
                0, 1, 3, // front
                1, 2, 3, // right
                2, 0, 3, // left
            ],
            material_indices: vec![0, 0, 0, 0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min,
            bounds_max,
        }
    }

    /// Helper: build a two-tetrahedron mesh with 8 vertices where
    /// the first 4 vertices belong to bone A and the last 4 to bone B.
    fn make_two_segment_mesh() -> MeshData {
        // Segment A: tetrahedron centered around x=-1
        let a0 = Vec3::new(-1.5, 0.0, 0.0);
        let a1 = Vec3::new(-0.5, 0.0, 0.0);
        let a2 = Vec3::new(-1.0, 1.0, 0.0);
        let a3 = Vec3::new(-1.0, 0.3, 0.8);

        // Segment B: tetrahedron centered around x=+1
        let b0 = Vec3::new(0.5, 0.0, 0.0);
        let b1 = Vec3::new(1.5, 0.0, 0.0);
        let b2 = Vec3::new(1.0, 1.0, 0.0);
        let b3 = Vec3::new(1.0, 0.3, 0.8);

        let positions = vec![a0, a1, a2, a3, b0, b1, b2, b3];
        let bounds_min = positions.iter().copied().reduce(Vec3::min).unwrap();
        let bounds_max = positions.iter().copied().reduce(Vec3::max).unwrap();

        MeshData {
            positions,
            normals: vec![Vec3::Y; 8],
            uvs: Vec::new(),
            // Segment A: 4 triangles (indices 0-3), Segment B: 4 triangles (indices 4-7)
            indices: vec![
                0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3, // segment A
                4, 6, 5, 4, 5, 7, 5, 6, 7, 6, 4, 7, // segment B
            ],
            material_indices: vec![0; 8],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min,
            bounds_max,
        }
    }

    // ---- extract_submesh tests ----

    #[test]
    fn extract_submesh_all_included() {
        let mesh = make_tetrahedron(1.0);
        let tri_count = mesh.triangle_count();
        let mask = vec![true; tri_count];
        let sub = extract_submesh(&mesh, &mask);

        assert_eq!(sub.positions.len(), mesh.positions.len());
        assert_eq!(sub.triangle_count(), mesh.triangle_count());
        assert_eq!(sub.normals.len(), mesh.normals.len());
    }

    #[test]
    fn extract_submesh_none_included() {
        let mesh = make_tetrahedron(1.0);
        let tri_count = mesh.triangle_count();
        let mask = vec![false; tri_count];
        let sub = extract_submesh(&mesh, &mask);

        assert_eq!(sub.positions.len(), 0);
        assert_eq!(sub.triangle_count(), 0);
        assert_eq!(sub.indices.len(), 0);
        assert_eq!(sub.bounds_min, Vec3::ZERO);
        assert_eq!(sub.bounds_max, Vec3::ZERO);
    }

    #[test]
    fn extract_submesh_partial() {
        let mesh = make_two_segment_mesh();
        assert_eq!(mesh.triangle_count(), 8);

        // Include only the first 4 triangles (segment A)
        let mask: Vec<bool> = (0..8).map(|i| i < 4).collect();
        let sub = extract_submesh(&mesh, &mask);

        assert_eq!(sub.triangle_count(), 4);
        // Only vertices 0-3 should be included (segment A)
        assert_eq!(sub.positions.len(), 4);
        // Indices should be remapped to 0-based
        for &idx in &sub.indices {
            assert!(idx < 4, "remapped index {idx} should be < 4");
        }
        // Bounds should only cover segment A
        assert!(sub.bounds_max.x <= 0.0, "segment A max x should be <= 0");
    }

    // ---- voxelize_segments tests ----

    #[test]
    fn voxelize_segments_single_segment() {
        let mesh = make_tetrahedron(1.0);
        let tri_count = mesh.triangle_count();

        // Uniform segmentation: all triangles belong to bone 0
        let segmentation = SegmentationResult {
            vertex_bones: vec![0; mesh.positions.len()],
            triangle_segments: vec![0; tri_count],
            segment_bones: vec![0],
            joint_pairs: Vec::new(),
            triangle_joint_index: vec![u32::MAX; tri_count],
        };

        let config = VoxelizeConfig {
            tier: 0,
            narrow_band_bricks: 2,
            compute_color: false,
        };

        let result = voxelize_segments(&mesh, &segmentation, &config);

        assert_eq!(result.segments.len(), 1, "should have exactly 1 segment");
        assert_eq!(result.joints.len(), 0, "should have 0 joints");
        assert_eq!(result.segments[0].segment.bone_index, 0);
        assert!(
            result.segments[0].segment.brick_count > 0,
            "segment should have bricks"
        );
        assert_eq!(result.segments[0].segment.brick_start, 0);
    }

    #[test]
    fn voxelize_segments_with_joint() {
        let mesh = make_two_segment_mesh();
        let tri_count = mesh.triangle_count();

        // First 4 triangles: bone 0, last 4: bone 1
        // But make one triangle a joint (mixed bones) to test joint voxelization
        let mut triangle_segments = vec![0u32; tri_count];
        let mut triangle_joint_index = vec![u32::MAX; tri_count];

        // Triangles 0-3: bone 0
        // Triangles 4-7: bone 1
        for i in 4..8 {
            triangle_segments[i] = 1;
        }

        // Make triangle 3 a joint triangle (between bone 0 and 1)
        triangle_segments[3] = u32::MAX;
        triangle_joint_index[3] = 0;

        let segmentation = SegmentationResult {
            vertex_bones: vec![0, 0, 0, 0, 1, 1, 1, 1],
            triangle_segments,
            segment_bones: vec![0, 1],
            joint_pairs: vec![(0, 1)],
            triangle_joint_index,
        };

        let config = VoxelizeConfig {
            tier: 0,
            narrow_band_bricks: 2,
            compute_color: false,
        };

        let result = voxelize_segments(&mesh, &segmentation, &config);

        assert_eq!(result.segments.len(), 2, "should have 2 segments");
        assert!(result.joints.len() >= 1, "should have at least 1 joint");

        // Check segment bone indices
        assert_eq!(result.segments[0].segment.bone_index, 0);
        assert_eq!(result.segments[1].segment.bone_index, 1);

        // Joint should reference the two segments
        let j = &result.joints[0].joint;
        assert_eq!(j.segment_a, 0);
        assert_eq!(j.segment_b, 1);
        assert_eq!(j.bone_index, 1); // child bone

        // blend_k should be in valid range
        assert!(j.blend_k >= 0.02 && j.blend_k <= 0.08, "blend_k={}", j.blend_k);
    }

    #[test]
    fn segment_data_brick_offset_sequential() {
        let mesh = make_two_segment_mesh();
        let tri_count = mesh.triangle_count();

        // Two segments, no joints
        let mut triangle_segments = vec![0u32; tri_count];
        for i in 4..8 {
            triangle_segments[i] = 1;
        }

        let segmentation = SegmentationResult {
            vertex_bones: vec![0, 0, 0, 0, 1, 1, 1, 1],
            triangle_segments,
            segment_bones: vec![0, 1],
            joint_pairs: Vec::new(),
            triangle_joint_index: vec![u32::MAX; tri_count],
        };

        let config = VoxelizeConfig {
            tier: 0,
            narrow_band_bricks: 2,
            compute_color: false,
        };

        let result = voxelize_segments(&mesh, &segmentation, &config);

        assert_eq!(result.segments.len(), 2);

        // First segment starts at 0
        assert_eq!(result.segments[0].segment.brick_start, 0);

        // Second segment starts after first
        let expected_start = result.segments[0].segment.brick_count;
        assert_eq!(
            result.segments[1].segment.brick_start, expected_start,
            "segment 1 should start at {} (after segment 0's {} bricks), got {}",
            expected_start,
            result.segments[0].segment.brick_count,
            result.segments[1].segment.brick_start
        );

        // No overlapping ranges
        let end_0 =
            result.segments[0].segment.brick_start + result.segments[0].segment.brick_count;
        let start_1 = result.segments[1].segment.brick_start;
        assert!(
            end_0 <= start_1,
            "segment ranges must not overlap: seg0 ends at {end_0}, seg1 starts at {start_1}"
        );
    }
}
