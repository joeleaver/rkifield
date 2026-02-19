//! Skeleton extraction from glTF and automatic segmentation.
//!
//! Extracts bone hierarchy, animations, and per-vertex weights from glTF files,
//! then segments the mesh into rigid body parts and joint regions for the
//! engine's segmented animation system.

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use std::collections::{HashMap, HashSet};

use rkf_animation::clip::{AnimationClip, BoneChannel, Keyframe};
use rkf_animation::skeleton::{Bone, Skeleton};

use crate::mesh::MeshData;

/// Per-vertex skinning data extracted from glTF.
#[derive(Debug, Clone)]
pub struct VertexSkinning {
    /// Bone indices (up to 4 per vertex). -1 means unused.
    pub joints: Vec<[i32; 4]>,
    /// Bone weights (up to 4 per vertex, normalized to sum to 1).
    pub weights: Vec<[f32; 4]>,
}

/// Result of skeleton extraction from a glTF file.
#[derive(Debug)]
pub struct SkeletonExtraction {
    /// The extracted skeleton hierarchy.
    pub skeleton: Skeleton,
    /// Per-vertex skinning weights.
    pub skinning: VertexSkinning,
    /// Extracted animation clips.
    pub clips: Vec<AnimationClip>,
}

/// Result of automatic mesh segmentation.
#[derive(Debug)]
pub struct SegmentationResult {
    /// Dominant bone index per vertex.
    pub vertex_bones: Vec<u32>,
    /// Per-triangle segment assignment (bone index of the segment, or `u32::MAX` for joint region).
    pub triangle_segments: Vec<u32>,
    /// Segment bone indices (unique bones that own at least one triangle).
    pub segment_bones: Vec<u32>,
    /// Joint region pairs: `(segment_a_bone, segment_b_bone)` for triangles with mixed dominant bones.
    pub joint_pairs: Vec<(u32, u32)>,
    /// Per-triangle: which joint pair index (or `u32::MAX` if not a joint triangle).
    pub triangle_joint_index: Vec<u32>,
}

/// Extract skeleton, skinning, and animation data from a glTF file.
///
/// Returns `None` if the file has no skins (not animated).
pub fn extract_skeleton(path: &str) -> Result<Option<SkeletonExtraction>> {
    let (document, buffers, _images) =
        gltf::import(path).with_context(|| format!("Failed to load glTF: {path}"))?;

    // Find the first skin
    let skin = match document.skins().next() {
        Some(s) => s,
        None => return Ok(None),
    };

    // Extract joints (bones)
    let joint_nodes: Vec<_> = skin.joints().collect();
    let joint_count = joint_nodes.len();

    // Build node index -> bone index map
    let mut node_to_bone: HashMap<usize, usize> = HashMap::new();
    for (bone_idx, node) in joint_nodes.iter().enumerate() {
        node_to_bone.insert(node.index(), bone_idx);
    }

    // Read inverse bind matrices
    let reader = skin.reader(|buf| Some(&buffers[buf.index()]));
    let inverse_binds: Vec<Mat4> = if let Some(ibm_iter) = reader.read_inverse_bind_matrices() {
        ibm_iter
            .map(|m| Mat4::from_cols_array_2d(&m))
            .collect::<Vec<_>>()
    } else {
        vec![Mat4::IDENTITY; joint_count]
    };

    // Build node parent map from document scene graph (children -> parent)
    let mut node_parent: HashMap<usize, usize> = HashMap::new();
    for node in document.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }

    // Build bones and hierarchy
    let mut bones = Vec::with_capacity(joint_count);
    let mut hierarchy = Vec::with_capacity(joint_count);

    for (bone_idx, node) in joint_nodes.iter().enumerate() {
        let (translation, rotation, scale) = node.transform().decomposed();
        let t = Vec3::from(translation);
        let r = Quat::from_array(rotation);
        let s = Vec3::from(scale);
        let bind_transform = Mat4::from_scale_rotation_translation(s, r, t);

        bones.push(Bone {
            name: node.name().unwrap_or("unnamed").to_string(),
            bind_transform,
            inverse_bind: inverse_binds[bone_idx],
        });

        // Walk up the scene graph to find nearest ancestor that is also a joint
        let mut current = node.index();
        let parent_bone = loop {
            match node_parent.get(&current) {
                Some(&parent_node) => {
                    if let Some(&parent_bi) = node_to_bone.get(&parent_node) {
                        break parent_bi as i32;
                    }
                    current = parent_node;
                }
                None => break -1i32,
            }
        };
        hierarchy.push(parent_bone);
    }

    let skeleton = Skeleton::new(bones, hierarchy)
        .map_err(|e| anyhow::anyhow!("Failed to construct skeleton from glTF joints: {e}"))?;

    // Extract per-vertex skinning
    let skinning = extract_skinning(&document, &buffers)?;

    // Extract animation clips
    let clips = extract_animations(&document, &buffers, &node_to_bone)?;

    Ok(Some(SkeletonExtraction {
        skeleton,
        skinning,
        clips,
    }))
}

/// Extract per-vertex skinning weights from all mesh primitives with joints.
fn extract_skinning(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<VertexSkinning> {
    let mut all_joints = Vec::new();
    let mut all_weights = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buf| Some(&buffers[buf.index()]));

            if let Some(joints_reader) = reader.read_joints(0) {
                for j in joints_reader.into_u16() {
                    all_joints.push([j[0] as i32, j[1] as i32, j[2] as i32, j[3] as i32]);
                }
            }

            if let Some(weights_reader) = reader.read_weights(0) {
                for w in weights_reader.into_f32() {
                    all_weights.push(w);
                }
            }
        }
    }

    // Pad if one is shorter (shouldn't happen in valid glTF)
    let len = all_joints.len().max(all_weights.len());
    all_joints.resize(len, [-1, -1, -1, -1]);
    all_weights.resize(len, [0.0, 0.0, 0.0, 0.0]);

    Ok(VertexSkinning {
        joints: all_joints,
        weights: all_weights,
    })
}

/// Extract animation clips from the glTF document.
fn extract_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    node_to_bone: &HashMap<usize, usize>,
) -> Result<Vec<AnimationClip>> {
    let mut clips = Vec::new();

    for anim in document.animations() {
        let name = anim.name().unwrap_or("unnamed").to_string();

        // Accumulate keyframe data per bone: time -> (pos, rot, scale)
        let mut bone_channels_map: HashMap<u32, Vec<(f32, Option<Vec3>, Option<Quat>, Option<Vec3>)>> =
            HashMap::new();

        for channel in anim.channels() {
            let target_node = channel.target().node().index();
            let bone_idx = match node_to_bone.get(&target_node) {
                Some(&bi) => bi as u32,
                None => continue, // Not a joint node
            };

            let reader = channel.reader(|buf| Some(&buffers[buf.index()]));
            let timestamps: Vec<f32> = reader
                .read_inputs()
                .map(|iter| iter.collect())
                .unwrap_or_default();

            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(translations)) => {
                    let values: Vec<_> = translations.collect();
                    let entries = bone_channels_map.entry(bone_idx).or_default();
                    for (i, t) in timestamps.iter().enumerate() {
                        if i < values.len() {
                            let pos = Vec3::from(values[i]);
                            if let Some(entry) =
                                entries.iter_mut().find(|e| (e.0 - t).abs() < 1e-6)
                            {
                                entry.1 = Some(pos);
                            } else {
                                entries.push((*t, Some(pos), None, None));
                            }
                        }
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(rotations)) => {
                    let values: Vec<_> = rotations.into_f32().collect();
                    let entries = bone_channels_map.entry(bone_idx).or_default();
                    for (i, t) in timestamps.iter().enumerate() {
                        if i < values.len() {
                            let rot = Quat::from_array(values[i]);
                            if let Some(entry) =
                                entries.iter_mut().find(|e| (e.0 - t).abs() < 1e-6)
                            {
                                entry.2 = Some(rot);
                            } else {
                                entries.push((*t, None, Some(rot), None));
                            }
                        }
                    }
                }
                Some(gltf::animation::util::ReadOutputs::Scales(scales)) => {
                    let values: Vec<_> = scales.collect();
                    let entries = bone_channels_map.entry(bone_idx).or_default();
                    for (i, t) in timestamps.iter().enumerate() {
                        if i < values.len() {
                            let scale = Vec3::from(values[i]);
                            if let Some(entry) =
                                entries.iter_mut().find(|e| (e.0 - t).abs() < 1e-6)
                            {
                                entry.3 = Some(scale);
                            } else {
                                entries.push((*t, None, None, Some(scale)));
                            }
                        }
                    }
                }
                _ => {} // MorphTargetWeights handled separately
            }
        }

        // Convert map to BoneChannels
        let mut channels = Vec::new();
        for (bone_idx, mut entries) in bone_channels_map {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let keyframes: Vec<Keyframe> = entries
                .iter()
                .map(|(time, pos, rot, scale)| Keyframe {
                    time: *time,
                    position: pos.unwrap_or(Vec3::ZERO),
                    rotation: rot.unwrap_or(Quat::IDENTITY),
                    scale: scale.unwrap_or(Vec3::ONE),
                })
                .collect();

            channels.push(BoneChannel {
                bone_index: bone_idx,
                keyframes,
            });
        }

        let duration = channels
            .iter()
            .flat_map(|c| c.keyframes.last())
            .map(|kf| kf.time)
            .fold(0.0f32, f32::max);

        clips.push(AnimationClip::new(name, duration, channels));
    }

    Ok(clips)
}

/// Compute dominant bone per vertex (highest weight).
pub fn compute_dominant_bones(skinning: &VertexSkinning) -> Vec<u32> {
    skinning
        .weights
        .iter()
        .zip(&skinning.joints)
        .map(|(w, j)| {
            let mut best_idx = 0u32;
            let mut best_weight = 0.0f32;
            for i in 0..4 {
                if j[i] >= 0 && w[i] > best_weight {
                    best_weight = w[i];
                    best_idx = j[i] as u32;
                }
            }
            best_idx
        })
        .collect()
}

/// Automatically segment a mesh based on per-vertex skinning weights.
///
/// - Triangles where all 3 vertices have the same dominant bone -> rigid segment
/// - Triangles with mixed dominant bones -> joint region between those segments
pub fn auto_segment(mesh: &MeshData, skinning: &VertexSkinning) -> SegmentationResult {
    let vertex_bones = compute_dominant_bones(skinning);
    let tri_count = mesh.triangle_count();

    let mut triangle_segments = Vec::with_capacity(tri_count);
    let mut triangle_joint_index = Vec::with_capacity(tri_count);
    let mut joint_pairs_set: Vec<(u32, u32)> = Vec::new();
    let mut segment_bones_set: HashSet<u32> = HashSet::new();

    for i in 0..tri_count {
        let base = i * 3;
        let i0 = mesh.indices[base] as usize;
        let i1 = mesh.indices[base + 1] as usize;
        let i2 = mesh.indices[base + 2] as usize;

        let b0 = if i0 < vertex_bones.len() {
            vertex_bones[i0]
        } else {
            0
        };
        let b1 = if i1 < vertex_bones.len() {
            vertex_bones[i1]
        } else {
            0
        };
        let b2 = if i2 < vertex_bones.len() {
            vertex_bones[i2]
        } else {
            0
        };

        if b0 == b1 && b1 == b2 {
            // Rigid segment
            triangle_segments.push(b0);
            triangle_joint_index.push(u32::MAX);
            segment_bones_set.insert(b0);
        } else {
            // Joint region: find the two different bones (use min/max for canonical pair)
            let mut bones = [b0, b1, b2];
            bones.sort();
            let pair = (
                bones[0],
                *bones.iter().find(|&&b| b != bones[0]).unwrap_or(&bones[0]),
            );

            // Find or create joint pair
            let jp_idx = joint_pairs_set
                .iter()
                .position(|p| *p == pair)
                .unwrap_or_else(|| {
                    joint_pairs_set.push(pair);
                    joint_pairs_set.len() - 1
                });

            triangle_segments.push(u32::MAX); // joint, not a rigid segment
            triangle_joint_index.push(jp_idx as u32);
            segment_bones_set.insert(pair.0);
            segment_bones_set.insert(pair.1);
        }
    }

    let mut segment_bones: Vec<u32> = segment_bones_set.into_iter().collect();
    segment_bones.sort();

    SegmentationResult {
        vertex_bones,
        triangle_segments,
        segment_bones,
        joint_pairs: joint_pairs_set,
        triangle_joint_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Helper: build a MeshData with given vertex count, indices, etc.
    fn make_mesh(vertex_count: usize, indices: Vec<u32>) -> MeshData {
        MeshData {
            positions: vec![Vec3::ZERO; vertex_count],
            normals: vec![Vec3::Y; vertex_count],
            uvs: Vec::new(),
            indices,
            material_indices: Vec::new(),
            materials: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ONE,
        }
    }

    #[test]
    fn compute_dominant_bones_selects_highest_weight() {
        let skinning = VertexSkinning {
            joints: vec![[0, 1, 2, 3]],
            weights: vec![[0.1, 0.5, 0.3, 0.1]],
        };
        let dominant = compute_dominant_bones(&skinning);
        assert_eq!(dominant.len(), 1);
        assert_eq!(dominant[0], 1); // bone 1 has highest weight 0.5
    }

    #[test]
    fn compute_dominant_bones_ignores_negative_joints() {
        let skinning = VertexSkinning {
            joints: vec![[2, -1, -1, -1]],
            weights: vec![[0.3, 0.7, 0.0, 0.0]],
        };
        let dominant = compute_dominant_bones(&skinning);
        assert_eq!(dominant[0], 2); // bone -1 is ignored despite weight 0.7
    }

    #[test]
    fn auto_segment_uniform_mesh() {
        // All 4 vertices assigned to bone 5; two triangles -> all rigid, no joints
        let mesh = make_mesh(4, vec![0, 1, 2, 1, 3, 2]);
        let skinning = VertexSkinning {
            joints: vec![[5, -1, -1, -1]; 4],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 4],
        };
        let result = auto_segment(&mesh, &skinning);

        assert_eq!(result.triangle_segments.len(), 2);
        assert_eq!(result.triangle_segments[0], 5);
        assert_eq!(result.triangle_segments[1], 5);
        assert!(result.joint_pairs.is_empty());
        assert_eq!(result.triangle_joint_index[0], u32::MAX);
        assert_eq!(result.triangle_joint_index[1], u32::MAX);
    }

    #[test]
    fn auto_segment_mixed_bones() {
        // Triangle with vertices on different bones -> joint region
        let mesh = make_mesh(3, vec![0, 1, 2]);
        let skinning = VertexSkinning {
            joints: vec![[0, -1, -1, -1], [1, -1, -1, -1], [0, -1, -1, -1]],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        };
        let result = auto_segment(&mesh, &skinning);

        assert_eq!(result.triangle_segments[0], u32::MAX); // joint, not rigid
        assert_eq!(result.joint_pairs.len(), 1);
        assert_eq!(result.joint_pairs[0], (0, 1));
        assert_eq!(result.triangle_joint_index[0], 0);
    }

    #[test]
    fn auto_segment_joint_pair_dedup() {
        // Two triangles both spanning bones 2 and 3 -> single joint pair, not duplicated
        let mesh = make_mesh(6, vec![0, 1, 2, 3, 4, 5]);
        let skinning = VertexSkinning {
            joints: vec![
                [2, -1, -1, -1],
                [3, -1, -1, -1],
                [2, -1, -1, -1],
                [3, -1, -1, -1],
                [2, -1, -1, -1],
                [3, -1, -1, -1],
            ],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        };
        let result = auto_segment(&mesh, &skinning);

        assert_eq!(result.joint_pairs.len(), 1); // deduplicated
        assert_eq!(result.joint_pairs[0], (2, 3));
        assert_eq!(result.triangle_joint_index[0], 0);
        assert_eq!(result.triangle_joint_index[1], 0); // same pair index
    }

    #[test]
    fn vertex_skinning_empty() {
        let skinning = VertexSkinning {
            joints: Vec::new(),
            weights: Vec::new(),
        };
        let dominant = compute_dominant_bones(&skinning);
        assert!(dominant.is_empty());
    }

    #[test]
    fn segmentation_result_segment_bones_sorted() {
        // Vertices on bones 7, 3, and 1 -> segment_bones should be sorted [1, 3, 7]
        let mesh = make_mesh(9, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let skinning = VertexSkinning {
            joints: vec![
                [7, -1, -1, -1],
                [7, -1, -1, -1],
                [7, -1, -1, -1],
                [3, -1, -1, -1],
                [3, -1, -1, -1],
                [3, -1, -1, -1],
                [1, -1, -1, -1],
                [1, -1, -1, -1],
                [1, -1, -1, -1],
            ],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 9],
        };
        let result = auto_segment(&mesh, &skinning);

        assert_eq!(result.segment_bones, vec![1, 3, 7]);
    }

    #[test]
    fn auto_segment_triangle_count_matches() {
        // 4 triangles -> output arrays must all have length 4
        let mesh = make_mesh(6, vec![0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]);
        let skinning = VertexSkinning {
            joints: vec![[0, -1, -1, -1]; 6],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 6],
        };
        let result = auto_segment(&mesh, &skinning);

        assert_eq!(result.triangle_segments.len(), 4);
        assert_eq!(result.triangle_joint_index.len(), 4);
    }

    #[test]
    fn compute_dominant_bones_multiple_vertices() {
        let skinning = VertexSkinning {
            joints: vec![
                [0, 1, 2, -1],
                [3, 4, -1, -1],
                [0, 5, 6, 7],
            ],
            weights: vec![
                [0.2, 0.6, 0.2, 0.0], // bone 1 wins
                [0.4, 0.6, 0.0, 0.0], // bone 4 wins
                [0.1, 0.1, 0.1, 0.7], // bone 7 wins
            ],
        };
        let dominant = compute_dominant_bones(&skinning);
        assert_eq!(dominant, vec![1, 4, 7]);
    }

    #[test]
    fn auto_segment_mixed_and_rigid() {
        // First triangle: all bone 0 (rigid). Second triangle: bones 0 and 1 (joint).
        let mesh = make_mesh(5, vec![0, 1, 2, 2, 3, 4]);
        let skinning = VertexSkinning {
            joints: vec![
                [0, -1, -1, -1],
                [0, -1, -1, -1],
                [0, -1, -1, -1],
                [1, -1, -1, -1],
                [1, -1, -1, -1],
            ],
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 5],
        };
        let result = auto_segment(&mesh, &skinning);

        // Triangle 0: rigid (bone 0)
        assert_eq!(result.triangle_segments[0], 0);
        assert_eq!(result.triangle_joint_index[0], u32::MAX);

        // Triangle 1: joint (bones 0 and 1)
        assert_eq!(result.triangle_segments[1], u32::MAX);
        assert_eq!(result.triangle_joint_index[1], 0);
        assert_eq!(result.joint_pairs[0], (0, 1));

        // Both bones present in segment_bones
        assert!(result.segment_bones.contains(&0));
        assert!(result.segment_bones.contains(&1));
    }
}
