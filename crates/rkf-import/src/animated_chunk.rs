//! Animated .rkf asset format — extends the base chunk format with animation data.
//!
//! Layout:
//! ```text
//! [Standard .rkf chunk data]
//! [Animation header: 32 bytes]
//!   magic: b"ANIM"
//!   bone_count: u32
//!   segment_count: u32
//!   joint_count: u32
//!   clip_count: u32
//!   blend_shape_count: u32
//!   _reserved: [u32; 2]
//! [Bone data: per bone]
//!   name_len: u32, name_bytes: [u8; name_len]
//!   bind_transform: [f32; 16] (column-major Mat4)
//!   inverse_bind: [f32; 16]
//!   parent_index: i32
//! [Segment data: per segment]
//!   bone_index: u32, brick_start: u32, brick_count: u32
//!   aabb_min: [f32; 3], aabb_max: [f32; 3]
//! [Joint data: per joint]
//!   segment_a: u32, segment_b: u32, bone_index: u32
//!   brick_start: u32, brick_count: u32
//!   aabb_min: [f32; 3], aabb_max: [f32; 3]
//!   blend_k: f32
//! [Clip data: LZ4 compressed blob]
//!   uncompressed: per clip:
//!     name_len: u32, name_bytes, duration: f32, channel_count: u32
//!     per channel: bone_index: u32, keyframe_count: u32
//!       per keyframe: time: f32, pos: [f32;3], rot: [f32;4], scale: [f32;3]
//! [Blend shape data: per shape]
//!   name_len: u32, name_bytes
//!   brick_offset: u32, brick_count: u32
//!   aabb_min: [f32; 3], aabb_max: [f32; 3]
//! ```

use std::io::{Read, Write};

use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};

use rkf_animation::blend_shape::BlendShape;
use rkf_animation::clip::{AnimationClip, BoneChannel, Keyframe};
use rkf_animation::segment::{JointRegion, Segment};
use rkf_animation::skeleton::{Bone, Skeleton};
use rkf_core::aabb::Aabb;
use rkf_core::chunk::{load_chunk, save_chunk, Chunk};

/// Magic bytes identifying the animation extension.
const ANIM_MAGIC: [u8; 4] = *b"ANIM";

/// An animated .rkf asset with skeleton, segments, joints, clips, blend shapes.
#[derive(Debug)]
pub struct AnimatedAsset {
    /// The base chunk with voxel data for all segments + joints.
    pub chunk: Chunk,
    /// Skeleton hierarchy.
    pub skeleton: Skeleton,
    /// Rigid body segments.
    pub segments: Vec<Segment>,
    /// Joint blending regions.
    pub joints: Vec<JointRegion>,
    /// Animation clips.
    pub clips: Vec<AnimationClip>,
    /// Blend shape descriptors.
    pub blend_shapes: Vec<BlendShape>,
}

/// Save an animated asset to a writer.
pub fn save_animated_asset(asset: &AnimatedAsset, writer: &mut impl Write) -> Result<()> {
    // 1. Write standard chunk
    save_chunk(&asset.chunk, writer).context("failed to write chunk data")?;

    // 2. Animation header
    writer.write_all(&ANIM_MAGIC)?;
    write_u32(writer, asset.skeleton.bones.len() as u32)?;
    write_u32(writer, asset.segments.len() as u32)?;
    write_u32(writer, asset.joints.len() as u32)?;
    write_u32(writer, asset.clips.len() as u32)?;
    write_u32(writer, asset.blend_shapes.len() as u32)?;
    write_u32(writer, 0)?; // reserved
    write_u32(writer, 0)?; // reserved

    // 3. Bone data
    for (i, bone) in asset.skeleton.bones.iter().enumerate() {
        write_string(writer, &bone.name)?;
        write_mat4(writer, &bone.bind_transform)?;
        write_mat4(writer, &bone.inverse_bind)?;
        write_i32(writer, asset.skeleton.hierarchy[i])?;
    }

    // 4. Segment data
    for seg in &asset.segments {
        write_u32(writer, seg.bone_index)?;
        write_u32(writer, seg.brick_start)?;
        write_u32(writer, seg.brick_count)?;
        write_vec3(writer, seg.rest_aabb.min)?;
        write_vec3(writer, seg.rest_aabb.max)?;
    }

    // 5. Joint data
    for joint in &asset.joints {
        write_u32(writer, joint.segment_a)?;
        write_u32(writer, joint.segment_b)?;
        write_u32(writer, joint.bone_index)?;
        write_u32(writer, joint.brick_start)?;
        write_u32(writer, joint.brick_count)?;
        write_vec3(writer, joint.rest_aabb.min)?;
        write_vec3(writer, joint.rest_aabb.max)?;
        write_f32(writer, joint.blend_k)?;
    }

    // 6. Clip data (LZ4 compressed)
    let mut clip_buf = Vec::new();
    for clip in &asset.clips {
        write_string(&mut clip_buf, &clip.name)?;
        write_f32(&mut clip_buf, clip.duration)?;
        write_u32(&mut clip_buf, clip.channels.len() as u32)?;
        for ch in &clip.channels {
            write_u32(&mut clip_buf, ch.bone_index)?;
            write_u32(&mut clip_buf, ch.keyframes.len() as u32)?;
            for kf in &ch.keyframes {
                write_f32(&mut clip_buf, kf.time)?;
                write_vec3(&mut clip_buf, kf.position)?;
                write_quat(&mut clip_buf, kf.rotation)?;
                write_vec3(&mut clip_buf, kf.scale)?;
            }
        }
    }
    let compressed_clips = lz4_flex::compress_prepend_size(&clip_buf);
    write_u32(writer, compressed_clips.len() as u32)?;
    writer.write_all(&compressed_clips)?;

    // 7. Blend shape data
    for bs in &asset.blend_shapes {
        write_string(writer, &bs.name)?;
        write_u32(writer, bs.brick_offset)?;
        write_u32(writer, bs.brick_count)?;
        write_vec3(writer, bs.bounding_box.min)?;
        write_vec3(writer, bs.bounding_box.max)?;
    }

    Ok(())
}

/// Load an animated asset from a reader.
pub fn load_animated_asset(reader: &mut impl Read) -> Result<AnimatedAsset> {
    // 1. Read standard chunk
    let chunk = load_chunk(reader).context("failed to read chunk data")?;

    // 2. Animation header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    anyhow::ensure!(magic == ANIM_MAGIC, "invalid animation magic: expected ANIM");
    let bone_count = read_u32(reader)? as usize;
    let segment_count = read_u32(reader)? as usize;
    let joint_count = read_u32(reader)? as usize;
    let clip_count = read_u32(reader)? as usize;
    let blend_shape_count = read_u32(reader)? as usize;
    let _reserved0 = read_u32(reader)?;
    let _reserved1 = read_u32(reader)?;

    // 3. Bone data
    let mut bones = Vec::with_capacity(bone_count);
    let mut hierarchy = Vec::with_capacity(bone_count);
    for _ in 0..bone_count {
        let name = read_string(reader)?;
        let bind_transform = read_mat4(reader)?;
        let inverse_bind = read_mat4(reader)?;
        let parent = read_i32(reader)?;
        bones.push(Bone {
            name,
            bind_transform,
            inverse_bind,
        });
        hierarchy.push(parent);
    }
    let skeleton =
        Skeleton::new(bones, hierarchy).context("invalid skeleton hierarchy in animated asset")?;

    // 4. Segment data
    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let bone_index = read_u32(reader)?;
        let brick_start = read_u32(reader)?;
        let brick_count = read_u32(reader)?;
        let aabb_min = read_vec3(reader)?;
        let aabb_max = read_vec3(reader)?;
        segments.push(Segment {
            bone_index,
            brick_start,
            brick_count,
            rest_aabb: Aabb::new(aabb_min, aabb_max),
        });
    }

    // 5. Joint data
    let mut joints = Vec::with_capacity(joint_count);
    for _ in 0..joint_count {
        let segment_a = read_u32(reader)?;
        let segment_b = read_u32(reader)?;
        let bone_index = read_u32(reader)?;
        let brick_start = read_u32(reader)?;
        let brick_count = read_u32(reader)?;
        let aabb_min = read_vec3(reader)?;
        let aabb_max = read_vec3(reader)?;
        let blend_k = read_f32(reader)?;
        joints.push(JointRegion {
            segment_a,
            segment_b,
            bone_index,
            brick_start,
            brick_count,
            rest_aabb: Aabb::new(aabb_min, aabb_max),
            blend_k,
        });
    }

    // 6. Clip data (LZ4 compressed)
    let compressed_len = read_u32(reader)? as usize;
    let mut compressed = vec![0u8; compressed_len];
    reader.read_exact(&mut compressed)?;
    let clip_buf = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| anyhow::anyhow!("LZ4 decompression error: {e}"))?;
    let mut clip_cursor = std::io::Cursor::new(&clip_buf);
    let mut clips = Vec::with_capacity(clip_count);
    for _ in 0..clip_count {
        let name = read_string(&mut clip_cursor)?;
        let duration = read_f32(&mut clip_cursor)?;
        let channel_count = read_u32(&mut clip_cursor)? as usize;
        let mut channels = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            let bone_index = read_u32(&mut clip_cursor)?;
            let kf_count = read_u32(&mut clip_cursor)? as usize;
            let mut keyframes = Vec::with_capacity(kf_count);
            for _ in 0..kf_count {
                let time = read_f32(&mut clip_cursor)?;
                let position = read_vec3(&mut clip_cursor)?;
                let rotation = read_quat(&mut clip_cursor)?;
                let scale = read_vec3(&mut clip_cursor)?;
                keyframes.push(Keyframe {
                    time,
                    position,
                    rotation,
                    scale,
                });
            }
            channels.push(BoneChannel {
                bone_index,
                keyframes,
            });
        }
        clips.push(AnimationClip {
            name,
            duration,
            channels,
        });
    }

    // 7. Blend shape data
    let mut blend_shapes = Vec::with_capacity(blend_shape_count);
    for _ in 0..blend_shape_count {
        let name = read_string(reader)?;
        let brick_offset = read_u32(reader)?;
        let brick_count = read_u32(reader)?;
        let aabb_min = read_vec3(reader)?;
        let aabb_max = read_vec3(reader)?;
        blend_shapes.push(BlendShape::new(
            name,
            brick_offset,
            brick_count,
            Aabb::new(aabb_min, aabb_max),
        ));
    }

    Ok(AnimatedAsset {
        chunk,
        skeleton,
        segments,
        joints,
        clips,
        blend_shapes,
    })
}

// --- Binary write helpers ---

fn write_u32(w: &mut impl Write, v: u32) -> Result<()> {
    Ok(w.write_all(&v.to_le_bytes())?)
}

fn write_i32(w: &mut impl Write, v: i32) -> Result<()> {
    Ok(w.write_all(&v.to_le_bytes())?)
}

fn write_f32(w: &mut impl Write, v: f32) -> Result<()> {
    Ok(w.write_all(&v.to_le_bytes())?)
}

fn write_vec3(w: &mut impl Write, v: Vec3) -> Result<()> {
    write_f32(w, v.x)?;
    write_f32(w, v.y)?;
    write_f32(w, v.z)
}

fn write_quat(w: &mut impl Write, q: Quat) -> Result<()> {
    write_f32(w, q.x)?;
    write_f32(w, q.y)?;
    write_f32(w, q.z)?;
    write_f32(w, q.w)
}

fn write_mat4(w: &mut impl Write, m: &Mat4) -> Result<()> {
    for f in m.to_cols_array() {
        write_f32(w, f)?;
    }
    Ok(())
}

fn write_string(w: &mut impl Write, s: &str) -> Result<()> {
    write_u32(w, s.len() as u32)?;
    Ok(w.write_all(s.as_bytes())?)
}

// --- Binary read helpers ---

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32(r: &mut impl Read) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_vec3(r: &mut impl Read) -> Result<Vec3> {
    Ok(Vec3::new(read_f32(r)?, read_f32(r)?, read_f32(r)?))
}

fn read_quat(r: &mut impl Read) -> Result<Quat> {
    Ok(Quat::from_xyzw(
        read_f32(r)?,
        read_f32(r)?,
        read_f32(r)?,
        read_f32(r)?,
    ))
}

fn read_mat4(r: &mut impl Read) -> Result<Mat4> {
    let mut cols = [0.0f32; 16];
    for c in &mut cols {
        *c = read_f32(r)?;
    }
    Ok(Mat4::from_cols_array(&cols))
}

fn read_string(r: &mut impl Read) -> Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).context("invalid UTF-8 in string")?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use std::io::Cursor;

    /// Create an empty chunk for testing.
    fn empty_chunk() -> Chunk {
        Chunk::new(IVec3::ZERO)
    }

    /// Create a bone with a translation-only bind pose.
    fn bone(name: &str, translation: Vec3) -> Bone {
        let bind = Mat4::from_translation(translation);
        Bone {
            name: name.to_string(),
            bind_transform: bind,
            inverse_bind: bind.inverse(),
        }
    }

    /// Helper to compare Vec3 values with tolerance.
    fn assert_vec3_eq(a: Vec3, b: Vec3, label: &str) {
        assert!(
            (a - b).length() < 1e-5,
            "{label}: expected {b:?}, got {a:?}"
        );
    }

    /// Helper to compare Mat4 values with tolerance.
    fn assert_mat4_eq(a: &Mat4, b: &Mat4, label: &str) {
        let diff = (*a - *b).abs().to_cols_array();
        let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "{label}: max diff {max_diff}");
    }

    // ------ 1. Empty skeleton roundtrip ------

    #[test]
    fn animated_asset_roundtrip_empty() {
        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton: Skeleton::new(vec![], vec![]).unwrap(),
            segments: vec![],
            joints: vec![],
            clips: vec![],
            blend_shapes: vec![],
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.skeleton.bone_count(), 0);
        assert!(loaded.segments.is_empty());
        assert!(loaded.joints.is_empty());
        assert!(loaded.clips.is_empty());
        assert!(loaded.blend_shapes.is_empty());
    }

    // ------ 2. Skeleton roundtrip ------

    #[test]
    fn animated_asset_roundtrip_skeleton() {
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("spine", Vec3::new(0.0, 1.0, 0.0)),
            bone("head", Vec3::new(0.0, 0.5, 0.0)),
        ];
        let hierarchy = vec![-1, 0, 1];
        let skeleton = Skeleton::new(bones, hierarchy).unwrap();

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments: vec![],
            joints: vec![],
            clips: vec![],
            blend_shapes: vec![],
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.skeleton.bone_count(), 3);
        assert_eq!(loaded.skeleton.bones[0].name, "root");
        assert_eq!(loaded.skeleton.bones[1].name, "spine");
        assert_eq!(loaded.skeleton.bones[2].name, "head");
        assert_eq!(loaded.skeleton.hierarchy, vec![-1, 0, 1]);

        // Verify transforms
        for i in 0..3 {
            assert_mat4_eq(
                &loaded.skeleton.bones[i].bind_transform,
                &asset.skeleton.bones[i].bind_transform,
                &format!("bone {i} bind_transform"),
            );
            assert_mat4_eq(
                &loaded.skeleton.bones[i].inverse_bind,
                &asset.skeleton.bones[i].inverse_bind,
                &format!("bone {i} inverse_bind"),
            );
        }
    }

    // ------ 3. Segments roundtrip ------

    #[test]
    fn animated_asset_roundtrip_segments() {
        let bones = vec![bone("root", Vec3::ZERO), bone("arm", Vec3::X)];
        let skeleton = Skeleton::new(bones, vec![-1, 0]).unwrap();

        let segments = vec![
            Segment {
                bone_index: 0,
                brick_start: 0,
                brick_count: 10,
                rest_aabb: Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)),
            },
            Segment {
                bone_index: 1,
                brick_start: 10,
                brick_count: 5,
                rest_aabb: Aabb::new(Vec3::new(0.5, -0.5, -0.5), Vec3::new(2.0, 0.5, 0.5)),
            },
        ];

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments,
            joints: vec![],
            clips: vec![],
            blend_shapes: vec![],
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.segments.len(), 2);

        assert_eq!(loaded.segments[0].bone_index, 0);
        assert_eq!(loaded.segments[0].brick_start, 0);
        assert_eq!(loaded.segments[0].brick_count, 10);
        assert_vec3_eq(
            loaded.segments[0].rest_aabb.min,
            Vec3::new(-1.0, -1.0, -1.0),
            "seg0 min",
        );
        assert_vec3_eq(
            loaded.segments[0].rest_aabb.max,
            Vec3::new(1.0, 1.0, 1.0),
            "seg0 max",
        );

        assert_eq!(loaded.segments[1].bone_index, 1);
        assert_eq!(loaded.segments[1].brick_start, 10);
        assert_eq!(loaded.segments[1].brick_count, 5);
        assert_vec3_eq(
            loaded.segments[1].rest_aabb.min,
            Vec3::new(0.5, -0.5, -0.5),
            "seg1 min",
        );
        assert_vec3_eq(
            loaded.segments[1].rest_aabb.max,
            Vec3::new(2.0, 0.5, 0.5),
            "seg1 max",
        );
    }

    // ------ 4. Joints roundtrip ------

    #[test]
    fn animated_asset_roundtrip_joints() {
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("upper", Vec3::Y),
            bone("lower", Vec3::new(0.0, 0.5, 0.0)),
        ];
        let skeleton = Skeleton::new(bones, vec![-1, 0, 1]).unwrap();

        let joints = vec![JointRegion {
            segment_a: 0,
            segment_b: 1,
            bone_index: 1,
            brick_start: 20,
            brick_count: 4,
            rest_aabb: Aabb::new(Vec3::new(-0.1, 0.8, -0.1), Vec3::new(0.1, 1.2, 0.1)),
            blend_k: 0.05,
        }];

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments: vec![],
            joints,
            clips: vec![],
            blend_shapes: vec![],
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.joints.len(), 1);
        let j = &loaded.joints[0];
        assert_eq!(j.segment_a, 0);
        assert_eq!(j.segment_b, 1);
        assert_eq!(j.bone_index, 1);
        assert_eq!(j.brick_start, 20);
        assert_eq!(j.brick_count, 4);
        assert_vec3_eq(j.rest_aabb.min, Vec3::new(-0.1, 0.8, -0.1), "joint min");
        assert_vec3_eq(j.rest_aabb.max, Vec3::new(0.1, 1.2, 0.1), "joint max");
        assert!((j.blend_k - 0.05).abs() < 1e-6, "blend_k = {}", j.blend_k);
    }

    // ------ 5. Clips roundtrip ------

    #[test]
    fn animated_asset_roundtrip_clips() {
        let bones = vec![bone("root", Vec3::ZERO), bone("child", Vec3::Y)];
        let skeleton = Skeleton::new(bones, vec![-1, 0]).unwrap();

        let clips = vec![AnimationClip {
            name: "walk".to_string(),
            duration: 1.5,
            channels: vec![
                BoneChannel {
                    bone_index: 0,
                    keyframes: vec![
                        Keyframe {
                            time: 0.0,
                            position: Vec3::ZERO,
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 0.5,
                            position: Vec3::new(1.0, 0.0, 0.0),
                            rotation: Quat::from_rotation_y(1.0),
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 1.5,
                            position: Vec3::new(2.0, 0.0, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                    ],
                },
                BoneChannel {
                    bone_index: 1,
                    keyframes: vec![
                        Keyframe {
                            time: 0.0,
                            position: Vec3::new(0.0, 1.0, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 1.5,
                            position: Vec3::new(0.0, 1.0, 0.0),
                            rotation: Quat::from_rotation_z(0.5),
                            scale: Vec3::splat(1.2),
                        },
                    ],
                },
            ],
        }];

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments: vec![],
            joints: vec![],
            clips,
            blend_shapes: vec![],
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.clips.len(), 1);
        let clip = &loaded.clips[0];
        assert_eq!(clip.name, "walk");
        assert!((clip.duration - 1.5).abs() < 1e-6);
        assert_eq!(clip.channels.len(), 2);

        // Channel 0: 3 keyframes
        let ch0 = &clip.channels[0];
        assert_eq!(ch0.bone_index, 0);
        assert_eq!(ch0.keyframes.len(), 3);
        assert!((ch0.keyframes[0].time).abs() < 1e-6);
        assert_vec3_eq(ch0.keyframes[1].position, Vec3::new(1.0, 0.0, 0.0), "kf1 pos");
        // Verify quaternion roundtrip
        let orig_rot = Quat::from_rotation_y(1.0);
        let loaded_rot = ch0.keyframes[1].rotation;
        assert!(
            (orig_rot - loaded_rot).length() < 1e-5 || (orig_rot + loaded_rot).length() < 1e-5,
            "rotation mismatch: expected {orig_rot:?}, got {loaded_rot:?}"
        );

        // Channel 1: 2 keyframes
        let ch1 = &clip.channels[1];
        assert_eq!(ch1.bone_index, 1);
        assert_eq!(ch1.keyframes.len(), 2);
        assert_vec3_eq(
            ch1.keyframes[1].scale,
            Vec3::splat(1.2),
            "ch1 kf1 scale",
        );
    }

    // ------ 6. Blend shapes roundtrip ------

    #[test]
    fn animated_asset_roundtrip_blend_shapes() {
        let skeleton = Skeleton::new(vec![bone("root", Vec3::ZERO)], vec![-1]).unwrap();

        let blend_shapes = vec![
            BlendShape::new(
                "smile",
                0,
                8,
                Aabb::new(Vec3::new(-0.5, -0.2, -0.3), Vec3::new(0.5, 0.3, 0.1)),
            ),
            BlendShape::new(
                "blink_L",
                8,
                4,
                Aabb::new(Vec3::new(-0.3, 0.1, -0.1), Vec3::new(-0.1, 0.4, 0.1)),
            ),
        ];

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments: vec![],
            joints: vec![],
            clips: vec![],
            blend_shapes,
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        assert_eq!(loaded.blend_shapes.len(), 2);

        let bs0 = &loaded.blend_shapes[0];
        assert_eq!(bs0.name, "smile");
        assert_eq!(bs0.brick_offset, 0);
        assert_eq!(bs0.brick_count, 8);
        assert_vec3_eq(
            bs0.bounding_box.min,
            Vec3::new(-0.5, -0.2, -0.3),
            "bs0 min",
        );
        assert_vec3_eq(
            bs0.bounding_box.max,
            Vec3::new(0.5, 0.3, 0.1),
            "bs0 max",
        );
        // BlendShape::new initializes weight to 0.0
        assert!((bs0.weight).abs() < 1e-6);

        let bs1 = &loaded.blend_shapes[1];
        assert_eq!(bs1.name, "blink_L");
        assert_eq!(bs1.brick_offset, 8);
        assert_eq!(bs1.brick_count, 4);
    }

    // ------ 7. Full roundtrip ------

    #[test]
    fn animated_asset_roundtrip_full() {
        let bones = vec![
            bone("root", Vec3::ZERO),
            bone("torso", Vec3::new(0.0, 0.8, 0.0)),
            bone("head", Vec3::new(0.0, 0.4, 0.0)),
        ];
        let skeleton = Skeleton::new(bones, vec![-1, 0, 1]).unwrap();

        let segments = vec![
            Segment {
                bone_index: 0,
                brick_start: 0,
                brick_count: 20,
                rest_aabb: Aabb::new(Vec3::new(-0.5, -1.0, -0.3), Vec3::new(0.5, 0.0, 0.3)),
            },
            Segment {
                bone_index: 1,
                brick_start: 20,
                brick_count: 15,
                rest_aabb: Aabb::new(Vec3::new(-0.4, 0.0, -0.25), Vec3::new(0.4, 0.8, 0.25)),
            },
            Segment {
                bone_index: 2,
                brick_start: 35,
                brick_count: 10,
                rest_aabb: Aabb::new(Vec3::new(-0.2, 0.8, -0.2), Vec3::new(0.2, 1.2, 0.2)),
            },
        ];

        let joints = vec![
            JointRegion {
                segment_a: 0,
                segment_b: 1,
                bone_index: 1,
                brick_start: 45,
                brick_count: 6,
                rest_aabb: Aabb::new(Vec3::new(-0.3, -0.1, -0.2), Vec3::new(0.3, 0.1, 0.2)),
                blend_k: 0.04,
            },
            JointRegion {
                segment_a: 1,
                segment_b: 2,
                bone_index: 2,
                brick_start: 51,
                brick_count: 4,
                rest_aabb: Aabb::new(Vec3::new(-0.15, 0.7, -0.15), Vec3::new(0.15, 0.9, 0.15)),
                blend_k: 0.06,
            },
        ];

        let clips = vec![
            AnimationClip {
                name: "idle".to_string(),
                duration: 2.0,
                channels: vec![BoneChannel {
                    bone_index: 1,
                    keyframes: vec![
                        Keyframe {
                            time: 0.0,
                            position: Vec3::new(0.0, 0.8, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 1.0,
                            position: Vec3::new(0.0, 0.82, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 2.0,
                            position: Vec3::new(0.0, 0.8, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                    ],
                }],
            },
            AnimationClip {
                name: "nod".to_string(),
                duration: 0.8,
                channels: vec![BoneChannel {
                    bone_index: 2,
                    keyframes: vec![
                        Keyframe {
                            time: 0.0,
                            position: Vec3::new(0.0, 0.4, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 0.4,
                            position: Vec3::new(0.0, 0.4, 0.0),
                            rotation: Quat::from_rotation_x(-0.3),
                            scale: Vec3::ONE,
                        },
                        Keyframe {
                            time: 0.8,
                            position: Vec3::new(0.0, 0.4, 0.0),
                            rotation: Quat::IDENTITY,
                            scale: Vec3::ONE,
                        },
                    ],
                }],
            },
        ];

        let blend_shapes = vec![
            BlendShape::new(
                "smile",
                55,
                8,
                Aabb::new(Vec3::new(-0.15, 0.85, -0.1), Vec3::new(0.15, 1.05, 0.05)),
            ),
            BlendShape::new(
                "blink_L",
                63,
                3,
                Aabb::new(Vec3::new(-0.12, 1.0, -0.05), Vec3::new(-0.02, 1.1, 0.05)),
            ),
        ];

        let asset = AnimatedAsset {
            chunk: empty_chunk(),
            skeleton,
            segments,
            joints,
            clips,
            blend_shapes,
        };

        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_animated_asset(&mut cursor).unwrap();

        // Verify all counts
        assert_eq!(loaded.skeleton.bone_count(), 3);
        assert_eq!(loaded.segments.len(), 3);
        assert_eq!(loaded.joints.len(), 2);
        assert_eq!(loaded.clips.len(), 2);
        assert_eq!(loaded.blend_shapes.len(), 2);

        // Spot-check skeleton
        assert_eq!(loaded.skeleton.bones[0].name, "root");
        assert_eq!(loaded.skeleton.bones[2].name, "head");
        assert_eq!(loaded.skeleton.hierarchy, vec![-1, 0, 1]);

        // Spot-check segments
        assert_eq!(loaded.segments[2].bone_index, 2);
        assert_eq!(loaded.segments[2].brick_count, 10);

        // Spot-check joints
        assert_eq!(loaded.joints[1].segment_a, 1);
        assert_eq!(loaded.joints[1].segment_b, 2);
        assert!((loaded.joints[1].blend_k - 0.06).abs() < 1e-6);

        // Spot-check clips
        assert_eq!(loaded.clips[0].name, "idle");
        assert!((loaded.clips[0].duration - 2.0).abs() < 1e-6);
        assert_eq!(loaded.clips[0].channels[0].keyframes.len(), 3);
        assert_eq!(loaded.clips[1].name, "nod");
        assert!((loaded.clips[1].duration - 0.8).abs() < 1e-6);

        // Spot-check blend shapes
        assert_eq!(loaded.blend_shapes[0].name, "smile");
        assert_eq!(loaded.blend_shapes[0].brick_offset, 55);
        assert_eq!(loaded.blend_shapes[1].name, "blink_L");
        assert_eq!(loaded.blend_shapes[1].brick_count, 3);

        // Verify entire buffer was consumed
        assert_eq!(
            cursor.position() as usize,
            buf.len(),
            "not all bytes consumed"
        );
    }

    // ------ 8. Full animated pipeline integration test ------

    /// Integration test: full animated import pipeline.
    /// Build skinned mesh -> segment -> voxelize -> animated asset -> .rkf roundtrip.
    #[test]
    fn full_animated_pipeline_roundtrip() {
        use crate::mesh::{ImportMaterial, MeshData};
        use crate::skeleton_extract::{auto_segment, VertexSkinning};
        use crate::segment_voxelize::voxelize_segments;
        use crate::voxelize::{VoxelizeConfig, to_chunk};
        use rkf_animation::blend_shape::BlendShape;
        use rkf_animation::clip::{AnimationClip, BoneChannel, Keyframe};
        use rkf_animation::segment::{JointRegion, Segment};
        use rkf_animation::skeleton::{Bone, Skeleton};
        use rkf_core::aabb::Aabb;
        use rkf_core::chunk::Chunk;
        use glam::{IVec3, Mat4, Quat, Vec3};

        // Build a simple two-segment mesh: left box + right box
        // Left box vertices (indices 0-7) attached to bone 0
        // Right box vertices (indices 8-15) attached to bone 1
        let half = 0.25f32;

        // Left box centered at (-0.5, 0, 0)
        let left_center = Vec3::new(-0.5, 0.0, 0.0);
        let left_verts = box_vertices(left_center, half);
        let right_center = Vec3::new(0.5, 0.0, 0.0);
        let right_verts = box_vertices(right_center, half);

        let mut positions: Vec<Vec3> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let left_offset = positions.len() as u32;
        positions.extend_from_slice(&left_verts);
        indices.extend(box_indices(left_offset));

        let right_offset = positions.len() as u32;
        positions.extend_from_slice(&right_verts);
        indices.extend(box_indices(right_offset));

        let tri_count = indices.len() / 3;
        let mesh = MeshData {
            normals: vec![Vec3::Y; positions.len()],
            uvs: Vec::new(),
            material_indices: vec![0; tri_count],
            materials: vec![ImportMaterial {
                name: "skin".to_string(),
                base_color: [0.8, 0.6, 0.5],
                metallic: 0.0,
                roughness: 0.7,
                albedo_texture: None,
            }],
            bounds_min: Vec3::new(-0.75, -0.25, -0.25),
            bounds_max: Vec3::new(0.75, 0.25, 0.25),
            indices,
            positions,
        };

        // Skinning: left box (verts 0-7) -> bone 0, right box (verts 8-15) -> bone 1
        let vert_count = mesh.positions.len();
        let mut joints = Vec::with_capacity(vert_count);
        let mut weights = Vec::with_capacity(vert_count);
        for i in 0..vert_count {
            if i < 8 {
                joints.push([0, -1, -1, -1]); // bone 0
            } else {
                joints.push([1, -1, -1, -1]); // bone 1
            }
            weights.push([1.0, 0.0, 0.0, 0.0]);
        }
        let skinning = VertexSkinning { joints, weights };

        // Segment
        let segmentation = auto_segment(&mesh, &skinning);
        assert_eq!(segmentation.segment_bones.len(), 2, "should have 2 segments");
        assert!(
            segmentation.joint_pairs.is_empty(),
            "no joint triangles for separate boxes"
        );

        // Voxelize segments
        let config = VoxelizeConfig {
            tier: 0,
            narrow_band_bricks: 2,
            compute_color: false,
        };
        let seg_vox = voxelize_segments(&mesh, &segmentation, &config);
        assert_eq!(seg_vox.segments.len(), 2);

        // Build skeleton: 2 bones, bone 0 is root, bone 1 is child
        let skeleton = Skeleton::new(
            vec![
                Bone {
                    name: "root".to_string(),
                    bind_transform: Mat4::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
                    inverse_bind: Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0)),
                },
                Bone {
                    name: "child".to_string(),
                    bind_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
                    inverse_bind: Mat4::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
                },
            ],
            vec![-1, 0],
        )
        .unwrap();

        // Build a simple animation clip
        let clip = AnimationClip {
            name: "wave".to_string(),
            duration: 1.0,
            channels: vec![BoneChannel {
                bone_index: 1,
                keyframes: vec![
                    Keyframe {
                        time: 0.0,
                        position: Vec3::new(1.0, 0.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    },
                    Keyframe {
                        time: 0.5,
                        position: Vec3::new(1.0, 0.3, 0.0),
                        rotation: Quat::from_rotation_z(0.3),
                        scale: Vec3::ONE,
                    },
                    Keyframe {
                        time: 1.0,
                        position: Vec3::new(1.0, 0.0, 0.0),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::ONE,
                    },
                ],
            }],
        };

        // Build chunk from all segment voxelization results
        // Collect grids from all segments into a single chunk
        let mut all_tier_grids = Vec::new();
        for sd in &seg_vox.segments {
            let sub_chunk = to_chunk(&sd.voxelize_result, &[], config.tier, IVec3::ZERO);
            all_tier_grids.extend(sub_chunk.grids);
        }
        let total_bricks: u32 = all_tier_grids
            .iter()
            .map(|tg| tg.bricks.len() as u32)
            .sum();
        let chunk = Chunk {
            coords: IVec3::ZERO,
            grids: all_tier_grids,
            brick_count: total_bricks,
        };

        // Build segment descriptors from voxelization
        let segments: Vec<Segment> = seg_vox.segments.iter().map(|sd| sd.segment.clone()).collect();
        let joints_desc: Vec<JointRegion> = seg_vox.joints.iter().map(|jd| jd.joint.clone()).collect();

        // Build animated asset
        let asset = AnimatedAsset {
            chunk,
            skeleton,
            segments,
            joints: joints_desc,
            clips: vec![clip],
            blend_shapes: vec![BlendShape::new(
                "smile",
                0,
                1,
                Aabb::new(Vec3::splat(-0.1), Vec3::splat(0.1)),
            )],
        };

        // Roundtrip
        let mut buf = Vec::new();
        save_animated_asset(&asset, &mut buf).unwrap();
        assert!(!buf.is_empty());

        let loaded = load_animated_asset(&mut std::io::Cursor::new(&buf)).unwrap();

        // Verify all components
        assert_eq!(loaded.skeleton.bones.len(), 2);
        assert_eq!(loaded.skeleton.bones[0].name, "root");
        assert_eq!(loaded.skeleton.bones[1].name, "child");
        assert_eq!(loaded.skeleton.hierarchy, vec![-1, 0]);

        assert_eq!(loaded.segments.len(), 2);
        assert_eq!(loaded.joints.len(), 0);

        assert_eq!(loaded.clips.len(), 1);
        assert_eq!(loaded.clips[0].name, "wave");
        assert_eq!(loaded.clips[0].channels.len(), 1);
        assert_eq!(loaded.clips[0].channels[0].keyframes.len(), 3);

        assert_eq!(loaded.blend_shapes.len(), 1);
        assert_eq!(loaded.blend_shapes[0].name, "smile");

        assert_eq!(loaded.chunk.brick_count, asset.chunk.brick_count);
    }

    // Helper: generate 8 vertices of an axis-aligned box
    fn box_vertices(center: Vec3, half: f32) -> [Vec3; 8] {
        [
            center + Vec3::new(-half, -half, -half),
            center + Vec3::new(half, -half, -half),
            center + Vec3::new(half, half, -half),
            center + Vec3::new(-half, half, -half),
            center + Vec3::new(-half, -half, half),
            center + Vec3::new(half, -half, half),
            center + Vec3::new(half, half, half),
            center + Vec3::new(-half, half, half),
        ]
    }

    // Helper: generate 36 indices (12 triangles) for a box with vertex offset
    fn box_indices(offset: u32) -> Vec<u32> {
        let faces: [[u32; 6]; 6] = [
            [0, 1, 2, 0, 2, 3], // front
            [5, 4, 7, 5, 7, 6], // back
            [4, 0, 3, 4, 3, 7], // left
            [1, 5, 6, 1, 6, 2], // right
            [3, 2, 6, 3, 6, 7], // top
            [4, 5, 1, 4, 1, 0], // bottom
        ];
        faces.iter().flatten().map(|&i| offset + i).collect()
    }
}
