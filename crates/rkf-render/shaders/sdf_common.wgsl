// sdf_common.wgsl — Shared types and SDF utilities for the v2 object-centric ray marcher.

// ---------- Type Constants ----------

const SDF_TYPE_NONE: u32       = 0u;
const SDF_TYPE_ANALYTICAL: u32 = 1u;
const SDF_TYPE_VOXELIZED: u32  = 2u;

const BLEND_SMOOTH_UNION: u32 = 0u;
const BLEND_UNION: u32        = 1u;
const BLEND_SUBTRACT: u32     = 2u;
const BLEND_INTERSECT: u32    = 3u;

const PRIM_SPHERE: u32   = 0u;
const PRIM_BOX: u32      = 1u;
const PRIM_CAPSULE: u32  = 2u;
const PRIM_TORUS: u32    = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_PLANE: u32    = 5u;

const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const MAX_FLOAT: f32 = 3.402823e+38;

// ---------- GPU Structs ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, bits 16-21 = material_id, bits 22-27 = secondary_material_id
    word1: u32, // byte 3 = blend_weight, bytes 0-2 reserved
}

struct GpuObject {
    inverse_world: mat4x4<f32>,  // 64 bytes @ offset 0
    aabb_min: vec4<f32>,         // 16 bytes @ offset 64
    aabb_max: vec4<f32>,         // 16 bytes @ offset 80
    brick_map_offset: u32,       // 4 bytes @ offset 96
    brick_map_dims_x: u32,       // 4 bytes @ offset 100
    brick_map_dims_y: u32,       // 4 bytes @ offset 104
    brick_map_dims_z: u32,       // 4 bytes @ offset 108
    voxel_size: f32,             // 4 bytes @ offset 112
    material_id: u32,            // 4 bytes @ offset 116
    sdf_type: u32,               // 4 bytes @ offset 120
    blend_mode: u32,             // 4 bytes @ offset 124
    blend_radius: f32,           // 4 bytes @ offset 128
    sdf_param_0: f32,            // 4 bytes @ offset 132
    sdf_param_1: f32,            // 4 bytes @ offset 136
    sdf_param_2: f32,            // 4 bytes @ offset 140
    sdf_param_3: f32,            // 4 bytes @ offset 144
    accumulated_scale_x: f32,    // 4 bytes @ offset 148
    accumulated_scale_y: f32,    // 4 bytes @ offset 152
    accumulated_scale_z: f32,    // 4 bytes @ offset 156
    lod_level: u32,              // 4 bytes @ offset 160
    object_id: u32,              // 4 bytes @ offset 164
    primitive_type: u32,         // 4 bytes @ offset 168
    // Tight local-space AABB of allocated bricks (for empty-space skipping).
    geometry_aabb_min_x: f32,    // 4 bytes @ offset 172
    geometry_aabb_min_y: f32,    // 4 bytes @ offset 176
    geometry_aabb_min_z: f32,    // 4 bytes @ offset 180
    geometry_aabb_max_x: f32,    // 4 bytes @ offset 184
    geometry_aabb_max_y: f32,    // 4 bytes @ offset 188
    geometry_aabb_max_z: f32,    // 4 bytes @ offset 192
    // 60 bytes of padding (15 × f32)
    _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: f32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32,
}

struct BvhNode {
    aabb_min: vec3<f32>,
    left: u32,
    aabb_max: vec3<f32>,
    right_or_object: u32,
}

struct CameraUniforms {
    position: vec4<f32>,
    forward:  vec4<f32>,
    right:    vec4<f32>,
    up:       vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
}

struct SceneUniforms {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

// ---------- VoxelSample Helpers ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return (word0 >> 16u) & 0x3Fu;
}


// ---------- SDF Primitives ----------

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let q = vec3<f32>(p.x, max(abs(p.y) - half_height, 0.0), p.z);
    return length(q) - radius;
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normal) + dist;
}

/// Evaluate an analytical SDF primitive using the object's sdf_params and primitive_type.
fn evaluate_analytical_sdf(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    switch obj.primitive_type {
        case PRIM_SPHERE: {
            return sdf_sphere(local_pos, obj.sdf_param_0);
        }
        case PRIM_BOX: {
            return sdf_box(local_pos, vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2));
        }
        case PRIM_CAPSULE: {
            return sdf_capsule(local_pos, obj.sdf_param_0, obj.sdf_param_1);
        }
        case PRIM_TORUS: {
            return sdf_torus(local_pos, obj.sdf_param_0, obj.sdf_param_1);
        }
        case PRIM_CYLINDER: {
            return sdf_cylinder(local_pos, obj.sdf_param_0, obj.sdf_param_1);
        }
        case PRIM_PLANE: {
            return sdf_plane(local_pos, normalize(vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)), 0.0);
        }
        default: {
            return MAX_FLOAT;
        }
    }
}

// ---------- Blend Modes ----------

struct BlendResult {
    distance: f32,
    material_id: u32,
}

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / max(k, 1e-6);
    return min(a, b) - h * h * k * 0.25;
}

fn apply_blend(dist_a: f32, mat_a: u32, dist_b: f32, mat_b: u32,
               mode: u32, radius: f32) -> BlendResult {
    var result: BlendResult;

    switch mode {
        case BLEND_SMOOTH_UNION: {
            result.distance = smooth_min(dist_a, dist_b, radius);
            result.material_id = select(mat_b, mat_a, dist_a < dist_b);
        }
        case BLEND_UNION: {
            result.distance = min(dist_a, dist_b);
            result.material_id = select(mat_b, mat_a, dist_a < dist_b);
        }
        case BLEND_SUBTRACT: {
            result.distance = max(dist_a, -dist_b);
            result.material_id = mat_a;
        }
        case BLEND_INTERSECT: {
            result.distance = max(dist_a, dist_b);
            result.material_id = select(mat_b, mat_a, dist_a > dist_b);
        }
        default: {
            result.distance = min(dist_a, dist_b);
            result.material_id = select(mat_b, mat_a, dist_a < dist_b);
        }
    }

    return result;
}

// ---------- Brick Map Sampling ----------

/// Sample the brick map for a voxelized object at a local-space position.
///
/// Returns vec2(distance, material_id_as_float). Returns (MAX_FLOAT, 0) if
/// the position is outside the brick map or the brick slot is empty.
fn sample_brick_map(local_pos: vec3<f32>, obj: GpuObject,
                    brick_pool: ptr<storage, array<VoxelSample>, read>,
                    brick_maps: ptr<storage, array<u32>, read>) -> vec2<f32> {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;

    // Object AABB in local space (aabb_min/max are camera-relative, but for
    // voxelized objects the brick map covers a local-space grid).
    // The brick grid starts at local origin (0,0,0) and extends dims * brick_extent.
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    // Offset local_pos relative to grid start (assume grid centered at object origin).
    let grid_pos = local_pos + grid_size * 0.5;

    // Out of bounds check.
    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }

    // Brick coordinate.
    let brick_coord = vec3<u32>(floor(grid_pos / brick_extent));
    let flat_brick = brick_coord.x + brick_coord.y * dims.x + brick_coord.z * dims.x * dims.y;

    // Look up slot from brick map.
    let slot = (*brick_maps)[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }
    if slot == INTERIOR_SLOT {
        return vec2<f32>(-(obj.voxel_size * 2.0), 0.0);
    }

    // Local position within brick.
    let brick_min = vec3<f32>(brick_coord) * brick_extent - grid_size * 0.5;

    // Trilinear interpolation within brick.
    let brick_local = (local_pos - brick_min) / vs - vec3<f32>(0.5);
    let f = clamp(brick_local, vec3<f32>(0.0), vec3<f32>(6.9999));
    let i0 = vec3<u32>(floor(f));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(7u));
    let t = f - floor(f);

    let base = slot * 512u;
    let c000 = extract_distance((*brick_pool)[base + i0.x + i0.y*8u + i0.z*64u].word0);
    let c100 = extract_distance((*brick_pool)[base + i1.x + i0.y*8u + i0.z*64u].word0);
    let c010 = extract_distance((*brick_pool)[base + i0.x + i1.y*8u + i0.z*64u].word0);
    let c110 = extract_distance((*brick_pool)[base + i1.x + i1.y*8u + i0.z*64u].word0);
    let c001 = extract_distance((*brick_pool)[base + i0.x + i0.y*8u + i1.z*64u].word0);
    let c101 = extract_distance((*brick_pool)[base + i1.x + i0.y*8u + i1.z*64u].word0);
    let c011 = extract_distance((*brick_pool)[base + i0.x + i1.y*8u + i1.z*64u].word0);
    let c111 = extract_distance((*brick_pool)[base + i1.x + i1.y*8u + i1.z*64u].word0);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    let dist = mix(c0, c1, t.z);

    // Material from nearest voxel.
    let nearest = vec3<u32>(round(f));
    let mat_sample = (*brick_pool)[base + nearest.x + nearest.y*8u + nearest.z*64u];
    let mat_id = f32(extract_material_id(mat_sample.word0));

    return vec2<f32>(dist, mat_id);
}

// ---------- Ray-AABB Intersection ----------

fn ray_aabb_intersect(origin: vec3<f32>, inv_dir: vec3<f32>,
                      box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t1 = (box_min - origin) * inv_dir;
    let t2 = (box_max - origin) * inv_dir;
    let t_lo = min(t1, t2);
    let t_hi = max(t1, t2);
    let t_near = max(t_lo.x, max(t_lo.y, t_lo.z));
    let t_far  = min(t_hi.x, min(t_hi.y, t_hi.z));
    return vec2<f32>(t_near, t_far);
}
