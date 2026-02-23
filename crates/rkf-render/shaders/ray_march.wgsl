// Ray march compute shader — v2 object-centric ray marcher.
//
// One thread per pixel at internal resolution. Generates a camera ray,
// traverses the BVH to find candidate objects, sphere-traces each object's
// SDF (analytical or voxelized), and writes G-buffer data.

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
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const MAX_FLOAT: f32 = 3.402823e+38;
const HIT_EPSILON: f32 = 0.001;
const MIN_STEP: f32 = 0.0005;

// BVH traversal stack depth.
const BVH_STACK_SIZE: u32 = 32u;

// ---------- GPU Structs ----------

struct VoxelSample {
    word0: u32,
    word1: u32,
}

// GpuObject: 256 bytes, must match Rust #[repr(C)] layout exactly.
// WGSL vec4<f32> has 16-byte alignment, but Rust [f32; 4] has 4-byte alignment.
// So we use scalar fields for fields not naturally aligned to 16.
struct GpuObject {
    inverse_world: mat4x4<f32>,  // 64 bytes @ offset 0 (mat4 = 16-byte aligned)
    aabb_min: vec4<f32>,         // 16 bytes @ offset 64 (16-byte aligned ✓)
    aabb_max: vec4<f32>,         // 16 bytes @ offset 80 (16-byte aligned ✓)
    brick_map_offset: u32,       // 4 bytes @ offset 96
    brick_map_dims_x: u32,       // 4 bytes @ offset 100
    brick_map_dims_y: u32,       // 4 bytes @ offset 104
    brick_map_dims_z: u32,       // 4 bytes @ offset 108
    voxel_size: f32,             // 4 bytes @ offset 112
    material_id: u32,            // 4 bytes @ offset 116
    sdf_type: u32,               // 4 bytes @ offset 120
    blend_mode: u32,             // 4 bytes @ offset 124
    blend_radius: f32,           // 4 bytes @ offset 128
    sdf_param_0: f32,            // 4 bytes @ offset 132 (scalar, no alignment issue)
    sdf_param_1: f32,            // 4 bytes @ offset 136
    sdf_param_2: f32,            // 4 bytes @ offset 140
    sdf_param_3: f32,            // 4 bytes @ offset 144
    accumulated_scale: f32,      // 4 bytes @ offset 148
    lod_level: u32,              // 4 bytes @ offset 152
    object_id: u32,              // 4 bytes @ offset 156
    primitive_type: u32,         // 4 bytes @ offset 160
    // 92 bytes of padding (23 × f32)
    _pad0: f32, _pad1: f32, _pad2: f32, _pad3: f32,
    _pad4: f32, _pad5: f32, _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: f32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32, _pad21: f32, _pad22: f32,
}

struct BvhNode {
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_min_z: f32,
    left: u32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    aabb_max_z: f32,
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

struct MarchResult {
    t: f32,
    hit: bool,
    material_id: u32,
    object_id: u32,
    normal: vec3<f32>,
}

// ---------- Bindings ----------

// Group 0: scene data (v2 object-centric layout)
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;
@group(0) @binding(3) var<uniform>       camera: CameraUniforms;
@group(0) @binding(4) var<uniform>       scene: SceneUniforms;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 1: G-buffer output textures
@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<r32uint, write>;
@group(1) @binding(3) var gbuf_motion:   texture_storage_2d<rgba32float, write>;

// ---------- VoxelSample Helpers ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return word0 >> 16u;
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

// ---------- Object Evaluation ----------

/// Evaluate an analytical SDF primitive at a local-space position.
fn evaluate_analytical(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
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

/// Sample a voxelized object's brick map at a local-space position.
/// Returns the SDF distance, or MAX_FLOAT if outside the brick map.
fn sample_voxelized(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    // Grid starts at -grid_size/2 (object origin = center of grid).
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return MAX_FLOAT;
    }

    let brick_coord = vec3<u32>(floor(grid_pos / brick_extent));
    let flat_brick = brick_coord.x + brick_coord.y * dims.x + brick_coord.z * dims.x * dims.y;

    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return MAX_FLOAT;
    }

    // Brick minimum in local space.
    let brick_min = vec3<f32>(brick_coord) * brick_extent - grid_size * 0.5;

    // Trilinear interpolation within brick.
    let brick_local = (local_pos - brick_min) / vs - vec3<f32>(0.5);
    let f = clamp(brick_local, vec3<f32>(0.0), vec3<f32>(6.9999));
    let i0 = vec3<u32>(floor(f));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(7u));
    let t = f - floor(f);

    let base = slot * 512u;
    let c000 = extract_distance(brick_pool[base + i0.x + i0.y*8u + i0.z*64u].word0);
    let c100 = extract_distance(brick_pool[base + i1.x + i0.y*8u + i0.z*64u].word0);
    let c010 = extract_distance(brick_pool[base + i0.x + i1.y*8u + i0.z*64u].word0);
    let c110 = extract_distance(brick_pool[base + i1.x + i1.y*8u + i0.z*64u].word0);
    let c001 = extract_distance(brick_pool[base + i0.x + i0.y*8u + i1.z*64u].word0);
    let c101 = extract_distance(brick_pool[base + i1.x + i0.y*8u + i1.z*64u].word0);
    let c011 = extract_distance(brick_pool[base + i0.x + i1.y*8u + i1.z*64u].word0);
    let c111 = extract_distance(brick_pool[base + i1.x + i1.y*8u + i1.z*64u].word0);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}

/// Get material ID from a voxelized object at a local position (nearest neighbor).
fn sample_voxelized_material(local_pos: vec3<f32>, obj: GpuObject) -> u32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return 0u;
    }

    let brick_coord = vec3<u32>(floor(grid_pos / brick_extent));
    let flat_brick = brick_coord.x + brick_coord.y * dims.x + brick_coord.z * dims.x * dims.y;
    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return 0u;
    }

    let brick_min = vec3<f32>(brick_coord) * brick_extent - grid_size * 0.5;
    let brick_local = (local_pos - brick_min) / vs;
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let idx = slot * 512u + voxel.x + voxel.y * 8u + voxel.z * 64u;
    return extract_material_id(brick_pool[idx].word0);
}

/// Evaluate a single object at a camera-relative position.
/// Returns (distance_in_world_space, material_id).
fn evaluate_object(cam_rel_pos: vec3<f32>, obj_idx: u32) -> vec2<f32> {
    let obj = objects[obj_idx];

    if obj.sdf_type == SDF_TYPE_NONE {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }

    // Transform to object-local space.
    let local_pos = (obj.inverse_world * vec4<f32>(cam_rel_pos, 1.0)).xyz;

    var dist: f32;
    var mat_id: u32;

    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        dist = evaluate_analytical(local_pos, obj);
        mat_id = obj.material_id;
    } else {
        // SDF_TYPE_VOXELIZED
        dist = sample_voxelized(local_pos, obj);
        if dist < MAX_FLOAT * 0.5 {
            mat_id = sample_voxelized_material(local_pos, obj);
        } else {
            mat_id = 0u;
        }
    }

    // Scale correction: SDF distance in local space needs to be scaled back
    // to world space by the accumulated uniform scale.
    dist = dist * obj.accumulated_scale;

    return vec2<f32>(dist, f32(mat_id));
}

// ---------- BVH Traversal ----------

/// Traverse the BVH to find all candidate objects for a ray.
/// Uses an iterative stack-based traversal. Returns the minimum distance
/// and best material/object info by sphere-tracing through candidates.
fn ray_march_bvh(origin: vec3<f32>, dir: vec3<f32>) -> MarchResult {
    var result: MarchResult;
    result.t = MAX_FLOAT;
    result.hit = false;
    result.material_id = 0u;
    result.object_id = 0u;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    if scene.num_objects == 0u {
        return result;
    }

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let inv_dir = 1.0 / safe_dir;
    let cam_pos = camera.position.xyz;

    var t = 0.0;

    for (var step = 0u; step < scene.max_steps; step++) {
        if t > scene.max_distance {
            break;
        }

        let pos = origin + safe_dir * t;
        // Camera-relative position for object evaluation.
        // inverse_world matrices are built in camera-relative space by flatten_object().
        let cam_rel = pos - cam_pos;

        // Find minimum distance across all objects using BVH.
        var min_dist = MAX_FLOAT;
        var best_mat = 0u;
        var best_obj_id = 0u;

        // BVH stack-based traversal.
        var stack: array<u32, 32>;
        var stack_ptr = 0u;
        stack[0] = 0u;
        stack_ptr = 1u;

        while stack_ptr > 0u {
            stack_ptr -= 1u;
            let node_idx = stack[stack_ptr];
            let node = bvh_nodes[node_idx];

            let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
            let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

            // Quick AABB-point distance check: if the current marching position
            // is far from this node's AABB, we can skip it.
            let closest = clamp(pos, node_min, node_max);
            let box_dist = length(closest - pos);
            if box_dist > min_dist {
                continue;
            }

            if node.left == BVH_INVALID {
                // Leaf node — evaluate the object.
                let obj_idx = node.right_or_object;
                if obj_idx < scene.num_objects {
                    let eval = evaluate_object(cam_rel, obj_idx);
                    if eval.x < min_dist {
                        min_dist = eval.x;
                        best_mat = u32(eval.y);
                        best_obj_id = objects[obj_idx].object_id;
                    }
                }
            } else {
                // Internal node — push children.
                if stack_ptr < BVH_STACK_SIZE - 1u {
                    stack[stack_ptr] = node.left;
                    stack_ptr += 1u;
                    stack[stack_ptr] = node.right_or_object;
                    stack_ptr += 1u;
                }
            }
        }

        if min_dist < scene.hit_threshold {
            result.t = t;
            result.hit = true;
            result.material_id = best_mat;
            result.object_id = best_obj_id;
            return result;
        }

        // Step forward by the minimum distance (sphere tracing).
        t += max(min_dist, MIN_STEP);
    }

    return result;
}

/// Simple brute-force march (no BVH) for small object counts.
fn ray_march_brute(origin: vec3<f32>, dir: vec3<f32>) -> MarchResult {
    var result: MarchResult;
    result.t = MAX_FLOAT;
    result.hit = false;
    result.material_id = 0u;
    result.object_id = 0u;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let cam_pos = camera.position.xyz;

    var t = 0.0;

    for (var step = 0u; step < scene.max_steps; step++) {
        if t > scene.max_distance {
            break;
        }

        let pos = origin + safe_dir * t;
        let cam_rel = pos - cam_pos;

        var min_dist = MAX_FLOAT;
        var best_mat = 0u;
        var best_obj_id = 0u;

        for (var i = 0u; i < scene.num_objects; i++) {
            let eval = evaluate_object(cam_rel, i);
            if eval.x < min_dist {
                min_dist = eval.x;
                best_mat = u32(eval.y);
                best_obj_id = objects[i].object_id;
            }
        }

        if min_dist < scene.hit_threshold {
            result.t = t;
            result.hit = true;
            result.material_id = best_mat;
            result.object_id = best_obj_id;
            return result;
        }

        t += max(min_dist, MIN_STEP);
    }

    return result;
}

// ---------- Normal Computation ----------

/// Compute SDF at a world position by evaluating all objects (brute force).
/// Converts to camera-relative internally for correct inverse_world transforms.
fn sample_scene(pos: vec3<f32>) -> f32 {
    let cam_rel = pos - camera.position.xyz;
    var min_dist = MAX_FLOAT;
    for (var i = 0u; i < scene.num_objects; i++) {
        let eval = evaluate_object(cam_rel, i);
        min_dist = min(min_dist, eval.x);
    }
    return min_dist;
}

/// Compute surface normal via central differences (6 SDF evaluations).
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = scene.hit_threshold * 10.0;
    let nx = sample_scene(pos + vec3<f32>(e, 0.0, 0.0)) - sample_scene(pos - vec3<f32>(e, 0.0, 0.0));
    let ny = sample_scene(pos + vec3<f32>(0.0, e, 0.0)) - sample_scene(pos - vec3<f32>(0.0, e, 0.0));
    let nz = sample_scene(pos + vec3<f32>(0.0, 0.0, e)) - sample_scene(pos - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(nx, ny, nz));
}

// ---------- Entry Point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(gbuf_position));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);

    // Generate UV in [0, 1], then NDC in [-1, 1].
    let uv = (vec2<f32>(pixel.xy) + 0.5 + camera.jitter) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Camera ray.
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz
    );

    // Choose march strategy.
    var result: MarchResult;
    if arrayLength(&bvh_nodes) > 0u {
        result = ray_march_bvh(ray_origin, ray_dir);
    } else {
        result = ray_march_brute(ray_origin, ray_dir);
    }

    if result.hit {
        let hit_pos = ray_origin + ray_dir * result.t;
        let normal = compute_normal(hit_pos);

        // Write G-buffer.
        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, result.t));
        textureStore(gbuf_normal, coord, vec4<f32>(normal, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(result.material_id, 0u, 0u, 0u));

        // Motion vectors.
        let prev_clip = camera.prev_vp * vec4<f32>(hit_pos, 1.0);
        var motion = vec2<f32>(0.0);
        if prev_clip.w > 0.0 {
            let prev_ndc = prev_clip.xy / prev_clip.w;
            let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
            let curr_uv = (vec2<f32>(pixel.xy) + 0.5) / vec2<f32>(dims);
            motion = curr_uv - prev_uv;
        }
        textureStore(gbuf_motion, coord, vec4<f32>(motion, 1.0, 0.0));
    } else {
        // Sky / miss.
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
