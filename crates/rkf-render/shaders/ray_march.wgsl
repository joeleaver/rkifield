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
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const MAX_FLOAT: f32 = 3.402823e+38;
const HIT_EPSILON: f32 = 0.001;
const MIN_STEP: f32 = 0.0005;

// BVH traversal stack depth.
const BVH_STACK_SIZE: u32 = 32u;

// Tile culling constants (must match tile_object_cull.wgsl and Rust TileObjectCullPass).
const OBJECT_TILE_SIZE: u32 = 16u;
const TILE_MAX_OBJECTS: u32 = 32u;

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
    accumulated_scale_x: f32,    // 4 bytes @ offset 148
    accumulated_scale_y: f32,    // 4 bytes @ offset 152
    accumulated_scale_z: f32,    // 4 bytes @ offset 156
    lod_level: u32,              // 4 bytes @ offset 160
    object_id: u32,              // 4 bytes @ offset 164
    primitive_type: u32,         // 4 bytes @ offset 168
    // Tight local-space AABB of allocated bricks (for empty-space skipping).
    // Zero for analytical objects (disables optimization).
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
    obj_idx: u32,
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

// Group 2: per-tile object lists (from tile_object_cull pass, read-only)
@group(2) @binding(0) var<storage, read> tile_object_indices: array<u32>;
@group(2) @binding(1) var<storage, read> tile_object_counts: array<u32>;

// Group 3: coarse acceleration field (for empty-space skipping)
struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,  // camera-relative origin (minimum corner)
    dims: vec4<u32>,            // field dimensions in cells (x, y, z, 0)
    voxel_size: f32,
    inv_voxel_size: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(3) @binding(0) var coarse_field: texture_3d<f32>;
@group(3) @binding(1) var coarse_sampler: sampler;
@group(3) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

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

// ---------- Coarse Field Sampling ----------

/// Sample the coarse acceleration field at a camera-relative position.
/// Returns the conservative distance to the nearest surface (based on AABB distance).
/// Returns 0.0 if the position is outside the field bounds (no skip info).
fn sample_coarse_field(cam_rel_pos: vec3<f32>) -> f32 {
    let field_pos = cam_rel_pos - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0; // Outside field — no skip info, fall through to per-object eval.
    }
    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
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

/// Sample a single voxel from a brick map at global grid coordinates.
/// Returns the SDF distance, or a large fallback for empty/out-of-bounds slots.
/// Sample a single voxel for ray-march stepping.
/// EMPTY_SLOT returns brick_extent so rays skip the entire empty brick in one step.
fn sample_voxel_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                    total_voxels: vec3<i32>, vs: f32) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        // Empty brick: no geometry here. Return brick_extent so the ray steps
        // completely over this brick in one iteration — empty slots must never
        // slow down the ray marcher or contribute shadow penumbra.
        return vs * 8.0;
    }
    if slot == INTERIOR_SLOT {
        return -(vs * 2.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_distance(brick_pool[idx].word0);
}


/// Catmull-Rom basis weights for parameter t in [0,1].
/// Returns weights for the 4 control points: p[-1], p[0], p[1], p[2].
fn catmull_rom_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,           // w[-1]
         1.5 * t3 - 2.5 * t2 + 1.0,          // w[0]
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,      // w[1]
         0.5 * t3 - 0.5 * t2,                 // w[2]
    );
}

/// Catmull-Rom basis weight derivatives for parameter t in [0,1].
/// Returns d(weight)/dt for the 4 control points.
fn catmull_rom_dweights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    return vec4<f32>(
        -1.5 * t2 + 2.0 * t - 0.5,           // dw[-1]/dt
         4.5 * t2 - 5.0 * t,                  // dw[0]/dt
        -4.5 * t2 + 4.0 * t + 0.5,            // dw[1]/dt
         1.5 * t2 - t,                         // dw[2]/dt
    );
}

/// 1D Catmull-Rom interpolation over 4 values using precomputed weights.
fn cr_interp(w: vec4<f32>, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    return w.x * v0 + w.y * v1 + w.z * v2 + w.w * v3;
}

/// Sample a voxelized object's brick map at a local-space position.
/// Uses trilinear interpolation (8 voxel samples) for conservative distance
/// values that never overshoot — critical for safe ray march stepping.
///
/// Surface normals use sample_density_gradient (binary sign field) which is
/// immune to SDF magnitude artifacts and medial-axis creases.
fn sample_voxelized(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    // Grid starts at -grid_size/2 (object origin = center of grid).
    let grid_pos = local_pos + grid_size * 0.5;

    // Clamp grid_pos to valid range and compute out-of-bounds distance.
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);

    // If we're far outside the grid (> 2 brick extents), early-out with large value.
    if outside_dist > brick_extent * 2.0 {
        return outside_dist;
    }

    // Geometry AABB early-out: if a tight allocated-brick AABB is set (non-zero),
    // and the sample point is outside it by more than one brick extent, return the
    // distance to the geometry AABB directly. This skips the slow vs*2.0 stepping
    // through the empty expanded region that grow_brick_map_if_needed adds.
    let geom_min = vec3<f32>(obj.geometry_aabb_min_x, obj.geometry_aabb_min_y, obj.geometry_aabb_min_z);
    let geom_max = vec3<f32>(obj.geometry_aabb_max_x, obj.geometry_aabb_max_y, obj.geometry_aabb_max_z);
    if geom_max.x > geom_min.x {
        let geom_closest = clamp(local_pos, geom_min, geom_max);
        let geom_dist = length(local_pos - geom_closest);
        if geom_dist > brick_extent {
            return geom_dist + outside_dist;
        }
    }

    // Convert to continuous voxel coordinates. Voxel centers are at integers
    // (the -0.5 shifts so that the center of voxel [0] is at coordinate 0.0).
    let voxel_coord = clamped / vs - vec3<f32>(0.5);

    // Lower corner of the trilinear cell (integer voxel coordinates).
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);

    let total_voxels = vec3<i32>(dims) * 8;

    // Sample the 8 corners — each may resolve to a different brick.
    let c000 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 0), dims, total_voxels, vs);
    let c100 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 0), dims, total_voxels, vs);
    let c010 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 0), dims, total_voxels, vs);
    let c110 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 0), dims, total_voxels, vs);
    let c001 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 1), dims, total_voxels, vs);
    let c101 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 1), dims, total_voxels, vs);
    let c011 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 1), dims, total_voxels, vs);
    let c111 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 1), dims, total_voxels, vs);

    // Trilinear interpolation.
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z) + outside_dist;
}

/// Sample a voxelized object as a binary density field (solid=1, empty=0).
///
/// Converts each of the 8 trilinear corner samples to binary (sign-only)
/// BEFORE interpolating.  The resulting smooth density field has no
/// dependence on stored SDF magnitudes — only the sign (solid vs empty)
/// matters.  This eliminates medial-axis creases from overlapping brush
/// strokes and brick-boundary artifacts from magnitude drift.
fn sample_density(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;

    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));

    // Outside grid → empty.
    if any(grid_pos < vec3<f32>(-brick_extent)) || any(grid_pos > grid_size + vec3<f32>(brick_extent)) {
        return 0.0;
    }

    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t  = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;

    // Sample 8 corners, convert to binary (solid=1, empty=0) before lerping.
    let b000 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0,0,0), dims, total_voxels, vs) < 0.0);
    let b100 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1,0,0), dims, total_voxels, vs) < 0.0);
    let b010 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0,1,0), dims, total_voxels, vs) < 0.0);
    let b110 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1,1,0), dims, total_voxels, vs) < 0.0);
    let b001 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0,0,1), dims, total_voxels, vs) < 0.0);
    let b101 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1,0,1), dims, total_voxels, vs) < 0.0);
    let b011 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0,1,1), dims, total_voxels, vs) < 0.0);
    let b111 = select(0.0, 1.0, sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1,1,1), dims, total_voxels, vs) < 0.0);

    let b00 = mix(b000, b100, t.x);
    let b10 = mix(b010, b110, t.x);
    let b01 = mix(b001, b101, t.x);
    let b11 = mix(b011, b111, t.x);
    let b0  = mix(b00,  b10,  t.y);
    let b1  = mix(b01,  b11,  t.y);
    return mix(b0, b1, t.z);
}

/// Compute the surface normal gradient via 6-sample central difference.
///
/// Gradient of stored SDF values via 6-sample central difference.
fn sample_voxelized_gradient(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let eps = obj.voxel_size * 1.5;
    let gx = sample_voxelized(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_voxelized(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_voxelized(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_voxelized(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_voxelized(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_voxelized(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    return vec3<f32>(gx, gy, gz);
}

/// Compute surface normal from the binary density field gradient.
///
/// Samples the binary density (solid=1, empty=0) at ±eps offsets and returns
/// the central-difference gradient.  Because it uses only the SIGN of stored
/// distances — never the magnitude — it is immune to:
///   - medial-axis SDF discontinuities between overlapping brush strokes
///   - EDT magnitude drift / incorrect distances from binary stamp
///   - EMPTY_SLOT / INTERIOR_SLOT sentinel value artifacts
///
/// The density field is 1 inside (solid) and 0 outside (empty).  Its gradient
/// points inward; negate to obtain the outward surface normal.
fn sample_density_gradient(local_pos: vec3<f32>, obj: GpuObject) -> vec3<f32> {
    let eps = obj.voxel_size * 2.0;
    let gx = sample_density(local_pos + vec3<f32>(eps, 0.0, 0.0), obj)
           - sample_density(local_pos - vec3<f32>(eps, 0.0, 0.0), obj);
    let gy = sample_density(local_pos + vec3<f32>(0.0, eps, 0.0), obj)
           - sample_density(local_pos - vec3<f32>(0.0, eps, 0.0), obj);
    let gz = sample_density(local_pos + vec3<f32>(0.0, 0.0, eps), obj)
           - sample_density(local_pos - vec3<f32>(0.0, 0.0, eps), obj);
    // Density increases toward interior → gradient is inward.  Negate for outward normal.
    return -vec3<f32>(gx, gy, gz);
}

/// Get material ID from a voxelized object at a local position (nearest neighbor).
/// Uses global grid coordinates for correct cross-brick lookup.
fn sample_voxelized_material(local_pos: vec3<f32>, obj: GpuObject) -> u32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;

    if any(grid_pos < vec3<f32>(0.0)) || any(grid_pos >= grid_size) {
        return 0u;
    }

    // Nearest voxel in global grid coordinates.
    let voxel_coord = grid_pos / vs;
    let vc = clamp(vec3<i32>(floor(voxel_coord)), vec3<i32>(0), vec3<i32>(dims) * 8 - vec3<i32>(1));

    let brick = vec3<u32>(vc / vec3<i32>(8));
    let local = vec3<u32>(vc % vec3<i32>(8));

    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj.brick_map_offset + flat_brick];
    if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
        return 0u;
    }

    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_material_id(brick_pool[idx].word0);
}

/// Evaluate a single object at a world-space position.
/// Returns (distance_in_world_space, material_id).
fn evaluate_object(world_pos: vec3<f32>, obj_idx: u32) -> vec2<f32> {
    let obj = objects[obj_idx];

    if obj.sdf_type == SDF_TYPE_NONE {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }

    // Transform world-space position to object-local space.
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;

    var dist: f32;
    var mat_id: u32;

    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        dist = evaluate_analytical(local_pos, obj);
        mat_id = obj.material_id;
    } else {
        // SDF_TYPE_VOXELIZED
        dist = sample_voxelized(local_pos, obj);
        // Only look up material for near-surface hits (dist is finite for all
        // positions now — large values mean we're far from the surface).
        if dist < obj.voxel_size * 16.0 {
            mat_id = sample_voxelized_material(local_pos, obj);
        } else {
            mat_id = 0u;
        }
    }

    // Scale correction: conservative min-component scale preserves SDF validity.
    // Never overshoots; extra march steps proportional to scale ratio.
    let min_scale = min(obj.accumulated_scale_x, min(obj.accumulated_scale_y, obj.accumulated_scale_z));
    dist = dist * min_scale;

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
    result.obj_idx = 0u;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    if scene.num_objects == 0u {
        return result;
    }

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let inv_dir = 1.0 / safe_dir;

    var t = 0.0;

    for (var step = 0u; step < scene.max_steps; step++) {
        if t > scene.max_distance {
            break;
        }

        // pos is in world-space (ray_origin is camera world position).
        let pos = origin + safe_dir * t;

        // Find minimum distance across all objects using BVH.
        // BVH AABBs and object inverse_world are both in world space.
        var min_dist = MAX_FLOAT;
        var best_mat = 0u;
        var best_obj_id = 0u;
        var best_obj_idx = 0u;

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
                // Leaf node — evaluate the object (world-space position).
                let leaf_obj_idx = node.right_or_object;
                if leaf_obj_idx < scene.num_objects {
                    let eval = evaluate_object(pos, leaf_obj_idx);
                    if eval.x < min_dist {
                        min_dist = eval.x;
                        best_mat = u32(eval.y);
                        best_obj_id = objects[leaf_obj_idx].object_id;
                        best_obj_idx = leaf_obj_idx;
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
            result.obj_idx = best_obj_idx;
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
    result.obj_idx = 0u;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));

    var t = 0.0;

    for (var step = 0u; step < scene.max_steps; step++) {
        if t > scene.max_distance {
            break;
        }

        let pos = origin + safe_dir * t;

        var min_dist = MAX_FLOAT;
        var best_mat = 0u;
        var best_obj_id = 0u;
        var best_obj_idx = 0u;

        for (var i = 0u; i < scene.num_objects; i++) {
            let eval = evaluate_object(pos, i);
            if eval.x < min_dist {
                min_dist = eval.x;
                best_mat = u32(eval.y);
                best_obj_id = objects[i].object_id;
                best_obj_idx = i;
            }
        }

        if min_dist < scene.hit_threshold {
            result.t = t;
            result.hit = true;
            result.material_id = best_mat;
            result.object_id = best_obj_id;
            result.obj_idx = best_obj_idx;
            return result;
        }

        t += max(min_dist, MIN_STEP);
    }

    return result;
}

/// Threshold below which we switch from coarse field to per-object evaluation.
/// When the coarse field distance is below this, objects are nearby and we need
/// precise SDF evaluation from the tile's object list.
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

/// Tiled ray march with coarse field acceleration.
///
/// Two-phase marching:
/// 1. When coarse field reports large distance, step by that amount (empty-space skip).
/// 2. When coarse field distance is small (near surfaces), evaluate per-tile objects.
///
/// The tile_object_cull pass projects object AABBs and writes per-tile object
/// index lists. Phase 2 reads the list for the current pixel's tile and only
/// evaluates those objects, typically 3-5 instead of all scene objects.
fn ray_march_tiled(origin: vec3<f32>, dir: vec3<f32>, pixel: vec2<u32>) -> MarchResult {
    var result: MarchResult;
    result.t = MAX_FLOAT;
    result.hit = false;
    result.material_id = 0u;
    result.object_id = 0u;
    result.obj_idx = 0u;
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    // Compute tile ID from pixel coordinates.
    let tile_x = pixel.x / OBJECT_TILE_SIZE;
    let tile_y = pixel.y / OBJECT_TILE_SIZE;
    let dims = vec2<u32>(textureDimensions(gbuf_position));
    let num_tiles_x = (dims.x + OBJECT_TILE_SIZE - 1u) / OBJECT_TILE_SIZE;
    let tile_id = tile_y * num_tiles_x + tile_x;

    let count = tile_object_counts[tile_id];
    if count == 0u {
        return result; // No objects in this tile — sky.
    }

    let base = tile_id * TILE_MAX_OBJECTS;
    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let cam_pos = camera.position.xyz;

    var t = 0.0;

    for (var step = 0u; step < scene.max_steps; step++) {
        if t > scene.max_distance {
            break;
        }

        let pos = origin + safe_dir * t;

        // Phase 1: coarse field empty-space skipping (camera-relative).
        // The coarse field is centered on the camera — convert to cam-rel.
        let cam_rel = pos - cam_pos;
        let coarse_dist = sample_coarse_field(cam_rel);
        if coarse_dist > COARSE_NEAR_THRESHOLD {
            t += coarse_dist;
            continue;
        }

        // Phase 2: near surfaces — evaluate per-tile objects (world-space).
        var min_dist = MAX_FLOAT;
        var best_mat = 0u;
        var best_obj_id = 0u;
        var best_obj_idx = 0u;

        for (var i = 0u; i < count; i++) {
            let obj_idx = tile_object_indices[base + i];
            let eval = evaluate_object(pos, obj_idx);
            if eval.x < min_dist {
                min_dist = eval.x;
                best_mat = u32(eval.y);
                best_obj_id = objects[obj_idx].object_id;
                best_obj_idx = obj_idx;
            }
        }

        if min_dist < scene.hit_threshold {
            result.t = t;
            result.hit = true;
            result.material_id = best_mat;
            result.object_id = best_obj_id;
            result.obj_idx = best_obj_idx;
            return result;
        }

        t += max(min_dist, MIN_STEP);
    }

    return result;
}

// ---------- Normal Computation ----------

/// Evaluate a single object's SDF at a world-space position.
/// Used for per-object normal computation.
fn sample_object(pos: vec3<f32>, obj_idx: u32) -> f32 {
    return evaluate_object(pos, obj_idx).x;
}

/// Compute surface normal for a specific object at a world-space hit position.
///
/// Voxelized objects use the analytical gradient of the trilinear interpolation
/// field. This avoids the finite-difference artifact where epsilon-offset samples
/// cross into EMPTY_SLOT bricks and return the large background value (vs*32),
/// producing incorrect gradient directions at narrow-band boundaries.
///
/// Analytical objects use central finite differences with a small epsilon.
fn compute_normal_for_object(pos: vec3<f32>, obj_idx: u32) -> vec3<f32> {
    let obj = objects[obj_idx];

    if obj.sdf_type == SDF_TYPE_VOXELIZED {
        // Transform hit position (world-space) to object-local space.
        let local_pos = (obj.inverse_world * vec4<f32>(pos, 1.0)).xyz;

        // Gradient in local space from trilinear SDF values.
        let local_grad = sample_voxelized_gradient(local_pos, obj);

        // Transform gradient from local → world space.
        let world_grad = (transpose(obj.inverse_world) * vec4<f32>(local_grad, 0.0)).xyz;

        let len = length(world_grad);
        if len < 1e-10 {
            return vec3<f32>(0.0, 1.0, 0.0);
        }
        return world_grad / len;
    } else {
        // Small epsilon for analytical SDFs (smooth gradients everywhere).
        let e = scene.hit_threshold * 10.0;
        let nx = sample_object(pos + vec3<f32>(e, 0.0, 0.0), obj_idx)
               - sample_object(pos - vec3<f32>(e, 0.0, 0.0), obj_idx);
        let ny = sample_object(pos + vec3<f32>(0.0, e, 0.0), obj_idx)
               - sample_object(pos - vec3<f32>(0.0, e, 0.0), obj_idx);
        let nz = sample_object(pos + vec3<f32>(0.0, 0.0, e), obj_idx)
               - sample_object(pos - vec3<f32>(0.0, 0.0, e), obj_idx);
        return normalize(vec3<f32>(nx, ny, nz));
    }
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

    // Choose march strategy: prefer tiled > BVH > brute.
    var result: MarchResult;
    if arrayLength(&tile_object_counts) > 0u {
        result = ray_march_tiled(ray_origin, ray_dir, pixel.xy);
    } else if arrayLength(&bvh_nodes) > 0u {
        result = ray_march_bvh(ray_origin, ray_dir);
    } else {
        result = ray_march_brute(ray_origin, ray_dir);
    }

    if result.hit {
        let hit_pos = ray_origin + ray_dir * result.t;
        let normal = compute_normal_for_object(hit_pos, result.obj_idx);

        // Write G-buffer.
        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, result.t));
        textureStore(gbuf_normal, coord, vec4<f32>(normal, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(result.material_id | (result.object_id << 24u), 0u, 0u, 0u));

        // Re-evaluate SDF distance at hit point for debug visualization.
        var sdf_at_hit = 0.0;
        let hit_obj = objects[result.obj_idx];
        if hit_obj.sdf_type == SDF_TYPE_VOXELIZED {
            let local_hit = (hit_obj.inverse_world * vec4<f32>(hit_pos, 1.0)).xyz;
            sdf_at_hit = sample_voxelized(local_hit, hit_obj);
        }

        // Motion vectors.
        let prev_clip = camera.prev_vp * vec4<f32>(hit_pos, 1.0);
        var motion = vec2<f32>(0.0);
        if prev_clip.w > 0.0 {
            let prev_ndc = prev_clip.xy / prev_clip.w;
            let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
            let curr_uv = (vec2<f32>(pixel.xy) + 0.5) / vec2<f32>(dims);
            motion = curr_uv - prev_uv;
        }
        textureStore(gbuf_motion, coord, vec4<f32>(motion, 1.0, sdf_at_hit));
    } else {
        // Sky / miss.
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
