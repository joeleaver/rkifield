// Ray march compute shader — Phase 6 (G-buffer output).
//
// One thread per pixel at internal resolution. Generates a camera ray,
// marches through the sparse grid via Amanatides & Woo 3D DDA,
// sphere-traces within occupied bricks, and writes G-buffer data
// (position, normal, material ID, motion vectors) to 4 output textures.

// ---------- Types ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 16 = u16 material_id
    word1: u32, // byte0 = blend_weight, byte1 = secondary_id, byte2 = flags, byte3 = reserved
}

struct CameraUniforms {
    position: vec4<f32>,   // xyz + pad
    forward:  vec4<f32>,   // xyz + pad  (unit)
    right:    vec4<f32>,   // xyz + pad  (scaled by fov * aspect)
    up:       vec4<f32>,   // xyz + pad  (scaled by fov)
    resolution: vec2<f32>, // width, height
    _pad: vec2<f32>,
}

struct SceneUniforms {
    grid_dims:    vec4<u32>,   // xyz = dimensions, w = unused
    grid_origin:  vec4<f32>,   // xyz = origin, w = brick_extent
    params:       vec4<f32>,   // x = voxel_size, yzw = unused
}

struct MarchResult {
    t: f32,
    hit: bool,
    material_id: u32,
    secondary_id_and_flags: u32,
    blend_weight: f32,
}

// ---------- Bindings ----------

// Group 0: scene data
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> occupancy:  array<u32>;
@group(0) @binding(2) var<storage, read> slots:      array<u32>;
@group(0) @binding(3) var<uniform>       camera:     CameraUniforms;
@group(0) @binding(4) var<uniform>       scene:      SceneUniforms;

// Group 1: G-buffer output textures
@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<rg16uint, write>;
@group(1) @binding(3) var gbuf_motion:   texture_storage_2d<rg16float, write>;

// ---------- Constants ----------

const MAX_DDA_STEPS: u32 = 256u;
const MAX_BRICK_STEPS: u32 = 64u;
const MAX_DISTANCE: f32 = 100.0;
const MAX_FLOAT: f32 = 3.402823e+38;
const HIT_EPSILON: f32 = 0.001;
const MIN_STEP: f32 = 0.0005;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// CellState values (2-bit, matching Rust CellState enum)
const CELL_EMPTY: u32      = 0u;
const CELL_SURFACE: u32    = 1u;
const CELL_INTERIOR: u32   = 2u;
const CELL_VOLUMETRIC: u32 = 3u;

// ---------- Helpers ----------

/// Extract f16 distance from word0 of a VoxelSample.
fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

/// Extract u16 material_id from word0 of a VoxelSample.
fn extract_material_id(word0: u32) -> u32 {
    return word0 >> 16u;
}

/// Extract blend_weight (byte 0 of word1) as f32 in [0, 1].
fn extract_blend_weight(word1: u32) -> f32 {
    return f32(word1 & 0xFFu) / 255.0;
}

/// Extract secondary_id (byte 1 of word1).
fn extract_secondary_id(word1: u32) -> u32 {
    return (word1 >> 8u) & 0xFFu;
}

/// Extract flags (byte 2 of word1).
fn extract_flags(word1: u32) -> u32 {
    return (word1 >> 16u) & 0xFFu;
}

// Accessor helpers for packed SceneUniforms.
fn grid_dims() -> vec3<u32>  { return scene.grid_dims.xyz; }
fn grid_origin() -> vec3<f32> { return scene.grid_origin.xyz; }
fn brick_extent() -> f32     { return scene.grid_origin.w; }
fn voxel_size() -> f32       { return scene.params.x; }

/// Convert a world-space position to grid cell coordinates.
fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - grid_origin();
    return vec3<i32>(floor(local / brick_extent()));
}

/// Check if cell coordinates are within grid bounds.
fn cell_in_bounds(cell: vec3<i32>) -> bool {
    return all(cell >= vec3<i32>(0)) && all(vec3<u32>(cell) < grid_dims());
}

/// Get the cell state (2-bit) for a flat cell index.
fn get_cell_state(flat: u32) -> u32 {
    let word_idx = flat / 16u;
    let bit_offset = (flat % 16u) * 2u;
    return (occupancy[word_idx] >> bit_offset) & 3u;
}

/// Flat index from cell coordinates.
fn cell_flat_index(cell: vec3<u32>) -> u32 {
    let d = grid_dims();
    return cell.x + cell.y * d.x + cell.z * d.x * d.y;
}

// ---------- DDA Ray March ----------

/// Ray-AABB intersection using the slab method.
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

/// Read the full VoxelSample at a world position within a brick.
/// Returns (distance, sample_index) where sample_index can be used
/// to read material data.
fn sample_brick_full(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> u32 {
    let brick_local = (pos - brick_min) / voxel_size();
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
    return slot * 512u + voxel_idx;
}

/// Read SDF distance from a specific brick slot at a world position.
fn sample_brick(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> f32 {
    let idx = sample_brick_full(pos, brick_min, slot);
    return extract_distance(brick_pool[idx].word0);
}

/// Sphere trace within a single brick, returning full material info on hit.
fn sphere_trace_brick(origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t_enter: f32, cell: vec3<u32>, flat: u32) -> MarchResult {
    let slot = slots[flat];
    if slot == EMPTY_SLOT {
        return MarchResult(-1.0, false, 0u, 0u, 0.0);
    }

    let be = brick_extent();
    let brick_min = grid_origin() + vec3<f32>(cell) * be;
    let brick_max = brick_min + vec3<f32>(be);

    let aabb_t = ray_aabb_intersect(origin, inv_dir, brick_min, brick_max);
    let t_start = max(t_enter, max(aabb_t.x, 0.0));
    let t_end = aabb_t.y;

    if t_start > t_end {
        return MarchResult(-1.0, false, 0u, 0u, 0.0);
    }

    var t = t_start;
    for (var i = 0u; i < MAX_BRICK_STEPS; i++) {
        let pos = origin + dir * t;
        let clamped = clamp(pos, brick_min + vec3<f32>(MIN_STEP),
                                 brick_max - vec3<f32>(MIN_STEP));

        let sample_idx = sample_brick_full(clamped, brick_min, slot);
        let sample = brick_pool[sample_idx];
        let d = extract_distance(sample.word0);

        if d < HIT_EPSILON {
            let mat_id = extract_material_id(sample.word0);
            let blend = extract_blend_weight(sample.word1);
            let sec_id = extract_secondary_id(sample.word1);
            let flags = extract_flags(sample.word1);
            // Pack secondary_id in lower 8 bits, flags in upper 8 bits
            let sec_and_flags = sec_id | (flags << 8u);
            return MarchResult(t, true, mat_id, sec_and_flags, blend);
        }

        t += max(abs(d), MIN_STEP);

        if t > t_end {
            break;
        }
    }

    return MarchResult(t_end, false, 0u, 0u, 0.0);
}

/// DDA ray march through the sparse grid. Returns full MarchResult with material info.
fn ray_march_dda(origin: vec3<f32>, dir: vec3<f32>) -> MarchResult {
    let be = brick_extent();
    let dims_f = vec3<f32>(grid_dims());
    let g_origin = grid_origin();
    let grid_max = g_origin + dims_f * be;

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let inv_dir = 1.0 / safe_dir;

    let aabb_t = ray_aabb_intersect(origin, inv_dir, g_origin, grid_max);
    var t_near = max(aabb_t.x, 0.0);
    let t_far = aabb_t.y;

    if t_near > t_far || t_far < 0.0 {
        return MarchResult(-1.0, false, 0u, 0u, 0.0);
    }

    t_near += HIT_EPSILON;

    let entry = origin + safe_dir * t_near;
    let dims_i = vec3<i32>(grid_dims());
    var cell = vec3<i32>(floor((entry - g_origin) / be));
    cell = clamp(cell, vec3<i32>(0), dims_i - vec3<i32>(1));

    let step = vec3<i32>(
        select(-1, 1, safe_dir.x >= 0.0),
        select(-1, 1, safe_dir.y >= 0.0),
        select(-1, 1, safe_dir.z >= 0.0)
    );

    let t_delta = abs(vec3<f32>(be) * inv_dir);

    let cell_min = g_origin + vec3<f32>(cell) * be;
    let cell_max = cell_min + vec3<f32>(be);
    let next_boundary = select(cell_min, cell_max, safe_dir >= vec3<f32>(0.0));
    var t_max = (next_boundary - origin) * inv_dir;

    var t = t_near;

    for (var i = 0u; i < MAX_DDA_STEPS; i++) {
        if !cell_in_bounds(cell) || t > MAX_DISTANCE {
            break;
        }

        let ucell = vec3<u32>(cell);
        let flat = cell_flat_index(ucell);
        let state = get_cell_state(flat);

        if state == CELL_SURFACE {
            let result = sphere_trace_brick(origin, safe_dir, inv_dir, t, ucell, flat);
            if result.hit {
                return result;
            }
        }

        if t_max.x < t_max.y && t_max.x < t_max.z {
            t = t_max.x;
            t_max.x += t_delta.x;
            cell.x += step.x;
        } else if t_max.y < t_max.z {
            t = t_max.y;
            t_max.y += t_delta.y;
            cell.y += step.y;
        } else {
            t = t_max.z;
            t_max.z += t_delta.z;
            cell.z += step.z;
        }
    }

    return MarchResult(-1.0, false, 0u, 0u, 0.0);
}

// ---------- Normal computation ----------

/// Sample the SDF at a world-space position using the sparse grid + brick pool.
fn sample_sdf(pos: vec3<f32>) -> f32 {
    let cell_i = world_to_cell(pos);
    let be = brick_extent();

    if !cell_in_bounds(cell_i) {
        return be;
    }

    let cell = vec3<u32>(cell_i);
    let flat = cell_flat_index(cell);
    let state = get_cell_state(flat);

    if state == CELL_EMPTY {
        return be * 0.5;
    }
    if state == CELL_INTERIOR {
        return -be * 0.5;
    }
    if state == CELL_SURFACE {
        let slot = slots[flat];
        if slot == EMPTY_SLOT {
            return be * 0.5;
        }
        let brick_min = grid_origin() + vec3<f32>(cell) * be;
        return sample_brick(pos, brick_min, slot);
    }
    return be * 0.5;
}

/// Compute surface normal via central differences (6 SDF evaluations).
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = voxel_size() * 1.5;
    let nx = sample_sdf(pos + vec3<f32>(e, 0.0, 0.0)) - sample_sdf(pos - vec3<f32>(e, 0.0, 0.0));
    let ny = sample_sdf(pos + vec3<f32>(0.0, e, 0.0)) - sample_sdf(pos - vec3<f32>(0.0, e, 0.0));
    let nz = sample_sdf(pos + vec3<f32>(0.0, 0.0, e)) - sample_sdf(pos - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(nx, ny, nz));
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(gbuf_position));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);

    // Generate UV in [0, 1], then NDC in [-1, 1]
    let uv = (vec2<f32>(pixel.xy) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;

    // Camera ray
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz
    );

    // March via DDA
    let result = ray_march_dda(ray_origin, ray_dir);

    if result.hit {
        let hit_pos = ray_origin + ray_dir * result.t;
        let normal = compute_normal(hit_pos);

        // Write G-buffer
        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, result.t));
        textureStore(gbuf_normal, coord, vec4<f32>(normal, result.blend_weight));
        textureStore(gbuf_material, coord, vec2<u32>(result.material_id, result.secondary_id_and_flags));
        textureStore(gbuf_motion, coord, vec2<f32>(0.0, 0.0));
    } else {
        // Sky / miss — encode as MAX_FLOAT hit distance
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec2<u32>(0u, 0u));
        textureStore(gbuf_motion, coord, vec2<f32>(0.0, 0.0));
    }
}
