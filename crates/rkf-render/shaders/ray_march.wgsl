// Ray march compute shader — Phase 5 (DDA traversal through sparse grid).
//
// One thread per pixel at internal resolution. Generates a camera ray,
// marches through the sparse grid via Amanatides & Woo 3D DDA,
// sphere-traces within occupied bricks, and writes the shaded result
// to the output texture.

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
}

// ---------- Bindings ----------

// Group 0: scene data
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> occupancy:  array<u32>;
@group(0) @binding(2) var<storage, read> slots:      array<u32>;
@group(0) @binding(3) var<uniform>       camera:     CameraUniforms;
@group(0) @binding(4) var<uniform>       scene:      SceneUniforms;

// Group 1: output texture
@group(1) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;

// ---------- Constants ----------

const MAX_DDA_STEPS: u32 = 256u;
const MAX_BRICK_STEPS: u32 = 64u;
const MAX_DISTANCE: f32 = 100.0;
const HIT_EPSILON: f32 = 0.001;
const MIN_STEP: f32 = 0.0005;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// CellState values (2-bit, matching Rust CellState enum)
const CELL_EMPTY: u32      = 0u;
const CELL_SURFACE: u32    = 1u;
const CELL_INTERIOR: u32   = 2u;
const CELL_VOLUMETRIC: u32 = 3u; // TODO(Phase 9): volumetric march with density accumulation

// Debug: set to true to output a heat-map of SDF evaluations per ray
// (blue = few, green = moderate, red = many). Useful for profiling.
const DEBUG_STEPS: bool = false;
const DEBUG_MAX_EVALS: f32 = 100.0; // evaluations at which heat-map saturates to red

// Per-thread counter for debug step visualization
var<private> debug_sdf_evals: u32;

// ---------- Helpers ----------

/// Extract f16 distance from word0 of a VoxelSample.
fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

// Accessor helpers for packed SceneUniforms.
fn grid_dims() -> vec3<u32>  { return scene.grid_dims.xyz; }
fn grid_origin() -> vec3<f32> { return scene.grid_origin.xyz; }
fn brick_extent() -> f32     { return scene.grid_origin.w; }
fn voxel_size() -> f32       { return scene.params.x; }

/// Convert a world-space position to grid cell coordinates.
/// Returns vec3<i32> — caller must bounds-check.
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
/// Takes precomputed inv_dir = 1.0 / direction.
/// Returns vec2(t_near, t_far). Miss when t_near > t_far or t_far < 0.
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

/// Read SDF distance from a specific brick slot at a world position.
fn sample_brick(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> f32 {
    let brick_local = (pos - brick_min) / voxel_size();
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
    let sample_idx = slot * 512u + voxel_idx;
    return extract_distance(brick_pool[sample_idx].word0);
}

/// Sphere trace within a single brick.
/// Returns MarchResult — hit=true with distance, or hit=false.
fn sphere_trace_brick(origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t_enter: f32, cell: vec3<u32>, flat: u32) -> MarchResult {
    let slot = slots[flat];
    if slot == EMPTY_SLOT {
        return MarchResult(-1.0, false);
    }

    let be = brick_extent();
    let brick_min = grid_origin() + vec3<f32>(cell) * be;
    let brick_max = brick_min + vec3<f32>(be);

    // Clip to brick AABB for precise entry/exit
    let aabb_t = ray_aabb_intersect(origin, inv_dir, brick_min, brick_max);
    let t_start = max(t_enter, max(aabb_t.x, 0.0));
    let t_end = aabb_t.y;

    if t_start > t_end {
        return MarchResult(-1.0, false);
    }

    var t = t_start;
    for (var i = 0u; i < MAX_BRICK_STEPS; i++) {
        let pos = origin + dir * t;
        let d = sample_brick(pos, brick_min, slot);
        debug_sdf_evals += 1u;

        if d < HIT_EPSILON {
            return MarchResult(t, true);
        }

        t += max(abs(d), MIN_STEP);

        if t > t_end {
            break;
        }
    }

    // Miss — return brick exit t so DDA can continue from there
    return MarchResult(t_end, false);
}

/// DDA ray march through the sparse grid.
/// Amanatides & Woo traversal: skips empty cells, sphere-traces in surface bricks.
/// Returns distance to hit or -1.0 on miss.
fn ray_march_dda(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let be = brick_extent();
    let dims_f = vec3<f32>(grid_dims());
    let g_origin = grid_origin();
    let grid_max = g_origin + dims_f * be;

    // Guard against near-zero direction components to avoid NaN from 0/0
    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let inv_dir = 1.0 / safe_dir;

    // Clip ray to grid AABB
    let aabb_t = ray_aabb_intersect(origin, inv_dir, g_origin, grid_max);
    var t_near = max(aabb_t.x, 0.0);
    let t_far = aabb_t.y;

    if t_near > t_far || t_far < 0.0 {
        return -1.0;  // ray misses grid entirely
    }

    // Nudge slightly inside to land cleanly in the first cell
    t_near += HIT_EPSILON;

    // Entry point and starting cell
    let entry = origin + safe_dir * t_near;
    let dims_i = vec3<i32>(grid_dims());
    var cell = vec3<i32>(floor((entry - g_origin) / be));
    cell = clamp(cell, vec3<i32>(0), dims_i - vec3<i32>(1));

    // DDA per-axis setup
    let step = vec3<i32>(
        select(-1, 1, safe_dir.x >= 0.0),
        select(-1, 1, safe_dir.y >= 0.0),
        select(-1, 1, safe_dir.z >= 0.0)
    );

    let t_delta = abs(vec3<f32>(be) * inv_dir);

    // t_max: parametric distance to the next cell boundary on each axis
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
            // Surface brick: sphere trace within this brick for precise hit
            let result = sphere_trace_brick(origin, safe_dir, inv_dir, t, ucell, flat);
            if result.hit {
                return result.t;
            }
            // Miss within brick — continue DDA from brick exit
        }
        // CELL_EMPTY: no geometry, skip cheaply
        // CELL_INTERIOR: fully inside an object, skip (future: could skip
        //   multiple consecutive interior cells in one step)
        // CELL_VOLUMETRIC: TODO(Phase 9) — accumulate density via volumetric march

        // Step to the axis with the smallest t_max (next cell boundary)
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

    return -1.0;
}

// ---------- Normal computation ----------

/// Sample the SDF at a world-space position using the sparse grid + brick pool.
/// Returns a large positive distance if outside the grid or in empty space.
/// Used for normal computation via central differences.
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

/// Compute a cheap normal via central differences.
fn compute_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = voxel_size() * 0.5;
    let nx = sample_sdf(pos + vec3<f32>(e, 0.0, 0.0)) - sample_sdf(pos - vec3<f32>(e, 0.0, 0.0));
    let ny = sample_sdf(pos + vec3<f32>(0.0, e, 0.0)) - sample_sdf(pos - vec3<f32>(0.0, e, 0.0));
    let nz = sample_sdf(pos + vec3<f32>(0.0, 0.0, e)) - sample_sdf(pos - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(nx, ny, nz));
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(output));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    // Generate UV in [0, 1], then NDC in [-1, 1]
    let uv = (vec2<f32>(pixel.xy) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;

    // Camera ray
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz
    );

    // March via DDA
    debug_sdf_evals = 0u;
    let t = ray_march_dda(ray_origin, ray_dir);

    var color: vec3<f32>;
    if DEBUG_STEPS {
        // Heat-map: blue (0) → green (50%) → red (100%) based on SDF evaluations
        let ratio = clamp(f32(debug_sdf_evals) / DEBUG_MAX_EVALS, 0.0, 1.0);
        if ratio < 0.5 {
            let s = ratio * 2.0;
            color = vec3<f32>(0.0, s, 1.0 - s); // blue → green
        } else {
            let s = (ratio - 0.5) * 2.0;
            color = vec3<f32>(s, 1.0 - s, 0.0); // green → red
        }
    } else if t >= 0.0 {
        // Hit — basic directional lighting
        let hit_pos = ray_origin + ray_dir * t;
        let normal = compute_normal(hit_pos);
        let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
        let ndl = max(dot(normal, light_dir), 0.0);
        let ambient = 0.1;
        color = vec3<f32>(ambient + ndl * 0.9);
    } else {
        // Miss — dark background
        color = vec3<f32>(0.02, 0.02, 0.03);
    }

    textureStore(output, vec2<i32>(pixel.xy), vec4<f32>(color, 1.0));
}
