// Ray march compute shader — Phase 6+ (G-buffer output, clipmap LOD).
//
// One thread per pixel at internal resolution. Generates a camera ray,
// marches through the sparse grid via Amanatides & Woo 3D DDA,
// sphere-traces within occupied bricks, and writes G-buffer data
// (position, normal, material ID, motion vectors) to 4 output textures.
//
// When clipmap.num_levels > 0, uses multi-level LOD traversal instead
// of the single-grid path.

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
    jitter: vec2<f32>,     // sub-pixel jitter in pixel units
    prev_vp: mat4x4<f32>, // previous frame view-projection
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
    level: u32,
}

// ---------- Clipmap types ----------

struct ClipmapLevel {
    params: vec4<f32>,      // voxel_size, brick_extent, radius, 0
    grid_dims: vec4<u32>,   // dim_x, dim_y, dim_z, total_cells
    grid_origin: vec4<f32>, // origin_x, origin_y, origin_z, 0
    offsets: vec4<u32>,     // occupancy_offset, slot_offset, 0, 0
}

struct ClipmapUniforms {
    num_levels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    levels: array<ClipmapLevel, 5>,
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
@group(1) @binding(2) var gbuf_material: texture_storage_2d<r32uint, write>;
@group(1) @binding(3) var gbuf_motion:   texture_storage_2d<rg32float, write>;

// Group 2: Clipmap data
@group(2) @binding(0) var<storage, read> cm_occupancy: array<u32>;
@group(2) @binding(1) var<storage, read> cm_slots:     array<u32>;
@group(2) @binding(2) var<uniform>       clipmap:      ClipmapUniforms;

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

// Accessor helpers for packed SceneUniforms (single-grid path).
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

// ---------- Clipmap accessor helpers ----------

/// Grid dimensions for a clipmap level.
fn cm_grid_dims(level: u32) -> vec3<u32> {
    return clipmap.levels[level].grid_dims.xyz;
}

/// Grid origin for a clipmap level.
fn cm_grid_origin(level: u32) -> vec3<f32> {
    return clipmap.levels[level].grid_origin.xyz;
}

/// Brick extent (world-space size of one brick) for a clipmap level.
fn cm_brick_extent(level: u32) -> f32 {
    return clipmap.levels[level].params.y;
}

/// Voxel size for a clipmap level.
fn cm_voxel_size(level: u32) -> f32 {
    return clipmap.levels[level].params.x;
}

/// Radius (half-extent of the level's coverage) for a clipmap level.
fn cm_radius(level: u32) -> f32 {
    return clipmap.levels[level].params.z;
}

/// Check if cell coordinates are within bounds for a clipmap level.
fn cm_cell_in_bounds(cell: vec3<i32>, level: u32) -> bool {
    return all(cell >= vec3<i32>(0)) && all(vec3<u32>(cell) < cm_grid_dims(level));
}

/// Flat index from cell coordinates for a clipmap level.
fn cm_cell_flat_index(cell: vec3<u32>, level: u32) -> u32 {
    let d = cm_grid_dims(level);
    return cell.x + cell.y * d.x + cell.z * d.x * d.y;
}

/// Get cell state from the combined clipmap occupancy buffer.
fn cm_get_cell_state(flat: u32, level: u32) -> u32 {
    let base = clipmap.levels[level].offsets.x;
    let word_idx = (base + flat) / 16u;
    let bit_offset = ((base + flat) % 16u) * 2u;
    return (cm_occupancy[word_idx] >> bit_offset) & 3u;
}

/// Get slot index from the combined clipmap slot buffer.
fn cm_get_slot(flat: u32, level: u32) -> u32 {
    let base = clipmap.levels[level].offsets.y;
    return cm_slots[base + flat];
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
/// Returns the flat index into brick_pool for the sample.
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

/// Level-parameterized version of sample_brick_full for clipmap levels.
fn sample_brick_full_cm(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32, level: u32) -> u32 {
    let brick_local = (pos - brick_min) / cm_voxel_size(level);
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
    return slot * 512u + voxel_idx;
}

/// Level-parameterized version of sample_brick for clipmap levels.
fn sample_brick_cm(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32, level: u32) -> f32 {
    let idx = sample_brick_full_cm(pos, brick_min, slot, level);
    return extract_distance(brick_pool[idx].word0);
}

/// Sphere trace within a single brick, returning full material info on hit.
fn sphere_trace_brick(origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                      t_enter: f32, cell: vec3<u32>, flat: u32) -> MarchResult {
    let slot = slots[flat];
    if slot == EMPTY_SLOT {
        return MarchResult(-1.0, false, 0u, 0u, 0.0, 0u);
    }

    let be = brick_extent();
    let brick_min = grid_origin() + vec3<f32>(cell) * be;
    let brick_max = brick_min + vec3<f32>(be);

    let aabb_t = ray_aabb_intersect(origin, inv_dir, brick_min, brick_max);
    let t_start = max(t_enter, max(aabb_t.x, 0.0));
    let t_end = aabb_t.y;

    if t_start > t_end {
        return MarchResult(-1.0, false, 0u, 0u, 0.0, 0u);
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
            return MarchResult(t, true, mat_id, sec_and_flags, blend, 0u);
        }

        t += max(abs(d), MIN_STEP);

        if t > t_end {
            break;
        }
    }

    return MarchResult(t_end, false, 0u, 0u, 0.0, 0u);
}

/// Sphere trace within a single brick using clipmap-level parameters.
fn sphere_trace_brick_cm(origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                         t_enter: f32, cell: vec3<u32>, flat: u32,
                         level: u32) -> MarchResult {
    let slot = cm_get_slot(flat, level);
    if slot == EMPTY_SLOT {
        return MarchResult(-1.0, false, 0u, 0u, 0.0, level);
    }

    let be = cm_brick_extent(level);
    let brick_min = cm_grid_origin(level) + vec3<f32>(cell) * be;
    let brick_max = brick_min + vec3<f32>(be);

    let aabb_t = ray_aabb_intersect(origin, inv_dir, brick_min, brick_max);
    let t_start = max(t_enter, max(aabb_t.x, 0.0));
    let t_end = aabb_t.y;

    if t_start > t_end {
        return MarchResult(-1.0, false, 0u, 0u, 0.0, level);
    }

    var t = t_start;
    for (var i = 0u; i < MAX_BRICK_STEPS; i++) {
        let pos = origin + dir * t;
        let clamped = clamp(pos, brick_min + vec3<f32>(MIN_STEP),
                                 brick_max - vec3<f32>(MIN_STEP));

        let sample_idx = sample_brick_full_cm(clamped, brick_min, slot, level);
        let sample = brick_pool[sample_idx];
        let d = extract_distance(sample.word0);

        if d < HIT_EPSILON {
            let mat_id = extract_material_id(sample.word0);
            let blend = extract_blend_weight(sample.word1);
            let sec_id = extract_secondary_id(sample.word1);
            let flags = extract_flags(sample.word1);
            let sec_and_flags = sec_id | (flags << 8u);
            return MarchResult(t, true, mat_id, sec_and_flags, blend, level);
        }

        t += max(abs(d), MIN_STEP);

        if t > t_end {
            break;
        }
    }

    return MarchResult(t_end, false, 0u, 0u, 0.0, level);
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
        return MarchResult(-1.0, false, 0u, 0u, 0.0, 0u);
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

    return MarchResult(-1.0, false, 0u, 0u, 0.0, 0u);
}

// ---------- Clipmap DDA Ray March ----------

/// DDA ray march through a single clipmap level's grid.
///
/// Uses the clipmap accessor functions to read occupancy and slots from the
/// combined buffers at the correct offsets for the given level. The brick pool
/// (group 0) is shared across all levels.
fn ray_march_dda_level(origin: vec3<f32>, dir: vec3<f32>,
                       level: u32, t_start: f32, t_end: f32) -> MarchResult {
    let be = cm_brick_extent(level);
    let dims = cm_grid_dims(level);
    let dims_f = vec3<f32>(dims);
    let g_origin = cm_grid_origin(level);
    let grid_max = g_origin + dims_f * be;

    let safe_dir = select(dir, vec3<f32>(1e-10), abs(dir) < vec3<f32>(1e-10));
    let inv_dir = 1.0 / safe_dir;

    // Intersect ray with this level's grid AABB.
    let aabb_t = ray_aabb_intersect(origin, inv_dir, g_origin, grid_max);
    var t_near = max(max(aabb_t.x, 0.0), t_start);
    let t_far = min(aabb_t.y, t_end);

    if t_near > t_far || t_far < 0.0 {
        return MarchResult(-1.0, false, 0u, 0u, 0.0, level);
    }

    t_near += HIT_EPSILON;

    let entry = origin + safe_dir * t_near;
    let dims_i = vec3<i32>(dims);
    var cell = vec3<i32>(floor((entry - g_origin) / be));
    cell = clamp(cell, vec3<i32>(0), dims_i - vec3<i32>(1));

    let step = vec3<i32>(
        select(-1, 1, safe_dir.x >= 0.0),
        select(-1, 1, safe_dir.y >= 0.0),
        select(-1, 1, safe_dir.z >= 0.0)
    );

    let t_delta = abs(vec3<f32>(be) * inv_dir);

    let cell_min_pos = g_origin + vec3<f32>(cell) * be;
    let cell_max_pos = cell_min_pos + vec3<f32>(be);
    let next_boundary = select(cell_min_pos, cell_max_pos, safe_dir >= vec3<f32>(0.0));
    var t_max_axis = (next_boundary - origin) * inv_dir;

    var t = t_near;

    for (var i = 0u; i < MAX_DDA_STEPS; i++) {
        if !cm_cell_in_bounds(cell, level) || t > t_far {
            break;
        }

        let ucell = vec3<u32>(cell);
        let flat = cm_cell_flat_index(ucell, level);
        let state = cm_get_cell_state(flat, level);

        if state == CELL_SURFACE {
            let result = sphere_trace_brick_cm(origin, safe_dir, inv_dir, t, ucell, flat, level);
            if result.hit {
                return result;
            }
        } else if state == CELL_EMPTY && level + 1u < clipmap.num_levels {
            // Finest level has no data here — check coarser levels for surface data.
            let fallback = probe_coarser_levels(origin, safe_dir, inv_dir, t, level);
            if fallback.hit {
                return fallback;
            }
        }

        if t_max_axis.x < t_max_axis.y && t_max_axis.x < t_max_axis.z {
            t = t_max_axis.x;
            t_max_axis.x += t_delta.x;
            cell.x += step.x;
        } else if t_max_axis.y < t_max_axis.z {
            t = t_max_axis.y;
            t_max_axis.y += t_delta.y;
            cell.y += step.y;
        } else {
            t = t_max_axis.z;
            t_max_axis.z += t_delta.z;
            cell.z += step.z;
        }
    }

    return MarchResult(-1.0, false, 0u, 0u, 0.0, level);
}

/// Probe coarser clipmap levels for surface data at the current ray position.
///
/// When the finest level covering a position has an empty cell, coarser levels
/// may still have surface data (e.g., coarse terrain where fine detail wasn't placed).
/// Checks levels from `current_level + 1` to `num_levels - 1`.
fn probe_coarser_levels(origin: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>,
                        t: f32, current_level: u32) -> MarchResult {
    let world_pos = origin + dir * t;

    for (var lvl = current_level + 1u; lvl < clipmap.num_levels; lvl++) {
        let be = cm_brick_extent(lvl);
        let g_origin = cm_grid_origin(lvl);
        let local = world_pos - g_origin;
        let cell_i = vec3<i32>(floor(local / be));

        if !cm_cell_in_bounds(cell_i, lvl) {
            continue;
        }

        let cell = vec3<u32>(cell_i);
        let flat = cm_cell_flat_index(cell, lvl);
        let state = cm_get_cell_state(flat, lvl);

        if state == CELL_SURFACE {
            let result = sphere_trace_brick_cm(origin, dir, inv_dir, t, cell, flat, lvl);
            if result.hit {
                return result;
            }
        }
    }

    return MarchResult(-1.0, false, 0u, 0u, 0.0, current_level);
}

/// Multi-level clipmap ray march.
///
/// Iterates LOD levels from finest (0) to coarsest. Each level covers
/// a distance band from the previous level's radius to its own radius.
/// Level 0 starts at t=0. Returns on the first hit.
fn ray_march_clipmap(origin: vec3<f32>, dir: vec3<f32>) -> MarchResult {
    var t_start = 0.0;

    for (var lvl = 0u; lvl < clipmap.num_levels; lvl++) {
        let t_end = cm_radius(lvl);

        if t_start < t_end {
            let result = ray_march_dda_level(origin, dir, lvl, t_start, t_end);
            if result.hit {
                return result;
            }
        }

        t_start = t_end;
    }

    return MarchResult(-1.0, false, 0u, 0u, 0.0, 0u);
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

/// Sample the SDF at a world-space position using a specific clipmap level.
fn sample_sdf_cm(pos: vec3<f32>, level: u32) -> f32 {
    let be = cm_brick_extent(level);
    let g_origin = cm_grid_origin(level);
    let local = pos - g_origin;
    let cell_i = vec3<i32>(floor(local / be));

    if !cm_cell_in_bounds(cell_i, level) {
        return be;
    }

    let cell = vec3<u32>(cell_i);
    let flat = cm_cell_flat_index(cell, level);
    let state = cm_get_cell_state(flat, level);

    if state == CELL_EMPTY {
        return be * 0.5;
    }
    if state == CELL_INTERIOR {
        return -be * 0.5;
    }
    if state == CELL_SURFACE {
        let slot = cm_get_slot(flat, level);
        if slot == EMPTY_SLOT {
            return be * 0.5;
        }
        let brick_min = g_origin + vec3<f32>(cell) * be;
        return sample_brick_cm(pos, brick_min, slot, level);
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

/// Compute surface normal via central differences using a specific clipmap level.
fn compute_normal_cm(pos: vec3<f32>, level: u32) -> vec3<f32> {
    let e = cm_voxel_size(level) * 1.5;
    let nx = sample_sdf_cm(pos + vec3<f32>(e, 0.0, 0.0), level)
           - sample_sdf_cm(pos - vec3<f32>(e, 0.0, 0.0), level);
    let ny = sample_sdf_cm(pos + vec3<f32>(0.0, e, 0.0), level)
           - sample_sdf_cm(pos - vec3<f32>(0.0, e, 0.0), level);
    let nz = sample_sdf_cm(pos + vec3<f32>(0.0, 0.0, e), level)
           - sample_sdf_cm(pos - vec3<f32>(0.0, 0.0, e), level);
    return normalize(vec3<f32>(nx, ny, nz));
}

/// Sample SDF at a position using the finest clipmap level with data.
///
/// Iterates levels from finest (0) to coarsest. Returns the SDF distance
/// from the first level that has surface or interior data at this position.
/// Used for accurate normal computation when multiple tiers coexist.
fn sample_sdf_finest(pos: vec3<f32>) -> f32 {
    for (var lvl = 0u; lvl < clipmap.num_levels; lvl++) {
        let be = cm_brick_extent(lvl);
        let g_origin = cm_grid_origin(lvl);
        let local = pos - g_origin;
        let cell_i = vec3<i32>(floor(local / be));

        if !cm_cell_in_bounds(cell_i, lvl) {
            continue;
        }

        let cell = vec3<u32>(cell_i);
        let flat = cm_cell_flat_index(cell, lvl);
        let state = cm_get_cell_state(flat, lvl);

        if state == CELL_SURFACE {
            let slot = cm_get_slot(flat, lvl);
            if slot != EMPTY_SLOT {
                let brick_min = g_origin + vec3<f32>(cell) * be;
                return sample_brick_cm(pos, brick_min, slot, lvl);
            }
        }
        if state == CELL_INTERIOR {
            return -be * 0.5;
        }
    }

    // No data at any level — return large positive distance.
    return cm_brick_extent(clipmap.num_levels - 1u);
}

/// Compute surface normal using the finest available clipmap level at each sample point.
///
/// The central difference epsilon is based on the hit level's voxel size,
/// but each SDF sample uses the finest level with data at that position.
fn compute_normal_finest(pos: vec3<f32>, hit_level: u32) -> vec3<f32> {
    let e = cm_voxel_size(hit_level) * 1.5;
    let nx = sample_sdf_finest(pos + vec3<f32>(e, 0.0, 0.0))
           - sample_sdf_finest(pos - vec3<f32>(e, 0.0, 0.0));
    let ny = sample_sdf_finest(pos + vec3<f32>(0.0, e, 0.0))
           - sample_sdf_finest(pos - vec3<f32>(0.0, e, 0.0));
    let nz = sample_sdf_finest(pos + vec3<f32>(0.0, 0.0, e))
           - sample_sdf_finest(pos - vec3<f32>(0.0, 0.0, e));
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

    // Generate UV in [0, 1], then NDC in [-1, 1].
    // Y is flipped: pixel.y=0 is screen top -> ndc.y=+1 (camera up).
    let uv = (vec2<f32>(pixel.xy) + 0.5 + camera.jitter) / vec2<f32>(dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    // Camera ray
    let ray_origin = camera.position.xyz;
    let ray_dir = normalize(
        camera.forward.xyz + ndc.x * camera.right.xyz + ndc.y * camera.up.xyz
    );

    // Choose march strategy based on clipmap availability.
    var result: MarchResult;
    if clipmap.num_levels > 0u {
        result = ray_march_clipmap(ray_origin, ray_dir);
    } else {
        result = ray_march_dda(ray_origin, ray_dir);
    }

    if result.hit {
        let hit_pos = ray_origin + ray_dir * result.t;

        // Compute normal using the appropriate path.
        // compute_normal_finest checks all levels for the finest data at each
        // sample point, producing accurate normals even when the hit came from
        // a coarser fallback level.
        var normal: vec3<f32>;
        if clipmap.num_levels > 0u {
            normal = compute_normal_finest(hit_pos, result.level);
        } else {
            normal = compute_normal(hit_pos);
        }

        // Write G-buffer
        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, result.t));
        textureStore(gbuf_normal, coord, vec4<f32>(normal, result.blend_weight));
        // Pack material_id (lower 16) + secondary_id_and_flags (upper 16) into R32Uint
        let packed_mat = result.material_id | (result.secondary_id_and_flags << 16u);
        textureStore(gbuf_material, coord, vec4<u32>(packed_mat, 0u, 0u, 0u));
        // Motion vector: reproject hit position through previous VP
        let prev_clip = camera.prev_vp * vec4<f32>(hit_pos, 1.0);
        var motion = vec2<f32>(0.0, 0.0);
        if prev_clip.w > 0.0 {
            let prev_ndc = prev_clip.xy / prev_clip.w;
            // NDC [-1,1] -> UV [0,1], Y flipped (screen top = y=0, NDC top = y=+1)
            let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
            let curr_uv = (vec2<f32>(pixel.xy) + 0.5) / vec2<f32>(dims);
            motion = curr_uv - prev_uv;
        }
        textureStore(gbuf_motion, coord, vec4<f32>(motion, 0.0, 0.0));
    } else {
        // Sky / miss — encode as MAX_FLOAT hit distance
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, MAX_FLOAT));
        textureStore(gbuf_normal, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
