// Ray march compute shader — Phase 4 (basic sphere tracing, no DDA).
//
// One thread per pixel at internal resolution. Generates a camera ray,
// marches through the sparse grid sampling SDF bricks, and writes
// white on hit / black on miss to the output texture.

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

const MAX_STEPS: u32 = 256u;
const MAX_DISTANCE: f32 = 100.0;
const HIT_EPSILON: f32 = 0.001;
const MIN_STEP: f32 = 0.0005;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// CellState values (2-bit, matching Rust CellState enum)
const CELL_EMPTY: u32     = 0u;
const CELL_SURFACE: u32   = 1u;
const CELL_INTERIOR: u32  = 2u;

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

/// Sample the SDF at a world-space position using the sparse grid + brick pool.
/// Returns a large positive distance if outside the grid or in empty space.
fn sample_sdf(pos: vec3<f32>) -> f32 {
    let cell_i = world_to_cell(pos);

    let be = brick_extent();

    // Outside grid bounds — return large distance
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

        // Convert to voxel coordinates within this brick
        let brick_min = grid_origin() + vec3<f32>(cell) * be;
        let brick_local = (pos - brick_min) / voxel_size();
        let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));

        // Read the voxel sample from the brick pool
        let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
        let sample_idx = slot * 512u + voxel_idx;
        let sample = brick_pool[sample_idx];

        return extract_distance(sample.word0);
    }

    // Volumetric or unknown — skip
    return be * 0.5;
}

/// March a ray through the scene, returning distance to hit or -1.0 on miss.
fn ray_march(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    var t = 0.0;

    for (var i = 0u; i < MAX_STEPS; i++) {
        let pos = origin + direction * t;
        let d = sample_sdf(pos);

        if d < HIT_EPSILON {
            return t;
        }

        // Step by SDF distance, but never less than MIN_STEP
        t += max(abs(d), MIN_STEP);

        if t > MAX_DISTANCE {
            return -1.0;
        }
    }

    return -1.0;
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

    // March
    let t = ray_march(ray_origin, ray_dir);

    var color: vec3<f32>;
    if t >= 0.0 {
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
