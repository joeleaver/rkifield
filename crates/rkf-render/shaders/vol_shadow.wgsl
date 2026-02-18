// Volumetric shadow map compute shader — Phase 11 task 11.1.
//
// Fills a 3D R32Float texture with per-texel transmittance values.
// Marches from each voxel toward the sun; if the march hits SDF geometry,
// transmittance drops. Used by the volumetric march pass for sun visibility.

// ---------- Params ----------

struct VolShadowParams {
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    sun_dir: vec3<f32>,
    _pad2: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    max_steps: u32,
    step_size: f32,
    extinction_coeff: f32,
    _pad3: u32,
    _pad4: u32,
}

struct VoxelSample {
    word0: u32,
    word1: u32,
}

struct SceneUniforms {
    grid_dims:   vec4<u32>,
    grid_origin: vec4<f32>,
    params:      vec4<f32>,
}

// Dummy camera struct — we must declare binding 3 to match the scene bind
// group layout, but the volumetric shadow shader does not use camera data.
struct DummyCamera {
    _unused: vec4<f32>,
}

// ---------- Bindings ----------

// Group 0: vol shadow data
@group(0) @binding(0) var<uniform> params: VolShadowParams;
@group(0) @binding(1) var shadow_map: texture_storage_3d<r32float, write>;

// Group 1: scene data (same layout as ray_march group 0)
@group(1) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(1) @binding(1) var<storage, read> occupancy:  array<u32>;
@group(1) @binding(2) var<storage, read> slots:      array<u32>;
@group(1) @binding(3) var<uniform>       _camera:    DummyCamera;
@group(1) @binding(4) var<uniform>       scene:      SceneUniforms;

// ---------- Constants ----------

const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const CELL_EMPTY: u32 = 0u;
const CELL_INTERIOR: u32 = 2u;

// ---------- SDF helpers ----------

fn grid_dims() -> vec3<u32>  { return scene.grid_dims.xyz; }
fn grid_origin() -> vec3<f32> { return scene.grid_origin.xyz; }
fn brick_extent() -> f32     { return scene.grid_origin.w; }
fn voxel_size() -> f32       { return scene.params.x; }

fn world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - grid_origin();
    return vec3<i32>(floor(local / brick_extent()));
}

fn cell_in_bounds(cell: vec3<i32>) -> bool {
    return all(cell >= vec3<i32>(0)) && all(vec3<u32>(cell) < grid_dims());
}

fn cell_flat_index(cell: vec3<u32>) -> u32 {
    let d = grid_dims();
    return cell.x + cell.y * d.x + cell.z * d.x * d.y;
}

fn get_cell_state(flat: u32) -> u32 {
    let word_idx = flat / 16u;
    let bit_offset = (flat % 16u) * 2u;
    return (occupancy[word_idx] >> bit_offset) & 3u;
}

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

/// Trilinear SDF distance sample within a brick.
fn sample_brick_trilinear(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> f32 {
    let vs = voxel_size();
    let brick_local = (pos - brick_min) / vs - vec3<f32>(0.5);
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

/// Check if world position `pos` is inside SDF geometry.
/// Returns a high density (1.0) if inside, 0.0 if outside or in empty space.
fn sample_sdf_density(pos: vec3<f32>) -> f32 {
    let cell = world_to_cell(pos);
    if !cell_in_bounds(cell) {
        return 0.0;
    }

    let ucell = vec3<u32>(cell);
    let flat = cell_flat_index(ucell);
    let state = get_cell_state(flat);

    // Empty cells have no geometry
    if state == CELL_EMPTY {
        return 0.0;
    }

    // Interior cells are fully solid
    if state == CELL_INTERIOR {
        return 1.0;
    }

    // Surface cells: sample the brick pool for the actual SDF distance (trilinear)
    let slot = slots[flat];
    if slot == EMPTY_SLOT {
        return 0.0;
    }

    let cell_origin = grid_origin() + vec3<f32>(ucell) * brick_extent();
    let dist = sample_brick_trilinear(pos, cell_origin, slot);

    // If SDF distance is negative, we are inside geometry — opaque
    if dist < 0.0 {
        return 1.0;
    }

    // Close to surface, partial density (soft shadow edges)
    let vs = voxel_size();
    if dist < vs * 2.0 {
        return 1.0 - dist / (vs * 2.0);
    }

    return 0.0;
}

// ---------- Entry point ----------

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dim_x || id.y >= params.dim_y || id.z >= params.dim_z {
        return;
    }

    let dim = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z));
    let uv = (vec3<f32>(id) + vec3<f32>(0.5)) / dim;
    let world_pos = params.volume_min + uv * (params.volume_max - params.volume_min);

    var transmittance: f32 = 1.0;
    var pos: vec3<f32> = world_pos;

    for (var step = 0u; step < params.max_steps; step++) {
        pos += params.sun_dir * params.step_size;

        let density = sample_sdf_density(pos);
        if density > 0.0 {
            transmittance *= exp(-density * params.extinction_coeff * params.step_size);
        }

        if transmittance < 0.01 {
            transmittance = 0.0;
            break;
        }
    }

    let texel = vec3<i32>(id);
    textureStore(shadow_map, texel, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
