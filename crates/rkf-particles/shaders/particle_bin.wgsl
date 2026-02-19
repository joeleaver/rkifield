// Particle binning compute shader — 3-pass GPU spatial binning.
//
// Bins volumetric particles (render_type == 0) into a 3D grid so the
// volumetric march shader can efficiently query nearby particles.
//
// Pass 1 (count):       Count particles per cell via atomicAdd.
// Pass 2 (prefix_sum):  Exclusive prefix sum over cell_counts → cell_offsets.
// Pass 3 (scatter):     Write particle indices into sorted array at cell offsets.
//
// Each pass is a separate dispatch with @workgroup_size(256).

// Particle layout — matches Rust repr(C) / particle_simulate.wgsl.
struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    max_lifetime: f32,
    color_emission: vec2<u32>,
    size_render_flags_mat_pad: vec2<u32>,
}

struct GridParams {
    cell_count: u32,
    cell_size: f32,
    origin: vec3<f32>,
    _pad0: f32,
    dims: vec3<u32>,
    _pad1: u32,
}

// Bindings — shared across all three passes.
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> grid: GridParams;
@group(0) @binding(2) var<uniform> particle_count: u32;
@group(0) @binding(3) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> particle_indices: array<u32>;

const RENDER_TYPE_VOLUMETRIC: u32 = 0u;
const FLAG_ALIVE: u32 = 8u;

// Extract render_type byte from packed field.
// size_render_flags_mat_pad[0]: bits [0:15]=size, [16:23]=render_type, [24:31]=flags
fn get_render_type(p: Particle) -> u32 {
    return (p.size_render_flags_mat_pad[0] >> 16u) & 0xFFu;
}

fn get_flags(p: Particle) -> u32 {
    return (p.size_render_flags_mat_pad[0] >> 24u) & 0xFFu;
}

// Convert world position to flat cell index.
fn world_to_cell_flat(pos: vec3<f32>) -> u32 {
    let rel = pos - grid.origin;
    let inv = 1.0 / grid.cell_size;
    let cx = clamp(u32(floor(rel.x * inv)), 0u, grid.dims.x - 1u);
    let cy = clamp(u32(floor(rel.y * inv)), 0u, grid.dims.y - 1u);
    let cz = clamp(u32(floor(rel.z * inv)), 0u, grid.dims.z - 1u);
    return cx + cy * grid.dims.x + cz * grid.dims.x * grid.dims.y;
}

// -------------------------------------------------------------------
// Pass 1: Count particles per cell.
// Dispatch: ceil(particle_count / 256) workgroups.
// -------------------------------------------------------------------
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= particle_count { return; }

    let p = particles[idx];
    let f = get_flags(p);
    if (f & FLAG_ALIVE) == 0u { return; }
    if get_render_type(p) != RENDER_TYPE_VOLUMETRIC { return; }

    let cell = world_to_cell_flat(p.position);
    atomicAdd(&cell_counts[cell], 1u);
}

// -------------------------------------------------------------------
// Pass 2: Exclusive prefix sum over cell_counts → cell_offsets.
// Simple serial scan — dispatched with 1 workgroup of 1 thread.
// For production use a parallel Blelloch scan; this is correct and
// sufficient for moderate cell counts (e.g. 32x16x32 = 16384).
// -------------------------------------------------------------------
@compute @workgroup_size(1)
fn prefix_sum() {
    var running = 0u;
    for (var i = 0u; i < grid.cell_count; i++) {
        let c = atomicLoad(&cell_counts[i]);
        cell_offsets[i] = running;
        running += c;
    }
    // Sentinel at end.
    cell_offsets[grid.cell_count] = running;
}

// -------------------------------------------------------------------
// Pass 3: Scatter particle indices into sorted array.
// Uses atomicAdd on cell_counts (reset to 0 between pass 2 and 3,
// or re-used as write cursors starting from cell_offsets values).
//
// IMPORTANT: Before dispatching pass 3, the host must copy
// cell_offsets[0..cell_count] into cell_counts (or reset cell_counts
// to 0 and use cell_offsets + atomicAdd on a separate cursor buffer).
// Here we use cell_counts as write cursors initialized to cell_offsets.
// -------------------------------------------------------------------
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= particle_count { return; }

    let p = particles[idx];
    let f = get_flags(p);
    if (f & FLAG_ALIVE) == 0u { return; }
    if get_render_type(p) != RENDER_TYPE_VOLUMETRIC { return; }

    let cell = world_to_cell_flat(p.position);
    // cell_counts has been re-initialized to cell_offsets values before this pass.
    let slot = atomicAdd(&cell_counts[cell], 1u);
    particle_indices[slot] = idx;
}
