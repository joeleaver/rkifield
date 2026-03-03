// 3D Jump Flooding Algorithm for SDF repair after sculpt strokes.
//
// Computes correct signed Euclidean distances from the binary solid/empty
// surface encoded in `solid[]`. Three entry points are dispatched in sequence:
//
//   1. jfa_init     — seed the surface voxels (sign-change neighbors)
//   2. jfa_pass     — propagate nearest-seed info (dispatch ceil(log2(N)) times)
//   3. jfa_writeback — convert seed coordinates to signed distances, write to dist_out
//
// Step encoding in uniforms.step:
//   bit 31 = parity (0 → read seeds_a / write seeds_b; 1 → read seeds_b / write seeds_a)
//   bits 0-30 = actual JFA step size (power of 2, e.g. 64, 32, 16 ... 1)
//
// For jfa_writeback the step field encodes only the parity bit (bits 0-30 unused).

struct JfaUniforms {
    gw:         u32,   // grid width  (voxels)
    gh:         u32,   // grid height (voxels)
    gd:         u32,   // grid depth  (voxels)
    step:       u32,   // JFA step size + parity flag (see above)
    voxel_size: f32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<uniform>            u:        JfaUniforms;
@group(0) @binding(1) var<storage, read>       solid:    array<u32>; // 1 = solid, 0 = empty
@group(0) @binding(2) var<storage, read_write> seeds_a:  array<u32>; // flat-index of nearest seed; 0xFFFFFFFF = none
@group(0) @binding(3) var<storage, read_write> seeds_b:  array<u32>;
@group(0) @binding(4) var<storage, read_write> dist_out: array<f32>; // signed distance output

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn flat_idx(x: u32, y: u32, z: u32) -> u32 {
    return x + y * u.gw + z * u.gw * u.gh;
}

fn seed_coords(idx: u32) -> vec3<u32> {
    let z = idx / (u.gw * u.gh);
    let r = idx % (u.gw * u.gh);
    return vec3<u32>(r % u.gw, r / u.gw, z);
}

fn read_seed(idx: u32) -> u32 {
    if (u.step & 0x80000000u) == 0u { return seeds_a[idx]; }
    else                             { return seeds_b[idx]; }
}

fn write_seed(idx: u32, val: u32) {
    if (u.step & 0x80000000u) == 0u { seeds_b[idx] = val; }
    else                             { seeds_a[idx] = val; }
}

// ---------------------------------------------------------------------------
// Entry 1: jfa_init
//
// A voxel is a surface seed if any of its 6 face-adjacent neighbors differs in
// solid/empty classification.  Seeds initialise seeds_a with their own flat
// index; all other voxels get 0xFFFFFFFF (invalid).
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 4)
fn jfa_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.gw || gid.y >= u.gh || gid.z >= u.gd { return; }

    let idx  = flat_idx(gid.x, gid.y, gid.z);
    let self_solid = solid[idx];

    var is_surf = false;
    if !is_surf && gid.x > 0u          { is_surf = solid[flat_idx(gid.x - 1u, gid.y,      gid.z     )] != self_solid; }
    if !is_surf && gid.x < u.gw - 1u   { is_surf = solid[flat_idx(gid.x + 1u, gid.y,      gid.z     )] != self_solid; }
    if !is_surf && gid.y > 0u          { is_surf = solid[flat_idx(gid.x,      gid.y - 1u, gid.z     )] != self_solid; }
    if !is_surf && gid.y < u.gh - 1u   { is_surf = solid[flat_idx(gid.x,      gid.y + 1u, gid.z     )] != self_solid; }
    if !is_surf && gid.z > 0u          { is_surf = solid[flat_idx(gid.x,      gid.y,      gid.z - 1u)] != self_solid; }
    if !is_surf && gid.z < u.gd - 1u   { is_surf = solid[flat_idx(gid.x,      gid.y,      gid.z + 1u)] != self_solid; }

    seeds_a[idx] = select(0xFFFFFFFFu, idx, is_surf);
    seeds_b[idx] = 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Entry 2: jfa_pass
//
// Standard 3D JFA: each voxel probes its 26 neighbours at ±step_actual.
// Reads from seeds_a or seeds_b (selected by parity bit), writes to the other.
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 4)
fn jfa_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.gw || gid.y >= u.gh || gid.z >= u.gd { return; }

    let idx  = flat_idx(gid.x, gid.y, gid.z);
    let step = i32(u.step & 0x7FFFFFFFu);

    var best_seed = read_seed(idx);
    var best_dsq: f32;

    if best_seed != 0xFFFFFFFFu {
        let sc  = seed_coords(best_seed);
        let d   = vec3<f32>(gid) - vec3<f32>(sc);
        best_dsq = dot(d, d);
    } else {
        best_dsq = 1e30;
    }

    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                if dx == 0 && dy == 0 && dz == 0 { continue; }

                let nx = i32(gid.x) + dx * step;
                let ny = i32(gid.y) + dy * step;
                let nz = i32(gid.z) + dz * step;

                if nx < 0 || ny < 0 || nz < 0 { continue; }
                let ux = u32(nx); let uy = u32(ny); let uz = u32(nz);
                if ux >= u.gw || uy >= u.gh || uz >= u.gd { continue; }

                let nseed = read_seed(flat_idx(ux, uy, uz));
                if nseed == 0xFFFFFFFFu { continue; }

                let sc  = seed_coords(nseed);
                let d   = vec3<f32>(gid) - vec3<f32>(sc);
                let dsq = dot(d, d);
                if dsq < best_dsq {
                    best_dsq = dsq;
                    best_seed = nseed;
                }
            }
        }
    }

    write_seed(idx, best_seed);
}

// ---------------------------------------------------------------------------
// Entry 3: jfa_writeback
//
// Reads the final seed buffer (parity-selected), converts to a signed float
// distance and writes to dist_out[].
//
//   dist = sign × (euclidean_distance_in_voxels − 0.5) × voxel_size
//
// The 0.5-voxel offset places the zero-crossing at the voxel face (midpoint
// between adjacent solid/empty voxel centres), matching the EDT convention.
// Sign: exterior voxels (solid=0) → positive; interior (solid=1) → negative.
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 4)
fn jfa_writeback(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.gw || gid.y >= u.gh || gid.z >= u.gd { return; }

    let idx     = flat_idx(gid.x, gid.y, gid.z);
    let is_sol  = solid[idx] != 0u;
    let seed    = read_seed(idx);

    var dist: f32;
    if seed == 0xFFFFFFFFu {
        // No surface seed was found: clamp to a large value with correct sign.
        dist = select(4.0 * u.voxel_size, -4.0 * u.voxel_size, is_sol);
    } else {
        let sc       = seed_coords(seed);
        let d        = vec3<f32>(gid) - vec3<f32>(sc);
        let mag_vox  = length(d) - 0.5;                         // subtract half-voxel
        let mag      = max(mag_vox, 0.01) * u.voxel_size;
        dist = select(mag, -mag, is_sol);
    }

    dist_out[idx] = dist;
}
