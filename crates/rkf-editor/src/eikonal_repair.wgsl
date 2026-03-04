// Eikonal PDE re-initialization for SDF repair.
//
// Sussman-Smereka-Osher (1994): ∂d/∂τ = S(d₀)(1 − |∇d|)
// Iteratively drives |∇d| → 1 everywhere while preserving zero-crossings.
//
// Ping-pong between dist_a and dist_b via parity bit.
// d0 is a read-only copy of the initial distances (for sign stability).
//
// SENTINEL HANDLING: Voxels with |d₀| >= narrow_band are "frozen" (EMPTY_SLOT
// at +8h, INTERIOR_SLOT at -8h). When computing finite differences, frozen
// neighbors are treated as Neumann boundaries (replicate center value) to
// prevent false zero-crossings from injecting massive gradients into the PDE.

struct Uniforms {
    gw:          u32,
    gh:          u32,
    gd:          u32,
    parity:      u32,   // 0 = read A / write B; 1 = read B / write A
    voxel_size:  f32,
    dt:          f32,   // Δτ = 0.45 * voxel_size
    narrow_band: f32,   // 6.0 * voxel_size
    _pad:        u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read_write> dist_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dist_b: array<f32>;
@group(0) @binding(3) var<storage, read>       d0:     array<f32>;

fn idx(x: u32, y: u32, z: u32) -> u32 {
    return z * u.gw * u.gh + y * u.gw + x;
}

// Smoothed sign function: S(d₀) = d₀ / sqrt(d₀² + h²)
fn smoothed_sign(d: f32, h: f32) -> f32 {
    return d / sqrt(d * d + h * h);
}

// Load from the "read" buffer based on parity.
fn load_dist(i: u32) -> f32 {
    if u.parity == 0u {
        return dist_a[i];
    } else {
        return dist_b[i];
    }
}

// Store to the "write" buffer based on parity.
fn store_dist(i: u32, val: f32) {
    if u.parity == 0u {
        dist_b[i] = val;
    } else {
        dist_a[i] = val;
    }
}

// Load a neighbor's distance for gradient computation.
// If the neighbor is frozen (sentinel: |d₀| >= narrow_band), return the
// CENTER value instead — Neumann BC.  This prevents ±8h sentinels from
// creating false zero-crossings that blow up the Godunov upwind gradient.
fn load_neighbor(ci: u32, ni: u32) -> f32 {
    let nd0 = d0[ni];
    if abs(nd0) >= u.narrow_band {
        return load_dist(ci);
    }
    return load_dist(ni);
}

@compute @workgroup_size(8, 8, 4)
fn eikonal_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;

    if x >= u.gw || y >= u.gh || z >= u.gd {
        return;
    }

    let ci = idx(x, y, z);
    let d_initial = d0[ci];
    let d_curr = load_dist(ci);

    // Outside narrow band: frozen — copy through unchanged.
    if abs(d_initial) >= u.narrow_band {
        store_dist(ci, d_curr);
        return;
    }

    let h = u.voxel_size;
    let S = smoothed_sign(d_initial, h);

    // Load neighbors with Neumann BC at grid edges AND sentinel clamping.
    // Grid edge: replicate center. Sentinel neighbor: replicate center.
    var d_xm: f32; var d_xp: f32;
    var d_ym: f32; var d_yp: f32;
    var d_zm: f32; var d_zp: f32;

    if x == 0u     { d_xm = d_curr; } else { d_xm = load_neighbor(ci, idx(x - 1u, y, z)); }
    if x >= u.gw-1u { d_xp = d_curr; } else { d_xp = load_neighbor(ci, idx(x + 1u, y, z)); }
    if y == 0u     { d_ym = d_curr; } else { d_ym = load_neighbor(ci, idx(x, y - 1u, z)); }
    if y >= u.gh-1u { d_yp = d_curr; } else { d_yp = load_neighbor(ci, idx(x, y + 1u, z)); }
    if z == 0u     { d_zm = d_curr; } else { d_zm = load_neighbor(ci, idx(x, y, z - 1u)); }
    if z >= u.gd-1u { d_zp = d_curr; } else { d_zp = load_neighbor(ci, idx(x, y, z + 1u)); }

    // Backward and forward finite differences.
    let inv_h = 1.0 / h;
    let Dmx = (d_curr - d_xm) * inv_h;  // D⁻x
    let Dpx = (d_xp - d_curr) * inv_h;  // D⁺x
    let Dmy = (d_curr - d_ym) * inv_h;  // D⁻y
    let Dpy = (d_yp - d_curr) * inv_h;  // D⁺y
    let Dmz = (d_curr - d_zm) * inv_h;  // D⁻z
    let Dpz = (d_zp - d_curr) * inv_h;  // D⁺z

    // Godunov upwind gradient magnitude squared.
    var grad_sq: f32;
    if S > 0.0 {
        // Exterior: information flows outward from zero-crossing.
        grad_sq = max(max(Dmx, 0.0) * max(Dmx, 0.0), min(Dpx, 0.0) * min(Dpx, 0.0))
                + max(max(Dmy, 0.0) * max(Dmy, 0.0), min(Dpy, 0.0) * min(Dpy, 0.0))
                + max(max(Dmz, 0.0) * max(Dmz, 0.0), min(Dpz, 0.0) * min(Dpz, 0.0));
    } else {
        // Interior: information flows inward from zero-crossing.
        grad_sq = max(min(Dmx, 0.0) * min(Dmx, 0.0), max(Dpx, 0.0) * max(Dpx, 0.0))
                + max(min(Dmy, 0.0) * min(Dmy, 0.0), max(Dpy, 0.0) * max(Dpy, 0.0))
                + max(min(Dmz, 0.0) * min(Dmz, 0.0), max(Dpz, 0.0) * max(Dpz, 0.0));
    }

    let grad_mag = sqrt(grad_sq);

    // PDE update: d_new = d + Δτ * S(d₀) * (1 − |∇d|)
    let d_new = d_curr + u.dt * S * (1.0 - grad_mag);

    store_dist(ci, d_new);
}
