// Volumetric particle helper functions for the vol_march shader.
//
// These are meant to be included (copy-pasted or @import'd) into the
// volumetric march pass, not dispatched standalone. They sample the
// density field contributed by volumetric particles at any world
// position using the binned particle grid.
//
// Bindings expected from the including shader:
//   @group(N) @binding(0) var<storage, read> vp_particles: array<VpParticle>;
//   @group(N) @binding(1) var<storage, read> vp_cell_offsets: array<u32>;
//   @group(N) @binding(2) var<storage, read> vp_particle_indices: array<u32>;
//   @group(N) @binding(3) var<uniform> vp_grid: VpGridParams;

struct VpParticle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    max_lifetime: f32,
    color_emission: vec2<u32>,          // packed f16x4
    size_render_flags_mat_pad: vec2<u32>,
}

struct VpGridParams {
    cell_count: u32,
    cell_size: f32,
    origin: vec3<f32>,
    _pad0: f32,
    dims: vec3<u32>,
    _pad1: u32,
}

const VP_FLAG_FADE_OUT: u32 = 4u;
const VP_FLAG_ALIVE: u32 = 8u;

// ---- Field extraction helpers ----

fn vp_get_size(p: VpParticle) -> f32 {
    let bits = p.size_render_flags_mat_pad[0] & 0xFFFFu;
    return unpack2x16float(bits).x;
}

fn vp_get_flags(p: VpParticle) -> u32 {
    return (p.size_render_flags_mat_pad[0] >> 24u) & 0xFFu;
}

fn vp_get_color(p: VpParticle) -> vec3<f32> {
    let rg = unpack2x16float(p.color_emission[0]);
    let ba = unpack2x16float(p.color_emission[1]);
    return vec3<f32>(rg.x, rg.y, ba.x);
}

fn vp_get_emission(p: VpParticle) -> f32 {
    let ba = unpack2x16float(p.color_emission[1]);
    return ba.y;
}

// ---- Core functions ----

/// WGSL smoothstep (built-in exists, but explicit for clarity).
fn vp_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Compute density and emission from a single particle at `pos`.
/// Returns vec2(density, emission).
fn particle_density_at(pos: vec3<f32>, p: VpParticle) -> vec2<f32> {
    let size = vp_get_size(p);
    let d = distance(pos, p.position);

    if d > size {
        return vec2<f32>(0.0, 0.0);
    }

    let falloff = 1.0 - vp_smoothstep(0.0, size, d);

    var age_fade = 1.0;
    let f = vp_get_flags(p);
    if (f & VP_FLAG_FADE_OUT) != 0u {
        if p.max_lifetime > 0.0 {
            age_fade = clamp(p.lifetime / p.max_lifetime, 0.0, 1.0);
        } else {
            age_fade = 0.0;
        }
    }

    let density = falloff * age_fade;
    let emission = falloff * vp_get_emission(p);
    return vec2<f32>(density, emission);
}

/// Convert world position to integer cell coordinates.
fn world_to_cell(pos: vec3<f32>, origin: vec3<f32>, cell_size: f32, dims: vec3<u32>) -> vec3<i32> {
    let rel = pos - origin;
    let inv = 1.0 / cell_size;
    let cx = clamp(i32(floor(rel.x * inv)), 0, i32(dims.x) - 1);
    let cy = clamp(i32(floor(rel.y * inv)), 0, i32(dims.y) - 1);
    let cz = clamp(i32(floor(rel.z * inv)), 0, i32(dims.z) - 1);
    return vec3<i32>(cx, cy, cz);
}

/// Flat cell index from 3D coordinates.
fn cell_flat_index(c: vec3<i32>, dims: vec3<u32>) -> u32 {
    return u32(c.x) + u32(c.y) * dims.x + u32(c.z) * dims.x * dims.y;
}

/// Accumulate particle contributions at `pos` from the binned grid.
/// Returns vec4(scatter_rgb, density). Emission is packed into alpha
/// of a second return — callers use particle_density_at for per-particle
/// emission if needed.
///
/// For simplicity this returns vec4(weighted_color_r, weighted_color_g, weighted_color_b, total_density).
/// The caller divides color by density to get the average scatter color.
fn accumulate_particle_contributions(
    pos: vec3<f32>,
) -> vec4<f32> {
    let cell = world_to_cell(pos, vp_grid.origin, vp_grid.cell_size, vp_grid.dims);

    var total_density = 0.0;
    var weighted_r = 0.0;
    var weighted_g = 0.0;
    var weighted_b = 0.0;

    // 3x3x3 neighborhood search.
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nc = cell + vec3<i32>(dx, dy, dz);

                // Bounds check.
                if nc.x < 0 || nc.y < 0 || nc.z < 0 { continue; }
                if nc.x >= i32(vp_grid.dims.x) || nc.y >= i32(vp_grid.dims.y) || nc.z >= i32(vp_grid.dims.z) { continue; }

                let flat = cell_flat_index(nc, vp_grid.dims);
                let start = vp_cell_offsets[flat];
                let end = vp_cell_offsets[flat + 1u];

                for (var i = start; i < end; i++) {
                    let pidx = vp_particle_indices[i];
                    let p = vp_particles[pidx];
                    let de = particle_density_at(pos, p);
                    let density = de.x;

                    if density > 0.0 {
                        total_density += density;
                        let col = vp_get_color(p);
                        weighted_r += col.x * density;
                        weighted_g += col.y * density;
                        weighted_b += col.z * density;
                    }
                }
            }
        }
    }

    return vec4<f32>(weighted_r, weighted_g, weighted_b, total_density);
}
