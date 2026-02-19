// Particle simulation compute shader.
// Each thread processes one particle: integrate, apply forces, age, kill.
//
// Particle layout (48 bytes, matches Rust repr(C)):
//   position: vec3<f32> + lifetime: f32        = 16 bytes (offset 0)
//   velocity: vec3<f32> + max_lifetime: f32    = 16 bytes (offset 16)
//   color_emission: vec2<u32>                  = 8 bytes  (offset 32, packed f16x4)
//   size_render_flags_mat_pad: vec2<u32>       = 8 bytes  (offset 40, packed fields)
// Total: 48 bytes
//
// size_render_flags_mat_pad[0]: size(u16 low) | render_type(u8) | flags(u8 high)
// size_render_flags_mat_pad[1]: material_id(u16 low) | _pad(u16 high)

struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    max_lifetime: f32,
    color_emission: vec2<u32>,          // packed f16x4 as 2x u32
    size_render_flags_mat_pad: vec2<u32>,  // packed fields
}

struct SimParams {
    dt: f32,
    gravity_scale: f32,
    wind_x: f32,
    wind_y: f32,
    wind_z: f32,
    particle_count: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;
@group(0) @binding(2) var<storage, read_write> alive_counter: atomic<u32>;

const GRAVITY: vec3<f32> = vec3<f32>(0.0, -9.81, 0.0);
const FLAG_GRAVITY: u32 = 1u;
const FLAG_COLLISION: u32 = 2u;
const FLAG_FADE_OUT: u32 = 4u;
const FLAG_ALIVE: u32 = 8u;

// Extract flags byte from packed field.
// size_render_flags_mat_pad[0] layout: size(u16) | render_type(u8) | flags(u8)
// In little-endian u32: bits [0:15] = size, [16:23] = render_type, [24:31] = flags
fn get_flags(p: Particle) -> u32 {
    return (p.size_render_flags_mat_pad[0] >> 24u) & 0xFFu;
}

// Set flags byte in the packed field, preserving other bits.
fn set_flags(packed: u32, new_flags: u32) -> u32 {
    return (packed & 0x00FFFFFFu) | ((new_flags & 0xFFu) << 24u);
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.particle_count { return; }

    var p = particles[idx];
    let f = get_flags(p);
    if (f & FLAG_ALIVE) == 0u { return; }

    let dt = params.dt;
    let wind = vec3<f32>(params.wind_x, params.wind_y, params.wind_z);

    // Gravity
    if (f & FLAG_GRAVITY) != 0u {
        p.velocity += GRAVITY * params.gravity_scale * dt;
    }

    // Wind drag: simple linear drag towards wind velocity
    let wind_force = (wind - p.velocity) * 0.1;
    p.velocity += wind_force * dt;

    // Integrate position
    p.position += p.velocity * dt;

    // Age
    p.lifetime -= dt;
    if p.lifetime <= 0.0 {
        // Kill particle by clearing alive flag
        let new_flags = f & ~FLAG_ALIVE;
        p.size_render_flags_mat_pad[0] = set_flags(p.size_render_flags_mat_pad[0], new_flags);
    } else {
        // Count alive particles
        atomicAdd(&alive_counter, 1u);
    }

    particles[idx] = p;
}
