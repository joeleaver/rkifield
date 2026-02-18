// Cloud shadow map compute shader.
//
// Projects cloud density downward from the sun onto a 2D texture.
// Each texel stores transmittance (1.0 = fully lit, 0.0 = fully shadowed).
// Updated once per frame.

struct CloudShadowParams {
    // World-space coverage (center + extent)
    center: vec4<f32>,          // xyz = center position (camera XZ), w = unused
    sun_dir: vec4<f32>,         // xyz = direction toward sun (normalized), w = unused
    // Cloud altitude band
    cloud_min: f32,
    cloud_max: f32,
    // Map settings
    resolution: u32,
    coverage: f32,              // world-space extent of the shadow map
    // March settings
    march_steps: u32,
    extinction: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: CloudShadowParams;
@group(0) @binding(1) var output_shadow: texture_storage_2d<r32float, write>;

// Simple 3D hash for noise
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise 3D
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            mix(hash3(i + vec3(0.0, 0.0, 0.0)), hash3(i + vec3(1.0, 0.0, 0.0)), u.x),
            mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), u.x),
            mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

fn fbm_3d(p: vec3<f32>, octaves: u32) -> f32 {
    var val: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    for (var i = 0u; i < octaves; i++) {
        val += amp * value_noise_3d(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return val;
}

// Simplified cloud density for shadow map (no detail noise — just shape)
fn cloud_density_simple(pos: vec3<f32>) -> f32 {
    if (pos.y < params.cloud_min || pos.y > params.cloud_max) { return 0.0; }
    let height_frac = (pos.y - params.cloud_min) / (params.cloud_max - params.cloud_min);
    let height_gradient = smoothstep(0.0, 0.1, height_frac)
                        * smoothstep(1.0, 0.6, height_frac);
    let shape = fbm_3d(pos * 0.0003, 3u);
    return max(0.0, shape * height_gradient - 0.4);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.resolution || gid.y >= params.resolution) { return; }

    let coord = vec2<i32>(gid.xy);
    let res_f = f32(params.resolution);

    // Map texel to world XZ position
    let uv = (vec2<f32>(gid.xy) + 0.5) / res_f;
    let world_xz = params.center.xz + (uv - 0.5) * params.coverage;

    // March upward through cloud layer along sun direction
    // Start at cloud_min, march toward sun through the cloud band
    var transmittance: f32 = 1.0;
    let step_size = (params.cloud_max - params.cloud_min) / f32(params.march_steps);

    for (var i = 0u; i < params.march_steps; i++) {
        let t = f32(i) * step_size;
        let pos = vec3(world_xz.x, params.cloud_min + t, world_xz.y);

        let density = cloud_density_simple(pos);
        if (density > 0.001) {
            transmittance *= exp(-density * params.extinction * step_size);
        }
        if (transmittance < 0.01) { break; }
    }

    textureStore(output_shadow, coord, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
