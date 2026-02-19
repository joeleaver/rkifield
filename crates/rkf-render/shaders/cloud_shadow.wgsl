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

// Cloud noise parameters — must match the CloudParams struct in vol_march.wgsl.
struct CloudNoiseParams {
    // x=cloud_min, y=cloud_max, z=threshold, w=density_scale
    altitude: vec4<f32>,
    // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    noise: vec4<f32>,
    // x=wind_dir.x, y=wind_dir.y, z=wind_speed, w=time
    wind: vec4<f32>,
    // x=procedural_enable (0/1), y=shadow_coverage, z=shadow_res, w=brick_clouds_enable (0/1)
    flags: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: CloudShadowParams;
@group(0) @binding(1) var output_shadow: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> cloud_noise: CloudNoiseParams;

// Simple 3D hash for noise — identical to vol_march.wgsl.
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise 3D — identical to vol_march.wgsl.
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

// Cloud density for shadow map — uses same noise as vol_march cloud_density().
// Omits detail noise (subtractive erosion) since the shadow map doesn't need
// the fine-grained edges — shape + weather is sufficient for shadow projection.
fn cloud_density_shadow(pos: vec3<f32>) -> f32 {
    let cloud_min = cloud_noise.altitude.x;
    let cloud_max = cloud_noise.altitude.y;

    if (pos.y < cloud_min || pos.y > cloud_max) { return 0.0; }
    if (cloud_noise.flags.x < 0.5)              { return 0.0; }

    let height_frac     = (pos.y - cloud_min) / (cloud_max - cloud_min);
    let height_gradient = smoothstep(0.0, 0.1, height_frac)
                        * smoothstep(1.0, 0.6, height_frac);

    // Wind scrolling — must match vol_march exactly.
    let wind_dir    = vec2<f32>(cloud_noise.wind.x, cloud_noise.wind.y);
    let wind_offset = wind_dir * cloud_noise.wind.z * cloud_noise.wind.w;
    let scrolled    = pos + vec3<f32>(wind_offset.x, 0.0, wind_offset.y);

    // Noise offset — avoid hash3 zero-point at origin. Must match vol_march.
    let noise_offset = vec3<f32>(173.5, 247.3, 391.7);

    // Shape noise — 4-octave FBM (matches vol_march).
    let shape_freq = cloud_noise.noise.x;
    let shape      = fbm_3d(scrolled * shape_freq + noise_offset, 4u);

    // Weather modulation — 2-octave FBM (matches vol_march).
    let weather_scale = cloud_noise.noise.w;
    let weather       = fbm_3d(
        vec3<f32>(scrolled.x / weather_scale, 0.0, scrolled.z / weather_scale) + noise_offset,
        2u
    );

    let threshold     = cloud_noise.altitude.z;
    let density_scale = cloud_noise.altitude.w;

    return max(0.0, shape * weather * height_gradient - threshold) * density_scale;
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

        let density = cloud_density_shadow(pos);
        if (density > 0.001) {
            transmittance *= exp(-density * params.extinction * step_size);
        }
        if (transmittance < 0.01) { break; }
    }

    textureStore(output_shadow, coord, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
