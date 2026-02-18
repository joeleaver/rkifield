// Procedural cloud density functions.
//
// High-altitude clouds evaluated analytically using FBM noise,
// weather map modulation, and height gradient shaping.
// Wind scrolling provides animation over time.
//
// These functions are designed to be called from the volumetric march shader.
// When vol_march.wgsl integrates cloud support (Phase 12), the CloudParams
// uniform and these functions will be inlined there.

struct CloudParams {
    // x=cloud_min, y=cloud_max, z=threshold, w=density_scale
    altitude: vec4<f32>,
    // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    noise: vec4<f32>,
    // x=wind_dir.x, y=wind_dir.y, z=wind_speed, w=time
    wind: vec4<f32>,
    // x=procedural_enable (0/1), y=shadow_coverage, z=shadow_res, w=brick_clouds_enable (0/1)
    flags: vec4<f32>,
}

// ── Noise primitives ─────────────────────────────────────────────────────────

// Simple 3D hash for noise generation.
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise 3D — tri-linearly interpolated hash lattice.
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep interpolation
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 0.0)),
                hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)),
                hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)),
                hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)),
                hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// Fractal Brownian Motion — sums `octaves` octaves of value noise.
// Amplitude halves and frequency doubles each octave.
fn fbm_3d(p: vec3<f32>, octaves: u32) -> f32 {
    var val: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        val += amp * value_noise_3d(pos * freq);
        freq *= 2.0;
        amp  *= 0.5;
    }
    return val;
}

// ── Cloud density ────────────────────────────────────────────────────────────

// Evaluate procedural cloud density at a world-space position.
//
// Returns a non-negative density value. Zero means no cloud.
// The caller is responsible for integrating this along the ray.
//
// Algorithm:
//   1. Height gradient shapes coverage — fade in at bottom, out at top.
//   2. Shape FBM (4 octaves) × weather noise × height gradient → base density.
//   3. Threshold clips sparse regions for crisp cloud edges.
//   4. Detail FBM (3 octaves) subtractively erodes edges.
//   5. Wind offset scrolls all noise with time.
fn cloud_density(pos: vec3<f32>, cloud_params: CloudParams) -> f32 {
    let cloud_min = cloud_params.altitude.x;
    let cloud_max = cloud_params.altitude.y;

    // Early out: outside altitude band or feature disabled.
    if pos.y < cloud_min || pos.y > cloud_max { return 0.0; }
    if cloud_params.flags.x < 0.5            { return 0.0; }

    // Height gradient: ramp up over bottom 10 %, ramp down over top 40 %.
    let height_frac     = (pos.y - cloud_min) / (cloud_max - cloud_min);
    let height_gradient = smoothstep(0.0, 0.1, height_frac)
                        * smoothstep(1.0, 0.6, height_frac);

    // Wind scrolling (XZ plane).
    let wind_dir    = vec2<f32>(cloud_params.wind.x, cloud_params.wind.y);
    let wind_offset = wind_dir * cloud_params.wind.z * cloud_params.wind.w;
    let scrolled    = pos + vec3<f32>(wind_offset.x, 0.0, wind_offset.y);

    // Shape noise — 4-octave FBM at low frequency.
    let shape_freq = cloud_params.noise.x;
    let shape      = fbm_3d(scrolled * shape_freq, 4u);

    // Weather modulation — 2-octave FBM tiled at weather_scale.
    let weather_scale = cloud_params.noise.w;
    let weather       = fbm_3d(
        vec3<f32>(scrolled.x / weather_scale, 0.0, scrolled.z / weather_scale),
        2u
    );

    // Base density after threshold.
    let threshold = cloud_params.altitude.z;
    let base      = saturate(shape * weather * height_gradient - threshold);

    // Subtractive detail — 3-octave FBM erodes cloud edges.
    let detail_freq   = cloud_params.noise.y;
    let detail_weight = cloud_params.noise.z;
    let detail        = fbm_3d(scrolled * detail_freq, 3u);

    let density_scale = cloud_params.altitude.w;
    return max(0.0, base - detail * detail_weight) * density_scale;
}
