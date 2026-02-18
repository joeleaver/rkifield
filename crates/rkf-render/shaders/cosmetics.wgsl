// Cosmetic post-processing effects (post-upscale).
//
// Three optional effects in a single compute pass:
// - Vignette: radial darkening toward screen edges
// - Film grain: luminance-weighted temporal noise
// - Chromatic aberration: radial RGB channel offset
//
// All effects are gated on intensity > 0.0 to skip cost when disabled.

struct CosmeticsParams {
    width: u32,
    height: u32,
    vignette_intensity: f32,    // 0.0 = off, ~0.5 = subtle, ~1.0 = strong
    grain_intensity: f32,       // 0.0 = off, ~0.05 = subtle, ~0.2 = strong
    chromatic_aberration: f32,  // 0.0 = off, ~0.002 = subtle, ~0.01 = strong
    frame_index: u32,           // for grain temporal variation
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var input_color: texture_2d<f32>;
@group(0) @binding(1) var output_color: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: CosmeticsParams;

// Hash function for film grain pseudo-random noise
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2<i32>(gid.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dims;

    // --- Chromatic aberration ---
    var color: vec3<f32>;
    if (params.chromatic_aberration > 0.0) {
        // Radial offset: stronger at edges
        let center = uv - 0.5;
        let dist = length(center);
        let offset = center * dist * params.chromatic_aberration;

        // Sample R, G, B at slightly different positions
        let r_uv = uv + offset;
        let b_uv = uv - offset;

        let r_coord = clamp(vec2<i32>(r_uv * dims), vec2(0), vec2<i32>(dims) - 1);
        let g_coord = coord;
        let b_coord = clamp(vec2<i32>(b_uv * dims), vec2(0), vec2<i32>(dims) - 1);

        color = vec3(
            textureLoad(input_color, r_coord, 0).r,
            textureLoad(input_color, g_coord, 0).g,
            textureLoad(input_color, b_coord, 0).b,
        );
    } else {
        color = textureLoad(input_color, coord, 0).rgb;
    }

    // --- Vignette ---
    if (params.vignette_intensity > 0.0) {
        let center = uv - 0.5;
        let dist = length(center) * 1.414; // normalize: corner = 1.0
        let vignette = 1.0 - params.vignette_intensity * dist * dist;
        color *= max(vignette, 0.0);
    }

    // --- Film grain ---
    if (params.grain_intensity > 0.0) {
        let noise_seed = vec2<f32>(gid.xy) + vec2<f32>(f32(params.frame_index) * 1.37, f32(params.frame_index) * 0.73);
        let noise = hash(noise_seed) - 0.5;  // [-0.5, 0.5]

        // Luminance-weighted: less grain on bright areas, more on dark
        let luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
        let grain_weight = 1.0 - saturate(luminance * 2.0);
        color += noise * params.grain_intensity * grain_weight;
    }

    textureStore(output_color, coord, vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0));
}
