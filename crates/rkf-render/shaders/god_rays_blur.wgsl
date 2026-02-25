// Screen-space radial blur god rays (pre-upscale compute pass).
//
// Radially blurs bright sky/sun pixels toward the projected sun position,
// creating visible light shaft streaks.  Based on GPU Gems 3 / Crytek
// screen-space light shaft technique adapted for SDF volumetric output.
//
// Only SKY samples contribute — the G-buffer position.w is checked to
// distinguish sky (MAX_FLOAT) from geometry.  This prevents emissive
// materials from producing erroneous radial streaks.

struct GodRaysBlurParams {
    width: u32,
    height: u32,
    num_samples: u32,      // default 64
    _pad0: u32,
    sun_uv: vec2<f32>,     // sun screen position [0,1]
    intensity: f32,        // overall strength (default 0.5)
    decay: f32,            // per-sample falloff (default 0.97)
    density: f32,          // sample weight (default 1.0)
    threshold: f32,        // luminance threshold for shaft contribution (default 0.8)
    sun_dot: f32,          // dot(sun_dir, cam_forward) for fade control
    _pad1: f32,
}

@group(0) @binding(0) var input_hdr: texture_2d<f32>;
@group(0) @binding(1) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: GodRaysBlurParams;
@group(0) @binding(3) var gbuf_position: texture_2d<f32>;  // G-buffer position (.w = MAX_FLOAT for sky)

// Depth threshold: anything beyond this is sky (ray marcher writes MAX_FLOAT = 3.4e38).
const SKY_DEPTH_THRESHOLD: f32 = 1.0e6;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2<i32>(gid.xy);
    let original = textureLoad(input_hdr, coord, 0).rgb;

    // Sun behind camera or below horizon → passthrough.
    let fade = smoothstep(0.0, 0.3, params.sun_dot);
    if (fade < 0.001 || params.intensity < 0.001) {
        textureStore(output_hdr, coord, vec4<f32>(original, 1.0));
        return;
    }

    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let pixel_uv = (vec2<f32>(gid.xy) + 0.5) / dims;

    // Direction from pixel toward sun screen position.
    let delta = params.sun_uv - pixel_uv;
    let dist_to_sun = length(delta);

    // Step length: march toward sun, scaled by distance so near-sun pixels
    // sample a shorter range (avoids over-blurring near the source).
    let step_len = min(dist_to_sun, 1.0) / f32(params.num_samples);
    let step_dir = normalize(delta + vec2<f32>(0.00001, 0.00001)) * step_len;

    var shaft_accum = vec3<f32>(0.0);
    var sample_uv = pixel_uv;
    var weight = 1.0;

    let idims = vec2<i32>(textureDimensions(input_hdr));

    for (var i = 0u; i < params.num_samples; i++) {
        sample_uv += step_dir;

        // Clamp to texture bounds.
        let sample_coord = clamp(
            vec2<i32>(sample_uv * dims),
            vec2<i32>(0),
            idims - vec2<i32>(1),
        );

        // Only accumulate SKY samples — skip geometry (including emissives).
        let depth = textureLoad(gbuf_position, sample_coord, 0).w;
        if (depth > SKY_DEPTH_THRESHOLD) {
            let sample_color = textureLoad(input_hdr, sample_coord, 0).rgb;
            let lum = luminance(sample_color);
            if (lum > params.threshold) {
                shaft_accum += sample_color * weight * params.density;
            }
        }

        weight *= params.decay;
    }

    // Normalize by sample count and apply intensity + sun visibility fade.
    let shafts = shaft_accum / f32(params.num_samples) * params.intensity * fade;

    let result = original + shafts;
    textureStore(output_hdr, coord, vec4<f32>(result, 1.0));
}
