// Motion blur compute shader (pre-upscale).
//
// Samples along the per-pixel motion vector from the G-buffer.
// Sample count is proportional to motion magnitude (max params.max_samples).
// Depth-aware weighting uses motion vector similarity to avoid
// bleeding between objects moving in different directions.

struct MotionBlurParams {
    width: u32,
    height: u32,
    intensity: f32,   // motion blur strength multiplier (default 1.0)
    max_samples: u32, // max blur samples per pixel (default 16)
}

@group(0) @binding(0) var input_hdr: texture_2d<f32>;     // HDR color
@group(0) @binding(1) var motion_tex: texture_2d<f32>;     // G-buffer motion vectors (RG = screen-space velocity in UV units)
@group(0) @binding(2) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params: MotionBlurParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let center = vec2<i32>(gid.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let center_color = textureLoad(input_hdr, center, 0).rgb;

    // Motion vector in UV space → pixel space
    let motion_uv = textureLoad(motion_tex, center, 0).rg;
    let motion_px = motion_uv * dims * params.intensity;
    let motion_len = length(motion_px);

    // No blur needed for near-stationary pixels
    if (motion_len < 0.5) {
        textureStore(output_hdr, center, vec4<f32>(center_color, 1.0));
        return;
    }

    // Sample count proportional to motion magnitude
    let num_samples = clamp(u32(motion_len + 0.5), 2u, params.max_samples);
    let step = motion_px / f32(num_samples);

    var total_color = center_color;
    var total_weight = 1.0;
    let idims = vec2<i32>(textureDimensions(input_hdr));

    // Sample along motion vector in both directions
    for (var i = 1u; i < num_samples; i++) {
        let t = f32(i) / f32(num_samples) - 0.5;  // range [-0.5, 0.5)
        let offset = vec2<i32>(step * t * 2.0);
        let sample_pos = clamp(center + offset, vec2<i32>(0), idims - vec2<i32>(1));

        // Depth-aware weighting: check if sample has similar motion
        let sample_motion = textureLoad(motion_tex, sample_pos, 0).rg;
        let sample_motion_px = sample_motion * dims * params.intensity;
        let motion_similarity = dot(normalize(motion_px), normalize(sample_motion_px + vec2<f32>(0.0001)));
        let w = max(motion_similarity, 0.1);

        total_color += textureLoad(input_hdr, sample_pos, 0).rgb * w;
        total_weight += w;
    }

    let result = total_color / total_weight;
    textureStore(output_hdr, center, vec4<f32>(result, 1.0));
}
