// Bilateral upscale for volumetric scattering buffer.
//
// Upsamples half-res (480×270) → full internal res (960×540) using
// depth-guided bilateral weights to prevent fog bleeding across edges.

struct VolUpscaleParams {
    out_width: u32,       // full internal res width (960)
    out_height: u32,      // full internal res height (540)
    in_width: u32,        // half-res width (480)
    in_height: u32,       // half-res height (270)
    depth_sigma: f32,     // depth similarity sigma (controls edge sharpness)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: VolUpscaleParams;
@group(0) @binding(1) var half_res_scatter: texture_2d<f32>;  // half-res volumetric scatter (Rgba16Float)
@group(0) @binding(2) var full_res_depth: texture_2d<f32>;    // full-res depth (position.w from G-buffer)
@group(0) @binding(3) var output_scatter: texture_storage_2d<rgba16float, write>;  // full-res output

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.out_width || gid.y >= params.out_height) { return; }

    let out_coord = vec2<i32>(gid.xy);

    // Full-res depth at this pixel
    let ref_depth = textureLoad(full_res_depth, out_coord, 0).w;

    // Map to continuous half-res coordinate
    let half_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.out_width), f32(params.out_height))
                  * vec2<f32>(f32(params.in_width), f32(params.in_height)) - 0.5;

    // 4 nearest low-res texels (2×2 neighborhood)
    let base = vec2<i32>(floor(half_uv));
    let frac = half_uv - vec2<f32>(base);

    var total_weight: f32 = 0.0;
    var total_color: vec4<f32> = vec4<f32>(0.0);

    for (var dy = 0; dy <= 1; dy++) {
        for (var dx = 0; dx <= 1; dx++) {
            let sample_coord = clamp(
                base + vec2<i32>(dx, dy),
                vec2(0),
                vec2<i32>(i32(params.in_width) - 1, i32(params.in_height) - 1)
            );

            let sample_val = textureLoad(half_res_scatter, sample_coord, 0);

            // Corresponding full-res depth for this low-res texel
            let full_sample_coord = clamp(
                sample_coord * 2,
                vec2(0),
                vec2<i32>(i32(params.out_width) - 1, i32(params.out_height) - 1)
            );
            let sample_depth = textureLoad(full_res_depth, full_sample_coord, 0).w;

            // Bilinear weight
            let bx = select(1.0 - frac.x, frac.x, dx == 1);
            let by = select(1.0 - frac.y, frac.y, dy == 1);
            let bilinear_w = bx * by;

            // Depth similarity weight (Gaussian)
            let depth_diff = abs(ref_depth - sample_depth) / max(ref_depth, 0.001);
            let depth_w = exp(-depth_diff * depth_diff / (2.0 * params.depth_sigma * params.depth_sigma));

            let w = bilinear_w * depth_w;
            total_weight += w;
            total_color += sample_val * w;
        }
    }

    // Normalize or fall back to nearest if weights too small
    if (total_weight > 0.001) {
        textureStore(output_scatter, out_coord, total_color / total_weight);
    } else {
        // Fallback: nearest neighbor
        let nearest = clamp(vec2<i32>(round(half_uv)), vec2(0),
            vec2<i32>(i32(params.in_width) - 1, i32(params.in_height) - 1));
        textureStore(output_scatter, out_coord, textureLoad(half_res_scatter, nearest, 0));
    }
}
