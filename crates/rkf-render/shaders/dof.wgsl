// Depth of Field compute shader — two entry points:
//   1. coc_compute — compute per-pixel circle of confusion from G-buffer depth
//   2. dof_blur    — gather disc-kernel blur weighted by CoC

struct DofParams {
    focus_distance: f32,  // world-space distance to focus plane
    focus_range: f32,     // depth range over which focus transitions (aperture)
    max_coc: f32,         // maximum CoC radius in pixels
    width: u32,
    height: u32,
    near_start: f32,      // near field starts blurring at this depth
    near_end: f32,        // near field fully blurred at this depth
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Entry point 1: coc_compute
// Computes signed CoC from G-buffer depth.
// Negative = near field (in front of focus plane), positive = far field.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var position_tex: texture_2d<f32>;
@group(0) @binding(1) var coc_tex: texture_storage_2d<r16float, write>;
@group(0) @binding(2) var<uniform> params: DofParams;

@compute @workgroup_size(8, 8, 1)
fn coc_compute(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let pos = textureLoad(position_tex, vec2<i32>(gid.xy), 0);
    let depth = pos.w;  // distance from G-buffer (ray march hit distance)

    // Skip sky pixels (depth = 0 or very large)
    if (depth <= 0.0 || depth > 1000.0) {
        textureStore(coc_tex, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Signed CoC: negative = near field, positive = far field
    let coc = clamp(
        (depth - params.focus_distance) / params.focus_range,
        -1.0,
        1.0,
    ) * params.max_coc;

    textureStore(coc_tex, vec2<i32>(gid.xy), vec4<f32>(coc, 0.0, 0.0, 0.0));
}

// ---------------------------------------------------------------------------
// Entry point 2: dof_blur
// Gather blur using a disc kernel. Samples weighted by CoC to model
// near-field bleeding and far-field separation.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var input_hdr: texture_2d<f32>;
@group(0) @binding(1) var coc_tex_read: texture_2d<f32>;
@group(0) @binding(2) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> blur_params: DofParams;

@compute @workgroup_size(8, 8, 1)
fn dof_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= blur_params.width || gid.y >= blur_params.height) { return; }

    let center = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(input_hdr));
    let center_color = textureLoad(input_hdr, center, 0).rgb;
    let center_coc = textureLoad(coc_tex_read, center, 0).r;
    let abs_coc = abs(center_coc);

    // No blur needed if CoC is tiny
    if (abs_coc < 0.5) {
        textureStore(output_hdr, center, vec4<f32>(center_color, 1.0));
        return;
    }

    // Disc kernel — sample in concentric rings scaled to CoC radius
    let radius = min(abs_coc, blur_params.max_coc);
    // Number of rings: 1-3, proportional to radius
    let num_rings = u32(clamp(radius / 2.0, 1.0, 3.0));

    var total_color = center_color;
    var total_weight = 1.0;

    // Samples per ring (ring 1 = 6, ring 2 = 12, ring 3 = 18)
    // We iterate manually because WGSL arrays in loop bodies must be statically sized
    for (var ring = 1u; ring <= num_rings; ring++) {
        let ring_radius = radius * f32(ring) / f32(num_rings);
        var num_samples: u32;
        if (ring == 1u) {
            num_samples = 6u;
        } else if (ring == 2u) {
            num_samples = 12u;
        } else {
            num_samples = 18u;
        }
        let angle_step = 6.283185307 / f32(num_samples);

        for (var s = 0u; s < num_samples; s++) {
            let angle = f32(s) * angle_step;
            let offset = vec2<i32>(vec2<f32>(cos(angle), sin(angle)) * ring_radius);
            let sample_pos = clamp(center + offset, vec2<i32>(0), dims - vec2<i32>(1));

            let sample_color = textureLoad(input_hdr, sample_pos, 0).rgb;
            let sample_coc = textureLoad(coc_tex_read, sample_pos, 0).r;

            // Near-field bleeding: near samples (negative CoC) can bleed over
            // in-focus regions. Far-field samples should not bleed into near.
            var w = 1.0;
            if (sample_coc < 0.0 && center_coc >= 0.0) {
                // Near field bleeding into focused area — reduced weight
                w = 0.5 * saturate(abs(sample_coc) / blur_params.max_coc);
            } else if (sample_coc > 0.0 && center_coc < 0.0) {
                // Far field should not bleed into near field
                w = 0.1;
            }

            total_color += sample_color * w;
            total_weight += w;
        }
    }

    let result = total_color / total_weight;
    textureStore(output_hdr, center, vec4<f32>(result, 1.0));
}
