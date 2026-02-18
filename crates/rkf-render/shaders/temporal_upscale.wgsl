// Temporal upscale compute shader.
//
// Runs at display resolution. Bilinear-samples the internal-resolution
// HDR frame, reprojects history via motion vectors, applies neighborhood
// clipping in YCoCg, and blends for temporal super-sampling.

// ---------- Uniforms ----------

struct UpscaleUniforms {
    display_width:  u32,
    display_height: u32,
    internal_width: u32,
    internal_height: u32,
    jitter_x: f32,
    jitter_y: f32,
    _pad0: u32,
    _pad1: u32,
}

// ---------- Bindings ----------

// Group 0: current frame (internal resolution)
@group(0) @binding(0) var hdr_color: texture_2d<f32>;          // HDR from shade pass
@group(0) @binding(1) var gbuf_position: texture_2d<f32>;      // world pos + hit dist
@group(0) @binding(2) var gbuf_normal: texture_2d<f32>;        // normal + blend
@group(0) @binding(3) var gbuf_material: texture_2d<u32>;      // packed material IDs
@group(0) @binding(4) var gbuf_motion: texture_2d<f32>;        // motion vectors
@group(0) @binding(5) var bilinear_sampler: sampler;            // bilinear filtering

// Group 1: history read (display resolution, previous frame)
@group(1) @binding(0) var history_color: texture_2d<f32>;      // accumulated HDR
@group(1) @binding(1) var history_meta: texture_2d<u32>;       // packed depth + material

// Group 2: outputs (display resolution)
@group(2) @binding(0) var output_color: texture_storage_2d<rgba16float, write>;  // upscaled result
@group(2) @binding(1) var out_history_color: texture_storage_2d<rgba16float, write>;  // new history
@group(2) @binding(2) var out_history_meta: texture_storage_2d<rg32uint, write>;      // new metadata

// Group 3: uniforms
@group(3) @binding(0) var<uniform> params: UpscaleUniforms;

// ---------- Constants ----------

const MAX_FLOAT: f32 = 3.402823e+38;
const BLEND_FACTOR_BASE: f32 = 0.9;       // high = smoother, more history
const DEPTH_REJECT_THRESHOLD: f32 = 0.1;  // relative depth difference for rejection
const MOTION_DAMPEN_SCALE: f32 = 20.0;    // motion magnitude → trust reduction

// ---------- Color space helpers ----------

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y  =  rgb.r * 0.25 + rgb.g * 0.5 + rgb.b * 0.25;
    let co =  rgb.r * 0.5                 - rgb.b * 0.5;
    let cg = -rgb.r * 0.25 + rgb.g * 0.5 - rgb.b * 0.25;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y  = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    return vec3<f32>(y + co - cg, y + cg, y - co - cg);
}

// ---------- Helpers ----------

/// Pack depth (as f32 bits) and material_id into Rg32Uint.
fn pack_metadata(depth: f32, material_id: u32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(depth), material_id);
}

/// Unpack depth from the R channel of metadata.
fn unpack_depth(packed: vec2<u32>) -> f32 {
    return bitcast<f32>(packed.x);
}

/// Unpack material_id from the G channel of metadata.
fn unpack_material(packed: vec2<u32>) -> u32 {
    return packed.y;
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let display_dims = vec2<u32>(params.display_width, params.display_height);
    if gid.x >= display_dims.x || gid.y >= display_dims.y {
        return;
    }

    let internal_dims = vec2<f32>(f32(params.internal_width), f32(params.internal_height));
    let display_dims_f = vec2<f32>(display_dims);
    let coord = vec2<i32>(gid.xy);

    // UV in [0, 1] for this display pixel
    let uv = (vec2<f32>(gid.xy) + 0.5) / display_dims_f;

    // ----- Sample current frame at internal resolution -----
    // Unjitter: the internal frame was rendered with sub-pixel jitter applied
    // to ray generation. Compensate by shifting the sample position so the
    // output is stable across frames (jitter accumulates via history instead).
    let jitter_uv = vec2<f32>(params.jitter_x, params.jitter_y) / internal_dims;
    let current_hdr = textureSampleLevel(hdr_color, bilinear_sampler, uv - jitter_uv, 0.0).rgb;

    // Sample G-buffer at nearest internal-res pixel
    let internal_coord = vec2<i32>(vec2<f32>(uv * internal_dims));
    let internal_coord_clamped = clamp(internal_coord,
        vec2<i32>(0),
        vec2<i32>(i32(params.internal_width) - 1, i32(params.internal_height) - 1));

    let position_sample = textureLoad(gbuf_position, internal_coord_clamped, 0);
    let hit_dist = position_sample.w;
    let normal_sample = textureLoad(gbuf_normal, internal_coord_clamped, 0);
    let material_packed = textureLoad(gbuf_material, internal_coord_clamped, 0).r;
    let material_id = material_packed & 0xFFFFu;
    let motion = textureLoad(gbuf_motion, internal_coord_clamped, 0).rg;

    // ----- Sky pixels: no history, just output current -----
    if hit_dist >= MAX_FLOAT * 0.5 {
        textureStore(output_color, coord, vec4<f32>(current_hdr, 1.0));
        textureStore(out_history_color, coord, vec4<f32>(current_hdr, 1.0));
        textureStore(out_history_meta, coord, vec4<u32>(pack_metadata(MAX_FLOAT, 0u), 0u, 0u));
        return;
    }

    // ----- Reproject via motion vector -----
    let prev_uv = uv - motion;

    // ----- 3×3 neighborhood AABB in YCoCg for color clipping -----
    var aabb_min = vec3<f32>(1e10);
    var aabb_max = vec3<f32>(-1e10);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor_coord = clamp(
                internal_coord_clamped + vec2<i32>(dx, dy),
                vec2<i32>(0),
                vec2<i32>(i32(params.internal_width) - 1, i32(params.internal_height) - 1)
            );
            let neighbor_rgb = textureLoad(hdr_color, neighbor_coord, 0).rgb;
            let neighbor_ycocg = rgb_to_ycocg(neighbor_rgb);
            aabb_min = min(aabb_min, neighbor_ycocg);
            aabb_max = max(aabb_max, neighbor_ycocg);
        }
    }

    // ----- Sample history -----
    var blend_factor = BLEND_FACTOR_BASE;

    // Check if reprojected UV is within screen bounds
    let in_bounds = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv < vec2<f32>(1.0));

    var history_rgb = current_hdr; // fallback if no valid history

    if in_bounds {
        let prev_coord = vec2<i32>(prev_uv * display_dims_f);
        let prev_coord_clamped = clamp(prev_coord,
            vec2<i32>(0),
            vec2<i32>(i32(display_dims.x) - 1, i32(display_dims.y) - 1));

        history_rgb = textureLoad(history_color, prev_coord_clamped, 0).rgb;
        let prev_meta_packed = textureLoad(history_meta, prev_coord_clamped, 0).rg;
        let prev_depth = unpack_depth(prev_meta_packed);
        let prev_material = unpack_material(prev_meta_packed);

        // ----- Material rejection -----
        if prev_material != material_id {
            blend_factor = 0.0;
        }

        // ----- Depth rejection -----
        let depth_diff = abs(hit_dist - prev_depth) / max(hit_dist, 0.001);
        if depth_diff > DEPTH_REJECT_THRESHOLD {
            blend_factor = 0.0;
        }

        // ----- Motion magnitude trust reduction -----
        let motion_len = length(motion);
        blend_factor *= saturate(1.0 - motion_len * MOTION_DAMPEN_SCALE);

        // ----- Neighborhood clipping in YCoCg -----
        let history_ycocg = rgb_to_ycocg(history_rgb);
        let clipped_ycocg = clamp(history_ycocg, aabb_min, aabb_max);
        history_rgb = ycocg_to_rgb(clipped_ycocg);
    } else {
        // Off-screen: no history available
        blend_factor = 0.0;
    }

    // ----- Blend -----
    let result = mix(current_hdr, history_rgb, blend_factor);

    // ----- Write outputs -----
    textureStore(output_color, coord, vec4<f32>(result, 1.0));
    textureStore(out_history_color, coord, vec4<f32>(result, 1.0));
    textureStore(out_history_meta, coord, vec4<u32>(pack_metadata(hit_dist, material_id), 0u, 0u));
}
