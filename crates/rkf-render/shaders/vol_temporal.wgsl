// Volumetric temporal reprojection compute shader.
//
// Blends the current frame's half-res volumetric buffer with reprojected
// history for temporal stability. Uses depth-based validation to reject
// disoccluded pixels.

struct VolTemporalParams {
    width: u32,             // half-res width
    height: u32,            // half-res height
    blend_factor: f32,      // history weight (0.9 = 90% history)
    depth_threshold: f32,   // max relative depth difference for validity
    frame_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: VolTemporalParams;
@group(0) @binding(1) var current_scatter: texture_2d<f32>;    // current frame vol scatter (rgba16float)
@group(0) @binding(2) var history_scatter: texture_2d<f32>;    // previous frame vol scatter (rgba16float)
@group(0) @binding(3) var motion_vectors: texture_2d<f32>;     // full-res motion vectors (Rg32Float from G-buffer)
@group(0) @binding(4) var output_scatter: texture_storage_2d<rgba16float, write>;  // blended result

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2<i32>(gid.xy);
    let dims = vec2<f32>(f32(params.width), f32(params.height));
    let uv = (vec2<f32>(gid.xy) + 0.5) / dims;

    // Current frame scatter
    let current = textureLoad(current_scatter, coord, 0);

    // Sample motion vectors at corresponding full-res coordinate
    let full_coord = vec2<i32>(vec2<f32>(gid.xy) * 2.0);
    let motion = textureLoad(motion_vectors, full_coord, 0).xy;

    // Reproject to previous frame UV
    let prev_uv = uv - motion;

    // Bounds check — reject if reprojected UV is outside [0, 1]
    if (prev_uv.x < 0.0 || prev_uv.x >= 1.0 || prev_uv.y < 0.0 || prev_uv.y >= 1.0) {
        textureStore(output_scatter, coord, current);
        return;
    }

    // Sample history at reprojected position (nearest neighbor at half res)
    let prev_coord = vec2<i32>(prev_uv * dims);
    let history = textureLoad(history_scatter, prev_coord, 0);

    // Depth-based validation: compare transmittance as a proxy for depth consistency
    // Large transmittance difference suggests disocclusion
    let transmittance_diff = abs(current.w - history.w);
    let valid = transmittance_diff < params.depth_threshold;

    // Blend: high history weight for valid, current-only for invalid
    let blend = select(0.0, params.blend_factor, valid);
    let result = mix(current, history, blend);

    textureStore(output_scatter, coord, result);
}
