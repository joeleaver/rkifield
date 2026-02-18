// Spatial upscale compute shader.
//
// Runs at display resolution. Bilinear-samples the internal-resolution
// HDR frame and writes it to the output. No temporal accumulation —
// the SDF ray marcher's per-frame variation from jitter is too large
// for simple TAA to absorb without visible wobble.
//
// Temporal super-resolution (FSR2-style Lanczos + lock + per-signal
// filtering) is a future upgrade path. For now, spatial bilinear at
// 1.33x is sufficient quality.

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
@group(0) @binding(0) var hdr_color: texture_2d<f32>;
@group(0) @binding(1) var gbuf_position: texture_2d<f32>;
@group(0) @binding(2) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(3) var gbuf_material: texture_2d<u32>;
@group(0) @binding(4) var gbuf_motion: texture_2d<f32>;
@group(0) @binding(5) var bilinear_sampler: sampler;

// Group 1: history (unused, kept for bind group layout compatibility)
@group(1) @binding(0) var history_color: texture_2d<f32>;
@group(1) @binding(1) var history_meta: texture_2d<u32>;

// Group 2: outputs (display resolution)
@group(2) @binding(0) var output_color: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var out_history_color: texture_storage_2d<rgba16float, write>;
@group(2) @binding(2) var out_history_meta: texture_storage_2d<rg32uint, write>;

// Group 3: uniforms
@group(3) @binding(0) var<uniform> params: UpscaleUniforms;

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let display_dims = vec2<u32>(params.display_width, params.display_height);
    if gid.x >= display_dims.x || gid.y >= display_dims.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(display_dims);

    // Bilinear upscale from internal to display resolution
    let color = textureSampleLevel(hdr_color, bilinear_sampler, uv, 0.0).rgb;

    textureStore(output_color, coord, vec4<f32>(color, 1.0));
    // Write to history outputs too (bind group expects writes)
    textureStore(out_history_color, coord, vec4<f32>(color, 1.0));
    textureStore(out_history_meta, coord, vec4<u32>(0u, 0u, 0u, 0u));
}
