// Color grading compute shader (post-upscale).
//
// Applies a 3D lookup table (LUT) as a final color transform. The LUT
// maps input RGB to output RGB via trilinear sampling of a 3D texture.
// Intensity controls blend between original and graded colors.

struct ColorGradeParams {
    width: u32,
    height: u32,
    lut_size: u32,     // LUT dimension (e.g. 32 or 64)
    intensity: f32,    // blend factor: 0.0 = no grading, 1.0 = full LUT
}

@group(0) @binding(0) var input_ldr: texture_2d<f32>;
@group(0) @binding(1) var lut_texture: texture_3d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var output_ldr: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: ColorGradeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let color = textureLoad(input_ldr, vec2<i32>(gid.xy), 0).rgb;

    // Scale color to LUT UV coordinates.
    // Offset by half-texel to sample at cell centers.
    let lut_size_f = f32(params.lut_size);
    let scale = (lut_size_f - 1.0) / lut_size_f;
    let offset_val = 0.5 / lut_size_f;
    let lut_uv = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)) * scale + offset_val;

    // Sample LUT with trilinear filtering.
    let graded = textureSampleLevel(lut_texture, lut_sampler, lut_uv, 0.0).rgb;

    // Blend between original and graded based on intensity.
    let result = mix(color, graded, params.intensity);

    textureStore(output_ldr, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
