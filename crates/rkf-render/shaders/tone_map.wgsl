// Tone mapping compute shader — Phase 6.
//
// Reads HDR Rgba16Float, applies ACES tone mapping and gamma correction,
// writes LDR Rgba8Unorm for display.

// ---------- Bindings ----------

@group(0) @binding(0) var hdr_input: texture_2d<f32>;
@group(1) @binding(0) var ldr_output: texture_storage_2d<rgba8unorm, write>;

// ---------- ACES Tone Mapping ----------

// ACES filmic tone mapping curve (Krzysztof Narkowicz approximation).
// Simple, good-looking, widely used in game engines.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Linear to sRGB gamma correction.
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, linear <= cutoff);
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(ldr_output));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);
    let hdr = textureLoad(hdr_input, coord, 0).rgb;

    // Apply ACES tone mapping
    let mapped = aces_tonemap(hdr);

    // Apply sRGB gamma correction
    let srgb = linear_to_srgb(mapped);

    textureStore(ldr_output, coord, vec4<f32>(srgb, 1.0));
}
