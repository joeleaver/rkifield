// Volumetric compositing compute shader — Phase 11 task 11.8.
//
// Composites upscaled volumetric scattering over the shaded scene color.
// Formula: final = shaded_color * transmittance + scatter_color
//
// The volumetric scatter buffer encodes:
//   .rgb — in-scattered light accumulated along the view ray
//   .a   — transmittance (fraction of scene light that reaches the camera)
//
// A transmittance of 1.0 means fully clear (no fog); 0.0 means fully opaque
// (scene is completely occluded by the medium). The composite blends the two
// contributions in a physically correct front-to-back manner.

struct VolCompositeParams {
    width:  u32,
    height: u32,
    _pad0:  u32,
    _pad1:  u32,
}

@group(0) @binding(0) var<uniform> params: VolCompositeParams;
/// Scene color output from the shading pass (Rgba16Float, internal resolution).
@group(0) @binding(1) var shaded_color: texture_2d<f32>;
/// Upscaled volumetric scatter buffer (Rgba16Float: rgb = scatter, a = transmittance).
@group(0) @binding(2) var vol_scatter:  texture_2d<f32>;
/// Composited output written by this pass (Rgba16Float, internal resolution).
@group(0) @binding(3) var output_color: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2<i32>(gid.xy);

    let scene = textureLoad(shaded_color, coord, 0);
    let vol   = textureLoad(vol_scatter,  coord, 0);

    let scatter       = vol.rgb;
    let transmittance = vol.a;

    // Front-to-back compositing:
    //   final_rgb = scene_rgb * transmittance + in_scattered_light
    let composited = scene.rgb * transmittance + scatter;

    textureStore(output_color, coord, vec4<f32>(composited, scene.a));
}
