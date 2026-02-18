// Radiance mip generation — downsample Level N → Level N+1
//
// 2×2×2 box filter: radiance = average(8 children), opacity = max(8 children).
// Workgroup size: 4×4×4 → dispatch 32×32×32 for 128³ output.

@group(0) @binding(0) var src_level: texture_3d<f32>;
@group(0) @binding(1) var dst_level: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> mip_params: vec4<u32>; // x = dim (128)

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = mip_params.x;
    if gid.x >= dim || gid.y >= dim || gid.z >= dim { return; }

    // The 8 source texels forming the 2×2×2 block for this output texel.
    // Source coord = output_coord * 2 + offset, clamped to [0, dim-1].
    let base = vec3<i32>(gid) * 2;
    let max_coord = vec3<i32>(i32(dim) - 1);

    var sum_radiance = vec3<f32>(0.0);
    var max_opacity: f32 = 0.0;

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let src_coord = clamp(base + vec3<i32>(dx, dy, dz), vec3<i32>(0), max_coord);
                let texel = textureLoad(src_level, src_coord, 0);
                sum_radiance += texel.rgb;
                max_opacity = max(max_opacity, texel.a);
            }
        }
    }

    let avg_radiance = sum_radiance / 8.0;
    textureStore(dst_level, vec3<i32>(gid), vec4<f32>(avg_radiance, max_opacity));
}
