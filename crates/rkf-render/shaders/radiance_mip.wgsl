// Radiance mip generation — clipmap resampling Level N → Level N+1
//
// Each clipmap level covers 4× more space than the previous. This shader
// maps L(N+1) texels back to L(N) coordinate space and box-filters a 2×2×2
// neighborhood. Texels outside L(N) coverage are zeroed out.
//
// Both radiance and opacity are averaged (not max). This provides correct
// pre-integration: thin bright surfaces appear as dim, spread-out glow at
// coarser levels, preventing sharp artifacts in cone tracing.
//
// Workgroup size: 4×4×4 → dispatch 32×32×32 for 128³ output.

@group(0) @binding(0) var src_level: texture_3d<f32>;
@group(0) @binding(1) var dst_level: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> mip_params: vec4<u32>; // x = dim (128)

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = mip_params.x;
    if gid.x >= dim || gid.y >= dim || gid.z >= dim { return; }

    // Map this L(N+1) texel back to L(N) coordinate space.
    // Each level covers 4× more space, so L(N) data occupies the center 1/4
    // of L(N+1) in each dimension.
    let center = f32(dim) * 0.5;           // 64.0 for dim=128
    let ratio = 4.0;                        // clipmap level ratio
    let src_f = (vec3<f32>(gid) + 0.5 - center) * ratio + center;

    // Smooth fade at L(N) projection boundary instead of hard zero.
    // norm is [0,1] in L(N) texel space; fade over 15% margin at each edge.
    let norm = src_f / f32(dim);
    let FADE = 0.15;
    let fx = smoothstep(0.0, FADE, norm.x) * smoothstep(1.0, 1.0 - FADE, norm.x);
    let fy = smoothstep(0.0, FADE, norm.y) * smoothstep(1.0, 1.0 - FADE, norm.y);
    let fz = smoothstep(0.0, FADE, norm.z) * smoothstep(1.0, 1.0 - FADE, norm.z);
    let mip_fade = fx * fy * fz;

    if mip_fade < 0.001 {
        textureStore(dst_level, vec3<i32>(gid), vec4<f32>(0.0));
        return;
    }

    // Clamp src_f to valid range — edge texels extend outward via ClampToEdge
    let clamped = clamp(src_f, vec3<f32>(0.5), vec3<f32>(f32(dim) - 0.5));
    let base = vec3<i32>(floor(clamped - 0.5));
    let max_c = vec3<i32>(i32(dim) - 1);

    var sum = vec4<f32>(0.0);

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let c = clamp(base + vec3<i32>(dx, dy, dz), vec3<i32>(0), max_c);
                sum += textureLoad(src_level, c, 0);
            }
        }
    }

    textureStore(dst_level, vec3<i32>(gid), sum / 8.0 * mip_fade);
}
