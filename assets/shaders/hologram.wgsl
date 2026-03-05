// Hologram shading model — glowing scanlines with fresnel edge glow.
//
// A demo custom shader showing how user-authored shading models work.
// Place .wgsl files in assets/shaders/ with a function matching
// `fn shade_<name>(ctx: ShadingContext) -> vec3<f32>` and they are
// automatically registered and hot-reloaded.

fn shade_hologram(ctx: ShadingContext) -> vec3<f32> {
    // Horizontal scanline pattern modulated by jitter for animation.
    let scan = sin(ctx.world_pos.y * 50.0 + ctx.jitter * 6.28) * 0.5 + 0.5;

    // Fresnel-based edge glow — strong at grazing angles.
    let fresnel = pow(1.0 - ctx.n_dot_v, 3.0);

    // Base hologram color (cyan) with fresnel glow and scanline modulation.
    let glow = vec3<f32>(0.0, 0.8, 1.0) * (fresnel * 2.0 + scan * 0.3);

    // Combine with material albedo tint and emission.
    return glow * ctx.albedo + ctx.emission * ctx.emission_strength;
}
