// Emissive shading model — emission-focused, bloom-friendly.
//
// Primarily driven by emission color and strength, with minimal ambient
// contribution from albedo. Designed for high HDR values that drive bloom.

fn shade_emissive(ctx: ShadingContext) -> vec3<f32> {
    let emission = ctx.emission * ctx.emission_strength;
    let ambient = ctx.albedo * 0.05;  // minimal ambient contribution
    return emission + ambient;
}
