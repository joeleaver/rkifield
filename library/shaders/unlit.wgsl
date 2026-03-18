// Unlit shading model — flat color, no lighting.
//
// Returns albedo + emission. No shadows, no AO, no GI.

fn shade_unlit(ctx: ShadingContext) -> vec3<f32> {
    return ctx.albedo + ctx.emission * ctx.emission_strength;
}
