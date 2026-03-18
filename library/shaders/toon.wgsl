// Toon/cel shading model — quantized lighting with rim highlight.
//
// Uses soft_shadow and sdf_ao from shade_common.wgsl.

fn shade_toon(ctx: ShadingContext) -> vec3<f32> {
    // Iterate lights for N·L, using just the first directional light for simplicity
    var n_dot_l = 0.0;
    var shadow = 1.0;
    let num_lights = shade_uniforms.num_lights;
    for (var li = 0u; li < num_lights; li++) {
        let light = lights[li];
        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            let light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            n_dot_l = max(dot(ctx.normal, light_dir), 0.0);
            if light.shadow_caster == 1u {
                let shadow_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
                shadow = soft_shadow(shadow_origin, light_dir, SHADOW_MAX_DIST, SHADOW_K);
            }
            break;  // Use first directional light only
        }
    }

    // Quantize N·L into bands
    let band = floor(n_dot_l * 4.0) / 4.0;

    // Hard shadow threshold
    let hard_shadow = select(1.0, 0.3, shadow < 0.5);

    // Ambient occlusion
    let ao = sdf_ao(ctx.world_pos + ctx.normal * SHADOW_BIAS, ctx.normal);

    // Rim lighting
    let rim = pow(1.0 - ctx.n_dot_v, 3.0) * 0.5;

    // Flat ambient
    let ambient = 0.15;

    let lit = ctx.albedo * (band * hard_shadow + ambient) * ao + vec3<f32>(rim);
    return lit + ctx.emission * ctx.emission_strength;
}
