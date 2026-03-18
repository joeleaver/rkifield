// PBR shading model — full physically-based rendering with Cook-Torrance GGX.
//
// Evaluates all lights (directional, point, spot) with SDF soft shadows,
// ambient occlusion, subsurface scattering, GI via voxel cone tracing,
// atmospheric ambient, and contact shadows.

fn shade_pbr(ctx: ShadingContext) -> vec3<f32> {
    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    var sss_total = vec3<f32>(0.0);
    var shadow_count = 0u;
    let shadow_budget = shade_uniforms.shadow_budget_k;

    // Iterate all lights (no tile culling — simple for small counts)
    let num_lights = shade_uniforms.num_lights;
    for (var li = 0u; li < num_lights; li++) {
        let light = lights[li];
        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);
        let radiance = light_color * light.intensity;

        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            let light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let half_vec = normalize(ctx.view_dir + light_dir);

            let n_dot_l = max(dot(ctx.normal, light_dir), 0.0);
            let n_dot_h = max(dot(ctx.normal, half_vec), 0.0);
            let h_dot_v = max(dot(half_vec, ctx.view_dir), 0.0);

            let d = distribution_ggx(n_dot_h, ctx.roughness);
            let v = visibility_smith_ggx(ctx.n_dot_v, n_dot_l, ctx.roughness);
            let f = fresnel_schlick(h_dot_v, ctx.f0);

            let specular_brdf = d * v * f;
            let ks = f;
            let kd = (vec3<f32>(1.0) - ks) * (1.0 - ctx.metallic);
            let diffuse_brdf = kd * ctx.albedo / PI;

            var shadow = 1.0;
            if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                let shadow_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
                shadow = soft_shadow(shadow_origin, light_dir, SHADOW_MAX_DIST, SHADOW_K);
                shadow_count += 1u;
            }
            shadow = mix(shadow, 1.0, ctx.atmo_shadow_fill);

            total_diffuse += diffuse_brdf * radiance * n_dot_l * shadow;
            total_specular += specular_brdf * radiance * n_dot_l * shadow;
            sss_total += sss_contribution(ctx.world_pos, ctx.normal, light_dir, ctx.subsurface, ctx.sss_color)
                         * radiance * shadow;

        } else if light.light_type == LIGHT_TYPE_POINT {
            // Light positions are camera-relative in the buffer; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + shade_uniforms.camera_pos.xyz;
            let to_light = light_pos - ctx.world_pos;
            let dist = length(to_light);
            let light_dir = to_light / max(dist, 0.0001);

            let atten = distance_attenuation(dist, light.range);
            if atten > 0.001 {
                let half_vec = normalize(ctx.view_dir + light_dir);

                let n_dot_l = max(dot(ctx.normal, light_dir), 0.0);
                let n_dot_h = max(dot(ctx.normal, half_vec), 0.0);
                let h_dot_v = max(dot(half_vec, ctx.view_dir), 0.0);

                let d = distribution_ggx(n_dot_h, ctx.roughness);
                let v = visibility_smith_ggx(ctx.n_dot_v, n_dot_l, ctx.roughness);
                let f = fresnel_schlick(h_dot_v, ctx.f0);

                let specular_brdf = d * v * f;
                let ks = f;
                let kd = (vec3<f32>(1.0) - ks) * (1.0 - ctx.metallic);
                let diffuse_brdf = kd * ctx.albedo / PI;

                var shadow = 1.0;
                if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                    let shadow_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
                    shadow = soft_shadow(shadow_origin, light_dir, min(dist, SHADOW_MAX_DIST), SHADOW_K);
                    shadow_count += 1u;
                }
                shadow = mix(shadow, 1.0, ctx.atmo_shadow_fill);

                let attenuated_radiance = radiance * atten;
                total_diffuse += diffuse_brdf * attenuated_radiance * n_dot_l * shadow;
                total_specular += specular_brdf * attenuated_radiance * n_dot_l * shadow;
                sss_total += sss_contribution(ctx.world_pos, ctx.normal, light_dir, ctx.subsurface, ctx.sss_color)
                             * attenuated_radiance * shadow;
            }

        } else if light.light_type == LIGHT_TYPE_SPOT {
            // Light positions are camera-relative in the buffer; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + shade_uniforms.camera_pos.xyz;
            let spot_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let to_light = light_pos - ctx.world_pos;
            let dist = length(to_light);
            let light_dir = to_light / max(dist, 0.0001);

            let cos_angle = dot(-light_dir, spot_dir);
            let cos_outer = cos(light.outer_angle);
            let cos_inner = cos(light.inner_angle);
            let spot = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);

            let atten = spot * distance_attenuation(dist, light.range);
            if atten > 0.001 {
                let half_vec = normalize(ctx.view_dir + light_dir);

                let n_dot_l = max(dot(ctx.normal, light_dir), 0.0);
                let n_dot_h = max(dot(ctx.normal, half_vec), 0.0);
                let h_dot_v = max(dot(half_vec, ctx.view_dir), 0.0);

                let d = distribution_ggx(n_dot_h, ctx.roughness);
                let v = visibility_smith_ggx(ctx.n_dot_v, n_dot_l, ctx.roughness);
                let f = fresnel_schlick(h_dot_v, ctx.f0);

                let specular_brdf = d * v * f;
                let ks = f;
                let kd = (vec3<f32>(1.0) - ks) * (1.0 - ctx.metallic);
                let diffuse_brdf = kd * ctx.albedo / PI;

                var shadow = 1.0;
                if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                    let shadow_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS + light_dir * SHADOW_BIAS * 0.5;
                    shadow = soft_shadow(shadow_origin, light_dir, min(dist, SHADOW_MAX_DIST), SHADOW_K);
                    shadow_count += 1u;
                }
                shadow = mix(shadow, 1.0, ctx.atmo_shadow_fill);

                let attenuated_radiance = radiance * atten;
                total_diffuse += diffuse_brdf * attenuated_radiance * n_dot_l * shadow;
                total_specular += specular_brdf * attenuated_radiance * n_dot_l * shadow;
                sss_total += sss_contribution(ctx.world_pos, ctx.normal, light_dir, ctx.subsurface, ctx.sss_color)
                             * attenuated_radiance * shadow;
            }
        }
    }

    // SDF ambient occlusion
    let ao = sdf_ao(ctx.world_pos + ctx.normal * SHADOW_BIAS, ctx.normal);

    // GI via voxel cone tracing (6 diffuse + 1 specular cone).
    let gi_origin = ctx.world_pos + ctx.normal * SHADOW_BIAS * 2.0;
    let gi_diffuse_raw = cone_trace_diffuse(gi_origin, ctx.normal, ctx.jitter);
    let kd_gi = (1.0 - ctx.metallic);
    let gi_diffuse = gi_diffuse_raw * ctx.albedo * kd_gi * ao * GI_STRENGTH;

    let gi_specular_raw = cone_trace_specular(gi_origin, ctx.reflect_dir, ctx.roughness, ctx.jitter);
    let gi_fresnel = fresnel_schlick(ctx.n_dot_v, ctx.f0);
    let gi_specular = gi_specular_raw * gi_fresnel * ao * GI_STRENGTH;

    // Minimal ambient fallback (for areas outside radiance volume coverage).
    // Derive ambient from sky color so it responds to sun position.
    var ambient_sky_color: vec3<f32>;
    var ambient_reflect_color: vec3<f32>;
    if shade_uniforms.sky_params.z > 0.5 {
        let sun_d = normalize(shade_uniforms.sun_dir.xyz);
        ambient_sky_color = atmosphere_sky(vec3<f32>(0.0, 1.0, 0.0), sun_d) * 0.1;
        let reflect_env = reflect(-ctx.view_dir, ctx.normal);
        ambient_reflect_color = atmosphere_sky(reflect_env, sun_d);
    } else {
        ambient_sky_color = AMBIENT_COLOR;
        let reflect_env = reflect(-ctx.view_dir, ctx.normal);
        let sky_up_frac = clamp(reflect_env.y * 0.5 + 0.5, 0.0, 1.0);
        ambient_reflect_color = mix(SKY_HORIZON, SKY_ZENITH, sky_up_frac);
    }
    let kd_ambient = 1.0 - ctx.metallic;
    let ambient_diffuse = ambient_sky_color * ctx.albedo * ao * 0.15 * kd_ambient;
    let ambient_fresnel = fresnel_schlick(ctx.n_dot_v, ctx.f0);
    let ambient_specular = ambient_reflect_color * ambient_fresnel * ao * SKY_REFLECT_STRENGTH * 0.5;
    let ambient = ambient_diffuse + ambient_specular;

    // Final color = direct + SSS + GI + ambient + emission
    let emission = ctx.emission * ctx.emission_strength;
    let direct = (total_diffuse + total_specular) * ctx.contact;
    return direct + sss_total + (gi_diffuse + gi_specular + ambient) * ctx.contact + emission;
}
