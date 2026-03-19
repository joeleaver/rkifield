// Volumetric march compute shader (half resolution) — Phase 11 task 11.2 / 11.4.
//
// Front-to-back fixed-step compositing through participating media.
// Jittered step offset (interleaved gradient noise) for temporal accumulation.
//
// Task 11.4 adds analytic height fog and distance fog density functions.
// Fog parameters are packed into three vec4 fields in VolMarchParams.
//
// Output: Rgba16Float texture (scattering_rgb, transmittance).

struct VolMarchParams {
    // Camera (vec4 packed for alignment)
    cam_pos:     vec4<f32>,  // xyz = camera position, w = unused
    cam_forward: vec4<f32>,  // xyz = forward, w = unused
    cam_right:   vec4<f32>,  // xyz = right (scaled by fov), w = unused
    cam_up:      vec4<f32>,  // xyz = up (scaled by fov), w = unused
    // Sun
    sun_dir:   vec4<f32>,    // xyz = direction toward sun, w = unused
    sun_color: vec4<f32>,    // xyz = sun color * intensity, w = unused
    // Resolution and march settings
    width:       u32,        // half-res width
    height:      u32,        // half-res height
    full_width:  u32,        // full internal res width (for depth sampling)
    full_height: u32,        // full internal res height
    max_steps:   u32,        // max march steps (32-64)
    step_size:   f32,        // world-space step size (metres)
    near:        f32,        // near plane distance
    far:         f32,        // max march distance
    // Fog settings (task 11.4)
    // fog_color:    xyz = scattering RGB, w = height_fog_enable (0.0 / 1.0)
    // fog_height:   x = base_density, y = base_height, z = height_falloff,
    //               w = distance_fog_enable (0.0 / 1.0)
    // fog_distance: x = distance_density, y = distance_falloff,
    //               z = ambient_dust_density, w = ambient_dust_g
    fog_color:    vec4<f32>,
    fog_height:   vec4<f32>,
    fog_distance: vec4<f32>,
    frame_index:       u32,
    vol_ambient_r:     f32,        // Volumetric ambient sky color R
    vol_ambient_g:     f32,        // Volumetric ambient sky color G
    vol_ambient_b:     f32,        // Volumetric ambient sky color B
    // Volumetric shadow map volume bounds (world space)
    vol_shadow_min: vec4<f32>,  // xyz = min corner, w = unused
    vol_shadow_max: vec4<f32>,  // xyz = max corner, w = unused
}

@group(0) @binding(0) var<uniform> params: VolMarchParams;
// Full-res depth: the .w channel stores scene distance from the G-buffer position pass.
@group(0) @binding(1) var depth_buffer:    texture_2d<f32>;
// Volumetric shadow map from task 11.1 (transmittance per world position).
@group(0) @binding(2) var vol_shadow_map:  texture_3d<f32>;
// Linear-clamp sampler for the shadow map.
@group(0) @binding(3) var vol_shadow_smp:  sampler;
// Output half-res scatter texture: (scatter_r, scatter_g, scatter_b, transmittance).
@group(0) @binding(4) var output_scatter:  texture_storage_2d<rgba16float, write>;

// Cloud parameters (procedural FBM clouds).
struct CloudParams {
    // x=cloud_min, y=cloud_max, z=threshold, w=density_scale
    altitude: vec4<f32>,
    // x=shape_freq, y=detail_freq, z=detail_weight, w=weather_scale
    noise: vec4<f32>,
    // x=wind_dir.x, y=wind_dir.y, z=wind_speed, w=time
    wind: vec4<f32>,
    // x=procedural_enable (0/1), y=shadow_coverage, z=shadow_res, w=brick_clouds_enable (0/1)
    flags: vec4<f32>,
}
@group(0) @binding(5) var<uniform> cloud_params: CloudParams;
// Cloud shadow map (2D R32Float, camera-centered transmittance).
@group(0) @binding(6) var cloud_shadow_tex: texture_2d<f32>;

const PI: f32 = 3.14159265359;

// ---------------------------------------------------------------------------
// Interleaved gradient noise — jitters step offset per pixel per frame.
// Pixel coords are offset by a frame-dependent value to give temporal variety.
// ---------------------------------------------------------------------------
fn interleaved_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let magic = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    let offset = 5.588238 * f32(frame % 64u);
    let p = pixel + vec2<f32>(offset, offset);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

// ---------------------------------------------------------------------------
// Henyey-Greenstein phase function.
// cos_theta: cosine of the angle between the view ray and the sun direction.
// g: asymmetry parameter (-1 = full backscatter, 0 = isotropic, 1 = full forward).
// ---------------------------------------------------------------------------
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-6), 1.5));
}

// ---------------------------------------------------------------------------
// Sample volumetric shadow map at a world position.
// Returns transmittance in [0, 1] (1 = fully lit, 0 = fully shadowed).
//
// Computes UVW coordinates from the world position relative to the shadow
// volume bounds stored in params. If the volume is unconfigured (zero size)
// or the position lies outside the volume, returns 1.0 (fully lit).
// ---------------------------------------------------------------------------
fn sample_vol_shadow(pos: vec3<f32>) -> f32 {
    // Compute UVW from world position relative to shadow volume bounds.
    let vol_min = params.vol_shadow_min.xyz;
    let vol_max = params.vol_shadow_max.xyz;
    let vol_size = vol_max - vol_min;

    // If volume size is zero (not configured), return fully lit.
    if (vol_size.x <= 0.0 || vol_size.y <= 0.0 || vol_size.z <= 0.0) {
        return 1.0;
    }

    let uvw = (pos - vol_min) / vol_size;

    // If outside the shadow volume, return fully lit.
    if (any(uvw < vec3(0.0)) || any(uvw > vec3(1.0))) {
        return 1.0;
    }

    // Edge fade: smoothstep from 0 at boundary → 1 at 15% inward, per axis.
    // Eliminates the hard rectangular brightness cutoff at shadow volume edges.
    let FADE = 0.15;
    let fade_x = smoothstep(0.0, FADE, uvw.x) * smoothstep(1.0, 1.0 - FADE, uvw.x);
    let fade_y = smoothstep(0.0, FADE, uvw.y) * smoothstep(1.0, 1.0 - FADE, uvw.y);
    let fade_z = smoothstep(0.0, FADE, uvw.z) * smoothstep(1.0, 1.0 - FADE, uvw.z);
    let edge_fade = fade_x * fade_y * fade_z;

    // Sample the 3D shadow map with trilinear filtering.
    let shadow_val = textureSampleLevel(vol_shadow_map, vol_shadow_smp, uvw, 0.0).r;
    // Blend from fully lit (1.0) at edges to actual shadow value in center.
    return mix(1.0, shadow_val, edge_fade);
}

// ---------------------------------------------------------------------------
// Sample cloud shadow map at a world position (vertical projection).
// Returns transmittance in [0, 1] (1 = fully lit, 0 = fully shadowed).
//
// The cloud shadow map is a 2D texture centered on the camera, covering
// `cloud_params.flags.y` metres of world XZ space. Positions outside the map
// smoothly fade to 1.0 (fully lit) to avoid hard cutoff.
// ---------------------------------------------------------------------------
fn sample_cloud_shadow(pos: vec3<f32>) -> f32 {
    return sample_cloud_shadow_at(pos.xz);
}

// ---------------------------------------------------------------------------
// Directional cloud shadow: traces from a fog position toward the sun to
// find where the ray enters the cloud layer, then looks up the shadow map
// at that XZ coordinate.  This produces correct shadow columns for low-angle
// sunlight, which is what creates visible god rays in volumetric fog.
//
// For positions inside or above the cloud layer, falls back to vertical.
// ---------------------------------------------------------------------------
fn sample_cloud_shadow_directional(pos: vec3<f32>) -> f32 {
    let coverage = cloud_params.flags.y;
    if (coverage <= 0.0 || cloud_params.flags.x < 0.5) { return 1.0; }

    let cloud_base = cloud_params.altitude.x;
    let dy = cloud_base - pos.y;

    var query_xz = pos.xz;

    // Only offset when below the cloud layer and sun has a horizontal component.
    let sun_horiz = length(params.sun_dir.xz);
    if (dy > 0.0 && sun_horiz > 0.001) {
        let sun_vert    = max(params.sun_dir.y, 0.001);
        let offset_dist = dy * sun_horiz / sun_vert;
        let offset_dir  = params.sun_dir.xz / sun_horiz;
        query_xz = pos.xz + offset_dir * offset_dist;
    }

    return sample_cloud_shadow_at(query_xz);
}

// ---------------------------------------------------------------------------
// Core cloud shadow lookup at an arbitrary XZ position.
// ---------------------------------------------------------------------------
fn sample_cloud_shadow_at(query_xz: vec2<f32>) -> f32 {
    let coverage = cloud_params.flags.y;
    if (coverage <= 0.0 || cloud_params.flags.x < 0.5) { return 1.0; }

    let cloud_uv = (query_xz - params.cam_pos.xz) / coverage + 0.5;

    // Edge fade: smoothstep from boundary to 10% inward.
    let FADE = 0.1;
    let fade_x = smoothstep(0.0, FADE, cloud_uv.x) * smoothstep(1.0, 1.0 - FADE, cloud_uv.x);
    let fade_y = smoothstep(0.0, FADE, cloud_uv.y) * smoothstep(1.0, 1.0 - FADE, cloud_uv.y);
    let edge_fade = fade_x * fade_y;

    // Out of bounds → fully lit.
    if (edge_fade <= 0.0) { return 1.0; }

    let transmittance = textureSampleLevel(cloud_shadow_tex, vol_shadow_smp, cloud_uv, 0.0).r;
    return mix(1.0, transmittance, edge_fade);
}

// ---------------------------------------------------------------------------
// Noise primitives for procedural clouds (ported from clouds.wgsl).
// ---------------------------------------------------------------------------

// Simple 3D hash for noise generation.
fn hash3(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Value noise 3D — tri-linearly interpolated hash lattice.
fn value_noise_3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 0.0)),
                hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)),
                hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)),
                hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)),
                hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// Fractal Brownian Motion — sums octaves of value noise.
fn fbm_3d(p: vec3<f32>, octaves: u32) -> f32 {
    var val: f32 = 0.0;
    var amp: f32 = 0.5;
    var freq: f32 = 1.0;
    var pos = p;
    for (var i = 0u; i < octaves; i++) {
        val += amp * value_noise_3d(pos * freq);
        freq *= 2.0;
        amp  *= 0.5;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Procedural cloud density (ported from clouds.wgsl).
//
// Evaluates FBM-based cloud density at a world position.
// Returns non-negative density; zero means no cloud.
// ---------------------------------------------------------------------------
fn cloud_density(pos: vec3<f32>) -> f32 {
    let cloud_min = cloud_params.altitude.x;
    let cloud_max = cloud_params.altitude.y;

    // Early out: outside altitude band or feature disabled.
    if (pos.y < cloud_min || pos.y > cloud_max) { return 0.0; }
    if (cloud_params.flags.x < 0.5)             { return 0.0; }

    // Height gradient: ramp up over bottom 50 m, fade out over top 200 m.
    // Absolute distances so the dense region stays near the base altitude
    // regardless of slab thickness.
    let height_above_base = pos.y - cloud_min;
    let height_below_top  = cloud_max - pos.y;
    let height_gradient = smoothstep(0.0, 50.0, height_above_base)
                        * smoothstep(0.0, 200.0, height_below_top);

    // Wind scrolling (XZ plane).
    let wind_dir    = vec2<f32>(cloud_params.wind.x, cloud_params.wind.y);
    let wind_offset = wind_dir * cloud_params.wind.z * cloud_params.wind.w;
    let scrolled    = pos + vec3<f32>(wind_offset.x, 0.0, wind_offset.y);

    // Large offset avoids hash3 zero-point at world origin.
    let noise_offset = vec3<f32>(173.5, 247.3, 391.7);

    // Shape noise — 4-octave FBM at low frequency.
    let shape_freq = cloud_params.noise.x;
    let shape      = fbm_3d(scrolled * shape_freq + noise_offset, 4u);

    // Weather modulation — 2-octave FBM tiled at weather_scale.
    let weather_scale = cloud_params.noise.w;
    let weather       = fbm_3d(
        vec3<f32>(scrolled.x / weather_scale, 0.0, scrolled.z / weather_scale) + noise_offset,
        2u
    );

    // Base density after threshold.
    let threshold = cloud_params.altitude.z;
    let base      = saturate(shape * weather * height_gradient - threshold);

    // Subtractive detail — 3-octave FBM erodes cloud edges.
    let detail_freq   = cloud_params.noise.y;
    let detail_weight = cloud_params.noise.z;
    let detail        = fbm_3d(scrolled * detail_freq + noise_offset, 3u);

    let density_scale = cloud_params.altitude.w;

    return max(0.0, base - detail * detail_weight) * density_scale;
}

// ---------------------------------------------------------------------------
// Height fog: exponential density falloff above the base height.
//
// fog_height.x = base_density
// fog_height.y = base_height  (world-space Y, metres)
// fog_height.z = height_falloff exponent
// ---------------------------------------------------------------------------
fn height_fog_density(pos: vec3<f32>) -> f32 {
    let base_density = params.fog_height.x;
    let base_height  = params.fog_height.y;
    let falloff      = params.fog_height.z;
    return base_density * exp(-falloff * max(pos.y - base_height, 0.0));
}

// ---------------------------------------------------------------------------
// Distance fog: density increases monotonically with camera distance t.
//
// fog_distance.x = distance_density scale
// fog_distance.y = distance_falloff exponent
// ---------------------------------------------------------------------------
fn distance_fog_density(t: f32) -> f32 {
    let density = params.fog_distance.x;
    let falloff  = params.fog_distance.y;
    return density * (1.0 - exp(-falloff * t));
}

// ---------------------------------------------------------------------------
// Sample fog and cloud density separately for different scattering treatment.
//
// Returns vec2(fog_density, cloud_density).
// Fog uses the configurable fog scattering color; clouds use white albedo.
// ---------------------------------------------------------------------------
fn sample_density_split(pos: vec3<f32>, t: f32) -> vec2<f32> {
    // Fog: ambient dust + height fog + distance fog.
    var fog: f32 = params.fog_distance.z;
    if (params.fog_color.w > 0.5) {
        fog += height_fog_density(pos);
    }
    if (params.fog_height.w > 0.5) {
        fog += distance_fog_density(t);
    }

    // Clouds: FBM noise within altitude band.
    let cloud = cloud_density(pos);

    return vec2(fog, cloud);
}

// ---------------------------------------------------------------------------
// Main compute entry point.
// Dispatched at half internal resolution (e.g. 480×270 for a 960×540 target).
// Workgroup: 8×8×1.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let pixel = vec2<f32>(f32(gid.x), f32(gid.y));
    let dims  = vec2<f32>(f32(params.width), f32(params.height));

    // UV in [0, 1]
    let uv  = (pixel + 0.5) / dims;
    // NDC in [-1, 1]
    let ndc = uv * 2.0 - 1.0;

    // Reconstruct view ray at half resolution using the same camera basis as
    // the full-res ray march pass.
    let ray_dir = normalize(
        params.cam_forward.xyz
        + ndc.x * params.cam_right.xyz
        - ndc.y * params.cam_up.xyz
    );

    // Sample full-res depth to get per-pixel scene distance.
    // Map half-res pixel index to the nearest full-res coordinate.
    let full_coord  = vec2<i32>(vec2<f32>(f32(gid.x), f32(gid.y)) * 2.0);
    let depth_value = textureLoad(depth_buffer, full_coord, 0).w;  // .w = distance
    // If depth is 0 (sky/miss), march to the far plane.
    let max_t = select(params.far, depth_value, depth_value > 0.0);

    // Jittered start offset keeps the first step from always landing at t=near.
    let jitter = interleaved_gradient_noise(pixel, params.frame_index);

    // Scattering angle cosine for phase function.
    // Standard HG convention: cos_theta = dot(incident, scattered).
    // incident = -sun_dir (light travels from sun), scattered = -ray_dir (toward camera).
    // cos_theta = dot(-sun_dir, -ray_dir) = dot(sun_dir, ray_dir).
    // Forward scattering (g>0) peaks at cos=1, i.e. looking toward the sun.
    let cos_sun = dot(ray_dir, params.sun_dir.xyz);

    // Fog scattering color acts as the single-scattering albedo.
    let scatter_albedo = params.fog_color.xyz;

    // Henyey-Greenstein asymmetry for ambient dust (fog_distance.w).
    let dust_g = params.fog_distance.w;

    var scatter:      vec3<f32> = vec3<f32>(0.0);
    var transmittance: f32      = 1.0;

    // Cloud scattering: white albedo, low asymmetry to approximate multi-scatter.
    // Real clouds scatter isotropically after many internal bounces.
    // Two-lobe: mild forward peak (silver lining) + strong isotropic base.
    let cloud_albedo = vec3<f32>(1.0, 1.0, 1.0);
    let cloud_g_forward: f32 = 0.6;  // forward lobe for silver lining
    let cloud_g_back: f32 = -0.2;    // slight back-scatter lobe
    let cloud_forward_weight: f32 = 0.3; // 30% directional, 70% isotropic-ish
    // Ambient sky light illuminates clouds from all directions (multi-scatter approx).
    let sky_ambient = vec3<f32>(params.vol_ambient_r, params.vol_ambient_g, params.vol_ambient_b);

    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if (t >= max_t) { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;

        let densities = sample_density_split(pos, t);
        let fog_dens   = densities.x;
        let cloud_dens = densities.y;
        let total      = fog_dens + cloud_dens;
        if (total <= 0.001) { continue; }

        // Beer-Lambert extinction over this step (total medium).
        let step_transmittance = exp(-total * params.step_size);

        // In-scattering: fog and clouds use different albedo and phase.
        // Combine object volumetric shadows with cloud shadow map.
        // Directional cloud shadow: traces toward the sun so cloud gaps
        // create proper light shafts (god rays) in the fog volume.
        let sun_vis = sample_vol_shadow(pos) * sample_cloud_shadow_directional(pos);
        let fog_in  = fog_dens * sun_vis * henyey_greenstein(cos_sun, dust_g)
                    * params.sun_color.xyz * scatter_albedo;
        // Two-lobe HG phase for clouds (multi-scatter approximation).
        let cloud_phase = mix(
            henyey_greenstein(cos_sun, cloud_g_back),
            henyey_greenstein(cos_sun, cloud_g_forward),
            cloud_forward_weight
        );
        let cloud_sun  = cloud_dens * sun_vis * cloud_phase * params.sun_color.xyz * cloud_albedo;
        let cloud_sky  = cloud_dens * sky_ambient * cloud_albedo;
        let cloud_in   = cloud_sun + cloud_sky;
        let in_scatter = fog_in + cloud_in;

        // Front-to-back accumulation.
        scatter      += in_scatter * transmittance * params.step_size;
        transmittance *= step_transmittance;

        // Early exit once transmittance is negligible.
        if (transmittance < 0.01) { break; }
    }

    // ── Dedicated cloud march ───────────────────────────────────────────────
    // The fog loop above covers 0..64 m (32 steps × 2 m).  Clouds typically
    // live at higher altitudes, so we intersect the cloud altitude slab and
    // march through it separately with its own step budget.
    if (cloud_params.flags.x > 0.5 && transmittance > 0.01) {
        let cloud_min = cloud_params.altitude.x;
        let cloud_max = cloud_params.altitude.y;
        let cam_y = params.cam_pos.y;
        let dir_y = ray_dir.y;

        // Ray–slab intersection.
        var cloud_t_enter: f32 = 0.0;
        var cloud_t_exit:  f32 = 0.0;
        var cloud_valid = false;

        if (abs(dir_y) > 1e-5) {
            let t_lo = (cloud_min - cam_y) / dir_y;
            let t_hi = (cloud_max - cam_y) / dir_y;
            cloud_t_enter = max(min(t_lo, t_hi), 0.0);
            cloud_t_exit  = max(max(t_lo, t_hi), 0.0);
            cloud_valid   = cloud_t_exit > cloud_t_enter;
        } else if (cam_y >= cloud_min && cam_y <= cloud_max) {
            // Ray is horizontal and camera is inside the cloud band.
            cloud_t_enter = 0.0;
            cloud_t_exit  = 100000.0;
            cloud_valid   = true;
        }

        if (cloud_valid) {
            // Clamp to scene depth; sky pixels (depth=0) get full cloud range.
            // No artificial distance cap — the slab intersection already limits
            // the march to the cloud volume, and transmittance drops to zero
            // long before all steps complete for distant clouds.
            let cloud_cap = select(100000.0, max_t, depth_value > 0.0);
            let t_end = min(cloud_t_exit, cloud_cap);
            let slab  = t_end - cloud_t_enter;

            if (slab > 0.0) {
                // March through at most 6 km of the slab — the height gradient
                // concentrates density in the 10-60% height range, so anything
                // beyond ~6 km is negligible.  48 steps keeps the per-step
                // spacing ≤ 125 m, avoiding FBM aliasing / visible banding.
                let cloud_steps = 48u;
                let effective_slab = min(slab, 6000.0);
                let cloud_step = effective_slab / f32(cloud_steps);

                // Temporally stable jitter — no frame_index, so the offset
                // is fixed per pixel. Prevents flicker when steps are large.
                let cloud_jitter = interleaved_gradient_noise(pixel, 0u);

                for (var ci = 0u; ci < cloud_steps; ci++) {
                    let t = cloud_t_enter + (f32(ci) + cloud_jitter) * cloud_step;
                    let pos = params.cam_pos.xyz + ray_dir * t;

                    var cd = cloud_density(pos);
                    if (cd <= 0.001) { continue; }

                    // Fade density near the camera so being inside the cloud
                    // band shows wisps rather than a solid gray wall.
                    cd *= smoothstep(0.0, 150.0, t);

                    let step_tr = exp(-cd * cloud_step);
                    let sun_vis = sample_cloud_shadow(pos);
                    let cloud_phase = mix(
                        henyey_greenstein(cos_sun, cloud_g_back),
                        henyey_greenstein(cos_sun, cloud_g_forward),
                        cloud_forward_weight
                    );
                    let cloud_sun = cd * sun_vis * cloud_phase * params.sun_color.xyz * cloud_albedo;
                    let cloud_sky = cd * sky_ambient * cloud_albedo;

                    scatter      += (cloud_sun + cloud_sky) * transmittance * cloud_step;
                    transmittance *= step_tr;
                    if (transmittance < 0.01) { break; }
                }
            }
        }
    }

    textureStore(output_scatter, vec2<i32>(gid.xy), vec4<f32>(scatter, transmittance));
}
