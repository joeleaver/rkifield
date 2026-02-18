// Volumetric march compute shader (half resolution) — Phase 11 task 11.2.
//
// Front-to-back fixed-step compositing through participating media.
// Jittered step offset (interleaved gradient noise) for temporal accumulation.
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
    // Fog / dust settings (placeholder — expanded in tasks 11.4, 11.5)
    ambient_dust_density: f32,
    ambient_dust_g:       f32,  // Henyey-Greenstein asymmetry for dust
    frame_index:          u32,
    _pad0:                u32,
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

    // Sample the 3D shadow map with trilinear filtering.
    return textureSampleLevel(vol_shadow_map, vol_shadow_smp, uvw, 0.0).r;
}

// ---------------------------------------------------------------------------
// Sample total extinction density at a world position.
// Placeholder: returns only the uniform ambient dust density.
// Individual density sources (height fog, cloud SDF) are added in tasks 11.4+.
// ---------------------------------------------------------------------------
fn sample_density(pos: vec3<f32>) -> f32 {
    let _ = pos;
    return params.ambient_dust_density;
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

    // Angle between view ray and sun direction (for phase function).
    // sun_dir points *toward* the sun, ray_dir points *away* from camera.
    let cos_sun = dot(-ray_dir, params.sun_dir.xyz);

    var scatter:      vec3<f32> = vec3<f32>(0.0);
    var transmittance: f32      = 1.0;

    for (var i = 0u; i < params.max_steps; i++) {
        let t = params.near + (f32(i) + jitter) * params.step_size;
        if (t >= max_t) { break; }

        let pos = params.cam_pos.xyz + ray_dir * t;

        let density = sample_density(pos);
        if (density <= 0.001) { continue; }

        // Beer-Lambert extinction over this step.
        let step_transmittance = exp(-density * params.step_size);

        // In-scattering from the sun: visibility × phase × sun radiance.
        let sun_vis   = sample_vol_shadow(pos);
        let sun_phase = henyey_greenstein(cos_sun, params.ambient_dust_g);
        let in_scatter = density * sun_vis * sun_phase * params.sun_color.xyz;

        // Front-to-back accumulation.
        scatter      += in_scatter * transmittance * params.step_size;
        transmittance *= step_transmittance;

        // Early exit once transmittance is negligible.
        if (transmittance < 0.01) { break; }
    }

    textureStore(output_scatter, vec2<i32>(gid.xy), vec4<f32>(scatter, transmittance));
}
