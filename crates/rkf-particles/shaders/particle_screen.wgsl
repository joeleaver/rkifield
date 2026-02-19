// Screen-space particle composite shader.
//
// Full-screen compute dispatch that composites screen-space particles
// (rain, snow, dust) over the shaded image. Each pixel iterates all
// particles and accumulates the closest visible ones with alpha blending.
//
// This naive per-pixel iteration is a correctness placeholder. Production
// would use tile binning for O(particles_per_tile) instead of O(N).

struct ScreenParticle {
    screen_pos: vec2<f32>,
    depth: f32,
    size: f32,
    color: vec3<f32>,
    alpha: f32,
    velocity_screen: vec2<f32>,
    emission: f32,
    _pad: f32,
}

struct ScreenParticleParams {
    particle_count: u32,
    screen_width: f32,
    screen_height: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> screen_particles: array<ScreenParticle>;
@group(0) @binding(1) var<uniform> params: ScreenParticleParams;
@group(0) @binding(2) var depth_buffer: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, read_write>;

// ---- Helpers ----

/// Elongate distance along velocity direction for streak shapes.
/// Returns the anisotropic distance from pixel to particle center,
/// compressed along the velocity axis to create a streak effect.
fn streak_distance(
    pixel: vec2<f32>,
    center: vec2<f32>,
    velocity: vec2<f32>,
    base_size: f32,
) -> f32 {
    let delta = pixel - center;
    let speed = length(velocity);

    if speed < 0.1 {
        // No significant velocity -- circular dot.
        return length(delta);
    }

    let dir = velocity / speed;
    let perp = vec2<f32>(-dir.y, dir.x);

    // Project delta onto streak direction and perpendicular.
    let along = abs(dot(delta, dir));
    let across = abs(dot(delta, perp));

    // Stretch along velocity: effective length is base_size + speed * streak_factor.
    let streak_factor = 0.02;
    let streak_len = base_size + speed * streak_factor;

    // Ellipse-like distance: along axis stretched, across axis at base size.
    let norm_along = along / max(streak_len, 0.001);
    let norm_across = across / max(base_size, 0.001);

    return length(vec2<f32>(norm_along, norm_across)) * base_size;
}

// ---- Main compute entry point ----

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(output));

    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    // Read existing color from output (for blending).
    let existing = textureLoad(output, pixel);

    // Pixel center in screen [0,1] space.
    let pixel_uv = vec2<f32>(
        (f32(pixel.x) + 0.5) / params.screen_width,
        (f32(pixel.y) + 0.5) / params.screen_height,
    );

    // Pixel center in pixel coordinates.
    let pixel_pos = vec2<f32>(f32(pixel.x) + 0.5, f32(pixel.y) + 0.5);

    // Read scene depth at this pixel (linear depth).
    let scene_depth = textureLoad(depth_buffer, pixel, 0).r;

    var accum_color = vec3<f32>(0.0, 0.0, 0.0);
    var accum_alpha = 0.0;

    for (var i = 0u; i < params.particle_count; i++) {
        let sp = screen_particles[i];

        // Particle center in pixel coordinates.
        let center = vec2<f32>(
            sp.screen_pos.x * params.screen_width,
            sp.screen_pos.y * params.screen_height,
        );

        // Quick bounding box reject (max possible extent).
        let max_extent = sp.size * 3.0; // streak can be up to 3x base size
        if abs(pixel_pos.x - center.x) > max_extent || abs(pixel_pos.y - center.y) > max_extent {
            continue;
        }

        // Compute anisotropic streak distance.
        let d = streak_distance(pixel_pos, center, sp.velocity_screen, sp.size);

        if d >= sp.size {
            continue;
        }

        // Depth test: particle must be in front of scene geometry.
        // Allow a small bias so particles on surfaces still show.
        let depth_bias = 0.05;
        if sp.depth > scene_depth + depth_bias {
            continue;
        }

        // Soft falloff within particle radius.
        let falloff = 1.0 - (d / sp.size);
        let contrib_alpha = sp.alpha * falloff;

        // Accumulate with front-to-back alpha compositing.
        let color_contrib = sp.color + sp.color * sp.emission;
        accum_color += color_contrib * contrib_alpha * (1.0 - accum_alpha);
        accum_alpha += contrib_alpha * (1.0 - accum_alpha);

        // Early out if nearly opaque.
        if accum_alpha > 0.95 {
            break;
        }
    }

    // Blend accumulated particles over existing image.
    let final_color = existing.rgb * (1.0 - accum_alpha) + accum_color;
    let final_alpha = existing.a;

    textureStore(output, pixel, vec4<f32>(final_color, final_alpha));
}
