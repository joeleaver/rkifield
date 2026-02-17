# Volumetric Effects

> **Status: DECIDED**

### Decision: Separate Volumetric Pass at Half Resolution

**Chosen over:** Unified march (complicates primary ray march shader, can't run at different resolution, conflicting step-size requirements).

Volumetrics run as a dedicated compute pass after shading at half internal resolution (480×270), then bilateral-upscaled to internal resolution (960×540) before compositing. This gives a 4× cost reduction on a naturally low-frequency effect.

**Core algorithm — front-to-back compositing with fixed step size:**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn volumetric_march(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let ray = camera_ray(pixel);
    let max_t = depth_buffer[pixel * 2];  // sample full-res depth
    var color = vec3f(0.0);
    var transmittance = 1.0;

    // Jittered step offset for temporal accumulation
    let jitter = interleaved_gradient_noise(pixel, frame_index);

    for (var t = NEAR + jitter * STEP_SIZE; t < max_t; t += STEP_SIZE) {
        let pos = ray.origin + ray.dir * t;

        // Density from all sources
        var density = 0.0;
        density += sample_brick_density(pos);       // local fog volumes + brick-backed clouds
        density += height_fog_density(pos);          // global height fog
        density += cloud_density(pos);               // procedural high-altitude clouds
        density += ambient_dust_density;             // constant dust (for god rays)

        if density <= 0.001 { continue; }

        // Extinction
        let step_transmittance = exp(-density * extinction_coeff * STEP_SIZE);

        // In-scattering: sun (volumetric shadow map) + secondary lights (radiance opacity)
        var scatter = vec3f(0.0);
        let sun_vis = sample_volumetric_shadow(pos);
        let sun_phase = henyey_greenstein(dot(-ray.dir, sun_dir), phase_g);
        scatter += density * sun_vis * sun_phase * sun_color * sun_intensity;

        for (var i = 0u; i < vol_light_count; i++) {
            let light = vol_lights[i];
            let vis = sample_radiance_opacity(pos, light.position);
            let phase = henyey_greenstein(dot(-ray.dir, normalize(light.position - pos)), phase_g);
            scatter += density * vis * phase * light.color * distance_attenuation(light, pos);
        }

        // Emission (fire, glow)
        let emission = sample_brick_emission(pos);

        // Accumulate front-to-back
        color += (scatter + emission) * transmittance * STEP_SIZE;
        transmittance *= step_transmittance;
        if transmittance < 0.01 { break; }
    }

    scattering_buffer[pixel] = vec4f(color, transmittance);
}
```

**Compositing:** `final_color = shaded_color * vol_transmittance + vol_color`

### Decision: God Rays — Emergent from Participating Media

God rays are not a separate system. They emerge when participating media (even thin ambient dust) exists and the sun partially illuminates the medium through gaps in geometry. The per-step shadow check (via volumetric shadow map) creates the visible light shafts.

For god rays to appear, you need:
- `ambient_dust_density > 0` (very low — 0.001-0.01 range)
- A directional light (sun)
- Geometry that partially occludes the light

Quality driven by volumetric shadow map resolution, phase function `g` (higher = brighter forward glow), and dust density.

### Decision: Volumetric Shadow Map

Pre-computed per frame for the primary directional light:

```
Volumetric Shadow Map (compute, once per frame):
  - 3D texture, 256×256×128 (configurable)
  - Covers camera-centered volume
  - Each texel: transmittance from sun to that point
  - Computed by marching SDF from each texel toward sun
```

All volumetric steps sample this texture for sun visibility — one texture read vs a full shadow ray per step.

| Shadow Approach | Per-Step Cost | Quality |
|----------------|--------------|---------|
| SDF shadow ray | ~32-64 SDF evals | Exact |
| Volumetric shadow map | 1 texture sample | Very good |
| Radiance volume opacity | 1 texture sample | Approximate |

Sun uses the shadow map. Secondary lights use radiance volume opacity.

**Upgrade path — Cascaded volumetric shadow:**
Multiple shadow volumes at increasing resolution near the camera (like cascaded shadow maps). Sharper nearby shafts, softer distant ones. Only needed if single volume resolution causes visible banding.

### Decision: Local Fog Volumes (Brick-Backed)

Uses the volumetric companion pool from [Core Data Structure](./01-core-data-structure.md):
```
Volumetric Data Pool:
  Word 0 (u32): [ f16 density | f16 emission_intensity ]
  2KB per brick, ~64MB budget → ~32K bricks
```

Fog volumes are regions with occupied volumetric bricks. Placed by level designers or spawned dynamically:
- Cave fog: static bricks, low density, no emission
- Mist over a lake: slowly animated (noise-modulated density per frame)
- Torch glow: emission-only bricks near light sources
- Smoke/fire: dynamic bricks updated by fluid sim or procedural noise

Material's albedo serves as scattering color — a green-tinted fog material scatters green light.

### Decision: Global Fog (Analytic)

Scene-wide atmospheric effects without brick storage:

**Height fog:**
```wgsl
fn height_fog_density(pos: vec3f) -> f32 {
    return fog_base_density * exp(-fog_height_falloff * max(pos.y - fog_base_height, 0.0));
}
```

**Distance fog:**
```wgsl
fn distance_fog_density(t: f32) -> f32 {
    return fog_distance_density * (1.0 - exp(-fog_distance_falloff * t));
}
```

Configurable per-scene:
```rust
struct FogSettings {
    height_fog_enabled: bool,
    fog_base_density: f32,
    fog_base_height: f32,
    fog_height_falloff: f32,
    fog_color: [f32; 3],

    distance_fog_enabled: bool,
    fog_distance_density: f32,
    fog_distance_falloff: f32,

    ambient_dust_density: f32,   // for god rays (very low)
    ambient_dust_g: f32,         // phase asymmetry for dust
}
```

### Decision: Dynamic Volumes — Smoke and Fire

**Source A: GPU Fluid Simulation (hero effects)**
```
Eulerian fluid sim on GPU per active effect:
  1. Advect density by velocity field
  2. Apply forces (buoyancy, wind, turbulence)
  3. Diffuse (viscosity)
  4. Write density + temperature → volumetric bricks

Temperature → emission: intensity = max(0, temperature - ignition) * scale
```

Per-effect simulation grid (32³ or 64³) centered on the emitter. Multiple simultaneous effects each get their own grid.

**Source B: Procedural Noise Animation (ambient effects)**
```wgsl
fn procedural_smoke(pos: vec3f, time: f32, emitter: Emitter) -> f32 {
    let scroll = pos + vec3(wind.x * time, -rise_speed * time, wind.z * time);
    let n = fbm_noise(scroll * frequency, octaves);
    let falloff = 1.0 - smoothstep(0.0, emitter.radius, distance(pos, emitter.position));
    return max(0.0, n - threshold) * falloff * base_density;
}
```

No simulation state, evaluated inline during volumetric march. Good for distant smoke, ambient haze, wisps.

Both sources feed the same volumetric brick pool. The rendering path doesn't care where density came from.

**Upgrade path — Full Navier-Stokes solver:**
Replace simplified Eulerian sim with full incompressible N-S solver (pressure projection, proper boundary conditions). Better physical accuracy for large-scale smoke plumes.

### Decision: Clouds — Dual System (Procedural + Brick-Backed)

Two cloud systems that complement each other:

**High-altitude procedural clouds (evaluated analytically):**
```wgsl
fn cloud_density(pos: vec3f) -> f32 {
    if pos.y < cloud_min || pos.y > cloud_max { return 0.0; }

    let height_frac = (pos.y - cloud_min) / (cloud_max - cloud_min);
    let height_gradient = smoothstep(0.0, 0.1, height_frac)
                        * smoothstep(1.0, 0.6, height_frac);

    let weather = textureSample(weather_map, weather_sampler, pos.xz / weather_scale).r;
    let shape = fbm_noise_3d(pos * shape_frequency, 4);
    let detail = fbm_noise_3d(pos * detail_frequency, 3);

    let base = saturate(shape * weather * height_gradient - cloud_threshold);
    return max(0.0, base - detail * detail_weight) * cloud_density_scale;
}
```

No bricks needed. Scrolled by wind over time. Controlled by weather map texture for spatial variation.

**Low-altitude brick-backed clouds (terrain interaction):**

For dense fog banks, low clouds, and mist that wrap around mountains and flow through valleys — stored in the volumetric brick pool. These clouds:
- Interact with terrain geometry (flow around hills, pool in valleys)
- Are lit and shadowed correctly by the volumetric shadow map
- Can be placed by level designers or generated procedurally and baked to bricks
- Use the same density/emission format as local fog volumes

Brick-backed clouds enable effects impossible with analytic evaluation: fog rolling over a cliff edge, mist threading through a forest, low clouds parting around a mountain peak.

**Cloud shadows on terrain:**

A 2D cloud shadow map is rendered each frame by projecting cloud density (both procedural and brick-backed) downward from the sun's direction:

```
Cloud Shadow Map (compute, once per frame):
  - 2D texture, 1024×1024 (covers large area around camera)
  - For each texel: project upward from terrain into the cloud layer
  - Sample cloud density along sun direction
  - Store transmittance (0 = fully shadowed, 1 = fully lit)
```

The shading pass samples this map for terrain and outdoor surfaces:
```wgsl
let cloud_shadow = textureSample(cloud_shadow_map, shadow_sampler, world_pos.xz / shadow_scale).r;
direct_light *= cloud_shadow;
```

This creates moving shadow patterns on the ground as clouds drift — a dramatic visual with minimal cost.

**Cloud parameters:**
```rust
struct CloudSettings {
    // High-altitude procedural
    procedural_enabled: bool,
    cloud_min: f32,              // lower altitude (e.g., 1000m)
    cloud_max: f32,              // upper altitude (e.g., 3000m)
    cloud_threshold: f32,        // coverage control
    cloud_density_scale: f32,
    shape_frequency: f32,
    detail_frequency: f32,
    detail_weight: f32,
    weather_scale: f32,
    wind_direction: [f32; 2],
    wind_speed: f32,

    // Low-altitude brick-backed
    brick_clouds_enabled: bool,  // uses volumetric brick pool

    // Cloud shadows
    shadow_enabled: bool,
    shadow_map_resolution: u32,  // default 1024
    shadow_coverage: f32,        // world-space extent of shadow map
}
```

### Decision: Phase Function — Henyey-Greenstein, Per-Volume

```wgsl
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}
```

| g | Effect | Use case |
|---|--------|----------|
| 0.0 | Isotropic | Thin fog, generic haze |
| 0.3 | Mild forward scatter | Ground fog, dust |
| 0.6 | Strong forward scatter | Clouds, god ray glow |
| 0.8 | Very forward | Dense clouds facing sun |
| -0.2 | Back scatter | Snow, retroreflective media |

Per-volume configurable `g` parameter. Global fog, local fog volumes, and clouds each have their own setting.

**Upgrade path — Dual-lobe (Cornette-Shanks):**
Blend two HG lobes for more realistic cloud scattering: `0.7 * HG(g=0.6) + 0.3 * HG(g=-0.3)`. Brighter silver lining + gentle backlit glow.

### Decision: Temporal Reprojection for Volumetrics

Reuse previous frame's volumetric results with temporal blending:

```
Per frame:
  1. Reproject previous volumetric buffer using motion vectors
  2. Validate (depth consistency, disocclusion check)
  3. Valid: result = lerp(current, history, 0.9)  // 90% history
  4. Invalid: result = current only
```

Jittered step offset (different per frame) + temporal accumulation smooths stepping artifacts over time.

| Setting | Steps/ray | Effective samples (with temporal) |
|---------|-----------|----------------------------------|
| Low | 16 | ~160 over 10 frames |
| Medium | 32 | ~320 over 10 frames |
| High | 64 | ~640 over 10 frames |

### Decision: Bilateral Upscale for Volumetric Buffer

Edge-aware upsampling from 480×270 to 960×540, guided by full-res depth buffer:

```wgsl
fn bilateral_upscale(pixel: vec2u) -> vec4f {
    // Sample 4 nearest low-res volumetric pixels
    // Weight each by depth similarity to full-res depth at this pixel
    // Depth-similar neighbors contribute more → clean silhouette edges
}
```

Prevents fog from bleeding across sharp depth discontinuities (character silhouettes, object edges).

### Complete Volumetric Pipeline Per Frame

```
Pre-compute:
  a. Volumetric shadow map from sun (compute, ~256³ 3D texture)
  b. Cloud shadow map from sun (compute, ~1024² 2D texture)
  c. Update fluid sim grids for active effects (compute, per effect)
  d. Write sim results to volumetric brick pool

Pass 6a: VOLUMETRIC MARCH (compute, 480×270)
  - Fixed-step march with jittered offset
  - Sample density: bricks + height fog + distance fog + procedural clouds + ambient dust
  - In-scattering: sun (shadow map) + secondary lights (radiance opacity)
  - Emission: brick pool
  - Phase function: per-volume HG
  - Temporal blend with reprojected history

Pass 6b: VOLUMETRIC UPSCALE (compute, 480×270 → 960×540)
  - Bilateral upscale guided by full-res depth

Compositing (in post-process):
  final = shaded_color * vol_transmittance + vol_color
```

### Session 6 Summary: All Volumetric Effects Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| Pass architecture | Separate half-res pass (480×270) | — |
| God rays | Emergent from participating media + shadow map | — |
| Volumetric shadows | 3D shadow map (sun) + radiance opacity (secondary) | Cascaded shadow volumes |
| Local fog | Brick-backed (volumetric companion pool) | — |
| Global fog | Analytic height fog + distance fog | — |
| Dynamic volumes | GPU fluid sim (hero) + procedural noise (ambient) | Full N-S solver |
| Clouds (high) | Procedural noise + weather map | Dual-lobe phase function |
| Clouds (low) | Brick-backed, terrain interaction | — |
| Cloud shadows | 2D shadow map projected from sun | — |
| Phase function | Henyey-Greenstein, per-volume g | Dual-lobe Cornette-Shanks |
| Temporal | Reprojection with 90% history blend | — |
| Upscale | Bilateral, depth-guided | — |
