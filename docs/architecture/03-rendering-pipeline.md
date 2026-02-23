> **SUPERSEDED** by [v2 Architecture](../v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# Rendering Pipeline

> **Status: DECIDED**

All rendering is done via compute shaders. No rasterization pipeline is used.

### Full Pipeline Overview

```
Pass 1: RAY MARCH (compute)                    ← Session 3 — DECIDED
Pass 2: TILE LIGHT CULLING (compute)            ← Session 4 — DECIDED
Pass 3: RADIANCE INJECTION (compute)            ← Session 4 — DECIDED
Pass 4: RADIANCE MIP GENERATION (compute, ×3)   ← Session 4 — DECIDED
Pass 5: SHADE (compute, merged material+light)   ← Session 4 — DECIDED
Pass 6: VOLUMETRICS (compute)                    ← Session 6
Pass 7: POST-PROCESS (compute, low-res stack)    ← Session 9
Pass 8: UPSCALE (compute)                        ← Session 9
Pass 9: POST-PROCESS (compute, final-res stack)  ← Session 9
```

---

### Pass 1: Ray March — Detailed Design

#### Decision: Compute Dispatch — One Thread Per Pixel, Flat Dispatch

```wgsl
@compute @workgroup_size(8, 8, 1)
fn ray_march(@builtin(global_invocation_id) pixel: vec3<u32>) { ... }

// Dispatch: ceil(width/8) × ceil(height/8) × 1
// At 960×540: 120×68 = 8,160 workgroups, 522,240 threads
```

Each thread independently generates and marches one camera ray. Simple and correct baseline.

**Upgrade path — Tile-based with shared memory:**
If profiling shows excessive redundant brick reads among neighboring pixels, upgrade to cooperative tile loading. Threads in a workgroup pre-load brick data for their region of interest into shared memory before marching. Benefits depend on ray coherence (good near camera, poor at grazing angles). Implementation: shared memory array sized for expected brick overlap, barrier sync after load, then individual marching against shared data.

**Upgrade path — Early-out compaction:**
If warp divergence is severe (some rays finish in 5 steps, others take 200), add a compaction pass. First dispatch: all pixels march for N steps. Pixels that resolve write results and mark done. Compaction pass generates a new dispatch list of unresolved pixels. Repeat. Eliminates idle threads waiting for long-marching neighbors. Cost: multiple dispatches + indirect dispatch buffer management.

#### Decision: Ray Generation — Pinhole Camera with Sub-Pixel Jitter

```wgsl
let jitter = halton_2d(frame_index);  // sub-pixel offset, changes per frame
let uv = (vec2<f32>(pixel.xy) + jitter) / resolution;
let ndc = uv * 2.0 - 1.0;
let ray_origin = camera.position;
let ray_direction = normalize(camera.forward + ndc.x * camera.right + ndc.y * camera.up);
```

Sub-pixel jitter (Halton sequence) gives the temporal upscaler different sample positions each frame, improving effective resolution through accumulation. The jitter offset is passed as a uniform alongside the camera matrices.

#### Decision: March Algorithm — Multi-Level DDA + Sphere Tracing

Two-phase traversal exploiting the multi-level sparse grid:

**Phase 1: DDA through spatial index (empty-space skipping)**
```
DDA through Level 2 (coarse occupancy bitfield):
  → Large steps through empty regions, one per coarse cell
  → On hitting occupied Level 2 cell:

    DDA through Level 1 (block index) within that cell:
      → Medium steps through empty blocks
      → On hitting occupied Level 1 cell (brick exists):
        → Enter Phase 2
```

**Phase 2: Sphere tracing within brick**
```
While ray is inside this brick:
  p = ray_origin + ray_direction * t
  d = trilinear_sample_brick(brick, local_position(p))
  if d < EPSILON:
    return hit(p, t, brick)
  t += d
  if ray exits brick bounds:
    return to Phase 1 (continue DDA)
```

Most ray travel is through empty space, handled by cheap DDA cell-stepping. Expensive SDF evaluation only happens near surfaces inside occupied bricks. The multi-level grid makes this efficient: Level 2 skips entire 32m+ regions, Level 1 skips individual brick-sized blocks.

**LOD transitions during march:** When the ray's `t` value crosses a clipmap boundary, the marcher switches to the next coarser LOD level's spatial index. Step sizes naturally increase in coarser levels (larger voxels → larger safe distances), making distant marching cheaper.

**Upgrade path — Relaxed sphere tracing:**
Standard sphere tracing steps by exactly `d` (the SDF value). Relaxed sphere tracing (Keinert et al. 2014) steps by `d * relaxation_factor` (e.g., 1.6×), overshooting slightly, then validates. Reduces total step count by ~40% for most scenes. Requires validation step to catch overshoots — if `sdf(new_pos) > new_pos - old_pos`, we overshot and must backtrack. Safe with Lipschitz-1 SDFs; may need smaller relaxation factor for animated/warped SDFs that violate Lipschitz.

**Upgrade path — Cone tracing for LOD:**
Instead of point-sampling the SDF, trace a cone that widens with distance. At each step, compare the cone radius against the SDF value. This naturally selects the appropriate LOD — wide cone at distance evaluates coarser bricks. More complex step logic but better LOD integration. Relevant if LOD transitions produce visible popping.

#### Decision: Brick Sampling — Trilinear Distance, Nearest-Neighbor Material

**Distance field:** Trilinear interpolation from the 8 surrounding voxels within the brick. Produces smooth surfaces without staircase artifacts. Cost: 8 voxel reads + 7 lerps per evaluation.

**Material IDs:** Nearest-neighbor (closest voxel). Material IDs are categorical — interpolating between material 5 and material 12 is meaningless. The hit point's material is the nearest voxel's material.

**Brick boundary handling:** Clamp at brick edges. The SDF is continuous by construction — values at shared boundaries between adjacent bricks should already agree. No ghost voxels (would nearly double brick memory from 512 to 1000 samples) and no neighbor lookups (would add branching and indirection).

**Upgrade path — Neighbor lookup at boundaries:**
If clamping produces visible seams between bricks (unlikely for well-constructed SDFs, but possible after editing/deformation), add neighbor brick references. Each brick stores 6 neighbor IDs (±x, ±y, ±z). When trilinear sampling within 1 voxel of a brick face, fetch the needed samples from the neighbor brick. Branchy but geometrically correct. Only needed for pathological cases.

**Upgrade path — Tricubic interpolation:**
If trilinear normals are too faceted (visible flat-shading on curved surfaces), upgrade to tricubic. Reads 64 surrounding voxels, produces C1-continuous surfaces with smooth normals. Very expensive — only consider if normal quality is visibly lacking after procedural noise perturbation is applied.

#### Decision: Normal Computation — Central Differences

Surface normals computed from the SDF gradient at the hit point:

```wgsl
let eps = voxel_size * 1.5;  // step size: 1.5× current tier's voxel width
let normal = normalize(vec3<f32>(
    sample_sdf(p + vec3(eps, 0, 0)) - sample_sdf(p - vec3(eps, 0, 0)),
    sample_sdf(p + vec3(0, eps, 0)) - sample_sdf(p - vec3(0, eps, 0)),
    sample_sdf(p + vec3(0, 0, eps)) - sample_sdf(p - vec3(0, 0, eps))
));
```

6 additional SDF evaluations at the hit point. Since the hit is inside a brick, all 6 samples are likely within the same brick or immediate neighbors — cheap trilinear lookups, no spatial index traversal.

Step size of 1.5× voxel width balances detail capture (not too large) against f16 numerical stability (not too small).

**Upgrade path — Tetrahedron gradient:**
Reduce from 6 to 4 SDF evaluations using a tetrahedral sampling pattern. Slightly noisier normals but ~33% fewer samples. Worth considering if normal computation becomes a bottleneck.

#### Decision: G-Buffer Output — Four Render Targets, Unpacked

The ray march writes per-pixel results to a G-buffer at internal resolution (960×540). The shading pass reads these.

| Target | Format | Content | Bytes/px |
|--------|--------|---------|----------|
| 0 | RGBA32Float | `world_position.xyz`, `hit_distance` | 16 |
| 1 | RGBA16Float | `normal.xyz`, `material_blend_weight` | 8 |
| 2 | RG16Uint | `material_id` (u16), `secondary_id_and_flags` (u16) | 4 |
| 3 | RG16Float | `motion_vector.xy` | 4 |

- **Total: 32 bytes/pixel × 518,400 pixels = ~16MB** — trivial
- **Sky/miss pixels:** Encoded as `hit_distance = MAX_FLOAT`. Shading pass skips them.
- **Motion vectors:** Computed by reprojecting the hit position through the previous frame's view-projection matrix. One matrix multiply per pixel, critical for temporal upscaling.
- `secondary_id_and_flags` packs u8 secondary_id in lower 8 bits, flags (has_color, has_bone, etc.) in upper 8 bits.

**Upgrade path — Packed two-target G-buffer:**
Combine into 2 targets by bit-packing normals (octahedral encoding, 2×16 bits), material IDs, and motion vectors. Saves ~8MB and halves the number of texture bindings in the shading pass. Only worth it if G-buffer bandwidth becomes a bottleneck — unlikely at 960×540.

**Upgrade path — Visibility buffer:**
Instead of a full G-buffer, store only (brick_id, local_position) per pixel. The shading pass re-evaluates everything from the voxel data. Eliminates G-buffer entirely but doubles SDF evaluation cost. Only useful if G-buffer memory or bandwidth is critical — not expected at our target resolution.

#### Ray March Constants and Limits

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MAX_STEPS` | 256 | Maximum march iterations per ray |
| `MAX_DISTANCE` | 2048m | Matches furthest clipmap level |
| `EPSILON` | 0.0005 | Surface hit threshold (half of finest voxel size) |
| `NORMAL_EPS` | voxel_size × 1.5 | Central difference step for gradient |
| Internal resolution | 960×540 | 1/4 of 1080p, 1/16 of 4K |

These are configurable at runtime for quality/performance tuning.

### Session 3 Summary: All Ray March Pipeline Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| Dispatch | One thread per pixel, 8×8 workgroups | Tile-based shared memory; early-out compaction |
| Ray generation | Pinhole camera + Halton sub-pixel jitter | — |
| March algorithm | Multi-level DDA + sphere tracing in bricks | Relaxed sphere tracing; cone tracing for LOD |
| Brick sampling | Trilinear distance, nearest-neighbor material | Neighbor lookup at boundaries; tricubic interpolation |
| Boundary handling | Clamp at brick edges | Neighbor brick references |
| Normals | Central differences, 6 samples, 1.5× voxel step | Tetrahedron gradient (4 samples) |
| G-buffer | 4 targets, 32 bytes/pixel, ~16MB at 960×540 | Packed 2-target; visibility buffer |

---

### Passes 2–5: Shading Pipeline — Detailed Design

#### Decision: PBR Shading Model — Cook-Torrance GGX

Standard microfacet BRDF used by every modern production engine (UE5, Unity, Filament):

```
Diffuse:  Lambertian (albedo / PI) × (1 - metallic)
Specular: Cook-Torrance (D_GGX × F_Schlick × G_Smith) / (4 × NdotL × NdotV)
```

Roughness drives GGX distribution width. Metallic blends between dielectric and conductor Fresnel. Our material struct was designed for this model — direct mapping, no adaptation needed.

#### Decision: Shadows — SDF Soft Shadows

SDF soft shadows computed inline during shading. No shadow maps, no aliasing, no resolution limits.

```wgsl
fn soft_shadow(origin: vec3f, light_dir: vec3f, max_dist: f32, k: f32) -> f32 {
    var shadow = 1.0;
    var t = 0.01;
    for (var i = 0; i < MAX_SHADOW_STEPS; i++) {
        let d = evaluate_sdf(origin + light_dir * t);
        if d < EPSILON { return 0.0; }
        shadow = min(shadow, k * d / t);  // k controls penumbra softness
        t += d;
        if t > max_dist { break; }
    }
    return clamp(shadow, 0.0, 1.0);
}
```

The ratio `d/t` naturally produces physically plausible penumbrae — nearby occluders cast hard shadows, distant occluders cast soft ones. Parameter `k` tunes overall softness (higher = sharper, typical range 8-64).

Shadow rays use relaxed settings vs primary rays:
- `MAX_SHADOW_STEPS`: 64 (vs 256 for primary)
- Larger epsilon (only need occlusion, not exact hit)
- No material/normal computation

**Upgrade path — Improved penumbra (Quilez 2018):**
The basic `k * d / t` formula can produce banding in penumbrae. Quilez's improved formula tracks the previous step's distance for smoother results: `shadow = min(shadow, k * d * d / (2.0 * prev_d * t))`. One extra variable, significantly smoother penumbrae. Drop-in replacement.

#### Decision: Ambient Occlusion — SDF-Based, 6 Samples

```wgsl
fn sdf_ao(pos: vec3f, normal: vec3f) -> f32 {
    var ao = 0.0;
    var scale = 1.0;
    for (var i = 1u; i <= 6u; i++) {
        let dist = AO_STEP_SIZE * f32(i);
        let d = evaluate_sdf(pos + normal * dist);
        ao += scale * (dist - d);
        scale *= 0.5;
    }
    return clamp(1.0 - AO_STRENGTH * ao, 0.0, 1.0);
}
```

6 SDF evaluations along the normal. Each sample compares expected vs actual distance — discrepancy indicates nearby occluding geometry. Exponentially decaying weights emphasize near-field occlusion.

Not screen-space limited — detects occlusion from geometry behind the camera or outside the frame. No noise, no blur pass needed.

#### Decision: Subsurface Scattering — SDF Thickness-Based

```wgsl
fn sss_contribution(pos: vec3f, normal: vec3f, light_dir: vec3f, mat: Material) -> vec3f {
    if mat.subsurface <= 0.0 { return vec3f(0.0); }

    // Estimate thickness by sampling SDF in the interior
    let interior_pos = pos - normal * SSS_MAX_THICKNESS;
    let thickness = clamp(-evaluate_sdf(interior_pos), 0.0, SSS_MAX_THICKNESS);

    // Beer's law attenuation through the material
    let attenuation = exp(-thickness * SSS_SIGMA);

    // Light wrapping — illumination continues past the terminator
    let wrap = max(0.0, dot(normal, light_dir) + SSS_WRAP) / (1.0 + SSS_WRAP);

    return mat.subsurface_color * attenuation * wrap * mat.subsurface;
}
```

Cost: 1 additional SDF evaluation per pixel (only for materials with `subsurface > 0`). Convincing for skin, leaves, wax, thin cloth.

**Upgrade path — Multi-sample thickness:**
March 4-8 steps through the interior for a thickness profile instead of a single probe. Better accuracy for complex geometry (e.g., fingers creating variable thickness in a hand). Cost: 4-8 extra SDF evals per SSS pixel.

#### Decision: Lighting Architecture — Tiled, Many Lights

**Light types supported:**

```rust
struct Light {                              // 64 bytes (GPU-aligned)
    light_type: u32,                        // 0 = directional, 1 = point, 2 = spot
    position: [f32; 3],
    direction: [f32; 3],                    // for directional and spot
    color: [f32; 3],                        // linear RGB
    intensity: f32,                         // multiplier
    range: f32,                             // attenuation cutoff
    inner_angle: f32,                       // spot: full-intensity cone half-angle
    outer_angle: f32,                       // spot: falloff cone half-angle
    cookie_index: i32,                      // -1 = none, else index into cookie texture array
    shadow_caster: u32,                     // bool: casts SDF soft shadows?
    _padding: [f32; 2],
}
```

**Spot light with cookie:**
```wgsl
fn evaluate_spot(light: Light, hit_pos: vec3f) -> f32 {
    let to_light = normalize(light.position - hit_pos);
    let angle = acos(dot(-to_light, light.direction));
    let spot = smoothstep(light.outer_angle, light.inner_angle, angle);

    var cookie = 1.0;
    if light.cookie_index >= 0 {
        let local_uv = project_to_light_space(hit_pos, light);
        cookie = textureSample(cookie_array, cookie_sampler, local_uv, light.cookie_index).r;
    }

    return spot * cookie * distance_attenuation(light, hit_pos);
}
```

Cookies stored in a `texture_2d_array` (64×64 or 128×128 per cookie). Supports window patterns, gobo projections, flashlight shapes.

**Tiled light culling (Pass 2):**
```
Pre-shade compute pass:
  1. Divide screen into 16×16 pixel tiles
     At 960×540: 60×34 = 2,040 tiles
  2. Per tile: compute min/max depth from G-buffer → tile frustum
  3. Per light: test bounding sphere against each tile frustum
  4. Write per-tile light lists:
     tile_light_indices[tile_id][i] = light_index
     tile_light_count[tile_id] = N
```

**Shadow budget per tile:**
- All lights in tile contribute diffuse + specular (cheap, no shadow ray)
- Top K lights per tile (ranked by intensity / distance²) cast SDF soft shadows
- Configurable K: 2 (low), 4 (medium), 8 (high)

**Upgrade path — Clustered shading:**
Add depth slicing to tiles → 3D clusters. Reduces false-positive light assignments in deep tiles. Standard technique from DOOM 2016. Only needed if scenes have extreme depth complexity with many overlapping lights.

**Upgrade path — Shadow atlas / cached shadows:**
For static lights on static geometry, cache shadow results per light. Invalidate per-brick on SDF modification. Significant savings for mostly-static scenes with many lights.

#### Decision: Global Illumination — SDF Voxel Cone Tracing

Non-negotiable first-class feature. Our voxel clipmap structure is the data that VXGI engines build as an extra step — we get it for free.

**Radiance Volume (separate from geometry):**

A clipmap of 3D textures storing outgoing radiance + opacity:

| Level | Resolution | Voxel Size | Coverage | Memory (RGBA16F) |
|-------|-----------|------------|----------|-----------------|
| 0 | 128³ | ~4cm | ~5m radius | 16MB |
| 1 | 128³ | ~16cm | ~20m radius | 16MB |
| 2 | 128³ | ~64cm | ~80m radius | 16MB |
| 3 | 128³ | ~256cm | ~320m radius | 16MB |
| **Total** | | | | **64MB** |

Format: `RGBA16F` — RGB radiance + A opacity. Stored as wgpu `texture_3d` with hardware trilinear filtering for fast cone sampling.

**Pass 3 — Radiance Injection (compute):**
```
For each surface voxel (dispatch over occupied geometry bricks):
  1. Compute voxel surface position + normal (SDF gradient)
  2. Evaluate direct illumination from key lights (sun + top N)
     - Simplified shadow (fewer steps, coarser)
     - Emissive materials self-inject their emission
  3. Write radiance + opacity into radiance 3D texture level 0
```

**Pass 4 — Radiance Mip Generation (compute, ×3 dispatches):**
```
Downsample level 0 → 1 → 2 → 3:
  - Radiance: average of 8 children
  - Opacity: max of 8 children (conservative — ensures cone tracing doesn't leak light)
```

**Pass 5 — Cone Tracing (within shade pass):**
```wgsl
fn trace_gi_cone(origin: vec3f, dir: vec3f, half_angle: f32) -> vec3f {
    var radiance = vec3f(0.0);
    var occlusion = 0.0;
    var t = CONE_START;

    for (var i = 0; i < CONE_STEPS; i++) {
        let pos = origin + dir * t;
        let radius = t * tan(half_angle);
        let mip = log2(radius / finest_voxel_size);
        let sample = sample_radiance_volume(pos, mip);  // hardware trilinear + mip

        let a = sample.a * (1.0 - occlusion);
        radiance += sample.rgb * a;
        occlusion += a;
        if occlusion > 0.95 { break; }
        t += radius * STEP_MULT;
    }
    return radiance;
}

fn compute_indirect(pos: vec3f, normal: vec3f) -> vec3f {
    var indirect = vec3f(0.0);
    // 6 cones in cosine-weighted hemisphere
    for (var i = 0; i < 6; i++) {
        indirect += trace_gi_cone(pos, cone_dirs[i], CONE_HALF_ANGLE) * cone_weights[i];
    }
    return indirect;
}
```

6 cones × ~16 steps × 1 texture sample = ~96 texture samples per pixel. At 960×540: ~50M texture samples. Manageable — 3D texture sampling is highly GPU-optimized.

**Upgrade path — Directional radiance:**
Store 6 directional components (±X, ±Y, ±Z) instead of omnidirectional. 6 RGBA16F textures per level (384MB total) or L1 spherical harmonics (4 RGB coefficients per texel). Eliminates light leaking through thin walls. Only pursue if omnidirectional produces visible artifacts.

**Upgrade path — Multi-bounce GI:**
Feed previous frame's indirect radiance back into injection pass. Each frame accumulates one more bounce, converging over ~4 frames. Cheap to implement — add previous GI to injection. Creates warm interiors with color bleeding.

**Upgrade path — Specular GI:**
Trace a single narrow cone in the reflection direction for glossy reflections. Uses the same radiance volume. Cost: 1 extra cone trace for reflective surfaces. Quality limited by radiance volume resolution — works for rough specular, not mirror reflections.

#### Decision: Atmospheric Scattering — Precomputed LUT (Bruneton Model)

Physically-based sky rendering and aerial perspective using precomputed lookup tables.

**Precomputed LUTs (generated once at startup or when atmosphere params change):**

| LUT | Dimensions | Content |
|-----|-----------|---------|
| Transmittance | 256×64 (2D) | Optical depth through atmosphere for any view angle × altitude |
| Scattering | 256×128×32 (3D) | Single-scatter in-scattering for any view/sun angle × altitude |
| Irradiance | 64×16 (2D) | Ground-level irradiance for any sun angle |

Total memory: ~4MB. Generated by a compute shader from atmosphere parameters (Rayleigh/Mie coefficients, planet radius, atmosphere height, ozone density).

**Runtime usage (in shade pass):**

```
For sky pixels (hit_distance == MAX_FLOAT):
  - Sample scattering LUT based on view direction + sun direction
  - Output sky color directly (Rayleigh blue + Mie sun halo)

For geometry pixels (aerial perspective):
  - Compute transmittance from camera to hit point through atmosphere
  - Compute in-scattering along the view ray
  - final_color = surface_color * transmittance + in_scattering
  - Distant objects naturally fade toward sky color
```

Aerial perspective uses the hit distance from the G-buffer — which we already have. No extra ray marching needed, just LUT samples.

**Parameters (configurable for time-of-day):**

```rust
struct Atmosphere {
    rayleigh_coefficients: [f32; 3],  // wavelength-dependent scattering
    mie_coefficient: f32,             // aerosol/haze scattering
    mie_direction: f32,               // Henyey-Greenstein asymmetry (sun halo tightness)
    planet_radius: f32,               // for horizon curvature
    atmosphere_height: f32,           // scale height
    sun_direction: [f32; 3],          // drives everything — animate for day/night cycle
    sun_intensity: f32,
    sun_color: [f32; 3],
}
```

**Upgrade path — Multi-scattering LUT:**
Add a 4th LUT for multiple scattering (Hillaire 2020). Captures light that has scattered more than once — improves sky brightness near the horizon and in shadowed atmosphere. Small additional precompute cost, significant quality improvement for sunset/sunrise.

#### Decision: Post-Processing Stack — Split Low-Res / Final-Res

Post-processing is split across the upscale boundary. Effects that benefit from depth/motion vectors or can tolerate lower resolution run before upscaling. Cosmetic effects that need full resolution run after.

**Pre-upscale stack (at 960×540 internal resolution):**

| Effect | Input | Notes |
|--------|-------|-------|
| Bloom extract | HDR color | Threshold bright pixels |
| Bloom blur | Bloom buffer | Gaussian blur (separable, multi-pass) |
| Depth of field | HDR color + depth | Circle-of-confusion from depth buffer |
| Motion blur | HDR color + motion vectors | Per-pixel directional blur |

**Post-upscale stack (at display resolution):**

| Effect | Input | Notes |
|--------|-------|-------|
| Bloom composite | Upscaled color + bloom | Add blurred bloom back |
| Tone mapping | HDR color | ACES, AgX, or configurable curve |
| Color grading | LDR color | 3D LUT-based color transform |
| Vignette | Final color | Darken edges (optional) |
| Film grain | Final color | Noise overlay (optional) |
| Chromatic aberration | Final color | RGB channel offset (optional) |

**Architecture:** Post-processing is a configurable chain of compute passes. Each effect is a module that can be enabled/disabled and reordered. The engine provides a `PostProcessStack` that iterates the chain, ping-ponging between two color buffers.

```rust
struct PostProcessStack {
    pre_upscale: Vec<Box<dyn PostProcessEffect>>,   // run at internal resolution
    post_upscale: Vec<Box<dyn PostProcessEffect>>,  // run at display resolution
}

trait PostProcessEffect {
    fn execute(&self, encoder: &mut CommandEncoder, input: &Texture, output: &Texture);
    fn enabled(&self) -> bool;
}
```

Detailed per-effect design deferred to Session 9. The key architectural decision (split stack, configurable chain) is locked now.

#### Volumetric Effects — Forward Reference

The following are designed in [Volumetric Effects](./05-volumetric-effects.md) and execute between shading and post-processing:

- **God rays / volumetric light shafts:** Emerge naturally from marching through participating media with shadow checks. March camera rays through density field, evaluate light visibility at each step, accumulate in-scattered light. Sun god rays are the primary use case.
- **Fog volumes (local):** Regions with nonzero density in the volumetric brick pool. Ray marches through them accumulating extinction and scattering.
- **Global fog (height/distance):** Analytic density function — no brick data needed. Computed per-pixel from distance and altitude.
- **Smoke, fire, clouds:** Dynamic density + emission fields in the volumetric brick pool. Updated per frame from simulation or procedural noise.

#### Merged Shade Pass — Full Flow

```
SHADE (compute, Pass 5):
  Input:  G-buffer, material table, light buffer, tile light lists,
          radiance volume, atmosphere LUTs, cookie textures, voxel data
  Output: HDR color buffer at internal resolution

  1. Read G-buffer (position, normal, material IDs, blend weight, motion vectors)
  2. Skip sky pixels → write sky color from atmosphere scattering LUT
  3. Resolve material (table lookup + lerp if blend_weight > 0)
  4. Apply procedural noise perturbation (if noise_strength > 0)
  5. If has_color_data: sample companion color brick, modulate albedo
  6. Compute SDF-based AO (6 samples along normal)
  7. Read tile light list for this pixel's tile
  8. For each light in tile:
     a. Evaluate light type (directional / point / spot + cookie)
     b. If shadow_caster and within shadow budget: SDF soft shadow ray
     c. Cook-Torrance BRDF (diffuse + specular)
     d. SSS contribution (if subsurface > 0)
  9. Cone-trace GI (6 cones through radiance volume)
  10. Apply aerial perspective (atmosphere transmittance + in-scattering)
  11. Combine: direct + indirect + emission + ambient
  12. Write HDR color
```

### Session 4 Summary: All Shading Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| PBR model | Cook-Torrance GGX | — (industry standard) |
| Shadows | SDF soft shadows, inline | Improved penumbra (Quilez 2018) |
| AO | SDF-based, 6 samples | — |
| SSS | SDF thickness-based, Beer's law | Multi-sample thickness profile |
| Light types | Directional + point + spot with cookies | — |
| Light management | Tiled culling (16×16 tiles, 2040 tiles) | Clustered shading; shadow caching |
| Shadow budget | Top K lights/tile (configurable 2-8) | — |
| GI | SDF voxel cone tracing (64MB radiance volume, 6 cones) | Directional radiance; multi-bounce; specular GI |
| Sky/atmosphere | Precomputed Bruneton LUTs (~4MB) | Multi-scattering LUT |
| Aerial perspective | LUT-based transmittance + in-scattering from depth | — |
| Material resolve | Merged into shade pass | Split for editor debug views |
| Post-processing | Split stack: pre-upscale (bloom, DoF, motion blur) + post-upscale (tone map, color grade, film grain) | Per-effect detail in Session 9 |
