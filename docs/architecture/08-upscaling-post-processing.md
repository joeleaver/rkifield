# Upscaling and Post-Processing

> **Status: DECIDED**

### Decision: Dual Upscaler Backend — DLSS (Preferred) + Custom (Fallback)

**DLSS** is the preferred upscaler on NVIDIA hardware via the `dlss_wgpu` crate (proven wgpu integration). **Custom temporal upscaler** serves as the cross-platform fallback for AMD/Intel/other hardware, leveraging SDF-specific data (material IDs, SDF normals) that DLSS cannot use.

```rust
enum UpscaleBackend {
    DLSS,    // preferred on NVIDIA, auto-detected
    Custom,  // fallback, always available
}

// At startup:
let backend = if dlss_wgpu::is_available(&device) {
    UpscaleBackend::DLSS
} else {
    UpscaleBackend::Custom
};
```

Both backends consume the same inputs (G-buffer + HDR color) and produce the same output (display-resolution HDR color). The post-processing stack is backend-agnostic.

| Backend | Platform | Strengths | Weaknesses |
|---------|----------|-----------|------------|
| DLSS | NVIDIA RTX | AI detail hallucination, battle-tested, excellent disocclusion handling | Can't use material IDs, proprietary |
| Custom | All GPUs | Material ID rejection (perfect edges), SDF normal awareness, no vendor dependency | No AI hallucination, needs battle-testing |

### Decision: Resolution Strategy — Configurable Ratio

| Quality Mode | Ratio | Internal @ 1080p | Internal @ 4K | Pixel Reduction |
|-------------|-------|-------------------|---------------|-----------------|
| Ultra | 1/1 | 1920×1080 | 3840×2160 | None |
| Quality | 3/4 | 1440×810 | 2880×1620 | 1.8× |
| Balanced | 1/2 | 960×540 | 1920×1080 | 4× |
| Performance | 1/3 | 640×360 | 1280×720 | 9× |
| Potato | 1/4 | 480×270 | 960×540 | 16× |

**Default: Balanced (1/2).** Runtime-configurable. Both DLSS and Custom upscaler support all ratios (DLSS maps these to its Quality/Balanced/Performance/Ultra Performance modes).

**Dynamic resolution (optional):** Adjust ratio per-frame to maintain target frametime. Smooth interpolation of scale factor to avoid visual popping.

### DLSS Integration Path

```rust
// DLSS inputs — all available from our G-buffer
struct DlssInputs {
    color: Texture,          // HDR color at internal resolution
    depth: Texture,          // linear depth from ray march
    motion_vectors: Texture, // screen-space motion from G-buffer
    exposure: f32,           // from auto-exposure or manual setting
    jitter: Vec2,            // sub-pixel jitter offset (Halton 2,3)
    reset: bool,             // true on camera cut (discard history)
}

// Per frame:
dlss.render(&device, &queue, DlssInputs {
    color: shaded_hdr,
    depth: gbuffer_depth,
    motion_vectors: gbuffer_motion,
    exposure: tone_mapping.exposure,
    jitter: halton_jitter(frame_index),
    reset: camera_cut_this_frame,
});
// Output: display-resolution HDR color in dlss.output_texture()
```

DLSS handles its own temporal history, jitter expectations, and spatial upscale internally. We provide the inputs; it returns the upscaled result.

### Custom Temporal Upscaler — Full Design

For non-NVIDIA hardware. Exploits our SDF-specific data.

**Persistent buffers (at display resolution):**
- `history_color`: RGBA16F — accumulated color from previous frames
- `history_metadata`: RG32Uint — packed depth (f32 as u32) + material_id (u16) + flags

**Jitter pattern — Halton 2,3 sequence (16-frame cycle):**
```wgsl
fn jitter(frame: u32) -> vec2f {
    return vec2f(halton_2(frame % 16), halton_3(frame % 16)) - 0.5;
}
```
At 1/2 resolution, 16 jittered frames effectively super-sample each display pixel with ~16 sub-pixel positions.

**Temporal accumulation + spatial upscale (single pass, internal → display):**

```wgsl
@compute @workgroup_size(8, 8, 1)
fn temporal_upscale(@builtin(global_invocation_id) display_pixel: vec3<u32>) {
    // Sample current frame (bilinear from internal resolution)
    let current_uv = display_to_internal_uv(display_pixel);
    let current_color = bilinear_sample(current_hdr, current_uv);
    let current_depth = sample_gbuffer_depth(current_uv);
    let current_material = sample_gbuffer_material(current_uv);
    let current_normal = sample_gbuffer_normal(current_uv);

    // Reproject into previous frame's history
    let motion = sample_gbuffer_motion(current_uv);
    let history_uv = vec2f(display_pixel.xy) / display_resolution - motion;
    let history_color = bilinear_sample(history_buffer, history_uv);
    let history_meta = sample_history_metadata(history_uv);

    // --- Multi-signal rejection ---
    var blend = HISTORY_WEIGHT;  // default 0.9-0.95

    // Signal 1: Material ID (binary, perfect — our unique advantage)
    if current_material != history_meta.material_id {
        blend = 0.0;  // guaranteed different surface
    }

    // Signal 2: Depth discontinuity
    let depth_diff = abs(current_depth - history_meta.depth) / max(current_depth, 0.001);
    if depth_diff > DEPTH_REJECT_THRESHOLD { blend = 0.0; }

    // Signal 3: Normal discontinuity
    if dot(current_normal, history_meta.normal) < NORMAL_REJECT_THRESHOLD { blend = 0.0; }

    // Signal 4: Neighborhood color clipping (catches subtle disocclusion)
    let (color_min, color_max) = neighborhood_aabb_ycocg_3x3(current_uv);
    let clipped = clamp(rgb_to_ycocg(history_color), color_min, color_max);
    let history_clipped = ycocg_to_rgb(clipped);

    // Reduce trust with motion magnitude
    blend *= saturate(1.0 - length(motion) * MOTION_SENSITIVITY);

    let result = mix(current_color, history_clipped, blend);

    // Write to history for next frame
    history_buffer[display_pixel.xy] = result;
    history_metadata[display_pixel.xy] = pack(current_depth, current_material);

    output_hdr[display_pixel.xy] = result;
}
```

**Edge-aware sharpening (post-accumulation, at display resolution):**

```wgsl
fn edge_aware_sharpen(pixel: vec2u) -> vec3f {
    let center = output_hdr[pixel];
    let center_material = upscaled_material[pixel];

    // 5×5 cross kernel, weighted by material + depth similarity
    var blur = vec3f(0.0);
    var weight_sum = 0.0;
    for neighbor in cross_5x5 {
        var w = neighbor.kernel_weight;
        let n_material = upscaled_material[pixel + neighbor.offset];
        if n_material != center_material { w = 0.0; }  // hard stop at material edges
        w *= depth_similarity(pixel, pixel + neighbor.offset);
        blur += output_hdr[pixel + neighbor.offset] * w;
        weight_sum += w;
    }
    blur /= weight_sum;

    // Unsharp mask
    return center + (center - blur) * SHARPEN_STRENGTH;
}
```

Material ID boundaries create perfect sharpening edges — no haloing.

### Decision: Post-Processing Stack — Split Pre/Post Upscale

**Pre-upscale (internal resolution):**

**Bloom — Physically-based multi-mip:**
```
1. Extract: threshold HDR → bloom_buffer (pixels above bloom_threshold)
2. Downsample chain: 4 mip levels, 13-tap tent filter
3. Blur: separable Gaussian at each mip (5-9 taps)
4. Upsample: progressive bilinear upsample with additive blend
   Result stored at internal res for compositing after upscale
```

**Depth of Field:**
```rust
struct DepthOfFieldSettings {
    enabled: bool,
    focus_distance: f32,    // world-space sharp focus distance
    aperture: f32,          // blur amount (larger = more)
    focal_length: f32,      // lens focal length
    max_blur_radius: f32,   // cap in pixels
}
```
Circle of confusion from depth buffer. Two-layer gather: far field (increasing blur past focus) and near field (separate buffer, composited over sharp). At internal resolution, this is cheap.

**Motion Blur:**
```rust
struct MotionBlurSettings {
    enabled: bool,
    strength: f32,          // motion vector multiplier
    max_samples: u32,       // per-pixel cap (default 16)
}
```
Per-pixel directional blur along motion vector. Sample count proportional to motion magnitude.

**Post-upscale (display resolution):**

**Bloom Composite:** Add bilinear-upsampled bloom buffer to upscaled color.

**Tone Mapping:**
```rust
struct ToneMappingSettings {
    curve: ToneMappingCurve,    // ACES, AgX
    exposure: f32,              // manual EV
    auto_exposure: bool,
    auto_exposure_speed: f32,
    auto_exposure_min: f32,
    auto_exposure_max: f32,
}
```
Two built-in curves: ACES (industry standard) and AgX (better highlight preservation). Auto-exposure via HDR luminance reduction.

**Color Grading:** 3D LUT-based (32³ or 64³). Artist-authored, loaded as a 3D texture. Handles any color transform.

**Optional cosmetic effects:** Vignette, film grain (luminance-weighted), chromatic aberration (radial RGB offset). All configurable, all off by default.

**Stack architecture:**
```rust
struct PostProcessStack {
    pre_upscale: Vec<Box<dyn PostProcessEffect>>,   // at internal res
    post_upscale: Vec<Box<dyn PostProcessEffect>>,  // at display res
}

trait PostProcessEffect {
    fn execute(&self, ctx: &PostProcessContext);
    fn enabled(&self) -> bool;
}
```
Configurable chain, ping-pong buffers, effects enable/disable and reorder at runtime.

### Complete Frame Pipeline

```
 1. Ray March                       → G-buffer (internal res)
 2. Tile Light Culling              → per-tile light lists
 3. Radiance Injection              → radiance volume L0
 4. Radiance Mip Generation ×3      → radiance volume L1-L3
 5. Shade (merged material+light+GI) → HDR color (internal res)
 6a. Volumetric Shadow Map          → 3D shadow texture
 6b. Cloud Shadow Map               → 2D shadow texture
 6c. Volumetric March               → scattering (half-internal res)
 6d. Volumetric Bilateral Upscale   → scattering (internal res)
 7. Volumetric Composite            → composited HDR (internal res)
 8. Pre-Upscale Post-Process        → bloom extract/blur, DoF, motion blur
 9. Upscale (DLSS or Custom)        → HDR color (display res)
10. Edge-Aware Sharpen (Custom only) → sharpened (display res)
11. Post-Upscale Post-Process       → bloom composite, tone map, color grade, etc.
12. Present to swapchain
```

### Session 9 Summary: All Upscaling and Post-Processing Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Upscale backend | DLSS (preferred, NVIDIA) + Custom (fallback, all GPUs) | `dlss_wgpu` crate for DLSS integration |
| Resolution | Configurable ratio (1/4 to 1/1), default 1/2 | Dynamic resolution optional |
| Custom temporal | Multi-signal rejection (material ID, depth, normal, color clip) | Material ID is unique SDF advantage |
| Custom spatial | Edge-aware sharpen guided by material + depth | Perfect material edge sharpening |
| Jitter | Halton 2,3, 16-frame cycle | Shared between DLSS and Custom |
| Bloom | Multi-mip physically-based, pre-upscale | — |
| DoF | CoC from depth, two-layer gather | — |
| Motion blur | Per-pixel directional, pre-upscale | — |
| Tone mapping | ACES / AgX, auto-exposure | — |
| Color grading | 3D LUT | — |
| Stack architecture | Configurable chain, split pre/post upscale | — |
