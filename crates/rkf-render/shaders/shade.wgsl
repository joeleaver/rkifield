// Shading compute shader — Phase 7 (Cook-Torrance GGX + SDF shadows).
//
// Reads the G-buffer, looks up materials from the material table,
// evaluates PBR shading with SDF soft shadows from the directional sun light,
// and writes HDR color to the output texture.

// ---------- Material struct (must match Rust Material, 96 bytes) ----------

struct Material {
    // PBR baseline (0–15)
    albedo_r: f32,
    albedo_g: f32,
    albedo_b: f32,
    roughness: f32,
    // 16–31
    metallic: f32,
    emission_r: f32,
    emission_g: f32,
    emission_b: f32,
    // 32–35
    emission_strength: f32,
    // SSS (36–55)
    subsurface: f32,
    subsurface_r: f32,
    subsurface_g: f32,
    subsurface_b: f32,
    opacity: f32,
    ior: f32,
    // Noise (60–71)
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    // Padding (72–95)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

// ---------- Scene data types (must match ray_march.wgsl) ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 16 = u16 material_id
    word1: u32, // byte0 = blend_weight, byte1 = secondary_id, byte2 = flags, byte3 = reserved
}

struct SceneUniforms {
    grid_dims:    vec4<u32>,   // xyz = dimensions, w = unused
    grid_origin:  vec4<f32>,   // xyz = origin, w = brick_extent
    params:       vec4<f32>,   // x = voxel_size, yzw = unused
}

// ---------- Light type (must match Rust Light, 64 bytes) ----------

struct Light {
    light_type: u32,  // 0=directional, 1=point, 2=spot
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    cookie_index: i32,
    shadow_caster: u32,
}

// ---------- Bindings ----------

// Group 0: G-buffer read (sampled textures)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;    // r32uint: packed material data
@group(0) @binding(3) var gbuf_motion:   texture_2d<f32>;   // rg32float: motion vectors

// Group 1: material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: HDR output
@group(2) @binding(0) var hdr_output: texture_storage_2d<rgba16float, write>;

// Group 3: Shade uniforms (debug mode + camera position)
struct ShadeUniforms {
    debug_mode: u32, // 0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse only, 5=specular only
    num_lights: u32,
    num_tiles_x: u32,
    shadow_budget_k: u32, // max shadow-casting lights per pixel (0 = unlimited)
    camera_pos: vec4<f32>, // xyz = world-space camera position, w = unused
}
@group(3) @binding(0) var<uniform> shade_uniforms: ShadeUniforms;

// Group 4: Scene SDF data (same layout as ray march group 0)
@group(4) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(4) @binding(1) var<storage, read> occupancy:  array<u32>;
@group(4) @binding(2) var<storage, read> slots:      array<u32>;
// binding 3 = camera uniforms (not used here, but part of the bind group)
// binding 4 = scene uniforms
@group(4) @binding(4) var<uniform> scene: SceneUniforms;

// Group 5: Light / tile data (read-only for shading)
@group(5) @binding(0) var<storage, read> lights: array<Light>;
@group(5) @binding(1) var<storage, read> tile_light_indices: array<u32>;
@group(5) @binding(2) var<storage, read> tile_light_counts: array<u32>;

// Group 6: Radiance volume for voxel cone tracing GI
struct RadianceVolumeUniforms {
    center:      vec4<f32>,   // xyz = volume centre, w = unused
    voxel_sizes: vec4<f32>,   // per-level voxel sizes [L0, L1, L2, L3]
    inv_extents: vec4<f32>,   // 1/(voxel_size * dim) per level
    params:      vec4<u32>,   // x = dim, y = num_levels
}
@group(6) @binding(0) var radiance_L0: texture_3d<f32>;
@group(6) @binding(1) var radiance_L1: texture_3d<f32>;
@group(6) @binding(2) var radiance_L2: texture_3d<f32>;
@group(6) @binding(3) var radiance_L3: texture_3d<f32>;
@group(6) @binding(4) var radiance_sampler: sampler;
@group(6) @binding(5) var<uniform> vol: RadianceVolumeUniforms;

// Group 7: Color pool (per-voxel color companion data)
@group(7) @binding(0) var<storage, read> color_pool: array<u32>;
@group(7) @binding(1) var<storage, read> color_companion: array<u32>;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// Voxel flags
const FLAG_HAS_COLOR_DATA: u32 = 4u; // bit 2

// CellState values (2-bit, matching Rust CellState enum)
const CELL_EMPTY: u32      = 0u;
const CELL_SURFACE: u32    = 1u;
const CELL_INTERIOR: u32   = 2u;

// Light types
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;

// Ambient/sky
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.15, 0.18, 0.25);

// Sky gradient
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.15, 0.25, 0.55);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.6, 0.7, 0.85);

// Shadow parameters
const MAX_SHADOW_STEPS: u32 = 64u;
const SHADOW_EPSILON: f32 = 0.005;
const SHADOW_K: f32 = 16.0; // Penumbra softness (higher = sharper)
const SHADOW_MAX_DIST: f32 = 50.0;
const SHADOW_BIAS: f32 = 0.02; // Normal-direction bias to avoid self-shadowing

// Ambient occlusion parameters
const AO_STEP_SIZE: f32 = 0.03; // Distance between AO samples along normal
const AO_STRENGTH: f32 = 1.5;   // AO intensity multiplier

// Subsurface scattering parameters
const SSS_MAX_THICKNESS: f32 = 0.3;  // Maximum probed thickness
const SSS_SIGMA: f32 = 8.0;          // Extinction coefficient for Beer's law
const SSS_WRAP: f32 = 0.3;           // Light wrapping past terminator

// ---------- SDF Sampling (duplicated from ray_march.wgsl) ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn sdf_grid_dims() -> vec3<u32>  { return scene.grid_dims.xyz; }
fn sdf_grid_origin() -> vec3<f32> { return scene.grid_origin.xyz; }
fn sdf_brick_extent() -> f32     { return scene.grid_origin.w; }
fn sdf_voxel_size() -> f32       { return scene.params.x; }

fn sdf_world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - sdf_grid_origin();
    return vec3<i32>(floor(local / sdf_brick_extent()));
}

fn sdf_cell_in_bounds(cell: vec3<i32>) -> bool {
    return all(cell >= vec3<i32>(0)) && all(vec3<u32>(cell) < sdf_grid_dims());
}

fn sdf_get_cell_state(flat: u32) -> u32 {
    let word_idx = flat / 16u;
    let bit_offset = (flat % 16u) * 2u;
    return (occupancy[word_idx] >> bit_offset) & 3u;
}

fn sdf_cell_flat_index(cell: vec3<u32>) -> u32 {
    let d = sdf_grid_dims();
    return cell.x + cell.y * d.x + cell.z * d.x * d.y;
}

fn sdf_sample_brick(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> f32 {
    let brick_local = (pos - brick_min) / sdf_voxel_size();
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
    let idx = slot * 512u + voxel_idx;
    return extract_distance(brick_pool[idx].word0);
}

/// Sample the SDF at a world-space position using the sparse grid + brick pool.
fn sample_sdf(pos: vec3<f32>) -> f32 {
    let cell_i = sdf_world_to_cell(pos);
    let be = sdf_brick_extent();

    if !sdf_cell_in_bounds(cell_i) {
        return be;
    }

    let cell = vec3<u32>(cell_i);
    let flat = sdf_cell_flat_index(cell);
    let state = sdf_get_cell_state(flat);

    if state == CELL_EMPTY {
        return be * 0.5;
    }
    if state == CELL_INTERIOR {
        return -be * 0.5;
    }
    if state == CELL_SURFACE {
        let slot = slots[flat];
        if slot == EMPTY_SLOT {
            return be * 0.5;
        }
        let brick_min = sdf_grid_origin() + vec3<f32>(cell) * be;
        return sdf_sample_brick(pos, brick_min, slot);
    }
    return be * 0.5;
}

// ---------- SDF Soft Shadow ----------

/// SDF soft shadow with penumbra from d/t ratio.
/// Returns shadow factor in [0, 1] where 1 = fully lit, 0 = fully shadowed.
fn soft_shadow(origin: vec3<f32>, light_dir: vec3<f32>, max_dist: f32, k: f32) -> f32 {
    var shadow = 1.0;
    var t = SHADOW_EPSILON;
    for (var i = 0u; i < MAX_SHADOW_STEPS; i++) {
        let d = sample_sdf(origin + light_dir * t);
        if d < SHADOW_EPSILON {
            return 0.0;
        }
        shadow = min(shadow, k * d / t);
        t += max(d, SHADOW_EPSILON);
        if t > max_dist {
            break;
        }
    }
    return clamp(shadow, 0.0, 1.0);
}

// ---------- SDF Ambient Occlusion ----------

/// SDF-based ambient occlusion — 6 samples along the surface normal.
/// Compares expected vs actual SDF distance at each sample point.
/// Exponentially decaying weights emphasize near-field occlusion.
/// Returns AO factor in [0, 1] where 1 = no occlusion, 0 = fully occluded.
fn sdf_ao(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    var ao = 0.0;
    var scale = 1.0;
    for (var i = 1u; i <= 6u; i++) {
        let dist = AO_STEP_SIZE * f32(i);
        let d = sample_sdf(pos + normal * dist);
        ao += scale * (dist - d);
        scale *= 0.5;
    }
    return clamp(1.0 - AO_STRENGTH * ao, 0.0, 1.0);
}

// ---------- Subsurface Scattering ----------

/// SDF thickness-based subsurface scattering contribution.
/// Probes the SDF interior to estimate thickness, applies Beer's law
/// attenuation, and wraps light past the terminator for translucency.
/// Only evaluates for materials with subsurface > 0.
fn sss_contribution(pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>,
                    subsurface: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
    if subsurface <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Estimate thickness by sampling SDF in the interior
    let interior_pos = pos - normal * SSS_MAX_THICKNESS;
    let thickness = clamp(-sample_sdf(interior_pos), 0.0, SSS_MAX_THICKNESS);

    // Beer's law attenuation through the material
    let attenuation = exp(-thickness * SSS_SIGMA);

    // Light wrapping — illumination continues past the terminator
    let wrap = max(0.0, dot(normal, light_dir) + SSS_WRAP) / (1.0 + SSS_WRAP);

    return subsurface_color * attenuation * wrap * subsurface;
}

// ---------- Light Attenuation ----------

/// Windowed inverse-square distance attenuation for point/spot lights.
/// Smooth falloff to zero at the light's range. Prevents singularity at distance=0.
fn distance_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = dist * dist;
    let r2 = range * range;
    let factor = d2 / r2;
    // Smooth window: (1 - (d/r)^2)^2, clamped to [0,1]
    let window = clamp(1.0 - factor, 0.0, 1.0);
    return (window * window) / max(d2, 0.0001);
}

// ---------- PBR Functions ----------

/// GGX/Trowbridge-Reitz normal distribution function (D term).
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

/// Schlick approximation of Fresnel reflectance (F term).
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/// Smith's method using Schlick-GGX for geometry term (G term).
/// Combined G1(N,V) * G1(N,L).
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;

    let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return ggx_v * ggx_l;
}

// ---------- 3D Simplex Noise (Ashima Arts webgl-noise) ----------

fn mod289_3(x: vec3<f32>) -> vec3<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn permute(x: vec4<f32>) -> vec4<f32> {
    return mod289_4(((x * 34.0) + 10.0) * x);
}

fn taylor_inv_sqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 - 0.85373472095314 * r;
}

/// 3D simplex noise. Returns value in approximately [-1, 1].
fn simplex3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x));

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + vec3<f32>(C.x);
    let x2 = x0 - i2 + vec3<f32>(C.y);
    let x3 = x0 - D.yyy;

    // Permutations
    i = mod289_3(i);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    let n_ = 0.142857142857; // 1.0/7.0
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + vec4<f32>(ns.y);
    let y = y_ * ns.x + vec4<f32>(ns.y);
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4<f32>(x.xy, y.xy);
    let b1 = vec4<f32>(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4<f32>(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3<f32>(a0.xy, h.x);
    var p1 = vec3<f32>(a0.zw, h.y);
    var p2 = vec3<f32>(a1.xy, h.z);
    var p3 = vec3<f32>(a1.zw, h.w);

    // Normalise gradients
    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix contributions from the four corners
    var m = max(vec4<f32>(0.5) - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 105.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Noise channel bit flags (matching Rust NOISE_CHANNEL_* constants)
const NOISE_CHANNEL_ALBEDO: u32 = 1u;    // bit 0
const NOISE_CHANNEL_ROUGHNESS: u32 = 2u; // bit 1
const NOISE_CHANNEL_NORMAL: u32 = 4u;    // bit 2

// ---------- Voxel Cone Tracing GI ----------

// Cone tracing parameters
const GI_CONE_STEPS: u32 = 16u;
const GI_CONE_HALF_ANGLE: f32 = 0.5236; // ~30 degrees
const GI_STEP_MULT: f32 = 1.0;          // Step by cone diameter
const GI_OCCLUSION_CUTOFF: f32 = 0.95;
const GI_STRENGTH: f32 = 1.0;

/// Sample the radiance volume at a world-space position and mip level.
/// Returns vec4(radiance.rgb, opacity).
fn sample_radiance(pos: vec3<f32>, level: i32) -> vec4<f32> {
    let dim = f32(vol.params.x);
    // Convert world position to [0, 1] UV for the given level
    let inv_ext = vol.inv_extents[level];
    let uv = (pos - vol.center.xyz) * inv_ext + 0.5;

    // Out-of-bounds check
    if any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0)) {
        return vec4<f32>(0.0);
    }

    // Sample the appropriate level with trilinear filtering
    switch level {
        case 0: { return textureSampleLevel(radiance_L0, radiance_sampler, uv, 0.0); }
        case 1: { return textureSampleLevel(radiance_L1, radiance_sampler, uv, 0.0); }
        case 2: { return textureSampleLevel(radiance_L2, radiance_sampler, uv, 0.0); }
        case 3: { return textureSampleLevel(radiance_L3, radiance_sampler, uv, 0.0); }
        default: { return vec4<f32>(0.0); }
    }
}

/// Trace a single GI cone through the radiance volume.
/// Front-to-back alpha compositing, automatic mip selection from cone radius.
fn trace_gi_cone(origin: vec3<f32>, dir: vec3<f32>, half_angle: f32) -> vec4<f32> {
    let tan_half = tan(half_angle);
    let finest_voxel = vol.voxel_sizes.x;

    var color = vec3<f32>(0.0);
    var alpha = 0.0;
    // Start a few voxels out to avoid self-intersection
    var t = finest_voxel * 3.0;

    for (var i = 0u; i < GI_CONE_STEPS; i++) {
        let sample_pos = origin + dir * t;
        let cone_radius = t * tan_half;

        // Select mip level based on cone radius vs finest voxel size
        let mip_f = log2(max(cone_radius / finest_voxel, 1.0));
        let level = clamp(i32(mip_f), 0, 3);

        let s = sample_radiance(sample_pos, level);

        // Front-to-back compositing
        let a = s.a * (1.0 - alpha);
        color += s.rgb * a;
        alpha += a;

        if alpha >= GI_OCCLUSION_CUTOFF {
            break;
        }

        // Advance by cone diameter (or at least one finest voxel)
        t += max(cone_radius * 2.0 * GI_STEP_MULT, finest_voxel);
    }

    return vec4<f32>(color, alpha);
}

/// Build an orthonormal basis (tangent, bitangent) from a normal vector.
fn build_tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    // Pick a non-parallel reference vector
    let ref_vec = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let t = normalize(cross(n, ref_vec));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

/// Compute indirect illumination via 6-cone hemisphere trace.
/// Returns indirect radiance (RGB).
fn compute_indirect(pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let frame = build_tangent_frame(normal);

    // 6 cosine-weighted cones: 1 centre + 5 side cones
    // Centre cone: along normal
    // Side cones: ~57° from normal (cos(57°) ≈ 0.545), evenly spaced azimuthally

    let center_weight = 1.0;
    let side_weight = 0.545; // cos(~57°)
    let total_weight = center_weight + 5.0 * side_weight;

    // Side cone directions in tangent space (57° from Z, 72° apart in azimuth)
    let sin_a = 0.8387; // sin(57°)
    let cos_a = 0.5446; // cos(57°)

    var result = vec3<f32>(0.0);

    // Centre cone (along normal)
    let c0 = trace_gi_cone(pos, normal, GI_CONE_HALF_ANGLE);
    result += c0.rgb * center_weight;

    // 5 side cones
    for (var i = 0u; i < 5u; i++) {
        let angle = f32(i) * 1.2566371; // 2π/5 = 72° in radians
        let local_dir = vec3<f32>(sin_a * cos(angle), sin_a * sin(angle), cos_a);
        let world_dir = frame * local_dir;
        let c = trace_gi_cone(pos, world_dir, GI_CONE_HALF_ANGLE);
        result += c.rgb * side_weight;
    }

    return result * GI_STRENGTH / total_weight;
}

// ---------- Material Blending ----------

struct ResolvedMaterial {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    emission_strength: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
}

fn resolve_material_from(m: Material) -> ResolvedMaterial {
    return ResolvedMaterial(
        vec3<f32>(m.albedo_r, m.albedo_g, m.albedo_b),
        m.roughness,
        m.metallic,
        vec3<f32>(m.emission_r, m.emission_g, m.emission_b),
        m.emission_strength,
        m.subsurface,
        vec3<f32>(m.subsurface_r, m.subsurface_g, m.subsurface_b),
        m.opacity,
        m.ior,
        m.noise_scale,
        m.noise_strength,
        m.noise_channels,
    );
}

fn blend_materials(primary_id: u32, secondary_id: u32, weight: f32) -> ResolvedMaterial {
    let a = resolve_material_from(materials[primary_id]);
    if weight <= 0.0 {
        return a;
    }
    let b = resolve_material_from(materials[secondary_id]);
    return ResolvedMaterial(
        mix(a.albedo, b.albedo, weight),
        mix(a.roughness, b.roughness, weight),
        mix(a.metallic, b.metallic, weight),
        mix(a.emission, b.emission, weight),
        mix(a.emission_strength, b.emission_strength, weight),
        mix(a.subsurface, b.subsurface, weight),
        mix(a.subsurface_color, b.subsurface_color, weight),
        mix(a.opacity, b.opacity, weight),
        mix(a.ior, b.ior, weight),
        mix(a.noise_scale, b.noise_scale, weight),
        mix(a.noise_strength, b.noise_strength, weight),
        select(a.noise_channels, b.noise_channels, weight > 0.5),
    );
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = vec2<u32>(textureDimensions(hdr_output));
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);

    // Read G-buffer
    let pos_data = textureLoad(gbuf_position, coord, 0);
    let hit_dist = pos_data.w;

    // Sky pixel — write sky color and return
    if hit_dist >= MAX_FLOAT * 0.5 {
        // Simple sky gradient based on ray direction (approximated from UV)
        let uv_y = f32(pixel.y) / f32(dims.y);
        let sky = mix(SKY_HORIZON, SKY_ZENITH, uv_y);
        textureStore(hdr_output, coord, vec4<f32>(sky, 1.0));
        return;
    }

    let world_pos = pos_data.xyz;
    let normal_data = textureLoad(gbuf_normal, coord, 0);
    var normal = normalize(normal_data.xyz);
    let blend_weight = normal_data.w;

    let packed_mat = textureLoad(gbuf_material, coord, 0).r;
    let material_id = packed_mat & 0xFFFFu;
    let secondary_id = (packed_mat >> 16u) & 0xFFu;
    let voxel_flags = (packed_mat >> 24u) & 0xFFu;

    // Resolve material with blending
    let resolved = blend_materials(material_id, secondary_id, blend_weight);
    var albedo = resolved.albedo;
    var roughness = clamp(resolved.roughness, 0.04, 1.0);
    let metallic = resolved.metallic;
    let emission = resolved.emission * resolved.emission_strength;

    // Per-voxel color modulation (companion color pool)
    if (voxel_flags & FLAG_HAS_COLOR_DATA) != 0u {
        let cell_i = sdf_world_to_cell(world_pos);
        if sdf_cell_in_bounds(cell_i) {
            let cell = vec3<u32>(cell_i);
            let flat = sdf_cell_flat_index(cell);
            let sdf_slot = slots[flat];
            if sdf_slot != EMPTY_SLOT {
                let color_slot = color_companion[sdf_slot];
                if color_slot != EMPTY_SLOT {
                    let be = sdf_brick_extent();
                    let brick_min = sdf_grid_origin() + vec3<f32>(cell) * be;
                    let brick_local = (world_pos - brick_min) / sdf_voxel_size();
                    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
                    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
                    let packed_color = color_pool[color_slot * 512u + voxel_idx];
                    let cr = f32(packed_color & 0xFFu) / 255.0;
                    let cg = f32((packed_color >> 8u) & 0xFFu) / 255.0;
                    let cb = f32((packed_color >> 16u) & 0xFFu) / 255.0;
                    let ci = f32((packed_color >> 24u) & 0xFFu) / 255.0;
                    // Multiply mode: tint albedo, weighted by intensity
                    let voxel_color = vec3<f32>(cr, cg, cb);
                    albedo = mix(albedo, albedo * voxel_color, ci);
                }
            }
        }
    }

    // Procedural noise variation
    if resolved.noise_channels != 0u && resolved.noise_scale > 0.0 && resolved.noise_strength > 0.0 {
        let noise_pos = world_pos * resolved.noise_scale;
        let n = simplex3d(noise_pos);
        let ns = n * resolved.noise_strength;

        // Albedo perturbation: vary brightness/color
        if (resolved.noise_channels & NOISE_CHANNEL_ALBEDO) != 0u {
            albedo = clamp(albedo + albedo * ns, vec3<f32>(0.0), vec3<f32>(1.0));
        }

        // Roughness perturbation
        if (resolved.noise_channels & NOISE_CHANNEL_ROUGHNESS) != 0u {
            roughness = clamp(roughness + ns * 0.5, 0.04, 1.0);
        }

        // Normal perturbation (bump mapping via noise gradient)
        if (resolved.noise_channels & NOISE_CHANNEL_NORMAL) != 0u {
            let eps = 0.01 / resolved.noise_scale;
            let dnx = simplex3d(noise_pos + vec3<f32>(eps, 0.0, 0.0)) - simplex3d(noise_pos - vec3<f32>(eps, 0.0, 0.0));
            let dny = simplex3d(noise_pos + vec3<f32>(0.0, eps, 0.0)) - simplex3d(noise_pos - vec3<f32>(0.0, eps, 0.0));
            let dnz = simplex3d(noise_pos + vec3<f32>(0.0, 0.0, eps)) - simplex3d(noise_pos - vec3<f32>(0.0, 0.0, eps));
            let bump = vec3<f32>(dnx, dny, dnz) * resolved.noise_strength * 0.5;
            normal = normalize(normal + bump);
        }
    }

    // F0: reflectance at normal incidence
    // Dielectric: 0.04, Metal: albedo color
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // View direction (from surface toward camera)
    let view_dir = normalize(shade_uniforms.camera_pos.xyz - world_pos);
    let n_dot_v = max(dot(normal, view_dir), 0.001);

    // SSS color lookup
    let sss_color = resolved.subsurface_color;

    // Tile-based light iteration
    let tile_x = pixel.x / 16u;
    let tile_y = pixel.y / 16u;
    let tile_id = tile_y * shade_uniforms.num_tiles_x + tile_x;
    let tile_base = tile_id * MAX_LIGHTS_PER_TILE;
    let tile_count = tile_light_counts[tile_id];

    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);
    var sss_total = vec3<f32>(0.0);
    var shadow_count = 0u;
    let shadow_budget = shade_uniforms.shadow_budget_k;

    for (var li = 0u; li < tile_count; li++) {
        let light_idx = tile_light_indices[tile_base + li];
        let light = lights[light_idx];

        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);
        let radiance = light_color * light.intensity;

        if light.light_type == LIGHT_TYPE_DIRECTIONAL {
            let light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let half_vec = normalize(view_dir + light_dir);

            let n_dot_l = max(dot(normal, light_dir), 0.0);
            let n_dot_h = max(dot(normal, half_vec), 0.0);
            let h_dot_v = max(dot(half_vec, view_dir), 0.0);

            // Cook-Torrance specular BRDF
            let d = distribution_ggx(n_dot_h, roughness);
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let f = fresnel_schlick(h_dot_v, f0);

            let numerator = d * g * f;
            let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
            let specular_brdf = numerator / denominator;

            let ks = f;
            let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
            let diffuse_brdf = kd * albedo / PI;

            // SDF soft shadow (if light casts shadows and budget allows)
            var shadow = 1.0;
            if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                let shadow_origin = world_pos + normal * SHADOW_BIAS;
                shadow = soft_shadow(shadow_origin, light_dir, SHADOW_MAX_DIST, SHADOW_K);
                shadow_count += 1u;
            }

            total_diffuse += diffuse_brdf * radiance * n_dot_l * shadow;
            total_specular += specular_brdf * radiance * n_dot_l * shadow;

            // SSS for this light
            sss_total += sss_contribution(world_pos, normal, light_dir, resolved.subsurface, sss_color)
                         * radiance * shadow;
        } else if light.light_type == LIGHT_TYPE_POINT {
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z);
            let to_light = light_pos - world_pos;
            let dist = length(to_light);
            let light_dir = to_light / max(dist, 0.0001);

            let atten = distance_attenuation(dist, light.range);
            if atten > 0.001 {
                let half_vec = normalize(view_dir + light_dir);

                let n_dot_l = max(dot(normal, light_dir), 0.0);
                let n_dot_h = max(dot(normal, half_vec), 0.0);
                let h_dot_v = max(dot(half_vec, view_dir), 0.0);

                // Cook-Torrance specular BRDF
                let d = distribution_ggx(n_dot_h, roughness);
                let g = geometry_smith(n_dot_v, n_dot_l, roughness);
                let f = fresnel_schlick(h_dot_v, f0);

                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
                let specular_brdf = numerator / denominator;

                let ks = f;
                let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
                let diffuse_brdf = kd * albedo / PI;

                // SDF soft shadow (if light casts shadows and budget allows)
                var shadow = 1.0;
                if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                    let shadow_origin = world_pos + normal * SHADOW_BIAS;
                    shadow = soft_shadow(shadow_origin, light_dir, min(dist, SHADOW_MAX_DIST), SHADOW_K);
                    shadow_count += 1u;
                }

                let attenuated_radiance = radiance * atten;
                total_diffuse += diffuse_brdf * attenuated_radiance * n_dot_l * shadow;
                total_specular += specular_brdf * attenuated_radiance * n_dot_l * shadow;

                // SSS for this light
                sss_total += sss_contribution(world_pos, normal, light_dir, resolved.subsurface, sss_color)
                             * attenuated_radiance * shadow;
            }
        } else if light.light_type == LIGHT_TYPE_SPOT {
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z);
            let spot_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let to_light = light_pos - world_pos;
            let dist = length(to_light);
            let light_dir = to_light / max(dist, 0.0001);

            // Cone factor: angle between -light_dir and spot direction
            let cos_angle = dot(-light_dir, spot_dir);
            let cos_outer = cos(light.outer_angle);
            let cos_inner = cos(light.inner_angle);
            let spot = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);

            let atten = spot * distance_attenuation(dist, light.range);
            if atten > 0.001 {
                let half_vec = normalize(view_dir + light_dir);

                let n_dot_l = max(dot(normal, light_dir), 0.0);
                let n_dot_h = max(dot(normal, half_vec), 0.0);
                let h_dot_v = max(dot(half_vec, view_dir), 0.0);

                // Cook-Torrance specular BRDF
                let d = distribution_ggx(n_dot_h, roughness);
                let g = geometry_smith(n_dot_v, n_dot_l, roughness);
                let f = fresnel_schlick(h_dot_v, f0);

                let numerator = d * g * f;
                let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
                let specular_brdf = numerator / denominator;

                let ks = f;
                let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);
                let diffuse_brdf = kd * albedo / PI;

                // SDF soft shadow (if light casts shadows and budget allows)
                var shadow = 1.0;
                if light.shadow_caster == 1u && (shadow_budget == 0u || shadow_count < shadow_budget) {
                    let shadow_origin = world_pos + normal * SHADOW_BIAS;
                    shadow = soft_shadow(shadow_origin, light_dir, min(dist, SHADOW_MAX_DIST), SHADOW_K);
                    shadow_count += 1u;
                }

                let attenuated_radiance = radiance * atten;
                total_diffuse += diffuse_brdf * attenuated_radiance * n_dot_l * shadow;
                total_specular += specular_brdf * attenuated_radiance * n_dot_l * shadow;

                // SSS for this light
                sss_total += sss_contribution(world_pos, normal, light_dir, resolved.subsurface, sss_color)
                             * attenuated_radiance * shadow;
            }
        }
    }

    // SDF ambient occlusion
    let ao = sdf_ao(world_pos + normal * SHADOW_BIAS, normal);

    // Voxel cone tracing indirect illumination
    let indirect = compute_indirect(world_pos + normal * SHADOW_BIAS, normal);
    let gi_contribution = indirect * albedo * (1.0 - metallic);

    // Ambient approximation (hemisphere) — modulated by AO, reduced by GI
    // When GI is active, reduce ambient to avoid double-counting
    let ambient = AMBIENT_COLOR * albedo * ao * 0.3;

    // Final color = direct + indirect GI + SSS + ambient + emission
    let direct = total_diffuse + total_specular;
    var color = direct + gi_contribution * ao + sss_total + ambient + emission;

    // Debug visualization modes
    switch shade_uniforms.debug_mode {
        case 1u: {
            // Normals: remap [-1,1] → [0,1] for visualization
            color = normal * 0.5 + 0.5;
        }
        case 2u: {
            // World positions: scale to visible range
            color = abs(world_pos) * 0.5;
        }
        case 3u: {
            // Material IDs: distinct colors per ID
            let mid = material_id;
            color = vec3<f32>(
                f32((mid * 7u + 3u) % 11u) / 10.0,
                f32((mid * 13u + 5u) % 11u) / 10.0,
                f32((mid * 19u + 7u) % 11u) / 10.0,
            );
        }
        case 4u: {
            // Diffuse only (no specular, no emission)
            color = total_diffuse + ambient;
        }
        case 5u: {
            // Specular only
            color = total_specular;
        }
        default: {
            // Normal shading (already computed)
        }
    }

    textureStore(hdr_output, coord, vec4<f32>(color, 1.0));
}
