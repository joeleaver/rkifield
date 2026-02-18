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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;

// CellState values (2-bit, matching Rust CellState enum)
const CELL_EMPTY: u32      = 0u;
const CELL_SURFACE: u32    = 1u;
const CELL_INTERIOR: u32   = 2u;

// Sun light parameters
const SUN_DIR: vec3<f32> = vec3<f32>(0.4, 0.8, 0.3);
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const SUN_INTENSITY: f32 = 3.0;
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
    let normal = normalize(normal_data.xyz);
    let blend_weight = normal_data.w;

    let packed_mat = textureLoad(gbuf_material, coord, 0).r;
    let material_id = packed_mat & 0xFFFFu;

    // Look up material
    let mat = materials[material_id];
    let albedo = vec3<f32>(mat.albedo_r, mat.albedo_g, mat.albedo_b);
    let roughness = clamp(mat.roughness, 0.04, 1.0);
    let metallic = mat.metallic;
    let emission = vec3<f32>(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;

    // F0: reflectance at normal incidence
    // Dielectric: 0.04, Metal: albedo color
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // View direction (from surface toward camera)
    let view_dir = normalize(shade_uniforms.camera_pos.xyz - world_pos);

    // Lighting
    let light_dir = normalize(SUN_DIR);
    let half_vec = normalize(view_dir + light_dir);

    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let h_dot_v = max(dot(half_vec, view_dir), 0.0);

    // Cook-Torrance specular BRDF
    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    // Energy conservation: diffuse is reduced by what specular reflects
    let ks = f;
    let kd = (vec3<f32>(1.0) - ks) * (1.0 - metallic);

    // Lambertian diffuse
    let diffuse = kd * albedo / PI;

    // SDF soft shadow from sun
    let shadow_origin = world_pos + normal * SHADOW_BIAS;
    let shadow = soft_shadow(shadow_origin, light_dir, SHADOW_MAX_DIST, SHADOW_K);

    // Combined direct lighting (modulated by shadow)
    let radiance = SUN_COLOR * SUN_INTENSITY;
    let direct = (diffuse + specular) * radiance * n_dot_l * shadow;

    // SDF ambient occlusion
    let ao = sdf_ao(world_pos + normal * SHADOW_BIAS, normal);

    // Ambient approximation (hemisphere) — modulated by AO
    let ambient = AMBIENT_COLOR * albedo * ao;

    // Final color = direct + ambient + emission
    var color = direct + ambient + emission;

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
            color = diffuse * radiance * n_dot_l * shadow + ambient;
        }
        case 5u: {
            // Specular only
            color = specular * radiance * n_dot_l * shadow;
        }
        default: {
            // Normal shading (already computed)
        }
    }

    textureStore(hdr_output, coord, vec4<f32>(color, 1.0));
}
