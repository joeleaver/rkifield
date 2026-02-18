// Shading compute shader — Phase 6 (Cook-Torrance GGX BRDF).
//
// Reads the G-buffer, looks up materials from the material table,
// evaluates PBR shading with a single directional light (sun),
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

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;

// Sun light parameters
const SUN_DIR: vec3<f32> = vec3<f32>(0.4, 0.8, 0.3);
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const SUN_INTENSITY: f32 = 3.0;
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.15, 0.18, 0.25);

// Sky gradient
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.15, 0.25, 0.55);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.6, 0.7, 0.85);

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

    // Combined direct lighting
    let radiance = SUN_COLOR * SUN_INTENSITY;
    let direct = (diffuse + specular) * radiance * n_dot_l;

    // Ambient approximation (hemisphere)
    let ambient = AMBIENT_COLOR * albedo;

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
            color = diffuse * radiance * n_dot_l + ambient;
        }
        case 5u: {
            // Specular only
            color = specular * radiance * n_dot_l;
        }
        default: {
            // Normal shading (already computed)
        }
    }

    textureStore(hdr_output, coord, vec4<f32>(color, 1.0));
}
