// preview_shade.wgsl — Simplified PBR shading for material preview thumbnails.
//
// Reads G-buffer from the preview march pass, looks up material properties,
// and applies a two-light PBR setup. No shadows, GI, AO, or SDF queries —
// just pure material appearance.

// ---------- Material struct (must match Rust Material, 96 bytes) ----------

struct Material {
    albedo_r: f32,
    albedo_g: f32,
    albedo_b: f32,
    roughness: f32,
    metallic: f32,
    emission_r: f32,
    emission_g: f32,
    emission_b: f32,
    emission_strength: f32,
    subsurface: f32,
    subsurface_r: f32,
    subsurface_g: f32,
    subsurface_b: f32,
    opacity: f32,
    ior: f32,
    noise_scale: f32,
    noise_strength: f32,
    noise_channels: u32,
    shader_id: u32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

// ---------- Uniforms ----------

struct PreviewShadeUniforms {
    camera_pos: vec4<f32>,
    resolution: vec2<f32>,
    _pad: vec2<f32>,
}

// ---------- Bind groups ----------

// Group 0: G-buffer (read)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;

// Group 1: Material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: HDR output + uniforms
@group(2) @binding(0) var hdr_output: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var<uniform> shade_uniforms: PreviewShadeUniforms;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;

// Key light: warm directional from upper-right
const KEY_DIR: vec3<f32> = vec3<f32>(0.5, 0.8, 0.3);
const KEY_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.9);
const KEY_INTENSITY: f32 = 2.5;

// Fill light: cool from lower-left
const FILL_DIR: vec3<f32> = vec3<f32>(-0.3, 0.2, -0.5);
const FILL_COLOR: vec3<f32> = vec3<f32>(0.6, 0.7, 1.0);
const FILL_INTENSITY: f32 = 0.8;

// Ambient
const AMBIENT: vec3<f32> = vec3<f32>(0.08, 0.09, 0.12);

// ---------- Environment ----------

// Procedural studio environment for reflections.
// Bright overall with directional variation so metals read well from all angles.
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let y = dir.y;

    // Base: bright neutral grey everywhere — metals need something to reflect.
    var env = vec3<f32>(0.4, 0.4, 0.42);

    // Vertical gradient: warm below, cool above
    let sky_mix = clamp(y * 0.5 + 0.5, 0.0, 1.0); // 0 = down, 1 = up
    env = mix(
        vec3<f32>(0.35, 0.3, 0.25),  // warm ground
        vec3<f32>(0.45, 0.5, 0.6),   // cool sky
        sky_mix,
    );

    // Bright horizon band — wraps around all sides (key for side-face reflections)
    let horizon_band = exp(-y * y * 8.0); // peaks at y=0, gaussian falloff
    env += vec3<f32>(0.5, 0.48, 0.45) * horizon_band;

    // Key light area: bright spot matching the key directional light direction
    let key_dot = max(dot(dir, normalize(KEY_DIR)), 0.0);
    env += KEY_COLOR * pow(key_dot, 8.0) * 1.5;

    // Fill light area: softer, broader
    let fill_dot = max(dot(dir, normalize(FILL_DIR)), 0.0);
    env += FILL_COLOR * pow(fill_dot, 4.0) * 0.6;

    // Subtle rim from behind camera
    let back = max(-dir.z, 0.0);
    env += vec3<f32>(0.25, 0.25, 0.3) * back * back * 0.4;

    return env;
}

// ---------- PBR Functions ----------

fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn F_Schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    let gv = NdotV / (NdotV * (1.0 - k) + k);
    let gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

fn shade_light(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    albedo: vec3<f32>, roughness: f32, metallic: f32,
    light_color: vec3<f32>, intensity: f32
) -> vec3<f32> {
    let H = normalize(V + L);
    let NdotL = max(dot(N, L), 0.0);
    let NdotV = max(dot(N, V), 0.001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    if NdotL <= 0.0 {
        return vec3<f32>(0.0);
    }

    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    let D = D_GGX(NdotH, roughness);
    let F = F_Schlick(HdotV, F0);
    let G = G_Smith(NdotV, NdotL, roughness);

    let spec = (D * F * G) / (4.0 * NdotV * NdotL + 0.0001);
    let kd = (1.0 - F) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    return (diffuse + spec) * light_color * intensity * NdotL;
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(shade_uniforms.resolution.x), u32(shade_uniforms.resolution.y));
    if gid.x >= res.x || gid.y >= res.y {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let pos_data = textureLoad(gbuf_position, coord, 0);
    let hit_dist = pos_data.w;

    // Sky / background — transparent so panel background shows through.
    if hit_dist < 0.0 {
        textureStore(hdr_output, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Load G-buffer
    let world_pos = pos_data.xyz;
    let normal_data = textureLoad(gbuf_normal, coord, 0);
    let N = normalize(normal_data.xyz);
    let packed_mat = textureLoad(gbuf_material, coord, 0).r;
    let material_id = packed_mat & 0x3Fu;

    // Material lookup
    let mat = materials[material_id];
    let albedo = vec3<f32>(mat.albedo_r, mat.albedo_g, mat.albedo_b);
    let roughness = clamp(mat.roughness, 0.04, 1.0);
    let metallic = mat.metallic;

    // View direction
    let V = normalize(shade_uniforms.camera_pos.xyz - world_pos);

    // Key light
    let key_L = normalize(KEY_DIR);
    var color = shade_light(N, V, key_L, albedo, roughness, metallic, KEY_COLOR, KEY_INTENSITY);

    // Fill light
    let fill_L = normalize(FILL_DIR);
    color += shade_light(N, V, fill_L, albedo, roughness, metallic, FILL_COLOR, FILL_INTENSITY);

    // Environment reflection — critical for metallic materials.
    let R = reflect(-V, N);
    let env = sample_environment(R);
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    // Roughness attenuates environment sharpness (approximate with lerp to diffuse irradiance).
    let env_diffuse = sample_environment(N) * 0.3;
    let env_spec = mix(env, env_diffuse, roughness * roughness);
    let F_env = F_Schlick(max(dot(N, V), 0.0), F0);
    color += env_spec * F_env;

    // Ambient (non-metallic diffuse fill)
    color += AMBIENT * albedo * (1.0 - metallic);

    // Emission
    let emission = vec3<f32>(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;
    color += emission;

    // Subsurface scattering approximation (wrap lighting on key light)
    if mat.subsurface > 0.0 {
        let sss_color = vec3<f32>(mat.subsurface_r, mat.subsurface_g, mat.subsurface_b);
        let wrap = (dot(N, key_L) + 1.0) * 0.5; // [0,1] wrap diffuse
        color += sss_color * mat.subsurface * wrap * 0.5 * KEY_INTENSITY;
    }

    // Simple tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));

    textureStore(hdr_output, coord, vec4<f32>(color, 1.0));
}
