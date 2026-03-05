// Shared shading infrastructure — structs, bindings, constants, and utility functions.
//
// This file is concatenated with per-model shader files (shade_pbr.wgsl, etc.)
// and shade_main.wgsl by the CPU-side ShaderComposer to produce the final
// uber-shader. All functions defined here are available to shading model functions.

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
    // Shader selection (72–75)
    shader_id: u32,
    // Padding (76–95)
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
}

// ---------- v2 Scene data types (must match ray_march.wgsl) ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 16 = u16 material_id
    word1: u32,
}

struct GpuObject {
    inverse_world: mat4x4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    brick_map_offset: u32,
    brick_map_dims_x: u32,
    brick_map_dims_y: u32,
    brick_map_dims_z: u32,
    voxel_size: f32,
    material_id: u32,
    sdf_type: u32,
    blend_mode: u32,
    blend_radius: f32,
    sdf_param_0: f32,
    sdf_param_1: f32,
    sdf_param_2: f32,
    sdf_param_3: f32,
    accumulated_scale_x: f32,
    accumulated_scale_y: f32,
    accumulated_scale_z: f32,
    lod_level: u32,
    object_id: u32,
    primitive_type: u32,
    geometry_aabb_min_x: f32, geometry_aabb_min_y: f32, geometry_aabb_min_z: f32,
    geometry_aabb_max_x: f32, geometry_aabb_max_y: f32, geometry_aabb_max_z: f32,
    _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: f32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32,
}

struct BvhNode {
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_min_z: f32,
    left: u32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    aabb_max_z: f32,
    right_or_object: u32,
}

struct SceneUniformsV2 {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

// ---------- Light type (must match Rust Light, 64 bytes) ----------

struct Light {
    light_type: u32,
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

// ---------- Coarse field info ----------

struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,
    dims: vec4<u32>,
    voxel_size: f32,
    inv_voxel_size: f32,
    _cf_pad0: f32,
    _cf_pad1: f32,
}

struct RadianceVolumeUniforms {
    center:      vec4<f32>,
    voxel_sizes: vec4<f32>,
    inv_extents: vec4<f32>,
    params:      vec4<u32>,   // x = dim, y = num_levels
}

// ---------- ShadingContext — the user shader API ----------

struct ShadingContext {
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    n_dot_v: f32,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emission: vec3<f32>,
    emission_strength: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    opacity: f32,
    ior: f32,
    f0: vec3<f32>,
    reflect_dir: vec3<f32>,
    cam_dist: f32,
    jitter: f32,
    contact: f32,
    atmo_shadow_fill: f32,
    pixel: vec2<u32>,
    material_id: u32,
    sss_color: vec3<f32>,
}

// ---------- Bindings ----------

// Group 0: G-buffer read (sampled textures)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;
@group(0) @binding(3) var gbuf_motion:   texture_2d<f32>;

// Group 1: material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: HDR output
@group(2) @binding(0) var hdr_output: texture_storage_2d<rgba16float, write>;

// Group 3: Shade uniforms
struct ShadeUniforms {
    debug_mode: u32,
    num_lights: u32,
    _su_pad0: u32,
    shadow_budget_k: u32,
    camera_pos: vec4<f32>,
    // Atmosphere
    sun_dir: vec4<f32>,        // xyz = direction toward sun, w = sun_intensity
    sun_color: vec4<f32>,      // xyz = sun color (linear RGB), w = unused
    sky_params: vec4<f32>,     // x = rayleigh_scale, y = mie_scale, z = atmosphere_enabled, w = unused
    // Camera basis for sky ray reconstruction
    cam_forward: vec4<f32>,    // xyz = camera forward (unit), w = unused
    cam_right: vec4<f32>,      // xyz = camera right * tan(fov/2) * aspect, w = unused
    cam_up: vec4<f32>,         // xyz = camera up * tan(fov/2), w = unused
}
@group(3) @binding(0) var<uniform> shade_uniforms: ShadeUniforms;

// Group 4: v2 Scene data (same layout as ray march group 0)
@group(4) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(4) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(4) @binding(2) var<storage, read> objects: array<GpuObject>;
// binding 3 = camera uniforms (not used here — shade_uniforms has camera_pos)
// binding 4 = scene uniforms
@group(4) @binding(4) var<uniform> v2_scene: SceneUniformsV2;
@group(4) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 5: Light buffer
@group(5) @binding(0) var<storage, read> lights: array<Light>;

// Group 6: Coarse acceleration field
@group(6) @binding(0) var coarse_field: texture_3d<f32>;
@group(6) @binding(1) var coarse_sampler: sampler;
@group(6) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// Group 7: Radiance volume (4 clipmap levels + sampler + uniforms)
@group(7) @binding(0) var radiance_L0: texture_3d<f32>;
@group(7) @binding(1) var radiance_L1: texture_3d<f32>;
@group(7) @binding(2) var radiance_L2: texture_3d<f32>;
@group(7) @binding(3) var radiance_L3: texture_3d<f32>;
@group(7) @binding(4) var radiance_sampler: sampler;
@group(7) @binding(5) var<uniform> radiance_vol: RadianceVolumeUniforms;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const BVH_STACK_SIZE: u32 = 32u;

const SDF_TYPE_NONE: u32       = 0u;
const SDF_TYPE_ANALYTICAL: u32 = 1u;
const SDF_TYPE_VOXELIZED: u32  = 2u;

const PRIM_SPHERE: u32   = 0u;
const PRIM_BOX: u32      = 1u;
const PRIM_CAPSULE: u32  = 2u;
const PRIM_TORUS: u32    = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_PLANE: u32    = 5u;

// Light types
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;

// Ambient/sky
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.03, 0.035, 0.05);
const SKY_ZENITH: vec3<f32> = vec3<f32>(0.12, 0.18, 0.45);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.95, 0.6, 0.3);
const SKY_REFLECT_STRENGTH: f32 = 0.15;

// Shadow parameters
const MAX_SHADOW_STEPS: u32 = 32u;
const SHADOW_EPSILON: f32 = 0.005;
const SHADOW_K: f32 = 16.0;
const SHADOW_MAX_DIST: f32 = 25.0;
const SHADOW_BIAS: f32 = 0.08;
const SHADOW_ATMO_DENSITY: f32 = 0.04;
const SHADOW_ATMO_MAX_FILL: f32 = 0.65;

// AO parameters
const AO_STEP_SIZE: f32 = 0.12;
const AO_STRENGTH: f32 = 0.7;

// SSS parameters
const SSS_MAX_THICKNESS: f32 = 0.3;
const SSS_SIGMA: f32 = 8.0;
const SSS_WRAP: f32 = 0.3;

// Coarse field threshold for switching to per-object evaluation
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

// GI cone tracing parameters
const GI_CONE_STEPS: u32 = 24u;
const GI_MAX_STEP: f32 = 0.16;
const GI_STRENGTH: f32 = 2.0;
const GI_DIFFUSE_MAX_DIST: f32 = 5.0;
const GI_SPECULAR_MAX_DIST: f32 = 8.0;

// Noise channel constants
const NOISE_CHANNEL_ALBEDO: u32 = 1u;
const NOISE_CHANNEL_ROUGHNESS: u32 = 2u;
const NOISE_CHANNEL_NORMAL: u32 = 4u;

// ---------- VoxelSample Helpers ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

// ---------- SDF Primitives ----------

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let q = vec3<f32>(p.x, max(abs(p.y) - half_height, 0.0), p.z);
    return length(q) - radius;
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normal) + dist;
}

// ---------- Object Evaluation (from ray_march.wgsl) ----------

fn evaluate_analytical(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    switch obj.primitive_type {
        case PRIM_SPHERE: { return sdf_sphere(local_pos, obj.sdf_param_0); }
        case PRIM_BOX: { return sdf_box(local_pos, vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)); }
        case PRIM_CAPSULE: { return sdf_capsule(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_TORUS: { return sdf_torus(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_CYLINDER: { return sdf_cylinder(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_PLANE: { return sdf_plane(local_pos, normalize(vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)), 0.0); }
        default: { return MAX_FLOAT; }
    }
}

fn sample_voxel_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                    total_voxels: vec3<i32>, vs: f32) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        // Empty brick: no geometry. Return brick_extent so shadow rays step
        // completely over this brick — empty slots must never cast shadows.
        return vs * 8.0;
    }
    if slot == INTERIOR_SLOT {
        return -(vs * 2.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_distance(brick_pool[idx].word0);
}

fn sample_voxelized(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);
    if outside_dist > brick_extent * 2.0 {
        return outside_dist;
    }
    // Geometry AABB early-out: skip empty expanded brick-map region.
    let geom_min = vec3<f32>(obj.geometry_aabb_min_x, obj.geometry_aabb_min_y, obj.geometry_aabb_min_z);
    let geom_max = vec3<f32>(obj.geometry_aabb_max_x, obj.geometry_aabb_max_y, obj.geometry_aabb_max_z);
    if geom_max.x > geom_min.x {
        let geom_closest = clamp(local_pos, geom_min, geom_max);
        let geom_dist = length(local_pos - geom_closest);
        if geom_dist > brick_extent {
            return geom_dist + outside_dist;
        }
    }
    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;
    let c000 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 0), dims, total_voxels, vs);
    let c100 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 0), dims, total_voxels, vs);
    let c010 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 0), dims, total_voxels, vs);
    let c110 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 0), dims, total_voxels, vs);
    let c001 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 1), dims, total_voxels, vs);
    let c101 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 1), dims, total_voxels, vs);
    let c011 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 1), dims, total_voxels, vs);
    let c111 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 1), dims, total_voxels, vs);
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z) + outside_dist;
}

/// Evaluate a single object at a world-space position. Returns world-space distance.
fn evaluate_object_dist(world_pos: vec3<f32>, obj_idx: u32) -> f32 {
    let obj = objects[obj_idx];
    if obj.sdf_type == SDF_TYPE_NONE {
        return MAX_FLOAT;
    }
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    var dist: f32;
    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        dist = evaluate_analytical(local_pos, obj);
    } else {
        dist = sample_voxelized(local_pos, obj);
    }
    let min_scale = min(obj.accumulated_scale_x, min(obj.accumulated_scale_y, obj.accumulated_scale_z));
    return dist * min_scale;
}

// ---------- Coarse Field Sampling ----------

fn sample_coarse_field(cam_rel_pos: vec3<f32>) -> f32 {
    let field_pos = cam_rel_pos - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }
    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
}

// ---------- v2 SDF Point Query (coarse field + BVH) ----------

/// Sample the SDF at a world-space position using the v2 coarse field + BVH.
/// Returns the minimum signed distance to any object surface.
///
/// Coarse field uses camera-relative coordinates (centered on camera).
/// Object evaluation uses world-space (inverse_world is world-space).
/// BVH traversal uses world-space positions against world-space AABBs.
fn sample_sdf(pos: vec3<f32>) -> f32 {
    // Coarse field is camera-relative (centered on camera).
    let cam_rel = pos - shade_uniforms.camera_pos.xyz;
    let coarse_dist = sample_coarse_field(cam_rel);
    if coarse_dist > COARSE_NEAR_THRESHOLD {
        return coarse_dist;
    }

    // BVH traversal for precise per-object distance (world-space).
    if v2_scene.num_objects == 0u {
        return MAX_FLOAT;
    }

    var min_dist = MAX_FLOAT;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
        let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

        let closest = clamp(pos, node_min, node_max);
        let box_dist = length(closest - pos);
        if box_dist > min_dist {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let d = evaluate_object_dist(pos, leaf_obj_idx);
                min_dist = min(min_dist, d);
            }
        } else {
            if stack_ptr < BVH_STACK_SIZE - 1u {
                stack[stack_ptr] = node.left;
                stack_ptr += 1u;
                stack[stack_ptr] = node.right_or_object;
                stack_ptr += 1u;
            }
        }
    }

    return min_dist;
}

// ---------- SDF Soft Shadow ----------

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

fn sdf_ao(pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    var ao = 0.0;
    var scale = 1.0;
    for (var i = 1u; i <= 4u; i++) {
        let dist = AO_STEP_SIZE * f32(i);
        let d = sample_sdf(pos + normal * dist);
        ao += scale * (dist - d);
        scale *= 0.5;
    }
    return clamp(1.0 - AO_STRENGTH * ao, 0.0, 1.0);
}

// ---------- Subsurface Scattering ----------

fn sss_contribution(pos: vec3<f32>, normal: vec3<f32>, light_dir: vec3<f32>,
                    subsurface: f32, subsurface_color: vec3<f32>) -> vec3<f32> {
    if subsurface <= 0.0 {
        return vec3<f32>(0.0);
    }
    let interior_pos = pos - normal * SSS_MAX_THICKNESS;
    let thickness = clamp(-sample_sdf(interior_pos), 0.0, SSS_MAX_THICKNESS);
    let attenuation = exp(-thickness * SSS_SIGMA);
    let wrap = max(0.0, dot(normal, light_dir) + SSS_WRAP) / (1.0 + SSS_WRAP);
    return subsurface_color * attenuation * wrap * subsurface;
}

// ---------- Light Attenuation ----------

fn distance_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = dist * dist;
    let r2 = range * range;
    let factor = d2 / r2;
    let window = clamp(1.0 - factor, 0.0, 1.0);
    return (window * window) / max(d2, 0.0001);
}

// ---------- PBR Functions ----------

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn visibility_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a2 = roughness * roughness;
    let ggxv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let ggxl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(ggxv + ggxl, 0.0001);
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

fn simplex3d(v: vec3<f32>) -> f32 {
    let C = vec2<f32>(1.0 / 6.0, 1.0 / 3.0);
    let D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    var i = floor(v + dot(v, vec3<f32>(C.y)));
    let x0 = v - i + dot(i, vec3<f32>(C.x));

    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + vec3<f32>(C.x);
    let x2 = x0 - i2 + vec3<f32>(C.y);
    let x3 = x0 - D.yyy;

    i = mod289_3(i);
    let p = permute(permute(permute(
        i.z + vec4<f32>(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4<f32>(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    let n_ = 0.142857142857;
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

    let norm = taylor_inv_sqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    var m = max(vec4<f32>(0.5) - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4<f32>(0.0));
    m = m * m;
    return 105.0 * dot(m * m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
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

// ---------- Radiance Volume Cone Tracing ----------

/// Sample a single level of the radiance volume.
/// `cam_rel_pos` is camera-relative (since volume center = camera world pos,
/// this is the same as volume-center-relative).
fn sample_radiance_level(cam_rel_pos: vec3<f32>, level: u32) -> vec4<f32> {
    let inv_ext = radiance_vol.inv_extents[level];
    let uvw = cam_rel_pos * inv_ext + 0.5;
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    switch level {
        case 0u: { return textureSampleLevel(radiance_L0, radiance_sampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(radiance_L1, radiance_sampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(radiance_L2, radiance_sampler, uvw, 0.0); }
        default: { return textureSampleLevel(radiance_L3, radiance_sampler, uvw, 0.0); }
    }
}

/// Sample the radiance volume with continuous mip level (interpolate between levels).
fn sample_radiance(cam_rel_pos: vec3<f32>, mip_f: f32) -> vec4<f32> {
    let lo = u32(floor(mip_f));
    let hi = u32(ceil(mip_f));
    let lo_clamped = min(lo, 3u);
    let hi_clamped = min(hi, 3u);
    let s_lo = sample_radiance_level(cam_rel_pos, lo_clamped);
    if lo_clamped == hi_clamped {
        return s_lo;
    }
    let s_hi = sample_radiance_level(cam_rel_pos, hi_clamped);
    return mix(s_lo, s_hi, fract(mip_f));
}

/// Trace a cone through the radiance volume using front-to-back compositing.
/// `origin` is in world-space; positions are converted to camera-relative
/// before sampling the radiance volume (which is centered at the camera).
fn trace_cone(origin: vec3<f32>, dir: vec3<f32>, tan_half_angle: f32, max_dist: f32, jitter: f32) -> vec4<f32> {
    var color = vec3<f32>(0.0);
    var opacity = 0.0;
    // Start past L0 voxel to avoid self-illumination, with jitter to break banding.
    var t = radiance_vol.voxel_sizes.x * (2.0 + jitter);

    for (var i = 0u; i < GI_CONE_STEPS; i++) {
        if opacity > 0.95 || t > max_dist {
            break;
        }
        let pos = origin + dir * t;
        let cone_radius = t * tan_half_angle;
        // Mip selection: *0.5 accounts for 4× (not 2×) clipmap ratio between levels.
        let mip_f = log2(max(cone_radius / radiance_vol.voxel_sizes.x, 1.0)) * 0.5;

        // Convert world-space position to camera-relative for radiance volume sampling.
        let cam_rel_pos = pos - shade_uniforms.camera_pos.xyz;
        let s = sample_radiance(cam_rel_pos, mip_f);
        let step_opacity = s.a;

        // Front-to-back compositing.
        let w = (1.0 - opacity) * step_opacity;
        color += s.rgb * w;
        opacity += w;

        // Step size increases with cone radius.
        let step = max(cone_radius * 0.5, radiance_vol.voxel_sizes.x);
        t += min(step, GI_MAX_STEP);
    }

    return vec4<f32>(color, opacity);
}

/// Trace 6 diffuse cones in a hemisphere around the surface normal.
fn cone_trace_diffuse(pos: vec3<f32>, normal: vec3<f32>, jitter: f32) -> vec3<f32> {
    // Build tangent frame.
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    let tan_half = 0.577; // tan(30°) for ~60° opening cones
    var gi = vec3<f32>(0.0);

    // 6 cones tilted 30° from normal, evenly spaced azimuthally.
    let cos30 = 0.866;
    let sin30 = 0.5;

    for (var i = 0u; i < 6u; i++) {
        let angle = f32(i) * PI / 3.0 + jitter * 0.5;
        let dir = normalize(
            normal * cos30
            + (tangent * cos(angle) + bitangent * sin(angle)) * sin30
        );
        let result = trace_cone(pos, dir, tan_half, GI_DIFFUSE_MAX_DIST, jitter);
        gi += result.rgb;
    }

    return gi / 6.0;
}

/// Trace 1 specular cone along the reflection direction.
fn cone_trace_specular(pos: vec3<f32>, reflect_dir: vec3<f32>, roughness: f32, jitter: f32) -> vec3<f32> {
    // Narrower cone for smoother surfaces.
    let tan_half = max(roughness * 0.5, 0.02);
    let result = trace_cone(pos, reflect_dir, tan_half, GI_SPECULAR_MAX_DIST, jitter);
    return result.rgb;
}

// ---------- Analytic Atmosphere ----------

/// Henyey-Greenstein phase function for sky Mie scattering.
fn henyey_greenstein_sky(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * pow(max(denom, 1e-6), 1.5));
}

/// Compute sky color for a given view ray using analytic Rayleigh + Mie scattering.
/// Returns linear HDR RGB.
fn atmosphere_sky(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let cos_theta = ray_dir.y;  // elevation: dot(ray_dir, up)
    let cos_sun = dot(ray_dir, sun_dir);

    let rayleigh_scale = shade_uniforms.sky_params.x;
    let mie_scale = shade_uniforms.sky_params.y;
    let sun_intensity = shade_uniforms.sun_dir.w;
    let sun_col = shade_uniforms.sun_color.xyz;

    // Height-integrated optical depth coefficients (β × scale_height).
    // Rayleigh: β_R(λ) × H_R(8400m), Mie: β_M × H_M(1200m).
    let tau_r = vec3<f32>(0.032, 0.114, 0.278) * rayleigh_scale;
    let tau_m = 0.025 * mie_scale;

    // Path length through atmosphere (longer at horizon, ~1 at zenith).
    let path = 1.0 / max(cos_theta + 0.025, 0.01);

    // Extinction along the view ray.
    let total_tau = tau_r + vec3<f32>(tau_m);
    let extinction = exp(-total_tau * path);

    // Phase functions.
    let phase_r = 0.75 * (1.0 + cos_sun * cos_sun);
    let g = 0.76;
    let phase_m = henyey_greenstein_sky(cos_sun, g);

    // In-scattered radiance (single scattering).
    // inscatter_i = (β_i * phase_i / Σβ) * (1 - exp(-Σβ * path)) * L_sun
    let scatter_r = tau_r * phase_r;
    let scatter_m = vec3<f32>(tau_m * phase_m);
    let safe_total = max(total_tau, vec3<f32>(1e-6));
    let inscatter = (scatter_r + scatter_m) / safe_total * (vec3<f32>(1.0) - extinction);
    let sky = inscatter * sun_col * sun_intensity;

    // Sun disk + bloom.
    let sun_angular_radius = 0.00465;  // ~0.267 degrees
    let sun_disk = smoothstep(cos(sun_angular_radius * 3.0), cos(sun_angular_radius), cos_sun);
    let sun_bloom = pow(max(cos_sun, 0.0), 256.0) * 2.0;
    let sun_contribution = (sun_disk * 50.0 + sun_bloom) * sun_col * sun_intensity * extinction;

    return sky + sun_contribution;
}

/// Compute view ray direction from pixel coordinates using camera basis.
fn compute_view_ray(pixel: vec2<u32>, dims: vec2<u32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(dims);
    let ndc = uv * 2.0 - 1.0;
    // cam_right and cam_up are pre-scaled by tan(fov/2)*aspect and tan(fov/2).
    return normalize(
        shade_uniforms.cam_forward.xyz
        + ndc.x * shade_uniforms.cam_right.xyz
        - ndc.y * shade_uniforms.cam_up.xyz  // -y: screen y is top-down
    );
}
